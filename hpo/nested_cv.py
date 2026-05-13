"""Nested cross-validation with multi-fidelity HPO for circuit-discovery algorithms.

Usage (single GPU):
    python hpo/nested_cv.py --algo "DiscoGP Edge Only" --task IOI

Usage (multi-GPU, 4 workers sharing one study):
    #SBATCH --gres=gpu:4
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python hpo/nested_cv.py \\
            --algo "DiscoGP Edge Only" --task IOI \\
            --output-dir runs/disco_ioi \\
            --worker-id $i &
    done
    wait

The script:
  1. Loads a base prune-scores cache (or runs other algos fresh).
  2. Strips the target algo from the cache so it can be tuned from scratch.
  3. Builds outer K-fold splits over task.train_loader.
  4. For each outer fold:
       a. Splits outer_train into inner_train / inner_val (holdout or K-fold).
       b. Runs an Optuna study with Hyperband (or configured pruner).
       c. Refits the best hyperparameters on full outer_train.
       d. Evaluates on outer_test (once — avoids selection bias).
  5. Saves per-fold best_hparams, PruneScores, Optuna CSV, and combined pkl.
  6. Generates comparison figures (ROC + AUC metric plots) alongside baseline algos.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import torch as t
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

# Repo root on sys.path so we can always import auto_circuit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_circuit.metrics.official_circuits.measure_roc import measure_roc
from auto_circuit.metrics.official_circuits.roc_plot import roc_plot
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
)
from auto_circuit.prune_algos.prune_algos import (
    ACDC_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_HARD_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
    DISCO_GP_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
    PruneAlgo,
    run_prune_algos,
)
from auto_circuit.tasks import (
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    TASK_DICT,
    Task,
)
from auto_circuit.types import AblationType, PatchType, PruneScores, TaskPruneScores
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache

from hpo.cv_utils import holdout_split, kfold_prompt_loaders, register_fold_kv_caches
from hpo.scoring import DEFAULT_EVAL_EDGE_COUNTS, METRIC_REGISTRY, score_all
from hpo.search_spaces import SEARCH_SPACES

# ---------------------------------------------------------------------------
# Task registry (short CLI alias → Task)
# ---------------------------------------------------------------------------
TASK_REGISTRY: Dict[str, Task] = {
    "IOI": IOI_COMPONENT_CIRCUIT_TASK,
    "GREATERTHAN": GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    "DOCSTRING": DOCSTRING_TOKEN_CIRCUIT_TASK,
}

# ---------------------------------------------------------------------------
# Default comparison algos (run once, not tuned)
# ---------------------------------------------------------------------------
DEFAULT_COMPARISON_ALGOS = [
    GROUND_TRUTH_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    ACDC_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_HARD_PRUNE_ALGO,
    DISCO_GP_PRUNE_ALGO,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested CV + multi-fidelity HPO for prune algos")

    # Core
    p.add_argument("--algo", default="DiscoGP Edge Only",
                   help="Algo key to tune (must be in SEARCH_SPACES). Default: 'DiscoGP Edge Only'")
    p.add_argument("--task", default="IOI",
                   choices=list(TASK_REGISTRY.keys()),
                   help="Task to run on. Default: IOI")

    # Cache
    p.add_argument("--base-cache", default=None,
                   help="Filename (inside .prune_scores_cache/) of existing TaskPruneScores pkl "
                        "to use as baseline. If omitted, comparison algos are re-run from scratch.")
    p.add_argument("--run-comparison-algos", action="store_true",
                   help="Force re-running comparison algos even if --base-cache is given.")

    # CV structure
    p.add_argument("--outer-k", type=int, default=3,
                   help="Number of outer folds. Set 1 to do HPO once without outer CV.")
    p.add_argument("--inner-mode", choices=["holdout", "kfold"], default="holdout")
    p.add_argument("--inner-k", type=int, default=3,
                   help="Number of inner folds (only used if --inner-mode=kfold).")
    p.add_argument("--inner-val-frac", type=float, default=0.25,
                   help="Fraction of outer-train to hold out for inner validation.")
    p.add_argument("--cv-batch-size", type=int, default=None,
                   help="Batch size for fold loaders. Auto-detected if omitted.")

    # HPO
    p.add_argument("--n-trials", type=int, default=30,
                   help="Total Optuna trials per outer fold.")
    p.add_argument("--selection-metric",
                   choices=["kl_div_auc", "logit_diff_auc", "roc_auc"],
                   default="kl_div_auc",
                   help="Scalar metric used to rank trials in the inner loop.")
    p.add_argument("--pruner",
                   choices=["hyperband", "successive_halving", "median", "none"],
                   default="hyperband",
                   help="Optuna pruner for multi-fidelity early stopping.")
    p.add_argument("--min-resource", type=int, default=10,
                   help="Minimum epochs/steps for Hyperband/SHA.")
    p.add_argument("--reduction-factor", type=int, default=3,
                   help="Reduction factor η for Hyperband/SHA.")
    p.add_argument("--max-epochs-cap", type=int, default=None,
                   help="Override the algo's default max epochs/steps (the Hyperband budget).")
    p.add_argument("--eval-every", type=int, default=10,
                   help="How often (in epochs/steps) the eval_callback is fired.")

    # Evaluation
    p.add_argument("--ablation-type",
                   nargs="+",
                   choices=["resample", "tokenwise_mean_corrupt", "zero"],
                   default=["zero", "resample"],
                   help="Ablation type(s) used for comparison figures. Multiple values produce"
                        " separate figure panels. Default: zero resample")
    p.add_argument("--eval-edge-counts", default=None,
                   help="Comma-separated list of edge counts for AUC sweep, e.g. '0,10,50,200'.")

    # I/O
    p.add_argument("--output-dir", default=None,
                   help="Directory for all outputs. Auto-named if omitted.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worker-id", type=int, default=0,
                   help="GPU worker index. Added to --seed for per-worker sampling diversity.")
    p.add_argument("--storage-backend", choices=["sqlite", "journal"], default="journal",
                   help="Optuna study storage backend. Default: journal (recommended for multi-process).")
    p.add_argument("--no-figs", action="store_true",
                   help="Skip generating comparison figures (faster).")

    return p.parse_args()


_ABLATION_MAP = {
    "resample": AblationType.RESAMPLE,
    "tokenwise_mean_corrupt": AblationType.TOKENWISE_MEAN_CORRUPT,
    "zero": AblationType.ZERO,
}


def _parse_ablation_types(names: List[str]) -> List[AblationType]:
    return [_ABLATION_MAP[n] for n in names]


def _make_pruner(args: argparse.Namespace, max_resource: int) -> optuna.pruners.BasePruner:
    if args.pruner == "hyperband":
        return HyperbandPruner(
            min_resource=args.min_resource,
            max_resource=max_resource,
            reduction_factor=args.reduction_factor,
        )
    elif args.pruner == "successive_halving":
        return SuccessiveHalvingPruner(
            min_resource=args.min_resource,
            reduction_factor=args.reduction_factor,
        )
    elif args.pruner == "median":
        return MedianPruner()
    else:
        return NopPruner()


def _make_storage(storage_dir: Path, fold_idx: int, algo_key: str, task_key: str,
                  backend: str) -> str:
    safe_algo = algo_key.replace(" ", "_").replace("/", "-")
    safe_task = task_key.replace(" ", "_").replace("/", "-")
    study_name = f"{safe_algo}_{safe_task}_outer{fold_idx}"
    if backend == "sqlite":
        db_path = storage_dir / f"optuna_outer{fold_idx}.db"
        return f"sqlite:///{db_path}"
    else:  # journal
        log_path = storage_dir / f"optuna_outer{fold_idx}.log"
        return optuna.storages.JournalStorage(
            optuna.storages.JournalFileBackend(str(log_path))
        )


def _make_eval_callback(trial, model, val_loader, true_edges, edge_counts, ablation_type,
                        selection_metric: str):
    """Build the per-epoch callback that reports to Optuna and checks should_prune()."""
    lower_better = METRIC_REGISTRY[selection_metric][1]

    def callback(step: int, prune_scores: PruneScores) -> bool:
        scores = score_all(model, prune_scores, val_loader, true_edges,
                           edge_counts, ablation_type)
        val_score = scores.get(selection_metric)
        if val_score is not None and not (val_score != val_score):  # not NaN
            trial.report(val_score, step)
        return trial.should_prune()

    return callback


def _register_fold_algos(algo_key: str, fold_prune_scores: Dict[str, PruneScores]) -> None:
    """Add fold refit PruneAlgos to PRUNE_ALGO_DICT so measurement_figs can find them."""
    base_algo = PRUNE_ALGO_DICT.get(algo_key)
    base_short = base_algo._short_name if (base_algo and base_algo._short_name) else algo_key.split()[0]
    for fold_key, ps in fold_prune_scores.items():
        captured = ps  # closure capture
        fold_num = fold_key.rsplit("fold", 1)[-1]  # "0", "1", "2" from "... / fold0"
        if fold_key not in PRUNE_ALGO_DICT:
            PRUNE_ALGO_DICT[fold_key] = PruneAlgo(
                key=fold_key,
                name=fold_key,
                _short_name=f"{base_short}/f{fold_num}",
                func=lambda *a, _ps=captured: _ps,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    t.manual_seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ------------------------------------------------------------------
    # Validate algo
    # ------------------------------------------------------------------
    if args.algo not in SEARCH_SPACES:
        print(f"[ERROR] '{args.algo}' is not in SEARCH_SPACES. Available:")
        for k in SEARCH_SPACES:
            print(f"  {k}")
        sys.exit(1)

    ss = SEARCH_SPACES[args.algo]

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir is None:
        safe_algo = args.algo.replace(" ", "_")
        out_root = repo_path_to_abs_path(f"hpo/results/{safe_algo}_{args.task}_{timestamp}")
    else:
        out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
    task = TASK_REGISTRY[args.task]
    print(f"[nested_cv] Task: {task.name}")
    print(f"[nested_cv] Algo: {args.algo}")

    # Init task (loads model + data) — idempotent
    task.init_task()
    model = task.model
    true_edges = task.true_edges
    ablation_types = _parse_ablation_types(args.ablation_type)
    # HPO scoring uses the algo's preferred ablation (e.g. ZERO for DiscoGP),
    # falling back to RESAMPLE — independent of --ablation-type which controls figures only.
    algo_eval_ablation = ss.eval_ablation_type if ss.eval_ablation_type is not None else AblationType.RESAMPLE
    print(f"[nested_cv] Algo eval ablation (HPO): {algo_eval_ablation} | Figure ablations: {ablation_types}")

    # ------------------------------------------------------------------
    # Edge counts for AUC sweep
    # ------------------------------------------------------------------
    if args.eval_edge_counts:
        eval_edge_counts = [int(x) for x in args.eval_edge_counts.split(",")]
    else:
        eval_edge_counts = DEFAULT_EVAL_EDGE_COUNTS

    # ------------------------------------------------------------------
    # Max-epochs budget for Hyperband
    # ------------------------------------------------------------------
    fidelity_default_map = {
        "e_train_epochs": 500,
        "w_train_epochs": 500,
        "steps": 3000,
    }
    fidelity_param = ss.fidelity_param
    if fidelity_param is not None:
        max_resource = args.max_epochs_cap or fidelity_default_map.get(fidelity_param, 500)
    else:
        max_resource = 1  # unused

    # ------------------------------------------------------------------
    # Load / build baseline prune scores
    # ------------------------------------------------------------------
    cache_folder = ".prune_scores_cache"
    task_prune_scores: TaskPruneScores = defaultdict(dict)

    if args.base_cache:
        print(f"[nested_cv] Loading base cache: {args.base_cache}")
        loaded = load_cache(cache_folder, args.base_cache)
        task_prune_scores = defaultdict(dict, loaded)

    # Strip the target algo from the loaded scores (we'll relearn it)
    algo_ps = task_prune_scores.get(task.key, {})
    if args.algo in algo_ps:
        print(f"[nested_cv] Removing '{args.algo}' from loaded cache for retuning.")
        del algo_ps[args.algo]
    task_prune_scores[task.key] = algo_ps

    if args.run_comparison_algos:
        # Run comparison algos (excluding the one being tuned) on task.train_loader
        comparison_algos = [
            a for a in DEFAULT_COMPARISON_ALGOS
            if a.key != args.algo and a.key not in task_prune_scores.get(task.key, {})
        ]
        if comparison_algos:
            print(f"[nested_cv] Running {len(comparison_algos)} comparison algo(s)...")
            fresh = run_prune_algos([task], comparison_algos)
            for algo_key, ps in fresh.get(task.key, {}).items():
                task_prune_scores[task.key][algo_key] = ps

    # Save the baseline copy (stripped of target algo)
    baseline_path = out_root / "baseline_prune_scores.pkl"
    save_cache(dict(task_prune_scores), cache_folder, "baseline_prune_scores")
    print(f"[nested_cv] Baseline algos: {list(task_prune_scores.get(task.key, {}).keys())}")

    # Save config
    config = vars(args)
    config["max_resource"] = max_resource
    (out_root / "config.json").write_text(json.dumps(config, indent=2))

    # ------------------------------------------------------------------
    # Outer folds
    # ------------------------------------------------------------------
    is_lower_better = METRIC_REGISTRY[args.selection_metric][1]
    optuna_direction = "minimize" if is_lower_better else "maximize"

    if args.outer_k > 1:
        outer_folds = kfold_prompt_loaders(
            task.train_loader, k=args.outer_k,
            seed=args.seed, batch_size=args.cv_batch_size,
        )
    else:
        # Use the full train/test split as one fold
        outer_folds = [(task.train_loader, task.test_loader)]

    outer_scores: Dict[str, Dict[str, Optional[float]]] = {}
    best_hparams_per_fold: List[Dict[str, Any]] = []
    fold_prune_scores: Dict[str, PruneScores] = {}

    # ------------------------------------------------------------------
    # Outer fold loop
    # ------------------------------------------------------------------
    for fold_idx, (outer_train, outer_test) in enumerate(outer_folds):
        print(f"\n[nested_cv] === Outer fold {fold_idx + 1}/{len(outer_folds)} ===")
        fold_dir = out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Build inner split
        if args.inner_mode == "holdout":
            inner_train, inner_val = holdout_split(
                outer_train,
                val_frac=args.inner_val_frac,
                seed=args.seed + fold_idx,
                batch_size=args.cv_batch_size,
            )
            inner_splits = [(inner_train, inner_val)]
        else:
            inner_splits = kfold_prompt_loaders(
                outer_train, k=args.inner_k,
                seed=args.seed + fold_idx,
                batch_size=args.cv_batch_size,
            )

        # Register all fold KV caches into the model so its forward pass can find them
        all_fold_loaders = [outer_train, outer_test] + [
            loader for pair in inner_splits for loader in pair
        ]
        register_fold_kv_caches(model, *all_fold_loaders)

        # ------------------------------------------------------------------
        # Build Optuna study
        # ------------------------------------------------------------------
        safe_algo = args.algo.replace(" ", "_").replace("/", "-")
        safe_task = task.key.replace(" ", "_").replace("/", "-")
        study_name = f"{safe_algo}_{safe_task}_outer{fold_idx}"

        if args.storage_backend == "sqlite":
            db_path = fold_dir / "optuna_study.db"
            storage_url = f"sqlite:///{db_path}"
        else:
            log_path = fold_dir / "optuna_study.log"
            storage_url = JournalStorage(JournalFileBackend(str(log_path)))

        pruner = (
            _make_pruner(args, max_resource)
            if ss.supports_intermediate
            else NopPruner()
        )

        study = optuna.create_study(
            direction=optuna_direction,
            pruner=pruner,
            sampler=TPESampler(seed=args.seed + args.worker_id + fold_idx * 100),
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
        )

        # ------------------------------------------------------------------
        # Objective
        # ------------------------------------------------------------------
        def objective(trial: optuna.Trial) -> float:
            hparams = ss.suggest(trial)

            # Cap fidelity if requested
            if fidelity_param is not None and args.max_epochs_cap is not None:
                hparams[fidelity_param] = args.max_epochs_cap

            inner_scores_list: List[Optional[float]] = []

            for inner_idx, (in_train, in_val) in enumerate(inner_splits):
                # Build the eval callback for multi-fidelity pruning (only on first inner fold)
                callback = None
                if ss.supports_intermediate and inner_idx == 0:
                    callback = _make_eval_callback(
                        trial, model, in_val, true_edges,
                        eval_edge_counts, algo_eval_ablation, args.selection_metric,
                    )
                    extra = dict(eval_callback=callback, eval_every=args.eval_every)
                else:
                    extra = {}

                try:
                    ps = ss.base_func(model, in_train, true_edges, **hparams, **extra)
                except optuna.TrialPruned:
                    raise
                finally:
                    # Release the CUDA memory pool so successive Hyperband trials
                    # don't accumulate allocations (expandable_segments retains them).
                    t.cuda.empty_cache()

                fold_scores = score_all(
                    model, ps, in_val, true_edges, eval_edge_counts, algo_eval_ablation
                )
                sel = fold_scores.get(args.selection_metric)
                inner_scores_list.append(sel)

                # Log all metrics as user attrs
                for metric_name, val in fold_scores.items():
                    if val is not None:
                        trial.set_user_attr(f"inner_fold{inner_idx}_{metric_name}", val)

            # Average valid scores across inner folds
            valid = [s for s in inner_scores_list if s is not None]
            if not valid:
                raise optuna.TrialPruned()
            return sum(valid) / len(valid)

        # Bind fold-level variables into the objective (needed because Python closures
        # capture by reference and fold_idx / inner_splits / outer_train would change)
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        # ------------------------------------------------------------------
        # Save Optuna results
        # ------------------------------------------------------------------
        trials_df = study.trials_dataframe()
        trials_df.to_csv(fold_dir / "optuna_trials.csv", index=False)

        best_hparams = study.best_params
        best_hparams_per_fold.append(best_hparams)
        (fold_dir / "best_hparams.json").write_text(json.dumps(best_hparams, indent=2))
        print(f"[nested_cv] Fold {fold_idx}: best inner score = {study.best_value:.4f}")
        print(f"[nested_cv] Best hparams: {best_hparams}")

        # ------------------------------------------------------------------
        # Refit + outer-test evaluation: only worker 0 does this to avoid
        # redundant re-training when multiple workers share the same study.
        # ------------------------------------------------------------------
        if args.worker_id != 0:
            print(f"[nested_cv] Worker {args.worker_id}: skipping refit (worker 0 will do it).")
            continue

        print(f"[nested_cv] Refitting on full outer_train ({fold_idx})...")
        refit_hparams = dict(best_hparams)
        if fidelity_param is not None and args.max_epochs_cap is not None:
            refit_hparams[fidelity_param] = args.max_epochs_cap
        refit_ps = ss.base_func(model, outer_train, true_edges, **refit_hparams)

        fold_key = f"{args.algo} / fold{fold_idx}"
        fold_prune_scores[fold_key] = refit_ps
        save_cache(
            {task.key: {fold_key: refit_ps}},
            cache_folder,
            f"fold{fold_idx}_prune_scores",
        )

        # ------------------------------------------------------------------
        # Evaluate refit model on outer_test (one-shot, unbiased estimate)
        # ------------------------------------------------------------------
        print(f"[nested_cv] Evaluating on outer_test ({fold_idx})...")
        outer_fold_scores = score_all(
            model, refit_ps, outer_test, true_edges, eval_edge_counts, algo_eval_ablation
        )
        outer_scores[f"fold{fold_idx}"] = outer_fold_scores
        (fold_dir / "outer_test_scores.json").write_text(
            json.dumps(outer_fold_scores, indent=2)
        )
        print(f"[nested_cv] Outer-test scores: {outer_fold_scores}")

    # Non-primary workers only contributed HPO trials; refit and reporting
    # are handled by worker 0 to avoid duplicate work.
    if args.worker_id != 0:
        print(f"\n[nested_cv] Worker {args.worker_id}: done contributing trials. Exiting.")
        return

    # ------------------------------------------------------------------
    # Aggregate outer scores
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "per_fold": outer_scores,
        "best_hparams_per_fold": best_hparams_per_fold,
    }
    # Compute mean ± std across folds for each metric
    for metric_name in ["kl_div_auc", "logit_diff_auc", "roc_auc"]:
        vals = [
            outer_scores[f"fold{i}"].get(metric_name)
            for i in range(len(outer_folds))
            if outer_scores.get(f"fold{i}", {}).get(metric_name) is not None
        ]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary[f"{metric_name}_mean"] = mean
            summary[f"{metric_name}_std"] = std
    (out_root / "outer_scores.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[nested_cv] === Summary ===")
    for k, v in summary.items():
        if k not in ("per_fold", "best_hparams_per_fold"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Save combined prune scores (baseline + all fold refits)
    # ------------------------------------------------------------------
    _register_fold_algos(args.algo, fold_prune_scores)
    combined: TaskPruneScores = {
        task.key: {**task_prune_scores.get(task.key, {}), **fold_prune_scores}
    }
    save_cache(combined, cache_folder, "combined_prune_scores")
    print(f"[nested_cv] Saved combined_prune_scores.pkl with keys: {list(combined[task.key].keys())}")

    # ------------------------------------------------------------------
    # Comparison figures
    # ------------------------------------------------------------------
    if not args.no_figs:
        fig_dir = out_root / "comparison_figs"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # ROC plot
        try:
            if true_edges is not None:
                roc_measurements = measure_roc(combined)
                fig_roc = roc_plot(roc_measurements)
                fig_roc.write_image(str(fig_dir / "roc.pdf"))
                print("[nested_cv] Saved roc.pdf")
        except Exception as e:
            print(f"[nested_cv] ROC figure skipped: {e}")

        # Prune metric plots
        try:
            _metrics = [CLEAN_KL_DIV_METRIC, LOGIT_DIFF_PERCENT_METRIC]

            # Non-GT algos: sweep over the configured edge counts
            gt_key = GROUND_TRUTH_PRUNE_ALGO.key
            non_gt = {task.key: {k: v for k, v in combined[task.key].items() if k != gt_key}}
            prune_measurements = measure_prune_metrics(
                ablation_types=ablation_types,
                metrics=_metrics,
                task_prune_scores=non_gt,
                patch_type=PatchType.TREE_PATCH,
                test_edge_counts=eval_edge_counts or None,
            )

            # GT algo: single edge count so it renders as a marker (black X), not a line
            if gt_key in combined.get(task.key, {}) and task.true_edge_count is not None:
                gt_only = {task.key: {gt_key: combined[task.key][gt_key]}}
                gt_measurements = measure_prune_metrics(
                    ablation_types=ablation_types,
                    metrics=_metrics,
                    task_prune_scores=gt_only,
                    patch_type=PatchType.TREE_PATCH,
                    test_edge_counts=[task.true_edge_count],
                )
                for abl in gt_measurements:
                    for mk in gt_measurements[abl]:
                        for tk in gt_measurements[abl][mk]:
                            prune_measurements[abl][mk][tk].update(gt_measurements[abl][mk][tk])

            for i, (fig, meta) in enumerate(measurement_figs(prune_measurements)):
                fname = f"metric_{meta['metric_key'].replace(' ', '_')}_{i}.pdf"
                fig.write_image(str(fig_dir / fname))
            print(f"[nested_cv] Saved metric figures to {fig_dir}")
        except Exception as e:
            print(f"[nested_cv] Metric figures skipped: {e}")

    print("\n[nested_cv] Done.")


if __name__ == "__main__":
    main()
