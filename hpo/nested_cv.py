"""Nested cross-validation with multi-fidelity HPO for circuit-discovery algorithms.

This script runs trial-only HPO workers. Refit, outer-test evaluation, and
comparison figures are produced by ``hpo/finalize.py`` after all parallel
workers have exited.

Usage (single GPU, single worker):
    python hpo/nested_cv.py --algo "DiscoGP Edge Only" --task IOI

Usage (multi-GPU, 4 workers sharing one study via JournalStorage):
    #SBATCH --gres=gpu:4
    OUT_DIR=runs/disco_ioi
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python hpo/nested_cv.py \\
            --algo "DiscoGP Edge Only" --task IOI \\
            --output-dir $OUT_DIR --worker-id $i &
    done
    wait
    python hpo/finalize.py --output-dir $OUT_DIR

The worker script:
  1. Loads a base prune-scores cache (or runs other algos fresh).
  2. Strips the target algo from the cache so it can be tuned from scratch.
  3. Builds outer K-fold splits over ``task.train_loader``.
  4. For each outer fold:
       a. Splits ``outer_train`` into ``inner_train`` / ``inner_val``.
       b. Runs an Optuna study (Hyperband by default) up to ``--n-trials``
          globally across all workers. Stops contributing once the global
          count of completed/pruned trials is reached.

The finalize script then loads each fold's study, picks ``best_params``,
refits, evaluates on ``outer_test``, and generates ROC / metric figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import torch as t
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

# Repo root on sys.path so we can always import auto_circuit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_circuit.prune_algos.prune_algos import (
    ACDC_PRUNE_ALGO,
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
    Task,
)
from auto_circuit.types import AblationType, PruneScores, TaskPruneScores
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path

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
                   help="Target total trials per outer fold across ALL workers "
                        "(complete+pruned). Idempotent: on SLURM requeue, existing trials "
                        "in the journal are counted, and only the remainder is run.")
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


def _make_eval_callback(trial, model, val_loader, true_edges, edge_counts, ablation_type,
                        selection_metric: str):
    """Per-epoch callback that reports intermediate scores to Optuna.

    Captures the prune decision in ``state["pruned"]`` at the moment
    ``should_prune()`` first returns True, so the objective can definitively
    raise ``TrialPruned()`` afterward. Re-querying ``should_prune()`` later is
    NOT reliable with parallel workers: Hyperband's rung-promotion threshold
    is recomputed from all trials in the journal each call, so other workers'
    in-flight reports can flip a True back to False between the callback and
    the post-algo re-check. That is exactly what produced 47/53 trials being
    Hyperband-aborted but still recorded as COMPLETE in run 388147.
    """
    state: Dict[str, Any] = {"pruned": False}

    def callback(step: int, prune_scores: PruneScores) -> bool:
        scores = score_all(model, prune_scores, val_loader, true_edges,
                           edge_counts, ablation_type)
        val_score = scores.get(selection_metric)
        if val_score is not None and not (val_score != val_score):  # not NaN
            trial.report(val_score, step)
        if trial.should_prune():
            state["pruned"] = True
            return True
        return False

    callback.state = state  # let objective() inspect after the algo returns
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

    # Save the baseline copy (stripped of target algo) and config — worker 0 only,
    # to avoid concurrent writes from parallel SLURM workers to the same path.
    # Scoped to the run dir (not the shared cache) so a later run can't load a
    # stale baseline from a different experiment.
    if args.worker_id == 0:
        t.save(dict(task_prune_scores), out_root / "baseline_prune_scores.pkl")
        config = vars(args)
        config["max_resource"] = max_resource
        (out_root / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[nested_cv] Baseline algos: {list(task_prune_scores.get(task.key, {}).keys())}")

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

    # ------------------------------------------------------------------
    # Outer fold loop — workers only contribute trials. Refit, outer-test
    # evaluation, aggregation, and figures are handled by hpo/finalize.py
    # after the SLURM `wait` barrier, so that the refit's `study.best_params`
    # reflects every trial completed by every worker.
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

            # finalize.py refits from this, not study.best_params (which drops
            # non-suggested kwargs like use_weight_masks / loss_mode / tao_exps).
            trial.set_user_attr("full_hparams", dict(hparams))

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
                finally:
                    # Release the CUDA memory pool so successive Hyperband trials
                    # don't accumulate allocations (expandable_segments retains them).
                    t.cuda.empty_cache()

                # If the eval_callback saw should_prune()==True at any point during
                # training, raise TrialPruned now so Optuna records the trial as
                # PRUNED. We rely on the callback's captured flag (callback.state)
                # rather than re-querying trial.should_prune() — with parallel
                # workers, Hyperband's rung threshold can shift between calls and
                # flip a True back to False, which is exactly what caused run
                # 388147 to record 47/53 Hyperband-aborted trials as COMPLETE.
                if callback is not None and callback.state["pruned"]:
                    raise optuna.TrialPruned()

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
        #
        # Resumability: --n-trials is the GLOBAL target per outer fold (across all
        # workers). On a fresh run the study is empty; on a SLURM requeue, Optuna
        # loads the existing trials from the journal and we only run the remainder.
        def _count_done() -> int:
            return sum(
                1 for trial_obj in study.get_trials(deepcopy=False)
                if trial_obj.state in (TrialState.COMPLETE, TrialState.PRUNED)
            )

        def _stop_cb(study_inner, _trial) -> None:
            if _count_done() >= args.n_trials:
                study_inner.stop()

        already_done = _count_done()
        remaining = max(0, args.n_trials - already_done)
        print(
            f"[nested_cv] Fold {fold_idx}: target={args.n_trials} completed/pruned trials, "
            f"existing={already_done}, this worker will run up to {remaining}."
        )
        if remaining > 0:
            study.optimize(
                objective,
                n_trials=remaining,
                callbacks=[_stop_cb],
                show_progress_bar=True,
            )

        # Live progress (no file writes here — hpo/finalize.py is the single
        # writer for trials CSV, best_hparams.json, refit caches, and figures).
        try:
            print(f"[nested_cv] Fold {fold_idx}: best inner score so far = {study.best_value:.4f}")
        except ValueError:
            # No completed trials yet (e.g., worker started after `_stop_cb` already fired)
            print(f"[nested_cv] Fold {fold_idx}: no completed trials by this worker.")

    print(f"\n[nested_cv] Worker {args.worker_id}: done contributing trials.")
    print(f"[nested_cv] Run `python hpo/finalize.py --output-dir {out_root}` after all "
          f"workers exit to produce refit / outer-test eval / figures.")


if __name__ == "__main__":
    main()
