"""Post-HPO finalize: refit each outer fold on its best hparams, evaluate on
outer_test, generate comparison figures.

This runs ONCE after all parallel HPO workers have finished contributing trials.
Splitting refit/eval from the trial-running phase fixes a race where worker 0
would start refit before workers 1..N had completed their final in-flight trials,
causing the refit to use a stale ``study.best_params``.

Usage:
    python hpo/finalize.py --output-dir hpo/results/<run_dir>

Resumability: if ``fold_<i>/outer_test_scores.json`` exists AND a matching
``fold<i>_prune_scores-*.pkl`` is present in the cache folder, the fold's refit
is skipped (cache hit). Pass ``--force-refit`` to override.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

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
from auto_circuit.prune_algos.prune_algos import GROUND_TRUTH_PRUNE_ALGO
from auto_circuit.types import AblationType, PatchType, PruneScores, TaskPruneScores
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache

from hpo.cv_utils import kfold_prompt_loaders, register_fold_kv_caches
from hpo.nested_cv import (
    TASK_REGISTRY,
    _parse_ablation_types,
    _register_fold_algos,
)
from hpo.scoring import DEFAULT_EVAL_EDGE_COUNTS, score_all
from hpo.search_spaces import SEARCH_SPACES

CACHE_FOLDER = ".prune_scores_cache"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize an HPO run: refit, eval, figures")
    p.add_argument("--output-dir", required=True,
                   help="Path to the run output dir (must contain config.json)")
    p.add_argument("--no-figs", action="store_true",
                   help="Skip generating comparison figures.")
    p.add_argument("--rebuild-figs", action="store_true",
                   help="Re-render figures even if outputs already exist.")
    p.add_argument("--force-refit", action="store_true",
                   help="Re-refit every fold even if outer_test_scores.json exists.")
    return p.parse_args()


def _cache_dir() -> Path:
    return repo_path_to_abs_path(CACHE_FOLDER)


def _find_fold_cache(fold_idx: int) -> Optional[Path]:
    """Return most-recent fold-cache pkl path, or None."""
    matches = sorted(_cache_dir().glob(f"fold{fold_idx}_prune_scores-*.pkl"))
    return matches[-1] if matches else None


def main() -> None:
    cli_args = _parse_args()
    out_root = Path(cli_args.output_dir)
    config_path = out_root / "config.json"
    if not config_path.exists():
        sys.exit(f"[finalize] config.json not found in {out_root}")
    config = json.loads(config_path.read_text())

    algo_key = config["algo"]
    task_key = config["task"]
    seed = config["seed"]
    outer_k = config["outer_k"]
    cv_batch_size = config.get("cv_batch_size")
    max_epochs_cap = config.get("max_epochs_cap")

    ss = SEARCH_SPACES[algo_key]
    task = TASK_REGISTRY[task_key]
    print(f"[finalize] Task: {task.name}")
    print(f"[finalize] Algo: {algo_key}")
    print(f"[finalize] Output dir: {out_root}")

    task.init_task()
    model = task.model
    true_edges = task.true_edges
    ablation_types = _parse_ablation_types(config["ablation_type"])
    algo_eval_ablation = (
        ss.eval_ablation_type if ss.eval_ablation_type is not None else AblationType.RESAMPLE
    )

    if config.get("eval_edge_counts"):
        eval_edge_counts = [int(x) for x in config["eval_edge_counts"].split(",")]
    else:
        eval_edge_counts = DEFAULT_EVAL_EDGE_COUNTS

    fidelity_param = ss.fidelity_param

    # Load baseline (saved by worker 0 during HPO). save_cache appends a
    # timestamp suffix, so glob for the newest match.
    task_prune_scores: TaskPruneScores = defaultdict(dict)
    baseline_matches = sorted(_cache_dir().glob("baseline_prune_scores-*.pkl"))
    if baseline_matches:
        loaded = load_cache(CACHE_FOLDER, baseline_matches[-1].name)
        task_prune_scores = defaultdict(dict, loaded)
        print(f"[finalize] Loaded baseline from {baseline_matches[-1].name}")
        print(f"[finalize] Baseline algos: {list(task_prune_scores.get(task.key, {}).keys())}")
    else:
        print(f"[finalize] No baseline cache found; figures will compare only fold refits.")

    # Build outer folds with the same seed used by the worker run
    if outer_k > 1:
        outer_folds = kfold_prompt_loaders(
            task.train_loader, k=outer_k, seed=seed, batch_size=cv_batch_size,
        )
    else:
        outer_folds = [(task.train_loader, task.test_loader)]

    # Register outer fold KV caches (idempotent if batch sizes match existing)
    all_outer_loaders = [loader for pair in outer_folds for loader in pair]
    register_fold_kv_caches(model, *all_outer_loaders)

    outer_scores: Dict[str, Dict[str, Optional[float]]] = {}
    best_hparams_per_fold: List[Dict[str, Any]] = []
    fold_prune_scores: Dict[str, PruneScores] = {}

    for fold_idx, (outer_train, outer_test) in enumerate(outer_folds):
        print(f"\n[finalize] === Outer fold {fold_idx + 1}/{len(outer_folds)} ===")
        fold_dir = out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_key = f"{algo_key} / fold{fold_idx}"

        # Open the per-fold journal study
        safe_algo = algo_key.replace(" ", "_").replace("/", "-")
        safe_task = task.key.replace(" ", "_").replace("/", "-")
        study_name = f"{safe_algo}_{safe_task}_outer{fold_idx}"
        log_path = fold_dir / "optuna_study.log"
        if not log_path.exists():
            print(f"[finalize] Fold {fold_idx}: no optuna_study.log found, skipping fold.")
            continue
        storage = JournalStorage(JournalFileBackend(str(log_path)))
        study = optuna.load_study(study_name=study_name, storage=storage)

        best_hparams = study.best_params
        best_hparams_per_fold.append(best_hparams)

        # Single-writer (finalize is a single process), safe to write directly.
        trials_df = study.trials_dataframe()
        trials_df.to_csv(fold_dir / "optuna_trials.csv", index=False)
        (fold_dir / "best_hparams.json").write_text(json.dumps(best_hparams, indent=2))
        print(f"[finalize] Fold {fold_idx}: best inner score = {study.best_value:.4f}")
        print(f"[finalize] Best hparams: {best_hparams}")

        # Cache-hit: prior finalize already produced refit + eval for this fold
        scores_path = fold_dir / "outer_test_scores.json"
        cache_pkl = _find_fold_cache(fold_idx)
        if not cli_args.force_refit and scores_path.exists() and cache_pkl is not None:
            print(f"[finalize] Fold {fold_idx}: cache hit ({cache_pkl.name}), skipping refit "
                  f"(use --force-refit to override).")
            outer_scores[f"fold{fold_idx}"] = json.loads(scores_path.read_text())
            cached = load_cache(CACHE_FOLDER, cache_pkl.name)
            if task.key in cached and fold_key in cached[task.key]:
                fold_prune_scores[fold_key] = cached[task.key][fold_key]
            continue

        # Refit
        print(f"[finalize] Refitting on full outer_train ({fold_idx})...")
        refit_hparams = dict(best_hparams)
        if fidelity_param is not None and max_epochs_cap is not None:
            refit_hparams[fidelity_param] = max_epochs_cap
        refit_ps = ss.base_func(model, outer_train, true_edges, **refit_hparams)

        fold_prune_scores[fold_key] = refit_ps
        save_cache(
            {task.key: {fold_key: refit_ps}},
            CACHE_FOLDER,
            f"fold{fold_idx}_prune_scores",
        )

        # Eval outer_test (one-shot unbiased estimate)
        print(f"[finalize] Evaluating on outer_test ({fold_idx})...")
        outer_fold_scores = score_all(
            model, refit_ps, outer_test, true_edges, eval_edge_counts, algo_eval_ablation
        )
        outer_scores[f"fold{fold_idx}"] = outer_fold_scores
        scores_path.write_text(json.dumps(outer_fold_scores, indent=2))
        print(f"[finalize] Outer-test scores: {outer_fold_scores}")

    # ------------------------------------------------------------------
    # Aggregate outer scores
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "per_fold": outer_scores,
        "best_hparams_per_fold": best_hparams_per_fold,
    }
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
    print(f"\n[finalize] === Summary ===")
    for k, v in summary.items():
        if k not in ("per_fold", "best_hparams_per_fold"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Save combined prune scores (baseline + all fold refits)
    # ------------------------------------------------------------------
    _register_fold_algos(algo_key, fold_prune_scores)
    combined: TaskPruneScores = {
        task.key: {**task_prune_scores.get(task.key, {}), **fold_prune_scores}
    }
    save_cache(combined, CACHE_FOLDER, "combined_prune_scores")
    print(f"[finalize] Saved combined_prune_scores.pkl with keys: {list(combined[task.key].keys())}")

    # ------------------------------------------------------------------
    # Comparison figures
    # ------------------------------------------------------------------
    if cli_args.no_figs:
        print("[finalize] --no-figs: skipping figures.")
        return

    fig_dir = out_root / "comparison_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figs_exist = (fig_dir / "roc.pdf").exists()
    if figs_exist and not cli_args.rebuild_figs:
        print(f"[finalize] Figures already exist in {fig_dir}; pass --rebuild-figs to regenerate.")
        return

    # ROC
    try:
        if true_edges is not None:
            roc_measurements = measure_roc(combined)
            fig_roc = roc_plot(roc_measurements)
            fig_roc.write_image(str(fig_dir / "roc.pdf"))
            fig_roc.write_html(str(fig_dir / "roc.html"))
            print("[finalize] Saved roc.pdf and roc.html")
    except Exception as e:
        print(f"[finalize] ROC figure skipped: {e}")

    # Prune metric plots
    try:
        _metrics = [CLEAN_KL_DIV_METRIC, LOGIT_DIFF_PERCENT_METRIC]
        gt_key = GROUND_TRUTH_PRUNE_ALGO.key
        non_gt = {task.key: {k: v for k, v in combined[task.key].items() if k != gt_key}}
        prune_measurements = measure_prune_metrics(
            ablation_types=ablation_types,
            metrics=_metrics,
            task_prune_scores=non_gt,
            patch_type=PatchType.TREE_PATCH,
            test_edge_counts=eval_edge_counts or None,
        )
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
            base = f"metric_{meta['metric_key'].replace(' ', '_')}_{i}"
            fig.write_image(str(fig_dir / f"{base}.pdf"))
            fig.write_html(str(fig_dir / f"{base}.html"))
        print(f"[finalize] Saved metric figures (pdf + html) to {fig_dir}")
    except Exception as e:
        print(f"[finalize] Metric figures skipped: {e}")

    print("\n[finalize] Done.")


if __name__ == "__main__":
    main()
