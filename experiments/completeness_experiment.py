"""Completeness experiment: Same Under Knockouts.

Measures both FAITHFULNESS and COMPLETENESS of circuits discovered by multiple
algorithms on the IOI Component Circuit task.

Faithfulness: Does the circuit reproduce the full model's behaviour?
Completeness: Does the circuit capture ALL of the model's mechanism?
             (Tested via adversarial knockout of circuit edges.)

The experiment pipeline:
    Phase 1: Circuit discovery (GT, EAP, TP, EP)
    Phase 2: Knockout training (adversarial search for damaging knockouts)
    Phase 3: Completeness evaluation (same-under-knockouts plot)
    Phase 4: Faithfulness curves (KL, Logit Diff %)

Reference: Wang et al. (2022) — "Interpretability in the Wild"
           https://arxiv.org/abs/2211.00593

Usage:
    # Fresh run (compute everything from scratch):
    python experiments/completeness_experiment.py

    # Load cached prune scores, recompute knockouts + faithfulness:
    python experiments/completeness_experiment.py --load-prune-scores <filename.pkl>

    # Load cached knockouts too (only recompute faithfulness + plots):
    python experiments/completeness_experiment.py \\
        --load-prune-scores <filename.pkl> \\
        --load-knockouts <filename.pkl>

    # Skip faithfulness curves:
    python experiments/completeness_experiment.py --no-faithfulness

    # Skip saving caches:
    python experiments/completeness_experiment.py --no-save
"""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch as t

from auto_circuit.metrics.completeness_metrics.same_under_knockouts import (
    TaskCompletenessScores,
    measure_same_under_knockouts,
    same_under_knockouts_fig,
)
from auto_circuit.metrics.completeness_metrics.train_same_under_knockouts import (
    train_same_under_knockouts,
)
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PruneMetric,
)
from auto_circuit.prune_algos.prune_algos import (
    CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
    DISCO_GP_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    PruneAlgo,
    run_prune_algos,
)
from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK, Task
from auto_circuit.types import (
    AblationType,
    PatchType,
    PruneScores,
    TaskPruneScores,
)
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache

# ============================================================================
# Configuration (defaults — overridden by CLI args)
# ============================================================================

# --- Task ---
TASK: Task = IOI_COMPONENT_CIRCUIT_TASK

# --- Algorithms ---
PRUNE_ALGOS: List[PruneAlgo] = [
    DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
    DISCO_GP_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO
]

# --- Knockout training hyperparameters ---
KNOCKOUT_LR: float = 0.02
KNOCKOUT_EPOCHS: int = 300
KNOCKOUT_REG_LAMBDA: float = 0.0
KNOCKOUT_FAITHFULNESS_TARGET: str = "kl_div"

# --- Faithfulness metrics ---
PRUNE_METRICS: List[PruneMetric] = [
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
]

# --- Output directories ---
FIGURES_DIR = "results/completeness_experiment"
PRUNE_SCORES_CACHE = ".prune_scores_cache"
MEASUREMENT_CACHE = ".measurement_cache"


# ============================================================================
# CLI argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Completeness experiment: measure circuit faithfulness "
                    "and completeness via same-under-knockouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Cache loading ---
    parser.add_argument(
        "--load-prune-scores",
        type=str,
        default=None,
        metavar="FILE",
        help="Load prune scores from a cached .pkl file instead of computing "
             "them from scratch. File is loaded from the prune scores cache "
             f"directory ({PRUNE_SCORES_CACHE}/). Example: prune-scores-27-04-2026.pkl",
    )
    parser.add_argument(
        "--load-knockouts",
        type=str,
        default=None,
        metavar="FILE",
        help="Load knockout scores from a cached .pkl file instead of training "
             "them. File is loaded from the knockout cache directory "
             f"({PRUNE_SCORES_CACHE}/). Example: kl_div-knockout-scores-27-04-2026.pkl",
    )

    # --- Phase toggles ---
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Skip Phase 4 (faithfulness curve evaluation).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save intermediate caches (.pkl files).",
    )

    return parser.parse_args()


# ============================================================================
# Main experiment
# ============================================================================

def main():
    args = parse_args()

    # Derive flags from CLI
    load_ps_file: str | None = args.load_prune_scores
    load_ko_file: str | None = args.load_knockouts
    compute_prune_scores: bool = load_ps_file is None
    compute_knockouts: bool = load_ko_file is None
    compute_faithfulness: bool = not args.no_faithfulness
    save_caches: bool = not args.no_save

    print("=" * 70)
    print("Completeness Experiment: Same Under Knockouts")
    print("=" * 70)

    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: {TASK.name}")
    print(f"Mode: {'FRESH RUN' if compute_prune_scores else 'LOADING CACHES'}")
    if load_ps_file:
        print(f"  Prune scores cache: {load_ps_file}")
    if load_ko_file:
        print(f"  Knockout cache: {load_ko_file}")
    print(f"Phases: discovery={'compute' if compute_prune_scores else 'cached'}, "
          f"knockouts={'compute' if compute_knockouts else 'cached'}, "
          f"faithfulness={'yes' if compute_faithfulness else 'skip'}")
    print()

    task = TASK
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = Path(repo_path_to_abs_path(FIGURES_DIR)) / f"run_{timestamp}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Phase 1: Circuit Discovery
    # ==================================================================
    print("=" * 70)
    print("Phase 1: Circuit Discovery")
    print("=" * 70)

    task_prune_scores: TaskPruneScores = defaultdict(dict)

    if compute_prune_scores:
        # Run discovery algorithms
        print(f"\nRunning {len(PRUNE_ALGOS)} discovery algorithms...")
        task_prune_scores = run_prune_algos([task], PRUNE_ALGOS)


        # Ensure float32
        for tk, algo_ps in task_prune_scores.items():
            for ak, ps in algo_ps.items():
                for mod, scores in ps.items():
                    task_prune_scores[tk][ak][mod] = scores.float()

        if save_caches:
            save_cache(task_prune_scores, PRUNE_SCORES_CACHE, "prune-scores")

    else:
        # Load from cache
        assert load_ps_file is not None
        print(f"\nLoading prune scores from: {PRUNE_SCORES_CACHE}/{load_ps_file}")
        task_prune_scores = load_cache(PRUNE_SCORES_CACHE, load_ps_file)

        # Ensure float32
        for tk, algo_ps in task_prune_scores.items():
            for ak, ps in algo_ps.items():
                for mod, scores in ps.items():
                    task_prune_scores[tk][ak][mod] = scores.float()

    # Print summary
    print("\n--- Circuit Discovery Summary ---")
    for algo_key in task_prune_scores[task.key]:
        ps = task_prune_scores[task.key][algo_key]
        nonzero = sum((v.abs() > 1e-6).sum().item() for v in ps.values())
        print(f"  {algo_key:40s}: {nonzero:.0f} non-zero edges")
    print()

    # ==================================================================
    # Phase 2: Knockout Training
    # ==================================================================
    print("=" * 70)
    print("Phase 2: Knockout Training (adversarial search)")
    print("=" * 70)
    print(f"  LR={KNOCKOUT_LR}, epochs={KNOCKOUT_EPOCHS}, "
          f"reg_lambda={KNOCKOUT_REG_LAMBDA}")
    print(f"  faithfulness_target={KNOCKOUT_FAITHFULNESS_TARGET}")
    print()

    completeness_prune_scores: TaskPruneScores = {}

    if compute_knockouts:
        # Determine which algo keys to train knockouts for
        algo_keys_for_ko = list(task_prune_scores[task.key].keys())
        print(f"Training knockouts for: {algo_keys_for_ko}")

        completeness_prune_scores = train_same_under_knockouts(
            task_prune_scores,
            algo_keys=algo_keys_for_ko,
            learning_rate=KNOCKOUT_LR,
            epochs=KNOCKOUT_EPOCHS,
            regularize_lambda=KNOCKOUT_REG_LAMBDA,
            faithfulness_target=KNOCKOUT_FAITHFULNESS_TARGET,
        )

        if save_caches:
            save_cache(
                completeness_prune_scores,
                PRUNE_SCORES_CACHE,
                f"{KNOCKOUT_FAITHFULNESS_TARGET}-knockout-scores",
            )

    else:
        # Load from cache
        assert load_ko_file is not None
        print(f"\nLoading knockout scores from: {PRUNE_SCORES_CACHE}/{load_ko_file}")
        completeness_prune_scores = load_cache(PRUNE_SCORES_CACHE, load_ko_file)

    # ==================================================================
    # Phase 3: Completeness Evaluation
    # ==================================================================
    print()
    print("=" * 70)
    print("Phase 3: Completeness Evaluation (same under knockouts)")
    print("=" * 70)

    if completeness_prune_scores:
        task_completeness_scores: TaskCompletenessScores = (
            measure_same_under_knockouts(
                circuit_ps=task_prune_scores,
                knockout_ps=completeness_prune_scores,
            )
        )

        # Print numeric results
        print("\n--- Completeness Results ---")
        print(f"{'Algorithm':40s} | {'Circuit':>8s} | {'Knockouts':>9s} | "
              f"{'Normal KL':>10s} | {'Knockout KL':>11s} | {'Gap':>8s}")
        print("-" * 100)

        for tk, algo_scores in task_completeness_scores.items():
            for algo_key, scores in algo_scores.items():
                for circ_size, n_ko, normal_kl, ko_kl in scores:
                    gap = ko_kl - normal_kl
                    print(f"  {algo_key:40s} | {circ_size:>8d} | {n_ko:>9d} | "
                          f"{normal_kl:>10.4f} | {ko_kl:>11.4f} | {gap:>+8.4f}")

        # Generate completeness figure
        completeness_fig = same_under_knockouts_fig(task_completeness_scores)
        completeness_path = fig_dir / "completeness_same_under_knockouts.pdf"
        completeness_fig.write_image(str(completeness_path))
        print(f"\nSaved: {completeness_path}")

        # Also save interactive HTML
        completeness_html = fig_dir / "completeness_same_under_knockouts.html"
        completeness_fig.write_html(str(completeness_html))
        print(f"Saved: {completeness_html}")
    else:
        print("No knockout scores available — skipping completeness evaluation.")

    # ==================================================================
    # Phase 4: Faithfulness Curves
    # ==================================================================
    print()
    print("=" * 70)
    print("Phase 4: Faithfulness Curves")
    print("=" * 70)

    if compute_faithfulness and task_prune_scores:
        n_edges = task.model.n_edges
        test_edge_counts = sorted(set([
            1, 5, 10, 25, 50, 100, 200, 300, 500,
            int(n_edges * 0.01),
            int(n_edges * 0.02),
            int(n_edges * 0.05),
            int(n_edges * 0.10),
            int(n_edges * 0.20),
            int(n_edges * 0.50),
            n_edges,
        ]))

        prune_metric_measurements = measure_prune_metrics(
            ablation_types=[AblationType.RESAMPLE],
            metrics=PRUNE_METRICS,
            task_prune_scores=task_prune_scores,
            patch_type=PatchType.TREE_PATCH,
            reverse_clean_corrupt=False,
            test_edge_counts=test_edge_counts,
        )

        if save_caches:
            save_cache(
                prune_metric_measurements,
                MEASUREMENT_CACHE,
                "faithfulness-measurements",
            )

        # Generate faithfulness figures
        fig_pairs = list(measurement_figs(prune_metric_measurements))
        for fig, meta in fig_pairs:
            abl = str(meta["ablation_type"]).replace(" ", "_").lower()
            metric = meta["metric_name"].replace(" ", "_").lower()
            auc = "_auc" if meta.get("is_auc") else ""
            filepath = fig_dir / f"faithfulness_{abl}_{metric}{auc}.pdf"
            fig.write_image(str(filepath))
            print(f"Saved: {filepath}")

            # Also save HTML
            html_path = filepath.with_suffix(".html")
            fig.write_html(str(html_path))
    else:
        print("Skipping faithfulness curves.")

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 70)
    print("Experiment Complete")
    print("=" * 70)
    print(f"All outputs saved to: {fig_dir}")
    if save_caches:
        print(f"Caches saved to:")
        print(f"  Prune scores: {PRUNE_SCORES_CACHE}")
        print(f"  Measurements: {MEASUREMENT_CACHE}")

    if task.true_edge_count is not None:
        print(f"\nGround truth circuit size: {task.true_edge_count} edges")
        print(f"Total model edges: {task.model.n_edges}")
        print(f"Sparsity: {100 * (1 - task.true_edge_count / task.model.n_edges):.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
