"""Validate Edge Pruning against existing AutoCircuit algorithms on IOI.

This script:
1. Loads GPT-2 Small with the IOI component circuit task
2. Runs Edge Pruning, EAP, Subnetwork Probing, and Random baselines
3. Evaluates all algorithms at multiple sparsity levels
4. Produces sparsity vs. faithfulness (KL divergence) comparison plots
5. Saves prune scores for further analysis

Usage:
    python experiments/validate_edge_pruning.py
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch as t

from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
    PruneMetric,
)
from auto_circuit.prune_algos.prune_algos import (
    ACDC_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    PruneAlgo,
    run_prune_algos,
)
from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK, TASK_DICT, Task
from auto_circuit.types import (
    AblationType,
    PatchType,
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.misc import repo_path_to_abs_path, save_cache

# ============================================================================
# Configuration
# ============================================================================

# Which algorithms to compare
ALGOS: List[PruneAlgo] = [
    RANDOM_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,          # Fast, one-pass baseline
    # ACDC_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,  # Closest existing multi-epoch method
    EDGE_PRUNING_PRUNE_ALGO,             # Our integration
]

# Which tasks to evaluate on
TASKS: List[Task] = [
    IOI_COMPONENT_CIRCUIT_TASK,
]

# Evaluation configuration
ABLATION_TYPES: List[AblationType] = [
    AblationType.RESAMPLE,
]
PRUNE_METRICS: List[PruneMetric] = [
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
]

# Save configuration
SAVE_PRUNE_SCORES = True
SAVE_FIGURES = True
FIGURES_DIR = "figures/edge_pruning_validation"

# ============================================================================
# Step 1: Compute prune scores
# ============================================================================
print("=" * 60)
print("Edge Pruning Validation Experiment")
print("=" * 60)

device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Tasks: {[t.name for t in TASKS]}")
print(f"Algorithms: {[a.name for a in ALGOS]}")
print()

task_prune_scores: TaskPruneScores = run_prune_algos(TASKS, ALGOS)

# Convert to float32 for consistent evaluation
for task_key, algo_prune_scores in task_prune_scores.items():
    for algo_key, prune_scores in algo_prune_scores.items():
        for mod_name, scores in prune_scores.items():
            task_prune_scores[task_key][algo_key][mod_name] = scores.float()

# ============================================================================
# Step 2: Save prune scores
# ============================================================================
if SAVE_PRUNE_SCORES:
    cache_folder = ".prune_scores_cache"
    base_filename = "ep-validation"
    save_cache(task_prune_scores, cache_folder, base_filename)

# ============================================================================
# Step 3: Evaluate with prune metrics
# ============================================================================
prune_metric_measurements: TaskMeasurements = measure_prune_metrics(
    ablation_types=ABLATION_TYPES,
    metrics=PRUNE_METRICS,
    task_prune_scores=task_prune_scores,
    patch_type=PatchType.EDGE_PATCH,
    reverse_clean_corrupt=False,
)

# ============================================================================
# Step 4: Generate and display figures
# ============================================================================
fig_pairs = list(measurement_figs(prune_metric_measurements))

if SAVE_FIGURES:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = Path(repo_path_to_abs_path(FIGURES_DIR)) / f"run_{timestamp}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fig, meta in fig_pairs:
        abl = meta["ablation_type"].replace(" ", "_").lower()
        metric = meta["metric_name"].replace(" ", "_").lower()
        auc = "_auc" if meta.get("is_auc") else ""
        filepath = fig_dir / f"ep_validation_{abl}_{metric}{auc}.pdf"
        fig.write_image(str(filepath))
        print(f"Saved: {filepath}")

# ============================================================================
# Step 5: Print summary statistics
# ============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for task_key, algo_prune_scores in task_prune_scores.items():
    task = TASK_DICT[task_key]
    print(f"\nTask: {task.name} ({task.model.n_edges} total edges)")
    for algo_key, prune_scores in algo_prune_scores.items():
        nonzero = sum((ps.abs() > 1e-6).sum().item() for ps in prune_scores.values())
        sparsity_pct = 100.0 * (1.0 - nonzero / task.model.n_edges)
        print(f"  {algo_key:40s} | Hard-Pruned: {sparsity_pct:>5.1f}% | Non-zero continuous scores: {nonzero:5d}/{task.model.n_edges}")

print("\nDone!")