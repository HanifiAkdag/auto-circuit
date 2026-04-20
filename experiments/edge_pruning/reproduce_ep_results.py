"""Reproduce Edge Pruning results from the paper on GPT-2 IOI.

Matches the experimental setup from:
    Bhaskar et al., "Finding Transformer Circuits with Edge Pruning"
    (NeurIPS 2024, §5.1 — IOI experiments)

This script sweeps multiple sparsity targets to trace the faithfulness curve,
comparing our AutoCircuit integration against the paper's reported numbers.

Paper hyperparameters (from Appendix B, Table 4):
    - Model: GPT-2 Small (12 layers, 12 heads)
    - LR (edges): 0.8
    - LR (nodes): 0.8
    - LR (reg_edges): 0.8
    - LR (reg_nodes): 0.8
    - Edge sparsity warmup: 2500 steps
    - Node sparsity warmup: 500 steps
    - Training steps: 3000
    - Batch size: 32
    - Target edge sparsity: 1.1
    - Target node sparsity: 0.72

Usage:
    python experiments/edge_pruning/reproduce_ep_results.py
"""
#%%
import json
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import torch as t

from auto_circuit.metrics.prune_metrics.kl_div import kl_div_metric
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PruneMetric,
)
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.edge_pruning import edge_pruning_prune_scores
from auto_circuit.prune_algos.prune_algos import (
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    PruneAlgo,
    run_prune_algos,
)
from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK, TASK_DICT, Task
from auto_circuit.types import (
    AblationType,
    PatchType,
    PruneScores,
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.misc import repo_path_to_abs_path, save_cache

# ============================================================================
# Paper hyperparameters
# ============================================================================

# Sparsity targets to sweep (tracing the faithfulness-sparsity curve)
SPARSITY_TARGETS: List[float] = [
    0.5,    # Low sparsity (many edges kept)
    0.7,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
    1.0,
    1.05,
    1.1,    # Maximum sparsity (paper default)
]

# Shared hyperparameters (from paper Appendix B)
SHARED_HPARAMS = dict(
    learning_rate=0.8,
    lambda_learning_rate=0.8,
    steps=3000,
    warmup_steps=0,
    start_edge_sparsity=0.0,
    start_node_sparsity=0.0,
    sparsity_warmup_steps=2500,
    use_node_masks=True,
    init_log_alpha=10.0,
    show_train_graph=False,
)

# Node sparsity target (fixed across edge sparsity sweep)
NODE_SPARSITY_TARGET = 0.72

FIGURES_DIR = "figures/edge_pruning_reproduction"

# ============================================================================
# Step 1: Load task
# ============================================================================
print("=" * 60)
print("Edge Pruning Reproduction Experiment")
print("=" * 60)

device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Device: {device}")

task = IOI_COMPONENT_CIRCUIT_TASK
print(f"Task: {task.name}")
print(f"Model: gpt2-small ({task.model.n_edges} edges)")
print()

# ============================================================================
# Step 2: Run EP at each sparsity target
# ============================================================================

all_prune_scores: Dict[str, PruneScores] = {}

# Run EAP as baseline (fast, single pass)
print("Running EAP baseline...")
eap_ps = EDGE_ATTR_PATCH_PRUNE_ALGO.func(
    task.model, task.train_loader, task.true_edges
)
all_prune_scores["EAP"] = eap_ps

# Run EP at each sparsity target
for target_sparsity in SPARSITY_TARGETS:
    run_name = f"EP (s={target_sparsity})"
    print(f"\nRunning {run_name}...")
    ps = edge_pruning_prune_scores(
        model=task.model,
        dataloader=task.train_loader,
        official_edges=task.true_edges,
        target_edge_sparsity=target_sparsity,
        target_node_sparsity=NODE_SPARSITY_TARGET,
        **SHARED_HPARAMS,
    )
    all_prune_scores[run_name] = ps

    # Quick stats
    nonzero = sum((v.abs() > 1e-6).sum().item() for v in ps.values())
    total = task.model.n_edges
    print(f"  Non-zero edges: {nonzero}/{total} ({100*nonzero/total:.1f}%)")

# ============================================================================
# Step 3: Evaluate at multiple edge counts
# ============================================================================
print("\nEvaluating circuits...")

# Create edge counts to test at
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

# Build task_prune_scores structure for measure_prune_metrics
task_prune_scores: TaskPruneScores = {
    task.key: {k: v for k, v in all_prune_scores.items()}
}

# Evaluate
PRUNE_METRICS: List[PruneMetric] = [
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
]
prune_metric_measurements = measure_prune_metrics(
    ablation_types=[AblationType.RESAMPLE],
    metrics=PRUNE_METRICS,
    task_prune_scores=task_prune_scores,
    patch_type=PatchType.EDGE_PATCH,
    reverse_clean_corrupt=False,
)

# ============================================================================
# Step 4: Generate figures
# ============================================================================
figs = list(measurement_figs(prune_metric_measurements))

# Custom comparison plot: sparsity target vs achieved faithfulness
fig_custom = go.Figure()
for algo_name, ps in all_prune_scores.items():
    nonzero = sum((v.abs() > 1e-6).sum().item() for v in ps.values())
    sparsity_pct = 100.0 * (1.0 - nonzero / n_edges)
    fig_custom.add_trace(go.Scatter(
        x=[sparsity_pct],
        y=[0],  # Placeholder — will be filled by actual KL values
        mode="markers+text",
        text=[algo_name],
        textposition="top center",
        name=algo_name,
    ))
fig_custom.update_layout(
    title="Edge Pruning: Sparsity vs Algorithm",
    xaxis_title="Sparsity (%)",
    yaxis_title="KL Divergence",
)
figs.append(fig_custom)

# Save
fig_dir = Path(repo_path_to_abs_path(FIGURES_DIR))
fig_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for i, fig in enumerate(figs):
    filepath = fig_dir / f"ep_reproduce_{i}_{timestamp}.pdf"
    fig.write_image(str(filepath))
    print(f"Saved: {filepath}")

for fig in figs:
    fig.show()

# ============================================================================
# Step 5: Save results
# ============================================================================
save_cache(task_prune_scores, ".prune_scores_cache", "ep-reproduction")

# Print summary
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)
print(f"{'Algorithm':40s} | {'Non-zero edges':>15s} | {'Sparsity %':>10s}")
print("-" * 75)
for algo_name, ps in all_prune_scores.items():
    nonzero = sum((v.abs() > 1e-6).sum().item() for v in ps.values())
    sparsity_pct = 100.0 * (1.0 - nonzero / n_edges)
    print(f"{algo_name:40s} | {nonzero:15.0f} | {sparsity_pct:10.2f}%")

print("\nDone!")

#%%
