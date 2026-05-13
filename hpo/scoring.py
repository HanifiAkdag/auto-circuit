"""Scalar evaluation functions for HPO trial ranking.

Each function takes a trained PruneScores and an evaluation loader and returns
a single float suitable for Optuna's ``trial.report`` and objective return.

Three metrics are available:
- ``kl_div_auc``:     log-log AUC of KL divergence vs edges. Lower is better.
                      Uses raw area (no baseline subtraction) so a circuit with
                      low KL at all edge counts scores lower. Works on any task.
- ``logit_diff_auc``: baseline-subtracted AUC of logit-diff % vs edges.
                      Higher is better. Best for IOI.
- ``roc_auc``:        ROC AUC against the official ground-truth circuit.
                      Higher is better. Returns None if task has no true_edges.

``score_all`` computes all three and returns a dict keyed by metric name.
"""
import math
from typing import Dict, List, Optional, Set

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.metrics.area_under_curve import measurements_auc
from auto_circuit.metrics.official_circuits.measure_roc import measure_task_roc
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PruneMetric,
)
from auto_circuit.prune import run_circuits
from auto_circuit.types import AblationType, Edge, Measurements, PatchType, PruneScores
from auto_circuit.utils.patchable_model import PatchableModel

# Consistent set of edge-count sweep points used for AUC integration.
# Must start at 0 (required by measurements_auc) and span a logarithmic range.
DEFAULT_EVAL_EDGE_COUNTS: List[int] = [0, 5, 10, 20, 50, 100, 200, 500, 1000]


def _run_sweep(
    model: PatchableModel,
    prune_scores: PruneScores,
    eval_loader: PromptDataLoader,
    edge_counts: List[int],
    ablation_type: AblationType,
    metric: PruneMetric,
) -> Optional[Measurements]:
    """Run the circuit sweep and return raw (edge_count, metric_value) measurements."""
    try:
        n_edges = model.n_edges
        counts = [c for c in edge_counts if c <= n_edges]
        if not counts or counts[0] != 0:
            counts = [0] + [c for c in counts if c > 0]

        circuit_outs = run_circuits(
            model=model,
            dataloader=eval_loader,
            test_edge_counts=counts,
            prune_scores=prune_scores,
            patch_type=PatchType.TREE_PATCH,
            ablation_type=ablation_type,
        )
        measurements: Measurements = metric.metric_func(model, eval_loader, circuit_outs)
        del circuit_outs
        t.cuda.empty_cache()
        return measurements if len(measurements) > 1 else None
    except Exception:
        return None


def _raw_log_auc(measurements: Measurements, metric: PruneMetric) -> float:
    """Log-log AUC without baseline subtraction.

    Integrates the raw metric curve in log-x / log-y space. For lower-is-better
    metrics (KL div), a smaller area means consistently low values = better circuit.
    """
    pts = sorted(measurements, key=lambda p: p[0])
    area = 0.0
    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        if metric.log_y and metric.y_min is not None:
            y1 = max(y1, metric.y_min)
            y2 = max(y2, metric.y_min)
        if metric.log_x and (x1 == 0 or x2 == 0):
            continue
        if metric.log_x:
            x1 = math.log10(x1)
            x2 = math.log10(x2)
        if metric.log_y and metric.y_min is not None:
            y1 = math.log10(y1) - math.log10(metric.y_min)
            y2 = math.log10(y2) - math.log10(metric.y_min)
        area += (x2 - x1) * (y1 + y2) / 2.0
    return max(area, 1e-3)


def _metric_auc(
    model: PatchableModel,
    prune_scores: PruneScores,
    eval_loader: PromptDataLoader,
    edge_counts: List[int],
    ablation_type: AblationType,
    metric: PruneMetric,
) -> Optional[float]:
    """Baseline-subtracted AUC (for higher-is-better metrics like logit_diff)."""
    measurements = _run_sweep(model, prune_scores, eval_loader, edge_counts, ablation_type, metric)
    if measurements is None:
        return None
    try:
        return measurements_auc(
            measurements,
            log_x=metric.log_x,
            log_y=metric.log_y,
            y_min=metric.y_min,
        )
    except Exception:
        return None


def kl_div_auc(
    model: PatchableModel,
    prune_scores: PruneScores,
    eval_loader: PromptDataLoader,
    edge_counts: List[int] = DEFAULT_EVAL_EDGE_COUNTS,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> Optional[float]:
    """Raw log-log AUC of KL divergence vs edges kept. Lower is better.

    Uses raw area (no baseline subtraction) because KL always decreases as edges
    are added, making the baseline-subtracted area always negative and uninformative.
    A circuit that achieves low KL at few edges → small area → lower score → better.
    """
    measurements = _run_sweep(model, prune_scores, eval_loader, edge_counts, ablation_type, CLEAN_KL_DIV_METRIC)
    if measurements is None:
        return None
    return _raw_log_auc(measurements, CLEAN_KL_DIV_METRIC)


def logit_diff_auc(
    model: PatchableModel,
    prune_scores: PruneScores,
    eval_loader: PromptDataLoader,
    edge_counts: List[int] = DEFAULT_EVAL_EDGE_COUNTS,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> Optional[float]:
    """AUC of logit-difference % vs edges kept. Higher is better."""
    return _metric_auc(model, prune_scores, eval_loader, edge_counts, ablation_type, LOGIT_DIFF_PERCENT_METRIC)


def roc_auc(
    model: PatchableModel,
    prune_scores: PruneScores,
    true_edges: Optional[Set[Edge]],
) -> Optional[float]:
    """ROC AUC against the official ground-truth circuit. Higher is better.

    Returns None if ``true_edges`` is None (task has no official circuit).
    """
    if true_edges is None:
        return None
    try:
        points = measure_task_roc(model, true_edges, prune_scores)
        points = sorted(points, key=lambda p: p[0])
        area = 0.0
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            area += (x2 - x1) * (y1 + y2) / 2.0
        return area
    except Exception:
        return None


# Maps metric name -> (function, lower_is_better)
METRIC_REGISTRY: Dict[str, tuple] = {
    "kl_div_auc":      (kl_div_auc,       True),
    "logit_diff_auc":  (logit_diff_auc,    False),
    "roc_auc":         (roc_auc,           False),
}


def score_all(
    model: PatchableModel,
    prune_scores: PruneScores,
    eval_loader: PromptDataLoader,
    true_edges: Optional[Set[Edge]],
    edge_counts: List[int] = DEFAULT_EVAL_EDGE_COUNTS,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> Dict[str, Optional[float]]:
    """Compute all three scalar scores and return as a dict.

    Keys: ``"kl_div_auc"``, ``"logit_diff_auc"``, ``"roc_auc"``.
    Values may be None if a metric is unavailable (e.g., no true_edges for roc_auc,
    or incompatible task for logit_diff_auc).
    """
    return {
        "kl_div_auc":     kl_div_auc(model, prune_scores, eval_loader, edge_counts, ablation_type),
        "logit_diff_auc": logit_diff_auc(model, prune_scores, eval_loader, edge_counts, ablation_type),
        "roc_auc":        roc_auc(model, prune_scores, true_edges),
    }
