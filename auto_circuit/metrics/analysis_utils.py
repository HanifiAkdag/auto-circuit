"""Post-hoc analysis utilities for sub-RQ3 multi-run comparisons.

Used by the analysis pass that loads separately-saved PruneScores caches and
computes Jaccard stability, cross-method overlap, and metric extraction.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import json
import numpy as np

from auto_circuit.types import (
    AblationType,
    AblationMeasurements,
    Edge,
    PruneScores,
    TaskPruneScores,
)
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.misc import load_cache
from auto_circuit.utils.tensor_ops import prune_scores_threshold


def top_k_edge_set(prune_scores: PruneScores, k: int, edges: Set[Edge]) -> Set[Edge]:
    """Return the set of top-k edges by absolute prune score.

    Uses the same ranking as run_circuits / prune.py (abs threshold via
    prune_scores_threshold from tensor_ops).
    """
    if k == 0:
        return set()
    threshold = prune_scores_threshold(prune_scores, k)
    return {e for e in edges if abs(e.prune_score(prune_scores).item()) >= threshold.item()}


def jaccard(a: Set[Edge], b: Set[Edge]) -> float:
    """Jaccard similarity between two edge sets. Returns 1.0 if both are empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def overlap_matrix(
    runs: List[PruneScores], k: int, edges: Set[Edge]
) -> np.ndarray:
    """Pairwise Jaccard matrix for N runs at a fixed edge budget k.

    Returns an (N, N) array where entry [i, j] is jaccard(run_i @ k, run_j @ k).
    """
    edge_sets = [top_k_edge_set(ps, k, edges) for ps in runs]
    n = len(edge_sets)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = jaccard(edge_sets[i], edge_sets[j])
    return mat


def matched_budgets(edges: Set[Edge], true_edge_count: Optional[int] = None) -> List[int]:
    """Return the representative edge-count grid used by measure_prune_metrics.

    Wraps edge_counts_util to ensure Jaccard comparisons use the same grid as
    faithfulness curves (so sparsity levels are comparable across figures).
    """
    return edge_counts_util(edges=edges, true_edge_count=true_edge_count)


def load_many_caches(
    folder: str, glob_pattern: str
) -> List[Tuple[Optional[Dict], TaskPruneScores]]:
    """Load all PruneScores caches matching a glob pattern, with their manifests.

    For each matched .pkl file, attempts to load a sibling manifest.json in the
    same directory. Returns a list of (manifest_dict_or_None, task_prune_scores).

    Example:
        caches = load_many_caches(".prune_scores_cache", "cell_ioi_ep_resample_seed*")
    """
    results = []
    cache_dir = Path(folder)
    for pkl_path in sorted(cache_dir.glob(glob_pattern + "*.pkl")):
        task_prune_scores: TaskPruneScores = load_cache(folder, pkl_path.name)
        manifest_path = pkl_path.parent / pkl_path.name.replace(".pkl", "_manifest.json")
        manifest = None
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        results.append((manifest, task_prune_scores))
    return results


def extract_metric(
    measurements: AblationMeasurements,
    ablation: AblationType,
    metric_key: str,
    task_key: str,
    algo_key: str,
) -> List[Tuple[float, float]]:
    """Extract the (edge_count, score) series from AblationMeasurements.

    AblationMeasurements is Dict[AblationType, Dict[metric_key, Dict[task_key,
    Dict[algo_key, List[Tuple[int|float, int|float]]]]]] as returned by
    measure_prune_metrics.
    """
    return measurements[ablation][metric_key][task_key][algo_key]
