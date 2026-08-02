"""Post-hoc analysis utilities for sub-RQ3 multi-run comparisons.

Used by the analysis pass that loads separately-saved PruneScores caches and
computes Jaccard stability, cross-method overlap, and metric extraction.
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

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


@dataclass(frozen=True)
class Budget:
    """A circuit-size specification that resolves against one run's scores.

    ``top_k`` matches the number of edges; ``threshold`` matches the score cutoff. The
    second exists for ACDC, whose score is the tau at which an edge was pruned: with only
    ~21 distinct levels over 32k edges, a top-k request admits the straddling tie block
    whole and overshoots by a seed-dependent amount, which caps Jaccard by arithmetic.
    Matching tau is ACDC's native parameterisation, not a workaround.
    """

    kind: Literal["top_k", "threshold"]
    value: float

    @property
    def label(self) -> str:
        return str(int(self.value)) if self.kind == "top_k" else f"{self.value:.1e}"

    @property
    def slug(self) -> str:
        return f"k{self.label}" if self.kind == "top_k" else f"tau{self.label}"


def as_budget(budget: "Budget | int") -> Budget:
    return budget if isinstance(budget, Budget) else Budget("top_k", budget)


def budget_threshold(prune_scores: PruneScores, budget: "Budget | int") -> float:
    """The absolute-score cutoff this budget implies, in run_circuits' own terms."""
    budget = as_budget(budget)
    if budget.kind == "threshold":
        return budget.value
    return prune_scores_threshold(prune_scores, int(budget.value)).item()


def select_edges(
    prune_scores: PruneScores,
    budget: "Budget | int",
    edges: Set[Edge],
    exclude_zero: bool = False,
) -> Set[Edge]:
    """The circuit this budget selects, using run_circuits' `|score| >= threshold` rule."""
    budget = as_budget(budget)
    if budget.kind == "top_k" and budget.value == 0:
        return set()
    threshold = budget_threshold(prune_scores, budget)
    return {
        e
        for e in edges
        if (s := abs(e.prune_score(prune_scores).item())) >= threshold
        and (s > 0 or not exclude_zero)
    }


def realised_edge_count(
    prune_scores: PruneScores, budget: "Budget | int", edges: Set[Edge]
) -> int:
    """How many edges the budget actually selects.

    The bridge to the GPU pass: `run_circuits` only accepts edge counts, and converts one
    back to a threshold via the k-th largest score. Because a tau budget sits on an
    attained score level, handing it this count reproduces exactly the same edge set.
    """
    return len(select_edges(prune_scores, budget, edges))


def top_k_edge_set(
    prune_scores: PruneScores, k: int, edges: Set[Edge], exclude_zero: bool = False
) -> Set[Edge]:
    """Return the set of top-k edges by absolute prune score.

    Uses the same ranking as run_circuits / prune.py (abs threshold via
    prune_scores_threshold from tensor_ops).

    With ``exclude_zero``, edges with an exactly-zero score are never included even
    when the k-th largest |score| is 0 (binary GT scores or ACDC's mostly-zero
    scores at k above their nonzero count would otherwise admit every edge); the
    circuit then simply stops growing at the number of nonzero-score edges.
    """
    return select_edges(prune_scores, Budget("top_k", k), edges, exclude_zero)


def jaccard(a: Set[Edge], b: Set[Edge]) -> float:
    """Jaccard similarity between two edge sets. Returns 1.0 if both are empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass(frozen=True)
class EdgeIndex:
    """A canonical edge order plus the gather needed to read scores into that order.

    Edges are ordered by name, so the index space is defined by edge identity rather than
    by `PruneScores` dict iteration order. Comparing two runs through it is therefore safe
    even for caches written under different key orders, which flattening the dicts is not.
    Built once per model; reused for every run and every budget.
    """

    edges: Tuple[Edge, ...]
    gather: Dict[str, Tuple[np.ndarray, np.ndarray]]

    def __len__(self) -> int:
        return len(self.edges)


def edge_index(edges: Set[Edge], prune_scores: PruneScores) -> EdgeIndex:
    order = sorted(edges, key=lambda e: e.name)
    per_module: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for pos, edge in enumerate(order):
        shape = tuple(prune_scores[edge.dest.module_name].shape)
        flat = int(np.ravel_multi_index(edge.patch_idx, shape))
        per_module[edge.dest.module_name].append((flat, pos))
    gather = {
        mod: (np.array([f for f, _ in v]), np.array([p for _, p in v]))
        for mod, v in per_module.items()
    }
    return EdgeIndex(tuple(order), gather)


def edge_score_vector(index: EdgeIndex, prune_scores: PruneScores) -> np.ndarray:
    """Absolute scores of every edge, in the index's canonical order."""
    scores = np.empty(len(index), dtype=np.float64)
    for mod, (flat, pos) in index.gather.items():
        flat_scores = prune_scores[mod].detach().abs().flatten().cpu().numpy()
        scores[pos] = flat_scores[flat]
    return scores


def vector_threshold(scores: np.ndarray, budget: "Budget | int") -> float:
    budget = as_budget(budget)
    if budget.kind == "threshold":
        return budget.value
    k = int(budget.value)
    return float(-np.partition(-scores, k - 1)[k - 1])


def budget_mask(
    scores: np.ndarray, budget: "Budget | int", exclude_zero: bool = False
) -> np.ndarray:
    budget = as_budget(budget)
    if budget.kind == "top_k" and budget.value == 0:
        return np.zeros(scores.shape, dtype=bool)
    mask = scores >= vector_threshold(scores, budget)
    return mask & (scores > 0) if exclude_zero else mask


def pairwise_jaccard(masks: np.ndarray) -> np.ndarray:
    """(S, E) boolean circuits -> (S, S) Jaccard matrix."""
    counts = masks.astype(np.int64)
    intersection = counts @ counts.T
    sizes = counts.sum(1)
    union = sizes[:, None] + sizes[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1), 1.0)


def pairwise_stats(matrix: np.ndarray) -> Dict[str, float]:
    """Summarise the S(S-1)/2 distinct pairs of a Jaccard matrix.

    ``n_pairs`` is reported because the pairs are not independent (each run appears in
    S-1 of them), so the spread must be read as a range over dependent comparisons rather
    than as a sampling interval.
    """
    i, j = np.triu_indices(matrix.shape[0], k=1)
    pairs = matrix[i, j]
    if pairs.size == 0:
        return dict(mean=1.0, sd=0.0, cv=0.0, min=1.0, max=1.0, n_pairs=0)
    mean = float(pairs.mean())
    sd = float(pairs.std(ddof=0))
    return dict(
        mean=mean,
        sd=sd,
        cv=sd / mean if mean else 0.0,
        min=float(pairs.min()),
        max=float(pairs.max()),
        n_pairs=int(pairs.size),
    )


def mutual_iou(masks: np.ndarray) -> Tuple[float, int, int]:
    """Agreement of all S runs at once: |intersection| / |union|, with both raw counts."""
    intersection = int(masks.all(0).sum())
    union = int(masks.any(0).sum())
    return (intersection / union if union else 1.0), intersection, union


def selection_frequency(masks: np.ndarray) -> np.ndarray:
    """How many edges were selected by exactly m of the S runs, for m = 0..S.

    Entry 0 is always 0: edges no run selected are not part of the union.
    """
    per_edge = masks.sum(0)
    freq = np.bincount(per_edge[per_edge > 0], minlength=masks.shape[0] + 1)
    return freq[: masks.shape[0] + 1]


def overlap_matrix(
    runs: List[PruneScores],
    budget: "Budget | int",
    edges: Set[Edge],
    exclude_zero: bool = False,
) -> np.ndarray:
    """Pairwise Jaccard matrix for N runs at a fixed budget.

    Returns an (N, N) array where entry [i, j] is the Jaccard of runs i and j.
    """
    if not runs:
        return np.zeros((0, 0))
    index = edge_index(edges, runs[0])
    masks = np.stack(
        [budget_mask(edge_score_vector(index, ps), budget, exclude_zero) for ps in runs]
    )
    return pairwise_jaccard(masks)


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


def latest_per_cell(
    loaded: List[Tuple[Optional[Dict], TaskPruneScores]],
) -> List[Tuple[Optional[Dict], TaskPruneScores]]:
    """Deduplicate loaded cells, keeping the newest per cell identity.

    A cell is identified by its manifest's ``cell`` block
    (task, algo, ablation, seed, null_test); the newest is chosen by ``saved_at``.
    Re-running a failed/OOM cell just adds a newer timestamped file, so the analysis
    pass transparently picks it up. Cells without a manifest ``cell`` block are kept
    as-is (keyed by their object id, never deduplicated).
    """
    best: Dict[tuple, Tuple[str, Optional[Dict], TaskPruneScores]] = {}
    for manifest, tps in loaded:
        cell = (manifest or {}).get("cell")
        if not cell:
            best[(id(tps),)] = ("", manifest, tps)
            continue
        key = (
            cell.get("task"),
            cell.get("algo"),
            cell.get("ablation"),
            cell.get("seed"),
            cell.get("null_test"),
        )
        saved = (manifest or {}).get("saved_at", "")
        if key not in best or saved > best[key][0]:
            best[key] = (saved, manifest, tps)
    return [(m, t) for _, m, t in best.values()]


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
