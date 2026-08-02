#%%
import json

import numpy as np
import torch as t

from auto_circuit.metrics.analysis_utils import (
    Budget,
    budget_mask,
    budget_threshold,
    edge_index,
    edge_score_vector,
    extract_metric,
    jaccard,
    latest_per_cell,
    load_many_caches,
    mutual_iou,
    overlap_matrix,
    pairwise_jaccard,
    pairwise_stats,
    realised_edge_count,
    select_edges,
    selection_frequency,
    top_k_edge_set,
)
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.types import AblationType
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.patchable_model import PatchableModel
from tests.conftest import DEVICE


def _micro_patchable() -> PatchableModel:
    return patchable_model(
        model=MicroModel(n_layers=2),
        factorized=True,
        slice_output="last_seq",
        seq_len=None,
        separate_qkv=True,
        device=DEVICE,
    )


def test_jaccard():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0          # identical
    assert jaccard({1, 2}, {3, 4}) == 0.0                # disjoint
    assert jaccard({1, 2, 3, 4}, {3, 4, 5, 6}) == 2 / 6  # partial
    assert jaccard(set(), set()) == 1.0                  # both empty -> 1.0
    assert jaccard({1}, set()) == 0.0                    # one empty -> 0.0


def test_top_k_edge_set():
    model = _micro_patchable()
    edges = model.edges
    ps = model.new_prune_scores()
    # Assign three distinct, decreasing scores; everything else stays 0.
    chosen = sorted(edges, key=lambda e: e.name)[:3]
    for i, e in enumerate(chosen):
        ps[e.dest.module_name][e.patch_idx] = float(10 - i)  # 10, 9, 8

    assert top_k_edge_set(ps, 0, edges) == set()
    assert top_k_edge_set(ps, 1, edges) == {chosen[0]}
    assert top_k_edge_set(ps, 3, edges) == set(chosen)


def test_overlap_matrix():
    model = _micro_patchable()
    edges = model.edges
    es = sorted(edges, key=lambda e: e.name)

    ps_a = model.new_prune_scores()
    ps_b = model.new_prune_scores()
    for i, e in enumerate(es[:3]):       # top-3 of A
        ps_a[e.dest.module_name][e.patch_idx] = float(10 - i)
    for i, e in enumerate(es[3:6]):      # top-3 of B (disjoint from A)
        ps_b[e.dest.module_name][e.patch_idx] = float(10 - i)

    m = overlap_matrix([ps_a, ps_a, ps_b], budget=3, edges=edges)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 1.0)   # self-overlap
    assert m[0, 1] == 1.0                  # identical runs
    assert m[0, 2] == 0.0                  # disjoint runs
    assert np.allclose(m, m.T)             # symmetric


def _scored(model: PatchableModel, values: dict) -> dict:
    ps = model.new_prune_scores()
    es = sorted(model.edges, key=lambda e: e.name)
    for i, v in values.items():
        ps[es[i].dest.module_name][es[i].patch_idx] = float(v)
    return ps


def test_edge_score_vector_is_key_order_invariant():
    """The vectorised path must key on edge identity, never on dict iteration order.

    Cached cells on disk were written under many different PruneScores key orders, so a
    comparison that indexed by dict order would silently compare unrelated edges.
    """
    model = _micro_patchable()
    ps = _scored(model, {0: 10.0, 1: 9.0, 2: 8.0})
    shuffled = dict(reversed(list(ps.items())))
    assert list(shuffled) != list(ps)

    index = edge_index(model.edges, ps)
    assert np.array_equal(
        edge_score_vector(index, ps), edge_score_vector(index, shuffled)
    )
    # Same multiset of scores as the flat tensors: the order is a permutation, not a subset.
    flat = t.cat([v.flatten().abs() for v in ps.values()]).cpu().numpy()
    assert len(index) == model.n_edges
    assert np.array_equal(
        np.sort(edge_score_vector(index, ps)), np.sort(flat.astype(np.float64))
    )


def test_budget_mask_matches_edge_set_path_across_a_tie_block():
    model = _micro_patchable()
    es = sorted(model.edges, key=lambda e: e.name)
    ps = _scored(model, {0: 10.0, 1: 5.0, 2: 5.0, 3: 5.0, 4: 1.0})
    index = edge_index(model.edges, ps)
    scores = edge_score_vector(index, ps)

    for k in (1, 2, 3, 4, 5):
        budget = Budget("top_k", k)
        from_edges = select_edges(ps, budget, model.edges)
        from_vector = {index.edges[i] for i in np.flatnonzero(budget_mask(scores, budget))}
        assert from_edges == from_vector
    # Requesting 2 admits the whole 3-edge tie block, exactly as run_circuits does.
    assert len(select_edges(ps, Budget("top_k", 2), model.edges)) == 4
    assert realised_edge_count(ps, Budget("threshold", 5.0), model.edges) == 4
    assert select_edges(ps, Budget("threshold", 5.0), model.edges) == set(es[:4])


def test_pairwise_stats_uses_only_distinct_pairs():
    mat = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    stats = pairwise_stats(mat)
    assert stats["n_pairs"] == 3                       # not 9, and not 6
    assert np.isclose(stats["mean"], 0.4)              # diagonal excluded
    assert stats["min"] == 0.2 and stats["max"] == 0.6


def test_mutual_iou_and_selection_frequency():
    identical = np.array([[True, True, False]] * 3)
    assert mutual_iou(identical) == (1.0, 2, 2)
    assert list(selection_frequency(identical)) == [0, 0, 0, 2]

    disjoint = np.array(
        [[True, False, False], [False, True, False], [False, False, True]]
    )
    assert mutual_iou(disjoint) == (0.0, 0, 3)
    assert list(selection_frequency(disjoint)) == [0, 3, 0, 0]
    assert np.allclose(np.diag(pairwise_jaccard(disjoint)), 1.0)
    assert pairwise_jaccard(disjoint)[0, 1] == 0.0


def test_load_many_caches_and_extract_metric(tmp_path):
    """The analysis pass must load >=2 separately-saved cells and read their metadata.

    Mirrors run_cell.py output: cell_<...>-<dt>.pkl + sibling _manifest.json. No model
    needed — PruneScores are plain dict-of-tensors.
    """
    def fake_ps(val: float) -> dict:
        return {"taskA": {"algoX": {"blocks.0.hook_q_input": t.full((2, 3), val)}}}

    cells = [(42, 1.0, "01-01-2026_00-00-00"), (99, 2.0, "01-01-2026_00-00-01")]
    for seed, val, dt in cells:
        stem = f"cell_taskA_algoX_resample_seed{seed}-{dt}"
        t.save(fake_ps(val), tmp_path / f"{stem}.pkl")
        with open(tmp_path / f"{stem}_manifest.json", "w") as f:
            json.dump({"cell": {"seed": seed, "ablation": "resample"}}, f)

    loaded = load_many_caches(str(tmp_path), "cell_")
    assert len(loaded) == 2
    assert sorted(m["cell"]["seed"] for m, _ in loaded) == [42, 99]
    # PruneScores round-trip, matched to their manifest seed.
    for manifest, tps in loaded:
        v = tps["taskA"]["algoX"]["blocks.0.hook_q_input"]
        expected = 1.0 if manifest["cell"]["seed"] == 42 else 2.0
        assert t.allclose(v, t.full((2, 3), expected))

    # extract_metric pulls the (edge_count, score) series from the nested measurements.
    measurements = {
        AblationType.RESAMPLE: {"Clean KL Div": {"taskA": {"algoX": [(0, 0.0), (10, 0.5)]}}}
    }
    series = extract_metric(
        measurements, AblationType.RESAMPLE, "Clean KL Div", "taskA", "algoX"
    )
    assert series == [(0, 0.0), (10, 0.5)]


def test_latest_per_cell():
    """Dedup keeps the newest file per cell identity; distinct cells are all kept."""
    def cell(task, algo, abl, saved, tag):
        manifest = {
            "cell": {"task": task, "algo": algo, "ablation": abl, "seed": 42, "null_test": False},
            "saved_at": saved,
        }
        return (manifest, {"tps": tag})

    loaded = [
        cell("ioi", "ep", "zero", "2026-06-08T10:00:00", "old"),
        cell("ioi", "ep", "zero", "2026-06-08T12:00:00", "new"),  # newer -> wins
        cell("ioi", "ep", "resample", "2026-06-08T10:00:00", "other-abl"),
        cell("gt", "acdc", "zero", "2026-06-08T10:00:00", "other-cell"),
    ]
    kept = latest_per_cell(loaded)
    assert len(kept) == 3  # the two ioi/ep/zero collapse to one
    tags = {tps["tps"] for _, tps in kept}
    assert "new" in tags and "old" not in tags
    assert {"new", "other-abl", "other-cell"} == tags
