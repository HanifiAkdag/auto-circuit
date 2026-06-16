#%%
import json

import numpy as np
import torch as t

from auto_circuit.metrics.analysis_utils import (
    extract_metric,
    jaccard,
    latest_per_cell,
    load_many_caches,
    overlap_matrix,
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

    m = overlap_matrix([ps_a, ps_a, ps_b], k=3, edges=edges)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 1.0)   # self-overlap
    assert m[0, 1] == 1.0                  # identical runs
    assert m[0, 2] == 0.0                  # disjoint runs
    assert np.allclose(m, m.T)             # symmetric


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
