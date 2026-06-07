#%%
import numpy as np

from auto_circuit.metrics.analysis_utils import (
    jaccard,
    overlap_matrix,
    top_k_edge_set,
)
from auto_circuit.model_utils.micro_model_utils import MicroModel
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
