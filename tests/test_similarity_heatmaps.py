"""Tests for the cross-cell circuit-similarity heatmaps (faithfulness spine, #4)."""
import pytest

from auto_circuit.metrics.analysis_utils import top_k_edge_set  # noqa: E402
from auto_circuit.model_utils.micro_model_utils import MicroModel  # noqa: E402
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT  # noqa: E402
from auto_circuit.utils.graph_utils import patchable_model  # noqa: E402
from tests.conftest import DEVICE  # noqa: E402

from experiments.faithfulness_cells import (  # noqa: E402
    ANCHOR_ALGOS,
    DISCOVERY_ABLATIONS,
    SWEPT_ALGOS,
)
from experiments.run_cell import _ALGO_MAP  # noqa: E402
from experiments.similarity_heatmaps import (  # noqa: E402
    _jaccard_matrix,
    collect_labeled_scores,
    save_similarity_heatmaps,
)


@pytest.fixture(scope="module")
def patch_micro():
    # Fresh copy: wrapping mutates the module tree, don't touch the shared fixture.
    return patchable_model(
        model=MicroModel(n_layers=2).to(DEVICE),
        factorized=True,
        slice_output=None,
        seq_len=None,
        device=DEVICE,
    )


def _scores_for(model, edges, value=1.0):
    """PruneScores with `value` on exactly `edges` and 0 elsewhere."""
    ps = model.new_prune_scores()
    for e in edges:
        ps[e.dest.module_name][e.patch_idx] = value
    return ps


def test_top_k_edge_set_exclude_zero_caps_at_nonzero_count(patch_micro):
    model = patch_micro
    edges = sorted(model.edges, key=lambda e: e.name)
    nonzero = edges[:3]
    ps = _scores_for(model, nonzero)
    k = model.n_edges  # k-th largest |score| is 0 -> threshold 0

    # Existing behavior: threshold 0 admits every edge.
    assert len(top_k_edge_set(ps, k, model.edges)) == model.n_edges
    # exclude_zero: the circuit stops growing at its nonzero-score count.
    assert top_k_edge_set(ps, k, model.edges, exclude_zero=True) == set(nonzero)
    # Below the nonzero count, exclude_zero changes nothing.
    assert len(top_k_edge_set(ps, 2, model.edges, exclude_zero=True)) == len(
        top_k_edge_set(ps, 2, model.edges)
    )


def test_jaccard_matrix_properties():
    a, b, empty = {1, 2, 3}, {3, 4, 5}, set()
    mat = _jaccard_matrix([a, a, b, empty])
    n = 4
    for i in range(n):
        assert mat[i][i] == 1.0
        for j in range(n):
            assert mat[i][j] == mat[j][i]
    assert mat[0][1] == 1.0  # identical sets
    assert mat[0][2] == pytest.approx(1 / 5)  # {3} over {1,2,3,4,5}
    assert mat[0][3] == 0.0  # disjoint from empty


def test_collect_labeled_scores_ordering_and_anchor_dedup():
    ep, acdc = _ALGO_MAP["ep"].key, _ALGO_MAP["acdc"].key
    gt, rnd = _ALGO_MAP["gt"].key, _ALGO_MAP["random"].key
    s_ep_zero, s_ep_res, s_acdc_zero = object(), object(), object()
    s_gt, s_rnd = object(), object()
    task = "IOI"
    groups = {
        "zero": {task: {acdc: s_acdc_zero, ep: s_ep_zero, gt: s_gt, rnd: s_rnd}},
        "resample": {task: {ep: s_ep_res, gt: s_gt, rnd: s_rnd}},
        "mean": {task: {gt: s_gt, rnd: s_rnd}},
    }

    labeled = collect_labeled_scores(
        groups,
        DISCOVERY_ABLATIONS,
        algo_order=[_ALGO_MAP[a].key for a in SWEPT_ALGOS],
        anchor_algo_keys=[_ALGO_MAP[a].key for a in ANCHOR_ALGOS],
    )[task]

    ep_short = PRUNE_ALGO_DICT[ep].short_name
    acdc_short = PRUNE_ALGO_DICT[acdc].short_name
    gt_short = PRUNE_ALGO_DICT[gt].short_name
    rnd_short = PRUNE_ALGO_DICT[rnd].short_name

    # Algo-major order (SWEPT_ALGOS puts EP before ACDC), ablations in
    # DISCOVERY_ABLATIONS order, missing cells absent, anchors deduped and last.
    assert [label for label, _ in labeled] == [
        f"{ep_short} [zero]", f"{ep_short} [resample]", f"{acdc_short} [zero]",
        gt_short, rnd_short,
    ]
    scores = dict(labeled)
    assert scores[f"{ep_short} [zero]"] is s_ep_zero
    assert scores[f"{ep_short} [resample]"] is s_ep_res
    assert scores[gt_short] is s_gt and scores[rnd_short] is s_rnd


def test_save_similarity_heatmaps_writes_outputs(patch_micro, tmp_path):
    model = patch_micro
    edges = sorted(model.edges, key=lambda e: e.name)
    labeled = [
        ("A [zero]", _scores_for(model, edges[:3])),
        ("B [zero]", _scores_for(model, edges[1:4])),
        ("GT", _scores_for(model, edges[:2])),
    ]
    huge = model.n_edges + 1000  # over budget -> silently skipped
    written = save_similarity_heatmaps(labeled, [3, 2, huge], model, "Micro", tmp_path)

    assert (tmp_path / "jaccard_k2.pdf").exists()
    assert (tmp_path / "jaccard_k3.pdf").exists()
    assert (tmp_path / "jaccard_k2.png").exists()
    assert (tmp_path / "jaccard_k3.png").exists()
    assert not (tmp_path / f"jaccard_k{huge}.pdf").exists()
    assert (tmp_path / "jaccard.html").exists()
    assert (tmp_path / "jaccard.csv").exists()
    assert set(written) == {
        tmp_path / "jaccard_k2.pdf", tmp_path / "jaccard_k3.pdf",
        tmp_path / "jaccard_k2.png", tmp_path / "jaccard_k3.png",
        tmp_path / "jaccard.html", tmp_path / "jaccard.csv",
    }

    import csv as csv_mod
    with open(tmp_path / "jaccard.csv") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 2 * 3 * 3  # 2 budgets x 3x3 matrix
    at3 = {
        (r["label_a"], r["label_b"]): float(r["jaccard"])
        for r in rows if r["edge_count"] == "3"
    }
    # A={e0,e1,e2}, B={e1,e2,e3} at k=3 -> |∩|=2, |∪|=4.
    assert at3[("A [zero]", "B [zero]")] == pytest.approx(0.5)
    # GT has only 2 nonzero scores: capped by exclude_zero, subset of A.
    gt_rows = [r for r in rows if r["label_a"] == "GT" and r["edge_count"] == "3"]
    assert all(r["size_a"] == "2" for r in gt_rows)
    assert at3[("GT", "A [zero]")] == pytest.approx(2 / 3)


def test_save_similarity_heatmaps_degenerate_inputs(patch_micro, tmp_path):
    model = patch_micro
    one = [("A [zero]", model.new_prune_scores())]
    assert save_similarity_heatmaps(one, [2], model, "Micro", tmp_path) == []
    two = one + [("B [zero]", model.new_prune_scores())]
    assert save_similarity_heatmaps(two, [model.n_edges + 1], model, "Micro", tmp_path) == []
