"""Tier B of the DiscoGP port fidelity study: same-mask function equivalence.

A fast regression version of ``experiments/port_comparison/forward_equivalence.py``.
The script produces the numbers reported in the thesis; these tests pin the
conclusions so a later refactor cannot silently break them.

The load-bearing claim is the ``b_O``-zeroed arm: with the one known
decomposition difference removed, the two implementations agree to float
precision under masking. That is what establishes the edge bijection and the
mask-convention inversion as correct.

Marked ``slow`` (needs GPT-2 and a forward pass per mask); run with ``--runslow``.
"""

import sys
from pathlib import Path

import pytest
import torch as t

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO, REPO / "third_party" / "disco_gp"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytestmark = pytest.mark.slow

fe = pytest.importorskip("experiments.port_comparison.forward_equivalence")
from experiments.port_comparison.edge_map import (  # noqa: E402
    ac_to_orig,
    expected_shapes,
    orig_to_ac,
    total_edges,
)

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
N_PROMPTS = 8
# Densities where the LayerNorm of a near-zero residual sum does not dominate;
# see the density-0.05/0.1 rows of the full sweep.
DENSITIES = (0.25, 0.5, 0.9)


@pytest.fixture(scope="module")
def bench():
    """(ac_model, orig_model, tokens, ablations) with `b_O` removed from both."""
    from auto_circuit.experiment_utils import load_tl_model
    from auto_circuit.types import AblationType
    from auto_circuit.utils.ablation_activations import src_ablations
    from auto_circuit.utils.graph_utils import patchable_model

    if not (REPO / "third_party" / "disco_gp").is_dir():
        pytest.skip("vendored original not found")

    tl_model = load_tl_model(fe.MODEL, DEVICE)
    orig_model = fe.build_original(DEVICE)
    fe.zero_attn_out_bias(tl_model, orig_model)
    tokens = fe.load_tokens(N_PROMPTS, DEVICE)
    ac_model = patchable_model(
        tl_model, factorized=True, slice_output="last_seq",
        separate_qkv=True, device=DEVICE,
    )
    ablations = src_ablations(ac_model, tokens, AblationType.ZERO)
    return ac_model, orig_model, tokens, ablations


# --- B0: the bijection -------------------------------------------------------


def test_edge_map_round_trips():
    n_layers, n_heads = 12, 12
    g = t.Generator().manual_seed(0)
    ac = {
        k: t.randn(spec["ac"], generator=g)
        for k, spec in expected_shapes(n_layers, n_heads).items()
    }
    back = orig_to_ac(ac_to_orig(ac, n_layers, n_heads), n_layers, n_heads)
    assert set(back) == set(ac)
    for k in ac:
        assert t.equal(back[k], ac[k])


def test_edge_map_covers_every_edge():
    assert total_edges(12, 12) == 32491


def test_edge_map_shapes_match_the_model(bench):
    ac_model, _, _, _ = bench
    n_layers, n_heads = ac_model.cfg.n_layers, ac_model.cfg.n_heads
    assert total_edges(n_layers, n_heads) == ac_model.n_edges == 32491
    wrappers = {w.module_name: w for w in ac_model.ordered_dest_wrappers}
    shapes = expected_shapes(n_layers, n_heads)
    assert set(shapes) == set(wrappers)
    for key, spec in shapes.items():
        assert tuple(wrappers[key].patch_mask.shape) == spec["ac"], key


# --- B1: the numerical floor -------------------------------------------------


def test_full_density_agrees(bench):
    """Both graphs are mathematically identical at full density."""
    ac_model, orig_model, tokens, ablations = bench
    n_layers, n_heads = ac_model.cfg.n_layers, ac_model.cfg.n_heads

    keep = {w.module_name: t.ones_like(w.patch_mask) for w in ac_model.ordered_dest_wrappers}
    fe.install_original_mask(orig_model, ac_to_orig(keep, n_layers, n_heads))
    dev = fe.deviation(
        fe.ac_logits(ac_model, tokens, keep, ablations), fe.orig_logits(orig_model, tokens)
    )
    assert dev["max_abs"] < 1e-3, dev
    assert dev["argmax_agree"] == 1.0


# --- B2: masked forwards -----------------------------------------------------


@pytest.mark.parametrize("density", DENSITIES)
def test_masked_forward_agrees_without_b_O(bench, density):
    """The decisive test: same circuit in, same logits out.

    This simultaneously validates the edge bijection (a permutation error inside
    a shape could not survive it), the `AC_mask = 1 - disco_mask` inversion, and
    the equivalence of AutoCircuit's zero-ablation with the original's dropping
    of terms from the node sum.
    """
    ac_model, orig_model, tokens, ablations = bench
    n_layers, n_heads = ac_model.cfg.n_layers, ac_model.cfg.n_heads
    gen = t.Generator(device=DEVICE).manual_seed(int(density * 1000))

    for _ in range(3):
        keep = fe.random_keep_masks(ac_model, density, gen)
        fe.install_original_mask(orig_model, ac_to_orig(keep, n_layers, n_heads))
        dev = fe.deviation(
            fe.ac_logits(ac_model, tokens, keep, ablations), fe.orig_logits(orig_model, tokens)
        )
        assert dev["max_abs"] < 1e-2, (density, dev)
        assert dev["argmax_agree"] == 1.0, (density, dev)


def test_single_edge_probes_cover_each_source_kind(bench):
    """Ablating index 0 of a mask only ever hits the embedding source, which
    carries no `b_O`. Probes must be selected by source kind or they silently
    pass while testing nothing."""
    ac_model, orig_model, tokens, ablations = bench
    n_layers, n_heads = ac_model.cfg.n_layers, ac_model.cfg.n_heads
    mlp_dest = f"blocks.{n_layers - 1}.hook_mlp_in"

    base = {w.module_name: t.ones_like(w.patch_mask) for w in ac_model.ordered_dest_wrappers}
    for src_idx in (0, 1, n_heads + 1):  # embed, a head, an MLP
        keep = {k: v.clone() for k, v in base.items()}
        keep[mlp_dest][src_idx] = 0.0
        fe.install_original_mask(orig_model, ac_to_orig(keep, n_layers, n_heads))
        dev = fe.deviation(
            fe.ac_logits(ac_model, tokens, keep, ablations), fe.orig_logits(orig_model, tokens)
        )
        assert dev["max_abs"] < 1e-3, (src_idx, dev)


# --- B3: the known decomposition difference ----------------------------------


def test_head_node_gap_is_exactly_b_O_over_n_heads():
    """With `b_O` intact the two decompositions differ, by a quantity that is
    exactly the original's per-head share of the attention output bias
    (`modules/attention.py:150`). Not a port defect, but not negligible either:
    on GPT-2 small the layer-11 term is ~20 logits.
    """
    from auto_circuit.experiment_utils import load_tl_model

    if not (REPO / "third_party" / "disco_gp").is_dir():
        pytest.skip("vendored original not found")

    tl_model = load_tl_model(fe.MODEL, DEVICE)
    orig_model = fe.build_original(DEVICE)
    tokens = fe.load_tokens(N_PROMPTS, DEVICE)

    report = fe.compare_node_decomposition(tl_model, orig_model, tokens)

    assert report["embed_node_gap"] < 1e-3
    assert report["mlp_node_gap"] < 1e-3
    # The head gap is large, and is accounted for entirely by b_O / n_heads.
    assert report["head_node_gap"] > 1.0
    assert report["head_node_gap_minus_b_O_over_n_heads"] < 1e-2
