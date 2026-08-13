"""Tier B of the Edge Pruning port fidelity study: same-mask function equivalence.

Given the same circuit, the port and ``princeton-nlp/Edge-Pruning`` @ ``a918efd``
must compute the same function. The heavy tests are marked slow and need one GPU;
the map's algebraic properties are checked without either.

Skipped automatically when ``third_party/edge_pruning`` is absent.
"""

import sys
from pathlib import Path

import pytest
import torch as t

REPO = Path(__file__).resolve().parents[2]
_EP_ROOT = REPO / "third_party" / "edge_pruning"
for _extra in (REPO, _EP_ROOT / "src" / "modeling", _EP_ROOT / "src" / "prune"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from experiments.port_comparison.ep_edge_map import (  # noqa: E402
    ac_to_orig,
    expected_shapes,
    n_sources,
    n_writers,
    node_ac_to_orig,
    node_orig_to_ac,
    orig_key_attn_write,
    orig_key_mlp_write,
    orig_to_ac,
    total_edges_autocircuit,
    total_edges_original,
)

N_LAYERS, N_HEADS = 12, 12
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
N_PROMPTS = 8
# Very low densities are excluded: the LayerNorm of a near-empty residual sum
# dominates, which measures float conditioning rather than port fidelity.
DENSITIES = (0.25, 0.5, 0.9)

# The measured floor from the full sweep (`raw` arm, 64 prompts x 40 masks x 5
# densities): worst max|delta logprob| = 3.7e-04. Deviations are judged against this,
# not against zero.
FLOOR = 1e-3


# --- map algebra: no model, no third_party ----------------------------------


def test_edge_counts_reconcile():
    """AutoCircuit's edge set is the reference's plus one embed edge per reader."""
    ac = total_edges_autocircuit(N_LAYERS, N_HEADS)
    ref = total_edges_original(N_LAYERS, N_HEADS)
    n_readers = N_LAYERS * (3 * N_HEADS + 1) + 1
    assert ac == 32491
    assert ref == 32046
    assert ac - ref == n_readers == 445


def _random_orig(seed: int = 0) -> dict:
    gen = t.Generator().manual_seed(seed)
    total = n_writers(N_LAYERS, N_HEADS)
    orig = {}
    for block in range(N_LAYERS):
        for letter in ("q", "k", "v"):
            orig[f"h.{block}.{letter}_read_log_alphas"] = t.randn(
                (total, N_HEADS), generator=gen
            )
        orig[f"h.{block}.mlp_read_log_alphas"] = t.randn((total,), generator=gen)
    orig["final_read_log_alphas"] = t.randn((total,), generator=gen)
    return orig


def test_edge_map_round_trips():
    """orig -> ac -> orig is the identity on every causally legal entry."""
    orig = _random_orig()
    ac = orig_to_ac(orig, N_LAYERS, N_HEADS, embed_fill=0.0)
    back = ac_to_orig(ac, N_LAYERS, N_HEADS, illegal_fill=0.0)

    for ac_key, spec in expected_shapes(N_LAYERS, N_HEADS).items():
        legal = spec["legal"]
        assert t.equal(back[spec["orig_key"]][:legal], orig[spec["orig_key"]][:legal]), (
            ac_key
        )


def test_edge_map_produces_autocircuit_shapes():
    orig = _random_orig()
    ac = orig_to_ac(orig, N_LAYERS, N_HEADS, embed_fill=1.0)
    for ac_key, spec in expected_shapes(N_LAYERS, N_HEADS).items():
        assert tuple(ac[ac_key].shape) == spec["ac"], ac_key
    assert sum(v.numel() for v in ac.values()) == 32491


def test_embed_column_is_the_only_unmapped_entry():
    """Every AutoCircuit entry except the embed column comes from the reference."""
    orig = _random_orig()
    ac = orig_to_ac(orig, N_LAYERS, N_HEADS, embed_fill=float("nan"))
    n_nan = sum(int(v.isnan().sum()) for v in ac.values())
    assert n_nan == 445


def test_node_map_round_trips():
    gen = t.Generator().manual_seed(1)
    orig = {}
    for block in range(N_LAYERS):
        orig[orig_key_attn_write(block)] = t.randn((N_HEADS,), generator=gen)
        orig[orig_key_mlp_write(block)] = t.randn((1,), generator=gen)
    vec = node_orig_to_ac(orig, N_LAYERS, N_HEADS, embed_fill=0.0)
    assert vec.numel() == n_sources(N_LAYERS, N_HEADS) == 157
    back = node_ac_to_orig(vec, N_LAYERS, N_HEADS)
    for key, value in orig.items():
        assert t.equal(back[key], value), key


# --- forward equivalence: needs a GPU and the reference ----------------------


def _require_reference():
    if not _EP_ROOT.is_dir():
        pytest.skip(f"reference implementation not found at {_EP_ROOT}")


@pytest.fixture(scope="module")
def bench():
    _require_reference()
    fe = pytest.importorskip("experiments.port_comparison.ep_forward_equivalence")

    from auto_circuit.experiment_utils import load_tl_model
    from auto_circuit.utils.graph_utils import patchable_model

    tl = load_tl_model(fe.MODEL, DEVICE)
    ac_model = patchable_model(
        tl, factorized=True, slice_output="last_seq", separate_qkv=True, device=DEVICE
    )
    orig_model = fe.build_original(DEVICE)
    orig_full = fe.build_original(DEVICE)
    clean, corrupt = fe.raw_tokens(N_PROMPTS, DEVICE)
    return fe, tl, ac_model, orig_model, orig_full, clean, corrupt


@pytest.mark.slow
def test_map_matches_the_live_models(bench):
    fe, tl, ac_model, _, _, _, _ = bench
    assert ac_model.n_edges == total_edges_autocircuit(N_LAYERS, N_HEADS)
    assert len(ac_model.srcs) == n_sources(N_LAYERS, N_HEADS)
    for ac_key, spec in expected_shapes(N_LAYERS, N_HEADS).items():
        assert tuple(ac_model.patch_masks[ac_key].shape) == spec["ac"], ac_key
    # AutoCircuit's src ordering must be embed, then per block [heads..., mlp],
    # since the map relies on `src_idx i` <-> `writer i - 1` with no remapping.
    names = [s.name for s in sorted(ac_model.srcs, key=lambda s: s.src_idx)]
    assert names[1:14] == [f"A0.{h}" for h in range(12)] + ["MLP 0"]
    assert names[-1] == "MLP 11"


@pytest.mark.slow
def test_full_density_agrees(bench):
    """With nothing ablated both are plain GPT-2, so this is the float floor."""
    fe, _, ac_model, orig_model, orig_full, clean, corrupt = bench
    from auto_circuit.types import AblationType
    from auto_circuit.utils.ablation_activations import src_ablations

    keep = fe.all_keep_masks(ac_model)
    fe.install_original_mask(orig_model, keep, N_LAYERS, N_HEADS)
    ablations = src_ablations(ac_model, corrupt, AblationType.RESAMPLE)
    corr_x = fe.original_writer_states(orig_full, corrupt)

    dev = fe.deviation(
        fe.ac_logits(ac_model, clean, keep, ablations),
        fe.orig_logits(orig_model, clean, corr_x, -1),
    )
    assert dev["max_abs"] < FLOOR
    assert dev["argmax_agree"] == 1.0


@pytest.mark.slow
@pytest.mark.parametrize("density", DENSITIES)
def test_masked_forward_agrees(bench, density):
    """The core Tier B claim: same circuit, same function, under resample."""
    fe, _, ac_model, orig_model, orig_full, clean, corrupt = bench
    from auto_circuit.types import AblationType
    from auto_circuit.utils.ablation_activations import src_ablations

    ablations = src_ablations(ac_model, corrupt, AblationType.RESAMPLE)
    corr_x = fe.original_writer_states(orig_full, corrupt)
    gen = t.Generator().manual_seed(int(density * 100))

    for _ in range(3):
        keep = fe.random_keep_masks(ac_model, density, gen, N_HEADS)
        fe.install_original_mask(orig_model, keep, N_LAYERS, N_HEADS)
        dev = fe.deviation(
            fe.ac_logits(ac_model, clean, keep, ablations),
            fe.orig_logits(orig_model, clean, corr_x, -1),
        )
        assert dev["max_abs"] < FLOOR, (density, dev)


@pytest.mark.slow
@pytest.mark.parametrize("ablation_name", ["RESAMPLE", "BATCH_TOKENWISE_MEAN"])
def test_no_attention_bias_signature(bench, ablation_name):
    """The pre-registered `b_O` prediction, and its generalisation.

    The reference folds ``b_O / n_heads`` into each head inside the masked path;
    AutoCircuit adds ``b_O`` once outside it. Under *zero* ablation that difference
    was the entire divergence of the DiscoGP study, and it showed up as head-source
    edges disagreeing far more than embedding- or MLP-source edges. Under
    *resample* both branches carry the bias share, so it must cancel: head-source
    probes must be indistinguishable from the others.

    Parametrised over the mean ablation too, because the property that actually
    discriminates is the **replacement value**, not the ablation's name: whether the
    substituted activation is itself computed through the same per-head decomposition
    and so carries its own ``b_O / H`` share. Every AutoCircuit ablation except
    ``ZERO`` substitutes a mean or a sample of genuine activations, so every one of
    them should be inert here, and only ``ZERO`` should bite.
    """
    from auto_circuit.types import AblationType

    fe, _, ac_model, orig_model, orig_full, clean, corrupt = bench
    probes = fe.single_edge_probes(
        ac_model, orig_model, orig_full, clean, corrupt, N_LAYERS, N_HEADS,
        ablation_type=getattr(AblationType, ablation_name),
    )
    assert set(probes) == {
        "embed_src", "head_src_layer0", "mlp_src_layer0", "head_src_layer10"
    }
    for name, value in probes.items():
        assert value < FLOOR, (name, value)

    non_head = max(probes["embed_src"], probes["mlp_src_layer0"])
    for name in ("head_src_layer0", "head_src_layer10"):
        assert probes[name] < 3 * non_head, (
            f"{name} shows a head-specific excess: {probes[name]} vs {non_head}"
        )


# --- the input-convention factor, isolated ----------------------------------


@pytest.mark.slow
def test_bos_accounts_for_the_data_path_gap(bench):
    """Prepending BOS on the AutoCircuit side alone, with no padding involved,
    reproduces the gap seen when both sides run their own dataloaders. So the
    remaining Tier B divergence is the BOS convention, not the masking."""
    fe, _, ac_model, orig_model, orig_full, clean, corrupt = bench
    _, summary = fe.run_arm(
        ac_model, orig_model, orig_full, clean, corrupt,
        (0.5,), 2, 0, N_LAYERS, N_HEADS, ac_prepend_bos=True,
    )
    assert summary["full_density_max_abs"] > 100 * FLOOR


@pytest.mark.slow
def test_left_padding_is_batch_invariant_only_when_tl_is_told_about_bos():
    """A defect in AutoCircuit's data path, found while building this tier.

    ``data.py`` tokenises with its own ``prepend_bos``, then calls ``model(tokens)``
    without passing it on. ``HookedTransformer`` resolves ``prepend_bos`` from
    ``cfg.default_prepend_bos`` (True) when it rebuilds the attention mask
    (``HookedTransformer.py:286-294``), and ``get_attention_mask`` un-masks the last
    leading pad on the assumption that it is a prepended BOS
    (``transformer_lens/utils.py:976-978``). GPT-2's pad token *is* its BOS token, so
    with ``prepend_bos=False`` one genuine pad per padded row gets attended.

    The 2x2 below isolates it: exactly one cell breaks, and forcing
    ``cfg.default_prepend_bos`` to match the tokenisation repairs it. That the repair
    works is what makes this a call-site defect in AutoCircuit rather than a
    TransformerLens bug -- TL behaves correctly given what it is told.

    Severity is latent: ``tasks.py`` passes ``prepend_bos=has_tokenizer``, so no
    shipped task reaches this path. It bites only explicit ``prepend_bos=False``
    callers, needs padding to be present, and touches at most one token per row.
    """
    import json

    from auto_circuit.data import load_datasets_from_json
    from auto_circuit.experiment_utils import load_tl_model
    from auto_circuit.utils.misc import repo_path_to_abs_path

    path = repo_path_to_abs_path("datasets/port_fidelity/ioi_frozen.json")
    if not path.exists():
        pytest.skip("frozen split not built yet")

    tl = load_tl_model("gpt2", DEVICE)
    rows = json.loads(path.read_text())["prompts"][:16]
    original_default = tl.cfg.default_prepend_bos

    def batched_vs_solo(prepend_bos: bool, default_prepend_bos: bool) -> float:
        tl.cfg.default_prepend_bos = default_prepend_bos
        loader, _ = load_datasets_from_json(
            model=tl, path=path, device=DEVICE, batch_size=16,
            train_test_size=(16, 16), shuffle=False, random_seed=0,
            prepend_bos=prepend_bos,
        )
        clean = next(iter(loader)).clean
        # Each row's real content is the suffix of known length. Stripping tokens
        # equal to the pad id would also strip the BOS, since they are the same id,
        # and would silently compare BOS against no-BOS instead.
        lens = [
            len(tl.tokenizer(r["clean"])["input_ids"]) + (1 if prepend_bos else 0)
            for r in rows
        ]
        assert any(n < clean.shape[1] for n in lens), "need ragged prompts to test this"
        with t.no_grad():
            batched = t.log_softmax(tl(clean)[:, -1].float(), dim=-1)
            solo = t.stack([
                t.log_softmax(
                    tl(clean[i, -lens[i]:].unsqueeze(0))[0, -1].float(), dim=-1
                )
                for i in range(clean.shape[0])
            ])
        return float((batched - solo).abs().max())

    try:
        results = {
            (tokenised, default): batched_vs_solo(tokenised, default)
            for tokenised in (True, False)
            for default in (True, False)
        }
    finally:
        tl.cfg.default_prepend_bos = original_default

    broken = (False, True)  # tokens have no BOS; TL assumes one
    assert results[broken] > 1.0, results
    for cell, value in results.items():
        if cell != broken:
            assert value < 1e-3, (cell, results)
