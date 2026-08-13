"""Tier A of the Edge Pruning port fidelity study: component equivalence.

Each test pairs one component of the port against the corresponding component of
the original implementation (``princeton-nlp/Edge-Pruning`` @ ``a918efd``, expected
at ``third_party/edge_pruning``) and asserts they agree exactly.

Where a component cannot be isolated as a function, the original's *real* code is
driven on a tiny randomly-initialised model rather than transcribed into the test:
a transcription would only check that the test author copied correctly.

These tests close a coverage gap that exists regardless of the comparison —
``edge_pruning.py`` had no tests at all, and every "faithful" row of
``docs/related_work/edge_pruning.md`` rested on reading source.

Skipped automatically when ``third_party/edge_pruning`` is absent.
"""

import json
import sys
from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as F

from auto_circuit.prune_algos.edge_pruning import (
    _lagrangian,
    _sample_mask,
    _soft_mask,
    _target_sparsity,
    edge_pruning_prune_scores,
)
from auto_circuit.utils.l0_utils import (
    FACTOR,
    LIMIT_LEFT,
    LIMIT_RIGHT,
    TEMPERATURE,
    cdf_stretched_concrete,
    deterministic_z,
    get_linear_schedule_with_warmup,
)
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.tensor_ops import multibatch_kl_div, sample_hard_concrete

_ORIG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "edge_pruning"


def _import_original():
    """Import the reference implementation, or skip the module if unavailable."""
    if not _ORIG_ROOT.is_dir():
        pytest.skip(f"reference implementation not found at {_ORIG_ROOT}")
    for extra in (_ORIG_ROOT / "src" / "modeling", _ORIG_ROOT / "src" / "prune"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    import l0 as orig_l0
    import modeling_fpt2 as orig_modeling
    from fpt2_ioi import FPT2InfoTrainer, get_optimizers

    return {
        "l0": orig_l0,
        "modeling": orig_modeling,
        "trainer_cls": FPT2InfoTrainer,
        "get_optimizers": get_optimizers,
    }


@pytest.fixture(scope="module")
def orig():
    return _import_original()


@pytest.fixture(scope="module")
def tiny(orig):
    """A 2-layer randomly-initialised FPT2, so the reference's real forward path can
    be exercised on CPU in milliseconds instead of downloading GPT-2."""
    from transformers import GPT2Config

    cfg = GPT2Config(
        n_layer=2, n_head=2, n_embd=32, vocab_size=64, n_positions=16,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    t.manual_seed(0)
    model = orig["modeling"].FPT2LMHeadModel(cfg, with_embedding_nodes=False)
    model.reset_all_log_alphas()
    model.eval()
    return model


# --- A1: the stretched-concrete CDF -----------------------------------------


@pytest.mark.parametrize("x", [0.0])
@pytest.mark.parametrize(
    "log_alpha",
    [
        t.tensor([10.0, -10.0, 0.0, 5.0, -5.0]),
        t.linspace(-40.0, 40.0, 501),  # spans both EPS clamp boundaries
        t.randn(97, generator=t.Generator().manual_seed(3)) * 8,
    ],
)
def test_cdf_stretched_concrete_bitwise(orig, x, log_alpha):
    ours = cdf_stretched_concrete(x, log_alpha)
    theirs = orig["l0"].cdf_stretched_concrete(x, log_alpha)
    assert t.equal(ours, theirs)


def test_l0_constants_match(orig):
    assert (LIMIT_LEFT, LIMIT_RIGHT) == (orig["l0"].LIMIT_LEFT, orig["l0"].LIMIT_RIGHT)
    assert TEMPERATURE == orig["l0"].TEMPERATURE
    assert FACTOR == orig["l0"].FACTOR


# --- A2: deterministic (evaluation-time) discretisation ----------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [0.5, 3.0, 12.0])
def test_deterministic_z_bitwise(orig, seed, scale):
    log_alpha = t.randn(512, generator=t.Generator().manual_seed(seed)) * scale
    ours = deterministic_z(log_alpha.clone())
    theirs = orig["l0"].deterministic_z_from_log_alpha(log_alpha.clone())
    assert t.equal(ours, theirs)


def test_deterministic_z_bitwise_multidim(orig):
    log_alpha = t.randn(11, 7, generator=t.Generator().manual_seed(9)) * 4
    ours = deterministic_z(log_alpha.clone())
    theirs = orig["l0"].deterministic_z_from_log_alpha(log_alpha.clone())
    assert ours.shape == log_alpha.shape
    assert t.equal(ours, theirs)


def test_deterministic_z_soft_value_is_sigmoid_1p2_log_alpha(orig):
    """The kept entries are ``σ(log_alpha · FACTOR / TEMPERATURE)`` = ``σ(1.2·α)``,
    not the standard hard-concrete deterministic value. This is the quantity the
    port exports as PruneScores, so it is asserted rather than assumed."""
    log_alpha = t.tensor([8.0, 9.0, 10.0])  # all kept: expected zeros rounds to 0
    ours = deterministic_z(log_alpha.clone())
    assert t.allclose(ours, t.sigmoid(log_alpha * 1.2))
    assert t.equal(ours, orig["l0"].deterministic_z_from_log_alpha(log_alpha.clone()))


def test_port_num_zeros_guard_is_inert(orig):
    """The port adds ``k = min(num_zeros, numel)`` where the reference has bare
    ``num_zeros``. The clamp in ``cdf_stretched_concrete`` bounds the expected
    non-zero count away from 0, so the guard can never bind; assert that, so it
    stays a guard rather than a silent divergence."""
    for scale in (0.1, 1.0, 10.0, 100.0):
        for seed in range(5):
            la = t.randn(256, generator=t.Generator().manual_seed(seed)) * scale
            size = la.numel()
            expected_nonzero = t.sum(1 - cdf_stretched_concrete(0, la))
            num_zeros = int(t.round(size - expected_nonzero).item())
            assert 0 <= num_zeros <= size
            assert t.equal(
                deterministic_z(la.clone()),
                orig["l0"].deterministic_z_from_log_alpha(la.clone()),
            )


def test_deterministic_z_zero_dim_agrees_and_the_scalar_branch_is_dead(orig):
    """Both implementations guard a ``soft_mask.ndim == 0`` case that differs (the
    reference returns an integer ``tensor(0)``, the port a float). The guard is
    unreachable in both: ``.reshape(-1)`` runs first, so a 0-dim ``log_alpha``
    already has ndim 1 by the time the branch is tested. Assert the reachable
    behaviour agrees, and pin the reason the divergence cannot bite."""
    la = t.tensor(-20.0)
    ours = deterministic_z(la.clone())
    theirs = orig["l0"].deterministic_z_from_log_alpha(la.clone())

    assert t.equal(ours, theirs)
    assert ours.dtype == theirs.dtype == t.float32
    assert ours.shape == theirs.shape == la.shape

    # the branch is dead because reshape(-1) precedes it
    assert t.sigmoid(la / TEMPERATURE * FACTOR).reshape(-1).ndim == 1


# --- A3: hard-concrete sampling ---------------------------------------------


@pytest.mark.parametrize("shape", [(157,), (32, 12), (4, 8, 16)])
def test_hard_concrete_sampling_matches_under_shared_u(orig, shape):
    """The two draw ``u`` differently (see the port-fidelity spec, difference 2):
    the reference draws one sample per forward on CPU clamped to (1e-6, 1-1e-6),
    the port draws one per batch element on device clamped to (1e-4, 0.9999).
    Feeding a shared ``u`` isolates the transform from the RNG divergence."""
    gen = t.Generator().manual_seed(11)
    log_alpha = t.randn(shape, generator=gen) * 3
    u = t.rand(shape, generator=gen).clamp(1e-4, 0.9999)

    # The port's transform, with its own `u` substituted in.
    s = t.sigmoid((u.log() - (1 - u).log() + log_alpha) / TEMPERATURE)
    ours = (s * (LIMIT_RIGHT - LIMIT_LEFT) + LIMIT_LEFT).clamp(min=0.0, max=1.0)

    theirs = F.hardtanh(orig["l0"].sample_z_from_u(u, log_alpha), 0, 1)

    assert t.equal(ours, theirs)


def test_hard_concrete_sampler_distribution_matches(orig):
    """The samplers as actually called cannot agree bitwise -- different RNG
    streams, different clamp bounds, and the port adds a batch dimension the
    reference does not have. Compare the distributions they induce instead."""
    log_alpha = t.tensor([5.0, 0.0, -5.0])

    t.manual_seed(0)
    ours = sample_hard_concrete(log_alpha, batch_size=200_000)  # (200000, 3)

    t.manual_seed(0)
    theirs = t.stack(
        [orig["l0"].sample_z_from_log_alpha(log_alpha) for _ in range(50_000)]
    )

    # Tolerance is set by Monte-Carlo error on the smaller sample (n = 50,000,
    # sd <= 0.5, so the sd of the mean is <= 2.3e-3); 1e-2 is ~4 of those.
    assert t.allclose(ours.mean(0), theirs.mean(0), atol=1e-2)
    assert t.allclose(ours.std(0), theirs.std(0), atol=1e-2)


def test_port_samples_per_batch_element_reference_does_not(orig):
    """Pin the divergence itself, so it cannot regress silently into 'fixed'."""
    log_alpha = t.zeros(64)

    t.manual_seed(0)
    ours = sample_hard_concrete(log_alpha, batch_size=8)
    assert ours.shape == (8, 64)
    assert not t.equal(ours[0], ours[1]), "port must draw independently per example"

    t.manual_seed(0)
    theirs = orig["l0"].sample_z_from_log_alpha(log_alpha)
    assert theirs.shape == (64,), "reference draws one sample, shared across the batch"


# --- A4: sparsity warmup schedule -------------------------------------------


@pytest.mark.parametrize(
    "start, target, warmup",
    [
        (0.0, 1.1, 2500),   # the paper's IOI sweep
        (0.0, 0.72, 2500),  # its node counterpart
        (0.3, 0.9, 100),
    ],
)
def test_sparsity_warmup_schedule_matches(orig, start, target, warmup):
    """Driven against the reference's *real* unbound method, with a stub `self`
    carrying only the attributes it reads."""

    class _Stub:
        num_edge_sparsity_warmup_steps = warmup
        warmup_type = "linear"
        start_edge_sparsity = start
        target_edge_sparsity = target

    theirs_fn = orig["trainer_cls"].get_current_edge_target_sparsity
    for step in list(range(0, 3000)):
        ours = _target_sparsity(step, start, target, warmup)
        theirs = theirs_fn(_Stub(), step)
        assert ours == pytest.approx(theirs, abs=0.0, rel=0.0), f"step {step}"


def test_sparsity_warmup_holds_target_after_warmup(orig):
    class _Stub:
        num_edge_sparsity_warmup_steps = 2500
        warmup_type = "linear"
        start_edge_sparsity = 0.0
        target_edge_sparsity = 1.1

    theirs_fn = orig["trainer_cls"].get_current_edge_target_sparsity
    for step in (2500, 2501, 2999):
        assert _target_sparsity(step, 0.0, 1.1, 2500) == theirs_fn(_Stub(), step) == 1.1


# --- A5: the Lagrangian term and the sparsity it is applied to ---------------


def test_lagrangian_matches_reference_forward(orig, tiny):
    """Compare against the value the reference's own forward computes, rather than
    against a transcription of its expression."""
    cfg = tiny.config
    n_edges = orig["modeling"].get_num_edges(cfg, False)

    with t.no_grad():
        for lam1, lam2 in [(0.0, 0.0), (1.5, 0.0), (0.0, 2.5), (-0.7, 3.1)]:
            tiny.transformer.sparsity_lambda_edges_1.data.fill_(lam1)
            tiny.transformer.sparsity_lambda_edges_2.data.fill_(lam2)
            ids = t.randint(0, 64, (2, 8))
            out = tiny(input_ids=ids, target_edge_sparsity=0.4, target_node_sparsity=0.4)

            sparsity = out.model_edge_sparsity.reshape([])
            ours = _lagrangian(
                sparsity, 0.4, t.tensor(lam1), t.tensor(lam2)
            )
            assert t.allclose(ours, out.edge_loss.reshape([]), atol=0, rtol=0)

    assert n_edges > 0


def test_sparsity_conventions_are_complementary(orig, tiny):
    """The reference reports ``1 - Σz_keep/n_edges``; the port sums z in *ablation*
    convention and divides by the same n_edges. On matched masks the two must be
    the same number."""
    with t.no_grad():
        ids = t.randint(0, 64, (2, 8))
        out = tiny(input_ids=ids, target_edge_sparsity=0.0, target_node_sparsity=0.0)
        their_sparsity = float(out.model_edge_sparsity)

    n_edges = orig["modeling"].get_num_edges(tiny.config, False)
    kept = tiny.transformer.get_edge_sparsity  # exists on the reference model
    assert callable(kept)

    # At init every log_alpha is +10, so σ(1.2·10) = 0.999994 of each edge is kept.
    keep_frac = float(t.sigmoid(t.tensor(12.0)))
    expected = 1.0 - keep_frac
    assert their_sparsity == pytest.approx(expected, abs=1e-6)

    # The port's ablation-convention sum over the same masks gives the same value.
    z_ablate = t.full((n_edges,), 1.0 - keep_frac)
    ours = float(z_ablate.sum() / n_edges)
    assert ours == pytest.approx(their_sparsity, abs=1e-6)


# --- A6: the KL objective ----------------------------------------------------


def test_kl_matches_reference_on_single_token_span(orig, tiny):
    """``multibatch_kl_div`` at a last-token slice equals the reference's
    per-example ``batchmean`` over a one-token answer span. This is the condition
    under which port-fidelity difference 4 (KL span) vanishes."""
    gen = t.Generator().manual_seed(5)
    batch, vocab = 6, 64
    circuit_logits = t.randn(batch, vocab, generator=gen)
    full_logits = t.randn(batch, vocab, generator=gen)

    ours = multibatch_kl_div(
        F.log_softmax(circuit_logits, dim=-1), F.log_softmax(full_logits, dim=-1)
    )

    # The reference's loop, over spans of length 1.
    theirs = 0.0
    for i in range(batch):
        a = F.log_softmax(circuit_logits[i : i + 1], dim=-1)
        b = F.log_softmax(full_logits[i : i + 1], dim=-1)
        theirs = theirs + F.kl_div(a, b, reduction="batchmean", log_target=True)
    theirs = theirs / batch

    assert t.allclose(ours, theirs, atol=1e-6)


def test_kl_diverges_from_reference_on_multi_token_span(orig):
    """The agreement above is conditional, so pin the condition. With a two-token
    span the reference averages over both positions while the port scores only the
    last, and the two genuinely differ."""
    gen = t.Generator().manual_seed(6)
    batch, span, vocab = 4, 2, 64
    circuit = t.randn(batch, span, vocab, generator=gen)
    full = t.randn(batch, span, vocab, generator=gen)

    port_last_token = multibatch_kl_div(
        F.log_softmax(circuit[:, -1], dim=-1), F.log_softmax(full[:, -1], dim=-1)
    )

    theirs = 0.0
    for i in range(batch):
        a = F.log_softmax(circuit[i], dim=-1)
        b = F.log_softmax(full[i], dim=-1)
        theirs = theirs + F.kl_div(a, b, reduction="batchmean", log_target=True)
    theirs = theirs / batch

    assert not t.allclose(port_last_token, theirs, atol=1e-3)


def test_frozen_split_answers_are_single_token():
    """The removal of difference 4 depends on every answer being one token, so
    check the artifact rather than trusting the claim."""
    path = repo_path_to_abs_path("datasets/port_fidelity/ioi_frozen.json")
    if not path.exists():
        pytest.skip("frozen split not built yet")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    data = json.loads(path.read_text())
    for row in data["prompts"]:
        for key in ("answers", "wrong_answers"):
            for answer in row[key]:
                assert answer.startswith(" "), f"{answer!r} lost its leading space"
                assert len(tok(answer)["input_ids"]) == 1, f"{answer!r} is multi-token"


# --- A7: learning-rate schedule ---------------------------------------------


def test_lr_schedule_matches_huggingface(orig):
    from transformers import get_linear_schedule_with_warmup as hf_schedule

    def _build(factory):
        p = t.nn.Parameter(t.zeros(1))
        opt = t.optim.AdamW([{"params": [p], "lr": 0.8}], lr=0.8)
        return opt, factory(opt, 200, 3000)

    our_opt, our_sched = _build(get_linear_schedule_with_warmup)
    hf_opt, hf_sched = _build(hf_schedule)

    for _ in range(3000):
        assert our_sched.get_last_lr() == hf_sched.get_last_lr()
        our_opt.step(), hf_opt.step()
        our_sched.step(), hf_sched.step()


# --- A8: optimiser wiring ----------------------------------------------------


def test_optimizer_param_groups_match_reference(orig, tiny):
    """The reference's real ``get_optimizers`` decides the group layout; assert the
    properties the port must reproduce (four groups, ascent on the lambdas only,
    torch's default weight decay)."""
    opt, _ = orig["get_optimizers"](
        tiny,
        edges_lr=0.8, layers_lr=0.8, reg_edges_lr=0.8, reg_layers_lr=0.8,
        num_training_steps=3000, warmup_steps=200, disable_node_loss=True,
    )
    groups = opt.param_groups
    assert len(groups) == 4
    # `lr` reads 0 here: get_optimizers also builds the LambdaLR, whose warmup
    # factor at step 0 is 0/200. `initial_lr` is the configured value.
    assert [g["initial_lr"] for g in groups] == [0.8, 0.8, 0.8, 0.8]
    assert [g.get("maximize", False) for g in groups] == [False, True, False, True]
    assert all(g["weight_decay"] == 0.01 for g in groups), "torch AdamW default"

    # --disable_node_loss empties the node-lambda group rather than removing it.
    assert len(groups[3]["params"]) == 0
    assert len(groups[1]["params"]) == 2  # lambda_edges_1 and _2
    assert len(groups[0]["params"]) > 0 and len(groups[2]["params"]) > 0


def test_only_mask_and_lambda_parameters_train(orig, tiny):
    from fpt2_ioi import freeze_all_except_pruning_params

    freeze_all_except_pruning_params(tiny)
    for name, p in tiny.named_parameters():
        expected = ("log_alpha" in name) or ("sparsity_lambda" in name)
        assert p.requires_grad == expected, name


# --- A9: node-mask composition ----------------------------------------------


def test_node_mask_composition_is_the_product_of_keep_masks():
    """The port composes in ablation space with De Morgan; the reference multiplies
    keep masks (``get_effective_edge_sparsity``, ``modeling_fpt2.py:971-994``).
    These are the same operation."""
    gen = t.Generator().manual_seed(21)
    z_edge_ablate = t.rand(1000, generator=gen)
    z_node_ablate = t.rand(1000, generator=gen)

    port_ablate = 1.0 - (1.0 - z_edge_ablate) * (1.0 - z_node_ablate)
    reference_keep = (1.0 - z_edge_ablate) * (1.0 - z_node_ablate)

    # Equal in exact arithmetic; `1 - (1 - x)` costs an ulp, hence the tolerance.
    assert t.allclose(1.0 - port_ablate, reference_keep, atol=1e-7)


# --- A10: mask initialisation -----------------------------------------------


def test_log_alpha_init_matches_under_the_sign_flip(orig, tiny):
    """The reference initialises at N(10, 0.01²) (keep); the port at N(-10, 0.01²),
    because AutoCircuit's convention is 1 = ablate. Compared by moments."""
    theirs = t.cat(
        [p.data.reshape(-1) for n, p in tiny.named_parameters() if "log_alpha" in n]
    )
    assert theirs.numel() > 0

    ours = t.empty(theirs.numel())
    t.nn.init.normal_(ours, mean=-10.0, std=0.01)

    assert float(theirs.mean()) == pytest.approx(10.0, abs=0.01)
    assert float(ours.mean()) == pytest.approx(-10.0, abs=0.01)
    assert float(ours.std()) == pytest.approx(float(theirs.std()), abs=0.005)


def test_soft_mask_is_the_reference_score_under_the_sign_flip(orig):
    """The port exports ``1 - σ(α/T·F)`` on a sign-flipped ``log_alpha``, which is
    ``σ(α/T·F)`` on the reference's -- so the two rank edges identically."""
    log_alpha = t.randn(500, generator=t.Generator().manual_seed(31)) * 5
    port_score = 1.0 - _soft_mask(-log_alpha)
    reference_soft = t.sigmoid(log_alpha / TEMPERATURE * FACTOR)
    assert t.allclose(port_score, reference_soft, atol=1e-7)


# --- A11: mask-sampling granularity, and which one is the default -----------


def test_shared_sampling_matches_the_reference_shape_convention(orig):
    """`shared=True` must reproduce the reference: one draw, reused for every
    example. The reference's `sample_z_from_log_alpha` has no batch dimension at
    all (`l0.py:43-48`), so every row of the port's batch must be identical."""
    log_alpha = t.zeros(64)

    t.manual_seed(0)
    shared = _sample_mask(log_alpha, batch_size=8, shared=True)
    assert shared.shape == (8, 64)
    for row in range(1, 8):
        assert t.equal(shared[0], shared[row]), "shared draw must not vary by example"

    t.manual_seed(0)
    per_example = _sample_mask(log_alpha, batch_size=8, shared=False)
    assert per_example.shape == (8, 64)
    assert not t.equal(per_example[0], per_example[1])

    # the reference draws exactly one vector of this shape
    t.manual_seed(0)
    theirs = orig["l0"].sample_z_from_log_alpha(log_alpha)
    assert theirs.shape == (64,)


def test_paper_faithful_sampling_is_the_default():
    """Pins the 2026-08-12 decision: the spine runs Edge Pruning as published.

    Tier C showed this single flag accounts for the entire discrepancy against the
    reference, so a silent flip of the default would silently change every EP result
    in the thesis. If this test fails, that is what happened.
    """
    import inspect

    default = inspect.signature(edge_pruning_prune_scores).parameters[
        "shared_mask_sample"
    ].default
    assert default is True

    from auto_circuit.prune_algos.prune_algos import EDGE_PRUNING_PRUNE_ALGO

    bound = EDGE_PRUNING_PRUNE_ALGO.func.keywords
    assert bound.get("shared_mask_sample", default) is True
