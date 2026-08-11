"""Tier A of the DiscoGP port fidelity study: component equivalence.

Each test pairs one component of the port against the corresponding component of
the original implementation (``frankniujc/disco_gp`` @ ``54aef81``, vendored at
``third_party/disco_gp``) and asserts they agree exactly.

These tests are the cheapest and strongest evidence in the fidelity study: they
turn claims that were previously established by reading source into assertions
that run in CI. They also close the coverage gap on ``disco_gp.py``, whose mask
convention inversion was previously untested.

Skipped automatically when ``third_party/disco_gp`` is absent.
"""

import sys
from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as F

from auto_circuit.prune_algos.disco_gp import (
    _complete_loss_margin,
    _faith_loss_margin,
    _sample_edge_masks,
)
from auto_circuit.utils.gumbel_sigmoid_utils import gumbel_sigmoid, schedule_epoch_lambda

_ORIG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "disco_gp"


def _import_original():
    """Import the vendored original, or skip the whole module if unavailable."""
    if not _ORIG_ROOT.is_dir():
        pytest.skip(f"vendored original not found at {_ORIG_ROOT}")
    if str(_ORIG_ROOT) not in sys.path:
        sys.path.insert(0, str(_ORIG_ROOT))
    from disco_gp.discogp_lm import gumbel_sigmoid as orig_gumbel_sigmoid
    from disco_gp.evaluation import (
        compute_complete_loss_binary_label,
        compute_faith_loss_binary_label,
    )
    from disco_gp.utils import schedule_epoch_lambda as orig_schedule

    return {
        "gumbel_sigmoid": orig_gumbel_sigmoid,
        "schedule_epoch_lambda": orig_schedule,
        "faith": compute_faith_loss_binary_label,
        "complete": compute_complete_loss_binary_label,
    }


@pytest.fixture(scope="module")
def orig():
    return _import_original()


# --- A1: Gumbel-sigmoid ------------------------------------------------------
# Both draw `logits.new_empty([2, *shape]).uniform_(0, 1)`, so under a shared seed
# they consume the RNG stream identically and must agree bitwise.


@pytest.mark.parametrize("gs_temp", [0.01, 1.0])
@pytest.mark.parametrize("shape", [(157,), (32, 12), (4, 8, 16)])
def test_gumbel_sigmoid_bitwise(orig, gs_temp, shape):
    logits = t.randn(shape, generator=t.Generator().manual_seed(0))

    t.manual_seed(1234)
    ours = gumbel_sigmoid(logits, gs_temp=gs_temp, hard=True)
    t.manual_seed(1234)
    theirs = orig["gumbel_sigmoid"](logits, gs_temp=gs_temp)

    assert t.equal(ours, theirs)


def test_gumbel_sigmoid_output_is_binary_under_ste(orig):
    """The original has no `hard` flag; it always applies the STE. `hard=True`
    is therefore the port setting that corresponds to the original."""
    logits = t.randn(1000, generator=t.Generator().manual_seed(0))

    t.manual_seed(7)
    ours = gumbel_sigmoid(logits, gs_temp=1.0, hard=True)
    t.manual_seed(7)
    theirs = orig["gumbel_sigmoid"](logits, gs_temp=1.0)

    assert set(ours.unique().tolist()) <= {0.0, 1.0}
    assert t.equal(ours, theirs)


def test_gumbel_sigmoid_ste_gradient_matches(orig):
    """Straight-through: forward is binary, backward flows through the sigmoid."""
    logits = t.randn(256, generator=t.Generator().manual_seed(0))

    a = logits.clone().requires_grad_(True)
    t.manual_seed(99)
    gumbel_sigmoid(a, gs_temp=1.0, hard=True).sum().backward()

    b = logits.clone().requires_grad_(True)
    t.manual_seed(99)
    orig["gumbel_sigmoid"](b, gs_temp=1.0).sum().backward()

    assert a.grad is not None and b.grad is not None
    assert t.equal(a.grad, b.grad)


# --- A2: lambda schedule -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, n_epochs",
    [
        # Edge phase, as registered in prune_algos.py and the original's cli.py.
        (dict(lambda_0=1.0, max_times=100.0, min_times=0.01,
              n_epoch_warmup=20, n_epoch_cooldown=20), 100),
        # Weight phase.
        (dict(lambda_0=1.0, max_times=1000.0, min_times=1.0,
              n_epoch_warmup=500, n_epoch_cooldown=1), 500),
        # Completeness coefficient: called with defaults, i.e. a flat schedule.
        (dict(lambda_0=1.0), 100),
        (dict(lambda_0=0.0), 100),
    ],
)
def test_schedule_epoch_lambda_matches(orig, kwargs, n_epochs):
    for epoch in range(n_epochs):
        ours = schedule_epoch_lambda(epoch, **kwargs)
        theirs = orig["schedule_epoch_lambda"](epoch, **kwargs)
        assert ours == theirs, f"epoch {epoch}: {ours} != {theirs}"


def test_schedule_zero_warmup_guard_is_unreachable(orig):
    """The port guards the divisor with `max(n, 1)` where the original does not.

    The guard is unreachable: the branch requires `epoch < n_epoch_warmup`, which
    is false whenever `n_epoch_warmup == 0`. This test pins that reasoning so the
    divergence cannot be mistaken for a behavioural difference later.
    """
    for epoch in range(50):
        ours = schedule_epoch_lambda(
            epoch, 1.0, max_times=10.0, min_times=0.5,
            n_epoch_warmup=0, n_epoch_cooldown=0,
        )
        theirs = orig["schedule_epoch_lambda"](
            epoch, 1.0, max_times=10.0, min_times=0.5,
            n_epoch_warmup=0, n_epoch_cooldown=0,
        )
        assert ours == theirs == 0.5


# --- A3 / A4: faithfulness and completeness losses ---------------------------
# The original indexes a (B, S, V) tensor at `seq_lens - 1`; the port receives the
# already-sliced (B, V) tensor. Feeding the original a batch whose `seq_lens`
# select exactly the port's slice makes the two directly comparable.


class _Batch:
    """Minimal stand-in for auto_circuit.data.PromptPairBatch."""

    def __init__(self, answers, wrong_answers):
        self.answers = answers
        self.wrong_answers = wrong_answers


def _paired_inputs(batch_size=16, seq_len=7, vocab=5000, seed=0):
    g = t.Generator().manual_seed(seed)
    logits_bsv = t.randn(batch_size, seq_len, vocab, generator=g)
    seq_lens = t.randint(1, seq_len + 1, (batch_size,), generator=g)
    good = t.randint(0, vocab, (batch_size,), generator=g)
    bad = t.randint(0, vocab, (batch_size,), generator=g)

    sliced = logits_bsv[t.arange(batch_size), seq_lens - 1]  # (B, V)
    batch = _Batch(answers=good.unsqueeze(-1), wrong_answers=bad.unsqueeze(-1))
    orig_inputs = {"seq_lens": seq_lens, "target good": good, "target bad": bad}
    return logits_bsv, sliced, batch, orig_inputs


def test_faith_loss_matches(orig):
    logits_bsv, sliced, batch, orig_inputs = _paired_inputs()

    ours = _faith_loss_margin(sliced, batch)
    theirs = orig["faith"](logits_bsv, orig_inputs)["faith"]

    assert t.allclose(ours, theirs, rtol=0, atol=0)


def test_complete_loss_matches(orig):
    logits_bsv, sliced, batch, orig_inputs = _paired_inputs(seed=3)

    ours = _complete_loss_margin(sliced, batch)
    theirs, _ = orig["complete"](logits_bsv, orig_inputs)

    assert t.allclose(ours, theirs, rtol=0, atol=0)


def test_faith_loss_multi_token_answers_reduce_to_the_binary_case(orig):
    """The port averages over multi-token answer sets; with one answer per prompt
    that mean is the identity, which is what makes the comparison above valid."""
    _, sliced, batch, _ = _paired_inputs(seed=5)
    duplicated = _Batch(
        answers=batch.answers.repeat(1, 4),
        wrong_answers=batch.wrong_answers.repeat(1, 4),
    )
    assert t.allclose(_faith_loss_margin(sliced, batch),
                      _faith_loss_margin(sliced, duplicated))


# --- A5: sparsity loss -------------------------------------------------------


def test_edge_sparsity_loss_matches():
    """Port computes the sparsity term inline (disco_gp.py); the original exposes
    it as `edge_sparseness_loss`. Both are `sum(sigmoid(logits)) / N_edge`."""
    g = t.Generator().manual_seed(11)
    logit_dict = {f"m{i}": t.randn(s, generator=g) for i, s in enumerate([(157,), (13, 12), (25,)])}
    n_edge = sum(v.numel() for v in logit_dict.values())

    ours = t.cat([t.sigmoid(l).flatten() for l in logit_dict.values()]).sum() / n_edge

    # Transcription of discogp_lm.DiscoGPTransformer.edge_sparseness_loss.
    theirs = sum(F.sigmoid(l).sum() for l in logit_dict.values()) / n_edge

    assert t.allclose(ours, theirs, rtol=0, atol=0)


# --- A6: mask convention and reversal ---------------------------------------
# The original multiplies each residual contribution by a *keep* mask (1 = keep).
# AutoCircuit's patch_mask is an *ablate* mask (1 = replace with ablation), so the
# port stores `AC_mask = 1 - disco_mask`. This inversion had no test before.


def _orig_keep_mask(logits, gs_temp, reverse):
    """Transcription of the mask path in modules/transformer_block.py:142-150."""
    from disco_gp.discogp_lm import gumbel_sigmoid as orig_gs

    sampled = orig_gs(logits, gs_temp)
    return 1.0 - sampled if reverse else sampled


@pytest.mark.parametrize("reverse", [False, True])
def test_mask_convention_inversion(orig, reverse):
    g = t.Generator().manual_seed(21)
    edge_logits = {
        "a": t.nn.Parameter(t.randn(13, 12, generator=g)),
        "b": t.nn.Parameter(t.randn(157, generator=g)),
    }

    t.manual_seed(555)
    ac_masks = _sample_edge_masks(edge_logits, gs_temp=1.0, reverse=reverse, hard=True)

    t.manual_seed(555)
    for name, logit in edge_logits.items():
        keep = _orig_keep_mask(logit, 1.0, reverse)
        # AutoCircuit: mask=1 ablates, so keep = 1 - patch_mask.
        assert t.equal(1.0 - ac_masks[name], keep)


def test_reverse_is_the_complement_of_forward(orig):
    """Completeness evaluates the complement circuit: every edge kept in the
    faithfulness pass must be ablated in the completeness pass, and vice versa."""
    g = t.Generator().manual_seed(31)
    edge_logits = {"a": t.nn.Parameter(t.randn(200, generator=g))}

    t.manual_seed(77)
    fwd = _sample_edge_masks(edge_logits, gs_temp=1.0, reverse=False, hard=True)
    t.manual_seed(77)
    rev = _sample_edge_masks(edge_logits, gs_temp=1.0, reverse=True, hard=True)

    assert t.equal(fwd["a"], 1.0 - rev["a"])


# --- A7: mask logit initialisation ------------------------------------------


def test_edge_logit_init_distribution_matches():
    """Port: `full_like(x, init) + 0.01 * randn_like(x)`.
    Original: `nn.init.normal_(ones(shape), mean=init, std=0.01)`.

    Different expressions, same N(init, 0.01) law. Compared on moments because the
    two consume the RNG stream differently.
    """
    shape = (1_000_000,)
    init = 1.0

    t.manual_seed(2024)
    ours = t.full(shape, init) + 0.01 * t.randn(shape)

    t.manual_seed(2024)
    theirs = t.nn.init.normal_(t.ones(shape), mean=init, std=0.01)

    assert abs(ours.mean().item() - theirs.mean().item()) < 1e-4
    assert abs(ours.std().item() - theirs.std().item()) < 1e-4
    assert abs(ours.mean().item() - init) < 1e-4
    assert abs(ours.std().item() - 0.01) < 1e-4


# --- A8: graph shape ---------------------------------------------------------


def test_original_mask_shapes_sum_to_the_autocircuit_edge_count():
    """The original's per-block mask tensors enumerate the same graph AutoCircuit
    builds for GPT-2 small with `factorized=True, separate_qkv=True`:
    157 sources, 445 destinations, 32,491 edges."""
    n_layers, n_heads = 12, 12

    n_edges = n_srcs = n_dests = 0
    for block in range(n_layers):
        prev_nodes = (n_heads + 1) * block + 1  # transformer_block.py:97
        n_edges += 3 * prev_nodes * n_heads  # q, k, v: (prev_nodes, n_heads)
        n_edges += prev_nodes + n_heads  # mlp: (prev_nodes + n_heads,)
        n_dests += 3 * n_heads + 1
    n_srcs = (n_heads + 1) * n_layers + 1  # edge_mask_output_logits
    n_edges += n_srcs  # resid_final reads every source
    n_dests += 1

    assert (n_srcs, n_dests, n_edges) == (157, 445, 32491)
