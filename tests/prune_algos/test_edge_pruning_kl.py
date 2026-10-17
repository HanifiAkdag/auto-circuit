"""Vocabulary selection in Edge Pruning's training objective."""

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch as t
import torch.nn.functional as F

import auto_circuit.prune_algos.edge_pruning as ep
from auto_circuit.data import PromptDataLoader, PromptDataset
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.tensor_ops import multibatch_kl_div


def test_full_vocabulary_kl_preserves_existing_calculation():
    generator = t.Generator().manual_seed(7)
    circuit = t.randn(3, 2, 11, generator=generator, dtype=t.float64)
    full = t.randn(3, 2, 11, generator=generator, dtype=t.float64)
    expected = multibatch_kl_div(F.log_softmax(circuit, -1), F.log_softmax(full, -1))
    assert t.equal(ep._kl_divergence(circuit, full), expected)


@pytest.mark.parametrize("repeats", [1, 3])
def test_restricted_kl_normalizes_candidates_and_has_correct_gradient(repeats):
    circuit = t.tensor([[2.0, 3.0, 100.0]], dtype=t.float64).log()
    circuit = circuit.repeat(repeats, 1).requires_grad_()
    full = t.tensor([[4.0, 1.0, 0.1]], dtype=t.float64).log().repeat(repeats, 1)
    indices = t.tensor([0, 1])

    loss = ep._kl_divergence(circuit, full, indices)
    expected = 0.8 * math.log(0.8 / 0.4) + 0.2 * math.log(0.2 / 0.6)
    assert loss.item() == pytest.approx(expected, abs=1e-14)
    loss.backward()
    expected_gradient = t.tensor([[-0.4, 0.4, 0.0]], dtype=t.float64)
    t.testing.assert_close(circuit.grad, expected_gradient.repeat(repeats, 1) / repeats)


def test_restricted_kl_ignores_logits_outside_candidate_set():
    circuit = t.tensor([[1.0, -2.0, 8.0, 0.0]])
    full = t.tensor([[-2.0, 5.0, 0.0, 3.0]])
    indices = t.tensor([3, 0])
    original = ep._kl_divergence(circuit, full, indices)
    circuit[:, 1:3] += 1000
    full[:, 1:3] -= 1000
    assert t.equal(ep._kl_divergence(circuit, full, indices), original)


class _Tokenizer:
    def encode(self, token, **kwargs):
        return {"00": [9], "01": [3], "alias": [9], "multi": [4, 5]}[token]


def test_resolve_vocabulary_preserves_token_order():
    model = SimpleNamespace(tokenizer=_Tokenizer())
    indices = ep._resolve_kl_vocab_indices(model, ("00", "01"), t.device("cpu"))
    assert indices.tolist() == [9, 3]
    assert indices.dtype == t.long


def test_default_requires_no_tokenizer_or_task_configuration():
    assert ep._resolve_kl_vocab_indices(SimpleNamespace(), None, t.device("cpu")) is None


@pytest.mark.parametrize("tokens", [(), "00", ("multi",), ("00", "alias"), ("00", 7)])
def test_invalid_vocabulary_is_rejected(tokens):
    with pytest.raises(ValueError):
        ep._resolve_kl_vocab_indices(SimpleNamespace(tokenizer=_Tokenizer()), tokens, t.device("cpu"))


def test_explicit_vocabulary_requires_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        ep._resolve_kl_vocab_indices(SimpleNamespace(), ("00",), t.device("cpu"))


def _training_setup(mini_tl_transformer):
    model = patchable_model(
        deepcopy(mini_tl_transformer), factorized=True,
        slice_output="last_seq", separate_qkv=True,
        device=next(mini_tl_transformer.parameters()).device,
    )
    device = next(model.parameters()).device
    clean = t.tensor([[10, 20, 30], [40, 50, 60]], device=device)
    corrupt = clean.flip(0)
    dataset = PromptDataset(
        clean, corrupt,
        [t.tensor([1], device=device) for _ in range(2)],
        [t.tensor([2], device=device) for _ in range(2)],
    )
    loader = PromptDataLoader(dataset, seq_len=None, diverge_idx=0, batch_size=2)
    return model, loader


def test_tokenizer_entries_outside_model_vocabulary_fail_before_optimization(mini_tl_transformer):
    model, loader = _training_setup(mini_tl_transformer)
    token = "<ep_test_token_outside_model_vocab>"
    assert model.tokenizer.add_tokens([token]) == 1
    requires_grad = [param.requires_grad for param in model.parameters()]
    with pytest.raises(ValueError, match="output vocabulary"):
        ep.edge_pruning_prune_scores(
            model, loader, None, steps=1, kl_vocab_tokens=(token,),
        )
    assert [param.requires_grad for param in model.parameters()] == requires_grad


@pytest.mark.parametrize("tokens", [None, tuple(f"{i:02d}" for i in range(100))])
def test_training_loop_uses_configured_vocabulary(mini_tl_transformer, monkeypatch, tokens):
    model, loader = _training_setup(mini_tl_transformer)
    observed = []
    calculate = ep._kl_divergence

    def record(circuit_logits, full_logits, vocab_indices=None):
        loss = calculate(circuit_logits, full_logits, vocab_indices)
        observed.append((circuit_logits.detach(), full_logits.detach(), vocab_indices, loss.detach()))
        assert loss.requires_grad
        return loss

    monkeypatch.setattr(ep, "_kl_divergence", record)
    t.manual_seed(5)
    scores = ep.edge_pruning_prune_scores(
        model, loader, None, steps=2, init_log_alpha=0.0,
        disable_node_loss=True, kl_vocab_tokens=tokens,
    )

    assert len(observed) == 2
    for circuit, full, indices, loss in observed:
        if tokens is None:
            assert indices is None
        else:
            expected_ids = [model.tokenizer.encode(f"{i:02d}")[0] for i in range(100)]
            assert indices.tolist() == expected_ids
            assert len(set(expected_ids)) == 100
            circuit, full = circuit[:, expected_ids], full[:, expected_ids]
        target = F.softmax(full, dim=-1)
        expected = (target * (F.log_softmax(full, -1) - F.log_softmax(circuit, -1))).sum(-1).mean()
        t.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)

    assert set(scores) == set(model.patch_masks)
    assert sum(value.numel() for value in scores.values()) == model.n_edges
    assert all(t.isfinite(value).all() and ((0 <= value) & (value <= 1)).all() for value in scores.values())
