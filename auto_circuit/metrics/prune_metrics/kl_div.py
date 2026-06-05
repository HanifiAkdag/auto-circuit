from typing import Dict, List

import torch as t
import torch.nn.functional as F
from torch.nn.functional import log_softmax

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import BatchKey, CircuitOutputs, Measurements
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import multibatch_kl_div


def _answer_pair_logprobs(logits: t.Tensor, batch) -> t.Tensor:
    """Log-softmax over the (mean_correct, mean_wrong) logit pair.

    Mirrors DiscoGP evaluation.py: stack [logit_good, logit_bad] at the answer
    position, apply log_softmax over the 2-class pair — NOT the full vocabulary.

    Returns:
        (B, 2) log-probabilities over the (correct, wrong) pair.
    """
    answers = batch.answers
    wrong_answers = batch.wrong_answers
    if isinstance(answers, t.Tensor) and isinstance(wrong_answers, t.Tensor):
        correct_mean = t.gather(logits, dim=-1, index=answers).mean(dim=-1)
        wrong_mean = t.gather(logits, dim=-1, index=wrong_answers).mean(dim=-1)
    else:
        assert isinstance(answers, list) and isinstance(wrong_answers, list)
        correct_mean = t.stack(
            [logits[i].index_select(-1, a).mean() for i, a in enumerate(answers)]
        )
        wrong_mean = t.stack(
            [logits[i].index_select(-1, w).mean() for i, w in enumerate(wrong_answers)]
        )
    return F.log_softmax(t.stack([correct_mean, wrong_mean], dim=-1), dim=-1)


def measure_kl_div(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    compare_to_clean: bool = True,
) -> Measurements:
    """
    Average KL divergence between the full model and the circuits.

    Args:
        model: The model on which `circuit_outs` was calculated.
        dataloader: The dataloader on which the `circuit_outs` was calculated.
        circuit_outs: The outputs of the ablated model for each circuit size.
        compare_to_clean: Whether to compare the circuit output to the full model on the
            clean (`True`) or corrupt (`False`) prompt.

    Returns:
        A list of tuples, where the first element is the number of edges pruned and the
            second element is the average KL divergence for that number of edges.
    """
    circuit_kl_divs: Measurements = []
    default_logprobs: Dict[BatchKey, t.Tensor] = {}
    with t.inference_mode():
        for batch in dataloader:
            default_batch = batch.clean if compare_to_clean else batch.corrupt
            logits = model(default_batch)[model.out_slice]
            default_logprobs[batch.key] = log_softmax(logits, dim=-1)

    for edge_count, circuit_out in (pruned_out_pbar := tqdm(circuit_outs.items())):
        pruned_out_pbar.set_description_str(f"KL Div for {edge_count} edges")
        circuit_logprob_list: List[t.Tensor] = []
        default_logprob_list: List[t.Tensor] = []
        for batch in dataloader:
            circuit_logprob_list.append(log_softmax(circuit_out[batch.key], dim=-1))
            default_logprob_list.append(default_logprobs[batch.key])
        kl = multibatch_kl_div(t.cat(circuit_logprob_list), t.cat(default_logprob_list))

        # Numerical errors can cause tiny negative values in KL divergence
        circuit_kl_divs.append((edge_count, max(kl.item(), 0)))
    return circuit_kl_divs


def measure_answer_pair_kl_div(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    circuit_outs: CircuitOutputs,
    compare_to_clean: bool = True,
) -> Measurements:
    """KL divergence restricted to the (correct, wrong) answer logit pair.

    Both distributions are log_softmax over [mean_correct_logit, mean_wrong_logit]
    rather than the full vocabulary — directly comparable to DiscoGP's reported KL.

    Args:
        model: The patchable model (used for full-model forward pass).
        dataloader: Dataloader on which circuit_outs was calculated.
        circuit_outs: Ablated model outputs for each circuit size.
        compare_to_clean: Compare circuit to clean (True) or corrupt (False) prompt.

    Returns:
        List of (edge_count, kl_div) tuples.
    """
    default_logprobs: Dict[BatchKey, t.Tensor] = {}
    with t.inference_mode():
        for batch in dataloader:
            default_batch = batch.clean if compare_to_clean else batch.corrupt
            logits = model(default_batch)[model.out_slice]
            default_logprobs[batch.key] = _answer_pair_logprobs(logits, batch)

    kl_divs: Measurements = []
    for edge_count, circuit_out in (pbar := tqdm(circuit_outs.items())):
        pbar.set_description_str(f"Answer-Pair KL Div for {edge_count} edges")
        circuit_logprob_list: List[t.Tensor] = []
        default_logprob_list: List[t.Tensor] = []
        for batch in dataloader:
            circuit_logprob_list.append(_answer_pair_logprobs(circuit_out[batch.key], batch))
            default_logprob_list.append(default_logprobs[batch.key])
        kl = multibatch_kl_div(t.cat(circuit_logprob_list), t.cat(default_logprob_list))
        kl_divs.append((edge_count, max(kl.item(), 0)))
    return kl_divs
