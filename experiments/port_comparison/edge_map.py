"""Bijection between the original DiscoGP's edge masks and AutoCircuit's PruneScores.

The two implementations enumerate the *same* graph for GPT-2 small
(157 sources, 32,491 edges), but store it differently:

===================  ==============================  =========================
edge group           original                        AutoCircuit
===================  ==============================  =========================
Q / K / V input      ``blocks.{b}.edge_mask_         ``blocks.{b}.hook_{q,k,v}
                     attention_{q,k,v}_logits``      _input``
                     ``(prev_nodes, n_heads)``       ``(n_heads, prev_nodes)``
MLP input            ``blocks.{b}.edge_mask_         ``blocks.{b}.hook_mlp_in``
                     mlp_logits``
                     ``(prev_nodes + n_heads,)``     ``(prev_nodes + n_heads,)``
final residual       ``edge_mask_output_logits``     ``blocks.{L-1}.hook_
                     ``(n_srcs,)``                   resid_post`` ``(n_srcs,)``
===================  ==============================  =========================

Source indices coincide exactly. Both order sources as ``embed``, then per block
``[head_0 .. head_{H-1}, mlp]``, so the original's ``prev_nodes`` index equals
AutoCircuit's ``SrcNode.src_idx`` with no remapping. The only transformation is a
transpose on the Q/K/V tensors.

Do not trust this module on the strength of its shape assertions alone: a
permutation error within a shape would pass every check here. It is verified
*functionally* in ``tests/prune_algos/test_disco_gp_equivalence.py``, by running
both implementations under the same random masks and comparing logits.
"""

from typing import Dict

import torch as t

QKV_LETTERS = ("q", "k", "v")


def n_prev_nodes(block: int, n_heads: int) -> int:
    """Sources visible to `block`'s attention: embed plus every earlier node.

    Mirrors ``modules/transformer_block.py:97``.
    """
    return (n_heads + 1) * block + 1


def n_sources(n_layers: int, n_heads: int) -> int:
    """Total source nodes, i.e. the width of the final-residual mask."""
    return (n_heads + 1) * n_layers + 1


def orig_key_qkv(block: int, letter: str) -> str:
    return f"blocks.{block}.edge_mask_attention_{letter}_logits"


def orig_key_mlp(block: int) -> str:
    return f"blocks.{block}.edge_mask_mlp_logits"


ORIG_KEY_OUTPUT = "edge_mask_output_logits"


def ac_key_qkv(block: int, letter: str) -> str:
    return f"blocks.{block}.hook_{letter}_input"


def ac_key_mlp(block: int) -> str:
    return f"blocks.{block}.hook_mlp_in"


def ac_key_output(n_layers: int) -> str:
    return f"blocks.{n_layers - 1}.hook_resid_post"


def expected_shapes(n_layers: int, n_heads: int) -> Dict[str, Dict[str, tuple]]:
    """Per-tensor shapes on both sides, keyed by AutoCircuit module name."""
    shapes: Dict[str, Dict[str, tuple]] = {}
    for block in range(n_layers):
        prev = n_prev_nodes(block, n_heads)
        for letter in QKV_LETTERS:
            shapes[ac_key_qkv(block, letter)] = {
                "orig_key": orig_key_qkv(block, letter),
                "orig": (prev, n_heads),
                "ac": (n_heads, prev),
                "transpose": True,
            }
        shapes[ac_key_mlp(block)] = {
            "orig_key": orig_key_mlp(block),
            "orig": (prev + n_heads,),
            "ac": (prev + n_heads,),
            "transpose": False,
        }
    n_srcs = n_sources(n_layers, n_heads)
    shapes[ac_key_output(n_layers)] = {
        "orig_key": ORIG_KEY_OUTPUT,
        "orig": (n_srcs,),
        "ac": (n_srcs,),
        "transpose": False,
    }
    return shapes


def total_edges(n_layers: int, n_heads: int) -> int:
    return sum(int(t.tensor(s["ac"]).prod()) for s in expected_shapes(n_layers, n_heads).values())


def orig_to_ac(
    orig: Dict[str, t.Tensor], n_layers: int, n_heads: int, strict: bool = True
) -> Dict[str, t.Tensor]:
    """Convert original edge-mask tensors to AutoCircuit module-keyed tensors.

    Values pass through untouched; only the layout changes. Callers decide whether
    the values are logits, keep-masks, or scores.
    """
    out: Dict[str, t.Tensor] = {}
    for ac_key, spec in expected_shapes(n_layers, n_heads).items():
        tensor = orig[spec["orig_key"]]
        if strict and tuple(tensor.shape) != spec["orig"]:
            raise ValueError(
                f"{spec['orig_key']}: expected {spec['orig']}, got {tuple(tensor.shape)}"
            )
        out[ac_key] = tensor.T.contiguous() if spec["transpose"] else tensor.clone()
    if strict:
        unmapped = set(orig) - {s["orig_key"] for s in expected_shapes(n_layers, n_heads).values()}
        if unmapped:
            raise ValueError(f"unmapped original keys: {sorted(unmapped)}")
    return out


def ac_to_orig(
    ac: Dict[str, t.Tensor], n_layers: int, n_heads: int, strict: bool = True
) -> Dict[str, t.Tensor]:
    """Inverse of :func:`orig_to_ac`."""
    out: Dict[str, t.Tensor] = {}
    for ac_key, spec in expected_shapes(n_layers, n_heads).items():
        tensor = ac[ac_key]
        if strict and tuple(tensor.shape) != spec["ac"]:
            raise ValueError(f"{ac_key}: expected {spec['ac']}, got {tuple(tensor.shape)}")
        out[spec["orig_key"]] = tensor.T.contiguous() if spec["transpose"] else tensor.clone()
    return out


def keep_mask_to_prune_scores(
    orig_logits: Dict[str, t.Tensor], n_layers: int, n_heads: int
) -> Dict[str, t.Tensor]:
    """Original mask logits to AutoCircuit ``PruneScores``.

    The port returns ``sigmoid(logit)`` as the score (higher = more important =
    keep), which is monotone in the logit and so induces the same ranking the
    original's ``logit > 0`` threshold produces. Applying the same transform here
    keeps the two arms on one scale.
    """
    return {k: t.sigmoid(v) for k, v in orig_to_ac(orig_logits, n_layers, n_heads).items()}
