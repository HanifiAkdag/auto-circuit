"""Bijection between the original Edge Pruning's masks and AutoCircuit's PruneScores.

The two enumerate almost the same graph for GPT-2 small, but store it differently
and disagree about the embeddings.

**Layout.** The original keeps every read mask at full writer width and enforces
causality with a buffer (``attn_read_common_mask``, ``modeling_fpt2.py:510-527``),
so entries above the layer's writer offset exist as parameters but are multiplied
by zero. AutoCircuit allocates exactly the legal width. The map therefore *slices*
before transposing; it is not the pure transpose the DiscoGP map was.

=================  ================================  ==========================
edge group         original                          AutoCircuit
=================  ================================  ==========================
Q / K / V input    ``h.{b}.{q,k,v}_read_log_alphas`` ``blocks.{b}.hook_{q,k,v}
                   ``(n_writers, n_head)``           _input``
                   legal rows ``[:13b]``             ``(n_head, 13b + 1)``
MLP input          ``h.{b}.mlp_read_log_alphas``     ``blocks.{b}.hook_mlp_in``
                   ``(n_writers,)``                  ``(13b + 13,)``
                   legal ``[:13b + 12]``
final residual     ``final_read_log_alphas``         ``blocks.{L-1}.hook_
                   ``(n_writers,)``                  resid_post`` ``(n_srcs,)``
node / write       ``h.{b}.attn_write_log_alphas``   folded into the port's
                   ``(n_head,)``,                    per-``src_idx`` node scalars
                   ``h.{b}.mlp_write_log_alphas``
                   ``(1,)``
=================  ================================  ==========================

**Embeddings.** Run without ``--with_embedding_nodes`` (the configuration this
study uses, see the spec §2.1) the original has **156** writers and **32,046**
edges; AutoCircuit has **157** sources and **32,491**. The difference is exactly
``n_readers = 445``: AutoCircuit models a combined embed source, the original adds
the embeddings unconditionally at every read site
(``x_q = x_q + embeds.unsqueeze(0)``, ``modeling_fpt2.py:562-564``) -- which *is*
"embed edges always kept".

So the correspondence is: AutoCircuit ``src_idx i`` ↔ original writer ``i - 1``,
with AutoCircuit's ``src_idx 0`` (the embed column, index 0 of every mask's source
axis) having no counterpart and being kept in converted reference scores.
This conversion does not freeze the port's own embedding masks during training.

Do not trust this module on its shape assertions alone: a permutation error within
a shape would pass every check here. It is verified *functionally* in
``tests/prune_algos/test_edge_pruning_equivalence.py``, by running both
implementations under the same random masks and comparing logits.
"""

from typing import Dict, List, Tuple

import torch as t

QKV_LETTERS = ("q", "k", "v")


# --- counts ------------------------------------------------------------------


def n_writers(n_layers: int, n_heads: int) -> int:
    """Original writer count without embedding nodes (``get_num_writers``)."""
    return n_layers * (n_heads + 1)


def n_sources(n_layers: int, n_heads: int) -> int:
    """AutoCircuit source count: the writers plus one combined embed source."""
    return n_writers(n_layers, n_heads) + 1


def attn_writer_offset(block: int, n_heads: int) -> int:
    """First writer index belonging to `block` (`get_base_indices_for_layer`)."""
    return block * (n_heads + 1)


def mlp_writer_offset(block: int, n_heads: int) -> int:
    return attn_writer_offset(block, n_heads) + n_heads


def ac_prev_nodes(block: int, n_heads: int) -> int:
    """AutoCircuit sources visible to `block`'s attention: embed plus every earlier."""
    return attn_writer_offset(block, n_heads) + 1


def n_readers(n_layers: int, n_heads: int) -> int:
    return n_layers * (3 * n_heads + 1) + 1


def total_edges_original(n_layers: int, n_heads: int) -> int:
    """Mirrors ``get_num_edges(cfg, with_embedding_nodes=False)``."""
    total = 0
    for block in range(n_layers):
        contribution = block * (n_heads + 1)
        total += 3 * n_heads * contribution
        total += contribution + n_heads
    return total + n_writers(n_layers, n_heads)


# --- key names ---------------------------------------------------------------


def orig_key_qkv(block: int, letter: str) -> str:
    return f"h.{block}.{letter}_read_log_alphas"


def orig_key_mlp(block: int) -> str:
    return f"h.{block}.mlp_read_log_alphas"


ORIG_KEY_FINAL = "final_read_log_alphas"


def orig_key_attn_write(block: int) -> str:
    return f"h.{block}.attn_write_log_alphas"


def orig_key_mlp_write(block: int) -> str:
    return f"h.{block}.mlp_write_log_alphas"


def ac_key_qkv(block: int, letter: str) -> str:
    return f"blocks.{block}.hook_{letter}_input"


def ac_key_mlp(block: int) -> str:
    return f"blocks.{block}.hook_mlp_in"


def ac_key_resid(n_layers: int) -> str:
    return f"blocks.{n_layers - 1}.hook_resid_post"


# --- the map itself ----------------------------------------------------------


def expected_shapes(n_layers: int, n_heads: int) -> Dict[str, dict]:
    """Per-tensor specs, keyed by AutoCircuit module name.

    ``legal`` is how many leading entries of the original's source axis are causally
    permitted; ``ac`` is AutoCircuit's full shape, whose source axis is one wider
    because of the embed column at index 0.
    """
    spec: Dict[str, dict] = {}
    for block in range(n_layers):
        legal = attn_writer_offset(block, n_heads)
        for letter in QKV_LETTERS:
            spec[ac_key_qkv(block, letter)] = {
                "orig_key": orig_key_qkv(block, letter),
                "orig_full": (n_writers(n_layers, n_heads), n_heads),
                "legal": legal,
                "ac": (n_heads, legal + 1),
                "transpose": True,
            }
        mlp_legal = mlp_writer_offset(block, n_heads)
        spec[ac_key_mlp(block)] = {
            "orig_key": orig_key_mlp(block),
            "orig_full": (n_writers(n_layers, n_heads),),
            "legal": mlp_legal,
            "ac": (mlp_legal + 1,),
            "transpose": False,
        }
    total = n_writers(n_layers, n_heads)
    spec[ac_key_resid(n_layers)] = {
        "orig_key": ORIG_KEY_FINAL,
        "orig_full": (total,),
        "legal": total,
        "ac": (total + 1,),
        "transpose": False,
    }
    return spec


def total_edges_autocircuit(n_layers: int, n_heads: int) -> int:
    return sum(
        int(t.tensor(s["ac"]).prod()) for s in expected_shapes(n_layers, n_heads).values()
    )


def orig_to_ac(
    orig: Dict[str, t.Tensor],
    n_layers: int,
    n_heads: int,
    embed_fill: float,
    strict: bool = True,
) -> Dict[str, t.Tensor]:
    """Convert original read-mask tensors to AutoCircuit module-keyed tensors.

    Values pass through untouched; only the layout changes. ``embed_fill`` is
    written into AutoCircuit's embed column, which the original does not model --
    callers decide what "always kept" means on their scale (1.0 for a keep mask,
    0.0 for an ablation mask).
    """
    out: Dict[str, t.Tensor] = {}
    for ac_key, s in expected_shapes(n_layers, n_heads).items():
        tensor = orig[s["orig_key"]]
        if strict and tuple(tensor.shape) != s["orig_full"]:
            raise ValueError(
                f"{s['orig_key']}: expected {s['orig_full']}, got {tuple(tensor.shape)}"
            )
        legal = tensor[: s["legal"]]
        legal = legal.T.contiguous() if s["transpose"] else legal.clone()

        embed = t.full(
            (*legal.shape[:-1], 1), embed_fill, dtype=legal.dtype, device=legal.device
        )
        merged = t.cat([embed, legal], dim=-1)
        if strict and tuple(merged.shape) != s["ac"]:
            raise ValueError(f"{ac_key}: built {tuple(merged.shape)}, expected {s['ac']}")
        out[ac_key] = merged
    return out


def ac_to_orig(
    ac: Dict[str, t.Tensor],
    n_layers: int,
    n_heads: int,
    illegal_fill: float,
    strict: bool = True,
) -> Dict[str, t.Tensor]:
    """Inverse of :func:`orig_to_ac`, dropping AutoCircuit's embed column.

    The original's tensors are full writer width, so the causally illegal entries
    have to be given some value. They are multiplied by ``*_read_common_mask``
    before use, so the value cannot affect the forward pass; ``illegal_fill`` makes
    the choice explicit and reproducible instead of leaving uninitialised memory.
    """
    out: Dict[str, t.Tensor] = {}
    for ac_key, s in expected_shapes(n_layers, n_heads).items():
        tensor = ac[ac_key]
        if strict and tuple(tensor.shape) != s["ac"]:
            raise ValueError(f"{ac_key}: expected {s['ac']}, got {tuple(tensor.shape)}")
        body = tensor[..., 1:]  # drop the embed column
        body = body.T.contiguous() if s["transpose"] else body.clone()

        full = t.full(
            s["orig_full"], illegal_fill, dtype=tensor.dtype, device=tensor.device
        )
        full[: s["legal"]] = body
        out[s["orig_key"]] = full
    return out


# --- node (write) masks ------------------------------------------------------


def node_orig_to_ac(
    orig: Dict[str, t.Tensor], n_layers: int, n_heads: int, embed_fill: float
) -> t.Tensor:
    """Per-block write log-alphas → one vector indexed by AutoCircuit ``src_idx``."""
    parts: List[t.Tensor] = []
    for block in range(n_layers):
        parts.append(orig[orig_key_attn_write(block)].reshape(-1))
        parts.append(orig[orig_key_mlp_write(block)].reshape(-1))
    body = t.cat(parts)
    assert body.numel() == n_writers(n_layers, n_heads)
    embed = t.full((1,), embed_fill, dtype=body.dtype, device=body.device)
    return t.cat([embed, body])


def node_ac_to_orig(
    vec: t.Tensor, n_layers: int, n_heads: int
) -> Dict[str, t.Tensor]:
    """Inverse of :func:`node_orig_to_ac`; ``vec[0]`` (embed) is dropped."""
    assert vec.numel() == n_sources(n_layers, n_heads)
    body = vec[1:]
    out: Dict[str, t.Tensor] = {}
    for block in range(n_layers):
        start = attn_writer_offset(block, n_heads)
        out[orig_key_attn_write(block)] = body[start : start + n_heads].clone()
        out[orig_key_mlp_write(block)] = body[
            start + n_heads : start + n_heads + 1
        ].clone()
    return out


# --- convenience for Tier C --------------------------------------------------


def read_log_alphas(model, n_layers: int) -> Dict[str, t.Tensor]:
    """Harvest the original's read log-alphas from a live model, prefix-stripped."""
    prefix = "transformer."
    wanted = {"read_log_alphas"}
    out: Dict[str, t.Tensor] = {}
    for name, param in model.named_parameters():
        short = name[len(prefix):] if name.startswith(prefix) else name
        if any(w in short for w in wanted):
            out[short] = param.detach().float().cpu()
    return out


def write_log_alphas(model, n_layers: int) -> Dict[str, t.Tensor]:
    prefix = "transformer."
    out: Dict[str, t.Tensor] = {}
    for name, param in model.named_parameters():
        short = name[len(prefix):] if name.startswith(prefix) else name
        if "write_log_alpha" in short:
            out[short] = param.detach().float().cpu()
    return out


def _strip_embedding_writers(
    read: Dict[str, t.Tensor], n_embed: int
) -> Tuple[Dict[str, t.Tensor], Dict[str, t.Tensor]]:
    """Split the reference's embedding-writer rows off its read masks.

    Only relevant for the ``orig-withembed`` arm. Run with
    ``--with_embedding_nodes`` the reference has ``n_embed = 2`` extra writers
    (``tok_embeds``, ``pos_embeds``) at the front of every read mask's source axis,
    against AutoCircuit's single combined embed source. There is no bijection, so
    the study's primary arm drops the flag entirely; this path exists only to
    measure what the flag costs.

    Reduction rule, fixed in the spec rather than chosen here: AutoCircuit's embed
    edge is kept if the reference keeps *either* of its embedding edges, i.e. the
    elementwise maximum of the two keep values.
    """
    body: Dict[str, t.Tensor] = {}
    embed: Dict[str, t.Tensor] = {}
    for key, tensor in read.items():
        embed[key] = tensor[:n_embed].max(dim=0).values
        body[key] = tensor[n_embed:]
    return body, embed


def log_alphas_to_prune_scores(
    read: Dict[str, t.Tensor],
    write: Dict[str, t.Tensor],
    n_layers: int,
    n_heads: int,
    fold_node_masks: bool = True,
    n_embed_writers: int = 0,
) -> Dict[str, t.Tensor]:
    """Original log-alphas → AutoCircuit ``PruneScores`` (higher = keep).

    The score is ``σ(log_alpha / T · F)``, the same monotone transform the port
    exports, so both arms land on one scale and rank edges identically to the
    original's own thresholding (see ``thesis_notes/limitations.md``, 2026-07-23).
    Node masks are folded in as the port does, ``keep = keep_edge · keep_node``,
    so the two arms' scores mean the same thing.

    Without separate embedding writers, converted reference scores keep the embedding
    edges, matching unconditional addition. The port's trained scores remain learned.
    """
    from auto_circuit.utils.l0_utils import FACTOR, TEMPERATURE

    keep = {k: t.sigmoid(v / TEMPERATURE * FACTOR) for k, v in read.items()}
    embed_keep: Dict[str, t.Tensor] = {}
    if n_embed_writers:
        keep, embed_keep = _strip_embedding_writers(keep, n_embed_writers)

    scores = orig_to_ac(keep, n_layers, n_heads, embed_fill=1.0)
    for ac_key, spec in expected_shapes(n_layers, n_heads).items():
        if spec["orig_key"] in embed_keep:
            # Reducing over the embedding writers already removed that axis, so the
            # column is (n_head,) for Q/K/V and scalar for MLP/final -- exactly the
            # shape of `[..., 0]` in both cases, with no transpose needed.
            scores[ac_key][..., 0] = embed_keep[spec["orig_key"]]

    if not fold_node_masks:
        return scores

    write_body = {
        k: v for k, v in write.items() if ("attn_write" in k or "mlp_write" in k)
    }
    node_keep = t.sigmoid(
        node_orig_to_ac(write_body, n_layers, n_heads, embed_fill=1e9)
        / TEMPERATURE
        * FACTOR
    )
    for ac_key, spec in expected_shapes(n_layers, n_heads).items():
        width = spec["ac"][-1]
        scores[ac_key] = scores[ac_key] * node_keep[:width]
    return scores
