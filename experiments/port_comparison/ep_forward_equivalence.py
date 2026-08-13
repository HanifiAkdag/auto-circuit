"""Tier B of the Edge Pruning port fidelity study: same-mask function equivalence.

Given the *same* circuit, do the two implementations compute the same function?
This tier carries the fidelity argument because it is deterministic: unlike a
training comparison, it has a right answer.

Edge Pruning uses **resample** ablation, so the decomposition argument from the
DiscoGP study does not transfer and is re-derived here. The original interpolates
rather than subtracts (``modeling_fpt2.py:555-571``)::

    x_read = Σ_w [ z_w · y_w  +  (1 − z_w) · y_w^corrupt ]

AutoCircuit computes ``Σ_w y_w^clean + Σ_w mask_w · (y_w^corrupt − y_w^clean)``,
which is the same expression with ``mask = 1 − z``. They coincide only if the two
decompositions of the residual stream agree term by term, which is what this script
measures.

**The attention output bias.** The original folds ``b_O / n_heads`` into each head
inside the masked path (``_apply_c_proj``, ``modeling_fpt2.py:403-414``); AutoCircuit
adds ``b_O`` once, outside. Under *zero* ablation that was the entire divergence of
the DiscoGP study. Under *resample* it should cancel, because every writer
contributes its full value (bias share included) in both the clean and the corrupt
branch::

    Σ_h [ z(y_h + b_O/H) + (1−z)(y_h^corr + b_O/H) ] = Σ_h y_h^{...} + b_O

That is a pre-registered prediction, and this script is how it gets tested.

Two arms, because a test that bypasses a code path says nothing about that path:

* ``raw``      - identical ``input_ids`` to both models. Isolates the masking math.
* ``pipeline`` - each side builds its inputs through its own real data path
  (AutoCircuit's ``PromptDataLoader``; the original's ``DataCollatorIOI``), so the
  BOS/padding conventions and the dataloader are inside the test rather than
  excluded from it.

Run::

    srun --gres=gpu:1 --mem=48G python -m experiments.port_comparison.ep_forward_equivalence
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch as t

REPO = Path(__file__).resolve().parents[2]
_EP_ROOT = REPO / "third_party" / "edge_pruning"
for _extra in (REPO, _EP_ROOT / "src" / "modeling", _EP_ROOT / "src" / "prune"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from auto_circuit.experiment_utils import load_tl_model  # noqa: E402
from auto_circuit.types import AblationType  # noqa: E402
from auto_circuit.utils.ablation_activations import src_ablations  # noqa: E402
from auto_circuit.utils.graph_utils import patch_mode, patchable_model  # noqa: E402
from auto_circuit.utils.misc import repo_path_to_abs_path  # noqa: E402

from experiments.port_comparison.ep_edge_map import (  # noqa: E402
    ac_to_orig,
    expected_shapes,
    n_sources,
    orig_key_attn_write,
    orig_key_mlp_write,
    total_edges_autocircuit,
    total_edges_original,
)

MODEL = "gpt2"

# +/- this in log-alpha space, then threshold at 0.5, gives exactly binary masks:
# sigma(1.2 * 10) = 0.999994 > 0.5 and sigma(-12) = 6e-06 < 0.5.
LOGIT_MAG = 10.0
DET_THRESHOLD = 0.5


# --- model construction ------------------------------------------------------


def build_original(device: t.device, dtype=t.float32, with_embedding_nodes: bool = False):
    """The reference GPT-2, with masks forced deterministic and binary."""
    from modeling_fpt2 import FPT2LMHeadModel

    model = FPT2LMHeadModel.from_pretrained(
        MODEL, with_embedding_nodes=with_embedding_nodes
    )
    model.reset_all_log_alphas()
    model.set_edge_threshold_for_deterministic(DET_THRESHOLD)
    model.set_node_threshold_for_deterministic(DET_THRESHOLD)
    return model.to(device, dtype=dtype).eval()


def install_original_mask(
    model, keep_ac: Dict[str, t.Tensor], n_layers: int, n_heads: int
) -> None:
    """Force an exact binary circuit into the reference.

    The original samples masks inside ``forward``, so a mask cannot simply be passed
    in. Writing +/-``LOGIT_MAG`` into the log-alphas and thresholding the
    deterministic mask at 0.5 makes ``get_mask`` return exactly 0/1.

    Causally illegal entries are set to "drop". They are multiplied by
    ``*_read_common_mask`` before use so they cannot affect the forward pass, but
    they *do* take part in ``deterministic_z``'s bottom-k count, and making them
    losers keeps that count aligned with the real drops.
    """
    orig = ac_to_orig(keep_ac, n_layers, n_heads, illegal_fill=0.0)
    params = dict(model.named_parameters())
    with t.no_grad():
        for short, keep in orig.items():
            param = params[f"transformer.{short}"]
            param.copy_(
                t.where(keep.to(param.device) > 0.5, LOGIT_MAG, -LOGIT_MAG).to(param.dtype)
            )
        # Node/write masks stay fully on: this tier tests the edge decomposition.
        for block in range(n_layers):
            for key in (orig_key_attn_write(block), orig_key_mlp_write(block)):
                params[f"transformer.{key}"].fill_(LOGIT_MAG)


def original_writer_states(model, tokens: t.Tensor) -> t.Tensor:
    """``corr_x`` as the reference's own trainer builds it (``fpt2_ioi.py:139-143``)."""
    with t.no_grad():
        states = model(input_ids=tokens, output_writer_states=True).writer_states
    return states.transpose(0, 1)


def orig_logits(model, tokens: t.Tensor, corr_x: t.Tensor, idx) -> t.Tensor:
    with t.no_grad():
        out = model(input_ids=tokens, corr_x=corr_x)
    logits = out.logits
    if isinstance(idx, int):
        return logits[:, idx].float()
    return logits[t.arange(logits.shape[0], device=logits.device), idx].float()


# --- AutoCircuit side --------------------------------------------------------


def ac_logits(
    ac_model, tokens: t.Tensor, keep_ac: Dict[str, t.Tensor], ablations: t.Tensor
) -> t.Tensor:
    """Run AutoCircuit with the given keep-mask, through its ordinary patching path."""
    with t.no_grad():
        for wrapper in ac_model.dest_wrappers:
            keep = keep_ac[wrapper.module_name].to(wrapper.patch_mask.device)
            wrapper.patch_mask.data = (1.0 - keep).to(wrapper.patch_mask.dtype)
        with patch_mode(ac_model, ablations):
            out = ac_model(tokens)[ac_model.out_slice]
    return out.float()


def random_keep_masks(
    ac_model, density: float, gen: t.Generator, n_heads: int
) -> Dict[str, t.Tensor]:
    """i.i.d. Bernoulli(density) keep-masks, with the embed column forced kept.

    The embed column has no counterpart in the reference run without
    ``--with_embedding_nodes``, where the embeddings are added unconditionally at
    every read site. Forcing it kept is what makes the two graphs correspond.
    """
    masks: Dict[str, t.Tensor] = {}
    for wrapper in ac_model.ordered_dest_wrappers:
        shape = wrapper.patch_mask.shape
        keep = (
            t.rand(shape, generator=gen, device="cpu") < density
        ).float().to(wrapper.patch_mask.device)
        keep[..., 0] = 1.0  # embed source
        masks[wrapper.module_name] = keep
    return masks


def all_keep_masks(ac_model) -> Dict[str, t.Tensor]:
    return {
        w.module_name: t.ones_like(w.patch_mask) for w in ac_model.ordered_dest_wrappers
    }


# --- measurement -------------------------------------------------------------


def logprobs(logits: t.Tensor) -> t.Tensor:
    """Compare in log-probability space, not raw logits.

    ``load_tl_model`` builds the model with ``center_unembed=True``, so AutoCircuit's
    logits are centred per position while the reference's HF GPT-2 are not. The gap
    is a per-position constant: it leaves argmax untouched and vanishes under
    softmax, but it swamps a raw-logit comparison (measured: 114 against a logit
    scale of 21). Log-probabilities are also what both objectives consume, since
    both are KL divergences.
    """
    return t.log_softmax(logits.float(), dim=-1)


def deviation(a: t.Tensor, b: t.Tensor) -> Dict[str, float]:
    la, lb = logprobs(a), logprobs(b)
    return {
        "max_abs": float((la - lb).abs().max()),
        "mean_abs": float((la - lb).abs().mean()),
        "logprob_scale": float(la.abs().max()),
        "raw_logit_max_abs": float((a - a.mean(-1, keepdim=True)
                                    - (b - b.mean(-1, keepdim=True))).abs().max()),
        "argmax_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
    }


def load_frozen_prompts(n_prompts: int) -> Tuple[List[str], List[str]]:
    """Clean and corrupt prompt strings from the frozen split."""
    path = repo_path_to_abs_path("datasets/port_fidelity/ioi_frozen.json")
    data = json.loads(path.read_text())
    rows = data["prompts"][:n_prompts]
    return [r["clean"] for r in rows], [r["corrupt"] for r in rows]


def raw_tokens(n_prompts: int, device: t.device) -> Tuple[t.Tensor, t.Tensor]:
    """Matched clean/corrupt token ids at a single uniform length.

    Choosing one length means position -1 is the scored position for both
    implementations, so this arm isolates the masking math from every tokenisation
    difference. The `pipeline` arm is what puts those differences back in.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    clean_s, corrupt_s = load_frozen_prompts(n_prompts * 4)

    by_len: Dict[int, List[Tuple[List[int], List[int]]]] = {}
    for c, x in zip(clean_s, corrupt_s):
        ci, xi = tok(c)["input_ids"], tok(x)["input_ids"]
        if len(ci) != len(xi):
            continue
        by_len.setdefault(len(ci), []).append((ci, xi))
    length = max(by_len, key=lambda k: len(by_len[k]))
    pairs = by_len[length][:n_prompts]
    clean = t.tensor([p[0] for p in pairs], device=device)
    corrupt = t.tensor([p[1] for p in pairs], device=device)
    return clean, corrupt


def pipeline_inputs(
    tl_model, n_prompts: int, device: t.device, prepend_bos: bool = True
):
    """Build each side's inputs through its own real data path.

    The DiscoGP study's Tier B fed raw token ids to both models to neutralise the
    BOS and padding conventions, and thereby excluded the dataloader from the test.
    A real defect was hiding there. This arm keeps both data paths inside the test:
    AutoCircuit's ``load_datasets_from_json`` (BOS prepended, left padding, KV-cache
    prefix logic) and the reference's ``DataCollatorIOI`` (no BOS, right padding to
    ``max_seq_length``, no attention mask).

    ``shuffle=False`` is what makes the two orderings correspond row for row, so the
    comparison stays paired.
    """
    from transformers import AutoTokenizer

    from auto_circuit.data import load_datasets_from_json
    from fpt2_ioi import DataCollatorIOI

    path = repo_path_to_abs_path("datasets/port_fidelity/ioi_frozen.json")
    train_loader, _ = load_datasets_from_json(
        model=tl_model,
        path=path,
        device=device,
        batch_size=n_prompts,
        train_test_size=(n_prompts, n_prompts),
        shuffle=False,
        random_seed=0,
        prepend_bos=prepend_bos,
    )
    batch = next(iter(train_loader))

    rows = json.loads(path.read_text())["prompts"][:n_prompts]
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    examples = [
        {
            "ioi_sentences": r["clean"] + r["answers"][0],
            "corr_ioi_sentences": r["corrupt"] + r["answers"][0],
        }
        for r in rows
    ]
    collated = DataCollatorIOI(tok, max_length=64)(examples)
    return batch, {k: v.to(device) for k, v in collated.items()}


def run_pipeline_arm(
    ac_model, orig_model, orig_full, batch, collated, densities, n_masks, seed,
    n_layers, n_heads,
) -> Tuple[List[dict], dict]:
    """Same comparison as `run_arm`, but each side scored at its *own* index."""
    records: List[dict] = []
    gen = t.Generator().manual_seed(seed)

    ablations = src_ablations(ac_model, batch.corrupt, AblationType.RESAMPLE)
    corr_x = original_writer_states(orig_full, collated["corr_input_ids"])
    start = collated["start_idxes"]

    keep_all = all_keep_masks(ac_model)
    install_original_mask(orig_model, keep_all, n_layers, n_heads)
    a_full = ac_logits(ac_model, batch.clean, keep_all, ablations)
    o_full = orig_logits(orig_model, collated["input_ids"], corr_x, start)
    full_density = deviation(a_full, o_full)

    for density in densities:
        for i in range(n_masks):
            keep = random_keep_masks(ac_model, density, gen, n_heads)
            install_original_mask(orig_model, keep, n_layers, n_heads)
            a = ac_logits(ac_model, batch.clean, keep, ablations)
            o = orig_logits(orig_model, collated["input_ids"], corr_x, start)
            rec = {"density": density, "mask": i}
            rec.update(deviation(a, o))
            records.append(rec)

    by_density = {}
    for density in densities:
        vals = sorted(r["max_abs"] for r in records if r["density"] == density)
        by_density[str(density)] = vals[len(vals) // 2]

    return records, {
        "full_density_max_abs": full_density["max_abs"],
        "full_density_mean_abs": full_density["mean_abs"],
        "full_density_argmax_agree": full_density["argmax_agree"],
        "logprob_scale": full_density["logprob_scale"],
        "worst_max_abs": max(r["max_abs"] for r in records),
        "median_mean_abs": sorted(r["mean_abs"] for r in records)[len(records) // 2],
        "median_max_abs": sorted(r["max_abs"] for r in records)[len(records) // 2],
        "min_argmax_agreement": min(r["argmax_agree"] for r in records),
        "median_max_abs_by_density": by_density,
        "ac_seq_len": int(batch.clean.shape[1]),
        "ref_seq_len": int(collated["input_ids"].shape[1]),
        # Fingerprints, so two arms that should differ cannot silently report the
        # same numbers because one of them reused the other's inputs.
        "ac_logits_fingerprint": float(a_full.abs().sum()),
        "ref_logits_fingerprint": float(o_full.abs().sum()),
        "ac_first_row_ids": batch.clean[0, :4].tolist(),
    }


def run_arm(
    ac_model,
    orig_model,
    orig_full,
    clean: t.Tensor,
    corrupt: t.Tensor,
    densities: Tuple[float, ...],
    n_masks: int,
    seed: int,
    n_layers: int,
    n_heads: int,
    ac_prepend_bos: bool = False,
) -> Tuple[List[dict], dict]:
    """``ac_prepend_bos`` gives *only* AutoCircuit a leading BOS, on prompts that are
    already uniform length. That isolates the BOS convention from padding, which
    matters because the two cannot be separated in a padded batch (see
    ``pipeline_inputs``)."""
    records: List[dict] = []
    gen = t.Generator().manual_seed(seed)

    if ac_prepend_bos:
        bos = t.full((clean.shape[0], 1), 50256, dtype=clean.dtype, device=clean.device)
        ac_clean = t.cat([bos, clean], dim=1)
        ac_corrupt = t.cat([bos, corrupt], dim=1)
    else:
        ac_clean, ac_corrupt = clean, corrupt

    ablations = src_ablations(ac_model, ac_corrupt, AblationType.RESAMPLE)
    corr_x = original_writer_states(orig_full, corrupt)

    # --- reference 1: full density. Nothing is ablated, so any gap here is pure
    # float arithmetic between two different implementations of GPT-2.
    keep_all = all_keep_masks(ac_model)
    install_original_mask(orig_model, keep_all, n_layers, n_heads)
    a_full = ac_logits(ac_model, ac_clean, keep_all, ablations)
    o_full = orig_logits(orig_model, clean, corr_x, -1)
    full_density = deviation(a_full, o_full)

    # --- reference 2: within-implementation nondeterminism. One implementation run
    # twice on a permuted batch, un-permuted before comparing: reduction-order noise
    # only. This is the floor the deviations below are read against.
    perm = t.randperm(clean.shape[0], device=clean.device)
    abl_perm = src_ablations(ac_model, ac_corrupt[perm], AblationType.RESAMPLE)
    a_perm = ac_logits(ac_model, ac_clean[perm], keep_all, abl_perm)
    nondet = deviation(a_full, a_perm[t.argsort(perm)])

    # --- the sweep
    for density in densities:
        for i in range(n_masks):
            keep = random_keep_masks(ac_model, density, gen, n_heads)
            install_original_mask(orig_model, keep, n_layers, n_heads)
            a = ac_logits(ac_model, ac_clean, keep, ablations)
            o = orig_logits(orig_model, clean, corr_x, -1)
            rec = {"density": density, "mask": i}
            rec.update(deviation(a, o))
            records.append(rec)

    by_density = {}
    for density in densities:
        vals = [r["max_abs"] for r in records if r["density"] == density]
        vals.sort()
        by_density[str(density)] = vals[len(vals) // 2]

    summary = {
        "full_density_max_abs": full_density["max_abs"],
        "full_density_argmax_agree": full_density["argmax_agree"],
        "within_impl_nondeterminism_max_abs": nondet["max_abs"],
        "logprob_scale": full_density["logprob_scale"],
        "worst_max_abs": max(r["max_abs"] for r in records),
        "median_max_abs": sorted(r["max_abs"] for r in records)[len(records) // 2],
        "min_argmax_agreement": min(r["argmax_agree"] for r in records),
        "median_max_abs_by_density": by_density,
    }
    return records, summary


def single_edge_probes(
    ac_model, orig_model, orig_full, clean, corrupt, n_layers, n_heads,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> Dict[str, float]:
    """Ablate exactly one edge, chosen by *source kind*.

    Index 0 of any mask is always the embedding source, which carries no ``b_O``, so
    probes picked by position alone would silently test nothing. This is the probe
    that decides the ``b_O`` prediction: a head source is the only kind whose value
    differs between the two decompositions.
    """
    ablations = src_ablations(ac_model, corrupt, ablation_type)
    corr_x = original_writer_states(orig_full, corrupt)
    if ablation_type == AblationType.BATCH_TOKENWISE_MEAN:
        # Mirror `src_out_hook`'s BATCH_TOKENWISE_MEAN on the reference side: the mean
        # over the batch at each token position. `corr_x` is (batch, writers, seq, d)
        # after `original_writer_states` transposes it.
        corr_x = corr_x.mean(dim=0, keepdim=True).repeat(
            corr_x.shape[0], *([1] * (corr_x.ndim - 1))
        )
    dest = f"blocks.{n_layers - 1}.hook_mlp_in"
    probes = {"embed_src": 0, "head_src_layer0": 1, "mlp_src_layer0": n_heads + 1,
              "head_src_layer10": 1 + 10 * (n_heads + 1)}
    out: Dict[str, float] = {}
    for name, idx in probes.items():
        keep = all_keep_masks(ac_model)
        keep[dest][idx] = 0.0
        install_original_mask(orig_model, keep, n_layers, n_heads)
        a = ac_logits(ac_model, clean, keep, ablations)
        o = orig_logits(orig_model, clean, corr_x, -1)
        out[name] = deviation(a, o)["max_abs"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--n-masks", type=int, default=40)
    ap.add_argument("--densities", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/port_fidelity/ep_forward_equivalence")
    args = ap.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    tl = load_tl_model(MODEL, device)
    ac_model = patchable_model(
        tl, factorized=True, slice_output="last_seq", separate_qkv=True, device=device
    )
    n_layers, n_heads = tl.cfg.n_layers, tl.cfg.n_heads

    assert total_edges_autocircuit(n_layers, n_heads) == ac_model.n_edges, (
        total_edges_autocircuit(n_layers, n_heads), ac_model.n_edges
    )
    assert len(ac_model.srcs) == n_sources(n_layers, n_heads)
    for ac_key, spec in expected_shapes(n_layers, n_heads).items():
        assert tuple(ac_model.patch_masks[ac_key].shape) == spec["ac"], ac_key
    edge_gap = ac_model.n_edges - total_edges_original(n_layers, n_heads)
    print(f"AutoCircuit edges {ac_model.n_edges}, reference edges "
          f"{total_edges_original(n_layers, n_heads)}, gap {edge_gap} (= n_readers)")

    orig_model = build_original(device)
    orig_full = build_original(device)  # unmasked, supplies corr_x

    clean, corrupt = raw_tokens(args.n_prompts, device)
    print(f"raw arm: {clean.shape[0]} prompts of length {clean.shape[1]}")

    probes = single_edge_probes(
        ac_model, orig_model, orig_full, clean, corrupt, n_layers, n_heads
    )
    print(f"single-edge probes: {probes}")

    records, summary = run_arm(
        ac_model, orig_model, orig_full, clean, corrupt,
        tuple(args.densities), args.n_masks, args.seed, n_layers, n_heads,
    )
    summary["single_edge_max_abs"] = probes

    # BOS isolated: uniform-length prompts, so no padding is involved on either side.
    bos_records, bos_summary = run_arm(
        ac_model, orig_model, orig_full, clean, corrupt,
        tuple(args.densities), args.n_masks, args.seed, n_layers, n_heads,
        ac_prepend_bos=True,
    )
    print("raw-bos arm (AutoCircuit prepends BOS, no padding anywhere):")
    print(json.dumps(bos_summary, indent=2))
    summary["edge_counts"] = {
        "autocircuit": ac_model.n_edges,
        "reference": total_edges_original(n_layers, n_heads),
        "gap": edge_gap,
    }
    print(json.dumps(summary, indent=2))

    # --- pipeline arms: each side through its own dataloader.
    # The no-BOS variant attributes whatever gap the first one shows: if it is the
    # BOS/left-pad convention (difference 5) and not the masking machinery, dropping
    # BOS must collapse it to the raw arm's floor.
    for rec in records:
        rec["arm"] = "raw"
    for rec in bos_records:
        rec["arm"] = "raw-bos"
    all_records = list(records) + list(bos_records)
    arm_summaries = {"raw": summary, "raw-bos": bos_summary}

    for arm_name, prepend_bos in (("pipeline", True),):
        batch, collated = pipeline_inputs(tl, args.n_prompts, device, prepend_bos)
        pipe_records, pipe_summary = run_pipeline_arm(
            ac_model, orig_model, orig_full, batch, collated,
            tuple(args.densities), args.n_masks, args.seed, n_layers, n_heads,
        )
        print(f"{arm_name} arm:")
        print(json.dumps(pipe_summary, indent=2))
        for rec in pipe_records:
            rec["arm"] = arm_name
        all_records += pipe_records
        arm_summaries[arm_name] = pipe_summary

    out_dir = repo_path_to_abs_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_records for k in r})
    with open(out_dir / "deviations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_records)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"n_prompts": int(clean.shape[0]), "n_masks_per_density": args.n_masks,
             "arms": arm_summaries},
            f, indent=2,
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
