"""Tier B of the DiscoGP port fidelity study: same-mask function equivalence.

Given the *same* circuit, do the two implementations compute the same function?
This tier carries the fidelity argument because it is deterministic: unlike a
training comparison, it has a right answer.

The two implementations decompose the residual stream differently in one respect.
The original folds the attention output bias into each head
(``modules/attention.py:150`` adds ``b_O / n_heads`` per head), so ablating a head
edge also removes ``b_O / n_heads``. AutoCircuit's ``hook_result`` excludes
``b_O``, which is added once outside the maskable decomposition, so ablating a
head edge leaves the full ``b_O`` in place. The two agree at full density (the
sums are identical) and diverge under masking.

That difference is real and not a port defect, so the script measures two arms:

* ``bias_intact`` - both models as shipped. Quantifies the divergence.
* ``b_O_zeroed`` - ``b_O`` set to zero in *both* models, which removes the term
  by construction. Any remaining deviation here would be a genuine port defect,
  so this arm is the actual test of the bijection and the mask semantics.

Deviations are only interpretable against a scale, so two references are
established: the full-density float floor, and the within-implementation
nondeterminism from re-running one implementation on a permuted batch.

Run::

    srun --gres=gpu:1 --mem=48G python -m experiments.port_comparison.forward_equivalence
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch as t

from ..config import ROOT as REPO, reference_root
if str(reference_root("disco_gp")) not in sys.path:
    sys.path.insert(0, str(REPO / "third_party" / "disco_gp"))

from auto_circuit.experiment_utils import load_tl_model  # noqa: E402
from auto_circuit.prune_algos.disco_gp import _forward_with_masks  # noqa: E402
from auto_circuit.types import AblationType  # noqa: E402
from auto_circuit.utils.ablation_activations import src_ablations  # noqa: E402
from auto_circuit.utils.graph_utils import patchable_model  # noqa: E402

from .edge_map import (  # noqa: E402
    ac_to_orig,
    expected_shapes,
    total_edges,
)

MODEL = "gpt2"

# TransformerLens version compat: the original targets TL 2.x, this env runs 1.19.
# Standard GPT-2 values, so the arithmetic is unchanged.
TL_COMPAT = {
    "attn_scale": float(np.sqrt(64)),  # sqrt(d_head)
    "use_qk_norm": False,  # TL 2.x feature; GPT-2 does not use QK-norm
    "init_range": 0.02,  # unused: init_weights=False
}


# --- model construction ------------------------------------------------------


def build_original(device: t.device, dtype=t.float32):
    """Build the original DiscoGPTransformer without invoking its data pipeline.

    `DiscoGPTransformer.from_pretrained` also runs `setup_task`, which builds the
    IOI dataset (and carries its answer-token defect). Tier B needs only the
    model, so just the weight-loading half of that classmethod is reproduced.
    """
    from disco_gp import Config, DiscoGPTransformer
    from transformer_lens import HookedTransformer

    model_cfg = Config.from_tl(MODEL, dtype=dtype)
    for key, value in TL_COMPAT.items():
        if key not in model_cfg._values:
            setattr(model_cfg, key, value)

    cfg = Config.from_configs(
        model=model_cfg,
        weight=Config(use_weight_masks=False, gs_temp_weight=0.01, logits_w_init=1.0),
        edge=Config(use_edge_masks=True, gs_temp_edge=1.0, logits_e_init=1.0),
        exp=Config(print_results=False),
    )

    model = DiscoGPTransformer(cfg)
    state_dict = HookedTransformer.from_pretrained(MODEL).state_dict()
    model.hook_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.setup_mask()
    return model.to(device, dtype=dtype).eval()


def zero_attn_out_bias(tl_model, orig_model) -> None:
    """Remove `b_O` from both models, eliminating the decomposition difference."""
    with t.no_grad():
        for block in tl_model.blocks:
            block.attn.b_O.zero_()
        for block in orig_model.blocks:
            block.attn.b_O.zero_()


def split_attn_out_bias(tl_model) -> None:
    """Give each attention head its own share of `b_O`, as the original does.

    TransformerLens computes ``result = hook_result(z @ W_O)`` per head and adds
    ``b_O`` only afterwards, so ``b_O`` sits outside the part AutoCircuit can mask.
    Moving ``b_O / n_heads`` into ``hook_result`` and zeroing ``b_O`` leaves the
    full-model output unchanged (the shares sum back to ``b_O``) but makes each
    head's *contribution* carry its share, so ablating a head removes it. That
    reproduces `modules/attention.py:150` of the original.

    Not the default: this changes what an edge means for every algorithm, so
    enabling it globally would break comparability with the spine results and with
    the convention the rest of the literature uses. Opt in per experiment.

    Works before or after `patchable_model`: once wrapped, `hook_result` is a
    `PatchWrapperImpl` whose `.module` is the real `HookPoint`. The wrapper calls
    that inner module and records *its* output as the head's source activation, so
    the hook must go on the inner one.
    """
    n_heads = tl_model.cfg.n_heads
    with t.no_grad():
        for block in tl_model.blocks:
            share = (block.attn.b_O.data / n_heads).clone()
            block.attn.b_O.data.zero_()
            hook_point = getattr(block.attn.hook_result, "module", block.attn.hook_result)
            hook_point.add_perma_hook(lambda tensor, hook, s=share: tensor + s)


def load_tokens(n_prompts: int, device: t.device) -> t.Tensor:
    """Uniform-length IOI prompts, so position -1 is the answer position for both.

    Feeding identical token ids sidesteps the BOS / padding-side difference by
    construction; that difference is a Tier C factor, not a Tier B one.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = json.loads((REPO / "datasets" / "ioi" / "ioi_prompts.json").read_text())["prompts"]

    by_len: Dict[int, List[List[int]]] = {}
    for p in prompts:
        ids = tok(p["clean"])["input_ids"]
        by_len.setdefault(len(ids), []).append(ids)
    length = max(by_len, key=lambda k: len(by_len[k]))
    chosen = by_len[length][:n_prompts]
    print(f"[data] {len(chosen)} prompts of uniform length {length}")
    return t.tensor(chosen, device=device)


# --- forward passes ----------------------------------------------------------


def install_original_mask(orig_model, keep_orig: Dict[str, t.Tensor]) -> None:
    """Force an exact binary mask into the original via deterministic thresholding.

    The original samples masks inside `forward`, so an arbitrary mask cannot be
    passed in. Setting logits to +1/-1 and enabling `deterministic=True` makes
    `torch.where(logits > 0, 1., 0.)` reproduce the requested mask exactly.
    """
    with t.no_grad():
        for name, param in orig_model.named_parameters():
            if "edge" not in name:
                continue
            keep = keep_orig[name].to(param.device)
            param.copy_(t.where(keep > 0.5, 1.0, -1.0).to(param.dtype))
    orig_model.turn_on_edge_masks(deterministic=True, reverse=False)


def orig_logits(orig_model, tokens: t.Tensor) -> t.Tensor:
    with t.no_grad():
        return orig_model(tokens)[0][:, -1].float()


def ac_logits(ac_model, tokens, keep_ac: Dict[str, t.Tensor], ablations) -> t.Tensor:
    """AutoCircuit convention: patch_mask=1 ablates, so patch_mask = 1 - keep."""
    wrappers = {w.module_name: w for w in ac_model.ordered_dest_wrappers}
    masks = {name: (1.0 - keep_ac[name]).to(w.patch_mask.dtype) for name, w in wrappers.items()}
    with t.no_grad():
        return _forward_with_masks(
            ac_model, tokens, ablations, masks, wrappers, ac_model.out_slice
        ).float()


def random_keep_masks(ac_model, density: float, gen: t.Generator) -> Dict[str, t.Tensor]:
    return {
        w.module_name: (
            t.rand(w.patch_mask.shape, generator=gen, device=w.patch_mask.device) < density
        ).float()
        for w in ac_model.ordered_dest_wrappers
    }


def deviation(a: t.Tensor, b: t.Tensor) -> Dict[str, float]:
    d = (a - b).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "logit_scale": a.abs().max().item(),
        "argmax_agree": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
    }


# --- node decomposition ------------------------------------------------------


def compare_node_decomposition(tl_model, orig_model, tokens) -> Dict[str, float]:
    """Check every source node's contribution at full density.

    If the graphs agree edge-by-edge then each of the 157 node contributions must
    match. Where they do not, the gap is exactly what masking will expose.
    """
    n_layers, n_heads = tl_model.cfg.n_layers, tl_model.cfg.n_heads
    orig_model.turn_off_edge_masks()
    with t.no_grad():
        states = orig_model(tokens, return_states=True).float()  # [B, P, n_srcs, d_model]
        _, cache = tl_model.run_with_cache(tokens)

    nodes = [cache["blocks.0.hook_resid_pre"].float()]
    for block in range(n_layers):
        result = cache[f"blocks.{block}.attn.hook_result"].float()
        nodes.extend(result[:, :, h] for h in range(n_heads))
        nodes.append(cache[f"blocks.{block}.hook_mlp_out"].float())
    ac = t.stack(nodes, dim=2)

    head_gap_vs_bias, head_gap, mlp_gap = 0.0, 0.0, 0.0
    for block in range(n_layers):
        b_o_per_head = tl_model.blocks[block].attn.b_O.float() / n_heads
        for head in range(n_heads):
            idx = 1 + block * (n_heads + 1) + head
            gap = states[:, :, idx] - ac[:, :, idx]
            head_gap = max(head_gap, gap.abs().max().item())
            head_gap_vs_bias = max(head_gap_vs_bias, (gap - b_o_per_head).abs().max().item())
        mlp_idx = 1 + block * (n_heads + 1) + n_heads
        mlp_gap = max(mlp_gap, (states[:, :, mlp_idx] - ac[:, :, mlp_idx]).abs().max().item())

    return {
        "embed_node_gap": (states[:, :, 0] - ac[:, :, 0]).abs().max().item(),
        "mlp_node_gap": mlp_gap,
        "head_node_gap": head_gap,
        "head_node_gap_minus_b_O_over_n_heads": head_gap_vs_bias,
        "max_b_O_over_n_heads": max(
            (b.attn.b_O.float() / n_heads).abs().max().item() for b in tl_model.blocks
        ),
    }


# --- main --------------------------------------------------------------------


def run_arm(ac_model, orig_model, tokens, ablations, densities, n_masks, seed
            ) -> Tuple[List[dict], dict]:
    n_layers, n_heads = ac_model.cfg.n_layers, ac_model.cfg.n_heads
    records: List[dict] = []

    keep_all = {w.module_name: t.ones_like(w.patch_mask) for w in ac_model.ordered_dest_wrappers}
    install_original_mask(orig_model, ac_to_orig(keep_all, n_layers, n_heads))
    a_full = ac_logits(ac_model, tokens, keep_all, ablations)
    floor = deviation(a_full, orig_logits(orig_model, tokens))
    records.append({"kind": "B1_full_density", "density": 1.0, **floor})

    perm = t.randperm(tokens.shape[0], device=tokens.device)
    abl_perm = src_ablations(ac_model, tokens[perm], AblationType.ZERO)
    a_perm = ac_logits(ac_model, tokens[perm], keep_all, abl_perm)
    nondet = deviation(a_full, a_perm[t.argsort(perm)])
    records.append({"kind": "within_impl_nondeterminism", "density": 1.0, **nondet})

    # Single-edge probes, one per source *kind*. Ablating index 0 of a mask only
    # ever hits the embedding source, which carries no b_O, so the probes must be
    # chosen by source kind or they silently test nothing.
    mlp_dest = f"blocks.{n_layers - 1}.hook_mlp_in"
    probes = {"embed_src": 0, "head_src": 1, "mlp_src": n_heads + 1}
    for label, src_idx in probes.items():
        keep = {k: v.clone() for k, v in keep_all.items()}
        keep[mlp_dest][src_idx] = 0.0
        install_original_mask(orig_model, ac_to_orig(keep, n_layers, n_heads))
        dev = deviation(ac_logits(ac_model, tokens, keep, ablations),
                        orig_logits(orig_model, tokens))
        records.append({"kind": f"B2_single_edge_{label}", "density": 1.0, **dev})

    gen = t.Generator(device=tokens.device).manual_seed(seed)
    for density in densities:
        for i in range(n_masks):
            keep = random_keep_masks(ac_model, density, gen)
            install_original_mask(orig_model, ac_to_orig(keep, n_layers, n_heads))
            dev = deviation(ac_logits(ac_model, tokens, keep, ablations),
                            orig_logits(orig_model, tokens))
            records.append({"kind": "B2_masked", "density": density, "mask": i, **dev})

    masked = [r for r in records if r["kind"] == "B2_masked"]
    summary = {
        "b1_full_density_max_abs": floor["max_abs"],
        "within_impl_nondeterminism_max_abs": nondet["max_abs"],
        "logit_scale": floor["logit_scale"],
        "single_edge_max_abs": {
            label: next(r["max_abs"] for r in records if r["kind"] == f"B2_single_edge_{label}")
            for label in probes
        },
        "b2_worst_max_abs": max(r["max_abs"] for r in masked),
        "b2_median_max_abs": float(np.median([r["max_abs"] for r in masked])),
        "b2_min_argmax_agreement": min(r["argmax_agree"] for r in masked),
        "b3_median_max_abs_by_density": {
            str(d): float(np.median([r["max_abs"] for r in masked if r["density"] == d]))
            for d in densities
        },
    }
    return records, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--n-masks", type=int, default=40, help="random masks per density")
    ap.add_argument("--densities", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/port_fidelity/forward_equivalence")
    args = ap.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.manual_seed(args.seed)

    all_records: List[dict] = []
    summaries: Dict[str, dict] = {}
    node_report: Dict[str, float] = {}

    for arm in ("bias_intact", "b_O_zeroed"):
        tl_model = load_tl_model(MODEL, device)
        orig_model = build_original(device)
        if arm == "b_O_zeroed":
            zero_attn_out_bias(tl_model, orig_model)
        tokens = load_tokens(args.n_prompts, device)

        if arm == "bias_intact":
            node_report = compare_node_decomposition(tl_model, orig_model, tokens)
            print("\n[B1b] node decomposition at full density")
            for k, v in node_report.items():
                print(f"       {k:38s} {v:.4e}")

        ac_model = patchable_model(
            tl_model, factorized=True, slice_output="last_seq",
            separate_qkv=True, device=device,
        )
        ablations = src_ablations(ac_model, tokens, AblationType.ZERO)

        n_layers, n_heads = ac_model.cfg.n_layers, ac_model.cfg.n_heads
        assert total_edges(n_layers, n_heads) == ac_model.n_edges == 32491
        for ac_key, spec in expected_shapes(n_layers, n_heads).items():
            wrapper = next(w for w in ac_model.ordered_dest_wrappers if w.module_name == ac_key)
            assert tuple(wrapper.patch_mask.shape) == spec["ac"], ac_key
        if arm == "bias_intact":
            print(f"[B0] bijection covers {ac_model.n_edges} edges; all shapes agree")

        print(f"\n=== arm: {arm} ===")
        records, summary = run_arm(
            ac_model, orig_model, tokens, ablations,
            args.densities, args.n_masks, args.seed,
        )
        for r in records:
            r["arm"] = arm
        all_records.extend(records)
        summaries[arm] = summary

        print(f"  full density        max|dlogit|={summary['b1_full_density_max_abs']:.3e} "
              f"(logit scale {summary['logit_scale']:.1f})")
        print(f"  within-impl floor   max|dlogit|={summary['within_impl_nondeterminism_max_abs']:.3e}")
        for label, val in summary["single_edge_max_abs"].items():
            print(f"  single edge {label:10s} max|dlogit|={val:.3e}")
        for d, val in summary["b3_median_max_abs_by_density"].items():
            sub = [r for r in records if r["kind"] == "B2_masked" and r["density"] == float(d)]
            print(f"  density {d:>5}         median max|dlogit|={val:.3e}  "
                  f"argmax agree {np.mean([r['argmax_agree'] for r in sub]):.2%}")

        del ac_model, orig_model, tl_model
        t.cuda.empty_cache()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_records for k in r})
    with open(out_dir / "deviations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_records)
    (out_dir / "summary.json").write_text(json.dumps(
        {"n_prompts": args.n_prompts, "n_masks_per_density": args.n_masks,
         "node_decomposition": node_report, "arms": summaries}, indent=2))
    print(f"\n[done] wrote {out_dir}/deviations.csv and summary.json")


if __name__ == "__main__":
    main()
