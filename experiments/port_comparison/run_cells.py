"""Tier C of the port fidelity study: one matched training run per invocation.

Runs edge-only DiscoGP once, in either implementation, on the frozen IOI split,
and writes `PruneScores` in AutoCircuit layout plus a manifest. The original's
masks are converted through `edge_map`, which Tier B verified functionally, so
both arms land in one comparable format.

Arms (see `docs/superpowers/specs/2026-08-08-discogp-port-fidelity.md`):

===============  =========================================================
``port``         primary
``orig``         primary
``*-bOzero``     mechanism arm: `b_O` removed, the one Tier B divergence
``port-nobos``   measures the port's BOS-prepend / left-pad factor
``port-neginv``  negative control: mask-convention inversion removed
``orig-asshipped``  negative control: the original's unspaced answer tokens
===============  =========================================================

Run::

    srun --gres=gpu:1 python -m experiments.port_comparison.run_cells \
        --arm port --seed 0
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch as t

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO, REPO / "experiments", REPO / "third_party" / "disco_gp"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK, Task  # noqa: E402
from auto_circuit.types import AblationType  # noqa: E402
from auto_circuit.utils.misc import repo_path_to_abs_path  # noqa: E402
from experiment_utils import git_state  # noqa: E402

from experiments.port_comparison.edge_map import keep_mask_to_prune_scores  # noqa: E402
from experiments.port_comparison.forward_equivalence import (  # noqa: E402
    MODEL,
    TL_COMPAT,
)
from experiments.port_comparison.frozen_ioi import fixed_split_loader  # noqa: E402

FROZEN_DATASET = "port_fidelity/ioi_frozen"
ARMS = (
    "port", "orig",
    "port-split",  # port adopting the original's bias convention, model unchanged
    "port-bOzero", "orig-bOzero",
    "port-nobos",
    "port-neginv", "orig-asshipped",
)

# Edge-phase hyperparameters, as registered in both `prune_algos.py` and the
# original's `cli.py`. Tier A verified the schedule and losses agree; these are the
# values that make the two runs comparable.
EDGE_HPARAMS = dict(
    gs_temp_edge=1.0,
    logits_e_init=1.0,
    lr=0.1,
    lambda_sparse_init=1.0,
    lambda_complete_init=1.0,
    min_times_lambda_sparse=0.01,
    max_times_lambda_sparse=100.0,
    train_epochs=100,
    n_epoch_warmup_lambda_sparse=20,
    n_epoch_cooldown_lambda_sparse=20,
)
BATCH_SIZE = 64


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


# --- port arm ----------------------------------------------------------------


def run_port(arm: str, seed: int, device: t.device) -> Dict[str, t.Tensor]:
    from auto_circuit.prune_algos import disco_gp as dg

    if arm == "port-neginv":
        # Negative control: drop the AutoCircuit convention flip, so kept edges are
        # ablated and vice versa. The study must detect this.
        def _broken(edge_logits, gs_temp, reverse, hard):
            out = {}
            for name, logit in edge_logits.items():
                mask = dg.gumbel_sigmoid(logit, gs_temp=gs_temp, hard=hard)
                out[name] = 1.0 - mask if reverse else mask
            return out

        dg._sample_edge_masks = _broken

    tmpl = IOI_COMPONENT_CIRCUIT_TASK
    task = Task(
        key=tmpl.key, name=tmpl.name,
        batch_size=BATCH_SIZE, batch_count=(11, 4),
        token_circuit=tmpl.token_circuit,
        _dataset_name=FROZEN_DATASET,
        factorized=tmpl.factorized, separate_qkv=tmpl.separate_qkv,
        _true_edge_func=tmpl._true_edge_func, slice_output=tmpl.slice_output,
        dtype=tmpl.dtype, seed=seed, _model_def=tmpl._model_def,
    )
    prepend_bos = False if arm == "port-nobos" else None
    with fixed_split_loader(prepend_bos=prepend_bos):
        task.init_task()
    # Dataset construction sets global RNG state. Make the run seed's role in
    # mask initialisation and stochastic optimisation explicit.
    seed_everything(seed)

    if arm == "port-bOzero":
        with t.no_grad():
            for block in task.model.blocks:
                block.attn.b_O.zero_()
    elif arm == "port-split":
        from experiments.port_comparison.forward_equivalence import split_attn_out_bias

        split_attn_out_bias(task.model)

    scores = dg.disco_gp_prune_scores(
        task.model, task.train_loader, None,
        use_weight_masks=False, use_edge_masks=True, hard=True,
        loss_mode="answer_margin_ce", ablation_type=AblationType.ZERO,
        e_lr=EDGE_HPARAMS["lr"],
        gs_temp_edge=EDGE_HPARAMS["gs_temp_edge"],
        logits_e_init=EDGE_HPARAMS["logits_e_init"],
        e_lambda_sparse_init=EDGE_HPARAMS["lambda_sparse_init"],
        e_lambda_complete_init=EDGE_HPARAMS["lambda_complete_init"],
        e_min_times_lambda_sparse=EDGE_HPARAMS["min_times_lambda_sparse"],
        e_max_times_lambda_sparse=EDGE_HPARAMS["max_times_lambda_sparse"],
        e_train_epochs=EDGE_HPARAMS["train_epochs"],
        e_n_epoch_warmup=EDGE_HPARAMS["n_epoch_warmup_lambda_sparse"],
        e_n_epoch_cooldown=EDGE_HPARAMS["n_epoch_cooldown_lambda_sparse"],
    )
    return {k: v.detach().cpu() for k, v in scores.items()}


# --- original arm ------------------------------------------------------------


def _build_dls(model, as_shipped: bool):
    """Attach dataloaders built from the frozen split to the original model."""
    from argparse import Namespace

    from datasets import Dataset
    from torch.utils.data import DataLoader

    from experiments.port_comparison.frozen_ioi import FROZEN, load_for_original

    def make(split: str):
        d = load_for_original(model.tokenizer, split=split, path=FROZEN)
        if as_shipped:
            # Negative control: reproduce the original's defect by re-tokenizing the
            # answers without their leading space, which fragments ~48% of names.
            data = json.loads(FROZEN.read_text())
            meta = data["port_fidelity"]
            rows = (data["prompts"][: meta["n_train"]] if split == "train"
                    else data["prompts"][meta["n_train"]: meta["n_train"] + meta["n_test"]])
            d["target good"] = t.tensor(
                [model.tokenizer(r["answers"][0].lstrip())["input_ids"][0] for r in rows]
            )
            d["target bad"] = t.tensor(
                [model.tokenizer(r["wrong_answers"][0].lstrip())["input_ids"][0] for r in rows]
            )
        ds = Dataset.from_dict({k: v for k, v in d.items() if k != "prompt"})
        return DataLoader(ds.with_format("torch"), batch_size=BATCH_SIZE, shuffle=False)

    train = make("train")
    test = make("test")
    return Namespace(train=train, eval=test, test=test)


def run_original(arm: str, seed: int, device: t.device) -> Dict[str, t.Tensor]:
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer

    from disco_gp import Config, DiscoGPTransformer

    model_cfg = Config.from_tl(MODEL, dtype=t.float32)
    for key, value in TL_COMPAT.items():
        if key not in model_cfg._values:
            setattr(model_cfg, key, value)

    cfg = Config.from_configs(
        model=model_cfg,
        wandb=Config(use_wandb=False, wandb_project_name=None, wandb_entity=None),
        weight=Config(use_weight_masks=False, gs_temp_weight=0.01, logits_w_init=1.0),
        edge=Config(use_edge_masks=True, **EDGE_HPARAMS),
        task=Config(task_type="ioi", batch_size=BATCH_SIZE),
        exp=Config(evaluate_every=100, save_every=None, output_dir_path=None,
                   exp_name="port_fidelity", print_results=True),
    )

    model = DiscoGPTransformer(cfg)
    state_dict = HookedTransformer.from_pretrained(MODEL).state_dict()
    model.hook_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.setup_mask()
    model = model.to(device, dtype=t.float32)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer

    if arm == "orig-bOzero":
        with t.no_grad():
            for block in model.blocks:
                block.attn.b_O.zero_()

    model.dls = _build_dls(model, as_shipped=(arm == "orig-asshipped"))
    model.setup_experiment()
    model.evaluate_and_report(epoch=0, mode="baseline")
    model.search(modes="e")
    model.evaluate_and_report(epoch="final", mode="pruned")

    logits = {name: p.detach().float().cpu()
              for name, p in model.named_parameters() if "edge" in name}
    return keep_mask_to_prune_scores(logits, cfg.n_layers, cfg.n_heads)


# --- entry point -------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", default=".port_fidelity_cache")
    ap.add_argument("--skip-if-exists", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    stem = f"cell_{args.arm.replace('-', '_')}_seed{args.seed}"
    folder = repo_path_to_abs_path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    if args.skip_if_exists and not args.force:
        # A .pkl only counts as cached once its manifest landed; a job killed between
        # the two writes leaves an orphan that would silently drop the cell.
        done = [p for p in sorted(folder.glob(f"{stem}-*.pkl"))
                if (folder / f"{p.stem}_manifest.json").exists()]
        if done:
            print(f"Cell already cached ({done[-1].name}); skipping.")
            return

    seed_everything(args.seed)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Running arm={args.arm} seed={args.seed} on {device}")

    runner = run_original if args.arm.startswith("orig") else run_port
    scores = runner(args.arm, args.seed, device)

    density = float(
        sum((v > 0.5).sum() for v in scores.values())
        / sum(v.numel() for v in scores.values())
    )
    print(f"edge density at sigmoid>0.5 (i.e. logit>0): {density:.4%}")

    dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    full_stem = f"{stem}-{dt}"
    cache_path = folder / f"{full_stem}.pkl"
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    t.save({IOI_COMPONENT_CIRCUIT_TASK.key: {args.arm: scores}}, tmp)
    os.replace(tmp, cache_path)

    manifest = {
        "cell": {"arm": args.arm, "seed": args.seed, "task": "ioi",
                 "algo": "discogp-ep", "ablation": "zero"},
        "cache_file": cache_path.name,
        "edge_density_at_threshold": density,
        "n_edges": sum(v.numel() for v in scores.values()),
        "dataset": FROZEN_DATASET,
        "dataset_shuffle": False,
        "batch_size": BATCH_SIZE,
        "edge_hparams": EDGE_HPARAMS,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
    }
    mpath = folder / f"{full_stem}_manifest.json"
    tmpm = mpath.with_name(mpath.name + ".tmp")
    with open(tmpm, "w") as f:
        json.dump(manifest, f, indent=2, default=repr)
    os.replace(tmpm, mpath)
    print(f"Saved {cache_path.name} and manifest")


if __name__ == "__main__":
    main()
