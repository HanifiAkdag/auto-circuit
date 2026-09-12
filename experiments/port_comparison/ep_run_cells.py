"""Tier C of the Edge Pruning port fidelity study: one matched training run.

Runs Edge Pruning once, in either implementation, on the frozen IOI split, and
writes ``PruneScores`` in AutoCircuit layout plus a manifest. The reference's masks
are converted through ``ep_edge_map``, which Tier B verified functionally, so both
arms land in one comparable format.

The reference is driven through its own ``src/prune/fpt2_ioi.py`` as a subprocess,
not through a re-expressed training loop. That is deliberate: a control that
reconstructs the thing under test cannot detect a defect in how the real code
constructs it.

Arms (see ``docs/superpowers/specs/2026-08-11-edge-pruning-port-fidelity.md``):

==================  ========================================================
``port``            primary
``orig``            primary, without ``--with_embedding_nodes``
``orig-withembed``  mechanism: the paper's default embedding-node flag
``port-sharedz``    mechanism: one mask sample per step, as the reference does
                    (this is what the library default became after the study)
``port-nobos``      the port's input convention (BOS prepend + left pad)
``port-neginv``     negative control: score-sign convention removed
``port-negkl``      negative control: KL direction reversed
``port-nolambda``   negative control: Lagrangian frozen, no sparsity pressure
==================  ========================================================

Run::

    srun --gres=gpu:1 --mem=48G python -m experiments.port_comparison.ep_run_cells \\
        --arm port --seed 0
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch as t

REPO = Path(__file__).resolve().parents[2]
_EP_ROOT = REPO / "third_party" / "edge_pruning"
for _extra in (REPO, REPO / "experiments", _EP_ROOT / "src" / "modeling"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK, Task  # noqa: E402
from auto_circuit.types import AblationType  # noqa: E402
from auto_circuit.utils.misc import repo_path_to_abs_path  # noqa: E402
from experiment_utils import git_state, hardware_state  # noqa: E402

from experiments.port_comparison.ep_edge_map import (  # noqa: E402
    log_alphas_to_prune_scores,
    read_log_alphas,
    write_log_alphas,
)
from experiments.port_comparison.frozen_ioi import fixed_split_loader  # noqa: E402

FROZEN_DATASET = "port_fidelity/ioi_frozen"
FROZEN_EP_DIR = repo_path_to_abs_path("datasets/port_fidelity/ioi_frozen_ep")

ARMS = (
    "port", "orig",
    "orig-withembed",
    "port-sharedz",
    "port-nobos",
    "port-neginv", "port-negkl", "port-nolambda",
)

# The paper's IOI sweep (`run_scripts/ioi_sweep.sh`), which both arms follow.
BATCH_SIZE = 32
BATCH_COUNT = (22, 8)  # 704 train / 256 test at batch size 32
HPARAMS = dict(
    learning_rate=0.8,
    lambda_learning_rate=0.8,
    steps=3000,
    warmup_steps=200,
    target_edge_sparsity=1.1,
    target_node_sparsity=0.72,
    start_edge_sparsity=0.0,
    start_node_sparsity=0.0,
    sparsity_warmup_steps=2500,
    use_node_masks=True,
    disable_node_loss=True,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


# --- port arms ---------------------------------------------------------------


def run_port(arm: str, seed: int, device: t.device) -> Dict[str, t.Tensor]:
    from auto_circuit.prune_algos import edge_pruning as ep

    # Mask sampling is pinned explicitly on both arms rather than inherited from the
    # library default. The default changed to the reference's behaviour on 2026-08-12
    # *as a result of this study*, so leaving it implicit would silently redefine what
    # the cached `port` cells mean and make the published Tier C numbers
    # irreproducible.
    shared_mask_sample = arm == "port-sharedz"

    if arm == "port-negkl":
        # Negative control: KL(circuit || full) instead of KL(full || circuit).
        base_kl = ep.multibatch_kl_div
        ep.multibatch_kl_div = lambda a, b: base_kl(b, a)

    if arm == "port-nolambda":
        # Negative control: no sparsity pressure at all.
        ep._lagrangian = lambda sparsity, target, l1, l2: t.zeros(
            (), device=sparsity.device, dtype=sparsity.dtype
        )

    tmpl = IOI_COMPONENT_CIRCUIT_TASK
    task = Task(
        key=tmpl.key, name=tmpl.name,
        batch_size=BATCH_SIZE, batch_count=BATCH_COUNT,
        token_circuit=tmpl.token_circuit,
        _dataset_name=FROZEN_DATASET,
        factorized=tmpl.factorized, separate_qkv=tmpl.separate_qkv,
        _true_edge_func=tmpl._true_edge_func, slice_output=tmpl.slice_output,
        dtype=tmpl.dtype, seed=seed, _model_def=tmpl._model_def,
    )
    prepend_bos = False if arm == "port-nobos" else None
    with fixed_split_loader(prepend_bos=prepend_bos):
        task.init_task()

    # Task construction may consume or reset random-number streams. The run seed
    # controls optimisation and mask sampling independently of loader construction.
    seed_everything(seed)

    scores = ep.edge_pruning_prune_scores(
        task.model, task.train_loader, None,
        ablation_type=AblationType.RESAMPLE,
        show_train_graph=False,
        shared_mask_sample=shared_mask_sample,
        **HPARAMS,
    )
    scores = {k: v.detach().cpu() for k, v in scores.items()}

    if arm == "port-neginv":
        # Negative control: export the ablation probability as the score, i.e. drop
        # the `1 - z` convention flip that makes high = keep.
        scores = {k: 1.0 - v for k, v in scores.items()}
    return scores


# --- original arm ------------------------------------------------------------


def run_original(arm: str, seed: int, device: t.device) -> Dict[str, t.Tensor]:
    with_embedding_nodes = arm == "orig-withembed"
    out_dir = REPO / ".scratch" / "ep_port" / "runs" / f"{arm}_seed{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "src/prune/fpt2_ioi.py",
        "--report_to", "none", "--do_train",
        "--dataset_path", str(FROZEN_EP_DIR), "--train_split", "train",
        "--initialize_from", "gpt2", "--max_seq_length", "64",
        "--per_device_train_batch_size", str(BATCH_SIZE),
        "--per_device_eval_batch_size", "16",
        "--gradient_accumulation_steps", "1", "--eval_accumulation_steps", "16",
        "--edge_learning_rate", "0.8", "--layer_learning_rate", "0.8",
        "--reg_edge_learning_rate", "0.8", "--reg_layer_learning_rate", "0.8",
        "--max_steps", str(HPARAMS["steps"]),
        "--warmup_steps", str(HPARAMS["warmup_steps"]),
        "--eval_strategy", "no", "--save_strategy", "no",
        "--logging_steps", "500",
        "--start_edge_sparsity", str(HPARAMS["start_edge_sparsity"]),
        "--target_edge_sparsity", str(HPARAMS["target_edge_sparsity"]),
        "--start_layer_sparsity", str(HPARAMS["start_node_sparsity"]),
        "--target_layer_sparsity", str(HPARAMS["target_node_sparsity"]),
        "--num_sparsity_warmup_steps", str(HPARAMS["sparsity_warmup_steps"]),
        "--max_train_samples", "1000000", "--max_eval_samples", "256",
        "--output_dir", str(out_dir),
        "--remove_unused_columns", "false", "--dataloader_num_workers", "0",
        "--warmup_type", "linear", "--disable_node_loss",
        "--seed", str(seed), "--overwrite_output_dir",
    ]
    if with_embedding_nodes:
        cmd.append("--with_embedding_nodes")

    env = dict(os.environ, WANDB_MODE="disabled")
    print("running reference:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(_EP_ROOT), env=env, check=True)

    from modeling_fpt2 import FPT2LMHeadModel

    model = FPT2LMHeadModel.from_pretrained(
        str(out_dir), with_embedding_nodes=with_embedding_nodes
    )
    n_layers, n_heads = model.config.n_layer, model.config.n_head
    scores = log_alphas_to_prune_scores(
        read_log_alphas(model, n_layers),
        write_log_alphas(model, n_layers),
        n_layers, n_heads,
        n_embed_writers=2 if with_embedding_nodes else 0,
    )
    shutil.rmtree(out_dir, ignore_errors=True)
    return {k: v.detach().cpu() for k, v in scores.items()}


# --- entry point -------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--output-dir", default=".prune_scores/final/port_fidelity/edge_pruning"
    )
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

    n_edges = sum(v.numel() for v in scores.values())
    density = float(sum((v > 0.5).sum() for v in scores.values()) / n_edges)
    print(f"edge density at score>0.5: {density:.4%}  ({n_edges} edges)")

    dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    full_stem = f"{stem}-{dt}"
    cache_path = folder / f"{full_stem}.pkl"
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    t.save({IOI_COMPONENT_CIRCUIT_TASK.key: {args.arm: scores}}, tmp)
    os.replace(tmp, cache_path)

    manifest = {
        "cell": {"arm": args.arm, "seed": args.seed, "task": "ioi",
                 "algo": "ep", "ablation": "resample"},
        "cache_file": cache_path.name,
        "edge_density_at_threshold": density,
        "n_edges": n_edges,
        "dataset": FROZEN_DATASET,
        "dataset_shuffle": False,
        "batch_size": BATCH_SIZE,
        "batch_count": list(BATCH_COUNT),
        "hparams": HPARAMS,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware_state(),
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
