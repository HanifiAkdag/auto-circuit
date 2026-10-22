"""Small, explicitly non-production discovery checks on all benchmark models."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import random


def _run_cell(task_short, algo_short, ablation, seed, init_seed, output, protocol):
    import json
    import torch
    from auto_circuit.prune import run_circuits
    from auto_circuit.prune_algos.prune_algos import make_ablation_variant
    from auto_circuit.types import PatchType
    from .pipeline import atomic_json
    from .run_cell import _ALGO_MAP, _ABLATION_MAP, _bind_task_hyperparams, _build_task

    suffix = "" if (seed, init_seed) == (42, 42) else f"_seed{seed}_init{init_seed}"
    path = output / f"{task_short}_{algo_short}_{ablation}{suffix}.json"
    if path.exists():
        if not json.loads(path.read_text()).get("complete"):
            raise ValueError(f"Incomplete smoke result: {path}")
        return False

    split_seed = None if protocol == "thesis" else 42
    task = replace(_build_task(task_short, seed, False, split_seed=split_seed),
                   batch_size=1, batch_count=(1, 1), __init_complete__=False)
    if init_seed != seed:
        task.init_task()
        torch.manual_seed(init_seed)
        random.seed(init_seed)
    algo = _bind_task_hyperparams(_ALGO_MAP[algo_short], algo_short, task_short)
    algo = make_ablation_variant(algo, _ABLATION_MAP[ablation])
    settings = {"steps": 2, "warmup_steps": 0, "sparsity_warmup_steps": 1} if algo_short == "ep" else (
        {"e_train_epochs": 1} if algo_short == "discogp-ep" else
        {"tao_bases": [1], "tao_exps": [-5]} if algo_short == "acdc" else {})
    scores = partial(algo.func, **settings)(task.model, task.train_loader, task.true_edges)
    assert set(scores) == set(task.model.patch_masks)
    assert all(scores[k].shape == task.model.patch_masks[k].shape for k in scores)
    assert all(not torch.isnan(value).any() for value in scores.values())
    outputs = run_circuits(task.model, task.test_loader, [task.true_edge_count, task.model.n_edges],
                           scores, patch_type=PatchType.TREE_PATCH,
                           ablation_type=_ABLATION_MAP[ablation], render_graph=False)
    assert outputs
    record = dict(task=task_short, algo=algo_short, ablation=ablation,
                  seed=seed, init_seed=init_seed, smoke=True, complete=True,
                  graph_edges=task.model.n_edges, overrides=settings)
    atomic_json(path, record)
    print(f"Smoke completed {task_short}/{algo_short}/{ablation}/{seed}/{init_seed}", flush=True)
    return True


def run(args, registry):
    from .config import freeze_run, sha256
    from .pipeline import atomic_json, hpo, code_identity

    output = args.run_dir / "smoke"
    settings = {"smoke": True, "protocol": args.protocol, "train_batch_size": 1,
                "train_batches": 1, "ep_steps": 2, "discogp_epochs": 1,
                "acdc_tao_bases": [1], "acdc_tao_exps": [-5],
                "seed_decomposition": [[42, 43], [43, 42]],
                "registry_sha256": sha256(args.registry), "code": code_identity()}
    freeze_run(output, settings)
    completed = 0
    for task_short in ("ioi", "gt", "docstring"):
        for algo_short in ("ep", "discogp-ep", "eap", "acdc"):
            for ablation in ("zero", "resample", "mean"):
                completed += _run_cell(task_short, algo_short, ablation, 42, 42, output, args.protocol)
    for seed, init_seed in ((42, 43), (43, 42)):
        completed += _run_cell("ioi", "ep", "zero", seed, init_seed, output, args.protocol)
    hpo(args, registry, smoke=True)
    atomic_json(output / "complete.json", {"complete": True, "smoke": True,
                "cells": 38, "new_cells": completed, "seed_decomposition": 2})
