"""Per-cell discovery entry point for sub-RQ3 experiments.

Runs exactly one (method, task, discovery-ablation, seed) cell and saves the resulting
PruneScores to a uniquely-named cache file alongside a manifest JSON.

SLURM-array usage (one cell per array task):
    srun python experiments/run_cell.py --task ioi --algo ep --ablation resample --seed 42

For the null test (randomly-initialized model weights):
    srun python experiments/run_cell.py --task ioi --algo ep --ablation resample \\
        --seed 42 --null-test

See docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md §3
(Operating model) for the full design rationale.
"""
import argparse
import json
import os
import random
import socket
import sys
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch as t
import transformer_lens as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.prune_algos.prune_algos import (
    ACDC_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
    DISCO_GP_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    PruneAlgo,
    make_ablation_variant,
    run_prune_algos,
)
from auto_circuit.tasks import (
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    Task,
)
from auto_circuit.types import AblationType
from auto_circuit.utils.misc import repo_path_to_abs_path
from experiment_utils import algo_hyperparams, git_state

# Algos that accept an ablation_type parameter (EP, DiscoGP variants, ACDC, EAP).
# GT and Random are ablation-independent.
_ABLATION_AWARE_ALGOS = {"ep", "discogp", "discogp-e", "discogp-ep", "acdc", "eap"}

_ALGO_MAP: dict[str, PruneAlgo] = {
    "ep": EDGE_PRUNING_PRUNE_ALGO,
    "discogp": DISCO_GP_PRUNE_ALGO,
    "discogp-e": DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
    "discogp-ep": DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO,
    "acdc": ACDC_PRUNE_ALGO,
    "eap": EDGE_ATTR_PATCH_PRUNE_ALGO,
    "gt": GROUND_TRUTH_PRUNE_ALGO,
    "random": RANDOM_PRUNE_ALGO,
}

_ABLATION_MAP: dict[str, AblationType] = {
    "zero": AblationType.ZERO,
    "resample": AblationType.RESAMPLE,
    "mean": AblationType.TOKENWISE_MEAN_CORRUPT,
}

# Task templates — seed and (optionally) _model_def are overridden at runtime.
_TASK_TEMPLATES: dict[str, Task] = {
    "ioi": IOI_COMPONENT_CIRCUIT_TASK,
    "gt": GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    "docstring": DOCSTRING_COMPONENT_CIRCUIT_TASK,  # component, matches the spine (not token)
}

# Model strings for each task (used to build a random-weight model for the null test).
_TASK_MODEL_DEF: dict[str, str] = {
    "ioi": "gpt2-small",
    "gt": "gpt2-small",
    "docstring": "attn-only-4l",
}


def _serialize(obj: Any) -> Any:
    """`json.dump(default=...)` fallback for the manifest: renders an AblationType (or
    any other enum) as its member name and a Path as a plain string, otherwise falls
    back to `repr` rather than raising on an unrecognised type."""
    import enum
    if isinstance(obj, enum.Enum):
        return str(obj).split(".")[-1]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _build_null_model(model_def: str, device: t.device) -> t.nn.Module:
    """Load an architecture, randomize all weights, set required cfg flags."""
    model = tl.HookedTransformer.from_pretrained(
        model_def, device=device, fold_ln=True,
        center_writing_weights=True, center_unembed=True,
    )
    # Randomize every parameter tensor in-place.
    for param in model.parameters():
        t.nn.init.normal_(param, mean=0.0, std=0.02)
    # The nn.Module branch in Task.init_task skips these cfg flags — set them manually.
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def _cache_stem(
    task: str,
    algo: str,
    ablation: str,
    seed: int,
    init_seed: Optional[int],
    null_test: bool = False,
    partial_tag: Optional[str] = None,
) -> str:
    """Cache identity. The init tag appears only when the init seed differs from the data
    seed, so joint cells keep the names the existing cells already use and --skip-if-exists
    stays idempotent against them."""
    init_tag = "" if init_seed is None or init_seed == seed else f"_init{init_seed}"
    null_tag = "_null" if null_test else ""
    ptag = f"__{partial_tag}" if partial_tag else ""
    algo_slug = algo.replace("-", "_")
    return f"cell_{task}_{algo_slug}_{ablation}_seed{seed}{init_tag}{null_tag}{ptag}"


def _hardware() -> dict[str, Any]:
    """Seed repeats are only comparable within one GPU architecture and nothing else in the
    cache records which one ran them."""
    return {
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": t.cuda.get_device_name(0) if t.cuda.is_available() else None,
    }


def _data_budget(task: Task) -> dict[str, Any]:
    """Record the data budget: nothing else in the cache identifies it, so cells from
    different tasks.py settings are otherwise indistinguishable. Both fields may be a
    scalar (same for train and test) or a (train, test) tuple, as Task.init_task reads
    them."""
    pair = lambda v: v if isinstance(v, tuple) else (v, v)  # noqa: E731
    bs, bc = pair(task.batch_size), pair(task.batch_count)
    return {
        "batch_size": task.batch_size,
        "batch_count": task.batch_count,
        "train_examples": bs[0] * bc[0],
        "test_examples": bs[1] * bc[1],
    }


def _build_task(task_short: str, seed: int, null_test: bool) -> Task:
    """Construct a Task from the template, overriding seed (and model for null test)."""
    tmpl = _TASK_TEMPLATES[task_short]
    kwargs = {
        "key": tmpl.key,
        "name": tmpl.name,
        "batch_size": tmpl.batch_size,
        "batch_count": tmpl.batch_count,
        "token_circuit": tmpl.token_circuit,
        "_dataset_name": tmpl._dataset_name,
        "factorized": tmpl.factorized,
        "separate_qkv": tmpl.separate_qkv,
        "_true_edge_func": tmpl._true_edge_func,
        "slice_output": tmpl.slice_output,
        "dtype": tmpl.dtype,
        "seed": seed,
    }
    if null_test:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        kwargs["_model_def"] = _build_null_model(_TASK_MODEL_DEF[task_short], device)
    else:
        kwargs["_model_def"] = tmpl._model_def
    return Task(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run one (method, task, ablation, seed) discovery cell.")
    parser.add_argument("--task", required=True, choices=list(_TASK_TEMPLATES), help="Task short name")
    parser.add_argument("--algo", required=True, choices=list(_ALGO_MAP), help="Algorithm short name")
    parser.add_argument("--ablation", required=True, choices=list(_ABLATION_MAP), help="Discovery ablation type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (data shuffle + mask init)")
    parser.add_argument("--init-seed", type=int, default=None, help="Mask-init seed, decoupled from --seed (data draw). Omit for the legacy coupled path.")
    parser.add_argument("--output-dir", default=".prune_scores_cache", help="Cache folder (relative to repo root)")
    parser.add_argument("--null-test", action="store_true", help="Use randomly-initialized model weights (dead-salmon null)")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip if this cell is already cached (requeue-safe)")
    parser.add_argument("--force", action="store_true", help="Recompute even if a cache exists")
    # ACDC-only: split the tao sweep across parallel jobs. Each partial computes a
    # disjoint subset of tao; the tao passes are independent and merge by element-wise
    # min (see merge_acdc_partials.py), which is identical to one process running all
    # tao. --partial-tag makes the cache name unique so partials do not collide.
    parser.add_argument("--tao-bases", default=None, help="ACDC only: comma-separated tao bases override (e.g. '1,3')")
    parser.add_argument("--tao-exps", default=None, help="ACDC only: comma-separated tao exps override (e.g. '-5,-4,-3,-2')")
    parser.add_argument("--partial-tag", default=None, help="Suffix appended to the cache stem (for tao-split partials)")
    args = parser.parse_args()

    # ── Cache identity (computed up front so we can skip/resume) ──
    base_filename = _cache_stem(
        args.task, args.algo, args.ablation, args.seed, args.init_seed,
        args.null_test, args.partial_tag,
    )
    folder = repo_path_to_abs_path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    # Requeue/restart-safe: a cancelled-then-requeued array task whose cell already
    # finished exits immediately instead of recomputing.
    if args.skip_if_exists and not args.force:
        # A .pkl is only "cached" if its manifest landed too. A job killed between the
        # two writes leaves an orphan .pkl; treating that as done would silently drop
        # the cell from the analysis, which keys off manifests.
        existing = [
            p for p in sorted(folder.glob(f"{base_filename}-*.pkl"))
            if (folder / f"{p.stem}_manifest.json").exists()
        ]
        if existing:
            print(f"Cell already cached ({existing[-1].name}); skipping. Use --force to recompute.")
            return

    # ── Seed everything before any model/data initialization ──
    t.manual_seed(args.seed)
    random.seed(args.seed)

    ablation_type = _ABLATION_MAP[args.ablation]
    base_algo = _ALGO_MAP[args.algo]

    # ACDC tao-sweep override (partial jobs). partial() lets outer kwargs win, so this
    # overrides the tao_bases/tao_exps already bound in ACDC_PRUNE_ALGO.func.
    if args.tao_bases is not None or args.tao_exps is not None:
        assert args.algo == "acdc", "--tao-bases/--tao-exps are ACDC-only"
        tao_kwargs: dict[str, Any] = {}
        if args.tao_bases is not None:
            tao_kwargs["tao_bases"] = [int(b) for b in args.tao_bases.split(",")]
        if args.tao_exps is not None:
            tao_kwargs["tao_exps"] = [int(e) for e in args.tao_exps.split(",")]
        base_algo = PruneAlgo(
            key=base_algo.key, name=base_algo.name,
            func=partial(base_algo.func, **tao_kwargs), _short_name=base_algo.short_name,
        )

    if args.algo in _ABLATION_AWARE_ALGOS:
        algo = make_ablation_variant(base_algo, ablation_type)
    else:
        algo = base_algo  # GT / Random: ablation is irrelevant

    task = _build_task(args.task, args.seed, args.null_test)

    # --init-seed decouples mask initialisation from the prompt draw. load_datasets_from_json
    # re-seeds torch itself, so the seed has to land after the loaders are built; init_task is
    # idempotent. Omitting the flag leaves the legacy path untouched.
    if args.init_seed is not None:
        task.init_task()
        t.manual_seed(args.init_seed)
        random.seed(args.init_seed)

    print(f"Running: task={args.task}  algo={algo.key}  ablation={args.ablation}  seed={args.seed}  init_seed={args.init_seed}  null_test={args.null_test}")
    task_prune_scores = run_prune_algos([task], [algo])

    # ── Save cache + manifest ──
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    full_stem = f"{base_filename}-{dt_string}"
    cache_path = folder / f"{full_stem}.pkl"
    # Write-then-rename: on a preemptible partition a direct write can be truncated by
    # SIGKILL, and a torn .pkl at the final name would be mistaken for a finished cell.
    tmp_cache = cache_path.with_name(cache_path.name + ".tmp")
    t.save(task_prune_scores, tmp_cache)
    os.replace(tmp_cache, cache_path)
    print(f"Saved cache to {cache_path}")

    manifest = {
        "cell": {
            "task": args.task,
            "algo": args.algo,
            "algo_key": algo.key,
            "ablation": args.ablation,
            "seed": args.seed,
            "init_seed": args.seed if args.init_seed is None else args.init_seed,
            "null_test": args.null_test,
        },
        "cache_file": cache_path.name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "hardware": _hardware(),
        "data_budget": _data_budget(task),
        "algo_hyperparams": algo_hyperparams(algo),
    }
    manifest_path = folder / f"{full_stem}_manifest.json"
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=_serialize)
    os.replace(tmp_manifest, manifest_path)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
