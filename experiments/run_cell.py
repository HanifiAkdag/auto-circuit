"""Per-cell discovery entry point for sub-RQ3 experiments.

Runs exactly one (method, task, discovery-ablation, seed) cell and saves the resulting
PruneScores to a uniquely-named cache file alongside a manifest JSON.

SLURM-array usage (one cell per array task):
    srun --mem=64G python -m experiments.run_cell --task ioi --algo ep --ablation resample --seed 42

The maintained matrix and frozen inputs are documented in docs/reproduction.md.
"""
import argparse
import hashlib
import json
import os
import random
import socket
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch as t
import transformer_lens as tl

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
from .experiment_utils import algo_hyperparams, git_state

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

# Released EP settings: GT normalizes its training KL over the 100 number tokens.
# Unlisted tasks retain the registered algorithm's defaults.
_EP_TASK_HPARAMS: dict[str, dict[str, Any]] = {
    "ioi": {"target_edge_sparsity": 1.1, "target_node_sparsity": 0.72},
    "gt": {
        "target_edge_sparsity": 1.1, "target_node_sparsity": 0.68,
        "kl_vocab_tokens": tuple(f"{i:02d}" for i in range(100)),
    },
    "docstring": {"target_edge_sparsity": 1.1, "target_node_sparsity": 0.72},
}


def _bind_task_hyperparams(algo: PruneAlgo, algo_short: str, task_short: str) -> PruneAlgo:
    """Bind released task-specific settings without changing the algorithm identity."""
    if algo_short != "ep":
        return algo
    return PruneAlgo(
        key=algo.key,
        name=algo.name,
        func=partial(algo.func, **_EP_TASK_HPARAMS.get(task_short, {})),
        _short_name=algo.short_name,
    )


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
    split_seed: Optional[int] = None,
) -> str:
    """Cache identity. The init tag appears only when the init seed differs from the data
    seed, so joint cells keep the names the existing cells already use and --skip-if-exists
    stays idempotent against them."""
    init_tag = "" if init_seed is None or init_seed == seed else f"_init{init_seed}"
    null_tag = "_null" if null_test else ""
    ptag = f"__{partial_tag}" if partial_tag else ""
    split_tag = "" if split_seed is None else f"_holdout{split_seed}"
    vocab = _EP_TASK_HPARAMS.get(task, {}).get("kl_vocab_tokens") if algo == "ep" else None
    vocab_tag = ""
    if vocab is not None:
        digest = hashlib.sha256(json.dumps(vocab).encode()).hexdigest()[:12]
        vocab_tag = f"_klvocab{digest}"
    algo_slug = algo.replace("-", "_")
    return f"cell_{task}_{algo_slug}_{ablation}_seed{seed}{init_tag}{null_tag}{split_tag}{vocab_tag}{ptag}"


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
        "batch_size": list(task.batch_size) if isinstance(task.batch_size, tuple) else task.batch_size,
        "batch_count": list(task.batch_count) if isinstance(task.batch_count, tuple) else task.batch_count,
        "train_examples": bs[0] * bc[0],
        "test_examples": bs[1] * bc[1],
    }


def _can_reuse_cache(manifest: dict, task_short: str, split_seed: Optional[int]) -> bool:
    template = _TASK_TEMPLATES[task_short]
    if manifest.get('cell', {}).get('split_seed') != split_seed or manifest.get('data_budget') != _data_budget(template):
        return False
    if manifest.get("cell", {}).get("algo") == "ep":
        vocab = _EP_TASK_HPARAMS.get(task_short, {}).get("kl_vocab_tokens")
        expected = None if vocab is None else list(vocab)
        if manifest.get("algo_hyperparams", {}).get("kl_vocab_tokens") != expected:
            return False
    if split_seed is None:
        return True
    path = repo_path_to_abs_path(f'datasets/{template._dataset_name}.json')
    records = json.loads(path.read_text())['prompts']
    pool_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return (manifest.get('data_split') or {}).get('pool_sha256') == pool_hash


def _build_task(task_short: str, seed: int, null_test: bool, split_seed: Optional[int] = 42) -> Task:
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
        "split_seed": split_seed,
    }
    if null_test:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        kwargs["_model_def"] = _build_null_model(_TASK_MODEL_DEF[task_short], device)
    else:
        kwargs["_model_def"] = tmpl._model_def
        if task_short == "docstring" and os.environ.get("THESIS_MODEL_MIRROR"):
            from .model_assets import pinned_docstring_model
            kwargs["_model_def"] = pinned_docstring_model("cuda" if t.cuda.is_available() else "cpu")
    return Task(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run one (method, task, ablation, seed) discovery cell.")
    parser.add_argument("--task", required=True, choices=list(_TASK_TEMPLATES), help="Task short name")
    parser.add_argument("--algo", required=True, choices=list(_ALGO_MAP), help="Algorithm short name")
    parser.add_argument("--ablation", required=True, choices=list(_ABLATION_MAP), help="Discovery ablation type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (data shuffle + mask init)")
    parser.add_argument("--init-seed", type=int, default=None, help="Mask-init seed, decoupled from --seed (data draw). Omit for the legacy coupled path.")
    parser.add_argument("--protocol", choices=("thesis", "fixed-holdout"), default="thesis")
    parser.add_argument("--split-seed", type=int, default=42, help="Holdout seed for --protocol fixed-holdout")
    parser.add_argument("--legacy-split", action="store_true", help="Reproduce historical per-run train/test membership; unsuitable for common held-out evaluation")
    parser.add_argument("--output-dir", default="results/discovery/scores", help="Cache folder (relative to repo root)")
    parser.set_defaults(null_test=False)
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
    split_seed = None if args.legacy_split or args.protocol == "thesis" else args.split_seed

    # ── Cache identity (computed up front so we can skip/resume) ──
    base_filename = _cache_stem(
        args.task, args.algo, args.ablation, args.seed, args.init_seed,
        args.null_test, args.partial_tag,
        split_seed=split_seed,
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
            and _can_reuse_cache(json.loads((folder / f"{p.stem}_manifest.json").read_text()), args.task, split_seed)
        ]
        if existing:
            print(f"Cell already cached ({existing[-1].name}); skipping. Use --force to recompute.")
            return

    # ── Seed everything before any model/data initialization ──
    t.manual_seed(args.seed)
    random.seed(args.seed)

    ablation_type = _ABLATION_MAP[args.ablation]
    base_algo = _ALGO_MAP[args.algo]
    base_algo = _bind_task_hyperparams(base_algo, args.algo, args.task)

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

    task = _build_task(args.task, args.seed, args.null_test, split_seed=split_seed)

    # --init-seed decouples mask initialisation from the prompt draw. load_datasets_from_json
    # re-seeds torch itself, so the seed has to land after the loaders are built; init_task is
    # idempotent. Omitting the flag, or passing it equal to --seed, leaves the legacy path
    # untouched -- _cache_stem already gives both the same filename, so this makes that
    # equivalence enforced rather than incidental.
    if args.init_seed is not None and args.init_seed != args.seed:
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
            "split_seed": split_seed,
        },
        "cache_file": cache_path.name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "hardware": _hardware(),
        "data_budget": _data_budget(task),
        "algo_hyperparams": algo_hyperparams(algo),
        "data_split": getattr(task.train_loader, "split_metadata", None),
    }
    manifest_path = folder / f"{full_stem}_manifest.json"
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=_serialize)
    os.replace(tmp_manifest, manifest_path)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
