"""Merge tao-split ACDC partial caches into one canonical cell cache.

ACDC's tao passes are independent: each tao runs a fresh greedy sweep and writes
`min(tao, current)` into a prune-scores tensor initialised to +inf, with non-removed
edges left at +inf. Splitting the tao sweep across parallel jobs (run_cell.py with
--tao-bases/--tao-exps + --partial-tag) and taking the element-wise min across the
partials is therefore identical to one process running all tao (min is commutative /
associative, +inf is its identity, and each tao pass is independent of the others).

Usage:
    python -m experiments.merge_acdc_partials --task ioi --ablation resample \\
        --partials-dir results/new-run/partials --output-dir results/new-run/scores
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch as t

from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path
from .experiment_utils import git_state
from .run_cell import _cache_stem
from .config import sha256


def _serialize(obj: Any) -> Any:
    return str(obj)


def validate_partial_protocols(manifests: list[dict]) -> dict | None:
    first = manifests[0]
    for other in manifests[1:]:
        if (other['cell'].get('split_seed'), other.get('data_split')) != (first['cell'].get('split_seed'), first.get('data_split')):
            raise ValueError('ACDC partial split identities disagree')
        if other.get('data_budget') != first.get('data_budget'):
            raise ValueError('ACDC partial data budgets disagree')
    return first.get('data_split')


def main():
    p = argparse.ArgumentParser(description="Merge tao-split ACDC partials by element-wise min.")
    p.add_argument("--task", required=True)
    p.add_argument("--ablation", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--legacy-split", action="store_true")
    p.add_argument("--partials-dir", required=True,
                   help="Directory containing the explicitly selected partial score files")
    p.add_argument("--output-dir", required=True,
                   help="Destination directory for the merged scores")
    p.add_argument("--expect", type=int, default=5, help="Required partial count (one per tao base)")
    p.add_argument("--output-name", help="Stable cache stem for an idempotent pipeline merge")
    args = p.parse_args()

    split_seed = None if args.legacy_split else args.split_seed
    base_stem = _cache_stem(args.task, 'acdc', args.ablation, args.seed, None, split_seed=split_seed)
    stem = base_stem + '__'
    pdir = repo_path_to_abs_path(args.partials_dir)
    # Ambiguous partials require an explicit inventory rather than mtime selection.
    newest: dict = {}
    for f in pdir.glob(f"{stem}*.pkl"):
        if not f.with_name(f.stem + '_manifest.json').exists():
            continue
        tag = f.name[len(stem):].split("-")[0]
        if tag in newest:
            raise ValueError(f"Duplicate ACDC partial identity: {tag}")
        newest[tag] = f
    partials = [newest[k] for k in sorted(newest)]
    assert partials, f"No partials matching {stem}* in {pdir}"
    assert len(partials) == args.expect, (
        f"Expected {args.expect} tao bases for {args.task}/{args.ablation}, found "
        f"{sorted(newest)}. Merging an incomplete tao sweep would silently produce a wrong cell."
    )
    print(f"Merging {len(partials)} partials for {args.task}/{args.ablation}:")
    for f in partials:
        print(f"  {f.name}")
    partial_manifests = [json.loads(f.with_name(f.stem + '_manifest.json').read_text()) for f in partials]
    data_split = validate_partial_protocols(partial_manifests)
    if partial_manifests[0]['cell'].get('split_seed') != split_seed:
        raise ValueError('Requested split does not match ACDC partials')
    merged_inputs = {f.name: {"scores": sha256(f), "manifest": sha256(f.with_name(f.stem + '_manifest.json'))}
                     for f in partials}
    if args.output_name:
        if Path(args.output_name).name != args.output_name:
            raise ValueError('Output name must be a filename stem')
        previous_score = repo_path_to_abs_path(args.output_dir) / (args.output_name + '.pkl')
        previous_manifest = previous_score.with_name(args.output_name + '_manifest.json')
        if previous_score.exists() and previous_manifest.exists():
            previous = json.loads(previous_manifest.read_text())
            if previous.get('merged_inputs') != merged_inputs or previous.get('scores_sha256') != sha256(previous_score):
                raise ValueError('ACDC merge inputs or completed score changed; use a new run directory')
            print(f'Reusing verified ACDC merge: {previous_score}')
            return

    # Element-wise min across partials (init +inf identity is already baked into each).
    merged: dict = t.load(partials[0], map_location="cpu")
    for f in partials[1:]:
        other = t.load(f, map_location="cpu")
        for task_key, algo_ps in other.items():
            for algo_key, ps in algo_ps.items():
                dst = merged[task_key][algo_key]
                for mod_name, tensor in ps.items():
                    dst[mod_name] = t.minimum(dst[mod_name], tensor)

    n_levels = None
    for task_key, algo_ps in merged.items():
        for algo_key, ps in algo_ps.items():
            flat = t.cat([v.flatten().float() for v in ps.values()])
            n_levels = flat.unique().numel()
            print(f"  merged '{algo_key}': {flat.numel()} edges, {n_levels} score levels, "
                  f"core(inf)={int(t.isinf(flat).sum())}")

    out = repo_path_to_abs_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    stem_out = args.output_name or f"{base_stem}-{dt}"
    cache_path = out / f"{stem_out}.pkl"
    # Write-then-rename, as in run_cell.py: a direct write truncated by preemption would
    # leave a torn .pkl under the final name.
    tmp_cache = cache_path.with_name(cache_path.name + ".tmp")
    t.save(merged, tmp_cache)
    os.replace(tmp_cache, cache_path)
    print(f"Saved merged cache to {cache_path}")

    # Partials all ran the same task, so carry their data budget onto the merged cell;
    # without it the merged cells would be the only ones with no budget recorded.
    partial_manifest = partials[0].with_name(partials[0].name.replace(".pkl", "_manifest.json"))
    budget = None
    if partial_manifest.exists():
        with open(partial_manifest) as f:
            budget = json.load(f).get("data_budget")

    manifest = {
        "cell": {
            "task": args.task, "algo": "acdc",
            "algo_key": list(list(merged.values())[0].keys())[0],
            "ablation": args.ablation, "seed": args.seed, "null_test": False,
            "split_seed": split_seed,
        },
        "cache_file": cache_path.name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "data_budget": budget,
        "data_split": data_split,
        "merged_from": [f.name for f in partials],
        "merged_inputs": merged_inputs,
        "scores_sha256": sha256(cache_path),
        "note": "tao-split ACDC partials merged by element-wise min (identical to single-process 20-tau sweep).",
    }
    manifest_path = out / f"{stem_out}_manifest.json"
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=_serialize)
    os.replace(tmp_manifest, manifest_path)
    print(f"Saved manifest to {out / (stem_out + '_manifest.json')}")


if __name__ == "__main__":
    main()
