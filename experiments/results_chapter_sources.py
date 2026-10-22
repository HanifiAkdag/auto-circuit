"""Resolve frozen scientific inputs shared by the Results exporters."""

import json
from pathlib import Path

from .config import ROOT, active_sources, load_registry, resolve_path, sha256

EXPORT_ROOT = ROOT / "results/reproduction/exports"


def bundle(name="main", registry=None):
    return active_sources(registry or load_registry())["bundles"][name]


def frozen_manifest_index(inputs, tasks, methods, ablations, seeds, *, registry=None,
                          root=ROOT, include_manual=False):
    """Select exact joint-seed inputs; duplicate identities are always errors."""
    records = {}
    for item in inputs:
        cell = item["cell"]
        task, method, ablation, seed = (cell.get(k) for k in ("task", "algo", "ablation", "seed"))
        if task not in tasks or cell.get("null_test"):
            continue
        if method == "gt" and include_manual:
            if ablation != "resample" or seed != 42:
                continue
        elif method not in methods or ablation not in ablations or seed not in seeds:
            continue
        if cell.get("init_seed", seed) != seed:
            continue
        identity = task, method, ablation, seed
        if identity in records:
            raise ValueError(f"Duplicate frozen discovery identity: {identity}")
        paths = {key: resolve_path(item[key], registry, root=root) for key in ("scores", "manifest")}
        for key, path in paths.items():
            if sha256(path) != item[key + "_sha256"]:
                raise ValueError(f"Frozen input hash mismatch: {path}")
        manifest = json.loads(paths["manifest"].read_text())
        if manifest["cell"] != cell:
            raise ValueError(f"Frozen manifest identity mismatch: {identity}")
        records[identity] = manifest, paths["scores"], paths["manifest"]
    return records


def provenance_path(path, root=ROOT):
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
