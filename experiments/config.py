"""Portable paths and immutable inputs for the thesis reproduction commands.

This module deliberately has no scientific or GPU dependencies, so inventory and
configuration checks can run before importing the experiment machinery.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "artifacts/final/registry.json"
REFERENCE_REVISIONS = {
    "disco_gp": ("https://github.com/frankniujc/disco_gp.git", "54aef81e7c3803ed7daf09fcf41cb54604bdd272"),
    "edge_pruning": ("https://github.com/princeton-nlp/Edge-Pruning.git", "a918efd8fffcabe9b5853f6e236722e9ec0f4688"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_root(name):
    return Path(os.environ.get("THESIS_REFERENCE_ROOT", ROOT / "third_party")) / name


def load_registry(path=None, *, verify=False):
    path = Path(path or os.environ.get("THESIS_REGISTRY", DEFAULT_REGISTRY)).resolve()
    registry = json.loads(path.read_text())
    if registry.get("schema") != 1:
        raise ValueError("Unsupported thesis registry schema")
    registry["_path"] = str(path)
    if verify:
        for name, checksum in registry["files"].items():
            if sha256(ROOT / name) != checksum:
                raise ValueError(f"Artifact hash mismatch: {name}")
        validate_inventory(registry["benchmark"]["inputs"])
    return registry


def resolve_path(value, registry=None, *, root=ROOT):
    """Resolve published paths, including absolute paths in original provenance."""
    root = Path(root).resolve()
    text = str(value)
    if Path(text).is_absolute():
        absolute = Path(text).resolve()
        if absolute == root or root in absolute.parents:
            return absolute
        for prefix in ("/mnt/beegfs/work/akdag/thesis/auto-circuit/",
                       "/storage/ukp/work/akdag/thesis/auto-circuit/"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
    mapping = (registry or {}).get("path_map", {})
    text = mapping.get(text, text)
    return root / text


def active_sources(registry=None):
    registry = registry or load_registry()
    bundles = {}
    for name, item in registry["benchmark"]["bundles"].items():
        bundles[name] = dict(item)
        for key in ("source_dir", "selected_inputs", "evaluation_manifest", "exhibits_dir"):
            if key in item:
                bundles[name][key] = str(resolve_path(item[key], registry))
    return {"bundles": bundles}


def thesis_evaluation_inputs(registry=None):
    """Return the frozen primary evaluation manifest and score inventory paths."""
    registry = registry or load_registry(DEFAULT_REGISTRY)
    bundle = registry["benchmark"]["bundles"]["main"]
    return (resolve_path(bundle["evaluation_manifest"], registry),
            resolve_path(bundle["selected_inputs"], registry))


def cell_identity(cell):
    seed = int(cell["seed"])
    return (cell["task"], cell["algo"], cell["ablation"], seed,
            int(cell.get("init_seed", seed)), cell.get("split_seed"))


def validate_inventory(items):
    seen = set()
    for item in items:
        identity = cell_identity(item["cell"])
        if identity in seen:
            raise ValueError(f"Duplicate discovery identity: {identity}")
        seen.add(identity)
    return seen


def require_unique_caches(loaded):
    """Reject ambiguous cache directories instead of silently choosing by timestamp."""
    seen = set()
    for manifest, _ in loaded:
        cell = (manifest or {}).get("cell", {})
        identity = cell_identity(cell)
        if identity in seen:
            raise ValueError(f"Duplicate discovery cache identity: {identity}")
        seen.add(identity)
    return loaded


def discovery_cells():
    from .stability_cells import CELLS, DECOMP_CELLS
    cells = [dict(c) for c in CELLS + DECOMP_CELLS]
    cells += [dict(task=task, algo="gt", ablation="resample", seed=42)
              for task in ("ioi", "gt", "docstring")]
    for index, cell in enumerate(cells):
        cell["index"] = index
        cell["seed"] = int(cell["seed"])
        cell["init_seed"] = int(cell.get("init_seed", cell["seed"]))
        cell["split_acdc"] = int(cell["algo"] == "acdc" and cell["task"] in ("ioi", "gt"))
    return cells


def freeze_run(folder, configuration):
    """Refuse to resume a run after scientific inputs or settings change."""
    folder = Path(folder)
    encoded = json.dumps(configuration, sort_keys=True, indent=2) + "\n"
    path = folder / "configuration.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError(f"Run configuration changed: use a new directory ({folder})")
    folder.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_text() != encoded:
                    raise ValueError(f"Run configuration changed: {folder}")
        finally:
            temporary.unlink()
    return path
