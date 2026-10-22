#!/usr/bin/env python3
"""Publish the selected final thesis inputs without changing their original bytes.

Run Python on a compute node, for example::

    srun --mem=4G python scripts/package_final_artifacts.py
    srun --mem=4G python scripts/package_final_artifacts.py --verify

Historical provenance is copied verbatim. Only derivative selected-input inventories
and registry.json contain portable published paths. No pickle is deserialized.
"""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


ACTIVE = "results/results_chapter/active_sources.json"
AUDIT = "results/port_fidelity/statistical_audit_20260930"
HEAD = "results/ioi_head_diagnostics/20261011_mean_v3"
HPO = "hpo/results/hpo_disco_paper_ioi_401205_2026-06-17_15-11-54"
PREFIX = "artifacts/final"
EXPECTED_COUNTS = {"benchmark": 342, "main": 228, "docstring": 114,
                   "discogp": 55, "ep": 50}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def relative_source(value):
    """Normalize historical machine roots while rejecting paths outside the repo."""
    path = str(value)
    if Path(path).is_absolute():
        marker = "/auto-circuit/"
        if marker not in path:
            raise ValueError(f"Unrecognized historical path: {path}")
        path = path.split(marker, 1)[1]
    if ".." in Path(path).parts:
        raise ValueError(f"Unsafe path: {path}")
    return path


def verify_bundle(source_root, registry_path=None, expected_counts=None):
    """Verify all published bytes and inventory references using only the stdlib."""
    root = Path(source_root).resolve()
    registry_path = Path(registry_path or root / PREFIX / "registry.json")
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != 1 or registry.get("purpose") != "final_thesis_reproduction":
        raise ValueError("Unsupported final artifact registry")
    def published_path(path):
        if "artifact_root" in registry:
            return registry_path.parent / Path(path).relative_to(registry["artifact_root"])
        return root / path

    for path, expected in registry["files"].items():
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"Unsafe published path: {path}")
        actual = sha256(published_path(path))
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
    counts = {"benchmark": len(registry["benchmark"]["inputs"])}
    inventories = [registry["benchmark"]["inputs"]]
    for name, bundle in registry["benchmark"]["bundles"].items():
        inputs = json.loads(published_path(bundle["selected_inputs"]).read_text())["inputs"]
        counts[name] = len(inputs)
        inventories.append(inputs)
    for name, study in registry["port_fidelity"].items():
        counts[name] = len(study["inputs"])
        inventories.append(study["inputs"])
    if counts != (EXPECTED_COUNTS if expected_counts is None else expected_counts):
        raise ValueError(f"Unexpected selected-input counts: {counts}")
    for inputs in inventories:
        identities = set()
        for item in inputs:
            if item["scores"] in identities:
                raise ValueError(f"Duplicate selected score: {item['scores']}")
            identities.add(item["scores"])
            for kind in ("scores", "manifest"):
                if registry["files"].get(item[kind]) != item[kind + "_sha256"]:
                    raise ValueError(f"Inventory hash mismatch: {item[kind]}")
    return {"files": len(registry["files"]), **counts}


def package(source_root, output_dir):
    root = Path(source_root).resolve()
    output = Path(output_dir).resolve()
    # Registry paths stay repository relative even when staging a comparison copy.
    try:
        prefix = output.relative_to(root).as_posix()
    except ValueError:
        prefix = PREFIX
    files, path_map, planned = {}, {}, {}

    def publish(value, destination, expected=None):
        source = relative_source(value)
        path = root / source
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Source SHA256 mismatch: {source}")
        target = f"{prefix}/{destination}"
        if target in files and files[target] != digest:
            raise ValueError(f"Conflicting artifact destination: {target}")
        files[target], path_map[source], planned[destination] = digest, target, path
        return target

    def derived(destination, value):
        data = json_bytes(value)
        target = f"{prefix}/{destination}"
        files[target] = hashlib.sha256(data).hexdigest()
        planned[destination] = data
        return target

    def publish_inputs(inputs, destination):
        result = copy.deepcopy(inputs)
        for item in result:
            for kind in ("scores", "manifest"):
                item[kind] = publish(item[kind], f"{destination}/{Path(item[kind]).name}",
                                     item[kind + "_sha256"])
        return result

    active = json.loads((root / ACTIVE).read_text())
    publish(ACTIVE, "provenance/active_sources.json")
    benchmark = {"inputs": [], "bundles": {}}
    for name, original in active["bundles"].items():
        bundle = copy.deepcopy(original)
        inventory = relative_source(original["selected_inputs"])
        publish(inventory, f"provenance/{name}_selected_inputs.json")
        inputs = publish_inputs(json.loads((root / inventory).read_text())["inputs"], "benchmark")
        benchmark["inputs"].extend(inputs)
        destination = f"results_chapter/{name}"
        for filename in ("functional.csv", "structural.csv", "summary.csv", "manifest.json"):
            key = filename.split(".")[0] + "_sha256"
            publish(f"{original['source_dir']}/{filename}", f"{destination}/{filename}",
                    original.get(key))
        exhibits = root / original["exhibits_dir"]
        for path in sorted(exhibits.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json"):
                publish(path.relative_to(root), f"{destination}/exhibits/{path.relative_to(exhibits)}")
        bundle.update(source_dir=f"{prefix}/{destination}",
                      exhibits_dir=f"{prefix}/{destination}/exhibits",
                      evaluation_manifest=relative_source(original["evaluation_manifest"]),
                      selected_inputs=derived(f"{destination}/selected_inputs.json", {"inputs": inputs}))
        benchmark["bundles"][name] = bundle

    provenance = json.loads((root / AUDIT / "provenance.json").read_text())
    for path in sorted((root / AUDIT).rglob("*")):
        if path.is_file() and path.suffix in (".csv", ".json"):
            publish(path.relative_to(root), f"port_fidelity/statistical_audit/{path.relative_to(root / AUDIT)}")
    port = {}
    for name, study in provenance["studies"].items():
        directory = "edge_pruning" if name == "ep" else name
        summary = publish(study["summary"], f"port_fidelity/calibration/{name}/summary.csv",
                          study["summary_sha256"])
        port[name] = {
            "inputs": publish_inputs(study["selected_manifests"], f"port_fidelity/{directory}"),
            "source_dir": f"{prefix}/port_fidelity/statistical_audit/{name}",
            "calibration_summary": summary,
            "deltas": study["deltas"],
        }
    calibration = "results/stability/stability_v5_2026-09-19"
    for filename in ("functional.csv", "structural.csv", "summary.csv", "manifest.json"):
        publish(f"{calibration}/{filename}", f"port_fidelity/calibration/stability/{filename}")
    publish("results/port_fidelity/calibrated_margins_20260921.json",
            "port_fidelity/calibration/calibrated_margins_20260921.json")

    head = {"source_dir": f"{prefix}/head_diagnostics"}
    for filename in ("inventory.json", "masks.npz", "per_run.csv", "paired_effects.csv",
                     "effect_summary.csv", "exact_per_run.csv", "complete.json", "smoke.json",
                     "reference.json", "measurement_verification.json", "artifact_verification.json",
                     "active_source_audit.json"):
        head[filename.split(".")[0]] = publish(f"{HEAD}/{filename}", f"head_diagnostics/{filename}")
    protocol = "docs/writing/ioi_retained_head_mean_protocol_20261011.md"
    head["protocol_file"] = publish(protocol, "head_diagnostics/protocol.md")
    hpo = {"source_dir": f"{prefix}/hpo", "original_study": HPO}
    for filename in ("config.json", "outer_scores.json"):
        hpo[filename.split(".")[0]] = publish(f"{HPO}/{filename}", f"hpo/{filename}")
    for fold in range(5):
        for filename in ("optuna_trials.csv", "best_hparams.json", "outer_test_scores.json", "optuna_study.log"):
            publish(f"{HPO}/fold_{fold}/{filename}", f"hpo/fold_{fold}/{filename}")
    models = json.loads((root / "scripts/model_assets.json").read_text())
    publish("scripts/model_assets.json", "provenance/model_assets.json")
    registry = {"schema": 1, "purpose": "final_thesis_reproduction", "artifact_root": prefix,
                "benchmark": benchmark,
                "port_fidelity": port, "head_diagnostics": head, "hpo": hpo,
                "models": models, "path_map": path_map, "files": files}
    planned["registry.json"] = json_bytes(registry)
    existing = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    unexpected = existing - planned.keys()
    if unexpected:
        raise ValueError(f"Unexpected existing artifacts: {sorted(unexpected)}")
    # Validate the entire destination before writing anything.
    for destination, source in planned.items():
        target = output / destination
        expected = hashlib.sha256(source).hexdigest() if isinstance(source, bytes) else sha256(source)
        if target.exists() and sha256(target) != expected:
            raise ValueError(f"Refusing to overwrite differing artifact: {target}")
    for destination, source in planned.items():
        target = output / destination
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, bytes):
                target.write_bytes(source)
            else:
                shutil.copyfile(source, target)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.source_root / PREFIX
    if not args.verify:
        package(args.source_root, output)
    print(json.dumps(verify_bundle(args.source_root, output / "registry.json"), sort_keys=True))


if __name__ == "__main__":
    main()
