"""Reproduce the thesis from immutable inputs or a newly completed discovery run.

Invoke through scripts/thesis.sh so Python always runs on a compute node.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .config import (ROOT, DEFAULT_REGISTRY, active_sources, cell_identity,
                     discovery_cells, freeze_run, load_registry, resolve_path,
                     sha256, validate_inventory, REFERENCE_REVISIONS)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def registry_inputs(path):
    registry = load_registry(path, verify=True)
    for item in registry["benchmark"]["inputs"]:
        for field in ("scores", "manifest"):
            source = resolve_path(item[field], registry)
            if sha256(source) != item[field + "_sha256"]:
                raise ValueError(f"Selected {field} hash mismatch: {source}")
    return registry


def action_registry(action, run_dir, requested):
    """Prefer the registry produced by this discovery run for its follow-up stages."""
    generated = Path(run_dir) / "discovery/registry.json"
    if action == "evaluate" and not generated.exists():
        raise ValueError("Complete discover first; use reproduce to evaluate bundled caches")
    if action in ("evaluate", "export") and generated.exists():
        return generated
    return Path(requested)


def validate_protocol(registry, requested):
    registered = registry.get("protocol", "thesis")
    if registered != requested:
        raise ValueError(
            f"Run protocol mismatch: registry uses {registered!r}, "
            f"but --protocol is {requested!r}; use a separate run directory"
        )


def require_evaluation_for_export(registry, run_dir):
    if registry.get("purpose") != "new_thesis_discovery":
        return
    for name in ("main", "docstring"):
        manifest_path = Path(run_dir) / "evaluation" / name / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Evaluate this discovery run before export: missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            raise ValueError(f"Evaluation is incomplete: {manifest_path.parent}")
        if manifest.get("protocol") != registry.get("protocol"):
            raise ValueError(f"Evaluation protocol does not match discovery registry: {manifest_path}")


def compare_value(metric, expected, actual):
    expected, actual = float(expected), float(actual)
    if not math.isfinite(expected) or not math.isfinite(actual):
        return expected == actual
    if "Correct" in metric or "Accuracy" in metric:
        return abs(expected - actual) < 1e-10
    tolerance = 1e-4 if "KL" in metric else 1e-3 if "Logit" in metric else 1e-12
    float_slack = 4 * max(math.ulp(expected), math.ulp(actual))
    return abs(expected - actual) <= tolerance + float_slack


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def compare_csv(expected_path, actual_path):
    expected, actual = read_csv(expected_path), read_csv(actual_path)
    identity_fields = ("task", "algo", "discovery_ablation", "condition", "eval_ablation",
                       "measure", "selector", "budget_label", "metric")
    key = lambda row: tuple(row.get(k, "") for k in identity_fields)
    old, new = {key(r): r for r in expected}, {key(r): r for r in actual}
    if len(old) != len(expected) or len(new) != len(actual) or old.keys() != new.keys():
        raise ValueError(f"CSV identities disagree: {expected_path} vs {actual_path}")
    mismatches = []
    for identity, before in old.items():
        after = new[identity]
        for field, value in before.items():
            if field in identity_fields or field in ("metric_cv", "jaccard_cv"):
                continue
            if field == "metric_values":
                left, right = value.split(";"), after[field].split(";")
                # Historical per-seed exports use six significant figures.
                good = len(left) == len(right) and all(
                    compare_value(before.get("metric", ""), a, b) or
                    (float(a) != 0 and abs(float(a) - float(b)) <=
                     .5 * 10 ** (math.floor(math.log10(abs(float(a)))) - 5))
                    for a, b in zip(left, right))
            else:
                try:
                    good = compare_value(before.get("metric", "") if field.startswith("metric_") else "",
                                         value, after[field])
                except ValueError:
                    good = value == after[field]
            if not good:
                mismatches.append({"identity": identity, "field": field,
                                   "expected": value, "actual": after.get(field)})
    if mismatches:
        report = Path(actual_path).with_suffix(".mismatches.json")
        atomic_json(report, mismatches)
        raise ValueError(f"{len(mismatches)} numerical mismatches: {report}")
    return {"rows": len(expected), "matched": True}


def compare_supplement(reference, actual, keys):
    """Compare supplementary outputs without relaxing recorded metric tolerances."""
    before, after = read_csv(reference), read_csv(actual)
    identity = lambda row: tuple(row.get(k, "") for k in keys)
    old, new = {identity(r): r for r in before}, {identity(r): r for r in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError(f"Supplementary output identities disagree: {actual}")
    mismatches = []
    for key, expected in old.items():
        for field, value in expected.items():
            measured = new[key].get(field)
            if value == measured or field in keys:
                continue
            # Older frozen summaries leave later-added audit metadata blank.
            if value == "" and field not in keys:
                continue
            metric = expected.get("endpoint", expected.get("metric", field)).lower()
            if field.startswith(("p_", "n_")) or field in ("exact", "equivalent", "reading"):
                tolerance = 1e-12
            elif "accuracy" in metric:
                tolerance = 1e-10
            elif "logit_difference_percent" in metric or "logit diff" in metric:
                tolerance = 1e-3
            else:
                tolerance = 1e-4
            try:
                if ";" in value:
                    left, right = value.split(";"), measured.split(";")
                    good = len(left) == len(right) and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))
                else:
                    good = abs(float(value) - float(measured)) <= tolerance
            except (ValueError, TypeError):
                good = False
            if not good:
                mismatches.append({"identity": key, "field": field, "expected": value, "actual": measured})
    if mismatches:
        report = Path(actual).with_suffix(".mismatches.json")
        atomic_json(report, mismatches)
        raise ValueError(f"Supplementary measurements changed: {report}")
    return {"matched": True, "rows": len(before)}


@contextlib.contextmanager
def arguments(module, args):
    old = sys.argv
    sys.argv = [module.__name__, *map(str, args)]
    try:
        yield
    finally:
        sys.argv = old


def invoke(name, args=()):
    module = importlib.import_module("experiments." + name)
    with arguments(module, args):
        module.main()


def code_identity():
    files = sorted((ROOT / "auto_circuit").rglob("*.py"))
    files += sorted((ROOT / "experiments").rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def configure_models(registry, run_dir):
    """Prepare and verify pinned assets before importing any model libraries."""
    from .model_assets import install_model_pins
    return install_model_pins(registry, run_dir)


def verify_environment():
    from importlib.metadata import PackageNotFoundError, version
    expected = {}
    for line in (ROOT / "requirements-thesis.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, pinned = line.split("==", 1)
        try:
            actual = version(name)
        except PackageNotFoundError as error:
            raise ValueError(f"Required package is missing: {name}=={pinned}") from error
        if actual != pinned:
            raise ValueError(f"Pinned dependency mismatch: {name}=={pinned}, found {actual}")
        expected[name] = actual
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, check=True)
    return expected

def discover(args, registry):
    cells = discovery_cells()
    configuration = {"protocol": args.protocol, "smoke": False, "models": registry["models"],
                     "code": code_identity(), "cells": cells}
    freeze_run(args.run_dir / "discovery", configuration)
    split = [c for c in cells if c["split_acdc"]]
    if args.finalize:
        for cell in split:
            command = ["--task", cell["task"], "--ablation", cell["ablation"],
                       "--seed", cell["seed"], "--partials-dir", args.run_dir / "discovery/partials",
                       "--output-dir", args.run_dir / "discovery/scores",
                       "--output-name", f"cell_{cell['task']}_acdc_{cell['ablation']}_seed{cell['seed']}" + ("_holdout42" if args.protocol == "fixed-holdout" else "")]
            if args.protocol == "thesis":
                command.append("--legacy-split")
            invoke("merge_acdc_partials", command)
        publish_discovery(args, registry, cells)
        return
    if args.partial_index is not None:
        cell = split[args.partial_index // 5]
        base = (1, 3, 5, 7, 9)[args.partial_index % 5]
    elif args.cell_index is not None:
        cell = cells[args.cell_index]
        if cell["split_acdc"]:
            raise ValueError("Use the ACDC partial array for this cell")
        base = None
    else:
        raise ValueError("Discovery requires --cell-index, --partial-index or --finalize")
    command = ["--task", cell["task"], "--algo", cell["algo"], "--ablation", cell["ablation"],
               "--seed", cell["seed"], "--init-seed", cell["init_seed"], "--skip-if-exists",
               "--output-dir", args.run_dir / ("discovery/partials" if base else "discovery/scores")]
    if args.protocol == "thesis":
        command.append("--legacy-split")
    else:
        command += ["--protocol", "fixed-holdout"]
    if base:
        command += ["--tao-bases", base, "--partial-tag", f"base{base}"]
    invoke("run_cell", command)


def publish_discovery(args, base_registry, cells):
    inputs = []
    directory = args.run_dir / "discovery/scores"
    for manifest in sorted(directory.glob("*_manifest.json")):
        record = json.loads(manifest.read_text())
        score = directory / record["cache_file"]
        inputs.append({"cell": record["cell"], "data_budget": record["data_budget"],
                       "scores": str(score), "scores_sha256": sha256(score),
                       "manifest": str(manifest), "manifest_sha256": sha256(manifest)})
    identities = validate_inventory(inputs)
    holdout = None if args.protocol == "thesis" else 42
    expected = {cell_identity({**c, "split_seed": holdout}) for c in cells}
    if identities != expected:
        raise ValueError(f"Incomplete discovery inventory: {len(identities)} of {len(expected)} identities")
    registry = dict(base_registry)
    registry.pop("_path", None)
    registry["purpose"] = "new_thesis_discovery"
    registry["protocol"] = args.protocol
    registry["files"] = {}
    registry["benchmark"] = {"inputs": inputs, "bundles": {}}
    for name, shorts in (("main", ("ioi", "gt")), ("docstring", ("docstring",))):
        selected = [i for i in inputs if i["cell"]["task"] in shorts]
        inventory = args.run_dir / "discovery" / name / "selected_inputs.json"
        atomic_json(inventory, {"inputs": selected})
        registry["benchmark"]["bundles"][name] = {
            **base_registry["benchmark"]["bundles"][name], "selected_inputs": str(inventory)}
    atomic_json(args.run_dir / "discovery/registry.json", registry)


def evaluate_bundle(args, registry, name):
    import torch
    from auto_circuit.metrics.analysis_utils import Budget, edge_index
    from auto_circuit.types import PatchType
    from . import stability_analysis as sa
    from .evaluation_protocol import build_evaluation_tasks, load_protocol, load_frozen_inputs, verify_evaluation_task
    bundle = registry["benchmark"]["bundles"][name]
    task_shorts = ["ioi", "gt"] if name == "main" else ["docstring"]
    inputs_path = resolve_path(bundle["selected_inputs"], registry)
    inputs = load_frozen_inputs(inputs_path)
    protocol = (load_protocol(resolve_path(bundle["evaluation_manifest"], registry))
                if args.protocol == "thesis" else None)
    split_seed = None if protocol else 42
    tasks = build_evaluation_tasks(protocol, task_shorts)
    output = args.run_dir / "evaluation" / name
    specs = sa._cell_specs(task_shorts, sa.SWEPT_ALGOS + sa.ANCHOR_ALGOS,
                           sa.DISCOVERY_ABLATIONS, sa.CONDITIONS)
    freeze_run(output, {"protocol": args.protocol, "inventory_sha256": sha256(inputs_path),
                        "protocol_id": protocol["protocol_id"] if protocol else "fixed_holdout42",
                        "code": code_identity(), "models": registry["models"]})
    structural, functional, cells = [], [], []
    for task_short, algo, ablation, condition in specs:
        spec = (task_short, algo, ablation, condition)
        cell = sa._load_cell("", spec, sa.SEEDS, split_seed=split_seed, frozen_inputs=inputs)
        if cell.seeds != sa.SEEDS:
            raise ValueError(f"Incomplete cell: {spec}: {cell.seeds}")
        cells.append(dict(task=task_short, algo=algo, ablation=ablation, condition=condition, seeds=cell.seeds))
        effective = (task_short, algo, ablation, sa._effective_condition(algo, condition))
        group_path = output / "cells" / ("_".join(effective) + ".json")
        if group_path.exists():
            group = json.loads(group_path.read_text())
            if not group.get("complete"):
                raise ValueError(f"Incomplete cell output: {group_path}")
        else:
            task = tasks[cell.task_key]
            if protocol:
                verify_evaluation_task(task, protocol, task_short)
            else:
                from .evaluation_protocol import verify_holdout_identity
                from .run_cell import _data_budget
                verify_holdout_identity(cell.manifests, task.test_loader.split_metadata, _data_budget(task))
            index = edge_index(task.model.edges, cell.scores[0])
            budgets = sa._budgets(cell, task.model.edges, task.true_edge_count)
            structural_rows = sa._structural_rows(cell, index, budgets, len(sa.SEEDS))
            if algo != "acdc":
                requested = sorted(set(sa.FUNCTIONAL_BUDGETS + [task.true_edge_count]))
                budgets = [Budget("top_k", n) for n in requested if 0 < n < len(index)] + [Budget("top_k", len(index))]
            device = next(task.model.parameters()).device
            scores = [{k: v.float().to(device) for k, v in ps.items()} for ps in cell.scores]
            functional_rows = sa._functional_rows(cell, index, budgets, scores,
                [PatchType.TREE_PATCH, PatchType.EDGE_PATCH], evaluation_tasks=tasks)
            group = {"complete": True, "structural": structural_rows, "functional": functional_rows}
            atomic_json(group_path, group)
            print(f"Evaluated {spec}: {len(functional_rows)} endpoints", flush=True)
        structural.extend(dict(row, condition=condition) for row in group["structural"])
        functional.extend(dict(row, condition=condition) for row in group["functional"])
    sa._write_csv(output / "structural.csv", structural, sa.STRUCTURAL_FIELDS)
    sa._write_csv(output / "functional.csv", functional, sa.FUNCTIONAL_FIELDS)
    task_keys = {task.name: key for key, task in tasks.items()}
    summary = sa._summary_rows(structural, functional, task_keys, evaluation_tasks=tasks)
    sa._write_csv(output / "summary.csv", summary, list(summary[0]))
    manifest = {"complete": False, "tasks": task_shorts, "seeds": sa.SEEDS, "conditions": sa.CONDITIONS,
                "cells": cells, "selected_inputs": str(inputs_path), "selected_inputs_sha256": sha256(inputs_path),
                "evaluation_protocol_id": protocol["protocol_id"] if protocol else sa.fixed_evaluation_id(tasks),
                "functional_rows": len(functional), "structural_rows": len(structural),
                "stability_metric": sa.STABILITY_METRIC, "protocol": args.protocol}
    atomic_json(output / "manifest.json", manifest)
    if registry.get("purpose") == "final_thesis_reproduction":
        reference = resolve_path(bundle["source_dir"], registry)
        checks = {field: compare_csv(reference / (field + ".csv"), output / (field + ".csv"))
                  for field in ("structural", "functional")}
        atomic_json(output / "comparison.json", checks)
    manifest["complete"] = True
    atomic_json(output / "manifest.json", manifest)
    from . import results_chapter_manual_baseline as manual
    manual.main(eval_manifest=resolve_path(bundle["evaluation_manifest"], registry) if protocol else None,
                selected_inputs=inputs_path,
                output_dir=output / "exhibits/size_curves",
                split_seed=split_seed, task_keys=task_shorts, source_dir=output)
    return output


def evaluate(args, registry):
    for name in ("main", "docstring"):
        evaluate_bundle(args, registry, name)


def export(args, registry):
    require_evaluation_for_export(registry, args.run_dir)
    from . import results_chapter_size_curves as curves
    for name, tasks in (("main", ["ioi", "gt"]), ("docstring", ["docstring"])):
        source = args.run_dir / "evaluation" / name
        if not (source / "manifest.json").is_file():
            if source.exists():
                raise ValueError(f"Evaluation is incomplete: {source}")
            # Export-only commands use published measurements, with no GPU forward calls.
            source = resolve_path(registry["benchmark"]["bundles"][name]["source_dir"], registry)
            output = args.run_dir / "exports" / name
            import shutil
            baseline = resolve_path(registry["benchmark"]["bundles"][name]["exhibits_dir"], registry) / "size_curves"
            (output / "size_curves").mkdir(parents=True, exist_ok=True)
            for file in baseline.glob("manual*.*"):
                shutil.copyfile(file, output / "size_curves" / file.name)
        else:
            output = source / "exhibits"
        curves = importlib.reload(curves)
        curves.main(source_dir=source, output_dir=output, task_keys=tasks)
        if name == "docstring":
            from . import results_chapter_stability as stability
            rows = [stability.select_cell(read_csv(source / "summary.csv"), read_csv(source / "structural.csv"),
                    "Docstring", method, ablation, "joint", 24)
                    for method in stability.METHODS for ablation in stability.ABLATIONS]
            (output / "stability").mkdir(parents=True, exist_ok=True)
            stability._write_csv(output / "stability/within_ablation_selected.csv", rows)
        if name == "main":
            from . import results_chapter_controlled_comparison as quality
            from . import results_chapter_stability as stability
            from . import results_chapter_evaluation_ablation as crossed
            from . import results_chapter_completeness as completeness
            quality.main(output_root=output)
            stability.main(source_dir=source, output_root=output)
            crossed.main(source_dir=source, output_root=output)
            completeness.main(source_dir=source, output_root=output)
            from . import results_chapter_cross_discovery as overlap
            overlap.main(source_dir=source, output_root=output, inputs=registry["benchmark"]["inputs"])
    from . import results_chapter_native_roc as roc
    roc.main(registry=registry, output_root=args.run_dir / "exports")
    from . import results_chapter_structure_function as paired
    paired.run(ROOT, args.run_dir / "exports/structure_function", registry=registry,
               source_dir=args.run_dir / "evaluation/main" if (args.run_dir / "evaluation/main/manifest.json").exists() else None)
    paired.plot(args.run_dir / "exports/structure_function")
    source = resolve_path(registry["hpo"]["source_dir"], registry)
    historical = []
    for fold in range(5):
        folder = source / f"fold_{fold}"
        trials = read_csv(folder / "optuna_trials.csv")
        winner = max((r for r in trials if r["state"] == "COMPLETE"), key=lambda r: float(r["value"]))
        outer = json.loads((folder / "outer_test_scores.json").read_text())
        historical.append(dict(fold=fold, trials=len(trials), best_inner_logit_diff_auc=float(winner["value"]), **outer))
    destination = args.run_dir / "exports/hpo/fold_results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(historical[0]))
        writer.writeheader()
        writer.writerows(historical)


def references(args):
    destination = args.run_dir / "references"
    for name, (url, revision) in REFERENCE_REVISIONS.items():
        path = destination / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
        if actual != revision or dirty:
            raise ValueError(f"Reference checkout differs from pinned source: {path}")
    os.environ["THESIS_REFERENCE_ROOT"] = str(destination)


def port_validation(args, registry, *, rediscover=False):
    from .port_comparison import fidelity_analysis as analysis
    if rediscover:
        references(args)
        from .port_comparison.ep_cells import CELLS as ep_cells
        from .port_comparison.run_cells import ARMS
        for study, cells, module in (("ep", ep_cells, "ep_run_cells"),
                                    ("discogp", [(a, s) for a in ARMS for s in range(10 if a in ("port", "orig") else 5)], "run_cells")):
            for arm, seed in cells:
                invoke("port_comparison." + module, ["--arm", arm, "--seed", seed, "--output-dir", args.run_dir / "port_discovery" / study])
    for study in ("discogp", "ep"):
        inputs = registry["port_fidelity"][study]["inputs"]
        cache = (args.run_dir / "port_discovery" / study if rediscover else
                 resolve_path(inputs[0]["scores"], registry).parent)
        output = args.run_dir / "port_validation" / study
        with arguments(analysis, ["--study", study, "--cache-dir", cache, "--out", output]):
            analysis.main()
        atomic_json(output / "provenance.json", {"complete": True, "inputs": inputs,
                    "models": registry["models"], "rediscovered": rediscover})
        if not rediscover:
            reference = resolve_path(registry["port_fidelity"][study]["source_dir"], registry)
            comparison = compare_supplement(reference / "summary.csv", output / "summary.csv", ("comparison",))
            atomic_json(output / "comparison.json", comparison)


def head_diagnostics(args, registry):
    from .ioi_head_diagnostics import protocol, evaluation, analysis
    output = args.run_dir / "head_diagnostics"
    inventory = protocol.inventory(output)
    historical = json.loads(resolve_path(registry["head_diagnostics"]["source_dir"], registry).joinpath("inventory.json").read_text())
    import numpy as np
    expected_masks = np.load(resolve_path(registry["head_diagnostics"]["source_dir"], registry) / "masks.npz")["masks"]
    actual_masks = np.load(output / "masks.npz")["masks"]
    if inventory["graph"] != historical["graph"] or inventory["conditions"] != historical["conditions"] or not np.array_equal(actual_masks, expected_masks):
        raise ValueError("Head diagnostic scientific inventory changed")
    evaluation.smoke(output)
    for phase in ("screen", "causal", "size-sensitivity"):
        for cell in range(20):
            evaluation.worker(output, phase, cell)
    analysis.summarize(output)
    reference = resolve_path(registry["head_diagnostics"]["source_dir"], registry)
    comparisons = {}
    for filename, keys in (("per_run.csv", ("condition_id",)),
                           ("paired_effects.csv", ("algo", "seed", "label", "phase", "realization", "budget")),
                           ("effect_summary.csv", ("algo", "budget", "label", "metric")),
                           ("exact_per_run.csv", ("algo", "seed", "budget", "label"))):
        comparisons[filename] = compare_supplement(reference / filename, output / filename, keys)
    atomic_json(output / "comparison.json", comparisons)


def hpo(args, registry, *, smoke=False):
    source = resolve_path(registry["hpo"]["source_dir"], registry)
    config = json.loads((source / "config.json").read_text())
    output = args.run_dir / ("hpo_smoke" if smoke else "hpo")
    effective_config = {key: config[key] for key in (
        "algo", "task", "outer_k", "inner_mode", "inner_k", "inner_val_frac",
        "cv_batch_size", "n_trials", "selection_metric", "pruner", "min_resource",
        "reduction_factor", "max_epochs_cap", "eval_every", "ablation_type",
        "eval_edge_counts", "seed", "worker_id", "storage_backend")}
    effective_config.update({"output_dir": str(output), "smoke": smoke,
                             "base_cache": None, "run_comparison_algos": False})
    if smoke:
        effective_config.update({"outer_k": 2, "n_trials": 1, "pruner": "none",
                                 "min_resource": 1, "max_epochs_cap": 1,
                                 "eval_every": 1, "eval_edge_counts": "10,739,2000"})
    run_config = {"effective_config": effective_config, "protocol": args.protocol,
                  "source_config_sha256": sha256(source / "config.json"),
                  "source_trial_hashes": {f"fold_{fold}": sha256(source / f"fold_{fold}/optuna_trials.csv")
                                           for fold in range(5)},
                  "source_outer_hashes": {f"fold_{fold}": sha256(source / f"fold_{fold}/outer_test_scores.json")
                                           for fold in range(5)},
                  "registry_sha256": sha256(args.registry), "code": code_identity()}
    freeze_run(output, run_config)
    command = ["--algo", config["algo"], "--task", config["task"], "--output-dir", output,
               "--outer-k", 2 if smoke else config["outer_k"],
               "--inner-mode", config["inner_mode"], "--inner-k", config["inner_k"],
               "--inner-val-frac", config["inner_val_frac"],
               "--n-trials", 1 if smoke else config["n_trials"],
               "--max-epochs-cap", 1 if smoke else config["max_epochs_cap"],
               "--selection-metric", config["selection_metric"],
               "--pruner", config["pruner"],
               "--min-resource", config["min_resource"],
               "--reduction-factor", config["reduction_factor"],
               "--eval-every", config["eval_every"],
               "--ablation-type", *config["ablation_type"],
               "--storage-backend", config["storage_backend"],
               "--seed", config["seed"], "--worker-id", config["worker_id"]]
    if config.get("cv_batch_size") is not None:
        command += ["--cv-batch-size", config["cv_batch_size"]]
    if config.get("eval_edge_counts"):
        command += ["--eval-edge-counts", config["eval_edge_counts"]]
    if config.get("run_comparison_algos"):
        command.append("--run-comparison-algos")
    if smoke:
        command += ["--pruner", "none", "--min-resource", 1, "--eval-every", 1, "--eval-edge-counts", "10,739,2000"]
    import hpo.nested_cv as worker
    with arguments(worker, command):
        worker.main()
    import hpo.finalize as finalize
    with arguments(finalize, ["--output-dir", args.run_dir / ("hpo_smoke" if smoke else "hpo")]):
        finalize.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "reproduce", "discover", "evaluate", "export", "port-validation", "head-diagnostics", "hpo", "smoke", "test"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/reproduction")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--protocol", choices=("thesis", "fixed-holdout"), default="thesis")
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--partial-index", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runexternal", action="store_true")
    parser.add_argument("--runslow", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.dry_run:
        print(json.dumps({"action": args.action, "protocol": args.protocol, "registry": str(args.registry), "run_dir": str(args.run_dir), "discovery_cells": len(discovery_cells())}, indent=2))
        return
    args.registry = action_registry(args.action, args.run_dir, args.registry)
    registry = registry_inputs(args.registry)
    if args.action in ("evaluate", "export"):
        validate_protocol(registry, args.protocol)
    os.environ["THESIS_REGISTRY"] = str(args.registry.resolve())
    os.environ["THESIS_RUN_DIR"] = str(args.run_dir)
    if args.action == "verify":
        packages = verify_environment()
        snapshots = str(configure_models(registry, args.run_dir))
        atomic_json(args.run_dir / "verification/preflight.json", {"complete": True,
                    "benchmark_runs": len(registry["benchmark"]["inputs"]),
                    "packages": packages, "models": snapshots})
        print("Verified final inputs and pinned model snapshots")
        return
    if args.action == "test":
        command = [sys.executable, "-m", "pytest", "-q"]
        command += [flag for flag in ("--runexternal", "--runslow") if getattr(args, flag[2:])]
        subprocess.run(command, cwd=ROOT, check=True)
        return
    configure_models(registry, args.run_dir)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.protocol != "thesis" and args.action in ("reproduce", "head-diagnostics", "port-validation"):
        raise ValueError("The packaged thesis evidence requires --protocol thesis")
    if args.action == "reproduce":
        evaluate(args, registry)
        export(args, registry)
        port_validation(args, registry)
        head_diagnostics(args, registry)
        atomic_json(args.run_dir / "verification/complete.json", {"complete": True, "registry_sha256": sha256(args.registry), "code": code_identity()})
    elif args.action == "discover":
        discover(args, registry)
    elif args.action == "evaluate":
        evaluate(args, registry)
    elif args.action == "export":
        export(args, registry)
    elif args.action == "port-validation":
        port_validation(args, registry, rediscover=True)
    elif args.action == "head-diagnostics":
        head_diagnostics(args, registry)
    elif args.action == "hpo":
        hpo(args, registry)
    elif args.action == "smoke":
        from .smoke import run
        run(args, registry)


if __name__ == "__main__":
    main()
