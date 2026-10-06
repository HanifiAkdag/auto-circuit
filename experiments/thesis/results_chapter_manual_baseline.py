"""Evaluate deterministic official circuits at their native task ablations.

Run on a CPU compute node before `experiments.thesis.results_chapter_size_curves`.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

from auto_circuit.metrics.analysis_utils import extract_metric
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import measure_prune_metrics
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
)
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import AblationType, PatchType


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".prune_scores/final/benchmark"
OUT = ROOT / "results/results_chapter/size_curves"
TASKS = {
    "ioi": ("cell_ioi_gt_resample_seed42", "mean", 739),
    "gt": ("cell_gt_gt_resample_seed42", "resample", 235),
    "docstring": ("cell_docstring_gt_resample_seed42", "resample", 24),
}
TASK_METRICS = {
    "ioi": [CLEAN_KL_DIV_METRIC, LOGIT_DIFF_PERCENT_METRIC,
            CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC],
    "gt": [CLEAN_KL_DIV_METRIC, LOGIT_DIFF_PERCENT_METRIC],
    "docstring": [CLEAN_KL_DIV_METRIC,
                   CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC],
}


def main(eval_manifest=None, selected_inputs=None, output_dir=None):
    global OUT
    from .evaluation_protocol import build_evaluation_tasks, load_frozen_inputs, load_protocol, verify_evaluation_task
    from .run_cell import _TASK_TEMPLATES
    protocol = load_protocol(eval_manifest) if eval_manifest else None
    if protocol and (selected_inputs is None or output_dir is None):
        raise ValueError('Temporary repair requires frozen inputs and a separate output directory')
    inputs = load_frozen_inputs(selected_inputs) if selected_inputs else None
    task_shorts = list(protocol['tasks']) if protocol else list(TASKS)
    evaluation_tasks = (build_evaluation_tasks(protocol, task_shorts) if protocol else
                        {template.key: TASK_DICT[template.key] for template in _TASK_TEMPLATES.values()})
    if output_dir:
        OUT = (ROOT / output_dir).resolve()
    rows = []
    completeness_rows = []
    source_records = []
    for task_key, (prefix, ablation_name, expected_edges) in TASKS.items():
        if task_key not in task_shorts:
            continue
        candidates = ([ROOT / item['manifest'] for item in inputs
                       if item['cell'].get('task') == task_key and item['cell'].get('algo') == 'gt']
                      if inputs else sorted(CACHE.glob(f"{prefix}*_manifest.json")))
        assert len(candidates) == 1, (task_key, candidates)
        manifest_path = candidates[0]
        cache_manifest = json.loads(manifest_path.read_text())
        assert cache_manifest["cell"]["algo"] == "gt"
        cache_path = CACHE / cache_manifest["cache_file"]
        task_prune_scores = torch.load(cache_path, weights_only=True, map_location="cpu")
        task_key_loaded = next(iter(task_prune_scores))
        algo_key = next(iter(task_prune_scores[task_key_loaded]))
        task = evaluation_tasks[task_key_loaded]
        if protocol:
            verify_evaluation_task(task, protocol, task_key)
        assert task.true_edge_count == expected_edges
        scores = task_prune_scores[task_key_loaded][algo_key]
        assert sum(int((tensor == 1).sum()) for tensor in scores.values()) == expected_edges
        device = next(task.model.parameters()).device
        task_prune_scores[task_key_loaded][algo_key] = {name: value.to(device) for name, value in scores.items()}
        ablation = (AblationType.TOKENWISE_MEAN_CORRUPT if ablation_name == "mean"
                    else getattr(AblationType, ablation_name.upper()))
        measurements = measure_prune_metrics(
            ablation_types=[ablation], metrics=TASK_METRICS[task_key],
            task_prune_scores=task_prune_scores,
            patch_type=PatchType.TREE_PATCH, test_edge_counts=[expected_edges],
            evaluation_tasks=evaluation_tasks,
        )
        for metric in TASK_METRICS[task_key]:
            series = extract_metric(measurements, ablation, metric.key,
                                    task_key_loaded, algo_key)
            assert len(series) == 1 and int(series[0][0]) == expected_edges, series
            value = float(series[0][1])
            rows.append({"task": task.name, "algo": "Manual circuit",
                         "discovery_ablation": ablation_name, "eval_ablation": ablation_name,
                         "measure": "faithfulness", "selector": "manual",
                         "budget_label": str(expected_edges), "realised_mean": expected_edges,
                         "metric": metric.key, "n_seeds": 1,
                         "metric_mean": value, "metric_sd": 0.0,
                         "metric_min": value, "metric_max": value})
        if protocol:
            completeness = measure_prune_metrics(
                ablation_types=[ablation], metrics=TASK_METRICS[task_key], task_prune_scores=task_prune_scores,
                patch_type=PatchType.EDGE_PATCH, reverse_clean_corrupt=True, test_edge_counts=[expected_edges],
                evaluation_tasks=evaluation_tasks)
            for metric in TASK_METRICS[task_key]:
                value = extract_metric(completeness, ablation, metric.key, task_key_loaded, algo_key)[0][1]
                completeness_rows.append({'task': task.name, 'algo': 'GT', 'measure': 'completeness',
                    'discovery_ablation': ablation_name, 'eval_ablation': ablation_name,
                    'metric': metric.key, 'budget_edges': expected_edges, 'value': value})
        source_records.append({"task_key": task_key, "task_name": task.name,
                               "cache": str(cache_path.relative_to(ROOT)),
                               "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                               "manifest": str(manifest_path.relative_to(ROOT)),
                               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                               "ablation": ablation_name, "positive_edges": expected_edges})

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUT / "manual_baselines.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "evaluation_protocol_id": protocol['protocol_id'] if protocol else 'legacy_template',
        "temporary_historical_repair": bool(protocol),
        "sources": source_records,
        "metrics": {k: [m.key for m in v] for k, v in TASK_METRICS.items()},
        "test_loader": "Explicit temporary evaluation Task (256 examples)" if protocol else "Historical TASK_DICT template test_loader (256 examples)",
        "patch_type": str(PatchType.TREE_PATCH),
        "selection": "all binary-positive edges in each canonical deterministic official circuit cache",
    }
    (OUT / "manual_baselines_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if completeness_rows:
        with (OUT / 'manual_completeness.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(completeness_rows[0]))
            writer.writeheader()
            writer.writerows(completeness_rows)
    print(f"Wrote {len(rows)} manual baseline rows")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-manifest', type=Path)
    parser.add_argument('--selected-inputs', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    main(args.eval_manifest, args.selected_inputs, args.output_dir)
