"""Evaluate deterministic official circuits at their native task ablations.

Run on a CPU compute node before results_chapter_size_curves.py.
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


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".prune_scores/final/benchmark"
OUT = ROOT / "results/results_chapter/size_curves"
TASKS = {
    "ioi": ("cell_ioi_gt_resample_seed42", "mean", 739),
}
METRICS = [CLEAN_KL_DIV_METRIC, LOGIT_DIFF_PERCENT_METRIC,
           CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC]


def main():
    rows = []
    source_records = []
    for task_key, (prefix, ablation_name, expected_edges) in TASKS.items():
        candidates = sorted(CACHE.glob(f"{prefix}*_manifest.json"))
        assert len(candidates) == 1, (task_key, candidates)
        manifest_path = candidates[0]
        cache_manifest = json.loads(manifest_path.read_text())
        assert cache_manifest["cell"]["algo"] == "gt"
        cache_path = CACHE / cache_manifest["cache_file"]
        task_prune_scores = torch.load(cache_path, weights_only=True, map_location="cpu")
        task_key_loaded = next(iter(task_prune_scores))
        algo_key = next(iter(task_prune_scores[task_key_loaded]))
        task = TASK_DICT[task_key_loaded]
        assert task.true_edge_count == expected_edges
        scores = task_prune_scores[task_key_loaded][algo_key]
        assert sum(int((tensor == 1).sum()) for tensor in scores.values()) == expected_edges
        ablation = (AblationType.TOKENWISE_MEAN_CORRUPT if ablation_name == "mean"
                    else getattr(AblationType, ablation_name.upper()))
        measurements = measure_prune_metrics(
            ablation_types=[ablation], metrics=METRICS,
            task_prune_scores=task_prune_scores,
            patch_type=PatchType.TREE_PATCH, test_edge_counts=[expected_edges],
        )
        for metric in METRICS:
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
        "sources": source_records,
        "metrics": [m.key for m in METRICS],
        "test_loader": "TASK_DICT task template test_loader (256 examples)",
        "patch_type": str(PatchType.TREE_PATCH),
        "selection": "all binary-positive edges in each canonical deterministic official circuit cache",
    }
    (OUT / "manual_baselines_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} manual baseline rows")


if __name__ == "__main__":
    main()
