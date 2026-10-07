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


def _official_input(task_short, prefix, split_seed, cache_dir, inputs):
    """Select the latest complete official identity without crossing split protocols."""
    candidates = ([(ROOT / item['manifest'], ROOT / item['scores']) for item in inputs
                   if item['cell'].get('task') == task_short and item['cell'].get('algo') == 'gt']
                  if inputs is not None else
                  [(path, None) for path in cache_dir.glob(f'{prefix}*_manifest.json')])
    matching = []
    for path, frozen_scores in candidates:
        manifest = json.loads(path.read_text())
        cell = manifest.get('cell', {})
        if (cell.get('task'), cell.get('algo'), cell.get('ablation'), cell.get('seed'),
            cell.get('init_seed', cell.get('seed')), cell.get('split_seed')) != (
                task_short, 'gt', 'resample', 42, 42, split_seed) or cell.get('null_test'):
            continue
        scores = frozen_scores if frozen_scores is not None else path.parent / manifest['cache_file']
        if scores.is_file():
            matching.append((manifest.get('saved_at', ''), str(path), path, scores, manifest))
    if not matching:
        raise ValueError(f'No complete official baseline cache for {task_short}, split seed {split_seed}')
    _, _, manifest_path, cache_path, manifest = max(matching, key=lambda item: item[:2])
    return manifest_path, cache_path, manifest


def main(eval_manifest=None, selected_inputs=None, output_dir=None, *, split_seed=None,
         cache_dir=None, task_keys=None, source_dir=None):
    from .evaluation_protocol import (build_evaluation_tasks, fixed_evaluation_id,
        load_frozen_inputs, load_protocol, require_same_protocol,
        verify_evaluation_task, verify_holdout_identity)
    from .run_cell import _TASK_TEMPLATES, _data_budget
    if eval_manifest is not None and split_seed is not None:
        raise ValueError('Choose a temporary repair or fixed holdout, not both')
    if selected_inputs is not None and eval_manifest is None:
        raise ValueError('Frozen inputs require a temporary evaluation manifest')
    if split_seed is not None and (source_dir is None or output_dir is None):
        raise ValueError('Fixed holdout requires an analysis source and separate output directory')
    protocol = load_protocol(eval_manifest) if eval_manifest else None
    if protocol and (selected_inputs is None or output_dir is None):
        raise ValueError('Temporary repair requires frozen inputs and a separate output directory')
    inputs = load_frozen_inputs(selected_inputs) if selected_inputs else None
    source_path = (ROOT / source_dir).resolve() / 'manifest.json' if source_dir is not None else None
    source_meta = json.loads(source_path.read_text()) if source_path else None
    if source_meta is not None and split_seed is not None and source_meta.get('complete') is not True:
        raise ValueError('Incomplete analysis source bundle')
    source_tasks = (set(source_meta.get('tasks') or [cell['task'] for cell in source_meta['cells']])
                    if source_meta is not None else None)
    task_shorts = (list(task_keys) if task_keys is not None else
                   list(protocol['tasks']) if protocol else
                   [task for task in TASKS if task in source_tasks] if source_tasks is not None else list(TASKS))
    if not task_shorts or len(set(task_shorts)) != len(task_shorts) or not set(task_shorts) <= set(TASKS):
        raise ValueError('Specify distinct supported baseline tasks')
    if source_tasks is not None and set(task_shorts) != source_tasks:
        raise ValueError('Baseline and analysis task scopes disagree')
    if protocol and not set(task_shorts) <= set(protocol['tasks']):
        raise ValueError('Baseline tasks are absent from the evaluation artifact')
    if protocol or split_seed is not None:
        evaluation_tasks = build_evaluation_tasks(protocol, task_shorts, 42 if split_seed is None else split_seed)
    else:
        evaluation_tasks = {_TASK_TEMPLATES[short].key: TASK_DICT[_TASK_TEMPLATES[short].key]
                            for short in task_shorts}
    evaluation_id = (protocol['protocol_id'] if protocol else
                     fixed_evaluation_id(evaluation_tasks) if split_seed is not None else 'legacy_template')
    if source_meta is not None:
        require_same_protocol({**source_meta,
            'evaluation_protocol_id': source_meta.get('evaluation_protocol_id', 'legacy_template')},
            {'evaluation_protocol_id': evaluation_id})
    cache = (ROOT / cache_dir).resolve() if cache_dir is not None else CACHE
    out = (ROOT / output_dir).resolve() if output_dir is not None else OUT
    rows = []
    completeness_rows = []
    source_records = []
    for task_key, (prefix, ablation_name, expected_edges) in TASKS.items():
        if task_key not in task_shorts:
            continue
        manifest_path, cache_path, cache_manifest = _official_input(
            task_key, prefix, split_seed, cache, inputs)
        task_prune_scores = torch.load(cache_path, weights_only=True, map_location="cpu")
        task_key_loaded = next(iter(task_prune_scores))
        algo_key = next(iter(task_prune_scores[task_key_loaded]))
        task = evaluation_tasks[task_key_loaded]
        if protocol:
            verify_evaluation_task(task, protocol, task_key)
        elif split_seed is not None:
            verify_holdout_identity([cache_manifest], task.test_loader.split_metadata, _data_budget(task))
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
        if protocol or split_seed is not None:
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
                               "cache": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path),
                               "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                               "manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
                               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                               "ablation": ablation_name, "positive_edges": expected_edges,
                               "evaluation_clean_sha256": (task.test_loader.split_metadata['test_clean_sha256']
                                                           if split_seed is not None else
                                                           protocol['tasks'][task_key]['identity_checks']['ordered_clean_tokens_sha256']
                                                           if protocol else None)})

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "evaluation_protocol_id": evaluation_id,
        "complete": False,
        "split_seed": split_seed,
        "temporary_historical_repair": bool(protocol),
        "analysis_manifest": str(source_path) if source_path else None,
        "analysis_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path else None,
        "sources": source_records,
        "metrics": {k: [m.key for m in v] for k, v in TASK_METRICS.items()},
        "test_loader": ("Explicit temporary evaluation Task (256 examples)" if protocol else
                        "Explicit common fixed-holdout evaluation Tasks" if split_seed is not None else
                        "Historical TASK_DICT template test_loader (256 examples)"),
        "patch_type": str(PatchType.TREE_PATCH),
        "selection": "all binary-positive edges in each canonical deterministic official circuit cache",
    }
    manifest_path = out / 'manual_baselines_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with (out / "manual_baselines.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if completeness_rows:
        with (out / 'manual_completeness.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(completeness_rows[0]))
            writer.writeheader()
            writer.writerows(completeness_rows)
    manifest['complete'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Wrote {len(rows)} manual baseline rows")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-manifest', type=Path)
    parser.add_argument('--selected-inputs', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--split-seed', type=int, help='Fixed holdout seed; requires --source-dir and --output-dir')
    parser.add_argument('--cache-dir', type=Path, help='Discovery cache folder containing the official circuit cells')
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS), help='Tasks; must match the analysis source scope')
    parser.add_argument('--source-dir', type=Path, help='Analysis bundle whose evaluation inputs the baseline must match')
    args = parser.parse_args()
    main(args.eval_manifest, args.selected_inputs, args.output_dir,
         split_seed=args.split_seed, cache_dir=args.cache_dir, task_keys=args.tasks,
         source_dir=args.source_dir)
