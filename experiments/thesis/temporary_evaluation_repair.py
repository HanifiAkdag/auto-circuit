"""Temporary evaluation of frozen historical circuits; never a discovery data source."""
import argparse
import json
import csv
import os
from datetime import datetime
from pathlib import Path

from .evaluation_protocol import ROOT, build_evaluation_tasks, load_frozen_inputs, load_protocol, verify_evaluation_task

HISTORICAL = ROOT / 'results/stability/stability_v5_2026-09-19'


def _rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _atomic_output(path, write, *args):
    staged = path.with_name('.' + path.name + '.tmp')
    write(staged, *args)
    os.replace(staged, path)


def repair_specs(task_shorts):
    from .stability_analysis import _effective_condition
    specs = {}
    for cell in json.loads((HISTORICAL / 'manifest.json').read_text())['cells']:
        if cell['task'] not in task_shorts:
            continue
        key = (cell['task'], cell['algo'], cell['ablation'], _effective_condition(cell['algo'], cell['condition']))
        specs.setdefault(key, cell)
    return list(specs.values())


def run_cell(index, protocol_path, inputs_path, output, task_shorts):
    import torch
    from auto_circuit.metrics.analysis_utils import Budget, edge_index
    from auto_circuit.types import PatchType
    from .evaluation_protocol import sha256
    from .stability_analysis import _functional_rows, _load_cell
    spec = repair_specs(task_shorts)[index]
    protocol = load_protocol(protocol_path)
    inputs = load_frozen_inputs(inputs_path)
    complete_id = {'evaluation_protocol_id': protocol['protocol_id'],
                   'selected_inputs_sha256': sha256(inputs_path),
                   'historical_functional_sha256': sha256(HISTORICAL / 'functional.csv'),
                   'spec': spec}
    path = output / 'cells' / f'{index:03d}.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous.get(k) != v for k, v in complete_id.items()) or not previous.get('complete'):
            raise ValueError('Repair resume inputs disagree')
        print('Already evaluated', spec)
        return
    cell = _load_cell(str(ROOT / '.prune_scores/final/benchmark'),
        (spec['task'], spec['algo'], spec['ablation'], spec['condition']), spec['seeds'], frozen_inputs=inputs)
    assert cell.seeds == spec['seeds']
    tasks = build_evaluation_tasks(protocol, [cell.task])
    task = tasks[cell.task_key]
    verify_evaluation_task(task, protocol, cell.task)
    index_map = edge_index(task.model.edges, cell.scores[0])
    old_rows = [r for r in _rows(HISTORICAL / 'functional.csv') if
        (r['task'], r['algo'], r['discovery_ablation'], r['condition']) ==
        (task.name, cell.short_name, 'anchor' if cell.is_anchor else cell.ablation, cell.condition)]
    assert old_rows
    selections = sorted({(r['selector'], r['budget_label']) for r in old_rows})
    budgets = [Budget(kind, float(label)) for kind, label in selections]
    device = next(task.model.parameters()).device
    on_device = [{name: value.float().to(device) for name, value in ps.items()} for ps in cell.scores]
    assert all(set(ps) == set(task.model.patch_masks) and all(ps[k].shape == task.model.patch_masks[k].shape for k in ps) for ps in on_device)
    rows = _functional_rows(cell, index_map, budgets, on_device,
                           [PatchType.TREE_PATCH, PatchType.EDGE_PATCH], evaluation_tasks=tasks)
    key = lambda row: tuple(str(row[k]) for k in ('task','algo','discovery_ablation','condition','eval_ablation','measure','selector','budget_label','metric'))
    old_lookup = {key(r): r for r in old_rows}
    assert {key(r) for r in rows} == set(old_lookup)
    assert all(float(r['realised_mean']) == float(old_lookup[key(r)]['realised_mean']) for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({**complete_id, 'complete': True, 'rows': rows}, indent=2))
    os.replace(temporary, path)
    print('Completed', spec, len(rows))


def merge(protocol_path, inputs_path, output, task_shorts):
    started_at = datetime.now().isoformat()
    from .evaluation_protocol import sha256
    from .run_cell import _TASK_TEMPLATES
    from .experiment_utils import git_state
    from .stability_plots import _read
    from .stability_analysis import FUNCTIONAL_FIELDS, STRUCTURAL_FIELDS, _summary_rows, _write_csv, _write_summary_md, _effective_condition
    protocol = load_protocol(protocol_path)
    load_frozen_inputs(inputs_path)
    partials = {}
    for index, spec in enumerate(repair_specs(task_shorts)):
        item = json.loads((output / 'cells' / f'{index:03d}.json').read_text())
        assert item['complete'] and item['spec'] == spec
        assert item['evaluation_protocol_id'] == protocol['protocol_id']
        assert item['selected_inputs_sha256'] == sha256(inputs_path)
        assert item['historical_functional_sha256'] == sha256(HISTORICAL / 'functional.csv')
        key = (spec['task'], spec['algo'], spec['ablation'], _effective_condition(spec['algo'], spec['condition']))
        partials[key] = item['rows']
    old_manifest = json.loads((HISTORICAL / 'manifest.json').read_text())
    specs = [c for c in old_manifest['cells'] if c['task'] in task_shorts]
    functional = []
    for spec in specs:
        key = (spec['task'], spec['algo'], spec['ablation'], _effective_condition(spec['algo'], spec['condition']))
        functional.extend(dict(r, condition=spec['condition']) for r in partials[key])
    names = {_TASK_TEMPLATES[k].name for k in task_shorts}
    structural = [r for r in _rows(HISTORICAL / 'structural.csv') if r['task'] in names]
    task_keys = {_TASK_TEMPLATES[k].name: _TASK_TEMPLATES[k].key for k in task_shorts}
    numeric_structural = [r for r in _read(HISTORICAL / 'structural.csv') if r['task'] in names]
    summary = _summary_rows(numeric_structural, functional, task_keys)
    manifest = {**old_manifest, 'experiment': output.name, 'git': git_state(), 'started_at': started_at,
                'historical_analysis_started_at': old_manifest.get('started_at'), 'tasks': task_shorts,
                'stability_metric': {k: v for k, v in old_manifest.get('stability_metric', {}).items() if k in task_keys.values()},
                'complete': False, 'cells': specs, 'functional_rows': len(functional),
                'structural_rows': len(structural), 'temporary_historical_repair': True,
                'evaluation_protocol_id': protocol['protocol_id'], 'evaluation_manifest': str(protocol_path),
                'selected_inputs': str(inputs_path), 'selected_inputs_sha256': sha256(inputs_path), 'eval_data': 'Temporary unseen-clean evaluation, 256 prompts per main task',
                'structural_source_sha256': sha256(HISTORICAL / 'structural.csv'),
                'historical_manifest_sha256': sha256(HISTORICAL / 'manifest.json')}
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2))
    _atomic_output(output / 'manifest.json', write_json, manifest)
    _atomic_output(output / 'structural.csv', _write_csv, structural, STRUCTURAL_FIELDS)
    _atomic_output(output / 'functional.csv', _write_csv, functional, FUNCTIONAL_FIELDS)
    _atomic_output(output / 'summary.csv', _write_csv, summary, list(summary[0]))
    _atomic_output(output / 'summary.md', _write_summary_md, summary)
    manifest.update(complete=True, completed_at=datetime.now().isoformat())
    _atomic_output(output / 'manifest.json', write_json, manifest)
    print('Merged', len(functional), 'functional rows and', len(structural), 'unchanged structural rows')


def smoke(protocol_path: Path, inputs_path: Path, output: Path, task_shorts: list[str]) -> None:
    import numpy as np
    import torch
    from auto_circuit.metrics.analysis_utils import Budget, edge_index
    from auto_circuit.types import PatchType
    from .stability_analysis import _functional_rows, _load_cell
    protocol = load_protocol(protocol_path)
    inputs = load_frozen_inputs(inputs_path)
    tasks = build_evaluation_tasks(protocol, task_shorts)
    rows, checks = [], {}
    for short in task_shorts:
        cell = _load_cell(str(ROOT / '.prune_scores/final/benchmark'), (short, 'ep', 'resample', 'joint'), [42], frozen_inputs=inputs)
        task = tasks[cell.task_key]
        verify_evaluation_task(task, protocol, short)
        assert len(task.train_loader) == 0 and len(task.test_loader) == 4
        model = task.model
        scores = {name: value.to(next(model.parameters()).device) for name, value in cell.scores[0].items()}
        assert set(scores) == set(model.patch_masks)
        assert all(scores[k].shape == model.patch_masks[k].shape for k in scores)
        index = edge_index(model.edges, scores)
        rows.extend(_functional_rows(cell, index,
            [Budget('top_k', task.true_edge_count), Budget('top_k', model.n_edges)],
            [scores], [PatchType.TREE_PATCH], evaluation_tasks=tasks))
        full = [r for r in rows if r['task'] == task.name and r['metric'] == 'Clean KL Div' and r['realised_mean'] == model.n_edges]
        assert len(full) == 3 and max(r['metric_mean'] for r in full) < 1e-5, full
        max_error = 0.0
        for batch in task.test_loader:
            with torch.inference_mode():
                cached = model(batch.clean)[model.out_slice]
                if task.test_loader.diverge_idx:
                    prefix = torch.tensor(model.tokenizer.encode(model.tokenizer.bos_token + 'The'), device=batch.clean.device)
                    assert len(prefix) == task.test_loader.diverge_idx
                    raw = torch.cat([prefix.expand(len(batch.clean), -1), batch.clean], dim=1)
                    uncached = model.wrapped_model(raw)[model.out_slice]
                else:
                    uncached = model.wrapped_model(batch.clean)[model.out_slice]
                max_error = max(max_error, float((cached - uncached).abs().max()))
        assert max_error < 1e-3
        checks[short] = {'graph_edges': model.n_edges, 'prefix_length': task.test_loader.diverge_idx,
                         'cached_uncached_max_logit_error': max_error,
                         'full_model_clean_kl_max': max(r['metric_mean'] for r in full)}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'smoke.json').write_text(json.dumps({'evaluation_protocol_id': protocol['protocol_id'], 'checks': checks, 'rows': rows}, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-manifest', type=Path, default=ROOT / 'datasets/evaluation/temporary_repair_20261006/manifest.json')
    parser.add_argument('--selected-inputs', type=Path, default=ROOT / 'results/results_chapter/evaluation_overlap_integration_20261006/selected_inputs.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/results_chapter/temporary_repair_20261006')
    parser.add_argument('--tasks', nargs='+', default=['ioi', 'gt'])
    parser.add_argument('--cell-index', type=int)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(repair_specs(args.tasks), indent=2))
    elif args.merge:
        merge(args.eval_manifest, args.selected_inputs, args.output_dir, args.tasks)
    elif args.cell_index is not None:
        run_cell(args.cell_index, args.eval_manifest, args.selected_inputs, args.output_dir, args.tasks)
    else:
        smoke(args.eval_manifest, args.selected_inputs, args.output_dir, args.tasks)
