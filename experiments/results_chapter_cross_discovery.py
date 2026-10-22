"""Source-backed same-seed circuit overlaps for Results Section 4.3.2.

Run on a CPU compute node with python -m experiments.results_chapter_cross_discovery.
"""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from auto_circuit.metrics.analysis_utils import Budget, budget_mask, jaccard
from experiments.experiment_utils import git_state
from .config import ROOT, load_registry
from .results_chapter_sources import EXPORT_ROOT, bundle, frozen_manifest_index, provenance_path

TASKS = {'ioi': ('IOI', 'Indirect Object Identification', 739),
         'gt': ('Greater-Than', 'Greaterthan', 235)}
METHODS = {'acdc': 'ACDC', 'eap': 'EAP', 'ep': 'EP', 'discogp-ep': 'DiscoGP'}
ABLATIONS = ('zero', 'resample', 'mean')
SEEDS = (42, 43, 44, 45, 46)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_scores(scores, keys, shapes):
    if set(scores) != set(keys):
        raise ValueError('Score module names differ')
    values = []
    for key in keys:
        if tuple(scores[key].shape) != shapes[key]:
            raise ValueError(f'Score tensor shape differs for {key}')
        values.append(scores[key].detach().abs().reshape(-1).to(torch.float64).cpu().numpy())
    result = np.concatenate(values)
    if np.isnan(result).any():
        raise ValueError('NaN score')
    return result


def select_mask(scores, selector, value):
    vector = np.abs(np.asarray(scores, dtype=np.float64))
    return budget_mask(vector, Budget(selector, float(value)))


def paired_rows(first, second):
    if set(first) != set(second):
        raise ValueError('Dataset/PyTorch seed identities differ')
    rows = []
    for data_seed, init_seed in sorted(first):
        a, b = first[data_seed, init_seed], second[data_seed, init_seed]
        if a.shape != b.shape:
            raise ValueError('Circuit shapes differ')
        intersection, union = int((a & b).sum()), int((a | b).sum())
        rows.append({'data_seed': data_seed, 'init_seed': init_seed,
                     'size_a': int(a.sum()), 'size_b': int(b.sum()),
                     'intersection': intersection, 'union': union,
                     'jaccard': intersection / union if union else 1.0})
    return rows


def table_source(rows):
    lookup = {(r['task'], r['algo'], r['ablation_a'], r['ablation_b']): r for r in rows}
    lines = [r'\begin{table}[!ht]', r'\centering', r'\small',
             r'\caption{Edge-set agreement across discovery ablations at the requested comparison sizes of 739 edges for the IOI task and 235 edges for the Greater-Than task. Entries summarise five comparisons pairing identical dataset and PyTorch seeds (42--46). Ranges are descriptive. ACDC is approximately size matched using its selected threshold for each ablation; attained sizes are reported in \Cref{tab:within-ablation-stability}.}',
             r'\label{tab:cross-discovery-stability}', r'\setlength{\tabcolsep}{5pt}',
             r'\renewcommand{\arraystretch}{1.10}', r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
             r'& \multicolumn{3}{c}{Jaccard: average [minimum, maximum]} \\',
             r'\cmidrule(lr){2-4}', r'Method & Zero--Resample & Zero--Mean & Resample--Mean \\', r'\midrule']
    for task in ('IOI', 'Greater-Than'):
        lines.append(r'\multicolumn{4}{@{}l}{\textbf{' + task + r'}} \\')
        for method in METHODS.values():
            cells = []
            for a, b in combinations(ABLATIONS, 2):
                row = lookup[task, method, a, b]
                cells.append(f"${row['jaccard_average']:.3f}\\,[{row['jaccard_min']:.3f},{row['jaccard_max']:.3f}]$")
            label = 'Edge Pruning' if method == 'EP' else method
            lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        if task == 'IOI':
            lines.append(r'\midrule')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\end{table}']) + '\n'


def main(source_dir=None, output_root=None, inputs=None):
    registry = load_registry()
    source = Path(source_dir) if source_dir is not None else Path(bundle(registry=registry)['source_dir'])
    output = Path(output_root) if output_root is not None else EXPORT_ROOT / 'main'
    output = (ROOT / output).resolve() / 'stability'
    source = (ROOT / source).resolve()
    hashes = {name: digest(source / name) for name in ('manifest.json', 'structural.csv', 'summary.csv', 'functional.csv')}
    output.mkdir(parents=True, exist_ok=True)
    # An export-only invocation can use the immutable selected threshold table.
    selected_root = output if (output / 'within_ablation_selected.csv').exists() else source / 'exhibits/stability'
    selected_path = selected_root / 'within_ablation_selected.csv'
    selected = read_csv(selected_path)
    previous = json.loads((selected_root / 'exhibit_manifest.json').read_text())
    assert digest(selected_path) == previous['output_sha256'][selected_path.name]
    lookup = {(r['task_label'], r['algo'], r['discovery_ablation']): r for r in selected}
    records = frozen_manifest_index(registry['benchmark']['inputs'] if inputs is None else inputs,
                                    TASKS, METHODS, ABLATIONS, SEEDS, registry=registry)
    assert len(records) == 120
    groups, sources, selected_runs, schemas = {}, [], [], {}
    for (task, method, ablation, seed), (manifest, scores_path, path) in sorted(records.items()):
        payload = torch.load(scores_path, map_location='cpu', weights_only=True)
        scores = next(iter(next(iter(payload.values())).values()))
        if task not in schemas:
            schemas[task] = tuple(sorted(scores)), {k: tuple(v.shape) for k, v in scores.items()}
        keys, shapes = schemas[task]
        vector = flatten_scores(scores, keys, shapes)
        assert vector.size == 32491
        label, _, target = TASKS[task]
        row = lookup[label, METHODS[method], ablation]
        mask = select_mask(vector, row['selector'], float(row['budget_value']))
        groups.setdefault((label, METHODS[method], ablation), {})[seed, seed] = mask
        source = {'task': label, 'algo': METHODS[method], 'discovery_ablation': ablation,
                  'data_seed': seed, 'init_seed': seed, 'saved_at': manifest['saved_at'],
                  'manifest': provenance_path(path), 'manifest_sha256': digest(path),
                  'scores': provenance_path(scores_path), 'scores_sha256': digest(scores_path),
                  'training_git': manifest.get('git', {})}
        sources.append(source)
        selected_runs.append({k: source[k] for k in ('task', 'algo', 'discovery_ablation', 'data_seed', 'init_seed', 'scores')}
                             | {'target': target, 'selector': row['selector'], 'budget_value': float(row['budget_value']),
                                'realised_size': int(mask.sum())})
    for key, masks in groups.items():
        assert sorted(masks) == [(s, s) for s in SEEDS]
        expected = lookup[key]
        sizes = [int(mask.sum()) for mask in masks.values()]
        assert [np.average(sizes), min(sizes), max(sizes)] == [float(expected[k]) for k in ('realised_mean', 'realised_min', 'realised_max')]
        overlaps = [int((a & b).sum()) / int((a | b).sum()) for a, b in combinations(masks.values(), 2)]
        for value, field in zip((np.average(overlaps), np.std(overlaps), min(overlaps), max(overlaps)),
                                ('jaccard_mean', 'jaccard_sd', 'jaccard_min', 'jaccard_max')):
            assert abs(value - float(expected[field])) < 1e-12, (key, field)
    pairwise, aggregates = [], []
    for label, _, target in TASKS.values():
        for method in METHODS.values():
            for a, b in combinations(ABLATIONS, 2):
                first, second = groups[label, method, a], groups[label, method, b]
                rows = paired_rows(first, second)
                assert len(rows) == 5
                for row in rows:
                    identity = row['data_seed'], row['init_seed']
                    expected = jaccard(set(np.flatnonzero(first[identity])), set(np.flatnonzero(second[identity])))
                    assert abs(row['jaccard'] - expected) < 1e-15
                    pairwise.append({'task': label, 'algo': method, 'ablation_a': a, 'ablation_b': b,
                                     'target': target, **row,
                                     'selector_a': lookup[label, method, a]['selector'],
                                     'budget_a': float(lookup[label, method, a]['budget_value']),
                                     'selector_b': lookup[label, method, b]['selector'],
                                     'budget_b': float(lookup[label, method, b]['budget_value'])})
                values = [r['jaccard'] for r in rows]
                aggregates.append({'task': label, 'algo': method, 'ablation_a': a, 'ablation_b': b,
                                   'target': target, 'n_pairs': 5, 'jaccard_average': float(np.average(values)),
                                   'jaccard_min': min(values), 'jaccard_max': max(values),
                                   'size_a_average': float(np.average([r['size_a'] for r in rows])),
                                   'size_b_average': float(np.average([r['size_b'] for r in rows]))})
    assert len(pairwise) == 120 and len(aggregates) == 24
    write_csv(output / 'cross_discovery_selected_runs.csv', selected_runs)
    write_csv(output / 'cross_discovery_pairwise.csv', pairwise)
    write_csv(output / 'cross_discovery_summary.csv', aggregates)
    (output / 'cross_discovery_table.tex').write_text(table_source(aggregates))
    outputs = ('cross_discovery_selected_runs.csv', 'cross_discovery_pairwise.csv',
               'cross_discovery_summary.csv', 'cross_discovery_table.tex')
    manifest = {'section': 'Results 4.3.2', 'generator': str(Path(__file__).relative_to(ROOT)),
                'generator_sha256': digest(Path(__file__)), 'git': git_state(), 'source_sha256': hashes,
                'selected_rows_sha256': digest(selected_path), 'cache_sources': sources,
                'seeds': list(SEEDS), 'selection': 'absolute magnitude >= float64 threshold; retain all ties',
                'pairing': 'equal dataset and PyTorch seeds, joint condition only',
                'edge_alignment': {t: {'module_order': list(k), 'tensor_shapes': s} for t, (k, s) in schemas.items()},
                'within_ablation_cells_reproduced': 24, 'pairwise_rows': 120, 'aggregate_rows': 24,
                'output_sha256': {name: digest(output / name) for name in outputs}}
    (output / 'cross_discovery_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'selected_runs': len(selected_runs), 'within_cells_reproduced': 24,
                      'pairwise_rows': len(pairwise), 'summary_rows': len(aggregates)}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--inputs', type=Path, help='Exact selected-input JSON inventory')
    args = parser.parse_args()
    frozen = json.loads(args.inputs.read_text())['inputs'] if args.inputs else None
    main(args.source_dir, args.output_root, frozen)
