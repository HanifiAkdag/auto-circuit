"""Frozen-circuit paired stability analysis; run only on a CPU compute node."""
import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from .config import ROOT, load_registry, resolve_path
from .results_chapter_sources import EXPORT_ROOT, provenance_path

SEEDS = [42, 43, 44, 45, 46]
METHODS = {'acdc': 'ACDC', 'eap': 'EAP', 'ep': 'EP', 'discogp-ep': 'DiscoGP'}
TASKS = {'ioi': ('IOI', 'Indirect Object Identification', 'mean', 739),
         'gt': ('Greater-Than', 'Greaterthan', 'resample', 235)}
KL = 'Clean KL Div'
ACCURACY = 'Correct Answer Greater Than Incorrect Percent'


def paired_records(masks, values, seeds):
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or len(masks) != len(values) or len(values) != len(seeds):
        raise ValueError('Circuit and score identities do not align')
    if len(set(seeds)) != len(seeds) or not np.isfinite(values).all():
        raise ValueError('Duplicate seeds or nonfinite scores')
    rows = []
    for i, j in combinations(range(len(seeds)), 2):
        intersection = int((masks[i] & masks[j]).sum())
        union = int((masks[i] | masks[j]).sum())
        jac = intersection / union if union else 1.
        rows.append(dict(seed_i=int(seeds[i]), seed_j=int(seeds[j]),
                         size_i=int(masks[i].sum()), size_j=int(masks[j].sum()),
                         size_difference=abs(int(masks[i].sum())-int(masks[j].sum())),
                         intersection=intersection, union=union, jaccard=jac,
                         dissimilarity=1-jac, value_i=float(values[i]),
                         value_j=float(values[j]), gap=abs(float(values[i])-float(values[j]))))
    return rows


def correlation(pairs):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            for i in order[start:end]:
                result[i] = (start + 1 + end) / 2
            start = end
        return np.asarray(result)
    if len(pairs) < 3:
        return None
    x, y = (ranks([p[key] for p in pairs]) for key in ('dissimilarity', 'gap'))
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def omission_records(pairs, seeds):
    rows = []
    for seed in seeds:
        remaining = [p for p in pairs if seed not in (p['seed_i'], p['seed_j'])]
        rows.append(dict(omitted_seed=seed, n_pairs=len(remaining), rho=correlation(remaining)))
    return rows


def rounding_error(value):
    return 0. if value == 0 else 0.5 * 10 ** (math.floor(math.log10(abs(value))) - 5)


def accuracy_values(serialized, count=256):
    values = [float(x) for x in serialized.split(';')]
    result = [round(x * count / 100) * 100 / count for x in values]
    if any(not 0 <= x <= 100 or abs(x-y) > rounding_error(x)+1e-12
           for x, y in zip(values, result)):
        raise ValueError('Saved accuracy is incompatible with the verified prompt count')
    return result


def precision_audit(values, errors):
    intervals = []
    for i, j in combinations(range(len(values)), 2):
        gap, error = abs(values[i]-values[j]), errors[i]+errors[j]
        intervals.append((gap, max(0, gap-error), gap+error, error))
    ambiguous = 0
    for a, b in combinations(intervals, 2):
        if a[0] == b[0] and a[3] == b[3] == 0:
            continue
        if not (a[2] < b[1] or b[2] < a[1]):
            ambiguous += 1
    return dict(ambiguous_orders=ambiguous,
                average_gap_error_bound=float(np.mean([r[3] for r in intervals])),
                max_gap_error_bound=max(r[3] for r in intervals))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def one(rows):
    if len(rows) != 1:
        raise ValueError(f'Expected one source row, got {len(rows)}')
    return rows[0]


def validate_scores(vector, method):
    if np.isnan(vector).any() or (method != 'ACDC' and np.isinf(vector).any()):
        raise ValueError('Invalid scores; infinity is ACDC\'s retained-edge sentinel')


def load_selected_inputs(root, registry, final_registry=None):
    path = resolve_path(registry['selected_inputs'], final_registry, root=root)
    return path, json.loads(path.read_text())['inputs']


def run(root, output, registry=None, source_dir=None):
    import torch
    from experiments.results_chapter_cross_discovery import flatten_scores, select_mask
    from experiments.stability_plots import _spearman
    root, output = Path(root).resolve(), Path(output).resolve()
    final_registry = registry or load_registry()
    bundle = final_registry['benchmark']['bundles']['main']
    source = ((root / source_dir).resolve() if source_dir is not None else
              resolve_path(bundle['source_dir'], final_registry, root=root))
    source_meta = json.loads((source/'manifest.json').read_text())
    selected_bundle = (dict(bundle, selected_inputs=source_meta['selected_inputs'])
                       if source_dir is not None and 'selected_inputs' in source_meta else bundle)
    inputs_path, frozen = load_selected_inputs(root, selected_bundle, final_registry)
    paths = [source/'functional.csv', source/'structural.csv', source/'manifest.json', inputs_path,
             source/'exhibits/stability/within_ablation_selected.csv',
             source/'exhibits/controlled_comparison/matched_quality.csv']
    from .evaluation_protocol import require_same_protocol
    exhibit_manifest = paths[4].with_name('exhibit_manifest.json')
    quality_manifest = paths[5].with_name('matched_quality_manifest.json')
    exhibit_meta = json.loads(exhibit_manifest.read_text())
    require_same_protocol(source_meta, exhibit_meta)
    require_same_protocol(source_meta, json.loads(quality_manifest.read_text()))
    for filename, expected in exhibit_meta['source_sha256'].items():
        assert digest(source / filename) == expected
    assert digest(paths[4]) == exhibit_meta['output_sha256'][paths[4].name]
    paths.extend([exhibit_manifest, quality_manifest])
    for name in ('functional', 'structural', 'manifest'):
        path = source/(name + ('.json' if name == 'manifest' else '.csv'))
        if source_dir is None:
            assert digest(path) == bundle[name+'_sha256']
        elif name != 'manifest' and name+'_sha256' in source_meta:
            assert digest(path) == source_meta[name+'_sha256']
    if source_dir is not None and 'selected_inputs_sha256' in source_meta:
        assert digest(inputs_path) == source_meta['selected_inputs_sha256']
    assert source_meta['seeds'] == SEEDS and source_meta['complete']
    structural, functional, quality = (read_csv(p) for p in (paths[4], paths[0], paths[5]))
    runs, pairs, summaries, omissions, inputs, schemas = [], [], [], [], [], {}
    for task, (label, task_name, ablation, target) in TASKS.items():
        for slug, method in METHODS.items():
            selected = one([r for r in structural if (r['task'], r['algo'], r['discovery_ablation'], r['condition'])
                            == (task_name, method, ablation, 'joint')])
            assert int(selected['target']) == target and int(selected['n_seeds']) == 5
            masks = []
            identities = []
            for seed in SEEDS:
                item = one([r for r in frozen if (r['cell']['task'], r['cell']['algo'], r['cell']['ablation'], r['cell']['seed'])
                            == (task, slug, ablation, seed) and r['cell'].get('init_seed', seed) == seed
                            and not r['cell'].get('null_test')])
                for key in ('scores', 'manifest'):
                    assert digest(resolve_path(item[key], final_registry, root=root)) == item[key+'_sha256']
                manifest = json.loads(resolve_path(item['manifest'], final_registry, root=root).read_text())
                assert manifest['cell'] == item['cell']
                payload = torch.load(resolve_path(item['scores'], final_registry, root=root), map_location='cpu', weights_only=True)
                assert len(payload) == 1 and len(next(iter(payload.values()))) == 1
                scores = next(iter(next(iter(payload.values())).values()))
                if task not in schemas:
                    schemas[task] = (tuple(sorted(scores)), {k: tuple(v.shape) for k, v in scores.items()})
                keys, shapes = schemas[task]
                vector = flatten_scores(scores, keys, shapes)
                assert len(vector) == 32491
                validate_scores(vector, method)
                masks.append(select_mask(vector, selected['selector'], float(selected['budget_value'])))
                identities.append(item)
                inputs.append(item)
            masks = np.asarray(masks)
            sizes = masks.sum(1)
            assert [float(np.mean(sizes)), int(min(sizes)), int(max(sizes))] == [float(selected[k]) for k in ('realised_mean','realised_min','realised_max')]
            structural_pairs = paired_records(masks, [0]*5, SEEDS)
            jac = [p['jaccard'] for p in structural_pairs]
            for actual, field in zip([np.mean(jac), np.std(jac), min(jac), max(jac)],
                                     ['jaccard_mean','jaccard_sd','jaccard_min','jaccard_max']):
                assert abs(actual-float(selected[field])) < 1e-12, (task, method, field)
            for metric in ([KL, ACCURACY] if task == 'ioi' else [KL]):
                row = one([r for r in functional if (r['task'], r['algo'], r['discovery_ablation'], r['condition'], r['eval_ablation'], r['measure'], r['metric'], r['selector'])
                           == (task_name, method, ablation, 'joint', ablation, 'faithfulness', metric, selected['selector'])
                           and float(r['budget_label']) == float(selected['budget_value'])])
                q = one([r for r in quality if (r['task'], r['algo'], r['ablation'], r['metric']) == (task_name, method, ablation, metric)])
                for key in ('metric_mean','metric_sd','metric_min','metric_max','selector','budget_label','realised_mean'):
                    assert row[key] == q[key]
                assert int(row['n_seeds']) == 5
                raw = [float(x) for x in row['metric_values'].split(';')]
                assert len(raw) == 5
                values = accuracy_values(row['metric_values']) if metric == ACCURACY else raw
                errors = [0.]*5 if metric == ACCURACY else [rounding_error(v) for v in values]
                for actual, key in zip([np.mean(values), np.std(values), min(values), max(values)], ['metric_mean','metric_sd','metric_min','metric_max']):
                    assert abs(actual-float(row[key])) <= max(errors)+1e-12
                context = dict(task=label, algo=method, ablation=ablation, metric=metric, target=target)
                current = paired_records(masks, values, SEEDS)
                rho = correlation(current)
                assert rho == _spearman([p['dissimilarity'] for p in current], [p['gap'] for p in current]) or (rho is not None and abs(rho-_spearman([p['dissimilarity'] for p in current], [p['gap'] for p in current])) < 1e-12)
                reverse = paired_records(masks[::-1], values[::-1], SEEDS[::-1])
                assert rho == correlation(reverse) or (rho is not None and abs(rho-correlation(reverse)) < 1e-12)
                omitted = omission_records(current, SEEDS)
                assert all(r['n_pairs'] == 6 for r in omitted)
                defined = [r['rho'] for r in omitted if r['rho'] is not None]
                precision = precision_audit(values, errors)
                summaries.append(dict(**context, n_runs=5, n_pairs=10, jaccard_average=float(np.mean(jac)),
                                      size_average=float(np.mean(sizes)), size_min=int(min(sizes)), size_max=int(max(sizes)),
                                      score_average=float(row['metric_mean']), score_population_sd=float(row['metric_sd']),
                                      score_min=float(row['metric_min']), score_max=float(row['metric_max']),
                                      gap_average=float(np.mean([p['gap'] for p in current])), gap_min=min(p['gap'] for p in current),
                                      gap_max=max(p['gap'] for p in current), rho=rho,
                                      omission_min=min(defined) if defined else None, omission_max=max(defined) if defined else None,
                                      omission_undefined=5-len(defined), **precision))
                pairs.extend(dict(**context, **p) for p in current)
                omissions.extend(dict(**context, **p) for p in omitted)
                for i, seed in enumerate(SEEDS):
                    runs.append(dict(**context, seed=seed, selector=selected['selector'], budget_value=selected['budget_value'],
                                     attained_size=int(sizes[i]), score=values[i], saved_score=raw[i], rounding_error_bound=errors[i],
                                     scores_file=identities[i]['scores'], manifest_file=identities[i]['manifest']))
    assert len(inputs) == 40 and len(pairs) == 120 and len(summaries) == 12 and len(omissions) == 60
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('runs',runs),('pairs',pairs),('summary',summaries),('omissions',omissions)]:
        write_csv(output/(name+'.csv'), rows)
    manifest = dict(evaluation_protocol_id=source_meta['evaluation_protocol_id'], generator_sha256=digest(Path(__file__)),
                    source_hashes={provenance_path(p, root):digest(p) for p in paths}, frozen_inputs=inputs,
                    alignment={t:{'module_order':k,'tensor_shapes':s} for t,(k,s) in schemas.items()},
                    verified_structural_cells=8, n_circuits=40, n_metric_pairs=120, n_summaries=12, n_omissions=60,
                    accuracy_precision='Recovered exact 256-prompt counts; native aggregate fields reproduced',
                    kl_precision='Six significant digits; conservative pair-gap rank interval audit',
                    outputs={p.name:digest(p) for p in output.glob('*.csv')})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for r in summaries:
        print(r['task'], r['algo'], 'KL' if r['metric']==KL else 'Accuracy', 'rho',r['rho'],
              'omission',r['omission_min'],r['omission_max'],'rank ambiguities',r['ambiguous_orders'])


def run_points(pairs):
    runs, seen = {}, set()
    for pair in pairs:
        seeds = [int(pair['seed_i']), int(pair['seed_j'])]
        identity = tuple(sorted(seeds))
        if seeds[0] == seeds[1] or identity in seen:
            raise ValueError('Duplicate or self comparison')
        seen.add(identity)
        for seed, key in zip(seeds, ('value_i', 'value_j')):
            score = float(pair[key])
            current = runs.setdefault(seed, dict(score=score, distances=[]))
            if current['score'] != score:
                raise ValueError('Inconsistent score for a run')
            current['distances'].append(float(pair['dissimilarity']))
    if len(runs) < 2 or any(len(r['distances']) != len(runs)-1 for r in runs.values()):
        raise ValueError('Incomplete run comparisons')
    return [dict(seed=seed, score=runs[seed]['score'],
                 average_dissimilarity=float(np.mean(runs[seed]['distances'])))
            for seed in sorted(runs)]


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.results_chapter_size_curves import STYLE
    pairs = read_csv(output/'pairs.csv')
    plotted = []
    for task, metric, slug in [('IOI',KL,'ioi_kl'),('Greater-Than',KL,'gt_kl'),('IOI',ACCURACY,'ioi_accuracy')]:
        fig, axes = plt.subplots(2,2, figsize=(7.1,5.2), sharex=True, sharey=True)
        subset = [r for r in pairs if r['task']==task and r['metric']==metric]
        ymax = max(float(r[key]) for r in subset for key in ('value_i', 'value_j'))
        for ax, method in zip(axes.flat, METHODS.values()):
            rows = [r for r in subset if r['algo']==method]
            points = run_points(rows)
            assert [p['seed'] for p in points] == SEEDS
            color, name = STYLE[method]
            ax.scatter([p['average_dissimilarity'] for p in points], [p['score'] for p in points], s=45, c=color, edgecolors='white', linewidths=.6)
            for point in points:
                plotted.append(dict(task=task, algo=method, metric=metric, color=color, **point))
            ax.set_title(name, fontsize=10)
            ax.set_xlim(0,1)
            ax.set_ylim(-.02*max(ymax,.1), max(ymax,.1)*1.12)
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
        fig.supxlabel('Average structural dissimilarity to the other four runs (1 − Jaccard)', fontsize=10)
        fig.supylabel('KL divergence (nats)' if metric==KL else 'Correct-over-incorrect accuracy (%)', fontsize=10)
        fig.suptitle(task, fontsize=12, fontweight='bold')
        fig.tight_layout(rect=(.02,.02,1,1))
        for extension in ('pdf','png'):
            fig.savefig(output/(slug+'.'+extension), dpi=180, bbox_inches='tight')
        plt.close(fig)
    write_csv(output/'plotted_runs.csv', plotted)
    print('Exported three figures and 60 actual-score points using the size-figure colors.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=EXPORT_ROOT / 'structure_function')
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if args.plots_only:
        plot(args.output_dir)
    else:
        run(args.root, args.output_dir, registry=load_registry(args.registry), source_dir=args.source_dir)
