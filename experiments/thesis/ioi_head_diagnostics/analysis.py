"""Paired effects, crossed uncertainty, and standalone research figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .protocol import atomic_json, file_hash, load_inventory, normalized_margin, validate_reference
from .evaluation import METRICS

NAMES = {'acdc': 'ACDC', 'discogp-ep': 'DiscoGP', 'eap': 'EAP', 'ep': 'Edge Pruning'}
COLORS = {'acdc': '#009E73', 'discogp-ep': '#D55E00', 'eap': '#8A45BA', 'ep': '#0072B2'}


def crossed_interval(data, draws=10000, denominator=None):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('Crossed bootstrap requires finite run-by-prompt data')
    if denominator is not None:
        denominator = np.asarray(denominator, dtype=np.float64)
        if denominator.shape != (data.shape[1],) or not np.isfinite(denominator).all():
            raise ValueError('Invalid paired full-model denominator')
    gen = np.random.default_rng(20261011)
    seeds = gen.integers(0, len(data), size=(draws, len(data)))
    prompts = gen.integers(0, data.shape[1], size=(draws, data.shape[1]))
    values = []
    for start in range(0, draws, 256):
        s, p = seeds[start:start+256], prompts[start:start+256]
        value = data[s[:, :, None], p[:, None, :]].mean(axis=(1, 2))
        if denominator is not None:
            divisor = denominator[p].mean(axis=1)
            if np.any(divisor <= 0):
                raise ValueError('Nonpositive resampled full-model margin')
            value = 100 * value / divisor
        values.append(value)
    low, high = np.quantile(np.concatenate(values), [.025, .975])
    return float(low), float(high)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arrays_for(output, result):
    with np.load(Path(output) / 'conditions' / result['array_file']) as values:
        return {k: values[k].copy() for k in values.files}


def collect_primary(conditions, arrays, results):
    primary = {}
    baselines = {(c['case'], c['budget']): arrays[c['id']] for c in conditions if c['phase'] == 'screen'}
    native_exact = {(c['case'], c['budget']) for c in conditions if c['phase'] == 'screen' and c.get('attained') == c['budget']}
    groups = {}
    for c in conditions:
        if c['phase'] == 'causal':
            groups.setdefault((c['case'], c['budget']), []).append(c)
    for (case, budget), members in groups.items():
        baseline = baselines[(case, budget)]
        cell = results[members[0]['id']]['cell']
        if cell['ablation'] != 'mean':
            raise ValueError('Non-mean data cannot enter the replacement study')
        algo, seed = cell['algo'], cell['seed']
        def save(label, key, delta):
            primary.setdefault((algo, budget, label, key), {})[seed] = delta
        measures = [key for key in METRICS if key in baseline]
        for c in members:
            if c['label'].startswith(('control_', 'removal_', 'fixed_')):
                continue
            a = arrays[c['id']]
            for key in measures:
                save(c['label'], key, a[key] - baseline[key])
            for h, tag in enumerate(('10_7', '11_10')):
                delta = ((a['attention'][:, h, 0] - a['attention'][:, h, 1]) -
                         (baseline['attention'][:, h, 0] - baseline['attention'][:, h, 1]))
                save(c['label'], 'attention_' + tag, delta)
            if (case, budget) in native_exact and c['label'] in ('restore_q_both', 'drop_nnm_out', 'restore_q_drop_out'):
                for key in measures:
                    save('exact_' + c['label'], key, a[key] - baseline[key])
                for h, tag in enumerate(('10_7', '11_10')):
                    delta = ((a['attention'][:, h, 0] - a['attention'][:, h, 1]) -
                             (baseline['attention'][:, h, 0] - baseline['attention'][:, h, 1]))
                    save('exact_' + c['label'], 'attention_' + tag, delta)
        averaged = {}
        for label, prefix in [('control_average', 'control_'), ('removal_average', 'removal_'), ('fixed_q_average', 'fixed_')]:
            selected = [arrays[c['id']] for c in members if c['label'].startswith(prefix)]
            if selected:
                averaged[label] = {key: np.stack([a[key] for a in selected]).mean(0) for key in measures}
                for key in measures:
                    save(label, key, averaged[label][key] - baseline[key])
        if 'fixed_q_average' in averaged and 'removal_average' in averaged:
            for key in measures:
                save('fixed_paired_q', key, averaged['fixed_q_average'][key] - averaged['removal_average'][key])
        indexed = {c['label']: arrays[c['id']] for c in members}
        if all(k in indexed for k in ('restore_q_both', 'drop_nnm_out', 'restore_q_drop_out')):
            interaction = ((baseline['margin'] - indexed['drop_nnm_out']['margin']) -
                           (indexed['restore_q_both']['margin'] - indexed['restore_q_drop_out']['margin']))
            save('output_effect_change_after_q', 'margin', interaction)
    exact_baselines = {(c['case'], c['budget'], c['realization']): arrays[c['id']]
                       for c in conditions if c['label'] == 'exact_baseline'}
    exact = {}
    for c in conditions:
        if c['phase'] != 'size-sensitivity' or c['label'] == 'exact_baseline':
            continue
        cell = results[c['id']]['cell']
        if cell['ablation'] != 'mean':
            raise ValueError('Non-mean tie results cannot enter the replacement study')
        a = arrays[c['id']]
        b = exact_baselines[(c['case'], c['budget'], c['realization'])]
        for key in METRICS:
            if key in b:
                identity = (cell['algo'], c['budget'], c['label'], key, cell['seed'])
                exact.setdefault(identity, []).append(a[key] - b[key])
        for h, tag in enumerate(('10_7', '11_10')):
            delta = ((a['attention'][:, h, 0] - a['attention'][:, h, 1]) -
                     (b['attention'][:, h, 0] - b['attention'][:, h, 1]))
            identity = (cell['algo'], c['budget'], c['label'], 'attention_' + tag, cell['seed'])
            exact.setdefault(identity, []).append(delta)
    for (algo, budget, label, key, seed), realizations in exact.items():
        primary.setdefault((algo, budget, label, key), {})[seed] = np.stack(realizations).mean(0)
    return primary


def exact_run_rows(rows):
    groups = {}
    native_exact = {(r['algo'], r['seed'], r['budget']) for r in rows
                    if r['phase'] == 'screen' and r['attained'] == r['budget']}
    for row in rows:
        identity = (row['algo'], row['seed'], row['budget'])
        if row['phase'] == 'size-sensitivity':
            groups.setdefault((*identity, row['label']), []).append(row)
        elif identity in native_exact and row['label'] in ('baseline', 'restore_q_both', 'drop_nnm_out', 'restore_q_drop_out'):
            groups.setdefault((*identity, 'exact_' + row['label']), []).append(row)
    out = []
    for (algo, seed, budget, label), members in sorted(groups.items()):
        attained = [r['attained'] for r in members]
        row = dict(algo=algo, seed=seed, budget=budget, label=label, n_realizations=len(members),
                   baseline_attained=budget, attained=float(np.mean(attained)),
                   attained_min=min(attained), attained_max=max(attained))
        keys = list(METRICS) + ['logit_difference_percent']
        keys += [k for k in members[0] if k.startswith(('attention_', 'io_minus_', 'dla_', 'contrast_dla_'))]
        for key in keys:
            row[key] = float(np.mean([r[key] for r in members]))
        out.append(row)
    return out


def summarize(output):
    output = Path(output)
    manifest, _ = load_inventory(output)
    validate_reference(output, manifest)
    results, arrays, rows = {}, {}, []
    for condition in manifest['conditions']:
        path = output / 'conditions' / f'{condition["id"]:05d}.json'
        if not path.exists():
            raise ValueError(f'Missing predefined condition {condition["id"]}')
        result = json.loads(path.read_text())
        if not result.get('complete') or result['study_id'] != manifest['study_id'] or result['condition'] != condition:
            raise ValueError('Incomplete or mismatched scientific result')
        if file_hash(output / 'conditions' / result['array_file']) != result['arrays_sha256']:
            raise ValueError('Saved measurement checksum mismatch')
        a = arrays_for(output, result)
        if len(a['margin']) != 256 or any(not np.isfinite(v).all() for v in a.values()):
            raise ValueError('Missing/nonfinite per-prompt measurements')
        results[condition['id']], arrays[condition['id']] = result, a
        row = dict(condition_id=condition['id'], algo=result['cell']['algo'], seed=result['cell']['seed'],
                   ablation=result['cell']['ablation'], phase=condition['phase'], budget=condition['budget'],
                   label=condition['label'], realization=condition['realization'], attained=condition['attained'],
                   added=len(condition['added']), removed=len(condition['removed']), **result['summary'])
        row.pop('io_minus_s1')
        row['accuracy'] *= 100
        for h, tag in enumerate(('10_7', '11_10')):
            row['retained_' + tag] = result['output_retained'][h]
            for j, name in enumerate(('io', 's1', 's2')):
                row[f'attention_{tag}_{name}'] = float(a['attention'][:, h, j].mean())
            row['io_minus_s1_' + tag] = float((a['attention'][:, h, 0] - a['attention'][:, h, 1]).mean())
            row['dla_margin_' + tag] = float((a['dla'][:, h, 0] - a['dla'][:, h, 1]).mean())
            row['dla_io_' + tag] = float(a['dla'][:, h, 0].mean())
            row['dla_subject_' + tag] = float(a['dla'][:, h, 1].mean())
            row['contrast_dla_margin_' + tag] = float((a['contrast_dla'][:, h, 0] - a['contrast_dla'][:, h, 1]).mean())
        rows.append(row)
    full = dict(np.load(output / 'full_reference.npz'))
    exhibits = output / 'exhibits'
    exhibits.mkdir(exist_ok=True)
    write_csv(output / 'per_run.csv', rows)
    write_csv(output / 'exact_per_run.csv', exact_run_rows(rows))
    base_lookup = {(c['case'], c['budget']): c['id'] for c in manifest['conditions'] if c['phase'] == 'screen'}
    exact_lookup = {(c['case'], c['budget'], c['realization']): c['id'] for c in manifest['conditions'] if c['label'] == 'exact_baseline'}
    effects = []
    for c in manifest['conditions']:
        if c['phase'] == 'screen':
            continue
        baseline_id = exact_lookup[(c['case'], c['budget'], c['realization'])] if c['phase'] == 'size-sensitivity' else base_lookup[(c['case'], c['budget'])]
        a, b = arrays[c['id']], arrays[baseline_id]
        r = results[c['id']]
        effect = dict(algo=r['cell']['algo'], seed=r['cell']['seed'], label=c['label'], phase=c['phase'],
                      realization=c['realization'], budget=c['budget'], attained=c['attained'],
                      added=len(c['added']), removed=len(c['removed']))
        for key in METRICS:
            effect['delta_' + key] = float((a[key] - b[key]).mean())
        effect['delta_logit_difference_percent'] = normalized_margin(a['margin'] - b['margin'], full['margin'])
        for h, tag in enumerate(('10_7', '11_10')):
            effect['delta_attention_' + tag] = float(((a['attention'][:, h, 0] - a['attention'][:, h, 1]) -
                                                     (b['attention'][:, h, 0] - b['attention'][:, h, 1])).mean())
        effects.append(effect)
    primary = collect_primary(manifest['conditions'], arrays, results)
    summary = []
    for (algo, budget, label, metric), values in sorted(primary.items()):
        data = np.stack([values[s] for s in sorted(values)])
        means = data.mean(1)
        low, high = crossed_interval(data)
        summary.append(dict(algo=algo, budget=budget, label=label, metric=metric, n_runs=len(values),
                            average=float(means.mean()), sample_sd=float(means.std(ddof=1)) if len(values)>1 else 0.,
                            ci_low=low, ci_high=high, positive_runs=int((means>0).sum()),
                            negative_runs=int((means<0).sum()), values=';'.join(map(str, means))))
        if metric == 'margin':
            low, high = crossed_interval(data, denominator=full['margin'])
            percent = 100 * means / full['margin'].mean()
            summary.append(dict(algo=algo, budget=budget, label=label, metric='logit_difference_percent_points', n_runs=len(values),
                                average=float(percent.mean()), sample_sd=float(percent.std(ddof=1)) if len(values)>1 else 0.,
                                ci_low=low, ci_high=high, positive_runs=int((means>0).sum()), negative_runs=int((means<0).sum()),
                                values=';'.join(map(str, percent))))
    write_csv(output / 'paired_effects.csv', effects)
    write_csv(output / 'effect_summary.csv', summary)
    plots(exhibits, rows, effects, full, manifest['budgets'])
    report(output, manifest, rows, summary, full)
    atomic_json(output / 'complete.json', dict(study_id=manifest['study_id'], complete=True,
                predefined_conditions=len(manifest['conditions']), verified_conditions=len(results),
                source_unchanged=True, full_model_margin=float(full['margin'].mean()),
                n_prompts=256, n_discovery_runs=5, ablation='mean', budgets=manifest['budgets']))
    print('Complete study verified:', len(results), 'conditions', flush=True)


def plots(output, rows, effects, full, budgets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / (name + '.pdf'))
        fig.savefig(output / (name + '.png'), dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for h, tag in enumerate(('10_7', '11_10')):
        ax = axes[h]
        ax.axhline(float((full['attention'][:, h, 0] - full['attention'][:, h, 1]).mean()), color='black', linestyle='--', label='Full model')
        for algo, color in COLORS.items():
            group = [r for r in rows if r['phase'] == 'screen' and r['algo'] == algo]
            for seed in range(42, 47):
                points = sorted([r for r in group if r['seed'] == seed], key=lambda r: r['budget'])
                ax.plot([r['budget'] for r in points], [r['io_minus_s1_' + tag] for r in points], color=color, alpha=.3, marker='o', markersize=3)
            ax.plot(budgets, [np.mean([r['io_minus_s1_' + tag] for r in group if r['budget'] == k]) for k in budgets], color=color, label=NAMES[algo], linewidth=2)
        ax.axhline(0, color='grey', linewidth=.5)
        ax.set_title(f'Mean ablation: head {tag.replace("_", ".")}')
        ax.set_ylabel('Attention to IO − attention to S1')
        ax.set_xlabel('Requested edges (native thresholds)')
    axes[0].legend(fontsize=8)
    save(fig, 'recurrence')
    for budget in budgets:
        fig, axes = plt.subplots(4, 4, figsize=(15, 12))
        labels = ('baseline', 'restore_q_both', 'drop_nnm_out')
        for i, algo in enumerate(NAMES):
            group = [r for r in rows if r['algo'] == algo and r['budget'] == budget and r['phase'] != 'size-sensitivity']
            for j, key in enumerate(('io_minus_s1_10_7', 'io_minus_s1_11_10', 'margin', 'kl')):
                ax = axes[i, j]
                for seed in range(42, 47):
                    points = [next(r for r in group if r['seed'] == seed and r['label'] == label) for label in labels]
                    ax.plot(range(3), [r[key] for r in points], marker='o', color=COLORS[algo], alpha=.65)
                ax.set_xticks(range(3), ['Circuit', 'Restore Q', 'Mean outputs'], rotation=15)
                ax.set_title(f'{NAMES[algo]}: ' + key.replace('_', ' '))
                if j < 2:
                    ax.axhline(0, color='grey', linewidth=.5)
        fig.suptitle(f'Mean ablation, requested {budget:,} edges; separate paired interventions')
        save(fig, f'causal_comparison_{budget}')
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for i, algo in enumerate(NAMES):
            points = [r for r in rows if r['algo'] == algo and r['budget'] == budget and r['phase'] != 'size-sensitivity']
            labels = ('baseline', 'restore_q_both', 'drop_nnm_out', 'drop_nm_out', 'drop_bnm_out', 'drop_nnm_nm_out')
            for j, label in enumerate(labels):
                vals = [r['logit_difference_percent'] for r in points if r['label'] == label]
                axes[i].scatter(np.full(len(vals), j), vals, color=COLORS[algo], s=20)
                axes[i].plot([j-.15, j+.15], [np.mean(vals)]*2, color='black')
            axes[i].axhline(100, color='grey', linestyle='--')
            axes[i].set_xticks(range(len(labels)), ['Circuit', 'Q', '−NNM', '−NM', '−BNM', '−NNM−NM'], rotation=45)
            axes[i].set_title(NAMES[algo]); axes[i].set_ylabel('Logit difference (%)')
        fig.suptitle(f'Mean replacement interventions, requested {budget:,} edges')
        save(fig, f'remaining_overshoot_{budget}')
    fig, axes = plt.subplots(len(budgets), 2, figsize=(12, 5*len(budgets)), squeeze=False)
    for b, budget in enumerate(budgets):
        for i, algo in enumerate(NAMES):
            for label, offset in [('restore_q_both', -.15), ('control_', 0.), ('fixed_', .15)]:
                vals = [e['delta_margin'] for e in effects if e['algo'] == algo and e['phase'] == 'causal' and e['budget'] == budget and (e['label'].startswith(label) if label.endswith('_') else e['label'] == label)]
                if vals:
                    axes[b, 0].scatter(np.full(len(vals), i+offset), vals, color=COLORS[algo], s=10, alpha=.4, marker='o' if offset else 'x')
            exact = [r for r in exact_run_rows(rows) if r['algo'] == algo and r['label'] == 'exact_baseline' and r['budget'] == budget]
            native = [r for r in rows if r['algo'] == algo and r['phase'] == 'screen' and r['budget'] == budget]
            for group, offset in [(native, -.1), (exact, .1)]:
                if group:
                    axes[b, 1].scatter(np.full(len(group), i+offset), [r['logit_difference_percent'] for r in group], color=COLORS[algo], alpha=.5, s=15)
        axes[b, 0].set_title(f'{budget:,}: left Q / center controls / right fixed Q')
        axes[b, 0].set_ylabel('Change in raw margin'); axes[b, 0].axhline(0, color='grey')
        axes[b, 1].set_title(f'{budget:,}: left native / right exact size')
        axes[b, 1].set_ylabel('Logit difference (%)'); axes[b, 1].axhline(100, color='grey', linestyle='--')
        for ax in axes[b]:
            ax.set_xticks(range(4), [NAMES[a] for a in NAMES], rotation=15)
    save(fig, 'controls_and_size')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    terms = [('kl_partition', 'Pair-versus-rest mass', '#0072B2'), ('kl_pair', 'Within-name probabilities', '#D55E00'), ('kl_other', 'Remaining-vocabulary probabilities', '#009E73')]
    for b, budget in enumerate(budgets):
        selected = [r for r in rows if r['phase'] == 'screen' and r['budget'] == budget]
        bottom = np.zeros(4)
        for key, label, color in terms:
            vals = np.array([np.mean([r[key] for r in selected if r['algo'] == a]) for a in NAMES])
            axes[b].bar(range(4), vals, bottom=bottom, label=label, color=color)
            bottom += vals
        axes[b].set_xticks(range(4), [NAMES[a] for a in NAMES], rotation=25)
        axes[b].set_title(f'Mean ablation, {budget:,} requested edges')
        axes[b].set_ylabel('KL(full || circuit), nats')
    axes[0].legend(fontsize=7)
    save(fig, 'kl_partition')


def report(output, manifest, rows, summary, full):
    lines=['# IOI retained-head investigation', '',
           f'Study identity: `{manifest["study_id"]}`. All {len(manifest["conditions"])} predefined conditions verified.', '',
           'Mean discovery and mean evaluation only. Five joint discovery runs and 256 previously analyzed prompts were used. The primary requested budget is 2000; 739 and 3000 test size dependence. The former resample study is excluded.', '',
           f'Full-model average raw margin: {full["margin"].mean():.6f}. Percentages use this denominator, not per-prompt ratios.', '']
    for budget in manifest['budgets']:
        lines += [f'## Native mean circuits at requested {budget:,} edges', '',
                  '| Method | Attained size range | Logit difference % (average ± sample SD) | Q restoration % | NNM mean-output replacement % | NNM outputs retained |',
                  '|---|---:|---:|---:|---:|---:|']
        for algo in NAMES:
            selected=[r for r in rows if r['algo']==algo and r['budget']==budget and r['phase']!='size-sensitivity']
            base=[r for r in selected if r['label']=='baseline']
            vals=np.array([r['logit_difference_percent'] for r in base])
            q=np.mean([r['logit_difference_percent'] for r in selected if r['label']=='restore_q_both'])
            o=np.mean([r['logit_difference_percent'] for r in selected if r['label']=='drop_nnm_out'])
            retained=sum(r['retained_10_7'] and r['retained_11_10'] for r in base)
            lines.append(f'| {NAMES[algo]} | {min(r["attained"] for r in base)}–{max(r["attained"] for r in base)} | {vals.mean():.2f} ± {vals.std(ddof=1):.2f} | {q:.2f} | {o:.2f} | {retained}/5 |')
        lines += ['', '| Method | Baseline KL | Q restoration KL | NNM output replacement KL | Pair mass KL | Within-pair KL | Remaining-vocabulary KL |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for algo in NAMES:
            selected=[r for r in rows if r['algo']==algo and r['budget']==budget and r['phase']!='size-sensitivity']
            def avg(label, key):
                return np.mean([r[key] for r in selected if r['label']==label])
            values = [avg('baseline','kl'), avg('restore_q_both','kl'), avg('drop_nnm_out','kl')]
            values += [avg('baseline', key) for key in ('kl_partition','kl_pair','kl_other')]
            lines.append('| '+NAMES[algo]+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
        exact = [r for r in exact_run_rows(rows) if r['budget'] == budget]
        if exact:
            lines += ['', 'Exact-size sensitivity (tie realizations averaged within each discovery run; already-exact native runs included once):', '',
                      '| Method | Discovery runs | Baseline logit difference % | Q restoration % | NNM output replacement % | Baseline KL | Q restoration KL | NNM output replacement KL |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for algo in NAMES:
                group = [r for r in exact if r['algo'] == algo]
                if not group:
                    continue
                def avg_exact(label, key):
                    return np.mean([r[key] for r in group if r['label'] == label])
                values = [avg_exact(label, key) for key in ('logit_difference_percent', 'kl')
                          for label in ('exact_baseline', 'exact_restore_q_both', 'exact_drop_nnm_out')]
                n = sum(r['label'] == 'exact_baseline' for r in group)
                lines.append(f'| {NAMES[algo]} | {n} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['', '## Paired intervention evidence', '',
            '| Method | Requested edges | Intervention | Metric | Average change | 95% crossed bootstrap interval | Positive / negative runs |',
            '|---|---:|---|---|---:|---:|---:|']
    labels={'restore_q_both','drop_nnm_out','full_q_both','restore_q_nm','restore_q_si','restore_q_other',
            'control_average','fixed_paired_q','output_effect_change_after_q','drop_nm_out','drop_bnm_out','drop_nnm_nm_out',
            'exact_restore_q_both','exact_drop_nnm_out','exact_restore_q_drop_out'}
    for row in summary:
        if row['metric'] in ('margin','kl') and row['label'] in labels:
            lines.append(f'| {NAMES[row["algo"]]} | {row["budget"]} | {row["label"]} | {row["metric"]} | {row["average"]:.4f} | [{row["ci_low"]:.4f}, {row["ci_high"]:.4f}] | {row["positive_runs"]} / {row["negative_runs"]} |')
    lines+=['', '## Evidence boundaries', '',
            '- Attention, absolute output attribution, corrupt-replacement contrast, and causal output-edge effects are separate measurements in the saved arrays.',
            '- Inspect IO and subject logits separately before attributing a margin change to suppression. LayerNorm attribution is diagnostic, while intervention effects recompute normalization.',
            '- Group effects are conditional and may interact. Their sum is not an additive explanation of the spike. Removing a direct output edge preserves indirect edges.',
            '- `output_effect_change_after_q` = (baseline − output-ablated) − (Q-restored − Q-restored/output-ablated). Positive values mean the retained-output margin contribution is larger before Q restoration.',
            '- Missing-edge restoration and intact-query replacement answer different questions. No-op S-Inhibition restoration does not imply retained S-Inhibition signals are unchanged.',
            '- Controls and tie resolutions are averaged/summarized within runs. They are not additional discovery replications. Five runs limit uncertainty estimation.',
            '- Primary ACDC circuits preserve threshold ties. Exact-size subsets provide sensitivity evidence and are not the original discovery output. Exact size refers to the baseline; Q restoration and output replacement change its size, recorded separately in the endpoint CSV.',
            '- Every causal output replacement substitutes the fixed corrupt-tokenwise-mean source output, not an individual corrupt activation. `contrast_dla` uses this same mean reference.',
            '- The three KL partition terms are mathematical distribution-level terms, not additive causal head-group contributions. Conditional terms are weighted by full-model pair/rest masses.',
            '- This pooled tokenwise mean and component-edge intervention differ from Wang et al.\'s template-conditioned ABC mean and head/token-position knockouts. The replacement procedure has not been matched experimentally to that original protocol.',
            '- Previously analyzed prompts, five seeds and exploratory intervals limit generalization. No historical resample result is evidence for this mean-only study.', '',
            '## Matched-control availability', '',
            *[f'- {NAMES[c["cell"]["algo"]]}, seed {c["cell"]["seed"]}: {c["control_status"]}.' for c in manifest['cases']], '',
            '## Artifacts', '',
            '`per_run.csv`: all baseline and intervention endpoints, edge counts, retention, and head diagnostics.',
            '`paired_effects.csv`: individual intervention effects; `effect_summary.csv`: discovery-run summaries and crossed uncertainty.',
            '`exact_per_run.csv`: exact-size endpoints averaged over tie resolutions within discovery runs; exact paired effects in `effect_summary.csv` use the same within-run averaging.',
            '`conditions/`: exact condition metadata and per-prompt arrays; `inventory.json` and `masks.npz`: frozen inputs and connectivity.',
            '`exhibits/`: recurrence, causal comparison, remaining overshoot, and controls/size sensitivity figures.', '']
    (output/'report.md').write_text('\n'.join(lines))
