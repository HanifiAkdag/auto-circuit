"""Compare identical historical circuits across the original and temporary evaluations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation_protocol import ROOT, load_protocol, load_frozen_inputs, sha256, require_same_protocol
from .temporary_evaluation_repair import HISTORICAL, _rows


def compare_rows(original, repaired):
    fields = ('task', 'algo', 'discovery_ablation', 'condition', 'eval_ablation',
              'measure', 'selector', 'budget_label', 'metric')
    key = lambda row: tuple(str(row[k]) for k in fields)
    old = {key(row): row for row in original}
    new = {key(row): row for row in repaired}
    if len(old) != len(original) or len(new) != len(repaired) or set(old) != set(new):
        raise ValueError('Original and repaired circuit/metric identities disagree')
    result = []
    for identity, row in sorted(new.items()):
        before = old[identity]
        if float(row['realised_mean']) != float(before['realised_mean']) or int(row['n_seeds']) != int(before['n_seeds']):
            raise ValueError('Circuit sizes or discovery seeds changed')
        result.append({**{k: row[k] for k in fields}, 'realised_mean': row['realised_mean'],
                       'original_average': float(before['metric_mean']), 'repaired_average': float(row['metric_mean']),
                       'average_change': float(row['metric_mean']) - float(before['metric_mean']),
                       'original_sd': float(before['metric_sd']), 'repaired_sd': float(row['metric_sd'])})
    return result


def main(output: Path):
    manifest = json.loads((output / 'manifest.json').read_text())
    if not manifest.get('complete') or not manifest.get('temporary_historical_repair'):
        raise ValueError('Repair is not complete')
    protocol = load_protocol(Path(manifest['evaluation_manifest']))
    require_same_protocol(manifest, {'evaluation_protocol_id': protocol['protocol_id']})
    inputs_path = Path(manifest['selected_inputs'])
    load_frozen_inputs(inputs_path)
    if sha256(inputs_path) != manifest['selected_inputs_sha256']:
        raise ValueError('Selected score inventory changed')
    if sha256(HISTORICAL / 'structural.csv') != manifest['structural_source_sha256']:
        raise ValueError('Historical structural results changed')
    repaired = _rows(output / 'functional.csv')
    tasks = {row['task'] for row in repaired}
    original = [row for row in _rows(HISTORICAL / 'functional.csv') if row['task'] in tasks]
    original_structure = [row for row in _rows(HISTORICAL / 'structural.csv') if row['task'] in tasks]
    if _rows(output / 'structural.csv') != original_structure:
        raise ValueError('Structural measurements changed during functional repair')
    changes = compare_rows(original, repaired)
    with (output / 'evaluation_changes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    matched = defaultdict(list)
    for row in changes:
        if row['condition'] == 'joint' and row['measure'] == 'faithfulness' and row['discovery_ablation'] == row['eval_ablation']:
            matched[(row['task'], row['metric'])].append(row['average_change'])
    lines = ['# Temporary correction of historical evaluation', '',
             'This re-evaluates the existing IOI and Greater-Than circuits on 256 fresh clean prompts per task. Discovery scores, seeds, circuit selectors and attained sizes are unchanged. Historical result files and thesis figures are preserved.', '',
             '| Task | Clean prompts | Clean overlap with either original input role | Rows with a reused corrupt context |',
             '|---|---:|---:|---:|']
    for short, item in protocol['tasks'].items():
        checks = item['identity_checks']
        lines.append(f"| {short} | {checks['examples']} | {checks['clean_overlap_rows']} | {checks['corrupt_overlap_rows']} |")
    lines += ['', f"Verified {len(changes):,} corresponding functional rows and the frozen score/manifest hashes. Both faithfulness and completeness use the new clean inputs. Greater-Than corrupt-context reuse is permitted by the selected protocol; it does not establish unseen corrupt contexts.", '',
              '| Task | Metric | Median change across matched joint curve points | Largest absolute change |',
              '|---|---|---:|---:|']
    for (task, metric), values in sorted(matched.items()):
        lines.append(f'| {task} | {metric} | {np.median(values):.6g} | {max(abs(v) for v in values):.6g} |')
    lines += ['', 'Changes above are descriptive differences between two prompt samples, in each metric’s original units. They do not isolate a causal effect of evaluation overlap or establish that the earlier scores were inflated. See evaluation_changes.csv for every point and exhibits/ for regenerated figures and tables.', '',
              'New multi-seed discovery uses an independently fixed holdout, with clean test identities excluded from both training input roles. The temporary generated pool is never selected for new discovery. See ../../../docs/evaluation_protocol.md for the permanent protocol.', '',
              f"Evaluation protocol: `{protocol['protocol_id']}`."]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Compared {len(changes)} functional rows; wrote {output / "report.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/results_chapter/temporary_repair_20261006')
    main(parser.parse_args().output_dir)
