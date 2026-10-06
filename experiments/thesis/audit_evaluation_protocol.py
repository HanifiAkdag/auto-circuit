"""CPU production-loader confirmation and immutable historical input inventory."""
import json
import tempfile
from pathlib import Path

import torch

from auto_circuit.data import load_datasets_from_json
from .evaluation_protocol import ROOT, fingerprint, sha256


def audit(output: Path) -> dict:
    original_audit = ROOT / 'results/results_chapter/evaluation_overlap_reaudit_20261006'
    provenance = json.loads((original_audit / 'provenance.json').read_text())
    for check in provenance['dataset_checks']:
        assert sha256(ROOT / check['file']) == check['historical_sha256'], check
    selected = []
    for recorded in provenance['selected_runs']:
        path = ROOT / recorded['manifest']
        manifest = json.loads(path.read_text())
        assert manifest['cell']['seed'] == recorded['data_seed']
        assert manifest['data_budget'] == recorded['data_budget']
        assert manifest['git']['commit'] == recorded['commit']
        selected.append({'manifest': recorded['manifest'], 'sha256': sha256(path)})
    assert len(selected) == 120
    matrices = {}
    for task, path in [('ioi', ROOT / 'datasets/ioi/ioi_prompts.json'),
                       ('gt', ROOT / 'datasets/greaterthan_gpt2-small_prompts.json')]:
        original = json.loads(path.read_text())['prompts']
        marked = [dict(clean=[i, 1], corrupt=[i + 2000, 1], answers=[2], wrong_answers=[3])
                  for i in range(len(original))]
        own, counts, records, first, witnesses = {}, [], [], [], []
        with tempfile.TemporaryDirectory() as directory:
            tagged_path = Path(directory) / 'tagged.json'
            tagged_path.write_text(json.dumps({'prompts': marked}))
            for seed in range(42, 47):
                train, test = load_datasets_from_json(None, tagged_path, torch.device('cpu'),
                    prepend_bos=False, batch_size=64, train_test_size=(704, 256), random_seed=seed)
                ids = lambda loader: [int(i) for batch in loader for i in batch.clean[:, 0]]
                own[seed] = (ids(train), ids(test))
                assert len(own[seed][0]) == 704 and len(own[seed][1]) == 256
                assert set(own[seed][0]).isdisjoint(own[seed][1])
        common = own[42][1]
        for seed, (train, _) in own.items():
            shared = set(common).intersection(train)
            counts.append(len(shared))
            first.append(len(set(common).intersection(train[:64])))
            train_records = {fingerprint(original[i]) for i in train}
            records.append(sum(fingerprint(original[i]) in train_records for i in common))
            if shared:
                idx = min(shared)
                witnesses.append({'seed': seed, 'original_row': idx, 'evaluation_position': common.index(idx),
                                  'training_position': train.index(idx), 'record': original[idx]})
        assert counts == [0, 186, 186, 174, 187]
        assert first == [0, 17, 18, 12, 17]
        assert records == ([0, 186, 186, 174, 187] if task == 'ioi' else [4, 186, 188, 175, 189])
        matrices[task] = {'row_overlap': counts, 'complete_record_overlap': records,
                         'first_batch_overlap': first, 'witnesses': witnesses}
    cache = ROOT / '.prune_scores/final/benchmark'
    best = {}
    for path in sorted(cache.glob('*_manifest.json')):
        manifest = json.loads(path.read_text())
        cell = manifest.get('cell', {})
        if cell.get('task') not in ('ioi', 'gt') or cell.get('null_test') or cell.get('split_seed') is not None:
            continue
        if cell.get('algo') not in ('ep', 'discogp-ep', 'acdc', 'eap', 'gt', 'random'):
            continue
        identity = tuple(cell.get(k) for k in ('task', 'algo', 'ablation', 'seed')) + (cell.get('init_seed', cell.get('seed')),)
        if identity not in best or manifest['saved_at'] > best[identity][0]['saved_at']:
            best[identity] = manifest, path
    inputs = []
    for manifest, path in best.values():
        scores = path.parent / manifest['cache_file']
        inputs.append({'manifest': str(path.relative_to(ROOT)), 'manifest_sha256': sha256(path),
                       'scores': str(scores.relative_to(ROOT)), 'scores_sha256': sha256(scores),
                       'cell': manifest['cell'], 'data_budget': manifest.get('data_budget')})
    output.mkdir(parents=True, exist_ok=True)
    result = {'conclusion': 'confirmed', 'production_split': 'real load_datasets_from_json on unique row-token representations of the actual pools; content checks on unmodified original records',
              'selected_joint_provenance': selected, 'matrices': matrices,
              'exposure': 'source/configuration tracing from the historical audit; no algorithm training executed',
              'limitations': ['Historical dirty EP runner contents remain unavailable.',
                             'No claim about inflated scores, memorization or rankings.'],
              'frozen_inputs': inputs}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    (output / 'selected_inputs.json').write_text(json.dumps({'inputs': inputs}, indent=2))
    print(json.dumps({'conclusion': result['conclusion'], 'frozen_inputs': len(inputs),
                      'matrices': {k: {a: v for a, v in m.items() if a != 'witnesses'} for k, m in matrices.items()}}, indent=2))
    return result


if __name__ == '__main__':
    audit(ROOT / 'results/results_chapter/evaluation_overlap_integration_20261006')
