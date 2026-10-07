"""Baseline export must use the analysis's actual evaluation inputs."""
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.thesis import evaluation_protocol as protocol
from experiments.thesis import results_chapter_manual_baseline as baseline
from experiments.thesis.run_cell import _TASK_TEMPLATES, _data_budget


@pytest.fixture
def setup(monkeypatch, tmp_path):
    key = _TASK_TEMPLATES['ioi'].key
    batch = SimpleNamespace(**{name: torch.tensor([[i]]) for i, name in
        enumerate(('clean', 'corrupt', 'answers', 'wrong_answers'), 1)})

    class Loader(list):
        split_metadata = {'protocol': 'fixed_holdout_v1', 'split_seed': 42,
                          'pool_sha256': 'pool', 'test_clean_sha256': 'clean',
                          'train_row_ids': [1], 'test_row_ids': [2]}

    model = torch.nn.Linear(1, 1)
    model.tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 1})
    task = SimpleNamespace(key=key, name='IOI', model=model, true_edge_count=2,
        test_loader=Loader([batch]), batch_size=64, batch_count=(11, 4))
    tasks = {key: task}
    monkeypatch.setattr(baseline, 'ROOT', tmp_path)
    monkeypatch.setattr(baseline, 'CACHE', tmp_path / 'legacy')
    monkeypatch.setattr(baseline, 'OUT', tmp_path / 'historical-output')
    monkeypatch.setattr(baseline, 'TASKS', {'ioi': ('cell_ioi_gt_resample_seed42', 'mean', 2)})
    monkeypatch.setitem(baseline.TASK_DICT, key, task)
    monkeypatch.setattr(protocol, 'build_evaluation_tasks', lambda *args, **kwargs: tasks)
    cache = tmp_path / 'fresh'
    source = tmp_path / 'analysis'
    output = tmp_path / 'output'
    cache.mkdir()
    source.mkdir()
    reference = {'complete': True, 'cells': [{'task': 'ioi'}],
                 'evaluation_protocol_id': protocol.fixed_evaluation_id(tasks)}
    (source / 'manifest.json').write_text(json.dumps(reference))
    seen = []

    def measure(**kwargs):
        seen.append(kwargs)
        assert kwargs['evaluation_tasks'] == tasks
        return {}

    monkeypatch.setattr(baseline, 'measure_prune_metrics', measure)
    monkeypatch.setattr(baseline, 'extract_metric', lambda *args: [(2, 1.25)])

    def write(folder, split_seed=42, stamp='2026-10-07T10:00:00', orphan=False):
        folder.mkdir(exist_ok=True)
        tag = '' if split_seed is None else f'_holdout{split_seed}'
        stem = f'cell_ioi_gt_resample_seed42{tag}-{stamp.replace(":", "_")}'
        scores = folder / f'{stem}.pkl'
        torch.save({key: {'Ground Truth': {'m': torch.tensor([1., 1., 0.])}}}, scores)
        manifest = {'cell': {'task': 'ioi', 'algo': 'gt', 'ablation': 'resample',
                            'seed': 42, 'init_seed': 42, 'split_seed': split_seed},
                    'cache_file': scores.name, 'saved_at': stamp,
                    'data_budget': _data_budget(task), 'data_split': dict(task.test_loader.split_metadata)}
        path = folder / f'{stem}_manifest.json'
        if not orphan:
            path.write_text(json.dumps(manifest))
        return scores, path, manifest

    return SimpleNamespace(cache=cache, source=source, output=output, task=task,
        tasks=tasks, seen=seen, write=write, reference=reference, root=tmp_path)


def export(fixture, **kwargs):
    baseline.main(output_dir=fixture.output, split_seed=42,
                  cache_dir=fixture.cache, source_dir=fixture.source, **kwargs)


def test_fixed_baseline_uses_fresh_cache_and_exact_analysis_protocol(setup):
    scores, _, _ = setup.write(setup.cache)
    setup.write(setup.root / 'legacy', split_seed=None)
    export(setup)
    manifest = json.loads((setup.output / 'manual_baselines_manifest.json').read_text())
    assert manifest['evaluation_protocol_id'] == setup.reference['evaluation_protocol_id']
    assert not manifest['temporary_historical_repair'] and manifest['complete']
    assert manifest['sources'][0]['cache'] == str(scores.relative_to(setup.root))
    assert manifest['sources'][0]['evaluation_clean_sha256'] == 'clean'
    assert len(setup.seen) == 2
    assert setup.seen[1]['reverse_clean_corrupt'] is True
    with (setup.output / 'manual_baselines.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (setup.output / 'manual_completeness.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 3


@pytest.mark.parametrize('change', ['pool', 'clean', 'split', 'budget'])
def test_fixed_baseline_rejects_incompatible_discovery_before_measurement(setup, change):
    _, path, manifest = setup.write(setup.cache)
    if change == 'pool':
        manifest['data_split']['pool_sha256'] = 'other'
    elif change == 'clean':
        manifest['data_split']['test_clean_sha256'] = 'other'
    elif change == 'split':
        manifest['data_split']['split_seed'] = 43
    else:
        manifest['data_budget']['train_examples'] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='holdout|budget'):
        export(setup)
    assert not setup.seen and not setup.output.exists()


@pytest.mark.parametrize('change', ['protocol', 'incomplete', 'unmarked', 'tasks', 'corrupt_inputs'])
def test_baseline_rejects_analysis_mismatch_before_export(setup, change):
    setup.write(setup.cache)
    reference = dict(setup.reference)
    if change == 'protocol':
        reference['evaluation_protocol_id'] = 'legacy_template'
    elif change == 'incomplete':
        reference['complete'] = False
    elif change == 'unmarked':
        reference.pop('complete')
    elif change == 'tasks':
        reference['cells'] = [{'task': 'ioi'}, {'task': 'gt'}]
    else:
        setup.task.test_loader[0].corrupt = torch.tensor([[999]])
    (setup.source / 'manifest.json').write_text(json.dumps(reference))
    with pytest.raises(ValueError, match='protocol|Incomplete|task'):
        export(setup, task_keys=['ioi'])
    assert not setup.seen and not setup.output.exists()


def test_missing_fresh_baseline_cannot_fall_back_to_historical_cache(setup):
    setup.write(setup.root / 'legacy', split_seed=None)
    with pytest.raises(ValueError, match='official|baseline|cache'):
        export(setup)
    assert not setup.seen and not setup.output.exists()


def test_latest_complete_matching_identity_wins_over_orphan_or_wrong_split(setup):
    setup.write(setup.cache, stamp='2026-10-07T09:00:00')
    expected, _, _ = setup.write(setup.cache, stamp='2026-10-07T10:00:00')
    setup.write(setup.cache, stamp='2026-10-07T11:00:00', orphan=True)
    setup.write(setup.cache, split_seed=43, stamp='2026-10-07T12:00:00')
    export(setup)
    result = json.loads((setup.output / 'manual_baselines_manifest.json').read_text())
    assert result['sources'][0]['cache'] == str(expected.relative_to(setup.root))


def test_fixed_export_requires_separate_output_and_analysis_reference(setup):
    with pytest.raises(ValueError, match='output|source|analysis'):
        baseline.main(split_seed=42, cache_dir=setup.cache)
    assert not setup.seen


def test_historical_mode_ignores_fixed_holdout_cache_files(setup):
    cache = setup.root / 'legacy'
    expected, _, _ = setup.write(cache, split_seed=None)
    setup.write(cache, split_seed=42, stamp='2026-10-07T11:00:00')
    baseline.main()
    result = json.loads((setup.root / 'historical-output/manual_baselines_manifest.json').read_text())
    assert result['evaluation_protocol_id'] == 'legacy_template'
    assert result['sources'][0]['cache'] == str(expected.relative_to(setup.root))
    assert len(setup.seen) == 1


def test_temporary_repair_loads_frozen_score_from_inventory_directory(setup, monkeypatch):
    scores, path, cell = setup.write(setup.cache, split_seed=None)
    frozen = {'scores': str(scores), 'manifest': str(path), 'cell': cell['cell']}
    artifact = {'protocol_id': 'repair', 'tasks': {'ioi': {
        'identity_checks': {'ordered_clean_tokens_sha256': 'repair-clean'}}}}
    monkeypatch.setattr(protocol, 'load_protocol', lambda path: artifact)
    monkeypatch.setattr(protocol, 'load_frozen_inputs', lambda path: [frozen])
    monkeypatch.setattr(protocol, 'verify_evaluation_task', lambda *args: None)
    baseline.main('artifact.json', 'inventory.json', setup.output)
    result = json.loads((setup.output / 'manual_baselines_manifest.json').read_text())
    assert result['evaluation_protocol_id'] == 'repair'
    assert result['sources'][0]['cache'] == str(scores.relative_to(setup.root))
    assert result['sources'][0]['evaluation_clean_sha256'] == 'repair-clean'
    assert len(setup.seen) == 2


def test_zero_is_a_valid_holdout_seed(setup, monkeypatch):
    setup.task.test_loader.split_metadata['split_seed'] = 0
    setup.reference['evaluation_protocol_id'] = protocol.fixed_evaluation_id(setup.tasks)
    (setup.source / 'manifest.json').write_text(json.dumps(setup.reference))
    setup.write(setup.cache, split_seed=0)

    def build(artifact, tasks, split_seed):
        setup.task.test_loader.split_metadata['split_seed'] = split_seed
        return setup.tasks

    monkeypatch.setattr(protocol, 'build_evaluation_tasks', build)
    baseline.main(output_dir=setup.output, split_seed=0, cache_dir=setup.cache,
                  source_dir=setup.source)
    result = json.loads((setup.output / 'manual_baselines_manifest.json').read_text())
    assert result['split_seed'] == 0
    assert result['evaluation_protocol_id'] == setup.reference['evaluation_protocol_id']


def test_interrupted_completeness_write_marks_existing_output_incomplete(setup, monkeypatch):
    setup.write(setup.cache)
    setup.output.mkdir()
    path = setup.output / 'manual_baselines_manifest.json'
    path.write_text(json.dumps({'complete': True, 'evaluation_protocol_id': 'old'}))
    original = csv.DictWriter.writerows

    def write(writer, rows):
        if 'value' in rows[0]:
            raise OSError('Interrupted completeness output')
        return original(writer, rows)

    monkeypatch.setattr(csv.DictWriter, 'writerows', write)
    with pytest.raises(OSError, match='Interrupted'):
        export(setup)
    manifest = json.loads(path.read_text())
    assert manifest['complete'] is False
    with pytest.raises(ValueError, match='Incomplete'):
        protocol.require_same_protocol(manifest, setup.reference)


@pytest.mark.parametrize('incomplete', ['source', 'baseline'])
def test_curve_join_preserves_incomplete_bundle_flags(incomplete):
    from experiments.thesis.results_chapter_size_curves import validate_source_protocol
    source = {'evaluation_protocol_id': 'same', 'complete': True}
    manual = {'evaluation_protocol_id': 'same', 'complete': True}
    (source if incomplete == 'source' else manual)['complete'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        validate_source_protocol(source, manual)
