import json
from types import SimpleNamespace

import pytest


def test_source_join_rejects_old_baseline():
    from experiments.evaluation_protocol import require_same_protocol
    with pytest.raises(ValueError, match='protocol'):
        require_same_protocol({'evaluation_protocol_id': 'new'}, {'evaluation_protocol_id': 'old'})
    require_same_protocol({'evaluation_protocol_id': 'new'}, {'evaluation_protocol_id': 'new'})


def test_frozen_inventory_detects_changed_scores(tmp_path):
    from experiments.evaluation_protocol import load_frozen_inputs, sha256
    score = tmp_path / 'scores.pkl'
    manifest = tmp_path / 'run.json'
    score.write_bytes(b'initial')
    manifest.write_text('{}')
    path = tmp_path / 'selected.json'
    path.write_text(json.dumps({'inputs': [{'scores': str(score), 'scores_sha256': sha256(score),
        'manifest': str(manifest), 'manifest_sha256': sha256(manifest)}]}))
    assert len(load_frozen_inputs(path)) == 1
    score.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_frozen_inputs(path)


def test_functional_rows_explicit_evaluation_reaches_measurement(monkeypatch):
    import torch
    from auto_circuit.metrics.analysis_utils import Budget
    from auto_circuit.types import AblationType, PatchType
    from experiments import stability_analysis as module
    cell = module.Cell('ioi', 'ep', 'zero', seeds=[42], scores=[{'m': torch.ones(1)}])
    seen = []
    task = SimpleNamespace(name='T')
    tasks = {cell.task_key: task}
    metric = SimpleNamespace(key='M')
    monkeypatch.setattr(module, 'edge_score_vector', lambda index, ps: torch.ones(1))
    monkeypatch.setattr(module, 'budget_mask', lambda values, budget: torch.ones(1, dtype=torch.bool))
    monkeypatch.setattr(module, '_eval_ablations', lambda cell: [AblationType.ZERO])
    monkeypatch.setitem(module.TASK_METRICS, cell.task_key, [metric])
    def measure(**kwargs):
        seen.append(kwargs['evaluation_tasks'])
        return {AblationType.ZERO: {'M': {cell.task_key: {cell.algo_key: [(1, 0.)]}}}}
    monkeypatch.setattr(module, 'measure_prune_metrics', measure)
    module._functional_rows(cell, [None], [Budget('top_k', 1)], cell.scores,
                            [PatchType.TREE_PATCH], evaluation_tasks=tasks)
    assert seen == [tasks]


def test_acdc_merge_split_identity_must_match():
    from experiments.merge_acdc_partials import validate_partial_protocols
    shared = {'cell': {'split_seed': 42}, 'data_split': {'test_clean_sha256': 'a', 'pool_sha256': 'p'}}
    assert validate_partial_protocols([shared, shared]) == shared['data_split']
    with pytest.raises(ValueError, match='split'):
        validate_partial_protocols([shared, {'cell': {'split_seed': 43}, 'data_split': shared['data_split']}])
    with pytest.raises(ValueError, match='budget'):
        validate_partial_protocols([{**shared, 'data_budget': {'batch_size': 64}},
                                   {**shared, 'data_budget': {'batch_size': 32}}])


def test_cache_skip_requires_the_current_pool_and_budget(monkeypatch, tmp_path):
    from experiments import run_cell
    from experiments.evaluation_protocol import sha256
    path = tmp_path / 'pool.json'
    path.write_text(json.dumps({'prompts': [{'clean': 'a', 'corrupt': 'b'}]}))
    monkeypatch.setattr(run_cell, 'repo_path_to_abs_path', lambda _: path)
    import hashlib
    pool_hash = hashlib.sha256(json.dumps(json.loads(path.read_text())['prompts'], sort_keys=True).encode()).hexdigest()
    manifest = {'cell': {'split_seed': 42}, 'data_budget': run_cell._data_budget(run_cell._TASK_TEMPLATES['ioi']),
                'data_split': {'pool_sha256': pool_hash}}
    manifest = json.loads(json.dumps(manifest))
    assert run_cell._can_reuse_cache(manifest, 'ioi', 42)
    manifest['data_split']['pool_sha256'] = 'old'
    assert not run_cell._can_reuse_cache(manifest, 'ioi', 42)
    manifest['data_split']['pool_sha256'] = pool_hash
    manifest['data_budget']['train_examples'] = 1
    assert not run_cell._can_reuse_cache(manifest, 'ioi', 42)


def test_future_evaluation_checks_pool_and_holdout_identities():
    from experiments.evaluation_protocol import verify_holdout_identity
    expected = {'protocol': 'fixed_holdout_v1', 'split_seed': 42, 'pool_sha256': 'p', 'test_clean_sha256': 't'}
    source = {'cell': {'split_seed': 42}, 'data_split': dict(expected)}
    verify_holdout_identity([source], expected)
    source['data_split']['test_clean_sha256'] = 'old-test'
    with pytest.raises(ValueError, match='holdout'):
        verify_holdout_identity([source], expected)


def test_persisted_discovery_budget_matches_loader_budget():
    from experiments.evaluation_protocol import verify_holdout_identity
    from experiments.run_cell import _data_budget, _TASK_TEMPLATES
    expected = {'protocol': 'fixed_holdout_v1', 'split_seed': 42, 'pool_sha256': 'p', 'test_clean_sha256': 't'}
    budget = _data_budget(_TASK_TEMPLATES['ioi'])
    source = json.loads(json.dumps({'cell': {'split_seed': 42}, 'data_split': expected, 'data_budget': budget}))
    verify_holdout_identity([source], expected, budget)
    source['data_budget']['train_examples'] = 1
    with pytest.raises(ValueError, match='budget'):
        verify_holdout_identity([source], expected, budget)


def test_default_export_rejects_new_baseline_with_historical_curves():
    from experiments.results_chapter_size_curves import validate_source_protocol
    with pytest.raises(ValueError, match='protocol'):
        validate_source_protocol({}, {'evaluation_protocol_id': 'fixed-new'})
    validate_source_protocol({}, {})


def test_fixed_protocol_identity_tracks_content_not_training_draw():
    import torch
    from experiments.evaluation_protocol import fixed_evaluation_id
    batch = SimpleNamespace(**{k: torch.tensor([[1]]) for k in ('clean', 'corrupt', 'answers', 'wrong_answers')})
    class Loader(list):
        split_metadata = {'pool_sha256': 'p', 'test_clean_sha256': 't', 'train_row_ids': [1]}
    task = SimpleNamespace(test_loader=Loader([batch]), batch_size=1, batch_count=(1, 1),
                           model=SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: {'a': 1})))
    first = fixed_evaluation_id({'t': task})
    task.test_loader.split_metadata['train_row_ids'] = [2]
    assert fixed_evaluation_id({'t': task}) == first
    batch.corrupt = torch.tensor([[2]])
    assert fixed_evaluation_id({'t': task}) != first
    batch.answers = [torch.tensor([1, 2]), torch.tensor([3])]
    batch.wrong_answers = [torch.tensor([4]), torch.tensor([5, 6])]
    ragged = fixed_evaluation_id({'t': task})
    batch.answers[0][0] = 9
    assert fixed_evaluation_id({'t': task}) != ragged


def test_faithfulness_rejects_contradictory_evaluation_flags(monkeypatch):
    import sys
    from experiments.faithfulness_comparison import main
    monkeypatch.setattr(sys, 'argv', ['faithfulness', '--eval-manifest', 'nonexistent.json', '--legacy-eval'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_join_rejects_incomplete_functional_bundle():
    from experiments.evaluation_protocol import require_same_protocol
    with pytest.raises(ValueError, match='Incomplete'):
        require_same_protocol({'evaluation_protocol_id': 'p', 'complete': False}, {'evaluation_protocol_id': 'p'})
