import importlib
import json
from pathlib import Path

import pytest


def test_generator_imports_are_safe(monkeypatch):
    import transformer_lens
    monkeypatch.setattr(transformer_lens.HookedTransformer, 'from_pretrained',
                        lambda *a, **kw: pytest.fail('Generator import loaded a model'))
    root = Path(__file__).resolve().parents[1]
    for relative in ('datasets/greaterthan_generator.py', 'datasets/ioi/ioi_generator.py'):
        spec = importlib.util.spec_from_file_location('generator_under_test', root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_clean_identity_rejects_shared_clean_with_different_corruption():
    from experiments.thesis.evaluation_protocol import validate_records
    old = [dict(clean='old', corrupt='background', answers=['a'], wrong_answers=['b'])]
    new = [dict(clean='old', corrupt='new background', answers=['a'], wrong_answers=['b'])]
    with pytest.raises(ValueError, match='clean'):
        validate_records(new, old, lambda text: tuple(text.encode()))


def test_shared_corrupt_context_is_permitted_and_counted():
    from experiments.thesis.evaluation_protocol import validate_records
    old = [dict(clean='old', corrupt='background', answers=['a'], wrong_answers=['b'])]
    new = [dict(clean='new', corrupt='background', answers=['a'], wrong_answers=['b'])]
    assert validate_records(new, old, lambda text: tuple(text.encode()))['corrupt_overlap_rows'] == 1


def test_token_collision_and_internal_duplicate_are_rejected():
    from experiments.thesis.evaluation_protocol import validate_records
    old = [dict(clean='old', corrupt='background', answers=['a'], wrong_answers=['b'])]
    new = [dict(clean='new', corrupt='new background', answers=['a'], wrong_answers=['b'])]
    with pytest.raises(ValueError, match='clean'):
        validate_records(new, old, lambda text: (1,))
    with pytest.raises(ValueError, match='duplicate'):
        validate_records(new + new, old, lambda text: tuple(text.encode()))


def test_protocol_detects_changed_artifact_and_keeps_path_independent_id(tmp_path):
    from experiments.thesis.evaluation_protocol import protocol_id, load_protocol, sha256
    data = tmp_path / 'ioi.json'
    data.write_text(json.dumps({'prompts': []}))
    value = {'schema': 1, 'purpose': 'temporary_historical_repair',
             'tasks': {'ioi': {'dataset': 'ioi.json', 'sha256': sha256(data)}}}
    value['protocol_id'] = protocol_id(value)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(value))
    assert load_protocol(path)['protocol_id'] == value['protocol_id']
    data.write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        load_protocol(path)


def test_future_cache_identity_does_not_dedupe_with_legacy():
    from auto_circuit.metrics.analysis_utils import latest_per_cell
    cell = {'task': 'ioi', 'algo': 'ep', 'ablation': 'mean', 'seed': 42,
            'init_seed': 42, 'null_test': False}
    records = [({'cell': cell, 'saved_at': 'a'}, {}),
               ({'cell': {**cell, 'split_seed': 42}, 'saved_at': 'b'}, {})]
    assert len(latest_per_cell(records)) == 2
