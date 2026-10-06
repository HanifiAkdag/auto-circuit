import importlib
import json
import random
from types import SimpleNamespace

import pytest
import torch

from auto_circuit.data import load_datasets_from_json
from auto_circuit.prune_algos.prune_algos import PruneAlgo, run_prune_algos
from experiments.thesis.run_cell import _build_task, _cache_stem


def _pool(tmp_path, duplicate=False):
    records = [dict(clean=[i, 1], corrupt=[i + 2000, 1], answers=[2], wrong_answers=[3])
               for i in range(1000)]
    if duplicate:
        records[999] = records[0].copy()
    path = tmp_path / 'pool.json'
    path.write_text(json.dumps({'prompts': records}))
    return path


def _load(path, seed, **kwargs):
    return load_datasets_from_json(None, path, torch.device('cpu'), prepend_bos=False,
                                  batch_size=64, train_test_size=(704, 256),
                                  random_seed=seed, **kwargs)


def _ids(loader):
    return [int(value) for batch in loader for value in batch.clean[:, 0]]


def test_historical_production_membership_and_own_split(tmp_path):
    path = _pool(tmp_path)
    common = set(_ids(_load(path, 42)[1]))
    intersections = []
    first_batch = []
    for seed in range(42, 47):
        train, test = _load(path, seed)
        expected = list(range(1000))
        random.Random(seed).shuffle(expected)
        assert _ids(train) == expected[:704]
        assert _ids(test) == expected[704:960]
        assert set(_ids(train)).isdisjoint(_ids(test))
        intersections.append(len(common.intersection(_ids(train))))
        first_batch.append(len(common.intersection(_ids(train)[:64])))
    assert intersections == [0, 186, 186, 174, 187]
    assert first_batch == [0, 17, 18, 12, 17]


def test_discovery_boundary_uses_training_loader():
    seen = []
    train = object()
    task = SimpleNamespace(name='T', key='T', model=object(), train_loader=train,
                           test_loader=object(), true_edges=None)
    algo = PruneAlgo('A', 'A', lambda model, loader, official: seen.append(loader) or {})
    run_prune_algos([task], [algo])
    assert seen == [train]


def test_future_holdout_is_common_and_unseen_across_seeds(tmp_path):
    path = _pool(tmp_path, duplicate=True)
    memberships = []
    test_order = None
    for seed in range(42, 47):
        train, test = _load(path, seed, split_seed=42)
        current_test = _ids(test)
        assert len(current_test) == len(set(current_test)) == 256
        assert set(current_test).isdisjoint(_ids(train))
        assert len(_ids(train)) == 704
        if test_order is None:
            test_order = current_test
        assert current_test == test_order
        memberships.append(_ids(train))
    assert len({tuple(rows) for rows in memberships}) == 5


def test_fixed_holdout_excludes_cross_role_training_input(tmp_path):
    path = _pool(tmp_path)
    rows = json.loads(path.read_text())
    order = list(range(1000))
    random.Random(42).shuffle(order)
    rows['prompts'][order[300]]['corrupt'] = rows['prompts'][order[0]]['clean']
    path.write_text(json.dumps(rows))
    train, test = _load(path, 43, split_seed=42)
    eval_clean = {tuple(p.tolist()) for batch in test for p in batch.clean}
    training_inputs = {tuple(p.tolist()) for batch in train
                       for role in (batch.clean, batch.corrupt) for p in role}
    assert eval_clean.isdisjoint(training_inputs)


def test_holdout_protocol_in_task_and_cache_identity():
    task = _build_task('ioi', 43, False, split_seed=42)
    assert task.seed == 43 and task.split_seed == 42
    assert _cache_stem('ioi', 'ep', 'mean', 43, None, split_seed=42) != _cache_stem('ioi', 'ep', 'mean', 43, None)


def test_evaluation_boundary_uses_explicit_loader(monkeypatch):
    module = importlib.import_module('auto_circuit.metrics.prune_metrics.measure_prune_metrics')
    template = SimpleNamespace(key='T', name='old', test_loader=object(), model=object())
    loader = object()
    model = SimpleNamespace(edges=set())
    evaluation = SimpleNamespace(key='T', name='new', test_loader=loader, model=model, true_edge_count=1)
    metric_seen = []
    circuit_seen = []
    metric = SimpleNamespace(key='M', name='M', metric_func=lambda m, l, out: metric_seen.append(l) or [(1, 0.)])
    monkeypatch.setitem(module.TASK_DICT, 'T', template)
    monkeypatch.setitem(module.PRUNE_ALGO_DICT, 'A', SimpleNamespace(key='A', name='A'))
    monkeypatch.setattr(module, 'edge_counts_util', lambda **kwargs: [1])
    monkeypatch.setattr(module, 'run_circuits', lambda **kwargs: circuit_seen.append(kwargs['dataloader']) or {})
    module.measure_prune_metrics([], [], {}, None)
    from auto_circuit.types import AblationType, PatchType
    module.measure_prune_metrics([AblationType.ZERO], [metric], {'T': {'A': {}}},
                                 PatchType.TREE_PATCH, evaluation_tasks={'T': evaluation})
    assert circuit_seen == metric_seen == [loader]


def test_explicit_evaluation_mapping_cannot_fall_back(monkeypatch):
    module = importlib.import_module('auto_circuit.metrics.prune_metrics.measure_prune_metrics')
    with pytest.raises(KeyError):
        module.measure_prune_metrics([], [], {'missing': {}}, None, evaluation_tasks={})
