import importlib.util
import json

import numpy as np
import pytest
import torch


def derivation():
    name = 'experiments.results_chapter_cross_discovery'
    assert importlib.util.find_spec(name) is not None, 'Cross-discovery derivation is missing'
    return __import__(name, fromlist=['select_mask'])


def test_threshold_comparison_preserves_float64_boundary():
    scores = np.array([np.float32(.009), np.float32(.01)], dtype=np.float32)
    assert derivation().select_mask(scores, 'threshold', .009000000000000001).tolist() == [False, True]


def test_top_k_preserves_ties_and_uses_magnitudes():
    scores = np.array([-1., .8, -.8, .1])
    assert derivation().select_mask(scores, 'top_k', 2).tolist() == [True, True, True, False]


def test_pairing_uses_both_seed_identifiers_not_dictionary_order():
    a = {(42, 42): np.array([True, True, False]), (43, 43): np.array([False, False, True])}
    b = {(43, 43): np.array([False, False, True]), (42, 42): np.array([False, True, True])}
    rows = derivation().paired_rows(a, b)
    assert [(r['data_seed'], r['init_seed'], r['intersection'], r['union'], r['jaccard']) for r in rows] == [
        (42, 42, 1, 3, 1 / 3), (43, 43, 1, 1, 1)]


def test_pairing_rejects_mismatched_pytorch_seed():
    a = {(42, 42): np.array([True, False])}
    b = {(42, 43): np.array([True, False])}
    with pytest.raises(ValueError, match='seed'):
        derivation().paired_rows(a, b)


def test_tensor_alignment_ignores_dictionary_order_and_rejects_wrong_shape():
    module = derivation()
    scores = {'b': torch.tensor([-.4]), 'a': torch.tensor([[.1, .2]])}
    keys, shapes = ('a', 'b'), {'a': (1, 2), 'b': (1,)}
    assert module.flatten_scores(scores, keys, shapes).tolist() == pytest.approx([.1, .2, .4])
    with pytest.raises(ValueError, match='shape'):
        module.flatten_scores(scores, keys, {'a': (2,), 'b': (1,)})


def test_frozen_manifest_selection_excludes_isolated_seed_conditions(tmp_path):
    from experiments.config import sha256
    inputs = []
    for name, saved, init in [('selected', '2026-08-01', 42), ('isolated', '2026-09-02', 43)]:
        (tmp_path / (name + '.pkl')).write_bytes(b'cache')
        manifest = {
            'cell': {'task': 'ioi', 'algo': 'eap', 'ablation': 'mean', 'seed': 42, 'init_seed': init},
            'saved_at': saved, 'cache_file': name + '.pkl',
        }
        path = tmp_path / (name + '_manifest.json')
        path.write_text(json.dumps(manifest))
        scores = tmp_path / (name + '.pkl')
        inputs.append({'manifest': str(path), 'manifest_sha256': sha256(path),
                       'scores': str(scores), 'scores_sha256': sha256(scores), 'cell': manifest['cell']})
    records = derivation().frozen_manifest_index(inputs, {'ioi'}, {'eap'}, {'mean'}, {42})
    assert len(records) == 1
    assert records['ioi', 'eap', 'mean', 42][1].name == 'selected.pkl'
