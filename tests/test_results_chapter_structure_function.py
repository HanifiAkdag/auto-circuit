import importlib
import importlib.util

import numpy as np
import pytest


def analysis():
    name = 'experiments.thesis.results_chapter_structure_function'
    assert importlib.util.find_spec(name) is not None, 'Paired analysis module is missing'
    return importlib.import_module(name)


def test_pair_identity_and_overlap_are_preserved():
    masks = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=bool)
    rows = analysis().paired_records(masks, [1., 2., 4.], [42, 43, 44])
    first = rows[0]
    assert (first['seed_i'], first['seed_j']) == (42, 43)
    assert first['jaccard'] == pytest.approx(1 / 3)
    assert first['gap'] == 1
    assert len(rows) == 3
    assert rows[1]['size_difference'] == 1


def test_simultaneous_run_reordering_preserves_the_association():
    masks = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=bool)
    mod = analysis()
    a = mod.paired_records(masks, [1., 2., 4.], [42, 43, 44])
    order = [2, 0, 1]
    b = mod.paired_records(masks[order], np.array([1., 2., 4.])[order], np.array([42, 43, 44])[order])
    assert mod.correlation(a) == pytest.approx(mod.correlation(b))


def test_constant_performance_has_undefined_correlation():
    pairs = [{'dissimilarity': .1, 'gap': 0}, {'dissimilarity': .2, 'gap': 0}, {'dissimilarity': .3, 'gap': 0}]
    assert analysis().correlation(pairs) is None


def test_tied_ranks_are_averaged():
    pairs = [{'dissimilarity': x, 'gap': y} for x, y in zip([1, 2, 2, 4], [1, 3, 2, 4])]
    assert analysis().correlation(pairs) == pytest.approx(0.9486832980505138)


def test_omitting_a_run_removes_every_incident_pair():
    mod = analysis()
    pairs = mod.paired_records(np.eye(5, dtype=bool), [1, 2, 4, 8, 16], list(range(42, 47)))
    omitted = mod.omission_records(pairs, list(range(42, 47)))
    assert len(omitted) == 5
    assert all(r['n_pairs'] == 6 for r in omitted)
    assert all(r['rho'] is None for r in omitted)


def test_precision_audit_detects_unresolved_gap_order():
    result = analysis().precision_audit([1, 2, 3.00001], [1e-4] * 3)
    assert result['ambiguous_orders'] > 0


def test_exact_accuracy_grid_preserves_true_gap_ties():
    mod = analysis()
    values = mod.accuracy_values('87.5;81.6406;89.4531;78.9062;79.6875')
    assert values == [87.5, 81.640625, 89.453125, 78.90625, 79.6875]
    assert mod.precision_audit([0., 1., 2.], [0., 0., 0.])['ambiguous_orders'] == 0


def test_accuracy_grid_rejects_values_not_supported_by_rounding():
    with pytest.raises(ValueError):
        analysis().accuracy_values('50.2;100;100;100;100')


def test_only_acdc_accepts_its_infinite_retained_edge_sentinel():
    mod = analysis()
    assert hasattr(mod, 'validate_scores'), 'ACDC sentinel validation is missing'
    mod.validate_scores(np.array([np.inf, .003, .001]), 'ACDC')
    with pytest.raises(ValueError):
        mod.validate_scores(np.array([np.inf, .003]), 'EAP')
    with pytest.raises(ValueError):
        mod.validate_scores(np.array([np.nan, .003]), 'ACDC')


def test_run_points_preserve_actual_scores_and_average_incident_distances():
    mod = analysis()
    pairs = mod.paired_records(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
                               [1., 5., 9.], [42, 43, 44])
    assert hasattr(mod, 'run_points'), 'Run-level plotting transformation is missing'
    points = mod.run_points(pairs[::-1])
    assert [p['seed'] for p in points] == [42, 43, 44]
    assert [p['score'] for p in points] == [1., 5., 9.]
    assert [p['average_dissimilarity'] for p in points] == pytest.approx([.75, .75, 1.])


def test_run_points_reject_incomplete_comparisons():
    mod = analysis()
    pairs = mod.paired_records(np.eye(5, dtype=bool), [1., 2., 3., 4., 5.], list(range(42, 47)))
    assert hasattr(mod, 'run_points'), 'Run-level plotting transformation is missing'
    with pytest.raises(ValueError):
        mod.run_points(pairs[:-1])
