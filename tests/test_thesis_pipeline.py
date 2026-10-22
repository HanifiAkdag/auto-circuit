"""Acceptance comparisons and immutable scientific input selection."""
import csv

import pytest


def test_percentage_comparison_preserves_accuracy_counts():
    from experiments.pipeline import compare_value
    assert compare_value('Clean KL Div', '2.100000', '2.10005')
    assert compare_value('Logit Diff Percent', '101.100', '101.1008')
    assert not compare_value('Correct Answer Greater Than Incorrect Percent', '50', '50.390625')
    assert not compare_value('Clean KL Div', '2.100000', '2.101')


def test_comparison_accepts_values_on_decimal_tolerance_boundary():
    from experiments.pipeline import compare_value
    assert compare_value('Clean KL Div', '2.1000', '2.1001')
    assert compare_value('Logit Diff Percent', '247.816', '247.815')


def test_missing_functional_rows_are_a_failure(tmp_path):
    from experiments.pipeline import compare_csv
    expected = tmp_path / 'expected.csv'
    actual = tmp_path / 'actual.csv'
    expected.write_text('task,metric,metric_mean\nioi,Clean KL Div,1.0\n')
    actual.write_text('task,metric,metric_mean\n')
    with pytest.raises(ValueError, match='identit'):
        compare_csv(expected, actual)


def test_supplement_comparison_allows_new_metadata_when_reference_is_blank(tmp_path):
    from experiments.pipeline import compare_supplement
    expected = tmp_path / 'expected.csv'
    actual = tmp_path / 'actual.csv'
    expected.write_text('comparison,n_permutations,p_permutation\nrun,,0.5\n')
    actual.write_text('comparison,n_permutations,p_permutation\nrun,184756,0.5\n')
    assert compare_supplement(expected, actual, ('comparison',))['matched']


def test_frozen_registry_selection_does_not_fall_back_to_local_results(tmp_path):
    from experiments.pipeline import registry_inputs
    with pytest.raises((FileNotFoundError, ValueError)):
        registry_inputs(tmp_path / 'missing.json')


def test_export_consumes_the_registry_from_its_discovery_run(tmp_path):
    from experiments.pipeline import action_registry
    generated = tmp_path / 'discovery/registry.json'
    generated.parent.mkdir()
    generated.write_text('{}')
    assert action_registry('export', tmp_path, tmp_path / 'bundled.json') == generated


def test_followup_stages_reject_protocol_mixing():
    from experiments.pipeline import validate_protocol
    with pytest.raises(ValueError, match='protocol mismatch'):
        validate_protocol({'protocol': 'fixed-holdout'}, 'thesis')


def test_fresh_discovery_export_requires_both_complete_evaluations(tmp_path):
    import json
    from experiments.pipeline import require_evaluation_for_export
    registry = {'purpose': 'new_thesis_discovery', 'protocol': 'thesis'}
    with pytest.raises(ValueError, match='Evaluate this discovery run'):
        require_evaluation_for_export(registry, tmp_path)
    for name in ('main', 'docstring'):
        folder = tmp_path / 'evaluation' / name
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text(json.dumps({'complete': True, 'protocol': 'thesis'}))
    require_evaluation_for_export(registry, tmp_path)
