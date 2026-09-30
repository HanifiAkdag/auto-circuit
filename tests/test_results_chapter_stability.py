import pytest

from experiments.thesis.results_chapter_stability import select_cell


def _row(**changes):
    row = {
        "task": "Greaterthan",
        "algo": "ACDC",
        "discovery_ablation": "zero",
        "condition": "joint",
        "selector": "threshold",
        "budget_label": "7.0e-02",
        "budget_value": "0.07",
        "true_edge_count": "235",
        "n_seeds": "5",
        "n_pairs": "10",
        "realised_mean": "241",
        "realised_min": "226",
        "realised_max": "260",
        "jaccard_mean": "0.54",
        "jaccard_sd": "0.02",
        "jaccard_min": "0.50",
        "jaccard_max": "0.58",
    }
    row.update(changes)
    return row


def test_acdc_selects_one_group_threshold_and_keeps_its_attained_size():
    selected = _row()
    alternative = _row(budget_label="5.0e-02", budget_value="0.05",
                       realised_mean="270")
    result = select_cell(
        [selected], [alternative, selected],
        "Greaterthan", "ACDC", "zero", "joint", 235,
    )
    assert result["budget_label"] == "7.0e-02"
    assert result["realised_mean"] == 241.0
    assert result["realised_min"] == 226
    assert result["realised_max"] == 260
    assert result["jaccard_sd"] == 0.02


def test_selected_summary_must_match_structural_row_of_same_condition():
    selected = _row()
    other_condition = _row(condition="data", jaccard_mean="0.91")
    with pytest.raises(AssertionError):
        select_cell(
            [selected], [other_condition],
            "Greaterthan", "ACDC", "zero", "joint", 235,
        )
