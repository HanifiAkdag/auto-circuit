import pytest
from scipy.stats import spearmanr

from experiments.thesis import results_chapter_stability as exhibits
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


def _association_inputs():
    joint = exhibits._rows(exhibits.OUTPUT / "within_ablation_selected.csv")
    quality = exhibits._rows(exhibits.OUTPUT.parent / "controlled_comparison/matched_quality.csv")
    return joint, quality


def test_structure_function_uses_identical_selected_circuits_and_tied_ranks():
    assert callable(getattr(exhibits, "structure_function_records", None))
    selected, summary = exhibits.structure_function_records(*_association_inputs())
    assert len(selected) == 60 and len(summary) == 5
    expected = [-0.2517482517482518, -0.8951048951048951, 0.25874125874125875,
                -0.34265734265734266, -0.5244755244755245]
    for row, value in zip(summary, expected):
        points = [r for r in selected if (r["task"], r["metric"]) == (row["task"], row["metric"])]
        assert row["n_cells"] == len(points) == 12
        assert row["rho"] == pytest.approx(value)
        assert row["rho"] == pytest.approx(spearmanr(
            [r["jaccard_mean"] for r in points], [r["metric_sd"] for r in points]
        ).statistic)
        assert {r["algo"] for r in points} == set(exhibits.METHODS)


@pytest.mark.parametrize("change", [{"selector": "top_k"}, {"realised_mean": "739"},
                                   {"n_seeds": "4"}, {"budget_label": "0.02"}])
def test_structure_function_rejects_mismatched_selected_circuits(change):
    assert callable(getattr(exhibits, "structure_function_records", None))
    joint, quality = _association_inputs()
    quality[0].update(change)
    with pytest.raises(AssertionError):
        exhibits.structure_function_records(joint, quality)


def test_structure_function_rejects_duplicate_metric_rows():
    assert callable(getattr(exhibits, "structure_function_records", None))
    joint, quality = _association_inputs()
    with pytest.raises(AssertionError):
        exhibits.structure_function_records(joint, quality + [quality[0]])
