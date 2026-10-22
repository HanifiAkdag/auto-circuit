from experiments.stability_plots import (  # noqa: E402
    _diagonal_sd,
    _joint_only,
    _panel_points,
    _struct_rows,
    fig_variance_decomposition,
)


def _row(condition, algo="EP", task="IOI", jaccard=0.5, sd=0.1):
    return {
        "task": task, "algo": algo, "discovery_ablation": "resample",
        "condition": condition, "jaccard_mean": str(jaccard), "primary_sd": str(sd),
        "realised_mean": "200", "primary_metric": "Logit Diff %",
    }


def test_joint_only_filters_by_condition():
    rows = [_row("joint"), _row("data"), _row("init")]
    assert [r["condition"] for r in _joint_only(rows)] == ["joint"]


def test_joint_only_passes_through_legacy_rows_without_the_column():
    """CSVs written before the condition column are joint by construction."""
    legacy = {k: v for k, v in _row("joint").items() if k != "condition"}
    assert _joint_only([legacy]) == [legacy]


def test_structure_function_points_do_not_mix_seed_conditions():
    structural = [
        {
            "task": "IOI", "algo": "EP", "discovery_ablation": "resample",
            "condition": condition, "selector": "top_k", "budget_label": "200",
            "budget_value": 200, "realised_mean": 200,
            "jaccard_mean": jaccard, "jaccard_min": jaccard,
            "jaccard_max": jaccard,
        }
        for condition, jaccard in [("data", 0.9), ("joint", 0.4)]
    ]
    functional = [
        {
            "task": "IOI", "algo": "EP", "discovery_ablation": "resample",
            "condition": condition, "eval_ablation": "resample",
            "measure": "faithfulness", "budget_label": "200",
            "metric": "Clean KL Div", "metric_sd": sd,
        }
        for condition, sd in [("joint", 0.2), ("data", 0.8)]
    ]

    points = _panel_points(
        _struct_rows(structural), _diagonal_sd(functional),
        "IOI", "Clean KL Div", 200, ["EP"],
    )
    assert [(point["jaccard"], point["sd"]) for point in points] == [(0.4, 0.2)]


def test_variance_decomposition_figure_has_a_trace_per_condition():
    """`primary_sd` is a raw string here, as it is before `_read`'s own coercion runs.
    `_num` treats an un-coerced value as a missing measurement (None); a raw `float()`
    would silently parse it instead, so the None check pins that `_num` is what's used."""
    rows = [_row("data", sd=0.2), _row("init", sd=0.3), _row("joint", sd=0.4)]
    fig = fig_variance_decomposition(rows)
    assert {tr.name for tr in fig.data} == {"data", "init", "joint"}
    non_empty = [tr for tr in fig.data if len(tr.y) > 0]
    assert non_empty and all(y is None for tr in non_empty for y in tr.y)
