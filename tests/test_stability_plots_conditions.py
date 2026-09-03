import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from stability_plots import _joint_only, fig_variance_decomposition  # noqa: E402


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


def test_variance_decomposition_figure_has_a_trace_per_condition():
    """`primary_sd` is a raw string here, as it is before `_read`'s own coercion runs.
    `_num` treats an un-coerced value as a missing measurement (None); a raw `float()`
    would silently parse it instead, so the None check pins that `_num` is what's used."""
    rows = [_row("data", sd=0.2), _row("init", sd=0.3), _row("joint", sd=0.4)]
    fig = fig_variance_decomposition(rows)
    assert {tr.name for tr in fig.data} == {"data", "init", "joint"}
    non_empty = [tr for tr in fig.data if len(tr.y) > 0]
    assert non_empty and all(y is None for tr in non_empty for y in tr.y)
