"""Faithfulness at the published circuit size, under each matched protocol.

The main Sub-RQ 2 exhibit. One panel per (task, metric); rows are algorithms and the
x-axis is the metric's own value, so nothing is normalised away. Each algorithm carries
the three matched protocols, where the ablation used to evaluate is the one used to
discover:

    vertical offset + marker shape = the ablation, used at both discovery and evaluation

A row's horizontal spread is that algorithm's protocol sensitivity, and rows that overlap
mark an ordering the protocol can flip. Reading happens inside one panel, never across
two: the metrics carry different units, so the x-axes are deliberately independent.

Mismatched cells (discovery under one ablation, evaluation under another) are deliberately
excluded here and reported on their own, since they answer a different question: how the
evaluation protocol behaves as an instrument, rather than how the algorithms compare.
Pass --all-protocols to draw all nine cells instead, with fill giving the evaluation.

    python experiments/protocol_dotplot.py --summary <path/to/summary.csv>
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from auto_circuit.utils.plot_theme import (
    AXIS,
    DISC_SYMBOL,
    FONT_FAMILY,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    algo_colors,
    apply_theme,
    header,
)

# Row order follows ROW_METRICS in stability_plots.py, so the two chapters share a grid.
ROW_METRICS = [
    "Logit Diff Percent",
    "Clean KL Div",
    "Correct Answer Greater Than Incorrect Percent",
]
METRIC_LABEL = {
    "Logit Diff Percent": "Logit difference, % of full model",
    "Clean KL Div": "KL from the full model",
    "Correct Answer Greater Than Incorrect Percent": "Correct beats incorrect, %",
}
# Which direction is good, and how the axis has to be drawn to stay readable.
METRIC_BETTER = {
    "Logit Diff Percent": "higher",
    "Clean KL Div": "lower",
    "Correct Answer Greater Than Incorrect Percent": "higher",
}
METRIC_LOG = {"Clean KL Div"}
# The full model sits at 100% of itself; the reference line makes overshoot visible.
METRIC_REFERENCE = {"Logit Diff Percent": 100.0}

TASK_ORDER = ["Indirect Object Identification", "Greaterthan", "Docstring"]
TASK_DISPLAY = {
    "Indirect Object Identification": "IOI",
    "Greaterthan": "Greater-Than",
    "Docstring": "Docstring",
}
ALGO_ORDER = ["EP", "DiscoGP", "ACDC", "EAP", "GT", "Random"]
ABLATIONS = ["zero", "resample", "mean"]

# GT and Random are ablation-independent as edge sets, so their discovery column is an
# artefact of being reused in every panel; collapsing it stops them drawing three copies.
ANCHORS = {"GT", "Random"}
# Each hand-labelled circuit was established under one ablation. Scoring it under another
# measures protocol mismatch, not circuit quality, so those cells stay out of the figure.
GT_NATIVE_EVAL = {
    "Indirect Object Identification": "mean",
    "Greaterthan": "resample",
    "Docstring": "resample",
}

DISC_OFFSET = {"zero": 0.26, "resample": 0.0, "mean": -0.26}
ROW_PITCH = 1.0


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _fill(color: str, eval_ablation: str) -> Tuple[str, float]:
    """Evaluation ablation reads as fill weight: hollow, tinted, solid."""
    if eval_ablation == "zero":
        return "#ffffff", 1.9
    if eval_ablation == "mean":
        return _rgba(color, 0.42), 1.6
    return color, 1.2


def read_summary(path: Path) -> List[dict]:
    with path.open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["measure"] == "faithfulness"]


def points_for(
    rows: List[dict], task: str, metric: str, *, matched_only: bool
) -> Dict[str, List[dict]]:
    """One algorithm's protocol cells, keyed by algorithm and ordered for stable drawing.

    Anchors are deduplicated over the discovery axis they do not have, and GT is dropped
    outside the evaluation ablation its circuit was established under. Under
    `matched_only` the anchors keep their discovery label from the evaluation they are
    scored at, which is what puts them on the diagonal at all.
    """
    seen: set = set()
    out: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["task"] != task or r["metric"] != metric:
            continue
        algo = r["algo"]
        if algo == "GT" and r["eval_ablation"] != GT_NATIVE_EVAL.get(task):
            continue
        anchor = algo in ANCHORS
        if matched_only:
            # Anchors are discovered without an ablation, so every eval is "matched" for
            # them; the label follows the eval so they land on the diagonal.
            disc = r["eval_ablation"] if anchor else r["discovery_ablation"]
            if disc != r["eval_ablation"]:
                continue
        else:
            disc = "resample" if anchor else r["discovery_ablation"]
        key = (algo, disc, r["eval_ablation"])
        if key in seen:
            continue
        seen.add(key)
        out[algo].append(
            dict(discovery=disc, eval=r["eval_ablation"], value=float(r["value"]))
        )
    return out


def _panel_center(fig: go.Figure, row: int, col: int, n_cols: int) -> Tuple[float, float]:
    """Paper-space centre of one subplot.

    `add_annotation(row=, col=)` resolves against the subplot's own axes only for panels
    that carry traces; an empty panel silently takes the first subplot's axes instead.
    Reading the domains is unambiguous either way.
    """
    n = (row - 1) * n_cols + col
    sfx = "" if n == 1 else str(n)
    x0, x1 = fig.layout[f"xaxis{sfx}"].domain
    yaxis = fig.layout[f"yaxis{sfx}"]
    y0, y1 = yaxis.domain if yaxis.domain else fig.layout["yaxis"].domain
    return (x0 + x1) / 2, (y0 + y1) / 2


def _axis_range(values: List[float], metric: str) -> Optional[List[float]]:
    """Pad the data range so no marker is clipped in half by an axis line."""
    if not values:
        return None
    lo, hi = min(values), max(values)
    if metric in METRIC_LOG:
        lo = max(lo, 1e-3)
        import math

        span = math.log10(hi) - math.log10(lo)
        pad = max(span * 0.10, 0.08)
        return [math.log10(lo) - pad, math.log10(hi) + pad]
    span = hi - lo or abs(hi) or 1.0
    return [lo - span * 0.10, hi + span * 0.10]


def build(rows: List[dict], *, matched_only: bool = True) -> go.Figure:
    colors = algo_colors(ALGO_ORDER)
    n_rows, n_cols = len(ROW_METRICS), len(TASK_ORDER)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[TASK_DISPLAY[t] for t in TASK_ORDER] + [""] * (n_cols * (n_rows - 1)),
        horizontal_spacing=0.055,
        vertical_spacing=0.085,
        shared_yaxes=True,
    )
    fig.for_each_annotation(lambda a: a.update(font=dict(size=14, color=INK_PRIMARY)))

    y_of = {algo: -i * ROW_PITCH for i, algo in enumerate(ALGO_ORDER)}
    blank: List[Tuple[int, int]] = []

    for ri, metric in enumerate(ROW_METRICS, start=1):
        for ci, task in enumerate(TASK_ORDER, start=1):
            grouped = points_for(rows, task, metric, matched_only=matched_only)
            if not grouped:
                blank.append((ri, ci))
                fig.update_xaxes(visible=False, row=ri, col=ci)
                continue

            for algo, pts in grouped.items():
                color = colors[algo]
                for p in pts:
                    # Matched protocols need no second channel: one ablation, one mark.
                    fill, lw = (color, 1.2) if matched_only else _fill(color, p["eval"])
                    fig.add_trace(
                        go.Scatter(
                            x=[p["value"]],
                            y=[y_of[algo] + DISC_OFFSET[p["discovery"]]],
                            mode="markers",
                            marker=dict(
                                size=9,
                                symbol=DISC_SYMBOL[p["discovery"]],
                                color=fill,
                                line=dict(width=lw, color=color),
                            ),
                            showlegend=False,
                            hovertemplate=(
                                f"{algo}<br>"
                                + (
                                    f"{p['discovery']} ablation"
                                    if matched_only
                                    else f"discovery {p['discovery']} / eval {p['eval']}"
                                )
                                + "<br>%{x:.3f}<extra></extra>"
                            ),
                        ),
                        row=ri,
                        col=ci,
                    )

            values = [p["value"] for pts in grouped.values() for p in pts]
            rng = _axis_range(values, metric)
            is_log = metric in METRIC_LOG
            fig.update_xaxes(
                row=ri, col=ci,
                type="log" if is_log else "linear",
                range=rng,
                # One labelled tick per decade: Plotly's default log ticks label the
                # minors too, which reads as an unordered run of digits.
                dtick=1 if is_log else None,
                minor=dict(showgrid=False, ticks="outside", ticklen=2) if is_log else None,
                showgrid=True, gridcolor=GRID, zeroline=False,
                ticks="outside", tickcolor=AXIS, tickfont=dict(size=10, color=INK_MUTED),
            )
            ref = METRIC_REFERENCE.get(metric)
            if ref is not None and rng and rng[0] <= ref <= rng[1]:
                fig.add_vline(
                    x=ref, row=ri, col=ci,
                    line=dict(color=INK_MUTED, width=1, dash="dot"),
                )
            # Separator between algorithm bands, so the three sub-rows read as one row.
            for i in range(1, len(ALGO_ORDER)):
                fig.add_hline(
                    y=-i * ROW_PITCH + ROW_PITCH / 2, row=ri, col=ci,
                    line=dict(color=GRID, width=1),
                )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[y_of[a] for a in ALGO_ORDER],
        ticktext=ALGO_ORDER,
        range=[-(len(ALGO_ORDER) - 1) * ROW_PITCH - 0.55, 0.55],
        showgrid=False, zeroline=False, ticks="",
        tickfont=dict(size=12, color=INK_SECONDARY),
    )
    for ri, metric in enumerate(ROW_METRICS, start=1):
        arrow = "&#8594; better" if METRIC_BETTER[metric] == "higher" else "&#8592; better"
        fig.update_xaxes(
            title=dict(
                text=f"{METRIC_LABEL[metric]}  ({arrow})",
                font=dict(size=11.5, color=INK_SECONDARY),
            ),
            row=ri,
        )

    # After every axis exists, so the domains the labels are placed against are final.
    for ri, ci in blank:
        x, y = _panel_center(fig, ri, ci, n_cols)
        fig.add_annotation(
            text="not defined<br>for this task",
            xref="paper", yref="paper", x=x, y=y,
            xanchor="center", yanchor="middle",
            showarrow=False, font=dict(size=12, color=INK_MUTED),
        )

    _legend(fig, matched_only=matched_only)
    apply_theme(fig, height=1020, width=1500)
    # The legend has to clear the row-1 facet titles, which sit at the top of the plot area.
    if matched_only:
        title = "Faithfulness at the published circuit size, under each matched protocol"
        subtitle = (
            "Each algorithm carries the three ablations, each used for both discovery and "
            "evaluation. Shape and vertical offset give the ablation.<br>"
            "A wide row means the verdict is a property of the protocol; overlapping rows mean "
            "the ordering is not robust. Panels have independent x-axes and are read one at a time."
        )
    else:
        title = "Faithfulness at the published circuit size, under every protocol"
        subtitle = (
            "Each algorithm carries all nine (discovery, evaluation) ablation pairs. Shape and "
            "vertical offset give the discovery ablation, fill gives the evaluation ablation.<br>"
            "A wide row means the verdict is a property of the protocol; overlapping rows mean "
            "the ordering is not robust. Panels have independent x-axes and are read one at a time."
        )
    header(fig, title, subtitle, top=228, legend_y=1.055)
    return fig


def _legend(fig: go.Figure, *, matched_only: bool) -> None:
    """One block for the ablation shapes, and a second for the evaluation fills when the
    evaluation is free to differ from the discovery."""
    for ablation in ABLATIONS:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                name=ablation if matched_only else f"{ablation} discovery",
                legendgroup="discovery",
                marker=dict(
                    size=10, symbol=DISC_SYMBOL[ablation],
                    color=INK_SECONDARY if matched_only else "#ffffff",
                    line=dict(width=1.6, color=INK_SECONDARY),
                ),
            ),
            row=1, col=1,
        )
    if matched_only:
        return
    for ablation in ABLATIONS:
        fill, lw = _fill(INK_SECONDARY, ablation)
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                name=f"{ablation} evaluation",
                legendgroup="evaluation",
                marker=dict(
                    size=10, symbol="circle",
                    color=fill, line=dict(width=lw, color=INK_SECONDARY),
                ),
            ),
            row=1, col=1,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/faithfulness_comparison/"
            "faithfulness_seed42_2026-07-31_knockout/summary.csv"
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("results/faithfulness_comparison"))
    ap.add_argument("--stem", default="protocol_dotplot")
    ap.add_argument(
        "--all-protocols",
        action="store_true",
        help="Draw all nine (discovery, evaluation) cells instead of the matched three",
    )
    args = ap.parse_args()

    rows = read_summary(args.summary)
    fig = build(rows, matched_only=not args.all_protocols)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "html"):
        path = args.out_dir / f"{args.stem}.{ext}"
        fig.write_html(path) if ext == "html" else fig.write_image(path, scale=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
