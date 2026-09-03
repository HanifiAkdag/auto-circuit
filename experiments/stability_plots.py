"""Figures for the Pillar-2 stability pass.

Reads `structural.csv` and `functional.csv` from a results folder and writes every figure.
Nothing here touches the score caches, a model or a GPU, so figure design can be iterated
in seconds:

    srun python experiments/stability_plots.py --from results/stability/<experiment>

Figures, and the question each answers:

  jaccard_vs_budget        do five seeds find the same circuit, and does that depend on
                           how big a circuit you ask for?
  mutual_iou_vs_budget     how much survives agreement of all five at once, not just of
                           the best pair?
  selection_frequency      is there a reproducible core with a noisy periphery, or is the
                           disagreement spread evenly?
  stability_vs_function    where circuits disagree, does behaviour disagree too?
  faithfulness/*           the functional curves the previous figure summarises.

The uncertainty band throughout is a min-to-max **range** over the seed pairs, never a
confidence interval: with five seeds the ten pairs are dependent (each seed appears in
four of them), so a standard error has no sampling interpretation.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.utils.plot_theme import (
    AXIS,
    DISC_DASH,
    DISC_SYMBOL,
    FONT_FAMILY,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
    SLOTS,
    SURFACE,
    algo_colors,
    algo_symbols,
    apply_theme,
    header,
    is_anchor,
)

ALGO_ORDER = ["EP", "DiscoGP", "ACDC", "EAP", "Random"]
ABLATION_ORDER = ["zero", "resample", "mean"]
TASK_ORDER = ["Indirect Object Identification", "Greaterthan", "Docstring"]
TASK_DISPLAY = {"Indirect Object Identification": "IOI", "Greaterthan": "Greater-Than"}

_BAND_ALPHA = 0.13

# The two-class decision: does the correct answer outscore the incorrect one. Chen et al.
# report exactly this as "Acc.", so it is the axis their non-uniqueness claim is made on.
DECISION_METRIC = "Correct Answer Greater Than Incorrect Percent"

METRIC_GLOSS: Dict[str, str] = {
    "Logit Diff Percent": "logit diff, % of full model",
    "Correct Answer Percent": "correct answer top-1, %",
    DECISION_METRIC: "correct beats incorrect, %",
    "Clean KL Div": "KL from the full model",
    "Answer Pair KL Div": "answer-pair KL",
    "Prob Diff": "probability difference",
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _read(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key in ("metric_values",):
                continue
            try:
                row[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except (ValueError, TypeError):
                pass
    return rows


def _joint_only(rows: List[dict]) -> List[dict]:
    """The existing figures describe total run-to-run variance, which is the joint
    condition. A CSV written before the column existed is joint by construction."""
    return [r for r in rows if r.get("condition", "joint") == "joint"]


def _tasks(rows: List[dict]) -> List[str]:
    present = {r["task"] for r in rows}
    return [t for t in TASK_ORDER if t in present] + sorted(present - set(TASK_ORDER))


def _algos(rows: List[dict]) -> List[str]:
    present = {r["algo"] for r in rows}
    return [a for a in ALGO_ORDER if a in present] + sorted(present - set(ALGO_ORDER))


def _series(rows: List[dict], task: str, algo: str, ablation: str) -> List[dict]:
    """Rows for one line, ordered by circuit size. Anchors appear in every ablation panel."""
    want = "anchor" if algo == "Random" else ablation
    picked = [
        r for r in rows if r["task"] == task and r["algo"] == algo
        and r["discovery_ablation"] == want
    ]
    return sorted(picked, key=lambda r: r["realised_mean"])


def _band(fig, xs, lo, hi, color: str, row: int, col: int) -> None:
    """Draw a min-max range band as a filled polygon (forward along `hi`, back along
    reversed `lo`) — the seed-pair range used throughout, never a confidence interval
    (see module docstring)."""
    fig.add_trace(
        go.Scatter(
            x=list(xs) + list(xs)[::-1],
            y=list(hi) + list(lo)[::-1],
            mode="lines",
            fill="toself",
            fillcolor=_rgba(color, _BAND_ALPHA),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _facet_titles(fig, size: int = 14) -> None:
    """Subplot titles are annotations; only the column headers exist, so restyle them all."""
    fig.for_each_annotation(lambda a: a.update(font=dict(size=size, color=INK_PRIMARY)))


def _grid(tasks: List[str], ablations: List[str]):
    """Shared subplot scaffold for the ablation-rows x task-columns figures: axes are
    shared across panels so they read on one scale, and only the first row gets task
    titles (later rows are labelled by `_row_labels` instead of repeating them)."""
    fig = make_subplots(
        rows=len(ablations),
        cols=len(tasks),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[TASK_DISPLAY.get(t, t) for t in tasks] + [""]
        * (len(tasks) * (len(ablations) - 1)),
        horizontal_spacing=0.045,
        vertical_spacing=0.075,
    )
    return fig


def _row_labels(fig, ablations: List[str], n_cols: int, suffix: str = " discovery") -> None:
    """Name each row of the grid on its right edge, so the y-axis stays for the metric."""
    n_rows = len(ablations)
    for i, ablation in enumerate(ablations):
        fig.add_annotation(
            text=f"<b>{ablation}</b>{suffix}",
            xref="paper",
            yref=f"y{(i * n_cols) + 1} domain" if i else "y domain",
            x=1.005,
            xanchor="left",
            y=0.5,
            showarrow=False,
            textangle=90,
            font=dict(size=12, color=INK_SECONDARY, family=FONT_FAMILY),
        )


# ── Fig 1: pairwise Jaccard against circuit size ────────────────────────────────────


def fig_jaccard_vs_budget(rows: List[dict]) -> go.Figure:
    """Pairwise Jaccard vs. circuit size, faceted by task (columns) and discovery
    ablation (rows). Answers: do five seeds land on the same circuit, and does that
    agreement hold as the requested circuit grows or shrinks? Bands are the min-max
    range over the ten seed pairs; Random (dashed) is the chance floor."""
    tasks, algos = _tasks(rows), _algos(rows)
    colors = algo_colors(algos)
    symbols = algo_symbols(algos)
    fig = _grid(tasks, ABLATION_ORDER)

    for r_i, ablation in enumerate(ABLATION_ORDER, start=1):
        for task_i, task in enumerate(tasks, start=1):
            for algo in algos:
                series = _series(rows, task, algo, ablation)
                if not series:
                    continue
                xs = [s["realised_mean"] for s in series]
                mean = [s["jaccard_mean"] for s in series]
                anchor = is_anchor(algo)
                if not anchor:
                    _band(
                        fig, xs, [s["jaccard_min"] for s in series],
                        [s["jaccard_max"] for s in series], colors[algo], r_i, task_i,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=mean,
                        name=f"{algo} (chance floor)" if anchor else algo,
                        legendgroup=algo,
                        showlegend=(r_i == 1 and task_i == 1),
                        mode="lines+markers",
                        line=dict(
                            color=colors[algo],
                            width=1.5 if anchor else 2,
                            dash="dash" if anchor else "solid",
                        ),
                        marker=dict(
                            size=5 if anchor else 7,
                            symbol=symbols[algo],
                            line=dict(width=1.4, color="#ffffff"),
                        ),
                        hovertemplate=f"{algo}<br>%{{x:.0f}} edges<br>J=%{{y:.3f}}<extra></extra>",
                    ),
                    row=r_i,
                    col=task_i,
                )

    apply_theme(fig, width=1080, height=960, margin_left=96)
    header(
        fig,
        "Circuit agreement across five discovery seeds",
        "Jaccard overlap of the top-k edge sets, averaged over the ten seed pairs. "
        "Band spans those pairs.",
        top=170,
        right=120,
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", dtick=0.25)
    fig.update_xaxes(type="log", dtick=1, exponentformat="power", minor_ticks="outside")
    fig.update_yaxes(title=dict(text="Pairwise Jaccard"), col=1)
    for col, _ in enumerate(tasks, start=1):
        fig.update_xaxes(title=dict(text="Edges in circuit"), row=len(ABLATION_ORDER), col=col)
    _facet_titles(fig)
    _row_labels(fig, ABLATION_ORDER, len(tasks))
    return fig


def fig_mutual_iou_vs_budget(rows: List[dict]) -> go.Figure:
    """Mutual IoU (|intersection of all five seeds| / |union|) vs. circuit size, same
    task x ablation grid as `fig_jaccard_vs_budget`. Answers: how much survives
    agreement of every seed at once, a stricter bar than the best or average pairwise
    overlap."""
    tasks, algos = _tasks(rows), _algos(rows)
    colors, symbols = algo_colors(algos), algo_symbols(algos)
    fig = _grid(tasks, ABLATION_ORDER)

    for r_i, ablation in enumerate(ABLATION_ORDER, start=1):
        for task_i, task in enumerate(tasks, start=1):
            for algo in algos:
                series = _series(rows, task, algo, ablation)
                if not series:
                    continue
                anchor = is_anchor(algo)
                fig.add_trace(
                    go.Scatter(
                        x=[s["realised_mean"] for s in series],
                        y=[s["mutual_iou"] for s in series],
                        name=algo,
                        legendgroup=algo,
                        showlegend=(r_i == 1 and task_i == 1),
                        mode="lines+markers",
                        line=dict(
                            color=colors[algo],
                            width=1.5 if anchor else 2,
                            dash="dash" if anchor else "solid",
                        ),
                        marker=dict(
                            size=5 if anchor else 7,
                            symbol=symbols[algo],
                            line=dict(width=1.4, color="#ffffff"),
                        ),
                        customdata=[[s["n_intersect"], s["n_union"]] for s in series],
                        hovertemplate=f"{algo}<br>%{{x:.0f}} edges<br>"
                        "IoU=%{y:.3f}<br>%{customdata[0]} / %{customdata[1]} edges"
                        "<extra></extra>",
                    ),
                    row=r_i,
                    col=task_i,
                )

    apply_theme(fig, width=1080, height=960, margin_left=96)
    header(
        fig,
        "Edges recovered by all five seeds at once",
        "Size of the intersection of the five top-k edge sets, over the size of their union.",
        top=170,
        right=120,
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", dtick=0.25)
    fig.update_xaxes(type="log", dtick=1, exponentformat="power", minor_ticks="outside")
    fig.update_yaxes(title=dict(text="Mutual IoU"), col=1)
    for col, _ in enumerate(tasks, start=1):
        fig.update_xaxes(title=dict(text="Edges in circuit"), row=len(ABLATION_ORDER), col=col)
    _facet_titles(fig)
    _row_labels(fig, ABLATION_ORDER, len(tasks))
    return fig


def fig_jaccard_compact(rows: List[dict]) -> go.Figure:
    """One row of panels, every (algo × ablation) as a line, no bands. The overview."""
    tasks, algos = _tasks(rows), _algos(rows)
    colors, symbols = algo_colors(algos), algo_symbols(algos)
    fig = make_subplots(
        rows=1,
        cols=len(tasks),
        shared_yaxes=True,
        subplot_titles=[TASK_DISPLAY.get(t, t) for t in tasks],
        horizontal_spacing=0.04,
    )
    for task_i, task in enumerate(tasks, start=1):
        for algo in algos:
            for ablation in ABLATION_ORDER:
                if algo == "Random" and ablation != ABLATION_ORDER[0]:
                    continue
                series = _series(rows, task, algo, ablation)
                if not series:
                    continue
                anchor = is_anchor(algo)
                label = algo if anchor else f"{algo} | {ablation}"
                fig.add_trace(
                    go.Scatter(
                        x=[s["realised_mean"] for s in series],
                        y=[s["jaccard_mean"] for s in series],
                        name=label,
                        legendgroup=label,
                        showlegend=(task_i == 1),
                        mode="lines+markers",
                        line=dict(
                            color=colors[algo],
                            width=1.5 if anchor else 2,
                            dash=DISC_DASH["anchor" if anchor else ablation],
                        ),
                        marker=dict(
                            size=5,
                            symbol=DISC_SYMBOL["anchor" if anchor else ablation],
                            line=dict(width=1.2, color="#ffffff"),
                        ),
                    ),
                    row=1,
                    col=task_i,
                )
    apply_theme(fig, width=1300, height=560, margin_left=100)
    header(
        fig,
        "Circuit agreement across seeds, every discovery protocol",
        "Mean pairwise Jaccard of the top-k edge sets. Colour: algorithm. Dash: discovery "
        "ablation.",
        top=130,
        right=250,
    )
    fig.update_layout(
        legend=dict(orientation="v", x=1.01, y=1.0, xanchor="left", yanchor="top")
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", dtick=0.25)
    fig.update_xaxes(
        type="log", dtick=1, exponentformat="power", title=dict(text="Edges in circuit")
    )
    fig.update_yaxes(title=dict(text="Pairwise Jaccard"), col=1)
    _facet_titles(fig)
    return fig


# ── Fig 4: what the union of five circuits is made of ───────────────────────────────


def fig_selection_frequency(summary: List[dict]) -> go.Figure:
    """Stacked horizontal bars: for each (algo, discovery ablation), what share of the
    union of all five seeds' circuits was found by exactly 1..5 of them. Answers: is
    disagreement a reproducible core with a noisy periphery (mass in "5 of 5") or spread
    evenly across seeds (mass in "1 of 5")? Cut at each task's ground-truth circuit size."""
    tasks, algos = _tasks(summary), _algos(summary)
    # Name each panel with the circuit size it is cut at: the task's ground-truth count.
    sizes = {
        t: next((r["true_edge_count"] for r in summary if r["task"] == t), None)
        for t in tasks
    }
    fig = make_subplots(
        rows=1,
        cols=len(tasks),
        shared_yaxes=True,
        subplot_titles=[
            f"{TASK_DISPLAY.get(t, t)} · {sizes[t]:,.0f}-edge circuits"
            if sizes[t] else TASK_DISPLAY.get(t, t)
            for t in tasks
        ],
        # Wide enough that the union size printed past each bar clears the next panel.
        horizontal_spacing=0.075,
    )
    # Restyle the panel titles now, before the union-size labels are added: the helper
    # sweeps every annotation, and those labels must stay small and muted.
    _facet_titles(fig)
    labels: List[str] = []
    for algo in algos:
        for ablation in (["anchor"] if algo == "Random" else ABLATION_ORDER):
            labels.append(f"{algo} · {ablation}")
    shades = {5: SEQUENTIAL_BLUE[7], 4: SEQUENTIAL_BLUE[6], 3: SEQUENTIAL_BLUE[4],
              2: SEQUENTIAL_BLUE[2], 1: SEQUENTIAL_BLUE[1]}

    for task_i, task in enumerate(tasks, start=1):
        by_label = {
            f"{r['algo']} · {r['discovery_ablation']}": r
            for r in summary
            if r["task"] == task
        }
        for m in (5, 4, 3, 2, 1):
            xs, ys, text = [], [], []
            for label in labels:
                row = by_label.get(label)
                union = row["n_union"] if row else 0
                share = (row[f"freq_{m}"] / union) if row and union else 0.0
                xs.append(share)
                ys.append(label)
                text.append(f"{share:.0%}" if m == 5 and share > 0.08 else "")
            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    orientation="h",
                    name=f"{m} of 5",
                    legendgroup=f"{m}",
                    showlegend=(task_i == 1),
                    marker=dict(color=shades[m], line=dict(width=0)),
                    text=text,
                    textposition="inside",
                    insidetextfont=dict(color="#ffffff", size=11),
                    hovertemplate=f"%{{y}}<br>found by {m} of 5 seeds: %{{x:.1%}}"
                    "<extra></extra>",
                ),
                row=1,
                col=task_i,
            )
        # The union size is what every bar is normalised by, so it is printed per row
        # and named by a column header rather than left as a bare number.
        suffix = f"{task_i if task_i > 1 else ''}"
        fig.add_annotation(
            x=1.005, xref=f"x{suffix} domain", xanchor="left",
            y=1.02, yref=f"y{suffix} domain", yanchor="bottom",
            text="edges in<br>union", showarrow=False, align="left",
            font=dict(size=9.5, color=INK_MUTED, family=FONT_FAMILY),
        )
        for label in labels:
            row = by_label.get(label)
            if row:
                fig.add_annotation(
                    x=1.005, y=label, xref=f"x{suffix} domain",
                    yref=f"y{suffix}", xanchor="left",
                    text=f"{row['n_union']:,.0f}", showarrow=False,
                    font=dict(size=10.5, color=INK_MUTED, family=FONT_FAMILY),
                )

    apply_theme(fig, width=1180, height=680, margin_left=150)
    header(
        fig,
        "What the union of five seed circuits is made of",
        "Share of the union selected by exactly n of the five seeds. Union: every edge any "
        "seed selected. ACDC is matched by threshold, so its size is approximate.",
        top=200,
        right=60,
        legend_y=1.13,
    )
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title=dict(text="Share of union"))
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels)
    return fig


# ── Fig 2: does structural disagreement show up in behaviour? ───────────────────────


def _num(value) -> Optional[float]:
    """Numbers survive `_read` as int or float; anything else is a missing measurement."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if value != value else float(value)


def _ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for t in range(i, j + 1):
            out[order[t]] = shared
        i = j + 1
    return out


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rank correlation, hand-rolled (Pearson over `_ranks`, which averages
    tied ranks) to avoid a scipy dependency for this one statistic. None below 3 points
    or when either axis has zero variance."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def _diagonal_sd(functional: List[dict]) -> Dict[tuple, float]:
    """Across-seed sd keyed by cell and metric, each cell read at its own eval ablation.

    Faithfulness only. The stability-vs-function plane is an argument about how much of a
    circuit's behaviour survives reseeding, and mixing in the complement's behaviour would
    put two different questions on one axis.
    """
    out: Dict[tuple, float] = {}
    for r in functional:
        if r.get("measure", "faithfulness") != "faithfulness":
            continue
        if r["eval_ablation"] != r["discovery_ablation"]:
            continue
        sd = _num(r.get("metric_sd"))
        if sd is None:
            continue
        key = (r["task"], r["algo"], r["discovery_ablation"], r["budget_label"], r["metric"])
        out[key] = sd
    return out




# One metric per row, the same three for every task, so a row can be read across tasks.
ROW_METRICS = ["Logit Diff Percent", "Clean KL Div", DECISION_METRIC]


def _struct_rows(structural: List[dict]) -> Dict[tuple, List[dict]]:
    """Group structural.csv rows by (task, algo, discovery_ablation), each group sorted
    by circuit size — the per-cell budget grid `_pick_budget` searches over."""
    out: Dict[tuple, List[dict]] = {}
    for r in structural:
        out.setdefault((r["task"], r["algo"], r["discovery_ablation"]), []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r["realised_mean"])
    return out


_MATCH_TOLERANCE = 4.0


def _pick_budget(rows: List[dict], target: float) -> Optional[dict]:
    """Top-k methods must hit the budget exactly; ACDC is matched on realised size.

    ACDC scores are the tau an edge was pruned at, so its grid is ~20 tau values whose
    realised sizes fall where they fall. Asking for its nearest realised size is the same
    rule `select_at_budget` applies during the analysis pass. Nearest is refused past a
    factor of four, where the match would be to a circuit of a different order of size.
    """
    if not rows:
        return None
    if all(r["selector"] == "top_k" for r in rows):
        return next((r for r in rows if r["budget_value"] == target), None)
    best = min(rows, key=lambda r: abs(r["realised_mean"] - target))
    ratio = best["realised_mean"] / target if target else 0.0
    return best if 1 / _MATCH_TOLERANCE <= ratio <= _MATCH_TOLERANCE else None


def _panel_points(
    struct: Dict[tuple, List[dict]],
    sd_by: Dict[tuple, float],
    task: str,
    metric: str,
    target: float,
    algos: List[str],
) -> List[dict]:
    """One panel's (structural agreement, functional spread) points for `task`/`metric`:
    for each algo x discovery-ablation, pick the row nearest `target` edges (usually the
    task's published circuit size, via `_pick_budget`) and pair its Jaccard range with the
    across-seed sd of `metric` at that same cell. Skips combinations with no budget match
    or no sd on record."""
    points = []
    for algo in algos:
        for ablation in ABLATION_ORDER:
            rows = struct.get((task, algo, ablation))
            row = _pick_budget(rows, target) if rows else None
            if row is None:
                continue
            sd = sd_by.get((task, algo, ablation, row["budget_label"], metric))
            if sd is None:
                continue
            points.append(
                dict(
                    algo=algo,
                    ablation=ablation,
                    jaccard=row["jaccard_mean"],
                    lo=row["jaccard_mean"] - row["jaccard_min"],
                    hi=row["jaccard_max"] - row["jaccard_mean"],
                    sd=sd,
                    size=row["realised_mean"],
                )
            )
    return points


def _published_sizes(summary: List[dict], tasks: List[str]) -> Dict[str, float]:
    """Each task's ground-truth circuit size, read off summary.csv — the `target` budget
    `_panel_points` matches against, so structure and function are compared at the size
    the literature actually reports rather than an arbitrary edge count."""
    return {
        t: next((r["true_edge_count"] for r in summary if r["task"] == t), 0) for t in tasks
    }


def _panel_box(fig, row: int, col: int, n_cols: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Paper-space extent of one subplot.

    `add_annotation(row=, col=)` resolves against the subplot's own axes only for panels
    that carry traces, so an empty panel silently takes the first axis instead. Reading the
    domains is unambiguous either way.
    """
    n = (row - 1) * n_cols + col
    suffix = "" if n == 1 else str(n)
    return (
        tuple(fig.layout[f"xaxis{suffix}"].domain),
        tuple(fig.layout[f"yaxis{suffix}"].domain),
    )


def _at_panel(fig, row, col, n_cols, fx: float, fy: float) -> Tuple[float, float]:
    (x0, x1), (y0, y1) = _panel_box(fig, row, col, n_cols)
    return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)


def _stability_grid(tasks: List[str]):
    """Metric rows x task columns: task names head the columns, metrics label the rows."""
    titles = [TASK_DISPLAY.get(t, t) for t in tasks]
    titles += [""] * (len(tasks) * (len(ROW_METRICS) - 1))
    fig = make_subplots(
        rows=len(ROW_METRICS),
        cols=len(tasks),
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.09,
    )
    # Restyle the panel titles before any other annotation exists: the helper sweeps them all.
    _facet_titles(fig)
    return fig


def _metric_axes(fig, x_title: str = "Pairwise Jaccard", x_y: float = -0.055) -> None:
    """Name the metric on its row's y-axis, and the shared x once under the whole grid.

    Every panel shares one x quantity and one kind of y, so per-panel titles would repeat
    the same two strings nine times. The metric belongs on the axis it scales rather than
    in a separate strip down the edge.
    """
    for i, metric in enumerate(ROW_METRICS, start=1):
        fig.update_yaxes(
            title=dict(text=f"sd of {METRIC_GLOSS.get(metric, metric)}"), row=i, col=1
        )
    fig.add_annotation(
        text=x_title,
        xref="paper", yref="paper", x=0.5, y=x_y,
        xanchor="center", yanchor="top", showarrow=False,
        font=dict(size=13, color=INK_SECONDARY, family=FONT_FAMILY),
    )


def _ablation_key(fig) -> None:
    """Colour carries the algorithm, so the symbol key needs its own neutral legend block."""
    for ablation in ABLATION_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{ablation} discovery",
                legendgroup="ablation",
                marker=dict(
                    color=INK_MUTED,
                    size=10,
                    symbol=DISC_SYMBOL[ablation],
                    line=dict(width=1.4, color="#ffffff"),
                ),
            ),
            row=1,
            col=1,
        )


def fig_stability_vs_function(
    summary: List[dict], structural: List[dict], functional: List[dict]
) -> Optional[go.Figure]:
    """Structural against functional instability at the published circuit size.

    y is the spread in the metric's own units, not its coefficient of variation: a CV
    divides by a mean that several cells drive to zero (ACDC/IOI/mean means 4.15%), which
    manufactures enormous instability out of a circuit that is merely bad. The axis is
    linear because an sd of exactly zero is a real and interesting value here.
    """
    if not structural or not functional:
        return None
    tasks = _tasks(structural)
    algos = [a for a in _algos(structural) if a != "Random"]
    colors = algo_colors(_algos(structural))
    struct, sd_by = _struct_rows(structural), _diagonal_sd(functional)
    published = _published_sizes(summary, tasks)

    fig = _stability_grid(tasks)
    legend_seen: set = set()
    for row_i, metric in enumerate(ROW_METRICS, start=1):
        for col_i, task in enumerate(tasks, start=1):
            points = _panel_points(struct, sd_by, task, metric, published[task], algos)
            if not points:
                ax, ay = _at_panel(fig, row_i, col_i, len(tasks), 0.5, 0.5)
                fig.add_annotation(
                    text="not measured<br>for this task",
                    xref="paper", yref="paper", x=ax, y=ay,
                    showarrow=False, align="center",
                    font=dict(size=11.5, color=INK_MUTED, family=FONT_FAMILY),
                )
                fig.update_xaxes(showgrid=False, showticklabels=False, row=row_i, col=col_i)
                fig.update_yaxes(showgrid=False, showticklabels=False, row=row_i, col=col_i)
                continue

            for p in points:
                fig.add_trace(
                    go.Scatter(
                        x=[p["jaccard"]],
                        y=[p["sd"]],
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=[p["hi"]],
                            arrayminus=[p["lo"]],
                            color=_rgba(colors[p["algo"]], 0.4),
                            thickness=1,
                            width=0,
                        ),
                        mode="markers",
                        name=p["algo"],
                        legendgroup=p["algo"],
                        showlegend=p["algo"] not in legend_seen,
                        marker=dict(
                            color=colors[p["algo"]],
                            size=11,
                            symbol=DISC_SYMBOL[p["ablation"]],
                            line=dict(width=1.6, color="#ffffff"),
                        ),
                        customdata=[[p["size"]]],
                        hovertemplate=f"{p['algo']} · {p['ablation']}<br>mean J=%{{x:.3f}}"
                        "<br>sd=%{y:.3g}<br>%{customdata[0]:,.0f} edges<extra></extra>",
                    ),
                    row=row_i,
                    col=col_i,
                )
                legend_seen.add(p["algo"])

            rho = _spearman([p["jaccard"] for p in points], [p["sd"] for p in points])
            if rho is not None:
                ax, ay = _at_panel(fig, row_i, col_i, len(tasks), 0.97, 0.99)
                fig.add_annotation(
                    text=f"ρ = {rho:+.2f}",
                    xref="paper", yref="paper", x=ax, y=ay,
                    xanchor="right", yanchor="top", showarrow=False,
                    font=dict(size=11.5, color=INK_SECONDARY, family=FONT_FAMILY),
                )
            fig.update_yaxes(rangemode="tozero", row=row_i, col=col_i)

    _ablation_key(fig)
    apply_theme(fig, width=1180, height=1120, margin_left=115)
    header(
        fig,
        "Structural instability against functional instability",
        "One marker per algorithm and discovery ablation, at the published circuit size, "
        "faithfulness under the matched protocol.<br>"
        "Rows are metrics, columns are tasks; each panel carries its Spearman ρ. "
        "Bar: min-max Jaccard over the ten seed pairs.",
        top=215,
        right=40,
        legend_y=1.035,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    _metric_axes(fig)
    fig.update_layout(margin=dict(b=95))
    return fig


# ── Fig 2b: the same plane, with the circuit size on a slider ───────────────────────


def _slider_budgets(
    functional: List[dict], tasks: List[str], published: Dict[str, float]
) -> List[float]:
    """Edge counts the functional pass measured, minus each task's own published size.

    The published sizes differ per task, so as global steps they would empty out the other
    two columns. They get one shared step of their own instead. The unpruned model is
    dropped for the same reason, and it carries no information here: every method is the
    same circuit there.
    """
    # functional.csv carries only `budget_label`; for a top-k row that label is the count.
    per_task: Dict[str, set] = defaultdict(set)
    for r in functional:
        if r.get("measure", "faithfulness") != "faithfulness":
            continue
        count = _num(r["budget_label"]) if r["selector"] == "top_k" else None
        if count is not None and r["task"] in tasks:
            per_task[r["task"]].add(count)
    full = {max(v) for v in per_task.values() if v}
    counts = set().union(*per_task.values()) if per_task else set()
    return sorted(counts - full - set(published.values()))


def fig_stability_vs_function_slider(
    summary: List[dict], structural: List[dict], functional: List[dict]
) -> Optional[go.Figure]:
    """The same plane, swept over circuit size.

    A single budget is one slice of a moving picture: pairwise Jaccard rises with k for
    every method, so any ordering read off one panel has to survive the sweep to mean
    anything.
    """
    if not structural or not functional:
        return None
    tasks = _tasks(structural)
    algos = [a for a in _algos(structural) if a != "Random"]
    colors = algo_colors(_algos(structural))
    struct, sd_by = _struct_rows(structural), _diagonal_sd(functional)
    published = _published_sizes(summary, tasks)

    budgets: List[Tuple[str, Dict[str, float]]] = [
        ("published", {t: published[t] for t in tasks})
    ]
    for count in _slider_budgets(functional, tasks, published):
        budgets.append((f"{count:,.0f}", {t: count for t in tasks}))

    # One trace per (row, col, algo, ablation) plus one Random guide per panel, all built
    # once and then restyled by the slider, so a step is a data swap and not a redraw.
    slots: List[Tuple[int, int, str, str, Optional[str]]] = []
    for row_i, metric in enumerate(ROW_METRICS, start=1):
        for col_i, task in enumerate(tasks, start=1):
            for algo in algos:
                for ablation in ABLATION_ORDER:
                    slots.append((row_i, col_i, task, metric, f"{algo}|{ablation}"))

    frames: Dict[int, List[dict]] = {}
    ceilings: Dict[Tuple[int, int], float] = {}
    for step, (_, targets) in enumerate(budgets):
        payload = []
        for row_i, col_i, task, metric, who in slots:
            algo, ablation = who.split("|")
            found = [
                p for p in _panel_points(
                    struct, sd_by, task, metric, targets[task], [algo]
                ) if p["ablation"] == ablation
            ]
            if not found:
                payload.append(dict(x=[], y=[], lo=[], hi=[], size=[]))
                continue
            p = found[0]
            payload.append(
                dict(x=[p["jaccard"]], y=[p["sd"]], lo=[p["lo"]], hi=[p["hi"]],
                     size=[p["size"]])
            )
            key = (row_i, col_i)
            ceilings[key] = max(ceilings.get(key, 0.0), p["sd"])
        frames[step] = payload

    fig = _stability_grid(tasks)
    legend_seen: set = set()
    initial = frames[0]
    for slot, data in zip(slots, initial):
        row_i, col_i, task, metric, who = slot
        algo, ablation = who.split("|")
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                error_x=dict(
                    type="data", symmetric=False,
                    array=data.get("hi", []), arrayminus=data.get("lo", []),
                    color=_rgba(colors[algo], 0.4), thickness=1, width=0,
                ),
                mode="markers",
                name=algo,
                legendgroup=algo,
                showlegend=algo not in legend_seen,
                marker=dict(
                    color=colors[algo], size=11, symbol=DISC_SYMBOL[ablation],
                    line=dict(width=1.6, color="#ffffff"),
                ),
                customdata=[[s] for s in data.get("size", [])],
                hovertemplate=f"{algo} · {ablation}<br>mean J=%{{x:.3f}}<br>sd=%{{y:.3g}}"
                "<br>%{customdata[0]:,.0f} edges<extra></extra>",
            ),
            row=row_i, col=col_i,
        )
        legend_seen.add(algo)

    n_data = len(slots)
    steps = []
    for step, (label, _) in enumerate(budgets):
        payload = frames[step]
        steps.append(
            dict(
                method="restyle",
                label=label,
                args=[
                    {
                        "x": [d["x"] for d in payload],
                        "y": [d["y"] for d in payload],
                        "error_x.array": [d.get("hi", []) for d in payload],
                        "error_x.arrayminus": [d.get("lo", []) for d in payload],
                        "customdata": [
                            [[v] for v in d.get("size", [])] for d in payload
                        ],
                    },
                    list(range(n_data)),
                ],
            )
        )

    _ablation_key(fig)
    apply_theme(fig, width=1180, height=1180, margin_left=115)
    header(
        fig,
        "Structural instability against functional instability, by circuit size",
        "One marker per algorithm and discovery ablation, faithfulness under the matched "
        "protocol. Drag the slider to change the edge budget.<br>"
        "Rows are metrics, columns are tasks. "
        "Bar: min-max Jaccard over the ten seed pairs.",
        top=225,
        right=40,
        legend_y=1.03,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    _metric_axes(fig, x_y=-0.045)
    # Ranges are held across steps; autoscaling per step would make the sweep unreadable.
    for (row_i, col_i), top in ceilings.items():
        fig.update_yaxes(range=[0, top * 1.08], row=row_i, col=col_i)
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                steps=steps,
                x=0.08,
                len=0.84,
                y=-0.105,
                pad=dict(t=8, b=8),
                currentvalue=dict(
                    prefix="circuit size: ",
                    font=dict(size=13, color=INK_PRIMARY, family=FONT_FAMILY),
                ),
                font=dict(size=11, color=INK_MUTED, family=FONT_FAMILY),
                tickcolor=AXIS,
                bgcolor=GRID,
                activebgcolor=INK_SECONDARY,
                bordercolor=SURFACE,
            )
        ],
        margin=dict(b=150),
    )
    return fig





# ── Fig 5: the functional curves behind Fig 2 ───────────────────────────────────────


_MEASURE_GLOSS = {
    "faithfulness": "circuit kept, complement ablated",
    "completeness": "circuit ablated, complement kept",
}


def fig_measure_grid(
    functional: List[dict], task: str, metric: str, measure: str
) -> Optional[go.Figure]:
    """One task and metric over the whole protocol square.

    Rows are the ablation the circuit was discovered under, columns the ablation it is
    scored under, so the diagonal is the matched protocol and everything off it is a
    protocol mismatch. Folding both into one figure is what keeps nine combinations from
    becoming nine files.
    """
    rows = [
        r for r in functional
        if r["task"] == task and r["metric"] == metric
        and r.get("measure", "faithfulness") == measure
    ]
    if not rows:
        return None
    algos = _algos(rows)
    colors, symbols = algo_colors(algos), algo_symbols(algos)
    n_cols = len(ABLATION_ORDER)
    fig = make_subplots(
        rows=len(ABLATION_ORDER),
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"{a} eval" for a in ABLATION_ORDER]
        + [""] * (n_cols * (len(ABLATION_ORDER) - 1)),
        horizontal_spacing=0.045,
        vertical_spacing=0.075,
    )
    for r_i, discovery in enumerate(ABLATION_ORDER, start=1):
        for c_i, evaluation in enumerate(ABLATION_ORDER, start=1):
            for algo in algos:
                want = "anchor" if algo == "Random" else discovery
                series = sorted(
                    (
                        r for r in rows
                        if r["algo"] == algo and r["discovery_ablation"] == want
                        and r["eval_ablation"] == evaluation
                    ),
                    key=lambda r: r["realised_mean"],
                )
                if not series:
                    continue
                xs = [s["realised_mean"] for s in series]
                anchor = is_anchor(algo)
                if not anchor:
                    _band(
                        fig, xs, [s["metric_min"] for s in series],
                        [s["metric_max"] for s in series], colors[algo], r_i, c_i,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=[s["metric_mean"] for s in series],
                        name=algo,
                        legendgroup=algo,
                        showlegend=(r_i == 1 and c_i == 1),
                        mode="lines+markers",
                        line=dict(
                            color=colors[algo],
                            width=1.5 if anchor else 2,
                            dash="dash" if anchor else "solid",
                        ),
                        marker=dict(
                            size=5 if anchor else 7,
                            symbol=symbols[algo],
                            line=dict(width=1.4, color="#ffffff"),
                        ),
                    ),
                    row=r_i,
                    col=c_i,
                )
    apply_theme(fig, width=1180, height=980, margin_left=110)
    header(
        fig,
        f"{TASK_DISPLAY.get(task, task)} · {metric} · {measure}",
        f"Line: mean over the five seeds. Band: min-max. {_MEASURE_GLOSS[measure]}. "
        "Boxed panels are the matched protocol,<br>where the circuit is scored under the "
        "ablation it was discovered with.",
        top=185,
        right=40,
    )
    fig.update_xaxes(
        type="log", dtick=1, exponentformat="power", title=dict(text="Edges in circuit"),
        row=len(ABLATION_ORDER),
    )
    fig.update_yaxes(title=dict(text=metric), col=1)
    _row_labels(fig, ABLATION_ORDER, n_cols)
    _facet_titles(fig)
    _ring_diagonal(fig, n_cols)
    return fig


def _ring_diagonal(fig, n_cols: int) -> None:
    """Outline the panels where the eval ablation matches the discovery ablation."""
    for i in range(n_cols):
        n = (i * n_cols) + i + 1
        suffix = "" if n == 1 else str(n)
        fig.add_shape(
            type="rect",
            xref=f"x{suffix} domain",
            yref=f"y{suffix} domain",
            x0=0, x1=1, y0=0, y1=1,
            line=dict(color=AXIS, width=1.4),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        )


def fig_variance_decomposition(summary: List[dict]) -> go.Figure:
    """Spread of the task's primary metric under each condition, grouped by algorithm.

    Read together the three bars say how much of an algorithm's run-to-run variance is the
    prompt draw and how much is the mask init. Bars, not a stacked share: the components
    are estimated separately and need not sum to the joint value. Faceted by discovery
    ablation like the other grid figures, since `summary` holds one row per (task, algo,
    ablation, condition) and three conditions at one x position would overplot otherwise.
    Random has no ablation of its own (`_series`'s "anchor" convention), so its single
    value is repeated in every ablation row.
    """
    conditions = ["joint", "data", "init"]
    condition_colors = dict(zip(conditions, SLOTS))
    tasks, algos = _tasks(summary), _algos(summary)
    fig = _grid(tasks, ABLATION_ORDER)

    for r_i, ablation in enumerate(ABLATION_ORDER, start=1):
        for task_i, task in enumerate(tasks, start=1):
            for condition in conditions:
                by_algo = {}
                for r in summary:
                    want = "anchor" if r["algo"] == "Random" else ablation
                    if (
                        r["task"] == task
                        and r.get("condition", "joint") == condition
                        and r["discovery_ablation"] == want
                    ):
                        by_algo[r["algo"]] = r
                present = [a for a in algos if a in by_algo]
                fig.add_trace(
                    go.Bar(
                        name=condition,
                        x=present,
                        y=[_num(by_algo[a].get("primary_sd")) for a in present],
                        legendgroup=condition,
                        showlegend=(r_i == 1 and task_i == 1),
                        marker=dict(color=condition_colors[condition]),
                    ),
                    row=r_i,
                    col=task_i,
                )

    apply_theme(fig, width=1080, height=960, margin_left=96)
    header(
        fig,
        "Seed-variance decomposition",
        "Standard deviation of the primary metric across five seeds, split into the "
        "prompt draw (data) and the mask init, against the total (joint).",
        top=170,
        right=120,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=dict(text="sd of the primary metric"), col=1)
    _facet_titles(fig)
    _row_labels(fig, ABLATION_ORDER, len(tasks))
    return fig


# ── driver ──────────────────────────────────────────────────────────────────────────


def _save(
    fig: Optional[go.Figure],
    path: Path,
    formats: Tuple[str, ...] = ("pdf", "png", "html"),
) -> None:
    """PDF for the thesis, PNG to look at, HTML for hover.

    The grid figures opt out of HTML: nine panels of banded curves make a ~200 KB page
    that nobody opens twenty-eight of.
    """
    if fig is None:
        print(f"SKIP {path.name}: no data")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "pdf" in formats:
            fig.write_image(str(path))
        if "png" in formats:
            fig.write_image(str(path.with_suffix(".png")), scale=2)
    except Exception as e:
        print(f"WARNING: static export failed for {path.name} ({e}); writing HTML")
    if "html" in formats:
        fig.write_html(
            str(path.with_suffix(".html")), include_plotlyjs="cdn", full_html=True
        )
    print(f"  {path.relative_to(path.parents[1])}")


def write_all_figures(folder: Path) -> None:
    """Render every figure into `folder/figures`.

    All figures except the new decomposition one describe total run-to-run variance, so
    they are built from joint-condition rows only; a data/init cell existing alongside
    them changes nothing about the figures already in the write-up.
    """
    all_structural = _read(folder / "structural.csv")
    all_functional = _read(folder / "functional.csv")
    all_summary = _read(folder / "summary.csv")
    structural = _joint_only(all_structural)
    functional = _joint_only(all_functional)
    summary = _joint_only(all_summary)
    figures = folder / "figures"
    print(f"Figures -> {figures}")

    _save(fig_jaccard_vs_budget(structural), figures / "jaccard_vs_budget.pdf")
    _save(fig_mutual_iou_vs_budget(structural), figures / "mutual_iou_vs_budget.pdf")
    _save(fig_jaccard_compact(structural), figures / "compact" / "jaccard_vs_budget.pdf")
    _save(fig_selection_frequency(summary), figures / "selection_frequency.pdf")
    _save(
        fig_stability_vs_function(summary, structural, functional),
        figures / "stability_vs_function.pdf",
    )
    slider = fig_stability_vs_function_slider(summary, structural, functional)
    if slider is not None:
        path = figures / "stability_vs_function_by_size.html"
        slider.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        print(f"  {path.relative_to(path.parents[1])}")

    # The task lives in the filename, not a folder: two folders of fourteen beat six of
    # four or five, and the measure is the only dimension that earns a directory.
    seen = {
        (r.get("measure", "faithfulness"), r["task"], r["metric"]) for r in functional
    }
    for measure, task, metric in sorted(seen):
        slug = metric.lower().replace(" ", "_")
        task_slug = task.lower().replace(" ", "_")
        _save(
            fig_measure_grid(functional, task, metric, measure),
            figures / measure / f"{task_slug}__{slug}.pdf",
            formats=("pdf", "png"),
        )

    if {r.get("condition", "joint") for r in all_summary} != {"joint"}:
        _save(
            fig_variance_decomposition(all_summary),
            figures / "variance_decomposition.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="folder", required=True, help="Results folder")
    args = parser.parse_args()
    write_all_figures(Path(args.folder))


if __name__ == "__main__":
    main()
