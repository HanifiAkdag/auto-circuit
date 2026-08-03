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
import sys
from pathlib import Path
from typing import List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.utils.plot_theme import (
    DISC_DASH,
    DISC_SYMBOL,
    FONT_FAMILY,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
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


def _row_labels(fig, ablations: List[str], n_cols: int) -> None:
    """Name each row of the grid on its right edge, so the y-axis stays for the metric."""
    n_rows = len(ablations)
    for i, ablation in enumerate(ablations):
        fig.add_annotation(
            text=f"<b>{ablation}</b> discovery",
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


def fig_stability_vs_function(summary: List[dict]) -> Optional[go.Figure]:
    """y is the spread in the metric's own units, not its coefficient of variation.

    Both task metrics are percentages of the full model, so a standard deviation in
    percentage points is directly readable and comparable within a task. A CV is not: it
    divides by a mean that several cells drive to zero (ACDC/IOI/mean means 4.15%), which
    manufactures enormous instability out of a circuit that is merely bad.
    """
    usable = [r for r in summary if isinstance(r.get("primary_sd"), float)
              and r["primary_sd"] == r["primary_sd"]]
    if not usable:
        return None
    tasks, algos = _tasks(usable), _algos(usable)
    colors = algo_colors(algos)
    fig = make_subplots(
        rows=1,
        cols=len(tasks),
        subplot_titles=[TASK_DISPLAY.get(t, t) for t in tasks],
        horizontal_spacing=0.055,
    )
    for task_i, task in enumerate(tasks, start=1):
        trows = [r for r in usable if r["task"] == task]
        # Random is the x-floor, drawn as a guide rather than as a competing marker: its
        # metric is at chance for every seed, so its CV is 0 and has no place on a log axis.
        floor = next((r["jaccard_mean"] for r in trows if r["algo"] == "Random"), None)
        for algo in [a for a in algos if a != "Random"]:
            for ablation in ABLATION_ORDER:
                row = next(
                    (r for r in trows
                     if r["algo"] == algo and r["discovery_ablation"] == ablation),
                    None,
                )
                if row is None:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[row["jaccard_mean"]],
                        y=[row["primary_sd"]],
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=[row["jaccard_max"] - row["jaccard_mean"]],
                            arrayminus=[row["jaccard_mean"] - row["jaccard_min"]],
                            color=_rgba(colors[algo], 0.4),
                            thickness=1,
                            width=0,
                        ),
                        mode="markers",
                        name=algo,
                        legendgroup=algo,
                        showlegend=(task_i == 1 and ablation == ABLATION_ORDER[0]),
                        marker=dict(
                            color=colors[algo],
                            size=11,
                            symbol=DISC_SYMBOL[ablation],
                            line=dict(width=1.6, color="#ffffff"),
                        ),
                        hovertemplate=f"{algo} · {ablation}<br>mean J=%{{x:.3f}}<br>"
                        "spread (sd)=%{y:.3g}<extra></extra>",
                    ),
                    row=1,
                    col=task_i,
                )
        if floor is not None:
            fig.add_vline(
                x=floor,
                line=dict(color=INK_MUTED, width=1, dash="dot"),
                annotation=dict(
                    text="Random",
                    font=dict(size=10, color=INK_MUTED),
                    textangle=90,
                    yanchor="bottom",
                    y=0,
                ),
                row=1,
                col=task_i,
            )

    # Colour carries the algorithm, so the symbol key needs its own neutral legend block.
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

    apply_theme(fig, width=1180, height=580, margin_left=110)
    header(
        fig,
        "Structural instability against functional instability",
        "One marker per algorithm and discovery ablation, at the published circuit size. "
        "Horizontal bar: min-max Jaccard. Symbol: discovery ablation.",
        top=165,
        right=40,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title=dict(text="Pairwise Jaccard"))
    fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(
        title=dict(text="Task metric, sd across seeds"),
        col=1,
    )
    _facet_titles(fig)
    return fig


# ── Fig 5: the functional curves behind Fig 2 ───────────────────────────────────────


def fig_faithfulness_spread(
    functional: List[dict], task: str, metric: str
) -> Optional[go.Figure]:
    rows = [r for r in functional if r["task"] == task and r["metric"] == metric]
    if not rows:
        return None
    algos = _algos(rows)
    colors, symbols = algo_colors(algos), algo_symbols(algos)
    fig = make_subplots(
        rows=1,
        cols=len(ABLATION_ORDER),
        shared_yaxes=True,
        subplot_titles=[f"{a} discovery" for a in ABLATION_ORDER],
        horizontal_spacing=0.04,
    )
    for col, ablation in enumerate(ABLATION_ORDER, start=1):
        for algo in algos:
            want = "anchor" if algo == "Random" else ablation
            eval_want = ablation
            series = sorted(
                (
                    r for r in rows
                    if r["algo"] == algo and r["discovery_ablation"] == want
                    and r["eval_ablation"] == eval_want
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
                    [s["metric_max"] for s in series], colors[algo], 1, col,
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[s["metric_mean"] for s in series],
                    name=algo,
                    legendgroup=algo,
                    showlegend=(col == 1),
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
                row=1,
                col=col,
            )
    apply_theme(fig, width=1280, height=580, margin_left=110)
    header(
        fig,
        f"{TASK_DISPLAY.get(task, task)} · {metric} across five seeds",
        "Line: mean over the five seeds. Band: min-max. Each panel is evaluated under its "
        "own discovery ablation.",
        top=155,
        right=40,
    )
    fig.update_xaxes(
        type="log", dtick=1, exponentformat="power", title=dict(text="Edges in circuit")
    )
    fig.update_yaxes(title=dict(text=metric), col=1)
    _facet_titles(fig)
    return fig


# ── driver ──────────────────────────────────────────────────────────────────────────


def _save(fig: Optional[go.Figure], path: Path) -> None:
    if fig is None:
        print(f"SKIP {path.name}: no data")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path))
        fig.write_image(str(path.with_suffix(".png")), scale=2)
    except Exception as e:
        print(f"WARNING: static export failed for {path.name} ({e}); writing HTML")
    fig.write_html(str(path.with_suffix(".html")), include_plotlyjs="cdn", full_html=True)
    print(f"  {path.relative_to(path.parents[1])}")


def write_all_figures(folder: Path) -> None:
    structural = _read(folder / "structural.csv")
    functional = _read(folder / "functional.csv")
    summary = _read(folder / "summary.csv")
    figures = folder / "figures"
    print(f"Figures -> {figures}")

    _save(fig_jaccard_vs_budget(structural), figures / "jaccard_vs_budget.pdf")
    _save(fig_mutual_iou_vs_budget(structural), figures / "mutual_iou_vs_budget.pdf")
    _save(fig_jaccard_compact(structural), figures / "compact" / "jaccard_vs_budget.pdf")
    _save(fig_selection_frequency(summary), figures / "selection_frequency.pdf")
    _save(fig_stability_vs_function(summary), figures / "stability_vs_function.pdf")

    seen = {(r["task"], r["metric"]) for r in functional}
    for task, metric in sorted(seen):
        slug = metric.lower().replace(" ", "_")
        task_slug = task.lower().replace(" ", "_")
        _save(
            fig_faithfulness_spread(functional, task, metric),
            figures / "faithfulness" / task_slug / f"{slug}.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="folder", required=True, help="Results folder")
    args = parser.parse_args()
    write_all_figures(Path(args.folder))


if __name__ == "__main__":
    main()
