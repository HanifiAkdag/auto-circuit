from typing import Any, Dict, List, Optional, Set

import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import AblationType, TaskMeasurements
from auto_circuit.utils.plot_theme import (
    ANCHOR_SYMBOLS,
    INK_MUTED,
    INK_PRIMARY,
    algo_colors,
    algo_symbols,
    apply_theme,
    is_anchor,
    strip_facet_prefix,
    style_marks,
)

# Plain-language gloss for the y-axis, keyed by metric name. Tells the reader what
# the number means and which direction is good, so the axis is self-explanatory.
_METRIC_GLOSS: Dict[str, str] = {
    "Clean KL Div": "KL(full model ‖ circuit) · 0 = identical to the full model",
    "Answer Pair KL Div": "KL over the answer pair · 0 = identical to the full model",
    "Logit Diff Percent": "% of the full model's logit difference · 100 = full model",
    "Prob Diff": "answer-probability difference · higher = closer to full model",
    "Correct Answer Percent": "% of prompts answered correctly · higher is better",
    "Correct Answer Greater Than Incorrect Percent": (
        "% of prompts where the correct answer wins · higher is better"
    ),
    "Answer Prob": "probability of the correct answer · higher is better",
}


def _task_totals(task_measurements: TaskMeasurements) -> Dict[str, Dict[str, Any]]:
    """Per-task display info (name, total candidate edges), when the task is loaded.

    Tasks are lazy: touching `.model` on an uninitialised task would load a
    transformer just to draw a label, so only read tasks that are already live.
    """
    info: Dict[str, Dict[str, Any]] = {}
    for task_key in task_measurements:
        task = TASK_DICT.get(task_key)
        if task is None:
            continue
        entry: Dict[str, Any] = {"name": task_key}
        entry["name"] = task.name
        if getattr(task, "__init_complete__", False):
            entry["n_edges"] = task.model.n_edges
        info[task_key] = entry
    return info


def edge_patching_plot(
    data: List[Dict[str, Any]],
    task_measurements: TaskMeasurements,
    ablation_type: AblationType,
    metric_name: str,
    log_x: bool,
    log_y: bool,
    y_axes_match: bool,
    y_max: Optional[float],
    y_min: Optional[float],
    discovery_ablation: Optional[str] = None,
    omit_algos: Optional[Dict[str, Set[str]]] = None,
) -> go.Figure:
    """
    A figure showing the performance of the circuits produced by different
    [`PruneAlgos`][auto_circuit.prune_algos.prune_algos.PruneAlgo] on different tasks.
    The x-axis is the number of edges in the circuit and the y-axis is the performance.

    Args:
        data: A list of dictionaries in the following format:
                <pre><code>{
                "Task": str,
                "Algorithm": str,
                "X": Number,
                "Y": Number,
            }</code></pre>
        task_measurements: The measurements to plot
            (the same as `data` but in a different format).
        ablation_type: The type of ablation used to generate the data.
        metric_name: The name of the metric which the data represents.
        log_x: Whether to log the x-axis.
        log_y: Whether to log the y-axis.
        y_axes_match: Whether to use the same y-axis for all tasks.
        y_max: The maximum value for the y-axis.
        y_min: The minimum value for the y-axis.
        discovery_ablation: Ablation the circuits were discovered under. Named in
            the subtitle beside `ablation_type` (which is the eval ablation).
        omit_algos: Per task key, algorithm keys to leave out of this figure.

    Returns:
        A plotly figure.
    """
    # Facet order is fixed by task key so the single-point overlay below (which
    # indexes columns) always lands on the right panel.
    task_info = _task_totals(task_measurements)
    ordered_keys = sorted(task_measurements.keys())
    ordered_names = [task_info.get(k, {}).get("name", k) for k in ordered_keys]
    n_tasks = len(ordered_keys) or len(set(d["Task"] for d in data))

    # Every algorithm present (including single-point ones like GT) gets a colour,
    # so a hue always means the same algorithm across every figure.
    all_algo_names = sorted(
        {d["Algorithm"] for d in data}
        | {
            PRUNE_ALGO_DICT[a].short_name
            for algos in task_measurements.values()
            for a in algos
            if a in PRUNE_ALGO_DICT
        }
    )
    algo_names = sorted({d["Algorithm"] for d in data})
    colors = algo_colors(all_algo_names)
    symbols = algo_symbols(all_algo_names)

    if len(data) > 0:
        data = sorted(data, key=lambda x: (x["Algorithm"], x["Task"]))
        fig = px.line(
            data,
            x="X",
            y="Y",
            facet_col="Task",
            color="Algorithm",
            color_discrete_map=colors,
            symbol="Algorithm",
            symbol_map=symbols,
            category_orders={"Task": ordered_names, "Algorithm": algo_names},
            log_x=log_x,
            log_y=log_y,
            range_y=None if y_max is None else [y_min, y_max * 0.8],
            facet_col_spacing=0.04 if y_axes_match else 0.07,
            markers=True,
        )
    else:
        fig = subplots.make_subplots(rows=1, cols=len(task_measurements))

    # --- Single-measurement algorithms (e.g. GT) -------------------------------
    # A binary circuit yields one (edges, value) point, so it has no line. Draw it
    # as a large ringed marker with a direct label, otherwise it disappears.
    for task_idx, task_key in enumerate(ordered_keys):
        for algo_key, measurements in task_measurements[task_key].items():
            if len(measurements) != 1:
                continue
            if omit_algos and algo_key in omit_algos.get(task_key, set()):
                continue
            algo = PRUNE_ALGO_DICT[algo_key]
            name = algo.short_name or algo.name
            x, y = measurements[0]
            fig.add_scattergl(
                row=1,
                col=task_idx + 1,
                x=[x],
                y=[y],
                mode="markers+text",
                text=name,
                textposition="middle right" if name == "GT" else "middle left",
                textfont=dict(size=11, color=INK_PRIMARY),
                showlegend=task_idx == 0,
                marker=dict(
                    color=colors.get(name, INK_MUTED),
                    size=15,
                    symbol=symbols.get(name, ANCHOR_SYMBOLS.get(name, "star")),
                    line=dict(width=2, color="#ffffff"),
                ),
                name=name,
            )

    # --- Layout ---------------------------------------------------------------
    gloss = _METRIC_GLOSS.get(metric_name)
    y_title = metric_name if gloss is None else f"{metric_name}<br><sup>{gloss}</sup>"

    fig_width = max(430 * n_tasks + 140, 720)
    apply_theme(
        fig,
        width=fig_width,
        height=560,
        legend_title="",
        margin_left=110,
    )
    style_marks(fig, marker_size=6)
    # Anchors are reference lines, not competitors: dashed, thin, recessive.
    fig.for_each_trace(
        lambda tr: tr.update(line=dict(dash="dash", width=1.5), marker=dict(size=5))
        if is_anchor(tr.name) and tr.mode and "lines" in tr.mode
        else None
    )
    strip_facet_prefix(fig)

    # The task name is the title and the protocol is its subtitle, so the two read
    # as one block. The legend moves below the plot rather than stacking above them.
    ev = str(ablation_type).lower()
    protocol = f"{str(discovery_ablation).lower()} discovery · {ev} eval" \
        if discovery_ablation else f"{ev} eval"
    fig.for_each_annotation(
        lambda a: a.update(
            text=f"{a.text}<br><sup>{protocol}</sup>",
            font=dict(size=15, color=INK_PRIMARY),
        )
    )
    fig.update_layout(
        yaxis_title=y_title,
        margin=dict(t=76, b=150),
        legend=dict(
            orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0.0
        ),
    )

    # --- Axis formatting ------------------------------------------------------
    common_x_axes: Dict[str, Any] = dict(matches=None, ticks="outside")
    if log_x:
        common_x_axes.update(
            dict(type="log", dtick=1, minor_ticks="outside", exponentformat="power")
        )
    fig.update_xaxes(**common_x_axes)

    common_y_axes: Dict[str, Any] = dict(ticks="outside")
    if not y_axes_match:
        common_y_axes.update(dict(matches=None, showticklabels=True))
    if log_y:
        common_y_axes.update(
            dict(type="log", dtick=1, minor_ticks="outside", exponentformat="power")
        )
    fig.update_yaxes(**common_y_axes)

    # Per-facet x title answers "edges out of how many?".
    for task_idx, task_key in enumerate(ordered_keys):
        total = task_info.get(task_key, {}).get("n_edges")
        title = "Edges in circuit"
        if total:
            title += f"<br><sup>of {total:,} candidate edges</sup>"
        fig.update_xaxes(title=dict(text=title), col=task_idx + 1)

    return fig
