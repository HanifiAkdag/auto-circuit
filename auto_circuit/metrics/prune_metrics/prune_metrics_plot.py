from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.types import AblationType, TaskMeasurements


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

    Returns:
        A plotly figure.
    """
    n_tasks = (
        len(set(d["Task"] for d in data)) if data else len(task_measurements)
    )

    if len(data) > 0:
        data = sorted(data, key=lambda x: (x["Algorithm"], x["Task"]))
        fig = px.line(
            data,
            x="X",
            y="Y",
            facet_col="Task",
            color="Algorithm",
            log_x=log_x,
            log_y=log_y,
            range_y=None if y_max is None else [y_min, y_max * 0.8],
            facet_col_spacing=0.03 if y_axes_match else 0.06,
            markers=True,
        )
    else:
        fig = subplots.make_subplots(rows=1, cols=len(task_measurements))

    # Add single-point markers for algorithms with only one measurement
    task_measurements = dict(sorted(task_measurements.items(), key=lambda x: x[0]))
    for task_idx, algo_measurements in enumerate(task_measurements.values()):
        for algo_key, measurements in algo_measurements.items():
            algo = PRUNE_ALGO_DICT[algo_key]
            pos = "middle right" if algo.short_name == "GT" else "middle left"
            if len(measurements) == 1:
                x, y = measurements[0]
                fig.add_scattergl(
                    row=1,
                    col=task_idx + 1,
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    text=algo.short_name if algo.short_name else algo.name,
                    textposition=pos,
                    showlegend=task_idx == 0,
                    marker=dict(color="black", size=10, symbol="x-thin"),
                    marker_line_width=2,
                    name=algo.short_name,
                )

    # --- Layout -----------------------------------------------------------
    # Put the metric name and ablation type on separate lines so the y-axis
    # label never gets truncated, even for long names like
    # "Correct Answer Percent (Resample)".
    y_title = f"{metric_name}<br><sup>({ablation_type})</sup>"

    fig_width = max(450 * n_tasks + 80, 700)
    fig_height = 550

    fig.update_layout(
        yaxis_title=y_title,
        template="plotly",
        width=fig_width,
        height=fig_height,
        # Generous margins: left for the y-label, bottom for the legend,
        # top/right for breathing room.
        margin=dict(l=120, r=30, t=60, b=100),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
    )

    if not y_axes_match:
        fig.update_yaxes(matches=None, showticklabels=True, automargin=True)
    else:
        fig.update_yaxes(automargin=True)

    # --- Axis Formatting --------------------------------------------------
    common_x_axes = dict(matches=None, title="Circuit Edges", ticks="outside")
    if log_x:
        common_x_axes.update(dict(type="log", dtick=1, minor_ticks="outside", exponentformat="power"))
    fig.update_xaxes(**common_x_axes)

    common_y_axes = dict(automargin=True, ticks="outside")
    if not y_axes_match:
        common_y_axes.update(dict(matches=None, showticklabels=True))
    if log_y:
        common_y_axes.update(dict(type="log", dtick=1, minor_ticks="outside", exponentformat="power"))
    fig.update_yaxes(**common_y_axes)

    return fig
