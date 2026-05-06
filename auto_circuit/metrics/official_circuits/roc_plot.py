from collections import defaultdict
from typing import Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import (
    COLOR_PALETTE,
    Measurements,
    TaskMeasurements,
)


def task_roc_plot(task_measurements: TaskMeasurements) -> go.Figure:
    """
    Wrapper that takes the output of
    [`measure_roc`][auto_circuit.metrics.official_circuits.measure_roc.measure_roc],
    extracts the names of the tasks and algorithms, and plots the ROC curves
    using [`roc_plot`][auto_circuit.metrics.official_circuits.roc_plot.roc_plot].
    """
    task_algo_name_measurements: Dict[str, Dict[str, Measurements]] = defaultdict(dict)
    for task_key, algo_measurements in task_measurements.items():
        task_name = TASK_DICT[task_key].name
        for algo_key, measurements in algo_measurements.items():
            algo_name = PRUNE_ALGO_DICT[algo_key].short_name
            task_algo_name_measurements[task_name][algo_name] = measurements
    return roc_plot(task_algo_name_measurements)


def roc_plot(
    taskname_measurements: Dict[str, Dict[str, Measurements]],
    variable_width: bool = False,
) -> go.Figure:
    """
    Plots the pessimistic Receiver Operating Characteristic (ROC) curve for a nested
    dictionary of measurements. The outer dictionary has keys that are the task names
    and the inner dictionary has keys that are the algorithm names.

    Args:
        taskname_measurements: A nested dictionary with keys corresponding to task names
            (outer), algorithm names (inner), and values corresponding to the points of
            the ROC curve.
        variable_width: If True, the lines corresponding to different to different
            algorithms will have different widths. This helps distinguish overlapping
            lines.

    Returns:
        A plotly figure.
    """
    taskname_measurements = dict(
        sorted(taskname_measurements.items(), key=lambda x: x[0])
    )
    n_tasks = len(taskname_measurements)

    # Each subplot is a square of this size (pixels).
    CELL_SIZE = 420
    # Margins around the whole figure.
    MARGIN_L = 90   # room for y-axis title
    MARGIN_R = 30   # small right pad
    MARGIN_T = 80   # room for subplot titles
    MARGIN_B = 130  # room for x-axis title + bottom legend

    fig_width  = MARGIN_L + n_tasks * CELL_SIZE + MARGIN_R
    fig_height = MARGIN_T + CELL_SIZE + MARGIN_B

    # ---- Build subplots (no built-in subplot_titles — we'll add annotations) ----
    fig = make_subplots(
        rows=1,
        cols=n_tasks,
        shared_yaxes=False,
        horizontal_spacing=0.12 / max(n_tasks, 1),
    )

    # ---- Add traces ----
    for task_idx, (task_name, algo_measurements) in enumerate(
        taskname_measurements.items()
    ):
        for algo_idx, (algo_name, measurements) in enumerate(algo_measurements.items()):
            width_delta = 8
            max_width = (width_delta / 2) + (len(algo_measurements) - 1) * width_delta
            line_width = max_width - algo_idx * width_delta

            fig.add_scatter(
                row=1,
                col=task_idx + 1,
                x=[x for x, _ in measurements],
                y=[y for _, y in measurements],
                mode="markers+text" if len(measurements) == 1 else "lines",
                text=algo_name,
                line=dict(
                    shape="hv",
                    width=line_width if variable_width else 2,
                    color=COLOR_PALETTE[algo_idx],
                ),
                textposition="middle right",
                showlegend=task_idx == 0,
                marker_line_width=2,
                name=algo_name,
                marker_color=COLOR_PALETTE[algo_idx],
            )

    # ---- Axis styling ----
    fig.update_xaxes(
        matches=None,
        range=[-0.02, 1.02],
        title_text="False Positive Rate",
        title_standoff=12,
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        range=[-0.02, 1.02],
        showgrid=True,
        zeroline=False,
    )
    # Label the y-axis only on the leftmost subplot
    fig.update_yaxes(title_text="True Positive Rate", title_standoff=12, col=1)

    # ---- Subplot title annotations (wrapped, above each subplot) ----
    # Plotly places subplot titles automatically only when using subplot_titles=,
    # but that path clips long strings.  We place them manually as annotations.
    n_cols = n_tasks
    annotations = []
    for col_idx, task_name in enumerate(taskname_measurements.keys()):
        # x-position: centre of the column in paper coordinates
        domain_step = 1.0 / n_cols
        x_centre = domain_step * col_idx + domain_step / 2.0
        annotations.append(dict(
            text=f"<b>{task_name}</b>",
            x=x_centre,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=13),
        ))

    # ---- Overall layout ----
    fig.update_layout(
        template="plotly",
        width=fig_width,
        height=fig_height,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
        annotations=annotations,
        legend=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            yanchor="top",
            y=-0.22,         # below the x-axis title (sits at ~y=-0.12)
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    return fig
