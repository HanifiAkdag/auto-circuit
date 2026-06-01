import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import plotly.graph_objects as go
import torch as t

from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import prune_scores_threshold
from auto_circuit.visualize import net_viz


def _slugify(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return slug or "unnamed"


def _panel_intervals(
    model: PatchableModel,
    prune_scores: PruneScores,
    score_threshold: float,
    show_all_seq_pos: bool = False,
) -> Dict[Optional[int], Tuple[float, float]]:
    """Vertical (domain-y) interval per token position. Mirrors `draw_seq_graph`.

    Computed once (at the largest edge count) so panels stay fixed across slider steps.
    For `seq_len=None` models this is simply `{None: (0, 1)}`.
    """
    seq_len = model.seq_len or 1
    if seq_len <= 1:
        return {list(model.edge_dict.keys())[0]: (0.0, 1.0)}

    edge_scores = list(prune_scores.values())
    ps = [t.clamp(v.abs() - score_threshold, min=0).sum().item() for v in edge_scores]
    total_ps = max(sum(ps), 1e-2)

    sankey_heights: Dict[Optional[int], float] = {}
    for patch_mask in edge_scores:
        ps_seq_tots = t.clamp(patch_mask.abs() - score_threshold, min=0.0)
        ps_seq_tots = ps_seq_tots.sum(dim=list(range(1, patch_mask.ndim)))
        for seq_idx, ps_seq_tot in enumerate(ps_seq_tots):
            if ps_seq_tot > 0 or show_all_seq_pos:
                sankey_heights.setdefault(seq_idx, 0.0)
                sankey_heights[seq_idx] += ps_seq_tot.item()

    for seq_idx in model.edge_dict.keys():
        min_height = total_ps / (len(model.edge_dict) * 2)
        if show_all_seq_pos:
            sankey_heights[seq_idx] = max(sankey_heights.get(seq_idx, 0.0), min_height)

    margin_height = total_ps / ((n_figs := len(sankey_heights)) * 2)
    total_height = sum(sankey_heights.values()) + margin_height * (n_figs - 1)
    intervals: Dict[Optional[int], Tuple[float, float]] = {}
    interval_start = total_height
    for seq_idx, height in sorted(
        sankey_heights.items(), key=lambda x: (x[0] is None, x[0])
    ):
        interval_end = interval_start - (margin_height if len(intervals) > 0 else 0)
        interval_start = interval_end - height
        intervals[seq_idx] = (
            max(interval_start / total_height, 1e-6),
            min(interval_end / total_height, 1 - 1e-6),
        )
    return intervals


def save_circuit_sankey_slider(
    prune_scores: PruneScores,
    model: PatchableModel,
    algo_name: str,
    task_name: str,
    output_dir: Path,
    edge_counts: Optional[List[int]] = None,
    max_edges: int = 10000,
    label_max_edges: int = 1000,
    node_font_size: int = 8,
    true_edge_count: Optional[int] = None,
    seq_labels: Optional[List[str]] = None,
    orientation: Literal["h", "v"] = "h",
) -> Path:
    """Save a circuit as an interactive HTML Sankey with an edge-count slider.

    Each slider step shows the top-k edges by ``|score|``. The step values match the
    logarithmic grid used by ``measure_prune_metrics`` (via ``edge_counts_util``), so the
    slider lines up with the prune-metric curves. One self-contained HTML file is written.

    Per-edge hover labels are kept only for steps with at most ``label_max_edges`` edges;
    above that the labels are dropped (they are unreadable at that density and dominate
    the HTML file size).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if edge_counts is None:
        edge_counts = edge_counts_util(
            edges=model.edges,
            test_counts=None,
            prune_scores=prune_scores,
            true_edge_count=true_edge_count,
        )
    counts = sorted({k for k in edge_counts if 1 <= k <= max_edges})
    if not counts:
        counts = [min(max_edges, model.n_edges)]

    # Fixed panel layout, computed at the largest edge count for stability.
    largest_th = prune_scores_threshold(prune_scores, counts[-1]).item()
    intervals = _panel_intervals(model, prune_scores, score_threshold=largest_th)

    sankeys: List[go.Sankey] = []
    step_trace_ranges: List[Tuple[int, int]] = []  # [start, end) trace idx per step
    for k in counts:
        th = prune_scores_threshold(prune_scores, k).item()
        # Keep only the top-k edges (|score| >= th); a positive floor drops exact zeros.
        step_threshold = max(th, 1e-12)
        start = len(sankeys)
        for seq_idx, vert_interval in intervals.items():
            viz, _ = net_viz(
                model=model,
                seq_edges=set(model.edge_dict[seq_idx]),
                prune_scores=prune_scores,
                vert_interval=vert_interval,
                seq_idx=seq_idx,
                score_threshold=step_threshold,
                layer_order=True,
                orientation=orientation,
            )
            if k > label_max_edges:
                viz.link["label"] = []
                viz.link["hoverinfo"] = "skip"
            viz.node["pad"] = 15
            viz.node["thickness"] = 10
            viz.textfont = dict(size=node_font_size)
            sankeys.append(viz)
        step_trace_ranges.append((start, len(sankeys)))

    n_traces = len(sankeys)
    # Default step: nearest to true_edge_count, else the median count.
    if true_edge_count is not None:
        default_step = min(
            range(len(counts)), key=lambda i: abs(counts[i] - true_edge_count)
        )
    else:
        default_step = len(counts) // 2

    def _visible(step_idx: int) -> List[bool]:
        lo, hi = step_trace_ranges[step_idx]
        return [lo <= i < hi for i in range(n_traces)]

    for i, viz in enumerate(sankeys):
        viz.visible = _visible(default_step)[i]

    title = f"{task_name} · {algo_name} · top-{counts[default_step]} edges"
    steps = []
    for step_idx, k in enumerate(counts):
        steps.append(
            dict(
                method="update",
                label=str(k),
                args=[
                    {"visible": _visible(step_idx)},
                    {"title.text": f"{task_name} · {algo_name} · top-{k} edges"},
                ],
            )
        )

    # Fixed size that nearly fills the browser viewport (see write_html below).
    fig = go.Figure(
        data=sankeys,
        layout=go.Layout(
            autosize=True,
            title=dict(text=title),
            margin=dict(t=60, b=80),
            sliders=[
                dict(
                    active=default_step,
                    currentvalue={"prefix": "Edges: "},
                    pad={"t": 40},
                    steps=steps,
                )
            ],
        ),
    )

    if seq_labels is not None and (model.seq_len or 1) > 1:
        for seq_idx, (y0, y1) in intervals.items():
            label = "All tokens" if seq_idx is None else seq_labels[seq_idx]
            fig.add_annotation(
                x=-0.17,
                y=(y0 + y1) / 2,
                text=f"<b>{label}</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
            )

    out_path = output_dir / f"{_slugify(task_name)}__{_slugify(algo_name)}.html"
    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="96vh",
        config={"responsive": True},
    )
    return out_path
