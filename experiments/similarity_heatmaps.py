"""Cross-cell circuit-similarity heatmaps for the faithfulness spine (sub-RQ3, #4).

For each task, compares the circuits discovered by every (algorithm x discovery
ablation) cell at fixed edge budgets: pairwise Jaccard similarity between the top-k
edge sets, with the ablation-independent GT / Random anchors as reference rows.
Similarity is cross-discovery-ablation by construction (and independent of the eval
ablation), so outputs live beside — not inside — the discovery-<D>/ folders:

  <experiment>/similarity/<task>/jaccard_k<k>.pdf   one static heatmap per budget
  <experiment>/similarity/<task>/jaccard.html       interactive, slider over budgets
  <experiment>/similarity/<task>/jaccard.csv        tidy matrix rows

Called by faithfulness_comparison.py after the eval matrix; also runnable standalone
(CPU-only — models are loaded just for the edge graph) to regenerate heatmaps from
cached cells without re-measuring anything:

  srun python experiments/similarity_heatmaps.py --cache-dir .prune_scores_cache/faithfulness
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.metrics.analysis_utils import jaccard, top_k_edge_set
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import PruneScores
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.plot_theme import (
    FONT_FAMILY,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
)

LabeledScores = List[Tuple[str, PruneScores]]

DEFAULT_SIMILARITY_EDGE_COUNTS = [500, 1000, 2000, 5000]


def collect_labeled_scores(
    groups: Dict[str, Dict[str, Dict[str, PruneScores]]],
    discovery_ablations: List[str],
    algo_order: List[str],
    anchor_algo_keys: List[str],
) -> Dict[str, LabeledScores]:
    """Flatten the per-discovery-ablation groups into one labeled list per task.

    Axis order is algo-major ("ACDC [zero]", "ACDC [resample]", "ACDC [mean]",
    "EP [zero]", ...) with the ablation-independent anchors (present identically in
    every group) deduplicated and placed last. Missing cells are simply absent.
    """
    task_keys: List[str] = []
    algo_keys_seen: List[str] = []
    for d in discovery_ablations:
        for task_key, algo_ps in groups.get(d, {}).items():
            if task_key not in task_keys:
                task_keys.append(task_key)
            for ak in algo_ps:
                if ak not in algo_keys_seen:
                    algo_keys_seen.append(ak)

    anchors = [ak for ak in anchor_algo_keys if ak in algo_keys_seen]
    swept = [ak for ak in algo_order if ak in algo_keys_seen and ak not in anchors]
    swept += [ak for ak in algo_keys_seen if ak not in swept and ak not in anchors]

    labeled: Dict[str, LabeledScores] = defaultdict(list)
    for task_key in task_keys:
        for ak in swept:
            short = PRUNE_ALGO_DICT[ak].short_name
            for d in discovery_ablations:
                ps = groups.get(d, {}).get(task_key, {}).get(ak)
                if ps is not None:
                    labeled[task_key].append((f"{short} [{d}]", ps))
        for ak in anchors:
            for d in discovery_ablations:
                ps = groups.get(d, {}).get(task_key, {}).get(ak)
                if ps is not None:
                    labeled[task_key].append((PRUNE_ALGO_DICT[ak].short_name, ps))
                    break
    return dict(labeled)


def _jaccard_matrix(edge_sets: List[set]) -> List[List[float]]:
    n = len(edge_sets)
    mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mat[i][j] = mat[j][i] = jaccard(edge_sets[i], edge_sets[j])
    return mat


def _color_max(mats: List[List[List[float]]]) -> float:
    """Top of the colour range: the largest off-diagonal similarity, rounded up.

    Observed overlaps sit far below 1.0, so anchoring the scale at 100% would
    render every cell the same pale blue. The always-1.0 diagonal is excluded for
    the same reason; it simply saturates at the top of the ramp. One shared max
    across all budgets in a figure keeps colours comparable when the slider moves.
    """
    vals = [
        mat[i][j] for mat in mats for i in range(len(mat)) for j in range(i)
    ]
    hi = max(vals, default=1.0)
    return min(1.0, max(0.1, math.ceil(hi * 20) / 20))


def _heatmap_trace(
    labels: List[str], mat: List[List[float]], zmax: float, visible: bool
) -> go.Heatmap:
    # Sequential single-hue scale; plotly picks a contrasting per-cell text color
    # automatically when textfont.color is unset.
    return go.Heatmap(
        x=labels,
        y=labels,
        z=mat,
        zmin=0.0,
        zmax=zmax,
        colorscale=SEQUENTIAL_BLUE,
        colorbar=dict(
            title=dict(
                text="shared<br>edges",
                font=dict(size=13, color=INK_SECONDARY),
                side="top",
            ),
            tickformat=".0%",
            tickfont=dict(size=12, color=INK_SECONDARY),
            outlinewidth=0,
            thickness=18,
            len=0.55,
            xref="paper", x=1.015, xanchor="left",
            yref="paper", y=1.0, yanchor="top",
        ),
        xgap=2,
        ygap=2,
        texttemplate="%{z:.0%}",
        # 18 is the fit limit: "100%" is the widest possible label and at 19+ its
        # trailing "%" overflows the _CELL-wide box. Raise _CELL before raising this.
        textfont={"size": 18, "family": FONT_FAMILY},
        hovertemplate="%{y} ∩ %{x}<br>shared edges: %{z:.1%}<extra></extra>",
        visible=visible,
    )


# Ink colors for the ablation tag in tick labels (identity is still carried by the
# text itself; color is a redundant scanning aid). Chosen away from the Blues fill
# hue, >=4.5:1 contrast on the white surface.
_ABLATION_TAG_COLORS = {
    "zero": "#B84A00",      # dark orange
    "resample": "#00795E",  # dark teal-green
    "mean": "#7B3294",      # purple
}


def _split_label(label: str) -> Tuple[str, str]:
    """"EP [resample]" -> ("EP", "resample"); anchors ("GT") -> ("GT", "")."""
    if label.endswith("]") and " [" in label:
        algo, abl = label.rsplit(" [", 1)
        return algo, abl[:-1]
    return label, ""


def _styled_tick(label: str) -> str:
    """Tick shows only the ablation; the algorithm is named once per block outside
    the axis. Anchors have no ablation, so their tick is blank and the block label
    alone names them."""
    _, abl = _split_label(label)
    color = _ABLATION_TAG_COLORS.get(abl)
    return "" if color is None else f"<span style='color:{color}'>{abl}</span>"


def _blocks(labels: List[str]) -> List[Tuple[str, float, int]]:
    """Contiguous same-algorithm runs as (algo, centre position, length)."""
    algos = [_split_label(lb)[0] for lb in labels]
    out, start = [], 0
    for i in range(1, len(algos) + 1):
        if i == len(algos) or algos[i] != algos[start]:
            out.append((algos[start], (start + i - 1) / 2, i - start))
            start = i
    return out


def _block_boundaries(labels: List[str]) -> List[float]:
    """Category-axis positions between consecutive different-algo blocks."""
    algos = [_split_label(lb)[0] for lb in labels]
    return [i - 0.5 for i in range(1, len(algos)) if algos[i] != algos[i - 1]]


def _title(task_name: str, k: int) -> str:
    return (
        f"<b>{task_name} · circuit overlap · top-{k:,} edges</b><br>"
        "<sup>each cell: share of edges the two circuits have in common"
        " · 100% = identical</sup>"
    )


_MARGIN = dict(l=130, r=110, t=168, b=96)
_CELL = 58
# Distance from each axis to the block bracket and to the block name. The y-axis
# needs more room because its tick text runs sideways ("resample") instead of down.
_BRACKET_PX = dict(x=26, y=72)
_LABEL_PX = dict(x=32, y=80)


def _fig_layout(title: str, labels: List[str]) -> go.Layout:
    n = len(labels)
    width = max(_CELL * n + _MARGIN["l"] + _MARGIN["r"], 620)
    height = max(_CELL * n + _MARGIN["t"] + _MARGIN["b"], 560)
    # Offsets outside the axes are set in pixels, then converted to paper units:
    # a fixed paper fraction would drift with the figure size and open the gap
    # between a block name and the rows it labels.
    plot_w = width - _MARGIN["l"] - _MARGIN["r"]
    plot_h = height - _MARGIN["t"] - _MARGIN["b"]
    lo, hi = -0.5, n - 0.5

    rule = dict(color=INK_PRIMARY, width=3)
    # Rules between algorithm blocks, spanning the whole grid.
    sep = [
        dict(type="line", x0=b, x1=b, y0=lo, y1=hi,
             xref="x", yref="y", layer="above", line=rule)
        for b in _block_boundaries(labels)
    ] + [
        dict(type="line", y0=b, y1=b, x0=lo, x1=hi,
             xref="x", yref="y", layer="above", line=rule)
        for b in _block_boundaries(labels)
    ]
    # A bracket under / beside each block ties its name to exactly the columns and
    # rows it covers, so no cell is ambiguous about which method it belongs to.
    blocks = _blocks(labels)
    span = dict(color=INK_PRIMARY, width=2)
    for _, c, ln in blocks:
        a, b = c - (ln - 1) / 2 - 0.45, c + (ln - 1) / 2 + 0.45
        sep.append(dict(type="line", xref="x", yref="paper", x0=a, x1=b,
                        y0=-_BRACKET_PX["x"] / plot_h, y1=-_BRACKET_PX["x"] / plot_h,
                        line=span))
        sep.append(dict(type="line", xref="paper", yref="y", y0=a, y1=b,
                        x0=-_BRACKET_PX["y"] / plot_w, x1=-_BRACKET_PX["y"] / plot_w,
                        line=span))
    notes = [
        dict(text=f"<b>{algo}</b>", xref="x", x=c, yref="paper",
             y=-_LABEL_PX["x"] / plot_h, xanchor="center", yanchor="top",
             showarrow=False, font=dict(size=12, color=INK_PRIMARY, family=FONT_FAMILY))
        for algo, c, _ in blocks
    ] + [
        dict(text=f"<b>{algo}</b>", xref="paper", x=-_LABEL_PX["y"] / plot_w, yref="y",
             y=c, xanchor="center", yanchor="middle", textangle=-90,
             showarrow=False, font=dict(size=12, color=INK_PRIMARY, family=FONT_FAMILY))
        for algo, c, _ in blocks
    ]
    axis = dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(size=11, color=INK_SECONDARY, family=FONT_FAMILY),
    )
    # The title is an annotation, not layout.title: the latter caps y at 1.0, which
    # measures a two-line title from the top of the canvas and clips its first line.
    notes.append(dict(
        text=title, xref="paper", x=0.0, xanchor="left",
        yref="paper", y=1 + 46 / plot_h, yanchor="bottom",
        align="left", showarrow=False,
        font=dict(size=15, color=INK_PRIMARY, family=FONT_FAMILY),
    ))
    return go.Layout(
        font=dict(family=FONT_FAMILY, size=12, color=INK_SECONDARY),
        width=width,
        height=height,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=_MARGIN,
        # Explicit ranges (y inverted so the matrix reads top-left -> bottom-right)
        # keep the block brackets aligned with the cells they span.
        xaxis=dict(
            tickvals=labels, ticktext=[_styled_tick(lb) for lb in labels],
            range=[lo, hi], **axis,
        ),
        yaxis=dict(
            tickvals=labels, ticktext=[_styled_tick(lb) for lb in labels],
            range=[hi, lo], **axis,
        ),
        annotations=notes,
        shapes=sep,
    )


def save_similarity_heatmaps(
    labeled: LabeledScores,
    edge_counts: List[int],
    model: PatchableModel,
    task_name: str,
    out_dir: Path,
) -> List[Path]:
    """Write per-budget Jaccard heatmaps (PDF + PNG), a slider HTML, and a tidy CSV.

    The PNG is a 2x-scale copy of the same figure, for slides and anywhere a PDF
    cannot be dropped in directly.

    Budgets above the task's total edge count are skipped. Edge sets use
    exclude_zero so binary GT / mostly-zero ACDC scores cap at their nonzero count
    instead of admitting the whole graph when the k-th |score| is 0.
    """
    ks = sorted({k for k in edge_counts if 0 < k <= model.n_edges})
    if not ks or len(labeled) < 2:
        print(f"[similarity] {task_name}: skipping (budgets={ks}, cells={len(labeled)})")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in labeled]
    edges = model.edges
    written: List[Path] = []

    csv_rows: List[dict] = []
    mats: List[List[List[float]]] = []
    for k in ks:
        edge_sets = [
            top_k_edge_set(ps, k, edges, exclude_zero=True) for _, ps in labeled
        ]
        mat = _jaccard_matrix(edge_sets)
        mats.append(mat)
        sizes = [len(s) for s in edge_sets]
        for i, (label_a, size_a) in enumerate(zip(labels, sizes)):
            for j, (label_b, size_b) in enumerate(zip(labels, sizes)):
                csv_rows.append({
                    "task": task_name, "edge_count": k,
                    "label_a": label_a, "label_b": label_b,
                    "jaccard": mat[i][j], "size_a": size_a, "size_b": size_b,
                })

    zmax = _color_max(mats)
    traces: List[go.Heatmap] = []
    for step, (k, mat) in enumerate(zip(ks, mats)):
        fig = go.Figure(
            data=[_heatmap_trace(labels, mat, zmax, visible=True)],
            layout=_fig_layout(_title(task_name, k), labels),
        )
        for path, kwargs in (
            (out_dir / f"jaccard_k{k}.pdf", {}),
            (out_dir / f"jaccard_k{k}.png", {"scale": 2}),
        ):
            fig.write_image(str(path), **kwargs)
            written.append(path)
        traces.append(_heatmap_trace(labels, mat, zmax, visible=step == 0))

    steps = [
        dict(
            method="update",
            label=str(k),
            args=[
                {"visible": [i == step for i in range(len(ks))]},
                {"title.text": _title(task_name, k)},
            ],
        )
        for step, k in enumerate(ks)
    ]
    html_fig = go.Figure(
        data=traces,
        layout=_fig_layout(_title(task_name, ks[0]), labels),
    )
    html_fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Edges: "}, pad={"t": 40}, steps=steps)]
    )
    html_path = out_dir / "jaccard.html"
    html_fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    written.append(html_path)

    csv_path = out_dir / "jaccard.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    written.append(csv_path)
    return written


def write_all_similarity_heatmaps(
    groups: Dict[str, Dict[str, Dict[str, PruneScores]]],
    discovery_ablations: List[str],
    algo_order: List[str],
    anchor_algo_keys: List[str],
    edge_counts: List[int],
    experiment_folder: Path,
) -> None:
    """Entry point for the analysis pass: one similarity/<task>/ folder per task."""
    labeled_per_task = collect_labeled_scores(
        groups, discovery_ablations, algo_order, anchor_algo_keys
    )
    for task_key, labeled in labeled_per_task.items():
        task = TASK_DICT[task_key]
        task_slug = task.name.lower().replace(" ", "_")
        out_dir = experiment_folder / "similarity" / task_slug
        try:
            written = save_similarity_heatmaps(
                labeled, edge_counts, task.model, task.name, out_dir
            )
            if written:
                print(f"[similarity] {task.name}: {len(labeled)} cells -> {out_dir}")
        except Exception as e:
            print(f"WARNING: similarity heatmaps failed for {task.name}: {e}")


def main() -> None:
    from auto_circuit.metrics.analysis_utils import latest_per_cell, load_many_caches
    from auto_circuit.utils.misc import repo_path_to_abs_path
    from faithfulness_cells import ANCHOR_ALGOS, DISCOVERY_ABLATIONS, SWEPT_ALGOS
    from faithfulness_comparison import _group_cells_by_discovery_ablation
    from run_cell import _ALGO_MAP

    parser = argparse.ArgumentParser(description="Circuit-similarity heatmaps from cached cells.")
    parser.add_argument("--cache-dir", default=".prune_scores_cache/faithfulness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output folder (default: results/faithfulness_comparison/similarity_<date>)")
    parser.add_argument("--edge-counts", type=int, nargs="+", default=DEFAULT_SIMILARITY_EDGE_COUNTS)
    args = parser.parse_args()

    loaded = latest_per_cell(load_many_caches(args.cache_dir, "cell_"))
    groups, consumed = _group_cells_by_discovery_ablation(loaded, args.seed)
    print(f"Loaded {len(consumed)} cells from {args.cache_dir} (seed {args.seed}).")

    out = Path(args.out) if args.out else repo_path_to_abs_path(
        f"results/faithfulness_comparison/similarity_{datetime.now():%Y-%m-%d}"
    )
    write_all_similarity_heatmaps(
        groups=groups,
        discovery_ablations=DISCOVERY_ABLATIONS,
        algo_order=[_ALGO_MAP[a].key for a in SWEPT_ALGOS],
        anchor_algo_keys=[_ALGO_MAP[a].key for a in ANCHOR_ALGOS],
        edge_counts=args.edge_counts,
        experiment_folder=out,
    )
    print(f"Done. Results in: {out}")


if __name__ == "__main__":
    main()
