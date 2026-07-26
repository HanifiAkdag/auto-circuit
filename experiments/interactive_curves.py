"""Interactive overlay curves for cross-cell comparison (sub-RQ3 analysis pass).

Complements the static per-cell PDFs. Each HTML overlays *all* discovery cells for
one (task, measure, metric) so any subset can be compared interactively:

  - one line per (algo x discovery-ablation), e.g. "DiscoGP-EP | disc=zero";
  - a dropdown selects the eval-ablation (resample / zero / mean);
  - color encodes the algorithm, line dash encodes the discovery-ablation;
  - the Plotly legend toggles individual lines (click = show/hide,
    double-click = isolate), so you can pick exactly the curves to compare
    (e.g. DiscoGP-EP disc=zero vs disc=resample alongside EP disc=mean).

Anchor algorithms (GT / Random) have no discovery sweep, so they appear once
per eval-ablation as solid reference lines.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.colors as pc
import plotly.graph_objects as go

# One overlaid line: its algo, discovery label, eval label and (edges, value) series.
CurveEntry = Tuple[str, str, str, List[Tuple[float, float]]]

# Per discovery-ablation styling. Dash + marker symbol are varied together so
# same-colour lines (one algo, different discovery) stay distinguishable even when
# printed greyscale or zoomed out.
_DISC_DASH: Dict[str, str] = {
    "zero": "dash",
    "resample": "solid",
    "mean": "dot",
    "anchor": "solid",
}
_DISC_SYMBOL: Dict[str, str] = {
    "zero": "square",
    "resample": "circle",
    "mean": "diamond",
    "anchor": "x",
}
_DISC_ORDER = ["zero", "resample", "mean", "anchor"]


def _algo_color_map(algo_keys: List[str]) -> Dict[str, str]:
    palette = pc.qualitative.D3 + pc.qualitative.Set2 + pc.qualitative.Dark24
    return {a: palette[i % len(palette)] for i, a in enumerate(sorted(algo_keys))}


def _line_name(algo_display: str, disc: str) -> str:
    return algo_display if disc == "anchor" else f"{algo_display} | disc={disc}"


def save_interactive_curves(
    entries: List[CurveEntry],
    *,
    title: str,
    x_title: str,
    y_title: str,
    eval_labels: List[str],
    default_eval: str,
    out_path: Path,
    algo_display: Dict[str, str],
) -> Optional[Path]:
    """Write one interactive overlay HTML for a single (task, measure, metric).

    `entries` are (algo_key, discovery_label, eval_label, series) tuples; duplicates
    (same algo/disc/eval) are collapsed. Returns the written path, or None if empty.
    """
    # Deduplicate (anchors are recorded once per discovery group upstream).
    by_key: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    for algo_key, disc, ev, series in entries:
        if series:
            by_key[(algo_key, disc, ev)] = series
    if not by_key:
        return None

    algo_keys = sorted({k[0] for k in by_key})
    color = _algo_color_map(algo_keys)
    default_eval = default_eval if default_eval in eval_labels else eval_labels[0]

    # Stable trace order: algo, then discovery, then eval.
    def _sort_key(k: Tuple[str, str, str]):
        algo_key, disc, ev = k
        disc_rank = _DISC_ORDER.index(disc) if disc in _DISC_ORDER else len(_DISC_ORDER)
        ev_rank = eval_labels.index(ev) if ev in eval_labels else len(eval_labels)
        return (algo_display.get(algo_key, algo_key), disc_rank, ev_rank)

    fig = go.Figure()
    trace_eval: List[str] = []
    for algo_key, disc, ev in sorted(by_key, key=_sort_key):
        series = by_key[(algo_key, disc, ev)]
        xs = [pt[0] for pt in series]
        ys = [pt[1] for pt in series]
        name = _line_name(algo_display.get(algo_key, algo_key), disc)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                legendgroup=algo_display.get(algo_key, algo_key),
                line=dict(color=color[algo_key], dash=_DISC_DASH.get(disc, "solid"), width=2),
                marker=dict(size=6, symbol=_DISC_SYMBOL.get(disc, "circle")),
                visible=(ev == default_eval),
                hovertemplate=(
                    f"<b>{name}</b><br>eval={ev}<br>"
                    f"{x_title}=%{{x}}<br>{y_title}=%{{y:.4f}}<extra></extra>"
                ),
            )
        )
        trace_eval.append(ev)

    # Dropdown: pick the eval-ablation (shows only that eval's lines).
    buttons = []
    for ev in eval_labels:
        vis = [te == ev for te in trace_eval]
        if not any(vis):
            continue
        buttons.append(
            dict(
                label=f"eval = {ev}",
                method="update",
                args=[{"visible": vis}, {"title.text": f"{title} — eval-ablation: {ev}"}],
            )
        )
    active = next((i for i, b in enumerate(buttons) if b["label"] == f"eval = {default_eval}"), 0)

    fig.update_layout(
        title=dict(text=f"{title} — eval-ablation: {default_eval}"),
        # dtick=1 on a log axis = one label per power of ten (10, 100, 1000, …)
        # instead of Plotly's default 1,2,5 per-decade labels; minor ticks stay
        # as unlabeled gridlines for orientation.
        xaxis=dict(
            title=x_title,
            type="log",
            dtick=1,
            minor=dict(showgrid=True),
        ),
        yaxis=dict(title=y_title),
        legend=dict(title="click to toggle · double-click to isolate", groupclick="toggleitem"),
        template="plotly_white",
        updatemenus=[
            dict(
                buttons=buttons,
                active=active,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.12,
                yanchor="top",
            )
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path
