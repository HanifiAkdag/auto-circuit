"""Shared Plotly styling for the thesis / slide figures.

One light theme and one entity -> colour map, so every figure reads as the same
system. The four discovery algorithms own the hues; the anchors (GT, Random) stay
neutral ink so they read as reference rather than as another competitor.

Palette: first four slots of a CVD-validated categorical set (worst all-pairs
CVD dE 9.2, normal-vision dE 16.3 on a white surface). Marker symbols vary with
colour so the series stay separable in greyscale print.
"""
from typing import Dict, Iterable, List

import plotly.graph_objects as go

FONT_FAMILY = 'Inter, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

# Chart chrome / ink
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e9e8e2"
AXIS = "#c3c2b7"

# Categorical slots (validated order). Use SLOTS directly when the series are not
# algorithms (e.g. one line per discovery ablation).
SLOTS: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#eda100"]
_SLOTS = SLOTS
_SYMBOLS: List[str] = ["circle", "square", "triangle-up", "diamond", "cross", "pentagon"]

# Single-hue sequential ramp for magnitude (heatmaps). Light -> dark blue.
SEQUENTIAL_BLUE: List[str] = [
    "#f4f8fe",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

# Anchors are not series: ink, not hue.
ANCHOR_COLORS: Dict[str, str] = {"GT": INK_PRIMARY, "Random": INK_MUTED}
ANCHOR_SYMBOLS: Dict[str, str] = {"GT": "star", "Random": "x-thin"}

# Fixed assignment for the faithfulness spine, so a colour always means the same
# algorithm across every figure in the thesis.
_FIXED: Dict[str, str] = {
    "EP": _SLOTS[0],
    "DiscoGP": _SLOTS[1],
    "ACDC": _SLOTS[2],
    "EAP": _SLOTS[3],
    **ANCHOR_COLORS,
}
_FIXED_SYMBOLS: Dict[str, str] = {
    "EP": _SYMBOLS[0],
    "DiscoGP": _SYMBOLS[1],
    "ACDC": _SYMBOLS[2],
    "EAP": _SYMBOLS[3],
    **ANCHOR_SYMBOLS,
}

# Discovery-ablation styling (used where one algo is shown under several ablations).
DISC_DASH: Dict[str, str] = {
    "zero": "dash",
    "resample": "solid",
    "mean": "dot",
    "anchor": "solid",
}
DISC_SYMBOL: Dict[str, str] = {
    "zero": "square",
    "resample": "circle",
    "mean": "diamond",
    "anchor": "x-thin",
}


def _assign(names: Iterable[str], fixed: Dict[str, str], slots: List[str]) -> Dict[str, str]:
    """Fixed assignment where known, then unused slots in name order (never cycled by rank)."""
    out, taken = {}, set()
    extras = []
    for n in sorted(set(names)):
        if n in fixed:
            out[n] = fixed[n]
            taken.add(fixed[n])
        else:
            extras.append(n)
    free = [s for s in slots if s not in taken]
    for i, n in enumerate(extras):
        out[n] = free[i] if i < len(free) else INK_MUTED
    return out


def algo_colors(names: Iterable[str]) -> Dict[str, str]:
    return _assign(names, _FIXED, _SLOTS)


def algo_symbols(names: Iterable[str]) -> Dict[str, str]:
    return _assign(names, _FIXED_SYMBOLS, _SYMBOLS)


def is_anchor(name: str) -> bool:
    return name in ANCHOR_COLORS


def apply_theme(
    fig: go.Figure,
    *,
    width: int,
    height: int,
    legend_title: str = "",
    margin_left: int = 90,
) -> go.Figure:
    """White surface, hairline grid, recessive axes, legend above the plot."""
    fig.update_layout(
        template="none",
        font=dict(family=FONT_FAMILY, size=13, color=INK_SECONDARY),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        width=width,
        height=height,
        # Top margin clears the legend, which sits above the plot (and above any
        # facet titles) rather than stealing width from it.
        margin=dict(l=margin_left, r=28, t=118, b=64),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12), bordercolor=AXIS),
        legend=dict(
            title=dict(text=legend_title, font=dict(size=12, color=INK_MUTED)),
            orientation="h",
            yanchor="bottom",
            y=1.07,
            xanchor="left",
            x=0.0,
            font=dict(size=12, color=INK_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            itemsizing="constant",
            tracegroupgap=8,
        ),
    )
    axis = dict(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        ticklen=4,
        tickcolor=AXIS,
        tickfont=dict(size=11, color=INK_MUTED),
        title_font=dict(size=13, color=INK_SECONDARY),
        automargin=True,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


TITLE_SIZE = 19
SUBTITLE_SIZE = 12.5


def title_markup(title: str, subtitle: str = "") -> str:
    """A title line over a muted subtitle line, as one block of markup.

    Returned as markup rather than set on the figure so the heatmaps, which place
    their title in an annotation, use the same two tiers as everything else.
    """
    head = f"<b>{title}</b>"
    if not subtitle:
        return head
    return (
        f"{head}<br>"
        f'<span style="font-size:{SUBTITLE_SIZE}px;color:{INK_MUTED}">{subtitle}</span>'
    )


def header(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    *,
    top: int = 150,
    right: int = 28,
    legend_y: float = 1.03,
) -> go.Figure:
    """Centred title block above the legend.

    The subtitle says what is measured and how to read the marks. Findings belong in
    the caption, not here.
    """
    # Plotly places a multi-line title from the first line's baseline, so a fixed y in
    # paper units clips the ascenders on tall figures. Hold the gap constant in pixels.
    height = fig.layout.height or 600
    fig.update_layout(
        margin=dict(t=top, r=right),
        title=dict(
            text=title_markup(title, subtitle),
            font=dict(size=TITLE_SIZE, color=INK_PRIMARY, family=FONT_FAMILY),
            # Centred on the plot area, not the canvas, so a legend column on the right
            # does not push the title off the data it heads.
            x=0.5,
            xref="paper",
            xanchor="center",
            y=1 - (TITLE_SIZE + 15) / height,
            yanchor="top",
        ),
        legend=dict(y=legend_y),
    )
    return fig


def style_marks(fig: go.Figure, *, line_width: int = 2, marker_size: int = 7) -> go.Figure:
    """Thin lines, markers with a surface ring so overlaps stay readable."""
    fig.for_each_trace(
        lambda tr: tr.update(
            line=dict(width=line_width),
            marker=dict(size=marker_size, line=dict(width=1.4, color=SURFACE)),
        )
        if tr.type in ("scatter", "scattergl") and tr.mode and "lines" in tr.mode
        else None
    )
    return fig


def strip_facet_prefix(fig: go.Figure) -> go.Figure:
    """px writes facet titles as 'Task=IOI'; keep only the value."""
    fig.for_each_annotation(
        lambda a: a.update(
            text=f"<b>{a.text.split('=')[-1]}</b>",
            font=dict(size=14, color=INK_PRIMARY, family=FONT_FAMILY),
        )
    )
    return fig
