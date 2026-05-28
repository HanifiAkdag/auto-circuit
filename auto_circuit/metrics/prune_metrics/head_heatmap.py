import re
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import plotly.graph_objects as go

from auto_circuit.types import PruneScores
from auto_circuit.utils.patchable_model import PatchableModel


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def aggregate_scores_by_head(
    prune_scores: PruneScores,
    model: PatchableModel,
    aggregation: Literal["sum", "max", "mean"] = "sum",
) -> torch.Tensor:
    """Aggregate edge prune scores into a (n_transformer_layers, n_heads) grid.

    AutoCircuit attention layers are odd integers (1, 3, 5 … for GPT-2 Small).
    Transformer layer index = (dest.layer - 1) // 2.
    """
    attn_edges = [e for e in model.edges if e.dest.head_idx is not None]
    if not attn_edges:
        raise ValueError("No attention edges found in model.")

    n_layers = max((e.dest.layer - 1) // 2 for e in attn_edges) + 1
    n_heads = max(e.dest.head_idx for e in attn_edges) + 1  # type: ignore[type-var]

    totals = torch.zeros(n_layers, n_heads)
    counts = torch.zeros(n_layers, n_heads)
    maxes = torch.full((n_layers, n_heads), float("-inf"))

    for e in attn_edges:
        layer = (e.dest.layer - 1) // 2
        head = e.dest.head_idx
        score = e.prune_score(prune_scores).item()
        totals[layer, head] += score
        counts[layer, head] += 1
        if score > maxes[layer, head]:
            maxes[layer, head] = score

    if aggregation == "sum":
        return totals
    elif aggregation == "max":
        maxes[maxes == float("-inf")] = 0.0
        return maxes
    else:  # mean
        safe_counts = counts.clamp(min=1)
        return totals / safe_counts


def head_importance_heatmap(
    prune_scores: PruneScores,
    model: PatchableModel,
    algo_name: str,
    task_name: str,
    aggregation: Literal["sum", "max", "mean"] = "sum",
    output_dir: Optional[Path] = None,
    file_formats: Tuple[str, ...] = ("html", "png"),
) -> go.Figure:
    """Layer × head heatmap of aggregated edge prune scores.

    Styled after Figure 3(b) of the IOI paper (Interpretability in the Wild,
    arxiv 2211.00593). Works with any task/algorithm combination whose prune
    scores have already been computed.

    Args:
        prune_scores: Edge importance scores from any prune algorithm.
        model: The patchable model the scores were computed on.
        algo_name: Human-readable algorithm name (used in title and file path).
        task_name: Human-readable task name (used in title and file path).
        aggregation: How to combine multiple edges feeding the same head.
        output_dir: If provided, figures are saved under
            ``{output_dir}/{task_slug}/{algo_slug}/head_importance_{aggregation}.{ext}``.
        file_formats: Which formats to save. Use ``("html",)`` to skip kaleido.

    Returns:
        Plotly figure (always returned, regardless of whether saving is requested).
    """
    grid = aggregate_scores_by_head(prune_scores, model, aggregation)
    n_layers, n_heads = grid.shape

    fig = go.Figure(
        go.Heatmap(
            z=grid.tolist(),
            x=list(range(n_heads)),
            y=list(range(n_layers)),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title=f"Edge Score<br>({aggregation})"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=f"{algo_name} — {task_name}: Head Importance ({aggregation})",
        xaxis=dict(title="Head", tickmode="linear", dtick=1),
        yaxis=dict(title="Layer", tickmode="linear", dtick=1, autorange="reversed"),
        width=500,
        height=450,
        template="plotly",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    if output_dir is not None:
        save_dir = Path(output_dir) / _slug(task_name) / _slug(algo_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = save_dir / f"head_importance_{aggregation}"
        for fmt in file_formats:
            if fmt == "html":
                fig.write_html(str(stem) + ".html")
            else:
                fig.write_image(str(stem) + f".{fmt}")

    return fig
