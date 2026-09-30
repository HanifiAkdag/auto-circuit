"""Reproducible exhibits for Results Section 4.3.1."""

import csv
import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from auto_circuit.utils.plot_theme import FONT_FAMILY, GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SLOTS


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/stability/stability_v5_2026-09-19"
OUTPUT = ROOT / "results/results_chapter/stability"
SOURCE_HASHES = {
    "manifest.json": "d2b3ebb0f3b72e06263c5b839bdb0b3cf930155e3ea2eb475e7aae6064c8e13a",
    "structural.csv": "ef51c93308f91bfeb2dfa689e21bb721b1b8045f659c884ce4284f9bb0b51ff9",
    "summary.csv": "eed0490613bd253a170d35345c0780a20ddd2faa47ea4bc7c6a0af8039c1ec9e",
}
TASKS = {"Indirect Object Identification": ("IOI", 739), "Greaterthan": ("Greater-Than", 235)}
METHODS = ("ACDC", "EAP", "EP", "DiscoGP")
ABLATIONS = ("zero", "resample", "mean")
CONDITIONS = ("joint", "data", "init")
FIELDS = (
    "task", "task_label", "algo", "discovery_ablation", "condition", "target",
    "selector", "budget_label", "budget_value", "realised_mean", "realised_min",
    "realised_max", "jaccard_mean", "jaccard_sd", "jaccard_min", "jaccard_max", "n_seeds", "n_pairs",
)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def select_cell(summary_rows, structural_rows, task, method, ablation, condition, target):
    """Select the published budget and verify it against its structural sweep."""
    key = (task, method, ablation, condition)
    matches = [r for r in summary_rows if (
        r["task"], r["algo"], r["discovery_ablation"], r["condition"]
    ) == key]
    assert len(matches) == 1, key
    summary = matches[0]
    assert (int(summary["n_seeds"]), int(summary["n_pairs"]), int(summary["true_edge_count"])) == (5, 10, target), key
    group = [r for r in structural_rows if (
        r["task"], r["algo"], r["discovery_ablation"], r["condition"]
    ) == key]
    selected = [r for r in group if (
        r["selector"], r["budget_label"]
    ) == (summary["selector"], summary["budget_label"])]
    assert len(selected) == 1, key
    row = selected[0]
    for field in ("budget_value", "realised_mean", "realised_min", "realised_max",
                  "jaccard_mean", "jaccard_sd", "jaccard_min", "jaccard_max"):
        assert float(summary[field]) == float(row[field]), (key, field)
    if method == "ACDC":
        assert row["selector"] == "threshold", key
        nearest = min(group, key=lambda r: abs(float(r["realised_mean"]) - target))
        assert row["budget_label"] == nearest["budget_label"], key
    else:
        assert row["selector"] == "top_k" and float(row["budget_value"]) == target, key
    return {
        "task": task, "task_label": TASKS.get(task, (task, target))[0],
        "algo": method, "discovery_ablation": ablation, "condition": condition,
        "target": target, "selector": row["selector"],
        "budget_label": row["budget_label"], "budget_value": float(row["budget_value"]),
        "realised_mean": float(row["realised_mean"]),
        "realised_min": int(row["realised_min"]), "realised_max": int(row["realised_max"]),
        "jaccard_mean": float(row["jaccard_mean"]),
        "jaccard_sd": float(row["jaccard_sd"]),
        "jaccard_min": float(row["jaccard_min"]),
        "jaccard_max": float(row["jaccard_max"]),
        "n_seeds": 5, "n_pairs": 10,
    }


def _write_csv(path, records):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _table(records):
    lookup = {(r["task_label"], r["discovery_ablation"], r["algo"]): r for r in records}
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\caption{Pairwise edge-set Jaccard similarity across five joint-seed runs at the task-specific comparison size (739 edges for IOI and 235 for Greater-Than). Columns denote discovery ablation. Jaccard entries give the average and minimum--maximum of ten dependent run pairs. The indented ACDC rows give average attained edge counts and their five-run ranges. Other methods attain the target size exactly. All ranges are descriptive.}",
        r"\label{tab:within-ablation-stability}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Method & Zero & Resample & Mean \\",
        r"\midrule",
    ]
    for task_label in ("IOI", "Greater-Than"):
        lines.append(r"\multicolumn{4}{@{}l}{\textbf{" + task_label + r"}} \\")
        for method in METHODS:
            cells = []
            for ablation in ABLATIONS:
                row = lookup[(task_label, ablation, method)]
                value = (f"${row['jaccard_mean']:.3f}"
                         f"\\,[{row['jaccard_min']:.3f},{row['jaccard_max']:.3f}]$")
                cells.append(value)
            method_label = "Edge Pruning" if method == "EP" else method
            lines.append(method_label + " & " + " & ".join(cells) + r" \\")
            if method == "ACDC":
                sizes = [lookup[(task_label, ablation, "ACDC")] for ablation in ABLATIONS]
                size_cells = [f"${r['realised_mean']:.0f}"
                              f"\\,[{r['realised_min']},{r['realised_max']}]$" for r in sizes]
                lines.append(r"\quad attained edges & " + " & ".join(size_cells) + r" \\")
                lines.append(r"\addlinespace[2pt]")
        if task_label == "IOI":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _jaccard_decimal(value):
    return f"{value:.3f}".removeprefix("0")


def _condition_table(records):
    lookup = {(r["task_label"], r["algo"], r["discovery_ablation"], r["condition"]): r
              for r in records}
    assert len(lookup) == len(records) == 36
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\caption{Pairwise edge-set Jaccard similarity under joint and isolated seed variation. Entries give the average $\pm$ population SD of ten dependent run pairs from five circuits at 739 edges (IOI) or 235 edges (Greater-Than). In the joint condition both seeds vary. The data-only condition fixes the PyTorch seed, and the PyTorch-only condition fixes the dataset seed.}",
        r"\label{tab:seed-condition-jaccard}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}cclccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Task} & \multirow{2}{*}{Method} & \multirow{2}{*}{Ablation} & \multicolumn{3}{c}{Jaccard (average $\pm$ SD)} \\",
        r"\cmidrule(lr){4-6}",
        r"& & & Both seeds & Data only & PyTorch only \\",
        r"\midrule",
    ]
    for task in ("IOI", "Greater-Than"):
        for method in ("EP", "DiscoGP"):
            for ablation in ABLATIONS:
                task_cell = rf"\multirow{{6}}{{*}}{{{task}}}" if method == "EP" and ablation == "zero" else ""
                method_cell = (r"\multirow{3}{*}{Edge Pruning}" if method == "EP" else
                               r"\multirow{3}{*}{DiscoGP}") if ablation == "zero" else ""
                values = [lookup[(task, method, ablation, condition)] for condition in CONDITIONS]
                lines.append(f"{task_cell} & {method_cell} & {ablation.capitalize()} & "
                             + " & ".join(f"${_jaccard_decimal(value['jaccard_mean'])}"
                                          f"\\pm{_jaccard_decimal(value['jaccard_sd'])}$"
                                          for value in values) + r" \\")
            if method == "EP":
                lines.append(r"\cmidrule(lr){2-6}")
        if task == "IOI":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _condition_figure(records, output):
    lookup = {(r["task_label"], r["discovery_ablation"], r["algo"], r["condition"]): r
              for r in records}
    colors = {"joint": INK_PRIMARY, "data": SLOTS[0], "init": SLOTS[1]}
    symbols = {"joint": "circle", "data": "square", "init": "diamond"}
    names = {"joint": "Both seeds", "data": "Dataset seed only", "init": "PyTorch seed only"}
    offsets = {"joint": -0.16, "data": 0.0, "init": 0.16}
    panels = (("IOI", "EP"), ("Greater-Than", "EP"),
              ("IOI", "DiscoGP"), ("Greater-Than", "DiscoGP"))
    titles = tuple(f"{task} · {'Edge Pruning' if method == 'EP' else 'DiscoGP'}"
                   for task, method in panels)
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True,
                        horizontal_spacing=0.10, vertical_spacing=0.16,
                        subplot_titles=titles)
    for panel_index, (task, method) in enumerate(panels):
        row, col = divmod(panel_index, 2)
        for condition in CONDITIONS:
            selected = [lookup[(task, ablation, method, condition)] for ablation in ABLATIONS]
            fig.add_trace(go.Scatter(
                x=[index + offsets[condition] for index in range(3)],
                y=[r["jaccard_mean"] for r in selected],
                mode="markers", name=names[condition], legendgroup=condition,
                showlegend=panel_index == 0,
                marker=dict(symbol=symbols[condition], color=colors[condition], size=9,
                            line=dict(width=0.5, color="white")),
                error_y=dict(type="data", symmetric=False,
                             array=[r["jaccard_max"] - r["jaccard_mean"] for r in selected],
                             arrayminus=[r["jaccard_mean"] - r["jaccard_min"] for r in selected],
                             color=colors[condition], thickness=1.1, width=2.5),
                hovertemplate="Jaccard %{y:.3f}<extra>%{fullData.name}</extra>",
            ), row=row + 1, col=col + 1)
    fig.update_layout(
        template="none", width=1000, height=690, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=86, r=26, t=70, b=130),
        font=dict(family=FONT_FAMILY, size=16, color=INK_SECONDARY),
        legend=dict(orientation="h", x=0.5, y=-0.17, xanchor="center", yanchor="top",
                    font=dict(size=15, color=INK_SECONDARY), itemsizing="constant",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(range=[-.45, 2.45], tickvals=[0, 1, 2],
                     ticktext=["Zero", "Resample", "Mean"],
                     showgrid=False, zeroline=False, showline=False,
                     ticks="", tickfont=dict(size=15, color=INK_SECONDARY))
    fig.update_yaxes(range=[0, .72], tickvals=[0, .2, .4, .6],
                     tickformat=".1f", showgrid=True, gridcolor=GRID, gridwidth=1,
                     zeroline=False, showline=False, ticks="",
                     tickfont=dict(size=14, color=INK_MUTED))
    for row in (1, 2):
        fig.update_yaxes(title_text="Pairwise Jaccard", title_font=dict(size=16, color=INK_SECONDARY),
                         title_standoff=8, showticklabels=True, row=row, col=1)
        fig.update_yaxes(showticklabels=True, row=row, col=2)
    for col in (1, 2):
        fig.update_xaxes(title_text="Discovery ablation", title_font=dict(size=16, color=INK_SECONDARY),
                         title_standoff=10, row=2, col=col)
    for annotation, (row, col) in zip(fig.layout.annotations, ((1, 1), (1, 2), (2, 1), (2, 2))):
        domain = fig.get_subplot(row, col).xaxis.domain
        annotation.update(x=(domain[0] + domain[1]) / 2, xanchor="center",
                          font=dict(family=FONT_FAMILY, size=17, color=INK_PRIMARY))
    fig.write_image(output / "seed_condition_jaccard.pdf")
    fig.write_image(output / "seed_condition_jaccard.png", scale=2)


def main():
    actual = {name: _hash(SOURCE / name) for name in SOURCE_HASHES}
    assert actual == SOURCE_HASHES, actual
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert manifest["seeds"] == [42, 43, 44, 45, 46]
    assert manifest["discovery_cache_dir"] == ".prune_scores/final/benchmark"
    summary, structural = _rows(SOURCE / "summary.csv"), _rows(SOURCE / "structural.csv")
    assert len(summary) == 96 and len(structural) == 2588
    joint = [select_cell(summary, structural, task, method, ablation, "joint", target)
             for task, (_, target) in TASKS.items() for ablation in ABLATIONS for method in METHODS]
    conditions = [select_cell(summary, structural, task, method, ablation, condition, target)
                  for task, (_, target) in TASKS.items() for ablation in ABLATIONS
                  for method in ("EP", "DiscoGP") for condition in CONDITIONS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "within_ablation_selected.csv", joint)
    _write_csv(OUTPUT / "seed_condition_selected.csv", conditions)
    (OUTPUT / "within_ablation_table.tex").write_text(_table(joint))
    (OUTPUT / "seed_condition_table.tex").write_text(_condition_table(conditions))
    _condition_figure(conditions, OUTPUT)
    outputs = ("within_ablation_selected.csv", "seed_condition_selected.csv",
               "within_ablation_table.tex", "seed_condition_table.tex",
               "seed_condition_jaccard.pdf", "seed_condition_jaccard.png")
    (OUTPUT / "exhibit_manifest.json").write_text(json.dumps({
        "section": "Results 4.3.1", "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": actual, "source_row_counts": {"summary": len(summary), "structural": len(structural)},
        "seeds": manifest["seeds"], "target_edges": {label: target for label, target in TASKS.values()},
        "selected_cells": {"joint": len(joint), "seed_conditions": len(conditions)},
        "output_sha256": {name: _hash(OUTPUT / name) for name in outputs},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
