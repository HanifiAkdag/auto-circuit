"""Reproducible structural and functional stability exhibits for Results."""

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
    "functional.csv": "b41262bd64051a4614fb6dd22cfaddf0a997da181abe9160980ebc9505a3f145",
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
        r"\caption{Pairwise edge-set Jaccard similarity across five joint-seed runs at the task-specific comparison size (739 edges for the IOI task and 235 for the Greater-Than task). Columns denote discovery ablation. Jaccard entries give the average and minimum--maximum of ten dependent run pairs. The indented ACDC rows give average attained edge counts and their five-run ranges. Other methods attain the target size exactly. All ranges are descriptive.}",
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
        r"\caption{Pairwise edge-set Jaccard under joint and isolated seed variation. Entries give the average $\pm$ population SD of ten dependent pairs from five runs, at requested sizes of 739 edges for the IOI task and 235 for the Greater-Than task. Threshold ties give 739--741 edges for IOI Edge Pruning with resample discovery ablation under PyTorch-only variation, and 235--236 edges for Greater-Than Edge Pruning with mean discovery ablation under data-only variation. Both seeds vary in the joint condition; data-only fixes the PyTorch seed, and PyTorch-only fixes the dataset seed.}",
        r"\label{tab:seed-condition-jaccard}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}lllccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Task} & \multirow{2}{*}{Method} & \shortstack[l]{Discovery\\ablation} & \multicolumn{3}{c}{Jaccard (average $\pm$ SD)} \\",
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


def structure_function_records(joint, quality):
    from experiments.thesis.stability_plots import _spearman

    metrics = ("Clean KL Div", "Logit Diff Percent", "Correct Answer Greater Than Incorrect Percent")
    selected, summaries = [], []
    assert len(joint) == 24
    for task, (label, target) in TASKS.items():
        cells = [r for r in joint if r["task"] == task]
        assert len(cells) == 12
        assert {(r["algo"], r["discovery_ablation"]) for r in cells} == {
            (method, ablation) for method in METHODS for ablation in ABLATIONS
        }
        for metric in metrics if label == "IOI" else metrics[:2]:
            points = []
            for cell in cells:
                assert cell["condition"] == "joint" and int(cell["n_seeds"]) == 5
                assert int(cell["n_pairs"]) == 10 and int(cell["target"]) == target
                matches = [r for r in quality if (r["task"], r["algo"], r["ablation"], r["metric"]) ==
                           (task, cell["algo"], cell["discovery_ablation"], metric)]
                assert len(matches) == 1, (task, cell["algo"], cell["discovery_ablation"], metric)
                row = matches[0]
                assert row["selector"] == cell["selector"]
                assert float(row["budget_label"]) == float(cell["budget_label"])
                assert float(row["realised_mean"]) == float(cell["realised_mean"])
                assert int(row["n_seeds"]) == 5 and int(row["target_edges"]) == target
                points.append(dict(task=task, algo=cell["algo"],
                                   discovery_ablation=cell["discovery_ablation"], condition="joint",
                                   eval_ablation=cell["discovery_ablation"], target_edges=target,
                                   selector=cell["selector"], budget_label=cell["budget_label"],
                                   realised_mean=float(cell["realised_mean"]), metric=metric,
                                   n_seeds=5, n_pairs=10, jaccard_mean=float(cell["jaccard_mean"]),
                                   metric_mean=float(row["metric_mean"]), metric_sd=float(row["metric_sd"])))
            rho = _spearman([r["jaccard_mean"] for r in points], [r["metric_sd"] for r in points])
            assert rho is not None
            selected.extend(points)
            summaries.append(dict(task=task, task_label=label, metric=metric, n_cells=12, rho=rho))
    assert len(selected) == 60 and len(summaries) == 5
    return selected, summaries


def write_structure_function(joint):
    quality_path = OUTPUT.parent / "controlled_comparison/matched_quality.csv"
    quality_manifest = quality_path.with_name("matched_quality_manifest.json")
    provenance = json.loads(quality_manifest.read_text())
    source_meta = json.loads((SOURCE / 'manifest.json').read_text())
    if source_meta.get('evaluation_protocol_id'):
        from .evaluation_protocol import require_same_protocol
        require_same_protocol(source_meta, provenance)
    assert _hash(ROOT / provenance["source"]) == provenance["source_sha256"]
    assert _hash(ROOT / provenance["source_manifest"]) == provenance["source_manifest_sha256"]
    selected, summary = structure_function_records(joint, _rows(quality_path))
    for name, rows in (("structure_function_selected.csv", selected),
                       ("structure_function_summary.csv", summary)):
        with (OUTPUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lookup = {(r["task_label"], r["metric"]): r["rho"] for r in summary}
    lines = [r"\begin{table}[!ht]", r"\centering", r"\small",
             r"\caption{Descriptive Spearman correlations between average pairwise Jaccard and across-run population SD of each functional metric. Each coefficient compares 12 method--discovery-ablation cells under joint seed variation and matched evaluation ablation, at requested sizes of 739 edges for the IOI task and 235 for the Greater-Than task. ACDC uses the same nearest attained thresholds as the preceding tables. These are cell-level associations, not pairwise output-similarity tests. Correct-over-incorrect accuracy is not reported for the Greater-Than task.}",
             r"\label{tab:structure-function-association}", r"\begin{tabular}{@{}lccc@{}}",
             r"\toprule", r"& \multicolumn{3}{c}{$\rho$ with across-run SD} \\",
             r"\cmidrule(lr){2-4}",
             r"Task & KL divergence & \shortstack[c]{Logit difference\\percent} & \shortstack[c]{Correct-over-incorrect\\accuracy} \\",
             r"\midrule"]
    for label in ("IOI", "Greater-Than"):
        metrics = ("Clean KL Div", "Logit Diff Percent", "Correct Answer Greater Than Incorrect Percent")
        values = [f"${lookup[(label, m)]:.3f}$" if (label, m) in lookup else "--" for m in metrics]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUTPUT / "structure_function_table.tex").write_text("\n".join(lines) + "\n")
    names = ("structure_function_selected.csv", "structure_function_summary.csv", "structure_function_table.tex")
    inputs = (SOURCE / "functional.csv", SOURCE / "structural.csv", SOURCE / "manifest.json",
              OUTPUT / "within_ablation_selected.csv", quality_path, quality_manifest)
    (OUTPUT / "structure_function_manifest.json").write_text(json.dumps({
        "evaluation_protocol_id": source_meta.get('evaluation_protocol_id'),
        "inputs": {str(p.relative_to(ROOT)): _hash(p) for p in inputs},
        "selection": "Joint five-run cells; matched discovery/evaluation ablations; identical selected thresholds and budgets to quality and within-ablation tables; Random and manual excluded.",
        "measure": "Spearman with average ranks for ties; population metric SD; descriptive across 12 cells per task/metric.",
        "rows": len(selected), "coefficients": len(summary),
        "outputs": {name: _hash(OUTPUT / name) for name in names},
    }, indent=2) + "\n")


def main(source_dir=None, output_root=None):
    global SOURCE, OUTPUT
    if source_dir is not None:
        if output_root is None:
            raise ValueError('An alternate source requires separate output artifacts')
        SOURCE = (ROOT / source_dir).resolve()
        OUTPUT = (ROOT / output_root).resolve() / 'stability'
    actual = {name: _hash(SOURCE / name) for name in SOURCE_HASHES}
    if source_dir is None:
        assert actual == SOURCE_HASHES, actual
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert manifest["seeds"] == [42, 43, 44, 45, 46]
    if source_dir is None:
        assert manifest["discovery_cache_dir"] == ".prune_scores/final/benchmark"
    summary, structural = _rows(SOURCE / "summary.csv"), _rows(SOURCE / "structural.csv")
    if source_dir is None:
        assert len(summary) == 96 and len(structural) == 2588
    else:
        assert manifest['complete'] and manifest['evaluation_protocol_id']
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
    write_structure_function(joint)
    _condition_figure(conditions, OUTPUT)
    outputs = ("within_ablation_selected.csv", "seed_condition_selected.csv",
               "within_ablation_table.tex", "seed_condition_table.tex",
               "seed_condition_jaccard.pdf", "seed_condition_jaccard.png")
    (OUTPUT / "exhibit_manifest.json").write_text(json.dumps({
        "evaluation_protocol_id": manifest.get('evaluation_protocol_id'),
        "section": "Results 4.3.1", "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": actual, "source_row_counts": {"summary": len(summary), "structural": len(structural)},
        "seeds": manifest["seeds"], "target_edges": {label: target for label, target in TASKS.values()},
        "selected_cells": {"joint": len(joint), "seed_conditions": len(conditions)},
        "output_sha256": {name: _hash(OUTPUT / name) for name in outputs},
    }, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    main(args.source_dir, args.output_root)
