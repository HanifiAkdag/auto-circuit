"""Plot final cross-ablation circuit evaluations and write exact appendix tables.

Run on a CPU compute node:
    srun --mem=8G python -m experiments.thesis.results_chapter_evaluation_ablation
"""
from __future__ import annotations

import csv
import hashlib
import json
from math import isclose
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/stability/stability_v5_2026-09-19/functional.csv"
SOURCE_MANIFEST = SOURCE.with_name("manifest.json")
SELECTION = ROOT / "results/results_chapter/controlled_comparison/matched_quality.csv"
SELECTION_MANIFEST = SELECTION.with_name("matched_quality_manifest.json")
OUT = ROOT / "results/results_chapter/evaluation_ablation"
IMAGES = ROOT / "thesis-overleaf/Thesis/Images"
APPENDIX = ROOT / "thesis-overleaf/Thesis/chapters/appendix.tex"
TABLES_BEGIN = "% BEGIN GENERATED EVALUATION ABLATION TABLES"
TABLES_END = "% END GENERATED EVALUATION ABLATION TABLES"
TASKS = {
    "Indirect Object Identification": ("ioi", "IOI", 739,
        ("Clean KL Div", "Logit Diff Percent", "Correct Answer Greater Than Incorrect Percent")),
    "Greaterthan": ("greater_than", "Greater-Than", 235,
        ("Clean KL Div", "Logit Diff Percent")),
}
METRIC_LABELS = {
    "Clean KL Div": "Clean KL divergence",
    "Logit Diff Percent": "Logit-difference recovery (%)",
    "Correct Answer Greater Than Incorrect Percent": "Correct-over-incorrect accuracy (%)",
}
AXIS_LABELS = METRIC_LABELS
METRIC_SLUGS = {
    "Clean KL Div": "clean_kl",
    "Logit Diff Percent": "logit_difference",
    "Correct Answer Greater Than Incorrect Percent": "accuracy",
}
ABLATIONS = ("zero", "resample", "mean")
METHODS = ("ACDC", "EAP", "EP", "DiscoGP")
METHOD_LABELS = {"ACDC": "ACDC", "EAP": "EAP", "EP": "Edge Pruning", "DiscoGP": "DiscoGP"}
EVAL_STYLE = {
    "zero": ("#B44C46", "circle"),
    "resample": ("#336B9E", "diamond"),
    "mean": ("#32835F", "square"),
}
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select() -> list[dict[str, object]]:
    selection_meta = json.loads(SELECTION_MANIFEST.read_text())
    assert sha(ROOT / selection_meta["source"]) == selection_meta["source_sha256"]
    assert sha(ROOT / selection_meta["source_manifest"]) == selection_meta["source_manifest_sha256"]
    source_meta = json.loads(SOURCE_MANIFEST.read_text())
    assert source_meta["seeds"] == [42, 43, 44, 45, 46]
    assert source_meta["discovery_cache_dir"] == ".prune_scores/final/benchmark"
    all_rows = read_csv(SOURCE)
    selection = read_csv(SELECTION)
    selected = []
    for task, (_, _, target, metrics) in TASKS.items():
        for discovery in ABLATIONS:
            for method in METHODS:
                for metric in metrics:
                    chosen = [row for row in selection if row["task"] == task
                              and row["ablation"] == discovery and row["algo"] == method
                              and row["metric"] == metric]
                    assert len(chosen) == 1, (task, discovery, method, metric)
                    chosen = chosen[0]
                    assert int(chosen["target_edges"]) == target
                    for evaluation in ABLATIONS:
                        matches = [row for row in all_rows if row["task"] == task
                                   and row["algo"] == method
                                   and row["discovery_ablation"] == discovery
                                   and row["eval_ablation"] == evaluation
                                   and row["condition"] == "joint"
                                   and row["measure"] == "faithfulness"
                                   and row["metric"] == metric
                                   and row["selector"] == chosen["selector"]
                                   and row["budget_label"] == chosen["budget_label"]]
                        assert len(matches) == 1, (task, discovery, method, metric, evaluation)
                        row = matches[0]
                        assert int(row["n_seeds"]) == 5
                        assert len(row["metric_values"].split(";")) == 5
                        assert isclose(float(row["realised_mean"]), float(chosen["realised_mean"]), abs_tol=1e-9)
                        if evaluation == discovery:
                            assert isclose(float(row["metric_mean"]), float(chosen["metric_mean"]), abs_tol=1e-12)
                            assert isclose(float(row["metric_sd"]), float(chosen["metric_sd"]), abs_tol=1e-12)
                        selected.append({key: row[key] for key in (
                            "task", "algo", "discovery_ablation", "eval_ablation", "metric",
                            "selector", "budget_label", "realised_mean", "n_seeds", "metric_mean", "metric_sd",
                            "metric_min", "metric_max", "metric_values")})
    assert len(selected) == 180
    return selected


def render_figure(rows: list[dict[str, object]], task: str) -> tuple[Path, Path]:
    slug, title, _, metrics = TASKS[task]
    width = 1160 if len(metrics) == 3 else 920
    plot_width = width - 245
    table_left = -135 / plot_width
    strip_left, strip_right = -195 / plot_width, -157 / plot_width
    headings = {
        "Clean KL Div": "Clean KL divergence",
        "Logit Diff Percent": "Logit difference (%)",
        "Correct Answer Greater Than Incorrect Percent": "Correct-over-incorrect<br>accuracy (%)",
    }
    fig = make_subplots(rows=1, cols=len(metrics), shared_yaxes=True,
                        horizontal_spacing=0.065,
                        subplot_titles=[headings[metric] for metric in metrics])
    offsets = {"zero": 0.18, "resample": 0.0, "mean": -0.18}
    row_y = {
        (discovery, method): 12 - 5.6 * discovery_idx - method_idx
        for discovery_idx, discovery in enumerate(ABLATIONS)
        for method_idx, method in enumerate(METHODS)
    }
    tickvals = [row_y[(discovery, method)] for discovery in ABLATIONS for method in METHODS]
    ticktext = [METHOD_LABELS[method] for discovery in ABLATIONS for method in METHODS]
    ranges = {}
    for metric in metrics:
        values = [float(r["metric_mean"]) + sign * float(r["metric_sd"])
                  for r in rows if r["task"] == task and r["metric"] == metric
                  for sign in (-1, 1)]
        lower, upper = min(values), max(values)
        pad = max((upper - lower) * 0.07, 0.02)
        ranges[metric] = [-pad if metric == "Clean KL Div"
                          else (0 if metric == "Correct Answer Greater Than Incorrect Percent"
                                else lower - pad),
                          upper + pad]
        if metric == "Correct Answer Greater Than Incorrect Percent":
            ranges[metric][1] = max(100.0, ranges[metric][1])

    for col, metric in enumerate(metrics, 1):
        for evaluation in ABLATIONS:
            group = [r for r in rows if r["task"] == task and r["metric"] == metric
                     and r["eval_ablation"] == evaluation]
            group.sort(key=lambda r: (-ABLATIONS.index(r["discovery_ablation"]),
                                      -METHODS.index(r["algo"])))
            color, symbol = EVAL_STYLE[evaluation]
            fig.add_trace(go.Scatter(
                x=[float(r["metric_mean"]) for r in group],
                y=[row_y[(r["discovery_ablation"], r["algo"])] + offsets[evaluation]
                   for r in group],
                mode="markers", name=evaluation.capitalize(), legendgroup=evaluation,
                showlegend=col == 1,
                marker=dict(color=color, symbol=symbol, size=8,
                            line=dict(color="white", width=0.6)),
                error_x=dict(type="data", array=[float(r["metric_sd"]) for r in group],
                             symmetric=True, color=color, thickness=1.0, width=2),
                hovertemplate="%{x:.3f}<extra>%{fullData.name} evaluation</extra>",
            ), row=1, col=col)

        fig.update_xaxes(range=ranges[metric], showline=True, linecolor="#333333",
                         linewidth=1, ticks="outside", tickfont=dict(size=15),
                         showgrid=True, gridcolor="#E4E6E8", zeroline=False,
                         row=1, col=col)
        if task == "Indirect Object Identification" and metric == "Logit Diff Percent":
            fig.update_xaxes(tickvals=[0, 100, 200, 300, 400], row=1, col=col)
        fig.update_yaxes(range=[-2.85, 12.65], tickvals=tickvals, ticktext=ticktext,
                         showticklabels=col == 1, tickfont=dict(size=16), ticks="",
                         showgrid=False,
                         zeroline=False, showline=False, row=1, col=col)

    # Paper coordinates keep the table bands continuous across subplot gutters.
    for band_idx, y in enumerate(tickvals):
        if band_idx % 2 == 0:
            fig.add_shape(type="rect", xref="paper", yref="y",
                          x0=table_left, x1=1, y0=y - 0.48, y1=y + 0.48,
                          fillcolor="#F3F4F5", line_width=0, layer="below")
    for i, discovery in enumerate(ABLATIONS):
        y = 10.5 - 5.6 * i
        fig.add_annotation(x=(strip_left + strip_right) / 2, xref="paper", y=y, yref="y",
                           text=f"<b>{discovery.capitalize()} discovery</b>", textangle=-90,
                           showarrow=False, xanchor="center", yanchor="middle",
                           font=dict(size=13, color="#303030"))

    for annotation in fig.layout.annotations[:len(metrics)]:
        annotation.update(font=dict(size=17, color="#222222"), y=1.04)
    fig.update_layout(template="simple_white", width=width,
                      height=640, margin=dict(l=210, r=35, t=115, b=55),
                      font=dict(family="Arial", color="#222222", size=15),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.16, yanchor="bottom",
                                  title="Evaluation ablation: ", font=dict(size=15),
                                  title_font=dict(size=15), bgcolor="rgba(0,0,0,0)"),
                      paper_bgcolor="white", plot_bgcolor="rgba(0,0,0,0)")
    pdf = IMAGES / f"{slug}_evaluation_ablation.pdf"
    png = OUT / f"{slug}_evaluation_ablation.png"
    exported = OUT / f"{slug}_evaluation_ablation_export.pdf"
    fig.write_image(str(exported))
    pages = PdfReader(exported).pages
    assert len(pages) in (1, 2)
    if len(pages) == 2:
        assert not pages[1].extract_text() and len(pages[1].get_contents().get_data()) < 500
    writer = PdfWriter()
    writer.add_page(pages[0])
    with pdf.open("wb") as handle:
        writer.write(handle)
    exported.unlink()
    fig.write_image(str(png), scale=1.4)
    return pdf, png


def fmt(metric: str, row: dict[str, object]) -> str:
    mean, sd = float(row["metric_mean"]), float(row["metric_sd"])
    if metric == "Clean KL Div":
        decimals = 3 if max(abs(mean), sd) < 1 or 0 < sd < 0.005 else 2
    else:
        decimals = 1
    return f"${mean:.{decimals}f}\\pm{sd:.{decimals}f}$"


def render_table(rows: list[dict[str, object]], task: str, metric: str) -> str:
    slug, title, _, _ = TASKS[task]
    caption_metric = METRIC_LABELS[metric].replace("%", r"\%")
    lines = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
             "\\caption{" + title + " " + caption_metric[0].lower() + caption_metric[1:]
             + " under each evaluation ablation. Values are five-run average $\\pm$ population SD.}",
             "\\label{tab:" + slug.replace("_", "-") + "-evaluation-" + METRIC_SLUGS[metric].replace("_", "-") + "}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}", r"\toprule",
             "Discovery & Method & Zero & Resample & Mean \\\\", r"\midrule"]
    for discovery in ABLATIONS:
        for i, method in enumerate(METHODS):
            values = []
            for evaluation in ABLATIONS:
                matches = [r for r in rows if r["task"] == task and r["metric"] == metric
                           and r["discovery_ablation"] == discovery and r["algo"] == method
                           and r["eval_ablation"] == evaluation]
                assert len(matches) == 1
                values.append(fmt(metric, matches[0]))
            label = discovery.capitalize() if i == 0 else ""
            lines.append(f"{label} & {METHOD_LABELS[method]} & " + " & ".join(values) + " \\\\")
        if discovery != ABLATIONS[-1]:
            lines.append(r"\addlinespace[4pt]")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    return "\n".join(lines)


def update_appendix(tables: list[str]) -> None:
    appendix = APPENDIX.read_text()
    assert appendix.count(TABLES_BEGIN) == 1
    assert appendix.count(TABLES_END) == 1
    assert len(tables) == 5
    start = appendix.index(TABLES_BEGIN) + len(TABLES_BEGIN)
    end = appendix.index(TABLES_END, start)
    table_block = "\n\n".join(table.rstrip() for table in tables)
    appendix = (appendix[:start].rstrip("\n") + "\n\n" + table_block + "\n\n"
                + appendix[end:].lstrip("\n"))
    APPENDIX.write_text(appendix)


def main() -> None:
    for directory in (OUT, IMAGES):
        directory.mkdir(parents=True, exist_ok=True)
    rows = select()
    csv_path = OUT / "crossed_evaluation_values.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs = [csv_path]
    tables = []
    for task, (_, _, _, metrics) in TASKS.items():
        outputs.extend(render_figure(rows, task))
        tables.extend(render_table(rows, task, metric) for metric in metrics)
    update_appendix(tables)
    outputs.append(APPENDIX)
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "selection": str(SELECTION.relative_to(ROOT)), "selection_sha256": sha(SELECTION),
        "selection_manifest": str(SELECTION_MANIFEST.relative_to(ROOT)),
        "selection_manifest_sha256": sha(SELECTION_MANIFEST),
        "discovery_cache_dir": ".prune_scores/final/benchmark",
        "seeds": [42, 43, 44, 45, 46], "rows": len(rows),
        "selection_rule": "Same selector, budget, and five circuits as the matched-quality table; only evaluation ablation varies.",
        "outputs": {str(path.relative_to(ROOT)): sha(path) for path in outputs},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} crossed-evaluation rows, two figures, and five appendix tables")


if __name__ == "__main__":
    main()
