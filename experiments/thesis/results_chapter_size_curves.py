"""Render final five-run matched-ablation size curves for all tasks.

Run on a CPU compute node: srun --mem=8G python -m experiments.thesis.results_chapter_size_curves
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/stability/stability_v5_2026-09-19"
OUT = ROOT / "results/results_chapter"
SIZE_OUT = OUT / "size_curves"
INSPECTION = Path("/tmp/auto-circuit-results-inspection/size_curves")
ABLATIONS = ("zero", "resample", "mean")
TASKS = {
    "ioi": ("Indirect Object Identification", "IOI"),
    "gt": ("Greaterthan", "Greater-Than"),
    "docstring": ("Docstring", "Docstring"),
}
TASK_SLUGS = {"ioi": "ioi", "gt": "greater_than", "docstring": "docstring"}
METHODS = ("ACDC", "EAP", "EP", "DiscoGP")
STYLE = {
    "ACDC": ("#009E73", "ACDC"),
    "EAP": ("#7B2CBF", "EAP"),
    "EP": ("#0072B2", "Edge Pruning"),
    "DiscoGP": ("#D55E00", "DiscoGP"),
    "Random": ("#777777", "Random"),
    "Manual circuit": ("#222222", "Manual circuit"),
}
METRICS = (("Clean KL Div", "Clean KL divergence"),
           ("Logit Diff Percent", "Recovered logit difference (%)"),
           ("Correct Answer Greater Than Incorrect Percent",
            "Correct-over-incorrect accuracy (%)"))
TASK_METRICS = {
    "ioi": METRICS,
    "gt": METRICS[:2],
    "docstring": (METRICS[0], METRICS[2]),
}
MANUAL_BUDGETS = {"ioi": 739, "gt": 235, "docstring": 24}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metric_ylim(task_rows, metric):
    """Return one y-range for every ablation column in a metric row."""
    values = []
    for row in task_rows:
        if row["metric"] != metric:
            continue
        mean = float(row["metric_mean"])
        sd = float(row["metric_sd"])
        values.extend((mean - sd, mean + sd))
    assert values, f"No values found for {metric}"
    lower, upper = min(values), max(values)
    if metric == "Clean KL Div":
        lower = max(0.0, lower)
    if metric in {"Logit Diff Percent", "Correct Answer Greater Than Incorrect Percent"}:
        upper = max(upper, 100.0)
    padding = max((upper - lower) * 0.06, 1e-3)
    return [lower if lower == 0 else lower - padding, upper + padding]


def main():
    SIZE_OUT.mkdir(parents=True, exist_ok=True)
    INSPECTION.mkdir(parents=True, exist_ok=True)
    functional = SOURCE / "functional.csv"
    stability_manifest = SOURCE / "manifest.json"
    source_meta = json.loads(stability_manifest.read_text())
    assert source_meta["seeds"] == [42, 43, 44, 45, 46]
    assert len(source_meta["cells"]) == 96
    all_rows = read_csv(functional)
    manual_file = SIZE_OUT / "manual_baselines.csv"
    assert manual_file.exists(), "Run results_chapter_manual_baseline.py first"
    manual_rows = read_csv(manual_file)
    assert len(manual_rows) == sum(len(v) for v in TASK_METRICS.values())
    rows = []
    for task_key, (task_name, _) in TASKS.items():
        task_metrics = TASK_METRICS[task_key]
        rows.extend(dict(r, source_discovery_ablation=r["discovery_ablation"])
                     for r in all_rows
                     if r["task"] == task_name and r["algo"] in METHODS
                     and r["condition"] == "joint" and r["measure"] == "faithfulness"
                     and r["discovery_ablation"] == r["eval_ablation"]
                     and r["metric"] in dict(task_metrics)
                     and r["discovery_ablation"] in ABLATIONS)
        rows.extend(dict(r, discovery_ablation=r["eval_ablation"],
                         source_discovery_ablation="anchor")
                    for r in all_rows
                    if r["task"] == task_name and r["algo"] == "Random"
                    and r["condition"] == "joint" and r["measure"] == "faithfulness"
                    and r["discovery_ablation"] == "anchor"
                    and r["eval_ablation"] in ABLATIONS
                    and r["metric"] in dict(task_metrics))
        rows.extend(dict(r, condition="deterministic", source_discovery_ablation="manual",
                         metric_values=r["metric_mean"])
                    for r in manual_rows if r["task"] == task_name)
    assert rows and all(int(r["n_seeds"]) == (1 if r["algo"] == "Manual circuit" else 5)
                        for r in rows)
    for task_name, _ in TASKS.values():
        task_rows = [r for r in rows if r["task"] == task_name]
        seen = {(r["discovery_ablation"], r["algo"], r["metric"]) for r in task_rows}
        task_key = next(k for k, v in TASKS.items() if v[0] == task_name)
        task_metrics = TASK_METRICS[task_key]
        expected = (3 * len(METHODS) + 3) * len(task_metrics)
        expected += len(task_metrics)
        assert len(seen) == expected
        assert all((a, m, metric) in seen for a in ABLATIONS
                   for m in (*METHODS, "Random") for metric, _ in task_metrics)
        native_manual_ablation = "mean" if task_key == "ioi" else "resample"
        assert all((native_manual_ablation, "Manual circuit", metric) in seen
                   for metric, _ in task_metrics)
    for r in rows:
        r["realised_mean"] = float(r["realised_mean"])
        r["metric_mean"] = float(r["metric_mean"])
        r["metric_sd"] = float(r["metric_sd"])
        assert r["realised_mean"] > 0
        if r["algo"] == "ACDC" and r["selector"] == "threshold":
            assert r["budget_label"]
    order = (*METHODS, "Random", "Manual circuit")
    rows.sort(key=lambda r: (list(TASKS).index(next(k for k, v in TASKS.items()
                                                   if v[0] == r["task"])),
                             ABLATIONS.index(r["discovery_ablation"]),
                             order.index(r["algo"]), r["metric"], r["realised_mean"]))
    fields = ["task", "algo", "discovery_ablation", "eval_ablation", "condition",
              "source_discovery_ablation", "measure", "selector", "budget_label", "realised_mean", "metric",
              "n_seeds", "metric_mean", "metric_sd", "metric_min", "metric_max",
              "metric_values"]
    with (SIZE_OUT / "size_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)

    for task_key, (task_name, task_label) in TASKS.items():
        task_rows = [r for r in rows if r["task"] == task_name]
        task_metrics = TASK_METRICS[task_key]
        row_ranges = {metric: metric_ylim(task_rows, metric)
                      for metric, _ in task_metrics}
        manual_budget = MANUAL_BUDGETS[task_key]
        native_manual_ablation = "mean" if task_key == "ioi" else "resample"
        fig = make_subplots(rows=len(task_metrics), cols=3, horizontal_spacing=.075,
                            vertical_spacing=.09,
                            subplot_titles=[a.capitalize() for a in ABLATIONS]
                            + [""] * ((len(task_metrics) - 1) * len(ABLATIONS)))
        for i, (metric, ylabel) in enumerate(task_metrics):
            for j, ablation in enumerate(ABLATIONS):
                for method in (*METHODS, "Random", "Manual circuit"):
                    rr = sorted((r for r in task_rows if r["discovery_ablation"] == ablation
                             and r["algo"] == method and r["metric"] == metric),
                            key=lambda r: r["realised_mean"])
                    if not rr:
                        continue
                    color, label = STYLE[method]
                    x = [r["realised_mean"] for r in rr]
                    y = [r["metric_mean"] for r in rr]
                    if method == "Manual circuit":
                        fig.add_trace(go.Scatter(
                        x=x, y=y, mode="markers", name=label, legendgroup=method,
                        showlegend=i == 0 and ablation == native_manual_ablation,
                        marker=dict(symbol="star", size=11, color=color)),
                            row=i+1, col=j+1)
                        continue
                    upper = [r["metric_mean"] + r["metric_sd"] for r in rr]
                    lower = [r["metric_mean"] - r["metric_sd"] for r in rr]
                    rgb = tuple(int(color[k:k+2], 16) for k in (1, 3, 5))
                    fig.add_trace(go.Scatter(
                    x=x + x[::-1], y=upper + lower[::-1], fill="toself",
                    fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.14)",
                    line=dict(width=0), hoverinfo="skip", showlegend=False),
                        row=i+1, col=j+1)
                    fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines+markers", name=label,
                    legendgroup=method, showlegend=i == 0 and j == 0,
                    marker=dict(symbol="circle", size=4, color=color),
                    line=dict(color=color, width=2),
                    hovertemplate="%{x:.0f} edges<br>%{y:.3g}<extra>%{fullData.name}</extra>"
                    ), row=i+1, col=j+1)
                fig.add_vline(x=manual_budget, row=i+1, col=j+1, line_color="#999999",
                          line_width=.8, line_dash="dot", layer="below")
                if metric == "Logit Diff Percent":
                    fig.add_hline(y=100, row=i+1, col=j+1, line_color="#999999",
                              line_width=.8, line_dash="dot", layer="below")
                fig.update_xaxes(type="log", range=[.9, 4.65], row=i+1, col=j+1,
                             showline=True, linecolor="#666666", linewidth=.7,
                             ticks="outside", tickvals=[10, 100, 1000, 10000],
                             ticktext=["10", "100", "1,000", "10,000"],
                             showticklabels=i == len(task_metrics) - 1,
                             showgrid=False, tickfont=dict(size=15),
                             title_font=dict(size=15),
                                 title_text="Circuit size (edges)" if i == len(task_metrics) - 1 else None)
                fig.update_yaxes(title_text=ylabel if j == 0 else None, row=i+1, col=j+1,
                             range=row_ranges[metric],
                             showline=True, linecolor="#666666", linewidth=.7,
                             ticks="outside",
                             showticklabels=j == 0,
                             gridcolor="#e4e4e4", gridwidth=.5,
                             zeroline=False, tickfont=dict(size=15),
                                 title_font=dict(size=14))
        fig.update_layout(width=1060, height=900, margin=dict(l=83, r=28, t=40, b=116),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Arial", size=15, color="#222222"),
                      legend=dict(orientation="h", x=.5, xanchor="center", y=-.18,
                                  yanchor="top", font=dict(size=16), itemsizing="constant"))
        fig.update_annotations(font=dict(size=17), yshift=4)
        pdf_path = SIZE_OUT / f"{TASK_SLUGS[task_key]}_size_curves.pdf"
        fig.write_image(str(pdf_path))
        reader = PdfReader(pdf_path)
        if len(reader.pages) > 1:
            assert all(not (page.extract_text() or "").strip() and not list(page.images)
                       for page in reader.pages[1:])
            writer = PdfWriter()
            writer.add_page(reader.pages[0])
            single = pdf_path.with_suffix(".single.pdf")
            with single.open("wb") as f:
                writer.write(f)
            single.replace(pdf_path)
        fig.write_image(str(INSPECTION / f"{TASK_SLUGS[task_key]}_size_curves.png"), scale=1.4)
        print(f"Wrote {task_label} size curves")
    manifest = {
        "figures": [f"{TASK_SLUGS[k]}_size_curves.pdf" for k in TASKS],
        "tasks": {k: v[0] for k, v in TASKS.items()},
        "metrics": {k: [m for m, _ in TASK_METRICS[k]] for k in TASKS},
        "source": str(functional.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(functional.read_bytes()).hexdigest(),
        "stability_manifest": str(stability_manifest.relative_to(ROOT)),
        "stability_manifest_sha256": hashlib.sha256(stability_manifest.read_bytes()).hexdigest(),
        "filters": "All tasks; four methods at joint matched discovery/evaluation; ablation-independent Random at each evaluation; official circuit marker at each task's native ablation",
        "values": "functional.csv five-run metric_mean and population metric_sd (ddof=0); colored band is mean plus/minus 1 SD; manual is one deterministic evaluation",
        "random_source": str(functional.relative_to(ROOT)),
        "manual_source": str(manual_file.relative_to(ROOT)),
        "manual_source_sha256": hashlib.sha256(manual_file.read_bytes()).hexdigest(),
        "manual_provenance": "The ablation-independent official score cache uses a resample anchor filename; the manual circuit is displayed only under its task's native evaluation ablation (mean for IOI, resample for Greater-Than and Docstring)",
        "acdc": "Only recorded threshold boundaries and full-circuit point are plotted; lines connect observations without adding intermediate values",
        "rows": len(rows), "manual_budgets": MANUAL_BUDGETS,
    }
    (SIZE_OUT / "size_curves_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} plotted rows")


if __name__ == "__main__":
    main()
