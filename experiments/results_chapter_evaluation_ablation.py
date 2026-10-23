"""Plot final cross-ablation circuit evaluations and write exact appendix tables.

Run on a CPU compute node:
    srun --mem=8G python -m experiments.results_chapter_evaluation_ablation
"""
from __future__ import annotations

import csv
import hashlib
import json
from math import isclose
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from .config import ROOT, load_registry, resolve_path
from .results_chapter_sources import EXPORT_ROOT, bundle, provenance_path


SOURCE = ROOT / "artifacts/final/results_chapter/main/functional.csv"
SOURCE_MANIFEST = SOURCE.with_name("manifest.json")
SELECTION = SOURCE.parent / "exhibits/controlled_comparison/matched_quality.csv"
SELECTION_MANIFEST = SELECTION.with_name("matched_quality_manifest.json")
OUT = EXPORT_ROOT / 'main/evaluation_ablation'
IMAGES = OUT
TASKS = {
    "Indirect Object Identification": ("ioi", "IOI", 739,
        ("Clean KL Div", "Logit Diff Percent", "Correct Answer Greater Than Incorrect Percent")),
    "Greaterthan": ("greater_than", "Greater-Than", 235,
        ("Clean KL Div", "Logit Diff Percent")),
}
METRIC_LABELS = {
    "Clean KL Div": "KL divergence",
    "Logit Diff Percent": "Logit difference (% of full model)",
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
    registry = load_registry()
    assert sha(resolve_path(selection_meta["source"], registry)) == selection_meta["source_sha256"]
    assert sha(resolve_path(selection_meta["source_manifest"], registry)) == selection_meta["source_manifest_sha256"]
    source_meta = json.loads(SOURCE_MANIFEST.read_text())
    if source_meta.get('evaluation_protocol_id'):
        from .evaluation_protocol import require_same_protocol
        require_same_protocol(source_meta, selection_meta)
    assert source_meta["seeds"] == [42, 43, 44, 45, 46]
    if not source_meta.get('evaluation_protocol_id'):
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


EVAL_COLORS = tuple(EVAL_STYLE[a][0] for a in ABLATIONS)
EVAL_MARKERS = ("o", "D", "s")
KL = "Clean KL Div"
LD = "Logit Diff Percent"
ACC = "Correct Answer Greater Than Incorrect Percent"
LABELS = tuple(METHOD_LABELS[m] for m in METHODS)

PLOT_STYLE = {"font.family": "DejaVu Sans", "font.size": 10,
                     "text.color": "#252B33", "axes.labelcolor": "#252B33",
                     "xtick.color": "#59616B", "ytick.color": "#252B33",
                     "axes.edgecolor": "#C3C8CE", "axes.linewidth": .6,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"}

def values(rows, task, metric, discovery, method):
    group = {row["eval_ablation"]: row for row in rows
             if row["task"] == task and row["metric"] == metric
             and row["discovery_ablation"] == discovery and row["algo"] == method}
    assert set(group) == set(ABLATIONS)
    return np.asarray([(float(group[ev]["metric_mean"]), float(group[ev]["metric_sd"]))
                       for ev in ABLATIONS]).T


def scale(task, metric):
    # Identical linear scales across the three discovery panels of each metric.
    if metric == KL:
        return ((-.55, 13), [0, 4, 8, 12]) if task == "Indirect Object Identification" else ((-1, 24), [0, 5, 10, 15, 20])
    if metric == LD:
        return ((-60, 450), [0, 100, 200, 300, 400]) if task == "Indirect Object Identification" else ((-65, 150), [-50, 0, 50, 100, 150])
    return (-3, 104), [0, 25, 50, 75, 100]


def reference(ax, metric, vertical):
    if metric != LD:
        return
    draw = ax.axvline if vertical else ax.axhline
    draw(0, color="#AFB5BD", linewidth=.7, zorder=0)
    draw(100, color="#77818C", linewidth=.9, linestyle=(0, (3, 3)), zorder=0)


def discovery_header(fig, axes, height):
    for row in axes:
        for ax, discovery in zip(row, ABLATIONS):
            pos = ax.get_position()
            fig.text((pos.x0 + pos.x1) / 2, pos.y1 + .10 / height,
                     f"Discovery: $\\mathbf{{{discovery.title()}}}$", ha="center", va="center",
                     fontsize=9, color="#59616B")


def metric_heading(fig, axes, row, metric, height):
    left = axes[row, 0].get_position()
    fig.text(.5, left.y1 + .34 / height,
             ("Logit difference (%)" if metric == LD else METRIC_LABELS[metric]), ha="center", va="center", fontsize=10.5,
             weight="bold")


def _render_figure(rows, task):
    slug, title, budget, metrics = TASKS[task]
    height = 5.7 if len(metrics) == 3 else 3.9
    fig, axes = plt.subplots(len(metrics), 3, figsize=(6.6, height))
    fig.subplots_adjust(left=.165, right=.985, bottom=.65 / height,
                        top=1 - .60 / height, hspace=.85, wspace=.15)
    handles = [Line2D([], [], color=c, marker=m, linewidth=0, markersize=5, label=a.title())
               for a, c, m in zip(ABLATIONS, EVAL_COLORS, EVAL_MARKERS)]
    fig.text(.165, .16 / height, "Evaluation ablation", va="center", fontsize=9)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.43, .16 / height),
               ncol=3, frameon=False, fontsize=9, handletextpad=.4,
               handlelength=1, columnspacing=1.2, borderaxespad=0)
    discovery_header(fig, axes, height)
    for r, metric in enumerate(metrics):
        metric_heading(fig, axes, r, metric, height)
        limits, ticks = scale(task, metric)
        for c, discovery in enumerate(ABLATIONS):
            ax = axes[r, c]
            for y in range(4):
                if y % 2 == 0:
                    ax.axhspan(y-.47, y+.47, facecolor="#F4F6F8", zorder=-2)
            for m, method in enumerate(METHODS):
                av, sd = values(rows, task, metric, discovery, method)
                for e in range(3):
                    assert limits[0] <= av[e]-sd[e] <= av[e]+sd[e] <= limits[1], (task, metric, av, sd)
                    ax.errorbar(av[e], m + (e-1)*.245, xerr=sd[e], fmt=EVAL_MARKERS[e],
                                color=EVAL_COLORS[e], ms=4.3, elinewidth=.9, capsize=2,
                                capthick=.8, markeredgecolor="white", markeredgewidth=.5, zorder=3)
            reference(ax, metric, True)
            ax.set(xlim=limits, xticks=ticks, ylim=(3.57, -.57), yticks=range(4))
            ax.set_yticklabels(LABELS if c == 0 else [""]*4, fontsize=9)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.tick_params(axis="x", labelsize=9, length=3, pad=4)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E5E8EC", linewidth=.5)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
    pdf = IMAGES / f"{slug}_evaluation_ablation.pdf"
    png = OUT / f"{slug}_evaluation_ablation.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=190)
    plt.close(fig)
    return pdf, png



def render_figure(rows, task):
    with plt.rc_context(PLOT_STYLE):
        return _render_figure(rows, task)

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
             "\\caption{" + caption_metric + " for the " + title
             + " task under each evaluation ablation. Values are five-run average $\\pm$ population SD. Numeric columns identify the evaluation ablation.}",
             "\\label{tab:" + slug.replace("_", "-") + "-evaluation-" + METRIC_SLUGS[metric].replace("_", "-") + "}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc@{}}", r"\toprule",
             "Discovery ablation & Method & Zero & Resample & Mean \\\\", r"\midrule"]
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


def main(source_dir=None, output_root=None) -> None:
    global SOURCE, SOURCE_MANIFEST, SELECTION, SELECTION_MANIFEST, OUT, IMAGES
    source = (ROOT / source_dir).resolve() if source_dir is not None else Path(bundle()['source_dir'])
    SOURCE, SOURCE_MANIFEST = source / 'functional.csv', source / 'manifest.json'
    root = (ROOT / output_root).resolve() if output_root is not None else EXPORT_ROOT / 'main'
    SELECTION = root / 'controlled_comparison/matched_quality.csv'
    if not SELECTION.is_file():
        SELECTION = source / 'exhibits/controlled_comparison/matched_quality.csv'
    SELECTION_MANIFEST = SELECTION.with_name('matched_quality_manifest.json')
    OUT = IMAGES = root / 'evaluation_ablation'
    for directory in (OUT, IMAGES):
        directory.mkdir(parents=True, exist_ok=True)
    rows = select()
    source_meta = json.loads(SOURCE_MANIFEST.read_text())
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
    table_path = OUT / 'appendix_tables.tex'
    table_path.write_text('\n\n'.join(tables) + '\n')
    outputs.append(table_path)
    manifest = {
        "evaluation_protocol_id": source_meta.get('evaluation_protocol_id'),
        "source": provenance_path(SOURCE), "source_sha256": sha(SOURCE),
        "source_manifest": provenance_path(SOURCE_MANIFEST),
        "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "selection": provenance_path(SELECTION), "selection_sha256": sha(SELECTION),
        "selection_manifest": provenance_path(SELECTION_MANIFEST),
        "selection_manifest_sha256": sha(SELECTION_MANIFEST),
        "discovery_cache_dir": source_meta.get("discovery_cache_dir"),
        "selected_inputs": source_meta.get("selected_inputs"),
        "seeds": [42, 43, 44, 45, 46], "rows": len(rows),
        "selection_rule": "Same selector, budget, and five circuits as the matched-quality table; only evaluation ablation varies.",
        "outputs": {provenance_path(path): sha(path) for path in outputs},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} crossed-evaluation rows, two figures, and five appendix tables")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    main(args.source_dir, args.output_root)
