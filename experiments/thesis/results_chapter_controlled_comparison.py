"""Build the selected-budget table for the controlled comparison in Results.

Run on a CPU compute node:
    srun --mem=8G python -m experiments.thesis.results_chapter_controlled_comparison
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/results_chapter/size_curves/size_curves.csv"
SOURCE_MANIFEST = ROOT / "results/results_chapter/size_curves/size_curves_manifest.json"
OUT = ROOT / "results/results_chapter/controlled_comparison"

TASKS = (
    ("Indirect Object Identification", "IOI", 739),
    ("Greaterthan", "Greater-Than", 235),
)
ABLATIONS = ("zero", "resample", "mean")
METHODS = ("ACDC", "EAP", "EP", "DiscoGP")
METHOD_LABELS = {"ACDC": "ACDC", "EAP": "EAP", "EP": "Edge Pruning", "DiscoGP": "DiscoGP"}
METRICS = (
    ("Clean KL Div", "Clean KL divergence"),
    ("Logit Diff Percent", "Recovered logit difference (\\%)"),
    ("Correct Answer Greater Than Incorrect Percent", "Correct-over-incorrect accuracy (\\%)"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def selected_row(rows: list[dict[str, str]], task: str, ablation: str,
                 method: str, metric: str, target: int) -> dict[str, str]:
    candidates = [
        row for row in rows
        if row["task"] == task
        and row["algo"] == method
        and row["discovery_ablation"] == ablation
        and row["eval_ablation"] == ablation
        and row["condition"] == "joint"
        and row["measure"] == "faithfulness"
        and row["metric"] == metric
    ]
    assert candidates, (task, ablation, method, metric)
    if method != "ACDC":
        exact = [row for row in candidates if float(row["realised_mean"]) == target]
        assert len(exact) == 1, (task, ablation, method, metric, target)
        return exact[0]
    return min(candidates, key=lambda row: abs(float(row["realised_mean"]) - target))


def format_value(row: dict[str, str], metric: str, best: bool = False) -> str:
    mean = float(row["metric_mean"])
    sd = float(row["metric_sd"])
    if metric == "Clean KL Div" and max(abs(mean), abs(sd)) < 1:
        value = f"{mean:.3f} \\pm {sd:.3f}"
    elif metric == "Clean KL Div":
        value = f"{mean:.2f} \\pm {sd:.2f}"
    else:
        value = f"{mean:.1f} \\pm {sd:.1f}"
    return f"$\\mathbf{{{value}}}$" if best else f"${value}$"


def best_method(by_method: dict[str, dict[str, object]], metric: str) -> str:
    means = {method: float(rows[metric]["metric_mean"])
             for method, rows in by_method.items() if metric in rows}
    if metric == "Clean KL Div":
        return min(means, key=means.get)
    if metric == "Logit Diff Percent":
        return min(means, key=lambda method: abs(means[method] - 100.0))
    return max(means, key=means.get)


def latex_table(selected: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (r"\caption{Circuit quality under matched discovery and evaluation ablations. "
         r"Values are the mean $\pm$ population standard deviation across five runs at "
         r"the task-specific comparison size. Lower clean KL divergence is better; higher "
         r"accuracy is better, and recovered logit difference is best at $100\%$. Bold marks the "
         r"best mean for each task and ablation and does not indicate statistical significance. "
         r"Parenthetical values beside ACDC give its mean realised edge count. Accuracy is not "
         r"defined for Greater-Than.}"),
        r"\label{tab:matched-circuit-quality}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Task & Ablation & Algorithm & Clean KL & Logit difference (\%) & Accuracy (\%) \\",
        r"\midrule",
    ]
    for task_index, (task, task_label, _) in enumerate(TASKS):
        for ablation_index, ablation in enumerate(ABLATIONS):
            rows_by_method = {
                method: {
                    row["metric"]: row for row in selected
                    if row["task"] == task and row["ablation"] == ablation
                    and row["algo"] == method
                }
                for method in METHODS
            }
            winners = {metric: best_method(rows_by_method, metric)
                       for metric, _ in METRICS
                       if any(metric in rows for rows in rows_by_method.values())}
            for method_index, method in enumerate(METHODS):
                by_metric = rows_by_method[method]
                values = [
                    format_value(by_metric[metric], metric, winners[metric] == method)
                    if metric in by_metric else r"--"
                    for metric, _ in METRICS
                ]
                task_cell = (rf"\multirow{{12}}{{*}}{{{task_label}}}"
                             if ablation_index == 0 and method_index == 0 else "")
                ablation_cell = (rf"\multirow{{4}}{{*}}{{{ablation.capitalize()}}}"
                                 if method_index == 0 else "")
                method_label = METHOD_LABELS[method]
                if method == "ACDC":
                    realised = float(by_metric[METRICS[0][0]]["realised_mean"])
                    method_label += f" ({realised:.0f})"
                lines.append(
                    f"{task_cell} & {ablation_cell} & {method_label} & "
                    + " & ".join(values) + r" \\"
                )
            if ablation != ABLATIONS[-1]:
                lines.append(r"\cmidrule(lr){2-6}")
        if task_index != len(TASKS) - 1:
            lines.extend((r"\midrule", r"\midrule"))
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SOURCE)
    selected: list[dict[str, object]] = []
    for task, _, target in TASKS:
        for ablation in ABLATIONS:
            task_metrics = METRICS if task == TASKS[0][0] else METRICS[:2]
            for metric, _ in task_metrics:
                for method in METHODS:
                    row = selected_row(rows, task, ablation, method, metric, target)
                    selected.append({
                        "task": task,
                        "ablation": ablation,
                        "algo": method,
                        "target_edges": target,
                        "selector": row["selector"],
                        "budget_label": row["budget_label"],
                        "realised_mean": float(row["realised_mean"]),
                        "metric": metric,
                        "n_seeds": int(row["n_seeds"]),
                        "metric_mean": float(row["metric_mean"]),
                        "metric_sd": float(row["metric_sd"]),
                        "metric_min": float(row["metric_min"]),
                        "metric_max": float(row["metric_max"]),
                    })
    expected = len(ABLATIONS) * len(METHODS) * (len(METRICS) + len(METRICS[:2]))
    assert len(selected) == expected
    assert all(row["n_seeds"] == 5 for row in selected)

    fields = list(selected[0])
    with (OUT / "matched_quality.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(selected)
    (OUT / "matched_quality.tex").write_text(latex_table(selected))

    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        "tasks": [task for task, _, _ in TASKS],
        "ablations": list(ABLATIONS),
        "methods": list(METHODS),
        "metrics": {
            TASKS[0][0]: [metric for metric, _ in METRICS],
            TASKS[1][0]: [metric for metric, _ in METRICS[:2]],
        },
        "selection": "Exact task-specific target for continuous scores; nearest realised ACDC threshold boundary; joint five-run condition; matched discovery and evaluation ablation.",
        "rows": len(selected),
    }
    (OUT / "matched_quality_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(selected)} matched-quality rows to {OUT}")


if __name__ == "__main__":
    main()
