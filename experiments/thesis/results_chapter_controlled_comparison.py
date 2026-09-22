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
MANUAL_NATIVE_ABLATION = {"Indirect Object Identification": "mean", "Greaterthan": "resample"}
METRICS = (
    ("Clean KL Div", "Clean KL divergence"),
    ("Logit Diff Percent", "Recovered logit difference (\\%)"),
    ("Correct Answer Greater Than Incorrect Percent", "Correct-over-incorrect accuracy (\\%)"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def selected_row(rows: list[dict[str, str]], task: str, ablation: str,
                 method: str, metric: str, target: int, *, condition: str = "joint") -> dict[str, str]:
    candidates = [
        row for row in rows
        if row["task"] == task
        and row["algo"] == method
        and row["discovery_ablation"] == ablation
        and row["eval_ablation"] == ablation
        and row["condition"] == condition
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
    deterministic = int(row["n_seeds"]) == 1
    if metric == "Clean KL Div" and max(abs(mean), abs(sd)) < 1:
        value = f"{mean:.3f}" if deterministic else f"{mean:.3f} \\pm {sd:.3f}"
    elif metric == "Clean KL Div":
        value = f"{mean:.2f}" if deterministic else f"{mean:.2f} \\pm {sd:.2f}"
    else:
        value = f"{mean:.1f}" if deterministic else f"{mean:.1f} \\pm {sd:.1f}"
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
        (r"\caption{Circuit quality at the task-specific comparison size. The four discovery "
         r"methods report the mean $\pm$ population standard deviation across five runs under "
         r"matched discovery and evaluation ablations. Random circuit is a size-matched five-run "
         r"null evaluated under each ablation; the deterministic manual circuit is shown only at "
         r"its native ablation (mean for IOI and resample for Greater-Than). Lower clean KL "
         r"divergence is better; higher accuracy is better, and recovered logit difference is best "
         r"at $100\%$. Bold marks the best mean among the four discovery methods for each task and "
         r"ablation and does not indicate statistical significance. Parenthetical values give "
         r"edge counts; for ACDC the value is the mean realised count. Accuracy is not defined "
         r"for Greater-Than.}"),
        r"\label{tab:matched-circuit-quality}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Task & Ablation & Algorithm & Clean KL & Logit difference (\%) & Accuracy (\%) \\",
        r"\midrule",
    ]
    for task_index, (task, task_label, _) in enumerate(TASKS):
        manual_ablation = MANUAL_NATIVE_ABLATION[task]
        task_row_count = len(ABLATIONS) * (len(METHODS) + 1) + 1
        for ablation_index, ablation in enumerate(ABLATIONS):
            rows_by_method = {
                method: {
                    row["metric"]: row for row in selected
                    if row["task"] == task and row["ablation"] == ablation
                    and row["algo"] == method
                }
                for method in (*METHODS, "Random", "Manual circuit")
            }
            method_rows = {method: rows for method, rows in rows_by_method.items() if method in METHODS}
            winners = {metric: best_method(method_rows, metric)
                       for metric, _ in METRICS
                       if any(metric in rows for method, rows in rows_by_method.items()
                              if method in METHODS)}
            methods = (*METHODS, "Random")
            if ablation == manual_ablation:
                methods = (*methods, "Manual circuit")
            ablation_row_count = len(methods)
            for method_index, method in enumerate(methods):
                by_metric = rows_by_method[method]
                values = [
                    format_value(by_metric[metric], metric, winners[metric] == method)
                    if metric in by_metric else r"--"
                    for metric, _ in METRICS
                ]
                task_cell = (rf"\multirow{{{task_row_count}}}{{*}}{{{task_label}}}"
                             if ablation_index == 0 and method_index == 0 else "")
                ablation_cell = (rf"\multirow{{{ablation_row_count}}}{{*}}{{{ablation.capitalize()}}}"
                                 if method_index == 0 else "")
                method_label = ("Random circuit" if method == "Random" else
                                "Manual circuit" if method == "Manual circuit" else
                                METHOD_LABELS[method])
                if method == "ACDC":
                    realised = float(by_metric[METRICS[0][0]]["realised_mean"])
                    method_label += f" ({realised:.0f})"
                if method == "Manual circuit":
                    method_label += f" ({int(by_metric[METRICS[0][0]]['realised_mean'])})"
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
                for method in (*METHODS, "Random"):
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
                if ablation == MANUAL_NATIVE_ABLATION[task]:
                    method = "Manual circuit"
                    row = selected_row(rows, task, ablation, method, metric, target,
                                       condition="deterministic")
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
    expected_method_rows = len(ABLATIONS) * len(METHODS) * (len(METRICS) + len(METRICS[:2]))
    expected_random_rows = len(ABLATIONS) * (len(METRICS) + len(METRICS[:2]))
    expected_manual_rows = len(METRICS) + len(METRICS[:2])
    assert len(selected) == expected_method_rows + expected_random_rows + expected_manual_rows
    assert all(row["n_seeds"] == 5 for row in selected if row["algo"] != "Manual circuit")

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
        "baselines": {
            "random": "size-matched five-run null under each evaluation ablation",
            "manual": "deterministic native-ablation reference; IOI mean and Greater-Than resample",
        },
        "metrics": {
            TASKS[0][0]: [metric for metric, _ in METRICS],
            TASKS[1][0]: [metric for metric, _ in METRICS[:2]],
        },
        "selection": "Exact task-specific target for continuous scores; nearest realised ACDC threshold boundary; joint five-run condition; matched discovery and evaluation ablation for methods and Random; deterministic native-ablation condition for Manual circuit.",
        "rows": len(selected),
    }
    (OUT / "matched_quality_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(selected)} matched-quality rows to {OUT}")


if __name__ == "__main__":
    main()
