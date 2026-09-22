"""Build the matched-ablation completeness table for the appendix.

Run on a CPU compute node:
    srun --mem=2G python -m experiments.thesis.results_chapter_completeness
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.thesis.results_chapter_controlled_comparison import (
    ABLATIONS,
    MANUAL_NATIVE_ABLATION,
    METHODS,
    METHOD_LABELS,
    METRICS,
    ROOT,
    TASKS,
    format_value,
    read_csv,
)


FUNCTIONAL = ROOT / "results/stability/stability_v5_2026-09-19/functional.csv"
FUNCTIONAL_MANIFEST = FUNCTIONAL.with_name("manifest.json")
MANUAL = ROOT / "results/faithfulness_comparison/faithfulness_seed42_2026-09-19/summary.csv"
OUT = ROOT / "results/results_chapter/completeness"
APPENDIX = ROOT / "thesis-overleaf/Thesis/chapters/appendix.tex"
TABLE_START = "% BEGIN GENERATED MATCHED COMPLETENESS TABLE"
TABLE_END = "% END GENERATED MATCHED COMPLETENESS TABLE"


def select_functional(rows: list[dict[str, str]], task: str, ablation: str,
                      method: str, metric: str, target: int) -> dict[str, str]:
    candidates = [row for row in rows if row["task"] == task
                  and row["algo"] == method and row["measure"] == "completeness"
                  and row["condition"] == "joint" and row["eval_ablation"] == ablation
                  and row["discovery_ablation"] == ("anchor" if method == "Random" else ablation)
                  and row["metric"] == metric]
    assert candidates, (task, ablation, method, metric)
    if method != "ACDC":
        selected = [row for row in candidates if float(row["realised_mean"]) == target]
        assert len(selected) == 1, (task, ablation, method, metric, target)
        return selected[0]
    return min(candidates, key=lambda row: abs(float(row["realised_mean"]) - target))


def select_manual(rows: list[dict[str, str]], task: str, ablation: str,
                  metric: str, target: int) -> dict[str, str]:
    selected = [row for row in rows if row["task"] == task and row["algo"] == "GT"
                and row["measure"] == "completeness"
                and row["discovery_ablation"] == ablation
                and row["eval_ablation"] == ablation and row["metric"] == metric
                and int(row["budget_edges"]) == target]
    assert len(selected) == 1, (task, ablation, metric, target)
    return selected[0]


def latex_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        (r"\caption{Completeness at the task-specific comparison size under matched "
         r"discovery and evaluation ablations. The selected circuit edges are ablated "
         r"from the clean prompt, leaving their complement intact. Method and random "
         r"circuit entries give the mean $\pm$ population standard deviation across five "
         r"runs. The deterministic manual circuit is shown only under its native ablation. "
         r"Parenthetical values give edge counts; ACDC uses the mean realised count. "
         r"Clean KL divergence is from the unablated model; logit difference reports "
         r"the percentage of the full-model difference retained after knockout. "
         r"Accuracy is correct-over-incorrect accuracy and is undefined for Greater-Than.}" ),
        r"\label{tab:matched-completeness}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lllrrr@{}}", r"\toprule",
        r"Task & Ablation & Algorithm & Clean KL & Logit difference (\%) & Accuracy (\%) \\",
        r"\midrule",
    ]
    for task_index, (task, task_label, _) in enumerate(TASKS):
        for ablation_index, ablation in enumerate(ABLATIONS):
            methods = (*METHODS, "Random")
            if ablation == MANUAL_NATIVE_ABLATION[task]:
                methods += ("Manual circuit",)
            for method_index, method in enumerate(methods):
                by_metric = {row["metric"]: row for row in rows
                             if row["task"] == task and row["ablation"] == ablation
                             and row["algo"] == method}
                values = [format_value(by_metric[metric], metric).replace("$-0.0$", "$0.0$")
                          if metric in by_metric else "--" for metric, _ in METRICS]
                task_cell = (rf"\multirow{{16}}{{*}}{{{task_label}}}"
                             if ablation_index == 0 and method_index == 0 else "")
                ablation_cell = (rf"\multirow{{{len(methods)}}}{{*}}{{{ablation.capitalize()}}}"
                                 if method_index == 0 else "")
                label = ("Random circuit" if method == "Random" else
                         "Manual circuit" if method == "Manual circuit" else
                         METHOD_LABELS[method])
                if method in {"ACDC", "Manual circuit"}:
                    label += f" ({float(by_metric[METRICS[0][0]]['realised_mean']):.0f})"
                lines.append(f"{task_cell} & {ablation_cell} & {label} & "
                             + " & ".join(values) + r" \\")
            if ablation != ABLATIONS[-1]:
                lines.append(r"\cmidrule(lr){2-6}")
        if task_index != len(TASKS) - 1:
            lines.extend((r"\midrule", r"\midrule"))
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    return "\n".join(lines)


def main() -> None:
    functional = read_csv(FUNCTIONAL)
    manual = read_csv(MANUAL)
    rows: list[dict[str, object]] = []
    for task, _, target in TASKS:
        metrics = METRICS if task == TASKS[0][0] else METRICS[:2]
        for ablation in ABLATIONS:
            for metric, _ in metrics:
                for method in (*METHODS, "Random"):
                    source = select_functional(functional, task, ablation, method, metric, target)
                    assert int(source["n_seeds"]) == 5
                    rows.append({"task": task, "ablation": ablation, "algo": method,
                                 "target_edges": target, "selector": source["selector"],
                                 "budget_label": source["budget_label"],
                                 "realised_mean": float(source["realised_mean"]),
                                 "metric": metric, "n_seeds": 5,
                                 "metric_mean": float(source["metric_mean"]),
                                 "metric_sd": float(source["metric_sd"])})
                if ablation == MANUAL_NATIVE_ABLATION[task]:
                    source = select_manual(manual, task, ablation, metric, target)
                    rows.append({"task": task, "ablation": ablation,
                                 "algo": "Manual circuit", "target_edges": target,
                                 "selector": "manual", "budget_label": str(target),
                                 "realised_mean": float(target), "metric": metric,
                                 "n_seeds": 1, "metric_mean": float(source["value"]),
                                 "metric_sd": 0.0})
    assert len(rows) == 80, len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "matched_completeness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    appendix = APPENDIX.read_text()
    assert appendix.count(TABLE_START) == appendix.count(TABLE_END) == 1
    before, rest = appendix.split(TABLE_START, 1)
    _, after = rest.split(TABLE_END, 1)
    APPENDIX.write_text(before + TABLE_START + "\n" + latex_table(rows)
                        + TABLE_END + after)
    manifest = {
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (FUNCTIONAL, FUNCTIONAL_MANIFEST, MANUAL)},
        "selection": "Matched discovery/evaluation ablations at task-specific target; nearest realised ACDC boundary; Random anchor reused per evaluation ablation; deterministic manual circuit under native ablation.",
        "patch_direction": "EDGE_PATCH on clean prompts (circuit knockout)",
        "rows": len(rows),
    }
    (OUT / "matched_completeness_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} completeness rows to {OUT} and updated {APPENDIX}")


if __name__ == "__main__":
    main()
