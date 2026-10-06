"""Select matched-ablation circuit-quality values for Results.

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
MANUAL_NATIVE_ABLATION = {"Indirect Object Identification": "mean", "Greaterthan": "resample"}
METRICS = (
    "Clean KL Div",
    "Logit Diff Percent",
    "Correct Answer Greater Than Incorrect Percent",
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


def main(output_root=None) -> None:
    global SOURCE, SOURCE_MANIFEST, OUT
    if output_root is not None:
        root = (ROOT / output_root).resolve()
        SOURCE = root / 'size_curves/size_curves.csv'
        SOURCE_MANIFEST = root / 'size_curves/size_curves_manifest.json'
        OUT = root / 'controlled_comparison'
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SOURCE)
    selected: list[dict[str, object]] = []
    for task, _, target in TASKS:
        for ablation in ABLATIONS:
            task_metrics = METRICS if task == TASKS[0][0] else METRICS[:2]
            for metric in task_metrics:
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
    manifest = {
        "evaluation_protocol_id": json.loads(SOURCE_MANIFEST.read_text()).get('evaluation_protocol_id'),
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
            TASKS[0][0]: list(METRICS),
            TASKS[1][0]: list(METRICS[:2]),
        },
        "selection": "Exact task-specific target for continuous scores; nearest realised ACDC threshold boundary; joint five-run condition; matched discovery and evaluation ablation for methods and Random; deterministic native-ablation condition for Manual circuit.",
        "rows": len(selected),
    }
    (OUT / "matched_quality_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(selected)} matched-quality rows to {OUT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    main(args.output_root)
