"""Faithfulness spine — discovery x eval matrix ANALYSIS pass (sub-RQ3, issue #4).

Discovery is run separately as a SLURM array of per-cell jobs
(`faithfulness_discovery.sh` -> `run_cell.py`), producing one cached `PruneScores` per
(method, task, discovery-ablation) in `.prune_scores_cache/faithfulness/`. This script
LOADS those cells (no retraining), groups them by discovery-ablation, and produces the
full discovery x eval matrix:

  results/faithfulness_comparison/<name>/discovery-<D>/<task>/
      <faithfulness|completeness>/<eval-ablation>/<metric>.pdf
      circuit_graphs/<algo>.html
  results/faithfulness_comparison/<name>/discovery-<D>/roc.pdf
  results/faithfulness_comparison/<name>/summary.{csv,md}

Every number is tagged with its (discovery-ablation, eval-ablation, patch-direction)
triple via the directory path and the summary-table columns (Miller 2024).

Run discovery first:   sbatch faithfulness_discovery.sh
Then this analysis:    sbatch run_faithfulness.sh   (or: srun python experiments/faithfulness_comparison.py)

Spec: docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md
"""
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.metrics.analysis_utils import (
    extract_metric,
    latest_per_cell,
    load_many_caches,
)
from auto_circuit.metrics.official_circuits.measure_roc import measure_roc
from auto_circuit.metrics.official_circuits.roc_plot import roc_plot
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    ANSWER_PAIR_KL_DIV_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PROB_DIFF_METRIC,
)
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import (
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    TASK_DICT,
)
from auto_circuit.types import AblationType, PatchType
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize_circuit_slider import save_circuit_sankey_slider
from experiment_utils import git_state, write_manifest
from interactive_curves import save_interactive_curves

from faithfulness_cells import ANCHOR_ALGOS, DISCOVERY_ABLATIONS, SWEPT_ALGOS
from run_cell import _ALGO_MAP, _TASK_TEMPLATES
from similarity_heatmaps import (
    DEFAULT_SIMILARITY_EDGE_COUNTS,
    write_all_similarity_heatmaps,
)

# Base algo keys of the ablation-independent anchors (no discovery sweep).
_ANCHOR_ALGO_KEYS = {_ALGO_MAP[a].key for a in ANCHOR_ALGOS}

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

FAITHFULNESS_CACHE_DIR = ".prune_scores_cache/faithfulness"
RESULTS_DIR = "results/faithfulness_comparison"

# Eval ablations swept for free (no retraining) on every discovered circuit.
EVAL_ABLATION_TYPES: List[AblationType] = [
    AblationType.RESAMPLE,
    AblationType.ZERO,
    AblationType.TOKENWISE_MEAN_CORRUPT,
]

# Patch direction -> (measure, what-is-ablated) (Miller 2024 Table 3, clean input):
#   TREE_PATCH = ablate complement (keep circuit)   -> faithfulness  (Row 3)
#   EDGE_PATCH = ablate circuit    (keep complement) -> completeness (Row 1)
_PATCH_SEMANTICS = {
    PatchType.TREE_PATCH: "faithfulness",
    PatchType.EDGE_PATCH: "completeness",
}

# Per-task metric selection (spec §3 + hard compatibility). CORRECT_ANSWER_* assert a
# single tensor answer -> excluded for Greater-Than (list answers). LOGIT_DIFF_PERCENT
# is unstable on Docstring -> excluded there (KL is the Docstring primary).
_COMMON_METRICS = [CLEAN_KL_DIV_METRIC, ANSWER_PAIR_KL_DIV_METRIC]
TASK_METRICS: dict = {
    IOI_COMPONENT_CIRCUIT_TASK.key: _COMMON_METRICS + [
        LOGIT_DIFF_PERCENT_METRIC,
        PROB_DIFF_METRIC,
        CORRECT_ANSWER_PERCENT_METRIC,
        CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,
    ],
    GREATERTHAN_COMPONENT_CIRCUIT_TASK.key: _COMMON_METRICS + [
        LOGIT_DIFF_PERCENT_METRIC,
        PROB_DIFF_METRIC,
    ],
    DOCSTRING_COMPONENT_CIRCUIT_TASK.key: _COMMON_METRICS + [
        CORRECT_ANSWER_PERCENT_METRIC,
        CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,
    ],
}

# Eval ablation each official circuit was originally established under: IOI = mean
# over the ABC distribution (Wang et al. 2022), Greater-Than = resample (Hanna et al.
# 2023), Docstring = resample (Heimersheim & Janiak 2023). GT is plotted only at its
# own ablation; elsewhere the number reports protocol mismatch, not circuit quality.
GT_NATIVE_EVAL_ABLATION: Dict[str, str] = {
    IOI_COMPONENT_CIRCUIT_TASK.key: "mean",
    GREATERTHAN_COMPONENT_CIRCUIT_TASK.key: "resample",
    DOCSTRING_COMPONENT_CIRCUIT_TASK.key: "resample",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Short, filesystem-clean ablation slugs, consistent with the discovery-ablation names
# used by faithfulness_cells.py (zero/resample/mean). AblationType.__str__ is title-cased
# with spaces ("Tokenwise Mean Corrupt"), so normalise explicitly.
_ABL_SHORT = {
    AblationType.ZERO: "zero",
    AblationType.RESAMPLE: "resample",
    AblationType.TOKENWISE_MEAN_CORRUPT: "mean",
}
# measurement_figs stores the ablation as str(AblationType) in its meta dict.
_ABL_SHORT_FROM_STR = {str(a): s for a, s in _ABL_SHORT.items()}


def _task_slug(task_key: str) -> str:
    return TASK_DICT[task_key].name.lower().replace(" ", "_")


def _norm_slug(s: str) -> str:
    return s.split(".")[-1].lower().replace(" ", "_")


def _ablation_slug(ablation: AblationType) -> str:
    return _ABL_SHORT.get(ablation, _norm_slug(str(ablation)))


_GT_ALGO_KEY = _ALGO_MAP["gt"].key


def _omit_gt(task_key: str, algo_key: str, ablation: AblationType) -> bool:
    """True where GT should be hidden: any eval ablation but its own.

    See GT_NATIVE_EVAL_ABLATION. Tasks with no recorded native ablation keep GT
    everywhere rather than silently losing it.
    """
    if algo_key != _GT_ALGO_KEY:
        return False
    native = GT_NATIVE_EVAL_ABLATION.get(task_key)
    return native is not None and _ablation_slug(ablation) != native


def _value_at_budget(series, budget) -> Optional[float]:
    """Faithfulness value at the grid point nearest `budget` (e.g. true_edge_count)."""
    if not series:
        return None
    if budget is None:
        return series[len(series) // 2][1]  # mid grid point if no reference budget
    return min(series, key=lambda xy: abs(xy[0] - budget))[1]


def _experiment_folder(name: Optional[str], seed: int) -> Path:
    base = repo_path_to_abs_path(RESULTS_DIR)
    if name:
        folder = base / name
    else:
        folder = base / f"faithfulness_seed{seed}_{datetime.now():%Y-%m-%d}"
        if folder.exists():  # same-day re-run: disambiguate, don't clobber
            folder = base / f"{folder.name}_{datetime.now():%H%M}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _group_cells_by_discovery_ablation(loaded, seed: int):
    """Assemble {discovery-ablation D: {task_key: {base_algo_key: PruneScores}}}.

    Swept cells land in their own ablation's group; ablation-independent anchors
    (GT/Random) are added to *every* group as reference rows. Each loaded cell
    holds exactly one (task, algo) entry; we re-key it to the BASE algo key (the
    variant key "Edge Pruning [zero]" is not in PRUNE_ALGO_DICT, the base
    "Edge Pruning" is) so downstream lookups and legends are clean — the discovery
    ablation lives in the output path, not the algo name.
    """
    groups = {d: defaultdict(dict) for d in DISCOVERY_ABLATIONS}
    consumed: list = []
    for manifest, tps in loaded:
        cell = (manifest or {}).get("cell", {})
        task_short, algo_short, abl = cell.get("task"), cell.get("algo"), cell.get("ablation")
        if cell.get("seed") != seed or cell.get("null_test"):
            continue
        if task_short not in _TASK_TEMPLATES or algo_short not in _ALGO_MAP:
            continue
        task_key = _TASK_TEMPLATES[task_short].key
        base_algo_key = _ALGO_MAP[algo_short].key
        # The cell holds a single (task -> single algo -> PruneScores).
        scores = next(iter(next(iter(tps.values())).values()))
        target_ds = DISCOVERY_ABLATIONS if algo_short in ANCHOR_ALGOS else [abl]
        for d in target_ds:
            if d in groups:
                groups[d][task_key][base_algo_key] = scores
        consumed.append({
            "task": task_short, "algo": algo_short, "ablation": abl,
            "cache_file": (manifest or {}).get("cache_file"),
            "git": (manifest or {}).get("git", {}).get("commit"),
            "saved_at": (manifest or {}).get("saved_at"),
        })
    return groups, consumed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faithfulness spine analysis pass (sub-RQ3).")
    parser.add_argument("--cache-dir", default=FAITHFULNESS_CACHE_DIR, help="Discovery cell cache folder")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the cells to analyse")
    parser.add_argument("--name", default=None, help="Experiment folder name (default: faithfulness_seed<seed>_<date>)")
    parser.add_argument("--no-circuit-graphs", action="store_true", help="Skip circuit-graph HTML")
    parser.add_argument(
        "--similarity-edge-counts", type=int, nargs="+",
        default=DEFAULT_SIMILARITY_EDGE_COUNTS,
        help="Edge budgets for the cross-cell Jaccard similarity heatmaps",
    )
    args = parser.parse_args()

    loaded = latest_per_cell(load_many_caches(args.cache_dir, "cell_"))
    groups, consumed = _group_cells_by_discovery_ablation(loaded, args.seed)
    print(f"Loaded {len(consumed)} cells from {args.cache_dir} (seed {args.seed}).")

    experiment_folder = _experiment_folder(args.name, args.seed)
    print(f"Writing results to: {experiment_folder}")

    manifest: dict = {
        "experiment": experiment_folder.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "discovery_cache_dir": args.cache_dir,
        "seed": args.seed,
        "discovery_ablations": DISCOVERY_ABLATIONS,
        "eval_ablation_types": EVAL_ABLATION_TYPES,
        "patch_types": list(_PATCH_SEMANTICS.values()),
        "metrics_per_task": {k: [m.key for m in ms] for k, ms in TASK_METRICS.items()},
        "similarity_edge_counts": args.similarity_edge_counts,
        "consumed_cells": consumed,
    }
    write_manifest(experiment_folder / "manifest.json", manifest)

    summary_rows: list = []  # one row per (discovery, task, algo, measure, eval-ablation, metric)
    # Overlay curves for the interactive HTML: one figure per (task, measure, metric),
    # each holding every (algo x discovery-ablation x eval-ablation) series.
    overlay: dict = defaultdict(list)  # (task_key, measure, metric_key) -> [CurveEntry]

    for d in DISCOVERY_ABLATIONS:
        tps_d = groups[d]
        if not tps_d:
            print(f"[discovery={d}] no cells, skipping.")
            continue
        d_dir = experiment_folder / f"discovery-{d}"
        d_dir.mkdir(parents=True, exist_ok=True)  # ROC writes here directly

        # Move loaded scores onto each task's model device + float32.
        for task_key, algo_ps in tps_d.items():
            dev = next(TASK_DICT[task_key].model.parameters()).device
            for ak in list(algo_ps):
                algo_ps[ak] = {m: s.float().to(dev) for m, s in algo_ps[ak].items()}

        # ── Circuit graphs (per discovery-ablation × task × algo) ──
        if not args.no_circuit_graphs:
            for task_key, algo_ps in tps_d.items():
                task = TASK_DICT[task_key]
                for algo_key, ps in algo_ps.items():
                    try:
                        save_circuit_sankey_slider(
                            prune_scores=ps, model=task.model,
                            algo_name=PRUNE_ALGO_DICT[algo_key].name, task_name=task.name,
                            output_dir=d_dir / _task_slug(task_key) / "circuit_graphs",
                            max_edges=10000, true_edge_count=task.true_edge_count,
                            seq_labels=task.test_loader.seq_labels,
                        )
                    except Exception as e:
                        print(f"WARNING: circuit graph failed {d}/{task.name}/{algo_key}: {e}")

        # ── ROC vs ground-truth circuits ──
        try:
            roc_fig = roc_plot(measure_roc(tps_d))
            roc_fig.write_image(str(d_dir / "roc.pdf"))
        except Exception as e:
            print(f"WARNING: ROC failed for discovery={d}: {e}")

        # ── Eval matrix: per patch direction × task × eval-ablation × metric ──
        for patch_type, measure in _PATCH_SEMANTICS.items():
            for task_key, algo_ps in tps_d.items():
                task = TASK_DICT[task_key]
                metrics = TASK_METRICS.get(task_key, _COMMON_METRICS)
                try:
                    measurements = measure_prune_metrics(
                        ablation_types=EVAL_ABLATION_TYPES, metrics=metrics,
                        task_prune_scores={task_key: algo_ps}, patch_type=patch_type,
                    )
                except Exception as e:
                    print(f"WARNING: measure failed {d}/{measure}/{task.name}: {e}")
                    continue

                for fig, meta in measurement_figs(
                    measurements, auc_plots=False, discovery_ablation=d,
                    omit_algo=_omit_gt,
                ):
                    metric_slug = meta["metric_key"].lower().replace(" ", "_")
                    abl_slug = _ABL_SHORT_FROM_STR.get(
                        meta["ablation_type"], _norm_slug(meta["ablation_type"])
                    )
                    rel = d_dir / _task_slug(task_key) / measure / abl_slug / f"{metric_slug}.pdf"
                    rel.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        fig.write_image(str(rel))
                    except Exception as e:
                        print(f"WARNING: write figure failed {rel}: {e}")

                # Summary-table rows: faithfulness at the ground-truth edge count.
                budget = task.true_edge_count
                for ev in EVAL_ABLATION_TYPES:
                    for m in metrics:
                        for algo_key in algo_ps:
                            try:
                                series = extract_metric(measurements, ev, m.key, task_key, algo_key)
                            except KeyError:
                                continue
                            # Anchors have no discovery sweep → label "anchor" (stored once).
                            disc_label = "anchor" if algo_key in _ANCHOR_ALGO_KEYS else d
                            # Figures follow the GT-native-ablation rule; summary.csv
                            # keeps every row so the dropped values stay inspectable.
                            if not _omit_gt(task_key, algo_key, ev):
                                overlay[(task_key, measure, m.key)].append(
                                    (algo_key, disc_label, _ablation_slug(ev), series)
                                )
                            summary_rows.append({
                                "discovery_ablation": d,
                                "task": task.name,
                                "algo": PRUNE_ALGO_DICT[algo_key].short_name,
                                "measure": measure,
                                "eval_ablation": _ablation_slug(ev),
                                "metric": m.key,
                                "budget_edges": budget,
                                "value": _value_at_budget(series, budget),
                            })

    # ── Cross-cell circuit-similarity heatmaps (algo x discovery-ablation axes) ──
    write_all_similarity_heatmaps(
        groups=groups,
        discovery_ablations=DISCOVERY_ABLATIONS,
        algo_order=[_ALGO_MAP[a].key for a in SWEPT_ALGOS],
        anchor_algo_keys=[_ALGO_MAP[a].key for a in ANCHOR_ALGOS],
        edge_counts=args.similarity_edge_counts,
        experiment_folder=experiment_folder,
    )

    # ── Interactive overlay HTMLs (cross-cell comparison) ──
    _write_interactive(experiment_folder, overlay)

    # ── Summary tables ──
    _write_summary(experiment_folder, summary_rows)

    manifest.update({
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "summary_rows": len(summary_rows),
    })
    write_manifest(experiment_folder / "manifest.json", manifest)
    print(f"\nDone. {len(summary_rows)} summary rows. Results in: {experiment_folder}")


def _write_interactive(folder: Path, overlay: dict) -> None:
    """One interactive overlay HTML per (task, measure, metric) under interactive/.

    Also dumps the raw curve data to interactive/overlay.pkl so the plots can be
    re-styled later (replot_interactive.py) without re-measuring the metrics.
    """
    if not overlay:
        print("WARNING: no overlay curves produced.")
        return
    interactive_dir = folder / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    with open(interactive_dir / "overlay.pkl", "wb") as f:
        pickle.dump(dict(overlay), f)
    eval_labels = [_ablation_slug(a) for a in EVAL_ABLATION_TYPES]
    default_eval = _ablation_slug(AblationType.RESAMPLE)
    algo_display = {k: v.short_name for k, v in PRUNE_ALGO_DICT.items()}
    n = 0
    for (task_key, measure, metric_key), entries in overlay.items():
        metric_slug = metric_key.lower().replace(" ", "_")
        out = folder / "interactive" / _task_slug(task_key) / measure / f"{metric_slug}.html"
        try:
            written = save_interactive_curves(
                entries,
                title=f"{TASK_DICT[task_key].name} · {measure} · {metric_key}",
                x_title="edges",
                y_title=metric_key,
                eval_labels=eval_labels,
                default_eval=default_eval,
                out_path=out,
                algo_display=algo_display,
            )
            if written is not None:
                n += 1
        except Exception as e:
            print(f"WARNING: interactive plot failed {out}: {e}")
    print(f"Wrote {n} interactive overlay HTMLs under {folder / 'interactive'}")


def _write_summary(folder: Path, rows: list) -> None:
    if not rows:
        print("WARNING: no summary rows produced.")
        return
    cols = ["discovery_ablation", "task", "algo", "measure", "eval_ablation",
            "metric", "budget_edges", "value"]
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Headline pivot (markdown): faithfulness KL at matched eval==discovery ablation,
    # rows=algo, cols=discovery-ablation, per task — the discovery-relativity table.
    lines = ["# Faithfulness spine summary", "",
             f"Full tidy table: `summary.csv` ({len(rows)} rows).", "",
             "## Headline: Clean-KL faithfulness at the ground-truth edge count",
             "*(TREE-patch, eval-ablation matched to discovery-ablation)*", ""]
    headline = [r for r in rows
                if r["metric"] == CLEAN_KL_DIV_METRIC.key and r["measure"] == "faithfulness"
                and r["eval_ablation"] == r["discovery_ablation"]]
    tasks = sorted({r["task"] for r in headline})
    for task in tasks:
        lines += [f"### {task}", ""]
        algos = sorted({r["algo"] for r in headline if r["task"] == task})
        lines.append("| algo | " + " | ".join(DISCOVERY_ABLATIONS) + " |")
        lines.append("|" + "---|" * (len(DISCOVERY_ABLATIONS) + 1))
        for algo in algos:
            cells = []
            for d in DISCOVERY_ABLATIONS:
                v = next((r["value"] for r in headline
                          if r["task"] == task and r["algo"] == algo
                          and r["discovery_ablation"] == d), None)
                cells.append(f"{v:.4f}" if isinstance(v, float) else "-")
            lines.append(f"| {algo} | " + " | ".join(cells) + " |")
        lines.append("")
    (folder / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
