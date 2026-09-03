"""Pillar-2 stability analysis pass (sub-RQ3, issue #5).

Re-reads the cached discovery cells in `.prune_scores_cache/stability/` (4 algorithms ×
3 tasks × 3 discovery ablations × 5 seeds, plus a Random anchor × 3 tasks × 5 seeds) and
asks how much of a discovered circuit is an artifact of the random draw.

Two passes over the same cells:

  structural   edge-set agreement between seeds: pairwise Jaccard, mutual IoU across all
               five, and the selection-frequency histogram, as a function of budget.
               CPU only; models are loaded solely for the edge graph.
  functional   each seed's circuit re-scored with `measure_prune_metrics`, giving the
               spread of faithfulness across seeds at matched budgets. One call per
               (task, algo, ablation, seed) because ACDC's budgets are per-seed.

Structure without function cannot separate "the algorithm is unreliable" from "the task
admits many equally faithful circuits", which is the Pillar-2 question, so both passes feed
one table.

Writes `structural.csv`, `functional.csv`, `summary.csv`, `summary.md` and the figures.
`stability_plots.py` can redraw every figure from the two CSVs alone (no cache, no GPU).

    srun --gres=gpu:1 python experiments/stability_analysis.py
    srun python experiments/stability_analysis.py --skip-functional   # CPU only
"""
import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.metrics.analysis_utils import (
    Budget,
    EdgeIndex,
    budget_mask,
    edge_index,
    edge_score_vector,
    extract_metric,
    latest_per_cell,
    load_many_caches,
    matched_budgets,
    mutual_iou,
    pairwise_jaccard,
    pairwise_stats,
    selection_frequency,
)
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import measure_prune_metrics
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    CLEAN_KL_DIV_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
)
from auto_circuit.prune_algos.prune_algos import ACDC_PRUNE_ALGO, PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import AblationType, PatchType, PruneScores
from auto_circuit.utils.misc import repo_path_to_abs_path

from experiment_utils import algo_hyperparams, git_state, write_manifest
from faithfulness_comparison import (
    TASK_METRICS,
    _ABL_SHORT,
    _COMMON_METRICS,
    _PATCH_SEMANTICS,
    _ablation_slug,
    _reverse_clean_corrupt,
)
from run_cell import _ALGO_MAP, _TASK_TEMPLATES
from stability_cells import (
    ANCHOR_ABLATION,
    ANCHOR_ALGOS,
    ANCHOR_SEED,
    CONDITIONS,
    DISCOVERY_ABLATIONS,
    SEEDS,
    SWEPT_ALGOS,
    TASKS,
    condition_of,
)

STABILITY_CACHE_DIR = ".prune_scores_cache_v3/stability"
RESULTS_DIR = "results/stability"

# Méloux et al. (2025) propose mean pairwise Jaccard > 0.8 (at n >= 100 bootstrap
# resamples) as a tentative bar for calling a circuit stable. Drawn as their reference,
# not as a criterion of ours: n = 5 here, and the perturbation is reseeding.
STABLE_BAR = 0.80

# The metric whose spread across seeds is the headline. Not Clean KL: for a faithful
# circuit its mean approaches zero, so the coefficient of variation is dominated by its
# denominator. These are bounded and higher-better, so a CV means something. Clean KL is
# still recorded in functional.csv, reported as mean ± sd.
STABILITY_METRIC: Dict[str, str] = {
    _TASK_TEMPLATES["ioi"].key: LOGIT_DIFF_PERCENT_METRIC.key,
    _TASK_TEMPLATES["gt"].key: LOGIT_DIFF_PERCENT_METRIC.key,
    _TASK_TEMPLATES["docstring"].key: CORRECT_ANSWER_PERCENT_METRIC.key,
}

FUNCTIONAL_BUDGETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

_ABL_SHORT_INV = {v: k for k, v in _ABL_SHORT.items()}

STRUCTURAL_FIELDS = [
    "task", "algo", "discovery_ablation", "condition", "n_seeds", "selector",
    "budget_label", "budget_value", "realised_mean", "realised_min", "realised_max",
    "jaccard_mean", "jaccard_sd", "jaccard_cv", "jaccard_min", "jaccard_max", "n_pairs",
    "mutual_iou", "n_intersect", "n_union", "freq_5", "freq_4", "freq_3", "freq_2",
    "freq_1",
]
FUNCTIONAL_FIELDS = [
    "task", "algo", "discovery_ablation", "condition", "eval_ablation", "measure",
    "selector", "budget_label", "realised_mean", "metric", "n_seeds", "metric_mean",
    "metric_sd", "metric_cv", "metric_min", "metric_max", "metric_values",
]


@dataclass
class Cell:
    """One (task, algo, discovery-ablation, condition) group, over the seeds found."""

    task: str
    algo: str
    ablation: str
    condition: str = "joint"
    seeds: List[int] = field(default_factory=list)
    scores: List[PruneScores] = field(default_factory=list)

    @property
    def task_key(self) -> str:
        return _TASK_TEMPLATES[self.task].key

    @property
    def algo_key(self) -> str:
        return _ALGO_MAP[self.algo].key

    @property
    def short_name(self) -> str:
        return PRUNE_ALGO_DICT[self.algo_key].short_name

    @property
    def is_anchor(self) -> bool:
        return self.algo in ANCHOR_ALGOS

    def __str__(self) -> str:
        return f"{self.task}/{self.algo}/{self.ablation}/{self.condition}"


def _acdc_tau_grid() -> List[float]:
    """ACDC's tau sweep, taken from the algorithm rather than from a cell manifest.

    The 30 tau-split ACDC × {ioi, gt} cells are written by `merge_acdc_partials.py`, whose
    manifest carries no `algo_hyperparams` block, so the manifest cannot supply this.
    Deriving it from the data would be wrong too: a tau that pruned nothing in one seed is
    simply absent from that seed's score levels.
    """
    hp = algo_hyperparams(ACDC_PRUNE_ALGO)
    taus = [b * 10.0**e for b in hp["tao_bases"] for e in hp["tao_exps"]]
    return sorted(taus, reverse=True)


# Which of the two seeds each algorithm actually consumes. ACDC's only random() is a
# graph-rendering coin flip and resample ablation is deterministic given the batch, so
# neither ACDC nor EAP reads the torch RNG; Random is t.rand_like over masks that carry
# no sequence dimension at component granularity, so the data cannot reach it.
DATA_ONLY_ALGOS = {"acdc", "eap"}
INIT_ONLY_ALGOS = {"random"}


def _is_constant(algo: str, condition: str) -> bool:
    """True where a condition has nothing to vary, so it is not a cell at all."""
    return (algo in DATA_ONLY_ALGOS and condition == "init") or (
        algo in INIT_ONLY_ALGOS and condition == "data"
    )


def _effective_condition(algo: str, condition: str) -> str:
    """Which condition's files hold this algorithm's cells. An algorithm that reads only
    one seed already has that condition's runs: they are its joint cells."""
    if algo in DATA_ONLY_ALGOS and condition == "data":
        return "joint"
    if algo in INIT_ONLY_ALGOS and condition == "init":
        return "joint"
    return condition


def _cell_specs(
    tasks: List[str], algos: List[str], ablations: List[str], conditions: List[str]
) -> List[Tuple[str, str, str, str]]:
    """Every (task, algo, discovery_ablation, condition) group to load, swept algos plus
    anchors, minus the (algo, condition) combinations `_is_constant` rules out."""
    specs = [
        (task, algo, abl, cond)
        for task in tasks
        for algo in algos
        if algo in SWEPT_ALGOS
        for abl in ablations
        for cond in conditions
        if not _is_constant(algo, cond)
    ]
    specs += [
        (task, algo, ANCHOR_ABLATION, cond)
        for task in tasks
        for algo in algos
        if algo in ANCHOR_ALGOS
        for cond in conditions
        if not _is_constant(algo, cond)
    ]
    return specs


def _load_cell(cache_dir: str, spec: Tuple[str, str, str, str], seeds: List[int]) -> Cell:
    """Load one group's seeds.

    Globs narrowly per group rather than loading all caches at once, and re-keys the
    manifest's short algo name: the manifest's own `algo_key` is an ablation variant like
    "ACDC [tokenwise mean corrupt]", which is not in PRUNE_ALGO_DICT.

    The *varying* seed indexes every condition, so all three are keyed 42..46. The files
    for a degenerate condition are the algorithm's joint files (see _effective_condition).

    (ANCHOR_SEED, ANCHOR_SEED) sits on all three axes at once (stability_cells' docstring:
    "All three conditions contain (42, 42)"), but `condition_of` files an equal pair under
    "joint" only, so it is matched separately rather than through `condition_of`.
    """
    task, algo, ablation, condition = spec
    wanted = _effective_condition(algo, condition)
    pattern = f"cell_{task}_{algo.replace('-', '_')}_{ablation}_seed"
    loaded = latest_per_cell(load_many_caches(cache_dir, pattern))
    cell = Cell(task=task, algo=algo, ablation=ablation, condition=condition)
    by_seed: Dict[int, PruneScores] = {}
    for manifest, tps in loaded:
        meta = (manifest or {}).get("cell", {})
        if meta.get("null_test"):
            continue
        data_seed = int(meta.get("seed", -1))
        init_seed = int(meta.get("init_seed", data_seed))
        is_anchor_point = data_seed == init_seed == ANCHOR_SEED
        if condition_of(data_seed, init_seed) != wanted and not is_anchor_point:
            continue
        varying = init_seed if wanted == "init" else data_seed
        if varying not in seeds:
            continue
        by_seed[varying] = next(iter(next(iter(tps.values())).values()))
    for seed in sorted(by_seed):
        cell.seeds.append(seed)
        cell.scores.append(by_seed[seed])
    return cell


def _budgets(cell: Cell, edges: set, true_edge_count: Optional[int]) -> List[Budget]:
    """The budget grid for a cell: matched top-k, or matched tau where scores are tied.

    The grid always ends at the unpruned model. ACDC's coarsest tau admits only ~21k of
    the 32k edges, so without that terminal point its curve stops short of every other
    method's and the sweep never reaches the size where all of them coincide.
    """
    full = Budget("top_k", len(edges))
    if cell.algo == "acdc":
        return [Budget("threshold", tau) for tau in _acdc_tau_grid()] + [full]
    grid = matched_budgets(edges, true_edge_count)
    return [Budget("top_k", k) for k in grid if 10 <= k < len(edges)] + [full]


def _structural_rows(cell: Cell, index: EdgeIndex, budgets: List[Budget]) -> List[dict]:
    """One row per budget: pairwise Jaccard, mutual IoU and the selection-frequency
    histogram over the cell's five seeds, at that budget's realised edge set."""
    vectors = [edge_score_vector(index, ps) for ps in cell.scores]
    task_name = TASK_DICT[cell.task_key].name
    rows = []
    for budget in budgets:
        masks = np.stack([budget_mask(v, budget) for v in vectors])
        assert len(cell.seeds) == 5, (
            f"{cell} has {len(cell.seeds)} seeds; the freq_1..freq_5 columns assume 5 "
            f"and would silently mis-report"
        )
        sizes = masks.sum(1)
        stats = pairwise_stats(pairwise_jaccard(masks))
        iou, n_intersect, n_union = mutual_iou(masks)
        freq = selection_frequency(masks)
        row = dict(
            task=task_name,
            algo=cell.short_name,
            discovery_ablation="anchor" if cell.is_anchor else cell.ablation,
            condition=cell.condition,
            n_seeds=len(cell.seeds),
            selector=budget.kind,
            budget_label=budget.label,
            budget_value=budget.value,
            realised_mean=float(sizes.mean()),
            realised_min=int(sizes.min()),
            realised_max=int(sizes.max()),
            jaccard_mean=stats["mean"],
            jaccard_sd=stats["sd"],
            jaccard_cv=stats["cv"],
            jaccard_min=stats["min"],
            jaccard_max=stats["max"],
            n_pairs=stats["n_pairs"],
            mutual_iou=iou,
            n_intersect=n_intersect,
            n_union=n_union,
        )
        for m in range(1, 6):
            row[f"freq_{m}"] = int(freq[m]) if m < len(freq) else 0
        rows.append(row)
    return rows


def _eval_ablations(cell: Cell) -> List[AblationType]:
    """Every cell is evaluated under all three eval ablations.

    Pillar 1 sweeps the same cross with one seed per cell, so it reports how a score moves
    with the protocol but not how the spread across seeds moves. Only the cross answers
    whether a method's instability is a property of the method or of the protocol it was
    discovered under.
    """
    return [_ABL_SHORT_INV[a] for a in DISCOVERY_ABLATIONS]


def _functional_rows(
    cell: Cell,
    index: EdgeIndex,
    budgets: List[Budget],
    device_scores: List[PruneScores],
    patch_types: List[PatchType],
) -> List[dict]:
    """Re-score each seed's circuit, one `measure_prune_metrics` call per (seed, measure).

    Not one call per seed-group: ACDC resolves a budget to a different edge count in each
    seed, and `test_edge_counts` is shared across the algos in a single call. The patch
    type is a scalar argument, so the two measures are separate calls as well.
    """
    task = TASK_DICT[cell.task_key]
    metrics = TASK_METRICS.get(cell.task_key, _COMMON_METRICS)
    eval_ablations = _eval_ablations(cell)
    n_edges = len(index)

    # (measure, eval, metric, budget_label) -> {seed: value}; realised counts are per seed.
    collected: Dict[Tuple[str, str, str, str], Dict[int, float]] = defaultdict(dict)
    realised_by_budget: Dict[str, List[int]] = defaultdict(list)
    kind_by_budget = {b.label: b.kind for b in budgets}

    for seed, ps in zip(cell.seeds, device_scores):
        # Resolved on the vector, not per Edge: the loop over 32k Edge objects costs
        # ~200 ms a call and the answer is identical.
        scores = edge_score_vector(index, ps)
        realised = {b.label: int(budget_mask(scores, b).sum()) for b in budgets}
        for label, count in realised.items():
            realised_by_budget[label].append(count)
        counts = sorted({c for c in realised.values() if 0 < c <= n_edges})
        if not counts:
            print(f"WARNING: {cell} seed {seed}: no usable budgets, skipping")
            continue
        for patch_type in patch_types:
            measure = _PATCH_SEMANTICS[patch_type]
            measurements = measure_prune_metrics(
                ablation_types=eval_ablations,
                metrics=metrics,
                task_prune_scores={cell.task_key: {cell.algo_key: ps}},
                patch_type=patch_type,
                reverse_clean_corrupt=_reverse_clean_corrupt(patch_type),
                test_edge_counts=counts,
            )
            for ablation in eval_ablations:
                for metric in metrics:
                    series = dict(
                        extract_metric(
                            measurements, ablation, metric.key, cell.task_key,
                            cell.algo_key,
                        )
                    )
                    for label, count in realised.items():
                        if count in series:
                            key = (measure, _ablation_slug(ablation), metric.key, label)
                            collected[key][seed] = float(series[count])

    rows = []
    for (measure, eval_slug, metric_key, label), by_seed in sorted(collected.items()):
        values = np.array([by_seed[s] for s in sorted(by_seed)], dtype=np.float64)
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        rows.append(
            dict(
                task=task.name,
                algo=cell.short_name,
                discovery_ablation="anchor" if cell.is_anchor else cell.ablation,
                condition=cell.condition,
                eval_ablation=eval_slug,
                measure=measure,
                selector=kind_by_budget[label],
                budget_label=label,
                realised_mean=float(np.mean(realised_by_budget[label])),
                metric=metric_key,
                n_seeds=len(values),
                metric_mean=mean,
                metric_sd=sd,
                metric_cv=abs(sd / mean) if mean else float("nan"),
                metric_min=float(values.min()),
                metric_max=float(values.max()),
                metric_values=";".join(f"{v:.6g}" for v in values),
            )
        )
    return rows


def _reference_row(rows: List[dict], target: Optional[int]) -> Optional[dict]:
    """The row whose realised circuit size is closest to the published circuit size."""
    if not rows:
        return None
    if target is None:
        return rows[len(rows) // 2]
    return min(rows, key=lambda r: abs(r["realised_mean"] - target))


def _summary_rows(
    structural: List[dict], functional: List[dict], task_keys: Dict[str, str]
) -> List[dict]:
    """Join structure and function at the reference budget, one row per cell.

    Both reference papers print the two together; read apart, a low Jaccard cannot be
    distinguished from an unreliable algorithm. Faithfulness only, so the headline keeps
    the meaning it had before completeness was measured.
    """
    by_cell: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for row in structural:
        by_cell[
            (row["task"], row["algo"], row["discovery_ablation"],
             row.get("condition", "joint"))
        ].append(row)
    func_by_cell: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for row in functional:
        if row.get("measure", "faithfulness") != "faithfulness":
            continue
        func_by_cell[
            (row["task"], row["algo"], row["discovery_ablation"],
             row.get("condition", "joint"))
        ].append(row)

    rows = []
    for key, srows in sorted(by_cell.items()):
        # Resolve through the cells we loaded: TASK_DICT holds both the token and the
        # component variant of Docstring under the same display name.
        task_key = task_keys[key[0]]
        target = TASK_DICT[task_key].true_edge_count
        ref = _reference_row(srows, target)
        if ref is None:
            continue
        out = {k: ref[k] for k in STRUCTURAL_FIELDS if k in ref}
        out["condition"] = key[3]
        # `_rebuild_summary` re-reads budget_label from structural.csv through `_read`,
        # which coerces any value containing "e" or "." to float (needed for numeric
        # columns) and so mangles ACDC's threshold labels ("7.0e-02" -> 0.07). Rebuild
        # the canonical label from budget_value/selector rather than trusting the string.
        out["budget_label"] = Budget(out["selector"], float(out["budget_value"])).label
        out["true_edge_count"] = target
        # Anchors were evaluated under all three ablations; report them at resample.
        eval_slug = key[2] if key[2] in _ABL_SHORT_INV else "resample"
        primary = STABILITY_METRIC.get(task_key, CLEAN_KL_DIV_METRIC.key)
        for metric_key, prefix in ((primary, "primary"), (CLEAN_KL_DIV_METRIC.key, "kl")):
            candidates = [
                r
                for r in func_by_cell.get(key, [])
                if r["metric"] == metric_key and r["eval_ablation"] == eval_slug
            ]
            best = _reference_row(candidates, target)
            out[f"{prefix}_metric"] = metric_key
            for field_name in ("mean", "sd", "cv", "min", "max"):
                out[f"{prefix}_{field_name}"] = (
                    best[f"metric_{field_name}"] if best else float("nan")
                )
        rows.append(out)
    return rows


def _write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


def _write_summary_md(path: Path, rows: List[dict]) -> None:
    """Render summary.csv as a human-readable table, one section per (task, condition)."""
    lines = [
        "# Pillar-2 stability summary",
        "",
        "Circuits from five discovery runs differing only in random seed, compared at the",
        "published circuit size. `J` is the pairwise Jaccard over the seed pairs (the pairs",
        "are dependent, so the spread is a range, not an interval). `mutual IoU` is the",
        "agreement of all five at once. The functional column is the task's bounded metric;",
        "no CV is reported for Clean KL, whose mean approaches zero for faithful circuits.",
        "",
    ]
    by_task: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        by_task[(row["task"], row.get("condition", "joint"))].append(row)
    for (task, condition), trows in sorted(by_task.items()):
        lines += [
            f"## {task} — {condition}",
            "",
            "| algo | discovery | k | mean J | CV J | min J | mutual IoU | ∩ / ∪ "
            "| metric | µ | sd |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(trows, key=lambda r: (r["algo"] == "Random", r["algo"])):
            lines.append(
                f"| {r['algo']} | {r['discovery_ablation']} | {r['realised_mean']:.0f} "
                f"| {r['jaccard_mean']:.3f} | {r['jaccard_cv']:.3f} "
                f"| {r['jaccard_min']:.3f} | {r['mutual_iou']:.3%} "
                f"| {r['n_intersect']} / {r['n_union']} | {r['primary_metric']} "
                f"| {r['primary_mean']:.3g} | {r['primary_sd']:.3g} |"
            )
        lines.append("")
    lines += [
        "`CV` of the functional metric is unreliable wherever the mean approaches zero (it",
        "divides by it), so the figures plot the standard deviation in the metric's own",
        "units instead; both are in `functional.csv`.",
        "",
        "Reference: Méloux et al. (2025) report mean pairwise Jaccard 0.561 (CV 0.335) for",
        f"EAP-IG under bootstrap resampling, and propose {STABLE_BAR} as a stability bar at",
        "n >= 100 resamples. Our perturbation is a full method re-run at n = 5, so the bar is",
        "quoted for orientation only.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def _experiment_folder(name: Optional[str]) -> Path:
    base = repo_path_to_abs_path(RESULTS_DIR)
    folder = base / (name or f"stability_{datetime.now():%Y-%m-%d}")
    if folder.exists() and name is None:
        folder = base / f"{folder.name}_{datetime.now():%H%M}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _rebuild_summary(folder: Path, skip_figures: bool = False) -> None:
    """Re-join structural.csv and functional.csv without re-running either pass.

    Changing how the table is summarised should not cost a GPU run. Task keys come back
    from the display names in the CSVs, resolved against the three spine tasks.
    """
    from stability_plots import _read

    structural = _read(folder / "structural.csv")
    functional = _read(folder / "functional.csv")
    if not structural:
        raise SystemExit(f"No structural.csv in {folder}")
    task_keys = {
        TASK_DICT[t.key].name: t.key
        for t in (_TASK_TEMPLATES[name] for name in TASKS)
    }
    summary = _summary_rows(structural, functional, task_keys)
    _write_csv(folder / "summary.csv", summary, list(summary[0].keys()))
    _write_summary_md(folder / "summary.md", summary)
    if not skip_figures:
        from stability_plots import write_all_figures

        write_all_figures(folder)


def main() -> None:
    """CLI entry point: build the (task, algo, ablation, condition) cell matrix, run the
    structural/functional passes, and write the CSVs, summary and figures. With
    --summary-only, skips straight to rebuilding the summary from existing CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=STABILITY_CACHE_DIR)
    parser.add_argument("--name", default=None, help="Experiment folder name")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--algos", nargs="+", default=SWEPT_ALGOS + ANCHOR_ALGOS)
    parser.add_argument("--ablations", nargs="+", default=DISCOVERY_ABLATIONS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--functional-budgets", type=int, nargs="+", default=FUNCTIONAL_BUDGETS)
    parser.add_argument(
        "--measures",
        nargs="+",
        choices=sorted(_PATCH_SEMANTICS.values()),
        default=sorted(_PATCH_SEMANTICS.values()),
    )
    parser.add_argument("--skip-functional", action="store_true", help="CPU-only pass")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Inventory the cache and exit (no model load)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild summary.csv/summary.md and the figures from the CSVs already in "
        "--name, without touching the caches. Needs no GPU.",
    )
    args = parser.parse_args()

    if args.summary_only:
        _rebuild_summary(_experiment_folder(args.name), skip_figures=args.skip_figures)
        return

    specs = _cell_specs(args.tasks, args.algos, args.ablations, args.conditions)
    cells = [_load_cell(args.cache_dir, spec, args.seeds) for spec in specs]

    if args.dry_run:
        print(f"{'cell':<48} {'seeds':<20} n")
        for cell in cells:
            print(f"{str(cell):<48} {str(cell.seeds):<20} {len(cell.seeds)}")
        missing = [c for c in cells if len(c.seeds) != len(args.seeds)]
        print(f"\n{len(cells)} cells, {sum(len(c.seeds) for c in cells)} runs, "
              f"{len(missing)} incomplete")
        print(f"ACDC tau grid ({len(_acdc_tau_grid())}): {_acdc_tau_grid()[:3]} ... "
              f"{_acdc_tau_grid()[-1]:.1e}")
        return

    folder = _experiment_folder(args.name)
    print(f"Writing results to: {folder}")
    manifest = {
        "experiment": folder.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "discovery_cache_dir": args.cache_dir,
        "seeds": args.seeds,
        "conditions": args.conditions,
        "anchor_seed": ANCHOR_SEED,
        "eval_protocol": f"measures {sorted(args.measures)} x all three eval ablations, "
        "every cell",
        "eval_data": "TASK_DICT template test_loader, identical for every seed",
        "stability_metric": STABILITY_METRIC,
        "stable_bar": STABLE_BAR,
        "cells": [
            {
                "task": c.task, "algo": c.algo, "ablation": c.ablation,
                "condition": c.condition, "seeds": c.seeds,
            }
            for c in cells
        ],
    }
    write_manifest(folder / "manifest.json", manifest)

    structural: List[dict] = []
    functional: List[dict] = []
    patch_types = [p for p, m in _PATCH_SEMANTICS.items() if m in args.measures]
    for cell in cells:
        if len(cell.seeds) < 2:
            print(f"WARNING: {cell} has {len(cell.seeds)} seed(s), skipping")
            continue
        task = TASK_DICT[cell.task_key]
        edges = task.model.edges
        index = edge_index(edges, cell.scores[0])
        budgets = _budgets(cell, edges, task.true_edge_count)

        structural += _structural_rows(cell, index, budgets)
        print(f"[structural] {cell}: {len(cell.seeds)} seeds, {len(budgets)} budgets")

        if args.skip_functional:
            continue
        device = next(task.model.parameters()).device
        on_device = [{m: s.float().to(device) for m, s in ps.items()} for ps in cell.scores]
        fbudgets = budgets
        if cell.algo != "acdc":
            wanted = sorted(set(args.functional_budgets + ([task.true_edge_count] if task.true_edge_count else [])))
            fbudgets = [Budget("top_k", k) for k in wanted if 0 < k < len(edges)]
            fbudgets.append(Budget("top_k", len(edges)))
        functional += _functional_rows(cell, index, fbudgets, on_device, patch_types)

    _write_csv(folder / "structural.csv", structural, STRUCTURAL_FIELDS)
    if functional:
        _write_csv(folder / "functional.csv", functional, FUNCTIONAL_FIELDS)

    task_keys = {TASK_DICT[c.task_key].name: c.task_key for c in cells}
    summary = _summary_rows(structural, functional, task_keys)
    summary_fields = list(summary[0].keys()) if summary else STRUCTURAL_FIELDS
    _write_csv(folder / "summary.csv", summary, summary_fields)
    _write_summary_md(folder / "summary.md", summary)

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["structural_rows"] = len(structural)
    manifest["functional_rows"] = len(functional)
    write_manifest(folder / "manifest.json", manifest)

    if not args.skip_figures:
        from stability_plots import write_all_figures

        write_all_figures(folder)
    print(f"\nDone. Results in: {folder}")


if __name__ == "__main__":
    main()
