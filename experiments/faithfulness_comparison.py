"""Pillar 1 — Faithfulness spine for sub-RQ3.

Runs all four circuit discovery methods at their *native* discovery ablations:
  EP   → RESAMPLE   DiscoGP-EP → ZERO   ACDC → RESAMPLE   EAP → CORRUPT

Then evaluates every discovered circuit across all eval ablation types and both
patch directions.  This produces the full discovery × eval matrix needed to test
evaluation-relativity (Miller 2024) and cross-method faithfulness comparisons.

Design spec: docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md §4
"""
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.metrics.official_circuits.measure_roc import measure_roc
from auto_circuit.metrics.official_circuits.roc_plot import roc_plot
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    ANSWER_PAIR_KL_DIV_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PROB_DIFF_METRIC,
    PruneMetric,
)
from auto_circuit.prune_algos.prune_algos import (
    ACDC_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
    GROUND_TRUTH_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
    PruneAlgo,
    run_prune_algos,
)
from auto_circuit.tasks import (
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    TASK_DICT,
    Task,
)
from auto_circuit.types import AblationType, PatchType, TaskPruneScores
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache
from auto_circuit.visualize_circuit_slider import save_circuit_sankey_slider
from experiment_utils import algo_hyperparams, git_state, write_manifest

# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these to change what runs
# ─────────────────────────────────────────────────────────────────────────────

# Fresh discovery: prior caches predate the model-sharing fix (tasks.py deepcopy).
# IOI/GT shared one gpt2-small, double-wrapping GT's model. Measured effect is small
# (EAP/eval bit-identical; EP/DiscoGP/ACDC shift ~1% on masks, ~4x run-to-run noise) —
# not corruption, but load-order-dependent, so recompute for clean reproducible scores.
compute_prune_scores = True
save_prune_scores = True
load_prune_scores = False
load_prune_scores_file: str | None = None

# Component-level for all three tasks: granularity is held constant (Greater-Than only
# exists as a component circuit, so this is the sole uniform option). Token-level
# Docstring is reserved for a separate granularity sub-study — see thesis.md
# "Docstring granularity" note. (Heimersheim & Janiak 2023; Miller et al. 2024.)
TASKS: List[Task] = [
    IOI_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
]

# Native-ablation algos (Step 0).  EP=RESAMPLE, DiscoGP-EP=ZERO, ACDC=RESAMPLE,
# EAP=CORRUPT.  GT and Random are ablation-independent anchors.
PRUNE_ALGOS: List[PruneAlgo] = [
    GROUND_TRUTH_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    ACDC_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
]

# Eval ablations — sweep all three so the same discovered circuit is scored
# under each.  This is the free eval-axis sweep (no retraining).
EVAL_ABLATION_TYPES: List[AblationType] = [
    AblationType.RESAMPLE,
    AblationType.ZERO,
    AblationType.TOKENWISE_MEAN_CORRUPT,
]

# Per-task metric selection (spec §3 + hard compatibility).
# Hard incompatibility: CORRECT_ANSWER_PERCENT asserts batch.answers is a Tensor, which
# fails on Greater-Than (list-valued answers: "any year > XX"). LOGIT_DIFF_PERCENT is
# unstable/uninformative on Docstring (spec "avoid"). KL metrics and the answer-diff/val
# metrics handle both tensor- and list-valued answers, so they run on every task.
_COMMON_METRICS = [
    CLEAN_KL_DIV_METRIC,         # faithfulness — all tasks
    ANSWER_PAIR_KL_DIV_METRIC,   # DiscoGP paper comparison
]
TASK_METRICS: dict = {
    IOI_COMPONENT_CIRCUIT_TASK.key: _COMMON_METRICS + [
        LOGIT_DIFF_PERCENT_METRIC,      # primary (IOI)
        PROB_DIFF_METRIC,
        CORRECT_ANSWER_PERCENT_METRIC,
    ],
    GREATERTHAN_COMPONENT_CIRCUIT_TASK.key: _COMMON_METRICS + [
        LOGIT_DIFF_PERCENT_METRIC,      # primary (GT)
        PROB_DIFF_METRIC,               # secondary (GT)
        # CORRECT_ANSWER_PERCENT excluded — incompatible (list answers)
    ],
    DOCSTRING_COMPONENT_CIRCUIT_TASK.key: _COMMON_METRICS + [
        CORRECT_ANSWER_PERCENT_METRIC,  # secondary (Docstring)
        # LOGIT_DIFF_PERCENT excluded — unstable on Docstring (spec "avoid")
    ],
}

PRUNE_CACHE_DIR = ".prune_scores_cache"
RESULTS_DIR = "results/faithfulness_comparison"

# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

task_prune_scores: TaskPruneScores = defaultdict(dict)
_loaded_file: str | None = None
_algo_hyperparams: dict = {}

if compute_prune_scores:
    _algo_hyperparams = {a.key: algo_hyperparams(a) for a in PRUNE_ALGOS}
    task_prune_scores = run_prune_algos(TASKS, PRUNE_ALGOS)

if load_prune_scores:
    assert load_prune_scores_file is not None, "Set load_prune_scores_file"
    _loaded_file = load_prune_scores_file
    loaded = load_cache(PRUNE_CACHE_DIR, load_prune_scores_file)
    task_prune_scores = {k: v | task_prune_scores[k] for k, v in loaded.items()}

if save_prune_scores:
    save_cache(task_prune_scores, PRUNE_CACHE_DIR, "faithfulness-prune-scores")

# Cast all scores to float32 for consistent downstream arithmetic.
for task_key, algo_scores in task_prune_scores.items():
    for algo_key, prune_scores in algo_scores.items():
        for mod, scores in prune_scores.items():
            task_prune_scores[task_key][algo_key][mod] = scores.float()

# ─────────────────────────────────────────────────────────────────────────────
# Experiment folder + first manifest write
# ─────────────────────────────────────────────────────────────────────────────

experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_folder = repo_path_to_abs_path(RESULTS_DIR) / experiment_id
experiment_folder.mkdir(parents=True, exist_ok=True)

_manifest: dict = {
    "experiment_id": experiment_id,
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "git": git_state(),
    "flags": {
        "compute_prune_scores": compute_prune_scores,
        "save_prune_scores": save_prune_scores,
        "load_prune_scores": load_prune_scores,
    },
    "loaded_prune_scores_file": _loaded_file,
    "tasks": [{"key": k, "name": TASK_DICT[k].name} for k in task_prune_scores],
    "algos": sorted({a for ps in task_prune_scores.values() for a in ps}),
    "algo_hyperparams": _algo_hyperparams,
    "eval_ablation_types": EVAL_ABLATION_TYPES,
    "metrics_per_task": {k: [m.key for m in ms] for k, ms in TASK_METRICS.items()},
}
write_manifest(experiment_folder / "manifest.json", _manifest)

# ─────────────────────────────────────────────────────────────────────────────
# Circuit graph visualisations
# ─────────────────────────────────────────────────────────────────────────────

circuit_graph_paths: list = []
for task_key, algo_prune_scores in task_prune_scores.items():
    task = TASK_DICT[task_key]
    task_slug = task.name.lower().replace(" ", "_")
    for algo_key, ps in algo_prune_scores.items():
        algo = PRUNE_ALGO_DICT[algo_key]
        # A viz failure must never abort the run after discovery has completed.
        try:
            html_path = save_circuit_sankey_slider(
                prune_scores=ps,
                model=task.model,
                algo_name=algo.name,
                task_name=task.name,
                output_dir=experiment_folder / task_slug / "circuit_graphs",
                max_edges=10000,
                true_edge_count=task.true_edge_count,
                seq_labels=task.test_loader.seq_labels,
            )
            circuit_graph_paths.append(f"{task_slug}/circuit_graphs/{html_path.name}")
        except Exception as e:
            print(f"WARNING: circuit graph failed for {task.name} / {algo.name}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — free sweep over eval ablations × patch types
# ─────────────────────────────────────────────────────────────────────────────

# Each entry is (fig, relative_path). Figures are organised as
# <task>/<faithfulness|completeness>/<eval-ablation>/<metric>.pdf for easy navigation;
# ROC (spans all tasks) stays at the experiment root.
figs: list = []

# ROC vs ground-truth circuits (patch-type independent).
roc_measurements = measure_roc(task_prune_scores)
figs.append((roc_plot(roc_measurements), Path("roc.pdf")))

# Metric curves: call once per patch type so faithfulness and completeness figures
# are produced separately (measure_prune_metrics takes a single patch_type arg).
#
# Patch direction → what is ablated → what it measures (Miller 2024, Table 3, clean
# input / reverse_clean_corrupt=False):
#   TREE_PATCH = ablate the complement (keep circuit)  → faithfulness  (Row 3)
#   EDGE_PATCH = ablate the circuit    (keep complement) → completeness (Row 1)
# The ablation value (zero/resample/mean) is what the ablated edges are set to.
_PATCH_SEMANTICS = {
    PatchType.TREE_PATCH: ("faithfulness", "ablate-complement"),
    PatchType.EDGE_PATCH: ("completeness", "ablate-circuit"),
}
# Run per task so each task uses only its compatible metrics (TASK_METRICS).
# auc_plots=False: the AUC summary asserts a 0-edge baseline point, which binary-score
# algos (Ground Truth: 2 score-groups -> edge_counts_util omits 0) don't produce.
for patch_type in (PatchType.TREE_PATCH, PatchType.EDGE_PATCH):
    measure, _ablated = _PATCH_SEMANTICS[patch_type]  # _ablated documented in comment above
    for task_key, algo_ps in task_prune_scores.items():
        task_slug = TASK_DICT[task_key].name.lower().replace(" ", "_")
        metrics = TASK_METRICS.get(task_key, _COMMON_METRICS)
        try:
            measurements = measure_prune_metrics(
                ablation_types=EVAL_ABLATION_TYPES,
                metrics=metrics,
                task_prune_scores={task_key: algo_ps},
                patch_type=patch_type,
            )
            for fig, meta in measurement_figs(measurements, auc_plots=False):
                metric_slug = meta["metric_key"].lower().replace(" ", "_")
                ablation_slug = str(meta["ablation_type"]).split(".")[-1].lower()
                suffix = "_auc" if meta["is_auc"] else ""
                # <task>/<measure>/<ablation>/<metric>.pdf  e.g.
                # docstring/completeness/zero/correct_answer_percent.pdf
                rel = Path(task_slug) / measure / ablation_slug / f"{metric_slug}{suffix}.pdf"
                figs.append((fig, rel))
        except Exception as e:
            print(f"WARNING: metric figures failed for {task_slug} / {measure}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Save figures + final manifest
# ─────────────────────────────────────────────────────────────────────────────

figure_names = []
for fig, rel in figs:
    try:
        path = experiment_folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path))
        figure_names.append(str(rel))
    except Exception as e:
        print(f"WARNING: failed to write figure {rel}: {e}")

_manifest.update({
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "patch_types": [PatchType.TREE_PATCH, PatchType.EDGE_PATCH],
    "figures_saved": figure_names,
    "circuit_graphs": circuit_graph_paths,
})
write_manifest(experiment_folder / "manifest.json", _manifest)

print(f"\nDone. Results in: {experiment_folder}")
