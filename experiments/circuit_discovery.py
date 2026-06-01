#%%
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.metrics.completeness_metrics.same_under_knockouts import (
    TaskCompletenessScores,
    measure_same_under_knockouts,
    same_under_knockouts_fig,
)
from auto_circuit.metrics.completeness_metrics.train_same_under_knockouts import (
    train_same_under_knockouts,
)
from auto_circuit.metrics.official_circuits.measure_roc import measure_roc
from auto_circuit.metrics.official_circuits.roc_plot import roc_plot
from auto_circuit.metrics.prune_metrics.head_heatmap import head_importance_heatmap
from auto_circuit.metrics.prune_metrics.measure_prune_metrics import (
    measure_prune_metrics,
    measurement_figs,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    ANSWER_PROB_METRIC,
    CLEAN_KL_DIV_METRIC,
    CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,
    CORRECT_ANSWER_PERCENT_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
    PruneMetric,
)
from auto_circuit.metrics.prune_scores_similarity import prune_score_similarities_plotly
from auto_circuit.prune_algos.prune_algos import (
    GROUND_TRUTH_PRUNE_ALGO,
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
    DISCO_GP_PRUNE_ALGO,
    EDGE_PRUNING_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    ACDC_PRUNE_ALGO,
    DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO,
    PruneAlgo,
    run_prune_algos,
)
from auto_circuit.tasks import (
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    TASK_DICT,
    Task,
)
from auto_circuit.types import (
    AblationType,
    PatchType,
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache
from auto_circuit.visualize_circuit_slider import save_circuit_sankey_slider
from experiment_utils import git_state, write_manifest

figs: list = []  # list of (fig, name) tuples

# ------------------------------------ Prune Scores ------------------------------------

compute_prune_scores = False
save_prune_scores = False
load_prune_scores = True

task_prune_scores: TaskPruneScores = defaultdict(dict)
cache_folder_name = ".prune_scores_cache"
_loaded_prune_scores_file: str | None = None
if compute_prune_scores:
    TASKS: List[Task] = [
        # Token Circuits
        # SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
        # IOI_TOKEN_CIRCUIT_TASK,
        # DOCSTRING_TOKEN_CIRCUIT_TASK,
        # Component Circuits
        # SPORTS_PLAYERS_COMPONENT_CIRCUIT_TASK,
        IOI_COMPONENT_CIRCUIT_TASK,
        # DOCSTRING_COMPONENT_CIRCUIT_TASK,
        # GREATERTHAN_COMPONENT_CIRCUIT_TASK,
        # Autoencoder Component Circuits
        # IOI_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
        # GREATERTHAN_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK
        # ANIMAL_DIET_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
        # CAPITAL_CITIES_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    ]
    PRUNE_ALGOS: List[PruneAlgo] = [
        GROUND_TRUTH_PRUNE_ALGO,
        RANDOM_PRUNE_ALGO,
        EDGE_ATTR_PATCH_PRUNE_ALGO,
        # ACDC_PRUNE_ALGO,
        # DISCO_GP_EDGE_ONLY_PRUNE_ALGO,
        # DISCO_GP_PRUNE_ALGO,  # full joint weight+edge variant
        DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO,
        # EDGE_PRUNING_PRUNE_ALGO,
    ]
    task_prune_scores = run_prune_algos(TASKS, PRUNE_ALGOS)
if load_prune_scores:
    # 2000 epoch IOI Docstring tensor prune_scores post-kv-cache-fix
    # batch_size=128, batch_count=2, default seed (for both)
    filename = "task-prune-scores-28-05-2026_23-29-07.pkl"

    # 1000 epoch Sport Players tensor prune_scores post-kv-cache-fix
    # batch_size=(10, 20), batch_count=(10, 5), default seed
    # filename = "task-prune-scores-16-02-2024_22-22-43.pkl"

    _loaded_prune_scores_file = filename
    loaded_cache = load_cache(cache_folder_name, filename)
    task_prune_scores = {k: v | task_prune_scores[k] for k, v in loaded_cache.items()}
if save_prune_scores:
    base_filename = "task-prune-scores"
    save_cache(task_prune_scores, cache_folder_name, base_filename)

for task, algo_prune_scores in task_prune_scores.items():
    for algo, prune_scores in algo_prune_scores.items():
        for module_name, scores in prune_scores.items():
            # Convert dtype to float32
            task_prune_scores[task][algo][module_name] = scores.float()

# docstring_key = DOCSTRING_TOKEN_CIRCUIT_TASK.key
# task_prune_scores = {docstring_key: task_prune_scores[docstring_key]}

# Create a unique experiment folder once, shared by all output sections below
experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_folder: Path = repo_path_to_abs_path("results/circuit_discovery")
experiment_folder = base_folder / experiment_id
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
    "loaded_prune_scores_file": _loaded_prune_scores_file,
}
write_manifest(experiment_folder / "manifest.json", _manifest)

# ---------------------------------- Head Heatmaps -------------------------------------

for task_key, algo_prune_scores in task_prune_scores.items():
    task = TASK_DICT[task_key]
    for algo_key, ps in algo_prune_scores.items():
        algo = PRUNE_ALGO_DICT[algo_key]
        head_importance_heatmap(
            prune_scores=ps,
            model=task.model,
            algo_name=algo.name,
            task_name=task.name,
            aggregation="sum",
            output_dir=experiment_folder / "head_heatmaps",
            file_formats=("html", "png"),
        )

# -------------------------------- Draw Circuit Graphs ---------------------------------

circuit_graph_paths: list = []
if True:
    for task_key, algo_prune_scores in task_prune_scores.items():
        task = TASK_DICT[task_key]
        for algo_key, ps in algo_prune_scores.items():
            algo = PRUNE_ALGO_DICT[algo_key]
            html_path = save_circuit_sankey_slider(
                prune_scores=ps,
                model=task.model,
                algo_name=algo.name,
                task_name=task.name,
                output_dir=experiment_folder / "circuit_graphs",
                max_edges=10000,
                true_edge_count=task.true_edge_count,
                seq_labels=task.test_loader.seq_labels,
            )
            circuit_graph_paths.append(html_path.name)

# ------------------------------ Prune Scores Similarity -------------------------------

if True:
    prune_scores_similartity_fig = prune_score_similarities_plotly(
        task_prune_scores, [10, 100, 1000, 10000], ground_truths=True
    )
    figs.append((prune_scores_similartity_fig, "prune_scores_similarity"))

# ------------------------------------ Completeness ------------------------------------

compute_task_completeness_scores = False
save_task_completeness_scores = False
load_task_completeness_scores = False
completeness_prune_scores: TaskPruneScores = {}

faithfulness_target = "kl_div"
if compute_task_completeness_scores:
    completeness_prune_scores: TaskPruneScores = train_same_under_knockouts(
        task_prune_scores,
        algo_keys=["Official Circuit", "Random"],
        learning_rate=0.02,
        epochs=300,
        regularize_lambda=0,
        faithfulness_target=faithfulness_target,
    )

cache_folder_name = ".completeness_scores"
if save_task_completeness_scores:
    base_filename = f"{faithfulness_target}-task-completeness-prune-scores"
    save_cache(completeness_prune_scores, cache_folder_name, base_filename)

if load_task_completeness_scores:
    # for task-prune-scores-16-02-2024_23-27-49.pkl (IOI Docstring 2000 epochs)
    # IOI Docstring 100 epoch KL completeness
    filename = "task-completeness-prune-scores-20-02-2024_19-15-29.pkl"
    # IOI Docstring 300 epoch KL completeness
    filename = "task-completeness-prune-scores-23-02-2024_19-54-47.pkl"

    # for task-prune-scores-16-02-2024_22-22-43.pkl (Sports Players 1000 epochs)
    # Sports Players 100 epoch completeness
    # filename = "task-completeness-prune-scores-20-02-2024_22-55-47.pkl"
    completeness_prune_scores = load_cache(cache_folder_name, filename)

if completeness_prune_scores:
    task_completeness_scores: TaskCompletenessScores = measure_same_under_knockouts(
        circuit_ps=task_prune_scores,
        knockout_ps=completeness_prune_scores,
    )
    completeness_fig = same_under_knockouts_fig(task_completeness_scores)
    figs.append((completeness_fig, f"completeness_{faithfulness_target}"))

# ---------------------------------------- ROC -----------------------------------------

if True:
    roc_measurements: TaskMeasurements = measure_roc(task_prune_scores)
    roc_fig = roc_plot(roc_measurements)
    figs.append((roc_fig, "roc"))

# ----------------------------- Prune Metric Measurements ------------------------------

compute_prune_metric_measurements = True
save_prune_metric_measurements = False
load_prune_metric_measurements = False

cache_folder_name = ".measurement_cache"
prune_metric_measurements = None
_loaded_measurements_file: str | None = None
_metrics: list = []
_ablation_types: list = []
if compute_prune_metric_measurements:
    ABLATION_TYPES: List[AblationType] = [
        AblationType.RESAMPLE,  # Docstring
        AblationType.TOKENWISE_MEAN_CORRUPT,  # IOI and Sports Players
        AblationType.ZERO,
    ]
    PRUNE_METRICS: List[PruneMetric] = [
        CLEAN_KL_DIV_METRIC,
        # CORRUPT_KL_DIV_METRIC,
        ANSWER_PROB_METRIC,
        # ANSWER_LOGIT_METRIC,
        # WRONG_ANSWER_LOGIT_METRIC,
        # LOGIT_DIFF_METRIC,
        CORRECT_ANSWER_PERCENT_METRIC,  # Docstring
        LOGIT_DIFF_PERCENT_METRIC,  # IOI
        CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT_METRIC,  # Sports Players
    ]
    _metrics = [m.key for m in PRUNE_METRICS]
    _ablation_types = list(ABLATION_TYPES)
    prune_metric_measurements = measure_prune_metrics(
        ablation_types=ABLATION_TYPES,
        metrics=PRUNE_METRICS,
        task_prune_scores=task_prune_scores,
        patch_type=PatchType.TREE_PATCH,
        reverse_clean_corrupt=False,
    )
    if save_prune_metric_measurements:
        assert prune_metric_measurements is not None
        base_filename = "seq-circuit"
        save_cache(prune_metric_measurements, cache_folder_name, base_filename)
if load_prune_metric_measurements:
    # 2000 epoch IOI Docstring tensor prune_scores post-kv-cache-fix
    filename = "seq-circuit-22-02-2024_20-14-25.pkl"
    # filename = "seq-circuit-18-02-2024_17-20-57.pkl"

    # 1000 epoch Sport Players tensor prune_scores post-kv-cache-fix
    # batch_size=(10, 20), batch_count=(10, 5), default seed
    # filename="seq-circuit-18-02-2024_22-09-26.pkl"

    _loaded_measurements_file = filename
    prune_metric_measurements = load_cache(cache_folder_name, filename)

if prune_metric_measurements is not None:
    for fig, meta in measurement_figs(prune_metric_measurements):
        metric_slug = meta["metric_key"].lower().replace(" ", "_")
        ablation = str(meta["ablation_type"]).split(".")[-1].lower()
        name = f"metric_{metric_slug}_{ablation}"
        if meta["is_auc"]:
            name += "_auc"
        figs.append((fig, name))

# -------------------------------------- Figures ---------------------------------------

figure_names: list = []
for fig, name in figs:
    fig.write_image(str(experiment_folder / f"{name}.pdf"))
    figure_names.append(f"{name}.pdf")

_manifest.update({
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "flags": {
        **_manifest["flags"],
        "compute_prune_metric_measurements": compute_prune_metric_measurements,
        "save_prune_metric_measurements": save_prune_metric_measurements,
        "load_prune_metric_measurements": load_prune_metric_measurements,
        "compute_task_completeness_scores": compute_task_completeness_scores,
        "save_task_completeness_scores": save_task_completeness_scores,
        "load_task_completeness_scores": load_task_completeness_scores,
    },
    "loaded_measurements_file": _loaded_measurements_file,
    "tasks": [{"key": k, "name": TASK_DICT[k].name} for k in task_prune_scores],
    "algos": sorted({a for ps in task_prune_scores.values() for a in ps}),
    "metrics": _metrics,
    "ablation_types": _ablation_types,
    "patch_type": PatchType.TREE_PATCH,
    "figures_saved": figure_names,
    "circuit_graphs": circuit_graph_paths,
})
write_manifest(experiment_folder / "manifest.json", _manifest)

#%%
