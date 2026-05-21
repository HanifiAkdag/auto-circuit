"""Per-algorithm hyperparameter search spaces for Optuna.

Each entry in SEARCH_SPACES maps an algo key (matching PRUNE_ALGO_DICT) to a
SearchSpace dataclass that holds:

  base_func          The raw underlying function (not the partial-wrapped PruneAlgo.func).
  suggest            Callable that takes an optuna.Trial and returns a dict of kwargs.
  fidelity_param     Name of the kwarg that controls training duration (used by
                     Hyperband as the resource axis). None for non-iterative algos.
  supports_intermediate  Whether the algo's training loop can call trial.report()
                         via the eval_callback mechanism.

For non-iterative algos (ACDC, EAP) set supports_intermediate=False; the Optuna
pruner will be replaced with NopPruner automatically.

Edit this file to add new algos or adjust search ranges.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import optuna

from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.prune_algos.disco_gp import disco_gp_prune_scores
from auto_circuit.prune_algos.edge_attribution_patching import (
    edge_attribution_patching_prune_scores,
)
from auto_circuit.prune_algos.edge_pruning import edge_pruning_prune_scores
from auto_circuit.types import AblationType


@dataclass
class SearchSpace:
    base_func: Callable[..., Any]
    suggest: Callable[[optuna.Trial], Dict[str, Any]]
    fidelity_param: Optional[str]
    supports_intermediate: bool
    # Ablation type to use when *evaluating* this algo's prune scores (inner CV
    # scoring + outer-test scoring). Should match the ablation the algo was trained
    # with. None falls back to the --ablation-type CLI flag.
    eval_ablation_type: Optional[AblationType] = field(default=None)


# ---------------------------------------------------------------------------
# DiscoGP — edge-only (hard straight-through Gumbel-sigmoid)
# ---------------------------------------------------------------------------
def _suggest_disco_edge_only(trial: optuna.Trial) -> Dict[str, Any]:
    return dict(
        use_weight_masks=False,
        use_edge_masks=True,
        e_lr=trial.suggest_float("e_lr", 3e-2, 3e-1, log=True),
        e_lambda_sparse_init=trial.suggest_float("e_lambda_sparse_init", 3e-1, 3.0, log=True),
        e_lambda_complete_init=trial.suggest_float("e_lambda_complete_init", 3e-1, 3.0, log=True),
        e_max_times_lambda_sparse=trial.suggest_float("e_max_times_lambda_sparse", 3e1, 3e2, log=True),
        e_min_times_lambda_sparse=trial.suggest_float("e_min_times_lambda_sparse", 1e-3, 1e-1, log=True),
        gs_temp_edge=trial.suggest_float("gs_temp_edge", 0.5, 2.0, log=True),
        e_n_epoch_warmup=trial.suggest_int("e_n_epoch_warmup", 10, 30),
        e_n_epoch_cooldown=trial.suggest_int("e_n_epoch_cooldown", 10, 30),
        hard=True,
        show_train_graph=False,
    )


# ---------------------------------------------------------------------------
# DiscoGP — edge-only, paper-faithful (2-class answer-margin CE)
# ---------------------------------------------------------------------------
def _suggest_disco_edge_only_paper(trial: optuna.Trial) -> Dict[str, Any]:
    hparams = _suggest_disco_edge_only(trial)
    hparams["loss_mode"] = "answer_margin_ce"
    return hparams


# ---------------------------------------------------------------------------
# DiscoGP — joint weight + edge pruning
# ---------------------------------------------------------------------------
def _suggest_disco_joint(trial: optuna.Trial) -> Dict[str, Any]:
    edge_hparams = _suggest_disco_edge_only(trial)
    weight_hparams = dict(
        use_weight_masks=True,
        w_lr=trial.suggest_float("w_lr", 1e-3, 5e-1, log=True),
        w_lambda_sparse_init=trial.suggest_float("w_lambda_sparse_init", 1e-2, 1e1, log=True),
        w_max_times_lambda_sparse=trial.suggest_float("w_max_times_lambda_sparse", 1e1, 1e3, log=True),
        w_min_times_lambda_sparse=trial.suggest_float("w_min_times_lambda_sparse", 1e-2, 1e1, log=True),
        gs_temp_weight=trial.suggest_float("gs_temp_weight", 1e-3, 1.0, log=True),
        w_n_epoch_warmup=trial.suggest_int("w_n_epoch_warmup", 100, 500),
        w_n_epoch_cooldown=trial.suggest_int("w_n_epoch_cooldown", 1, 50),
    )
    return {**edge_hparams, **weight_hparams}


# ---------------------------------------------------------------------------
# Edge Pruning (L0 / Hard Concrete)
# ---------------------------------------------------------------------------
def _suggest_edge_pruning(trial: optuna.Trial) -> Dict[str, Any]:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 2.0, log=True),
        lambda_learning_rate=trial.suggest_float("lambda_learning_rate", 1e-2, 2.0, log=True),
        target_edge_sparsity=trial.suggest_float("target_edge_sparsity", 0.8, 1.2),
        target_node_sparsity=trial.suggest_float("target_node_sparsity", 0.5, 1.0),
        sparsity_warmup_steps=trial.suggest_int("sparsity_warmup_steps", 500, 3000),
        init_log_alpha=trial.suggest_float("init_log_alpha", 5.0, 15.0),
        show_train_graph=False,
    )


# ---------------------------------------------------------------------------
# ACDC (threshold sweep — no iterative training)
# ---------------------------------------------------------------------------
def _suggest_acdc(trial: optuna.Trial) -> Dict[str, Any]:
    min_exp = trial.suggest_int("tao_exp_min", -7, -3)
    max_exp = trial.suggest_int("tao_exp_max", min_exp + 1, -1)
    return dict(
        tao_exps=list(range(min_exp, max_exp + 1)),
        tao_bases=[1, 3, 5, 7, 9],
        show_graphs=False,
    )


# ---------------------------------------------------------------------------
# Edge Attribution Patching (nearly hyperparameter-free)
# ---------------------------------------------------------------------------
def _suggest_eap(trial: optuna.Trial) -> Dict[str, Any]:
    # Only knob: whether to use answer_diff or not
    answer_diff = trial.suggest_categorical("answer_diff", [True, False])
    return dict(answer_diff=answer_diff)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SEARCH_SPACES: Dict[str, SearchSpace] = {
    "DiscoGP Edge Only": SearchSpace(
        base_func=disco_gp_prune_scores,
        suggest=_suggest_disco_edge_only,
        fidelity_param="e_train_epochs",
        supports_intermediate=True,
        eval_ablation_type=AblationType.ZERO,
    ),
    "DiscoGP Edge Only Paper": SearchSpace(
        base_func=disco_gp_prune_scores,
        suggest=_suggest_disco_edge_only_paper,
        fidelity_param="e_train_epochs",
        supports_intermediate=True,
        eval_ablation_type=AblationType.ZERO,
    ),
    "DiscoGP": SearchSpace(
        base_func=disco_gp_prune_scores,
        suggest=_suggest_disco_joint,
        fidelity_param="e_train_epochs",
        supports_intermediate=True,
        eval_ablation_type=AblationType.ZERO,
    ),
    "Edge Pruning": SearchSpace(
        base_func=edge_pruning_prune_scores,
        suggest=_suggest_edge_pruning,
        fidelity_param="steps",
        supports_intermediate=True,
    ),
    "ACDC": SearchSpace(
        base_func=acdc_prune_scores,
        suggest=_suggest_acdc,
        fidelity_param=None,
        supports_intermediate=False,
    ),
    "Edge Attribution Patching": SearchSpace(
        base_func=edge_attribution_patching_prune_scores,
        suggest=_suggest_eap,
        fidelity_param=None,
        supports_intermediate=False,
    ),
}
