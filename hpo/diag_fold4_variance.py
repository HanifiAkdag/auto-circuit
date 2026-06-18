"""Fold-4 refit variance check: best_hparams scored 268 on inner_val but the
single finalize refit (seed=42+4=46) collapsed to ~1 on outer_test. Refit at
several seeds to show this is stochastic single-run instability, not a bug.
"""
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_circuit.prune_algos.disco_gp import disco_gp_prune_scores
from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK as TASK
from auto_circuit.types import AblationType
from hpo.cv_utils import kfold_prompt_loaders, register_fold_kv_caches
from hpo.scoring import DEFAULT_EVAL_EDGE_COUNTS, logit_diff_auc

FOLD = 4
EPOCHS = 100
BEST = dict(  # fold4 best_params (trial 42, inner_val logit_diff_auc=268.7)
    e_lr=0.11943022084945148,
    e_lambda_sparse_init=1.6672205658878068,
    e_lambda_complete_init=0.3352998567069518,
    e_max_times_lambda_sparse=49.424801197190924,
    e_min_times_lambda_sparse=0.03231893543076613,
    gs_temp_edge=0.960049117493492,
    e_n_epoch_warmup=23,
    e_n_epoch_cooldown=30,
)
FIXED = dict(use_weight_masks=False, use_edge_masks=True, hard=True,
             loss_mode="answer_margin_ce", logits_e_init=1.0, show_train_graph=False)


def main():
    t.manual_seed(42)
    TASK.init_task()
    model, true_edges = TASK.model, TASK.true_edges
    folds = kfold_prompt_loaders(TASK.train_loader, k=5, seed=42, batch_size=None)
    outer_train, outer_test = folds[FOLD]
    register_fold_kv_caches(model, *[l for p in folds for l in p])

    # seed 46 = finalize's exact refit seed (seed + fold_idx = 42 + 4)
    for seed in (46, 7, 123, 999):
        t.manual_seed(seed)
        ps = disco_gp_prune_scores(model, outer_train, true_edges,
                                   e_train_epochs=EPOCHS, **BEST, **FIXED)
        auc = logit_diff_auc(model, ps, outer_test, DEFAULT_EVAL_EDGE_COUNTS, AblationType.ZERO)
        tag = " (==finalize seed)" if seed == 46 else ""
        print(f"[fold4 refit seed={seed}{tag}] outer_test logit_diff_auc={auc:.3f}")


if __name__ == "__main__":
    main()
