"""Diagnose the HPO refit collapse: best_params score ~228 on inner_val but
~0.001 on outer_test after refit. Rebuild fold 0 exactly as finalize.py does,
refit with (a) HPO best_params at several seeds, (b) default-registration
hparams, and report the logit-diff curve + circuit edge count on outer_test.
"""
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_circuit.prune_algos.disco_gp import disco_gp_prune_scores
from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK as TASK
from auto_circuit.types import AblationType, PatchType
from auto_circuit.prune import run_circuits
from auto_circuit.metrics.prune_metrics.prune_metrics import LOGIT_DIFF_PERCENT_METRIC

from hpo.cv_utils import kfold_prompt_loaders, register_fold_kv_caches
from hpo.scoring import DEFAULT_EVAL_EDGE_COUNTS, logit_diff_auc

SEED = 42
FOLD = 0
EPOCHS = 100

BEST = dict(  # fold0 best_params (trial 35, inner_val logit_diff_auc=228)
    e_lr=0.2515987096353337,
    e_lambda_sparse_init=1.278134027531871,
    e_lambda_complete_init=1.4760090100955148,
    e_max_times_lambda_sparse=50.836074265520075,
    e_min_times_lambda_sparse=0.006622761219828983,
    gs_temp_edge=1.928463503612884,
    e_n_epoch_warmup=23,
    e_n_epoch_cooldown=29,
)
DEFAULT = dict(  # DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO registration
    e_lr=0.1,
    e_lambda_sparse_init=1.0,
    e_lambda_complete_init=1.0,
    e_max_times_lambda_sparse=100.0,
    e_min_times_lambda_sparse=0.01,
    gs_temp_edge=1.0,
    e_n_epoch_warmup=20,
    e_n_epoch_cooldown=20,
)
FIXED = dict(
    use_weight_masks=False,
    use_edge_masks=True,
    hard=True,
    loss_mode="answer_margin_ce",
    logits_e_init=1.0,
    show_train_graph=False,
)


def curve(model, ps, loader):
    counts = [c for c in DEFAULT_EVAL_EDGE_COUNTS if 0 < c <= model.n_edges]
    outs = run_circuits(model=model, dataloader=loader, test_edge_counts=[0] + counts,
                        prune_scores=ps, patch_type=PatchType.TREE_PATCH,
                        ablation_type=AblationType.ZERO)
    meas = LOGIT_DIFF_PERCENT_METRIC.metric_func(model, loader, outs)
    return sorted(meas)


def mask_stats(ps):
    # edges with positive score (kept by argmax of hard gumbel ≈ logit>0)
    tot = sum(v.numel() for v in ps.values())
    pos = sum((v > 0).sum().item() for v in ps.values())
    return pos, tot


def main():
    t.manual_seed(SEED)
    TASK.init_task()
    model = TASK.model
    true_edges = TASK.true_edges

    outer_folds = kfold_prompt_loaders(TASK.train_loader, k=5, seed=SEED, batch_size=None)
    outer_train, outer_test = outer_folds[FOLD]
    register_fold_kv_caches(model, *[l for p in outer_folds for l in p])
    print(f"outer_train batches={len(outer_train)} bs={outer_train.batch_size} | "
          f"outer_test batches={len(outer_test)} bs={outer_test.batch_size}")

    def run(tag, hp, seed):
        t.manual_seed(seed)
        ps = disco_gp_prune_scores(model, outer_train, true_edges,
                                   e_train_epochs=EPOCHS, **hp, **FIXED)
        pos, tot = mask_stats(ps)
        auc = logit_diff_auc(model, ps, outer_test, DEFAULT_EVAL_EDGE_COUNTS, AblationType.ZERO)
        c = curve(model, ps, outer_test)
        peak = max(y for _, y in c)
        print(f"\n[{tag} seed={seed}] kept_edges={pos}/{tot}  "
              f"outer_test logit_diff_auc={auc:.3f}  peak%={peak:.1f}")
        print("   curve:", [(x, round(y, 1)) for x, y in c])

    # Reproduce finalize.py refit EXACTLY: only best_params + e_train_epochs,
    # NO structural flags -> use_weight_masks defaults to True (joint pruning).
    def run_finalize_exact(seed):
        t.manual_seed(seed)
        ps = disco_gp_prune_scores(model, outer_train, true_edges,
                                   e_train_epochs=EPOCHS, **BEST)  # no FIXED!
        pos, tot = mask_stats(ps)
        auc = logit_diff_auc(model, ps, outer_test, DEFAULT_EVAL_EDGE_COUNTS, AblationType.ZERO)
        c = curve(model, ps, outer_test)
        peak = max(y for _, y in c)
        print(f"\n[FINALIZE_EXACT (use_weight_masks defaults True) seed={seed}] "
              f"kept_edges={pos}/{tot}  outer_test logit_diff_auc={auc:.3f}  peak%={peak:.1f}")
        print("   curve:", [(x, round(y, 1)) for x, y in c])

    run_finalize_exact(42)          # what finalize.py actually ran
    run("BEST_edge_only", BEST, 42)  # with use_weight_masks=False (correct)
    run("DEFAULT", DEFAULT, 42)


if __name__ == "__main__":
    main()
