import json

import pytest

from experiments.thesis.run_cell import (  # noqa: E402
    _EP_TASK_HPARAMS,
    _bind_task_hyperparams,
    _can_reuse_cache,
    _cache_stem,
    _data_budget,
    _serialize,
    _TASK_TEMPLATES,
)
from auto_circuit.prune_algos.prune_algos import EDGE_PRUNING_PRUNE_ALGO  # noqa: E402
from experiments.thesis.experiment_utils import algo_hyperparams


def test_joint_stem_matches_the_legacy_name():
    """The 195 existing cells must keep resolving, so a joint cell carries no init tag."""
    assert _cache_stem("ioi", "ep", "resample", 42, None) == "cell_ioi_ep_resample_seed42"
    assert _cache_stem("ioi", "ep", "resample", 42, 42) == "cell_ioi_ep_resample_seed42"


def test_decoupled_init_gets_a_tag():
    assert _cache_stem("ioi", "ep", "resample", 43, 42) == "cell_ioi_ep_resample_seed43_init42"
    assert _cache_stem("ioi", "ep", "resample", 42, 43) == "cell_ioi_ep_resample_seed42_init43"


def test_stem_slugs_algo_and_keeps_other_tags():
    assert _cache_stem("gt", "discogp-ep", "zero", 42, 43, null_test=True, partial_tag="a") == (
        "cell_gt_discogp_ep_zero_seed42_init43_null__a"
    )


def test_joint_glob_does_not_match_a_decoupled_cell():
    """--skip-if-exists globs '<stem>-*.pkl'; the separator after the seed keeps the two
    apart, which is what stops a decomposition cell being mistaken for a joint one."""
    joint = _cache_stem("ioi", "ep", "resample", 43, 43)
    decoupled = _cache_stem("ioi", "ep", "resample", 43, 42)
    assert not f"{decoupled}-2026.pkl".startswith(f"{joint}-")


def test_edge_pruning_uses_released_task_specific_targets():
    assert _EP_TASK_HPARAMS["ioi"]["target_node_sparsity"] == 0.72
    assert _EP_TASK_HPARAMS["gt"]["target_node_sparsity"] == 0.68
    assert _EP_TASK_HPARAMS["docstring"]["target_node_sparsity"] == 0.72

    gt = _bind_task_hyperparams(EDGE_PRUNING_PRUNE_ALGO, "ep", "gt")
    assert gt.func.keywords["target_edge_sparsity"] == 1.1
    assert gt.func.keywords["target_node_sparsity"] == 0.68
    assert gt.key == EDGE_PRUNING_PRUNE_ALGO.key


def test_gt_edge_pruning_records_all_reference_candidate_tokens():
    algo = _bind_task_hyperparams(EDGE_PRUNING_PRUNE_ALGO, "ep", "gt")
    params = json.loads(json.dumps(algo_hyperparams(algo), default=_serialize))
    assert params["kl_vocab_tokens"] == [f"{i:02d}" for i in range(100)]


@pytest.mark.parametrize("task", ["ioi", "docstring", "new_user_task"])
def test_other_tasks_default_to_full_vocabulary(task):
    algo = _bind_task_hyperparams(EDGE_PRUNING_PRUNE_ALGO, "ep", task)
    assert algo_hyperparams(algo)["kl_vocab_tokens"] is None
    assert algo.key == EDGE_PRUNING_PRUNE_ALGO.key


def test_gt_restricted_loss_cannot_collide_with_historical_cache():
    stem = _cache_stem("gt", "ep", "resample", 42, None)
    assert stem != "cell_gt_ep_resample_seed42"
    assert "_klvocab" in stem


def test_gt_cache_requires_matching_training_vocabulary():
    manifest = {
        "cell": {"task": "gt", "algo": "ep", "split_seed": None},
        "data_budget": _data_budget(_TASK_TEMPLATES["gt"]),
        "algo_hyperparams": {},
    }
    assert not _can_reuse_cache(manifest, "gt", None)
    manifest["algo_hyperparams"]["kl_vocab_tokens"] = [f"{i:02d}" for i in range(100)]
    assert _can_reuse_cache(manifest, "gt", None)
    manifest["algo_hyperparams"]["kl_vocab_tokens"].pop(0)
    assert not _can_reuse_cache(manifest, "gt", None)
