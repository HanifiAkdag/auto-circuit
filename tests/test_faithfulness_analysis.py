#%%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from faithfulness_cells import DISCOVERY_ABLATIONS  # noqa: E402
from faithfulness_comparison import _group_cells_by_discovery_ablation  # noqa: E402
from run_cell import _ALGO_MAP, _TASK_TEMPLATES  # noqa: E402


def _cell(task, algo, abl, scores, seed=42, null=False):
    """A fake loaded cell: (manifest, task_prune_scores). The inner algo key is the
    discovery-time key (a variant like 'Edge Pruning [zero]' for swept algos); the
    grouping must re-key it to the BASE algo key regardless."""
    manifest = {
        "cell": {"task": task, "algo": algo, "ablation": abl, "seed": seed, "null_test": null},
        "saved_at": "2026-06-08T00:00:00",
        "cache_file": f"cell_{task}_{algo}_{abl}.pkl",
    }
    task_key = _TASK_TEMPLATES[task].key
    return (manifest, {task_key: {f"{algo} [discovery-variant-key]": scores}})


def test_grouping_swept_rekeys_and_anchors_in_every_group():
    ep_zero, ep_res, eap_res, gt = object(), object(), object(), object()
    loaded = [
        _cell("ioi", "ep", "zero", ep_zero),
        _cell("ioi", "ep", "resample", ep_res),
        _cell("ioi", "eap", "resample", eap_res),
        _cell("ioi", "gt", "resample", gt),
        _cell("ioi", "ep", "zero", object(), seed=99),    # wrong seed -> excluded
        _cell("ioi", "ep", "mean", object(), null=True),  # null-test -> excluded
    ]
    groups, consumed = _group_cells_by_discovery_ablation(loaded, seed=42)

    ioi = _TASK_TEMPLATES["ioi"].key
    ep_key = _ALGO_MAP["ep"].key      # base key, e.g. "Edge Pruning"
    eap_key = _ALGO_MAP["eap"].key
    gt_key = _ALGO_MAP["gt"].key

    # Swept EP lands in exactly its own discovery-ablation, re-keyed to the BASE key.
    assert groups["zero"][ioi][ep_key] is ep_zero
    assert groups["resample"][ioi][ep_key] is ep_res
    assert ep_key not in groups["mean"][ioi]

    # EAP is swept too: its resample cell lands only in the resample group.
    assert groups["resample"][ioi][eap_key] is eap_res
    assert eap_key not in groups["zero"][ioi]
    assert eap_key not in groups["mean"][ioi]

    # Anchor GT is reused as a reference row in EVERY discovery-ablation group.
    for d in DISCOVERY_ABLATIONS:
        assert groups[d][ioi][gt_key] is gt

    # seed/null filtering: only the 4 valid cells are consumed.
    assert len(consumed) == 4
