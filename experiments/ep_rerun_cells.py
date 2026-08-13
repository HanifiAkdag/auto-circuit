"""EP cells invalidated by the 2026-08-12 mask-sampling change.

`edge_pruning_prune_scores` now defaults to the reference's sampling
(`shared_mask_sample=True`, one hard-concrete draw shared across the batch), so every
EP cell cached before that date was produced by a different estimator and has to be
recomputed. See `docs/port_validation/edge_pruning.md` §5.5.2.

Two pillars are affected, with different cache directories:

- Pillar 1 (faithfulness spine): 9 cells, `.prune_scores_cache/faithfulness_v2`
- Pillar 2 (stability): 45 cells, `.prune_scores_cache_v2/stability`

Pillar 3 (null test) has never been run, so there is nothing to invalidate there.

Cells are run with `--force`: the stale entries stay on disk for auditability and
`analysis_utils.latest_per_cell` picks the newer one by timestamp.
"""

from typing import List, Tuple

from experiments.faithfulness_cells import CELLS as FAITHFULNESS_CELLS
from experiments.stability_cells import CELLS as STABILITY_CELLS

FAITHFULNESS_DIR = ".prune_scores_cache/faithfulness_v2"
STABILITY_DIR = ".prune_scores_cache_v2/stability"

# (task, ablation, seed, output_dir)
CELLS: List[Tuple[str, str, int, str]] = [
    (c["task"], c["ablation"], 42, FAITHFULNESS_DIR)
    for c in FAITHFULNESS_CELLS
    if c["algo"] == "ep"
] + [
    (c["task"], c["ablation"], int(c["seed"]), STABILITY_DIR)
    for c in STABILITY_CELLS
    if c["algo"] == "ep"
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        task, ablation, seed, out = CELLS[int(sys.argv[1])]
        print(f"{task} {ablation} {seed} {out}")
    else:
        print(len(CELLS))
