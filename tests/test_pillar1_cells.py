#%%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from pillar1_cells import (  # noqa: E402
    ANCHOR_ABLATION,
    ANCHOR_ALGOS,
    CELLS,
    DISCOVERY_ABLATIONS,
    SWEPT_ALGOS,
    TASKS,
)


def test_cell_count_and_composition():
    # 3 tasks x (3 swept x 3 ablations + 3 anchors) = 36
    assert len(CELLS) == len(TASKS) * (len(SWEPT_ALGOS) * len(DISCOVERY_ABLATIONS) + len(ANCHOR_ALGOS))
    assert len(CELLS) == 36

    swept = [c for c in CELLS if c["algo"] in SWEPT_ALGOS]
    anchors = [c for c in CELLS if c["algo"] in ANCHOR_ALGOS]
    assert len(swept) == 27
    assert len(anchors) == 9

    # Swept algos cover every discovery ablation on every task.
    for task in TASKS:
        for algo in SWEPT_ALGOS:
            abls = {c["ablation"] for c in CELLS if c["task"] == task and c["algo"] == algo}
            assert abls == set(DISCOVERY_ABLATIONS)
    # Anchors run once per task at the anchor ablation.
    for c in anchors:
        assert c["ablation"] == ANCHOR_ABLATION


def test_no_duplicate_cells():
    keys = [(c["task"], c["algo"], c["ablation"]) for c in CELLS]
    assert len(keys) == len(set(keys))
