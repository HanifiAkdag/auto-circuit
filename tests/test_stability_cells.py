import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from stability_cells import (  # noqa: E402
    ANCHOR_SEED,
    CONDITIONS,
    DECOMP_ALGOS,
    DECOMP_CELLS,
    DISCOVERY_ABLATIONS,
    SEEDS,
    TASKS,
    condition_of,
    seeds_for,
)


def test_condition_of_classifies_the_three_conditions():
    assert condition_of(43, 43) == "joint"
    assert condition_of(43, ANCHOR_SEED) == "data"
    assert condition_of(ANCHOR_SEED, 43) == "init"
    assert condition_of(ANCHOR_SEED, ANCHOR_SEED) == "joint"


def test_condition_of_rejects_a_pair_on_no_axis():
    with pytest.raises(ValueError):
        condition_of(43, 44)


def test_seeds_for_round_trips_through_condition_of():
    for condition in CONDITIONS:
        for seed in SEEDS:
            data_seed, init_seed = seeds_for(condition, seed)
            if seed == ANCHOR_SEED:
                assert condition_of(data_seed, init_seed) == "joint"
            else:
                assert condition_of(data_seed, init_seed) == condition


def test_decomp_cell_count():
    # 2 conditions x 2 algos x 3 tasks x 3 ablations x 4 non-anchor seeds
    expected = 2 * len(DECOMP_ALGOS) * len(TASKS) * len(DISCOVERY_ABLATIONS) * (len(SEEDS) - 1)
    assert len(DECOMP_CELLS) == expected == 144


def test_decomp_cells_never_collide_with_a_joint_cell():
    """The anchor seed is excluded, so no decomposition cell has data == init and none can
    be written under a joint cell's stem."""
    for c in DECOMP_CELLS:
        assert int(c["seed"]) != int(c["init_seed"])


def test_decomp_cells_are_unique_and_optimization_only():
    keys = [(c["task"], c["algo"], c["ablation"], c["seed"], c["init_seed"]) for c in DECOMP_CELLS]
    assert len(keys) == len(set(keys))
    assert {c["algo"] for c in DECOMP_CELLS} == set(DECOMP_ALGOS)


def test_every_decomp_cell_reports_its_own_condition():
    for c in DECOMP_CELLS:
        assert condition_of(int(c["seed"]), int(c["init_seed"])) == c["condition"]
