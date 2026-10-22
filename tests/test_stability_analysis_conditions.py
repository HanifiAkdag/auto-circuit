import torch as t

import json  # noqa: E402

from experiments.run_cell import _cache_stem  # noqa: E402
from experiments.stability_analysis import (  # noqa: E402
    _cell_specs,
    _effective_condition,
    _is_constant,
    _load_cell,
)


def test_constant_combinations_are_excluded():
    """ACDC and EAP read no torch RNG, so their init condition has nothing to vary;
    Random ignores the data, so its data condition does. Neither is a cell."""
    assert _is_constant("acdc", "init") and _is_constant("eap", "init")
    assert _is_constant("random", "data")
    assert not _is_constant("ep", "init") and not _is_constant("ep", "data")
    assert not _is_constant("acdc", "data") and not _is_constant("random", "init")


def test_degenerate_conditions_alias_to_the_joint_files():
    """An algorithm that reads only one seed already has that condition's cells."""
    assert _effective_condition("eap", "data") == "joint"
    assert _effective_condition("random", "init") == "joint"
    assert _effective_condition("ep", "data") == "data"
    assert _effective_condition("ep", "joint") == "joint"


def test_cell_specs_skips_constant_combinations():
    specs = _cell_specs(["ioi"], ["ep", "eap", "random"], ["resample"], ["joint", "data", "init"])
    combos = {(algo, cond) for _, algo, _, cond in specs}
    assert ("eap", "init") not in combos
    assert ("random", "data") not in combos
    assert ("ep", "init") in combos and ("eap", "data") in combos


def _write_cell(folder, task, algo, ablation, seed, init_seed, value):
    """Writes a fake cached cell + manifest, naming the file via the production
    `_cache_stem` so this test would catch a stem/loader divergence instead of just
    re-verifying a hand-rolled stem built to match it by construction."""
    stem = _cache_stem(task, algo, ablation, seed, init_seed) + "-01-01-2026_00-00-00"
    t.save({"T": {"A": {"m": t.tensor([float(value)])}}}, folder / f"{stem}.pkl")
    manifest = {
        "cell": {
            "task": task, "algo": algo, "ablation": ablation,
            "seed": seed, "init_seed": init_seed, "null_test": False,
        },
        "saved_at": "2026-01-01T00:00:00",
    }
    (folder / f"{stem}_manifest.json").write_text(json.dumps(manifest))


def test_load_cell_separates_conditions(tmp_path):
    """The varying seed indexes every condition, so all three are keyed 42..44 and a
    data-only cell never lands in the joint cell's seed list."""
    for seed, init_seed, value in [
        (42, 42, 1.0),   # shared anchor
        (43, 43, 2.0),   # joint
        (43, 42, 3.0),   # data-only
        (42, 43, 4.0),   # init-only
    ]:
        _write_cell(tmp_path, "ioi", "ep", "resample", seed, init_seed, value)

    joint = _load_cell(str(tmp_path), ("ioi", "ep", "resample", "joint"), [42, 43])
    data = _load_cell(str(tmp_path), ("ioi", "ep", "resample", "data"), [42, 43])
    init = _load_cell(str(tmp_path), ("ioi", "ep", "resample", "init"), [42, 43])

    assert joint.seeds == [42, 43]
    assert [s["m"].item() for s in joint.scores] == [1.0, 2.0]
    assert data.seeds == [42, 43]
    assert [s["m"].item() for s in data.scores] == [1.0, 3.0]
    assert init.seeds == [42, 43]
    assert [s["m"].item() for s in init.scores] == [1.0, 4.0]
