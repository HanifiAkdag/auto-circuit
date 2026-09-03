"""Evidence that the degenerate conditions really are constant.

ACDC and EAP consume no torch RNG, so their circuits must not move with --init-seed;
Random ignores the data, so its circuit must not move with --seed. Sweeping all 90 such
cells would burn GPU hours on results knowable in advance, so this runs one pair per
algorithm on the cheapest task and asserts the pair is tensor-identical.

The two cells of each pair are produced beforehand by decomp_verify.sh (one `run_cell.py`
call per cell, into a throwaway --output-dir); this script only compares the results.

    srun --gres=gpu:1 python experiments/verify_degenerate_conditions.py
"""
from pathlib import Path

import torch as t

# (algo, why) — the actual seed/init/ablation values that produced each pair live in
# decomp_verify.sh, which owns the run configuration; this script only compares outputs.
PAIRS = [
    ("acdc", "ACDC reads no torch RNG"),
    ("eap", "EAP reads no torch RNG"),
    ("random", "Random ignores the data draw"),
]


def _scores(folder: Path) -> dict[str, t.Tensor]:
    """Load the single cached PruneScores dict from `folder` as CPU tensors.

    Reads tensor values via `t.load`, never compares `.pkl` bytes: pickle records
    metadata (e.g. save timestamps, storage layout) that is not value-determined, so a
    byte-level diff would report false mismatches between value-identical caches.
    """
    pkls = sorted(folder.glob("cell_*.pkl"))
    assert len(pkls) == 1, f"{folder}: expected 1 pkl, found {len(pkls)}"
    tps = t.load(pkls[0], map_location="cpu", weights_only=False)
    algo = next(iter(next(iter(tps.values())).values()))
    return {k: v.detach().cpu() for k, v in algo.items()}


def main() -> None:
    """Compare each degenerate pair's cached PruneScores and report OK/FAIL per algorithm.

    Exits non-zero (via SystemExit) if any pair differs, since that would falsify the
    cell-matrix assumption (`_is_constant` / `_effective_condition`) that the pair's
    condition is constant and can be skipped in the full decomposition sweep.
    """
    base = Path("/mnt/beegfs/work/akdag/tmp/degenerate")
    failures = []
    for algo, why in PAIRS:
        a, b = _scores(base / f"{algo}_a"), _scores(base / f"{algo}_b")
        assert a.keys() == b.keys(), f"{algo}: keyset mismatch"
        bad = {k: (a[k] - b[k]).abs().max().item() for k in a if not t.equal(a[k], b[k])}
        if bad:
            failures.append(f"{algo}: {len(bad)}/{len(a)} tensors differ, "
                            f"max|delta|={max(bad.values()):.3e} ({why})")
            print(f"FAIL {algo}: {why}")
        else:
            print(f"OK   {algo}: identical across the degenerate axis ({why})")
    if failures:
        raise SystemExit("\n".join(failures))
    print("\nAll degenerate conditions are constant, as the cell matrix assumes.")


if __name__ == "__main__":
    main()
