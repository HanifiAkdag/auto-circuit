"""Single source of truth for the Pillar-2 stability (seed-variance) cell list (#5).

Pillar 2 asks how much of a discovered circuit is an artifact of the random draw: each
cell of the faithfulness spine is re-run under N=5 seeds and the resulting circuits are
compared. Imported by the analysis pass and printed one cell per line by the SLURM array
launcher (`stability_discovery.sh`), so the launcher and the analysis can never disagree
on what the matrix is.

Differences from the faithfulness spine (`faithfulness_cells.py`):
  - a `seed` axis (5 levels) instead of the single seed 42;
  - Ground Truth is dropped: it is deterministic, so its seed variance is zero by
    construction and five runs would re-emit the same file;
  - Random is kept as the pure-noise ceiling, which is what tells "unstable" apart from
    "no better than chance".

Seed 42 is included deliberately: those cells double as a spine re-run under the
canonical-ordering fix (2026-08-01), before which no cell was reproducible at all.

ACDC on IOI and GT is the long pole (~13 GPU-h and ~4 GPU-h per cell). Those 30 cells are
excluded from `ARRAY_CELLS` and run tau-split via `stability_acdc_partials.sh`, exactly as
the spine does; `CELLS` still lists them so the analysis sees the full matrix.

Cell count: 4 swept x 3 tasks x 3 ablations x 5 seeds + 3 tasks x 5 seeds (Random) = 195.
"""
import argparse
from typing import Dict, List

TASKS: List[str] = ["ioi", "gt", "docstring"]
SWEPT_ALGOS: List[str] = ["ep", "discogp-ep", "acdc", "eap"]  # run under each ablation
ANCHOR_ALGOS: List[str] = ["random"]  # ablation-independent noise ceiling
DISCOVERY_ABLATIONS: List[str] = ["zero", "resample", "mean"]
ANCHOR_ABLATION: str = "resample"  # arbitrary; anchors don't depend on it
SEEDS: List[int] = [42, 43, 44, 45, 46]

# (algo, task) pairs too expensive for the main array; run tau-split instead.
PARTIAL_SPLIT_CELLS: List[tuple] = [("acdc", "ioi"), ("acdc", "gt")]


def _build_cells() -> List[Dict[str, str]]:
    cells: List[Dict[str, str]] = []
    for seed in SEEDS:
        for task in TASKS:
            for algo in SWEPT_ALGOS:
                for ablation in DISCOVERY_ABLATIONS:
                    cells.append(
                        {
                            "task": task,
                            "algo": algo,
                            "ablation": ablation,
                            "seed": str(seed),
                        }
                    )
            for algo in ANCHOR_ALGOS:
                cells.append(
                    {
                        "task": task,
                        "algo": algo,
                        "ablation": ANCHOR_ABLATION,
                        "seed": str(seed),
                    }
                )
    return cells


CELLS: List[Dict[str, str]] = _build_cells()
ARRAY_CELLS: List[Dict[str, str]] = [
    c for c in CELLS if (c["algo"], c["task"]) not in PARTIAL_SPLIT_CELLS
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--array",
        action="store_true",
        help="Print only the cells the main SLURM array runs (excludes tau-split ACDC).",
    )
    parser.add_argument(
        "--count", action="store_true", help="Print cell counts and exit."
    )
    args = parser.parse_args()

    if args.count:
        print(f"CELLS       {len(CELLS)}")
        print(f"ARRAY_CELLS {len(ARRAY_CELLS)}  (array bounds: 0-{len(ARRAY_CELLS) - 1})")
        print(f"tau-split   {len(CELLS) - len(ARRAY_CELLS)}")
        return

    for c in ARRAY_CELLS if args.array else CELLS:
        print(f"{c['task']} {c['algo']} {c['ablation']} {c['seed']}")


if __name__ == "__main__":
    main()
