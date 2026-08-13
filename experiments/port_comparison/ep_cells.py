"""The Tier C cell list for the Edge Pruning port fidelity study.

Single source of truth, imported by both the launcher and the analysis pass, so an
arm cannot be run under one definition and analysed under another.

Arm sizes are fixed in
``docs/superpowers/specs/2026-08-11-edge-pruning-port-fidelity.md``: n = 10 for the
two primary arms (which keeps ``C(20,10) = 184,756`` under
``fidelity_stats.MAX_EXACT``, so every permutation test stays exact), n = 5 for the
mechanism arms and negative controls.
"""

from typing import List, Tuple

PRIMARY_N = 10
SECONDARY_N = 5

ARM_SIZES = {
    "port": PRIMARY_N,
    "orig": PRIMARY_N,
    "orig-withembed": SECONDARY_N,
    "port-sharedz": SECONDARY_N,
    "port-nobos": SECONDARY_N,
    "port-neginv": SECONDARY_N,
    "port-negkl": SECONDARY_N,
    "port-nolambda": SECONDARY_N,
}

CELLS: List[Tuple[str, int]] = [
    (arm, seed) for arm, n in ARM_SIZES.items() for seed in range(n)
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arm, seed = CELLS[int(sys.argv[1])]
        print(f"{arm} {seed}")
    else:
        print(len(CELLS))
