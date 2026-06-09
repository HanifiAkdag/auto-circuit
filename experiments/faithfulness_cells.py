"""Single source of truth for the faithfulness-spine discovery cell list (sub-RQ3, #4).

The faithfulness spine is the full discovery x evaluation matrix comparing circuit
discovery methods. Imported by the analysis pass (`faithfulness_comparison.py`) and
printed one cell per line by the SLURM array launcher (`faithfulness_discovery.sh`),
so the launcher and the analysis can never disagree on what the matrix is.

Full discovery x eval matrix (single seed): EP / DiscoGP-EP / ACDC are each trained
under every discovery ablation {zero, resample, mean}. EAP (corrupt-only by
construction), Ground Truth and Random are ablation-independent — run once (at
`resample`) and reused as reference rows in every discovery-ablation panel by the
analysis pass.

Cell count: 3 tasks x (3 swept x 3 ablations + 3 anchors) = 36.
"""
from typing import Dict, List

TASKS: List[str] = ["ioi", "gt", "docstring"]
SWEPT_ALGOS: List[str] = ["ep", "discogp-ep", "acdc"]  # trained under each ablation
ANCHOR_ALGOS: List[str] = ["eap", "gt", "random"]      # ablation-independent
DISCOVERY_ABLATIONS: List[str] = ["zero", "resample", "mean"]
ANCHOR_ABLATION: str = "resample"  # arbitrary; anchors don't depend on it


def _build_cells() -> List[Dict[str, str]]:
    cells: List[Dict[str, str]] = []
    for task in TASKS:
        for algo in SWEPT_ALGOS:
            for ablation in DISCOVERY_ABLATIONS:
                cells.append({"task": task, "algo": algo, "ablation": ablation})
        for algo in ANCHOR_ALGOS:
            cells.append({"task": task, "algo": algo, "ablation": ANCHOR_ABLATION})
    return cells


CELLS: List[Dict[str, str]] = _build_cells()


if __name__ == "__main__":
    for c in CELLS:
        print(f"{c['task']} {c['algo']} {c['ablation']}")
