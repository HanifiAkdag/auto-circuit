"""Single source of truth for the faithfulness-spine discovery cell list (sub-RQ3, #4).

The faithfulness spine is the full discovery x evaluation matrix comparing circuit
discovery methods. Imported by the analysis pass (`faithfulness_comparison.py`) and
available as one cell per line when a cluster driver invokes this module, so the
discovery driver and the analysis can never disagree on what the matrix is.

Full discovery x eval matrix (single seed): EP / DiscoGP / ACDC / EAP are each
run under every discovery ablation {zero, resample, mean} (EAP via the mask-gradient
implementation, whose first-order estimate targets whichever ablation baseline is
chosen; resample reproduces Syed et al. 2023 exactly). Ground Truth and Random are
genuinely ablation-independent — run once (at `resample`) and reused as reference
rows in every discovery-ablation panel by the analysis pass.

Cell count: 3 tasks x (4 swept x 3 ablations + 2 anchors) = 42.
"""
from typing import Dict, List

TASKS: List[str] = ["ioi", "gt", "docstring"]
SWEPT_ALGOS: List[str] = ["ep", "discogp-ep", "acdc", "eap"]  # run under each ablation
ANCHOR_ALGOS: List[str] = ["gt", "random"]  # ablation-independent
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
