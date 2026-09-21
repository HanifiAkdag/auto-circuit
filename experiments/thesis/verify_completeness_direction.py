"""Assert the completeness curves are knockouts, and that faithfulness did not move.

Usage: python -m experiments.thesis.verify_completeness_direction NEW_TREE [OLD_TREE]
"""
import csv
import pickle
import sys
from pathlib import Path


def load_overlay(tree: Path) -> dict:
    with open(tree / "interactive" / "overlay.pkl", "rb") as f:
        return pickle.load(f)


def check_knockout(tree: Path) -> None:
    overlay = load_overlay(tree)
    zero_checked = end_checked = 0
    for (task, measure, metric), series in overlay.items():
        if measure != "completeness" or "KL" not in metric:
            continue
        for algo, disc, ev, pts in series:
            pts = sorted(pts)
            label = f"{task}/{algo}/disc={disc}/eval={ev}/{metric}"
            # Not every series has a 0-edge point: ACDC's edge grid comes from its tie
            # blocks and starts around 9-76 edges, and the GT anchor is a single point.
            # Those series open already saturated, so they cannot be checked for a rise.
            zeros = [v for b, v in pts if b == 0]
            if zeros:
                assert abs(zeros[0]) < 1e-4, (
                    f"{label}: knocking out 0 edges gave KL={zeros[0]:.4f}, expected 0"
                )
                if len(pts) >= 2:
                    assert pts[-1][1] > pts[0][1], (
                        f"{label}: KL fell from {pts[0][1]:.4f} to {pts[-1][1]:.4f}; "
                        "a knockout curve must rise from 0"
                    )
                zero_checked += 1
            # Knocking out every edge must destroy the model. Under the denoising
            # direction this endpoint is exactly 0, so it discriminates on its own.
            if len(pts) >= 2:
                assert pts[-1][1] > 0.1, (
                    f"{label}: KL at the full edge count is {pts[-1][1]:.4f}; "
                    "knocking out every edge cannot leave the model intact"
                )
                end_checked += 1
    assert zero_checked and end_checked, "no completeness KL series found"
    print(f"OK: {zero_checked} series start at KL=0 and rise; "
          f"{end_checked} series end above 0.1.")


def check_faithfulness_unchanged(new: Path, old: Path) -> None:
    def faith_rows(tree: Path) -> dict:
        with open(tree / "summary.csv") as f:
            return {
                (r["discovery_ablation"], r["task"], r["algo"], r["eval_ablation"],
                 r["metric"], r["budget_edges"]): r["value"]
                for r in csv.DictReader(f)
                if r["measure"] == "faithfulness"
            }

    a, b = faith_rows(new), faith_rows(old)
    assert a.keys() == b.keys(), (
        f"faithfulness row sets differ: {len(a)} new vs {len(b)} old, "
        f"symmetric difference {len(set(a) ^ set(b))}"
    )
    drifted = {k: (a[k], b[k]) for k in a if abs(float(a[k]) - float(b[k])) > 1e-6}
    assert not drifted, f"faithfulness moved on {len(drifted)} rows: {list(drifted)[:5]}"
    print(f"OK: all {len(a)} faithfulness rows identical to the previous tree.")


if __name__ == "__main__":
    new_tree = Path(sys.argv[1])
    check_knockout(new_tree)
    if len(sys.argv) > 2:
        check_faithfulness_unchanged(new_tree, Path(sys.argv[2]))
