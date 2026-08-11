"""Tier C analysis: is the port's output distribution equivalent to the original's?

Loads every cached Tier C cell, computes the four pre-registered endpoints, and
applies the pre-registered decision rules from
``docs/superpowers/specs/2026-08-08-discogp-port-fidelity.md``.

Endpoints are scored in **one** harness (this repo's) so a difference cannot be an
artifact of two different evaluation stacks. Scores are aligned through
``EdgeIndex``, which orders by edge identity rather than dict iteration order.

Run::

    srun --gres=gpu:1 --mem=64G python -m experiments.port_comparison.fidelity_analysis
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch as t

REPO = Path(__file__).resolve().parents[2]
for extra in (REPO, REPO / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from auto_circuit.metrics.analysis_utils import (  # noqa: E402
    budget_mask,
    edge_index,
    edge_score_vector,
    jaccard,
)
from auto_circuit.metrics.prune_metrics.prune_metrics import (  # noqa: E402
    ANSWER_PAIR_KL_DIV_METRIC,
    CLEAN_KL_DIV_METRIC,
    LOGIT_DIFF_PERCENT_METRIC,
)
from auto_circuit.prune import run_circuits  # noqa: E402
from auto_circuit.tasks import IOI_COMPONENT_CIRCUIT_TASK, Task  # noqa: E402
from auto_circuit.types import AblationType, PatchType  # noqa: E402
from auto_circuit.utils.misc import repo_path_to_abs_path  # noqa: E402

from experiments.port_comparison.fidelity_stats import (  # noqa: E402
    holm,
    pairwise_matrix,
    seed_label_permutation_test,
    summarise_endpoint,
    within_between_means,
)
from experiments.port_comparison.run_cells import BATCH_SIZE, FROZEN_DATASET  # noqa: E402

PRIMARY_K = 2000
BUDGETS = (500, 739, 1000, 2000, 5000)

# Pre-registered margins: the Pillar-2 within-implementation SD for
# DiscoGP x IOI x zero/zero x faithfulness at k=2000. The pre-registration fixed
# the *rule*, not the number, so the tail_divergence fix in data.py obliges
# recomputing them from the re-run Pillar 2
# (results/stability/stability_v2_2026-08-11/). Pre-fix values, from
# results/stability/stability_2026-08-05/, are kept for the record; every one of
# them was wider, so they made equivalence easier to declare.
#                        pre-fix     post-fix
#   Clean KL Div          1.0262  ->  0.8716
#   Logit Diff Percent   84.3293  -> 71.4485
#   Answer Pair KL Div    0.2404  ->  0.1792
#   jaccard               0.0337  ->  0.0116
# spearman and density are chosen bounds, not measured SDs, so they are unchanged.
DELTAS = {
    "Clean KL Div": 0.8716,
    "Logit Diff Percent": 71.4485,
    "Answer Pair KL Div": 0.1792,
    "jaccard": 0.0116,
    "spearman": 0.10,
    "density": 0.02,
}
METRICS = [CLEAN_KL_DIV_METRIC, LOGIT_DIFF_PERCENT_METRIC, ANSWER_PAIR_KL_DIV_METRIC]

PRIMARY_A, PRIMARY_B = "port", "orig"
CONTROLS = ("port-neginv", "orig-asshipped")


def load_cells(cache_dir: Path) -> Dict[str, Dict[int, dict]]:
    """Newest cache per (arm, seed). A .pkl without its manifest is an unfinished
    job and is ignored rather than silently treated as a result."""
    cells: Dict[str, Dict[int, tuple]] = defaultdict(dict)
    for manifest_path in sorted(cache_dir.glob("*_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        pkl = cache_dir / manifest["cache_file"]
        if not pkl.exists():
            continue
        arm, seed = manifest["cell"]["arm"], manifest["cell"]["seed"]
        stamp = manifest["saved_at"]
        if seed in cells[arm] and cells[arm][seed][0] >= stamp:
            continue
        cells[arm][seed] = (stamp, pkl)

    out: Dict[str, Dict[int, dict]] = {}
    for arm, by_seed in cells.items():
        out[arm] = {}
        for seed, (_, pkl) in sorted(by_seed.items()):
            payload = t.load(pkl, map_location="cpu", weights_only=False)
            (scores,) = next(iter(payload.values())).values()
            out[arm][seed] = scores
    return out


def build_task(seed: int = 0) -> Task:
    tmpl = IOI_COMPONENT_CIRCUIT_TASK
    task = Task(
        key=tmpl.key, name=tmpl.name, batch_size=BATCH_SIZE, batch_count=(11, 4),
        token_circuit=tmpl.token_circuit, _dataset_name=FROZEN_DATASET,
        factorized=tmpl.factorized, separate_qkv=tmpl.separate_qkv,
        _true_edge_func=tmpl._true_edge_func, slice_output=tmpl.slice_output,
        dtype=tmpl.dtype, seed=seed, _model_def=tmpl._model_def,
    )
    task.init_task()
    return task


def functional_scores(task: Task, scores: dict) -> Dict[str, float]:
    """Score one circuit at the primary budget, in this repo's harness only."""
    # Caches are loaded onto CPU so the analysis can run without a GPU; run_circuits
    # needs them beside the model.
    device = next(task.model.parameters()).device
    scores = {k: v.to(device) for k, v in scores.items()}
    circ_outs = run_circuits(
        model=task.model, dataloader=task.test_loader,
        test_edge_counts=[PRIMARY_K], prune_scores=scores,
        patch_type=PatchType.TREE_PATCH, ablation_type=AblationType.ZERO,
    )
    out = {}
    for metric in METRICS:
        measurements = metric.metric_func(task.model, task.test_loader, circ_outs)
        # metric_func returns [(edge_count, value), ...]; the sweep has one point.
        out[metric.key] = float(measurements[0][1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=".port_fidelity_cache")
    ap.add_argument("--out", default="results/port_fidelity/tier_c")
    ap.add_argument("--skip-functional", action="store_true",
                    help="structural / score-level / density only (no GPU needed)")
    args = ap.parse_args()

    cache_dir = repo_path_to_abs_path(args.cache_dir)
    cells = load_cells(cache_dir)
    if not cells:
        raise SystemExit(f"no cells found in {cache_dir}")
    print("loaded arms: " + ", ".join(f"{a}(n={len(s)})" for a, s in sorted(cells.items())))

    task = build_task()
    index = edge_index(task.model.edges, next(iter(cells[PRIMARY_A].values())))

    # --- per-cell quantities --------------------------------------------------
    vectors: Dict[str, List[np.ndarray]] = {}
    density: Dict[str, List[float]] = {}
    functional: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for arm, by_seed in sorted(cells.items()):
        vectors[arm], density[arm] = [], []
        for seed, scores in sorted(by_seed.items()):
            vec = edge_score_vector(index, scores)
            vectors[arm].append(vec)
            # The original thresholds at logit > 0, i.e. sigmoid(logit) > 0.5.
            density[arm].append(float((vec > 0.5).mean()))
            if not args.skip_functional:
                for key, value in functional_scores(task, scores).items():
                    functional[arm][key].append(value)
        print(f"  {arm}: n={len(vectors[arm])}, mean density "
              f"{np.mean(density[arm]):.4%}")

    # --- endpoints ------------------------------------------------------------
    reports: Dict[str, dict] = {}

    def compare(name: str, arm_a: str, arm_b: str, values: Dict[str, List[float]],
                delta: float) -> None:
        if arm_a not in values or arm_b not in values:
            return
        if not values[arm_a] or not values[arm_b]:
            return
        reports[f"{name}::{arm_a}_vs_{arm_b}"] = {
            "endpoint": name, "arm_a": arm_a, "arm_b": arm_b,
            **summarise_endpoint(values[arm_a], values[arm_b], delta),
        }

    # Each comparison must isolate exactly one factor. Pairing a b_O-zeroed arm
    # against a bias-intact one would confound the mechanism being tested with the
    # implementation, and a control is only a control against the arm it perturbs.
    candidate_comparisons = [
        (PRIMARY_A, PRIMARY_B),                # primary: implementation
        ("port-split", "orig"),                # matched bias convention, real GPT-2
        ("port-bOzero", "orig-bOzero"),        # mechanism: implementation, b_O removed
        ("port-nobos", "port"),                # factor: BOS prepend / left pad
        ("port-neginv", "port"),               # control: mask inversion removed
        ("orig-asshipped", "orig"),            # control: unspaced answer tokens
    ]
    comparisons = [(a, b) for a, b in candidate_comparisons if a in cells and b in cells]

    for arm_a, arm_b in comparisons:
        compare("density", arm_a, arm_b, density, DELTAS["density"])
        if not args.skip_functional:
            for metric in METRICS:
                compare(metric.key, arm_a, arm_b,
                        {a: functional[a][metric.key] for a in functional},
                        DELTAS[metric.key])

    # --- structural: within vs between, permuting seed labels -----------------
    structural: Dict[str, dict] = {}
    for arm_a, arm_b in comparisons:
        pooled = vectors[arm_a] + vectors[arm_b]
        masks = [budget_mask(v, PRIMARY_K) for v in pooled]
        sim = pairwise_matrix(masks, lambda x, y: float((x & y).sum() / max((x | y).sum(), 1)))
        n_a = len(vectors[arm_a])
        test = seed_label_permutation_test(sim, n_a=n_a)
        means = within_between_means(sim, n_a=n_a)
        structural[f"jaccard::{arm_a}_vs_{arm_b}"] = {
            "endpoint": "jaccard", "arm_a": arm_a, "arm_b": arm_b,
            **means, "gap_within_minus_between": test["observed"],
            "p_permutation": test["p_value"], "exact": bool(test["exact"]),
            "delta": DELTAS["jaccard"],
            "equivalent": abs(test["observed"]) < DELTAS["jaccard"],
        }

        from scipy.stats import spearmanr
        rho = pairwise_matrix(pooled, lambda x, y: float(spearmanr(x, y).statistic))
        rho_test = seed_label_permutation_test(rho, n_a=n_a)
        structural[f"spearman::{arm_a}_vs_{arm_b}"] = {
            "endpoint": "spearman", "arm_a": arm_a, "arm_b": arm_b,
            **within_between_means(rho, n_a=n_a),
            "gap_within_minus_between": rho_test["observed"],
            "p_permutation": rho_test["p_value"], "exact": bool(rho_test["exact"]),
            "delta": DELTAS["spearman"],
            "equivalent": abs(rho_test["observed"]) < DELTAS["spearman"],
        }

    # --- multiplicity: Holm within the primary comparison ---------------------
    primary_keys = [k for k in {**reports, **structural}
                    if k.endswith(f"::{PRIMARY_A}_vs_{PRIMARY_B}")]
    merged = {**reports, **structural}
    corrected = holm({k: merged[k]["p_permutation"] for k in primary_keys})

    # --- negative-control sensitivity ----------------------------------------
    sensitivity = []
    for control in CONTROLS:
        if control not in cells:
            continue
        for key, row in merged.items():
            if row["arm_a"] != control:
                continue
            flagged = row["p_permutation"] < 0.05 or not row["equivalent"]
            sensitivity.append({"control": control, "endpoint": row["endpoint"],
                                "detected": flagged,
                                "p_permutation": row["p_permutation"],
                                "equivalent": row["equivalent"]})

    # --- write ---------------------------------------------------------------
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, row in merged.items():
        row = dict(row)
        row["comparison"] = key
        if key in corrected:
            row.update(p_holm=corrected[key]["p_holm"])
        rows.append(row)
    fieldnames = sorted({k for r in rows for k in r})
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if sensitivity:
        with open(out_dir / "negative_controls.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sensitivity[0]))
            writer.writeheader()
            writer.writerows(sensitivity)

    labels = {
        (PRIMARY_A, PRIMARY_B): "Primary: implementation, both as shipped",
        ("port-split", "orig"): "Matched bias convention, on unmodified GPT-2",
        ("port-bOzero", "orig-bOzero"): "Mechanism: implementation with `b_O` removed from both",
        ("port-nobos", "port"): "Factor: the port's BOS prepend / left pad",
        ("port-neginv", "port"): "Control: port with the mask inversion removed",
        ("orig-asshipped", "orig"): "Control: original with unspaced answer tokens",
    }
    lines = ["# DiscoGP port fidelity, Tier C", "",
             f"All endpoints at k={PRIMARY_K}, scored in one harness.", ""]
    for arm_a, arm_b in comparisons:
        lines += [f"## `{arm_a}` vs `{arm_b}`", "",
                  labels.get((arm_a, arm_b), ""), "",
                  "| endpoint | mean A | mean B | diff | 90% CI | delta | equivalent | p | reading |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, row in merged.items():
            if (row["arm_a"], row["arm_b"]) != (arm_a, arm_b):
                continue
            if "ci90_low" in row:
                lines.append(
                    f"| {row['endpoint']} | {row['mean_a']:.4g} | {row['mean_b']:.4g} | "
                    f"{row['diff']:.4g} | [{row['ci90_low']:.3g}, {row['ci90_high']:.3g}] | "
                    f"{row['delta']:.4g} | {row['equivalent']} | "
                    f"{row['p_permutation']:.4g} | {row['reading']} |"
                )
            else:
                lines.append(
                    f"| {row['endpoint']} | within {row['within_a']:.4g} | "
                    f"within {row['within_b']:.4g} | between {row['between']:.4g} | "
                    f"gap {row['gap_within_minus_between']:.4g} | {row['delta']:.4g} | "
                    f"{row['equivalent']} | {row['p_permutation']:.4g} | |"
                )
        lines.append("")
    if sensitivity:
        lines += ["", "## Negative controls", "",
                  "An endpoint that detects nothing has no demonstrated power and is",
                  "reported as uninformative, not as evidence of equivalence.", "",
                  "| control | endpoint | detected |", "|---|---|---|"]
        lines += [f"| {s['control']} | {s['endpoint']} | {s['detected']} |"
                  for s in sensitivity]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n[done] wrote {out_dir}/summary.{{csv,md}}")


if __name__ == "__main__":
    main()
