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
from dataclasses import dataclass
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
    apply_holm_to_reports,
    pairwise_matrix,
    summarise_endpoint,
    summarise_similarity_endpoint,
)
from experiments.port_comparison.frozen_ioi import fixed_split_loader  # noqa: E402
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

# Edge Pruning margins: the Pillar-2 within-implementation SD for
# EP x IOI x resample/resample x faithfulness at k=2000. The rule is identical to
# DiscoGP's; only the cell it is read from differs, because resample is EP's native
# ablation where DiscoGP's is zero.
#
# The margins have been recomputed twice, and both columns are kept because the
# pre-registration binds the *rule*, not the numbers: changing what EP does obliges
# re-measuring its own spread before any verdict is read off.
#
#                        pre-fix     post-fix    post-sampling
#                        (v1)        (v2)        (v3, current)
#   Clean KL Div         0.00901  ->  0.02595  ->  0.03259
#   Logit Diff Percent    3.9621  ->   5.3445  ->  16.3605
#   Answer Pair KL Div   0.01413  ->  0.01314  ->  0.05142
#   jaccard             0.008928  -> 0.008246  ->  0.00540
#
# v1 -> v2 is the 2026-08-11 `data.py` fix. v2 -> v3 is this study's own change:
# `shared_mask_sample=True` became the default on 2026-08-12, so all 45 stability
# cells were re-run (`experiments/ep_rerun_cells.py`) and re-analysed as
# `results/stability/stability_v3_2026-08-13`.
#
# v3 was PROVISIONAL and is superseded. Its re-run constrained on GPU *memory* and
# scattered the 54 cells over four architectures (a180/ampere, a6000/ampere,
# l40/lovelace, titanrtx/turing) where the rest of Pillar 2 pins `gpu_model:a180`;
# since delta *is* a standard deviation, node variance entered the measured quantity.
# v4 re-runs the same cells pinned to a180, under the 2026-08-13 `no_padding` gate
# (`2a9c0e0`), with every manifest verified to carry that commit.
#
#                        v1 pre-fix  v2 post-fix  v3 provisional  v4 (current)
#   Clean KL Div         0.00901  ->  0.02595  ->  0.03259     ->  0.02808
#   Logit Diff Percent    3.9621  ->   5.3445  ->  16.3605     ->  10.8835
#   Answer Pair KL Div   0.01413  ->  0.01314  ->  0.05142     ->  0.05500
#   jaccard             0.008928  -> 0.008246  ->  0.00540     ->  0.00663
#
# What the v3 -> v4 move shows. Hardware scatter inflated the two margins that moved
# most (Clean KL and Logit Diff pull back by 14% and 33%), so v3 did overstate them.
# It did NOT manufacture the effect: mean pairwise Jaccard is 0.2409 at v3 against
# 0.2445 at v4, i.e. the fall from v2's 0.6224 survives pinning intact, and no EP cell
# moved by more than 0.028 across all nine task x ablation combinations. The drop is
# the sampling change, not the hardware and not the `no_padding` gate.
#
# Relative to v2 the three functional margins are still wider (1.1x, 2.0x, 4.2x): the
# reference's one-draw-per-step estimator is genuinely noisier across seeds. The
# jaccard margin is narrower, and that remains a floor effect rather than improved
# consistency, since overlap at 0.24 has less room to vary than at 0.62.
EP_DELTAS = {
    "Clean KL Div": 0.02808,
    "Logit Diff Percent": 10.8835,
    "Answer Pair KL Div": 0.05500,
    "jaccard": 0.00663,
    "spearman": 0.10,
    "density": 0.02,
}


@dataclass
class Study:
    """Everything that differs between the two port-fidelity studies.

    One code path serves both, so a change to the statistics or the reporting
    cannot apply to one study and silently not the other.
    """

    name: str
    deltas: Dict[str, float]
    controls: tuple
    comparisons: tuple
    labels: Dict[tuple, str]
    eval_ablation: AblationType
    batch_size: int
    batch_count: tuple
    density_rule: str  # "threshold" (score > 0.5) or "expected_l0"


STUDIES = {
    "discogp": Study(
        name="DiscoGP",
        deltas=DELTAS,
        controls=("port-neginv", "orig-asshipped"),
        comparisons=(
            ("port", "orig"),
            ("port-split", "orig"),
            ("port-bOzero", "orig-bOzero"),
            ("port-nobos", "port"),
            ("port-neginv", "port"),
            ("orig-asshipped", "orig"),
        ),
        labels={
            ("port", "orig"): "Primary: implementation, both as shipped",
            ("port-split", "orig"): "Matched bias convention, on unmodified GPT-2",
            ("port-bOzero", "orig-bOzero"):
                "Mechanism: implementation with `b_O` removed from both",
            ("port-nobos", "port"): "Factor: the port's BOS prepend / left pad",
            ("port-neginv", "port"): "Control: port with the mask inversion removed",
            ("orig-asshipped", "orig"):
                "Control: original with unspaced answer tokens",
        },
        eval_ablation=AblationType.ZERO,
        batch_size=64,
        batch_count=(11, 4),
        density_rule="threshold",
    ),
    "ep": Study(
        name="Edge Pruning",
        deltas=EP_DELTAS,
        controls=("port-neginv", "port-negkl", "port-nolambda"),
        comparisons=(
            ("port", "orig"),
            # POST-HOC, added after seeing the primary result and labelled as such.
            # The pre-registered verdict above is not restated from it: this pair only
            # asks whether the `port-sharedz` mechanism arm lands on the reference.
            ("port-sharedz", "orig"),
            ("orig-withembed", "orig"),
            ("port-sharedz", "port"),
            ("port-nobos", "port"),
            ("port-neginv", "port"),
            ("port-negkl", "port"),
            ("port-nolambda", "port"),
        ),
        labels={
            ("port", "orig"): "Primary: implementation, both as shipped",
            ("port-sharedz", "orig"):
                "POST-HOC mechanism: does shared-sample masking land on the reference?",
            ("orig-withembed", "orig"):
                "Mechanism: the paper's `--with_embedding_nodes` flag",
            ("port-sharedz", "port"):
                "Mechanism: one mask sample per step, as the reference draws it",
            ("port-nobos", "port"):
                "Factor: the port's input convention (BOS prepend + left pad)",
            ("port-neginv", "port"):
                "Control: port with the score-sign convention removed",
            ("port-negkl", "port"): "Control: KL direction reversed",
            ("port-nolambda", "port"):
                "Control: Lagrangian frozen, no sparsity pressure",
        },
        # EP's native protocol is resample on both axes, not DiscoGP's zero.
        eval_ablation=AblationType.RESAMPLE,
        batch_size=32,
        batch_count=(22, 8),
        density_rule="expected_l0",
    ),
}


def edge_density(scores: dict, rule: str) -> float:
    """Fraction of edges the implementation's own discretisation keeps.

    ``threshold``: DiscoGP thresholds at ``logit > 0``, i.e. ``sigmoid(logit) > 0.5``.

    ``expected_l0``: Edge Pruning does *not* use a fixed threshold. It zeroes, per
    mask tensor, as many entries as the stretched-concrete distribution expects to
    be zero (``l0.py:23-41``), then binary-searches the cut so the discrete sparsity
    matches the continuous one. Applying ``deterministic_z`` per tensor is that rule.
    Both arms export ``sigmoid(log_alpha / T * F)``, so the log-alphas are recovered
    by inverting that transform.
    """
    if rule == "threshold":
        total = sum(v.numel() for v in scores.values())
        return float(sum((v > 0.5).sum() for v in scores.values()) / total)

    from auto_circuit.utils.l0_utils import FACTOR, TEMPERATURE, deterministic_z

    kept = 0
    total = 0
    for tensor in scores.values():
        probs = tensor.detach().float().clamp(1e-6, 1 - 1e-6)
        log_alpha = t.log(probs / (1 - probs)) * TEMPERATURE / FACTOR
        kept += int((deterministic_z(log_alpha) > 0).sum())
        total += tensor.numel()
    return kept / total


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


def build_task(seed: int = 0, batch_size: int = BATCH_SIZE,
               batch_count: tuple = (11, 4)) -> Task:
    tmpl = IOI_COMPONENT_CIRCUIT_TASK
    task = Task(
        key=tmpl.key, name=tmpl.name, batch_size=batch_size, batch_count=batch_count,
        token_circuit=tmpl.token_circuit, _dataset_name=FROZEN_DATASET,
        factorized=tmpl.factorized, separate_qkv=tmpl.separate_qkv,
        _true_edge_func=tmpl._true_edge_func, slice_output=tmpl.slice_output,
        dtype=tmpl.dtype, seed=seed, _model_def=tmpl._model_def,
    )
    with fixed_split_loader():
        task.init_task()
    return task


def functional_scores(task: Task, scores: dict,
                      ablation_type: AblationType = AblationType.ZERO
                      ) -> Dict[str, float]:
    """Score one circuit at the primary budget, in this repo's harness only."""
    # Caches are loaded onto CPU so the analysis can run without a GPU; run_circuits
    # needs them beside the model.
    device = next(task.model.parameters()).device
    scores = {k: v.to(device) for k, v in scores.items()}
    circ_outs = run_circuits(
        model=task.model, dataloader=task.test_loader,
        test_edge_counts=[PRIMARY_K], prune_scores=scores,
        patch_type=PatchType.TREE_PATCH, ablation_type=ablation_type,
    )
    out = {}
    for metric in METRICS:
        measurements = metric.metric_func(task.model, task.test_loader, circ_outs)
        # metric_func returns [(edge_count, value), ...]; the sweep has one point.
        out[metric.key] = float(measurements[0][1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="discogp", choices=sorted(STUDIES))
    ap.add_argument(
        "--cache-dir", default=".prune_scores/final/port_fidelity/discogp"
    )
    ap.add_argument("--out", default="results/port_fidelity/tier_c")
    ap.add_argument("--skip-functional", action="store_true",
                    help="structural / score-level / density only (no GPU needed)")
    args = ap.parse_args()

    cfg = STUDIES[args.study]
    deltas = cfg.deltas
    cache_dir = repo_path_to_abs_path(args.cache_dir)
    cells = load_cells(cache_dir)
    if not cells:
        raise SystemExit(f"no cells found in {cache_dir}")
    print("loaded arms: " + ", ".join(f"{a}(n={len(s)})" for a, s in sorted(cells.items())))

    task = build_task(batch_size=cfg.batch_size, batch_count=cfg.batch_count)
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
            density[arm].append(edge_density(scores, cfg.density_rule))
            if not args.skip_functional:
                for key, value in functional_scores(
                    task, scores, cfg.eval_ablation
                ).items():
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
    comparisons = [(a, b) for a, b in cfg.comparisons if a in cells and b in cells]

    for arm_a, arm_b in comparisons:
        compare("density", arm_a, arm_b, density, deltas["density"])
        if not args.skip_functional:
            for metric in METRICS:
                compare(metric.key, arm_a, arm_b,
                        {a: functional[a][metric.key] for a in functional},
                        deltas[metric.key])

    # --- structural: within vs between, permuting seed labels -----------------
    structural: Dict[str, dict] = {}
    for arm_a, arm_b in comparisons:
        pooled = vectors[arm_a] + vectors[arm_b]
        masks = [budget_mask(v, PRIMARY_K) for v in pooled]
        sim = pairwise_matrix(masks, lambda x, y: float((x & y).sum() / max((x | y).sum(), 1)))
        n_a = len(vectors[arm_a])
        structural[f"jaccard::{arm_a}_vs_{arm_b}"] = {
            "endpoint": "jaccard", "arm_a": arm_a, "arm_b": arm_b,
            **summarise_similarity_endpoint(sim, n_a, deltas["jaccard"]),
        }

        from scipy.stats import spearmanr
        rho = pairwise_matrix(pooled, lambda x, y: float(spearmanr(x, y).statistic))
        structural[f"spearman::{arm_a}_vs_{arm_b}"] = {
            "endpoint": "spearman", "arm_a": arm_a, "arm_b": arm_b,
            **summarise_similarity_endpoint(rho, n_a, deltas["spearman"]),
        }

    # --- multiplicity: Holm within the primary comparison ---------------------
    primary_keys = [k for k in {**reports, **structural}
                    if k.endswith(f"::{PRIMARY_A}_vs_{PRIMARY_B}")]
    merged = {**reports, **structural}
    corrected = apply_holm_to_reports({k: merged[k] for k in primary_keys})
    merged.update(corrected)

    # --- negative-control sensitivity ----------------------------------------
    sensitivity = []
    for control in cfg.controls:
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

    labels = cfg.labels
    lines = [f"# {cfg.name} port fidelity, Tier C", "",
             f"All endpoints at k={PRIMARY_K}, scored in one harness.", ""]
    for arm_a, arm_b in comparisons:
        lines += [f"## `{arm_a}` vs `{arm_b}`", "",
                  labels.get((arm_a, arm_b), ""), "",
                  "| endpoint | arm summaries | difference or gap | 90% CI | delta | equivalent | p(diff) | p(TOST) | reading |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, row in merged.items():
            if (row["arm_a"], row["arm_b"]) != (arm_a, arm_b):
                continue
            p_diff = row.get("p_difference_holm", row["p_permutation"])
            p_tost = row.get("p_tost_holm", row["p_tost"])
            if "diff" in row:
                lines.append(
                    f"| {row['endpoint']} | {row['mean_a']:.4g} / {row['mean_b']:.4g} | "
                    f"{row['diff']:.4g} | [{row['ci90_low']:.3g}, {row['ci90_high']:.3g}] | "
                    f"{row['delta']:.4g} | {row['equivalent']} | "
                    f"{p_diff:.4g} | {p_tost:.4g} | {row['reading']} |"
                )
            else:
                lines.append(
                    f"| {row['endpoint']} | within {row['within_a']:.4g} / "
                    f"{row['within_b']:.4g}; between {row['between']:.4g} | "
                    f"{row['gap_within_minus_between']:.4g} | "
                    f"[{row['ci90_low']:.3g}, {row['ci90_high']:.3g}] | "
                    f"{row['delta']:.4g} | {row['equivalent']} | "
                    f"{p_diff:.4g} | {p_tost:.4g} | {row['reading']} |"
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
