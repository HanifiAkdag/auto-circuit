"""Tier C analysis: is the port's output distribution equivalent to the original's?

Loads every cached Tier C cell, computes the four pre-registered endpoints, and
applies the pre-registered decision rules from
``docs/superpowers/specs/2026-08-08-discogp-port-fidelity.md``.

Endpoints are scored in **one** harness (this repo's) so a difference cannot be an
artifact of two different evaluation stacks. Scores are aligned through
``EdgeIndex``, which orders by edge identity rather than dict iteration order.

Run::

    srun --gres=gpu:1 --mem=64G python -m experiments.thesis.port_comparison.fidelity_analysis
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

REPO = Path(__file__).resolve().parents[3]
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

from .fidelity_stats import (  # noqa: E402
    apply_holm_to_reports,
    control_outcomes,
    pairwise_matrix,
    summarise_endpoint,
    summarise_similarity_endpoint,
)
from .frozen_ioi import fixed_split_loader  # noqa: E402
from .run_cells import BATCH_SIZE, FROZEN_DATASET  # noqa: E402

PRIMARY_K = 2000
BUDGETS = (500, 739, 1000, 2000, 5000)

# Equivalence margins are one population SD of the relevant final IOI stability
# values at k=2000. For DiscoGP these are the joint-seed, zero-ablation runs;
# the functional and Jaccard values come from the final stability analysis and
# density and Spearman use the same rule applied to the final score tensors.
# The constants below are recorded from those final artifacts so that the
# comparison script is self-contained and reproducible.
DELTAS = {
    "Clean KL Div": 0.8716,
    "Logit Diff Percent": 71.4485,
    "Answer Pair KL Div": 0.1792,
    "jaccard": 0.0116,
    "spearman": 0.0144272053,
    "density": 0.0075117999,
}
METRICS = [CLEAN_KL_DIV_METRIC, LOGIT_DIFF_PERCENT_METRIC, ANSWER_PAIR_KL_DIV_METRIC]

PRIMARY_A, PRIMARY_B = "port", "orig"
CONTROLS = ("port-neginv", "orig-asshipped")

# Edge Pruning uses the same rule, with the joint-seed, resample-ablation IOI
# stability runs as its native reference. The values below are recorded from
# the final, GPU-pinned stability artifacts.
EP_DELTAS = {
    "Clean KL Div": 0.02808,
    "Logit Diff Percent": 10.8835,
    "Answer Pair KL Div": 0.05500,
    "jaccard": 0.00663,
    "spearman": 0.0171401920,
    "density": 0.0002179795,
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
            sensitivity.append({"control": control, "endpoint": row["endpoint"],
                                "comparator": row["arm_b"],
                                **control_outcomes(row),
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
                  "Detection requires a significant difference test. Failure to establish",
                  "equivalence alone is inconclusive and does not demonstrate detection power.", "",
                  "| control | comparator | endpoint | difference detected | equivalent | inconclusive |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {s['control']} | {s['comparator']} | {s['endpoint']} | "
                  f"{s['difference_detected']} | {s['equivalence_established']} | {s['inconclusive']} |"
                  for s in sensitivity]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n[done] wrote {out_dir}/summary.{{csv,md}}")


if __name__ == "__main__":
    main()
