"""Edge-probability sampling: draw alternative circuits from one trained DiscoGP mask.

Motivation (sub-RQ3 / supervisor Frank's suggestion; feeds Pillar 2, issue #5):
DiscoGP returns PruneScores that are exactly the per-edge keep-probabilities
p_e = sigmoid(logit_e) = sigma(l_e). Instead of hard-thresholding to ONE circuit, we
can draw *many* circuits from a single trained mask by sampling each edge independently,
mask_e ~ Bernoulli(p_e). This is a cheap alternative to the repulsion method of
"All Circuits Lead to Rome" (Chen et al. 2026), which instead retrains once per circuit.

Two modes:

  (default) DIAGNOSE — is sampling even viable? Reads cached p_e (no model needed) and
    reports polarization + the analytic expected Jaccard between two independent draws,
    plus a temperature sweep p_e(T) = sigma(l_e / T). E[Jaccard] ~ 1 => masks are
    polarized to 0/1 and sampling reproduces the same circuit; lower => genuine diversity.

  --evaluate  SAMPLE & SCORE — actually draw N circuits, measure each circuit's
    faithfulness (via measure_prune_metrics at its own edge count) and the pairwise
    Jaccard/IoU across the N samples. Answers: are the sampled circuits diverse AND
    faithful? Compare IoU against OASR's ~4% floor and Pillar-2 reseeding Jaccard.

Usage:
    # Diagnose all cached DiscoGP cells (CPU/GPU, just reads tensors):
    srun --gres=gpu:1 python experiments/edge_prob_diagnostic.py

    # Sample & evaluate one cell (needs the task model -> GPU). Prefix selects the cell:
    srun --gres=gpu:1 python experiments/edge_prob_diagnostic.py --evaluate \
        --prefix cell_docstring_discogp_ep_zero --n-samples 20 --temp 1.0

    # Train a fresh DiscoGP mask first instead of loading cache:
    srun --gres=gpu:1 python experiments/edge_prob_diagnostic.py --run-fresh --task ioi --ablation zero
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_circuit.metrics.analysis_utils import load_many_caches
from auto_circuit.utils.misc import repo_path_to_abs_path

FAITHFULNESS_CACHE_DIR = ".prune_scores_cache/faithfulness_v2"
RESULTS_DIR = "results/edge_prob_diagnostic"
EPS = 1e-6

# Temperatures for the softening sweep. T=1 is the trained mask; T>1 flattens p_e
# toward 0.5 (more stochastic), T<1 sharpens toward 0/1 (less stochastic).
TEMPS = [0.5, 1.0, 2.0, 4.0, 8.0]

# A "cell" is one trained mask: (task_key, algo_key, PruneScores{mod_name: tensor}).
Cell = Tuple[str, str, Dict[str, "object"]]


# ─────────────────────────────────────────────────────────────────────────────
# Probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_scores(prune_scores: Dict[str, "object"]) -> np.ndarray:
    """Concatenate all per-edge scores in a PruneScores dict into one 1-D array."""
    import torch as t
    return t.cat([v.detach().float().flatten().cpu() for v in prune_scores.values()]).numpy()


def soften(p: np.ndarray, temp: float) -> np.ndarray:
    """Recover logits l = logit(p), rescale by 1/temp, map back through sigmoid."""
    pc = np.clip(p, EPS, 1.0 - EPS)
    logit = np.log(pc / (1.0 - pc))
    return 1.0 / (1.0 + np.exp(-logit / temp))


def sampling_diagnostics(p: np.ndarray) -> dict:
    """Polarization + analytic diversity of Bernoulli(p) circuit sampling.

    For two independent draws A, B ~ Bernoulli(p) edge-wise:
      E[#edges that differ] = sum 2 p (1-p)
      E[|A & B|] = sum p^2 ,  E[|A | B|] = sum (2p - p^2)
      expected Jaccard ~= E[|A & B|] / E[|A | B|]   (ratio of expectations)
    Expected Jaccard near 1.0 => sampling reproduces the same circuit (no diversity);
    lower => sampled circuits genuinely differ.
    """
    n = p.size
    near0 = float(np.mean(p < 0.01))
    near1 = float(np.mean(p > 0.99))
    intermediate = float(np.mean((p >= 0.05) & (p <= 0.95)))
    # Per-edge binary entropy (bits): how much stochasticity each edge carries.
    pc = np.clip(p, EPS, 1.0 - EPS)
    ent = -(pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))
    exp_kept = float(p.sum())                       # expected soft circuit size
    exp_differ = float((2.0 * p * (1.0 - p)).sum())  # expected edges differing across draws
    inter = float((p ** 2).sum())
    union = float((2.0 * p - p ** 2).sum())
    exp_jaccard = inter / union if union > 0 else 1.0
    return {
        "n_edges": n,
        "polarized_frac": near0 + near1,
        "near0_frac": near0,
        "near1_frac": near1,
        "intermediate_frac": intermediate,
        "mean_entropy_bits": float(ent.mean()),
        "exp_kept_edges": exp_kept,
        "exp_differ_edges": exp_differ,
        "exp_differ_per_kept": exp_differ / exp_kept if exp_kept > 0 else 0.0,
        "exp_jaccard_two_samples": exp_jaccard,
    }


def feasibility_verdict(d: dict) -> str:
    """One-word verdict on whether per-edge sampling produces diverse circuits as-is."""
    j = d["exp_jaccard_two_samples"]
    if j > 0.9:
        return "INFEASIBLE (needs temperature)"
    if j > 0.6:
        return "MARGINAL"
    return "FEASIBLE"


def temp_sweep(p: np.ndarray) -> List[Tuple[float, float, float]]:
    """(temp, expected Jaccard, expected kept edges) for each temperature."""
    out = []
    for temp in TEMPS:
        ps = soften(p, temp)
        d = sampling_diagnostics(ps)
        out.append((temp, d["exp_jaccard_two_samples"], d["exp_kept_edges"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loading trained masks
# ─────────────────────────────────────────────────────────────────────────────

def _label(task_key: str, algo_key: str) -> str:
    return f"{task_key} | {algo_key}".replace(" ", "_")


def collect_cells(cache_dir: str, prefix: str, algo_filter: str) -> List[Cell]:
    """Load every (task, algo, PruneScores) whose algo_key matches `algo_filter`.

    `prefix` is matched as `<prefix>*.pkl` (load_many_caches appends the suffix, so it
    must not end in '*'). `algo_filter` is a case-insensitive substring on the algo key —
    defaults select DiscoGP cells, whose scores are sigma(l_e).
    """
    cells: List[Cell] = []
    flt = algo_filter.lower()
    for _manifest, tps in load_many_caches(cache_dir, prefix):
        for task_key, algo_ps in tps.items():
            for algo_key, ps in algo_ps.items():
                if flt and flt not in algo_key.lower():
                    continue
                cells.append((task_key, algo_key, ps))
    return cells


def run_fresh(task_short: str, ablation_short: str, seed: int) -> List[Cell]:
    """Train one DiscoGP edge-only mask and return it as a single cell."""
    import torch as t
    from auto_circuit.prune_algos.prune_algos import make_ablation_variant, run_prune_algos
    from run_cell import _ALGO_MAP, _ABLATION_MAP, _build_task

    t.manual_seed(seed)
    algo = make_ablation_variant(_ALGO_MAP["discogp-e"], _ABLATION_MAP[ablation_short])
    task = _build_task(task_short, seed, null_test=False)
    tps = run_prune_algos([task], [algo])
    return [
        (task_key, algo_key, ps)
        for task_key, algo_ps in tps.items()
        for algo_key, ps in algo_ps.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Sampling & evaluation (--evaluate)
# ─────────────────────────────────────────────────────────────────────────────

def sample_circuit(prune_scores: Dict[str, "object"], temp: float, device) -> Tuple[Dict[str, "object"], int]:
    """Draw a binary circuit: keep edge e with probability p_e(T) = sigma(l_e / T).

    Returns a PruneScores-shaped dict of 0/1 tensors (1 = kept) plus the kept-edge count.
    Seed torch before calling for reproducibility.
    """
    import torch as t
    sampled: Dict[str, t.Tensor] = {}
    n_kept = 0
    for mod, p in prune_scores.items():
        p = p.float().to(device)
        if temp != 1.0:
            pc = p.clamp(EPS, 1.0 - EPS)
            p = t.sigmoid(t.log(pc / (1.0 - pc)) / temp)
        mask = (t.rand_like(p) < p).float()
        sampled[mod] = mask
        n_kept += int(mask.sum().item())
    return sampled, n_kept


def evaluate_cell(
    cell: Cell, n_samples: int, temp: float, eval_ablation, metric, base_seed: int
) -> Tuple[List[dict], List[set]]:
    """Sample N circuits from a trained mask and score each for faithfulness.

    Each sampled circuit is evaluated at its OWN edge count via measure_prune_metrics
    (binary PruneScores + test_edge_counts=[n_kept], TREE_PATCH = keep circuit, ablate
    complement). Returns per-sample rows and the list of sampled edge sets (for overlap).
    """
    import re
    import torch as t
    from auto_circuit.metrics.analysis_utils import top_k_edge_set
    from auto_circuit.metrics.prune_metrics.measure_prune_metrics import measure_prune_metrics
    from auto_circuit.metrics.analysis_utils import extract_metric
    from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
    from auto_circuit.tasks import TASK_DICT
    from auto_circuit.types import PatchType

    task_key, algo_key, prune_scores = cell
    # Cached keys are ablation-tagged variants ("... [zero]") not in PRUNE_ALGO_DICT;
    # measure_prune_metrics looks the key up, so re-key to the registered base algo.
    base_algo_key = algo_key if algo_key in PRUNE_ALGO_DICT else re.sub(r"\s*\[[^\]]*\]$", "", algo_key)
    task = TASK_DICT[task_key]
    device = next(task.model.parameters()).device
    edges = task.model.edges
    ps_dev = {m: s.float().to(device) for m, s in prune_scores.items()}

    rows: List[dict] = []
    edge_sets: List[set] = []
    for i in range(n_samples):
        t.manual_seed(base_seed + i)
        binary_ps, n_kept = sample_circuit(ps_dev, temp, device)
        if n_kept == 0:
            continue
        measurements = measure_prune_metrics(
            ablation_types=[eval_ablation],
            metrics=[metric],
            task_prune_scores={task_key: {base_algo_key: binary_ps}},
            patch_type=PatchType.TREE_PATCH,
            test_edge_counts=[n_kept],
        )
        series = extract_metric(measurements, eval_ablation, metric.key, task_key, base_algo_key)
        faith = series[-1][1] if series else float("nan")
        eset = top_k_edge_set(binary_ps, n_kept, edges)
        edge_sets.append(eset)
        rows.append({
            "cell": _label(task_key, algo_key),
            "sample": i,
            "temp": temp,
            "n_kept": n_kept,
            "metric": metric.key,
            "eval_ablation": str(eval_ablation).split(".")[-1],
            "faithfulness": faith,
        })
    return rows, edge_sets


def pairwise_jaccard(edge_sets: List[set]) -> List[float]:
    """Off-diagonal pairwise Jaccard/IoU over sampled edge sets."""
    from auto_circuit.metrics.analysis_utils import jaccard
    return [
        jaccard(edge_sets[i], edge_sets[j])
        for i in range(len(edge_sets))
        for j in range(i + 1, len(edge_sets))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(cells: List[Cell], out_path: Path) -> None:
    """Grid of per-cell sigma(l_e) histograms (log-count y-axis)."""
    labels = [_label(tk, ak) for tk, ak, _ in cells]
    n = len(cells)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels)
    for i, (_tk, _ak, ps) in enumerate(cells):
        p = flatten_scores(ps)
        r, c = i // cols + 1, i % cols + 1
        fig.add_trace(
            go.Histogram(x=p, nbinsx=50, marker_color="#4C72B0", showlegend=False),
            row=r, col=c,
        )
        fig.update_xaxes(title_text="sigma(l_e)", range=[0, 1], row=r, col=c)
        fig.update_yaxes(type="log", title_text="count", row=r, col=c)
    fig.update_layout(
        title="DiscoGP per-edge keep-probabilities — polarization diagnostic",
        height=320 * rows, width=420 * cols, bargap=0.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path.with_suffix(".html")))
    try:
        fig.write_image(str(out_path.with_suffix(".pdf")))
    except Exception as e:
        print(f"WARNING: PDF export failed ({e}); HTML written.")


def plot_faith_vs_size(rows: List[dict], mean_jac: dict, out_path: Path) -> None:
    """Scatter of faithfulness vs sampled circuit size, one trace per cell."""
    fig = go.Figure()
    for cell in sorted({r["cell"] for r in rows}):
        rs = [r for r in rows if r["cell"] == cell]
        fig.add_trace(go.Scatter(
            x=[r["n_kept"] for r in rs], y=[r["faithfulness"] for r in rs],
            mode="markers", name=f"{cell} (IoU={mean_jac.get(cell, float('nan')):.2f})",
        ))
    fig.update_layout(
        title="Sampled circuits: faithfulness vs size (legend shows mean pairwise IoU)",
        xaxis_title="kept edges (n_kept)", yaxis_title="faithfulness (metric value)",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path.with_suffix(".html")))
    try:
        fig.write_image(str(out_path.with_suffix(".pdf")))
    except Exception as e:
        print(f"WARNING: PDF export failed ({e}); HTML written.")


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnose(cells: List[Cell], folder: Path) -> None:
    rows: List[dict] = []
    print(f"\n{'cell':<48} {'edges':>7} {'pol%':>6} {'mid%':>6} {'H_bits':>7} "
          f"{'E[kept]':>8} {'E[Jac]':>7}  verdict")
    print("-" * 116)
    for task_key, algo_key, ps in cells:
        label = _label(task_key, algo_key)
        p = flatten_scores(ps)
        d = sampling_diagnostics(p)
        verdict = feasibility_verdict(d)
        sweep = temp_sweep(p)
        print(f"{label:<48} {d['n_edges']:>7d} "
              f"{100*d['polarized_frac']:>5.1f} {100*d['intermediate_frac']:>5.1f} "
              f"{d['mean_entropy_bits']:>7.3f} {d['exp_kept_edges']:>8.1f} "
              f"{d['exp_jaccard_two_samples']:>7.3f}  {verdict}")
        sweep_str = "  ".join(f"T={tmp:g}:J={jac:.2f}(k={k:.0f})" for tmp, jac, k in sweep)
        print(f"{'  temp-sweep ->':<48} {sweep_str}")
        row = {"cell": label, **d, "verdict": verdict}
        for tmp, jac, k in sweep:
            row[f"jaccard_T{tmp:g}"] = jac
            row[f"kept_T{tmp:g}"] = k
        rows.append(row)

    csv_path = folder / "diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_histograms(cells, folder / "histograms")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {folder / 'histograms.html'} (+ .pdf if kaleido available)")
    print("\nReading the verdict:")
    print("  E[Jac] ~ 1.0  -> masks polarized to 0/1; sampling reproduces the same circuit")
    print("                   (needs temperature; see the temp-sweep for the T that helps).")
    print("  E[Jac] < ~0.6 -> learned probabilities retain entropy; sampling already diverse.")
    print("\nNext: rerun with --evaluate to confirm the sampled circuits are also FAITHFUL.")


def run_evaluate(cells: List[Cell], n_samples: int, temp: float, eval_ablation_short: str,
                 metric_short: str, base_seed: int, folder: Path) -> None:
    from auto_circuit.types import AblationType
    from auto_circuit.metrics.prune_metrics.prune_metrics import (
        ANSWER_PAIR_KL_DIV_METRIC,
        CLEAN_KL_DIV_METRIC,
        CORRECT_ANSWER_PERCENT_METRIC,
        LOGIT_DIFF_PERCENT_METRIC,
    )
    ablation_map = {
        "zero": AblationType.ZERO,
        "resample": AblationType.RESAMPLE,
        "mean": AblationType.TOKENWISE_MEAN_CORRUPT,
    }
    metric_map = {
        "clean_kl": CLEAN_KL_DIV_METRIC,
        "answer_pair_kl": ANSWER_PAIR_KL_DIV_METRIC,
        "logit_diff": LOGIT_DIFF_PERCENT_METRIC,
        "correct_pct": CORRECT_ANSWER_PERCENT_METRIC,
    }
    eval_ablation = ablation_map[eval_ablation_short]
    metric = metric_map[metric_short]

    all_rows: List[dict] = []
    mean_jac: dict = {}
    summary: List[dict] = []
    for cell in cells:
        label = _label(cell[0], cell[1])
        print(f"\n[evaluate] {label}: drawing {n_samples} circuits "
              f"(temp={temp}, eval={eval_ablation_short}, metric={metric_short}) ...")
        rows, edge_sets = evaluate_cell(cell, n_samples, temp, eval_ablation, metric, base_seed)
        if not rows:
            print("  all samples were empty circuits; skipping.")
            continue
        js = pairwise_jaccard(edge_sets)
        mj = float(np.mean(js)) if js else float("nan")
        mean_jac[label] = mj
        faiths = np.array([r["faithfulness"] for r in rows], dtype=float)
        sizes = np.array([r["n_kept"] for r in rows], dtype=float)
        all_rows.extend(rows)
        summary.append({
            "cell": label, "n_samples": len(rows), "temp": temp,
            "metric": metric.key, "eval_ablation": eval_ablation_short,
            "mean_pairwise_iou": mj,
            "faith_mean": float(np.nanmean(faiths)), "faith_std": float(np.nanstd(faiths)),
            "size_mean": float(sizes.mean()), "size_std": float(sizes.std()),
        })
        print(f"  mean pairwise IoU={mj:.3f}  |  faithfulness {np.nanmean(faiths):.4f}"
              f"±{np.nanstd(faiths):.4f}  |  size {sizes.mean():.0f}±{sizes.std():.0f}")

    if not all_rows:
        print("No non-empty sampled circuits produced; nothing to write.")
        return

    with open(folder / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(folder / "samples_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    plot_faith_vs_size(all_rows, mean_jac, folder / "faith_vs_size")

    print(f"\nWrote: {folder / 'samples.csv'}  ({len(all_rows)} sampled circuits)")
    print(f"Wrote: {folder / 'samples_summary.csv'}, {folder / 'faith_vs_size.html'}")
    print("\nReading the result (compare against your Pillar-2 baselines):")
    print("  mean pairwise IoU << 1  AND  low/stable faithfulness  -> sampled circuits are")
    print("    diverse AND faithful: a cheap single-run alternative to OASR's repulsion.")
    print("  Reference points: OASR engineered IoU ~ 0.04 (Chen 2026); reseeding Jaccard")
    print("    ~ 0.56 for EAP-IG (Meloux); your trained-mask analytic E[Jac] from --diagnose.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Per-edge circuit sampling from DiscoGP masks (diagnose + evaluate).")
    ap.add_argument("--cache-dir", default=FAITHFULNESS_CACHE_DIR, help="Discovery cell cache folder")
    ap.add_argument("--prefix", default="cell_", help="Cache filename prefix (no trailing '*'); narrow it to pick one cell")
    ap.add_argument("--algo-filter", default="discogp", help="Case-insensitive substring on algo key (DiscoGP scores are sigma(l_e))")
    ap.add_argument("--run-fresh", action="store_true", help="Train a DiscoGP mask instead of loading cache")
    ap.add_argument("--task", default="ioi", help="Task for --run-fresh (ioi/gt/docstring)")
    ap.add_argument("--ablation", default="zero", help="Discovery ablation for --run-fresh (zero/resample/mean)")
    ap.add_argument("--evaluate", action="store_true", help="Sample N circuits and score faithfulness + pairwise IoU")
    ap.add_argument("--n-samples", type=int, default=20, help="Circuits to draw per cell (--evaluate)")
    ap.add_argument("--temp", type=float, default=1.0, help="Sampling temperature T in p_e(T)=sigma(l_e/T)")
    ap.add_argument("--eval-ablation", default="resample", choices=["zero", "resample", "mean"], help="Ablation for faithfulness eval")
    ap.add_argument("--metric", default="clean_kl", choices=["clean_kl", "answer_pair_kl", "logit_diff", "correct_pct"], help="Faithfulness metric")
    ap.add_argument("--seed", type=int, default=42, help="Base seed (data + per-sample reproducibility)")
    ap.add_argument("--name", default=None, help="Output subfolder name")
    args = ap.parse_args()

    if args.run_fresh:
        print(f"Training fresh DiscoGP edge-only: task={args.task} ablation={args.ablation} seed={args.seed}")
        cells = run_fresh(args.task, args.ablation, args.seed)
    else:
        cells = collect_cells(args.cache_dir, args.prefix, args.algo_filter)

    if not cells:
        print(f"No cells matched prefix '{args.prefix}' + algo '{args.algo_filter}' in "
              f"{args.cache_dir}. Try --run-fresh, or adjust --prefix/--algo-filter.")
        return

    folder = repo_path_to_abs_path(RESULTS_DIR) / (args.name or f"seed{args.seed}")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(cells)} cell(s). Writing to: {folder}")

    if args.evaluate:
        run_evaluate(cells, args.n_samples, args.temp, args.eval_ablation,
                     args.metric, args.seed, folder)
    else:
        run_diagnose(cells, folder)


if __name__ == "__main__":
    main()
