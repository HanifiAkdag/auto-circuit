"""Exact full-ranking manual-circuit AUC and native-ablation ROC panels.

Run on a CPU compute node: srun python experiments/results_chapter_native_roc.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from scipy.stats import mannwhitneyu, rankdata


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".prune_scores/final/benchmark"
OUT = ROOT / "results/results_chapter"
ROC_OUT = OUT / "native_roc"
INSPECTION = Path("/tmp/auto-circuit-results-inspection/native_roc")
STABILITY = ROOT / "results/stability/decomposition_2026-09"
TASKS = {"ioi": ("IOI", "mean", 739, "Indirect Object Identification"),
         "gt": ("Greater-Than", "resample", 235, "Greaterthan"),
         "docstring": ("Docstring", "resample", 24, "Docstring")}
METHODS = {"acdc": ("ACDC", "#009E73"),
           "eap": ("EAP", "#7B2CBF"),
           "ep": ("Edge Pruning", "#0072B2"),
           "discogp-ep": ("DiscoGP", "#D55E00")}
ABLATIONS = ("zero", "resample", "mean")
SEEDS = (42, 43, 44, 45, 46)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def cache_index():
    best = {}
    for path in CACHE.glob("*_manifest.json"):
        meta = json.loads(path.read_text())
        c = meta.get("cell", {})
        task, algo, abl = c.get("task"), c.get("algo"), c.get("ablation")
        seed = c.get("seed")
        if task not in TASKS or c.get("null_test"):
            continue
        if algo == "gt":
            if abl != "resample" or seed != 42:
                continue
        elif algo not in METHODS or abl not in ABLATIONS or seed not in SEEDS:
            continue
        elif c.get("init_seed", seed) != seed:
            continue
        key = (task, algo, abl, seed)
        pkl = CACHE / meta["cache_file"]
        assert pkl.is_file(), pkl
        if key not in best or meta["saved_at"] > best[key][0]["saved_at"]:
            best[key] = (meta, pkl, path)
    expected = 3 + len(TASKS) * len(METHODS) * len(ABLATIONS) * len(SEEDS)
    assert len(best) == expected, (len(best), expected)
    return best


def score_dict(path):
    tps = torch.load(path, weights_only=True, map_location="cpu")
    ps = next(iter(next(iter(tps.values())).values()))
    assert isinstance(ps, dict) and all(isinstance(v, torch.Tensor) for v in ps.values())
    return ps


def flatten(ps, keys, shapes):
    assert set(ps) == set(keys)
    parts = []
    for k in keys:
        assert tuple(ps[k].shape) == shapes[k]
        parts.append(ps[k].detach().abs().reshape(-1).to(torch.float64).numpy())
    v = np.concatenate(parts)
    # ACDC may encode never-pruned edges as +inf; that is a valid top rank.
    assert not np.isnan(v).any()
    return v


def exact_auc(y, x):
    """Mann–Whitney AUC: average score ranks give half credit for ties."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = rankdata(x, method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def step_at(fpr, tpr, grid):
    return tpr[np.maximum(0, np.searchsorted(fpr, grid, side="right") - 1)]


def roc_points(labels, scores):
    """One threshold per distinct score, retaining tied blocks together."""
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    positive = labels[order]
    last = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(scores) - 1]
    tp = np.cumsum(positive)[last]
    fp = last + 1 - tp
    return np.r_[0, fp / (~labels).sum()], np.r_[0, tp / labels.sum()]


def main():
    ROC_OUT.mkdir(parents=True, exist_ok=True)
    INSPECTION.mkdir(parents=True, exist_ok=True)
    index = cache_index()
    assert exact_auc(np.array([True, False, True, False]), np.ones(4)) == .5
    summary = list(csv.DictReader((STABILITY / "summary.csv").open(newline="")))
    used = []
    auc_rows, mean_rows, seed_curves, plotted = [], [], [], []
    acdc_checks = []
    grid = np.linspace(0, 1, 1001)
    for task, (_, native, expected_pos, summary_name) in TASKS.items():
        manual_key = (task, "gt", "resample", 42)
        manual_meta, manual_path, manual_manifest = index[manual_key]
        manual = score_dict(manual_path)
        keys = sorted(manual)
        shapes = {k: tuple(manual[k].shape) for k in keys}
        label_values = flatten(manual, keys, shapes)
        assert np.all(np.isin(label_values, [0., 1.]))
        labels = label_values == 1
        assert int(labels.sum()) == expected_pos, (task, labels.sum())
        used.append((manual_meta, manual_path, manual_manifest))
        for method in METHODS:
            for ablation in ABLATIONS:
                aucs, curve_values = [], []
                for seed in SEEDS:
                    meta, path, manifest_path = index[(task, method, ablation, seed)]
                    scores = flatten(score_dict(path), keys, shapes)
                    auc = exact_auc(labels, scores)
                    if task == "ioi" and method == "acdc" and ablation == native and seed == 42:
                        trusted = float(mannwhitneyu(scores[labels], scores[~labels],
                                                     alternative="two-sided", method="asymptotic").statistic
                                        / (labels.sum() * (~labels).sum()))
                        assert abs(auc - trusted) < 1e-12
                        trusted_check = {"cell": f"{task}/{method}/{ablation}/{seed}",
                                         "rank_auc": auc,
                                         "scipy_mannwhitneyu_auc": trusted}
                    aucs.append(auc)
                    auc_rows.append({"task": task, "method": method, "discovery_ablation": ablation,
                                     "seed": seed, "auc": auc, "cache_file": path.name})
                    used.append((meta, path, manifest_path))
                    if method == "acdc":
                        ref = [r for r in summary if r["task"] == summary_name
                               if r["algo"] == "ACDC" and r["discovery_ablation"] == ablation
                               and r["condition"] == "joint"]
                        assert len(ref) == 1, (task, ablation, ref)
                        tau = float(ref[0]["budget_value"])
                        realised = int(np.count_nonzero(scores >= tau))
                        acdc_checks.append((task, ablation, seed, tau, realised,
                                            float(ref[0]["realised_mean"])))
                    if ablation == native:
                        fpr, tpr = roc_points(labels, scores)
                        assert fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1
                        assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
                        assert abs(float(np.trapz(tpr, fpr)) - auc) < 1e-10
                        vals = step_at(fpr, tpr, grid)
                        curve_values.append(vals)
                        seed_curves.extend({"task": task, "method": method,
                                            "discovery_ablation": ablation, "seed": seed,
                                            "fpr": float(f), "tpr": float(t)}
                                           for f, t in zip(fpr, tpr))
                mean_rows.append({"task": task, "method": method,
                                  "discovery_ablation": ablation,
                                  "n_seeds": len(aucs), "auc_mean": float(np.mean(aucs)),
                                  "auc_sd": float(np.std(aucs, ddof=1))})
                if ablation == native:
                    arr = np.stack(curve_values)
                    plotted.append({"task": task, "method": method,
                                    "discovery_ablation": ablation, "fpr": 0.0,
                                    "tpr_mean": 0.0, "tpr_min": 0.0, "tpr_max": 0.0})
                    plotted.extend({"task": task, "method": method,
                                    "discovery_ablation": ablation, "fpr": float(f),
                                    "tpr_mean": float(m), "tpr_min": float(lo),
                                    "tpr_max": float(hi)}
                                   for f, m, lo, hi in zip(grid, arr.mean(0),
                                                            arr.min(0), arr.max(0)))

    for task in TASKS:
        for abl in ABLATIONS:
            checks = [v for t, a, _, _, v, _ in acdc_checks if t == task and a == abl]
            expected = next(e for t, a, _, _, _, e in acdc_checks if t == task and a == abl)
            assert len(checks) == 5 and abs(np.mean(checks) - expected) < 1e-9, (task, abl, checks, expected)

    write_csv(ROC_OUT / "manual_circuit_auc_per_seed.csv", auc_rows,
              ["task", "method", "discovery_ablation", "seed", "auc", "cache_file"])
    write_csv(ROC_OUT / "manual_circuit_auc.csv", mean_rows,
              ["task", "method", "discovery_ablation", "n_seeds", "auc_mean", "auc_sd"])
    write_csv(ROC_OUT / "manual_native_roc_per_seed.csv", seed_curves,
              ["task", "method", "discovery_ablation", "seed", "fpr", "tpr"])
    write_csv(ROC_OUT / "manual_native_roc_plotted.csv", plotted,
              ["task", "method", "discovery_ablation", "fpr", "tpr_mean", "tpr_min", "tpr_max"])

    fig = make_subplots(rows=1, cols=3, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=.085,
                        subplot_titles=[f"{chr(65+i)}  {name} ({native})"
                                        for i, (_, (name, native, _, _)) in enumerate(TASKS.items())])
    for i, (task, (name, native, _, _)) in enumerate(TASKS.items()):
        col = i + 1
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999999",
                      width=1, dash="dot"), name="Chance", showlegend=i == 0,
                      legendgroup="chance"), row=1, col=col)
        for method, (label, color) in METHODS.items():
            rr = [r for r in plotted if r["task"] == task and r["method"] == method]
            fpr = np.array([r["fpr"] for r in rr])
            mean = np.array([r["tpr_mean"] for r in rr])
            lo = np.array([r["tpr_min"] for r in rr])
            hi = np.array([r["tpr_max"] for r in rr])
            # Fill is a five-run range at a common FPR grid. Mean is a step
            # function; no smoothing or sparse-budget trapezoids enter AUC.
            xx = np.r_[fpr, fpr[::-1]]
            yy = np.r_[hi, lo[::-1]]
            fig.add_trace(go.Scatter(x=xx, y=yy, fill="toself", fillcolor=color,
                          opacity=.085, line=dict(width=0), hoverinfo="skip",
                          showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(x=fpr, y=mean, mode="lines",
                          line=dict(color=color, width=2.3, shape="hv"),
                          name=label, legendgroup=method,
                          showlegend=i == 0), row=1, col=col)
        fig.update_xaxes(range=[0, 1], dtick=.25, title_text="False-positive rate",
                         showline=True, linecolor="#666666", linewidth=.7,
                         ticks="outside", gridcolor="#e8e8e8", gridwidth=.5,
                         title_font=dict(size=16), tickfont=dict(size=15),
                         row=1, col=col)
        fig.update_yaxes(range=[0, 1.025], dtick=.25,
                         title_text="True-positive rate" if i == 0 else None,
                         showline=True, linecolor="#666666", linewidth=.7,
                         ticks="outside", gridcolor="#e8e8e8", gridwidth=.5,
                         title_font=dict(size=16), tickfont=dict(size=15),
                         row=1, col=col)
    fig.update_layout(width=1060, height=405, margin=dict(l=78, r=25, t=50, b=85),
                      paper_bgcolor="white", plot_bgcolor="white", showlegend=True,
                      legend=dict(orientation="h", x=.5, xanchor="center", y=-.24,
                                  yanchor="top", font=dict(size=17),
                                  traceorder="normal", itemsizing="constant"),
                      font=dict(family="Arial", size=15, color="#222222"))
    for annotation in fig.layout.annotations[:3]:
        annotation.font.size = 17
    fig.write_image(str(ROC_OUT / "manual_native_roc.pdf"))
    fig.write_image(str(INSPECTION / "manual_native_roc.png"), scale=1.4)

    identities = [{"cache_file": p.name, "cache_manifest": mpath.name,
                   "cell": meta["cell"], "saved_at": meta["saved_at"]}
                  for meta, p, mpath in used]
    manifest = {
        "source_dir": str(CACHE.relative_to(ROOT)), "source_caches": identities,
        "manual_positives": {k: v[2] for k, v in TASKS.items()},
        "native_discovery_ablation": {k: v[1] for k, v in TASKS.items()},
        "seeds": list(SEEDS),
        "candidate_edges": "all entries of each PruneScores tensor, aligned by module name and tensor shape",
        "score": "absolute edge score", "labels": "positive iff deterministic official-circuit cache score is 1",
        "auc": "Mann-Whitney probability; scipy average ranks give half credit for ties; sample SD (ddof=1)",
        "roc": "all distinct scores, tied blocks together; step interpolation on 1001 common FPR points; mean and five-run min/max",
        "all_tied_auc": exact_auc(np.array([True, False, True, False]), np.ones(4)),
        "independent_auc_check": trusted_check,
        "acdc_selected_boundary_check": [{"task": t, "ablation": a, "seed": s,
                                          "tau": tau, "realised": v, "summary_mean": e}
                                         for t, a, s, tau, v, e in acdc_checks],
        "summary_sha256": hashlib.sha256((STABILITY / "summary.csv").read_bytes()).hexdigest(),
    }
    (ROC_OUT / "manual_native_roc_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Manual positives:", manifest["manual_positives"])
    print("Trusted AUC:", trusted_check)
    print("ACDC selected sizes matched summary.csv for 9 cells")
    print(f"Wrote ROC figure, {len(auc_rows)} per-seed AUCs, {len(mean_rows)} mean AUCs")


if __name__ == "__main__":
    main()
