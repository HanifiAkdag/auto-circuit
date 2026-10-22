"""Exact full-ranking manual-circuit AUC and native-ablation ROC panels.

Run on a CPU compute node: srun --mem=8G python -m experiments.results_chapter_native_roc
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from scipy.stats import mannwhitneyu, rankdata
from .config import ROOT, load_registry, resolve_path
from .results_chapter_sources import EXPORT_ROOT, frozen_manifest_index, provenance_path


OUT = EXPORT_ROOT
ROC_OUT = OUT / "native_roc"
INSPECTION = ROC_OUT / "inspection"
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


def cache_index(registry=None):
    registry = registry or load_registry()
    best = frozen_manifest_index(registry['benchmark']['inputs'], TASKS, METHODS,
                                 ABLATIONS, SEEDS, registry=registry, include_manual=True)
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


def vertical_at(fpr, tpr, grid):
    """Linearly interpolate the highest TPR at each common false-positive rate."""
    unique_fpr, starts = np.unique(fpr, return_index=True)
    highest_tpr = np.maximum.reduceat(tpr, starts)
    return np.interp(grid, unique_fpr, highest_tpr)


def roc_points(labels, scores):
    """One threshold per distinct score, retaining tied blocks together."""
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    positive = labels[order]
    last = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(scores) - 1]
    tp = np.cumsum(positive)[last]
    fp = last + 1 - tp
    return np.r_[0, fp / (~labels).sum()], np.r_[0, tp / labels.sum()]


def render_native_figures(plotted, mean_rows, output_root=None):
    """Render main-text IOI/Greater-Than panels and a separate Docstring panel."""
    names = {key: value[0] for key, value in METHODS.items()}
    native = {task: values[1] for task, values in TASKS.items()}
    auc = {(row["task"], row["method"], row["discovery_ablation"]): row
           for row in mean_rows}

    def make_figure(task_keys, pdf_path, png_path):
        fig = make_subplots(rows=1, cols=len(task_keys), shared_xaxes=True,
                            shared_yaxes=True, horizontal_spacing=.12,
                            subplot_titles=[TASKS[key][0] for key in task_keys])
        for col, task in enumerate(task_keys, start=1):
            ablation = native[task]
            suffix = "" if col == 1 else str(col)
            xref, yref = f"x{suffix} domain", f"y{suffix} domain"
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                          line=dict(color="#777777", width=1.5, dash="dot"),
                          hoverinfo="skip", showlegend=False), row=1, col=col)
            for method, (_, color) in METHODS.items():
                rows = [row for row in plotted
                        if row["task"] == task and row["method"] == method
                        and row["discovery_ablation"] == ablation]
                rows.sort(key=lambda row: float(row["fpr"]))
                fpr = np.array([float(row["fpr"]) for row in rows])
                avg = np.array([float(row["tpr_mean"]) for row in rows])
                low = np.array([float(row["tpr_min"]) for row in rows])
                high = np.array([float(row["tpr_max"]) for row in rows])
                assert len(fpr) >= 1000 and np.all(np.diff(fpr) >= 0)

                fig.add_trace(go.Scatter(
                    x=np.r_[fpr, fpr[::-1]], y=np.r_[high, low[::-1]],
                    fill="toself", fillcolor=color, opacity=.10,
                    line=dict(width=0), hoverinfo="skip", showlegend=False),
                    row=1, col=col)
                fig.add_trace(go.Scatter(
                    x=fpr, y=avg, mode="lines",
                    line=dict(color=color, width=3, shape="linear"),
                    name=names[method], hovertemplate=(
                        f"{names[method]}<br>FPR=%{{x:.3f}}<br>TPR=%{{y:.3f}}<extra></extra>"),
                    showlegend=False), row=1, col=col)

            # Panel-local key reports the AUC for the same task and native
            # discovery ablation, avoiding a misleading shared AUC legend.
            fig.add_shape(type="rect", xref=xref, yref=yref,
                          x0=.35, x1=.99, y0=.025, y1=.325,
                          line=dict(color="#999999", width=.8),
                          fillcolor="rgba(255,255,255,0.94)", layer="above")
            fig.add_annotation(x=.37, y=.292, xref=xref, yref=yref,
                               text="AUC (average ± SD)", showarrow=False,
                               xanchor="left", yanchor="middle",
                               font=dict(size=19, color="#222222"))
            for line, (method, (_, color)) in enumerate(METHODS.items()):
                stats = auc[(task, method, ablation)]
                text = (f"{names[method]}  "
                        f"{float(stats['auc_mean']):.3f} ± "
                        f"{float(stats['auc_sd']):.3f}")
                fig.add_annotation(x=.37, y=.238 - line * .058,
                                   xref=xref, yref=yref, text=text,
                                   showarrow=False, xanchor="left", yanchor="middle",
                                   font=dict(size=19, color=color))

            fig.update_xaxes(range=[0, 1], tickvals=[0, .25, .5, .75, 1],
                             tickformat=".2g", title_text="False-positive rate",
                             showline=True, linecolor="#555555", linewidth=1,
                             ticks="outside", tickfont=dict(size=24),
                             title_font=dict(size=26), gridcolor="#e5e5e5",
                             gridwidth=.7, row=1, col=col)
            fig.update_yaxes(range=[0, 1], tickvals=[0, .25, .5, .75, 1],
                             tickformat=".2g",
                             title_text="True-positive rate" if col == 1 else None,
                             showline=True, linecolor="#555555", linewidth=1,
                             ticks="outside", tickfont=dict(size=24),
                             title_font=dict(size=26), gridcolor="#e5e5e5",
                             gridwidth=.7, scaleanchor=f"x{suffix}", scaleratio=1,
                             row=1, col=col)

        compact = len(task_keys) == 1
        fig.update_layout(width=1450 if not compact else 620,
                          height=760 if not compact else 620,
                          margin=(dict(l=90, r=25, t=70, b=100) if not compact
                                  else dict(l=85, r=20, t=48, b=72)),
                          paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(family="Arial", size=19, color="#222222"),
                          showlegend=False)
        for annotation in fig.layout.annotations[:len(task_keys)]:
            annotation.font.size = 26
            annotation.font.color = "#222222"
        if pdf_path is not None:
            fig.write_image(str(pdf_path))
        fig.write_image(str(png_path), scale=1.2)

    roc_out = (ROOT / output_root).resolve() / 'native_roc' if output_root is not None else ROC_OUT
    inspection = roc_out / 'inspection'
    inspection.mkdir(parents=True, exist_ok=True)
    make_figure(("ioi",), roc_out / "ioi_native_roc.pdf",
                inspection / "ioi_native_roc.png")
    make_figure(("gt",), roc_out / "greater_than_native_roc.pdf",
                inspection / "greater_than_native_roc.png")
    make_figure(("ioi", "gt"), None, inspection / "manual_native_roc.png")
    make_figure(("docstring",), roc_out / "docstring_native_roc.pdf",
                inspection / "docstring_native_roc.png")


def replot_saved_results(output_root=None):
    """Re-render plots from the audited CSVs without loading score caches."""
    roc_out = (ROOT / output_root).resolve() / 'native_roc' if output_root is not None else ROC_OUT
    with (roc_out / "manual_native_roc_plotted.csv").open(newline="") as f:
        plotted = list(csv.DictReader(f))
    with (roc_out / "manual_circuit_auc.csv").open(newline="") as f:
        mean_rows = list(csv.DictReader(f))
    assert len(mean_rows) == len(TASKS) * len(METHODS) * len(ABLATIONS)
    for task, (_, ablation, _, _) in TASKS.items():
        assert sum(row["task"] == task and row["discovery_ablation"] == ablation
                   for row in plotted) >= 4 * 1000
        assert all((task, method, ablation) in {
            (row["task"], row["method"], row["discovery_ablation"])
            for row in mean_rows} for method in METHODS)
    render_native_figures(plotted, mean_rows, output_root)
    print("Rendered two single-panel main ROC PDFs, a composite preview, and a separate Docstring appendix panel.")


def summary_sources(registry, output_root):
    paths = []
    for name in ('main', 'docstring'):
        regenerated = Path(output_root).parent / 'evaluation' / name / 'summary.csv'
        path = (regenerated if regenerated.is_file() else
                resolve_path(registry['benchmark']['bundles'][name]['source_dir'], registry) / 'summary.csv')
        paths.append(path)
    return paths


def main(registry=None, output_root=None):
    registry = registry or load_registry()
    output_root = (ROOT / output_root).resolve() if output_root is not None else EXPORT_ROOT
    roc_out = output_root / 'native_roc'
    roc_out.mkdir(parents=True, exist_ok=True)
    index = cache_index(registry)
    assert exact_auc(np.array([True, False, True, False]), np.ones(4)) == .5
    tied_fpr, tied_tpr = roc_points(np.array([True, False, True, False]),
                                    np.ones(4))
    tie_grid = np.linspace(0, 1, 5)
    assert np.allclose(vertical_at(tied_fpr, tied_tpr, tie_grid), tie_grid)
    summary_paths = summary_sources(registry, output_root)
    summary = []
    for path in summary_paths:
        with path.open(newline='') as handle:
            summary.extend(csv.DictReader(handle))
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
                        vals = vertical_at(fpr, tpr, grid)
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

    write_csv(roc_out / "manual_circuit_auc_per_seed.csv", auc_rows,
              ["task", "method", "discovery_ablation", "seed", "auc", "cache_file"])
    write_csv(roc_out / "manual_circuit_auc.csv", mean_rows,
              ["task", "method", "discovery_ablation", "n_seeds", "auc_mean", "auc_sd"])
    write_csv(roc_out / "manual_native_roc_per_seed.csv", seed_curves,
              ["task", "method", "discovery_ablation", "seed", "fpr", "tpr"])
    write_csv(roc_out / "manual_native_roc_plotted.csv", plotted,
              ["task", "method", "discovery_ablation", "fpr", "tpr_mean", "tpr_min", "tpr_max"])

    render_native_figures(plotted, mean_rows, output_root)

    identities = [{"cache_file": p.name, "cache_manifest": mpath.name,
                   "cell": meta["cell"], "saved_at": meta["saved_at"]}
                  for meta, p, mpath in used]
    manifest = {
        "source_registry": provenance_path(Path(registry.get('_path', ROOT / 'artifacts/final/registry.json'))),
        "source_caches": identities,
        "manual_positives": {k: v[2] for k, v in TASKS.items()},
        "native_discovery_ablation": {k: v[1] for k, v in TASKS.items()},
        "seeds": list(SEEDS),
        "candidate_edges": "all entries of each PruneScores tensor, aligned by module name and tensor shape",
        "score": "absolute edge score", "labels": "positive iff deterministic official-circuit cache score is 1",
        "auc": "Mann-Whitney probability; scipy average ranks give half credit for ties; sample SD (ddof=1)",
        "roc": "all distinct scores, tied blocks together; linear interpolation at the highest TPR on 1001 common FPR points; average and five-run min-max",
        "all_tied_auc": exact_auc(np.array([True, False, True, False]), np.ones(4)),
        "independent_auc_check": trusted_check,
        "acdc_selected_boundary_check": [{"task": t, "ablation": a, "seed": s,
                                          "tau": tau, "realised": v, "summary_mean": e}
                                         for t, a, s, tau, v, e in acdc_checks],
        "summary_sha256": {provenance_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in summary_paths},
    }
    (roc_out / "manual_native_roc_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Manual positives:", manifest["manual_positives"])
    print("Trusted AUC:", trusted_check)
    print("ACDC selected sizes matched summary.csv for 9 cells")
    print(f"Wrote ROC figure, {len(auc_rows)} per-seed AUCs, {len(mean_rows)} mean AUCs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-render existing audited CSVs without loading score caches")
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.plot_only:
        replot_saved_results(args.output_root)
    else:
        main(registry=load_registry(args.registry), output_root=args.output_root)
