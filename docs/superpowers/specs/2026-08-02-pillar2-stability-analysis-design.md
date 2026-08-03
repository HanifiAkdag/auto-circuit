# Pillar 2 — stability analysis pass

**Date:** 2026-08-02
**Issue:** [#5](https://github.com/HanifiAkdag/auto-circuit/issues/5) (epic #2)
**Depends on:** the Pillar-2 discovery run (`2026-08-01-pillar2-stability-variance-design.md`)
**Scope:** analysis only. Discovery is specified separately and has already run.

## Goal

Turn the 195 cached discovery cells into an answer to the Pillar-2 question: how much of a
discovered circuit is an artifact of the random draw, and does structural instability show
up in behaviour?

Structure alone cannot separate "this algorithm is unreliable" from "this task admits many
equally faithful circuits". Both readings predict a low Jaccard; only the functional spread
tells them apart. So the pass has two stages and one joined table.

## What the reference papers do

| | Méloux et al. 2025 (arXiv 2510.00845) | Chen et al. 2026 (ICML, arXiv 2605.12671) |
|---|---|---|
| structural metric | mean and variance of pairwise Jaccard | pairwise IoU, min pairwise IoU, and mutual IoU = \|∩ of all runs\| / \|∪ of all runs\| with both raw counts |
| reported against | circuit error and KL, as µ / σ² / CV, in the same table | accuracy, completeness, edge density, mean \|E\|, in the same table |
| budget | fixed | **swept**: IoU falls 0.80 → 0.65 as top-k grows (Fig. 3b) |
| n | 100 bootstrap resamples | 20 discovery runs |
| ACDC | not covered | compared at **fixed τ = 0.00398**, not fixed size (Table 4) |
| shape of the distribution | split violins, full pairwise heatmap, 2-D projection | — |
| recommendation | report mean pairwise Jaccard + CV of the functional metric; µ_J > 0.8 at n ≥ 100 as a tentative stability bar | — |

Two design consequences. A single-k stability number would be an artifact of the k chosen,
so the budget is swept. And matching τ for ACDC is the published practice, not a workaround.

Neither paper measures plain seed-to-seed variance for all four algorithms under a matched
protocol, and neither has a chance floor. The Random anchor supplies one.

## Selection rule

`analysis_utils.Budget(kind, value)` with `kind ∈ {top_k, threshold}` is the single
abstraction; `select_edges` / `budget_mask` resolve it with the same `|score| >= threshold`
rule `run_circuits` uses (`prune.py:94`), so the structural and functional passes always
select the same circuits.

- **top-k** for EP, DiscoGP, EAP, Random. Their boundary tie block is exactly 1 edge
  (measured), so top-k lands on the requested size.
- **matched τ** for ACDC. Its score is the τ at which an edge was pruned, giving ~21
  distinct levels over 32 491 edges; a top-k request admits the straddling tie block whole
  and overshoots by a seed-dependent 8–165 edges, which caps Jaccard by arithmetic alone.
  All 20 τ levels are shared across seeds, and realised sizes come out tight
  (e.g. `[202, 203, 203, 206, 209]`).

The τ grid comes from `algo_hyperparams(ACDC_PRUNE_ALGO)`, **not** from a cell manifest: the
30 τ-split ACDC × {ioi, gt} cells are written by `merge_acdc_partials.py`, whose manifest has
no `algo_hyperparams` block. Deriving it from the data would also be wrong, because a τ that
pruned nothing in one seed is simply absent from that seed's score levels.

Realised sizes are reported for every algorithm in `structural.csv`
(`realised_mean/min/max`), so the ACDC caveat is a number in the table rather than a
footnote.

## Structural pass (CPU)

Per (task, algo, discovery ablation), over the 5 seeds and a swept budget grid:

- mean / sd / **CV** / min / max of the pairwise Jaccard, plus `n_pairs`
- mutual IoU with raw `|E∩|`, `|E∪|`
- selection frequency: edges found by exactly 5, 4, 3, 2, 1 of the seeds

**`n_pairs` is reported because the pairs are not independent** (each seed appears in four
of the ten), so the spread is a range over dependent comparisons and never a confidence
interval. Every figure states this.

**Vectorised, via `EdgeIndex`.** `top_k_edge_set` loops over every `Edge` with a `.item()`:
measured 204 ms per call at GPT-2 scale, and the pass needs ~5 200 of them. Instead the
scores are gathered once per (cell, seed) into a vector aligned to `sorted(edges, key=name)`,
after which every budget and every statistic is a numpy boolean op. This is **not** the
forbidden flattening: the index space is edge identity, not `PruneScores` dict order, so the
15 legacy EAP key orders on disk are irrelevant by construction. A key-order-shuffle
invariance test pins it.

## Functional pass (GPU)

Each seed's circuit re-scored with `measure_prune_metrics`, `PatchType.TREE_PATCH`
(faithfulness; completeness is out of scope), under **all three eval ablations**.

One call per (task, algo, ablation, seed), not per seed-group: ACDC's τ→k mapping is
per-seed and per-algo, so the calls cannot share a `test_edge_counts`. `test_edge_counts` is
always passed explicitly; left `None`, `measure_prune_metrics` infers a per-algo grid and
switches to `EdgeCounts.GROUPS` for tied scores, so seeds would not line up.

Evaluation reads `TASK_DICT[task_key].test_loader`, the unseeded template task, so **all
five seeds are scored on the same test set**: the functional spread is circuit variance, not
test-draw variance. The seed varies discovery only.

Random is discovered only at `resample` but is ablation-independent, so it is evaluated
under all three to act as a floor in every panel.

## The headline functional metric is not Clean KL

For a faithful circuit the mean Clean KL approaches zero, so `sd/mean` is dominated by its
denominator and yields meaningless three-digit CVs. `functional.csv` records
mean/sd/CV/min/max for every metric, but the CV reported in figures and `summary.md` is of a
bounded, higher-better metric: `Logit Diff Percent` for IOI and Greater-Than,
`Correct Answer Percent` for Docstring (where Logit-Diff-Percent is excluded as unstable).
Clean KL is carried as mean ± sd. Méloux et al. likewise use an accuracy-like functional
metric.

Consequence: the stability-vs-function figure is faceted by task and its y-axis is **not**
comparable across facets. The caption says so.

## Outputs

```
results/stability/<experiment>/
  manifest.json  structural.csv  functional.csv  summary.csv  summary.md
  figures/
    jaccard_vs_budget.{pdf,png,html}          3x3: rows = discovery ablation, cols = task
    mutual_iou_vs_budget.{pdf,png,html}       same layout, all-five agreement
    compact/jaccard_vs_budget.{pdf,png,html}  1x3 overview, every series, no bands
    selection_frequency.{pdf,png,html}        normalised stacked bars per cell
    stability_vs_function.{pdf,png,html}      the joined figure
    faithfulness/<task>/<metric>.{pdf,png,html}
```

`summary.csv` joins structure and function at the published circuit size, one row per cell,
which is the row format both reference papers print.

## Figure conventions

All figures go through `plot_theme` (`apply_theme`, fixed algorithm hues via `algo_colors`,
`DISC_SYMBOL`/`DISC_DASH` for the ablation axis, `SEQUENTIAL_BLUE` for ordered categories).
Random is drawn as a recessive dashed anchor, never as a competitor.

- **Band = min-to-max over the seed pairs.** Not a CI (dependent pairs); the lower edge is
  itself a published quantity, Chen et al. report min pairwise IoU as its own column.
- **The chance floor is empirical.** For independent draws E[J] ≈ k/(2E−k), which *rises*
  with k, so no constant analytic floor line is drawn; the Random series carries it.
- The 0.8 line is labelled as Méloux et al.'s bar, not as a criterion of this thesis: n = 5
  here and the perturbation is reseeding rather than bootstrap.
- `jaccard_vs_budget` uses a 3×3 grid rather than 1×3 with dashes because 13 series per
  panel cannot carry legible bands, and the band is the point.

## Interfaces

New generic helpers in `auto_circuit/metrics/analysis_utils.py` (next to `jaccard`):
`Budget`, `as_budget`, `budget_threshold`, `select_edges`, `realised_edge_count`,
`EdgeIndex`, `edge_index`, `edge_score_vector`, `vector_threshold`, `budget_mask`,
`pairwise_jaccard`, `pairwise_stats`, `mutual_iou`, `selection_frequency`. `top_k_edge_set`
and `overlap_matrix` are refactored onto them, so there is one implementation of each idea;
`overlap_matrix` gains the `exclude_zero` flag it was missing.

`experiments/stability_analysis.py` is the driver, `experiments/stability_plots.py` the
figures. The plots module reads only the CSVs, so `--from <folder>` redraws everything with
no cache, no model and no GPU.

## Running it

```bash
srun python experiments/stability_analysis.py --dry-run                 # cache inventory
srun python experiments/stability_analysis.py --skip-functional        # CPU, ~5 min
sbatch run_stability.sh                                                # full, ~40 min, 1 GPU
srun python experiments/stability_plots.py --from results/stability/<experiment>
```

No `gpu_model` constraint: evaluation need not be bitwise reproducible across nodes, only
discovery did.

## Out of scope

- **Completeness.** Faithfulness only.
- **Sub-experiment C** (Bernoulli sampling from DiscoGP's mask, `edge_prob_diagnostic.py`) —
  a separate instrument with its own run and framing.
- **Decomposing data-draw noise from init noise.** One `--seed` still drives both, which is
  what "re-run the method" means and what issue #5 specifies.
