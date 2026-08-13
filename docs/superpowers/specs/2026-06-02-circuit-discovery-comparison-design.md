# Experimental Design — Comparing Circuit Discovery Methods (sub-RQ3)

**Date:** 2026-06-02 (updated 2026-06-05)
**Status:** Design finalized; foundation (#0) implementation pending.
**Scope:** Defines the experimental core of sub-RQ3. Sub-RQ1 (EP/DiscoGP integration) and sub-RQ2 (Murano) are out of scope here except where a result doubles as a correctness check.

> **Corrections since drafting (2026-06 onward):**
> - **RQ numbering is outdated here.** Murano was dropped (supervisor decision). The final structure is one primary RQ + three sub-RQs: Sub-RQ 1 = Integration, **Sub-RQ 2 = Faithfulness**, **Sub-RQ 3 = Reliability** (stability + null test). Where this doc says "sub-RQ3", read it as the *Reliability* sub-RQ; "sub-RQ2 (Murano)" no longer exists.
> - **Null-test overclaim (§1, below).** "The null test has never been run on circuit-discovery algorithms" is too strong: Méloux et al. *prescribe* it but run it empirically on probing only, and a prior circuit-level hypothesis test exists (Shi et al. 2024). The defensible gap is *instantiating* the prescribed randomized-weight test for the gradient-based optimizers EP/DiscoGP (vs ACDC/EAP) in the unified harness. See `thesis_notes/introduction.md` (2026-06-19 dead-salmon entry) and [[subrq3-experiment-design]].

---

## 1. Motivation & the gap

A plain faithfulness/completeness/sparsity comparison of EP, DiscoGP, ACDC, and EAP would be a *descriptive benchmark* — exactly the kind of single-run comparison the 2025 statistical-rigor papers (Méloux et al.) argue is scientifically weak. The contribution is to reframe sub-RQ3 from *"which method is most faithful?"* to a hypothesis-driven study with three axes, two of which sit on a **verified, currently-unoccupied gap**:

- **Stability of optimization-based methods is untested.** The EAP-IG variance paper (arXiv 2510.00845 v4) limits its analysis to the EAP family and *explicitly names* characterizing the stability of other methods (incl. ACDC) as "future work." No optimization/pruning method (EP, DiscoGP, SP) has had its structural variance measured.
- **The null ("dead salmon") test has never been run on circuit-discovery algorithms.** Both Méloux papers run null controls only on probing/attribution/world-model experiments. The strongest control used for ACDC/EAP/EP/DiscoGP in the literature is a random-*edge* baseline on the *real* model — far weaker than discovery on randomly-*initialized weights*.

Gap verified 2026-06-02 against: EAP-IG variance (2510.00845v4), Dead Salmon (2512.18792), Certified Circuits (2602.22968 — vision/feature-channel, no overlap), DiscoGP (EMNLP 2025 — reports only faithfulness/sparsity), Have Faith in Faithfulness (2403.17806 — adjacent framing, no collision).

## 2. Research question

> How do optimization-based methods (EP, DiscoGP) compare to attribution/greedy methods (EAP, ACDC) along three axes — faithfulness under varied methodology, structural stability, and whether they recover genuine structure at all — and what do their (dis)agreements reveal about the non-identifiability of circuit discovery?

## 3. Common setup

- **Library:** AutoCircuit (this fork), edge-level factorized graphs, cached `PruneScores` per discovery run.
- **Methods:** EP, DiscoGP (registered variants), ACDC, EAP. Anchors: `RANDOM_PRUNE_ALGO`, `GROUND_TRUTH_PRUNE_ALGO`.
- **Tasks / models:** IOI, Greater-Than (GPT-2 Small); Docstring (4L attention-only `Attn_Only_4L512W_C4_Code`).
- **Task → metric mapping** (from `thesis_notes/experiments.md`, grounded in each task's origin paper):
  | Task | Primary | Secondary | Avoid |
  |---|---|---|---|
  | IOI | `LOGIT_DIFF_PERCENT_METRIC` | `CLEAN_KL_DIV_METRIC` | — |
  | Greater-Than | `LOGIT_DIFF_PERCENT_METRIC` | `CLEAN_KL_DIV_METRIC`, `PROB_DIFF_METRIC` | `CORRECT_ANSWER_PERCENT_METRIC` |
  | Docstring | `CLEAN_KL_DIV_METRIC` | `CORRECT_ANSWER_PERCENT_METRIC` | `LOGIT_DIFF_PERCENT_METRIC` (unstable) |
- **Sweep:** edge-count grid via `edge_counts_util` (the same log-grid `measure_prune_metrics` uses), so circuit-graph sliders, ROC, and metric curves line up tick-for-tick.

### Caching boundary (the cost model)

- **Discovery axis = expensive.** One training run per cell → one cached `PruneScores`. EP/DiscoGP/ACDC slow; EAP fast.
- **Evaluation axis = free.** Re-scoring a cached `PruneScores` under any ablation × patch-direction reuses the cache.
- Consequence: stability and the null test are **discovery-axis** experiments — each needs *new* training runs and produces *new* caches; neither can be reconstructed from the spine's caches. Analysis/plotting can be deferred, the runs cannot.

### Operating model (run-per-setting, decided 2026-06-02)

No monolithic orchestrator. Each experiment cell is run **one at a time** with its own settings, and comparison happens **after the fact**:

- **Immediate pre-step (no plumbing required):** run each method at its *native* discovery ablation (EP=resample, DiscoGP=zero, EAP=corrupt, ACDC=internal default) on all 3 tasks using the current `circuit_discovery.py` with just config changes (add all algos/tasks to the lists, add `TOKENWISE_MEAN_CORRUPT` to eval ablation types). This covers the **evaluation-relativity** result (same circuit, shifting eval methodology) and validates the pipeline end-to-end. No new code needed.
- **Discovery unit (post-plumbing):** one job = one `(method, task, discovery-ablation, seed)` → one cached `PruneScores`, tagged with its settings + `git_state()` in the manifest. **Entry point:** a small CLI/script (e.g. `experiments/run_cell.py`) that accepts `--method`, `--task`, `--ablation`, `--seed` and runs exactly one cell. SLURM-array friendly (one cell per array task).
- **Analysis pass:** a single post-hoc step loads all caches and produces the eval matrix, ROC overlay, stability/Jaccard, overlap heatmaps, and null p-values. Cheap; wants to see all circuits together. Lives in `auto_circuit/metrics/analysis_utils.py` (see §3 below).
- **What the foundation (#0) still must build:** the discovery-ablation + seed **parameters** (they don't yet exist — see code realities below), cache keys encoding `(ablation, seed)` so runs don't overwrite each other, the per-cell CLI entry point, and the analysis utils. What it removes is only the automated sweep loop.

### Code realities that shape the foundation (verified 2026-06-02)

- **Discovery ablation is hard-coded, not a parameter.** `edge_pruning.py:165` (`RESAMPLE`), `disco_gp.py:294` (`ZERO`); `PruneAlgo.func` is `(model, dataloader, official_edges) → PruneScores` with no ablation arg. Fix: add `ablation_type: AblationType` param to both algo functions; expose at call site via `functools.partial` to produce ablation-tagged `PruneAlgo` variants with unique `key`/`name` (e.g. `"Edge Pruning [zero]"`), keeping the native variants as defaults.
- **EAP/ACDC decision (resolved 2026-06-05):** **EAP stays corrupt-only** — extending it would require threading `src_ablations(...)` through `edge_attribution_patching.py:89–108`, adding noise for unclear gain. **[SUPERSEDED 2026-07-16 — see §12: the premise was wrong; no threading was needed because the test-verified equivalent `mask_gradient_prune_scores` already exposes `ablation_type`.]** **ACDC gets `ablation_type` threaded** similarly to EP/DiscoGP. The spine's discovery axis is effectively *EP/DiscoGP/ACDC × {zero, resample, mean}* **+ EAP × {corrupt}**, with GT/Random ablation-independent (≈30 discovery cells).
- **`Task` has no seed field**, and `DATASET_CACHE` (`tasks.py:192`) is keyed as `(dataset_name, batch_size, batch_count, token_circuit, device)` — excludes seed, so a new seed silently returns the cached loader. Fix: add seed to `DATASET_CACHE` key + `load_datasets_from_json(random_seed=seed)` call; also `torch.manual_seed(seed)` before EP/DiscoGP discovery (masks are randomly init'd). The seed axis is genuinely new plumbing.
- **Null-test model hook:** `Task._model_def` accepts `nn.Module` (`tasks.py:149`) but that branch **skips** the cfg flags set at lines 173-177 (`use_attn_result`, `use_attn_in`, `use_split_qkv_input`, `use_hook_mlp_in`) and skips `eval()`. Set these manually on the randomly-initialized model before passing it.
- **Cache identity:** encode `(discovery_ablation, seed)` into the cache filename (or algo key) so cells don't overwrite each other. The current timestamp-based naming in `save_cache` already prevents collisions in practice; the structured approach lets the analysis pass look up cells by identity rather than by timestamp.

### Analysis layer (`auto_circuit/metrics/analysis_utils.py`)

Post-hoc module used exclusively by the analysis pass. Functions needed (verified against issue #3):

- `top_k_edge_set(prune_scores, k)` — reuses `.abs()` ranking via `desc_prune_scores`/`prune_scores_threshold` (`tensor_ops.py`), exactly as `run_circuits`/`prune.py` do. Returns a `Set[Edge]`.
- `jaccard(a: Set[Edge], b: Set[Edge]) → float` — `|a ∩ b| / |a ∪ b|`.
- `overlap_matrix(runs: List[PruneScores], k: int) → np.ndarray` — pairwise Jaccard across N runs at fixed budget k.
- `matched_budgets(model, true_edge_count) → List[int]` — wraps `edge_counts_util` to return the representative grid points used for comparison.
- `load_many_caches(paths: List[Path]) → List[Tuple[meta, TaskPruneScores]]` — loads separately-saved caches and returns them with their manifest metadata for cross-run aggregation.
- `extract_metric(measurements: AblationMeasurements, ablation, metric_key, task_key, algo_key) → List[Tuple[int, float]]` — extracts the `(edge_count, score)` series from `measure_prune_metrics`'s nested return type.

## 4. Pillar 1 — Faithfulness spine (full discovery × eval matrix)

- **Discovery axis:** train each method under each ablation ∈ {zero, resample, mean}. Cell count is **not** a clean 4×3×3: EP/DiscoGP/ACDC sweep all three ablations (9 cells/task), **EAP is corrupt-only** (1 cell/task, decision locked in §3), GT/Random are ablation-independent (1 each). So ≈ (3 methods × 3 ablations + EAP × 1) × 3 tasks ≈ **30 discovery cells** + anchors. **[SUPERSEDED 2026-07-16 — see §12: EAP is swept too; realized matrix = 4 methods × 3 ablations × 3 tasks + GT/Random anchors = 42 cells.]**
- **Evaluation axis (cached, free):** score every circuit across {zero, resample, mean} × {tree-patch, edge-patch}.
- **Metrics:** task-primary/secondary above + completeness + sparsity, plus ROC-vs-ground-truth across the sweep.
- **Headline results:**
  - *Evaluation-relativity (Miller):* the same circuit's faithfulness shifting across evaluation methodologies.
  - *Discovery-relativity (Miller + Dead Salmon):* circuits *discovered* under different ablations diverging.
- **Note:** EP (native resample) and DiscoGP (native zero) are tested against different baselines by construction; every cross-algorithm table must state the (discovery-ablation, eval-ablation, patch-direction) triple.

## 5. Pillar 2 — Stability (attribution vs optimization vs greedy)

- **Breadth:** all 3 tasks, all 4 methods, **both** shared ablations {zero, resample}.
- **Repeats:** **N = 5** discovery runs per cell (dial up if wall-clock allows).
- **Perturbation sources:**
  - *Bootstrap-resampled dataset* — applies to all methods (the only variance source for EAP and ACDC, which are ~deterministic given data).
  - *Training seed / mask init* — additional variance source for EP and DiscoGP.
- **Metric:** pairwise **Jaccard** of edge sets at matched sparsity across the N runs → mean Jaccard + CV per (method, task, ablation). Directly comparable to Méloux's EAP-IG ≈ 0.56 (CV 0.335).
- **Hypothesis under test:** optimization methods (EP, DiscoGP), which train masks over epochs, yield higher Jaccard / lower CV than attribution (EAP) and greedy (ACDC).
- **Matched-sparsity definition:** compare edge sets at the same edge-count grid point(s); report a small set of representative budgets rather than a single one.

### Free byproduct — cross-method overlap matrix (Pillar 3a)

Same Jaccard machinery, applied *across methods* at matched sparsity, per task → are method disagreements structured (consistent head/layer preferences) or random? Frame with *Have Faith in Faithfulness* (2403.17806): overlap is reported *alongside* faithfulness, never instead of it.

## 6. Pillar 3 — Null test ("dead salmon")

- **Setup:** run all 4 methods on **randomly-initialized GPT-2**, all 3 tasks, **k = 10** independent re-initializations. Single shared ablation (proposed: **resample**, in-distribution) to bound cost; revisit if cheap.
- **Statistic & p-value:** for an explanatory-fit statistic `T` (e.g. faithfulness / negative KL of the discovered circuit at fixed sparsity), Monte Carlo p-value
  `p̂ = (1 + Σ_b 1{T_null^(b) ≥ T_obs}) / (k + 1)`,
  where `T_obs` is on the trained model and `T_null` on random re-inits.
- **Reading:** if a method reports low-KL / faithful-looking circuits on random weights (small effect vs. null), that is a dead-salmon result. Doubles as a sub-RQ1 correctness check — a pipeline that "finds circuits" in noise is one to distrust.

## 7. Phasing & run-count budget

Each "run" below is a **standalone job** (run-per-setting operating model); the analysis pass is separate and cheap.

- **Phase A — spine:** ~30 discovery jobs (see §4 cell count) + anchors; all evaluation free via cache.
- **Phase B — stability:** 4 × 3 × 2 × N = **120 jobs** at N=5.
- **Phase B — null:** 4 × 3 × k = **120 jobs** at k=10 (single ablation).
- **Total ≈ 270 discovery jobs** (excl. free eval). EAP cheap; EP/DiscoGP epoch-bound; **ACDC is the bottleneck** (~60 of these are greedy ACDC).

## 8. Risks & contingencies

- **ACDC wall-clock (primary risk).** ~60 ACDC runs in Phase B may dominate the budget. *Contingency:* time one ACDC run per task first; if a single run is too costly to repeat N× / k×, demote ACDC to anchor-only in Phase B and frame as a noted scoping decision (kept in the spine regardless).
- **DiscoGP paper-loss OOM on 40 GB nodes** — exclude bob/minerva; add `--constraint="gpu_mem:80gb"`.
- **Null-test interpretation.** Random-weight LayerNorm/activation statistics differ from trained; ensure `T` is computed identically (same sparsity, same ablation, same metric) on trained vs null so the comparison is apples-to-apples.
- **Matched-sparsity ambiguity for Jaccard.** Fix the representative edge-budget grid points up front; report Jaccard as a function of budget, not a single number.
- **Single shared ablation for the null test** is a scoping choice; note it explicitly.

## 9. Deliverables

- Faithfulness curves + tables across the discovery × eval matrix (per task, with the methodology triple stated).
- ROC-vs-ground-truth across the sweep.
- Stability table: mean Jaccard + CV per (method, task, ablation), with EP/DiscoGP vs EAP/ACDC contrast and side-by-side with Méloux's EAP-IG numbers.
- Cross-method overlap heatmaps per task.
- Dead-salmon Monte Carlo p-values per (method, task), with effect sizes vs the null.

## 10. Decisions locked in brainstorming

1. Core = faithfulness spine + stability + (overlap matrix + null test); hydra/overdetermination probe deferred.
2. Spine = **full discovery × eval** matrix (3 discovery ablations).
3. Stability/null slice = **all 3 tasks**, **both** ablations for stability, **ACDC kept** in Phase B (subject to the timing contingency above).
4. Stability comparison anchored to Méloux via resample-ablation Jaccard.
5. **Operating model = run-per-setting** (one job per `(method, task, ablation, seed)` cell, SLURM-array friendly) + a single post-hoc analysis pass; no monolithic orchestrator (see §3).

## 11. Open items for spec review

- Confirm N=5 / k=10 (vs higher) once a per-task ACDC timing result is available (see §8 ACDC wall-clock risk).
- Confirm `T` statistic for the null test: faithfulness at a fixed representative sparsity (simpler, easier to report) vs AUC over the full sweep (more robust but needs all k×sweep combinations).
- Confirm whether the null test stays single-ablation (resample) or also runs zero ablation.
- Confirm `functools.partial` (no signature change to `PruneAlgo.func`) vs a thin `PruneAlgoAblation` wrapper class for ablation-aware variants. Partial is simpler; a wrapper lets you carry metadata (ablation name, native vs swept) without monkey-patching.

**Resolved (2026-06-05):**
- EAP extension: ~~closed — EAP stays corrupt-only~~ **reopened and reversed 2026-07-16 (§12)** — EAP is swept over {zero, resample, mean} via `eap_prune_scores`; ACDC gets ablation_type threaded (see §3 code realities).

## 12. Amendment (2026-07-16): EAP joins the discovery sweep

The §3 decision "EAP stays corrupt-only" rested on a false premise. Rereading Syed et
al. (2023) and the AutoCircuit code showed:

- Resample ablation is part of EAP's *definition* in the paper (the score linearizes
  corrupted-prompt patching), but the Taylor machinery is baseline-agnostic: any
  ablation baseline yields the corresponding first-order estimate (precedent: Marks
  et al. 2024 use a zero baseline when counterfactual pairs don't exist).
- No `src_ablations` threading was ever required: `mask_gradient_prune_scores`
  (test-verified exactly equal to the reference EAP at resample) has exposed
  `ablation_type` all along. Only the slow reference implementation
  (`edge_attribution_patching.py`) hardcodes the corrupt cache.
- Smoke test (2026-07-16, Docstring + IOI): zero/mean EAP run correctly; resample
  reproduces the registered EAP exactly (corr 1.000000); zero reorders the top-100
  ranking heavily (overlap 0.25/0.30), mean moderately (0.85/0.58).

**Changes:** `EDGE_ATTR_PATCH_PRUNE_ALGO.func` → `eap_prune_scores` (mask-gradient
wrapper deriving `clean_corrupt` from the ablation type); EAP moved from
`ANCHOR_ALGOS` to `SWEPT_ALGOS` in `faithfulness_cells.py`; spine = **42 cells**
(4 swept × 3 ablations × 3 tasks + GT/Random anchors ×3 tasks). Old EAP resample
caches remain valid (scores identical) and are reused via `--skip-if-exists`.
EAP was never "ablation-independent": it was ablation-*bound* (to resample) by the
registered implementation — the opposite property. GT/Random are the only genuine
ablation-independent anchors.

## 13. Erratum (2026-07-31): completeness ran the denoising direction

§4 specifies the evaluation axis as `{zero, resample, mean} x {tree-patch, edge-patch}`
but does not fix which prompt each patch direction runs on. `run_circuits` pairs
`EDGE_PATCH` with the corrupt prompt by default (a denoising/sufficiency test), and
`faithfulness_comparison.py` did not pass `reverse_clean_corrupt`, so every completeness
number produced up to and including the `faithfulness_seed42_2026-07-27` results tree is
a denoising measurement, not the knockout stated in the code comment.

Diagnostic: completeness KL started at `KL(clean||corrupt)` (3.625 on IOI) at zero edges
and fell to 0 at the full edge count. A knockout must start at exactly 0 and rise.

Fixed by passing `reverse_clean_corrupt=(patch_type == PatchType.EDGE_PATCH)`.
Faithfulness, ROC, circuit graphs, and the similarity heatmaps are unaffected: they never
used `EDGE_PATCH`. Discovery is unaffected: it does not call `run_circuits`. The fix also
makes the zero and mean eval-ablation columns of the completeness panels interpretable,
since the ablation now chooses how the circuit is destroyed rather than what is written
into it.
