# Methodology

## Methodology — HPO

### Run 388576: nested-CV HPO collapsed despite C1 fix in place — 2026-05-17 / diagnosed 2026-05-18

- **Context:** `DiscoGP Edge Only Paper` (`answer_margin_ce`, hard STE), IOI · GPT-2 Small · 5 outer × 50 trials · max_epochs_cap=500 · Hyperband (min=10, η=3) · zero ablation.
- **Run:** `hpo/results/hpo_disco_paper_ioi_388576_2026-05-17_16-30-56/`
- **Observation:**
  - Only fold 1 produced a non-degenerate refit (`logit_diff_auc ≈ 1.03`); folds 0/2/3/4 all hit the `eps = 1e-3` floor — refit masks scored below the random baseline across the full edge-count sweep.
  - C1 bug (Hyperband-aborted trials recorded as `COMPLETE`) already fixed in commit `4b3059f`: ~80% of trials correctly marked `PRUNED` (fold 0: 44/53 PRUNED, 9 COMPLETE). Collapse is not the C1 pathology.
  - Two-seed diagnostic (job 389183, seeds 42 and 1042 on fold 0): both collapsed identically, monotonically-decreasing logit-diff (−1.44 → −9.15). Rules out RNG bistability.
  - Default-hparams sanity baseline (job 389375, fold 0): `logit_diff_auc = 92.79`, `roc_auc = 0.741`. Un-tuned algorithm works; HPO is actively selecting destructive hparams.
- **Diagnosis:** Pathological warmup interacts with Hyperband resource axis. Best hparams across folds show `e_n_epoch_warmup ≈ 200–270`; Hyperband rungs at {10, 30, 90, 270, 500}. Rung 0/1 reports happen before sparse loss has ramped in — TPE grades the *initialisation*, not a trained mask, and learns "high warmup ⇒ high score". At refit with `e_train_epochs=500`, peak sparse pressure ≈ `0.2 × 37 = 7.4` collapses the mask. Reproduces across all four failed folds.

| Fold | `e_n_epoch_warmup` | `e_lambda_sparse_init` | `e_max_times` | `e_lr` |
|---|---:|---:|---:|---:|
| 0 | 271 | 0.20 | 37 | 0.037 |
| 1 | 207 | 6.22 | 68 | 0.48 |
| 2 | 51 | 0.13 | 31 | 0.12 |
| 3 | 224 | 0.21 | 12 | 0.016 |
| 4 | 215 | 0.35 | 89 | 0.081 |

- **Significance:** HPO with a large warmup range and Hyperband creates a systematic bias toward degenerate hparams. Search-space fix required before results are meaningful.

### Search-space fix and paper-centred calibration — 2026-05-18

- **Context:** Comparison of implementation defaults against `frankniujc/disco_gp` repo (`cli.py`). Code: `hpo/search_spaces.py`, `auto_circuit/prune_algos/prune_algos.py`.
- **Paper defaults (`cli.py`):** `lr=0.1`, `lambda_sparse_init=1.0`, `lambda_complete_init=1.0`, `max_times=100`, `min_times=0.01`, `gs_temp=1.0`, `logits_init=1.0`, `warmup=20`, `cooldown=20`, `train_epochs=100` (≈20% warmup, 20% cooldown).
- **Search-space changes** (`_suggest_disco_edge_only`):

| Param | Old | New | Paper |
|---|---|---|---|
| `e_lr` | log[1e-3, 0.5] | log[0.03, 0.3] | 0.1 |
| `e_lambda_sparse_init` | log[0.01, 10] | log[0.3, 3.0] | 1.0 |
| `e_lambda_complete_init` | uniform[0, 2] | log[0.3, 3.0] | 1.0 |
| `e_max_times_lambda_sparse` | log[10, 100] | log[30, 300] | 100 |
| `e_min_times_lambda_sparse` | log[0.1, 10] | log[0.001, 0.1] | 0.01 |
| `gs_temp_edge` | log[0.01, 5] | log[0.5, 2.0] | 1.0 |
| `logits_e_init` | uniform[-2, 2] | fixed = 1.0 | 1.0 |
| `e_n_epoch_warmup` | int[20, 400] | int[50, 150] | 20 (≈20% of 100) |
| `e_n_epoch_cooldown` | int[10, 200] | int[50, 150] | 20 (≈20% of 100) |

- **Decisive change:** `e_min_times_lambda_sparse` old range forced lambda above its initial value, disabling the paper's lambda-decay release valve. Narrowed warmup/cooldown range prevents the Hyperband-rung-vs-warmup-epoch mismatch.
- **Registration fix** (`DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO`): warmup 400 → 100, `min_times_lambda_sparse` 1.0 → 0.01. Paper values pinned for reproducible paper-faithful baseline.
- **Smoke test result** (job 389669, fold 0, 500-epoch patched vs 100-epoch defaults):

| Run | `logit_diff_auc` | `roc_auc` |
|---|---|---|
| 100-ep defaults (job 389375) | 92.79 | 0.741 |
| 500-ep patched (job 389669) | 50.83 | 0.591 |

- **Significance:** 500-epoch budget with `max_times=100` applies peak sparse pressure for 300 epochs vs 60 in the paper — longer extreme pressure collapses more edges. The paper's 100-epoch budget is calibrated to the lambda schedule. **Implication for HPO:** set `--max-epochs-cap=100` (Hyperband rungs {10, 30, 100}), or keep 500 epochs but shrink `e_max_times_lambda_sparse` to log[5, 30] so peak-lambda energy matches the paper's 100-epoch budget.

### Refit collapse root cause: `study.best_params` drops structural flags (HPO harness bug, not DiscoGP) — 2026-06-17

- **Context:** Run 400047 (DiscoGP-EP, IOI, 5-fold nested CV). Outer-test `logit_diff_auc` was 0.001–3.0 across folds while the standalone `DISCO_GP_EDGE_ONLY_PAPER` registration reaches ~500% peak logit-diff. Investigated whether the DiscoGP implementation was wrong. Code: `hpo/finalize.py`, `hpo/nested_cv.py`, `hpo/search_spaces.py`.
- **Observation:** The DiscoGP edge-only implementation is **correct**. Inner HPO trials reach `logit_diff_auc` 100–228 (trials 11/21/31/35/39 in fold 0) — higher than the default registration — so search works. The collapse is purely a **refit-path bug**:
  - `study.best_params` contains **only** `trial.suggest_*` params (the 8 tunable floats/ints). The structural flags `use_weight_masks=False`, `use_edge_masks=True`, `hard=True`, `loss_mode` are plain dict entries in `_suggest_disco_edge_only*` and are **not** recorded.
  - `finalize.py` refit called `disco_gp_prune_scores(model, outer_train, true_edges, **best_params, e_train_epochs=100)`, and the function defaults `use_weight_masks=True`. So every refit silently ran **full joint weight+edge pruning** (weight hparams at defaults, `w_n_epoch_warmup=500` with only 100 epochs) → mask collapse.
  - **Reproduction (fold 0, best_params trial 35, seed 42, ZERO ablation):** `**best_params` only → `logit_diff_auc=0.001` (matches the recorded run exactly); add `use_weight_masks=False` → `86.3`; default registration → `186.6`. Run: `hpo/diag_refit_collapse.py` on lenny.
  - Latent same-class bug for ACDC (its `tao_exps`/`tao_bases` are also non-suggested → refit would crash or mis-run) and "DiscoGP Edge Only"/joint.
- **Fix:** Persist the full hparam dict at trial time (`trial.set_user_attr("full_hparams", dict(hparams))` in `nested_cv.objective`) and refit from `best_trial.user_attrs["full_hparams"]` in `finalize.py` (fallback to `best_trial.params` for pre-fix runs). General — covers structural flags **and** derived params for every algo.
- **Significance:** Corrects the earlier (run-388576) story that blamed warmup-vs-Hyperband-rung interaction for refit collapse. For run 400047 the dominant cause was the lost `use_weight_masks=False`. Any Optuna refit that reconstructs from `best_params` must carry non-suggested kwargs explicitly. Old runs (incl. 400047) must be **re-run**, not just re-finalized, since the `full_hparams` attr did not exist when they were recorded.
- **Validation (run 401205, patched, same config):** outer-test `logit_diff_auc` per fold = `141.3 / 188.6 / 164.8 / 199.1 / 0.995` (mean 138.9 ± 71.8) vs run 400047's `0.001 / 0.001 / 3.0 / 1.1 / 2.0` (mean 1.2). Four of five folds recovered to ≥141 — above the default registration's ~50 — confirming the structural-flag drop was the systematic cause. Verified fold 4's stored `full_hparams` carried `use_weight_masks=False`, so the fix is active. The lone fold-4 collapse (inner_val=268.7 on trial 42, refit→0.995) is residual single-run training variance, not the bug — see next entry.

### Fold-4 residual collapse is single-run refit variance, not the harness bug — 2026-06-17

- **Context:** After the structural-flag fix (run 401205), fold 4 still floored at `logit_diff_auc≈0.995` while folds 0–3 recovered to 141–199. Code: `hpo/diag_fold4_variance.py`.
- **Observation:** fold 4's best inner trial (42) scored 268.7 on inner_val with *moderate* hparams (`e_lr=0.12≈default`, `gs_temp=0.96≈1.0`), and the refit provably ran edge-only — so this is not the dropped-flag bug. It is the winner's-curse + single-run instability of DiscoGP's margin-maximizing objective: the "best" inner trial is the max over ~13–50 noisy trials, and finalize refits *once* (seed = `seed+fold_idx`); a fresh seed on the full `outer_train` can land in the all-masked collapse basin.
  - **Multi-seed refit (fold 4 best_params, outer_test, ZERO):** seeds {46, 7, 123, 999} → `logit_diff_auc` = {333.5, 30.5, 250.9, 222.1}, mean ≈209. The recorded finalize value (0.995) is an unlucky draw from this wide distribution. Note seed 46 = finalize's exact refit seed yet gave 333, not 0.995 → DiscoGP training is **not bit-reproducible** even at a fixed seed (CUDA nondeterminism in gumbel sampling + matmuls). Run: `hpo/diag_fold4_variance.py`.
- **Significance:** Outer-test scores for gradient circuit-discovery methods have non-trivial run-to-run variance; a single-run refit is a high-variance point estimate. Mitigation for the thesis HPO: refit best_params at k seeds and report median/best-of-k (or report mean±std over folds and flag collapse folds), rather than treating one refit as ground truth. Relates to the margin-maximizing instability already noted for DiscoGP-EP.

### HPO eval edge-count sweep extended to the full model — 2026-06-17

- **Context:** HPO comparison figures (and the `logit_diff_auc` AUC sweep) were capped at 1000 edges (`DEFAULT_EVAL_EDGE_COUNTS`), truncating the curve well below the model's ~32.5k edges. Code: `hpo/scoring.py` (`_run_sweep`), `hpo/finalize.py` (figure counts).
- **Change:** Extended `DEFAULT_EVAL_EDGE_COUNTS` to `[…, 2000, 5000, 10000, 20000, 30000]` and made both the scoring sweep and the figures **always cap to the task's `n_edges` and append the full-model point**, so curves span `0 → n_edges` for any task. Rebuilt run 401205 figures (cached refits, no retrain): all algos now run 0.5 → 32491.
- **Observation:** Extending the range is still useful, but the "flat / not faithful at intermediate sparsity" reading first drawn here was an artifact of the stale-cache bug below (the figure was plotting May-21 collapsed masks). After fixing that, the fold curves are strongly faithful at intermediate sparsity (see next entry). Net: the AUC selection metric now integrates the whole range (changes trial ranking for future runs — re-run, don't re-finalize, to apply to scoring).

### HPO comparison figures plotted stale circuits: lexical sort of DD-MM-YYYY cache names — 2026-06-17

- **Context:** User saw every DiscoGP-EP HPO fold sitting at ~0% logit-diff in the comparison figures, while standalone DiscoGP-EP and DiscoGP-E reach 275–470% at ~1000 edges — in the *same* figure (so task=IOI, patch=TREE_PATCH, ablation=Zero, metric=Logit-Diff-%, `task.test_loader` are all identical; the gap was the prune-scores, not the eval). Code: `hpo/finalize.py` (`_find_fold_cache`, baseline glob).
- **Root cause:** `_find_fold_cache` (and the baseline loader) did `sorted(glob("fold{i}_prune_scores-*.pkl"))[-1]`. The cache timestamp is `DD-MM-YYYY_HH-MM-SS`, which does **not** sort lexically by date: `"21-05-2026" > "17-06-2026"` as strings (day-of-month dominates). So the figure's cache-hit loaded a **May-21 fold** (a pre-fix joint-pruned collapsed mask) instead of run 401205's June-17 refit. `outer_scores.json` was unaffected (it scores the fresh in-memory refit at refit time), which is why outer-test AUC read 141–199 while the figure read ~0.
  - **Proof:** the same fold0 — 401205 June-17 pkl → `logit_diff_auc[0..1000]=340.8`; stale combined-cache mask → `60.6` (and `0.001` over full range); the two tensors are not equal.
- **Fix:** sort both globs by `mtime` (`max(..., key=lambda p: p.stat().st_mtime)`), not filename. Rebuilt 401205 figures (cache-hit now loads the June-17 folds).
- **Result (Zero panel, test_loader):** DiscoGP-EP folds f0–f3 peak **200 / 395 / 528 / 393%** at ~1000–2000 edges (≥ DiscoGP-E's 276% and ≈ standalone DiscoGP-EP's ~470%); f4 is the known variance-collapse draw (peak 111%). The folds were never flat.
- **Significance:** Two independent bugs masqueraded as "DiscoGP HPO is broken": (1) `best_params` dropping `use_weight_masks=False` (refit ran joint pruning), and (2) this lexical-sort stale-cache load in the *figures*. Both are harness bugs; the DiscoGP edge-only algorithm is faithful. General lesson: never `sorted()[-1]` over `DD-MM-YYYY` cache names.
- **Hardening (done):** Fold refits and the baseline are now written to **per-run paths** — `fold_{i}/prune_scores.pkl` and `<run>/baseline_prune_scores.pkl` — instead of the shared `.prune_scores_cache/` namespace, so caches can never leak across runs. `_find_fold_cache(fold_dir, fold_idx)` prefers the per-run file and only falls back to the legacy global glob (newest-by-mtime) for pre-existing runs. `nested_cv.py` writes the per-run baseline; verified per-run-preferred + legacy-fallback resolution.

## Methodology — Patching & Ablation Mechanics

### How factorized edge patching works, and how zero vs. resample change its meaning — 2026-06-01

- **Context:** Tracing the AutoCircuit patching path to understand what a "circuit edge" computes under tree patch. Code: `auto_circuit/utils/patch_wrapper.py` (`PatchWrapperImpl.forward`), `auto_circuit/utils/graph_utils.py` (`graph_edges`, `patch_mode`), `auto_circuit/utils/ablation_activations.py` (`src_ablations`), `auto_circuit/prune.py` (`run_circuits`).

- **Observation — factorized graph = residual-stream decomposition:** With `factorized=True` an edge exists for *every* `(src, dest)` with `src.layer < dest.layer` (`graph_edges`, the `s.layer < d.layer` filter). There is no literal wire (e.g. `A0.9 → A11.0.Q`); the edge is an *attribution*. Because every component writes to the shared residual stream **additively** and every component reads the whole stream, an upstream node's output is a separable summand inside every downstream node's input. AutoCircuit stores each source output separately (`curr_src_outs`) so it can add/remove that one summand from one reader's input independently.

- **Observation — the per-edge update:** At each dest, `arg_0 += mask · (patch_src − curr_src)` per incoming edge. `mask=0` → keep the live (clean) contribution; `mask=1` → swap it for the ablation value. `curr_src` is the *live* output on this pass, so cutting an edge into an intermediate node changes that node's output and the effect **cascades** to all its downstream kept edges — this is what makes it tree/path patching rather than edge-local activation patching.

- **Observation — zero ablation:** `patch_src = 0` (`src_ablations` ZERO branch returns `zeros_like`), so a cut edge *subtracts* the writer's contribution out of the reader's input. A node with **all** inputs cut receives exactly the zero residual → LayerNorm returns its bias (`β`), Q/K/V become position-independent constants → the node emits a **fixed, input-independent constant** (bias-driven), not random noise. Through an `edge → resid_final`, that is a constant logit offset identical across all prompts. The corrupt input is irrelevant under zero ablation (run but discarded).

- **Observation — resample ablation:** `patch_src` = source activations on `batch.corrupt`, so a cut edge **substitutes** clean→corrupt for that (writer, reader) pair. A node with all inputs cut receives the **full corrupt residual stream** (clean terms cancel, corrupt terms sum to `resid_corrupt`) → it computes its normal, coherent output **for the corrupt prompt** — prompt-dependent, correlated with the corrupt answer, never a bias constant. The off-circuit background is therefore "the model running on corrupt," and the circuit is the set of edges that route the clean-vs-corrupt *difference*. This keeps activations on-distribution (zero residual is never seen in training; corrupt residual is in-distribution), but makes results depend on the specific clean/corrupt pairing.

- **Observation — patch direction (AutoCircuit `PatchType`):** `TREE_PATCH` feeds `batch.clean`, ablates the **complement** (mask = sub-threshold edges), keeps circuit edges clean → sufficiency / "run the circuit in isolation"; reported edge count is the kept edges `(1−mask).sum()`. `EDGE_PATCH` feeds `batch.corrupt`, ablates the **circuit** edges → necessity. These two are exactly Miller et al.'s "direction" axis (ablate circuit vs. complement; clean vs. corrupt input).

- **Significance:** (i) A discovered circuit is only faithful if its kept edges form unbroken paths from the embeddings to `resid_final`; a dangling in-circuit edge contributes only ablation-artifact signal (zero: a constant; resample: corrupt-prompt info), never prompt-relevant computation. (ii) EP (resample) and DiscoGP (zero) are tested against **different baselines** and solve different causal queries, so their faithfulness numbers are not directly comparable — this is the concrete mechanism behind Miller et al.'s "faithfulness is methodology-relative." Every cross-algorithm comparison table must state the ablation type. See [`faithfulness_metrics_not_robust.md`](docs/related_work/faithfulness_metrics_not_robust.md) (six-dimension ablation framework).
