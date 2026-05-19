# Thesis Notes

## Results

### DiscoGP edge-only variants: KL collapse despite strong logit-difference (IOI, zero ablation) — 2026-05-15

- **Context:** IOI component circuit, GPT-2 Small, zero ablation for training and evaluation. Algorithms: `DiscoGP-E` (edge-only, `argmax_ce`, no completeness), `DiscoGP-EP` (edge-only, `answer_margin_ce`, completeness on), `DiscoGP` (joint weight+edge), EAP, Random, Ground Truth.
- **Run:** `figures/circuit_discovery/2026-05-15_17-57-22/`
- **Observation:**
  - Logit-difference percent ([figure_15.pdf](figures/circuit_discovery/2026-05-15_17-57-22/figure_15.pdf)): DiscoGP-E and DiscoGP-EP perform competitively across the edge-count sweep — recover most of the IOI margin at moderate sparsities, comparable to EAP.
  - Clean KL divergence ([figure_12.pdf](figures/circuit_discovery/2026-05-15_17-57-22/figure_12.pdf)): same two algorithms track the Random baseline closely across nearly the entire edge-count range; EAP/Ground Truth sit well below.
- **Significance:** A circuit that preserves the answer-margin but is essentially random on full-vocabulary divergence is exactly what the algorithm's objective predicts. Root-cause analysis in the Discussion section.

## Discussion

### Why edge-only DiscoGP collapses on KL — 2026-05-15

- **Context:** Follow-up analysis of the 2026-05-15 IOI result. Code: `auto_circuit/prune_algos/disco_gp.py`.
- **Observation:** Four compounding causes:
  - **Faithfulness loss only constrains the answer token.** `argmax_ce` (`_faith_loss`, lines 455–465): CE against one-hot argmax — zero gradient for the other ~50,256 vocab logits when top-1 token is preserved. `answer_margin_ce` (`_faith_loss_margin`, lines 486–498): two-class CE on (correct, wrong) margin only. `CLEAN_KL_DIV_METRIC` is KL over the full distribution (`kl_div.py` lines 46–48). Training objective never sees the evaluated quantity.
  - **Sparsity schedule prunes distribution-faithful edges.** Both edge-only variants use `e_max_times_lambda_sparse=100` with a 400-epoch warmup. During warmup, edges contributing spread mass across vocab get zero faith-loss gradient and positive sparsity-loss gradient — pruned without resistance. KL inflates; logit-diff is unchanged.
  - **Zero ablation amplifies the KL gap.** Zero ablation replaces dropped contributions with literal `0`, not in-distribution activations. The set of edges needed to preserve full-vocab KL is much larger under zero ablation than under resample/mean; the set needed for logit-diff is roughly unchanged. DiscoGP retains the latter, discards the former.
  - **Completeness term (DiscoGP-EP) does not fix KL.** `_complete_loss_margin` (lines 501–512) pushes the complement mask toward uniform 0.5/0.5 on the (correct, wrong) pair only — says nothing about the other ~50k vocab entries. EP is not measurably better than E on KL.
- **Significance:** This is not an implementation bug — it is the predicted behaviour of an answer-token-only objective.

### Random-like tail behaviour at high edge counts — 2026-05-15

- **Context:** `run_circuits` ranks all ~32k edges by score and sweeps a threshold (`prune.py` lines 84–95).
- **Observation:**
  - DiscoGP returns `sigmoid(logit)` from trained mask logits (`disco_gp.py` lines 443–446). Logits initialised at `logits_e_init=1.0 + 0.01 * randn` (line 305).
  - After training: edges the optimiser "cared about" sit at sigmoid ≈ 1; the remaining tens of thousands sit in a low blob whose ordering is dominated by init noise and AdamW jitter.
  - At small edge budgets: trained subset selected → good logit-diff, mediocre KL. At large budgets: random tail fills the budget → curve hugs `Random` baseline. Exactly the shape in [figure_12.pdf](figures/circuit_discovery/2026-05-15_17-57-22/figure_12.pdf).
- **Significance:** The plateau at high edge counts is a direct consequence of the objective, not noise or implementation error.

### Framing for the thesis — 2026-05-15

- **Observation:** DiscoGP optimises a task-success objective (argmax or answer-margin CE). Its circuits are tight along the task axis (logit-diff) and uninformative along the distribution axis (full KL). Algorithms whose objective tracks the full distribution — ACDC's KL/MSE target, EAP's logit gradients, Edge Pruning's L0 over `kl_div` — preserve KL because they were built to.
- **Significance:** The KL-vs-logit-diff gap is a clean empirical lever for one of the sub-RQs: which notion of *faithfulness* the user cares about determines which algorithm class is appropriate. DiscoGP-EP underperforming on KL is by design and acceptable when only behavioural reproduction on the task is needed.

### DiscoGP-EP answer probability collapses despite strong logit-diff — 2026-05-19

- **Context:** IOI, zero ablation. Figures from `figures/circuit_discovery/2026-05-15_17-57-22/`.
- **Observation:** DiscoGP-EP scores comparably to DiscoGP-E on logit-difference percent, but near zero on answer probability (indistinguishable from Random). DiscoGP-E scores high on both.
- **Mechanism:** `answer_margin_ce` is 2-class CE over `(logit_correct, logit_wrong)`. It constrains the *relative* ordering of those two tokens but not absolute scale. The circuit can satisfy `logit_IO > logit_S` while other tokens have higher logits still, giving near-zero full-vocab softmax probability on IO. `argmax_ce` is `-log P(argmax_token)` in the full-vocab softmax — to minimise it the circuit must suppress all other tokens, so high answer probability and good logit-diff both follow naturally.
- **Significance:** The paper-faithful loss (-EP) is strictly worse than argmax loss (-E) on two of three metrics (answer probability and KL), while only outperforming on logit-diff at some circuit sizes. The loss choice determines which faithfulness axes the circuit is trained to satisfy — a clean axis for the sub-RQ on loss-form comparison.

### Suggested follow-up experiments — 2026-05-15

- **Score-distribution histogram.** Plot `sigmoid(logit)` over all edges for each algorithm. Expect DiscoGP-E/EP to be bimodal (spike near 1, blob near 0); EAP/GT to be approximately continuous. The blob is the random tail responsible for the KL plateau.
- **Train-time KL probe.** At convergence of edge training, evaluate KL on the train set with Gumbel-sampled masks. Expectation: already at `Random` level, confirming the loss never minimised it.
- **Loss-swap ablation.** Re-run DiscoGP-E with `F.kl_div(log_softmax(train_logits), log_softmax(clean_logits.detach()))` as the faith loss. Expectation: KL curve jumps to ACDC/EAP level; logit-diff drops slightly. Confirms objective mismatch as root cause.
- **Sparsity decoupling.** Re-run with `e_max_times_lambda_sparse=1` (no ramp). If KL improves materially, the sparsity ramp contributes independently of the loss form.
- **Ablation-type cross-evaluation.** Train under zero, evaluate under resample/mean. If the KL gap narrows but logit-diff is unchanged, the zero-ablation amplification is confirmed.

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

## Experiments

### Benchmark task definitions and metric mapping — 2026-05-19

- **Context:** Reviewing the three benchmark papers (IOI: Wang et al. 2022, Greater-Than: Hanna et al. 2023, Docstring: Heimersheim & Janiak 2023 blog / Conmy et al. 2023 ACDC paper) to choose appropriate evaluation metrics in AutoCircuit.

#### IOI (Wang et al., 2022)
- **Task:** Predict IO name (appears once) over S name (appears twice) in sentences like *"When Mary and John went to the store, John gave a drink to"*.
- **Answer tokens:** 1 correct (IO), 1 wrong (S) — symmetric binary choice.
- **Paper metric:** Logit difference `logit(IO) − logit(S)`, reported as % of full model (87% recovery). Formula: difference of means (`mean(circuit diff) / mean(full model diff)`).
- **AutoCircuit mapping:** `LOGIT_DIFF_PERCENT_METRIC` (`diff_of_means=True`) — exact match.

#### Greater-Than (Hanna et al., 2023)
- **Task:** Predict a valid two-digit year > YY given *"The [noun] lasted from the year XXYY to the year XX"*.
- **Answer tokens:** ~36 correct (years > YY), ~64 wrong (years ≤ YY) — count varies per example, uses list branch in `batch_answer_diffs`.
- **Paper metric:** `ΣP(valid years) − ΣP(invalid years)` (probability difference, range [−1, 1]); circuit recovers 89.5%.
- **AutoCircuit mapping:** `LOGIT_DIFF_PERCENT_METRIC` — logit-space equivalent, comparable across algorithms. Also report `PROB_DIFF_METRIC` to anchor to paper's numbers.
- **Do not use:** `CORRECT_ANSWER_PERCENT_METRIC` — argmax accuracy is too coarse; GT is a distributional task over ~36 valid tokens.

#### Docstring (Heimersheim & Janiak 2023; used in Conmy et al., 2023)
- **Task:** Predict the next `:param` name in a Python docstring, given the function signature defines the ordering (induction task). Model: `Attn_Only_4L512W_C4_Code` (4-layer attention-only, **not GPT-2**).
- **Answer tokens:** 1 correct (next param in order), ~5 wrong (other param names from corrupt prompt).
- **Paper metric:** ACDC paper reports logit difference but notes it is **unstable** for this task (can be negative at intermediate circuit sizes); recommends KL divergence instead.
- **AutoCircuit mapping:** Primary = `CLEAN_KL_DIV_METRIC` (as recommended by ACDC paper). Secondary = `CORRECT_ANSWER_PERCENT_METRIC` (argmax accuracy — meaningful here since there is exactly 1 correct next token). `LOGIT_DIFF_PERCENT_METRIC` is noisy because wrong_answers are corrupt-prompt params, not a semantically clean foil.

#### Summary table

| Task | Primary metric | Secondary | Avoid |
|---|---|---|---|
| IOI | `LOGIT_DIFF_PERCENT_METRIC` | `CLEAN_KL_DIV_METRIC` | — |
| Greater-Than | `LOGIT_DIFF_PERCENT_METRIC` | `CLEAN_KL_DIV_METRIC`, `PROB_DIFF_METRIC` | `CORRECT_ANSWER_PERCENT_METRIC` |
| Docstring | `CLEAN_KL_DIV_METRIC` | `CORRECT_ANSWER_PERCENT_METRIC` | `LOGIT_DIFF_PERCENT_METRIC` (unstable) |

- **Significance:** The choice of primary metric is task-dependent and grounded in each paper's own evaluation. Using `LOGIT_DIFF_PERCENT_METRIC` uniformly across tasks would misrepresent Docstring results. The `answer_margin_ce` DiscoGP loss (optimises mean logit diff on correct/wrong pairs) aligns well with IOI and GT but is misaligned with Docstring's KL-preferred criterion.

## Notes

