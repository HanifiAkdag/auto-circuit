# Discussion

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

### Why DiscoGP-EP has high logit-diff AND high answer-pair KL (margin-maximize vs margin-match) — 2026-06-10

- **Context:** Apparent paradox — DiscoGP-EP achieves very high logit-difference but *also* high answer-pair KL vs. EAP/EP baselines, even though both metrics read the same `(logit_correct, logit_wrong)` pair. Code: [answer_diff_percent.py:113](auto_circuit/metrics/prune_metrics/answer_diff_percent.py#L113), [disco_gp.py:498](auto_circuit/prune_algos/disco_gp.py#L498), [kl_div.py:14](auto_circuit/metrics/prune_metrics/kl_div.py#L14).
- **Observation:** The two metrics are *not* similar — they reward opposite things in the overshoot regime. Both are functions of the margin `d = logit_correct − logit_wrong` (the 2-class softmax depends only on `d`), but:
  - **Logit-diff percent** `= 100·mean(d_circuit)/mean(d_full)`: linear, **unbounded, monotonic increasing** in `d_circuit`. No penalty for `d_circuit > d_full`; can read >100%.
  - **Answer-pair KL** `= KL(σ(d_full) ‖ σ(d_circuit))`: convex, **minimized only at `d_circuit = d_full`, penalizes deviation in both directions** — overshoot is punished just like undershoot.
  - So the metrics agree only while `d_circuit < d_full`; once the circuit margin *exceeds* the full model's, logit-diff keeps rising (good) while KL rises (bad).
- **Mechanism (why DiscoGP-EP overshoots):** its faith loss is `softplus(−d_circuit)` (2-class CE with target=correct), **monotonically decreasing in `d_circuit`** — a margin-*maximizer* with no anchor at `d_full`. It drives `d_circuit ≫ d_full`. Mechanistically this means shedding the full model's calibration: the negative name-mover / copy-suppression heads (10.7, 11.10) *subtract* from the IO logit to keep `d_full` moderate; a margin-maximizing circuit under-weights them and becomes *more confident than the full model itself*. KL flags this miscalibration; logit-diff structurally cannot see it. (Ties to [[negative_heads_score_sign]] and the 2026-05-28 head-heatmap entry.)
  - Worked example, `d_full = 3`: overshoot `d_c=8` → 267% logit-diff but KL=0.186; match `d_c=3` → 100% logit-diff, KL=0; undershoot `d_c=1.5` → 50% logit-diff, KL=0.082. Overshoot gives *worse* KL than undershoot despite far higher logit-diff.
- **Significance:** The high-logit-diff/high-KL combination is the **signature of a margin-maximizing objective**, not a metric bug or implementation error. KL-aligned methods (EAP, EP-on-KL, ACDC) reconstruct the full model (`d_circuit ≈ d_full`) → low answer-pair KL. This sharpens the loss-form sub-RQ: `answer_margin_ce` optimizes a quantity (margin size) that is *not* what answer-pair KL measures (margin match), so DiscoGP-EP's faithfulness ranking depends entirely on which axis the evaluator cares about. Cite as the clean mechanistic explanation of why "behaviorally correct" (wins the task) ≠ "faithful" (matches the model's calibrated distribution).
- **Code pointer / Run:** `auto_circuit/metrics/prune_metrics/answer_diff_percent.py`, `auto_circuit/prune_algos/disco_gp.py:498` (`_faith_loss_margin`), `auto_circuit/metrics/prune_metrics/kl_div.py` (`_answer_pair_logprobs`).

### Answer-pair KL: DiscoGP-EP circuit is real but AutoCircuit's top-k sweep misrepresents it — 2026-06-06

- **Context:** Answer-pair KL metric (KL over {IO, S} 2-class distribution, zero ablation) showed DiscoGP-EP tracking the Random baseline at 0.5 for most circuit sizes. Training curve from `disco_gp_edge_train_graph.html` showed: Faithfulness = 0.041 at end (min 0.000014 mid-run), Sparsity = 0.07 (7% mean sigmoid), Completeness = 0.74 (≈ ln 2, complement is near-uniform).
- **Observation:**
  - Training DID converge: faithfulness = 0.041 → stochastic circuit achieves ~96% IO-preference confidence. Completeness → ln(2) → complement is uninformative. The algorithm found a real circuit.
  - The U-shaped answer-pair KL (good at 200–1000 edges, bad elsewhere) is the expected behaviour of a **probabilistic mask model evaluated with a deterministic top-k sweep**. During training, edges with sigmoid = 0.2 are included 20% of the time; the model learns to be faithful given this probabilistic mixture. Top-k at k > the "core" set includes low-sigmoid marginal edges that were mostly excluded during training, breaking the circuit and pushing KL back to 0.5.
  - The circuit's **correct evaluation point is at `logit > 0` threshold** (paper's protocol), not at the sweep's k that happens to match that count. AutoCircuit's sweep passes through this as a single point.
  - **DiscoGP-EP DOES outperform EP under zero ablation**: DiscoGP-EP dips to KL ≈ 0.2 at 200–1000 edges; EP (trained with resample) stays at 0.5 throughout zero-ablation evaluation. This is the expected ablation-type ordering.
  - The gap between 0.2 (port) and 0.02 (paper): (1) edge-only without weight pruning is harder — circuit needs more edges to be self-contained under zero ablation; (2) mid-run over-pruning (sparsity dropped to 3.4% at midpoint, then recovered to 7%) prevented full faithfulness convergence; (3) top-k vs logit>0 threshold: even at the right k, top-k ordering may not exactly match the binary threshold.
- **Significance:** DiscoGP-EP is working as designed — it finds a real circuit visible at 200–1000 edges. The metric visualization is misleading because the top-k sweep buries the good operating point inside a U-shaped curve that looks random at large k. For thesis results, report the **minimum answer-pair KL** across the sweep (or the value at the logit>0 circuit size), not the full curve uncritically. The 0.02 paper number is not the right comparison target: it uses joint weight+edge pruning; the thesis uses edge-only.
- **Code pointer:** `disco_gp_edge_train_graph.html` (in repo root, generated by `show_train_graph=True`), `auto_circuit/metrics/prune_metrics/kl_div.py:measure_answer_pair_kl_div`

### DiscoGP paper KL (0.02) vs AutoCircuit KL: not the same metric — 2026-06-05

- **Context:** Traced why the paper reports ~0.02 KL at 2% edge density (IOI) but AutoCircuit's `CLEAN_KL_DIV_METRIC` shows DiscoGP at or above the Random baseline.
- **Observation:** The paper's KL is from `evaluation.py:compute_faith_loss_binary_label` — `F.kl_div(full_2class, circuit_2class, reduction='mean')` over a **(B, 2) tensor** of `[logit_IO, logit_S]` only. `reduction='mean'` divides by B×2. AutoCircuit's KL is `KL(full_model ‖ circuit)` over all **50,257 vocab tokens**, summed per sample then averaged over B. Three compounding differences: (1) projection scope: 2 tokens vs. 50,257; (2) direction: KL(circuit ‖ full) vs. KL(full ‖ circuit) (minor); (3) reduction scale: per-element vs. per-sample (factor of 2 in DiscoGP's case). Scope is the dominant factor.
- **Significance:** The paper's 0.02 KL and AutoCircuit's large KL are *both correct* — they measure different things. DiscoGP was trained to minimise 2-class CE on (IO, S) — it is faithful on that projection and makes no guarantees about the full distribution. EP was trained to minimise `KL(full ‖ circuit)` — directly matching what AutoCircuit measures. This is why EP and ACDC sit low on `CLEAN_KL_DIV_METRIC` and DiscoGP does not: it is the objective, not the algorithm quality. For the thesis this means: (a) DiscoGP's high AutoCircuit KL is expected and not a bug; (b) cross-algorithm KL comparisons must be caveated by the alignment between each algorithm's training objective and the evaluation metric; (c) the paper's 0.02 KL should *not* be cited as validation that AutoCircuit will recover similar numbers.
- **Code pointer:** `docs/related_work/disco_gp_sheaf_discovery.md` (⚠️ note under Loss Functions), `auto_circuit/metrics/prune_metrics/kl_div.py`, `frankniujc/disco_gp:evaluation.py`

### Suggested follow-up experiments — 2026-05-15

- **Score-distribution histogram.** Plot `sigmoid(logit)` over all edges for each algorithm. Expect DiscoGP-E/EP to be bimodal (spike near 1, blob near 0); EAP/GT to be approximately continuous. The blob is the random tail responsible for the KL plateau.
- **Train-time KL probe.** At convergence of edge training, evaluate KL on the train set with Gumbel-sampled masks. Expectation: already at `Random` level, confirming the loss never minimised it.
- **Loss-swap ablation.** Re-run DiscoGP-E with `F.kl_div(log_softmax(train_logits), log_softmax(clean_logits.detach()))` as the faith loss. Expectation: KL curve jumps to ACDC/EAP level; logit-diff drops slightly. Confirms objective mismatch as root cause.
- **Sparsity decoupling.** Re-run with `e_max_times_lambda_sparse=1` (no ramp). If KL improves materially, the sparsity ramp contributes independently of the loss form.
- **Ablation-type cross-evaluation.** Train under zero, evaluate under resample/mean. If the KL gap narrows but logit-diff is unchanged, the zero-ablation amplification is confirmed.
