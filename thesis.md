# Thesis Notes

## Related Work

### Non-identifiability of circuit discovery and hydra effects — 2026-05-28

- **Context:** Reading Méloux et al. (2025), "The Dead Salmons of AI Interpretability" (arXiv 2512.18792), while framing the multi-algorithm comparison chapter.
- **Observation:**
  - Circuit discovery is formally **non-identifiable**: multiple mutually incompatible circuits can explain the same observed model behavior equally well. The paper formalises this via a statistical-causal framework where an interpretability task is identifiable only when the query space, surrogate class, and discrepancy measure jointly determine a unique explanation.
  - Each algorithm's design choices (ablation type, loss function, optimization regime) constitute a different surrogate class and discrepancy measure, so EP, DiscoGP, ACDC, and EAP are formally answering slightly different causal queries — not approximating the same ground truth.
  - **Hydra effects**: models contain redundant causal pathways. Ablating edge A causes edge B to compensate, making A appear unimportant — until B is removed first. Edge importance is therefore context-dependent on what else remains active during measurement.
  - Comparison against a random baseline is a low bar. The meaningful test is whether a discovered circuit outperforms *other non-trivial computational explanations* of the same behavior (i.e., circuits from competing algorithms at the same sparsity level).
  - Faithfulness metrics (KL divergence, logit diff) measure how well a circuit reproduces the full model's output — not whether it identifies the correct mechanism. A faithful circuit is not necessarily causally correct.
- **Significance:** Provides theoretical grounding for why EP and DiscoGP return different circuits on IOI/GT/Docstring — the divergence is expected and does not indicate one algorithm is broken. Frames the multi-algorithm comparison as exploring the solution space of valid circuit explanations rather than a race to a unique ground truth. Directly relevant to the Discussion chapter when interpreting disagreements between algorithms.
- **Code pointer:** `docs/related_work/dead_salmon_interpretability.md`

### Circuit non-uniqueness is constructible: Functional Anisotropy rejected (OASR) — 2026-06-15

- **Context:** Reading Chen et al. (2026, ICML), "All Circuits Lead to Rome" (arXiv 2605.12671) — the DiscoGP authors' follow-up — while framing sub-RQ3's stability pillar.
- **Observation:**
  - Names and rejects the **Functional Anisotropy Hypothesis**: the implicit belief that each task localises to a unique/near-unique circuit. A single task is instead supported by many structurally distinct circuits/sheaves that are each faithful, sparse, and complete.
  - **Overlap-Aware Sheaf Repulsion (OASR)** *forces* this: it augments the DiscoGP objective with a penalty on edges from prior runs, repelling later runs to near-disjoint solutions at equal performance. This is the methodological inverse of Méloux's passive resampling — same conclusion reached by opposite means (triangulation).
  - Quantitatively: two IOI sheaves at 100% accuracy with **IoU = 4.1%**; over 20 runs mutual IoU drops to **0.15%** (OASR) / 0.30% (random init). Two-sheaf IoU stays 4.1–11.0% across IOI/BLiMP/Docstring.
  - A **3-edge IOI sheaf reaches 86.7%** yet template-decomposition (ABBA/BABA) yields functional circuits excluding all three — so even a minimal circuit has *no individually indispensable edge*.
  - Non-uniqueness persists across **DiscoGP, ACDC, EAP, and Edge Pruning** — exactly this thesis's comparison set — and is theoretically predicted by the **Distributive Dense Circuit Hypothesis** (superposition → non-canonical mechanisms).
- **Significance:** Sets an *upper bound* on Pillar-2 degeneracy: if circuits can be engineered to ~4% IoU at identical accuracy, instability measured under mere reseeding is a lower bound on true non-uniqueness. Reframes low circuit overlap as the theoretically *expected* outcome of superposition rather than algorithm failure, and warns against reading EP/DiscoGP sparsity as "the essential mechanism." Strongest citable statement that mechanistic explanations are non-canonical — and notable that DiscoGP's own authors now argue their sheaves are non-canonical.
- **Code pointer:** `docs/related_work/all_circuits_lead_to_rome.md`

### Faithfulness metrics are methodology-relative; AutoCircuit library origin — 2026-05-28

- **Context:** Reading Miller et al. (2024), "Transformer Circuit Faithfulness Metrics are not Robust" (arXiv 2407.08734, CoLM 2024), the paper that introduced the AutoCircuit library this thesis extends.
- **Observation:**
  - Faithfulness scores are not stable properties of a circuit — they are co-determined by six ablation dimensions: granularity (head/MLP/neuron), component type (node/edge/branch), ablation value (zero/mean/resample), token positions, ablation direction (circuit vs. complement), and component set selection.
  - IOI faithfulness under node ablation is ~87% logit difference; the same circuit under different ablation choices produces statistically significantly different values (mean vs. resample: p = 1e-5). IQR up to 50% within a single dataset.
  - Tracr ground-truth experiment: all three algorithms (ACDC, SP, HISP) achieve perfect circuit recovery when discovery and evaluation use the same ablation methodology — methodology alignment matters more than algorithm choice in the ideal case.
  - Zero vs. resample ablation on the X-Proportion Tracr model yield circuits with different compositions — circuits are ablation-relative objects, not unique ground truths.
  - AutoCircuit library released with the paper: efficient edge-patching, zero/mean/resample ablations, TransformerLens integration, faster than official ACDC. This is the codebase this thesis extends.
- **Significance:** Provides the methodological grounding for why EP (resample) and DiscoGP (zero ablation) produce different circuits by construction — the divergence is predicted by the framework, not evidence of a bug. Also frames why cross-algorithm faithfulness comparisons without ablation-type caveats are misleading. Primary citation for the AutoCircuit library in the Implementation chapter.
- **Code pointer:** `docs/related_work/faithfulness_metrics_not_robust.md`

### EAP-IG exhibits high structural variance and hyperparameter sensitivity — 2026-05-28

- **Context:** Reading Méloux et al. (2025), "Mechanistic Interpretability as Statistical Estimation: A Variance Analysis of EAP-IG" (arXiv 2510.00845), the empirical companion to the Dead Salmon paper, while assessing reliability of EAP as a comparison baseline.
- **Observation:**
  - Circuit discovery is a statistical estimation problem built on CMA. CMA scores are intrinsically volatile random variables, not stable properties — instability is not solely caused by approximation shortcuts but exists at the base layer.
  - EAP-IG bootstrap resampling yields Jaccard similarity ≈ **0.561** (CV = 0.335) — barely better than chance structural agreement. Meta-dataset and paraphrased prompts give Jaccard ≈ 0.79, showing data distribution matters more than within-distribution sampling noise.
  - Hyperparameter sensitivity spans Jaccard **0.086–1.0** across EAP variant × aggregation method × intervention type — nearly the full possible range.
  - Bootstrap samples cluster into **distinct circuit modes** rather than noise around one answer — empirical confirmation of the non-identifiability predicted by the Dead Salmon paper.
  - Larger models (Llama-3.2-1B) are more unstable than GPT-2-small.
- **Significance:** Single-run EAP results in the thesis are a point estimate in a high-variance landscape. Disagreements between EAP and EP/DiscoGP are ambiguous evidence — both the algorithm difference and EAP's own variance contribute. Optimization-based methods (EP, DiscoGP) train over epochs and may produce structurally more stable circuits than attribution-based EAP — a motivated hypothesis worth raising in Discussion if not tested directly. Must flag as a limitations caveat.
- **Code pointer:** `docs/related_work/eap_ig_variance_stability.md`

### DiscoGP: joint weight+edge pruning via Gumbel-sigmoid; sheaf discovery — 2026-06-03

- **Context:** Reading Yu et al. (2025), "Sheaf Discovery with Joint Computation Graph Pruning and Flexible Granularity" (EMNLP 2025, arXiv 2407.03779) — the paper this thesis ports to AutoCircuit as `disco_gp.py`.
- **Observation:**
  - DiscoGP's objective is `ℒ = ℒ_fidelity + λ_c · ℒ_completeness + λ_sparsity`. The **completeness term** (CE of the complement circuit against uniform) is what differentiates it from EP and ACDC — it explicitly forces removed components to be uninformative, not just that the kept components are faithful.
  - **Two-phase sequential training**: Phase 1 optimises per-element weight mask logits (excluding embeddings, layer norms); after convergence, masks are frozen via `logit > 0` threshold and baked into weights before Phase 2. Phase 2 optimises per-edge logits on the weight-sparse model. This is *sequential*, not simultaneous joint optimisation.
  - **Gumbel-sigmoid STE**: `s_i = σ(l_i - log(log U₁ / log U₂) / τ)` with hard binary mask via STE. Temperature τ = 0.01 for weights (near-deterministic), τ = 1.0 for edges (softer gradients). Lambda schedule with warmup + cooldown controls sparsity pressure.
  - **Zero ablation** is the deliberate choice: the paper argues that zero ablation makes sheaves *self-contained* (functional in isolation) whereas mean/resample ablation only identifies components that are *important relative to some counterfactual*. This is a philosophical design choice with direct implications for comparing DiscoGP circuits against EP (resample) circuits.
  - **Mask convention is inverted**: paper uses mask=1 → keep, mask=0 → ablate. AutoCircuit uses mask=1 → ablate. The port negates (`1 - disco_mask`) in `_sample_edge_masks` — a non-obvious correctness-critical detail.
  - **Results**: IOI = 100% accuracy at 1.79% weights / 2.03% edges. ACDC on PARAREL = only 1% accuracy vs. DiscoGP 93.1% — attributed to greedy threshold approach and mean ablation failing on complex tasks.
  - **QKV split**: Paper applies separate Q/K/V edge masks; the AutoCircuit port **matches this** by default (`separate_qkv=True`). The only granularity the port omits is *neuron-level* masking, which lives in the unused weight phase (see edge-only audit below).
  - **Composability**: Two sheaves from different tasks can be composed while maintaining 80–100% accuracy. Validates the modular-unit framing; not reproduced in thesis but motivates DiscoGP's design.
  - **Faithfulness audit of the port (vs. `frankniujc/disco_gp` @ `54aef81`), edge-only scope**: the thesis uses DiscoGP **edge-only** (`use_weight_masks=False`). For that path the port is a faithful reproduction — `gumbel_sigmoid`, `schedule_epoch_lambda`, all edge-phase defaults, zero ablation, the Q/K/V-split edge graph, and the `answer_margin_ce` faith/completeness losses are **exact/equivalent** to the original. Q/K/V splitting is on by default (`separate_qkv=True`, [tasks.py:95](auto_circuit/tasks.py#L95)), matching the original's separate `edge_mask_attention_{q,k,v}_logits` — an earlier note claiming "no QKV split" was wrong.
  - The original "fidelity = NLL of predicted label" is actually supervised CE against the *dataset* label (good=class 0 / `answer_idx`), not distribution distillation — so `answer_margin_ce` is the faithful mode and `argmax_ce` is a port-introduced task-agnostic relaxation.
  - **Edge-only divergences**: (1) port defaults to non-paper `argmax_ce` (fixed by `-EP`); (2) PARAREL multi-class loss not ported (moot — not a thesis task); (3) original reimplements blocks threading `prev_head_idx`, port wraps a frozen model (same math, enables AutoCircuit's eval pipeline). The only granularity the port lacks is *neuron-level* masking, which lives in the unused weight phase.
  - **Registration (clean single-axis contrast)**: `DISCO_GP_EDGE_ONLY` (`-E`) and `DISCO_GP_EDGE_ONLY_PAPER` (`-EP`) now both use `e_lambda_complete_init=1.0`, so they differ **only** on `loss_mode` (argmax_ce vs answer_margin_ce). Completeness is held fixed, so the `-E`/`-EP` contrast cleanly isolates the loss-form axis (updated 2026-06-03; an earlier version differed on completeness too).
  - **Weight phase (unused, for completeness)**: joint `DISCO_GP_PRUNE_ALGO` diverges from paper defaults (w_lr 1.0→0.1, weight-completeness 1.0→0.0, argmax_ce) but is out of scope.
- **Significance:**
  - The completeness objective and zero ablation together mean DiscoGP is answering a *different causal query* than EP/ACDC/EAP — "what is the minimal self-sufficient circuit?" vs. "what edges matter most relative to a corrupt baseline?". This is the primary framing for interpreting algorithm disagreements in the Discussion chapter.
  - The edge-only variants (`DISCO_GP_EDGE_ONLY_PRUNE_ALGO`) implemented in this thesis allow a controlled comparison against ACDC/EAP at the same edge-only level; the joint variant is a DiscoGP-exclusive condition.
  - The argmax_ce vs. answer_margin_ce loss-mode distinction (thesis-specific, not in paper) allows isolating loss-form effects from method effects — relevant to RQ2/RQ3 analysis.
- **Code pointer:** `docs/related_work/disco_gp_sheaf_discovery.md`, `auto_circuit/prune_algos/disco_gp.py`

### Edge Pruning: gradient-based L0 edge discovery; the EP baseline — 2026-06-03

- **Context:** Reading Bhaskar et al. (2024), "Finding Transformer Circuits with Edge Pruning" (NeurIPS 2024 Spotlight, arXiv 2406.16778) — the paper this thesis ports as `edge_pruning.py`. Verified against `princeton-nlp/Edge-Pruning` source (`src/modeling/l0.py`).
- **Observation:**
  - **Method**: frame circuit discovery as optimization. Learn a **hard-concrete L0** mask (Louizos; temp 2/3, stretch [-0.1,1.1]) over every edge, with sparsity enforced by a **Lagrangian dual** `λ₁(t−s)+λ₂(t−s)²` (λ maximized by gradient ascent, target sparsity `t` ramped). Faithfulness = **KL divergence** to the full model. **No completeness term; frozen weights; separate Q/K/V edges.**
  - **Ablation = interchange/resample**: a cut edge injects the corrupted-input activation `ỹ_j`, not zero — `f_i(Σ_j [z_ji·y_j + (1−z_ji)·ỹ_j])`. This is the defining contrast with DiscoGP's zero ablation. Port uses `AblationType.RESAMPLE` ([edge_pruning.py:165](auto_circuit/prune_algos/edge_pruning.py#L165)).
  - **Results**: 2.65× fewer edges than ACDC at equal faithfulness on IOI; faster and far more faithful than EAP at 100K examples (KL 0.20 vs 3.78); perfect Tracr ground-truth recovery; scales to CodeLlama-13B at >99.96% sparsity.
  - **DiscoGP Appendix C** calls EP "equivalent to our edge pruning setting" — true only at the family level (edge-only, frozen-weight, gradient-based, disentangled residual, Q/K/V-split). They differ on mask distribution (hard-concrete L0 vs Gumbel-sigmoid), **ablation (resample vs zero)**, sparsity (Lagrangian L0 vs scheduled L1-style), and objective (KL vs label-CE+completeness).
  - **Port specifics**: `EDGE_PRUNING_PRUNE_ALGO` uses `use_node_masks=True`, `disable_node_loss=True` (matches original IOI sweep), `target_edge_sparsity=1.1`, `steps=3000`, lr=0.8; returns soft mask `σ(log_alpha/β·F)` for AutoCircuit's threshold sweep rather than EP's binary-search `deterministic_z`. Known difference: edge/node masks are pre-composed before patching rather than applied sequentially.
- **Significance:** EP and DiscoGP are *genuinely different* edge-only methods (resample/KL vs zero/CE+completeness), so the EP-vs-DiscoGP comparison is not redundant — it contrasts two causal queries. EP's resample ablation and KL objective align it with AutoCircuit's `CLEAN_KL_DIV_METRIC`; its Tracr ground-truth recovery makes it the strong optimization baseline against ACDC (search) and EAP (attribution). Primary input for the RQ2 algorithm-comparison framing.
- **Code pointer:** `docs/related_work/edge_pruning.md`, `auto_circuit/prune_algos/edge_pruning.py`

### Docstring circuit: origin of the benchmark task and its ground-truth — 2026-06-07

- **Context:** Reading Heimersheim & Janiak (2023), "A circuit for Python docstrings in a 4-layer attention-only transformer" (SERI MATS, Neel Nanda) — the source of the Docstring benchmark task used in AutoCircuit.
- **Observation:**
  - Hand-found circuit in Neel Nanda's `attn-only-4l` toy model predicting repeated `:param` argument names. Found via **resample (corrupted→clean) activation patching** with a **logit-difference** metric over three corruptions (random answer / random def / random doc), batch of 50.
  - Components: Argument Mover Heads **3.0, 3.6** (copy answer ≈ IOI Name Movers); Fuzzy Previous Token Heads **1.4, 2.0** + Induction subcircuit (**0.2**+**1.4**) move "B"-information; Duplicate Token Heads **0.5** (mostly embedding transform), **1.2** (suppresses docstring attention); causal-mask positional head **0.4**; multi-function head **1.4** (Prev-Token in def context, Induction in doc context).
  - **Partial faithfulness:** resample-ablating everything outside the named head set recovers only **42%** (full model 56%, full minus negative heads 0.0/2.2 = 72%); adding 1.0/0.1/2.3 → 58%. Negative head **2.2** *improves* performance when patched.
- **Significance:** Defines the Docstring task and its hand-labeled ground-truth circuit — the ROC reference for circuit-discovery evaluation. The reference circuit itself recovers only ~42–58% of performance, reinforcing the methodology-relative view of faithfulness (Miller 2024) and the redundancy/negative-head motifs (Dead Salmon) that complicate ROC scoring. Its resample-ablation + logit-diff provenance aligns the Docstring ground truth with EP's resample ablation and logit-difference metrics in sub-RQ3.
- **Code pointer:** `docs/related_work/docstring_circuit.md`

## Results

### DiscoGP edge-only variants: KL collapse despite strong logit-difference (IOI, zero ablation) — 2026-05-15

- **Context:** IOI component circuit, GPT-2 Small, zero ablation for training and evaluation. Algorithms: `DiscoGP-E` (edge-only, `argmax_ce`, no completeness), `DiscoGP-EP` (edge-only, `answer_margin_ce`, completeness on), `DiscoGP` (joint weight+edge), EAP, Random, Ground Truth.
- **Run:** `figures/circuit_discovery/2026-05-15_17-57-22/`
- **Observation:**
  - Logit-difference percent ([figure_15.pdf](figures/circuit_discovery/2026-05-15_17-57-22/figure_15.pdf)): DiscoGP-E and DiscoGP-EP perform competitively across the edge-count sweep — recover most of the IOI margin at moderate sparsities, comparable to EAP.
  - Clean KL divergence ([figure_12.pdf](figures/circuit_discovery/2026-05-15_17-57-22/figure_12.pdf)): same two algorithms track the Random baseline closely across nearly the entire edge-count range; EAP/Ground Truth sit well below.
- **Significance:** A circuit that preserves the answer-margin but is essentially random on full-vocabulary divergence is exactly what the algorithm's objective predicts. Root-cause analysis in the Discussion section.

### Head importance heatmaps: EAP sign-flips negative movers — 2026-05-28

- **Context:** IOI component circuit, GPT-2 Small. Head importance heatmaps generated via `auto_circuit/metrics/prune_metrics/head_heatmap.py` (aggregation = sum over all incoming edges per head). Visualisation replicates IOI paper Figure 3(b) style: blue = positive, red = negative; `colorscale="RdBu", zmid=0`.
- **Crucial caveat — heatmap plots raw *signed* prune score, not selection-importance.** `head_heatmap.py:40` reads `e.prune_score(...)` with no `abs()`. But circuit *membership* in AutoCircuit is decided by **magnitude**: `desc_prune_scores` sorts on `.abs()` (`tensor_ops.py:247`) and `prune.py:94` thresholds on `patch_mask.abs() >= threshold`. So the colours are *not* the quantity that decides which edges are kept.
- **Observation:**
  - Heads 10.7 and 11.10 (known Negative Name Mover Heads; 10.7 is the canonical **copy-suppression head**, McDougall et al. 2023) appear **strongly positive (blue)** under GT, DiscoGP-E, DiscoGP-EP, and DiscoGP, and **negative (red)** under EAP.
  - This is largely a *representation* artefact, not a disagreement about importance: only EAP's prune score carries a **sign**. DiscoGP/EP mask values are sigmoids ∈ (0,1) — non-negative by construction — and GT is binary {0,1}, so none of them *can* be red. EAP scores 10.7/11.10 with large **|magnitude|**, so by `.abs()` selection EAP **also keeps them** — it does not exclude negative movers.
- **What negative heads do:** They attend to a token already present in context that the model is confident about (in IOI, the IO / correct answer) and write in the **negative** direction, *decreasing* that token's logit. Functional role is **calibration / anti-overconfidence / self-repair** — damping predictions the rest of the model makes too strongly. Copy suppression is task-general, which is why the same heads surface in GT, not only IOI.
- **Mechanism (why each method scores them differently):**
  - **GT**: Binary — hand-labelled as circuit members, so score = 1.
  - **DiscoGP / EP**: Mask values ∈ (0, 1) trained so the subnetwork **reproduces the full model's output** (faithfulness), not so it maximally solves the task. The full model's logits are calibrated *downward* by these heads. Dropping them makes the circuit **overshoot** the full model's confidence (correct-answer logit too high) → faithfulness loss rises (KL, or the `answer_margin_ce` margin-match). To minimise that loss the optimiser must **keep** them → high mask value → high (sign-agnostic) edge score. The score reflects *necessity to be faithful*, not task-helpfulness.
  - **EAP**: Signed first-order attribution — raw score ≈ the edge's marginal contribution to the metric. 10.7/11.10 suppress the **IO (correct) logit**. Ablating their inputs removes that suppression → IO logit **rises** → logit diff (IO − S) **rises** → marginal contribution is **negative** → EAP raw score **negative (red)**. The heatmap sums *signed* edge scores per head, so mixed-sign edges can also partially cancel. Selection still keeps them via `|score|`.
- **Significance:** The heatmap surfaces a *representation* difference, not a membership disagreement. Only EAP retains the **sign / direction** of an edge's effect; DiscoGP/EP discard it (masks ≥ 0) and GT is binary. So the red-vs-blue contrast on 10.7/11.10 should be read as "EAP additionally encodes that these heads act in the suppressive direction," not "EAP omits them" — all four methods include them by magnitude. The genuinely interesting cross-method point is the *direction* information EAP preserves and the mask methods structurally cannot express: useful when arguing what each score representation can and cannot tell you about a circuit. Faithfulness-vs-minimality still applies to whether a head is *kept*; the heatmap colour speaks to *direction*, a separate axis. Ties into the Méloux "different methods find different circuits" theme; relevant to the sub-RQ comparing algorithm outputs.

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

## Methodology — Patching & Ablation Mechanics

### How factorized edge patching works, and how zero vs. resample change its meaning — 2026-06-01

- **Context:** Tracing the AutoCircuit patching path to understand what a "circuit edge" computes under tree patch. Code: `auto_circuit/utils/patch_wrapper.py` (`PatchWrapperImpl.forward`), `auto_circuit/utils/graph_utils.py` (`graph_edges`, `patch_mode`), `auto_circuit/utils/ablation_activations.py` (`src_ablations`), `auto_circuit/prune.py` (`run_circuits`).

- **Observation — factorized graph = residual-stream decomposition:** With `factorized=True` an edge exists for *every* `(src, dest)` with `src.layer < dest.layer` (`graph_edges`, the `s.layer < d.layer` filter). There is no literal wire (e.g. `A0.9 → A11.0.Q`); the edge is an *attribution*. Because every component writes to the shared residual stream **additively** and every component reads the whole stream, an upstream node's output is a separable summand inside every downstream node's input. AutoCircuit stores each source output separately (`curr_src_outs`) so it can add/remove that one summand from one reader's input independently.

- **Observation — the per-edge update:** At each dest, `arg_0 += mask · (patch_src − curr_src)` per incoming edge. `mask=0` → keep the live (clean) contribution; `mask=1` → swap it for the ablation value. `curr_src` is the *live* output on this pass, so cutting an edge into an intermediate node changes that node's output and the effect **cascades** to all its downstream kept edges — this is what makes it tree/path patching rather than edge-local activation patching.

- **Observation — zero ablation:** `patch_src = 0` (`src_ablations` ZERO branch returns `zeros_like`), so a cut edge *subtracts* the writer's contribution out of the reader's input. A node with **all** inputs cut receives exactly the zero residual → LayerNorm returns its bias (`β`), Q/K/V become position-independent constants → the node emits a **fixed, input-independent constant** (bias-driven), not random noise. Through an `edge → resid_final`, that is a constant logit offset identical across all prompts. The corrupt input is irrelevant under zero ablation (run but discarded).

- **Observation — resample ablation:** `patch_src` = source activations on `batch.corrupt`, so a cut edge **substitutes** clean→corrupt for that (writer, reader) pair. A node with all inputs cut receives the **full corrupt residual stream** (clean terms cancel, corrupt terms sum to `resid_corrupt`) → it computes its normal, coherent output **for the corrupt prompt** — prompt-dependent, correlated with the corrupt answer, never a bias constant. The off-circuit background is therefore "the model running on corrupt," and the circuit is the set of edges that route the clean-vs-corrupt *difference*. This keeps activations on-distribution (zero residual is never seen in training; corrupt residual is in-distribution), but makes results depend on the specific clean/corrupt pairing.

- **Observation — patch direction (AutoCircuit `PatchType`):** `TREE_PATCH` feeds `batch.clean`, ablates the **complement** (mask = sub-threshold edges), keeps circuit edges clean → sufficiency / "run the circuit in isolation"; reported edge count is the kept edges `(1−mask).sum()`. `EDGE_PATCH` feeds `batch.corrupt`, ablates the **circuit** edges → necessity. These two are exactly Miller et al.'s "direction" axis (ablate circuit vs. complement; clean vs. corrupt input).

- **Significance:** (i) A discovered circuit is only faithful if its kept edges form unbroken paths from the embeddings to `resid_final`; a dangling in-circuit edge contributes only ablation-artifact signal (zero: a constant; resample: corrupt-prompt info), never prompt-relevant computation. (ii) EP (resample) and DiscoGP (zero) are tested against **different baselines** and solve different causal queries, so their faithfulness numbers are not directly comparable — this is the concrete mechanism behind Miller et al.'s "faithfulness is methodology-relative." Every cross-algorithm comparison table must state the ablation type. See [`faithfulness_metrics_not_robust.md`](docs/related_work/faithfulness_metrics_not_robust.md) (six-dimension ablation framework).

## Experiments

### Model-sharing double-wrap when two tasks share a model — 2026-06-08

- **Context:** Checking whether `faithfulness_comparison.py` handles per-task models correctly. IOI and Greater-Than both use `gpt2-small`; Docstring uses `attn-only-4l`.
- **Observation:**
  - `MODEL_CACHE` (`tasks.py`) keys on `(_model_def, dtype)`, so IOI and GT **shared one `gpt2-small` instance**. `patchable_model` **mutates the model in place** (injects `PatchWrapperImpl`s; its docstring says so), and its `assert not isinstance(model, PatchableModel)` guard misses this because the cache stores the raw `HookedTransformer`, not the `PatchableModel`.
  - The second task to build (**GT**) re-ran `make_model_patchable` on IOI's already-wrapped model → `PatchWrapperImpl(module=PatchWrapperImpl(...))` (**double-wrapped**); both tasks then share that object. Forward passes still succeed (the non-activated inner wrapper is a passthrough), so it was **silent**.
  - **Measured impact (controlled experiment, stash fix on/off, same seed):** the inner wrapper is effectively transparent, so the effect is *much smaller than the structure suggests*:
    - **EAP discovery** (gradient-based, bypasses `patch_mode`): **bit-identical** (max\|diff\| = 0).
    - **`run_circuits` evaluation** (official GT circuit, tree-patch): **bit-identical** (max\|diff\| = 0).
    - **Edge Pruning discovery** (patch-based, backprops through wrappers): a **small but real, reproducible** perturbation — max\|diff\| ≈ **0.0098** on soft-mask scores (∈ 0..1), vs ~0.0025 run-to-run nondeterminism (≈4× noise, consistent across fixed-vs-buggy and fixed2-vs-buggy2). The extra passthrough layer perturbs FP rounding which compounds over training steps; it does not corrupt the patching logic.
  - So: it is **not** a gross corruption — EAP/eval unaffected; patch-based discovery (EP/DiscoGP/ACDC) shifts at the ~1%-of-mask, near-noise level. The real problem is **non-reproducibility**: GT's patch-based scores depend on whether IOI was loaded first in the same process.
- **Fix:** `tasks.py` `init_task` now wraps `copy.deepcopy(model)`, keeping `MODEL_CACHE` pristine and giving each task an independent model. Distinct `wrapped_model` ids and `double_wrapped=False` for all three tasks. Regression test: `tests/test_seed_dataloaders.py::test_tasks_sharing_model_def_get_independent_models` (slow).
- **Significance:** Worth fixing for reproducibility, memory (avoids ~2× wrappers on the shared model), and clarity — but pre-fix GT results are *not invalid*, only slightly load-order-dependent for EP/DiscoGP/ACDC (EAP/eval identical). Re-running fresh with the fix is the clean choice (and the eval needed re-running anyway), not a rescue from corruption. Methodological note worth keeping: I initially overstated this as "silent corruption"; the controlled measurement showed the effect is ~4× noise on EP masks and zero on EAP/eval.
- **Code pointer:** `auto_circuit/tasks.py` (`init_task`, the `copy.deepcopy(model)` in the `patchable_model` call).

### Docstring granularity: component for the spine, token as a sub-study — 2026-06-07

- **Context:** Choosing between `DOCSTRING_TOKEN_CIRCUIT_TASK` and `DOCSTRING_COMPONENT_CIRCUIT_TASK` for the sub-RQ3 faithfulness spine (`experiments/faithfulness_comparison.py`). Read Heimersheim & Janiak (2023), "A circuit for Python docstrings in a 4-layer attention-only transformer."
- **Observation:**
  - The docstring circuit is **fundamentally position-specific**: heads are defined by query/key positions, information flows through specific residual-stream positions across layers (`B_def → ,_B → C_def → param_3`), head 1.4 does different jobs at different positions (fuzzy-prev-token vs. induction), and head 0.4 reads positional info off the causal mask. A component-level ground truth is therefore a *lossy projection* of the true circuit.
  - **But** the authors themselves validated at the **coarse head (component) level** — "we could expand this to take into account the heads position … but stick to the more coarse-grained check" — using **resample ablation, corrupt→clean** (same direction AutoCircuit uses). So the canonical published circuit is the component one.
  - **Decisive structural fact:** Greater-Than exists *only* as a component circuit (no `GREATERTHAN_TOKEN_CIRCUIT_TASK`). IOI and Docstring have both. So all-component is the *only* configuration with uniform granularity across the three spine tasks.
- **Decision:**
  - **Spine = component for all three tasks.** Granularity is held constant as a controlled variable (Miller et al. 2024 name token-position as one of six methodology dimensions that shift faithfulness). Mixing token-Docstring with component-IOI/GT would confound every cross-task comparison with a granularity difference.
  - **Token-Docstring is promoted to a dedicated granularity sub-study**, not buried in the spine. It is the only one of the three tasks with a genuinely position-specific circuit, so it is the ideal (and only) probe for: (a) does discovery granularity change which algorithm wins; (b) do optimization methods (EP, DiscoGP) recover fine-grained positional/induction structure that attribution/greedy methods (EAP, ACDC) smear across positions — directly the sub-RQ3 hypothesis.
- **Significance:** Turns an inherited template default into a defensible, paper-grounded design choice; isolates granularity as a studied axis rather than an accidental confound. Also explains the token-circuit-only viz failures (per-seq-position Sankey panels) seen on 2026-06-07.
- **Code pointer:** `experiments/faithfulness_comparison.py` (TASKS, TASK_METRICS); circuits in `auto_circuit/metrics/official_circuits/circuits/docstring_official.py` (`docstring_true_edges` vs `docstring_node_based_official_edges`, `token_positions` flag).

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

### Foundation (#0) implemented: ablation × seed as swept parameters — 2026-06-05

- **Context:** Implementing the prerequisite infrastructure for the three-pillar sub-RQ3 design; tracked in GitHub issue #3.
- **Observation:**
  - **Discovery ablation parameterized in EP, DiscoGP, ACDC.** Each function had `ablation_type` hard-coded (`edge_pruning.py:165` RESAMPLE, `disco_gp.py:294` ZERO, `ACDC.py:96` RESAMPLE). Added `ablation_type: AblationType` parameter with the original as default to all three — fully backward-compatible. `make_ablation_variant(base_algo, ablation_type)` in `prune_algos.py` composes a new `PruneAlgo` via `functools.partial` with the ablation pre-bound, allowing SLURM-array cells to differ only by ablation without touching registered algo defaults.
  - **ACDC ablation insight:** unlike EP/DiscoGP which use `batch_src_ablations` (returns `Dict[BatchKey, Tensor]`), ACDC uses `src_ablations` directly on a single-batch tensor (for a plain tensor compatible with `patch_mode`). `src_ablations` already handles all AblationType variants via its hook logic (including ZERO → zeros, RESAMPLE → corrupt activations, TOKENWISE_MEAN → dataset iteration via `ablation_type.mean_over_dataset`), so parameterization required only: `sample = dataloader if ablation_type.mean_over_dataset else corrupt_batch; patch_outs_tensor = src_ablations(model, sample, ablation_type=ablation_type)`.
  - **Seed plumbing:** `Task` dataclass got `seed: int = 42` field (backward-compatible). `DATASET_CACHE` key updated to 6-tuple (adds seed). `load_datasets_from_json(random_seed=self.seed)` is now threaded through `init_task`. Two Tasks with identical params but different seeds now produce different dataloaders.
  - **EAP stays corpus-only** (no change): its `edge_attribution_patching_prune_scores` computes `grad × (corrupt_act − clean_act)` — this is a gradient-based attribution, not a patching operation. Adding `ablation_type` would be conceptually mismatched.
  - **`run_cell.py`** (new): CLI accepting `--task {ioi,gt,docstring}`, `--algo {ep,discogp-e,discogp-ep,acdc,eap,...}`, `--ablation {zero,resample,mean}`, `--seed INT`, `--null-test`. Saves cache as `cell_{task}_{algo}_{ablation}_seed{N}-{timestamp}.pkl` with a sibling `_manifest.json` encoding the full cell identity + `git_state()` + `algo_hyperparams()`. Null-test mode loads the model architecture then randomizes all weights (normal init) and sets cfg flags manually (the `nn.Module` branch in `Task.init_task` skips them).
  - **`analysis_utils.py`** (new, `auto_circuit/metrics/`): `top_k_edge_set`, `jaccard`, `overlap_matrix`, `matched_budgets`, `load_many_caches`, `extract_metric` — reuses `prune_scores_threshold` from `tensor_ops.py` and `edge_counts_util` from `graph_utils.py` to ensure edge budgets are identical to those used in `measure_prune_metrics`.
- **Significance:** Unlocks all three pillars of the sub-RQ3 design. The ACDC `mean_over_dataset` insight is worth noting: TOKENWISE_MEAN ablation on ACDC iterates the whole dataloader once, which is cheap for ACDC's single-batch architecture but should be confirmed against wall-clock budgets before scheduling N=5 stability runs.
- **Code pointer:** `experiments/run_cell.py`, `auto_circuit/metrics/analysis_utils.py`, `auto_circuit/prune_algos/{ACDC,edge_pruning,disco_gp}.py`, `auto_circuit/prune_algos/prune_algos.py`, `auto_circuit/tasks.py`

### Sub-RQ3 reframed: faithfulness + stability + null test; gap verified — 2026-06-02

- **Context:** Designing the experimental core of sub-RQ3. The concern that triggered the redesign: a plain faithfulness/completeness/sparsity comparison of EP, DiscoGP, ACDC and EAP is a *descriptive benchmark* — precisely the single-run artifact that Miller et al. (2024) and Méloux et al. (2025a, 2025b) argue is methodology-relative and statistically fragile. Reframing turns the chapter from "which method is most faithful?" into a hypothesis-driven study. Full design (with §-level detail) in [`docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md`](docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md); execution tracked as GitHub issues (epic #2; foundation #3; pillars #4/#5/#6).
- **Observation — three-pillar design:**
  - **Pillar 1 — Faithfulness spine (discovery × eval matrix).** Discover each method's circuit under ablation ∈ {zero, resample, mean}, then re-score across {zero, resample, mean} × {tree-patch, edge-patch}. Two headline effects: *evaluation-relativity* (same circuit, faithfulness moves with the eval methodology — Miller et al. 2024) and *discovery-relativity* (circuits discovered under different ablations diverge). Metrics follow the per-task mapping (IOI/GT → logit-diff %; Docstring → KL), plus completeness, sparsity, ROC-vs-ground-truth.
  - **Pillar 2 — Stability.** Structural variance of optimization methods (EP, DiscoGP) vs attribution/greedy (EAP, ACDC): pairwise **Jaccard at matched sparsity** across N≈5 repeats, all 3 tasks × both ablations {zero, resample}. Perturbation sources: dataset re-draw / bootstrap (all methods) + mask-init seed (EP/DiscoGP only). Numbers reported against Méloux et al. (2025b)'s EAP-IG benchmark (Jaccard ≈ 0.561, CV 0.335). A cross-method **overlap matrix** is a free byproduct of the same machinery.
  - **Pillar 3 — Null / "dead salmon" test.** Run each method on a **randomly-initialized GPT-2** (k≈10 re-inits) and test whether the discovered circuit's faithfulness is distinguishable from chance via the Monte Carlo p-value `p̂ = (1 + Σ_b 1{T_null^(b) ≥ T_obs})/(k+1)` (Méloux et al. 2025a, Appendix A). Doubles as a sub-RQ1 correctness check.
- **Significance — the gap is verified, and partly named by Méloux themselves:**
  - Méloux et al. (2025b) (EAP-IG variance, arXiv:2510.00845 v4) restrict their analysis to the **EAP family** (EAP, EAP-IG inputs/activations, clean-corrupted) and *explicitly list* characterizing the stability of other methods (incl. **ACDC**) as "future work." **No optimization/pruning method (EP, DiscoGP, subnetwork probing) has had its structural variance measured.** Pillar 2 fills this.
  - The **null/dead-salmon test has never been applied to circuit-discovery algorithms** — Méloux et al. (2025a) and the probing/attribution literature run null controls only on probing, feature attribution, and "world-model" tasks. The strongest control used for ACDC/EAP/EP/DiscoGP to date is a random-*edge* baseline on the *real* model (AutoCircuit's `RANDOM_PRUNE_ALGO`), far weaker than discovery on random *weights*. Pillar 3 is first-of-kind for these algorithms.
  - No collisions with concurrent work: *Certified Circuits* (Anonymous 2026, arXiv:2602.22968) targets **vision feature-channel** circuits and OOD-dataset-transfer stability via randomized smoothing — different domain and stability notion. DiscoGP's own paper (Yu et al. 2025) reports only faithfulness/sparsity, no across-seed variance. Adjacent but non-overlapping: *Have Faith in Faithfulness* (arXiv:2403.17806) argues circuit *overlap* must be paired with faithfulness — cited when interpreting the Pillar-2 overlap matrix, not a competing experiment.
- **Methodology — operating model & code realities (gate the build):**
  - **Run-per-setting, not a monolithic orchestrator.** One job = one `(method, task, discovery-ablation, seed)` → one cached `PruneScores` (SLURM-array friendly), then a single post-hoc analysis pass loads all caches. Discovery is the expensive isolated unit; evaluation is free re-scoring of cached scores (`measure_prune_metrics` already sweeps eval ablations; call once per `patch_type`).
  - **Discovery ablation is hard-coded, not a parameter** — `edge_pruning.py:165` (RESAMPLE), `disco_gp.py:294` (ZERO); `PruneAlgo.func` takes no ablation arg. Making it a swept axis is the core of the foundation issue (#3).
  - **EAP is corrupt/resample-only by construction** (`edge_attribution_patching.py:89` uses `corrupt_cache`), so the spine's discovery axis is *EP/DiscoGP/ACDC × {zero, resample, mean} + EAP × {corrupt}* — ≈ **30 discovery cells**, not a clean 4×3×3.
  - **`Task` has no seed field** and `DATASET_CACHE` (`tasks.py:192`) is keyed without seed, so the N-repeat seed axis is genuinely new plumbing. Null-test hook: `Task._model_def` accepts an `nn.Module` (`tasks.py:149`) but that branch *skips* the `use_attn_result/use_attn_in/use_split_qkv_input/use_hook_mlp_in` flags — they must be set on the random model manually.
- **Cost:** ≈ **270 discovery jobs** (spine ≈30 + stability 120 + null 120; evaluation free via cache). **ACDC (~60 Phase-B jobs) is the wall-clock bottleneck** (greedy) → timing contingency to demote it to anchor-only in Phase B if a single run is too costly to repeat. DiscoGP paper-loss OOMs on 40 GB nodes (exclude bob/minerva).
- **Code pointers:** spec `docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md`; existing pipeline `experiments/circuit_discovery.py`; discovery driver `auto_circuit/prune_algos/prune_algos.py:60`; eval driver `auto_circuit/metrics/prune_metrics/measure_prune_metrics.py:35`.

#### References (sub-RQ3 design)

- Miller, J., Chughtai, B., & Saunders, W. (2024). *Transformer Circuit Faithfulness Metrics are not Robust.* CoLM 2024. arXiv:2407.08734. — origin of the AutoCircuit library; six-dimension ablation framework. See [`docs/related_work/faithfulness_metrics_not_robust.md`](docs/related_work/faithfulness_metrics_not_robust.md).
- Méloux, M., Dirupo, G., Portet, F., & Peyrard, M. (2025a). *The Dead Salmons of AI Interpretability.* arXiv:2512.18792. — non-identifiability; null-model significance test. See [`docs/related_work/dead_salmon_interpretability.md`](docs/related_work/dead_salmon_interpretability.md).
- Méloux, M., Portet, F., & Peyrard, M. (2025b). *Mechanistic Interpretability as Statistical Estimation: A Variance Analysis of EAP-IG.* arXiv:2510.00845. — EAP-family structural variance (Jaccard ≈ 0.561); names other-method stability as future work. See [`docs/related_work/eap_ig_variance_stability.md`](docs/related_work/eap_ig_variance_stability.md).
- Bhaskar, A., et al. (2024). *Finding Transformer Circuits with Edge Pruning.* NeurIPS 2024. arXiv:2406.16778.
- Yu, et al. (2025). *DiscoGP / Sheaf Discovery with Joint Computation Graph Pruning.* EMNLP 2025. aclanthology 2025.emnlp-main.446.
- *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms.* (2024) arXiv:2403.17806.
- *Certified Circuits: Stability Guarantees for Mechanistic Circuits.* (2026) arXiv:2602.22968 — vision domain; not a collision.
- Tasks: Wang et al. (2022) IOI; Hanna et al. (2023) Greater-Than; Heimersheim & Janiak (2023) Docstring.

## Limitations

### Implementation design differences from the original algorithms — 2026-06-03

- **Context:** Both EP and DiscoGP are re-implemented inside AutoCircuit's `PatchableModel`/wrapper infrastructure rather than each paper's bespoke codebase. Audited the ports against the original repos (`princeton-nlp/Edge-Pruning` @ `a918efd`, `frankniujc/disco_gp` @ `54aef81`). The cores are faithful; the differences below are deliberate adaptations or known gaps and should be stated as limitations.
- **Observation — common to both ports:**
  - **Single training run is faithful — the curve-tracing difference is evaluation-side, not a faithfulness gap.** EP and DiscoGP (like EAP) estimate edge importance from *one* optimization run; that run's continuous mask **is** the algorithm's native importance output, so returning it as `PruneScores` matches how each method decides which edges to keep — it is not a shortcut. AutoCircuit's `run_circuits` then traces the sparsity–faithfulness curve by top-k thresholding that single ranking (so the port's circuits are strictly nested). The EP paper instead runs independent per-target-sparsity training runs (`ioi_sweep.sh` bash loop; **no merge** — each checkpoint evaluated separately into one scatter point), so its circuits need not be nested. EP provides **no** mechanism to merge multiple runs; ACDC is the only algorithm that synthesizes a ranking across runs (~20 τ-values) and only because its per-τ output is a *binary* circuit with no continuous scores. The lone genuine design choice is the calibration of EP's single run via `target_edge_sparsity=1.1` (a tunable lever) — not whether single-run is correct, which it is.
  - **Pre-composed masks vs. sequential forward-pass masking.** AutoCircuit's wrapper applies one effective mask per edge before patching; the originals thread edge (and EP node) masks sequentially through the forward pass. Same intended math, possibly different gradient dynamics.
- **Observation — Edge Pruning specifics:**
  - **Faithful**: hard-concrete L0 sampling (temp 2/3, stretch [−0.1, 1.1]), `KL(full ‖ circuit)` direction, resample/interchange ablation, Lagrangian `λ₁(s−t)+λ₂(s−t)²` with λ maximized, node-mask composition `1−(1−z_e)(1−z_n)` ≡ paper's `z_edge·z_node`, all run-script LRs = 0.8, `steps=3000`, `sparsity_warmup_steps=2500`, `target_node_sparsity=0.72`, and **`weight_decay=0.01`** (matches the original, which relies on `torch.optim.AdamW`'s default; setting 0 departs from the paper and breaks learning).
  - **Aligned 2026-06-03**: added `warmup_steps=200` to the EP registration to match the original's `--warmup_steps 200` LR warmup (previously defaulted to 0).
  - **Remaining gaps**: KL is computed at the last-token `out_slice` rather than the original's answer-span `[start:end]` (near-identical for single-token IOI); the four original LR groups (edge/node × log_alpha/λ) are collapsed to two shared knobs (no effect at the paper's all-0.8 values, but cannot reproduce edge≠node LR runs).
- **Observation — DiscoGP specifics (edge-only scope):**
  - The thesis uses DiscoGP **edge-only** (`use_weight_masks=False`); the weight-pruning phase and the paper's neuron-level "flexible granularity" are not exercised, so DiscoGP here is not the paper's full joint method.
  - **Faithful** for the edge-only path: `gumbel_sigmoid` + `schedule_epoch_lambda` (verbatim), all edge-phase defaults, zero ablation, Q/K/V-split edges, and the `answer_margin_ce` faith/completeness losses.
  - The default `loss_mode="argmax_ce"` is a port-introduced, task-agnostic relaxation with no counterpart in the original; only `DISCO_GP_EDGE_ONLY_PAPER_PRUNE_ALGO` (`-EP`) uses the paper-faithful `answer_margin_ce`.
  - The `-E` vs `-EP` registrations now differ **only** on `loss_mode` (argmax_ce vs answer_margin_ce); both hold `e_lambda_complete_init=1.0`, so the contrast cleanly isolates the loss-form axis.
- **Significance:** These are the caveats that bound how directly the thesis's EP/DiscoGP numbers map onto the original papers' reported results. Running each gradient method once is faithful to how it estimates importance *and* makes the multi-algorithm comparison apples-to-apples on a shared edge-count grid; the genuine caveats are the items above (ablation type, loss form, KL span, single-run calibration). Ties into the broader non-identifiability framing (Méloux 2025) and ablation-relativity framing (Miller 2024).
- **Code pointer:** `docs/related_work/edge_pruning.md`, `docs/related_work/disco_gp_sheaf_discovery.md`, `auto_circuit/prune_algos/edge_pruning.py`, `auto_circuit/prune_algos/disco_gp.py`, `auto_circuit/prune_algos/prune_algos.py`

## Notes

- **2026-05-31** — Added `auto_circuit/visualize_circuit_slider.py` (`save_circuit_sankey_slider`): one interactive HTML Sankey per (task, algorithm) with an edge-count slider whose steps reuse `edge_counts_util` — the exact logarithmic grid `measure_prune_metrics` uses — so the graph slider lines up tick-for-tick with the prune-metric curves. Wired into `experiments/circuit_discovery.py` (replaces the old Sports-Players-only `draw_seq_graph` block).
- **2026-05-31** — Gotcha when reusing `net_viz` (`visualize.py`): its `score_threshold` filter is `abs(edge_score) < threshold → skip`, so passing `score_threshold=0.0` does **not** drop zero-valued edges. Zeroing sub-threshold scores in the `PruneScores` dict therefore does NOT shrink the rendered Sankey — every "circuit" still draws all ~32k edges. Correct top-k rendering requires passing the actual threshold (`prune_scores_threshold(ps, k)`, floored to a small positive eps) into `net_viz` directly. This bug inflated a 10k-edge HTML to ~90 MB; the fix brings the full log-grid file to ~3 MB.
- **2026-05-31** — The lost `circuit_graph_viz.py` (only `.pyc` survived, never committed) was rebuilt fresh rather than recovered; new module lives next to `visualize.py` as `visualize_circuit_slider.py`.
- **2026-06-15** — Supervisor proposed an alternative to OASR's repulsion (Chen et al. 2026): obtain multiple circuits by *sampling* each edge `mask_e ~ Bernoulli(p_e)` from a single trained DiscoGP mask, since PruneScores already equal `p_e = sigma(l_e)`. Built `experiments/edge_prob_diagnostic.py` to test feasibility (reads cached `p_e` straight from `.prune_scores_cache/faithfulness/`, no retraining; analytic expected-Jaccard of two independent draws + temperature sweep `p_e(T)=sigma(l_e/T)`). **Finding (reverses prior hypothesis that masks are polarized to 0/1):** the answer-margin DiscoGP masks (DiscoGP-EP) are *not* cleanly polarized — only 0.1–8.4% of edges sit at 0/1, with a large tail in `[0.01, 0.05]`. So per-edge sampling is already FEASIBLE at trained temperature (T=1): expected pairwise Jaccard between two sampled circuits is **0.15–0.34** across IOI/GT/Docstring × {zero, resample, mean}. Caveat for next step: that diversity is driven by the low-probability tail, so sampled circuits are *structurally* diverse but their *faithfulness* is not yet verified — the natural follow-up is to actually draw N circuits, run them through `measure_prune_metrics`, and check faithfulness-vs-IoU (compare against OASR's ~4% IoU floor and Pillar-2 reseeding Jaccard). Temperature sweep shows T=2 lowers Jaccard further (~0.1) but balloons kept-edge count ~3×, so T=1 is already a sensible operating point. Run: `results/edge_prob_diagnostic/seed42/`. **Update (same day):** the follow-up tooling is now built — `edge_prob_diagnostic.py` gained an `--evaluate` mode that draws N circuits, scores each one's faithfulness at its own edge count (`measure_prune_metrics`, TREE_PATCH, `test_edge_counts=[n_kept]`) and the pairwise IoU across samples. Smoke run (Docstring, discovery=zero, N=5, T=1, eval=resample): mean pairwise IoU = 0.28, clean-KL = 1.9 ± 0.25, size ≈ 60 — diverse circuits confirmed; faithfulness *spread* is the thing to characterize at scale. Framed as a third Pillar-2 instrument (alongside reseeding and OASR) and tracked in GitHub issue #5 (Sub-experiment C); full sweep deferred until Pillar 1 (#4) lands.
- **2026-06-15** — Added `experiments/interactive_curves.py` (`save_interactive_curves`): the faithfulness analysis pass now also writes one interactive Plotly HTML per `(task, measure, metric)` under `results/.../interactive/`, complementing the per-discovery static PDFs. Each line is one `(algo × discovery-ablation)` curve (color = algorithm, dash = discovery-ablation, name e.g. `DiscoGP-EP | disc=zero`); an eval-ablation dropdown switches the panel and the Plotly legend toggles individual lines (click = show/hide, double-click = isolate). The point: the *discovery-relativity* story (e.g. DiscoGP-EP trained on zero vs resample alongside EP trained on mean) is read off one figure instead of cross-referencing separate `discovery-<D>/` PDF trees. Anchors (EAP/GT/Random, no discovery sweep) appear once as solid reference lines. 28 HTMLs generated from the 36-cell seed-42 cache (run `400154`).
