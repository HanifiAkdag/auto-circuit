# Edge Pruning Port Fidelity Study — Pre-Registration

**Status:** Written before any Tier A, B or C run. The Tier C endpoints, margins and
decision rules in §6 are fixed **now** and must not be revised after seeing Tier C
results.

**Outcome (2026-08-12, appended after completion; nothing above was altered).** All
50 cells ran. Tiers A and B came out exact. Tier C **does not support port fidelity**
at these margins: Clean KL Div and Answer Pair KL Div both show a genuine discrepancy
(4.5x and 3.9x their margins, p = 1.1e-05 and 0.0015). The cause is difference 2, the
mask-sampling batch dimension, and the `port-sharedz` arm registered in §5 to price it
closes the gap. Full write-up: [`docs/port_validation/edge_pruning.md`](../../port_validation/edge_pruning.md).
One pairing (`port-sharedz` vs `orig`) was added post-hoc and is reported separately in
`results/port_fidelity/ep_tier_c_posthoc/`.

**Action taken (2026-08-12).** The port now defaults to the reference's sampling (`shared_mask_sample=True`) and the spine runs Edge Pruning as published; the per-example estimator is kept behind the flag as a measured variant. The Tier C arms pin the flag explicitly, so the numbers above remain reproducible against the changed default.

**Date:** 2026-08-11
**Comparison target:** `princeton-nlp/Edge-Pruning` @ `a918efd`, expected at
`third_party/edge_pruning`
**Port:** `auto_circuit/prune_algos/edge_pruning.py`

That directory is **gitignored on purpose**: the comparison code is a third-party
dependency and is kept out of this repo. Every result here was produced against commit
`a918efd` exactly; to reproduce, clone it yourself:

```bash
git clone https://github.com/princeton-nlp/Edge-Pruning.git third_party/edge_pruning
git -C third_party/edge_pruning checkout a918efd
unzip -q third_party/edge_pruning/data.zip -d third_party/edge_pruning/
```

The Tier A and Tier B tests skip automatically when it is absent.

## Terminology

This study measures **port fidelity**: whether two programs implement the same
algorithm. It is *not* the thesis's `KL(full‖circuit)` faithfulness. The two never share
a word in the write-up. See [`docs/port_validation/README.md`](../../port_validation/README.md).

## Question decomposition

Two claims requiring different evidence:

1. **Do the two programs compute the same thing?** Deterministic. Tiers A, B.
2. **Do they find the same circuits?** Distributional only, because EP is
   non-deterministic and circuits are non-identifiable. Tier C.

Most evidential weight sits on (1). A weak Tier C result is uninterpretable alone.

---

## 1. Completed: Phase 0, feasibility gate

Run before anything else, because it could have invalidated the schedule. The reference
README warns that `log_alpha` initialisation silently breaks with newer `transformers`
versions, and `requirements-experimental.txt` asks for `transformers==4.54.0` /
`torch==2.7.1` / `datasets==3.3.2` against this environment's `4.42.0` / `2.11.0` /
`4.8.4`.

Measured in the `autocircuit` env (`.scratch/ep_port/ep_smoke.py`, one GPU):

| gate | result |
|---|---|
| `FPT2LMHeadModel` imports and builds | OK |
| `reset_all_log_alphas()` → every one of the 73 (resp. 75) `log_alpha` tensors at `N(10, 0.01²)` | OK, **0 off-spec** — the documented breakage does not occur here |
| masked forward with `corr_x`, `output_writer_states` | OK, `writer_states` `(156, B, S, 768)` |
| `src/prune/fpt2_ioi.py`, 20 steps on the shipped `train` split | OK, 59 s, eval accuracy 0.845, `eval_kl_loss = −2.4e−11` at full density |

**Decision, recorded here rather than later: no separate `ep_ref` conda environment is
needed.** Both implementations run in one process in the `autocircuit` env, so Tier B can
compare them directly rather than through saved tensors.

Edge counts confirmed against `get_num_edges` at the same time, since they drive §2.1:

| configuration | writers | edges |
|---|---|---|
| reference, `--with_embedding_nodes` | 158 | 32,936 |
| reference, without the flag | 156 | **32,046** |
| AutoCircuit (`separate_qkv=True`) | 157 srcs | 32,491 |

`n_readers = 445` in every configuration, which is exactly the gap in both directions.

---

## 2. Catalogue of differences, established by source audit

Each row states the treatment: **removed** (made identical for the comparison),
**controlled** (held equal on both sides), or **remains**.

| # | difference | origin | treatment |
|---|---|---|---|
| 1 | **Embedding writers** | The reference models `tok_embeds` and `pos_embeds` as two separate writers under `--with_embedding_nodes` (`modeling_fpt2.py:54-68`); AutoCircuit has one combined embed source | **Removed** — see §2.1 |
| 2 | **Mask sampling has no batch dimension** | `sample_z_from_log_alpha` (`l0.py:43-48`) draws one `u` of shape `log_alpha.shape`, on CPU in fp32, clamped `(1e-6, 1−1e-6)`, shared across the batch; the port's `sample_hard_concrete` (`tensor_ops.py:12-35`) draws `batch_size` independent samples on device, clamped `(1e-4, 0.9999)` | **Remains**, measured by the `port-sharedz` arm |
| 3 | **Attention output bias `b_O`** | The reference folds `b_O / n_heads` into each head inside the masked path (`_apply_c_proj`, `modeling_fpt2.py:403-414`); AutoCircuit adds `b_O` once, outside the maskable decomposition | **Predicted to cancel under resample** — see §2.2. Measured in Tier B, not assumed |
| 4 | **KL span** | The reference averages over each example's answer span `[start:end]` (`fpt2_ioi.py:126-174`); the port takes the last-token `out_slice` | **Removed by construction** on a single-token split — see §2.3 |
| 5 | **BOS and padding** | The reference prepends no BOS, right-pads with EOS to 64, and passes **no** `attention_mask` (`fpt2_ioi.py:438-477`); the port prepends BOS and left-pads | **Controlled and measured**: the `port-nobos` arm estimates its contribution rather than disclaiming it |
| 6 | **Train/test split** | Reference splits are fixed on disk but are its own 200-example `train`; the port's task machinery builds its own | **Removed**: both sides consume one frozen 704/256 split |
| 7 | **Numeric precision** | No `--bf16`/`--fp16` in any GPT-2 script, so the reference is already fp32 | **Controlled**: fp32 on both sides |
| 8 | **Batch size** | The paper's IOI sweep uses 32 (`ioi_sweep.sh:52`); the thesis spine uses 64 | **Controlled**: 32 on both sides |
| 9 | **4 LR groups → 2** | The reference exposes edge/node × log_alpha/λ learning rates separately (`fpt2_ioi.py:524-565`); the port collapses them to two knobs | **Controlled**: no effect at the paper's all-0.8 values, which is what both sides use |
| 10 | **Pre-composed vs sequential masks** | The port composes node ∨ edge into one effective mask before patching; the reference applies write masks and read masks at different points in the forward pass | **Remains**; Tier B tests whether it changes the function |
| 11 | **Global vs per-tensor thresholding** | The reference calls `deterministic_z_from_log_alpha` per tensor, allocating zeros from that tensor's own expected L0; AutoCircuit's `run_circuits` thresholds globally | **Removed by construction**: Tier C re-scores both arms' scores in one harness, so the same global rule applies to both |
| 12 | **Score extraction** | The reference emits a binary mask via a binary-searched threshold; the port exports the continuous `σ(log_alpha/T·F)` | **Removed**: the two are monotone transforms of the same quantity, so they induce the same ranking (`thesis_notes/limitations.md`, 2026-07-23) |

Differences 1, 2 and 3 were established during this work; the rest were known from
[`docs/related_work/edge_pruning.md`](../../related_work/edge_pruning.md).

### 2.1 The embedding-writer mismatch and why dropping the flag removes it

`get_num_edges` (`modeling_fpt2.py:195-206`) gives 32,936 edges with the flag and 32,046
without, against AutoCircuit's 32,491. The gap is exactly `n_readers = 445` in both
directions, because one extra writer contributes one edge to every reader.

With the flag **off**, the reference adds the embeddings unconditionally at every read
site (`x_q = x_q + embeds.unsqueeze(0)`, `modeling_fpt2.py:562-564`). That *is* "embed
edges always kept". So the primary arm runs the reference without the flag, and
AutoCircuit's 445 embed edges are held always-kept, giving a true 1:1 bijection over
32,046 edges with matching semantics and no invented merge rule.

The paper's IOI sweep does set the flag, so this is a deliberate deviation from the
shipped configuration and becomes the `orig-withembed` mechanism arm rather than a
caveat.

### 2.2 Why the `b_O` argument does not carry over from the DiscoGP study

DiscoGP's Tier B found the entire divergence was `b_O / n_heads`, because zero ablation
*subtracts* a source's contribution and the two implementations disagree about whether
that contribution includes the bias share.

EP is a **resample** method, so the argument must be re-derived rather than inherited.
The reference interpolates instead of subtracting: at each read site every writer
contributes its full value, clean or corrupt, and both carry the bias share, so

```
Σ_h [ z·(y_h + b_O/12) + (1−z)·(y_h^corr + b_O/12) ]  =  Σ_h y_h^{clean or corr} + b_O
```

for **any** mask, which is what AutoCircuit computes with its single unconditional `b_O`.

**Pre-registered prediction:** the `b_O` divergence cancels exactly under resample, and
Tier B needs no `split_attn_out_bias` counterpart. This is a prediction, not a
conclusion. It is measured. If it fails, a bias-matching arm is added and the failure is
reported.

### 2.3 Why the KL-span difference disappears on this split

The reference computes `kl_div(pruned, full, reduction='batchmean', log_target=True)`
over `[start_idx:end_idx]`, then divides by the batch size. For a **one-token** answer
span, `batchmean` divides by 1 and the whole expression reduces to the mean per-example
KL, which is exactly `multibatch_kl_div` at `out_slice` (`tensor_ops.py:199-220`).

AutoCircuit's IOI answers are single-token by construction. Tier A asserts `span == 1`
for every row of the frozen split rather than assuming it, and separately asserts that
the two **do** diverge on a two-token span, so the condition under which they agree is
documented rather than hidden.

---

## 3. Tier A (component equivalence) — planned

`tests/prune_algos/test_edge_pruning_components.py`, CPU, seconds. Each test seeds both
implementations identically and asserts **bitwise** equality:

- `cdf_stretched_concrete`, including the `EPS` clamp boundaries
- `deterministic_z` vs `deterministic_z_from_log_alpha`, including `topk(largest=False)`
  tie behaviour, the 0-dim collapse, and the port's extra `k = min(num_zeros, numel)`
  guard which the reference lacks
- hard-concrete sampling **under a shared `u`**, which isolates the math from the two RNG
  divergences of difference 2; plus a distributional test of the sampler as actually
  called, where bitwise equality is impossible by construction
- the Lagrangian term: sign `(model − target)`, both λ initialised at 0, and the
  `disable_linear_regularization` path
- the sparsity warmup schedule over all 3000 steps, including past the warmup boundary
- the sparsity value: the port's `Σz/n_edges` (ablation convention) against the
  reference's `1 − Σz/n_edges` (keep convention), on matched masks, using the
  reference's own `n_edges` normaliser
- `KL(full‖circuit)` on a one-token span, and its documented divergence on two tokens
- the LR schedule against HF's, 3000 steps / 200 warmup, per parameter group
- optimiser wiring: group count, `maximize` flags, `weight_decay=0.01`, and that only
  mask and λ parameters carry `requires_grad`
- node-mask composition: the port's De Morgan `1−(1−z_e)(1−z_n)` against the reference's
  write-then-read composition
- `log_alpha` initialisation `N(10, 0.01²)` against the port's `N(−10, 0.01²)`, compared
  by moments under the sign flip

This closes a coverage gap that exists regardless of the comparison:
`edge_pruning.py` has no tests today. `tests/test_edge_pruning_smoke.py` is a print
script with no `def test_*` and no cross-implementation comparison; it is superseded.

## 4. Tier B (same-mask function equivalence) — planned

`experiments/port_comparison/{ep_edge_map,ep_forward_equivalence}.py`,
`tests/prune_algos/test_edge_pruning_equivalence.py`.

**The bijection** is not a transpose like DiscoGP's. The reference stores dense
`(n_writers, n_head)` read tensors and enforces causality with buffers
(`attn_read_common_mask`, `modeling_fpt2.py:510-527`), so the map slices to the legal
writers before transposing, handles `mlp_read` (which legally sees its own layer's heads)
and `final_read_log_alphas`, and maps per-block `attn_write_log_alphas (n_head,)` /
`mlp_write_log_alphas (1,)` onto the port's per-`src_idx` node scalars.

The map is **verified functionally, never by hand**: a permutation error inside a shape
survives every shape assertion but cannot survive hundreds of random-mask forward
comparisons.

**Forward equivalence** at densities {0.1, 0.25, 0.5, 0.75, 0.9}, reported against a
**measured numerical floor** (one implementation run twice on differently composed
batches), not against zero. Two arms:

- `raw` — identical `input_ids` to both models; isolates the masking math.
- `pipeline` — **each side builds its own inputs through its own real data path**
  (AutoCircuit `load_datasets_from_json` → `PromptDataLoader`; the reference's
  `DataCollatorIOI`).

The second arm exists because of a specific failure in the DiscoGP study: its Tier B fed
raw token ids to both models to neutralise BOS and padding, and thereby also excluded the
dataloader, where a real bug was hiding. A test that bypasses a code path says nothing
about that path.

Two EP-specific mechanics: resample needs matched **corrupt** tokens, and forcing an
exact binary mask into the reference goes through `get_mask`'s
`edge_threshold_for_deterministic` path (`modeling_fpt2.py:106-113`). Note that the
reference's deterministic mask is otherwise **soft**: at `log_alpha = 10` it is
`σ(12) = 0.999994`, not 1, which alone moves logits by ~3e-04 (measured in Phase 0).

---

## 5. Tier C — shared inputs and arms

### Shared inputs

Both arms consume one frozen split, `datasets/port_fidelity/ioi_frozen.json`
(704 train / 256 test, space-prefixed single-token answers), written for the reference as
an HF `DatasetDict` with `ioi_sentences` / `corr_ioi_sentences` so its own collator and
`Trainer` consume it unmodified.

The reference is driven through its own `src/prune/fpt2_ioi.py`, not a re-expressed
training loop. This is deliberate, and it is the second lesson carried over from the
DiscoGP study: a control that reconstructs the thing under test cannot detect a defect in
how the real code constructs it.

Controlled on both sides: batch size 32, fp32, 3000 optimizer steps, `warmup_steps=200`,
`sparsity_warmup_steps=2500`, `target_edge_sparsity=1.1`, `target_node_sparsity=0.72`,
`--disable_node_loss`, all four learning rates 0.8.

### Arms

| arm | n | purpose |
|---|---|---|
| `port` | 10 | primary |
| `orig` | 10 | primary, without `--with_embedding_nodes` (§2.1) |
| `orig-withembed` | 5 | mechanism: the paper's default flag |
| `port-sharedz` | 5 | mechanism: one mask sample per step, as the reference does (difference 2) |
| `port-nobos` | 5 | measures the BOS-prepend / left-pad factor (difference 5) |
| `port-neginv` | 5 | **negative control**: score-sign convention removed |
| `port-negkl` | 5 | **negative control**: KL direction reversed to `KL(circuit‖full)` |
| `port-nolambda` | 5 | **negative control**: λ frozen at 0, sparsity pressure removed |

50 runs total.

---

## 6. Tier C — endpoints, margins and decision rules (frozen)

### Endpoints

Unit of analysis is the **seed**, n = 10 per primary arm. `C(20,10) = 184,756`, so every
permutation test is **exact** (no Monte Carlo).

Primary budget **k = 2000 edges**, matching the Pillar-2 budget the margins come from.
All circuits are re-scored in **one** AutoCircuit harness under resample discovery /
**resample** evaluation, which is EP's native protocol (and differs from the DiscoGP
study's zero/zero for that reason).

1. **Functional (primary).** Clean KL Div, Logit Diff Percent, Answer Pair KL Div.
2. **Structural.** Jaccard at matched k (`auto_circuit/metrics/analysis_utils.py`).
3. **Score-level.** Spearman ρ over the aligned scores, aligned **by module name**
   (never flattened — `PruneScores` dict order is not stable across runs).
4. **Operating point.** Edge density under the reference's own discretisation rule,
   i.e. the fraction kept by `l0_utils.deterministic_z` applied per tensor. DiscoGP's
   `score > 0.5` rule is **not** reused: the reference's operating point is a binary
   search matching the expected L0 (`src/eval/ioi.py:195-232`), not a fixed threshold.

### Equivalence margins

δ is the **within-implementation SD already measured in Pillar 2** for
Edge Pruning × IOI × resample-discovery / resample-eval / faithfulness at k = 2000. The
claim therefore reads: *the shift between implementations is smaller than the method's
own run-to-run noise.*

**This fixes a rule, not a number.** A Pillar-2 re-run obliges recomputing every
data-derived margin before any Tier C verdict is read. That is not hypothetical: when the
2026-08-11 `data.py` fix forced Pillar 2 to be re-run, every DiscoGP margin changed and
the recomputation cost that study one endpoint.

Source: `results/stability/stability_v2_2026-08-11/` (`functional.csv` `metric_sd`;
`structural.csv` `jaccard_sd`), n = 5 seeds, verified post-fix via its `manifest.json`
git state.

| endpoint | Pillar-2 mean @ k=2000 | δ (= Pillar-2 SD) | pre-fix value, for the record |
|---|---|---|---|
| Clean KL Div | 0.15545 | **0.02595** | 0.00901 |
| Logit Diff Percent | 78.3484 | **5.3445** | 3.9621 |
| Answer Pair KL Div | 0.05028 | **0.01314** | 0.01413 |
| Jaccard @ k=2000 | 0.62238 | **0.008246** | 0.008928 |
| Spearman ρ | — | **0.10** (no counterpart; fixed a priori) | — |
| density | — | **0.02** absolute (no counterpart; fixed a priori) | — |

Unlike DiscoGP's, EP's post-fix margins are **wider** than pre-fix on three of four
endpoints. They remain roughly 30× tighter in absolute terms, because EP is a far more
stable method under resample (Pillar-2 Jaccard 0.62 against DiscoGP's 0.19).

### Decision rules

For each endpoint:

- **Difference test:** exact two-sample permutation test on the difference of arm means
  (all 184,756 label assignments).
- **Equivalence test:** 90% bootstrap CI for Δ (resampling **seeds**, 10,000 draws).
  Equivalence is declared iff the CI lies entirely within ±δ. This is TOST at α = 0.05.
- Report arm means with 95% bootstrap CIs alongside.

**Jaccard is a paired-dependent statistic** — each seed appears in many pairs, so
permuting *pairs* would be invalid. Permute **seed labels** and recompute the
between/within split each time, with statistic
`T = mean(within-pair J) − mean(between-pair J)`.

**Multiplicity:** Holm correction across the four endpoint families, applied to both the
difference tests and the TOST rejections.

### Interpretation grid, fixed in advance

| difference test | equivalence test | reading |
|---|---|---|
| not significant | equivalent | port fidelity supported |
| not significant | not equivalent | underpowered; report as inconclusive, **not** as support |
| significant | equivalent | a real but practically negligible shift |
| significant | not equivalent | a genuine discrepancy; investigate |

The second row is the one that must not be reported as success.

### Negative controls

Without demonstrated power the study proves nothing. Each control runs through the
identical pipeline and the achieved detections are reported as a sensitivity table
(endpoint × control).

- `port-neginv` — score-sign convention removed. Must be detected.
- `port-negkl` — KL direction reversed. A plausible real port bug, and the sharpest test
  of whether the endpoints have power against an objective-level defect.
- `port-nolambda` — λ frozen at 0, so nothing drives sparsity. Must be detected, most
  obviously on density.
- Random-baseline circuits — extreme control.

**Ship-blocking:** an endpoint that detects no control has no demonstrated power and is
reported as **uninformative** rather than as evidence of equivalence. This is not
optional and is decided here, before any result is seen.

### Power

n stays at **10 per primary arm**, chosen for exactness: `C(20,10) = 184,756` is under
`fidelity_stats.MAX_EXACT = 400,000`, while n = 11 exceeds it and would silently drop to
Monte Carlo.

The cost is stated in advance. Because δ *is* the per-seed SD, the 90% CI half-width is
about `1.645·σ·√(2/n) = 0.74δ` at n = 10, so TOST can only succeed if the true shift is
below roughly `0.26δ`. This is exactly what left DiscoGP's Clean KL inconclusive on
interval width rather than on measured disagreement, and EP's much tighter margins make
it more likely here, not less. The design is powered to rule out shifts of roughly one
pooled SD; it cannot certify bit-level identity and does not claim to. Achieved power is
evidenced empirically by the negative controls, not by a nominal calculation.

---

## 7. Scope and limits (fixed in advance)

- **IOI only.** The reference also ships Greater-Than (`fpt2_gt.py`) and Gendered
  Pronoun, and GT is genuinely reachable — its KL is restricted to the 100 two-digit
  tokens at a single index, so it would exercise a different objective path from IOI's
  single-token span. It is excluded to keep this study directly comparable to the DiscoGP
  one and because the Pillar-2 margins are best characterised on IOI. This is a chosen
  limit, not an unreachable one, and is stated as such.
- **Edge-only in effect.** EP never prunes weights, so unlike DiscoGP there is no
  scope gap on that axis.
- **RNG streams cannot be aligned** across implementations, so Tier C is distributional
  by necessity.
- **Low cross-implementation Jaccard would not be evidence of a port defect.** The EP
  paper's own Appendix D reports pairwise edge IoU of only 0.5–0.7 across 12 seeds of the
  *same* implementation. Pillar 2 measures the port's own within-implementation Jaccard
  at 0.62 (k = 2000), consistent with that. This is why the functional endpoints are
  primary and the structural one secondary, and it is decided here rather than after
  seeing results.
- **Metrics are never quoted across harnesses.** The reference's training KL is
  `batchmean` over an answer span and its eval KL uses `reduction="sum"` per example;
  neither is AutoCircuit's `multibatch_kl_div`. All numbers reported are recomputed in
  one harness.
- **Pillar-2 margins are estimated from n = 5 seeds** and are themselves noisy. They are
  pre-registered as-is rather than tuned.
- **Selection at matched k is deterministic, not randomised.** `budget_mask` and
  `prune_scores_threshold` admit a tie block straddling k whole. EP's continuous scores
  effectively never tie, so this is immaterial here; it is noted because
  `docs/port_validation/discogp.md` describes this path as "randomized-tie top-k", which
  is incorrect and is being corrected.
