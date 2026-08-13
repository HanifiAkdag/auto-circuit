# Edge Pruning port validation

Validation of this fork's Edge Pruning implementation
(`auto_circuit/prune_algos/edge_pruning.py`) against the authors' released code,
`princeton-nlp/Edge-Pruning` @ `a918efd`.

Written for reuse in the thesis Methodology chapter and appendix. Terminology, tier
design and the role of negative controls are defined once in
[`README.md`](README.md).

- **Pre-registration:** `docs/superpowers/specs/2026-08-11-edge-pruning-port-fidelity.md`, written before any run
- **Scope:** IOI only. Greater-Than is reachable in the reference (`fpt2_gt.py`) and is excluded by choice, not by necessity, to keep this study comparable to the DiscoGP one
- **Verdict:** Tiers A and B are **exact**: the components agree bitwise and, given the same circuit and inputs, the two programs compute the same function to the numerical floor, with no attention-bias correction needed. Tier C **does not support port fidelity** at the pre-registered margins, and the cause is identified: the port draws an independent hard-concrete mask sample per example where the reference draws one per forward and shares it (difference 2). Matching that single choice closes the gap (§5.5), and the reference's sampling has since become the library default. Margins were recomputed after that change, as the pre-registration requires, and all §5.4 verdicts survive it (§5.2)

---

## 1. Catalogue of differences between the implementations

Established by source audit and confirmed by measurement. Each row states whether it
was **removed** (made identical for the comparison), **controlled** (held equal on
both sides), or **remains**.

| # | difference | origin | treatment |
|---|---|---|---|
| 1 | **Embedding writers** | The reference models `tok_embeds` and `pos_embeds` as two separate writers under `--with_embedding_nodes` (`modeling_fpt2.py:54-68`); AutoCircuit has one combined embed source | **Removed** by dropping the flag (§1.1) |
| 2 | **Mask sampling has no batch dimension** | `sample_z_from_log_alpha` (`l0.py:43-48`) draws one `u` of shape `log_alpha.shape`, on CPU in fp32, clamped `(1e-6, 1−1e-6)`, shared across the batch; the port's `sample_hard_concrete` draws `batch_size` independent samples on device, clamped `(1e-4, 0.9999)` | **Remains**; measured by the `port-sharedz` arm |
| 3 | **Attention output bias `b_O`** | The reference folds `b_O / n_heads` into each head inside the masked path (`_apply_c_proj`, `modeling_fpt2.py:403-414`); AutoCircuit adds `b_O` once, outside it | **Cancels under resample** (§1.2), confirmed by measurement |
| 4 | **KL span** | The reference averages over each example's answer span; the port takes the last-token `out_slice` | **Removed by construction** on a single-token split (§1.3) |
| 5 | **BOS and padding** | The reference prepends no BOS, right-pads with EOS to 64, passes no `attention_mask`; the port prepends BOS and left-pads | **Remains and is measured**: it is the whole of the residual Tier B gap (§3.3) |
| 6 | **Train/test split** | The reference ships its own 200-example `train` split | **Removed**: both sides consume one frozen 704/256 split |
| 7 | **Numeric precision** | No `--bf16`/`--fp16` in any GPT-2 script | **Controlled**: fp32 on both sides (the reference already is) |
| 8 | **Batch size** | The paper's IOI sweep uses 32; the thesis spine uses 64 | **Controlled**: 32 on both sides |
| 9 | **4 LR groups → 2** | The reference exposes edge/node × log_alpha/λ rates separately | **Controlled**: no effect at the paper's all-0.8 values, which both sides use |
| 10 | **Pre-composed vs sequential masks** | The port composes node ∨ edge into one effective mask before patching; the reference applies write and read masks at different points in the forward pass | **Remains**; Tier B shows it does not change the function |
| 11 | **Global vs per-tensor thresholding** | The reference allocates zeros per mask tensor from that tensor's own expected L0; AutoCircuit thresholds globally | **Removed by construction**: Tier C re-scores both arms in one harness, so one global rule applies to both |
| 12 | **Score extraction** | The reference emits a binary mask via a binary-searched threshold; the port exports the continuous `σ(log_alpha/T·F)` | **Removed**: monotone transforms of the same quantity, so the induced ranking is identical |

Differences 1, 2 and 3 were established during this work; the rest were known from
[`docs/related_work/edge_pruning.md`](../related_work/edge_pruning.md).

### 1.1 The embedding-writer mismatch

`get_num_edges` gives **32,936** edges with `--with_embedding_nodes` (the paper's IOI
sweep default) and **32,046** without, against AutoCircuit's **32,491**. The gap is
exactly `n_readers = 445` in both directions, because one extra writer contributes
one edge to every reader.

With the flag off, the reference adds the embeddings unconditionally at every read
site (`x_q = x_q + embeds.unsqueeze(0)`, `modeling_fpt2.py:562-564`). That *is*
"embed edges always kept". So the primary arm drops the flag and holds AutoCircuit's
445 embed edges kept, giving a true 1:1 bijection over 32,046 edges with matching
semantics and no invented merge rule. The paper's configuration becomes the
`orig-withembed` mechanism arm rather than a caveat.

### 1.2 Why `b_O` cancels here, and why that could not be assumed

The DiscoGP study found `b_O / n_heads` to be its *entire* Tier B divergence. The
same structural difference exists in Edge Pruning, so the natural move is to inherit
the conclusion. That would have been wrong: the two studies differ in ablation type,
and the ablation type is exactly what decides the question.

Zero ablation *subtracts* a source's contribution, and the two implementations
disagree about whether that contribution includes the bias share. Resample
*interpolates*: every writer contributes its full value, clean or corrupt, and both
branches carry the share, so

```
Σ_h [ z·(y_h + b_O/H) + (1−z)·(y_h^corr + b_O/H) ]  =  Σ_h y_h^{clean or corr} + b_O
```

for any mask, which is what AutoCircuit computes with its single unconditional
`b_O`. This was pre-registered as a prediction and then measured (§3.2): the
head-source probes show no excess whatever.

The property that discriminates is the **replacement value**, not the ablation's
name: whether the substituted activation is itself computed through the same per-head
decomposition and so carries its own `b_O / H` share. Substituting zero does not.
Confirmed by re-running the same probe under `BATCH_TOKENWISE_MEAN`, which also lands
at the floor (head sources 2.59e-04 and 2.52e-04), so the divergence is unique to
zero ablation among every ablation AutoCircuit implements. See
[`README.md`](README.md).

### 1.3 Why the KL-span difference disappears on this split

The reference computes `kl_div(pruned, full, reduction='batchmean', log_target=True)`
over `[start_idx:end_idx]`, then divides by the batch size. For a **one-token**
answer span `batchmean` divides by 1, and the expression reduces to the mean
per-example KL, which is exactly `multibatch_kl_div` at `out_slice`.

AutoCircuit's IOI answers are single-token by construction. Tier A asserts
`span == 1` for every row of the frozen split rather than assuming it, and separately
asserts that the two **do** diverge on a two-token span, so the condition under which
they agree is documented rather than hidden.

---

## 2. Tier A: component equivalence

`tests/prune_algos/test_edge_pruning_components.py`, 37 tests, CPU, seconds.

Each test seeds both implementations identically and asserts **bitwise** equality.
Where a component could not be isolated as a function, the reference's real code is
driven on a 2-layer randomly-initialised `FPT2LMHeadModel` rather than transcribed
into the test, since a transcription would only check that the test author copied
correctly.

- the hard-concrete constants, and `cdf_stretched_concrete` across the `EPS` clamp boundaries
- `deterministic_z` vs `deterministic_z_from_log_alpha`, including `topk(largest=False)` tie behaviour and the kept value `σ(1.2·log_alpha)`
- hard-concrete sampling under a **shared `u`**, isolating the transform from the two RNG divergences of difference 2; plus a distributional test of the samplers as actually called, where bitwise equality is impossible by construction
- the Lagrangian term, compared against the value the reference's own forward computes
- the sparsity warmup schedule over all 3000 steps, against the reference's real unbound method
- the sparsity conventions (`Σz/n_edges` ablation vs `1 − Σz/n_edges` keep)
- `KL(full‖circuit)` on a one-token span, and its documented divergence on two tokens
- the LR schedule against HuggingFace's, per parameter group
- optimiser wiring: four groups, `maximize` on the λ groups only, `weight_decay=0.01`, and only mask/λ parameters trainable
- node-mask composition, and `log_alpha` initialisation under the sign flip

**Result: all pass**, after one fix to the port. These are the cheapest and strongest
evidence in the study, and they close a coverage gap that existed regardless of the
comparison: `edge_pruning.py` had no tests, and `tests/test_edge_pruning_smoke.py`
was a print script with no `def test_*` and no cross-implementation comparison at all.

### 2.1 The one thing Tier A changed in the port

The sparsity-target ramp was computed as `start + (target − start) · (step / warmup)`
where the reference computes `start + (target − start) · step / warmup`. The two
differ in the last ulp, which is invisible in any result but enough to fail a bitwise
test. The port now mirrors the reference's association exactly
(`edge_pruning.py:_target_sparsity`). This is a fidelity improvement of no practical
consequence, and it is recorded because "we made the test pass" and "we made the code
match" are different claims.

---

## 3. Tier B: same function given the same circuit

`experiments/port_comparison/{ep_edge_map,ep_forward_equivalence}.py` and
`tests/prune_algos/test_edge_pruning_equivalence.py`, 13 tests.

### 3.1 The edge bijection

Not the transpose the DiscoGP map was. The reference keeps every read mask at full
writer width and enforces causality with buffers (`attn_read_common_mask`), so the
map **slices** to the legal writers before transposing. Over GPT-2 small with
`separate_qkv=True`: **156 writers, 445 readers, 32,046 reference edges** against
AutoCircuit's **157 sources and 32,491 edges**, the difference being one embed edge
per reader.

Node masks map from per-block `attn_write_log_alphas (n_head,)` and
`mlp_write_log_alphas (1,)` onto the port's per-`src_idx` scalars.

The map is **not** hand-verified. It is verified functionally in §3.2, because a
permutation error within a shape would survive every shape check but cannot survive
hundreds of random-subset forward comparisons.

### 3.2 Forward equivalence

64 prompts × 40 random masks × 5 densities, with exact binary circuits forced into
the reference through its own `edge_threshold_for_deterministic` path. Reported
against a **measured floor**, not against zero: at full density nothing is ablated,
so both are plain GPT-2 and the gap there is pure float arithmetic between two
implementations.

Comparisons are in **log-probability** space. `load_tl_model` builds the model with
`center_unembed=True`, so AutoCircuit's logits are centred per position and the
reference's are not; the difference is a per-position constant that leaves argmax and
softmax untouched but swamps a raw-logit comparison (114 against a logit scale of
21). Log-probabilities are also what both objectives consume.

| arm | full density | median max&#124;Δ log p&#124; | worst | argmax |
|---|---|---|---|---|
| `raw` (identical ids) | 2.7e-04 | **2.6e-04** | 3.7e-04 | ≥98.4% |
| `raw-bos` (BOS on the port side only) | 2.55 | 3.39 | 5.36 | ≥65.1% |
| `pipeline` (each side through its own dataloader) | 2.76 | 2.83 | 5.45 | ≥59.4% |

**Result.** In the `raw` arm the implementations agree to the numerical floor at
every density from 0.1 to 0.9, and the masked deviations are indistinguishable from
the unmasked one. This single test validates the bijection, the mask convention, and
resample-ablation equivalence at once, since the reference masks by interpolating
toward the corrupted writer while AutoCircuit adds `mask·(corrupt − clean)` to a
clean sum, and those coincide only if the decomposition matches term for term.

Single-edge probes, chosen by **source kind** (an index-based probe would only ever
hit the embedding source and silently test nothing):

| source of the ablated edge | max &#124;Δ log p&#124; |
|---|---|
| embedding | 2.73e-04 |
| head, layer 0 | 2.57e-04 |
| MLP, layer 0 | 2.63e-04 |
| head, layer 10 | 2.63e-04 |

No head-specific excess. The `b_O` prediction of §1.2 holds: under resample the bias
share cancels exactly. Re-running the identical probe under `BATCH_TOKENWISE_MEAN`
gives the same answer (embed 2.21e-04, head L0 2.59e-04, MLP L0 2.56e-04, head L10
2.52e-04), so the cancellation is a property of substituting a genuine activation
rather than of resample specifically. For contrast, the same probe in the DiscoGP
study under zero ablation gave 1.6e-05 for an embedding source against 8.9e-02 and
7.4e-01 for head sources at layers 0 and 11.

### 3.3 What the remaining gap is, and what it is not

The `raw-bos` and `pipeline` arms are the same size, and both are **already present
at full density**, where nothing is ablated. So the residual is not the masking
machinery: it is difference 5, the input convention. Prepending BOS on the port side
alone, with no padding involved anywhere, reproduces the whole of the gap that
appears when both sides run their real dataloaders.

The `pipeline` arm exists because of a specific failure in the DiscoGP study, whose
Tier B fed raw token ids to both models to neutralise BOS and padding, and thereby
also excluded the dataloader, where a real bug was hiding. A test that bypasses a code
path says nothing about that path.

---

## 4. A second AutoCircuit defect, found while building Tier B

GPT-2's pad token **is** its BOS token (50256), and AutoCircuit left-pads.
`HookedTransformer.forward` recomputes the attention mask itself, and when it is
handed a raw token tensor it takes `prepend_bos` from **`cfg.default_prepend_bos`**
(True) rather than from however those tokens were actually built
(`HookedTransformer.py:286-294`). `get_attention_mask` then deliberately un-masks the
last leading pad, on the assumption that it is a prepended BOS
(`transformer_lens/utils.py:976-978`).

So when a caller asks `load_datasets_from_json` for `prepend_bos=False`, the tokens
have no BOS but the model is never told, and one genuine pad token per row is
re-enabled. Measured on 16 ragged IOI prompts, comparing a padded batch against the
same prompts run one at a time:

| tokens built with | `cfg.default_prepend_bos` | max &#124;Δ log p&#124;, batched vs solo |
|---|---|---|
| `prepend_bos=True` | True | 2.5e-05 |
| `prepend_bos=True` | False | 2.1e-05 |
| **`prepend_bos=False`** | **True** (the default) | **2.63** |
| `prepend_bos=False` | False | 2.9e-05 |

Visible directly in the mask: for a 14-token prompt padded to 20,
`get_attention_mask(..., prepend_bos=True)` attends **15** positions, against exactly
14 for `prepend_bos=False`.

This is an AutoCircuit defect rather than a TransformerLens one: TL behaves correctly
given what it is told, and AutoCircuit never threads its own `prepend_bos` through to
the forward (`data.py:334-339` tokenises with it, then calls `model(tokens)` without
it). The fix belongs at the call site, and AutoCircuit owns both halves of the
mismatch. **It is deliberately not applied here**, because Tier C was already running
when this was found and patching mid-flight would leave cached cells computed under
two different behaviours. Enforced meanwhile by
`test_left_padding_needs_bos_to_be_batch_invariant`.

**Severity: latent, not active.** `tasks.py:215` sets `prepend_bos=has_tokenizer`, so
every shipped tokenizer task passes `True` and no built-in task reaches this path. It
bites only callers who explicitly pass `prepend_bos=False`, which is exactly what the
`port-nobos` arms of both port-fidelity studies do. It is also conditional on padding
actually being present, and touches at most one token per padded row; the frozen IOI
split is ragged (15-21 tokens), so it is present here.

**It is a third, distinct trigger of the same TransformerLens heuristic**, and the
three should not be conflated:

| | trigger | status |
|---|---|---|
| (a) | KV-cache prefix constancy: `diverge_idx` computed within clean/corrupt pairs while one cache built from the first `batch_size` rows is reused | fixed by [#17](https://github.com/UFO-101/auto-circuit/issues/17) / PR #18 |
| (b) | `tail_divergence` stripping splits the leading pad run, so the fragment's last position is mistaken for the BOS | only *avoided* on GPT-2, by declining to strip at all |
| (c) | caller and TL disagree about `prepend_bos` (this one) | **unfixed**; lives on the ordinary forward path and would survive deleting `tail_divergence` outright |

Two consequences, both stated rather than absorbed:

- **`prepend_bos=False` is not a safe knob on ragged prompts.** It does not isolate
  the BOS factor; it adds a second, larger artifact. That is why the BOS factor is
  measured here on uniform-length prompts instead (§3.3), and why the Tier C
  `port-nobos` arm is reported as an input-convention arm rather than a clean BOS
  measurement.
- **The arm measures "what happens when BOS is turned off in this harness", defect
  included** -- not "what the BOS convention costs". The compound is not
  convention-plus-convention; it is one convention plus a mis-masking defect that
  switching BOS off activates.
- **The `orig` arm cannot be contaminated.** It runs through the reference's own
  dataloader, which neither prepends BOS nor left-pads, so it never enters this code
  path; and Tier C scoring happens in the AutoCircuit harness with BOS.
- **The DiscoGP study's `port-nobos` arm has the same confound** and its reading
  should be qualified the same way. Its primary and control arms are unaffected,
  since they all run with BOS.

This is separate from the `tail_divergence` defect reported as
[UFO-101/auto-circuit#17](https://github.com/UFO-101/auto-circuit/issues/17).

---

## 5. Tier C: matched training runs

`experiments/port_comparison/{ep_frozen_ioi,ep_run_cells,ep_cells}.py`, analysed by
`fidelity_analysis.py --study ep`.

### 5.1 Design

Both arms consume one frozen 704/256 split, written once in this repo's schema and
once as a HuggingFace `DatasetDict` so the reference's own collator and `Trainer`
consume it unmodified. The reference is driven through its own
`src/prune/fpt2_ioi.py` as a subprocess, not through a re-expressed training loop:
a control that reconstructs the thing under test cannot detect a defect in how the
real code constructs it.

Controlled on both sides: batch size 32, fp32, 3000 optimizer steps,
`warmup_steps=200`, `sparsity_warmup_steps=2500`, `target_edge_sparsity=1.1`,
`target_node_sparsity=0.72`, `--disable_node_loss`, all four learning rates 0.8.

| arm | n | purpose |
|---|---|---|
| `port` | 10 | primary |
| `orig` | 10 | primary, without `--with_embedding_nodes` |
| `orig-withembed` | 5 | mechanism: the paper's default flag |
| `port-sharedz` | 5 | mechanism: one mask sample per step, as the reference draws it |
| `port-nobos` | 5 | the port's input convention (see §4 for the confound) |
| `port-neginv` | 5 | **negative control**: score-sign convention removed |
| `port-negkl` | 5 | **negative control**: KL direction reversed |
| `port-nolambda` | 5 | **negative control**: Lagrangian frozen, no sparsity pressure |

The unit of analysis is the **seed**. With n = 10 per primary arm, C(20,10) = 184,756,
so every permutation test is **exact**. Jaccard values are pairwise and therefore
dependent, so seed *labels* are permuted and the within/between split recomputed each
time; permuting pairs would be wrong. Scores are aligned **by module name**, never
flattened.

Eval ablation is **resample**, EP's native protocol, not DiscoGP's zero.

### 5.2 Equivalence margins

δ is the **within-implementation standard deviation already measured in Pillar 2**
for EP × IOI × resample/resample × faithfulness at k = 2000. The claim then reads:
*the shift between implementations is smaller than the method's own run-to-run
noise.*

The pre-registration fixes this **rule**, not the numbers, so a Pillar-2 re-run
obliges recomputing every data-derived margin. That obligation has now been
discharged twice:

| endpoint | δ v1 pre-fix | δ v2 post-fix | δ v3 provisional | δ v4 current |
|---|---|---|---|---|
| Clean KL Div | 0.00901 | 0.02595 | 0.03259 | **0.02808** |
| Logit Diff Percent | 3.9621 | 5.3445 | 16.3605 | **10.8835** |
| Answer Pair KL Div | 0.01413 | 0.01314 | 0.05142 | **0.05500** |
| Jaccard | 0.008928 | 0.008246 | 0.00540 | **0.00663** |

v1 → v2 is the 2026-08-11 `data.py` fix. v2 → v3 is **this study's own change**:
`shared_mask_sample=True` became the library default on 2026-08-12 as a result of
§5.5, which invalidated all 45 EP stability cells.

**v3 was withdrawn as provisional.** The re-run that produced it constrained on GPU
*memory* and scattered the 54 cells across four architectures (a180/ampere,
a6000/ampere, l40/lovelace, titanrtx/turing), where the rest of Pillar 2 pins
`gpu_model:a180` for exactly this reason. δ *is* a standard deviation, so node
variance entered the quantity being measured. v4 re-runs the same cells pinned to
a180, under the 2026-08-13 `no_padding` gate (`2a9c0e0`), with every manifest
verified by `git merge-base` to carry that commit.

**Hardware scatter inflated the margins but did not manufacture the effect.** The two
margins that moved most pull back at v4 (Clean KL by 14%, Logit Diff by 33%), so v3
did overstate them. The structural result is untouched: mean pairwise Jaccard is
0.2409 at v3 against 0.2445 at v4, and no EP cell moved by more than 0.028 across all
nine task × ablation combinations.

**The v4 margins are what the verdicts below are read against, and three of the four
are still wider than v2** (1.1×, 2.0×, 4.2×). Adopting the reference's sampling made
EP genuinely noisier across seeds, so an equivalence claim is easier to make at v4
than at v2 for a reason that has nothing to do with the port being closer to the
reference. That ordering is stated before the results rather than after them.

The Jaccard margin instead *narrows*, and that one must not be read as improved
consistency: mean pairwise Jaccard fell from 0.622 to 0.241 over the same change, and
overlap that low has less room to vary. It is tightened by a floor effect.

This document previously called EP "a far more stable method under resample" than
DiscoGP, citing 0.62 against 0.19. **That comparison is withdrawn**, and not merely
updated: EP's 0.62 was measured under the port's own sampling, and the DiscoGP number
it was set against comes from cells discovered pre-`data.py`-fix, so the two sides
were never at one configuration. A replacement claim needs the non-EP cells re-run;
see §5.6.

**The power cost of that tightness is stated in advance.** Because δ *is* the
per-seed SD, the 90% CI half-width is about `1.645·σ·√(2/n) = 0.74δ` at n = 10, so
TOST can only succeed if the true shift is below roughly `0.26δ`. n stays at 10
because C(20,10) is under `fidelity_stats.MAX_EXACT = 400,000` and keeps every
permutation test exact; n = 11 would silently fall back to Monte Carlo.

### 5.3 Operating point

Density is **not** measured at DiscoGP's `score > 0.5`. Edge Pruning's own
discretisation is not a fixed threshold: it zeroes, per mask tensor, as many entries
as the stretched-concrete distribution expects to be zero, then binary-searches the
cut so the discrete sparsity matches the continuous one. The endpoint applies
`l0_utils.deterministic_z` per tensor, which is that rule.

### 5.4 Results, primary comparison

`port` vs `orig`, both as shipped, on the frozen split, all circuits re-scored in one
harness at k = 2000 under resample evaluation.

| endpoint | port | orig | difference | 90% CI | δ | verdict |
|---|---|---|---|---|---|---|
| edge density | 0.0108 | 0.0248 | −0.0141 | [−0.0143, −0.0138] | 0.02 | **differs** (p = 1.1e-05); see the margin caveat below |
| Clean KL Div | 0.1419 | 0.2582 | −0.1163 | [−0.144, −0.091] | 0.02808 | **genuine discrepancy** (p = 1.1e-05) |
| Logit Diff Percent | 78.29 | 89.87 | −11.58 | [−23.2, −0.85] | 10.8835 | inconclusive (p = 0.12) |
| Answer Pair KL Div | 0.0506 | 0.1014 | −0.0508 | [−0.082, −0.025] | 0.05500 | **genuine discrepancy** (p = 0.0015) |
| Jaccard (within − between) | 0.2133 | — | — | — | 0.00663 | not equivalent (p = 1.1e-05) |
| Spearman (within − between) | 0.0505 | — | — | — | 0.10 | equivalent (p = 1.1e-05) |

**Port fidelity is not supported at the pre-registered margins.** This is a real
disagreement, not an interval-width problem: the Clean KL gap is 4.1× its margin,
with p = 1.1e-05. Saying so is the point of having fixed the margins in advance.

**The density margin is mis-calibrated, and its `equivalent` flag must not be read as
fidelity support.** δ_density = 0.02 was chosen a priori on an absolute scale, unlike
the four data-derived margins. Observed densities live at 0.011 to 0.025, so ±0.02
spans the entire plausible range and almost nothing could fail it. What the test
actually reports here is *"a highly significant difference that happens to be smaller
than a margin we declared negligible"*: the CI is [−0.0137, −0.0134], nowhere near
zero, at p = 1.1e-05. In circuit terms the port keeps roughly 370 edges of ~32k where
the reference keeps roughly 800, a factor of 2.2. That is a substantive disagreement
about circuit size. The pre-registration locked the margin in, so it is reported as
specified and flagged as a design fault rather than silently rescaled. A relative
(ratio) margin, or one derived from measured seed spread like the others, would have
been correct.

**These verdicts are stable across all three margin revisions.** Clean KL is a
discrepancy at 4.5δ (v2), 3.6δ (v3) and 4.1δ (v4). Answer Pair KL stays significant
at p = 0.0015 throughout, though at v4 its gap (−0.0508) is only 0.92× its margin, so
it now fails TOST rather than clearing the difference test in the other direction:
read it as *not equivalent*, not as a large effect. Logit Diff Percent has been
inconclusive at every revision.

Two things it is **not**. It is not a defect hunt gone cold — §5.5 identifies the
cause. And it is not the port being worse: the port's circuits have *lower* KL to the
full model at matched k (0.142 against 0.258), i.e. they are more faithful, and they
are sparser at their own operating point (1.08% against 2.48% edge density).

### 5.5 The cause is a catalogued difference, and the arm that was pre-registered to measure it closes the gap

Difference 2: the reference draws **one** hard-concrete sample per forward and shares
it across the batch (`l0.py:43-48`); the port draws `batch_size` independent samples.
The `port-sharedz` arm changes only that, and was pre-registered before any Tier C
run precisely to price it.

| comparison | Clean KL difference | p | reading |
|---|---|---|---|
| `port` vs `orig` | **−0.1163** | 1.1e-05 | genuine discrepancy |
| `port-sharedz` vs `orig` | **−0.0078** | 0.77 | not significant, inside δ |

Adopting the reference's sampling moves Clean KL from 0.1419 to 0.2504 against the
reference's 0.2582 — from a 4.1δ discrepancy at p ≈ 1e-05 to a 0.28δ difference at
p = 0.77. Answer Pair KL moves the same way (0.0506 → 0.0884 against 0.1014). The
structural endpoints move too: `port-sharedz`'s within-arm Jaccard is 0.265 against
`orig`'s 0.392, where the port's own is 0.642.

At the v4 margins, one endpoint in this pairing clears TOST: **Answer Pair KL Div
reads *fidelity supported*** (difference −0.013, 90% CI [−0.046, 0.013], δ = 0.0550,
p = 0.71). Spearman is equivalent. Density is flagged equivalent too, but per the
caveat in §5.4 that flag is uninformative and the underlying difference is real
(0.0113 vs 0.0248, p = 0.0003). This verdict first appeared
at v3 and survived the correction to v4, which matters because v3's margins were the
inflated ones: it does not depend on the hardware artefact.

Clean KL still reads **inconclusive**, and the reason is worth naming precisely
because it was predicted in the pre-registration rather than discovered here. Its 90%
CI is [−0.040, 0.022], a half-width of 0.031 against δ = 0.0281, so the interval now
misses the margin by 0.0031 where at v3 it missed by 0.0016. Tightening the margins
made this endpoint slightly worse, not better. This is the `0.74δ` power ceiling of
§5.2 doing exactly what it
was said in advance it would do: when δ *is* the per-seed SD, TOST needs the true
shift under about `0.26δ`, and n = 5 on this arm makes the ceiling tighter still. The
same trap left DiscoGP's Clean KL inconclusive. It is a limit of the design, not
evidence against the port.

So the claim the data support is the weaker but sufficient one: **the discrepancy
disappears when the sampling is matched.** The difference test flips from p ≈ 1e-05
to p = 0.77, one endpoint reaches formal equivalence, and no endpoint detects a
residual gap.

*This pairing was added post-hoc, after seeing the primary result, and is reported in
`results/port_fidelity/ep_tier_c_posthoc/` rather than folded into the pre-registered
verdict above. The arm itself was pre-registered; the pairing of it against `orig` was
not.*

### 5.5.1 Does the paper say which one is right?

Checked against the paper (arXiv 2406.16778) rather than assumed. **It does not
specify, and neither implementation is violating a stated specification.**

The hard-concrete formulation in §3 is written as

```
u ~ Uniform(eps, 1 - eps)
s = sigma( (1/beta) * log(u / (1-u)) + log(alpha) )
s~ = s * (r - l) + l
z  = min(1, max(0, s~))
```

with **u** bold, i.e. a vector over edges, and `log alpha` applied element-wise.
**No batch index appears anywhere** — not in these four lines, not in the Lagrangian,
and not in Equation 3 itself. Appendix A lists learning rates, step counts, batch
sizes and sparsity targets, but never says how many mask samples are drawn per step.

So each side is a defensible reading of the same silent notation:

- **The reference's reading.** `u` has the shape of `log_alpha`, exactly as written,
  drawn once per forward. This also matches the lineage the paper builds on —
  it frames the method as structured pruning (Wang et al. 2020; Xia et al. 2022,
  CoFi), where the mask *is* the subnetwork being produced, a property of the model
  rather than of an example, so one draw per step is the natural convention.
- **The port's reading.** Equation 3 is written per example ("for each example `x`,
  the user provides a corrupted example `x~`"), so on a strict reading each example's
  forward pass carries its own `z`. AutoCircuit's wrapper samples this way for every
  hard-concrete algorithm it hosts (`patch_wrapper.py:135`, commented "Sample
  distribution for each batch element"), so the port inherited the house convention
  rather than inventing one.

Two details reinforce that the authors never had to decide. Their headline
CodeLlama-13B runs use **batch size 1** (Appendix A), where the two readings are
identical. And their GPT-2 runs use batch size 32, where the difference is real but
invisible unless you compare against another implementation.

**Verdict: an unstated implementation detail on both sides, not a bug and not a
documented design choice.** It inherits from the transplant of structured pruning
(one mask per subnetwork) into interpretability (one forward per example), where the
two conventions pull in different directions and the paper's notation does not
adjudicate.

**Why this is a substantive result rather than a bug report.** Independent per-example
sampling is a lower-variance gradient estimator: each step averages 32 mask draws
instead of committing to one. That is why the port converges to sparser, more faithful
circuits. It is a legitimate implementation choice, it is not what the authors did, and
until now it was invisible — it is one line in `l0.py` against one line in
`tensor_ops.py`, and the source audit had not caught it.

### 5.5.2 What was changed as a result

The port now defaults to the reference's sampling. `edge_pruning_prune_scores` gained
`shared_mask_sample: bool = True`, and the thesis spine runs Edge Pruning **as
published**.

The reasoning is about what the thesis is for. It compares circuit-discovery methods
and proposes improvements, so the EP baseline has to be the authors' EP. A
variance-reduced EP would have been compared against paper-faithful DiscoGP, ACDC and
EAP, crediting the method with an improvement that belongs to the port, and any later
improvement would have been measured against an already-improved baseline.

The per-example estimator is kept behind the flag rather than deleted, because it is
now a measured finding rather than an accident: the paper leaves the sampling
granularity unspecified, and resolving it the other way improves faithfulness from
0.258 to 0.142 at matched sparsity (~45%) on IOI. That is reportable as a small
result, and it is not available to a study that quietly runs the better variant as
its baseline.

**Consequences, stated rather than absorbed:**

- Every cached EP cell predated the change: **9** in the Pillar-1 spine and **45** in
  Pillar 2. Pillar 3 has not been run. Pillars 1 and 3 were already due for a re-run
  after the `data.py` fix, so the marginal cost was Pillar 2's EP cells. **All 54 were
  re-run on 2026-08-12/13** and both caches now hold a newer entry per cell; the stale
  entries are left on disk and `latest_per_cell` selects by the manifest's ISO
  `saved_at`.
- The Tier C numbers in this document describe the port **as it was**, which is why
  `ep_run_cells.py` pins `shared_mask_sample` explicitly per arm instead of
  inheriting the library default. The study stays reproducible after the change.
- Re-running Pillar 2 changed the EP margins, and the pre-registration required
  recomputing them. **Done: §5.2 carries the v4 column.** Three of four margins are
  wider than v2, so equivalence became easier to declare for a reason unrelated to
  the port, which is why §5.2 states the direction before §5.4 states the verdicts.
- EP's reported faithfulness across the thesis gets **worse**, and the measured
  amount is now known: Clean KL at k = 2000 rises from 0.155 to 0.260 on IOI
  resample. That is the correct baseline, not a regression.
- EP's **seed stability also falls**, on every task and ablation: mean pairwise
  Jaccard on IOI resample drops 0.622 → 0.245.
  - **That spine comparison is confounded and cannot carry the attribution on its
    own.** The v2 EP cells were discovered at `4d2d65e` (2026-08-05) and the v4 cells
    at `2a9c0e0`, so they differ by two `data.py` changes as well as by sampling.
  - Tier C settles it, because `port` and `port-sharedz` were run at one commit on
    one frozen split and differ only in sampling: within-arm Jaccard 0.642 against
    0.265. That is the same magnitude as the spine's 0.622 → 0.245, so the drop is
    attributable to sampling.
  - The v3 → v4 correction is independent evidence for the same conclusion. It
    changed both the GPU pool and the `data.py` gate at once, and moved the number by
    0.004 (0.2409 → 0.2445), with no EP cell moving more than 0.028 across all nine
    task × ablation combinations. Neither factor is doing the work.
- **A cross-method stability claim is *not* yet available and must not be written.**
  Every non-EP cell in both pillars is still at `4d2d65e`, i.e. pre-fix, so
  "EP now sits below DiscoGP's 0.375" would compare a post-fix EP against a pre-fix
  DiscoGP and conflate two changes. Until those cells are re-run, Pillar 2's ordering
  claim stands as last measured and EP's new number is reported only against EP's own
  history.

### 5.6 The other arms

| comparison | reading |
|---|---|
| `orig-withembed` vs `orig` | Functional endpoints all inconclusive (Clean KL difference −0.0009, p = 0.98); density differs, `real but negligible`. The paper's `--with_embedding_nodes` flag costs essentially nothing functionally, which retrospectively supports dropping it for the bijection (§1.1) |
| `port-nobos` vs `port` | Clean KL, Answer Pair KL and density all **fidelity supported**; Logit Diff inconclusive. The input convention barely moves the discovered circuit, even though §4 shows it moves the *forward pass* substantially |

### 5.7 Negative controls

| control | endpoints detecting it |
|---|---|
| `port-neginv` (score-sign convention removed) | **6 of 6** |
| `port-nolambda` (Lagrangian frozen) | **6 of 6** |
| `port-negkl` (KL direction reversed) | 1 of 6 (Logit Diff) |

**Every endpoint detects two of the three controls**, so none is reported as
uninformative. This is a materially stronger power position than the DiscoGP study,
where the density endpoint detected nothing and its "supported" readings had to be
discounted.

The `port-negkl` row is the interesting one, and it got more extreme from v3 onward:
Answer Pair KL detected it at v2 and no longer does, because that margin widened 4.2×
while the control's effect on it (−0.010) did not move. Reversing the KL direction to
`KL(circuit‖full)` now leaves five of six endpoints unmoved and shows up only in
Logit Diff Percent. The v3 → v4 correction does not disturb this: the row is 1 of 6
under both.

The reading that must be resisted here is "the endpoints are too weak to see it". The
same six endpoints detect `port-neginv` and `port-nolambda` decisively at p = 0.0003,
so weakness is ruled out for them. What remains is the substantive reading: the two
KL directions select nearly the same circuits by the yardstick this thesis reports as
faithfulness, and differ mainly in where they put answer-token probability mass. Worth
knowing before reading any KL-direction argument as decisive.

---

## 6. What this establishes, and what it cannot

**Establishes.** That the components are bitwise identical; that the edge bijection
is correct; and that the two programs compute the same function given the same
circuit, to within the numerical floor, at every density tested, with no
attention-bias correction required. The one divergence Tier B leaves is the input
convention, and it is fully present with nothing ablated.

It also establishes, and this is the study's main result, that **the two
implementations do not find equivalent circuits**, that the disagreement is far larger
than the method's own run-to-run noise, and that it is caused by a single catalogued
difference in how the hard-concrete mask is sampled. All three of those are claims the
source audit could not have made: the audit had marked the sampling "verbatim".

**Consequence for the thesis.** EP numbers reported here are not directly comparable
to the Edge Pruning paper's. The port is not wrong -- it is more faithful and sparser
at matched k -- but it is running a lower-variance gradient estimator than the authors
did. That belongs in the limitations table, and it is the kind of thing that stays
invisible without a Tier C.

**Cannot establish.** Bit-level identity of training: RNG streams cannot be aligned
across implementations, and the two do not even draw the same *shape* of mask sample
(difference 2), so Tier C is distributional by necessity. Nor is this a proof of
*correctness* against the paper: it is agreement between two programs, and where both
share an assumption neither this study nor the original would catch it.

**A low cross-implementation Jaccard would not be evidence of a port bug.** The Edge
Pruning paper's own Appendix D reports pairwise edge IoU of 0.5–0.7 across 12 seeds
of the *same* implementation. That is why the functional endpoints are primary and
the structural ones secondary, and it is decided in the pre-registration rather than
after seeing results.

**Metrics are not comparable across harnesses.** The reference's training KL is
`batchmean` over an answer span and its eval KL uses `reduction="sum"` per example;
neither is AutoCircuit's `multibatch_kl_div`. All numbers here are recomputed in one
harness; none is quoted across implementations.

**Selection at matched k is deterministic, not randomised.** `budget_mask` and
`prune_scores_threshold` admit a tie block straddling k whole. EP's continuous scores
effectively never tie, so this is immaterial here. It is noted because
[`discogp.md`](discogp.md) describes this path as "randomized-tie top-k", which is
incorrect.

---

## 7. Reproduction

The reference implementation is deliberately **not vendored** in this repo:

```bash
git clone https://github.com/princeton-nlp/Edge-Pruning.git third_party/edge_pruning
git -C third_party/edge_pruning checkout a918efd
unzip -q third_party/edge_pruning/data.zip -d third_party/edge_pruning/
```

Tiers A and B skip automatically when it is absent. No separate environment is
needed: despite the reference's README warning that `log_alpha` initialisation breaks
with newer `transformers`, all 73 mask tensors initialise correctly under this repo's
`transformers` 4.42 (checked as a gate before anything else was built).

```bash
# Tier A, CPU, seconds
pytest tests/prune_algos/test_edge_pruning_components.py -v

# Tier B, one GPU
pytest tests/prune_algos/test_edge_pruning_equivalence.py -v --runslow

# Tier B measurement sweep
python -m experiments.port_comparison.ep_forward_equivalence --n-prompts 64 --n-masks 40

# Tier C: the frozen split, then discovery (SLURM array), then the analysis pass
python -m experiments.port_comparison.ep_frozen_ioi
python -m experiments.port_comparison.ep_run_cells --arm port --seed 0
python -m experiments.port_comparison.fidelity_analysis \
    --study ep --cache-dir .port_fidelity_ep_cache \
    --out results/port_fidelity/ep_tier_c
```

The analysis pass is cheap and re-runnable: it re-scores cached `PruneScores` and
retrains nothing. Margins live in `fidelity_analysis.py` (`EP_DELTAS`), which records
all four columns (v1 pre-fix, v2 post-fix, v3 provisional, v4 current) so the change
stays auditable. The verdicts in §5.4 to §5.7 are read from
`results/port_fidelity/tier_c_ep_v4/`, produced by the command above with
`--out results/port_fidelity/tier_c_ep_v4` against the v4 margins; the v3 and v2 outputs are
retained in `results/port_fidelity/ep_tier_c{,_posthoc}/`.

Recomputing the margins after an EP change is a two-step chain, not an edit:

```bash
# 1. re-run the invalidated EP cells (see experiments/ep_rerun_cells.py for the list).
#    HARDWARE IS A CONTROLLED VARIABLE: submit with --constraint="gpu_model:a180",
#    the pin the rest of Pillar 2 uses. delta is a standard deviation, so letting
#    SLURM spread the cells across architectures measures node variance as if it
#    were seed variance. This is what made v3 provisional.
sbatch ep_rerun.sh

# 2. re-derive the SDs, then copy them into EP_DELTAS
python experiments/stability_analysis.py \
    --cache-dir .prune_scores_cache_v2/stability --name stability_v4_2026-08-13
```

The margin is the `metric_sd` / `jaccard_sd` for EP × IOI × resample/resample ×
faithfulness × `top_k` @ 2000 in that run's `functional.csv` and `structural.csv`.
