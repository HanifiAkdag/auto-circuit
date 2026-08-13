# DiscoGP port validation

Validation of this fork's DiscoGP implementation
(`auto_circuit/prune_algos/disco_gp.py`) against the authors' released code,
`frankniujc/disco_gp` @ `54aef81`.

Written for reuse in the thesis Methodology chapter and appendix. Terminology,
tier design and the role of negative controls are defined once in
[`README.md`](README.md).

- **Pre-registration:** `docs/superpowers/specs/2026-08-08-discogp-port-fidelity.md`, written before any Tier C run
- **Scope:** IOI only, edge-only pruning. The original supports IOI/BLiMP/PARAREL, so Greater-Than and Docstring are out of reach; the paper's joint weight+edge results are not comparable to the edge-only form the thesis reports
- **Verdict:** the port reproduces the original's behaviour on **3 of 6 endpoints** at the pre-registered margins. The two shortfalls are inconclusive on interval width, not on measured disagreement. The two structural endpoints are essentially exact

---

## 1. Catalogue of differences between the implementations

Established by source audit and confirmed by measurement. Each row states whether
it was **removed** (made identical for the comparison), **controlled** (held equal
on both sides), or **remains**.

| # | difference | origin | treatment |
|---|---|---|---|
| 1 | **Attention output bias `b_O`** | The original folds `b_O / n_heads` into each head's output (`modules/attention.py:150`); AutoCircuit adds `b_O` once into the residual stream, outside the maskable decomposition | **Removed** for the primary comparison by `split_attn_out_bias()`, which zeroes `b_O` and re-adds the per-head share through a permanent hook |
| 2 | **Answer-token spacing** | The original omits the leading space on IOI answer strings, so 46 of 99 names fragment and ~48% of answers are word fragments rather than whole names | **Controlled**: the corrected (spaced) tokenization is used on both sides in the primary arm; the as-shipped form becomes a negative control |
| 3 | **`tail_divergence` KV cache** | AutoCircuit strips the prompts' shared prefix into a KV cache; the original has no such mechanism and runs the full sequence | **Removed** by fixing the defect it contained (§4), after which it is inert on GPT-2 |
| 4 | **BOS and padding** | The port prepends BOS and left-pads; the original does neither | **Controlled and measured**: a 5-seed `port-nobos` arm estimates its contribution rather than disclaiming it. **Qualified 2026-08-12**: `prepend_bos=False` also breaks AutoCircuit's left-padded batching -- the forward reads `cfg.default_prepend_bos` (True) instead of how the tokens were built, so one genuine pad per row is attended -- meaning that arm measures "what happens when BOS is turned off in this harness", defect included, rather than what the BOS convention costs (see [`edge_pruning.md`](edge_pruning.md) §4). The primary and control arms all run with BOS and are unaffected, and `orig` cannot be contaminated at all: it runs through the original's own dataloader, which neither prepends BOS nor left-pads, while Tier C scoring happens in the AutoCircuit harness with BOS. The 3/6 verdict stands |
| 5 | **Train/test split** | The original's `train_test_split` is unseeded (`setup_task.py:175`) | **Removed**: both sides consume one frozen 704/256 split, `datasets/port_fidelity/ioi_frozen.json` |
| 6 | **Numeric precision** | The original defaults to bf16, including AdamW state | **Controlled**: fp32 on both sides |
| 7 | **Optimizer step count** | 704/64 with `drop_last` gives the port 11 batches × 100 epochs; the original's own split gives 13 × 100 | **Controlled** via the frozen split, so both reach the same number of steps |
| 8 | **Mask convention** | AutoCircuit's `patch_mask` is 1 = ablate; DiscoGP's is 1 = keep | **Removed** by the inversion `AC_mask = 1 - disco_mask`, itself validated in Tier B and as a negative control |

Differences 1, 3 and 5 were discovered during this work; the rest were known from
the source audit.

### 1.1 Why `b_O` matters, stated precisely

The two implementations assign different *values* to the same source node. The
original's head source is `result_h + b_O/n_heads`; AutoCircuit's is `result_h`.
Keeping an edge costs nothing, because the sums reconcile. Ablating an edge
`(head_h → D)` makes the original subtract `result_h + b_O/n_heads` from `D`'s
input while AutoCircuit subtracts only `result_h`, leaving the bias share behind.

Measured one edge at a time into `blocks.11.hook_resid_post`:

| source of the ablated edge | that layer's `b_O/n_heads` | max &#124;Δ logit&#124; |
|---|---|---|
| embedding | — | 1.6e-05 |
| head, layer 0 | 1.0351 | 8.90e-02 |
| head, layer 5 | 0.1993 | 2.83e-02 |
| head, layer 11 | 20.0493 | 7.41e-01 |
| MLP, layer 0 | — | 1.6e-05 |
| MLP, layer 11 | — | 1.7e-05 |

Only attention-head sources disagree, and the size tracks that layer's share. The
residue accumulates **per edge**, not per head: disagreement is linear in the
number of ablated head-source edges at first (0.089, 0.178, 0.357, 0.725 for
1/2/4/8) then sub-linear as `ln_final` compresses (1.43 at 32, 2.52 at 128).
Discovery keeps ~2,000 of 32,491 edges, so it ablates ~30,000, nearly all from
head sources: that is how a per-edge residue compounds into the 14-20 logit
disagreement seen on real discovered circuits.

---

## 2. Tier A: component equivalence

`tests/prune_algos/test_disco_gp_components.py`, 13 tests, CPU, seconds.

Each test seeds both implementations identically and asserts **bitwise** equality:

- `gumbel_sigmoid`, at `gs_temp` ∈ {0.01, 1.0}
- `schedule_epoch_lambda`, over all 100 edge epochs and 500 weight epochs, in both the warmup/cooldown and flat configurations
- the faithfulness loss (`_faith_loss_margin` vs `compute_faith_loss_binary_label`)
- the completeness loss (`_complete_loss_margin` vs `compute_complete_loss_binary_label`)
- the sparsity loss (`Σσ(l)/N_edge` vs `edge_sparseness_loss`)
- the mask-reversal path, after applying the convention flip of difference 8
- logit initialization, compared by moments over a large draw since the two draw from the same distribution rather than the same values

**Result: all pass.** These are the cheapest and strongest evidence in the study,
and they close a coverage gap that existed regardless of the comparison:
`disco_gp.py` previously had no tests at all, including none for the mask
inversion.

---

## 3. Tier B: same function given the same circuit

`tests/prune_algos/test_disco_gp_equivalence.py` and
`experiments/port_comparison/{edge_map,forward_equivalence}.py`.

### 3.1 The edge bijection

A mapping between the original's mask tensors and AutoCircuit `PruneScores`, over
GPT-2 small with `separate_qkv=True`: **157 sources, 445 head-level destinations,
49 destination modules, 32,491 edges**. Q/K/V masks transpose
`(prev_nodes, n_heads) ↔ (n_heads, prev_nodes)`; MLP and output masks are
identity. Source ordering is asserted, not assumed: embed, then per block
`[head_0 … head_11, mlp]`.

The map is **not** hand-verified. It is verified functionally in §3.2, because a
permutation error within a shape would survive a shape check but cannot survive
hundreds of random-subset forward comparisons.

### 3.2 Forward equivalence

Identical `input_ids` are fed to both models, bypassing differences 4 and 5 by
construction, at densities {0.1, 0.25, 0.5, 0.75, 0.9}. Reported against the
numerical floor (one implementation run twice on differently composed batches),
not against zero.

**Result:** once `b_O` is accounted for, the implementations agree to the
numerical floor, with **100% argmax agreement**. Node-by-node, the residual gap is
exactly `b_O / n_heads` (residual 1.4e-03 after subtracting it). This single test
validates the bijection, the mask inversion, and zero-ablation equivalence at
once, since the original masks by dropping terms from the `prev_head_idx` sum
while AutoCircuit substitutes a zero activation, and those coincide only if the
decomposition matches.

---

## 4. The AutoCircuit defect found along the way

Tier C initially failed to close, and the residual was traced to a bug in
AutoCircuit itself rather than in either DiscoGP implementation.

`load_datasets_from_json` computed `diverge_idx` by comparing each clean prompt
against **its own corrupt partner**, then built one KV cache from the first
`batch_size` rows and reused it for every batch. Different pairs diverge at
different positions, so the minimum over pairs can exceed the point where the
prompts still agree, and the cache then holds one prompt's keys and values while
other prompts are run against it.

Measured with **nothing ablated**, against each prompt run on its own:

| task | before | after |
|---|---|---|
| Greater-Than | 22.5883 | 0.0001 |
| Docstring | 7.8186 | 0.0000 |
| IOI | 3.1310 | 0.0001 |

Not a padding artefact: Greater-Than and Docstring have uniform prompt lengths
and no padding at all. Reported upstream as
[UFO-101/auto-circuit#17](https://github.com/UFO-101/auto-circuit/issues/17) with
the fix in [PR #18](https://github.com/UFO-101/auto-circuit/pull/18); enforced
locally by `tests/test_tail_divergence_correctness.py`.

**This is a load-bearing methodological point for the thesis.** The defect
affected every task and every algorithm, so all cached results predating the fix
were computed against a model that was not the one being claimed. Between-algorithm
comparisons stayed internally consistent, so their direction survives; absolute
faithfulness numbers do not.

---

## 5. Tier C: matched training runs

`experiments/port_comparison/{run_cells,fidelity_analysis,fidelity_stats}.py`.

### 5.1 Design

Both arms consume the frozen split. All circuits are scored in the **same**
AutoCircuit harness at matched edge count `k = 2000`.

> **Correction (2026-08-12).** This section previously said selection used
> "randomized-tie `top-k`". It does not: `budget_mask` and
> `prune_scores_threshold` both select with a deterministic `>=`, so a tie block
> straddling `k` is admitted whole. DiscoGP's continuous scores effectively never
> tie, so no number in this document changes; the claim was simply wrong. Ties are
> real only for ACDC.

| arm | n | purpose |
|---|---|---|
| `port` | 10 | the port, as shipped |
| `orig` | 10 | the original, as shipped |
| `port-split` | 10 | the port with the bias convention matched (**primary**) |
| `port-bOzero` / `orig-bOzero` | 5 + 5 | mechanism: `b_O` removed from both |
| `port-nobos` | 5 | the input convention (BOS/padding); see the note below |
| `port-neginv` | 5 | negative control: mask inversion removed |
| `orig-asshipped` | 5 | negative control: unspaced answer tokens |

The unit of analysis is the **seed**. With n = 10 per arm, C(20,10) = 184,756, so
every permutation test is **exact**, not Monte Carlo. Jaccard values are pairwise
and therefore dependent, so seed *labels* are permuted and the within/between
split recomputed each time, with statistic `T = mean(within J) − mean(between J)`;
permuting pairs would be wrong. Scores are aligned **by module name**, never
flattened, since `PruneScores` dict order is not stable across runs.

### 5.2 Equivalence margins

δ is the **within-implementation standard deviation already measured in Pillar 2**
for DiscoGP × IOI × zero/zero × faithfulness at k = 2000. The claim then reads:
*the shift between implementations is smaller than the method's own run-to-run
noise*, which is a principled margin rather than an invented threshold.

The pre-registration fixed this **rule**, not the numbers. When the defect in §4
forced Pillar 2 to be re-run, every data-derived margin had to be recomputed:

| endpoint | δ pre-fix | δ post-fix |
|---|---|---|
| Clean KL Div | 1.0262 | 0.8716 |
| Logit Diff Percent | 84.3293 | 71.4485 |
| Answer Pair KL Div | 0.2404 | 0.1792 |
| Jaccard | 0.0337 | 0.0116 |

Spearman (0.10) and density (0.02) are chosen bounds rather than measured SDs and
did not change. Recomputing was obligatory: a wider margin makes equivalence
*easier* to declare, so the stale margins would have inflated the verdict. Doing
so cost one endpoint (§5.3).

### 5.3 Results, primary comparison

`port-split` vs `orig`: bias convention matched, unmodified GPT-2, fixed
dataloader, corrected margins.

| endpoint | difference | 90% CI | δ | verdict |
|---|---|---|---|---|
| edge density | −0.0016 | [−0.012, 0.0092] | 0.02 | **fidelity supported** (p = 0.81) |
| Clean KL Div | 0.045 | [−0.877, 0.916] | 0.8716 | inconclusive (p = 0.94) |
| Logit Diff Percent | −29.6 | [−99.6, 38] | 71.45 | inconclusive (p = 0.52) |
| Answer Pair KL Div | 0.005 | [−0.113, 0.115] | 0.1792 | **fidelity supported** (p = 0.95) |
| Jaccard (within − between) | −0.0013 | — | 0.0116 | **equivalent** (p = 0.79) |
| Spearman (within − between) | 1.03e-05 | — | 0.10 | **equivalent** (p = 0.44) |

**How to state this in the thesis.** Three of six endpoints support fidelity. The
two inconclusive ones fail on *interval width*, not on measured disagreement:
Clean KL's point estimate is 0.045 against a margin of 0.8716, and its CI exceeds
the margin by about 5%. What is missing is precision at n = 5, not agreement.
The structural endpoints are the strongest result: Jaccard remained equivalent
even though its margin tightened by two thirds, and the Spearman gap over 32,491
aligned scores is 1e-05.

### 5.4 What each removed difference bought

| comparison | Clean KL difference |
|---|---|
| `port` vs `orig`, both as shipped | genuine discrepancy on all four functional endpoints |
| after matching the bias convention | 0.713 |
| after also fixing the dataloader | **0.045** |

The dataloader defect was named as the leading suspect for the residual gap
**before** the fix was applied, and fixing it closed the gap as predicted. That
is a confirmed prediction, not a post-hoc rationalisation.

### 5.5 Negative controls

| control | endpoints detecting it |
|---|---|
| `port-neginv` (mask inversion removed) | 5 of 6 (all but density) |
| `orig-asshipped` (unspaced answer tokens) | 2 of 6 (Clean KL, Logit-Diff) |

Density detects neither and is therefore **uninformative**, not evidence of
equivalence, wherever it reads "supported". `orig-asshipped` is only weakly
detected at n = 5, so the equivalence claims should be read with that limited
demonstrated power in mind. This is the study's main weakness and is stated as
such rather than omitted.

---

## 6. What this establishes, and what it cannot

**Establishes.** That the components are identical; that the two programs compute
the same function given the same circuit, to within the numerical floor; and that
circuits from matched runs are functionally equivalent within the method's own
run-to-run noise on three of six endpoints, with the structural endpoints
essentially exact.

**Cannot establish.** Bit-level identity of training: RNG streams cannot be
aligned across implementations, so Tier C is distributional by necessity. Nor is
this a proof of *correctness* against the paper: it is agreement between two
programs, and where both share an assumption neither this study nor the original
would catch it.

**A low cross-implementation Jaccard would not have been evidence of a port bug.**
Chen et al. (2026), by the same group as DiscoGP, report the method finding
near-disjoint circuits at equal performance (IoU 4.1% over 20 runs on IOI). That
is exactly why the functional endpoints are primary and the structural ones
secondary, and it is decided in the pre-registration rather than after seeing
results.

**Metrics are not comparable across harnesses.** The original's KL is a 2-class
`KL(circuit ‖ full)` with `reduction='mean'` (i.e. halved), not AutoCircuit's
full-vocab `KL(full ‖ circuit)`. All numbers here are recomputed in one harness;
none is quoted across implementations.

---

## 7. Reproduction

The reference implementation is deliberately **not vendored** in this repo:

```bash
git clone https://github.com/frankniujc/disco_gp.git third_party/disco_gp
git -C third_party/disco_gp checkout 54aef81
```

Tiers A and B skip automatically when it is absent.

```bash
# Tier A, CPU, seconds
pytest tests/prune_algos/test_disco_gp_components.py -v

# Tier B, one GPU
pytest tests/prune_algos/test_disco_gp_equivalence.py -v --runslow

# the statistics themselves, CPU
pytest tests/prune_algos/test_fidelity_stats.py -v

# Tier C: discovery (SLURM array) then the analysis pass
python -m experiments.port_comparison.frozen_ioi          # writes the frozen split
python -m experiments.port_comparison.run_cells --arm port-split --seed 42
python -m experiments.port_comparison.fidelity_analysis \
    --cache-dir .port_fidelity_cache_v2 \
    --out results/port_fidelity/tier_c_v3_tightmargins
```

The analysis pass is cheap and re-runnable: it re-scores cached `PruneScores` and
retrains nothing. Margins live in `fidelity_analysis.py` (`DELTAS`), which records
both the pre-fix and post-fix values so the change stays auditable.
