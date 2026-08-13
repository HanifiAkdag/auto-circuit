# DiscoGP Port Fidelity Study — Pre-Registration

**Status:** Tiers A and B complete. Tier C endpoints, margins, and decision rules
below are fixed **before** any Tier C run, and must not be revised after seeing
Tier C results.

**Date:** 2026-08-08
**Comparison target:** `frankniujc/disco_gp` @ `54aef81`, expected at `third_party/disco_gp`

That directory is **gitignored on purpose**: the comparison code is a third-party
dependency and is kept out of this repo. Every result here was produced against
commit `54aef81` exactly; to reproduce, clone it yourself:

```bash
git clone https://github.com/frankniujc/disco_gp.git third_party/disco_gp
git -C third_party/disco_gp checkout 54aef81
```

The Tier A and Tier B tests skip automatically when it is absent.
**Port:** `auto_circuit/prune_algos/disco_gp.py`

## Terminology

This study measures **port fidelity**: whether two programs implement the same
algorithm. It is *not* the thesis's `KL(full‖circuit)` faithfulness. The two never
share a word in the write-up.

## Question decomposition

Two claims requiring different evidence:

1. **Do the two programs compute the same thing?** Deterministic. Tiers A, B.
2. **Do they find the same circuits?** Distributional only, because DiscoGP is
   non-deterministic and circuits are non-identifiable. Tier C.

Most evidential weight sits on (1). A weak Tier C result is uninterpretable alone.

---

## Completed: Tier A (component equivalence)

`tests/prune_algos/test_disco_gp_components.py` — 22 assertions, all pass.
`gumbel_sigmoid` agrees bitwise (including its STE gradient); `schedule_epoch_lambda`,
the answer-margin faithfulness loss, the completeness loss, the sparsity
normalisation, the mask reversal, and the logit-init distribution all match.

## Completed: Tier B (same-mask function equivalence)

`experiments/port_comparison/{edge_map,forward_equivalence}.py`,
`tests/prune_algos/test_disco_gp_equivalence.py` — 9 assertions, all pass.
64 prompts × 40 random masks × 7 densities, two arms.

| arm | median max&#124;Δlogit&#124; @ d=0.5 | argmax agreement |
|---|---|---|
| `bias_intact` | 1.42e+01 | 30.4% |
| `b_O_zeroed` | 3.48e-05 | 100% |

**Result:** the edge bijection, the `AC_mask = 1 - disco_mask` inversion, and
zero-ablation ≡ dropping terms from the node sum are exact. The entire divergence
is that the original folds `b_O / n_heads` into each attention head
(`modules/attention.py:150`) while AutoCircuit adds `b_O` once outside the
maskable decomposition. Node-by-node the head gap is *exactly* that term
(residual 1.4e-03); `|b_O / n_heads|` at layer 11 is **20.05**, the same order as
the logit scale. Full write-up: `thesis_notes/experiments.md` (2026-08-08).

**Consequence for Tier C:** "ablate this head" is a different operation on each
side, so Tier C compares two subtly different objectives. This is handled by a
dedicated mechanism arm rather than a caveat.

---

## Tier C (pre-registered)

### Shared inputs

`experiments/port_comparison/frozen_ioi.py` writes one artifact consumed by both
arms: prompts, corrected (space-prefixed) answer token ids, and a fixed
704 train / 256 test split. This also bypasses the original's **unseeded**
`train_test_split` (`setup_task.py:175`).

Controls: identical prompts and answer tokens; matched optimiser step count
(the port's 11 batches × 100 epochs = 1100 steps; the frozen split sets the
original's batch count to match); fp32 on both sides (the original defaults to
bf16 including AdamW state); `hard=True` (the original has no soft variant).

### Arms

| arm | n | purpose |
|---|---|---|
| `PORT` | 10 | primary |
| `ORIG` | 10 | primary |
| `PORT-bOzero` / `ORIG-bOzero` | 5 / 5 | mechanism: is the residual gap `b_O` or the optimisation? |
| `PORT-noBOS` | 5 | measures the port's BOS-prepend / left-pad factor |
| `NEG-inversion` | 5 | negative control: port with the mask inversion removed |
| `NEG-asshipped` | 5 | negative control: original with unspaced answer tokens |

45 runs total.

### Endpoints

Unit of analysis is the **seed**, n=10 per primary arm. `C(20,10) = 184,756`, so
every permutation test is **exact** (no Monte Carlo).

Primary budget **k = 2000 edges**, the nearest Pillar-2 budget to DiscoGP's own
operating point (measured density 8.69% ⇒ ≈2823 edges); reported across a sweep
of k ∈ {500, 739, 1000, 2000, 5000}. Matched-k selection uses randomized-tie
`topk`.

1. **Functional (primary).** All circuits scored in one AutoCircuit harness under
   zero-ablation discovery / zero-ablation eval: Clean KL Div, Logit Diff Percent,
   Answer Pair KL Div.
2. **Structural.** Jaccard at matched k (`auto_circuit/metrics/analysis_utils.py:108`).
3. **Score-level.** Spearman ρ over the 32,491 aligned scores, aligned **by module
   name** (never flattened).
4. **Operating point.** Edge density at the `logit > 0` threshold.

### Equivalence margins

δ is the **within-implementation SD already measured in Pillar 2** for
DiscoGP × IOI × zero-discovery / zero-eval / faithfulness at k = 2000
(`results/stability/stability_2026-08-05/`, n = 5 seeds). The claim is therefore:
*the shift between implementations is smaller than the method's own run-to-run
noise.*

| endpoint | Pillar-2 mean @ k=2000 | δ (= Pillar-2 SD) |
|---|---|---|
| Clean KL Div | 11.3031 | **1.0262** |
| Logit Diff Percent | 369.3938 | **84.3293** |
| Answer Pair KL Div | 0.8529 | **0.2404** |
| Jaccard @ k=2000 | 0.1864 | **0.0337** |
| Spearman ρ | — | **0.10** (no Pillar-2 counterpart; fixed a priori) |
| density at `logit>0` | — | **0.02** absolute (no counterpart; fixed a priori) |

### Decision rules

For each endpoint:

- **Difference test:** exact two-sample permutation test on the difference of arm
  means (all 184,756 label assignments).
- **Equivalence test:** 90% bootstrap CI for Δ (resampling **seeds**, 10,000
  draws). Equivalence is declared iff the CI lies entirely within ±δ. This is
  TOST at α = 0.05.
- Report arm means with 95% bootstrap CIs alongside.

**Jaccard is a paired-dependent statistic** — each seed appears in many pairs, so
permuting *pairs* would be invalid. Permute **seed labels** and recompute the
between/within split each time, with statistic
`T = mean(within-pair J) − mean(between-pair J)`.

**Multiplicity:** Holm correction across the four endpoint families. Applied to
both the difference tests and the TOST rejections.

### Interpretation grid, fixed in advance

| difference test | equivalence test | reading |
|---|---|---|
| not significant | equivalent | port fidelity supported |
| not significant | not equivalent | underpowered; report as inconclusive, not as support |
| significant | equivalent | a real but practically negligible shift |
| significant | not equivalent | a genuine discrepancy; investigate |

The second row is the one that must not be reported as success.

### Negative controls

Without demonstrated power the study proves nothing. Each control is run through
the identical pipeline and the achieved detections are reported as a sensitivity
table (endpoint × control).

- `NEG-inversion` — must be detected.
- `NEG-asshipped` — known to reach a different operating point (10.48% vs 8.69%
  density), so must be detected.
- Random-baseline circuits — extreme control.

**Ship-blocking:** an endpoint that detects no control has no demonstrated power
and is reported as uninformative rather than as evidence of equivalence.

### Power

With n = 10 per arm the smallest attainable exact permutation p is
1/184,756 ≈ 5.4e-06. The design is powered to rule out shifts of roughly one
pooled SD; it cannot certify bit-level identity, and does not claim to. Achieved
power is evidenced empirically by the negative controls, not by a nominal
calculation.

---

## Scope and limits (fixed in advance)

- **IOI only.** The original supports IOI / BLiMP / PARAREL; Greater-Than and
  Docstring do not exist there.
- **Edge-only.** The paper's Table 2 is joint weight+edge and stays non-comparable.
- **RNG streams cannot be aligned** across implementations, so Tier C is
  distributional by necessity.
- **Low cross-implementation Jaccard is not evidence of a port defect.** Chen et
  al. (the same group) report DiscoGP finding near-disjoint circuits at equal
  performance (IoU 4.1% over 20 IOI runs). Pillar-2 measures DiscoGP's own
  within-implementation Jaccard at 0.186 (k=2000). This is why the functional
  endpoint is primary and the structural one secondary.
- **Metrics are never quoted across harnesses.** The original's KL is 2-class
  `KL(circuit‖full)` with `reduction='mean'` (hence halved), not AutoCircuit's
  full-vocab `KL(full‖circuit)`.
- **Pillar-2 margins are estimated from n = 5 seeds** and are themselves noisy.
  They are pre-registered as-is rather than tuned.
