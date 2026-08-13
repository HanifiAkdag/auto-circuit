# Port validation

How this fork establishes that its re-implementations of published circuit
discovery algorithms compute what the original implementations compute.

| algorithm | reference | status | document |
|---|---|---|---|
| DiscoGP | `frankniujc/disco_gp` @ `54aef81` | validated, 2026-08-11 | [`discogp.md`](discogp.md) |
| Edge Pruning | `princeton-nlp/Edge-Pruning` @ `a918efd` | validated, 2026-08-12 | [`edge_pruning.md`](edge_pruning.md) |

## What the two studies together show about the method

Running the same three tiers twice, on algorithms that differ mainly in their
ablation type, produced one result neither study could have reached alone: **a
port-fidelity conclusion is ablation-relative.**

Both reference implementations fold `b_O / n_heads` into each attention head inside
the masked path, where AutoCircuit adds `b_O` once outside it. Under DiscoGP's zero
ablation that single difference was the *entire* Tier B divergence and had to be
corrected before the study could close. Under Edge Pruning's resample ablation the
same difference cancels exactly.

The discriminating property is **not the ablation's name but the replacement value**:
whether the substituted activation is itself computed through the same per-head
decomposition, and so carries its own `b_O / H` share.

```
replacement carries the share:  Σ_h [z(y + b_O/H) + (1−z)(y' + b_O/H)]
                                  = Σ_h [z·y + (1−z)·y'] + b_O        (mask-free)
replacement is zero:            Σ_h [z(y + b_O/H) + (1−z)·0]
                                  = Σ_h z·y + (Σ_h z)·(b_O/H)         (mask-dependent)
```

Measured single-edge probes into `blocks.11.hook_mlp_in`, by source kind:

| ablation | embedding source | head source (L0) | head source (L10) |
|---|---|---|---|
| `RESAMPLE` | 2.21e-04 | 2.35e-04 | 2.58e-04 |
| `BATCH_TOKENWISE_MEAN` | 2.21e-04 | 2.59e-04 | 2.52e-04 |
| `ZERO` (DiscoGP study) | 1.6e-05 | 8.9e-02 | 7.4e-01 (L11) |

`AblationType` has **seven** members (`types.py:119-140`): `RESAMPLE`, `ZERO`,
`TOKENWISE_MEAN_CLEAN`, `TOKENWISE_MEAN_CORRUPT`,
`TOKENWISE_MEAN_CLEAN_AND_CORRUPT`, `BATCH_TOKENWISE_MEAN` and `BATCH_ALL_TOK_MEAN`.
Six of the seven substitute a mean or a sample of genuine activations; only `ZERO`
injects a value from outside the decomposition. So the divergence is **unique to
`ZERO`**.

**The identity is what carries the general claim; the probes are instances of it.**
A head's output carries `b_O / H` additively at every token, so any average or sample
of genuine head outputs carries it exactly once, whatever axis the average is taken
over:

```
mean_t ( y_t + b_O/H )  =  mean_t( y_t ) + b_O/H
```

Averaging over tokens as well as batch (`BATCH_ALL_TOK_MEAN`) changes *which* `y` is
substituted, not *whether* the bias share rides along. The property that matters is
whether the replacement is built from real head outputs, and that is true by
construction for six of the seven.

So the load sits on the algebra rather than on sample size, and two measured
instances is a design choice rather than a gap. Stated plainly all the same:
`RESAMPLE` and `BATCH_TOKENWISE_MEAN` were probed directly (above) and `ZERO` was
measured in the DiscoGP study; the remaining four were **not** probed.

This is the same distinction already recorded in the thesis notes as "zero
effectively subtracts, resample and mean substitute", surfacing here as a
port-fidelity artefact.

The practical rule: a difference catalogued for one port cannot be inherited by the
next, even when the two share the harness and the difference. It has to be re-derived
under the new ablation and then measured.

## Terminology, fixed once

**Port fidelity** is the question these documents answer: do two programs
implement the same algorithm? It is not **faithfulness**, which throughout the
thesis means `KL(full ‖ circuit)`, a property of a discovered circuit. Keeping the
two words apart matters, because a port can be perfectly faithful in the first
sense while discovering circuits with poor faithfulness in the second.

## Why a re-implementation needs validating at all

Both algorithms are re-implemented against AutoCircuit's edge-patching interface
rather than run as the authors shipped them. That is what makes them comparable
under a common protocol, and it is also what puts their correctness in question:
any difference in results could be a property of the method or an artefact of the
port. The thesis cannot use these algorithms as evidence about methods until that
ambiguity is closed.

## The general shape of the argument

Three tiers, because they are different kinds of claim needing different evidence:

| tier | claim | method | evidence |
|---|---|---|---|
| A | same components | unit tests, shared seed | bitwise equality |
| B | same function given the same circuit | paired forward passes | deviation vs numerical floor |
| C | same circuits from the same data | matched training runs | pre-registered equivalence tests |

Tier C is distributional by necessity: RNG streams cannot be aligned across two
independent implementations, so the claim is about distributions over seeds, not
about a single run matching.

A fourth element is not a tier but decides whether the other three mean anything:
**negative controls**. Deliberately broken variants are run through the identical
pipeline. An endpoint that fails to detect them has no demonstrated power and is
reported as uninformative rather than as evidence of equivalence.
