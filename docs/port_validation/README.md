# Port validation

How this fork establishes that its re-implementations of published circuit
discovery algorithms compute what the original implementations compute.

| algorithm | reference | status | document |
|---|---|---|---|
| DiscoGP | `frankniujc/disco_gp` @ `54aef81` | validated, 2026-08-11 | [`discogp.md`](discogp.md) |
| Edge Pruning | `princeton-nlp/Edge-Pruning` | **not yet validated** | — |

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
