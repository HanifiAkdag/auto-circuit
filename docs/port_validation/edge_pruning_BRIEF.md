# Brief: validate the Edge Pruning port

Paste this to a fresh agent.

---

Validate this fork's **Edge Pruning** implementation
(`auto_circuit/prune_algos/edge_pruning.py`) against the authors' released code
(`princeton-nlp/Edge-Pruning`, paper: https://arxiv.org/pdf/2406.16778), the same
way the DiscoGP port was validated last week.

**Read `docs/port_validation/discogp.md` first.** It is the completed template:
same three-tier structure, same pre-registration discipline, same statistics. Do
not redesign the method, reuse it. `docs/port_validation/README.md` fixes the
terminology (**port fidelity** ≠ **faithfulness**; never use the second word for
this).

## What to do

1. **Pre-register before running anything.** Write
   `docs/superpowers/specs/<date>-edge-pruning-port-fidelity.md` modelled on the
   DiscoGP spec: endpoints, arms, seeds, equivalence margins, decision rules,
   negative controls. Fix the margins as the **Pillar-2 within-implementation SD
   for EP**, and note in the spec that this is a rule, not a number, so a Pillar-2
   re-run obliges recomputing it.
2. **Tier A** — bitwise component tests against the original: hard-concrete L0
   gates, the Lagrangian dual update, the sparsity schedule, `KL(full ‖ circuit)`.
   EP has no test coverage of these today.
3. **Tier B** — the edge bijection plus forward equivalence under masking, against
   a measured numerical floor rather than against zero. Verify the map
   functionally; do not hand-check it.
4. **Tier C** — matched training runs on a frozen split, all circuits scored in one
   harness at matched `k`, exact permutation tests plus TOST, Holm-corrected.
   **Negative controls are not optional**: an endpoint that detects nothing has no
   demonstrated power.
5. Write `docs/port_validation/edge_pruning.md` in the same shape, and add a
   `thesis_notes/` entry.

## Known differences to start from, not to rediscover

`docs/related_work/edge_pruning.md` holds the existing source audit (note: that
directory is gitignored, so read it locally). It already records:

- KL taken at the last-token `out_slice` here vs an answer-span in the original
- 4 learning-rate groups collapsed to 2
- masks pre-composed before patching, rather than applied sequentially through the forward pass
- `target_edge_sparsity=1.1` as the calibration lever for the single-run design

Everything else it calls faithful (hard-concrete L0, resample ablation, Lagrangian,
node masks, all LRs 0.8, `weight_decay=0.01`, `warmup_steps=200`) is **audited by
reading only**. Tier A converts those from "read" to "tested" — that is the point
of the exercise.

## Two traps the DiscoGP study hit

- **A control that reconstructs the thing under test cannot detect a defect in how the real code constructs it.** One measurement script built its KV cache from the very tokens it evaluated, which structurally excluded the defect it was meant to find, and produced a confident wrong conclusion.
- **A test that bypasses a code path says nothing about that path.** Tier B fed raw token ids to both models to neutralise BOS/padding differences, and thereby also excluded the dataloader, where a real bug was hiding.

Also: EP uses **resample** ablation, where DiscoGP uses zero. Do not assume the
zero-ablation equivalence argument from Tier B carries over; resample substitutes
corrupt activations rather than subtracting, so the decomposition must be
re-checked.

## Environment

- `auto_circuit/data.py` was fixed on 2026-08-11 (upstream bug, reported as UFO-101/auto-circuit#17, PR #18). Any cached result older than that date is invalid. **Pillars 1 and 3 have not been re-run and their figures are stale.**
- Never run `python` on the SLURM login node; dispatch with `srun`/`sbatch` and always pass `--mem`.
- `yolo` partition is `sbatch`-only, and gives ~10 concurrent GPUs against 2 on `gpu`.
- Clone the reference implementation yourself; `third_party/` is gitignored by design.
- Full suite baseline: **107 passed**, 1 failed. That failure (`test_unfactorized_edges`) reproduces on unmodified upstream `main` and is not yours.
