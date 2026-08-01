# Pillar 2 — seed-variance (stability) discovery run

**Date:** 2026-08-01
**Issue:** [#5](https://github.com/HanifiAkdag/auto-circuit/issues/5) (epic #2)
**Depends on:** the canonical-iteration-order fix (`fix/deterministic-discovery-order`)
**Scope of this spec:** discovery only. The analysis pass is specified separately.

## Goal

Measure how much of a discovered circuit is an artifact of the random draw, by running
each discovery cell N=5 times under different seeds and comparing the resulting circuits.
Pairs with Pillar 1 (faithfulness) and Pillar 3 (null test) under the same
`(discovery-ablation, eval-ablation, patch-direction)` tagging.

## Precondition: runs must be reproducible

Until 2026-08-01 this experiment was not runnable as specified. Re-running one cell with
the same seed produced a different circuit (top-200 Jaccard 0.33–0.59 for EP/DiscoGP/ACDC)
because `model.edges` and `model.dest_wrappers` are sets whose iteration order varies per
process. A seed sweep would have measured that ordering noise, not the seed.

Fixed by canonical iteration order (`PatchableModel.ordered_dest_wrappers`, ACDC's
total-order sort key). Same node + same seed is now bitwise identical for all four
algorithms. Full write-up in `thesis.md`, "Run-to-run nondeterminism" and "Making
discovery reproducible" (both 2026-08-01).

## Hardware is a controlled variable

Cross-**architecture** runs are still not reproducible and cannot be: different GPUs round
differently and the difference compounds through training (EP top-200 Jaccard ≈ 0.65
between an H200 and a B6000; EAP's ranking is unaffected). Verified that two nodes of the
same compute capability agree bitwise: penelope (A100 80GB PCIe) and rubeus
(A100-SXM4-80GB) produced a byte-identical EP cell.

**Therefore every job in this experiment pins `--constraint="gpu_model:a180"`.** This keeps
16 GPUs of parallelism while holding the hardware axis fixed. Mixing architectures would
fold node variance into the seed-variance number and make the result uninterpretable.

## Cell list — 195 cells

`experiments/stability_cells.py` is the single source of truth, mirroring
`faithfulness_cells.py`.

| factor | levels | n |
|---|---|---|
| algo (swept) | `ep`, `discogp-ep`, `acdc`, `eap` | 4 |
| task | `ioi`, `gt`, `docstring` | 3 |
| discovery ablation | `zero`, `resample`, `mean` | 3 |
| seed | 42, 43, 44, 45, 46 | 5 |

4 × 3 × 3 × 5 = **180**, plus the `random` anchor (ablation-independent, run at `resample`)
× 3 tasks × 5 seeds = **15**. Total **195**.

- **All three ablations**, matching the faithfulness spine, so the stability panel lines up
  1:1 with the faithfulness panel.
- **Ground truth is excluded.** It is deterministic, so its seed variance is zero by
  construction and five runs would re-emit the same file.
- **Random is included** as the pure-noise ceiling: it bounds how low Jaccard can go, and is
  needed to tell "unstable" apart from "no better than chance".
- **Seed 42 is included deliberately.** Those cells are also a spine re-run under the
  ordering fix, so the faithfulness analysis can be re-pointed at this cache's seed-42
  slice rather than re-running the spine separately.

## Data budget

Unchanged from the spine: `SPINE_BATCH_SIZE=64`, `SPINE_BATCH_COUNT=(11, 4)` → 704 train /
256 test. Cache filenames do not encode the budget, so this run gets its own directory,
`.prune_scores_cache/stability/`, and must never be mixed with `faithfulness_v2/`.

## Job splitting

ACDC on IOI is ~13 GPU-h per cell unsplit, and `yolo` is preemptible with no checkpointing,
so a preempted long job restarts from zero. The spine's τ-split is reused for the two
expensive ACDC tasks.

| stage | script | jobs | per-job |
|---|---|---|---|
| A. main array | `stability_discovery.sh` | 165 | median ~3 min |
| B. ACDC partials | `stability_acdc_partials.sh` | 150 | ~51 min (IOI), ~16 min (GT) |
| C. merge | `stability_acdc_merge.sh` | 1 (30 merges) | ~1.6 min each |

Stage A is the 195 cells minus ACDC × {ioi, gt} × 3 ablations × 5 seeds (= 30), which move
to stage B. ACDC-docstring is fast and stays in stage A.

Stage B splits the 20-τ sweep by base (5 bases × 4 exponents each). The partials merge by
element-wise min, which is exactly equivalent to one process running all 20 τ: each τ pass
resets all masks and accumulates `min(tao, curr)` into a `+inf`-initialised tensor.

Total ≈ 100 GPU-h.

**Ordering:** A and B are independent and can run concurrently. C depends on B:

```
sbatch stability_discovery.sh
sbatch stability_acdc_partials.sh          # note the job id
sbatch --dependency=afterok:<id> stability_acdc_merge.sh
```

## Resilience

- `--skip-if-exists` on every `run_cell.py` call, so a preempted-and-requeued task
  recomputes nothing already finished and the whole thing is safely resubmittable.
- `--requeue` on stages A and B.
- `set -eo pipefail` in every script: without it a trailing `echo` makes a failed job report
  COMPLETED, which wrongly satisfies the `afterok` dependency of the merge.
- `--exclude=bob,minerva` is unnecessary here (the `a180` constraint already excludes the
  40 GB nodes that OOM DiscoGP), but the constraint is what carries that guarantee.

## Interfaces

`experiments/stability_cells.py`:

- `SEEDS`, `TASKS`, `SWEPT_ALGOS`, `ANCHOR_ALGOS`, `DISCOVERY_ABLATIONS` — the factors.
- `CELLS: List[Dict[str, str]]` — all 195, consumed by the analysis pass.
- `ARRAY_CELLS: List[Dict[str, str]]` — the 165 for stage A.
- CLI: prints one cell per line as `task algo ablation seed`. `--array` prints
  `ARRAY_CELLS` (what the launcher consumes), `--count` prints the counts for verifying the
  SLURM array bounds.

Cells are dicts with the same keys `run_cell.py` takes, so the launcher is a `read` into
four variables and no cell definition lives in bash.

## Out of scope

- **Analysis.** Pairwise Jaccard as a function of edge budget, mean Jaccard + CV per
  (algo, task, ablation), and cross-method overlap heatmaps. The pieces exist
  (`analysis_utils.jaccard` / `overlap_matrix` / `latest_per_cell`,
  `experiments/similarity_heatmaps.py`); the pass gets written once cells land.
- **Decomposing data-draw noise from init noise.** One `--seed` currently drives both. The
  coupled seed is what issue #5 specifies and is what "re-run the method" means. A
  `--data-seed`/`--algo-seed` split is a follow-up that reuses these same cells.
- **Sub-experiment C** (per-edge Bernoulli circuit sampling, `edge_prob_diagnostic.py`)
  remains tracked on #5 and is unaffected by this run.

## Acceptance criteria

- 195 cached `PruneScores` + manifests under `.prune_scores_cache/stability/`.
- Every manifest records `seed` and the `a180` run; no cell mixed in from another cache dir.
- Repeats confirmed to differ: two seeds of the same cell give different edge sets.
- Same-seed reproducibility confirmed: re-running one cell reproduces its cache bitwise.
