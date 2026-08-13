# Experiments

### The pad-id guard was unsound reasoning that happened to be safe; gating on the attention mask restores Greater-Than's prefix cache — 2026-08-13

- **Context:** deciding whether the stability pillar needs re-running, which required establishing what `data.py` on `main` actually does versus the version on the upstream PR branch (`55084a7`, review round 2).
- **Observation — the first fix (`05433b0`) reasons from token ids, which cannot decide the question.** It rejected any position where some row holds `pad_token_id`. On GPT-2 `pad == bos == 50256`, so the legitimate prepended BOS at position 0 was read as padding and rejected on every dataset. The guard was safe only because it over-rejected; the inference it makes ("this id is the pad id, therefore this position is padding") is not valid.
- **Observation — the second fix gates on the tokenizer's own `attention_mask` instead.** Strip only when no row is padded. Under that condition every row has the same length, so position *i* is the same absolute position for every prompt and equal token ids at *i* really are sufficient. This is an argument rather than a coincidence, and it no longer depends on `pad_id != bos_id`.
- **Observation — measured `diverge_idx` on the real spine loaders** (same fork code, only `data.py` swapped, three tasks, 704 train + 256 test):

    | task | pre-fix | pre-fix max &#124;Δlogit&#124; | v1 `05433b0` | PR `55084a7` |
    |---|---|---|---|---|
    | IOI | 3 | **16.01** | 0 | 0 |
    | Greater-Than | 8 | **24.56** | 0 | **2** |
    | Docstring | 11 | **12.87** | 3 | 3 |

    Post-fix error is 0.0 or 1e-4 (float floor) under either revision.
- **Significance — supersedes the 2026-08-11 entry below.** That entry concluded the optimization is "close to dead in this project" and that the write-up should not describe `tail_divergence` as active. Both were artifacts of the v1 guard. The feature is inert exactly on the ragged datasets (IOI) and active on the uniform ones (GT, Docstring). The correctness claim is unchanged; only the framing was wrong.
- **Significance — all three spine tasks were materially wrong pre-fix**, not just Greater-Than. No cell discovered before `05433b0` is comparable with one discovered after, which independently confirms the confounding recorded in the 2026-08-13 EP entry below.
- **Tests:** two fork tests failed on adopting the PR version, both because they pinned the v1 rule rather than a correctness property: one asserted `pad_token_id not in` the stripped prefix (the same unsound id-based inference), the other was a deliberate tripwire demanding the BOS-promotion interaction be re-checked before GPT-2 strips again. That re-check is the two behavioural tests, which pass for GT under the PR version: agreement with each prompt run alone (batch of one, no padding, no cache, no dataloader) and toggling `tail_divergence` being a no-op. Both tests rewritten to assert the padding gate. **18 passed, 2 skipped** (`--runslow`), the skips being IOI's genuinely inert cases.
- **Code pointer / Run:** `auto_circuit/data.py:342-370`; `tests/test_tail_divergence_correctness.py`; `tests/test_data.py::test_tail_divergence_only_strips_a_cacheable_prefix`.

### Adopting EP's paper-faithful mask sampling makes it the least stable method, not the second-most — 2026-08-13

- **Context:** the EP port-fidelity study (Tier C) found the port's per-example hard-concrete sampling to be the whole cause of its disagreement with the reference, so `shared_mask_sample=True` became the library default on 2026-08-12. That invalidated every cached EP cell. All 54 were re-run (9 Pillar-1 spine, 45 Pillar-2 stability; `experiments/ep_rerun_cells.py`) and Pillar 2 was re-analysed as `stability_v3_2026-08-13`.
- **Observation — the re-run touched only EP.** Every non-EP row in `structural.csv` is bit-identical to `stability_v2_2026-08-11` (delta exactly 0.000 for DiscoGP, EAP, ACDC and Random across all three tasks and ablations). That confirms the scope of the re-run. It is **not** a control for correctness: those rows are identical precisely because those cells were not recomputed.
- **Correction, found while writing this entry — the cached cells are not all at one configuration, and this was believed otherwise.** Checked by `git merge-base --is-ancestor` on each manifest's recorded commit rather than by file date. The `data.py` `tail_divergence` fix is `05433b0` (2026-08-11). Every non-EP cell in **both** pillars was discovered at `4d2d65e` (2026-08-05), which is **pre-fix**; only the re-run EP cells (`baeef64`) are post-fix.
    - The belief that "Pillar 2 is post-fix" came from the *analysis* run being dated 2026-08-11. The analysis date says nothing about when the cells it reads were discovered. `stability_v2_2026-08-11` reads cells discovered on 08-10.
    - Consequence: the v2 → v3 EP comparison below is **confounded**, differing by the `data.py` fix as well as by sampling. And any EP-vs-other-method claim from `stability_v3` compares a post-fix EP against pre-fix rivals.
    - **Lesson: date the cells by their recorded git commit, never by the results directory's name or mtime.** Same class of error as [[hpo-stale-cache-lexical-sort]], one level up: there the wrong file was loaded, here the right file was loaded and its provenance assumed.
- **Second correction, more serious — the re-run relaxed a controlled variable, so the v3 margins are provisional.** `stability_discovery_v2.sh` pins `#SBATCH --constraint="gpu_model:a180"` and says why in its own header: seed repeats are comparable only within one GPU architecture, and cross-architecture EP runs differ at about the magnitude of the effect being measured. The re-run script constrained on GPU **memory** instead, to dodge a 24 GB OOM, and SLURM scattered the 54 cells over **four architectures**: a180/ampere (penelope, rubeus, remus, severus), a6000/ampere (moe), l40/lovelace (itchy), titanrtx/turing (turtok).
    - δ *is* a standard deviation, so this put node variance directly into the quantity the port-fidelity margins are defined as. The 1.3× / 3.1× / 3.9× widening is an **upper bound** with an unknown hardware component, and the Jaccard drop is inflated by an unknown amount too.
    - Re-submitted pinned to `gpu_model:a180` (job 420561). Verified with `scontrol show node` that the four a180 machines are one architecture and that moe/itchy/turtok are genuinely not; the constraint is honoured, so `remus` and `severus` are legitimate a180 members that the launcher's header (written when the pool was penelope+rubeus) does not list.
    - **Lesson: when a launcher documents a constraint as a controlled variable, a new script for the same cells inherits the constraint, not just the cell list.** The OOM that motivated relaxing it is solved by the a180 pin as well, since every a180 node is 80 GB. Cost: one full re-run of 54 cells, and every number in this entry demoted to provisional.
- **Observation — EP's structural stability falls on all nine cells.** Mean pairwise Jaccard at the published budget:

    | task | ablation | v2 | v3 |
    |---|---|---|---|
    | IOI | resample | 0.622 | **0.241** |
    | IOI | mean | 0.512 | 0.327 |
    | IOI | zero | 0.333 | 0.202 |
    | Greater-Than | zero | 0.264 | 0.178 |
    | Docstring | resample | 0.708 | 0.668 |

    - **Attribution, given the confound above.** Tier C isolates the sampling factor cleanly, because `port` and `port-sharedz` ran at one commit on one frozen split and differ in nothing else: within-arm Jaccard 0.642 against 0.265. That is the same magnitude as the spine's 0.622 → 0.241, so the drop is the sampling change and the `data.py` fix contributes little. The spine comparison alone could not have established this.
    - **No cross-method claim yet.** It is tempting to write "EP is now least stable, below DiscoGP's 0.375", and that would be wrong: DiscoGP's 0.375 is a pre-fix number. The Pillar-2 headline stays "EAP most stable, DiscoGP least" until the non-EP cells are re-run. See [[stability_tracks_function]].
    - The damage is task-dependent: IOI resample loses 0.38, Docstring mean loses 0.005. It is not a uniform noise increase.
- **Observation — faithfulness gets worse by a measured amount.** Clean KL at matched budget, discovery = eval ablation: IOI resample 0.155 → 0.269, Greater-Than resample 0.034 → 0.051, Docstring resample 0.099 → 0.196. Under zero ablation the shift is larger still (IOI 0.99 → 2.45, GT 0.49 → 2.70).
- **Significance — this is the correct baseline, and the direction matters for the argument.** The thesis compares methods and proposes improvements, so EP has to be the authors' EP; a variance-reduced port would have credited the method with an improvement belonging to the port. The per-example estimator is kept behind a flag because it is now a finding rather than an accident: the paper leaves sampling granularity unspecified (no batch index anywhere in §3, the Lagrangian, or Eq. 3), and resolving it the other way improves faithfulness ~45% at matched sparsity on IOI.
- **Significance — the pre-registered margin rule did real work.** δ is defined as EP's own Pillar-2 within-implementation SD, so changing EP obliged recomputing it *before* reading any verdict. Three of four margins widened (Clean KL 0.0260 → 0.0326, Logit Diff 5.34 → 16.36, Answer Pair KL 0.0131 → 0.0514), which makes equivalence easier to declare for a reason unrelated to the port. Stating that ordering in advance is the only thing separating "the margins moved" from "the result improved". All §5.4 verdicts survive the widening.
    - The Jaccard margin instead *narrowed* (0.00825 → 0.00540), which is a floor effect and not improved consistency: mean overlap fell 0.622 → 0.241 over the same change and low overlap has less room to vary. Same level-versus-spread trap as the 2026-08-05 zero-ablation entry below.
- **Observation — one Tier C endpoint changed verdict, and one control went quiet.** `port-sharedz` vs `orig` Answer Pair KL now reads *fidelity supported* (diff −0.013, p = 0.71). Clean KL is still *inconclusive*, and by a hair: 90% CI half-width 0.031 against δ = 0.0326, missing by 0.0016. That is the `0.74δ` power ceiling the pre-registration named in advance doing exactly what was predicted, the same trap that left DiscoGP's Clean KL inconclusive. Separately, the `port-negkl` control fell from 2/6 endpoints to 1/6, because the Answer Pair KL margin widened past the control's effect.
- **Code pointer / Run:** `experiments/ep_rerun_cells.py`; `results/stability/stability_v3_2026-08-13/`; `results/port_fidelity/tier_c_ep_v3/`; margins in `experiments/port_comparison/fidelity_analysis.py` (`EP_DELTAS`, all three columns retained); write-up §5.2 in `docs/port_validation/edge_pruning.md`. SLURM jobs 419953, 420039, 420044.

### Status of the port-fidelity and `tail_divergence` work — as of 2026-08-11

The entries below run roughly newest-first and record the investigation as it
happened, including two framings that turned out to be wrong. This block is the
current state; where an entry disagrees with it, this block wins.

| question | answer | where |
|---|---|---|
| Is the DiscoGP port faithful? | **3 of 6 endpoints support fidelity** at the corrected margins; Clean KL and Logit-Diff are inconclusive on interval width, not on disagreement. Structural endpoints essentially exact (Spearman gap 1e-05). | margins erratum, 08-11 |
| What explained the port gap? | Two things, both removable: the `b_O/n_heads` bias convention, and AutoCircuit's `tail_divergence` dataloader defect. Neither is a DiscoGP implementation error. | 08-09 bias entries, 08-10 re-run |
| Is the `tail_divergence` defect fixed? | Yes, verified against a per-prompt reference on all three tasks. It is an upstream AutoCircuit bug, not a fork bug. | fix entry, 08-09 |
| Reported upstream? | [issue #17](https://github.com/UFO-101/auto-circuit/issues/17) and [PR #18](https://github.com/UFO-101/auto-circuit/pull/18). | 08-11 |
| Does Pillar 2 survive the fix? | Yes. Stability ordering unchanged (EAP > ACDC ≈ EP > DiscoGP), Random anchor and fixed budgets identical. | Methodology, Pillar-2 entries |
| What still needs re-running? | **Pillars 1 and 3.** Every figure under `results/faithfulness_comparison/` predates the fix. | fix entry, 08-09 |

Two claims made during this work were **withdrawn**, and are marked as such where
they appear: that the defect was a padding-mask problem (Docstring has zero padding
and was still wrong by 7.82), and that a per-batch cache was the correct fix (it
leaves IOI at 3.3437).

### Reported upstream: AutoCircuit issue #17 and PR #18 — 2026-08-11

- **Context:** the `tail_divergence` defect is in stock AutoCircuit, not in this fork. `auto_circuit/data.py` was byte-identical to upstream `main` (blob `cf9199d`) before the fix, so the patch applies cleanly despite this fork being far ahead.
- **Filed:** [issue #17](https://github.com/UFO-101/auto-circuit/issues/17) with the diagnosis, and [PR #18](https://github.com/UFO-101/auto-circuit/pull/18) with the fix plus a regression test, from the public fork `HanifiAkdag/auto-circuit-1` cut from upstream `8fc5afd`. The PR adds a test to upstream's own `tests/test_data.py` that fails on unmodified `main` and passes with the change.
- **Significance for the write-up:** the fork now diverges from released AutoCircuit on a correctness-relevant line. Anyone reproducing with a pip-installed `auto-circuit` gets the defective behaviour, so the Limitations section must say so. Upstream has had no functional commit since 2024-11 (the last five are README edits), so the fix should not be assumed to land.
- **Two upstream facts worth citing:** `tests/test_data.py::test_tail_divergence` passes only because `ioi_vanilla_template_prompts` happens to have a genuinely common prefix, uses `return_seq_length=True` (forbidding padding) and exactly one batch; and `test_unfactorized_edges` fails on unmodified upstream `main` (38 passed, 1 failed), so it is not a fork regression.

### The `tail_divergence` fix works by disabling the feature on GPT-2, and the tests said less than claimed — 2026-08-11

> **Partly superseded 2026-08-13** (see the top entry). The "inert on GPT-2" observations describe the v1 pad-id guard, not the version now on `main`: Greater-Than strips 2 tokens again. The coverage and ragged-input observations still hold.

- **Context:** auditing my own fix after the Pillar-2 re-run validated it. Two things did not hold up: how much of the feature survives, and how much the test suite actually covers.
- **Observation — the stripping path is inert on both GPT-2 tasks.** Measured `diverge_idx`: IOI 3 → **0**, GT 8 → **0**, Docstring 11 → **3**. GPT-2 sets `pad == bos == eos == 50256`, so a prepended BOS cannot be certified pad-free and stripping stops at position 0.
    - **The cost of the guard is 2 tokens, not the whole prefix.** Without the pad guard the prompts genuinely agree to index 2 (IOI, of 15) and index 2 (GT, of 11). So GT's pre-fix `diverge_idx=8` was never an 8-token saving: 2 tokens of real prefix plus 6 of corruption.
    - Being conservative here is the right call rather than merely the safe one: those are exactly the positions where TransformerLens' BOS-promotion rule misfires on a chunk, which is the mechanism that made the shared cache wrong.
- **Observation — two of the three "no-op" test cases were vacuous.** `test_disabling_tail_divergence_changes_nothing` passes trivially on IOI and GT because `diverge_idx=0` makes toggling a literal no-op. Real coverage was 3 structural + 3 correctness + **1** meaningful no-op (Docstring), not the 9 tests previously claimed.
- **Observation — the fix crashed on ragged input, and so did the code it replaced.** Clean and corrupt are tokenized separately, so they can be padded to different widths; `t.cat` then raises (`Sizes of tensors must match except in dimension 0`). The pre-fix `clean_prompts == corrupt_prompts` comparison would have raised too, so this is pre-existing fragility rather than a regression, but it is now guarded: unequal widths mean no position is comparable, so nothing is stripped.
- **Observation — any padding at all disables stripping.** Unequal prompt lengths put a pad at position 0 for the shorter rows under left-padding, so `diverge_idx` collapses to 0. This is correct, and it means the optimization is inert on most real datasets, not just on GPT-2.
- **Significance:** the correctness claim for the fix stands (end-to-end agreement with a per-prompt reference, all three tasks), but the *feature* it protects is now close to dead in this project. That is a defensible trade, and it should be stated as one rather than left implicit: the write-up should not describe `tail_divergence` as an active optimization.
- **Tests added:** a non-vacuity guard that skips rather than silently passes when `diverge_idx == 0`; a test pinning the GPT-2 pad/bos collision and asserting the prompts agree past position 0, so a future change that re-enables stripping trips it; a synthetic deep-prefix fixture on `attn-only-4l` (distinct pad/bos ids, 26 of 27 tokens cached) that exercises the caching path at a depth no stock task reaches; and a ragged-input test pinning the collapse to 0. **11 passed, 4 skipped** (the skips are the honest report of the inert cases).
- **Code pointer / Run:** `auto_circuit/data.py:354-377`; `tests/test_tail_divergence_correctness.py`.

### Erratum: the Tier C margins were derived from pre-fix Pillar-2 data; recomputing them costs one endpoint — 2026-08-11

- **Context:** the entry above scores Tier C against margins δ taken from the Pillar-2 within-implementation SD for DiscoGP × IOI × zero/zero × faithfulness at k=2000. Those SDs came from `stability_2026-08-05`, which was itself produced through the defective dataloader. The port cells were re-run on the fixed dataloader but the margins were not, so the verdicts mixed post-fix data with pre-fix noise estimates.
- **Observation — every data-derived margin shrank once Pillar 2 was re-run** (`stability_v2_2026-08-11`):

  | endpoint | δ pre-fix | δ post-fix | change |
  |---|---|---|---|
  | Clean KL Div | 1.0262 | 0.8716 | −15% |
  | Logit Diff Percent | 84.3293 | 71.4485 | −15% |
  | Answer Pair KL Div | 0.2404 | 0.1792 | −25% |
  | Jaccard | 0.0337 | 0.0116 | **−66%** |

  Spearman (0.10) and density (0.02) are chosen bounds rather than measured SDs, so they are unaffected.
- **Significance — the direction is unfavourable.** A wider margin makes TOST equivalence easier to declare, so every pre-fix verdict was scored against noise estimates that were too generous. Re-running with the corrected δ flips **Clean KL Div** from "fidelity supported" to inconclusive: its 90% CI upper bound is 0.916 against the new margin of 0.8716, missing by about 5%. The headline becomes **3/6 endpoints supporting fidelity**, not 4/6. Answer-Pair KL (0.115 vs 0.1792) and Jaccard (gap 0.0013 vs 0.0116) survive comfortably; Jaccard survives despite its margin tightening by two thirds.
- **Methodological point worth stating in the write-up:** the pre-registration fixed the *rule* for setting δ ("the Pillar-2 within-implementation SD"), not the literal numbers. Recomputing δ after Pillar 2 was corrected is what that rule requires; leaving the old numbers in place would have been the violation. Both value sets are recorded in the source so the change is auditable.
- **Unchanged:** the negative-control sensitivity table is identical under the tighter margins (`port-neginv` 5/6, `orig-asshipped` 2/6, density missed by both), so the limited demonstrated power noted above still applies and is not improved by the tightening.
- **Code pointer / Run:** job 418791; `experiments/port_comparison/fidelity_analysis.py:60-79` (DELTAS, both value sets); `results/port_fidelity/tier_c_v3_tightmargins/` alongside the superseded `tier_c_v2/`.

### Port fidelity re-run after both fixes: the implementations agree — 2026-08-10

- **Context:** the Tier C results were all discovered and scored through the defective dataloader, so they could not answer whether the port matches the original once the bias convention is matched. Re-ran the port arms (35 cells) on the fixed dataloader into a separate cache (`.port_fidelity_cache_v2`); the `orig*` arms build their own dataloaders (`run_cells.py:159-184`) and were carried over, though all arms were re-scored through the fixed harness. 55 cells, zero failures.
- **Result, `port-split` vs `orig` (bias convention matched, unmodified GPT-2):**

  | endpoint | pre-fix diff | post-fix diff | margin | verdict |
  |---|---|---|---|---|
  | edge density | −0.0013 | −0.0016 | 0.02 | fidelity supported (p=0.81) |
  | Clean KL Div | 0.713 | **0.045** | 1.026 | fidelity supported (p=0.94) |
  | Answer Pair KL | 0.008 | 0.005 | 0.240 | fidelity supported (p=0.95) |
  | Jaccard gap | 0.0074 | **−0.0013** | 0.0337 | equivalent |
  | Spearman gap | 0.0098 | **1.03e-05** | 0.10 | equivalent |
  | Logit Diff Percent | +102.5 | **−29.6** | 84.33 | inconclusive (p=0.52) |

  Four of six now support fidelity and the two structural endpoints are essentially exact (Spearman gap 1e-05). The remaining one is inconclusive on *interval width* only: its point estimate moved from +102.5, outside the margin, to −29.6, well inside it, so what is missing is precision rather than agreement.
  - **Superseded 2026-08-11:** the margins in this table were derived from pre-fix Pillar-2 data. Recomputing them from the re-run Pillar 2 drops Clean KL to inconclusive, so the headline is **3 of 6, not 4 of 6**. See the margins erratum above; the verdicts here should not be quoted.
- **The earlier hypothesis was right.** I had flagged the dataloader defect as the leading suspect for the two endpoints that stayed unresolved after matching the bias convention. Fixing it closed Clean KL (0.713 → 0.045) and moved logit-diff inside its margin. That is a prediction made before the fix and confirmed after it, not a post-hoc rationalisation.
- **`port` vs `orig` as shipped is still a genuine discrepancy on all four functional endpoints**, as expected: the bias convention still differs there. Notably the structural endpoints now read equivalent (Jaccard gap 0.0319 < 0.0337), where pre-fix Jaccard was not.
- **Control caveat:** with n=5, the negative controls are weaker than before. `port-neginv` is detected by 5/6 endpoints, but `orig-asshipped` by only 2/6 (Clean KL, Logit-Diff), down from 3/6, and density is now missed by both. The equivalence claims above should be read with that limited demonstrated power in mind.
- **Code pointer / Run:** arrays 418375 (port arms) and 418396 (controls) on the `yolo` partition, 10-wide; `results/port_fidelity/tier_c_v2/`.

### The `tail_divergence` defect is fixed, and it changes every cached result — 2026-08-09

- **The fix** (`auto_circuit/data.py`, in the `tail_divergence` branch): `diverge_idx` was the minimum *within-pair* divergence, i.e. where each clean prompt starts to differ from **its own** corrupt partner. But the prefix KV cache is built once and reused for every batch, so the prefix must be identical for **every** prompt. The new computation keeps only the positions where all clean and corrupt prompts agree **and** none of them pads, and never strips the whole sequence.
  - The pad-free condition matters independently: a padded position holds a real token for some rows and padding for others, so no single split index means the same thing for every row. This is also why the pad/BOS mask heuristic misfired, so one change fixes both defects.
- **Effect on the stripping:** Docstring `diverge_idx` 11 → **3** (a genuinely common prefix survives), Greater-Than 8 → **0**, IOI 3 → **0**. On the latter two the optimisation is now correctly disabled, because those prompts do not share a pad-free prefix at all.
- **Verified** by comparing a stock `Task` against each prompt run individually (batch of one, no padding, no cache); now enforced by `tests/test_tail_divergence_correctness.py`:

  | task | before | after |
  |---|---|---|
  | Docstring | 7.8186 | **0.0000** |
  | Greater-Than | 22.5883 | **0.0001** |
  | IOI | 3.1310 | **0.0001** |

  All three now agree with the reference on **100%** of top-1 predictions.
- **Rejected alternatives, both measured** (throwaway script, removed once the numbers below were recorded): a per-batch cache fixes the unpadded tasks completely (Docstring 0.0000, GT 0.0001) but leaves IOI wrong (3.3437), and adding a corrected mask on top makes IOI *worse* (3.9754). So "rebuild the cache per batch" is not sufficient, and the mask correction alone is actively harmful. The independent verifier's claim that a per-batch cache "recovers the truth exactly on all four datasets" does not hold.
- **Tests:** `tests/test_tail_divergence_correctness.py`. **Superseded 2026-08-11:** this originally claimed 9 tests giving three-way coverage; two of the three no-op cases were vacuous. See the audit entry above for the current 11-test suite and what it actually covers.
- **Consequence, and it is large:** every cached result in this project was produced through the defective path, on all three tasks. Between-algorithm comparisons were internally consistent so their *direction* should survive, but absolute faithfulness numbers were computed against a model that was not the one being claimed. Pillar 1, Pillar 2, Pillar 3 and the port-fidelity Tier C runs all need re-running before any number goes in the write-up.

### Correction: the dominant `tail_divergence` defect is a shared KV cache, not the padding mask — 2026-08-09

- **Context:** an independent agent was asked to verify the padding-mask finding below without being told the mechanism, and was instructed to try to refute it. It found a larger, separate defect. I verified the pivotal claim myself before accepting it.
- **The dominant defect:** `data.py` builds the prefix cache **once** from `clean_prompts[:batch_size, :diverge_idx]`, and `PatchableModel` (`patchable_model.py:129,143`) keeps exactly one cache per batch *size* and reuses it for **every batch**. The test loader's cache is built from the first `bs` entries of the same list, i.e. **train** prompts. `diverge_idx` is a *pairwise* quantity (clean vs corrupt of the same pair, `data.py:355-356`) and never checks the prefix is common *across* prompts, which it generally is not.
- **Measured on the library's own `Task` objects, stock config, nothing ablated** (test set, 4 batches of 64):

  | task | pad tokens in test set | max&#124;Δlogit&#124; | argmax agreement |
  |---|---|---|---|
  | IOI | 1148 | 2.67-3.13 | 90.6-96.9% |
  | Greater-Than | 256 | **22.59** | **14.1%** |
  | Docstring | **0** | **7.82** | 67-84% |

  **Docstring has zero padding across the whole test set and is still badly wrong.** That refutes the "padding is the trigger" framing in the entry below. Cross-batch prefix mismatch is 64/64 on GT and Docstring for batches 1-3.
- **Do not fix the mask alone.** The wrong mask incidentally masks out prefix positions holding another prompt's K/V; correcting it exposes those foreign values and makes IOI roughly 6x worse (max&#124;Δ&#124; ~2.2-3.1 rising to ~16.1-16.8).
  - **Correction (2026-08-09, later the same day):** this entry originally went on to say "the demonstrated correct fix is a per-batch cache, which recovers the truth exactly". That was the independent verifier's claim and it is **wrong**. Measured: a per-batch cache leaves IOI at 3.3437, and adding a corrected mask on top makes it 3.9754. See the fix entry above.
- **My methodological error:** `kv_effect_decomposition.py` built its cache from the very tokens it then evaluated, which structurally excludes the shared-cache defect, and I concluded from it that "the mask explains everything". A control that reconstructs the thing under test cannot detect a defect in how the real code constructs it. The same flaw invalidates the "uniform-length prompts agree to 2.0e-05" control.
- **The library's own test hides it:** `tests/test_data.py::test_tail_divergence` asserts exactly the equality that fails here, but only under `ioi_vanilla_template_prompts` + `return_seq_length=True` (which forbids padding) + a single batch, the one configuration where both defects vanish.
- **Consequence for the thesis:** unchanged in direction but larger in size than recorded below, and it now applies to Greater-Than and Docstring at least as strongly as to IOI. Discovery and evaluation both use the shared cache, so within-thesis between-algorithm comparisons stay like-for-like; absolute faithfulness numbers do not.
- **Definitive end-to-end test** (since folded into `tests/test_tail_divergence_correctness.py`), built to remove the flaws in the two earlier attempts. The library arm is a stock `Task` constructed exactly as `run_cell.py` does, with **nothing monkeypatched**; the truth arm runs each prompt **individually** (batch of one, no padding, no cache, no AutoCircuit dataloader), which is correct by construction; both use `load_tl_model`, so only the input path can differ. Nothing is ablated. Alignment is asserted rather than assumed (what the library feeds must be the tail of that prompt's own tokenization: 0/64 misaligned on every task).

  | task | pads in test set | control vs truth | **library vs truth** | argmax agreement |
  |---|---|---|---|---|
  | Docstring | **0** | 0.0000 | **7.8186** | 73.8% |
  | Greater-Than | 256 | 0.0001 | **22.5883** | **21.9%** |
  | IOI | 1148 | 0.0001 | **3.1310** | 93.4% |

  The control is the library's own loader with only `tail_divergence` disabled, and it reproduces truth exactly on all three tasks. So the dataloader, padding, batching and the `PatchableModel` wrapper are all fine, the defect is `tail_divergence` specifically, and nothing downstream corrects it.
  - Docstring is the cleanest single piece of evidence: zero padding, so the mask heuristic cannot fire, yet still wrong by 7.8 logits. That rules out the padding framing on its own.
- **Code pointer / Run:** `tests/test_tail_divergence_correctness.py` (the definitive test, folded in from a since-removed script); the scratch probes were throwaway and are gone.

### Superseded: the padding-mask account of the `tail_divergence` defect — 2026-08-09

This entry originally diagnosed the defect as a padding-mask problem. That framing
is **wrong** and was refuted the same day (see the correction entry above):
Docstring has zero padding and was still off by 7.82. Kept in compressed form
because three parts of it survived and are still load-bearing.

- **The mask defect is real, but secondary.** `transformer_lens/utils.py:get_attention_mask`
  has a rule for tokenizers where BOS and PAD share an id, which GPT-2's do (both 50256):
  `attention_mask[arange(batch), is_leading_pad.sum(-1) - 1] = 1`, i.e. "the last leading
  pad is really BOS". Correct for a whole sequence, but AutoCircuit calls the model once
  per chunk and TransformerLens rebuilds the mask **per chunk**, so the rule fires on a
  chunk containing no BOS and un-masks a genuine pad. Measured on one IOI batch:
  **42 of 64 rows** get a wrong mask, always in the same direction (a pad wrongly
  attended, never a real token wrongly ignored). Rows break iff their padding extends
  past `diverge_idx`. Some heads place their entire attention on a pad (max weight 1.000
  vs 0.000 on the whole-sequence path).
- **The naive fix direction does not work.** Passing `attention_mask=` to
  `HookedTransformer.forward` is silently discarded on the normal path: when
  `start_at_layer is None` it overwrites the caller's argument with `input_to_embed`'s
  return (`HookedTransformer.py:517-527`), and `input_to_embed` takes no such parameter.
  That is why `ACDC.py:145` can pass one (it sets `start_at_layer`). No chunk-local fix
  can work either, since whether a chunk's leading pads contain the BOS depends on tokens
  outside it.
- **Erratum on my own measurement.** An earlier version reported max 18.06 / 65.6% argmax
  agreement for the padded case. That run was confounded: `HookedTransformer.forward`
  computes a mask only when `padding_side == "left"` or a kv cache is passed, and a freshly
  loaded model defaults to right padding, so the uncached arm did no masking at all. With
  `padding_side="left"` on both arms, as AutoCircuit's loader does, the real batch gives
  max **1.82** and **96.9%** agreement. Direction unchanged, magnitude inflated ~10x.
- **Method note worth keeping:** the discarded decomposition script built its cache from
  the very tokens it evaluated, which structurally excludes the shared-cache defect, and I
  concluded from it that "the mask explains everything". A control that reconstructs the
  thing under test cannot detect a defect in how the real code constructs it.

### A third port difference: the KV-cache prefix is computed unmasked, and Tier B never tested it — 2026-08-09

- **Context:** asked whether fixing the answer-token spacing and the attention-bias convention makes the two implementations identical. Auditing for anything else surfaced a difference that neither earlier tier could have caught.
- **Observation:** `tasks.py:219` passes `tail_divergence=True` whenever a tokenizer is present, so the IOI loader strips the tokens shared by the clean and corrupt prompts into a KV cache. Measured on the frozen split: **`diverge_idx=3`, `kv_cache=yes`**, both splits. Those three positions are computed **once, unmasked**, and reused for every forward pass.
  - The original has no such mechanism. It runs the full sequence with masks applied at every position.
  - So edges feeding the first three token positions are effectively **un-ablatable in the port** and fully ablatable in the original.
- **Why it was invisible until now:** the forward-equivalence comparison fed raw token ids directly to both models, with no KV cache, precisely to bypass the BOS and padding difference. That also bypassed this. The training comparison *did* use `task.train_loader`, so it carries the difference uncontrolled.
- **Significance:** at the time this was a live candidate for the one endpoint that stayed non-equivalent after removing `b_O` (Answer-Pair KL, −0.212 against a 0.240 margin). **Resolved 2026-08-10:** it was the cause. Fixing the dataloader took Clean KL from 0.713 to 0.045 and moved logit-diff inside its margin, exactly as predicted before the fix.
- **Method note:** the useful generalisation is that a test which controls a confound by *bypassing* a code path cannot say anything about that path. Feeding identical token ids made the comparison clean and simultaneously excluded the dataloader from the evidence.
- **Code pointer / Run:** `auto_circuit/tasks.py:219`; the probe was throwaway and is gone.

### The attention-bias divergence is removable, and splitting it makes the implementations function-identical on real GPT-2 — 2026-08-09

- **Context:** follow-up to the `b_O` finding. The earlier proof that the port is exact relied on zeroing `b_O` in both implementations, which establishes agreement on a *modified* model. The question was whether the port can instead adopt the original's convention on the real model.
- **It can, in about twelve lines.** TransformerLens computes `result = hook_result(z @ W_O)` per head and adds `b_O` only afterwards, so `b_O` sits outside the maskable decomposition. Adding `b_O / n_heads` inside `hook_result` via a permanent hook and zeroing `b_O` reproduces the original's `modules/attention.py:150` exactly. AutoCircuit's `PatchWrapperImpl.forward` calls the wrapped `HookPoint` and records *its* output as the source activation, so the share lands inside the part an edge can ablate.
- **The model is unchanged:** full-model output before vs after differs by 2.1e-05 on a scale of 19, argmax identical on every prompt. The twelve shares sum back to `b_O`.
- **The implementations become function-identical under masking, with nothing removed from either:**

  | edge density | max&#124;Δlogit&#124; before splitting | after splitting | argmax agreement |
  |---|---|---|---|
  | 0.05 | 2.05e+01 | 2.9e-03 | 100% |
  | 0.25 | 1.65e+01 | 5.8e-05 | 100% |
  | 0.50 | 1.42e+01 | 3.3e-05 | 100% |
  | 0.90 | 6.0e+00 | 2.9e-05 | 100% |

  This is strictly stronger evidence than the zeroed-bias arm, which only showed agreement on a model neither codebase actually uses.
- **Deliberately not the default.** Enabling it globally would change what an edge means for every algorithm, invalidating the 42-cell spine, the stability pass, and the null test, all measured under the current convention, and diverging from both AutoCircuit and Miller (2024). There is also no principled winner: `b_O` is a constant that no head produces, so splitting it twelve ways is as arbitrary as leaving it outside. Exposed as opt-in `split_attn_out_bias()` for the port-comparison study only.
- **Code pointer:** `experiments/port_comparison/forward_equivalence.py`.

### Matching the bias convention closes most of the port gap on unmodified GPT-2 — 2026-08-09

- **Context:** the `b_O`-zeroed arm showed the implementations agree once the bias is removed, but on a model neither codebase uses. `split_attn_out_bias` instead gives each head its `b_O / n_heads` share, matching the original's convention while leaving GPT-2 exactly intact. Ten fresh discovery runs (`port-split`), compared against the existing ten `orig` runs.
- **Result:** four of six endpoints move to equivalence, and none remains a discrepancy.

  | endpoint | as shipped | with matched bias | margin | verdict |
  |---|---|---|---|---|
  | edge density | 5.00% vs 7.67% | **7.54% vs 7.67%** | 0.02 | fidelity supported (p=0.84) |
  | Answer Pair KL | 0.826 vs 0.442 | **0.450 vs 0.442** | 0.240 | fidelity supported (p=0.95) |
  | Jaccard gap | 0.0355 | **0.0074** | 0.0337 | equivalent |
  | Spearman gap | 0.0546 | **0.0098** | 0.10 | equivalent |
  | Clean KL Div | diff 3.293 | diff **0.713** | 1.026 | inconclusive (p=0.34) |
  | Logit Diff Percent | diff 285.4 | diff **102.5** | 84.33 | inconclusive (p=0.07) |

- **This resolves the residual the zeroed arm could not.** Answer-Pair KL was the one endpoint that stayed non-equivalent after removing `b_O` (−0.212 against a 0.240 margin); with the convention matched instead of the bias deleted it falls to 0.008. Confirms that splitting is the better control, and that the leftover was an artifact of running on a bias-free model rather than a real implementation difference.
- **Two endpoints stay unresolved, and not purely from low power.** Clean KL sits at +0.713 with CI [−0.445, 1.86]; Logit-Diff-% at +102.5 with CI [20.2, 187], a **point estimate above its own margin**. Neither difference test is significant, so by the pre-registered grid both read "inconclusive", but the logit-diff point estimate is large enough that a real remaining difference cannot be excluded.
  - Leading suspect is the KV-cache padding defect (next entry), which is uncontrolled in both arms and is known to change 3.1% of predictions with nothing ablated.
- **Code pointer / Run:** `experiments/port_comparison/run_cells.py --arm port-split`, array 418280 (10/10, no failures); `results/port_fidelity/tier_c/summary.md`.

### The bias divergence is per-edge, and only for edges out of attention heads — 2026-08-09

- **Context:** the divergence is easy to describe loosely as "deleting a head takes its bias share with it", but circuit discovery ablates **edges**, never heads. The per-edge behaviour was measured rather than inferred.
- **The mechanism stated precisely:** the two implementations assign different *values* to the same source node. The original's head source is `result_h + b_O/n_heads`; AutoCircuit's is `result_h`, with `b_O` added once into the residual stream outside the graph. Keeping an edge costs nothing because the sums reconcile. Ablating edge `(head_h -> D)` makes the original subtract `result_h + b_O/n_heads` from `D`'s input while AutoCircuit subtracts only `result_h`, leaving the share behind.
- **Measured, one edge at a time into `blocks.11.hook_resid_post`:**

  | source of the ablated edge | that layer's `b_O/n_heads` | max&#124;Δlogit&#124; |
  |---|---|---|
  | embedding | — | 1.6e-05 |
  | head, layer 0 | 1.0351 | 8.90e-02 |
  | head, layer 5 | 0.1993 | 2.83e-02 |
  | head, layer 11 | 20.0493 | 7.41e-01 |
  | MLP, layer 0 | — | 1.6e-05 |
  | MLP, layer 11 | — | 1.7e-05 |

  Edges from embeddings and MLPs disagree by nothing at all. Only attention-head sources do, and the size tracks that layer's share, compressed by `ln_final` (layer 11's share of 20 moves the logits by 0.74).
- **It accumulates per edge, not per head.** One head feeding four destinations is four edges, and ablating all four leaves four separate residues. Disagreement is exactly linear in the number of ablated head-source edges at first (0.089, 0.178, 0.357, 0.725 for 1/2/4/8) then sub-linear as LayerNorm compresses (1.43 at 32, 2.52 at 128).
- **Why the whole-circuit effect is large:** discovery keeps ~2,000 of 32,491 edges, so it ablates ~30,000, nearly all from head sources. That is how the per-edge residue compounds into the 14-20 disagreement seen on real discovered circuits.
- **Code pointer / Run:** the per-edge probe was throwaway; the mechanism it measured is covered by `tests/prune_algos/test_disco_gp_equivalence.py` and `experiments/port_comparison/forward_equivalence.py` (`split_attn_out_bias`).

### DiscoGP port fidelity, Tier C: the two implementations differ, and `b_O` explains it — 2026-08-08

- **Context:** 45 matched discovery runs on the frozen IOI split (10 PORT + 10 ORIG primary, 5+5 `b_O`-zeroed mechanism, 5 no-BOS, 5+5 negative controls), all scored in one harness at k=2000. Endpoints, TOST margins, and the interpretation grid were fixed in `docs/superpowers/specs/2026-08-08-discogp-port-fidelity.md` **before** any run. Array 418206, 45/45 cells, zero failures.
- **Primary result (both implementations as shipped): a genuine discrepancy on every functional endpoint.** Port reaches 5.00% edge density against the original's 7.67%; Clean KL 10.77 vs 7.47; Logit-Diff-% 389.5 vs 104.1; Answer-Pair KL 0.826 vs 0.442. All four survive Holm correction (p_holm 3.0e-04 to 3.6e-03), and none is equivalent at its pre-registered margin.
  - Structurally the gap is marginal rather than dramatic: within-port Jaccard 0.177, within-orig 0.135, between 0.121, so the within-minus-between gap is 0.0355 against a margin of 0.0337. Spearman's gap (0.0546) is *inside* its margin. The implementations are only slightly less like each other than each is like itself.
- **Mechanism arm settles the cause.** Removing `b_O` from **both** models collapses the discrepancy:

  | endpoint | as-shipped gap | `b_O`-zeroed gap | margin | verdict when zeroed |
  |---|---|---|---|---|
  | Clean KL Div | 3.293 | **0.023** | 1.026 | fidelity supported (p=0.92) |
  | density | 0.0268 | **0.0051** | 0.02 | fidelity supported (p=0.56) |
  | Logit Diff Percent | 285.4 | **22.2** | 84.33 | equivalent (negligible shift) |
  | Jaccard gap | 0.0355 | **0.0059** | 0.0337 | equivalent |
  | Spearman gap | 0.0546 | **0.0099** | 0.10 | equivalent |
  | Answer Pair KL Div | 0.385 | −0.212 | 0.2404 | still not equivalent |

  Five of six endpoints move from "genuine discrepancy" to equivalence. The Tier B finding therefore does not merely coexist with the Tier C gap, it **accounts for** it: the entire practical difference between the two implementations is the `b_O / n_heads` decomposition, not a defect in the port.
- **The residual:** Answer-Pair KL still differs by −0.212 with CI [−0.283, −0.148] against a margin of 0.240, so a small difference survives `b_O` removal on that endpoint alone. It is the metric most sensitive to the answer-pair margin the objective directly optimises, which is consistent with [[logit_diff_vs_answer_pair_kl]]. Not explained; report it rather than round it away.
- **BOS / left-padding contributes nothing detectable.** `port-nobos` vs `port`: density equivalent (p=0.85), Answer-Pair KL equivalent, KL and logit-diff inconclusive but with small point estimates. The factor was measured rather than disclaimed, and it is not the story.
  - Note this comparison is effectively **paired**: both arms are the port, so seed *s* gets the same mask init and batch order in each, and only BOS differs. That is why between-arm Jaccard (0.265) exceeds within-arm (0.177/0.190). Expected, not an artifact to fix.
- **Negative controls demonstrate the power.** `port-neginv` (mask inversion removed) is flagged by 5 of 6 endpoints, `orig-asshipped` (unspaced answer tokens) by 3 of 6. Both misses are structural: Spearman fails on both, Jaccard and density fail on `orig-asshipped`. This confirms the pre-registered hierarchy, that functional endpoints carry the argument and structural ones are secondary, and it means a structural null must never be read as evidence of fidelity.
  - Honest qualifier: with n=5 the control arms detect mostly by *rejecting equivalence* rather than by a significant difference test (e.g. `orig-asshipped` Clean KL has p=0.25 but a CI outside the margin). Detection here is the weaker of the two senses.
- **Significance:** the study can now say something stronger than "the port matches". It says the port is exactly correct as an implementation (Tiers A, B), that as-shipped the two codebases nonetheless produce measurably different circuits, and that this difference is fully localised to one documented architectural choice about where the attention output bias lives. That is a much more useful claim for the thesis than a bare equivalence result, and it is the kind of thing a reviewer can check.
- **Analysis bug caught before interpretation:** the first pass paired each `b_O`-zeroed arm against the *bias-intact* opposite implementation, which confounds the very factor the mechanism arm exists to isolate, and paired each negative control against the opposite implementation rather than the arm it perturbs. Every comparison must isolate exactly one factor. Fixed and re-run before any number was read.
- **Code pointer / Run:** `experiments/port_comparison/{run_cells,fidelity_analysis,fidelity_stats}.py`; `tests/prune_algos/test_fidelity_stats.py` (19 pass). Results: `results/port_fidelity/tier_c/{summary.md,summary.csv,negative_controls.csv}`.

### DiscoGP port fidelity, Tiers A and B: the port is exact, and the only divergence is `b_O / n_heads` — 2026-08-08

- **Context:** Building the port-vs-original comparison the supervisor asked for. The study separates two claims that need different evidence: whether the two programs implement the same algorithm (deterministic, Tiers A/B) and whether they find the same circuits from the same data (distributional, Tier C). Tiers A and B are complete; Tier C is not yet run.
  - Terminology: this measures **port fidelity**, not the thesis's `KL(full‖circuit)` faithfulness. The two must not share a word in the write-up.
- **Tier A — component equivalence (22 assertions, all pass).** `tests/prune_algos/test_disco_gp_components.py` pairs each component of the port against the original under a shared seed. `gumbel_sigmoid` agrees **bitwise** (both draw `logits.new_empty([2, *shape]).uniform_(0,1)`, so they consume the RNG stream identically), including its straight-through gradient. `schedule_epoch_lambda`, the answer-margin faithfulness loss, the completeness loss, the sparsity normalisation `Σσ(l)/N_edge`, and the mask reversal all match exactly. What was previously established by reading source is now asserted in CI.
  - The original's `gumbel_sigmoid` has **no `hard` flag** — it always applies the STE. So `hard=True` is the port setting that corresponds to the original, and the `hard=False` variant is an extension with no counterpart.
  - The port guards its schedule divisor with `max(n_epoch_warmup, 1)` where the original does not. The guard is unreachable (the branch requires `epoch < n_epoch_warmup`), so it is not a behavioural difference. Pinned by a test so it cannot later be mistaken for one.
- **Tier B — the edge bijection is exact.** The two implementations enumerate the same graph (157 sources, 445 head-level destinations, 32,491 edges) and order sources identically: embed, then per block `[head_0..head_11, mlp]`. The only layout difference is that the original stores Q/K/V masks as `(prev_nodes, n_heads)` where AutoCircuit uses `(n_heads, prev_nodes)`, so the map is a transpose for Q/K/V and the identity for MLP and final-residual masks (`experiments/port_comparison/edge_map.py`).
- **Observation (the finding):** with the same circuit installed on both sides, the implementations agree at full density (max|Δlogit| = 3.0e-05 on a logit scale of 19) but diverge sharply under masking — median max|Δlogit| ≈ 14–20 at densities 0.05–0.75, with argmax agreement as low as 3.6%.
  - Comparing all 157 node contributions directly localises it: the embedding node matches exactly (0.0), MLP nodes match to 2.3e-04, and **every attention-head node differs by exactly `b_O / n_heads`** (residual after subtracting that term: 1.4e-03).
  - The original folds the attention output bias into each head (`modules/attention.py:150` adds `b_O / n_heads` per head), so ablating a head edge also removes `b_O / n_heads`. AutoCircuit's `hook_result` excludes `b_O`, which is added once outside the maskable decomposition, so ablating a head edge leaves the full `b_O` in place. The sums agree at full density, which is why this is invisible until something is masked.
  - **Magnitude:** on GPT-2 small, `|b_O / n_heads|` at layer 11 is **20.05** — the same order as the entire logit scale. This is not a rounding detail.
- **Confirmation that it is the *only* divergence:** re-running the whole sweep with `b_O` zeroed in **both** models collapses the deviation to the float floor — median max|Δlogit| = 3.3e-04 at density 0.25 and 3.5e-05 at 0.5–0.99, with **100% argmax agreement**. Below density 0.25 it rises (2.9e-02 at 0.1, 0.90 at 0.05), which is LayerNorm amplifying a near-zero residual sum, not a mask error.
- **Significance:**
  - The port's edge bijection, its `AC_mask = 1 - disco_mask` inversion, and the equivalence of AutoCircuit's zero-ablation with the original's dropping of terms from the node sum are all **verified exactly**, not argued. A permutation error inside any mask shape could not survive 40 random masks per density at 100% argmax agreement.
  - But the two implementations do **not** compute the same function under masking. "Ablate this head" is a different operation on each side. Any port-vs-original circuit comparison inherits this, and it must be stated as a bound rather than disclaimed in passing.
  - This sharpens the existing `ablation_patching_mechanics` note from a qualitative caveat into a measured quantity, and it applies to the thesis's own DiscoGP results, whose zero-ablation leaves `b_O` in place.
- **Single-edge probes need care:** ablating index 0 of any mask only ever hits the *embedding* source, which carries no `b_O`, so such a probe passes at the float floor while testing nothing. The first version of this diagnostic did exactly that and reported a false all-clear. Probes must be selected by source **kind** (embed / head / MLP).
- **Code pointer / Run:** `experiments/port_comparison/{edge_map,forward_equivalence}.py`; `tests/prune_algos/test_disco_gp_{components,equivalence}.py` (22 + 9 pass). Results: `results/port_fidelity/forward_equivalence/{deviations.csv,summary.json}` (64 prompts, 40 masks per density, 7 densities, both arms).

### The original DiscoGP's IOI answer tokens omit the leading space, and ~48% of them are word fragments — 2026-08-07

- **Context:** Scoping a head-to-head comparison of our DiscoGP port against the original implementation (`frankniujc/disco_gp`, cloned at pinned commit `54aef81`, vendored under `third_party/disco_gp`). Before matching hyperparameters it was necessary to check that both sides optimize the same objective.
- **Observation:** they do not. The original builds the IOI prompt by truncating the full sentence immediately before the IO name (`setup_task.py:206`), so the prompt ends `"...gave a ring to"` with no trailing space and the correct continuation is `" Michael"`. But the answer token IDs are taken from the **bare** names (`setup_task.py:226-227`, `tokenizer(io_list)`), giving `"Michael"`. GPT-2 uses byte-level BPE, so these are different vocabulary entries.
    - Verified on 200 prompts generated by the original's own pipeline: `target good` differs from the true continuation token on **200/200** examples.
    - Under real GPT-2, their answer token is **never** the argmax (0.00%, median rank 724); the spaced token is the argmax 94% of the time (median rank 0).
    - Full-model IOI accuracy is **80.5%** under their metric versus **100%** under the standard one; mean logit-difference +2.725 versus +3.850.
- **Observation — the consequence the source alone does not reveal:** their own comment asserts "IO and S are expected to be single tokens", which holds only *with* the space. **46 of 99 names in their `NAMES` list fragment without it; 0 of 99 fragment with it** — the list was curated for the standard spaced convention. Because the code keeps only `token_ids[0]`, the answer becomes a word fragment: `Ashley→'Ash'`, `Stephanie→'Step'`, `Nicholas→'Nich'`, `Amber→'A'`.
    - On 1000 generated examples, **47.6%** of IO answers and **48.5%** of S answers are fragments.
    - Seven collision groups share a first token (`Daniel`/`Danielle`, `Amber`/`Alicia`, `Samantha`/`Sara`, …). In **0.4%** of examples `target_good == target_bad` exactly, so the 2-class cross-entropy is pinned at `log 2` and accuracy is forced to zero (the comparison is a strict `>`).
- **Observation — it is confined to IOI.** Their BLiMP path does `targets_good.append(' ' + target_good)` (`setup_task.py:109`) and PARAREL does `ds_dict['answer'].append(' ' + target)` (`setup_task.py:352`). IOI is the only pipeline that omits the space, which is what marks it as unintentional rather than a deliberate convention.
- **Observation — the defect is isolated to the answer token IDs; the rest of the data path is correct.** Audited 500 examples against explicit invariants (`.scratch/audit_data_path.py`): prompts never end with a trailing space (0/500), padding is right-side so real tokens are never displaced (0/500), and position `seq_lens-1` always holds the prompt's final token (0/500), i.e. the logits are read at the position that genuinely predicts the continuation. Meanwhile `target good` differs from the true next token on 500/500 and `target bad` on 500/500, with 245/500 (49%) of IO answers multi-token and 3/500 degenerate. So the fix is exactly the one line at `setup_task.py:226-227`, and a corrected arm is a clean single-variable change. (Note: the IO name does appear inside the prompt, but that is the IOI task by construction, not a leak, since the model must retrieve the earlier mention.)
- **Observation — the published 100% accuracy is nevertheless reproducible with the bug present.** Running their own `search(modes='e')` end to end (seed 42, fp32, wandb off, edge-only) on both token variants:

  | arm | baseline acc | pruned acc (train/eval/test) | edge density | KL (eval) |
  |---|---|---|---|---|
  | as-shipped (bare) | 85.1% | 99.5 / 100 / 100 | 10.48% | **0.189** |
  | corrected (spaced) | 99.4% | 100 / 100 / 100 | 8.69% | **0.020** |
  | paper Table 2 (joint) | — | 100 | 2.03% | **0.020** |

    - Pruning **raises** accuracy from 85% to ~100% under the as-shipped metric, because the search spends 100 epochs maximizing precisely that margin. So the paper's headline IOI accuracy is attainable either way, and the bug does not invalidate it.
    - The consequence is that **sheaf accuracy is an optimized quantity, not a preserved one**: the circuit outperforms the unpruned model by 15 points on the as-shipped metric. Accuracy alone therefore cannot separate a faithful circuit from an over-fitted mask, which is an argument for reporting KL alongside it (as the paper does) and for the spine's KL-primary convention.
    - **KL is where the tokenization actually shows**: 0.189 as-shipped versus 0.020 corrected, against the paper's reported 0.020. This is consistent with the published numbers having been produced under correct tokenization, with the defect entering at the September 2025 "DiscoGP Release" refactor, but it is not conclusive: the paper's row is the joint weight+edge run whereas this is edge-only, so the densities (2.03% vs 8.69%) are not comparable and the KL agreement may be coincidental.
    - The two arms find **measurably different circuits** from identical seeds and hyperparameters (10.48% vs 8.69% edge density), so the choice is not cosmetic for a structural comparison.
- **Significance:** on roughly half its training signal the original's IOI objective maximizes a margin between word fragments rather than names, so it is not the logit difference used by ACDC, EAP, EP or AutoCircuit. Two consequences for sub-RQ1: (i) a port-versus-original comparison run on identical prompts would still be comparing **different objectives**, so the answer tokens must be controlled before any gap is attributable to the implementation; (ii) the original's published IOI numbers are computed against this objective and are therefore not directly commensurable with the spine. Our port is unaffected — all 1000 answers in `datasets/ioi/ioi_prompts.json` carry the leading space.
- **Caveat on the baseline comparison in their Table 2:** ACDC (51.6%) and EAP (58.3%) do not optimize this margin, so they are scored at whatever accuracy their circuit happens to yield, against a full-model baseline that is only 85% under the as-shipped metric. Whether their reported baseline numbers were produced under the same tokenization cannot be determined from the release, and should not be asserted either way.
- **Code pointer / Run:** `third_party/disco_gp/disco_gp/data/setup_task.py:206,226-227`; verification scripts `.scratch/verify_answer_tokens.py`, `.scratch/verify_name_fragmentation.py`, `.scratch/run_original_baseline.py` (`--prune`, `--corrected`); logs `.scratch/orig_prune_{asshipped,corrected}.log`.
- **Environment note:** the release targets a newer TransformerLens than our 1.19. Three model-config attributes must be injected to run it: `attn_scale=sqrt(d_head)=8.0`, `use_qk_norm=False`, `init_range=0.02` (unused, `init_weights=False`). Also, `disco_gp.data.setup_task` resolves to the *function*, not the module (`data/__init__.py` re-exports it), so monkeypatching must go through `sys.modules`.

### Run-to-run nondeterminism is set-iteration order, not GPU floating point — 2026-08-01

- **Context:** Scoping Pillar 2 (stability). Before varying the seed, checked whether re-running the *identical* command reproduces a cell. Probe: `docstring`, seed 42, each cell run twice per node on two nodes (`nelson`, `carl`), caches compared bitwise and by top-k Jaccard.
- **Observation — same node, same seed, back-to-back runs are not reproducible for three of four algorithms:**

  | algo | bitwise equal | J@50 | J@200 | J@500 |
  |---|---|---|---|---|
  | EAP | **yes** | 1.000 | 1.000 | 1.000 |
  | EP | no | 0.75 / 0.64 | 0.42 / 0.48 | 0.41 / 0.43 |
  | DiscoGP (paper) | no | 0.33 / 0.16 | 0.40 / 0.33 | 0.46 / 0.36 |
  | ACDC | no | 0.37 / 0.08 | 0.59 / 0.58 | 0.79 / 0.81 |

  (two values = the nelson and carl replicate pairs.)
- **It is not GPU floating point.** EAP is *bitwise* identical across same-node repeats, so the forward/backward kernels are deterministic on a fixed device. Cross-node, EAP differs by only `max|Δ| = 3.6e-5` and its top-k ranking is **unchanged** (J = 1.000 at every k), so genuine FP differences are far too small to explain J ≈ 0.4.
- **Cause: container iteration order is not canonical, and it varies per process.**
    - `model.edges` is a plain `set` (`graph_utils.py:176`); `model.wrappers` / `model.dest_wrappers` are `Set[PatchWrapperImpl]`, i.e. sets of `nn.Module`, whose hash is identity-based and therefore cannot be pinned by `PYTHONHASHSEED`.
    - Verified across 3 processes on one node with one seed: `edges`, `srcs` and `dest_wrappers` iteration digests all differ, and ACDC's sorted edge list starts with 6 completely different edges each time.
    - **EP / DiscoGP:** the seeded RNG stream is consumed in `dest_wrappers` order, so the *same named mask* receives a *different* slice of the stream each run. Verified directly: `blocks.0.hook_k_input`'s first three uniforms differ between processes despite `t.manual_seed(42)`. Mask init and per-step Hard-Concrete/Gumbel noise therefore both change.
    - **ACDC:** `sorted(model.edges, key=lambda x: x.dest.layer, reverse=True)` (`ACDC.py:85`) sorts on a coarse key with a stable sort, so the many ties within a layer retain set order. The greedy removal order changes, and greedy decisions cascade.
    - **EAP is immune** because it writes each edge's score into its own slot, so order is irrelevant. `src_ablations` is likewise immune: it explicitly re-sorts by `src_idx` and asserts the result (`ablation_activations.py:129-130`).
- **`PYTHONHASHSEED=0` does not fix it** (tested end-to-end): EP J@200 = 0.49, DiscoGP 0.53, ACDC 0.56. Confirmed the variable was in effect (string hashes stable across processes). The `nn.Module` identity hashes remain unpinned, so the fix has to be code-level (sort by a total-order key such as `module_name` / `edge.name`), not an environment variable.
- **Significance:**
    - Pillar 2 as currently specified would not measure what it claims: re-running one command already gives J@200 ≈ 0.4–0.6, so a seed sweep conflates seed variance with uncontrolled ordering variance. Either pin the order and then vary the seed, or explicitly define and report the same-seed re-run as the noise floor that seed-variance is measured against.
    - Every cached spine cell is one draw from this distribution. The faithfulness curves remain valid (each cell is a real circuit, discovered under the stated protocol), but **no cell is reproducible from its manifest**, which is a reproducibility caveat the write-up must state.
    - It also sharpens the per-algorithm stability story: EAP is exactly reproducible, EP/DiscoGP inherit optimization noise, and ACDC's instability is an ordering effect rather than the arithmetic effect previously recorded (erratum applied to the 2026-07-27 entry).
- **Code pointer / Run:** SLURM jobs 416192 (nelson), 416193 (carl), 416270 (`PYTHONHASHSEED=0`); probe scripts were scratch under `.prune_scores_cache/_dettest/` (gitignored, since deleted). To reproduce: run the same cell twice into two `--output-dir`s and compare the cached `PruneScores` bitwise and by top-k Jaccard; to inspect ordering, digest `[w.module_name for w in model.dest_wrappers]` in several processes. Order-sensitive sites at the time: `graph_utils.py:176`, `ACDC.py:85`, `edge_pruning.py:182-195`, `disco_gp.py:313`.
- **Follow-up:** fixed the same day, see the next entry.

### Making discovery reproducible: canonical iteration order — 2026-08-01

- **Context:** Fix for the nondeterminism in the previous entry, so Pillar 2 can measure seed variance rather than ordering variance.
- **Change (3 sites, all "iterate a canonical order instead of a set"):**
    - `PatchableModel.ordered_dest_wrappers` (new property): `dest_wrappers` sorted by `module_name`. One place to state the invariant; `patch_masks` is now built from it, so its dict order is canonical and every `patch_masks.items()` consumer (EP init, Random, SP, the completeness metrics) inherits the fix.
    - `edge_pruning.py` and `disco_gp.py`: build `dest_wrapper_dict` from `model.ordered_dest_wrappers`.
    - `ACDC.py`: sort edges on `(-dest.layer, e.name, seq_idx)` instead of `dest.layer` alone, so the within-layer ties that dominate the greedy sweep have a total order.
- **Result — same node, same seed, back-to-back runs are now bitwise identical for all four algorithms** (docstring; verified on both nelson and carl, jobs 416289/416290):

  | algo | before (J@200) | after |
  |---|---|---|
  | ACDC | 0.59 / 0.58 | **bitwise identical** |
  | DiscoGP (paper) | 0.40 / 0.33 | **bitwise identical** |
  | EP | 0.42 / 0.48 | **bitwise identical** |
  | EAP | already identical | bitwise identical |
- **EP needed a second fix, and the diagnosis is the interesting part.** After the `patch_masks` fix, EP alone was still irreproducible. It was *not* a kernel: `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` raised no error and did not help, and running EP **twice inside one process** was bitwise identical. Every input (batch contents, weights, initial `edge_log_alphas`/`node_log_alphas`, `in_srcs`) was byte-identical across processes. The culprit was EP's own `dest_wrapper_dict`, still built from the raw set: the effective-mask loop builds the **autograd graph** in that order, so the backward accumulated gradients in a different order. The resulting float difference is invisible for ~138 steps and then compounds: the two runs' loss traces agree exactly to step 138 and diverge at step 139.
- **Cross-node is still not bitwise reproducible**, and cannot be: different GPU architectures give genuinely different rounding. EAP is unaffected in ranking (max|Δ| 3.6e-5, J = 1.000), but for the optimization-based methods the difference compounds (EP J@200 ≈ 0.65, DiscoGP ≈ 0.91). **Stability runs must therefore pin the node**, or report cross-node variance as a separate, named axis.
- **Consequence for existing results:** the cached spine cells were discovered under the old arbitrary ordering, so recomputing any cell now gives a different circuit. The curves remain valid (each cell is a real circuit under the stated protocol) but the cache and any re-run are no longer comparable edge-for-edge. Re-running the spine under the fix is what makes it reproducible going forward.
- **Regression test:** `tests/utils/test_canonical_iteration_order.py` (3 tests) pins `ordered_dest_wrappers` sortedness, `patch_masks` key order, and that ACDC's sort key is a total order with descending layers.
- **Code pointer / Run:** `auto_circuit/utils/patchable_model.py` (`ordered_dest_wrappers`), `auto_circuit/prune_algos/{edge_pruning,disco_gp,ACDC}.py`, `tests/utils/test_canonical_iteration_order.py`; jobs 416281/416282 (partial fix), 416289/416290 (full fix), 416296 (post-refactor re-check). The EP diagnosis used three throwaway probes (run EP twice in one process; digest every EP input across processes; run under `use_deterministic_algorithms(True)`), all scratch and since deleted.

### Pillar-2 analysis: seed stability measured, and it tracks functional instability — 2026-08-03

- **Context:** The analysis pass over the 195 cached stability cells (`experiments/stability_analysis.py`, figures in `experiments/stability_plots.py`, spec `docs/superpowers/specs/2026-08-02-pillar2-stability-analysis-design.md`). Structural pass (edge-set agreement between seeds) plus functional pass (each seed's circuit re-scored). Run 416493, 25 min on one A100. **Correction (2026-08-05):** this entry originally described the functional pass as covering "all three eval ablations". It did not. Swept cells were scored only on the diagonal (eval ablation = discovery ablation); only the Random anchor, being ablation-independent at discovery, got all three. The full cross was added later the same week, see the entry below.
- **Observation — a clear and consistent stability ordering, far above the chance floor.** Mean pairwise Jaccard at the published circuit size, worst→best across the three ablations:
    - **EAP 0.89–0.95** on every task and ablation. It is the only method that clears Méloux et al.'s 0.8 bar, and it does so nearly everywhere.
    - **ACDC 0.51–0.82**, **EP 0.31–0.90**, **DiscoGP 0.18–0.37**.
    - **Random 0.002–0.004**, so even DiscoGP is two orders of magnitude above chance: these are not noise, they are genuinely different circuits.
- **Observation — the discovery ablation moves stability as much as the algorithm does.** EP on Docstring goes 0.42 (zero) → 0.90 (resample); on IOI 0.37 → 0.67. The ablation is not a nuisance parameter for stability any more than it is for faithfulness, which is the Pillar-1×Pillar-2 interaction Chen et al. preview with their ACDC 28.6%/4.3% interchange-vs-zero split (Table 4).
- **Observation — structural instability shows up functionally on the primary metric, but on IOI it disappears on the two-class decision.** Spearman ρ between mean pairwise Jaccard and the across-seed standard deviation, over the 12 (algo × discovery-ablation) cells per task, each cell read at its own eval ablation and at its own headline budget:

    | y-metric | IOI | Greater-Than | Docstring |
    |---|---|---|---|
    | Logit Diff Percent | −0.80 | −0.80 | not measured |
    | Clean KL Div | −0.46 | −0.60 | −0.76 |
    | Correct Answer Greater Than Incorrect Percent | **+0.11** | not measured | −0.76 |

    (Metric coverage is not uniform: AutoCircuit registers no logit-diff for Docstring and no two-class accuracy for Greater-Than, `TASK_METRICS` in `faithfulness_comparison.py:110`. Those two panels are drawn empty rather than filled with a substitute.)
    - **Why Greater-Than has no two-class accuracy, and why it is not an oversight.** `correct_answer_greater_than_incorrect_proportion` asserts `answers.shape == (1,)` (`tensor_ops.py:180,190`): it needs exactly one designated correct token per prompt, then checks whether that token outscores every wrong one. Docstring and IOI supply one (`datasets/docstring_prompts.json`: 1 answer, 13 wrong answers per prompt). Greater-Than cannot: its correct answer is any year above the given one, so a prompt carries 2 to 57 correct and 43 to 98 wrong tokens, and the count varies per prompt. The task has no unique right answer, so the metric is undefined rather than unimplemented. A generalisation (max over correct beats max over wrong) is definable, but it would be a different quantity from the one scored on IOI and Docstring and would silently break the row's comparability, which is the whole reason the rows are fixed.

    - **The relationship vanishes exactly where the decision metric saturates.** On IOI every DiscoGP seed scores 98.4–100%, so the decision cannot separate circuits and only the margin does; on Docstring the circuits sit at 9–43%, far from the ceiling, so the decision still varies and tracks structure as strongly as KL does (−0.76 for both). Metric dependence here is a ceiling effect, not a uniform reversal: KL, which cannot saturate, holds a negative ρ on all three tasks.
    - Logit-diff percent is unbounded and measures how far the circuit pushes the margin. Circuits at 334% and 470% are different computations that emit identical predictions.
    - **Correction (2026-08-04):** an earlier version of this entry asserted that the lower-left quadrant of the stability-vs-function figure (structurally unstable, functionally equivalent) is "empty in all three tasks". It is not. It is occupied even on the primary axis (Docstring DiscoGP-zero J = 0.261 at sd 0.0; GT DiscoGP-resample J = 0.206 at sd 15.5), and on IOI's decision metric it holds high performers. The ρ = −0.80 figure itself was right; a first recomputation that reported −0.84/−0.53/+0.03 was joined without the budget column and is superseded by the table above.
    - **Significance — we replicate Chen et al.'s seed-only result rather than contradicting it.** Their Table 10 (DiscoGP, random seeds only, no OASR) reports IOI pairwise IoU **18.5% with both sheaves at 100.0% accuracy**; Table 3 extends this to 20 runs at mutual IoU 0.15–0.30% and 99.6–99.95% accuracy; Table 9 gives a worst-case run of 99.70%. Our DiscoGP/IOI cells give J = 0.176 (zero) at 99.61 ± 0.61% two-class accuracy and J = 0.343 (mean) at 100.0 ± 0.0%: five structurally distinct circuits, every one solving the task. Non-uniqueness at equal quality is not only constructible via OASR, it is what plain reseeding produces on a saturated decision metric. Their Appendix G anticipates our reading directly: "sheaf multiplicity is easily overlooked when only a small number of random seeds are evaluated, as an IoU of approximately 20% may be misinterpreted as algorithmic stability".
    - **For the write-up:** the stability-vs-function figure must name its y-metric in each panel, and any claim that instability is behaviourally consequential has to be stated relative to that metric and to whether the metric has headroom. Same margin-versus-match distinction as [[logit_diff_vs_answer_pair_kl]].
    - **Figure:** `results/stability/<run>/figures/stability_vs_function.pdf` is a 3 × 3 grid, rows = the three metrics above, columns = task, linear y (an sd of exactly zero is a real and interesting value, and a log axis cannot show it without an arbitrary floor), per-panel ρ. `stability_vs_function_by_size.html` is the same grid with a slider over the edge budget: the published size, then 50 to 10 000. Worth sweeping because pairwise Jaccard rises with k for every method including the Random anchor, so an ordering read off one budget has to survive the sweep. ACDC is matched on realised size (its ~20 τ levels cannot hit an arbitrary k) and is dropped from a step when its nearest realised size is off by more than 4×.
- **Observation — mutual IoU is much harsher than pairwise Jaccard.** Agreement of all five seeds at once: EAP 0.80–0.90, ACDC 0.23–0.66, EP 0.08–0.77, DiscoGP 0.02–0.13. DiscoGP's five Greater-Than circuits share 13 edges out of a 715-edge union under zero ablation. The gap between the pairwise mean and the all-five value is agreement that is pair-specific rather than common.
- **Method note — the coefficient of variation is unusable as the functional axis.** It divides by a mean that several cells drive to zero: ACDC/IOI/mean averages 4.15% logit-diff, giving CV 2.2 purely from its denominator. Both task metrics are percentages of the full model, so the figures report the standard deviation in percentage points instead; `functional.csv` keeps both. The same objection applies to Clean KL, which approaches zero for a faithful circuit, so KL is carried as mean ± sd and never as a CV.
- **Method note — ACDC is compared at matched τ, every other method at matched top-k**, decided per cell from the data (few distinct score levels ⇒ τ). This is what Chen et al. do (their Table 4 fixes τ = 0.00398). The τ grid must come from `algo_hyperparams(ACDC_PRUNE_ALGO)`: the 30 τ-split ACDC × {ioi, gt} cells are written by `merge_acdc_partials.py`, whose manifest has no `algo_hyperparams` block.
- **Method note — every budget sweep terminates at the unpruned model (added 2026-08-05).** The first analysis pass capped the functional grid at 10 000 edges and filtered `k < len(edges)` on both passes, so no curve reached the size where the methods coincide, and a reader could not tell an incomplete axis from a genuine failure to converge. ACDC stopped earlier still: the *smallest* τ is its right endpoint, and its grid tops out at 7 008 (IOI resample) to 21 000 (IOI zero) of 32 491 edges, so its line ended a fifth of the way along while the top-k methods ran to 30 000. Every grid now ends with a top-k of all edges, which is the unpruned model whatever the scores say. Faithfulness lands on KL = 0, structure on Jaccard = 1. Headline numbers at the published size are unchanged (verified identical across both runs).
- **Code pointer / Run:** `experiments/stability_analysis.py`, `experiments/stability_plots.py`, new helpers in `auto_circuit/metrics/analysis_utils.py` (`Budget`, `EdgeIndex`, `pairwise_stats`, `mutual_iou`, `selection_frequency`); results in `results/stability/stability_2026-08-05/` (supersedes `stability_2026-08-02/`, same numbers plus the terminal point); jobs 416493, 417094.

### Zero-ablation eval is a different measurement, but the cross-protocol claims do not hold — 2026-08-05

- **Context:** Pillar-2's functional pass extended from the diagonal (eval ablation = discovery ablation, faithfulness only) to the full square: every cell scored under all three eval ablations and both patch directions. Job 417313, 65 min on one A100, 13 572 functional rows. Motivation: Pillar 1 sweeps the same cross with **one seed per cell**, so it can report how a *score* moves with the protocol but nothing about how the *spread* moves.
- **Design note (the correction below turns on it) — the three eval ablations are not three independent protocols.** `RESAMPLE` substitutes activations sampled from the corrupt prompts and `TOKENWISE_MEAN_CORRUPT` substitutes the per-position mean of that same distribution, so mean is roughly the expectation of what resample draws; zero substitutes nothing from the data distribution. Measured on the same cells and budgets (Clean KL, faithfulness, n = 479): **resample ~ mean r = 0.914**, median |Δ| = 0.214; zero ~ resample r = 0.268, median |Δ| = 5.04; zero ~ mean r = 0.289. The 3 × 3 square is therefore 3 discovery ablations × **two** protocol families, with an internal replicate.
    - That replicate is not waste, it is a **null**. Any rank statistic computed across eval protocols must be read against the resample~mean value, which is close to a self-comparison, and not against 1.0.
- **Correction (2026-08-05, same day) — two claims first written here do not survive that null, and are withdrawn.** The user's objection was that resample and mean are near-identical protocols, so a cross-protocol agreement statistic is partly comparing a measurement with itself. Correct, and it removes the two stronger findings:
    - *Withdrawn: "which cells look unstable does not survive a protocol change".* Spearman ρ between 12-cell sd vectors, control first: Docstring control 0.90 vs zero~resample 0.78, zero~mean 0.73; Greater-Than control 0.74 vs 0.64, 0.41; IOI control **0.00** vs −0.29, −0.03. Where the statistic has power, zero agrees with resample about as well as the two corrupt-family protocols agree with each other. On IOI the control is 0.00, so nothing is measurable there and the −0.29 is noise.
    - *Withdrawn: "the ordering reverses".* The EP rank-4-to-rank-1 and DiscoGP rank-2-to-rank-4 swaps were read off IOI, the one task where the control says there is no signal. On Greater-Than and Docstring the ordering is largely preserved across protocols (EAP lowest sd, DiscoGP highest, in both families).
- **Observation (survives, weakened) — zero-ablation evaluation is a different measurement, mostly in level.** Median over the 36 cells at the published size: zero KL 7.621 with sd 0.852, resample 1.205 with sd 0.084, mean 1.157 with sd 0.054. The absolute spread is 10–16× larger under zero, but so is the level, and the **relative** spread is only ~1.5–1.8× larger (sd/mean 0.098 vs 0.066 / 0.054). Most of the apparent instability is the higher operating point, not extra seed-sensitivity. (The standing objection to CV, that the mean approaches zero for a faithful circuit, does not bite here: these means are 1.2–7.6 and the comparison is of the same cells across protocols, not of cells within one.)
- **Significance — the Pillar-2 headline stands unqualified.** "EAP most stable, DiscoGP least" was measured structurally, on edge sets, which have no eval ablation at all, and the functional evidence does not contradict it under any protocol. The methodological lesson is about **this experiment's own design**: with three ablations of which two are near-duplicates, any cross-protocol statistic needs its null stated alongside it, and a bare ρ between protocols is uninterpretable without one.
- **Method note — completeness is measured for Pillar 2 for the first time here.** It postdates the patch-direction fix of 2026-07-31, so it is not comparable to any completeness figure in a results tree dated earlier. Sanity confirmed on Docstring: ablating a discovered circuit drives correct-answer percent to ~0 for every method, while the Random anchor's complement retains performance at small k and only decays as k grows, which is the expected knockout signature.
- **Method note — presentation, so that nine protocol combinations cost no extra files.** Pillar 1 encodes four dimensions as directory levels (discovery / task / measure / eval) with the metric as filename and costs **367 files across 108 folders**. Here discovery × eval became the rows and columns of one 3 × 3 grid inside each figure, the task moved from a folder into the filename, and the grid figures skip the HTML sibling. Result: 14 figures → 28, folders 6 → 4, total 63 → 77 files, for 9× the protocol coverage and a second patch direction. Rule worth keeping: at most two dimensions become paths, the rest become geometry.
- **Code pointer / Run:** `experiments/stability_analysis.py` (`_eval_ablations`, `_functional_rows` now loops `_PATCH_SEMANTICS`), `experiments/stability_plots.py` (`fig_measure_grid`, `_ring_diagonal`); `results/stability/stability_2026-08-05/figures/{faithfulness,completeness}/<task>__<metric>.pdf`; job 417313.

### Near-identical eval baselines are not near-identical discovery baselines — 2026-08-05

- **Context:** direct follow-up to the entry above. Having found that resample and mean are near-duplicates as *evaluation* protocols (r = 0.914), the open question was whether the same collapse holds on the *discovery* axis. If it did, the spine's "three discovery ablations" would really be two, a third of the discovery compute would be a replicate, and every claim of the form "the result depends on the discovery ablation" would have to narrow to "depends on zero versus the corrupt-baseline family". The existing CSVs cannot answer it: they record structural agreement only *within* a discovery ablation.
- **Design — the within-ablation seed Jaccard is the null.** Two runs of the same ablation do not agree perfectly either, so a bare cross-ablation Jaccard cannot be read as "different". The comparison is cross-ablation agreement against same-ablation seed-to-seed agreement at matched budget, over all 12 (task, algo) cells × 3 ablations × 5 seeds. Exact top-k for EP/DiscoGP/EAP; matched τ for ACDC.
- **Observation — the collapse does not carry over.** J(resample, mean) never reaches the null: ratio 0.32 to 0.98, mean 0.70 over the 12 cells (IOI: DiscoGP 0.32, EAP 0.44, ACDC 0.62, EP 0.82; Greater-Than 0.58 to 0.76; Docstring 0.62 to 0.98). The two-family structure nevertheless survives, since res~mean is 2 to 5× higher than either zero pairing in every cell. Three discovery conditions, unevenly spaced, resample and mean adjacent.
    - A Spearman on the full score vectors reproduces the ordering in all 12 cells, so this is a shift in the ranking rather than a top-k boundary artifact.
    - ACDC's matched-τ comparison also mismatches circuit size (resample/mean size ratio 0.58 on IOI, 0.74 on Greater-Than, 1.19 on Docstring), so its two GPT-2 ratios are lower bounds.
- **Observation — seed-stability and protocol-insensitivity are separate properties, and EAP separates them cleanly.** EAP is the most seed-stable method in the study (within-ablation Jaccard 0.91 to 0.93) yet its IOI resample and mean circuits agree at only 0.41. Mechanically expected rather than anomalous: EAP scores grad·(corrupt − clean), so the baseline enters the score formula directly, shifting it systematically while adding no run-to-run variance. Nothing about a low seed spread predicts what happens when the baseline moves.
- **Significance.** Optimization amplifies a baseline difference that evaluation smooths over. Evaluation is a single forward pass, smooth in the substituted activation, so two baselines that differ only by within-distribution variance give nearly the same number. Discovery contains thresholds and greedy choices, so the same difference selects different edges. The practical consequence for the write-up is that the eval-side caveat (state the r = 0.914 wherever the three eval ablations appear as separate columns) does **not** extend to the discovery side: "three discovery ablations" stands, and no Pillar-1 or Pillar-2 claim needs weakening on that account.
- **Code pointer / Run:** `.scratch/discovery_ablation_null.py` (702 rows) and `.scratch/rank_null.py`, both CPU-only over `.prune_scores_cache/stability`; reuse `_load_cell` / `_acdc_tau_grid` from `experiments/stability_analysis.py`.

### Pillar-2 stability run complete: 195 cells, and two audit findings — 2026-08-02

- **Context:** Post-run audit of the seed-variance discovery run (`.prune_scores_cache/stability/`, 195 cells = 4 swept algos × 3 tasks × 3 ablations × 5 seeds + Random × 3 tasks × 5 seeds).
- **Run health:** all SLURM elements `COMPLETED 0:0` (jobs 416307/416309 array, 416310 partials, 416329 merge); 195 pkl + 195 manifests, 150 partials, zero orphans, zero stray `.tmp`, zero load failures. All 30 ACDC merges consumed exactly 5 partials and yielded 21 score levels (20 τ + `inf`). No NaN, no `-inf`, no `+inf` outside ACDC, no degenerate (constant-score) cell. Data budget 704/256 on every manifest.
- **Finding 1 — one site of the canonical-order fix was missed, and it is the one that silently corrupts *analysis*.** `mask_gradient.py:112` still iterated the raw `model.dest_wrappers` set when assembling its output dict, so EAP (which the spine runs via `mask_gradient_prune_scores`) emitted **15 different key orders across 15 cells**, while EP/DiscoGP/ACDC/Random all emitted the one sorted order.
    - **Values are unaffected.** The gradients already sit on each wrapper's `patch_mask.grad` before that loop runs, so only dict insertion order varies. This is consistent with EAP having been bitwise-reproducible throughout.
    - **The hazard is downstream.** Any analysis that flattens `PruneScores` into a single index space (concatenating tensors to compare two runs) silently compares mismatched edges. This produced a spurious "EAP is at the chance floor" reading (J@200 = 0.004 on IOI/GT, identical to Random) that contradicted the correct value.
    - `analysis_utils.top_k_edge_set` is **immune**, because it resolves scores through `Edge.prune_score(prune_scores)`, i.e. by module name. Pillar-1 results and the heatmaps are therefore unaffected, and the 195 cached cells need no re-run.
    - Fixed by iterating `ordered_dest_wrappers`; added `test_prune_scores_keys_are_sorted`, which asserts `new_prune_scores()` key order matches `patch_masks`.
- **Finding 2 — ACDC is the only method with real score ties, and the choice of how to cut them moves the stability number by ≤ 0.06.** ACDC assigns only 21 distinct values over 32 491 edges (score = the τ at which the edge was pruned; `+inf` = never pruned), so a rank-k cut lands inside a block of equal scores. Every other method has near-continuous float scores: the block on the k=200 boundary is **exactly 1 edge** for DiscoGP / EAP / Random, and 1–2 for EP (1–12 at k=1000 where its masks saturate). ACDC's is 8–165 edges, of which 2–91 must be picked.
    - **Selection is a threshold, not a top-k.** `run_circuits` reads the k-th largest `|score|` and sets `patch_mask = (|score| >= threshold)` per tensor (`prune.py:94`, noted at `:87`), so a straddling tie block is admitted whole. ACDC therefore overshoots the requested budget (request 200 → get 200–353; request 1000 → get 1001–1521) while every other method lands exactly on the request.
    - **Three defensible pairing rules for the stability comparison.** (a) *Threshold at fixed k*, as above. (b) *`topk` at fixed k*: exact sizes, tie broken by a rule carrying no information. (c) *Matched τ*: ACDC's score **is** τ and the τ sweep is a fixed list, so 20 of 20 levels are shared across seeds (17–18 on docstring, where some τ removed nothing); `{edges with score >= τ}` is then a circuit defined by a shared hyperparameter rather than by an arbitrary size.
    - **Measured, mean pairwise cross-seed Jaccard near 200 edges:**

      | task / ablation | (a) threshold | (b) topk | (c) matched τ |
      |---|---|---|---|
      | docstring / mean | 0.699 | 0.670 | 0.699 |
      | docstring / resample | 0.708 | 0.648 | 0.647 |
      | docstring / zero | 0.616 | 0.610 | 0.609 |
      | gt / mean | 0.746 | 0.747 | 0.746 |
      | gt / resample | 0.691 | 0.742 | 0.757 |
      | gt / zero | 0.503 | 0.493 | 0.487 |
      | ioi / mean | 0.602 | 0.635 | 0.602 |
      | ioi / resample | 0.653 | 0.656 | 0.653 |
      | ioi / zero | 0.451 | 0.447 | 0.425 |

    - **The tie question is second-order.** All three rules agree to ≤ 0.06, well inside the spread across ablations (0.43–0.76), so ACDC's stability signal does not depend on how ties are cut. Two secondary effects, both small: a deterministic (index-order) tie-break manufactures ≈ +0.026 median agreement (+0.108 worst, ioi/mean) relative to a randomized tie-break; and unequal sizes under rule (a) cap *individual pairs* at `|A|/|B|` (0.58 for the 204-vs-353 pair), though the mean over pairs barely moves because most pairs are similarly sized.
    - **Consequence for the Pillar-2 analysis:** use **matched τ** for ACDC. The gain is defensibility rather than accuracy: it compares circuits produced by the same hyperparameter, which is what "re-run the method" means, and its realized sizes come out tight anyway (`[202, 203, 203, 206, 209]` for gt/mean, `[200, 201, 206, 207, 207]` for ioi/resample). Other methods keep the shared k-grid. For cross-method panels, read off ACDC's realized sizes and evaluate the others at exactly those sizes, which they can hit since their boundary blocks are single edges.
    - **Pillar 1 needs no change, for the same reason.** `run_circuits` stores each result under `patch_edge_count`, the number of edges actually kept (`prune.py:102`), not the requested k. Since the kept set is always a complete tie block, ACDC's faithfulness points already *are* τ-circuits plotted at their true sizes. This is also the mechanism behind the known coarse ACDC curve: several requested budgets fall inside one tie block, produce the identical circuit, and collapse onto the same `patch_edge_count` key. The curve is honest, not degraded; a stability comparison differs only because it must fix a shared coordinate for *two* circuits before comparing them.
- **Code pointer / Run:** `auto_circuit/prune_algos/mask_gradient.py:112`, `tests/utils/test_canonical_iteration_order.py`; audit scripts kept at `.prune_scores_cache/stability/{audit_cells,validate_cells,seedcheck,overshoot,topk_option,natural_grid,tau_grid}.py`.

### What "completeness" means in the source literature, and how to measure it here — 2026-07-31

- **Context:** Designing the completeness experiment, following the finding that the completeness column ran the opposite patch direction to the one it documents (fixed 2026-07-31; see Notes, 2026-07-27 rev. 2026-07-31). Read Miller et al. (2024) §3.1.5 + Table 3, Wang et al. (2022) §4, the DiscoGP camera-ready (EMNLP 2025) §4-5, Bhaskar et al. (2024) and Conmy et al. (2023).
- **Observation:** the literature holds **two different things** under the word completeness, and only two of the five papers define it at all.
    - **Wang et al. (2022) — the original definition, subset-based.** "for every subset K ⊆ C, the incompleteness score |F(C∖K) − F(M∖K)| should be small." Measured by searching for a bad K three ways: uniformly at random, K = an entire class of circuit heads, and greedily node-by-node to *maximise* the incompleteness score. Random and class-based sampling suggested the IOI circuit was complete; greedy search found sets scoring **3.09**, 87% of the original logit difference. Ablation = mean over the ABC distribution; metric = logit difference. This is a search problem, not a single measurement.
    - **DiscoGP (Yu et al., 2025) — the practical definition, complement-based.** §4: "We also report completeness or the complement sheaf accuracy (i.e., the accuracy when the sheaf is zero-ablated and all other model components are kept on)." The Table 2 column header is `Comp. Acc. (%) (random* is better)`: the target is **chance**, 50% for IOI/BLiMP and ~0% for PARAREL, not a maximum or a minimum. Training-time counterpart is `L_complete = CE(complement output ‖ uniform over the label space)` on the reverse mask `m̃ = 1 − m` (Eq. 6). The paper attributes this fidelity/completeness/sparsity scheme to Wang 2022 + Conmy 2023 + Syed 2024 + Bhaskar 2024, but only Wang actually defines a completeness test.
    - **EP, ACDC, EAP measure no completeness at all.** EP reports output KL faithfulness plus an "edge faithfulness" check (an edge important for the model should be important for the circuit); ACDC evaluates by ROC against ground truth. So on IOI Table 2 the reported completeness numbers are 49.2 (DiscoGP), 49.9 (EP), 50.6 (ACDC), 55.2 (EAP): every method sits at chance and the column separates nothing. Completeness is only discriminative when read against a matched-size control.
    - **Miller et al. (2024) never uses the word** except once in passing ("Pareto frontier of faithfulness, completeness and simplicity"). What it supplies is the direction taxonomy, Table 3, four rows over (input, direction, set): Clean/Ablate/Circuit → *destroy performance*; Corrupt/Restore/Circuit → *restore performance*; Clean/Ablate/Complement → *maintain performance*; Corrupt/Restore/Complement → *maintain inefficacy*. §3.1.5 compares only rows 2 and 3, and notes the asymmetry: restoring clean activations into the circuit on a corrupt input lets "signal from the clean input flow through edges not included in the circuit", whereas ablating the complement "ensures signal from the input only flows through the circuit".
    - **DiscoGP's completeness is exactly Miller Row 1**, with ablation value = zero and metric = task accuracy. That is the row to target.
- **All four Miller rows are reachable from the existing machinery** with two booleans, no new code in `prune.py`: `(TREE_PATCH, reverse=False)` = Row 3 (current faithfulness, correct); `(EDGE_PATCH, reverse=True)` = Row 1 (knockout completeness, the one wanted); `(EDGE_PATCH, reverse=False)` = Row 2 (what the completeness column currently runs); `(TREE_PATCH, reverse=True)` = Row 4. Rows 2 and 4 are only interpretable at resample eval, since zero and mean supply no clean activations to restore; rows 1 and 3 are meaningful under all three eval ablations.
- **Significance:**
    - The metric direction has to change with the row. Under Row 1 the good outcome is *far from the full model*, so `CLEAN_KL_DIV_METRIC`'s "lower is better" gloss and the "0 = identical to the full model" axis annotation invert. The DiscoGP-matched readouts are the chance-referenced ones already registered: `CORRECT_ANSWER_GREATER_THAN_INCORRECT_PERCENT` → 50%, `LOGIT_DIFF_PERCENT` → 0%.
    - The curve shape mirrors faithfulness: at 0 edges ablated nothing is destroyed, so Row 1 starts at full-model performance and falls toward chance as the circuit grows. The informative statistic is the value at a matched sparsity budget, not an AUC.
    - **Completeness needs the Random anchor to mean anything.** Ablating k edges degrades the model partly just by volume; the Random baseline at the same k is the destruction-by-volume floor. Reporting completeness as a *margin over Random at matched edge count* is what turns it from a restatement of sparsity into a claim about the circuit. This is the design decision the DiscoGP table does not make, and it explains why its IOI completeness column is flat.
    - Wang-style subset-K incompleteness is a separate, stronger sub-study, and its cost scales with |C| per greedy step: cheap on Docstring (24 official edges), expensive on IOI (739). Scope it to Docstring plus the small end of the IOI curve rather than the full 42-cell matrix.
- **Code pointer:** `experiments/faithfulness_comparison.py:92-98` (`_PATCH_SEMANTICS`), `:319-322` (the `measure_prune_metrics` call missing `reverse_clean_corrupt`), `auto_circuit/prune.py:67-78` (the four-way input/patch-source branch), `auto_circuit/metrics/prune_metrics/measure_prune_metrics.py:40` (`reverse_clean_corrupt` already plumbed through).

### Data budget unified across spine tasks: "dataset size" is four different knobs — 2026-07-27

- **Context:** Reviewing whether IOI's larger dataset (768 train) should be matched by GT (128) and Docstring (256), and what the source papers use.
- **Observation:**
  - The four algorithms consume the train loader in incompatible ways, so a single "dataset size" number does not describe the experiment:
    - ACDC reads **only the first batch** (`ACDC.py:95`, `next(iter(dataloader))`) → its N is `batch_size`, and `batch_count` is irrelevant to it.
    - EP cycles the loader for a fixed 3000 steps (`edge_pruning.py:268`) → cost and per-step gradient signal depend on `batch_size` only; `batch_count` sets unique-prompt coverage.
    - DiscoGP runs 100 edge epochs over **all** batches (`disco_gp.py:328`) → its total optimizer step count is `train_batches × e_train_epochs`, i.e. proportional to `batch_count`.
    - EAP makes one pass over the whole train loader.
  - Consequence of the old per-task settings (IOI 128×(6,1), GT 64×2, Docstring 128×2): ACDC and EP saw half as much data on GT as on the other tasks, and **DiscoGP got 600 edge-phase steps on IOI but only 200 on GT/Docstring**. The second is the more damaging confound: DiscoGP was systematically under-trained on exactly the two tasks where it scores worse.
  - Reference protocols: ACDC uses 100 / 100 / 50 examples (IOI / GT / Docstring, single batch, `acdc/main.py`); DiscoGP uses `batch_size 64`, `n_ioi_data 1000`, 0.8 split → 800 train ≈ 12–13 batches × 100 epochs ≈ **1300 edge steps** (`cli.py` defaults); EP uses `per_device_train_batch_size 32`, `max_steps 3000` → ~96K example-presentations, the source of the paper's "100K examples" claim.
  - EP's 100K is presentations, not unique prompts: at `batch_size 64` × 3000 steps the port already exceeds it (192K). The binding constraint here is the 1000-prompt dataset pools, not gradient signal.
- **Decision:** all spine tasks now share `SPINE_BATCH_SIZE=64`, `SPINE_BATCH_COUNT=(11, 4)` → **704 train / 256 test**, identical across IOI / GT / Docstring (and Docstring-token, so the granularity sub-study stays controlled). This gives DiscoGP 1100 edge steps (≈ paper's 1300), ACDC a uniform N=64, EP 2× the paper's batch, and doubles the IOI/GT eval set to 256 (Miller's per-datapoint-unreliability point). Total 960 ≤ the 1000-prompt pools, so no dataset regeneration.
- **Cost:** invalidates all 42 cached cells and every downstream figure/CSV; cache filenames do not encode the data budget, so re-runs need `--force` or a fresh output dir.
- **Code pointer / Run:** `auto_circuit/tasks.py:263` (`SPINE_BATCH_SIZE` / `SPINE_BATCH_COUNT`); loader shapes verified on a compute node (11 batches / 704 ex train, 4 / 256 test on all four tasks). Full re-run into `.prune_scores_cache/faithfulness_v2/`: job 414762 (36-cell array) + 414785 / GT partials (IOI+GT ACDC, τ-split 5 ways, merged by element-wise min) → merge → analysis.

### Cross-cell circuit-similarity heatmaps: structural (dis)agreement axis added to the spine — 2026-07-20

- **Context:** Faithfulness-spine analysis pass. New `experiments/similarity_heatmaps.py` computes, per task, the pairwise Jaccard similarity between the top-k edge sets of all 14 cells (4 swept algos × 3 discovery ablations + GT + Random) at fixed budgets (default k ∈ {500, 1000, 2000, 5000}, capped to `n_edges`; Docstring only gets 500/1000). Wired into `faithfulness_comparison.py` (`--similarity-edge-counts`) and runnable standalone on CPU from cached cells. Outputs: per-k PDF heatmaps, slider HTML, tidy CSV under `<experiment>/similarity/<task>/`.
- **Observation:**
  - Smoke run on the 42 cached seed-42 cells (`results/faithfulness_comparison/similarity_smoke/`): structural agreement is low everywhere. IOI @ k=500: the largest off-diagonal Jaccard is 0.48 (EP[resample] × ACDC[resample]); most cross-method pairs sit at 0.10–0.35; Random pairs at ~0.01.
  - DiscoGP-EP is the structural outlier: its circuits overlap ≤ 0.18 with *everything* (including its own other-ablation variants, ≤ 0.12), consistent with its margin-maximizing objective finding a different region of circuit space.
  - Same-method-different-ablation is often no more similar than cross-method-same-ablation (EP[zero] × EP[resample] = 0.16 < EP[resample] × ACDC[resample] = 0.48 on IOI): the discovery ablation shifts circuits at least as much as the algorithm does — direct structural evidence for the ranking-robustness question.
  - GT overlap @ k=500 is highest for EP[resample]/EP[mean] on all three tasks (IOI 0.33, GT-task 0.31, Docstring 0.05); Random sits at 0.007–0.016.
  - Two artifacts handled explicitly: (a) tied scores make ACDC's top-k sets overshoot k (IOI zero: 4048 edges at k=500) — Jaccard is computed on the actual sets and the CSV records `size_a/size_b`, so the overshoot is visible, not hidden; (b) binary GT / mostly-zero ACDC scores at k above their nonzero count would admit the whole graph (threshold 0), so edge sets use the new `exclude_zero=True` mode of `top_k_edge_set` — a circuit stops growing at its nonzero-score count (GT stays 24 edges on Docstring at every k).
- **Significance:** Pillar 1 gains a structural-agreement axis alongside the behavioral (faithfulness) axis: methods that score similarly can now be shown to disagree structurally (the Dead-Salmon/Chen non-identifiability point) with one figure per task. The DiscoGP-EP isolation and the ablation-shift ≥ algorithm-shift pattern are candidate Results-chapter findings once confirmed at other budgets.
- **Code pointer / Run:** `experiments/similarity_heatmaps.py`, `auto_circuit/metrics/analysis_utils.py::top_k_edge_set(exclude_zero=True)`, `tests/test_similarity_heatmaps.py`; smoke run `results/faithfulness_comparison/similarity_smoke/`. Side fix: `auto_circuit/utils/misc.py::load_cache` now maps CUDA-saved caches to CPU when no GPU is present (CPU-only analysis nodes previously crashed).

### Erratum: EAP was never "ablation-independent" — promoted from anchor to swept (36→42 cells) — 2026-07-16

- **Context:** The 2026-06-05 spec decision "EAP stays corrupt-only" (and every claim downstream of it: `faithfulness_cells.py` anchors, `run_cell.py` comments, `make_ablation_variant` docstring, CLAUDE.md, Pillar-1 notes) rested on a false premise: that extending EAP to other ablations would require new plumbing in `edge_attribution_patching.py`. In fact the test-verified equivalent `mask_gradient_prune_scores` has exposed `ablation_type` all along.
- **Observation:**
  - The misunderstanding conflated two different properties: GT/Random are genuinely *ablation-independent* (the ablation never enters their computation); EAP was *ablation-bound* (the registered implementation hardcoded resample). The spine treated both as "anchors", so every analysis output re-used EAP's resample scores as reference rows in the zero and mean discovery panels, visually asserting EAP has no discovery-ablation axis.
  - Fix (branch `eap-ablation-audit`): `EDGE_ATTR_PATCH_PRUNE_ALGO.func` → new `eap_prune_scores` wrapper (mask-gradient EAP; derives `clean_corrupt` from the ablation type, so `make_ablation_variant` composes cleanly); EAP moved to `SWEPT_ALGOS`; spine = **42 cells** = 4 swept × 3 ablations × 3 tasks + GT/Random ×3. Old EAP resample caches stay valid (scores bit-equivalent up to 1.4e-6) and were reused via `--skip-if-exists`; job 411201 computed exactly the 6 new EAP zero/mean cells (36 skipped, 0 errors). Tests updated (42-cell composition; EAP grouped per-ablation, GT as the anchor example).
  - Spec carries a dated amendment (§12) rather than a rewritten decision; wrong claims in CLAUDE.md, `docs/related_work/edge_attribution_patching_syed2023.md`, `open_problems_mech_interp.md`, and the memory files were corrected in place.
- **Significance:** Sub-RQ2's discovery-relativity axis now covers all four algorithm families instead of three; the smoke-test divergence (zero reorders EAP's top-100 by 70–75%) suggests the EAP rows will show real discovery-ablation sensitivity, strengthening the ranking-robustness result. Methodology must present resample as EAP's *paper-native* setting (Syed et al. define the score as a linearization of corrupted-prompt patching) and zero/mean as first-order extensions with the §5.1 approximation caveat.
- **Code pointer / Run:** `auto_circuit/prune_algos/prune_algos.py` (`eap_prune_scores`), `experiments/faithfulness_cells.py`, spec §12; discovery job 411201.

### EAP under zero/mean discovery ablation: smoke test passes, rankings diverge — 2026-07-16

- **Context:** Follow-up to the 2026-07-15 finding that EAP's resample ablation is definitional but the Taylor machinery is baseline-agnostic. Smoke-tested EAP via `mask_gradient_prune_scores(grad_function="logit", answer_function="avg_diff", mask_val=0.0)` with `ablation_type` ∈ {RESAMPLE, ZERO, TOKENWISE_MEAN_CORRUPT} on Docstring (component, attn-only-4l) and IOI (component, gpt2-small).
- **Observation:**
  - All three ablations run to completion with finite, fully non-degenerate scores on both tasks; the resample run reproduces the registered `EDGE_ATTR_PATCH_PRUNE_ALGO` exactly (corr = 1.000000, max abs diff ≤ 1.4e-6), confirming the `mask_gradient` route is a faithful EAP.
  - Discovery ablation changes EAP's ranking substantially: top-100 |score| overlap vs resample is 0.25 (Docstring) / 0.30 (IOI) for zero and 0.85 / 0.58 for mean; score-vector correlations 0.32 / 0.52 (zero) and 0.83 / 0.71 (mean). Zero diverges most, consistent with it being the largest off-distribution perturbation for the linearization.
  - API gotcha: `batch_src_ablations` asserts `clean_corrupt` is set only for batch-specific ablations, so ZERO and TOKENWISE_MEAN_CORRUPT require `clean_corrupt=None` (the `mask_gradient_prune_scores` default `"corrupt"` must be overridden).
  - Resource note: the IOI mask-gradient backward OOMs on a 24 GB GPU; needs `--constraint="gpu_mem:80gb"` (or smaller batches).
- **Significance:** EAP-zero/EAP-mean are viable one-line discovery variants, so EAP need not stay a single-protocol anchor in the spine; the large ranking divergence under zero suggests adding them would strengthen the discovery-ablation axis with a fourth (attribution-based) algorithm family. `make_ablation_variant`'s "not meaningful for EAP" note only holds for the registered slow implementation.
- **Code pointer / Run:** smoke script (temp): `/mnt/beegfs/work/akdag/thesis/tmp_eap_ablation_smoke.py`; `auto_circuit/prune_algos/mask_gradient.py:27`

### Pillar 1 faithfulness spine implemented, run, and merged (#4) — 2026-06-16

- **Context:** Closing GitHub issue #4 (Pillar 1 of the sub-RQ3 design). Built the full discovery × eval matrix as a two-step pipeline, ran it end-to-end, reviewed, and merged via PR #8 into `main` (`pillar1-faithfulness` branch).
- **Observation:**
  - **Final cell count = 36** (the spec's ≈30 estimate assumed only EAP as an anchor). Realized composition: **27 swept** = EP / DiscoGP-EP / ACDC × {zero, resample, mean} × 3 tasks, plus **9 anchors** = EAP / GT / Random run once (at resample) per task and reused as reference rows in *every* discovery-ablation panel. Promoting GT+Random to anchors (not just EAP) keeps the matrix uniform. Single source of truth: `experiments/faithfulness_cells.py` (`CELLS`), imported by both the SLURM launcher and the analysis pass so they can never disagree.
  - **Operating model = run-per-setting, then one analysis pass.** Discovery is a SLURM array of `run_cell.py` jobs (`faithfulness_discovery.sh`), one cached `PruneScores` + manifest per cell in `.prune_scores_cache/faithfulness/`; idempotent via `--skip-if-exists` (a requeued task whose cell finished exits immediately). The analysis pass (`faithfulness_comparison.py`) loads the cells (no retraining), dedupes to the newest per cell identity (`analysis_utils.latest_per_cell`), groups by discovery-ablation, and **re-keys** each variant key (`"Edge Pruning [zero]"`) to its base key (`"Edge Pruning"`) so downstream lookups/legends are clean — the discovery ablation lives in the output path, not the algo name.
  - **Deliverables (triple-encoded path):** `discovery-<D>/<task>/<faithfulness|completeness>/<eval-ablation>/<metric>.pdf` (TREE_PATCH = faithfulness, EDGE_PATCH = completeness), `discovery-<D>/roc.pdf`, per-cell circuit-graph HTMLs, tidy `summary.csv` + headline `summary.md` (Clean-KL at the GT edge count), and **28 interactive overlay HTMLs** (`interactive_curves.py`) — one Plotly figure per (task, measure, metric) overlaying every (algo × discovery-ablation) as a legend-toggleable line (color = algo, dash + marker = discovery-ablation) with an eval-ablation dropdown. Raw curve data cached to `interactive/overlay.pkl` so styling tweaks re-render CPU-only via `replot_interactive.py`.
  - **Bug fixes folded in:** `clean_corrupt` is now set per ablation type in EP/DiscoGP (`None` for zero/dataset-mean, `"corrupt"` for batch-specific) — fixes the `batch_src_ablations` assertion under zero/mean discovery; Docstring discovered at **component** granularity (matches GT, see the granularity entry below); circuit-graph viz guards an empty-panel `min/max` and sanitizes ACDC's `+inf` scores to a large finite value.
  - **Refactor (code review pass):** the per-batch ablation list was triply duplicated (library helper + both algos with "must stay in sync" comments) → consolidated to a single exported `BATCH_SPECIFIC_ABLATIONS` in `ablation_activations.py`; the duplicated ablation-slug normalization extracted to `_norm_slug()`.
- **Significance:** Pillar 1 is the faithfulness spine the other two pillars hang off; it operationalizes both headline effects (evaluation-relativity à la Miller 2024 and discovery-relativity) in one navigable, reproducible artifact. The realized design also settled the anchor question (9 anchors, not EAP-only) and the idempotent dedup pattern (`latest_per_cell`) that Pillars 2/3 will reuse.
- **Tests:** `tests/test_faithfulness_cells.py` (36-cell composition), `tests/test_faithfulness_analysis.py` (grouping / anchor-reuse / variant→base re-keying / seed+null filtering), `tests/metrics/test_analysis_utils.py::test_latest_per_cell` — 8 passing.
- **Code pointer / Run:** `experiments/{faithfulness_cells,run_cell,faithfulness_comparison,interactive_curves,replot_interactive}.py`, `auto_circuit/metrics/analysis_utils.py`; latest run `results/faithfulness_comparison/faithfulness_seed42_2026-06-15_1754/` (36 cells, all deliverables). PR #8, closes #4.

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
