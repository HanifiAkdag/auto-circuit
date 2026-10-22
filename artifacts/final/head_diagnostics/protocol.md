# IOI retained-head study: mean ablation only

Supersedes the resample causal study for reporting, per explicit author instruction on 2026-10-11. Old score caches, experiment outputs and reports remain audit artifacts and are not imported into this study.

## Scope

Use four methods (ACDC, EAP, Edge Pruning, edge-only DiscoGP), five joint discovery runs (42–46), **mean discovery and mean evaluation only**, and requested budgets 739, 2000 and 3000. Baseline and causal measurements cover all 60 method/run/budget combinations. The 2000-edge condition targets the displayed mean-ablation KL/logit peak; neighboring budgets test size dependence. Apply production inclusive magnitude thresholds and record attained sizes. For tied thresholds, evaluate ten reproducible nested exact-size realizations at each affected budget and the Q/output probes in each realization.

Use the active 256-prompt IOI historical-repair holdout, corrupt counterparts, pretrained GPT-2 Small weights, float32, batch size 64, TF32 disabled, and component-level separate-Q/K/V graph. Every edge intervention acts at every token position; attention is observed at END with IO/S1/S2 identified separately per prompt. The data have already been analyzed; this is an exploratory follow-up, not an untouched confirmation.

Mean replacement is `AblationType.TOKENWISE_MEAN_CORRUPT`, computed from the full corrupt evaluation loader using the production averaging implementation. All output-edge removals substitute these mean source values. Restored edges transmit actual circuit source activations; intact-clean Q replacement remains a separately labeled hybrid. Attribution contrasts use the same mean replacement, not an individual corrupt activation.

## Interventions and controls

Preserve the predefined intervention families from the first protocol: individual/joint NNM Q restoration, joint K/V/QKV restoration, individual/joint direct-output replacement, Q restoration plus output replacement, intact-clean Q input hybrids, Name Mover/S-Inhibition/other source-group Q restoration, 10.7→11.10.Q retain/drop, and ordinary/backup/joint Name Mover direct-output replacements. Do not force NNM output retention; record applicability and no-ops.

At every method/run/budget, generate 20 count/stratum-matched absent-edge Q controls, matching destination layer, Q channel and source kind and excluding NNM destinations. Record unavailable strata without relaxation. Generate ten protected fixed-budget Q pairs by removing the same eligible retained edges in removal-only and restoration conditions, excluding every NNM-incident edge. Preserve the native attained count. Control randomization is keyed by discovery identity **and budget**. Ties are resolved with one random edge ordering per run/realization, preserving nesting across budgets. Exact-size realizations and controls are averaged within discovery runs, not treated as replications.

## Measurements and inference

Save per-prompt correct/wrong logits, raw margin, strict correct-over-incorrect accuracy, full-vocabulary `KL(full || circuit)`, IO/S1/S2 attention and QK scores, final LayerNorm scale, actual NNM outputs and LayerNorm-scaled diagnostic attribution. Save correct/wrong/pair probabilities and predictive entropy. Decompose KL exactly into the Bernoulli divergence for pair-versus-rest probability mass, plus full-model-pair-mass-weighted conditional pair KL and full-model-rest-mass-weighted conditional remaining-vocabulary KL. Validate that the three terms reconstruct total KL. These readouts diagnose distribution changes without asserting an additive causal share of a head group.

Use five-run averages and sample SD, paired effects and 10000 crossed discovery-seed/prompt bootstrap draws with shared indices and a resampled full-model denominator for logit difference percent. Keep budget explicit in every effect/interval. Report contrary findings and no-ops. Group interventions are conditional and nonadditive. Scope claims to mean ablation and this model/task/evaluation data.

## Verification and provenance

Freeze original score/manifest hashes, selected active inputs, graph, masks, exact changes, protocol/code identity and runtime package/model/tokenizer settings before measurement. New results must not overwrite original caches or active thesis plots. Resume only matching inventories and valid per-prompt checksums. Verify model weights are unchanged, observed forwards are neutral, full patched graph KL is below 1e-5, observer results match production `run_circuits` at **all three budgets**, OV outputs reconstruct, individual Q restoration preserves that head's K/V and direct-output replacement preserves NNM attention. Numerical tolerance is atol=rtol=1e-4. Test mean replacement and budget-specific inference before GPU execution. Use SLURM compute nodes only; six one-GPU yolo shards are authorized.

## Literature comparability limitation

Wang et al. (2022), Section 2.1, uses template-conditioned mean activations over ABC prompts for head/token-position knockouts, and Section 3.1 uses corrupted-example path patching. Our pooled tokenwise corrupt mean and component-edge interventions are not an exact reproduction of that procedure. Therefore matching the word “mean” does not justify treating either ablation replacement or intervention granularity as identical. Save this distinction in `thesis_notes/limitations.md`; do not edit manuscript prose without a separate request.
