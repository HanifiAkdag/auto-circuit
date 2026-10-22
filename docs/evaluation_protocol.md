# Evaluation protocols

The launcher exposes `--protocol thesis` (default) and `--protocol fixed-holdout`. Each output directory, selected registry, baseline, and export must use one consistent identity. Port validation uses its own frozen IOI protocol.

## Thesis protocol

The final thesis discoveries use the original prompt-pool splits and per-run discovery seeds. Their common evaluation sets were subsequently frozen with 256 unique clean inputs per task. Every evaluation clean token sequence is excluded from both clean and corrupt roles in the complete original discovery pool. This is a fresh evaluation set for existing discoveries, not a seed-42 pre-discovery split. The fresh pipeline reads the frozen inputs directly for both cached and new discovery outputs.

IOI and Greater-Than use GPT-2 Small and form the primary analysis. Docstring uses the four-layer attention-only model, component-level graphs, and `NeelNanda/gpt-neox-tokenizer-digits`; its results are appendix material. Greater-Than reuses 176 corrupt contexts, so the exclusion guarantee concerns clean evaluation inputs. Docstring has no corrupt-context overlap with either original input role.

Discovery has the shared budget of 704 training prompts in 11 batches of 64. The original pool loader also selected 256 test prompts; the thesis functional evaluation instead uses the separate frozen fresh set. Frozen dataset metadata records generators, seeds, tokenizers, pool hashes, answer identities, and token-sequence exclusion checks.

All four compared methods are discovered under zero, paired resample, and corrupt-tokenwise-mean ablation. Matched discovery/evaluation ablations form the primary comparison; off-diagonal evaluation is a sensitivity analysis. Manual circuits and random edge-score circuits provide task-specific baselines. The random baseline samples edges, not model weights.

Greater-Than Edge Pruning trains with KL restricted to the 100 candidate tokens `00`–`99`, following the released trainer. Evaluation Clean KL is full-vocabulary for every method. Its selected final discoveries and dependent analyses replace the earlier full-vocabulary-training GT EP runs; other methods, IOI, Docstring, and port validation are unchanged.

```bash
bash scripts/thesis.sh reproduce --protocol thesis --run-dir results/reproduction
```

The selected score registry is consumed directly. The public pipeline does not reconstruct historical result trees and then apply a repair. Re-exporting requires completed analyses and manual baselines with the same evaluation identity.

## Fixed-holdout protocol

For new studies, `fixed-holdout` reserves a common seed-42 test split before discovery. The data seed changes training selection and the PyTorch seed changes optimization; neither changes the held-out clean test inputs. The loader removes any training row whose clean or corrupt tokens match a held-out clean input. Shared corrupt contexts remain allowed. It fails if the remaining pool cannot supply the requested training budget.

```bash
bash scripts/thesis.sh discover --protocol fixed-holdout --run-dir results/new-holdout
```

Use the fresh run's registry for evaluation and export under the same protocol. Cache filenames and manifests distinguish the split identity, pool, budget, and ordered evaluation-clean hash. Analysis verifies those identities against the actual loader. Existing thesis caches do not satisfy fixed-holdout resume requirements.

Library callers retain upstream-compatible defaults: `split_seed=None` uses historical splitting. Opt in with `Task(..., split_seed=42)` or `load_datasets_from_json(..., random_seed=run_seed, split_seed=42)`. A frozen-file split uses a separate loader construction and must not also request randomized holdout selection.

## Measures and output identity

Clean KL divergence from the full model is the primary faithfulness measure. Logit difference percent is normalized by the full model, and task accuracy measures sufficiency. Completeness changes the ablated edge set and is reported separately in appendix analyses. Stability compares edge sets across five runs; the structure–function correlation uses the joint-seed condition at the manually identified circuit size and matched evaluation ablation.

Evaluation manifests bind ordered clean/corrupt inputs, answers, tokenizer and model identities, loader settings, and source hashes. Baseline and curve joins reject mismatched or incomplete sources. Neither historical exporter defaults nor an unrelated local results directory may silently fill missing inputs.

The head follow-up is confined to the retained mean-ablation analysis. The port-validation study preserves its independent frozen IOI protocol and must not inherit the thesis evaluation prompts. See [artifact provenance](artifacts.md).
