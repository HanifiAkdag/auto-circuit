# Shared held-out evaluation

New thesis discovery runs reserve a common test set before varying training seeds. The discovery data seed changes training selection and the PyTorch seed changes optimization; neither changes the held-out test inputs. The default fixed split seed is 42.

```bash
srun --gres=gpu:1 python -m experiments.thesis.run_cell --task ioi --algo ep --ablation resample --seed 43 --split-seed 42 --output-dir .prune_scores/new_comparison
srun --gres=gpu:1 python -m experiments.thesis.stability_analysis --cache-dir .prune_scores/new_comparison --tasks ioi --split-seed 42 --name new_comparison
```

Populate every requested method/ablation/seed before the analysis command. For several data seeds, use the **same `--split-seed`**, source pool, model/tokenizer and budgets in every run. The loader reserves unique clean test inputs using the split seed, removes every training record whose clean or corrupt tokens match those inputs, and samples the requested training rows using the per-run data seed. If the remaining pool is too small, it raises an error. Shared corrupt contexts are allowed: the held-out claim concerns clean inputs.

New cache filenames include `_holdout42`; manifests record the pool hash, split seed, original training/test row IDs, and ordered test-clean token hash. Analysis selects that split identity and verifies its held-out identities and data budget against the evaluation loader. Evaluation protocol IDs include the actual ordered clean/corrupt inputs, answers, pool and tokenizer identities, and loader settings. Cache resume rejects changed pools or budgets. ACDC partial merges preserve and check the split identity too. Historical caches cannot satisfy a new-protocol run through `--skip-if-exists`.

Library users can opt into the same protocol with `Task(..., split_seed=42)` or `load_datasets_from_json(..., random_seed=run_seed, split_seed=42)`. The library's default `split_seed=None` retains compatibility with existing callers; callers that vary data seeds and want a common held-out evaluation set must supply a fixed split seed. `shuffle=False` is incompatible with this option because frozen-file splits are a separate protocol.

`--legacy-split` on discovery and `--legacy-eval` on analysis reproduce the earlier protocol for audit. They do not make common evaluation across varied data seeds held out. Keep their outputs separate. The frozen-split port-validation study remains a separate protocol.

## Temporary repair of the 2026 historical comparison

The temporary artifact at `datasets/evaluation/temporary_repair_20261006/` evaluates existing IOI and Greater-Than caches on 256 fresh clean prompts per task. It is explicitly labeled `temporary_historical_repair` and is never selected by new discovery defaults. It excludes evaluation clean token sequences against both original input roles in the entire original pool; matching corrupt contexts are reported separately.

Reproduction commands, all on compute nodes:

```bash
srun --mem=8G python -m experiments.thesis.audit_evaluation_protocol
srun --mem=8G python -m experiments.thesis.build_evaluation_set --output-dir datasets/evaluation/a_new_temporary_artifact
srun --gres=gpu:1 --mem=32G python -m experiments.thesis.temporary_evaluation_repair --cell-index 0
```

The builder refuses to overwrite a frozen artifact. The existing artifact has its own generator seed, tokenizer vocabulary hash, dataset and record hashes, source pool/generator hashes and identity checks. The repair runner verifies the frozen score/manifest inventory, preserves the historical budgets and attained sizes, and writes one atomic output per independent seed group. Use `--inventory` to list groups and `--merge` after all groups finish. No discovery is repeated.

After all groups and the temporary manual baseline complete, run `python -m experiments.thesis.finalize_evaluation_repair` on a CPU compute node. It merges outputs, regenerates the IOI/Greater-Than exhibits, verifies unchanged structural rows and frozen caches, and writes `report.md` and `evaluation_changes.csv`. `finalization.json` records completion of each step. A failed step stops the pipeline. The no-argument manual-baseline exporter retains the historical protocol; use an explicit temporary manifest, frozen inventory and separate output directory for this repair.

Historical result CSVs and thesis figures remain intact. Preview outputs are written below `results/results_chapter/temporary_repair_20261006/`; exporters require matching evaluation protocol IDs so new measurements cannot be combined with historical baselines. Docstring and port-validation results are outside this temporary correction.
