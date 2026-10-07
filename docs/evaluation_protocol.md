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

Historical result CSVs remain intact. Corrected IOI/Greater-Than outputs are written below `results/results_chapter/temporary_repair_20261006/`; exporters require matching evaluation protocol IDs so new measurements cannot be combined with historical baselines. The manuscript integration is tracked separately in `results/results_chapter/complete_repair_integration_20261006/`. Port-validation results retain their separate original protocol.

## Docstring repair and manuscript integration

The separate frozen artifact at `datasets/evaluation/docstring_repair_20261006/` extends the historical repair to Docstring. It preserves the original rest-style generation geometry and metadata, uses generation seed 20261006, and contains 256 fresh clean prompts excluded from both original input roles. No corrupt contexts overlap either original role. Docstring uses **NeelNanda/gpt-neox-tokenizer-digits**, not GPT-2's tokenizer; identity and single-token-answer checks must use the actual model tokenizer. The builder requires a separate Docstring artifact so tokenizer provenance remains explicit.

```bash
srun --mem=8G python -m experiments.thesis.build_evaluation_set --tasks docstring --output-dir datasets/evaluation/a_new_docstring_artifact
```

Frozen historical Docstring inputs are recorded in `results/results_chapter/complete_repair_integration_20261006/docstring_selected_inputs.json`; functional outputs and matching manual baselines are in `results/results_chapter/docstring_repair_20261006/`. Only circuit evaluation is repeated. Original discovery scores, selectors, budgets and structural measurements remain unchanged. The task-specific smoke check also verifies the full circuit and cached-prefix computation.

The active thesis functional sources are recorded in `results/results_chapter/active_sources.json`. The main comparison uses the repaired IOI/Greater-Than bundle; the Docstring appendix uses its separate repaired bundle. Use explicit `--source-dir` and output arguments for re-exporting these sources. Historical exporter defaults remain available for audit and are not the active manuscript sources. Do not replace corrected thesis displays with historical exports. See the integration plan and report under `complete_repair_integration_20261006/` for the exact figure/table adoption and verification.

## Fresh-run verification boundary

The October 7 verification rebuilt all three repair datasets in empty output directories and exercised fresh EAP discovery, cache resume and stability analysis for all three tasks. This representative run does not rerun the complete method/ablation/seed matrix or test installation without downloaded model resources. Details and logs are in `results/results_chapter/fresh_run_verification_20261007/report.md`.

The manual-baseline limitation identified by that audit is addressed by the fixed-holdout export path below. Historical and repaired thesis baseline sources remain separate; reproducing those sources still requires their matching evaluation protocol.

## Manual baselines for new fixed-holdout runs

Generate the official circuit cell in the new cache folder with the same split seed, then export its baseline against the completed analysis source. For the IOI example above:

```bash
srun --gres=gpu:1 python -m experiments.thesis.run_cell --task ioi --algo gt --ablation resample --seed 42 --split-seed 42 --output-dir .prune_scores/new_comparison
srun --gres=gpu:1 python -m experiments.thesis.results_chapter_manual_baseline --split-seed 42 --cache-dir .prune_scores/new_comparison --source-dir results/stability/new_comparison --output-dir results/results_chapter/new_comparison/exhibits/size_curves
```

For a multi-task analysis, generate one official cell per task in that cache folder. Baseline task scope defaults to the analysis source; an explicit `--tasks` must match it. Fixed-holdout export requires a completed `--source-dir` and a separate `--output-dir`. It reconstructs the production evaluation Tasks and checks their actual content-derived protocol ID against the analysis before scoring. This identity includes ordered clean/corrupt inputs, answers, pool, tokenizer and loader construction. The selected official cache must also match the holdout identity and data budget. Missing inputs do not fall back to historical caches.

Official circuits retain their native evaluation ablations: mean ablation for IOI and resample ablation for Greater-Than and Docstring. Both faithfulness and completeness use the same clean evaluation set as the comparison. Output manifests record the evaluation-clean hash per task, the analysis manifest hash and completion status; curve joins reject incomplete or mismatched sources.

The no-argument exporter remains historical audit mode. `--eval-manifest` with `--selected-inputs` remains the frozen temporary-repair mode and is mutually exclusive with `--split-seed`. Repair score files are resolved from their inventory paths. The scoped implementation and verification report is `results/results_chapter/fixed_holdout_baseline_20261007/report.md`.
