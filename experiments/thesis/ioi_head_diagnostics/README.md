# IOI retained-head diagnostics

Controlling protocol: `docs/writing/ioi_retained_head_mean_protocol_20261011.md`.
Execution ledger: `docs/writing/ioi_retained_head_mean_execution_20261011.md`.

This is a **mean-only** study: four methods × five mean-discovery runs, with baseline and causal probes at 739/2000/3000 requested edges. Evaluation uses `TOKENWISE_MEAN_CORRUPT`, including every output replacement and attribution contrast. The 2000-edge condition targets the displayed peak. Prior resample outputs are audit-only and excluded. A new inventory is mandatory; old results cannot be resumed under changed scientific code.

The study re-evaluates frozen canonical discovery scores; it does not discover circuits, change pretrained weights, or edit the thesis. Connectivity is component-level, affecting every token position on an edge. Attention is measured at the final query position on each prompt's separately identified IO, S1 and S2 tokens.

## Local study inputs

The repository intentionally does not track scientific caches, evaluation results, cluster launch scripts or thesis-writing material. Before creating an inventory, supply the local protocol named above, `results/results_chapter/active_sources.json`, its referenced evaluation manifest and selected inputs, and the corresponding canonical `.prune_scores/final/` score/manifest pairs. A source checkout alone is insufficient to reproduce these measurements. The inventory records their content hashes, and completed studies must retain those exact inputs and scientific code for verification or resume. Keep generated outputs, model assets, plotting dependencies and session notes out of the code commit.

## Execution

Run Python only on SLURM compute nodes. Use a fresh output directory whenever scientific code or inputs change. The local, gitignored `run_job.sh` helper contains cluster-specific paths and requests one GPU, four CPUs and 32 GB RAM. The Python entry point below remains usable through your own compute-node launcher. Create the output log directory before submission.

```bash
sbatch --output=results/ioi_head_diagnostics/NEW/logs/preflight_%j.log experiments/thesis/ioi_head_diagnostics/run_job.sh preflight results/ioi_head_diagnostics/NEW
# Only after matching successful smoke validation:
sbatch --array=0-19%2 --output=results/ioi_head_diagnostics/NEW/logs/cell_%A_%a.log experiments/thesis/ioi_head_diagnostics/run_job.sh all results/ioi_head_diagnostics/NEW
# Only after every array task succeeds:
srun --cpus-per-task=4 --mem=32G /storage/ukp/work/akdag/miniconda3/envs/autocircuit/bin/python -m experiments.thesis.ioi_head_diagnostics --phase summarize --output-dir results/ioi_head_diagnostics/NEW
```

`--phase screen|causal|size-sensitivity --cell-index N` runs an individual phase. Existing matching condition results are checksum-verified and skipped. A completed array can be resubmitted safely. Concurrent duplicate workers for the same case are not supported; different case workers safely share the intact reference under a file lock.

For six-GPU yolo execution, use `sbatch --partition=yolo --qos=yolo --array=0-5%6 ... run_job.sh shard OUTPUT 6`. Each task processes the frozen cases at stride six. Stop previous overlapping jobs before changing the shard count; completed conditions resume without remeasurement. CPU summary jobs require `--partition=cpu --qos=cpu`.

## Output contracts

- `inventory.json`: content-derived study identity, input/software hashes, graph, cases and exact predefined conditions; `masks.npz`: Boolean retained-edge masks indexed by condition ID. Original score caches are never overwritten.
- `smoke.json`: observed production parity, graph and intervention numerical checks, immutable model-weight hash and GPU provenance.
- `reference.json`: atomic completion marker and hashes for the intact reference and ordered prompt/token-position metadata.
- `conditions/ID.json`: complete condition identity, attained size, changed edges, retained NNM output flags, per-prompt file hash, summary and optional alias. Identical mask/hybrid conditions share the measured array file without dropping experimental labels.
- Per-prompt NPZ fields: `correct`, `wrong`, `margin`, `kl`, `accuracy`; `kl_partition`, `kl_pair`, `kl_other` reconstruct full KL (nats): pair-versus-rest mass divergence and full-model-mass-weighted conditional pair/rest divergences. `correct_probability`, `wrong_probability`, `pair_probability` and `entropy` describe the full-vocabulary predictive distribution. `attention` and `scores` are `[prompt, head, IO/S1/S2]`; head order is 10.7, 11.10. `dla` and `contrast_dla` are `[prompt, head, IO/subject logit]`. `vectors` are actual head results and are omitted for randomized sensitivity repetitions. `final_scale` is the condition's final LayerNorm scale.
- `contrast_dla` compares actual head output with the fixed **mean replacement** head output. It is not an absolute output or the causal output-edge effect. No resample evaluation enters the study.
- `per_run.csv`, `paired_effects.csv`, `effect_summary.csv`: all endpoints, paired conditions and discovery-run uncertainty. `logit_difference_percent_points` intervals resample the shared intact-margin denominator.
- `exact_per_run.csv`: exact-size endpoints averaged within each discovery run. Already-exact native circuits enter once; affected tie resolutions are averaged, preserving all five discovery runs for every method/budget. Exact paired summaries follow the same rule.
- `exhibits/`: PDF/PNG figures; `report.md`: standalone evidence and limitations; `complete.json`: successful verification of every predefined condition, including aliases and no-ops.

Native thresholds use score magnitude and include all ties, including valid ACDC infinite scores. Requested and attained budgets are different fields. Exact-size circuits are randomized sensitivity subsets, not native outputs. Restored graph Q edges carry circuit source activations; `full_q_*` conditions instead replace the complete intact-clean pre-LayerNorm Q input and are hybrid diagnostics.

There are five discovery runs, not 20 control runs or ten tie-resolution runs. The crossed bootstrap resamples discovery-run blocks and shared prompt indices. Attention, diagnostic attribution and causal output-edge effects are distinct evidence. Interaction is `(baseline − output-ablated) − (Q-restored − Q-restored/output-ablated)`; group effects need not add.

Fixed-budget Q swaps preserve the baseline attained count by removing eligible non-NNM-incident edges before restoration. Their paired removal-only controls remove the same edges without restoration and are smaller circuits. The swap-minus-removal contrast measures restoration conditional on those removals, not a comparison of two equally sized circuits.

Keep budget explicit in every effect and summary. Output-removal labels mean replacement of direct contributions by their fixed mean source values, not deletion of complete heads. Mean replacement is pooled across the corrupt loader, unlike Wang et al.'s template-conditioned ABC means. The component graph also differs from their head/token-position knockouts; see `thesis_notes/limitations.md`.

## Figure-only dependencies

The shared autocircuit environment lacks Matplotlib and its plotting/parser dependencies. Keep it unchanged. Install `plotting_requirements.txt` using `pip install --no-deps --target OUTPUT/plotting_dependencies -r ...` on a compute node, then set `PYTHONPATH` to that absolute directory and `MPLCONFIGDIR` to a task-specific temporary directory when aggregating. Base NumPy, Pillow, packaging, python-dateutil and six remain in the original environment. Verify an Agg-backend PDF render before launching aggregation. The measured masks, weights and per-prompt arrays do not depend on these plotting packages.
