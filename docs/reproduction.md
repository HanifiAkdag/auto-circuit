# Reproducing the thesis experiments

The public entry point is `bash scripts/thesis.sh ACTION`. It submits work through SLURM and keeps Python, dependency installation, and tests off the login node. Run commands from the repository root. Setup and analysis outputs never target the frozen bundle or the separate Overleaf repository.

## Environment and resources

`setup` creates the Conda prefix `.venv-thesis` with Python 3.10.20, installs the pinned `requirements-thesis.txt` environment and this checkout in editable mode with the `thesis` extra. The pinned PyTorch build uses CUDA 13 (`cu130`); the compute-node driver must support it. Poetry remains the package build backend, rather than a second environment installation route.

```bash
bash scripts/thesis.sh setup --dry-run
bash scripts/thesis.sh setup
```

Models and reference repositories are downloaded at recorded revisions during setup/preflight. They are not bundled in Git. Preserve the setup log and recorded versions with the run. An existing environment can be selected with `THESIS_PYTHON`; consult `bash scripts/thesis.sh --help` for launcher and SLURM configuration. Never replace a model revision with a moving branch when reproducing the frozen measurements.

GPU jobs need enough memory for the selected method. DiscoGP's answer-margin objective requires more than 40 GB for the thesis GPT-2 workloads; select an 80 GB or larger GPU. CPU stages include integrity checks, statistics, and figure export. Port validation also needs the separately downloaded reference implementations.

## Commands

| Action | Inputs | Outputs beneath the run directory | Hardware |
| --- | --- | --- | --- |
| `setup` | Pinned dependencies and resource specifications | Setup logs; environment and resource preparation | CPU compute node with network |
| `verify` | Registry, selected score/manifest pairs, frozen datasets, hashes | Integrity report | CPU compute node |
| `reproduce` | Verified cached bundle | Evaluation, baselines, analyses, exports | GPU then CPU |
| `discover` | Original discovery pools and exact run matrix | Fresh scores, manifests, registry | GPU discovery jobs then CPU finalization |
| `evaluate` | Selected or fresh registry and chosen protocol | Functional curves, stability, matched baselines | GPU |
| `export` | Completed evaluation and structural sources | Tables, PDF figures, machine-readable exports | CPU |
| `port-validation` | Reference revisions, validation scores, frozen IOI inputs | Port comparisons, endpoint statistics | GPU then CPU |
| `head-diagnostics` | IOI caches and mean-ablation follow-up inputs | Retained-head conditions and analyses | GPU then CPU |
| `hpo` | Historical nested-CV search configuration | Trials, fold scores, refit artifacts | GPU |
| `smoke` | Small representative configuration | Pipeline smoke report | GPU |
| `test` | Library and pipeline test collection | Test logs | Compute node |

All actions accept `--run-dir`, `--registry`, `--protocol`, and `--dry-run`. Defaults are `results/reproduction`, `artifacts/final/registry.json`, and `thesis`. `--dry-run` prints planned submissions without launching them. Each output directory should contain only one protocol and one discovery identity.

The launcher returns job IDs after submission. Wait for successful completion before starting a dependent action. Discovery's internal arrays and finalization have explicit SLURM dependencies. Extra pipeline options follow `--`; use `bash scripts/thesis.sh --help` for the complete interface.

The default GPU request is one GPU constrained to at least 80 GB, with concurrency limited to two discovery tasks. Override placement with `THESIS_PARTITION`, `THESIS_QOS`, or `THESIS_ACCOUNT`; resource overrides include `THESIS_CPUS`, `THESIS_CPU_MEM`, `THESIS_GPU_MEM` (host RAM), `THESIS_GPU_COUNT`, and `THESIS_CONCURRENCY`. For UKP's lower-priority queue, set both `THESIS_PARTITION=yolo` and `THESIS_QOS=yolo`. The launcher uses `sbatch`, which supports this queue.

## Re-evaluate the selected thesis circuits

```bash
bash scripts/thesis.sh verify
bash scripts/thesis.sh reproduce --run-dir results/reproduction
```

This path reads the 342 selected benchmark score/manifest pairs and frozen fresh evaluation prompts. It does not repeat optimization. The resulting analyses use the frozen registry directly; no historical repair or overwrite step is required. Matching manual baselines are evaluated against the same clean prompts and protocol identity. Export reads those completed sources and writes a separate result tree.

To run these stages after a fresh discovery has finalized its registry:

```bash
bash scripts/thesis.sh evaluate --run-dir results/fresh-thesis
bash scripts/thesis.sh export --run-dir results/fresh-thesis
```

These stages require `results/fresh-thesis/discovery/registry.json` and reject a run without it. Use `reproduce` for the bundled final caches.

The main-task results are IOI and Greater-Than. Docstring remains separate appendix material. Matched discovery/evaluation ablations are primary; crossed evaluations test sensitivity. Faithfulness is measured with full-vocabulary Clean KL, with task accuracy and logit difference percent as additional functional measures. Stability uses five runs and pairwise edge-set Jaccard. A random edge-score baseline is retained; there is no random-weight null experiment in the thesis pipeline.

## Fresh discovery

```bash
bash scripts/thesis.sh discover --run-dir results/fresh-thesis --protocol thesis
```

The thesis matrix contains 342 discovery identities. Its SLURM plan consists of 312 regular jobs and 150 ACDC partial jobs: 30 ACDC identities each split into five independent threshold groups. A CPU finalization stage checks completion, merges the partials by element-wise minimum, and writes a registry pointing to the new files. Fresh evaluation must consume that run's registry, never silently fall back to the selected historical bundle.

Under `thesis`, discovery uses the original per-run pool splits and evaluation uses the immutable fresh 256-input set for each task. Under `fixed-holdout`, a common seed-42 split is reserved before discovery; train against and evaluate that split consistently. The latter is a new experimental protocol, not a reproduction of the thesis's selected scores. See [evaluation protocols](evaluation_protocol.md).

The shared discovery data budget is 704 training prompts (11 batches of 64). Different methods consume those batches differently. Greater-Than Edge Pruning uses the reference's `00`–`99` training-KL candidates, while Clean KL evaluation remains full-vocabulary. Changing pools, data budgets, objectives, or split protocol requires a new run directory and registry.

## Validation and follow-up analyses

```bash
bash scripts/thesis.sh port-validation --run-dir results/port-validation
bash scripts/thesis.sh head-diagnostics --run-dir results/head-diagnostics
```

Port validation is an IOI study with a separate frozen protocol. Component and fixed-circuit agreement do not establish unconditional complete-procedure equivalence. In particular, native DiscoGP comparisons depend on attention-output-bias decomposition; the decomposition-aligned comparison is validation-only. Use the registered endpoints and decision rules from the bundled validation evidence.

The retained-head follow-up uses mean ablation and the active `mean_v3` configuration (4,360 conditions). It is separate from the main discovery comparison. HPO remains available for historical exploratory analysis and did not select the main thesis comparison's configurations.

