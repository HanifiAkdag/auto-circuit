# Circuit Discovery in AutoCircuit

This master thesis fork of [AutoCircuit](https://github.com/UFO-101/auto-circuit) integrates [Edge Pruning](https://arxiv.org/abs/2406.16778) and [DiscoGP](https://aclanthology.org/2025.emnlp-main.446/) into a common circuit discovery and evaluation pipeline. The thesis compares Edge Pruning, edge-only DiscoGP with its answer-margin objective, ACDC, and EAP under zero, resample, and corrupt-tokenwise-mean ablation. IOI and Greater-Than on GPT-2 Small form the primary comparison; Docstring on a four-layer attention-only model is an appendix transfer study.

Hanifi Ibrahim Akdag maintains this thesis fork. The repository retains AutoCircuit's other algorithms, tasks, patching features, and tests. The thesis comparison covers faithfulness and stability. Hyperparameter search is preserved as historical exploratory work; it did not tune the main comparison.

The presentation follows the [UKP project template](https://github.com/UKPLab/ukp-project-template), adapted to this existing scientific codebase. It does not adopt the template's licensing terms.

## Getting started

Use Python 3.10 and SLURM compute nodes. On the UKP cluster, run every installation, Python command, and test through the launcher below; the login node has a two-core, 4 GB limit.

```bash
git clone https://github.com/HanifiAkdag/auto-circuit.git
cd auto-circuit
bash scripts/thesis.sh setup
bash scripts/thesis.sh verify
bash scripts/thesis.sh reproduce
```

`setup` submits environment installation and pinned model preparation to a compute node. `verify` checks the committed scientific bundle against its hashes. `reproduce` evaluates the selected cached scores and exports results without retraining. Outputs default to `results/reproduction/`; frozen inputs and the Overleaf project are never output targets.

The launcher prints SLURM job IDs and returns after submission. Wait for a stage to finish successfully before submitting a dependent action.

For resource settings, dependencies, and separate discovery runs, see [reproduction instructions](docs/reproduction.md).

## Usage

| Command | Inputs | Outputs | Hardware |
| --- | --- | --- | --- |
| `bash scripts/thesis.sh setup` | Dependency lock and pinned model/reference specifications | Environment and downloaded resources | CPU compute node, network |
| `bash scripts/thesis.sh verify` | `artifacts/final/registry.json` and frozen inputs | Integrity report | CPU compute node |
| `bash scripts/thesis.sh reproduce` | Selected cached scores and frozen evaluation prompts | Analyses and exported exhibits | GPU evaluation; CPU export |
| `bash scripts/thesis.sh discover` | Original prompt pools and discovery configurations | New scores, manifests, and run registry | GPU compute node |
| `bash scripts/thesis.sh evaluate` | Registry-selected scores and evaluation protocol | Functional and structural analyses | GPU compute node |
| `bash scripts/thesis.sh export` | Completed analyses and matching baselines | Tables and figures | CPU compute node |
| `bash scripts/thesis.sh port-validation` | Pinned reference implementations and frozen IOI inputs | Port-validation measurements and statistics | GPU compute node |
| `bash scripts/thesis.sh head-diagnostics` | IOI scores and frozen prompts | Mean-ablation retained-head follow-up | GPU compute node |
| `bash scripts/thesis.sh hpo` | Exploratory search spaces and nested-CV configuration | Search trials and refit results | GPU compute node |
| `bash scripts/thesis.sh smoke` | Minimal pipeline configuration | Smoke-test report | GPU compute node |
| `bash scripts/thesis.sh test` | Library and pipeline tests | Test report | Compute node; GPU for GPU tests |

The common options are `--run-dir`, `--registry`, `--protocol thesis|fixed-holdout`, and `--dry-run`. Defaults are `results/reproduction`, `artifacts/final/registry.json`, and `thesis`. For example:

```bash
bash scripts/thesis.sh reproduce --run-dir results/my-reproduction --dry-run
bash scripts/thesis.sh discover --run-dir results/new-discovery
```

The `thesis` protocol combines the original discovery pool splits with frozen fresh evaluation sets of 256 clean inputs per task. `fixed-holdout` reserves a common split before discovery and is a separate experiment. Greater-Than Edge Pruning trains KL on tokens `00`–`99`; evaluation Clean KL uses the full vocabulary. See [protocol details](docs/evaluation_protocol.md).

## Expected results and artifacts

The committed bundle selects 342 benchmark score/manifest pairs (228 main-task and 114 Docstring) and 105 port-validation pairs (55 DiscoGP and 50 Edge Pruning). [The artifact guide](docs/artifacts.md) explains provenance, hashes, immutable inputs, and the distinction between selected cached reproduction and fresh discovery. Model weights are downloaded at their pinned revisions and are not bundled.

## Development

Read [the architecture guide](docs/architecture.md), [library examples](<docs/guides/1)_Getting_Started.md>), and [contributing instructions](docs/guides/contributing.md). Documentation is ordinary Markdown readable on GitHub. Lightweight local hooks check source hygiene; compute-node tests and bundle integrity checks are separate gates.

## Cite and attribution

AutoCircuit was developed by Joseph Miller, Bilal Chughtai, and William Saunders. Cite their library paper when using its patching and evaluation machinery:

```bibtex
@inproceedings{miller2024transformer,
  title={Transformer Circuit Evaluation Metrics Are Not Robust},
  author={Joseph Miller and Bilal Chughtai and William Saunders},
  booktitle={First Conference on Language Modeling},
  year={2024},
  url={https://openreview.net/forum?id=zSf8PJyQb2}
}
```

Also cite the [Edge Pruning paper](https://arxiv.org/abs/2406.16778) and [DiscoGP paper](https://aclanthology.org/2025.emnlp-main.446/) when using those methods. [CITATION.cff](CITATION.cff) identifies this software fork and its maintainer; it does not change the authorship of the upstream AutoCircuit paper.

This is experimental research software. Results depend on the recorded models, inputs, seeds, optimization settings, and evaluation protocol.
