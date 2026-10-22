# Contributing

Use the [pinned thesis setup](../reproduction.md) and keep installations, imports, and tests on compute nodes. `scripts/thesis.sh test` submits the test workflow; use `--dry-run` to inspect it before launch. Development should preserve upstream algorithms, model/dataset capabilities, and public interfaces outside the explicitly scoped thesis additions.

## Local checks

Pre-commit performs lightweight static hygiene checks only. It does not run pytest, load models, or submit GPU work. It excludes immutable scientific artifact and frozen dataset JSON from formatting changes. Integrity checks remain a separate CI and compute-node gate.

```bash
pre-commit run --all-files
```

Install pre-commit in the compute-created environment. Hooks are optional developer tooling, not evidence that a scientific pipeline reproduces its outputs. Follow a passing hook run with relevant compute-node tests when changing behavior.

The default suite retains local synthetic sparse-autoencoder tests. Four optional pretrained-SAE integration tests are marked `external_resource`; enable them with `bash scripts/thesis.sh test -- --runexternal`. Their upstream checkpoints at `https://baulab.us/u/smarks/autoencoders/pythia-70m-deduped/mlp_out_layer{0..5}/0_8192/ae.pt` returned HTTP 404 in the baseline run. Explicitly requesting those tests reports unavailable checkpoint URLs as failures rather than treating them as passes. `--runslow` independently enables slow tests.

## Library and experiments

New pruning methods should implement the existing `PruneAlgo` interface and return `PruneScores`; circuit selection belongs in the shared evaluation code. Keep model mutation inside the existing context managers and restore temporary state. The patching layer necessarily uses module state, while experiment analysis should keep its data flow explicit.

Source documentation is ordinary Markdown and code docstrings. Link examples to existing source files, rather than using MkDocs-only snippet directives. See [architecture](../architecture.md) for ownership and [evaluation protocols](../evaluation_protocol.md) before changing loaders, cache identity, or baselines.

## Artifact changes

Treat `artifacts/final/` and frozen datasets as read-only scientific inputs. Generate new scores and reports into a separate run directory and registry. Do not update hashes merely to hide an unexpected change. Retain the original reference attribution and cite method papers for the exact claims they support.
