# Scientific artifacts

`artifacts/final/registry.json` is the default input registry. It selects the final scientific files retained in ordinary Git, with hashes and provenance. The frozen bundle is logically read-only: discovery, evaluation, and export write into a run directory such as `results/reproduction/`.

## Bundle contents

| Group | Selected score/manifest pairs | Scope |
| --- | ---: | --- |
| Benchmark | 342 | 228 IOI/Greater-Than main-task pairs and 114 Docstring appendix pairs |
| DiscoGP port validation | 55 | Frozen IOI reference/port study |
| Edge Pruning port validation | 50 | Frozen IOI reference/port study |

The bundle also carries the frozen inputs and selected provenance/evidence required by its registry. It preserves the active Greater-Than Edge Pruning discoveries trained with KL restricted to tokens `00`–`99`. Their evaluation Clean KL remains full-vocabulary. The selected bundle excludes superseded score histories, intermediate figures, and the separate thesis manuscript.

The selected cache count describes discovery identities rather than only the five-seed main comparison: it also includes variance-decomposition conditions and deterministic manually identified circuit baselines. The main experiment remains separate from Docstring and from the port-validation protocols.

## Integrity and immutable inputs

```bash
bash scripts/thesis.sh verify --registry artifacts/final/registry.json
```

The verifier checks membership, file hashes, and required provenance before scientific work. CI also checks integrity. JSON formatting hooks do not rewrite artifact or frozen dataset JSON, and large-file checks explicitly allow the selected scientific bundle. The exception is confined to the scientific artifact paths; it is not a blanket allowance for generated files.

Pickles are the retained native AutoCircuit score format. Load only the trusted repository's hash-verified files. Each selected `.pkl` has a matching manifest. A frozen registry must not be rewritten to make a modified artifact pass verification.

The original ignored `.prune_scores*`, `.port_fidelity*`, `results/`, and local document directories remain local historical material. Their selected copies have distinct paths in `artifacts/final/`; reproduction must resolve those copies through the registry, not depend on a maintainer's ignored tree.

## Models and reference implementations

Model weights are not included. Setup/preflight downloads pinned model and tokenizer revisions and records their identities. Docstring requires the model's GPT-NeoX digits tokenizer rather than GPT-2's tokenizer.

The EP and DiscoGP reference repositories are fetched at pinned revisions into ignored resource locations. No third-party reference checkout or credentials are committed. Their licenses and citations remain those of their original authors.

## Generated runs

`discover` creates new score/manifest files and a registry for that run. `evaluate` consumes a registry and records its evaluation protocol. `export` consumes completed matching analyses and manual baselines. A fresh run must use its own output registry and must not be patched into the selected bundle as a repair step.

The `thesis` protocol uses original discovery splits and frozen fresh evaluation prompts. `fixed-holdout` is a separate protocol with a common pre-discovery split. Never combine outputs across these identities. See [reproduction](reproduction.md) for commands.
