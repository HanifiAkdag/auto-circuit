# IOI retained-head diagnostics

Run the retained mean-ablation study through the repository launcher:

```bash
bash scripts/thesis.sh head-diagnostics --run-dir results/head-diagnostics
```

This action uses the frozen protocol, selected discovery caches, model revisions, scientific inventory, and masks declared by `artifacts/final/registry.json`. It writes a new result tree under the requested run directory and checks the regenerated inventory and reported tables against the published reference. Run Python only through SLURM compute jobs; the launcher handles submission.

The study uses four methods and five mean-discovery runs. It measures baseline and causal probes at 739, 2000, and 3000 requested edges using tokenwise-mean-corrupt evaluation. Connectivity is component-level. Attention is measured at the final query position on the separately identified IO, S1, and S2 tokens. Prior resample outputs are audit-only and are not part of the active study.

The output contains `inventory.json` and `masks.npz`, per-condition records and per-prompt arrays, summary CSVs, figures, a report, and a completion marker. `complete.json` is written only after each predefined condition and alias has passed input and output checks. The per-prompt arrays retain the KL partition fields needed to reconstruct full-vocabulary KL. Exact-size subsets are randomized sensitivity checks, not native outputs; requested and attained circuit sizes remain separate.

See [the reproduction guide](../../../docs/reproduction.md) for setup, registry, protocol, and SLURM details.
