# Ablation, caching, and metrics

The library supports cached prefixes, circuit evaluation, and KL measurement. The sections below describe these features; use the thesis [reproduction guide](../reproduction.md) for the complete registered workflows.

## Ablation types

[AblationType](../../auto_circuit/types.py) includes zero, paired resample, and several average-activation replacements. The thesis's mean ablation is `TOKENWISE_MEAN_CORRUPT`. [src_ablations](../../auto_circuit/utils/ablation_activations.py) computes the corresponding source activations from a batch or loader.

```python
ablations = src_ablations(model, test_loader, AblationType.RESAMPLE)
```

## Cached prefixes

With `tail_divergence=True`, [load_datasets_from_json](../../auto_circuit/data.py) computes key/value caches for the common prompt prefix and removes that prefix from the remaining prompt tokens. Pass the train/test loaders' `kv_cache` values to `patchable_model` through `kv_caches`. This keeps the library's prefix optimization available without changing the conceptual circuit evaluation.

## Evaluate circuits

[run_circuits](../../auto_circuit/prune.py) thresholds `PruneScores` into circuits at the requested edge counts. For example, with an already prepared model, loader, and score dictionary:

```python
from auto_circuit.prune import run_circuits
from auto_circuit.types import AblationType, PatchType

circuit_outputs = run_circuits(
    model=model,
    dataloader=test_loader,
    test_edge_counts=[100, 200],
    prune_scores=scores,
    patch_type=PatchType.EDGE_PATCH,
    ablation_type=AblationType.RESAMPLE,
)
```

The [pruning metric registry](../../auto_circuit/metrics/prune_metrics/prune_metrics.py) defines available metrics. [measure_kl_div](../../auto_circuit/metrics/prune_metrics/kl_div.py) compares circuit outputs with the full model. Scientific analyses also need matched input/protocol identities; the thesis [evaluation guide](../evaluation_protocol.md) describes those checks.
