# Getting started with patching

Install this checkout through [the compute-node setup](../reproduction.md). The PyPI package `auto-circuit` is the upstream distribution; it does not install this thesis fork's experiment pipeline.

This guide shows the core patching operation for a prepared TransformerLens model and paired prompt loader. For a complete thesis workflow, use the [reproduction commands](../reproduction.md).

```python
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model

# `model` is a loaded TransformerLens model and `test_loader` contains prompts.
model = patchable_model(
    model, factorized=True, slice_output="last_seq", separate_qkv=True
)
ablations = src_ablations(model, test_loader, AblationType.TOKENWISE_MEAN_CORRUPT)
patch_edges = ["Resid Start->MLP 1", "MLP 1->MLP 2"]
with patch_mode(model, ablations, patch_edges):
    for batch in test_loader:
        patched_out = model(batch.clean)
```

Datasets contain paired clean/corrupt prompts and correct/incorrect answers. See the loader in [data.py](../../auto_circuit/data.py) and [the IOI prompt pool](../../datasets/ioi/ioi_vanilla_template_prompts.json). Graph construction and wrapper types are documented in [graph_utils.py](../../auto_circuit/utils/graph_utils.py), [patchable_model.py](../../auto_circuit/utils/patchable_model.py), and [patch_wrapper.py](../../auto_circuit/utils/patch_wrapper.py).

For circuit discovery, continue to [the discovery guide](<2)_Circuit_Discovery.md>).

![Example patched circuit](../assets/Small_Circuit_Viz.png)
