# Patching internals

`PruneScores` is a dictionary mapping destination module names to tensors. Each tensor has one entry per incoming edge. The [Edge type](../../auto_circuit/types.py) supplies the destination module and the tensor index:

```python
score = prune_scores[edge.dest.module_name][edge.patch_idx]
```

[patchable_model](../../auto_circuit/utils/graph_utils.py) builds the source/destination graph and installs [PatchWrapperImpl](../../auto_circuit/utils/patch_wrapper.py) modules. [PatchableModel](../../auto_circuit/utils/patchable_model.py) exposes the graph, wrappers, and mask parameters.

Each destination wrapper's `patch_mask` has the same shape as its incoming-edge score tensor. During `patch_mode`, a mask value of zero preserves the current edge activation and one replaces it with the supplied ablated activation. Intermediate values interpolate. Scores used to retain edges therefore should not be mistaken for masks used to ablate them.

The replacement source activations are collected first, then the masked forward pass substitutes them at destination inputs. This permits efficient edge-level interventions without running a separate full forward pass for every edge. See [ablation_activations.py](../../auto_circuit/utils/ablation_activations.py) and [graph_utils.py](../../auto_circuit/utils/graph_utils.py) for the contexts and helpers.

![Edge patching](../assets/Edge_Patching.png)

![Factorized transformer](../assets/Factorized_Transformer.png)

For runnable examples, see [getting started](<1)_Getting_Started.md>) and [circuit discovery](<2)_Circuit_Discovery.md>). The [upstream announcement](https://www.lesswrong.com/posts/caZ3yR5GnzbZe2yJ3/how-to-do-patching-fast) provides additional background on the original library design.
