# Circuit discovery

Once a model and loader are prepared, a discovery algorithm returns edge scores. For example, the mask-gradient algorithm can be called as follows:

```python
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores

scores = mask_gradient_prune_scores(
    model=model,
    dataloader=train_loader,
    official_edges=None,
    grad_function="logit",
    answer_function="avg_diff",
    mask_val=0.0,
)
```

The [algorithm registry](../../auto_circuit/prune_algos/prune_algos.py) retains upstream algorithms and the EP/DiscoGP variants. Implementations include [ACDC](../../auto_circuit/prune_algos/ACDC.py), [mask-gradient EAP](../../auto_circuit/prune_algos/mask_gradient.py), [Subnetwork Probing](../../auto_circuit/prune_algos/subnetwork_probing.py), [Edge Pruning](../../auto_circuit/prune_algos/edge_pruning.py), and [DiscoGP](../../auto_circuit/prune_algos/disco_gp.py).

Each method returns one score per edge in a `PruneScores` dictionary. `run_circuits` derives circuits by thresholding score magnitudes at requested budgets; positive and negative attribution signs are not separate circuit-selection rules. In attribution visualizations, blue/red indicate score sign and line thickness indicates magnitude.

![IOI attribution graph](../assets/IOI_Attributions_Viz.png)

## Token-specific graphs

Token-specific graphs request `return_seq_length=True` from the dataset loader and supply that sequence length to `patchable_model`. All prompts must have the same sequence length. The resulting graph distinguishes edges at different token positions.

![Token-specific IOI graph](../assets/IOI_Tokenwise_Viz.png)

Use [the registry-based pipeline](../reproduction.md) for the thesis data budgets, objectives, ablations, and evaluation protocol.
