from copy import deepcopy

import torch as t
from transformer_lens import HookedTransformer

from auto_circuit.tasks import DATASET_CACHE, Task


def _task(model: HookedTransformer) -> Task:
    return Task(
        key="dataset-cache-identity",
        name="dataset-cache-identity",
        batch_size=1,
        batch_count=1,
        token_circuit=False,
        _model_def=model,
        _dataset_name="mini_prompts",
    )


def test_dataset_cache_separates_models(mini_tl_transformer: HookedTransformer):
    """A prefix cache contains model activations, not only prompt token IDs."""
    DATASET_CACHE.clear()
    first_model = deepcopy(mini_tl_transformer)
    second_model = deepcopy(mini_tl_transformer)
    with t.no_grad():
        second_model.blocks[0].attn.b_V.add_(1.0)
    first = _task(first_model)
    second = _task(second_model)

    assert first.train_loader is not second.train_loader
    assert first.train_loader.kv_cache is not second.train_loader.kv_cache
    assert not t.allclose(
        first.train_loader.kv_cache[0].past_values,
        second.train_loader.kv_cache[0].past_values,
    )
    repeated = _task(first_model)
    assert repeated.train_loader is first.train_loader


def test_dataset_cache_separates_tokenizers(mini_tl_transformer: HookedTransformer):
    DATASET_CACHE.clear()
    model = deepcopy(mini_tl_transformer)
    first = _task(model)
    first_loader = first.train_loader
    model.tokenizer = deepcopy(model.tokenizer)
    second = _task(model)

    assert second.train_loader is not first_loader
