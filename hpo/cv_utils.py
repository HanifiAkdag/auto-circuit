"""Cross-validation utilities for splitting PromptDataLoader into K folds.

Builds fold loaders by subsetting the underlying PromptDataset and wrapping
each subset in a new PromptDataLoader. Metadata (seq_len, diverge_idx,
seq_labels, word_idxs) is copied from the reference loader.

For tasks with tail_divergence=True (diverge_idx > 0), the KV cache is resized
to the new batch_size by slicing or repeating the batch dimension. This is
valid because all prompts share the same prefix (that's the definition of
diverge_idx), so every batch element has identical cached key-value pairs.
"""
from typing import List, Optional, Tuple

import torch as t
from torch.utils.data import Subset
from transformer_lens.past_key_value_caching import (
    HookedTransformerKeyValueCache,
    HookedTransformerKeyValueCacheEntry,
)

from auto_circuit.data import PromptDataLoader, PromptDataset
from auto_circuit.utils.patchable_model import PatchableModel


def _resize_kv_cache(
    ref_cache: HookedTransformerKeyValueCache,
    new_batch_size: int,
) -> HookedTransformerKeyValueCache:
    """Return a frozen KV cache with a different batch dimension.

    All batch elements in the original cache are identical (same prefix), so
    slicing is correct when shrinking, and repeating the first element is
    correct when expanding.
    """
    def _resize_tensor(x: t.Tensor, n: int) -> t.Tensor:
        if n <= x.shape[0]:
            return x[:n].clone()
        repeats = (n + x.shape[0] - 1) // x.shape[0]
        return x.repeat(repeats, *([1] * (x.ndim - 1)))[:n].clone()

    new_entries = []
    for entry in ref_cache.entries:
        new_entries.append(
            HookedTransformerKeyValueCacheEntry(
                past_keys=_resize_tensor(entry.past_keys, new_batch_size),
                past_values=_resize_tensor(entry.past_values, new_batch_size),
                frozen=True,
            )
        )

    new_cache = HookedTransformerKeyValueCache(
        entries=new_entries,
        previous_attention_mask=_resize_tensor(
            ref_cache.previous_attention_mask, new_batch_size
        ),
        frozen=True,
    )
    return new_cache


def _make_loader(
    subset: Subset,
    ref: PromptDataLoader,
    batch_size: int,
) -> PromptDataLoader:
    kv_cache = None
    if ref.diverge_idx > 0 and ref.kv_cache is not None:
        kv_cache = _resize_kv_cache(ref.kv_cache, batch_size)

    return PromptDataLoader(
        subset,
        seq_len=ref.seq_len,
        diverge_idx=ref.diverge_idx,
        kv_cache=kv_cache,
        seq_labels=ref.seq_labels,
        word_idxs=ref.word_idxs,
        batch_size=batch_size,
        shuffle=False,
    )


def _get_underlying(loader: PromptDataLoader) -> Tuple[PromptDataset, List[int]]:
    """Return (PromptDataset, index_list) from a loader whose dataset is a Subset."""
    subset = loader.dataset
    if isinstance(subset, Subset):
        return subset.dataset, list(subset.indices)
    # Fallback: dataset is already a PromptDataset
    return subset, list(range(len(subset)))


def kfold_prompt_loaders(
    ref_loader: PromptDataLoader,
    k: int,
    seed: int = 42,
    batch_size: Optional[int] = None,
) -> List[Tuple[PromptDataLoader, PromptDataLoader]]:
    """Split ref_loader into K stratified folds.

    Returns a list of (train_loader, val_loader) tuples, one per fold.
    Each fold's val set is 1/K of the data; train is the remaining (K-1)/K.
    Indices are shuffled once with the given seed.

    Args:
        ref_loader: The loader to split (typically task.train_loader).
        k: Number of folds.
        seed: Random seed for shuffling.
        batch_size: Batch size for the fold loaders. Defaults to half the
            val-fold size (so at least 2 batches), capped at the original
            batch size.
    """
    underlying, indices = _get_underlying(ref_loader)

    rng = t.Generator()
    rng.manual_seed(seed)
    perm = t.randperm(len(indices), generator=rng).tolist()
    shuffled = [indices[i] for i in perm]

    fold_size = len(shuffled) // k
    folds: List[List[int]] = [shuffled[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    for j, idx in enumerate(shuffled[k * fold_size:]):
        folds[j].append(idx)

    orig_bs = ref_loader.batch_size or 32

    result = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = [idx for j, fold in enumerate(folds) if j != i for idx in fold]

        val_bs = batch_size or min(orig_bs, max(1, len(val_idx) // 2))
        train_bs = batch_size or min(orig_bs, max(1, len(train_idx) // 2))

        result.append((
            _make_loader(Subset(underlying, train_idx), ref_loader, train_bs),
            _make_loader(Subset(underlying, val_idx), ref_loader, val_bs),
        ))
    return result


def holdout_split(
    ref_loader: PromptDataLoader,
    val_frac: float = 0.25,
    seed: int = 42,
    batch_size: Optional[int] = None,
) -> Tuple[PromptDataLoader, PromptDataLoader]:
    """Split ref_loader into train and validation sets.

    Args:
        ref_loader: The loader to split (typically an outer-fold train loader).
        val_frac: Fraction of data to use for validation.
        seed: Random seed for shuffling.
        batch_size: Batch size for the resulting loaders. Defaults to half the
            smaller split size, capped at the original batch size.
    """
    underlying, indices = _get_underlying(ref_loader)

    rng = t.Generator()
    rng.manual_seed(seed)
    perm = t.randperm(len(indices), generator=rng).tolist()
    shuffled = [indices[i] for i in perm]

    n_val = max(1, int(len(shuffled) * val_frac))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]

    orig_bs = ref_loader.batch_size or 32
    val_bs = batch_size or min(orig_bs, max(1, len(val_idx) // 2))
    train_bs = batch_size or min(orig_bs, max(1, len(train_idx) // 2))

    return (
        _make_loader(Subset(underlying, train_idx), ref_loader, train_bs),
        _make_loader(Subset(underlying, val_idx), ref_loader, val_bs),
    )


def register_fold_kv_caches(model: PatchableModel, *loaders: PromptDataLoader) -> None:
    """Register each loader's KV cache into model.kv_caches so the model's forward
    pass can look it up by batch_size.

    Call this once after building fold loaders when the task uses tail_divergence
    (diverge_idx > 0). Safe to call when model.kv_caches is None (no-op).
    """
    if model.kv_caches is None:
        return
    for loader in loaders:
        if loader.kv_cache is not None:
            bs = loader.kv_cache.previous_attention_mask.shape[0]
            if bs not in model.kv_caches:
                model.kv_caches[bs] = loader.kv_cache
