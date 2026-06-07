#%%
import pytest
import torch as t

from auto_circuit.tasks import Task


def test_task_has_seed_field():
    """Task exposes a seed field defaulting to 42 (backward-compatible)."""
    task = Task(
        key="seed-field-test",
        name="seed-field-test",
        _model_def="gpt2",
        _dataset_name="ioi/ioi_prompts",
        batch_size=1,
        batch_count=1,
        token_circuit=False,
    )
    assert task.seed == 42
    assert (
        Task(
            key="seed-field-test-2",
            name="seed-field-test-2",
            _model_def="gpt2",
            _dataset_name="ioi/ioi_prompts",
            batch_size=1,
            batch_count=1,
            token_circuit=False,
            seed=7,
        ).seed
        == 7
    )


@pytest.mark.slow
def test_task_seed_changes_dataloader_and_cache_key():
    """Two Tasks differing only in seed must yield different dataloaders.

    This guards the DATASET_CACHE-key fix: if seed were not part of the cache key,
    the seed=99 task would silently receive the seed=42 task's cached loader.
    Same seed must reproduce identical data (determinism + safe cache reuse).
    """
    def make(seed: int) -> Task:
        return Task(
            key=f"seed-dl-test-{seed}",
            name=f"seed-dl-test-{seed}",
            _model_def="gpt2",
            _dataset_name="ioi/ioi_prompts",
            batch_size=8,
            batch_count=1,
            token_circuit=False,
            seed=seed,
        )

    b42a = next(iter(make(42).train_loader))
    b42b = next(iter(make(42).train_loader))
    b99 = next(iter(make(99).train_loader))

    # Same seed -> identical data (determinism; cache reuse is correct here).
    assert t.equal(b42a.clean, b42b.clean)
    # Different seed -> different data (cache key must include seed, no collision).
    assert not t.equal(b42a.clean, b99.clean)
