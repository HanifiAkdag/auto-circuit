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


@pytest.mark.slow
def test_tasks_sharing_model_def_get_independent_models():
    """Two tasks with the same _model_def must each get an independent model.

    patchable_model mutates the model in place; if both tasks wrapped the same
    MODEL_CACHE instance, the second would double-wrap the first's, silently
    corrupting both. Guards the tasks.py deepcopy fix.
    """
    from auto_circuit.utils.graph_utils import module_by_name
    from auto_circuit.utils.patch_wrapper import PatchWrapperImpl

    def make(key: str) -> Task:
        return Task(
            key=key,
            name=key,
            _model_def="gpt2",
            _dataset_name="ioi/ioi_prompts",
            batch_size=4,
            batch_count=1,
            token_circuit=False,
        )

    a = make("share-test-a").model
    b = make("share-test-b").model

    # Independent underlying models (not the same mutated object).
    assert a.wrapped_model is not b.wrapped_model
    # Neither is double-wrapped (PatchWrapperImpl wrapping a PatchWrapperImpl).
    for m in (a, b):
        dw = next(iter(m.dest_wrappers))
        inner = module_by_name(m.wrapped_model, dw.module_name)
        assert isinstance(inner, PatchWrapperImpl)
        assert not isinstance(inner.module, PatchWrapperImpl)
