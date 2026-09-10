import json

import pytest
import torch as t

import auto_circuit.tasks as ac_tasks
from auto_circuit.prune_algos import disco_gp, edge_pruning
from experiments.experiment_utils import hardware_state
from experiments.port_comparison import ep_run_cells, fidelity_analysis, run_cells


RUNNERS = (
    (run_cells, disco_gp, "disco_gp_prune_scores"),
    (ep_run_cells, edge_pruning, "edge_pruning_prune_scores"),
)


def test_hardware_state_records_slurm_node_and_gpu(monkeypatch):
    monkeypatch.setenv("SLURMD_NODENAME", "test-node")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_name", lambda _: "Test GPU")

    assert hardware_state() == {
        "node": "test-node",
        "slurm_job_id": "123",
        "gpu": "Test GPU",
    }


@pytest.mark.parametrize(("runner", "algorithm", "prune_name"), RUNNERS)
def test_port_runner_preserves_frozen_membership(
    monkeypatch, tmp_path, runner, algorithm, prune_name
):
    rows = [
        {
            "clean": [idx, idx + 10],
            "corrupt": [idx + 20, idx + 30],
            "answers": [idx + 40],
            "wrong_answers": [idx + 50],
        }
        for idx in range(6)
    ]
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"prompts": rows}))
    observed = {}

    class FakeTask:
        current = None

        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]
            self.model = object()
            FakeTask.current = self

        def init_task(self):
            self.train_loader, self.test_loader = ac_tasks.load_datasets_from_json(
                model=None,
                path=path,
                device=t.device("cpu"),
                prepend_bos=False,
                batch_size=2,
                train_test_size=(4, 2),
                random_seed=self.seed,
            )

    def fake_prune_scores(model, train_loader, official_edges, **kwargs):
        observed["train"] = [
            batch.clean[:, 0].tolist() for batch in train_loader
        ]
        observed["test"] = [
            batch.clean[:, 0].tolist() for batch in FakeTask.current.test_loader
        ]
        return {"blocks.0.hook_resid_post": t.ones(1)}

    monkeypatch.setattr(runner, "Task", FakeTask)
    monkeypatch.setattr(algorithm, prune_name, fake_prune_scores)

    runner.run_port("port", 7, t.device("cpu"))

    assert observed["train"] == [[0, 1], [2, 3]]
    assert observed["test"] == [[4, 5]]


@pytest.mark.parametrize(("runner", "algorithm", "prune_name"), RUNNERS)
def test_port_nobos_overrides_task_preprocessing(
    monkeypatch, runner, algorithm, prune_name
):
    captured = {}

    def spy_loader(*args, **kwargs):
        captured.update(kwargs)
        return object(), object()

    class FakeTask:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]
            self.model = object()

        def init_task(self):
            self.train_loader, self.test_loader = ac_tasks.load_datasets_from_json(
                model=object(),
                path=object(),
                device=t.device("cpu"),
                prepend_bos=True,
                random_seed=self.seed,
            )

    monkeypatch.setattr(ac_tasks, "load_datasets_from_json", spy_loader)
    monkeypatch.setattr(runner, "Task", FakeTask)
    monkeypatch.setattr(
        algorithm,
        prune_name,
        lambda *args, **kwargs: {"blocks.0.hook_resid_post": t.ones(1)},
    )

    runner.run_port("port-nobos", 3, t.device("cpu"))

    assert captured["shuffle"] is False
    assert captured["prepend_bos"] is False


def test_analysis_uses_the_frozen_test_membership(monkeypatch, tmp_path):
    rows = [
        {
            "clean": [idx, idx + 10],
            "corrupt": [idx + 20, idx + 30],
            "answers": [idx + 40],
            "wrong_answers": [idx + 50],
        }
        for idx in range(6)
    ]
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({"prompts": rows}))

    class FakeTask:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]

        def init_task(self):
            self.train_loader, self.test_loader = ac_tasks.load_datasets_from_json(
                model=None,
                path=path,
                device=t.device("cpu"),
                prepend_bos=False,
                batch_size=2,
                train_test_size=(4, 2),
                random_seed=self.seed,
            )

    monkeypatch.setattr(fidelity_analysis, "Task", FakeTask)
    task = fidelity_analysis.build_task(seed=7, batch_size=2, batch_count=(2, 1))

    train_ids = [batch.clean[:, 0].tolist() for batch in task.train_loader]
    test_ids = [batch.clean[:, 0].tolist() for batch in task.test_loader]
    assert train_ids == [[0, 1], [2, 3]]
    assert test_ids == [[4, 5]]
