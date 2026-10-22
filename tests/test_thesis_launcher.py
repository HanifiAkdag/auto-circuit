"""Exercise SLURM submission and argument boundaries without a scheduler."""

import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "thesis.sh"


def launch(*args, env=None):
    return subprocess.run(
        ["bash", str(LAUNCHER), *map(str, args)],
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
        check=False,
    )


def scheduler(tmp_path):
    executable = tmp_path / "sbatch"
    record = tmp_path / "submissions.jsonl"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "p = pathlib.Path(os.environ['SBATCH_RECORD'])\n"
        "n = len(p.read_text().splitlines()) if p.exists() else 0\n"
        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(str(100 + n) + ';cluster')\n"
    )
    executable.chmod(0o755)
    return record, {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_RECORD": str(record),
    }


def test_dry_run_does_not_create_outputs_or_submit(tmp_path):
    run_dir = tmp_path / "new output with spaces"
    record, env = scheduler(tmp_path)
    result = launch("evaluate", "--run-dir", run_dir, "--dry-run", env=env)
    assert result.returncode == 0, result.stderr
    assert not run_dir.exists()
    assert not record.exists()
    assert "--constraint=gpu_mem:80gb" in result.stderr
    assert "experiments.pipeline" not in result.stderr  # Job template owns execution.


@pytest.mark.parametrize("action", ["setup", "verify", "export", "test"])
def test_cpu_actions_and_literal_arguments(tmp_path, action):
    record, env = scheduler(tmp_path)
    run_dir = tmp_path / "output ; literal $(do-not-execute)"
    registry = tmp_path / "registry with spaces.json"
    result = launch(action, "--run-dir", run_dir, "--registry", registry,
                    "--protocol", "fixed-holdout", env=env)
    assert result.returncode == 0, result.stderr
    args = json.loads(record.read_text())
    assert not any(arg.startswith("--gres") for arg in args)
    assert "--partition=cpu" in args
    assert "--qos=cpu" in args
    assert args[-3:] == [str(run_dir), str(registry), "fixed-holdout"]
    assert (run_dir / "logs").is_dir()
    assert f"{action}=100" in result.stdout
    assert "--open-mode=append" in args


def test_discovery_dependencies_and_default_concurrency(tmp_path):
    record, env = scheduler(tmp_path)
    result = launch("discover", "--run-dir", tmp_path / "run", env=env)
    assert result.returncode == 0, result.stderr
    regular, partial, finalize = map(json.loads, record.read_text().splitlines())
    assert "--dependency=afterok:100" in partial
    assert "--dependency=afterok:100:101" in finalize
    assert next(x for x in regular if x.startswith("--array=")).endswith("%2")
    assert "--array=0-149%2" in partial
    assert not any(x.startswith("--gres") for x in finalize)
    assert "--partition=gpu" in regular
    assert not any(x.startswith("--qos=") for x in regular)
    assert not any(x.startswith("--qos=") for x in partial)
    assert "--qos=cpu" in finalize
    assert "regular=100 partial=101 finalize=102" in result.stdout


def test_gpu_qos_override_is_preserved(tmp_path):
    record, env = scheduler(tmp_path)
    env.update(THESIS_PARTITION="yolo", THESIS_QOS="yolo")
    result = launch("smoke", "--run-dir", tmp_path / "run", env=env)
    assert result.returncode == 0, result.stderr
    args = json.loads(record.read_text())
    assert "--partition=yolo" in args
    assert "--qos=yolo" in args


@pytest.mark.parametrize("args", [
    ["verify", "--protocol", "legacy"],
    ["verify", "--run-dir"],
    ["verify", "--unexpected"],
    ["unknown"],
])
def test_invalid_arguments_do_not_submit(tmp_path, args):
    record, env = scheduler(tmp_path)
    result = launch(*args, env=env)
    assert result.returncode == 2
    assert not record.exists()


def test_scheduler_failure_stops_discovery_dependencies(tmp_path):
    executable = tmp_path / "sbatch"
    executable.write_text("#!/usr/bin/env bash\necho 'scheduler unavailable' >&2\nexit 23\n")
    executable.chmod(0o755)
    result = launch("discover", "--run-dir", tmp_path / "run",
                    env={"PATH": f"{tmp_path}:{os.environ['PATH']}"})
    assert result.returncode == 2
    assert "sbatch submission failed" in result.stderr
    assert "finalize=" not in result.stdout


def test_job_template_forwards_array_index_and_pipeline_options(tmp_path):
    interpreter = tmp_path / "capture-python"
    capture = tmp_path / "python-args"
    interpreter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    interpreter.chmod(0o755)
    env = {**os.environ, "THESIS_PYTHON": str(interpreter), "SLURM_JOB_ID": "10",
           "SLURM_ARRAY_TASK_ID": "42", "CAPTURE_ARGS": str(capture)}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/slurm/thesis_job.sh"), str(ROOT), "discover",
         "regular", str(tmp_path), "registry.json", "thesis", "--force"],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [
        "-m", "experiments.pipeline", "discover", "--run-dir", str(tmp_path),
        "--registry", "registry.json", "--protocol", "thesis", "--cell-index", "42", "--force",
    ]


def test_job_template_refuses_login_execution():
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/thesis_job.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "scripts/thesis.sh" in result.stderr


def test_setup_bootstraps_before_running_asset_verification(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bootstrap_marker = tmp_path / "bootstrapped"
    (scripts / "setup_environment.sh").write_text(
        '#!/usr/bin/env bash\ntouch "$BOOTSTRAP_MARKER"\n'
    )
    interpreter = tmp_path / "capture-python"
    capture = tmp_path / "python-args"
    interpreter.write_text(
        '#!/usr/bin/env bash\n[[ -f $BOOTSTRAP_MARKER ]] || exit 99\n'
        'printf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
    )
    interpreter.chmod(0o755)
    env = {**os.environ, "THESIS_PYTHON": str(interpreter), "SLURM_JOB_ID": "10",
           "BOOTSTRAP_MARKER": str(bootstrap_marker), "CAPTURE_ARGS": str(capture)}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/slurm/thesis_job.sh"), str(tmp_path), "setup",
         "single", str(tmp_path / "run"), "registry.json", "thesis"],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines()[:3] == ["-m", "experiments.pipeline", "verify"]
