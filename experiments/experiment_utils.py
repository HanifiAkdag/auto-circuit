import enum
import functools
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Base-function args that are runtime inputs, not hyperparameters.
_NON_HPARAM_ARGS = {"model", "dataloader", "official_edges"}


def git_state() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, cwd=_REPO_ROOT, text=True).strip()

    commit = run(["git", "rev-parse", "HEAD"])
    # Porcelain lines are "XY <path>"; don't strip, the leading space is significant.
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=_REPO_ROOT, text=True
    )
    dirty_files = [l[3:] for l in status.splitlines() if l.strip()]
    return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files}


def algo_hyperparams(algo: Any) -> dict[str, Any]:
    """Extract the effective hyperparameters of a ``PruneAlgo``.

    Merges the base function's default values with any keyword arguments bound
    via ``functools.partial`` (the partial's kwargs win). Runtime inputs
    (model/dataloader/official_edges) are excluded.
    """
    func = algo.func
    bound: dict[str, Any] = {}
    base = func
    if isinstance(func, functools.partial):
        base = func.func
        bound.update(func.keywords or {})
    params: dict[str, Any] = {}
    try:
        for name, p in inspect.signature(base).parameters.items():
            if p.default is not inspect.Parameter.empty:
                params[name] = p.default
    except (ValueError, TypeError):
        pass
    params.update(bound)
    return {k: v for k, v in params.items() if k not in _NON_HPARAM_ARGS}


def _serialize(obj: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return str(obj).split(".")[-1]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)
