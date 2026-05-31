import enum
import json
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def git_state() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, cwd=_REPO_ROOT, text=True).strip()

    commit = run(["git", "rev-parse", "HEAD"])
    status_lines = run(["git", "status", "--porcelain"]).splitlines()
    dirty_files = [l[3:] for l in status_lines if l.strip()]
    return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files}


def _serialize(obj: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return str(obj).split(".")[-1]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)
