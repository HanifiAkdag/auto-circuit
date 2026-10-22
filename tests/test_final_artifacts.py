"""Byte-level tests; import no model or training dependencies."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_final_artifacts", ROOT / "scripts/package_final_artifacts.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def test_published_final_inputs():
    result = PACKAGE.verify_bundle(ROOT)
    assert result["benchmark"] == 342
    assert result["main"] == 228
    assert result["docstring"] == 114
    assert result["discogp"] == 55
    assert result["ep"] == 50


def test_model_assets_are_pinned():
    registry = json.loads((ROOT / "artifacts/final/registry.json").read_text())
    assert registry["models"] == json.loads((ROOT / "scripts/model_assets.json").read_text())
    assert set(registry["models"]) == {"gpt2", "attn-only-4l", "docstring-tokenizer"}
    assert len(registry["models"]["gpt2"]["files"]["model.safetensors"]["sha256"]) == 64
    for model in registry["models"].values():
        assert len(model["revision"]) == 40
        assert all(item["bytes"] > 0 for item in model["files"].values())


@pytest.fixture
def tiny_bundle(tmp_path):
    path = tmp_path / "sample.pkl"
    path.write_bytes(b"opaque original bytes")
    registry = {"schema": 1, "purpose": "final_thesis_reproduction",
                "files": {"sample.pkl": PACKAGE.sha256(path)},
                "benchmark": {"inputs": [], "bundles": {}}, "port_fidelity": {}}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    return tmp_path, registry_path, path


def test_missing_file_is_rejected(tiny_bundle):
    root, registry, path = tiny_bundle
    path.unlink()
    with pytest.raises(FileNotFoundError):
        PACKAGE.verify_bundle(root, registry, {"benchmark": 0})


def test_corrupt_file_is_rejected(tiny_bundle):
    root, registry, path = tiny_bundle
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        PACKAGE.verify_bundle(root, registry, {"benchmark": 0})


def test_incorrect_inventory_count_is_rejected(tiny_bundle):
    root, registry, _ = tiny_bundle
    with pytest.raises(ValueError, match="counts"):
        PACKAGE.verify_bundle(root, registry, {"benchmark": 1})
