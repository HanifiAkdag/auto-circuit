"""Build an isolated, verified offline mirror of the thesis model snapshots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import load_registry, sha256


def prepare(registry_path, destination):
    from huggingface_hub import hf_hub_download
    registry = load_registry(registry_path)
    destination = Path(destination)
    for item in registry["models"].values():
        repository = destination / ("models--" + item["repo_id"].replace("/", "--"))
        snapshot = repository / "snapshots" / item["revision"]
        for filename, metadata in item["files"].items():
            target = snapshot / filename
            if not target.exists():
                source = Path(hf_hub_download(item["repo_id"], filename, revision=item["revision"]))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
            if sha256(target) != metadata["sha256"]:
                raise ValueError(f"Pinned model asset hash mismatch: {target}")
        reference = repository / "refs/main"
        reference.parent.mkdir(parents=True, exist_ok=True)
        if reference.exists() and reference.read_text() != item["revision"]:
            raise ValueError(f"Model mirror revision changed: {reference}")
        reference.write_text(item["revision"])
    (destination / "version.txt").write_text("1")


def install_model_pins(registry, run_dir):
    destination = Path(run_dir) / "models/hub"
    subprocess.run([sys.executable, "-m", __name__, "--registry", registry["_path"],
                    "--destination", str(destination)], check=True)
    # Set these before importing Transformers/TransformerLens. Reference code that
    # requests 'main' therefore sees exactly the verified, pinned snapshots too.
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        os.environ[variable] = str(destination)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["THESIS_MODEL_REGISTRY"] = registry["_path"]
    os.environ["THESIS_MODEL_MIRROR"] = str(destination)
    return destination


_DOCSTRING_MODELS = {}


def pinned_docstring_model(device):
    """Load the declared checkpoint directly, avoiding TL's remote file listing.

    Configuration conversion and weight processing are the same TransformerLens
    operations as from_pretrained. Only checkpoint selection uses the manifest.
    """
    import torch
    from transformer_lens import HookedTransformer
    from transformer_lens import loading_from_pretrained as loading
    from transformers import AutoTokenizer
    device = str(device)
    if device in _DOCSTRING_MODELS:
        return _DOCSTRING_MODELS[device]
    registry = load_registry(os.environ["THESIS_MODEL_REGISTRY"])
    mirror = Path(os.environ["THESIS_MODEL_MIRROR"])
    item = registry["models"]["attn-only-4l"]
    tokenizer_item = registry["models"]["docstring-tokenizer"]
    snapshot = mirror / ("models--" + item["repo_id"].replace("/", "--")) / "snapshots" / item["revision"]
    tokenizer_path = mirror / ("models--" + tokenizer_item["repo_id"].replace("/", "--")) / "snapshots" / tokenizer_item["revision"]
    cfg = loading.get_pretrained_model_config(item["repo_id"], fold_ln=True,
                                              device=device, dtype=torch.float32,
                                              revision=item["revision"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    state = torch.load(snapshot / "model_final.pth", map_location="cpu", weights_only=True)
    state = {key: value.to(torch.float32) for key, value in state.items()}
    model = HookedTransformer(cfg, tokenizer, move_to_device=False)
    model.load_and_process_state_dict(state, fold_ln=True, center_writing_weights=True,
                                      center_unembed=True, fold_value_biases=True,
                                      refactor_factored_attn_matrices=False)
    model.move_model_modules_to_device()
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    _DOCSTRING_MODELS[device] = model
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.registry, args.destination)
