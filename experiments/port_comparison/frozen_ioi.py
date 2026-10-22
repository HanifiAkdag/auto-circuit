"""Freeze one IOI train/test split for both arms of the port fidelity study.

Both implementations must see the same prompts, the same answer tokens, and the
same train/test membership, or a difference in discovered circuits cannot be
attributed to the implementations. Two confounds are removed here:

* The original's `train_test_split` (`setup_task.py:175`) is called **without a
  seed**, so its split differs run to run.
* The original's IOI answer tokens omit the leading space (`setup_task.py:226-227`),
  which makes ~48% of them word fragments and its objective different from the
  port's even on identical prompts. See `thesis_notes/experiments.md` (2026-08-07).

The artifact is written in this repo's own dataset schema, so the port consumes it
through the ordinary `Task` machinery with no special-casing. `load_for_original`
converts it into the dict the original's dataloader expects.

The artifact fixes train/test membership. AutoCircuit and the DiscoGP reference
preserve its row order, while the Edge Pruning reference retains its trainer's
native batching behaviour. Training order is not matched across implementations.

Run::

    srun --mem=32G python -m experiments.port_comparison.frozen_ioi
"""

import argparse
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "datasets" / "ioi" / "ioi_prompts.json"
FROZEN = REPO / "datasets" / "port_fidelity" / "ioi_frozen.json"

# Matches the spine budget (`tasks.py`): 64 x (11, 4).
N_TRAIN = 704
N_TEST = 256
SPLIT_SEED = 0  # fixed; the per-run seed varies mask init, not the split


@contextmanager
def fixed_split_loader(prepend_bos: bool | None = None) -> Iterator[None]:
    """Preserve the train/test membership encoded by the frozen artifact."""
    import auto_circuit.tasks as ac_tasks

    original_loader = ac_tasks.load_datasets_from_json

    def load_fixed(*args, **kwargs):
        kwargs["shuffle"] = False
        if prepend_bos is not None:
            kwargs["prepend_bos"] = prepend_bos
        return original_loader(*args, **kwargs)

    ac_tasks.load_datasets_from_json = load_fixed
    try:
        yield
    finally:
        ac_tasks.load_datasets_from_json = original_loader


def build(split_seed: int = SPLIT_SEED, n_train: int = N_TRAIN, n_test: int = N_TEST) -> dict:
    """Shuffle and split exactly as `load_datasets_from_json` does, then freeze."""
    data = json.loads(SOURCE.read_text())
    prompts = data["prompts"]
    random.seed(split_seed)
    random.shuffle(prompts)
    chosen = prompts[: n_train + n_test]
    assert len(chosen) == n_train + n_test, (
        f"{SOURCE} has {len(prompts)} prompts, need {n_train + n_test}"
    )
    frozen = dict(data)
    frozen["prompts"] = chosen
    frozen["port_fidelity"] = {
        "split_seed": split_seed,
        "n_train": n_train,
        "n_test": n_test,
        "source": str(SOURCE.relative_to(REPO)),
        "note": "train = prompts[:n_train], test = prompts[n_train:n_train+n_test]",
    }
    return frozen


def load_for_original(tokenizer, split: str = "train", path: Path = FROZEN) -> Dict[str, object]:
    """Convert the frozen artifact into the original's dataloader dict.

    Mirrors `disco_gp/data/setup_task.py:setup_ioi_dataset` in output shape
    (`input_ids`, `seq_lens`, `target good`, `target bad`) but takes the prompts and
    answer token ids from the frozen file, so the answer-token defect and the
    unseeded split are both bypassed.
    """
    import torch as t

    data = json.loads(Path(path).read_text())
    meta = data["port_fidelity"]
    prompts_all = data["prompts"]
    if split == "train":
        rows = prompts_all[: meta["n_train"]]
    elif split == "test":
        rows = prompts_all[meta["n_train"] : meta["n_train"] + meta["n_test"]]
    else:
        raise ValueError(f"unknown split {split!r}")

    texts: List[str] = [r["clean"] for r in rows]
    # The frozen answers already carry the leading space, so `[0]` is a whole name.
    good = [tokenizer(r["answers"][0])["input_ids"][0] for r in rows]
    bad = [tokenizer(r["wrong_answers"][0])["input_ids"][0] for r in rows]

    tokenizer.padding_side = "right"  # the original indexes logits at seq_lens - 1
    tokenized = tokenizer(texts, return_tensors="pt", padding=True)
    return {
        "prompt": texts,
        "input_ids": tokenized["input_ids"],
        "seq_lens": tokenized["attention_mask"].sum(-1),
        "target good": t.tensor(good),
        "target bad": t.tensor(bad),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--out", type=Path, default=FROZEN)
    args = ap.parse_args()

    frozen = build(args.split_seed, args.n_train, args.n_test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, indent=1))

    rows = frozen["prompts"]
    print(f"wrote {args.out.relative_to(REPO)}  ({len(rows)} prompts)")
    print(f"  train {args.n_train} / test {args.n_test}, split_seed={args.split_seed}")
    print(f"  example clean : {rows[0]['clean']!r}")
    print(f"  example answer: {rows[0]['answers']!r} / wrong {rows[0]['wrong_answers']!r}")

    # Every answer must carry the leading space, or the original's fragmenting
    # behaviour would re-enter through the frozen file.
    bad = [r for r in rows if not r["answers"][0].startswith(" ")]
    assert not bad, f"{len(bad)} answers lack a leading space, e.g. {bad[0]['answers']}"
    print(f"  all {len(rows)} answers carry a leading space")


if __name__ == "__main__":
    main()
