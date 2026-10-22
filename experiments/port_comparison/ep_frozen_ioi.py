"""Write the frozen IOI split in the format Edge Pruning's own trainer consumes.

Tier C gives both implementations the *same* prompts. The port reads
``datasets/port_fidelity/ioi_frozen.json`` through the ordinary ``Task`` machinery;
this script re-expresses the identical rows, in the identical order, as a
HuggingFace ``DatasetDict`` so that ``src/prune/fpt2_ioi.py`` can be run exactly as
the authors ship it -- collator, Trainer and all.

The reference's collator (``fpt2_ioi.py:438-477``) takes one field, splits the
answer off at the last space, and tokenises the whole sentence in one go. So the
sentence has to be reassembled as ``clean + answers[0]``, which is why the frozen
split's leading-space convention matters and is asserted here.

Run::

    srun --mem=16G python -m experiments.port_comparison.ep_frozen_ioi
"""

import json

from auto_circuit.utils.misc import repo_path_to_abs_path

FROZEN = repo_path_to_abs_path("datasets/port_fidelity/ioi_frozen.json")
OUT = repo_path_to_abs_path("datasets/port_fidelity/ioi_frozen_ep")


def build() -> "DatasetDict":  # noqa: F821
    from datasets import Dataset, DatasetDict

    data = json.loads(FROZEN.read_text())
    meta = data["port_fidelity"]
    n_train, n_test = meta["n_train"], meta["n_test"]
    rows = data["prompts"][: n_train + n_test]

    for row in rows:
        answer = row["answers"][0]
        assert answer.startswith(" "), f"{answer!r} lost its leading space"
        # The collator finds the answer with `text.rfind(' ')`, so a multi-word
        # answer would silently be truncated to its last word.
        assert " " not in answer.strip(), f"{answer!r} is more than one word"

    def to_split(subset):
        return Dataset.from_dict(
            {
                "ioi_sentences": [r["clean"] + r["answers"][0] for r in subset],
                "corr_ioi_sentences": [r["corrupt"] + r["answers"][0] for r in subset],
            }
        )

    return DatasetDict(
        {
            "train": to_split(rows[:n_train]),
            "validation": to_split(rows[n_train : n_train + n_test]),
        }
    )


def main() -> None:
    dsd = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(OUT))
    print(f"wrote {OUT}")
    for split, ds in dsd.items():
        print(f"  {split}: {len(ds)} rows")
    print(f"  example: {dsd['train'][0]['ioi_sentences']!r}")
    print(f"  corrupt: {dsd['train'][0]['corr_ioi_sentences']!r}")


if __name__ == "__main__":
    main()
