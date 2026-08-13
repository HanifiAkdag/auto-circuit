"""Inspect what tail_divergence actually caches, using the real loader.

Everything reported here comes from `load_datasets_from_json` itself - the
diverge_idx, the KV cache and the prompts are read off the returned loaders.
Nothing re-implements the rule.

Run it once per checkout and compare:

    W=/mnt/beegfs/work/akdag/thesis/auto-circuit/.worktrees
    S=/mnt/beegfs/work/akdag/thesis/auto-circuit/experiments/port_comparison/tail_divergence_cases.py
    git -C $W/.. worktree add $W/pre  upstream/main            # before the PR
    git -C $W/.. worktree add $W/v1   2fb93d8                  # first commit
    # $W/pr-tail is already the current commit

    for v in pre:$W/pre v1:$W/v1 now:$W/pr-tail; do
      n=${v%%:*}; d=${v##*:}
      srun --mem=32G --gres=gpu:1 -w penelope --chdir=$d \
        bash -c "source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh &&
                 conda activate autocircuit && PYTHONPATH=$d python $S --label $n \
                 --out /tmp/td_$n.json"
    done
    python $S --compare /tmp/td_pre.json /tmp/td_v1.json /tmp/td_now.json
"""
import argparse
import json
import tempfile
from pathlib import Path
from typing import List

import torch as t
import transformer_lens as tl
from transformer_lens.utils import get_attention_mask

from auto_circuit.data import load_datasets_from_json

# 16 prompts per case, loaded as 8 train + 8 test at batch_size 4, so the train
# loader has TWO batches and the cache built from the first is reused on the
# second. With a single batch the cache is built from the very rows it is used
# on and the cross-batch defect cannot appear at all.
NAMES = ["Mary", "John", "Anna", "Paul", "Lisa", "Mark", "Emma", "Luke",
         "Sara", "Adam", "Nina", "Carl", "Rosa", "Finn", "Iris", "Otto"]
YEARS = [f"{y:02d}" for y in range(11, 27)]
PREFIX = "The war lasted from the year 17"
TAILS = ["", " and then again", " in the north", " for a while"]


def _pairs(clean, corrupt):
    assert len(clean) == len(corrupt) == 16
    return clean, corrupt


CASES = {
    # Each pair agrees internally until the second name (late), but different
    # prompts differ at token 1. That gap is exactly the defect.
    "differs at token 1": _pairs(
        [f"{n} walked to the shop and {n} bought a" for n in NAMES],
        [f"{n} walked to the shop and {m} bought a"
         for n, m in zip(NAMES, NAMES[1:] + NAMES[:1])],
    ),
    # All 16 share a genuinely common 8-token prefix and are the same length.
    "same length / long shared prefix": _pairs(
        [PREFIX + f"{y} to 17" for y in YEARS],
        [PREFIX + f"{y} to 18" for y in YEARS],
    ),
    # Same common prefix, but the prompts differ in length.
    "ragged / same long prefix": _pairs(
        [PREFIX + f"{y} to 1745{TAILS[i % 4]}" for i, y in enumerate(YEARS)],
        [PREFIX + f"{y} to 1702{TAILS[i % 4]}" for i, y in enumerate(YEARS)],
    ),
    "ragged / no shared prefix": _pairs(
        [f"{n} {'went far ' * (i % 4)}about" for i, n in enumerate(NAMES)],
        [f"{n} {'went off ' * (i % 4)}around" for i, n in enumerate(NAMES)],
    ),
}


def build_dataset(clean: List[str], corrupt: List[str], path: Path) -> None:
    prompts = [
        {"clean": c, "corrupt": x, "answers": [" one"], "wrong_answers": [" two"]}
        for c, x in zip(clean, corrupt)
    ]
    path.write_text(json.dumps({"prompts": prompts}))


def run_case(model, name, clean, corrupt, tmpdir):
    path = Path(tmpdir) / f"{abs(hash(name))}.json"
    build_dataset(clean, corrupt, path)
    n = len(clean) // 2
    kwargs = dict(
        model=model, path=path, device=str(model.cfg.device), batch_size=n // 2,
        train_test_size=(n, n), return_seq_length=False, shuffle=False,
    )
    cached_loader, _ = load_datasets_from_json(tail_divergence=True, **kwargs)
    plain_loader, _ = load_datasets_from_json(tail_divergence=False, **kwargs)

    diverge_idx = cached_loader.diverge_idx
    kv_cache = cached_loader.kv_cache

    # The un-stripped prompts, straight from the loader that did not strip.
    # All batches, since padding width is set by the whole loaded set.
    full = t.cat([b.clean for b in plain_loader], dim=0)
    mask = get_attention_mask(model.tokenizer, full, prepend_bos=True)

    # Same equivalence check as tests/test_data.py.
    worst = 0.0
    with t.inference_mode():
        for cb, pb in zip(cached_loader, plain_loader):
            out_cached = model(cb.clean, past_kv_cache=kv_cache)
            out_plain = model(pb.clean)
            worst = max(worst, float((out_cached - out_plain[:, diverge_idx:]).abs().max()))

    pad_per_row = (mask == 0).sum(-1).tolist()
    any_padding = any(p > 0 for p in pad_per_row)
    return {
        "case": name,
        "n_batches": len(list(plain_loader)),
        "width": int(full.shape[1]),
        "padding_per_row": f"{min(pad_per_row)}..{max(pad_per_row)}",
        "any_padding": any_padding,
        "uniform_length": not any_padding,
        "mask_row_0": mask[0].tolist(),
        "diverge_idx": diverge_idx,
        "kv_cache": kv_cache is not None,
        "cached_len": 0 if kv_cache is None else int(kv_cache.previous_attention_mask.shape[1]),
        "cached_text": "" if not diverge_idx else model.tokenizer.decode(full[0, :diverge_idx]),
        "cached_nothing_when_padded": (not any_padding) or diverge_idx == 0,
        "max_abs_dlogit": round(worst, 5),
    }


def main_run(label: str, out: str) -> None:
    model = tl.HookedTransformer.from_pretrained("gpt2")
    model.tokenizer.padding_side = "left"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (clean, corrupt) in CASES.items():
            rows.append(run_case(model, name, clean, corrupt, tmp))
            r = rows[-1]
            print(f"\n=== {label} | {r['case']} ===")
            print(f"  width {r['width']}, {r['n_batches']} batches, "
                  f"padding per row {r['padding_per_row']}, "
                  f"uniform_length={r['uniform_length']}")
            print(f"  attention mask row 0 : {r['mask_row_0']}")
            print(f"  diverge_idx = {r['diverge_idx']}   kv_cache={r['kv_cache']}"
                  f"   cached positions={r['cached_len']}")
            if r["any_padding"]:
                print(f"  padding present -> cached nothing? "
                      f"{r['cached_nothing_when_padded']}")
            if r["cached_text"]:
                print(f"  cached text: {r['cached_text']!r}")
            print(f"  max |dlogit| cached vs uncached: {r['max_abs_dlogit']}")
    if out:
        Path(out).write_text(json.dumps({"label": label, "rows": rows}, indent=1))
        print(f"\nwrote {out}")


def main_compare(files: List[str]) -> None:
    runs = [json.loads(Path(f).read_text()) for f in files]
    labels = [r["label"] for r in runs]
    cases = [r["case"] for r in runs[0]["rows"]]
    w = 34
    for field, title in [("diverge_idx", "diverge_idx"),
                         ("cached_len", "positions cached"),
                         ("max_abs_dlogit", "max |dlogit| vs uncached")]:
        print(f"\n{title}")
        print(f"  {'case':<{w}}" + "".join(f"{la:>14}" for la in labels))
        print("  " + "-" * (w + 14 * len(labels)))
        for i, case in enumerate(cases):
            vals = "".join(f"{r['rows'][i][field]:>14}" for r in runs)
            print(f"  {case:<{w}}{vals}")
    print("\npadding per row (identical across variants, shown once)")
    for row in runs[0]["rows"]:
        print(f"  {row['case']:<{w}}{row['padding_per_row']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="current")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs="*", default=None)
    args = ap.parse_args()
    if args.compare:
        main_compare(args.compare)
    else:
        main_run(args.label, args.out)
