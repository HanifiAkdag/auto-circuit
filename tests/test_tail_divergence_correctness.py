"""`tail_divergence` must not change what the model computes.

The prefix KV cache is built once and reused for every batch, so the stripped
prefix has to be identical for *every* prompt. It previously was not: the
divergence index was computed within each clean/corrupt pair and never compared
across prompts, so the cache held one batch's prefix while other batches were run
against it. On stock tasks that moved the logits by up to ~22 with nothing
ablated.

The correctness test compares against the least ambiguous reference available:
each prompt run on its own, batch of one, no padding, no cache, no dataloader.

Measured against that reference before the fix, with nothing ablated:
Greater-Than 22.5883, Docstring 7.8186, IOI 3.1310 max |delta logit|; after it,
0.0000/0.0001 on all three with 100% top-1 agreement. Two candidate fixes were
rejected on the same measurement: a per-batch cache leaves IOI at 3.3437, and
adding a corrected attention mask on top makes it 3.9754.

Reported upstream as UFO-101/auto-circuit#17, fixed in PR #18.
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import pytest
import torch as t

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from auto_circuit.data import load_datasets_from_json  # noqa: E402
from auto_circuit.experiment_utils import load_tl_model  # noqa: E402
from auto_circuit.tasks import (  # noqa: E402
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    DATASET_CACHE,
    MODEL_CACHE,
    Task,
)
from auto_circuit.utils.misc import repo_path_to_abs_path  # noqa: E402

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
SEED = 0
TASKS = [
    ("docstring", DOCSTRING_COMPONENT_CIRCUIT_TASK, "attn-only-4l"),
    ("greaterthan", GREATERTHAN_COMPONENT_CIRCUIT_TASK, "gpt2"),
    ("ioi", IOI_COMPONENT_CIRCUIT_TASK, "gpt2"),
]


def _build(tmpl) -> Task:
    DATASET_CACHE.clear()
    MODEL_CACHE.clear()
    task = Task(
        key=tmpl.key, name=tmpl.name, batch_size=tmpl.batch_size,
        batch_count=tmpl.batch_count, token_circuit=tmpl.token_circuit,
        _dataset_name=tmpl._dataset_name, factorized=tmpl.factorized,
        separate_qkv=tmpl.separate_qkv, _true_edge_func=tmpl._true_edge_func,
        slice_output=tmpl.slice_output, dtype=tmpl.dtype, seed=SEED,
        _model_def=tmpl._model_def,
    )
    task.init_task()
    return task


def _skip_if_inert(diverge_idx: int, name: str) -> None:
    """Refuse to pass a test that the stripping path never reaches.

    On the stock GPT-2 tasks `diverge_idx` is 0, which makes `tail_divergence` a
    literal no-op, so any test of its behaviour would pass without exercising
    anything. Skipping keeps that visible instead of counting as coverage.
    """
    if diverge_idx == 0:
        pytest.skip(f"{name}: diverge_idx=0, stripping path inert (see DEEP_PREFIX tests)")


def _test_prompts(tmpl):
    """Clean prompts of the test split, in loader order."""
    path = repo_path_to_abs_path(f"datasets/{tmpl._dataset_name}.json")
    prompts = json.loads(Path(path).read_text())["prompts"]
    random.seed(SEED)
    random.shuffle(prompts)
    bs = tmpl.batch_size
    bc = tmpl.batch_count
    bs0, bs1 = (bs, bs) if isinstance(bs, int) else bs
    bc0, bc1 = (bc, bc) if isinstance(bc, int) else bc
    chosen = prompts[: bs0 * bc0 + bs1 * bc1]
    return [p["clean"] for p in chosen[bs0 * bc0 :]]


# --- structural: what may be stripped -----------------------------------------


@pytest.mark.parametrize("name,tmpl,_model", TASKS)
def test_stripped_prefix_is_common_to_every_prompt(name, tmpl, _model):
    """Whatever is stripped must be identical across all prompts and pad-free.

    Anything else makes the shared cache invalid for some row.
    """
    task = _build(tmpl)
    diverge_idx = task.train_loader.diverge_idx
    _skip_if_inert(diverge_idx, name)

    path = repo_path_to_abs_path(f"datasets/{tmpl._dataset_name}.json")
    prompts = json.loads(Path(path).read_text())["prompts"]
    tokenizer = task.model.tokenizer
    bos, pad = tokenizer.bos_token, tokenizer.pad_token_id

    prefixes = set()
    for p in prompts[:200]:
        for key in ("clean", "corrupt"):
            ids = tokenizer(bos + p[key])["input_ids"][:diverge_idx]
            assert pad not in ids, f"{name}: stripped prefix contains padding"
            prefixes.add(tuple(ids))
    assert len(prefixes) == 1, (
        f"{name}: {len(prefixes)} distinct prefixes are being stripped, but the "
        f"cache holds only one"
    )


# --- correctness: the model's output must be unchanged ------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name,tmpl,model_name", TASKS)
def test_forward_matches_unbatched_reference(name, tmpl, model_name):
    """The library's forward must equal each prompt run on its own.

    Nothing is ablated, so any difference is the dataloader altering the model's
    computation.
    """
    task = _build(tmpl)
    reference = load_tl_model(model_name, DEVICE)
    tokenizer = reference.tokenizer
    prompts = _test_prompts(tmpl)

    got = []
    with t.no_grad():
        for batch in task.test_loader:
            got.append(task.model(batch.clean)[task.model.out_slice].float().cpu())
    got = t.cat(got)
    assert got.shape[0] == len(prompts)

    want = []
    with t.no_grad():
        for s in prompts:
            ids = tokenizer(tokenizer.bos_token + s, return_tensors="pt")
            want.append(reference(ids["input_ids"].to(DEVICE))[0, -1].float().cpu())
    want = t.stack(want)

    max_abs = (got - want).abs().max().item()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    assert max_abs < 1e-2, f"{name}: max|dlogit|={max_abs:.4f}, argmax agree {agree:.1%}"


@pytest.mark.slow
@pytest.mark.parametrize("name,tmpl,model_name", TASKS)
def test_disabling_tail_divergence_changes_nothing(name, tmpl, model_name):
    """With the prefix stripping correct, turning it off must be a no-op.

    This is the property the feature is supposed to have, and the one that failed
    before: it is a speed optimisation, not a change of semantics.
    """
    path = repo_path_to_abs_path(f"datasets/{tmpl._dataset_name}.json")
    bs = tmpl.batch_size if isinstance(tmpl.batch_size, int) else tmpl.batch_size[0]
    model = load_tl_model(model_name, DEVICE)

    outs = {}
    for tail in (True, False):
        _, loader = load_datasets_from_json(
            model=model, path=path, device=DEVICE, batch_size=bs,
            train_test_size=(bs, bs * 2), tail_divergence=tail,
            random_seed=SEED, return_seq_length=False,
        )
        if tail:
            _skip_if_inert(loader.diverge_idx, name)
        kv = loader.kv_cache
        chunk = []
        with t.no_grad():
            for batch in loader:
                if kv is not None:
                    chunk.append(model(batch.clean, past_kv_cache=kv)[:, -1].float().cpu())
                else:
                    chunk.append(model(batch.clean)[:, -1].float().cpu())
        outs[tail] = t.cat(chunk)

    max_abs = (outs[True] - outs[False]).abs().max().item()
    assert max_abs < 1e-2, f"{name}: tail_divergence changed the output by {max_abs:.4f}"


# --- why the stock GPT-2 tasks strip nothing ----------------------------------


@pytest.mark.parametrize("name,tmpl", [(n, m) for n, m, k in TASKS if k == "gpt2"])
def test_gpt2_strips_nothing_because_pad_and_bos_share_an_id(name, tmpl):
    """On GPT-2 the guard gives up the whole prefix, and that is deliberate.

    GPT-2 sets pad == bos == eos == 50256, so a prepended BOS is indistinguishable
    from padding and position 0 can never be certified pad-free. Stripping stops
    there. The prompts themselves agree further than that, so this records the
    cost of the guard rather than a property of the data: it is the exact position
    where TransformerLens' BOS-promotion rule misfires on a chunk, which is what
    made the shared cache wrong in the first place.

    If a future change makes GPT-2 strip a prefix again, this test should fail and
    the BOS-promotion interaction must be re-checked before it is accepted.
    """
    task = _build(tmpl)
    tokenizer = task.model.tokenizer
    assert tokenizer.pad_token_id == tokenizer.bos_token_id, (
        "this test only describes tokenizers where pad and bos collide"
    )
    assert task.train_loader.diverge_idx == 0

    loader = task.train_loader
    prompts = t.cat(
        [t.cat([b.clean for b in loader]), t.cat([b.corrupt for b in loader])], dim=0
    )
    agree = (prompts == prompts[0]).all(dim=0)
    agree[-1] = False
    agree_only_idx = int((~agree).int().argmax().item())
    assert agree_only_idx > 0, (
        f"{name}: expected the prompts to agree past position 0, so that "
        f"diverge_idx=0 is attributable to the pad guard rather than the data"
    )


# --- the stripping path, exercised deeply -------------------------------------

# The stock tasks reach diverge_idx 0 (GPT-2) or 3 of 41 (Docstring), so nothing
# above tests a genuinely deep cache. This fixture builds one: a 26-token shared
# prefix on a model whose pad (2) and bos (1) ids differ, leaving a single token
# to run through the model.

DEEP_PREFIX = (
    "The archive records that the committee met in the spring and again in "
    "the autumn to review the annual accounts of the society. Item"
)
DEEP_PAIRS = [(" one", " two"), (" three", " four"), (" five", " six")]
DEEP_MODEL = "attn-only-4l"


def _write_deep_dataset(n: int = 64) -> Path:
    """Equal-length prompts, so no row is padded and the whole prefix is strippable."""
    prompts = []
    for i in range(n):
        clean_suf, corrupt_suf = DEEP_PAIRS[i % len(DEEP_PAIRS)]
        prompts.append({
            "clean": DEEP_PREFIX + clean_suf,
            "corrupt": DEEP_PREFIX + corrupt_suf,
            "answers": [" nine"],
            "wrong_answers": [" ten"],
        })
    path = Path(tempfile.mkdtemp()) / "deep_prefix.json"
    path.write_text(json.dumps({"prompts": prompts}))
    return path


@pytest.fixture(scope="module")
def deep_loaders():
    model = load_tl_model(DEEP_MODEL, DEVICE)
    path = _write_deep_dataset()
    kwargs = dict(
        model=model, path=path, device=DEVICE, batch_size=8,
        train_test_size=(32, 32), shuffle=False, random_seed=SEED,
    )
    _, on = load_datasets_from_json(tail_divergence=True, **kwargs)
    _, off = load_datasets_from_json(tail_divergence=False, **kwargs)
    return model, on, off


def test_deep_prefix_is_actually_cached(deep_loaders):
    """The fixture must reach the regime the stock tasks never enter."""
    _, on, off = deep_loaders
    assert on.diverge_idx >= 20, f"fixture only stripped {on.diverge_idx} tokens"
    assert on.kv_cache is not None
    run_len = next(iter(on)).clean.shape[1]
    full_len = next(iter(off)).clean.shape[1]
    assert full_len - run_len == on.diverge_idx
    assert run_len <= 2, "the prefix should account for nearly the whole sequence"


@pytest.mark.slow
def test_deep_prefix_forward_matches_unbatched_reference(deep_loaders):
    """A 26-token cached prefix must still reproduce each prompt run on its own."""
    model, on, off = deep_loaders

    got = []
    with t.no_grad():
        for batch in on:
            got.append(model(batch.clean, past_kv_cache=on.kv_cache)[:, -1].float().cpu())
    got = t.cat(got)

    # The reference is the same rows run one at a time with no cache. Taking the
    # tokens from the uncached loader rather than re-deriving them keeps the two
    # sides in the same order by construction.
    want = []
    with t.no_grad():
        for batch in off:
            for row in batch.clean:
                want.append(model(row.unsqueeze(0))[0, -1].float().cpu())
    want = t.stack(want)
    assert got.shape == want.shape

    max_abs = (got - want).abs().max().item()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    assert max_abs < 1e-2, f"deep prefix: max|dlogit|={max_abs:.4f}, argmax agree {agree:.1%}"


@pytest.mark.slow
def test_deep_prefix_toggle_is_a_noop(deep_loaders):
    """The no-op property, tested where it can actually fail."""
    model, on, off = deep_loaders
    outs = []
    with t.no_grad():
        for loader in (on, off):
            chunk = []
            for batch in loader:
                if loader.kv_cache is not None:
                    chunk.append(
                        model(batch.clean, past_kv_cache=loader.kv_cache)[:, -1].float().cpu()
                    )
                else:
                    chunk.append(model(batch.clean)[:, -1].float().cpu())
            outs.append(t.cat(chunk))
    max_abs = (outs[0] - outs[1]).abs().max().item()
    assert max_abs < 1e-2, f"deep prefix: tail_divergence changed the output by {max_abs:.4f}"


@pytest.mark.slow
def test_any_padded_row_disables_stripping():
    """Unequal prompt lengths must switch stripping off entirely.

    Left padding puts a pad at position 0 for the shorter rows, so no split index
    means the same thing for every row and nothing may be stripped. This is why
    the feature is inert on most real datasets, and it must stay inert rather than
    silently stripping a position that is padding for some prompts.
    """
    model = load_tl_model(DEEP_MODEL, DEVICE)
    prompts = []
    for i in range(32):
        # A deliberately ragged tail: the suffixes differ in token count.
        suffix = " one" if i % 2 else " one hundred and seventeen"
        prompts.append({
            "clean": DEEP_PREFIX + suffix,
            "corrupt": DEEP_PREFIX + " two",
            "answers": [" nine"],
            "wrong_answers": [" ten"],
        })
    path = Path(tempfile.mkdtemp()) / "ragged.json"
    path.write_text(json.dumps({"prompts": prompts}))

    _, loader = load_datasets_from_json(
        model=model, path=path, device=DEVICE, batch_size=8,
        train_test_size=(16, 16), tail_divergence=True, shuffle=False,
        random_seed=SEED,
    )
    assert loader.diverge_idx == 0
    assert loader.kv_cache is None
