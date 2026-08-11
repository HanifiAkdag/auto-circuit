# `tail_divergence` strips a prefix that is not common to every prompt

Filed upstream 2026-08-11 as
[UFO-101/auto-circuit#17](https://github.com/UFO-101/auto-circuit/issues/17),
with the fix in
[PR #18](https://github.com/UFO-101/auto-circuit/pull/18) (branch
`fix/tail-divergence-shared-prefix` on the public fork `HanifiAkdag/auto-circuit-1`,
cut from upstream `8fc5afd`). Kept here as the local record of the diagnosis.

## Summary

`load_datasets_from_json` computes `diverge_idx` by comparing each clean prompt
against **its own corrupt partner**:

```python
diverge_idxs = (~(clean_prompts == corrupt_prompts)).int().argmax(dim=1)
diverge_idx = int(diverge_idxs.min().item())
```

The KV cache for that prefix is then built **once**, from the first `batch_size`
rows, and reused for every batch. But different pairs diverge at different
positions, so the minimum over pairs can exceed the point where the prompts
themselves still agree. The cache then holds one prompt's keys and values while
other prompts are run against it.

This changes the logits with **nothing ablated**, so it affects both discovery
and evaluation. `Task.init_task` enables `tail_divergence`, so every
tokenizer-based task is affected.

## Impact

Reference is each prompt run on its own: batch of one, no padding, no cache.

| task | `diverge_idx` | prompts actually agree to | wrongly cached | max &#124;Δ logit&#124; |
|---|---|---|---|---|
| Greater-Than | 8 | 2 | 6 of 11 (55%) | **22.59** |
| Docstring | 11 | 3 | 8 of 41 (20%) | **7.82** |
| IOI (`ioi/ioi_prompts`) | 3 | 1 | 2 of 21 (10%) | **3.13** |

With `tail_divergence=False` all three match the reference (0.0000 / 0.0001).

Visible directly in the tokens, for Greater-Than:

```
row 0 stripped prefix : '<|endoftext|>The contact lasted from the year 17'
row 1 stripped prefix : '<|endoftext|>The development lasted from the year 12'
                                  ^^^^^^^ differs, yet row 0's K/V is cached for both
```

Pairs differ only in the final year digits, which pushes `diverge_idx` to 8,
while the noun varies across prompts at position 2.

Note this is **not** a padding problem: Greater-Than and Docstring have uniform
prompt lengths and no padding at all, and IOI is the mildest despite being the
only task with ragged lengths. The prompts are also not sorted or bucketed by
length anywhere (`shuffle=False` is always passed to `PromptDataLoader`), so the
prefix is not constant even within a single batch: 0 of 11 batches on all three
tasks.

## Why the existing test does not catch it

`tests/test_data.py::test_tail_divergence` loads
`ioi/ioi_vanilla_template_prompts`, which happens to have uniform length **and** a
genuinely common prefix, so the invariant holds for that one dataset. It also
uses `return_seq_length=True` (which forbids padding) and exactly one batch.
Swapping in `ioi/ioi_prompts`, the dataset `IOI_COMPONENT_CIRCUIT_TASK` uses,
breaks it immediately.

## Suggested fix

Require the stripped prefix to be common to every prompt and free of padding:

```python
if tail_divergence:
    # The prefix cache is built once and reused for every batch, so the
    # stripped prefix must be common to all prompts, not just to each pair.
    if clean_prompts.shape[1] == corrupt_prompts.shape[1]:
        all_prompts = t.cat([clean_prompts, corrupt_prompts], dim=0)
        same_everywhere = (all_prompts == all_prompts[0]).all(dim=0)
        pad_id = getattr(model.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            # A padded position is a real token for some rows only.
            same_everywhere &= (all_prompts != pad_id).all(dim=0)
        same_everywhere[-1] = False  # always run at least one token
        diverge_idx = int((~same_everywhere).int().argmax().item())
```

All three tasks then match the per-prompt reference and agree on 100% of top-1
predictions.

**This is a tradeoff, not a free fix.** GPT-2 sets `pad == bos == 50256`, so a
prepended BOS can never be certified pad-free and `diverge_idx` becomes 0 on
GPT-2 tasks, disabling the optimization there. The cost is small, since those
prompts genuinely agree only to index 1-2 anyway, but it is a real behaviour
change. An alternative that keeps the optimization alive would be to compute the
attention mask once over the full sequence and write both slices into the cache
explicitly, which needs a TransformerLens change as well.

Happy to submit either as a PR.

## Note on a related secondary defect

`transformer_lens.utils.get_attention_mask` is re-derived per chunk, and because
GPT-2's BOS and PAD ids coincide its "last leading pad is really BOS" rule
misfires on a chunk containing no BOS (42 of 64 rows on one IOI batch). This is
real but secondary, and fixing it **alone** makes things worse: the wrong mask
happens to be masking out prefix positions holding another prompt's K/V, so
repairing it exposes those values (IOI 3.34 with a per-batch cache, 3.98 with the
mask corrected on top).
