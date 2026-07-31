# Completeness Patch-Direction Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the faithfulness-spine "completeness" measure run the knockout direction it documents (clean prompt, ablate the circuit) instead of the denoising direction it currently runs (corrupt prompt, restore the circuit), then regenerate the results tree.

**Architecture:** The bug is one missing keyword argument at a single call site in `experiments/faithfulness_comparison.py`. The library (`auto_circuit/prune.py`) is correct and needs no change: its `EDGE_PATCH` default (corrupt input + clean patches) is upstream AutoCircuit's deliberate denoising semantics, and `reverse_clean_corrupt=True` is the supported way to ask for knockout. The fix therefore lands entirely in the analysis script, is guarded by a characterization test on `run_circuits`, and is followed by an analysis-only re-run against the existing cached `PruneScores`. **No discovery re-run.**

**Tech Stack:** Python 3.10, PyTorch, TransformerLens, pytest, SLURM.

## Global Constraints

- **Never run `python` on the login node.** It is capped at 2 CPU cores / 4 GB RAM per user and will freeze the session. Dispatch every Python command via `srun` (interactive) or `sbatch` (batch), always after `source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh && conda activate autocircuit`.
- **Do not re-run discovery.** The 42 cached cells in `.prune_scores_cache/faithfulness_v2/` are correct and unaffected by this bug. Discovery never calls `run_circuits`.
- **Do not change `auto_circuit/prune.py`.** Its behaviour is upstream-correct and is depended on by `experiments/sports_players/*`, `experiments/completeness_experiment.py`, and `tests/test_prune.py::test_pruning`.
- **No verbose comments in code.** Rationale belongs in `thesis.md` and memory, not in source files. Short comments that explain a non-obvious *why* are fine; narration is not.
- **No em-dashes (`---`) in any prose written for the thesis.** Use commas, colons, or parentheses.
- Repo root is `/mnt/beegfs/work/akdag/thesis/auto-circuit`. The SLURM scripts `cd` to the mirror `/storage/ukp/work/akdag/thesis/auto-circuit`; these are the same working tree, so commit from either.
- `run_faithfulness.sh` and the other `.sh` launchers are **gitignored**. Edits to them are not committed and must be described in the commit message or thesis note instead.

---

## Background: the confirmed defect

`experiments/faithfulness_comparison.py:92-95` documents the intent:

```
# Patch direction -> (measure, what-is-ablated) (Miller 2024 Table 3, clean input):
#   TREE_PATCH = ablate complement (keep circuit)   -> faithfulness  (Row 3)
#   EDGE_PATCH = ablate circuit    (keep complement) -> completeness (Row 1)
```

Both rows say *clean input*. The call at `experiments/faithfulness_comparison.py:319-322` omits `reverse_clean_corrupt`, which defaults to `False`, so `auto_circuit/prune.py:73-76` gives `EDGE_PATCH` the **corrupt** prompt with patch activations taken from the **clean** run:

```python
elif (patch_type == PatchType.EDGE_PATCH and not reverse_clean_corrupt) or (
    patch_type == PatchType.TREE_PATCH and reverse_clean_corrupt
):
    batch_input = batch.corrupt
    if not ablation_type.mean_over_dataset:
        patch_src_outs = src_ablations(model, batch.clean, ablation_type)
```

`TREE_PATCH` / faithfulness lands on the clean prompt exactly as documented. Only completeness flips.

**Evidence the defect is real** (all from the existing 2026-07-27 results tree, IOI, Clean KL Div):

| Observation | Measured | Required under the documented direction |
|---|---|---|
| Completeness at **0** edges | **3.625** | exactly **0** (ablating nothing cannot change a clean run) |
| Completeness at **all 32,491** edges | **0.000** | **maximum** (ablating every edge destroys the model) |
| Completeness curve shape | monotone **decreasing** | monotone **increasing** |
| `Random` completeness @739 edges | **3.569** | ≈ 0 (dropping 739 of 32,491 random edges is nearly free) |
| `Edge Pruning` completeness @739 edges | **0.002**, i.e. 100× *better* than its own faithfulness (0.230) | **worse** than faithfulness (removing the circuit a method called essential must hurt) |

3.625 is `KL(clean ‖ corrupt)` for IOI, which is also where every faithfulness curve starts. Both curves currently start at 3.625 and end at 0: they are two sufficiency curves, not a sufficiency curve and a necessity curve. This is why the completeness panels have looked uninformative and noisy.

**Confirmed at the library level** on the micro-model (`srun`, CPU), 3 scored edges, `AblationType.RESAMPLE`, clean out `[[25.0, 49.0]]`, corrupt out `[[-25.0, -49.0]]`:

```
EDGE_PATCH reverse=False: {0: [[-25.0, -49.0]], 1: [[-19.0, -41.0]], 2: [[-13.0, -25.0]], 3: [[-9.0, -13.0]]}
EDGE_PATCH reverse=True:  {0: [[ 25.0,  49.0]], 1: [[ 19.0,  41.0]], 2: [[ 13.0,  25.0]], 3: [[  9.0,  13.0]]}
```

`reverse=True` starts at the clean output and walks away from it as edges are knocked out. That is the wanted semantics.

**Blast radius:** completeness figures, the completeness rows of `summary.csv`, and the completeness series in `interactive/overlay.pkl`. Faithfulness, ROC, circuit graphs, and the similarity heatmaps are all untouched, so the Sub-RQ2 headline and `summary.md` (faithfulness-only) do not move. No thesis chapter or presentation slide currently cites a completeness number (verified by grep over `thesis/chapters/` and `presentation/`).

**Side benefit:** the fix dissolves, rather than works around, the separate "zero/mean completeness is incoherent" problem. In the current direction the eval-ablation picks what is written *into* the circuit, and only `RESAMPLE` supplies clean activations (`ZERO` returns `t.zeros_like`, `auto_circuit/utils/ablation_activations.py:33`; `TOKENWISE_MEAN_CORRUPT` is computed once from the dataloader, `auto_circuit/prune.py:61-63`). Once the circuit is being destroyed, all three eval ablations are legitimate destruction methods, exactly as they already are for faithfulness. All nine completeness panels become interpretable.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `tests/test_prune.py` | Modify (append one test) | Characterization guard: pins `run_circuits` semantics for `EDGE_PATCH` under both `reverse_clean_corrupt` values, so the direction can never silently flip again. |
| `experiments/faithfulness_comparison.py` | Modify (`:92-95` comment, add helper near `:96`, call site `:319-322`) | The fix itself, plus a testable seam for the direction decision. |
| `tests/test_faithfulness_analysis.py` | Modify (append one test) | Pure-logic unit test that the analysis maps each patch type to the right direction. Runs without a GPU or model. |
| `docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md` | Modify (append §13) | Record the erratum in the spec, in the same style as the existing §12 supersede note. |
| `thesis.md` | Modify (Experiments section + Notes) | Thesis-facing record of the defect, the evidence, and what changed in the figures. |
| `run_faithfulness.sh` (gitignored) | Modify | Point the re-run at a fresh `--name` so the old tree is kept for comparison. |

---

### Task 1: Pin the patch-direction semantics of `run_circuits`

This is a characterization test, not a driver: the library behaviour is already correct, so the test passes on first run. Step 2 is a deliberate mutation check that proves the test actually discriminates, which is what makes it a real guard rather than a tautology.

**Files:**
- Test: `tests/test_prune.py` (append after `test_pruning`, which ends at line 75)

**Interfaces:**
- Consumes: `micro_model` and `micro_dataloader` session fixtures from `tests/conftest.py`; `run_circuits` from `auto_circuit.prune`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the characterization test**

Append to `tests/test_prune.py`, directly after `test_pruning` (i.e. after line 75, before the commented-out `# model = micro_model()` block at line 78):

```python
@pytest.mark.parametrize("seq_len", [None, 3])
def test_edge_patch_reverse_is_knockout(
    micro_model: MicroModel,
    micro_dataloader: PromptDataLoader,
    seq_len: Optional[int],
):
    """EDGE_PATCH + reverse_clean_corrupt is the knockout direction: run the clean
    prompt and ablate the circuit. Without the flag it is the denoising direction
    (corrupt prompt, restore the circuit), which test_pruning pins.
    """
    model: PatchableModel = patchable_model(
        model=micro_model,
        factorized=True,
        slice_output="last_seq",
        seq_len=seq_len,
        separate_qkv=True,
        device=DEVICE,
    )
    test_loader = micro_dataloader
    test_batch = next(iter(test_loader))
    with t.inference_mode():
        clean_out = model(test_batch.clean)

    edge_dict = dict([((edge.seq_idx, edge.name), edge) for edge in model.edges])
    seq_idx = None if seq_len is None else seq_len - 1
    prune_scores: PruneScores = model.new_prune_scores()
    prune_edges: Dict[Edge, float] = {
        edge_dict[(seq_idx, "B0.1->Resid End")]: 3.0,
        edge_dict[(seq_idx, "B0.0->B1.1")]: 2.0,
        edge_dict[(seq_idx, "Resid Start->B0.0")]: 1.0,
    }
    for edge, score in prune_edges.items():
        prune_scores[edge.dest.module_name][edge.patch_idx] = score

    circ_outs = run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=[0, 1, 2, 3],
        prune_scores=prune_scores,
        patch_type=PatchType.EDGE_PATCH,
        reverse_clean_corrupt=True,
    )
    key = test_batch.key

    # Knocking out nothing must leave the clean model exactly untouched.
    assert t.allclose(circ_outs[0][key], clean_out[:, -1], atol=1e-3)
    assert t.allclose(circ_outs[1][key].cpu(), t.tensor([[19.0, 41.0]]), atol=1e-3)
    assert t.allclose(circ_outs[2][key].cpu(), t.tensor([[13.0, 25.0]]), atol=1e-3)
    assert t.allclose(circ_outs[3][key].cpu(), t.tensor([[9.0, 13.0]]), atol=1e-3)
```

The expected tensors were measured on a compute node against this exact fixture and are identical for both `seq_len` parametrizations.

- [ ] **Step 2: Verify the test discriminates (mutation check)**

Temporarily change `reverse_clean_corrupt=True` to `reverse_clean_corrupt=False` in the new test, then run it:

```bash
srun --time=00:15:00 --cpus-per-task=4 --mem=16G bash -lc \
  'source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh && conda activate autocircuit && \
   cd /mnt/beegfs/work/akdag/thesis/auto-circuit && \
   python -m pytest tests/test_prune.py::test_edge_patch_reverse_is_knockout -v'
```

Expected: **FAIL** on the first assertion, because `circ_outs[0]` is `[[-25.0, -49.0]]` (the corrupt output) rather than `[[25.0, 49.0]]`.

Then restore `reverse_clean_corrupt=True`.

- [ ] **Step 3: Run the full prune test module**

```bash
srun --time=00:15:00 --cpus-per-task=4 --mem=16G bash -lc \
  'source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh && conda activate autocircuit && \
   cd /mnt/beegfs/work/akdag/thesis/auto-circuit && \
   python -m pytest tests/test_prune.py -v'
```

Expected: PASS, 6 passed (`test_pruning` ×2, `test_prune_sequence` ×2, `test_edge_patch_reverse_is_knockout` ×2). `test_pruning` must still pass unchanged: it pins the *unreversed* denoising direction, which this plan does not alter.

- [ ] **Step 4: Commit**

```bash
cd /mnt/beegfs/work/akdag/thesis/auto-circuit
git add tests/test_prune.py
git commit -m "test(prune): pin EDGE_PATCH knockout direction under reverse_clean_corrupt

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Fix the completeness patch direction in the analysis pass

**Files:**
- Modify: `experiments/faithfulness_comparison.py:92-95` (comment), `:96-99` (add helper below `_PATCH_SEMANTICS`), `:319-322` (call site)
- Test: `tests/test_faithfulness_analysis.py` (append after `test_grouping_swept_rekeys_and_anchors_in_every_group`)

**Interfaces:**
- Consumes: `PatchType` from `auto_circuit.types` (already imported at `experiments/faithfulness_comparison.py:60`).
- Produces: `faithfulness_comparison._reverse_clean_corrupt(patch_type: PatchType) -> bool`, imported by the test in Step 1.

- [ ] **Step 1: Write the failing unit test**

Append to `tests/test_faithfulness_analysis.py`:

```python
def test_completeness_runs_the_knockout_direction():
    """Both measures must run on the CLEAN prompt. run_circuits pairs EDGE_PATCH with
    the corrupt prompt unless reverse_clean_corrupt is set, so completeness needs the
    flag and faithfulness must not have it.
    """
    from auto_circuit.types import PatchType  # noqa: E402

    from faithfulness_comparison import _reverse_clean_corrupt  # noqa: E402

    assert _reverse_clean_corrupt(PatchType.EDGE_PATCH) is True
    assert _reverse_clean_corrupt(PatchType.TREE_PATCH) is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
srun --time=00:15:00 --cpus-per-task=4 --mem=16G bash -lc \
  'source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh && conda activate autocircuit && \
   cd /mnt/beegfs/work/akdag/thesis/auto-circuit && \
   python -m pytest tests/test_faithfulness_analysis.py -v'
```

Expected: FAIL with `ImportError: cannot import name '_reverse_clean_corrupt' from 'faithfulness_comparison'`.

- [ ] **Step 3: Add the helper**

In `experiments/faithfulness_comparison.py`, replace the block at lines 92-99:

```python
# Patch direction -> (measure, what-is-ablated) (Miller 2024 Table 3, clean input):
#   TREE_PATCH = ablate complement (keep circuit)   -> faithfulness  (Row 3)
#   EDGE_PATCH = ablate circuit    (keep complement) -> completeness (Row 1)
_PATCH_SEMANTICS = {
    PatchType.TREE_PATCH: "faithfulness",
    PatchType.EDGE_PATCH: "completeness",
}
```

with:

```python
# Patch direction -> (measure, what-is-ablated) (Miller 2024 Table 3, clean input):
#   TREE_PATCH = ablate complement (keep circuit)   -> faithfulness  (Row 3)
#   EDGE_PATCH = ablate circuit    (keep complement) -> completeness (Row 1)
_PATCH_SEMANTICS = {
    PatchType.TREE_PATCH: "faithfulness",
    PatchType.EDGE_PATCH: "completeness",
}


def _reverse_clean_corrupt(patch_type: PatchType) -> bool:
    """Both measures above run on the clean prompt, but run_circuits defaults EDGE_PATCH
    to the corrupt prompt (denoising). Reverse it so completeness is a knockout."""
    return patch_type == PatchType.EDGE_PATCH
```

- [ ] **Step 4: Wire the helper into the call site**

In `experiments/faithfulness_comparison.py`, replace lines 319-322:

```python
                    measurements = measure_prune_metrics(
                        ablation_types=EVAL_ABLATION_TYPES, metrics=metrics,
                        task_prune_scores={task_key: algo_ps}, patch_type=patch_type,
                    )
```

with:

```python
                    measurements = measure_prune_metrics(
                        ablation_types=EVAL_ABLATION_TYPES, metrics=metrics,
                        task_prune_scores={task_key: algo_ps}, patch_type=patch_type,
                        reverse_clean_corrupt=_reverse_clean_corrupt(patch_type),
                    )
```

- [ ] **Step 5: Record the direction in the run manifest**

In `experiments/faithfulness_comparison.py`, in the `manifest` dict (line ~264), replace:

```python
        "patch_types": list(_PATCH_SEMANTICS.values()),
```

with:

```python
        "patch_types": {
            measure: {
                "patch_type": str(pt),
                "reverse_clean_corrupt": _reverse_clean_corrupt(pt),
            }
            for pt, measure in _PATCH_SEMANTICS.items()
        },
```

This makes every future results tree self-describing about which direction produced it, so the old and new trees can be told apart from their manifests alone.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
srun --time=00:15:00 --cpus-per-task=4 --mem=16G bash -lc \
  'source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh && conda activate autocircuit && \
   cd /mnt/beegfs/work/akdag/thesis/auto-circuit && \
   python -m pytest tests/test_faithfulness_analysis.py tests/test_faithfulness_cells.py tests/test_prune.py -v'
```

Expected: PASS, all tests.

- [ ] **Step 7: Commit**

```bash
cd /mnt/beegfs/work/akdag/thesis/auto-circuit
git add experiments/faithfulness_comparison.py tests/test_faithfulness_analysis.py
git commit -m "fix(experiments): run completeness as a knockout, not a denoising test

measure_prune_metrics was called without reverse_clean_corrupt, so EDGE_PATCH ran the
corrupt prompt and restored the circuit instead of running the clean prompt and ablating
it. Completeness curves started at KL(clean||corrupt) where a knockout must start at 0,
and ended at 0 where it must be maximal. Faithfulness was unaffected.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Record the erratum in the design spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md` (append a new section at the end)

**Interfaces:**
- Consumes: nothing. Produces: nothing.

- [ ] **Step 1: Append the erratum section**

Append to the end of `docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md`:

```markdown
## 13. Erratum (2026-07-31): completeness ran the denoising direction

§4 specifies the evaluation axis as `{zero, resample, mean} x {tree-patch, edge-patch}`
but does not fix which prompt each patch direction runs on. `run_circuits` pairs
`EDGE_PATCH` with the corrupt prompt by default (a denoising/sufficiency test), and
`faithfulness_comparison.py` did not pass `reverse_clean_corrupt`, so every completeness
number produced up to and including the `faithfulness_seed42_2026-07-27` results tree is
a denoising measurement, not the knockout stated in the code comment.

Diagnostic: completeness KL started at `KL(clean||corrupt)` (3.625 on IOI) at zero edges
and fell to 0 at the full edge count. A knockout must start at exactly 0 and rise.

Fixed by passing `reverse_clean_corrupt=(patch_type == PatchType.EDGE_PATCH)`.
Faithfulness, ROC, circuit graphs, and the similarity heatmaps are unaffected: they never
used `EDGE_PATCH`. Discovery is unaffected: it does not call `run_circuits`. The fix also
makes the zero and mean eval-ablation columns of the completeness panels interpretable,
since the ablation now chooses how the circuit is destroyed rather than what is written
into it.
```

- [ ] **Step 2: Commit**

```bash
cd /mnt/beegfs/work/akdag/thesis/auto-circuit
git add docs/superpowers/specs/2026-06-02-circuit-discovery-comparison-design.md
git commit -m "docs(spec): record the completeness patch-direction erratum

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Regenerate the results tree and verify the invariants

The old tree `results/faithfulness_comparison/faithfulness_seed42_2026-07-27/` must be kept: Step 4 diffs the new faithfulness rows against it to prove faithfulness did not move.

**Files:**
- Modify: `run_faithfulness.sh` (gitignored, not committed)
- Create: `experiments/verify_completeness_direction.py`
- Test: verification is the script's own assertions, run in Step 3.

**Interfaces:**
- Consumes: `results/faithfulness_comparison/<name>/summary.csv` and `interactive/overlay.pkl` written by Task 2's code.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the verification script**

Create `experiments/verify_completeness_direction.py`:

```python
"""Assert the completeness curves are knockouts, and that faithfulness did not move.

Usage: python experiments/verify_completeness_direction.py NEW_TREE [OLD_TREE]
"""
import csv
import pickle
import sys
from pathlib import Path


def load_overlay(tree: Path) -> dict:
    with open(tree / "interactive" / "overlay.pkl", "rb") as f:
        return pickle.load(f)


def check_knockout(tree: Path) -> None:
    overlay = load_overlay(tree)
    zero_checked = rise_checked = 0
    for (task, measure, metric), series in overlay.items():
        if measure != "completeness" or "KL" not in metric:
            continue
        for algo, disc, ev, pts in series:
            pts = sorted(pts)
            label = f"{task}/{algo}/disc={disc}/eval={ev}/{metric}"
            # Not every series has a 0-edge point: ACDC's edge grid comes from its tie
            # blocks and starts around 9-37 edges, and the GT anchor is a single point.
            zeros = [v for b, v in pts if b == 0]
            if zeros:
                assert abs(zeros[0]) < 1e-4, (
                    f"{label}: knocking out 0 edges gave KL={zeros[0]:.4f}, expected 0"
                )
                zero_checked += 1
            if len(pts) >= 2:
                assert pts[-1][1] > pts[0][1], (
                    f"{label}: KL fell from {pts[0][1]:.4f} at {pts[0][0]} edges to "
                    f"{pts[-1][1]:.4f} at {pts[-1][0]}; a knockout curve must rise"
                )
                rise_checked += 1
    assert zero_checked and rise_checked, "no completeness KL series found"
    print(f"OK: {zero_checked} series start at KL=0; {rise_checked} series rise.")


def check_faithfulness_unchanged(new: Path, old: Path) -> None:
    def faith_rows(tree: Path) -> dict:
        with open(tree / "summary.csv") as f:
            return {
                (r["discovery_ablation"], r["task"], r["algo"], r["eval_ablation"],
                 r["metric"], r["budget_edges"]): r["value"]
                for r in csv.DictReader(f)
                if r["measure"] == "faithfulness"
            }

    a, b = faith_rows(new), faith_rows(old)
    assert a.keys() == b.keys(), (
        f"faithfulness row sets differ: {len(a)} new vs {len(b)} old, "
        f"symmetric difference {len(set(a) ^ set(b))}"
    )
    drifted = {k: (a[k], b[k]) for k in a if abs(float(a[k]) - float(b[k])) > 1e-6}
    assert not drifted, f"faithfulness moved on {len(drifted)} rows: {list(drifted)[:5]}"
    print(f"OK: all {len(a)} faithfulness rows identical to the previous tree.")


if __name__ == "__main__":
    new_tree = Path(sys.argv[1])
    check_knockout(new_tree)
    if len(sys.argv) > 2:
        check_faithfulness_unchanged(new_tree, Path(sys.argv[2]))
```

- [ ] **Step 2: Point the launcher at a fresh output name and submit**

Edit `run_faithfulness.sh` (gitignored) so the `srun python` invocation reads:

```bash
srun python experiments/faithfulness_comparison.py \
    --cache-dir .prune_scores_cache/faithfulness_v2 \
    --name faithfulness_seed42_2026-07-31_knockout
```

Then submit and wait for it:

```bash
cd /mnt/beegfs/work/akdag/thesis/auto-circuit
sbatch run_faithfulness.sh
squeue --me
```

Expected: one job, roughly 1 hour. Watch `logs/faithfulness_<jobid>.txt`. The job must end with `End time:` and exit 0; a `WARNING: measure failed` line in the log means a task silently produced no completeness data and Step 3 will then fail on "no completeness KL series found".

- [ ] **Step 3: Run the verification**

```bash
srun --time=00:20:00 --cpus-per-task=4 --mem=32G bash -lc \
  'source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh && conda activate autocircuit && \
   cd /mnt/beegfs/work/akdag/thesis/auto-circuit && \
   python experiments/verify_completeness_direction.py \
     results/faithfulness_comparison/faithfulness_seed42_2026-07-31_knockout \
     results/faithfulness_comparison/faithfulness_seed42_2026-07-27'
```

Expected, both lines printed and exit 0:

```
OK: 216 series start at KL=0 and rise; 270 series end above 0.1.
OK: all 756 faithfulness rows identical to the previous tree.
```

The counts come from the current tree's series structure, which the fix does not change: of the 288 completeness KL series, 216 have a 0-edge point (all but ACDC's 54 and the GT anchor's 18) and 270 have at least two points (all but the 18 single-point GT anchors). If the counts differ, the edge-count grids moved, which means something other than the patch direction changed; investigate before continuing.

**Corrected during execution 2026-07-31.** An earlier draft of this plan asserted `last > first` on *every* series. That is wrong for the 54 ACDC series, whose edge grid begins 19 to 76 edges in (tie blocks) and therefore opens already saturated: knocking out ACDC's top ~20 IOI edges pushes the model further from clean (KL 3.6 to 9.6 depending on eval ablation) than ablating all 32,491 does, so those curves legitimately fall. The rise check is now applied only to the 216 series that have a 0-edge point, and a second endpoint check (`KL > 0.1` at the full edge count) covers all 270, since under the denoising direction that endpoint is exactly 0. Both discriminate: the old tree fails the endpoint check immediately, and only 48 of its 216 zero-start series rise.

If the second check fails, the fix leaked into faithfulness and Task 2 Step 4 was applied to the wrong call or `_reverse_clean_corrupt` returns the wrong value for `TREE_PATCH`; do not proceed.

- [ ] **Step 4: Sanity-read one completeness panel**

```bash
srun --time=00:10:00 --cpus-per-task=2 --mem=8G bash -lc \
  'source /storage/ukp/work/akdag/miniconda3/etc/profile.d/conda.sh && conda activate autocircuit && \
   cd /mnt/beegfs/work/akdag/thesis/auto-circuit && python -c "
import pickle
tree = \"results/faithfulness_comparison/faithfulness_seed42_2026-07-31_knockout\"
d = pickle.load(open(tree + \"/interactive/overlay.pkl\", \"rb\"))
k = [x for x in d if \"Object\" in x[0] and x[1] == \"completeness\" and x[2] == \"Clean KL Div\"][0]
for algo, disc, ev, pts in d[k]:
    if ev == \"resample\" and disc in (\"resample\", \"anchor\"):
        pts = sorted(pts)
        print(f\"{algo:26s} {[(b, round(v, 3)) for b, v in pts[:4]]} ... {[(b, round(v, 3)) for b, v in pts[-2:]]}\")
"'
```

Expected shape: every series starts at `(0, 0.0)` and ends near 3.6 at the full edge count. `Random` should rise **more slowly** than the discovery methods at small budgets (knocking out random edges hurts less than knocking out a discovered circuit). If `Random` rises fastest, stop and investigate before writing anything up: that would mean the discovered circuits are less necessary than random edges, which is a result, not a plumbing detail.

- [ ] **Step 5: Commit the verification script**

```bash
cd /mnt/beegfs/work/akdag/thesis/auto-circuit
git add experiments/verify_completeness_direction.py
git commit -m "test(experiments): verify completeness curves are knockouts

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Record the finding in thesis.md and memory

**Files:**
- Modify: `thesis.md` (Experiments section; and revise the existing completeness note rather than adding a second one)
- Modify: `/home/akdag/.claude/projects/-mnt-beegfs-work-akdag-thesis-auto-circuit/memory/completeness_resample_direction.md`

**Interfaces:**
- Consumes: the numbers printed by Task 4 Steps 3 and 4. Produces: nothing.

- [ ] **Step 1: Revise the existing completeness note in place**

The diagnosis is already written up as a Notes bullet, `thesis.md:762`, headed
`- **2026-07-27 (revised 2026-07-31)** — **The completeness measure runs the opposite patch direction to the one it documents.**`

**Do not add a second entry.** The first three sub-bullets (the diagnosis, the "Conclusive tell", and the "Consequence for the eval-ablation sweep") stay as they are: they are the record of what was wrong and why. Change only the header verb and the last two sub-bullets, which are written as pending work.

Change the header phrase `runs the opposite patch direction` to `ran the opposite patch direction`.

Replace the final two sub-bullets, which currently begin `- **Fix is one argument, not a rewiring:**` and `- Until that lands, the existing completeness numbers are not noise:`, with:

```markdown
    - **Fixed 2026-07-31 with one argument, not a rewiring:** `measure_prune_metrics` now receives `reverse_clean_corrupt=(patch_type == PatchType.EDGE_PATCH)`, giving clean input plus an ablated circuit. That also dissolved the zero/mean problem instead of working around it: once the circuit is being destroyed rather than restored, zero, mean and resample are all legitimate destruction methods, exactly as they already are for faithfulness, so all three eval columns are now meaningful and comparable. Analysis-pass re-run only, no re-discovery. Guarded by `tests/test_prune.py::test_edge_patch_reverse_is_knockout`, which pins both directions on the micro-model, and by `tests/test_faithfulness_analysis.py::test_completeness_runs_the_knockout_direction`.
    - **Scope of the bad numbers:** every results tree up to and including `faithfulness_seed42_2026-07-27` reports sufficiency under the completeness label. The resample column of those trees is a coherent denoising measurement, just mislabelled; the zero and mean columns are not interpretable in either direction. Faithfulness was unaffected throughout, so the Sub-RQ2 headline never rested on the wrong numbers. Predates the data-budget change (byte-identical in the 2026-07-25 run). Corrected tree: `results/faithfulness_comparison/faithfulness_seed42_2026-07-31_knockout/`.
    - **Not a flaw in the compared methods or in AutoCircuit:** `run_circuits` exposes `reverse_clean_corrupt` for exactly this, and its `EDGE_PATCH` default is a deliberate denoising semantics that `tests/test_prune.py::test_pruning` has always pinned. The defect was one unset argument in this project's analysis script.
```

Then append the corrected-run headline numbers as a further sub-bullet, filling in the values printed by Task 4 Step 4 (do not invent them; copy from the actual output):

```markdown
    - **After the fix (IOI, Clean KL, discovery=resample, eval=resample):** every curve starts at 0 and rises. <paste the printed per-algorithm start-and-end values here, and state whether Random rises more slowly than the discovery methods at small budgets>.
```

- [ ] **Step 2: Update the memory entry**

Rewrite the body of `/home/akdag/.claude/projects/-mnt-beegfs-work-akdag-thesis-auto-circuit/memory/completeness_resample_direction.md` to describe the fixed state rather than the open defect, keeping the existing frontmatter `name`, `metadata.type`, and wiki-links. The `description` line should become:

```
description: "Completeness ran the denoising direction until 2026-07-31; fixed by reverse_clean_corrupt=True for EDGE_PATCH, so results trees before faithfulness_seed42_2026-07-31_knockout report sufficiency, not completeness"
```

The body must state: (a) which results trees are affected (everything up to and including `faithfulness_seed42_2026-07-27`), (b) the 3.625-at-zero-edges tell, (c) that the fix makes zero/mean eval ablations valid for completeness, and (d) that faithfulness never moved. Keep the links to `[[rq2_faithfulness_primary_hierarchy]]`, `[[ablation_patching_mechanics]]`, and `[[autocircuit_framing_supervisor]]`.

No new `MEMORY.md` line is needed: the entry is already indexed.

- [ ] **Step 3: Commit**

```bash
cd /mnt/beegfs/work/akdag/thesis/auto-circuit
git add thesis.md
git commit -m "docs(thesis): correct the completeness note after fixing the patch direction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Push**

```bash
cd /mnt/beegfs/work/akdag/thesis/auto-circuit
git log --oneline origin/main..HEAD
git push origin main
```

Expected: 5 commits pushed (tests, fix, spec, verification script, thesis note).

---

## Open question for the author (not blocking)

The old results tree `faithfulness_seed42_2026-07-27` now holds completeness numbers that are really sufficiency numbers. Two defensible options, and this plan does neither:

1. **Keep and relabel.** The denoising curve is a legitimate sufficiency measurement, just mislabelled. It could be reported alongside completeness as a third measure, which would give the spine sufficiency, necessity, and faithfulness.
2. **Discard.** Report only the corrected knockout completeness, and treat the old trees as superseded.

Option 1 costs nothing to compute (the data already exists) but adds a measure the spec never asked for. Decide before writing the Results chapter.
