# Thesis Notes — Index

Running notes/observations log for the thesis. The entries live in
[`thesis_notes/`](thesis_notes/), one file per thesis chapter. This file is only
the index; do not add entries here.

> Not to be confused with `thesis/`, the LaTeX write-up (the final document).

| Section | File | Entries |
|---------|------|---------|
| Introduction | [`thesis_notes/introduction.md`](thesis_notes/introduction.md) | 10 |
| Related Work | [`thesis_notes/related-work.md`](thesis_notes/related-work.md) | 38 |
| Methodology | [`thesis_notes/methodology.md`](thesis_notes/methodology.md) | 7 (HPO · Patching & Ablation Mechanics) |
| Experiments | [`thesis_notes/experiments.md`](thesis_notes/experiments.md) | 37 |
| Results | [`thesis_notes/results.md`](thesis_notes/results.md) | 4 |
| Discussion | [`thesis_notes/discussion.md`](thesis_notes/discussion.md) | 8 |
| Limitations | [`thesis_notes/limitations.md`](thesis_notes/limitations.md) | 4 |
| Notes (unfiled) | [`thesis_notes/notes.md`](thesis_notes/notes.md) | 21 bullets |

## How to add an entry

Structured entry — append to the chapter file it belongs to, newest first is not
required (files are chronological only within a topic):

```
### Short descriptive title — YYYY-MM-DD

- **Context:** what task / experiment / code area was being worked on
- **Observation:** what was found or decided
- **Significance:** why it matters for the thesis argument or sub-RQ
- **Code pointer / Run:** file:line or run directory if relevant
```

Independent note — append to `thesis_notes/notes.md`:

```
- **YYYY-MM-DD** — One-sentence capture of a quick observation, open question,
  or reminder that doesn't yet belong to a chapter.
```

Rules: always date the entry; one logical fact per bullet; don't duplicate what
is already in `CLAUDE.md`, `docs/CODEBASE_MAP.md`, or the auto-memory index.

## Searching

```bash
grep -rn "tail_divergence" thesis_notes/     # across all chapters
grep -rln "2026-08-11" thesis_notes/         # which chapters got entries on a date
```
