# Scientific & Thesis Writing Guide

Reference for writing and revising every chapter of the thesis (`thesis-overleaf/Thesis/chapters/*.tex`).
Operationalized as a checklist by the `writing-thesis-prose` skill; this file is the depth
reference behind it.

The author's fixed style rules (no em-dashes, `\emph` sparingly, natbib, no invented
jargon, DiscoGP paper-faithful only) live in `CLAUDE.md` → **Thesis Writing Style**; this
guide complements them and does not repeat them.

This thesis's chapter order (from the outline in `introduction.tex`): Introduction →
Background → Related Work → Methodology → Experiments (setup) → Results → Discussion →
Conclusion. Sources are cross-validated across university writing centres and are listed at
the end.

---

## 1. Overall structure and the order to write in

Standard scientific order: Abstract, Introduction, Background/Related Work, Methodology,
Experiments/Results, Discussion, Conclusion. Two lessons all sources agree on:

- **Write the abstract last.** It mirrors the finished thesis; it cannot summarize what is
  not yet written.
- **Revisit the Introduction after the Conclusion.** The two must agree: what the intro
  promises, the conclusion must deliver. Check scope claims against what was actually done.

Each chapter has one job. A common failure is bleeding one chapter's job into another
(interpreting in Results, reviewing literature in Discussion, presenting new data in
Conclusion). Keep the jobs separate.

---

## 2. Scientific register (applies to every chapter)

- **Active, declarative statements.** "This thesis integrates / evaluates / compares...",
  not questions, not "we will try to".
- **No contractions, idioms, or colloquialisms.** "do not", not "don't". Avoid "taken at
  face value", "a small part of", "trading some", "at the end of the day", "a lot of".
- **Precision over vagueness.** Name the thing exactly. "a sparse subgraph of attention
  heads and MLPs", not "a small part of the network". Cut filler quantifiers ("some",
  "several", "apparent", "a number of").
- **One idea per sentence.** Three or more commas usually means split it.
- **Hedge only when warranted.** Prefer confident, citation-backed claims to "it could be
  argued" or "it is believed"; hedge where the evidence is genuinely uncertain.
- **Write for a reader who knows the field.** The graders are the supervisors, and the intended reader is someone taking a course on the topic. Presume transformers, attention, backpropagation and cross-entropy. Never explain what such a reader knows, never restate a displayed equation in prose, and never state one fact twice in different words. Supervisor review of Background Section 2.1 (2026-08-21) rejected the draft as "not professional and even childish" on exactly these three counts.
- **State the reason a practice exists, not its mechanics.** "An edge into a head terminates at one of the three" describes bookkeeping and says nothing. Read the source for why the practice exists, then give the consequence at the level the section is about. The reason must not itself be textbook material.
- **Cite or own every non-trivial claim.** A factual claim is either supported by a
  citation or presented explicitly as a contribution of this thesis. Nothing floats.

---

## 3. Sentence-level craft (Gopen & Swan, reader-expectation principles)

These are principles, not rules, but they resolve most "reads badly" problems:

- **Topic position (sentence start) = old / familiar information** that links back to what
  the reader already knows. **Stress position (sentence end) = the new information you want
  emphasized.** Put the payload last.
- **Keep the subject next to its verb.** Long interruptions between them force the reader to
  hold the subject in memory.
- **Old → new flow.** Each sentence should open with a link to the previous one and end
  with what advances the argument. This is what makes paragraphs feel connected.
- **One unit of discourse = one point.** A sentence, paragraph, or section should make a
  single point. If it makes several, divide it.

Self-check per sentence: Does the topic open it? Is the truly-new material at the end? Is
the verb close to the subject? Does it make one point?

---

## 4. Chapter-by-chapter guidance

### Abstract
A condensed mirror of the whole thesis: problem → approach → key findings → conclusion
(the labelled shape Background–Methods–Results–Conclusions). Short, self-contained, no
citations, written last.

### Introduction — Swales' CARS model
Follow the funnel general → specific through three moves:
- **Move 1 Establish the territory:** open with ONE concise centrality claim (never an
  overloaded first sentence), generalize the state of the art, review prior work briefly.
- **Move 2 Establish the niche:** name the gap with *neutral contrast* ("prior work focused
  on A rather than B"), not dismissal. At master's level the gap must be **methodological
  or conceptual**, not merely topical.
- **Move 3 Occupy the niche:** state the purpose (answer "so what?"), announce what the
  work does, summarize contributions, outline the structure.

Frame the correct **central object**: this thesis is about the controlled *comparison of
circuit-discovery methods*, not the circuit itself (the circuit is the artifact the methods
produce). Trace each RQ → a contribution → a chapter.

### Background — the machinery the reader must already have
Background and Related Work answer different questions. **Background: what must the reader
know to follow my methodology?** **Related Work: what have others done about my problem?**
Both are almost entirely external material and therefore almost entirely cited, but the
jobs do not overlap. The discovery algorithms (EP, DiscoGP, ACDC, EAP) are attempts at this
thesis's problem and belong in Related Work, not in Background.

- **Every item must be justified by downstream use.** State explicitly how each background
  concept is used later ("this is why the edge abstraction is well defined"). Per-subsection
  test: name the later chapter that breaks without it. If none, cut the subsection.
  Background is part of the tree of problems and solutions, not a laundry list.
- **Assume a competent CS/ML reader.** Do not explain backpropagation or softmax. Do explain
  the factorized residual-stream view, because the edge abstraction depends on it and a
  general ML reader does not carry it.
- **Organize by concept, never by paper or by chronology.** The layout is itself a
  contribution: it is the field seen from this thesis's vantage point.
- **Fix the notation and vocabulary here** (edge, circuit, ablation type, patch direction,
  faithfulness), define each once, and use it unchanged thereafter. This is the one place
  where `\emph` on a first definition is warranted.
- **Register:** present tense for established facts, past tense for what a specific work did.
  No hedging on textbook material. Borrowed figures are encouraged, cited in the caption.
- **Length:** roughly 10–15% of the body. Erring short is safer than erring long; this
  chapter is infrastructure, not the argument.

Planned order for this thesis: (1) transformers as a computational graph, (2) the circuit
abstraction and its quality properties, (3) activation patching and ablation types,
(4) AutoCircuit as the edge-patchable architecture standardization, (5) models and tasks.

### Related Work — synthesize, do not summarize
- A summary reports what each paper says one by one; a **synthesis** states *your* point and
  uses multiple papers as evidence for it. Organize by **theme / debate / method**, not one
  paragraph per paper.
- **Rearrangement test:** if a paragraph's sentences could be reordered without loss, it is
  a summary, not a synthesis. A synthesis paragraph breaks if you remove any of the linked
  studies.
- Start each paragraph with a claim sentence (your point); marshal the citations under it.
- Critically evaluate: expose gaps, assumptions, and biases — the niche the thesis fills.
  Use neutral contrastive language, not disparagement.

### Methodology — justify, do not merely describe
- Report the design, procedure, and analysis in enough detail that another researcher could
  **reproduce** the work.
- For every non-obvious choice, **justify it against alternatives** ("method X was chosen
  over Y because..."). Description alone is not a methodology.
- Acknowledge trade-offs, confounds, and how they are controlled (this thesis: the
  discovery/eval-ablation/patch-direction triple, seeds, null model).
- Aligns methods to the RQs: each RQ should have a method that answers it.

### Results / Experiments — report the "what", objectively
- State findings plainly, tied to each RQ; **no interpretation** here (that is the
  Discussion's job). Keep it dry and factual.
- Lead the reader through figures and tables with text; every figure/table is introduced,
  numbered, and referenced. Consistent formatting throughout.
- Report what happened, including negative or unexpected results, without explaining why.

### Discussion — interpret the "so what"
- Usually the longest analytical chapter. Move from data-gatherer to data-interpreter.
- For each key result: what it means, whether it **corroborates or contradicts** prior work
  (cite), and its theoretical/practical implications.
- Tie findings back explicitly to the research questions.
- State **limitations** transparently and what they do and do not undermine.

### Conclusion — answer the question, generally
- Concisely answer the main RQ; state what was done, found, and learned. More general and
  shorter than the Discussion.
- **No new data or new arguments.** Restate limitations at a high level; give concrete
  future-work directions.
- Must close the loop opened by the Introduction.

---

## 5. Figures, tables, and citations

- **Figure/table captions explain what to notice**, not just what the object is. Every
  figure and table is numbered, referenced in the text, and introduced before it appears.
- **natbib author-year** (see `CLAUDE.md`): `\citep{key}` parenthetical with a
  non-breaking space (`~\citep{...}`), `\citet{key}` textual. ASCII bib keys. Every `\cite`
  key must resolve in `UKPThesisTemplate.bib`.
- Prefer a **critical mass of quality sources** (the field's key papers) over incidental web
  pages; demonstrate engagement with the literature.

---

## 6. Smells to avoid

Recurring failure patterns and their fixes. Seeded from the supervisor's feedback on the
Introduction (2026-07-07); extend this table as new patterns are flagged.

| Smell (bad) | Rule violated | Fix (better) |
|---|---|---|
| "a small part of the network" | imprecise / vague (§2) | "a sparse subgraph of attention heads and MLPs" |
| "trading some accuracy for speed" | colloquial + vague (§2) | "at a cost to fidelity" (or quantify the trade-off) |
| "none of these methods can be taken at face value" | idiom, unscientific register (§2) | "the output of any single method cannot be treated as ground truth" |
| overloaded, casual first sentence | funnel violated (Move 1, §4) | split into one clean centrality claim, defer the rest |
| circuit as "the central object of this thesis" | wrong central object (§4 Intro) | recenter on the comparison/reliability of the methods |
| "apparent differences... confounded"; "a fair basis... the gap this thesis closes" | vague niche (Move 2, §4) | name the concrete confound (methodology-relative evaluation) and the concrete fix |
| "Two reliability questions therefore remain open, and neither..." | cataphora + repeated sentence structure (§3) | list the questions first, then characterize; vary sentence openings |
| new information buried mid-sentence | stress position wasted (§3) | move the payload to the sentence end |
| a Related-Work paragraph per paper | summary, not synthesis (§4) | organize by theme; one claim per paragraph, citations as evidence |
| interpreting results in the Results chapter | chapter-job bleed (§1, §4) | move the "why" to Discussion; keep Results factual |
| "The distribution is produced by a stack of blocks" | ELI5 register, informal noun (§2) | "Its computation is organized into blocks" |
| explaining what queries, keys and values do before saying why the split matters | over-explaining known material (§2) | give only the consequence for the section's own object: the split makes an edge record which input it enters |
| "Edges point only from earlier nodes to later ones, so the graph is a DAG", after already stating every node has an edge from each predecessor | one fact stated twice (§2) | fold the consequence into the sentence that establishes it |
| "The heads read $x_l$ in parallel, and the MLP reads the stream after they have written to it", printed above that equation | prose restating a display (§2) | delete; the equation carries it |
| "a running sum ...: it is initialized ..., and every component reads its current state and adds its own output back into it" | overlong sentence, two referents for "it" (§3) | split into two sentences, one referent per pronoun |
| "a connected pair need not interact", "a sum of this kind", "the same count at every node" | vague or unprofessional referent (§2) | name the mechanism, or delete the sentence if the point survives without it |
| "an edge into a head terminates at one of the three" | mechanics stated without the reason (§2) | read the source for why the practice exists (Wang §3.2: heads are influenced through value, key or query), then state the graph-level consequence |
| "the granularity is chosen to suit the explanation being sought" | ornate word choice (§2) | "the granularity of the graph is set by the level of detail an explanation requires" |
| a paragraph of true but unrelated facts (normalization, same-block heads, edge count) | disconnected paragraphs (§3) | move each fact beside what it qualifies; a fact with no home in the section belongs to a later chapter |
| a section-local fact that is really a later chapter's premise (quadratic edge count) | scope creep (§1) | cut it and record in `thesis_notes/` where it is owed |

---

## 7. Revision checklists

**Per chapter, confirm the chapter does its one job** (§4) and stays consistent with claims
made elsewhere (no promise unfulfilled, no result unpromised).

**Introduction rubric** (this thesis is graded on these):
- **9a Topic well-defined?** Subject and scope stated precisely and early.
- **9b Thesis well motivated?** Both the reliability concern and the comparison gap explicit.
- **9c State-of-the-art discussed?** Method families named and briefly positioned.
- **10 Research questions clear?** Crisp, answerable, non-overlapping.
- **11 Contributions clear?** Each concrete and verifiable in the thesis.

**Consistency:** every RQ ↔ a contribution ↔ a chapter; Introduction and Conclusion agree.

**Static compile check** (no TeX toolchain on the cluster; user compiles on Overleaf with
`lualatex`): cite keys resolve, `\label`/`\ref` match, no stray em-dashes
(`grep -n -- '---'`).

---

## 8. Resources and sources

- **Manchester Academic Phrasebank** (https://www.phrasebank.manchester.ac.uk/) —
  section-by-section ready-to-adapt phrases for every chapter. First stop when stuck on
  phrasing.
- Swales' CARS model — [USC writing guide](https://libguides.usc.edu/writingguide/CARS).
- Gopen & Swan, *The Science of Scientific Writing*, American Scientist 1990 —
  [PDF](https://www.gatsby.ucl.ac.uk/~pel/misc/gopen_swan.pdf) (topic/stress position).
- Literature review synthesis — [Purdue OWL](https://owl.purdue.edu/owl/research_and_citation/conducting_research/writing_a_literature_review.html),
  [Scribbr](https://www.scribbr.com/dissertation/literature-review/).
- Results vs Discussion — [Scribbr](https://www.scribbr.com/frequently-asked-questions/results-vs-discussion/),
  [Monash](https://www.monash.edu/student-academic-success/excel-at-writing/how-to-write/thesis-chapter/reporting-and-discussion-thesis-chapters).
- Methodology — [Monash](https://www.monash.edu/student-academic-success/excel-at-writing/how-to-write/thesis-chapter/methods-thesis-chapter),
  [Melbourne](https://students.unimelb.edu.au/academic-skills/graduate-research-services/writing-thesis-sections-part-2/choosing-and-justifying-your-methods).
- Abstract & overall structure — [SFU abstract guide](https://www.sfu.ca/~jcnesbit/HowToWriteAbstract.htm),
  [Paperpile thesis structure](https://paperpile.com/g/thesis-structure/).
- Register / taboo words — [Scribbr](https://www.scribbr.com/academic-writing/taboo-words/),
  [Sune Lehmann, *How to write a master's thesis*](https://sunelehmann.com/how-to-write-a-masters-thesis/).
- Background vs Related Work — [Claypool (WPI) one-pager](https://web.cs.wpi.edu/~claypool/one-pagers/background-rw.pdf)
  (the clearest CS-specific statement of the split),
  [Elphinstone (UNSW) thesis guide](https://cgi.cse.unsw.edu.au/~kevine/thesisguide.html)
  ("not a list of papers", "not a brain dump").
- CS technical writing — [Michael Ernst (UW)](https://homes.cs.washington.edu/~mernst/advice/write-technical-paper.html)
  (tree of problems and solutions; mini-introductions per section),
  [Ian Watson (Auckland), *How to Write a Master's Thesis in CS*](https://www.cs.auckland.ac.nz/~ian/msc/write)
  (build the chapter from an annotated bibliography).
- German CS-faculty norms — [TU Chemnitz CE guidelines (2025)](https://www.tu-chemnitz.de/informatik/ce/files/guidelines_2025.pdf)
  (Background only for specialized topics; assume common CS knowledge).
- Example chapter to imitate — [Colorado State, Background and Related Work](https://www.cs.colostate.edu/~christos/thesis/chp3.pdf).
