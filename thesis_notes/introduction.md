# Introduction

### Canonical gap / contributions / RQs — 2026-07-09

- **Gap** (three parts, one per sub-RQ; tooling is the enabler, not the payload):
  1. **Ranking robustness:** EP and DiscoGP were each shown superior only under their own protocol (EP: resample + full-vocab KL; DiscoGP: zero ablation + label accuracy) and the proposing method wins in each paper. No study varies the discovery/eval ablation as an experimental variable.
  2. **Stability:** per-run circuit variance quantified only for attribution methods; open for mask-optimization methods (Méloux).
  3. **Spuriousness:** no random-weight null test exists for EP/DiscoGP.
- **Contributions** (mirror the gap):
  1. EP + DiscoGP integrated into AutoCircuit; discovery/eval ablation becomes independently settable for these methods.
  2. Crossed discovery×eval ablation comparison of all four algorithms (IOI/GT/Docstring): do published orderings persist outside their origin protocols?
  3. Reliability characterization: run-to-run stability + random-weight null.
- **RQs:** central RQ = comparison under a common framework with identical ablations/metrics. Sub-RQ1 integration; Sub-RQ2 faithfulness + ranking persistence under ablation change; Sub-RQ3 reliability (stability, null).
- **Never claim** EP-vs-DiscoGP was "never compared" (see next entry).

### Sub-RQ1 is phrased so the audience answers "just implement them" — 2026-07-21

- **Context:** Midterm presentation review; supervisor read the research-questions slide, which carries the `introduction.tex` wording of Sub-RQ 1 verbatim.
- **Observation:** "How can Edge Pruning and DiscoGP be integrated into the edge-centric architecture of the AutoCircuit library...?" reads as a task, not a research question. The listener's instinctive answer is "just implement them", so the difficulty is never conveyed. The supervisor's phrasing: *"the answer to how is just implement them, because that's what the audience thinks; you never explain what is the difficulty of doing it."*
- **Significance:** The non-trivial content of Sub-RQ 1 is two things the current wording hides: (1) **port faithfulness** — each method must provably reproduce the original's behaviour inside a shared non-invasive wrapper, where DiscoGP's original does not wrap a model at all but rebuilds the transformer blocks to thread its own `prev_head_idx` residual axis, and its mask convention is inverted (1 = keep vs AutoCircuit's 1 = ablate, a silent correctness bug if missed); (2) **decoupling the ablation** — every original hardwires its own ablation, so lifting it into a free parameter is what makes the crossed discovery×evaluation matrix possible at all. Both were verified by line-by-line audits against the original repositories at fixed commits. Applies to `introduction.tex` §Research Questions, not only the slide: the written Sub-RQ 1 should foreground faithfulness and ablation-decoupling rather than "how can they be integrated".
- **Code pointer / Run:** `presentation/outline.md` slides 10–11; `docs/related_work/edge_pruning.md` and `disco_gp_sheaf_discovery.md` §Faithfulness of the AutoCircuit Port.

### EP Appendix D examines seed stability — soften the Pillar-2 gap claim — 2026-07-13

- **Context:** Midterm presentation prep; re-verified both papers' evaluation protocols in the primary sources (arXiv 2406.16778 HTML, 2407.03779 HTML, `disco_gp/evaluation.py`) after the user asked for citation checks.
- **Observation:** EP's Appendix D runs **12 seeds** at one representative sparsity per task (IOI 97.5%, GT 99.0%, GP 97.0%): resulting sparsity and KL are "remarkably consistent", but **pairwise edge IoU is only 0.5–0.7** ("high, but far from 1") over the 66 pairs; the authors attribute this to redundant components and defer the definitional question to future work.
  - Protocols re-confirmed: EP evaluates all methods (EP/ACDC/EAP) under **interchange ablation + full-vocab KL** (zero ablation only in the Tracr check); DiscoGP evaluates all methods (incl. EP) under **zero ablation + sheaf accuracy + label-space KL** (2-class softmax over (good,bad) logits; answer-index softmax for PARAREL) — verified in `compute_faith_loss_binary_label` / `_multi_label`.
- **Significance:** The Pillar-2 gap must be stated as "stability never *compared across methods or protocols*", never as "never examined for mask-optimization methods" (EP examined its own; DiscoGP averages 5 runs' performance without reporting circuit overlap). `introduction.tex:17` ("quantified mainly for attribution-based methods such as EAP") survives on the word "mainly" but should cite/acknowledge EP Appendix D when Related Work is written. Bonus: EP's IoU 0.5–0.7 is a published anchor to sanity-check Pillar 2's Jaccard numbers against.
- **Code pointer / Run:** `docs/related_work/edge_pruning.md` §Evaluation protocol (new); `presentation/outline.md` slide 12 say-notes.

### DiscoGP camera-ready DOES compare against EP — 2026-07-09

- **Context:** User challenged the "EP-vs-DiscoGP never compared" gap claim; verified against the EMNLP camera-ready (arXiv 2407.03779).
- **Observation:** Table 2 runs EP (original implementation, Appendix C), ACDC, and EAP as baselines on BLiMP/IOI/PARAREL, all scored under zero ablation with label-space accuracy/KL and favorable sparsity budgets for ACDC/EAP. EP nearly matches DiscoGP on its home criterion (IOI: 100%/KL 0.032 vs 100%/0.020); ACDC/EAP collapse (~52/58%).
- **Significance:** Grounds the "single-protocol, proposer wins" gap framing (canonical entry above); Introduction rewritten accordingly same day.
- **Code pointer / Run:** `docs/related_work/disco_gp_sheaf_discovery.md` §Comparisons (corrected); `thesis/chapters/introduction.tex`; CLAUDE.md index line; memories `project_context` + `edge_pruning_method_and_vs_discogp`.

### Method taxonomy corrected: EAP is gradient-based; EP/DiscoGP renamed "optimization-based" — 2026-07-09

- **Context:** Supervisor feedback on the Introduction's state-of-the-art paragraph.
- **Observation:** The old three-family taxonomy (search-based / attribution-based / gradient-based) was factually wrong on two counts. (1) **EAP is itself gradient-based**: attribution patching uses gradients to approximate the ablation effect in one backward pass, so "attribution" and "gradient" are not disjoint families. (2) **ACDC is greedy, not "search-based"**: it prunes edges one at a time against a threshold.
- **Significance:** Forces a terminology change across the thesis. The families are now distinguished by *how edge importance is estimated*: **greedy pruning** (ACDC), **gradient-based attribution** (EAP), **gradient-based mask optimization** (EP, DiscoGP). Because EP/DiscoGP can no longer be called "the gradient-based methods" without ambiguity, they are called **optimization-based** throughout, including in the central RQ, Problem Statement, and Contributions. `CLAUDE.md`, this file's older entries, and the memory files still say "gradient-based" for EP/DiscoGP and are not yet aligned.
- **Code pointer / Run:** `thesis/chapters/introduction.tex` (Motivation ¶3, Problem Statement, RQs, Contributions).

### Supervisor rejected Intro draft ("4/10"); scientific-register overhaul needed — 2026-07-07

- **Context:** Supervisor reviewed the drafted Introduction (`thesis/chapters/introduction.tex`) and rated it "4/10 at best." All five complaints target scientific register/precision, not content.
- **Observation:** Concrete flags: (1) informal/vague phrasing ("a small part of the network", "trading some accuracy", "none can be taken at face value"); (2) first sentence overloaded and too casual; (3) the circuit is wrongly framed as "the central object of this thesis" (the real object is the controlled *comparison of discovery methods*; the circuit is their artifact); (4) vague generic gap sentences ("apparent differences... confounded", "fair basis... the gap this thesis closes"); (5) "Two reliability questions therefore remain open, and neither..." uses cataphora ("Two" before the list) and repeats the prior sentence's structure.
- **Significance:** These map cleanly onto Swales' CARS model (intro structure) and standard scientific-register rules (precision, active claims, no idioms, one idea/sentence). Captured durably so they apply to all remaining chapters, not just a one-off intro fix.
- **Code pointer / Run:** Guidance distilled to `docs/writing/scientific_writing_guide.md` (CARS + register + smells table + rubric); operationalized as the `writing-thesis-prose` skill (auto-invokes on thesis prose). Intro rewrite is a deferred separate pass.

### Introduction chapter drafted; framing decisions — 2026-06-19

- **Context:** Writing the LaTeX Introduction (`thesis/UKPThesisTemplate.tex`).
- **Observation / decisions:**
  - **Spine:** the *comparison* of circuit-discovery algorithms is the payload; *integration* of EP/DiscoGP into AutoCircuit is the enabling contribution. Intro funnels: opaque LLMs → circuits → discovery methods (search/attribution/gradient) → results are hard to compare (methodology-relativity + non-identifiability) → this thesis unifies + compares.
  - **Murano dropped** (supervisor decision). *(Superseded by the next entry: the final structure is one primary RQ + three sub-RQs — Integration / Faithfulness / Reliability — not a "two-RQ" split.)*
  - **DiscoGP described only in its paper-faithful form** (answer-margin CE); the `argmax_ce` relaxation is omitted from the write-up.
  - Intro is **motivation/contribution-focused per the grading rubric** (9a topic, 9b motivation, 9c SOTA methods named-not-dissected, 10 RQs, 11 contributions). Algorithm internals → Background/Methodology; per-paper findings → Related Work. Citation budget kept minimal (ACDC, EAP, EP, DiscoGP, Miller 2024, Méloux 2025).
- **Significance:** Locks the thesis's top-level narrative and RQ structure; downstream chapters must match (each RQ ↔ a contribution ↔ a chapter).
- **Code pointer:** `thesis/UKPThesisTemplate.tex` (Introduction); blueprint at `~/.claude/plans/wiggly-napping-sky.md`.

### RQ structure finalized + AutoCircuit already has gradient methods (novelty scoping) — 2026-06-19

- **Context:** Revising the exposé research questions for the thesis intro after dropping Murano and DiscoGP weight-pruning; user flagged that AutoCircuit may already contain gradient-based pruning.
- **Observation:**
  - Verified in `auto_circuit/prune_algos/prune_algos.py` + `subnetwork_probing.py`: AutoCircuit **already ships gradient/optimization-based mask methods** — Subnetwork Edge Probing (SEP), Subnetwork/Circuit Tree Probing (STP/TP), Circuit Probing. `subnetwork_probing_prune_scores` optimizes edge masks via Adam + hard-concrete L0 to maximize faithfulness. SP is the literal basis for Edge Pruning.
  - **RQ restructured to one primary RQ + three sub-RQs** (Integration / Faithfulness / Reliability); the old Murano sub-RQ removed; primary RQ reworded from "unified within a standardized interpretability framework" (Murano allusion) to "integrated into a common evaluation framework," and comparison-forward to match the agreed spine.
- **Significance:** The contribution is **not** "bringing gradient-based discovery to AutoCircuit" (false — SP/TP exist) but integrating the two *specific recent published* algorithms EP (Bhaskar 2024) and DiscoGP (Yu 2025) so they can be compared on equal footing against ACDC/EAP/SP. Any "first gradient method in a standard framework" claim must be avoided. EP-as-descendant-of-SP and SP as an additional baseline are good Related-Work/Background points.
- **Code pointer:** `auto_circuit/prune_algos/subnetwork_probing.py`; intro §3 in `thesis/UKPThesisTemplate.tex`.

### Dead-salmon null-test gap corrected after reading Méloux 2025 in full — 2026-06-19

- **Context:** Verifying the Pillar-3 (null test) framing against the actual Dead Salmon paper (Méloux et al. 2025, arXiv 2512.18792).
- **Observation:** It is a position paper. It *prescribes* a randomized-weights null test (Appendix A eq. 4 Monte-Carlo p-value; Table 1 assigns the circuit-discovery row the error `KL(P_full‖P_circuit)`) but *empirically* runs it on **probing only** (sentiment / POS / world-models). For circuit discovery it cites toy-model artifacts (`melouxeverything`) and a **prior circuit-level hypothesis test, Shi et al. 2024** ("Hypothesis Testing the Circuit Hypothesis").
- **Significance:** "First null test on circuit discovery" would be an overclaim. The defensible gap is *instantiating Méloux's prescribed test for the gradient-based optimizers EP/DiscoGP (vs ACDC/EAP) inside the unified AutoCircuit harness on real GPT-2 tasks* — i.e. completing the circuit-discovery row of their Table 1. Cite Shi 2024 to differentiate. The thesis's KL faithfulness spine equals the discrepancy measure Méloux assigns to circuit discovery, so Pillars 1 and 3 sit inside their statistical-causal framework. **TODO:** locate the exact Shi 2024 reference and add to the bib before writing Methodology/Related Work.
- **Code pointer:** `docs/related_work/dead_salmon_interpretability.md` (summary may need this nuance added).

### Intro revised on supervisor feedback; three Méloux papers → three reliability axes — 2026-06-21

- **Context:** Reworking `thesis/chapters/introduction.tex` after supervisor feedback.
- **Observation:**
  - Circuit defined *loosely*, precise definition deferred to Background ("what counts as a circuit" is contested).
  - Discovery is not inherently automated: early work was manual patching; \citet{acdc} gave the *first* automated algorithm. Framed historically.
  - Models generalized: IOI/GT = `gpt2-small`, Docstring = `attn-only-4l` (`tasks.py:324,334`). Not "GPT-2 Small only."
  - Three Méloux papers now each anchor one axis: `meloux2025everything` = non-identifiability (Motivation), `meloux2025variance` = stability, `meloux2025deadsalmons` = "genuine learned structure vs spurious artifact" via random-weight null (both Sub-RQ 3). Previously only deadsalmons cited, for the wrong concept. NB: "non-spuriousness" was a coined draft label, replaced in prose with plain wording — the paper's own terms are "spurious" (adj.), "false positives", "dead salmon artifacts"; do not reuse the coined noun.
  - Problem-statement para 2 rewritten into a clean two-axis reliability gap (stability + genuine-vs-spurious structure; the coined noun "non-spuriousness" was dropped, see note above).
  - **Variance paper names the stability gap explicitly** (verified §6.3 Limitations, arXiv 2510.00845): *"our circuit discovery analysis focuses on the EAP family and its variants"*; non-EAP methods it lists (NAP/HAP/RelP) all rely on CMA; *"Characterizing their stability is an important direction for future work."* EP/DiscoGP are gradient-based optimization (neither EAP nor CMA) → their stability is unmeasured → direct citable support for Sub-RQ 3. Caveat: do **not** claim they are more stable a priori (paper argues step-(c) selection instability may still apply); it is untested, not known.
- **Significance:** Fixes flagged factual/framing errors; gives a Méloux-paper → reliability-axis → Pillar mapping the later chapters must mirror.
- **Code pointer:** `thesis/chapters/introduction.tex` §Motivation, §Problem Statement, Sub-RQ 3.
