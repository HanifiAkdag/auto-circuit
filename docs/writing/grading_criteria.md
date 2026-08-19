# UKP Lab Thesis Grading Criteria

Supplied by the author on 2026-08-17. This is the rubric the thesis is graded against. Reproduced
verbatim, then followed by the points that change how we work.

---

## Thesis

Summarizes the overall quality of the Thesis

### Formatting and overall appearance

1. Is the length appropriate? 30-50 (bachelor) and 60-80 (master):
   a. Is it too short? Is it unnecessarily too long? (redundancy and rephrasing same ideas over and over)
2. Language quality:
   a. Are grammar and spelling correct?
   b. Is the vocabulary that is used appropriate for the research area and scientific writing in general?
3. Do the references include basic meta data and are consistently added (e.g. abbreviation of venue, year, etc.)?
4. Do they use "we" or "I" instead of passive voice? Is it clear through the language which parts are contributions by the student?
5. Does every chapter have a clear goal represented by the introductory paragraph?
6. Is all new terminology well-introduced?

### Quality of the presented scientific work

7. Is the problem well-defined and formalized?
8. Quality of Abstract:
   a. Does the Abstract accurately describe the thesis?
9. Quality of Introduction:
   a. Is the topic well-defined?
   b. Is the thesis well motivated?
   c. Are state-of-art methods discussed?
10. Are the research questions clear?
11. Are the contributions clear?
12. Is the related work section well-structured?
13. Quality of Related Work:
    a. Are all relevant subtopics mentioned?
    b. Is the related work described thoroughly enough?
    c. Were they described in light of what the thesis is about? (pointing out similarities and differences)
14. Quality of Plots and Figures:
    a. Are the plots and figures readable?
    b. Do they have appropriate (expressive enough) captions?
    c. Are all plots, figures, and tables valuable?
    d. Are all plots, figures, and tables mentioned in the main text?
15. Did the thesis include both qualitative and quantitative results?
16. Critical thinking: Did they interpret the results of their thesis and include a discussion on results, limitations, and future work?
17. Do they have parts in the main text that would be better suited in the appendix or vice versa?
18. Was previously given feedback addressed properly?

## Final presentation

Summarizes the overall quality of the Final Presentation

1. Are research questions & objectives & contributions clearly presented?
2. Did they demonstrate understanding of relevant literature and problem scope? (background & context slides)
3. Are the results well-presented and critically discussed?
4. Was the contributed methodology well-explained?
5. Did the student manage to answer or comment on all the questions asked?
6. Did the student address the feedback provided on the presentation? (prior to the presentation)
7. Did the student organize their time well?
8. Was the presentation easy to follow? (confident, well-paced delivery)
9. Were the slides easy to read? (visual aids)
10. Quality of conclusion and future work slides?

---

## What this changes

### Open question that affects every chapter (criterion 4)

The rubric asks for **"we" or "I" instead of passive voice**, and asks whether the language makes
the student's own contributions identifiable. The write-up currently uses neither: it says
"this thesis integrates", "this thesis reports". That is active voice with an inanimate subject,
so it does not fail the passive-voice half of the criterion, but it never marks authorship either.
**Not resolved.** Decide before drafting more chapters, because retrofitting it across a finished
document is expensive and error-prone. The three options are: keep "this thesis" throughout; use
"we" throughout, which is the NLP convention; or use "we" only where a contribution is being
claimed, and impersonal constructions for background. The third matches what the criterion is
actually asking for.

### Already binding, restated

- **Criterion 2b (vocabulary).** Academic register is a grading criterion, not a preference. No
  colloquial verbs or nouns. See the register list below.
- **Criterion 6 (terminology).** Every new term defined at first use. Already the chapter rule.
- **Criterion 5 (chapter goal).** Every chapter opens with a paragraph stating its goal. This does
  not license meta-sentences inside sections; the goal statement belongs in the chapter opening.
- **Criterion 3 (references).** Entries need consistent metadata: venue, year, and so on. The bib
  currently has entries carrying only title/author/year. The author has decided `gpt2` stays in the
  form OpenAI's own repo publishes, which is that minimal form; that is a deliberate exception, not
  an oversight.
- **Criterion 14 (figures).** Every figure must be readable, carry an expressive caption, be
  referenced from the main text, and be worth its space. A figure that is not clearly valuable is
  better cut than kept. The §2.1 schematic was cut on exactly this ground.
- **Criterion 1 (length).** 60-80 pages for a master's thesis, with redundancy explicitly penalised.
- **Criterion 17 (appendix).** Material better suited to an appendix is a graded fault. Relevant to
  the port-faithfulness audits and the full result grids.
- **Criterion 18 (feedback).** Previously given feedback must be visibly addressed.

### Register: prefer the right column

The rubric's "vocabulary appropriate for the research area and scientific writing in general" rules
out the informal constructions that keep appearing in drafts.

| Avoid | Use |
|---|---|
| hand activations to one another | exchange activations |
| a shared running state | a shared representation |
| told apart | individually identifiable |
| what counts as a node | the node set |
| the layer stack it came from | the layered architecture from which it is derived |
| holds roughly 32,000 | contains approximately 32,000 |
| is built into | rests on |
| set aside | disregarded |
| a limit the authors state themselves | a limitation the authors state explicitly |
| work that follows this reading | research pursuing this view |
| show (of a paper's result) | demonstrate, report, establish |
| a graph rather than a chain | induces a graph structure |

The test: would this phrasing appear in an ACL or NeurIPS paper? If it reads as spoken explanation,
rewrite it. This does not license inflated vocabulary or long sentences; the standing rules on short
plain sentences and no invented jargon still hold, and criterion 1 penalises padding.

**A register fix is not a clarity fix, and substituting a formal word for a colloquial one can
disguise a sentence that never said anything.** "The terms of the sum can be told apart" became
"the terms of the sum are individually identifiable", which sounds like scientific writing and is
equally empty: it never states what the terms are or why separating them matters. The repair was to
say the actual fact, that each term is the output of exactly one earlier component, so a
component's input can be attributed to its sources term by term. When rewriting for register, check
afterwards that each sentence still makes a claim a reader could disagree with.
