# Supervisor / co-author review — `paperB_ecir` v1 (2026-09-08)

Read: the 16-page PDF (body ends p.12, references p.13–16 — inside the limit), `BRIEF.md`,
`ECIR_STYLE_GUIDE.md`, `chapterB_v2.tex`.

## 1. Verdict

**Submittable as an ECIR full paper after a focused revision; not submittable as it stands.** My
predicted outcome for the current PDF is **Borderline, with weak-reject risk**; with the "must" list
below done, **Weak accept**, and I think it can be pushed to Accept, because the evidence base is
unusual for this venue.

The upside is real: twenty-one selection rules re-implemented under one contract, 10,000 held-out
users, no sampled metrics, no simulator, unanswerable questions charged, a comparator that is
computed rather than chosen, and a one-variable ablation with two shuffle controls. Nothing in the
ECIR corpus does this. Style compliance is good — enumerated RQs, results sections named after them,
explicit "Answer to RQx" sentences, booktabs, statistics defined once in a caption.

Why it is borderline today. Three reviewer reflexes, in order of danger:

1. **"You may simply have implemented their methods badly."** The answer exists in the chapter
   (native validation gates, the coverage inventory, the Hu & Yu ML-1M replication, the declared EDDI
   failure) and almost all of it was cut. What survives is a table group header reading *"published
   methods, validated natively and ported"* — an assertion. That omission is, I think, the difference
   between weak accept and weak reject.
2. **"One dataset, one unpublished recommender of your own."** Worse: §4 itself reports a consumer on
   which the RQ1 ordering *reverses* (*"with Golbandi's leaf model, Thompson sampling beats the static
   order on full accuracy at every budget"*), and then the headline CASPER result is never checked on
   a second consumer. A reviewer will notice that asymmetry immediately.
3. **"The named contribution is dominated by your own control."** §6 says the bank change is worth
   +0.01430 with no conditioning against CASPER's +0.01035. Honest — but a reviewer reading "We
   introduce CASPER" in the abstract will ask why the paper is named after the weaker of the two fixes.

## 2. The story

Objective mismatch, not an adaptivity null, is the right frame and the strongest one the chapter
supports. Keep the disclaimer *"a fixed order is an adaptive policy that ignores its inputs"*; keep the
Golbandi-2011 concession — it is what makes the scoping credible rather than defensive. Two reframings:

**(a) Present two priced fixes, not one method plus a caveat.** The prescription is *"build the
selector around the quantity that will be reported and around nothing else"*, and the paper has two
independent demonstrations of it on one ruler: the objective (+0.01035) and the action space
(+0.01430). Say that in the abstract and in the contributions rather than letting the bank result
arrive in §6 as an objection to be survived. It converts the paper's weakest moment into its second
finding.

**(b) Soften the branding.** *"We introduce CASPER … which puts the measured metric itself into the
split criterion"* invites "the contribution is a name for changing one scalar". §6 handles this well
(*"The name states what is kept and what is changed"*); mirror it in the abstract: "a measured-metric
interview tree (CASPER)".

Strategic note: if the fidelity evidence genuinely cannot fit, the **reproducibility track** is a
natural home for this paper at the same deadline. My recommendation is still the full track — the
split-criterion ablation is a method result — but decide that in week one, not on 4 October.

## 3. What is missing that a reviewer will want, and what pays for it

**Must add**

- **A replication-fidelity table** (chapter `tab:coverage`, compressed): method / source / validated
  natively / compared against its own paper's static baseline. Four columns, ~14 rows, ~0.4 page.
  This is the answer to reflex 1 and nothing else substitutes for it.
- **Tuning protocol for the ported arms.** The style guide's §5.4 is the most reliably demanded item
  at ECIR and the paper says nothing. Worse, the untuned exploration weight is confessed only in
  Limitations (*"one free constant, the exploration weight, was never tuned"*), a page after "every
  published rule loses". State the protocol in §3 and put that admission beside the claim.
- **A consumer-robustness cell for the tree.** The interviews are banked; re-scoring the tree and its
  static comparator through RecVAE is a re-score, not a new campaign. One row, one sentence. If it is
  unfavourable, report it — but do not ship an asymmetry where RQ1 gets five consumers and the
  headline gets one.
- **Restore the EDDI honesty paragraph** (two sentences). The dagger says the row is *"a reduced
  latent-only acquisition failing native validation"*; a reviewer will ask why a self-declared failed
  replication is counted inside "twenty-one rules".
- **The multiplicity sentence** from the chapter's reproducibility section. Several hundred paired
  intervals, no correction, no statement.
- **The depth sentence**: depth 5 is the min-leaf rule, and *no node stopped for want of a
  positive-gain split*. One sentence, and it pre-empts "you tuned the depth".

**Should add**

- The chapter's *"the ground truth must not depend on what was asked"* result, now one clause in §3.
  It is the most portable finding in the chapter and this is an evaluation-minded venue.
- The dispersion guardrail that was dropped: the 4,431 request users are a *different population*, so
  the contrast is between regimes, not a within-user effect.

**Pay for it by cutting**

- The genre-conditioning paragraph in §4 (*"One arm clears the bar…"*, ~0.4 page). It is an upper
  bound with free side information, §6 removes the same confound more cleanly, and it blurs RQ1's
  answer. Two sentences and a forward pointer suffice.
- ~15 lines from §2; the RL and LLM paragraphs are longer than their role in the argument.

## 4. Clarity — passages an ECIR reader will stumble on

1. Abstract: *"the three published split criteria forgo 66–89% of what a random choice forgoes"* —
   reads as praise on first pass. → "their regret is only 11–34% smaller than that of picking at
   random".
2. *"twenty-one published selection rules"* / *"Twelve of the twenty-three arms are value-free"* /
   *"six of the thirteen published methods"* — three denominators, none reconciled, and Table 1 shows
   fewer rows than any of them. → define rules vs arms vs published methods once, in §3.
3. *"the greedy prefix-optimal question order"* — used as if standard. → "every prefix of the sequence
   is the best sequence of that length the search can build".
4. *"138 cells in all, the combined and concepts-only orders lead and separate in all 138"* — not
   recountable from Table 1. → footnote the arithmetic (14 adaptive arms × 5 budgets × 2 axes, less
   the tree's two missing K=16 cells).
5. *"The strongest items-only static order leads the best adaptive selector by .0281⋆… twice that
   selector's deficit against the heuristic."* Which selector? And "leads by +.0281" collides with the
   chapter's "trails by +.0281". → name the arm and fix the sign convention globally.
6. Table 1's *"Answer-free popularity reference .1626/.0262"* against §3's *"the empty interview on
   the test cohort reads 0.1586/0.0282"* — two no-information baselines, unexplained. → one clause
   distinguishing the popularity row from the model's own empty-interview prediction.
7. *"each a whitened centroid of its member films in the latent, answered with a signed behavioural
   affinity read from the user's watching"* → "each represented by the average latent position of the
   films tagged with it, and answered by a signed statistic computed from what the user has watched".
8. *"folding as the encoder's unseen-exposure or refusal token"* — tokens never introduced. → "the
   encoder receives a designated 'not seen' or 'no answer' input in place of a value".
9. *"all six static arms driven through the sequential loop reproduce the batch implementation exactly
   over 30 cells on all 10,000 users"* — the reader cannot tell what this gate is for. → "as an
   implementation check, the sequential harness reproduces the batch scoring of the six static arms
   exactly".
10. §5's only equation is inline, with ⊕ undefined and V glossed mid-sentence. → display it; define z,
    a, ⊕ and V; gloss "perplexity of the argmax distribution" as "the effective number of distinct
    optimal questions" at first use rather than after.
11. *"Placing 4,431 users by a real free-text request instead spreads the optimum to 442 questions"* —
    a second corpus appears with no provenance. → one sentence of description plus the
    different-population guardrail, or cut the row.
12. *"no ground truth is read at selection time, since the criterion is a cohort-level quantity on the
    validation cohort"* → say it plainly: "the criterion reads validation users' held-out ratings and
    never those of the user being scored". As written it invites a leakage question.
13. *"resumed from the tree root's own search heap"* → "the static search is initialised from the
    candidate scores computed at the tree's root, so its first pick is necessarily the root question".
14. *"the eighth excludes zero against it, since at two questions the tree is worse on the tail"* →
    "the eighth excludes zero in the static order's favour: at two questions the tree is worse on the
    tail."
15. *"a concept answered by 74.2% of the cohort, where the best item split would send 7,045…"* →
    "whereas".
16. *"16.6% of the static order's own lift over asking nothing"* — denominator never stated. → add
    "(the static order gains 0.0862 over asking nothing)".
17. *"concepts win the tail and tie on full (−0.00003 [−0.00315, +0.00310] full…)"* — state that the
    quantity is concepts minus items.
18. *"All 56 non-nested intervals in the six-arm matrix"* — "non-nested" and "six-arm matrix" each
    need a half-clause.
19. Conclusion: *"About 14% of cold users answer nothing across sixteen charged questions"* — a number
    that appears **nowhere else in the paper**. Support it in §6 or delete the sentence.
20. Abstract *"Every published rule loses"* vs conclusion *"Every published selector we could
    replicate loses"*. Choose the hedged form and use it in both.

## 5. Anonymity

Author, running head, institute and PDF metadata are clean; the code footnote is redacted; the
companion instrument is cited as an anonymous technical report. Two residual risks:

- **"CASPER" is a unique string.** If it appears in any public repository, preprint, thesis deposit,
  slide deck or talk before notification, the acronym alone deanonymises the submission. Check, and if
  it is exposed, delay the deposit or rename in the submitted version.
- The instrument paragraph plus reference [3] is identifying if the companion is on arXiv
  non-anonymously. Confirm before 5 October.

## 6. Presentation

Body 12.0 pp; floats sit near their references; captions are self-contained; RQ answers are explicit;
the abstract carries five numbers and repeats them in the conclusion. Improvements: underline
second-best per the corpus convention; add a "Δ vs greedy items-only" column to Table 1 so two prose
sentences can go; state in Table 1's caption why Golbandi has no k=16 cell and why the
concepts/combined cells tie at k=1–2 (equal by nesting, not by adjudication); Fig. 2's eight-arm
legend is at the limit of greyscale legibility — check a print.

## 7. Prioritised to-do

**Must** — (1) strip the editorial notes from the bibliography (§8); (2) fidelity/coverage table;
(3) tuning statement, with the untuned-constant admission moved next to the claim; (4) consumer
robustness for the tree, or an explicit limitation; (5) restore the EDDI validation paragraph;
(6) fix the unsupported 14% in the conclusion; (7) reconcile 21 / 23 / 13; (8) multiplicity sentence;
(9) depth sentence; (10) anonymity check on "CASPER".

**Should** — reframe abstract and contributions as two priced fixes; cut the genre paragraph and
~15 lines of §2; restore the "ground truth must not depend on what was asked" paragraph and the
dispersion population guardrail; clarity items 1–20.

**Could** — underline second-best; Δ column in Table 1; a one-line dataset row (users / items /
interactions / sparsity); an online-appendix footnote for the omitted battery rows.

## 8. What I would refuse to submit as written

- **The printed bibliography contains internal working notes.** Reference [14]: *"Title verified; FULL
  AUTHOR LIST UNVERIFIED – replace the and others before print."* Reference [9]: *"Full text closed on
  every route tried; mechanism reconstructed from … patent US8510248B2 …"*. Reference [27]: a
  three-line provenance note including *"The earlier [S] record was wrong twice"*. Reference [29]:
  *"arXiv preprint; not peer reviewed"*. These are in the PDF the reviewers receive. Non-negotiable.
- **The absolute claim "Every published rule loses"** while five arms are omitted from Table 1 for
  space, one counted arm is a declared failed replication, and no tuning protocol is stated. Restore
  the fidelity evidence and hedge the sentence, or I will not sign it.
- **The 14% figure in the conclusion**, which nothing in the paper supports.
