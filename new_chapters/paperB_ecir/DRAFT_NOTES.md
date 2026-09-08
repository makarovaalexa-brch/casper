# paperB_ecir — draft notes (2026-09-08)

Build: `tectonic paperB_ecir.tex` in this directory. Output 16 pages total, **body = 12.0 pages**
(References begin at the top of p.13). No overfull hboxes. 41 references, all cited, all present in
`refs.bib` (copied verbatim from `../references.bib` and `../chapterA_v2/chapterA_v2_new_refs.bib`,
plus one hand-written `anon_instrument` technical-report entry for double-blind).

## Page budget actually used

| section | starts | pages used |
|---|---|---|
| Title, abstract, keywords | p.1 | 0.45 |
| 1 Introduction (RQ1–RQ3 + 3 contribution bullets) | p.1 | 1.35 |
| 2 Related Work (3 run-in paragraphs) | p.2 | 1.25 |
| 3 Measurement Contract (protocol, instrument, answers, comparator) | p.3 | 1.55 + Fig. 1 |
| 4 RQ1: published selectors vs metric-optimised static order | p.5 | 3.4 incl. Table 1, Fig. 2 |
| 5 RQ2: how far the surrogates sit from the metric | p.8 | 1.3 incl. Table 2 |
| 6 RQ3: putting the reported metric in the split | p.9 | 3.0 incl. Tables 3–4 |
| 7 Limitations | p.12 | 0.4 |
| 8 Conclusion | p.12 | 0.4 |
| References (not counted) | p.13 | 3.5 |

Float placement: Fig. 1 + Fig. 2 on p.5, Table 1 on p.6, Table 2 on p.8, Table 3 on p.9,
Table 4 on p.10. Captions above tables, below figures, all self-contained. Preamble raises
`\topfraction`/`\floatpagefraction` so no float is deferred past its reference and no float page is
produced; `\emergencystretch=1.2em` removes the last overfull lines.

## Chapter floats included

| paper float | chapter source | note |
|---|---|---|
| Fig. 1 measurement contract | `fig:contract` (chapter ll. 196–222) | tikz reused, nodes tightened, caption cut to 3 lines |
| Fig. 2 full NDCG@10 vs budget | `fig:budgetcurves` (chapter ll. ~690–723) | pgfplots reused; legend trimmed from 10 to 8 arms (dropped `greedy concepts-only` and `latent-only info-gain`), moved below the axis, height 4.1 cm |
| Table 1 battery | `tab:battery` | rows kept in the chapter's grouping; **five rows dropped for space** (see below) |
| Table 2 criterion regret | `tab:criterionregret` | verbatim |
| Table 3 tree gains per K | `tab:tree` | verbatim |
| Table 4 split-criterion ablation | `tab:splitcrit` | verbatim |

## Chapter floats omitted

- `tab:greedy` (greedy static sequences): its numbers survive inside Table 1's three greedy rows and
  in the prose intercept `0.1586/0.0282`.
- `tab:coverage` (battery coverage ✔/✘ inventory): no room; the "we ran them rather than cite them"
  claim carries it in prose.
- `tab:prefixes` (distinct interviews): folded into one prose sentence (10,000 distinct interviews by
  Q4; 5,739 for GLM-UCB at Q16).
- `tab:dispersion`: folded into prose in §5 (all six rows' numbers are quoted).
- `tab:bankdecomp`: folded into prose in §6 (all three rows' gains, intervals and shares quoted).
- `tab:firstthree` (first three questions per genre order): dropped with most of the genre section.

## Row removals from Table 1 (the only place data was cut rather than prose)

Cut, only after ~1,600 words of prose had already been cut and the paper was still over budget:
`pure entropy`, `random (bank)`, `ConLinUCB-BS-static`, `DPE-ILD (WSDM'21)`, `EDDI three-outcome`
(five rows; the caption says "Five further degenerate or duplicate arms are omitted for space", and
the count was checked against the surviving table). None of the removed rows is referred to anywhere
in the prose. The `$^\ddagger$` marker was removed from the caption with the EDDI three-outcome row;
`$^\S$` (BALD, A-optimal) and `$^\dagger$` (latent-only info-gain) are still used by surviving rows.
`ConUCB (WWW'20)` and `A-optimal (WWW'15)` were kept.

## Places where I was not certain a number was copied correctly

1. **"138 cells in all"** (paper §4) — chapter l. ~760: "the table's thirteen adaptive rows plus the
   three-outcome sub-variant discussed in Section~\ref{sec:answerability}, $138$ cells in all". The
   claim is about the *whole battery*, but the paper's Table 1 now shows fewer adaptive rows than
   the battery contains, so a reader cannot recount 138 from the table. The sentence is still true;
   consider a footnote saying the count is over the full battery.
2. **"Twelve of the twenty-three arms are value-free"** — chapter says "twelve of twenty-three
   **rows**" (l. ~505). I changed "rows" to "arms" because the paper's table no longer has 23 rows.
   Same number, reworded.
3. **`+.0281` "trails by"** — chapter (l. ~600) writes "trails the strongest items-only static order
   by $+.0281$ $[+.0247,+.0314]$ full". Kept the chapter's sign convention (positive = static ahead),
   which reads oddly next to "trails". Worth a reviewer-proof rewording.
4. **Best cold-state concept gain: `+0.0238` vs `+0.0283`.** The chapter gives `+0.0238` against
   `+0.0019` at *cold* states (§act4) and `+0.0283` at *cold states after two openers* (§dispersion).
   Both are used, each with its own state named. Verify the two states are as I labelled them
   (chapter ll. ~1130 and ~1490).
5. **"eight stopped on size, seven on shape"** — chapter l. ~1400 gives five at $n<400$ plus three at
   $400\le n<600$. This sentence was cut from the final draft; no longer in the paper.
6. **Table 3 caption absolutes** — `tree 0.26936/0.13074, static 0.25900/0.12370, no questions
   0.15855/0.02819` copied from `tab:tree`'s caption. Note the static value there (`0.25900`) differs
   from Table 1's greedy-combined `k=16` cell (`.2590`) only in trailing zeros; they are the same arm
   measured under the tree's own comparator build, which the chapter treats as one number.
7. **Instrument paragraph** — `0.3496/0.2491`, RecVAE `0.3540/0.2497`, `1.2\%` relative shortfall,
   `1,031` genome tags: all from chapterA_v2 ll. 93/863/101. The four requirements are paraphrased in
   one clause rather than quoted; R1–R4 are not named as such (double-blind, and the labels mean
   nothing without Chapter A).

## Claims softened, reworded, or not fitted

- **Genre-conditioned greedy (chapter §groupgreedy)** is reduced to one paragraph. Dropped: the
  1.80× validation-to-test shrinkage across all seventeen groups (kept as "every group shrank"),
  the Horror/*Shrek 2* and Fantasy examples, the congruence-lift analysis, `all-groups` numbers.
- **The consumer flip (§flip)** is one sentence; the per-budget tail cells and the latent-only
  info-gain split at eight questions are gone. The chapter's mechanism explanation (spread) is kept.
- **The candidate-pool / target-set discussion** (chapter's "a published adaptive-degradation result
  is manufactured by it") is reduced to one clause in §3 ("Candidate formation is load-bearing and is
  held fixed"). The external preprint's ablation is not reported at all.
- **Answerability section (§act4)** is a single paragraph inside §6 rather than its own section.
- **Depth-five stopping-rule analysis (§depth)** is cut entirely, including the claim that no node
  stopped for want of a positive-gain split.
- **EDDI validation failure** survives only as the `$^\dagger$` caption marker; the paragraph
  explaining that the row measures our reduction rather than the published criterion is gone. This is
  the most significant honesty item lost to the page limit and should be restored as a footnote if
  any space is recovered.
- **Two- vs three-outcome response model** result is entirely cut (row and prose).
- **Hu & Yu native replication on ML-1M** (MAP, sampled negatives) is cut.
- **The requests/openers/policy section** is excluded per the brief; it appears only as the last
  clause of the conclusion ("reaching them with volunteered openings is ongoing work").
- **Multiplicity paragraph** from the chapter's reproducibility section is not included; the paper
  reports intervals per cell without a family-wise correction and does not say so.
- Never written: "adaptivity does not pay". The framing throughout is objective mismatch, with the
  measured-metric tree beating the static order (`+0.01035 [+0.00739,+0.01323]` full,
  `+0.00705 [+0.00394,+0.01012]` tail, K=16) stated in the abstract, §6 and the conclusion.

## Style compliance

RQ1–RQ3 enumerated in §1 and each results section named after its RQ, each ending in an explicit
"Answer to RQx" sentence. Three "We …" contribution bullets, the middle one carrying numbers.
Numbers in the abstract are repeated in the conclusion. Booktabs throughout, bold = best per column,
statistics defined once in Table 1's caption ("paired percentile bootstrap, 10,000 resamples
clustered over users; ⋆ marks an interval excluding zero"). One flat protocol sentence for
populations ("All orders and trees were fitted on the 10,000 validation users and scored once on the
disjoint 10,000 test users"). No em-dashes in body text, no rhetorical-question headings, no bold
run-in labels (italic `\paragraph` run-ins only). Anonymous author/institute, no acknowledgements,
redacted code footnote on p.2.
