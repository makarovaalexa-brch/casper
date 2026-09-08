# ECIR 2027 full paper from Chapter 2 — drafting brief

**Author ruling (2026-09-08):** *"prepare a draft publication for ecir conference. check what template needs to be
used, requirements for volume etc. … without changing chapter 2 write its content in a way to follow the style and
fill the template. include graphs etc. you will need a section on the instrument, but needs to be brief."*

## Venue facts (verified on ecir2027.co.uk, 2026-09-06)
- ECIR 2027, Southampton, 21–25 March 2027. Springer LNCS proceedings.
- Full papers: **12 pages including appendices, plus unlimited references**. Over-length = desk reject.
- **Abstract due 21 Sep 2026; paper due 5 Oct 2026** (11:59pm GMT); notification 7 Dec 2026.
- Double-blind, two-stage review. Avoid self-citation that breaks anonymity; preprints allowed.
- Template: `llncs.cls` v2.26 + `splncs04.bst` (in this directory). Single column, `\documentclass[runningheads]{llncs}`.
- AI tools: language editing only.

## What the paper is
Chapter 2's sections 1–4 and the answerability section, held-back requests work excluded (author's plan:
"ECIR 2027 full paper; sections 1–3 cut to ~12 pp"; the requests/openers/policy section goes to a second submission).

**Thesis of the paper (one sentence):** published adaptive question selectors for cold-start preference elicitation
optimise a surrogate of the reported metric and, re-implemented under one measurement contract on one frontier
recommender, every one of them loses to a fixed question order; putting the reported metric itself into the split
criterion reverses that, and changing what can be asked (concept questions) is worth more than adapting the order.

**Never write:** "adaptivity does not pay". The finding is *objective mismatch*: conditioning beats the static
order once the right objective is optimised (+0.01035 [+0.00739, +0.01323] full, +0.00705 tail, TEST, K=16).

## Source of truth
`new_chapters/chapterB_v2/chapterB_v2.tex` (46 pp, 24k words). **Do not change the chapter.** Every number in the
paper must come from it verbatim (or from `new_chapters/slides/part_ch2.py`, which was built from it). No new
experiments, no recomputation.

Line map (chapter → paper section):
| chapter | lines | paper |
|---|---|---|
| Introduction | 86–159 | §1 Introduction |
| Setup + answer model | 160–263 | §2 Measurement contract (+ brief instrument) |
| Frontier under standard practice (comparator, battery, one contract) | 264–620 | §3 Replication battery |
| Loss is not a failure to branch / budget spend / consumer flip / group greedy | 807–1038 | §3 (short) + appendix if room |
| Diagnosis: criterion regret, dispersion | 1039–1158 | §4 Diagnosis |
| Reported metric in the split; ablation; bank change; costs; depth; partition | 1159–1470 | §5 Measured-metric tree |
| Answerability orders the channels | 1471–1524 | §5 or §6 |
| Limitations | 2108–2129 | §7 |
| Positioning | 2130–2340 | §2 Related work (short) or after results |

Chapter 1 (the instrument) is `new_chapters/chapterA_v2/chapterA_v2.tex`; only the facts needed for the brief
instrument paragraph: certified two-tower set encoder, frozen; full/tail NDCG@10 0.3496/0.2491 on the Liang ML-25M
protocol vs RecVAE 0.3540/0.2497; fold-in with no per-user gradient step; concept channel = 1,031 genome tags with
whitened centroids and SEL affinity answers; four requirements R1–R4. Cite the instrument as an anonymous technical
report, not by title, or describe it in-paper without citation (double-blind).

## Page budget (12 pp LNCS ≈ 5,200 words + floats)
| section | pages |
|---|---|
| Title, abstract (≤150 words), keywords | 0.4 |
| 1 Introduction (with contributions list) | 1.4 |
| 2 Related work | 0.8 |
| 3 Measurement contract, instrument (brief), answer model | 1.4 (+ contract figure) |
| 4 Replication battery: comparator, battery table, budget curves, consumer flip | 2.6 |
| 5 Diagnosis: criterion vs measured gain; dispersion | 1.6 |
| 6 Measured metric in the split: tree, ablation, bank change, answerability | 2.4 |
| 7 Limitations and conclusion | 0.8 |
| References (not counted) | — |

## Floats (all data already in the chapter; pgfplots/tikz code reusable)
1. **Fig. 1** measurement contract (chapter lines 196–222, tikz).
2. **Fig. 2** full NDCG@10 vs budget, log axis (chapter lines ~690–723, pgfplots) — trim legend to ≤8 arms.
3. **Table 1** battery: one table, rows sorted by full@16, columns k=4/8/16 full/tail + answers@16 (from `tab:battery`).
4. **Table 2** criterion vs measured gain (`tab:criterionregret`).
5. **Table 3** split-criterion ablation (`tab:splitcrit`) or merged with Table 4.
6. **Table 4** bank change vs conditioning (`tab:bankdecomp`, two rows) + the tree's per-K gains (`tab:tree`).
7. Optional **Fig. 3**: dispersion (`tab:dispersion`) as a small bar/table.
Captions self-contained; booktabs; bold = best per column; intervals in brackets.

## Style (from `external_literature/ecir_style/ECIR_STYLE_GUIDE.md` — binding once it exists)
- Neutral academic register. No em-dashes in body text. No rhetorical questions in headings. No bold lead-ins.
- Enumerated contributions at the end of §1; RQs only if the corpus uses them.
- Statistics: paired percentile bootstrap, 10,000 resamples, clustered over users, seed 98765; report intervals,
  mark separation; state the test once in §3 and reuse.
- One flat protocol sentence for populations: fixed on validation, scored once on test. **No fitting-population
  narrative** (author writing rule, 2026-08-25).
- Name the system CASPER once (expanded), then plain. Never CASPER-R / CASPER-U.
- Code/data availability footnote with an anonymised placeholder URL.

## Build
`tectonic paperB_ecir.tex` from this directory (bib: `../references.bib`, `../chapterA_v2/chapterA_v2_new_refs.bib`;
copy needed entries into a local `refs.bib` if tectonic cannot resolve relative paths). Page count check after every
build; ≤ 12 pages before the references.
