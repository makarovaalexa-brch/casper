# TORS regular paper from Chapter A — drafting brief

**Author ruling (2026-09-08):** *"your colleague has drafted the ecir submission … can you do the same for TORS?
collect strong published papers, study style, find template requirements, and fold in instrument work, chapter A
in required format, required style."*

Standing rulings that bind this draft (memory, 2026-07/08):
- Paper A = **the instrument** (the recommender that makes policy measurement possible). Elicitation is evidence
  that it works on both channels, not a headline. No policy claim, no adaptivity claim, no "concepts beat items"
  headline (bank-conditional, second act).
- **Lead with the inversion** (Fable review 2026-08-09, ruling (b)): the introduction states the three orderings
  and the classic-ordering reproduction with zero instrument detail; the full table lands after the method.
- **Present the working version.** No error archaeology, no performed candour, no retraction paragraphs. Keep
  every fact whose omission would make a claim false, unreproducible or unfairly attributed (protocol, ruler,
  tail mask, substitutions in reconstructed baselines, the Mult-DAE drop, iALS as a floor).
- **The decision rule for self-critical sentences:** does the paper read better with it in or out?
- Never "parity"/"tie" for the tail: "statistically indistinguishable (Δ=−0.0006, CI [−0.0020,+0.0008])".
- Name the system **CASPER** once, expanded; never CASPER-R / CASPER-U. "This chapter" → "this article";
  "the companion chapter" → a companion study, outside this article's scope.
- Venue history: TORS regular paper is the settled target for A (`docs/PUBLICATION_PLAN.md` §TORS, 2026-07-29;
  two Fable reviews on file, 2026-08-05 and 2026-08-09, both "TORS regular").

## Source of truth
`new_chapters/chapterA_v2/chapterA_v2.tex` (41 pp article class, ~21,100 words, 50 references). **Do not change the
chapter.** Every number in the paper comes from it verbatim. No new experiments, no recomputation.

## Venue facts
See `TORS_VENUE_FACTS.md` in `external_literature/tors_style/` (fetched 2026-09-08) and the summary in
`DRAFT_NOTES.md`. Template: `acmart` v2.20 (2026/08/16), `\documentclass[acmsmall,review]{acmart}` +
`\acmJournal{TORS}`; the bundled Tectonic copy of acmart predates TORS, so the class and
`ACM-Reference-Format.bst` sit in this directory (copied from the TeX Live archive). Build:
`tectonic paperA_tors.tex` from this directory.

## Style
`external_literature/tors_style/TORS_STYLE_GUIDE.md` (binding once written). Until then: ACM journal
conventions — abstract 150–250 words, CCS concepts + keywords, enumerated RQs, contribution list, RQ-named
results subsections with explicit answers, booktabs tables with the legend in the caption, author-year
citations (acmart journal default), limitations as a named section, conclusion repeating the numbers.

## Chapter → paper map
| chapter | paper |
|---|---|
| §1 Introduction (confound, scope, R1–R4, bifurcation, contributions) | §1 Introduction, re-led with the inversion; RQ1–RQ4 |
| §2 Requirements and gates | §3 Requirements and verification |
| §3 Related work (taxonomy, two protocols, master table, gap map) + §4 Gap analysis | §2 Background and related work (shortened) + §4 Gap analysis |
| §5 The instrument | §5 The instrument |
| §6 Concept channel (estimator, fold, lineage, deployment currency) | §6 The concept channel |
| §7 Experimental design (bank, bridge, protocol, provenance, metrics) | §7 Experimental setup |
| §7.3 battery, §6.4 concept results, §8 inversion | §8 Results, ordered RQ1 (inversion) → RQ2/RQ3 (battery) → RQ4 (answerability) |
| §9 Limitations | §9 Limitations, §10 Conclusion |

## Floats
Fig. 1 architecture (tikz, reused) · Fig. 2 gap map (pgfplots, reused) · Fig. 3 graded curve (png) ·
Table 1 master table · Table 2 free-text illustration · Table 3 verification battery · Table 4 strategy inversion.
