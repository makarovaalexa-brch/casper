# paperA_tors — draft notes (2026-09-08)

Build: `tectonic paperA_tors.tex` in this directory (class and bst are local; see BRIEF.md).

**v2 (current, `paperA_tors.tex` = `paperA_tors_v2.tex`)**: `\documentclass[manuscript,screen]{acmart}` as the TORS
guidelines require (single-blind, no line numbers, `\setcopyright{none}`, numeric ACM references); **36 pages
total, body 32 (references p. 33, Appendix A p. 35)**; previewed in `[acmsmall]`: 37 total / 33 body, inside the
journal's usual 20–35. 59 references, all resolving, no leaked notes. What v2 changed and why is in
`REVIEW_SUMMARY.md`.

**v1 (`paperA_tors_v1.tex/.pdf`)**: 33 pages in `[acmsmall,review,anonymous]` mode (line-numbered; references
from p. 31), ~18,400 words including captions, 50 references, all resolving, no overfull boxes after
`\emergencystretch`. Bibliography copied verbatim from `../references.bib` and
`../chapterA_v2/chapterA_v2_new_refs.bib`; one working note removed (`qin2020attributepropensity`,
"WSDM/KDD line; Google unbiased-LTR"); arXiv-id and code-URL notes kept (they render as ordinary ACM
reference annotations).

## Structure (v1)

| § | title | pages | source in chapter |
|---|---|---|---|
| — | abstract, CCS, keywords | 1 | abstract, condensed and re-led with the inversion |
| 1 | Introduction (inversion first; scope; R1–R4 informal; bifurcation; RQ1–RQ4; 5 contributions) | 1–4 | §1 |
| 2 | Requirements and verification (formal R1–R4, Def. 2.1, per-requirement checks, integrity items) | 4–6 | §2 |
| 3 | Background: six-family taxonomy (two protocols, ratings decomposition, master table, corners, answerability lineage, gap map) | 7–12 | §3 + the answerability paragraph of §6.4 |
| 4 | Gap analysis (near-misses, three legs, feasibility, prior art) | 11–14 | §4 |
| 5 | The instrument (tower, observation model, architecture figure, learned-vs-closed-form fold, free-text table, provenance-independence, graded-curve figure) | 14–17 | §5 |
| 6 | The concept channel (SEL/VAL, whitening + gated fold, lineage + critiquing, deployment currency) | 17–20 | §6.1–6.3 |
| 7 | Experimental setup (ruler, bridge, baseline bank, answer provenance, "just add columns", interview protocol) | 21–23 | §7.1–7.2, §7.4 (firewall), §7.3(iv) |
| 8 | Results: 8.1 RQ1 inversion · 8.2 RQ2/RQ3 battery + R1 + exclusions + answerable-bank note · 8.3 RQ4 concept channel + answerability | 23–29 | §8, §7.3, §6.4 |
| 9 | Limitations and threats to validity | 29–30 | §9 |
| 10 | Conclusion | 30–31 | new, numbers repeated from abstract |

## Chapter → paper changes that are more than compression

- **Re-led with the inversion** (Fable review 2026-08-09 ruling (b)): the introduction states the three orderings
  and the classic-consumer reproduction with no instrument detail; the table lands in §8.1. The chapter opened
  with the same paragraph, so the numbers are unchanged; only the emphasis moved.
- **RQ1–RQ4 added** (journal convention); each results subsection ends in an explicit "Answer to RQx".
- **Answerability lineage** (Rashid / Golbandi / Elahi / Sun 2013 / Gharahighehi / Xia) moved from the concept
  results (chapter §6.4) into §3 (related work), where a journal reader expects it; the measured 25.2 % / 13.1 % /
  53.0 % numbers stay in §8.3.
- **"Just add genre columns"** (chapter §7.3(iv)) became a short subsection of §7 and is referenced from §4.
- **Deployment currency** (chapter §6.3 + the "methodological consequence" paragraph of §6.4) merged into one
  subsection, with the confession-shaped sentence ("a capability check passed on a fold that … never emitted a
  negative answer") rewritten as a forward-looking rule, per the working-version directive.
- **Limitations** rewritten as named run-in paragraphs (ACM style); the chapter's "metric non-comparability
  until the bridge is run" item folded into "bridged and unbridged rows"; BERT4Rec and graph-filter items
  merged; a new paragraph "The instrument is one of the consumers it is compared with" added because the
  chapter's inversion table makes our own model both a row and the reference column and a TORS reviewer will
  ask (the sentence's content is the chapter's own: the disagreement between the classic consumer and the
  likes-only frontier consumer does not depend on our row).
- **Provenance/firewall** promoted to a named subsection of the experimental setup (Fable review 2026-08-05
  §2.3), rather than a caveat.

## Chapter content omitted

- The **thesis table of contents**, "this chapter"/"companion chapter" framing, and every forward pointer to
  the policy chapter (replaced by "outside this article's scope" / "a study of that layer").
- The chapter's **"what each table is for"** paragraph (§3) — a journal reader does not need it.
- The **operator-ablation footnote's provenance clause** is kept (it names that the ablation ran on an earlier
  tower and settles only the operator choice) because omitting it would make the absolute numbers misleading.
- The chapter's **`\REDO`/`\TODO` machinery** — none present in the chapter any more.
- **Free-text screening rejects** kept (one sentence); the "we would not defend" sentence kept.
- The chapter's **NDCG@100 per-row list** kept (one sentence in §7.3) — it orders the LIANG column identically
  and a reviewer may ask for it.
- The "**within an answerable bank, question choice matters far less**" paragraph kept but compressed and
  merged with the one-level adaptive probe; the selection numbers (+0.0036, +0.0015/+0.0016) are stated as
  reported-not-claimed, as in the chapter.

## Numbers that are NOT from the chapter (author to verify)

- The **architecture and training paragraph** in §5.1 (d=200, ten half-star levels, width 512, 1.07 M trainable
  parameters, curriculum regimes, multinomial NLL, two learning-rate groups, one run selected on validation)
  comes from `src/instrument/i26_encoder.py`, `train_i26.py`, `interview_curriculum.py`,
  `experiments/instrument/t2i26c_run.out` and `t2i26c_selection.json`, not from the chapter. Check against
  `docs/design/RETRAIN_DESIGN_SHEET.md`.
- The **split definition** in §7.1 (18,430 / 18,359 / 140,768 / 10,000 / 10,000 / 80-20 / seed 98765 / 190 head
  items) is the project ruler from `CLAUDE.md`, also not stated in the chapter.
- **9,281 tail users** (§7.1): the harness's tail row-compaction (`a9_battery_policyB.py`, `policy_labels.py`);
  the chapter mentions 9,281 only in passing.

## Numbers I was not certain about in v1 (resolved in v2; see REVIEW_SUMMARY.md items 1–5)

1. **Two graded measurements.** The chapter reports three distinct short-interview premiums for graded input:
   `+0.0144` at k=2 (§2, R4 check: "adding the sub-threshold answers"), `+0.0102` at k=2 (Fig. gradedcurve and
   §3: "worth +0.0102 at k=2"), and `+0.0117`/`+0.0071` in older notes (not in the chapter). The paper uses
   `+0.0144` only where the chapter does (§2 R4 check, "adding the sub-threshold answers pays … +0.0144 at
   k=2") and `+0.0102` where the chapter does (figure caption, §3.2, battery table, R1 paragraph). The two are
   different comparisons in the chapter's own words; the paper keeps them apart, but a reader may still ask why
   two k=2 premiums exist. **Author to confirm the label of each** (`fig_graded_curve.png` plots the
   +0.0102 series).
2. **Two intercepts.** `0.1279/0.0192` (item interview harness, R1 identity read-out) and `0.1277/0.0267`
   (concept-channel matched protocol, zero-item). The paper states in §7.1 that the cold intercept is
   0.1279/0.0192 and uses 0.1277/0.0267 only in §8.2 ("deployed cold … against 0.1277/0.0267") and §8.3 (polar
   comparison). The chapter's sentence explaining that the two protocols carry their own intercepts was cut for
   length; if a reviewer trips on it, restore one clause in §7.6.
3. **"Six consumers realise three distinct orderings"** — counted from the chapter's table on full NDCG@10
   (E0>P>H, P>E0>H, H>P>E0; RBMF's E0≈P>H and TaNP's E0>P>H are not distinct from the first). The chapter's
   caption says the same.
4. **Master-table row count.** I did not print a count (the intro says "every reasonable candidate system");
   the table has 26 rows including ours.
5. **`+0.0293` vs `0.029`.** The introduction's "scores below our instrument by 0.029 full NDCG@10 after eight
   questions" is the chapter's own rounding of the table's `+.0293 [+.0267,+.0320]`; both appear as in the chapter.
6. **`0.15` vs `0.12`.** The chapter says both "forfeits 0.15 full NDCG" (§4.1) and "uniformity costs roughly
   0.12 full NDCG" (§3, §5) for the conjugate accumulator; 0.3496−0.2267 = 0.1229, so the paper uses **0.12**
   in both places.
7. **Battery R3 numbers** (`0.1574→0.2609`, rise `+0.1035`) and the R4 arms are copied from `tab:batteryresults`
   unchanged, including `n=7,563` for the sign-flip cohort.
8. **Bridge counts** (`9,990,682` events / `116,677` train users / `20,108` items) and the per-row NDCG@100
   list are verbatim.

## Style compliance (against `external_literature/tors_style/TORS_STYLE_GUIDE.md`)

Abstract two paragraphs, ~330 words (corpus 160–300; trim if the author wants). CCS concepts and keywords
present. Numeric ACM references (the class default; every corpus paper). RQ1–RQ4 enumerated; every results
subsection ends in an explicit answer and the conclusion repeats the answers with RQ tags. Roadmap paragraph at
the end of §1. Standalone Discussion (§9) with practical guidelines, then Limitations (§10) as named run-in
paragraphs each paired with future work. `\Description{}` on every figure. Statistics named once in §7.1 with
the multiplicity statement. Code sentence at the end of §7. Appendix A after the references. Booktabs tables, bold = best per column, statistics
defined in captions and once in §7.1 (paired per-user bootstrap, 10,000 resamples, 10,000 test users).
No em-dashes in prose (only `---` as table dashes). No "this chapter", no "companion chapter", no "parity",
no "tie", no CASPER-R/-U, no "first" claims. Anonymous author block; repository footnote withheld.

## Open items only the author can decide

- Whether TORS submission is double-anonymous for this call (`TORS_VENUE_FACTS.md`); the draft is anonymised
  and can be de-anonymised by removing `anonymous` from the class options and filling `\author`.
- Whether to keep §3.2 (the ratings decomposition, ~1 page) in the body or move it to an appendix.
- Whether to add the RecVAE re-score of the inversion at other budgets (k=2/16) — the chapter has k=8 only.
