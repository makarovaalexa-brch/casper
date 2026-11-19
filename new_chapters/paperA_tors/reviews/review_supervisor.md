# Supervisor pre-submission read: `paperA_tors.tex` against `chapterA_v2.tex`

Date: 2026-09-08. Manuscript 614 lines / 33 pp; chapter 1,779 lines. Rule under audit: every number
verbatim from the chapter, no recomputation, no re-rounding, no reinterpretation of what a number measures.

Method. A script (`scratchpad/audit_numbers.py`) tokenised every numeric literal in the manuscript body
(decimals, signed deltas, percentages, comma-grouped counts; small bare integers such as `k=8` excluded)
and tested value-presence in the chapter. A second script (`scratchpad/table_diff.py`) diffed the three
shared tables cell-by-cell. Everything the scripts passed I then re-read by hand for the *state* each number
is measured at, and I recomputed every derived delta in the inversion table from its printed cells.

---

## 1. Number fidelity audit

**Script result.** 916 numeric tokens in the manuscript. 19 have no value-match anywhere in the chapter;
14 of those are LaTeX layout (column widths, CCS concept ids, `leftmargin=3.2em`) and are not claims. The
remaining 5 are the split counts in §7.1 (below). All three tables (master, inversion, battery) are
**cell-identical** to the chapter, including bold placement and the check/cross flags (172 / 77 / 50
tokens compared).

**Hand result.** Value-presence is necessary, not sufficient. Reading each number at its state:

| number | manuscript location | chapter location | verdict |
|---|---|---|---|
| 0.3496/0.2491 vs 0.3540/0.2497; −0.0006 [−0.0020,+0.0008]; −0.0043 [−0.0059,−0.0028]; 1.2% | abstract, §1 contrib 3, §5.1, Table 3, §8.2, §10 | abstract, §method, §expdesign R1 para, Table battery | OK (state: LIANG, full 80/20 fold-in, 10k test) |
| 18,430 / 18,359 / ≥20 ratings / r>3.5 / ≥5 likes / 140,768 / 10,000 / 10,000 / 80/20 / seed 98765 / 190 head items | §7.1 Data and ruler | **not in chapter** (chapter never states the split counts; it defers to "§expdesign", which does not contain them) | NOT-IN-CHAPTER. Verified against `casper/CLAUDE.md` canonical-ruler line and `docs/design/DESIGN_PAPERB_REPLICATION.md` §83–84: all correct. Keep; a journal paper must define its split. Record the source. |
| 10,000 resamples | §7.1 | Table instrdep caption | OK |
| cold intercept 0.1279/0.0192 | §7.1 (interview curves) | §expdesign R1 para; §concept:results | OK, but see the missing sentence flagged in §3 below |
| concept intercept 0.1277/0.0267 | §8.2 "What the four criteria exclude"; §8.3 POLAR paragraph | §batteryresults; §concept:results | OK as numbers. **The manuscript now prints both intercepts and never says why they differ.** The chapter's sentence "the zero-item concept curves … carry their own intercept, 0.1277/0.0267, which is why the two are never compared across" was cut. A reviewer will read 0.1279 in §7.1 and 0.1277 in §8.3 as a typo. Must restore. |
| +0.0039 full, +0.0020 tail, 0.3496→0.3457, 0.2491→0.2471, −0.0083 | §2.2 R4, Table 3, §8.2, §9 | §reqs R4; Table battery; §expdesign; §limits | OK. Correctly labelled "identical support, re-valued by star level, at full profile" everywhere. |
| +0.0102 at k=2 [+0.0087,+0.0118]; +0.0080 at k=4; +0.0010 at k=8 | Fig. 3 caption, §3.3, Table 3, §8.2 | Fig. gradedcurve caption; §ratings; Table battery | OK as numbers. Labelled "graded vs likes-only (sub-threshold dropped)" = "fair binarising ablation". Consistent with the PNG (≈0.176 vs ≈0.165 at k=2). |
| **+0.0144 at k=2** | §2.2 R4 check: "adding the sub-threshold answers pays … +0.0144 at k=2 and nothing from k=8" | §reqs R4 (line 321) verbatim | REWORDED-but-same on fidelity, **but the manuscript prints two different k=2 premiums (+0.0144 in §2.2, +0.0102 in Fig. 3/Table 3/§3.3/§8.2) under descriptions a reader cannot tell apart** ("adding the sub-threshold answers" vs "graded over likes-only, sub-threshold answers dropped"). The chapter has the same collision. `docs/STATE.md` §1011 confirms +0.0144 is a real recorded figure but does not say what it contrasts. The figure supports +0.0102 as the graded-vs-likes-only premium. **Must resolve**: either state in §2.2 what +0.0144 varies that +0.0102 does not (the chapter's own automated-check source will say), or replace it with +0.0102 in §2.2 so the battery description and the battery table agree. |
| RecVAE tail delta +.0258 [+.0231,+.0285]; conjugate tail +.0492 [+.0459,+.0525] | Table 4 tail block, "Ours − this"; §8.1 "+0.0258 tail, each system on its own best strategy" | Table instrdep identical; §instrdep identical | numbers OK, **label WRONG**. Recomputed from the printed cells: .0792 − .0534 = .0258 uses RecVAE's POP arm, but RecVAE's tail-best (bold) is ENT0 .0576, which would give +.0216. Likewise .0792 − .0300 = .0492 uses the conjugate belief's ENT0 arm, but its tail-best (bold) is HELF .0338 (+.0454). The column pairs each rival at its **full-profile-best** strategy scored on the tail, not "at its own best strategy" as the caption and §8.1 say. Golbandi, RBMF, TaNP happen to coincide. Fix the caption and the §8.1 sentence to say what was measured ("each rival at the strategy that is best for it on full NDCG@10, scored on both metrics"). Do not change the cells. |
| +0.0105 full vs Golbandi; −.0000 tail; .0792 both | Table 4, §8.1 | identical | OK (recomputed: .2053−.1948; .0792−.0792) |
| +0.0293 full, 0.029 (intro) | §1, §8.1, Table 4 | intro, §instrdep | OK |
| −0.0042 / +0.0042; 9,281 users | §8.1 | §instrdep | OK |
| 1.59 / 3.10 / 3.05 / 0.05 / "1.6 answers of 8" | §8.1, Table 4 caption | §instrdep | OK |
| NDCG@100 list (RecVAE 0.4537 … Most-Popular 0.1975) | §7.3 | §expdesign (ii) | values OK; manuscript re-sorted Turbo-CF (0.3569) above RBMF (0.3533), which the chapter had out of order. **But the claim "orders the LIANG column identically" is false on the printed numbers, in both documents**: on NDCG@10 (LIANG) RBMF (.2915) sits 7th, above iALS, TaNP and Turbo-CF; on NDCG@100 it sits 10th, below all three. EDDI (.2153) and the conjugate belief (.2267) also swap. Soften to "orders the top of the column identically (RecVAE, ours, EASE, EDLAE, Mult-VAE, user-kNN)". |
| roughly 0.12 (uniform conjugate fold forfeits) | §4.2 and §5.2 | §method says 0.12; **§gap:why says 0.15** | REWORDED-but-same. The chapter is internally inconsistent; 0.3496−0.2267 = 0.1229, so the manuscript's 0.12 in both places is the arithmetically right choice. Note it so nobody "restores" 0.15. |
| +0.0064 full RecVAE over EASE; 1.8 unpaired SE | §7.3 | §expdesign | OK |
| bridge: 0.4203/0.420; 0.4425/0.442; 0.4144/0.414; 0.5525/0.553; 0.4223/0.426 (−0.0037); 0.3928/0.395; 0.5341/0.537; 0.3969/0.419; 0.358/0.386; 9,990,682 / 116,677 / 20,108 | §7.2 | §bridge | OK. Manuscript drops the miss magnitudes (−0.022, 0.028) and says "a larger miss"; fine. |
| Most-Popular +0.028, EASE +0.066 (pool effect) | §3.1 | §protocols | OK (recomputed from cells) |
| user-kNN 0.3756 / 0.2822 / −0.0934 / −0.0976; iALS +0.0194 [+0.0157,+0.0231]; 2.26×; +0.0257; −0.0088; +0.0282 (CIs); RBMF −0.070; −0.0045 / +0.0046 (CIs); 0.0981; 0.0499; +0.0454; +0.0601; +0.0281; +0.0267; +0.0194 vs +0.0022 [+0.0008,+0.0036] | §3.3 | §ratings | OK. State correct (NATIVE column, same 10k users). |
| iALS .3325/.2629, +0.0170; Mult-DAE 0.397 vs 0.419 | Table 1 caption | Table master caption | OK |
| TaNP 0.0060/0.0005; 0.1626; RMSE 0.696 / 0.955; 1,370 / 2,000; 0.862→0.697; 0.0111→0.0066; 0.0118 | §3.2 | §nativelimit | OK |
| 0.3065/0.1895; 0.2915/0.2071; 16 users, slates of 5; ~100 items | §3.4 | §related | OK |
| 0.420 / 0.426 / 0.442 / 0.401 / 0.386 (ML-20M published); 0.2622/0.1621 | §3.4 accuracy corner | chapter same para | OK |
| MRR@10 0.27 vs 0.17 (PEBOL) | §3.4 | same | OK |
| >80% unknown branch (Sun et al.) | §3.4 answerability | §answerability | OK |
| d=50; 164 tags; +0.2096 [+0.1986,+0.2208] | §4 | §gap | OK |
| Spearman 1.0000; |μ0|=0; AUC 0.44 → 0.94; +0.0015 [+0.0007,+0.0024]; +0.0036; 18,359 | §4.2 | §gap:why | OK |
| 0.2267/0.1300 | §5.2 | §method | OK |
| 1,031; 80% / 20%; 300×; 0.167 / 0.0005; cosine 0.638; cosine 0.6; table cos 0.47 / 0.39 / 0.54 / 0.54 | §5.2, Table 2 | §method, Table freetext | OK |
| λ=3; R² 0.376 / +0.042 / 0.398 / 0.017; 22× | §6.1 | §concept:est | OK |
| cosine ≈0.5; intercept 0.1279; +0.011 at two; 0.0991 at eight; 0.1759/0.0841; 0.1322/0.0503; 0.1685/0.0906; 0.1572/0.0777; AUC 0.820 / 0.626 | §6.2 | §concept:fold; §reqs R2 | OK. Footnote correctly keeps "earlier tower" provenance. |
| 40 or 75 keyphrases; ~57k entities, 46 relations; ~5 turns | §6.3 | §concept lineage | OK |
| −0.2012 (concept sign flip) | §6.4 | §concept:fail | OK |
| p=0.4; α=0.5, s=1; epoch 46 / 31 | §7.3 | §expdesign (ii) | OK |
| Table 3 (battery): all cells incl. 0.1574→0.2609, 0.0505→0.1062, +0.1035 [0.1000,0.1071], 0.4298→0.1345 (−0.2953, n=7,563), 0.4198→0.1714 (0.2484), 0.2999→0.0487, ρ=0.794 / 0.818, +0.0371, +0.0168, −0.0986, AUC 0.94, +0.1278 [+0.1195,+0.1361] | Table 3 | Table battery | OK (cell-identical) |
| 0.626 / 0.416 / 0.498; +0.0087; 0.1327/0.0412; 0.1498/0.0673; +0.0051; 0.79 / 1.97 | §8.2 | §batteryresults | OK |
| 0.2358 vs 0.2230 at q=8; +0.0022; +0.0036 [0.0016,0.0057]; +0.0016 tail [+0.0008,+0.0024] | §8.2 | §batteryresults; §adaptive headroom | OK |
| +0.0104 [+0.0075,+0.0134]; 0.1268; 0.1372; −0.045; +0.0335 | §8.3 | §concept:results | OK |
| POLAR: +0.0045 / +0.0109 / +0.0189 / +0.0310 / +0.0146 / +0.0265 with CIs; −0.0782 [−0.0857,−0.0707]; 4.8%; 0.1352/0.0409; 97.2% | §8.3 | §concept:results | OK |
| 25.2% / 13.1% / 53.0% / 1,031 | abstract, §1, §8.3, §10 | abstract, §answerability | OK |
| 55.7% | §9 | §limits | OK |
| 1.5% tolerance | §1, §2 | intro R1 | OK |

**Counts.** WRONG numbers: **0**. NOT-IN-CHAPTER: **1 group** (the §7.1 split definition; verified correct
against the project ruler). WRONG-LABEL (number verbatim, description of what it measures does not match
the printed cells): **2** (the tail "Ours − this" column; the NDCG@100 "orders identically" claim).
AMBIGUOUS (two chapter numbers printed under indistinguishable descriptions): **1** (+0.0144 vs +0.0102 at k=2).
REWORDED-but-same: **1** (0.12 where the chapter also says 0.15).

---

## 2. Claim fidelity

Sentences that go beyond the chapter or drop a qualifier it attaches:

1. **Abstract and §10**: "against a RecVAE bar of 0.3540/0.2497 reproduced to the published third decimal."
   The bar is an ML-25M NDCG@10 number; nothing published exists for it. What was reproduced to the third
   decimal is the ML-20M NDCG@100 anchor (0.4425 vs 0.442). As written, the sentence lets a reader believe
   0.3540 is a published figure. Chapter abstract says "our in-house RecVAE realisation sets the bar".
   Reword: "against a RecVAE bar of 0.3540/0.2497 from an implementation that reproduces the published
   ML-20M numbers to the third decimal".
2. **§1 para 2**: "Under the strongest full-profile model … restricted to the likes …, POPULARITY leads."
   True on full only; on tail that consumer has ENT0 > POP > HELF (Table 4). §8.1 says "on the full
   profile"; the intro drops it. Add the three words.
3. **Table 4 caption and §8.1**: "each system at its own best strategy" for the tail deltas. See §1 above;
   two of five tail deltas are not against the row's bold cell. This is the one place a careful reviewer can
   catch the paper contradicting its own table.
4. **§7.3**: "NDCG@100 orders the LIANG column identically." False on the printed numbers (RBMF, EDDI).
5. **§8.1 "Answer to RQ1"**: "the mechanism, what each consumer does with an unanswerable question, is
   measurable in graded-answer counts." The body correctly keeps the chapter's limit ("answerability explains
   why HELF divides the table, not the remaining orderings"); the RQ answer drops it. Add "for the HELF split".
6. **§3.3**: the chapter's explicit self-limit "we are not entitled to say that the canonical binarisation
   handicaps the elicitation family at full profile … nor to present graded input as something only the
   instrument turns to account" is gone. The manuscript's §3.1 argues the LIANG column reports the family
   "with its designed input removed", and §3.3 then shows restoring it does not help; without the chapter's
   sentence the two halves read as a contradiction rather than a finding. Restore one sentence.
7. **§8.3 first result paragraph**: "Re-running the pre-registered triple gate on the retrained model …
   ($0.1372$ in this gate's arm)". "The retrained model", "triple gate" and "this gate's arm" have no
   antecedent in the manuscript (the chapter defined them in §concept:fail, which was cut). This is the one
   residue of error archaeology and it now reads as an undefined reference. Rewrite as "The signed concept
   fold, driven down the deployment-currency bank, clears the three taste criteria of Section 6.4".
8. **§9 "The instrument is one of the consumers"** (new, not in chapter): "the classic consumer and the
   likes-only frontier consumer disagree with each other, and with the Golbandi node model, whether or not
   our row is present." The classic consumer *is* the Golbandi node model; the sentence is garbled. The
   point is right and supported by Table 4 (H>P>E0 vs P>E0>H); fix the sentence.

Qualifiers the chapter attaches that the manuscript **keeps** (checked): "we do not claim a selection
result" (§8.2, twice); tail non-win against Golbandi HELF (Table 4 caption and §8.1 last paragraph);
"this is not a controlled attribution … confounds the input contract with that training" (§8.1); the
mislabelling control "never as a headline" (§3.3); "we make no claim about why" (§3.3); "It does not show
that concepts beat items in general" (§8.3); "a designer whose production system is a tree should arguably
believe the tree" (§8.1); the operator-ablation "earlier tower" footnote (§6.2); Definition 2.1's "failure
to separate, not a demonstration of equivalence".

---

## 3. What was lost

| chapter content | manuscript | judgement |
|---|---|---|
| "the two intercepts are never compared across" sentence (0.1279/0.0192 vs 0.1277/0.0267) | cut | **must restore** (see §1) |
| "not entitled to say binarisation handicaps the elicitation family" | cut | **must restore** one sentence (see §2.6) |
| §concept:fail history of the sign-blind answer model, the "eval-only fold", the "signed retrain has since landed" | cut; general lesson kept in §6.4 | good cut, but leaves the dangling references in §8.3 (§2.7) |
| "methodological consequence for verification design" paragraph | folded into §6.4 last sentence | fine |
| +0.1032 / +0.0282 mislabelling-control paragraph ("one comparison must be kept out of that sentence") | cut; "never as a headline" kept | fine |
| chapter §related "What each table is for" | cut | fine |
| ICF ceiling = iALS row (0.2703) | reduced to "its ceiling is the measured iALS row" | fine |
| chapter's stale "EDDI … carries no number" | corrected (EDDI has a row) | improvement |
| substrate "RecVAE-class amortised encoder or a closed-form autoencoder" | narrowed to amortised | improvement; consistent with §4.2's own argument that EASE has no latent |
| "continuous direction … snapped to a nearby meaningful axis" | "snapped" dropped | fine |
| "we report any such step rather than suppress it" (monotonicity limitation) | cut | fine |
| "until it exists the quantitative inversion should be read as established for this ruler rather than for recommendation at large" | cut | restore (one clause); it is the honest reading of a one-dataset paper and the brief's own limitation |
| chapter's separate limitations: metric non-comparability, BERT4Rec, graph-filter admissibility | merged into "Bridged and unbridged rows" | fine |
| chapter "Sign is worth … an open quantity on this ruler" | kept | good |
| the four-reading structure "Five readings, and we are careful about which we claim" | converted to paragraphs | fine |
| companion-chapter pointers (policy, continuous-action interface, "what a policy should do with it") | replaced by "a policy that wants one can be built on top", "the natural next study" | fine, and required by the brief |
| "Sun et al. … diagnosed twelve years earlier and more sharply" | shortened | fine |

New in the manuscript and not in the chapter: the RQ framing, the §7.1 split definition (verified), and
the §9 limitation on the instrument being a row in its own table (sound, but the sentence needs fixing).

---

## 4. Attribution

- Master table: full-profile row printed as **user-kNN ~\citep{golbandi2011adaptive}** with the ‡ mark
  and the caption's "reduction we construct, not a result its paper reports" (matches chapter). Inversion
  table consumer printed as **Golbandi node model**. §3.4 calls it "the node recommender that is the
  Golbandi tree's core". Fig. 2 label "user-kNN". Correct on the naming rule.
- One drift: §9 "The Golbandi node model, RBMF and the conjugate belief are best-effort cores … with every
  substitution stated in Section 3" refers to the full-profile rows, where the chapter (and the manuscript's
  own §7.3) says "user-kNN". Change to "user-kNN".
- Bıyık and ConTS: dashes in both columns, with the chapter's reason ("a number that is ours, not theirs")
  and Bıyık's own metric described. Correct.
- Conjugate belief: master table "Conjugate belief (frozen MF)" with no rival citation; inversion table
  "after \citet{biyik2023soft}"; §3.4 says the mechanism is measured in §5 "as the rejected alternative".
  Correct.
- No sentence prints a reconstruction of ours under a rival's name. RBMF and TaNP are cited to their papers
  in Table 4 without a "best-effort" mark in that caption; §7.3 states it. Acceptable; a ‡ on those two
  rows in Table 4 would be cleaner.
- Frozen-tower design credited to Göpfert and Bıyık as convergent, not distinguishing (§5.1). Correct.

---

## 5. Style rulings

- **"parity"/"tie" for the tail**: the tail result itself is always "statistically indistinguishable"
  with the CI. Correct. But: §7.2 "no parity claim is made until the bridge is run"; §9 "the parity claim
  rests on the bridge"; §9 "no row in it can be a bridged **tie**". "Frontier-parity" as R1's name is the
  chapter's term and can stay; the three sentences above should say "the R1 claim" / "a bridged
  comparison". Definition label `def:tie` is invisible; fine.
- **"this chapter"/"companion chapter"**: zero occurrences. "this article" throughout. Fig. 2 label reads
  "CASPER (this article)". Good.
- **CASPER-R/CASPER-U**: zero occurrences. "CASPER" expanded once (§1 Contributions). Good.
- **Performed candour**: zero hits on honest/candid/awkward/retract/unflattering. Good.
- **Error archaeology**: (a) §3.3 "RBMF's graded arm was recorded at −0.070 against its binary build …
  at a twentieth of the magnitude reported previously … where the old number came from". "Recorded" and
  "reported previously" by whom? This is our own superseded configuration. The substantive fact is the
  decomposition (centring 0.0981, erased values 0.0499, restored +0.0454); the −0.070 is history. Cut
  "recorded at −0.070" and "reported previously"; keep the decomposition. (b) §3.3 "shared one hyperparameter
  setting selected by a validation sweep that optimised NDCG@100 on a previous dataset" is also history;
  keep only "re-selecting on the reported metric moves RBMF by +0.0601 …" if the re-selection deltas are
  needed (they are, for the attribution). (c) §5.1 "the graded-native set-encoder T2′, trained once in its
  pre-registered configuration": "T2′" is an internal checkpoint name; drop it. (d) §8.3 "retrained model",
  "triple gate" (see §2.7). (e) §6.2 footnote "earlier tower" is provenance, not archaeology; keep.
- **Em-dashes in body prose**: none (only the `---` table cells). Good.
- **"adaptivity does not pay"**: absent. §8.2 says the one-level probe "pays … with a confidence interval
  excluding zero" and that the selection problem is out of scope. Correct.
- **"first"/"novel"**: "not claimed as novel" (§1), "our defensible novelty is the conjunction" (§4.3),
  "the one leg of the conjunction we find nowhere in the literature" (§4), all bounded by "our survey" in §9.
  Defensible. No "first" claim.

---

## 6. Verdict

**Not as it stands, but close.** No number is wrong and every table is cell-identical to the chapter. What
would sink it at review is not fidelity but three places where the paper's own printed numbers contradict
its own sentence about them (the tail "Ours − this" label, the NDCG@100 "identically", and two k=2 premiums
under one description). Those are half a day's work and none requires a new run.

### Must change before submission (ordered)

1. Table 4 caption and §8.1: describe the "Ours − this" column as what it is (each rival at its
   full-profile-best strategy, scored on both metrics). Numbers stay.
2. Resolve +0.0144 vs +0.0102 at k=2 (§2.2 vs Fig. 3 / Table 3 / §3.3 / §8.2): state what each varies, or
   use +0.0102 in §2.2. Check the chapter's automated-check source file for what +0.0144 contrasts.
3. §7.3: "orders the LIANG column identically" → "orders the top of the column identically".
4. Restore the sentence that the concept curves carry their own intercept (0.1277/0.0267) and are never
   compared with 0.1279/0.0192; without it the paper appears to print a typo.
5. Abstract and §10: do not attach "reproduced to the published third decimal" to the ML-25M bar.
6. §8.3: remove "retrained model" / "triple gate" / "this gate's arm" (undefined references).
7. §9: fix the garbled "classic consumer … and with the Golbandi node model" sentence; change "Golbandi
   node model" to "user-kNN" where the full-profile rows are meant.
8. §7.2 / §9: "parity claim" → "R1 claim"; "bridged tie" → "bridged comparison".
9. §3.3: restore the chapter's one-sentence self-limit on binarisation not being shown to handicap the
   elicitation family; cut "recorded at −0.070" / "reported previously".
10. §1: "POPULARITY leads" → "POPULARITY leads on the full profile". §8.1 RQ1 answer: bound "the mechanism"
    to the HELF split.
11. §5.1: drop "T2′". §7.1: record where the split counts come from (project ruler), since they are the one
    group of numbers not in the chapter.

### Would strengthen (ordered)

1. Add ‡ to the RBMF and TaNP rows of Table 4 (best-effort / objective-ported), as the master table does.
2. Restore the one-dataset limitation's closing clause ("established for this ruler rather than for
   recommendation at large").
3. Note once, in §4.2 or §5.2, that 0.12 is 0.3496 − 0.2267, so a later editor does not reintroduce the
   chapter's 0.15.
4. §8.3 opening: the harness is "scored against the no-answer intercept" with no number; give 0.1277/0.0267
   there (this is the concept-protocol intercept) so the reader meets it before the POLAR paragraph.
5. The abstract is one 330-word paragraph; TORS asks for 150–250. Two sentences can go (the six-family
   taxonomy sentence duplicates the contribution list; the "reproduces the frozen decoder exactly" clause
   is in §10).
6. Table 4's `\S` footnote says only the HELF cell is "materially affected" by the missing floor and that
   orderings are identical under both floor contracts; the chapter said the same. If the floored numbers
   exist on disk, one clause giving the HELF cell under the floor would pre-empt the obvious reviewer request.
