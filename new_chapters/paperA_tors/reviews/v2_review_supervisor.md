# Supervisor pre-submission read: `paperA_tors.tex` (v2) against `chapterA_v2.tex`

Date: 2026-09-08. Read: manuscript source (684 lines, 36 pp PDF), chapter source (1,780 lines),
`src/instrument/i26_encoder.py`, `train_i26.py`, `interview_curriculum.py`, `train_i26_smoke.py` (loss),
`experiments/instrument/t2i26c_run.out`, `docs/design/RETRAIN_DESIGN_SHEET.md`, `refs.bib`,
`plot_graded_curve_tors.py` and its banked input. Reviews/ and notes files not read.
Audit script: `scratchpad/audit_v2.py` (token diff + recomputations; output reproduced in §1.1).

Headline: **numbers are faithful (0 WRONG cells in any table; 1,114 numeric tokens, every one traceable), but
the new §5.1 architecture-and-training paragraph contains one factually wrong statement (the checkpoint
selection rule), one misdescribed curriculum, one omitted loss term and one overstatement, and it introduces
an internal contradiction with the R1 identity claim that must be resolved before submission.** Two
inherited chapter defects that a TORS reviewer will catch are also listed as must-change.

---

## 1. Number fidelity audit

### 1.1 Method
`audit_v2.py` strips comments, extracts every numeric token (signed decimals, `{,}`-thousands, integers)
from both files, and prints each manuscript token absent from the chapter. Manuscript: 1,114 tokens
(497 distinct); chapter: 1,119 (501 distinct). 44 manuscript tokens are absent from the chapter; every one
is either LaTeX geometry (column widths, `0.92\linewidth`), CCS concept ids, or a figure coordinate whose
value *is* a master-table cell written with a leading zero (`0.1486` vs the table's `.1486`), or one of the
declared-exception numbers (§7.1 split, §5.1 architecture). The tables were then compared cell by cell by
hand and the ordering/difference sentences recomputed.

### 1.2 Table of examined numbers

| number | manuscript location | chapter location / source | verdict |
|---|---|---|---|
| 0.3496/0.2491 vs 0.3540/0.2497; −0.0006 [−0.0020,+0.0008]; −0.0043 [−0.0059,−0.0028]; 1.2% | abstract, §1, §5.1, Tab. battery, §8.2, §10 | ch. abstract, §R1 para, Tab. battery | OK (0.0043/0.3540 = 1.21%) |
| 0.1279/0.0192 (item-interview intercept) | §7 protocol, §6.2 additive ablation | ch. R1 para, §concept:fold | OK |
| 0.1277/0.0267 (concept-protocol intercept) | §7 protocol, §8.3 opening, Tab. polar, "one concept answer clears" | ch. R1 para, polar para, criteria para | OK where used; **but see next row** |
| 0.1277/0.0267 as the intercept of the *signed-triple-gate* harness | §8.3 first para ("against this protocol's no-answer intercept of 0.1277/0.0267") | ch. §concept:results opening says **0.1279/0.0192** for the same harness sentence | REWORDED — chapter is internally inconsistent (its R1 para says concept curves carry 0.1277/0.0267; its results opening says 0.1279/0.0192). The manuscript picked one. Author must confirm which floor the +0.0104 / 0.1268 / 0.1372 arm was scored against; 0.1268 sits below both, so the claim survives either way, but the printed intercept must be the measured one. |
| +0.0039 full / +0.0020 tail, identical support, CIs | §2.2 R4, §3.3, Tab. battery, §8.2, §9 limits | ch. R4 item, R1 para, limits | OK; comparison correctly named every time |
| +0.0102 at k=2 [+0.0087,+0.0118]; +0.0080 at k=4 [+0.0063,+0.0097] | §2.2, §3.3, Fig. graded caption, Tab. battery | ch. Fig. gradedcurve caption | OK; banked JSON `premium_vs_k_t2i26c_ep31.json` confirms |
| +0.0022 [+0.0008,+0.0036] graded-vs-likes-only at full profile | §3.3, Fig. graded caption | ch. ratings para | OK; JSON confirms (0.3489 vs 0.3466 — note the figure's "full" point is 0.3489, not the 0.3496 headline, because it is the graded-history fold; no caption says so) |
| k=8 premium "+0.0010 [−0.0007,+0.0026]" | dropped; caption says "indistinguishable from zero at k=8 and k=16" | ch. caption prints +0.0010 | REWORDED-but-same (k=16 is −0.0014 [−0.0031,+0.0003] in the JSON; "indistinguishable" holds) |
| Master table, all 21 measured cells, both columns | Tab. master | ch. Tab. master | OK, cell-identical. Bolding changed: instrument row un-bolded (consistent with the new caption "best measured value per column"). |
| .3325/.2629, +0.0170 | master caption ♮ | ch. caption ♮ | OK (0.3495−0.3325 = 0.0170) |
| 0.397 vs 0.419 (Mult-DAE) | master caption § | ch. caption | OK |
| Inversion table, 36 cells + 10 CIs | Tab. instrdep | ch. Tab. instrdep | OK, cell-identical |
| "Ours − row" column vs its caption | Tab. instrdep caption | ch. caption ("each system at its own best strategy") | **Recomputed: manuscript caption is correct and more precise than the chapter's.** Full: ours(E0) − row's best full cell → .0293/.0105/.0409/.0423(printed .0422, rounding)/.0311 ✓. Tail: ours(E0 tail .0792) − row's tail *at the row's best-FULL strategy* → RecVAE .0792−.0534(pop)=+.0258 ✓ (would be +.0216 at its best-tail strategy); conj .0792−.0300(E0)=+.0492 ✓ (+.0454 at its best-tail); Golbandi .0792−.0792(HELF)=0 ✓. So "each system at its own best full-profile strategy, scored on both metrics" is exactly what the column is. |
| Graded-answers row 1.59 / 3.10 / 3.05; "1.6 of 8" | Tab. instrdep new first row + caption | ch. text para "what consumers disagree about" | OK (moved into the table; 1.6 and 1.59 both in chapter) |
| −0.0042 / +0.0042; 10,000 and 9,281 | §8.1 metric para | ch. fourth reading | OK (.1718−.1760, .0576−.0534) |
| Battery table, all cells | Tab. battery | ch. Tab. battery | OK; R2 row *adds* 0.626 / 0.416 / 0.498 from ch. "criteria exclude" para — REWORDED-but-same |
| 0.626 (like-lift AUC) | Tab. battery R2 row **and** §6.2 operator ablation ("the gated fold sits at 0.626 and reaches 0.1759") | ch. carries both too (R2 item and criteria para) | **SUSPECT (inherited).** §6.2's footnote says its numbers are the *earlier tower's*; the battery row attributes the identical 0.626 to the certified fold. Either the number is stale in one place or the same measurement is being cited for two towers. Verify which tower produced 0.626 and print the certified value in the battery. |
| Polar table: 12 deltas/CIs, −0.0782 [−0.0857,−0.0707], .1352/.0409, .1498/.0673, 4.8%, 97.2% | Tab. polar + §8.3 | ch. polar para | OK (new table, verbatim numbers) |
| NDCG@100 list (15 values) + "orders the frontier band and the first six rows identically, reorders only the middle (RBMF, EDDI, conjugate belief move)" | §7.3 | ch. (ii) para | OK. Recomputed: the manuscript reorders the chapter's misplaced "RBMF 0.3533, Turbo-CF 0.3569" into descending order (correct); first six identical to LIANG@10 ✓; RBMF, EDDI and conj are exactly the rows that move ✓ |
| RecVAE − EASE +0.0064 | §7.3 | ch. | OK (chapter's "1.8 unpaired SE" dropped — fine) |
| Most-Popular +0.028, EASE +0.066 (pool effect) | §3.1 | ch. protocols | OK (0.0281, 0.0659) |
| "roughly 0.12" uniformity cost | §4.1 and §5.2 | ch. §method says 0.12; ch. §gap:why says **0.15** | REWORDED-but-same — manuscript harmonised to the arithmetically correct value (0.3496−0.2267 = 0.1229). Good. |
| +0.0087 popularity-projection margin; 0.44 → 0.94 AUC; ≈0.5 cosine | §4.1, §6.2, §8.2 | ch. | OK |
| Free-text table: 0.47/0.39/0.54/0.54; 0.167, 0.0005, 0.638, 300×, 80%/20%, cos 0.6 | §5.2 | ch. | OK verbatim |
| SEL 0.376, +0.042, 0.398, 0.017, 22× | §6.1 | ch. | OK |
| Additive-union +0.011 at 2, 0.0991 at 8; gated 0.1759/0.0841, 0.1322/0.0503, 0.1685/0.0906, 0.1572/0.0777, AUC 0.820 | §6.2 | ch. §concept:fold + R2 item | OK (earlier-tower footnote retained) |
| −0.2012 concept sign flip; −0.2953 item | §6.4 | ch. §concept:fail; battery | OK (cross-reference added) |
| +0.0104 [+0.0075,+0.0134], 0.1268, 0.1372, −0.045, +0.0335 | §8.3 | ch. §concept:results | OK |
| 25.2%, 13.1%, 53.0%, 1,031 | abstract, §1, §8.3, §10 | ch. | OK |
| R3: 0.1574→0.2609, 0.0505→0.1062, +0.1035 [0.1000,0.1071], ρ=1.0; random-bank +0.0051; 0.79 vs 1.97 | Tab. battery, §8.2 | ch. | OK |
| 0.2358 vs 0.2230 at q=8; +0.0022 at q=16; +0.0036 [0.0016,0.0057]; +0.0015 [+0.0007,+0.0024]; +0.0016 [+0.0008,+0.0024] | §4.2, §8.2 | ch. | OK |
| Controls +0.0371, +0.0168, −0.0986 | Tab. battery | ch. | OK |
| Sign flip 0.4298→0.1345 (n=7,563); neutralisation 0.4198→0.1714, 0.2484 [0.2436,0.2532], tail 0.2999→0.0487; ρ 0.794/0.818 | Tab. battery | ch. | OK |
| Bridge: 9,990,682 / 116,677 / 20,108; 0.4203 vs 0.420; 0.4425 vs 0.442; 0.4144/0.414; 0.5525/0.553; 0.4223 vs 0.426 (−0.0037); 0.3928/0.395; 0.5341/0.537; 0.3969 vs 0.419; 0.358 vs 0.386 | §7.2 | ch. bridge | OK |
| Ratings decomposition (§3.3 + App. A): −0.0934/−0.0976; 0.3756 vs 0.2822; +0.0194 [+0.0157,+0.0231]; 2.26×; +0.0257; −0.0088; +0.0282; 0.070; −0.0045 [−0.0069,−0.0022]; +0.0046 [+0.0021,+0.0071]; 0.0981; 0.0499; +0.0454 [+0.0428,+0.0479]; +0.0601/+0.0281/+0.0267 | §3.3, App. A | ch. ratings paras + master caption ¶ | OK |
| TaNP 0.0060/0.0005; floor 0.1626; RMSE 0.696 vs 0.955 | §3.1 | ch. | OK (1,370/2,000, loss 0.862→0.697, 0.0111→0.0066, 0.0118 dropped — good cut) |
| 55.7% | §9 limits | ch. limits | OK |
| PEBOL 0.27 vs 0.17; d=50; 164 tags; 16 users, slates of 5; 40/75 keyphrases; 57k/46; EASE .420, Mult-VAE .426, RecVAE .442, SLIM .401, WMF .386 | §3, §4, §6.3 | ch. | OK |
| Split: 18,430 / 18,359 / 140,768 / 10,000 / 10,000 / 80/20 / seed 98765 / 190 head items / 9,281 tail users / top third of mass | §7.1 (declared exception) | project ruler (CLAUDE.md, `liang_split.py`); 9,281 and 18,359 also in chapter | OK, consistent with the project ruler |
| §5.1: d=200, ten half-star levels, width 512, 1.07 M trainable, warm-path smaller LR | §5.1 (declared exception) | `i26_encoder.py` (d_lat=200, NLEV=10, h=512), run log `TRAINABLE=1,073,147`, `warm_lr_scale=0.1` | OK |
| §5.1: "checkpoint was selected on the validation users by full NDCG@10" | §5.1 | run log + `docs/archive/PAPER_A_STATE_2026-08-04.md`: pre-registered rule is **argmax(val_full + sel)** where sel = mean of the simulated k=2 and k=8 interview NDCG@10 on validation; ep31 wins it (0.3466+0.1407=0.4873, run max). ep31 is *not* the best val full (ep6 0.3479) nor the best sel (ep30 0.1408). | **WRONG** |
| §5.1: curriculum "full profiles, dropout-thinned profiles, short interviews of at most eight answered items in which unanswerable questions arrive as refusal tokens, and empty or single-answer sets" | §5.1 | run log `regime mixture full=0.25 DENSE-SMALL=0.3 interview=0.35 k0=0.05 k1=0.05`; `interview_curriculum.py`: the dense-small regime is k≤8 items from the user's own rated history, *all answered, no refusal tokens*; the interview regime draws strategy-selected budgets from {1,2,4,8,16,32} and is where refusal tokens occur | **WRONG** (two regimes conflated; interview budget is up to 32, not 8) |
| §5.1: objective "minimises the multinomial NLL of each user's held-out target items" | §5.1 | `train_i26_smoke.loss_vec`: NLL on held-out likes **plus** `W_NEG=0.1` × clamped (−8) mean log-probability of the user's held-out dislikes; design sheet §6 names both terms | INCOMPLETE — the dislike penalty is omitted |
| §5.1: "at initialisation the encoder is the frozen substrate exactly" | §5.1 | `forward`: at init z = a(n)·RecVAE_mean(likes), a(n)=1−exp(−softplus(−1.2586)·n) ≈ 1−e^{−0.25n}; ≈1 on a full profile, 0.22 at n=1 | OVERSTATED — exact only in the large-n limit; the anchor gate shrinks small sets (the run log's own phrase is "bit-identical to i25 at step 0", not to RecVAE) |
| §5.1: "learned prior latent supplies the belief with no evidence" vs battery/§8.2 "$|\bmu_0|=0$ to numerical precision … Spearman 1.0000" | §5.1 vs Tab. battery, §8.2, §4.2 | `z0 = nn.Parameter(zeros)` trainable in the NEW group at lr 3e-4; the empty-set output of the certified tower is z0, which need not be zero after 31 epochs | **INTERNAL CONTRADICTION introduced by the new paragraph.** Either z0 stayed at zero (then say "remains at zero after training, measured") or the identity claim holds only rank-wise (Spearman 1.0) and $|\bmu_0|=0$ is false for ep31. Must be re-measured on `t2i26c_ep31.pt` and one of the two sentences amended. |
| §5.1: "One run was trained" | §5.1 | log: one run (tag `t2i26c`), 35 epochs, resumed once at ep33 by the watchdog. Earlier runs of *other* configurations existed (i25-init run1; a warm_lr 0.01 run, deleted). | OK as written; safer as "one run of the pre-registered configuration" (which is what the chapter says) |

**Counts.** WRONG: 2 (both in §5.1: selection rule; curriculum description). INCOMPLETE/OVERSTATED: 2 (§5.1 loss; §5.1 init). NOT-IN-CHAPTER numbers: 0 outside the two declared exceptions (all §7.1 and §5.1 numbers check against the ruler and the code). Chapter-inherited suspect numbers: 2 (the 0.626 double attribution; the harness intercept in §8.3's first sentence).

### 1.3 Ordering sentences recomputed
* NDCG@100 sentence: holds (see table).
* "the six rows realise three distinct orderings" — full column: E0>P>H (ours, TaNP), P>E0>H (RecVAE), H>P>E0 (Golbandi); RBMF is E0≈P>H (unseparated, not a fourth), conj flat. Holds.
* "on the tail HELF is first for two of the six consumers" — Golbandi (.0792) and conj (.0338). Holds.
* "the strongest HELF cell in the table" (.1948) — holds; "its own strongest cell" — holds.
* "under CASPER entropy0 leads on both metrics; under Golbandi HELF leads on both; under likes-only RecVAE popularity leads on the full profile" — holds (RecVAE tail: E0 leads, and the text says "full profile").
* "RecVAE ahead of EASE by +0.0064" — holds.

---

## 2. Claim fidelity

Qualifiers checked and **retained**: "a selection result we report for completeness and do not claim" (§8.2); the tail non-win against the Golbandi HELF arm (Tab. instrdep caption *and* §8.1 "matches it exactly on tail"); "This is not a controlled attribution" (§8.1); "It does not show that concepts beat items in general" (§8.3); "We do not claim the heuristics are wrong, or that ours is the right consumer" (§8.1); "not a demonstration of equivalence" (Def. 1); "We are quantifying a property of generality, not discovering an advantage" (§8.3); "we make no claim about why" (§3.3).

Sentences that say something the chapter does not, or drop a qualifier:

1. **§1 (intro) and abstract:** "our reconstruction of the node model of Golbandi et al., *a neighbourhood-style consumer closest in kind to the user-based models the heuristics were validated on*". The chapter says only "the model closest to what the heuristics were validated on". "Neighbourhood-style" and "user-based models the heuristics were validated on" are new characterisations; Rashid et al. 2008 should be checked before asserting what recommender they validated on. Keep only if literally true of Rashid 2008.
2. **§3, master-table caption:** "R3 is marked met only where the system maintains a belief or posterior that its published update rule keeps consistent with a new answer." New sentence. By that rule ConTS (Thompson posterior), EDDI (Partial-VAE posterior) and PEBOL (per-item Beta) all maintain an answer-updated posterior, yet all three carry ✗ on R3, while the gap-analysis bullets (§4, inherited from the chapter) *credit* each of them with R3 ("ConTS: R3 plus a unified action space"; "EDDI: R3 (information gain over a belief)"; "PEBOL: R3 (per-item Beta)"). The table and the bullets contradict each other on R3 for three systems, and the added caption sentence makes the contradiction explicit rather than resolving it. This is a chapter defect the manuscript now advertises.
3. **§4 Bıyık bullet:** "(c) exact rather than approximate conjugacy *where a belief is maintained*" — inherited from the chapter's "(c) exact conjugacy vs. approximate posterior". The instrument maintains no belief (§2: "Nothing in what follows requires a distribution over u"; §3.2: "We do not maintain such an accumulator"). Separation (c) is not a property of the system described. Drop (c).
4. **§7.1 Statistics paragraph** (new): "The claims that rest on a single interval are the R1 contrasts …; every other claim in the article rests on directional consistency across budgets, metrics and controls rather than on one cell." Not in the chapter and not quite true: each deciding gap in Tab. instrdep is a single interval at one budget (k=8) and the RQ1 answer rests on those. Reword to what is true (the RQ1 gaps are reported with intervals and the *pattern* across six consumers is the claim).
5. **§3.1 TaNP native cell:** "and over training its loss fell while its validation NDCG@10 drifted down" — chapter gives the numbers; manuscript keeps the claim without them. Acceptable, but a reviewer may ask for the numbers; consider restoring "0.862→0.697 while 0.0111→0.0066".
6. **§3.2:** "*Not run here*: GF-CF, BSPM and BERT4Rec" — the chapter listed EDDI here too, wrongly (EDDI has a measured row). The manuscript silently corrected it. Good; no action.
7. **Abstract/§1/§10:** "under (the three) answerability-weighted rankings the item channel wins full accuracy" — chapter says "under the other three rankings the item channel wins full accuracy decisively". Calling ENTROPY "answerability-weighted" is a stretch (it is entropy over the four-way outcome including refusal, so it is answerability-*sensitive*). Minor; "answerability-sensitive" or "the other three rankings" is safer. **Separately, a supervisor's caution outside the chapter-verbatim rule:** the project record of 2026-08-12 (`CONCEPT_VS_ITEM_BANK_SWEEP.md`, memory note `certified-pair-and-cv5-null-result`) states that on the policy-B contract the secondary claim "off POLAR items win full decisively" is false at short budgets (entropy flips positive at m=1,2; pop/expinfo become n.s. at m=2,4) and that item full-axis dominance is confined to m≥8. The chapter still carries the earlier contract's wording. If the chapter is not going to be re-cut, the manuscript should at least confine the sentence to what its own printed cell supports ("at m=16 under POP, −0.0782") rather than "wins full accuracy decisively" at every budget. This affects the abstract, §1 contributions, §8.3 and §10.
8. **§8.2 ("Within an answerable bank…"):** "closing the gap is that study's job" — chapter says "the companion method chapter … the decision-tree acquisition line of Golbandi et al. on this instrument". The manuscript's version is fine, but "that study" has no antecedent in the sentence; say "a policy study's job".
9. **§5.1 (new):** see §1.2 — the selection-rule sentence makes a claim the run record contradicts; the identity sentence conflicts with the learned prior.

No dropped qualifier was found among the four the brief lists; all four survive verbatim or near-verbatim.

---

## 3. What was lost (chapter content absent from the manuscript)

| chapter content | judgement |
|---|---|
| Chapter abstract's "the field has largely reported the pair as though it were the strategy" | neutral (the intro carries the idea) |
| Contributions item 3's explicit "Freezing is *not* what separates us from the soft-attribute line" | neutral — §5.1 retains "a design choice we share with the soft-attribute line" |
| "What each table is for" paragraph | good cut |
| TaNP diagnostics: 1,370 distinct top-10s over 2,000 users; loss 0.862→0.697 vs NDCG 0.0111→0.0066; adversarial selection 0.0118 | neutral (see claim 5: consider restoring the loss/NDCG pair, it is the evidence for "the better it became at its own objective, the worse it ranked") |
| Master-table caption's "RBMF's graded seed on raw ratings comes within −0.0045 … centred variant loses a further 0.0981; user-kNN's centred build loses −0.0934" | neutral — moved to §3.3 and App. A |
| "+0.1032 for the instrument, +0.0282 for iALS" re-valuation against an every-answer-is-a-like control, and why it is not a headline | neutral, arguably a good cut; but the appendix still uses the iALS +0.0282 step, so the instrument's +0.1032 is the only number of that pair now unstated — consistent |
| "One row now also shows what the column was built to detect" (§3.3) | neutral — retained in App. A's last sentence |
| Gap analysis "Prior art, cited and not claimed" paragraph | neutral — §1 last paragraph carries the list |
| Chapter §gap:why "forfeits 0.15" | good cut (it was wrong; 0.12 is right) |
| §method "The tower is now selected and certified: T2′ … pre-registered configuration" | neutral (replaced by §5.1; but the phrase "pre-registered" was dropped — see §1.2 last row) |
| The critiquing "eight systems … without exception" count in the three-legs paragraph | good cut (count was not defensible from the list given) |
| Answerability section's "diagnosed twelve years earlier and more sharply"; "Both report the crossover we report" | neutral; the manuscript's Sun et al. paragraph keeps the substance |
| §concept:results "What we deliberately do not state … constructing *optimal* question sequences per bank would answer it, but that is a selection result" | neutral — §8.3 keeps the first half; the explicit "selection result" reason is gone. Consider restoring the clause; it is the article's cleanest statement of its scope boundary. |
| "The methodological consequence for verification design" paragraph: the story that the sign-flip *passed* on a fold whose answer model never emitted a negative, and only deployment currency exposed it | **must restore, condensed.** §6.4 keeps the abstract lesson ("a gate on what an operator can do must be paired with a gate on what the deployed system does") but drops the one concrete instance that justifies the whole deployment-currency apparatus. Two sentences suffice; without them §6.4 reads as a principle without evidence. (Keep it as a design fact, not as a history of a broken build.) |
| "The concept fold sits on the frozen tower, and inherits R1 by construction" | neutral — retained as §6.4's last paragraph |
| Experimental design (iii) "Symmetric access is enforced where a baseline can fold a channel; where it structurally cannot, that is reported as an architecture limit, not hidden" | neutral — §7.5 keeps the conclusion |
| Limitations: "Metric non-comparability until the bridge is run" bullet; "BERT4Rec replicability" and "Graph-filter admissibility" bullets | neutral — merged into "Bridged and unbridged rows" |
| Limitations: "we report any such step rather than suppress it" (R3) | neutral |
| Chapter R1 para: "RecVAE leads EASE by +0.0064 full, about 1.8 unpaired standard errors, the same order as the tower's own −0.0043 separation" | neutral; the "narrow band" point survives |
| Chapter's "Five readings, and we are careful about which we claim" framing | good cut |

New in the manuscript and welcome: Tab. polar; the graded-answers row in Tab. instrdep; the corrected "Ours − row" caption; the fuller gap map; the Discussion section (§9 "three practices" + "Certifying another consumer" recipe); the "One training run" and "concept channel depends on curated member lists" limitations; the statistics paragraph (needs the wording fix above).

---

## 4. Attribution

* Full-profile rows: **user-kNN [golbandi2011adaptive]** with ‡¶ and the text "the node recommender that is the Golbandi tree's core" — as the chapter. OK.
* Interview consumer: "Golbandi node model [golbandi2011adaptive] ‡" with ‡ = "best-effort core of the published system" — as the chapter (the chapter table did not carry ‡ on that row; the manuscript adds it, which is more honest). OK.
* RBMF [liu2011wisdom, zhou2011functional] ‡¶ — as the chapter. OK.
* Bıyık and ConTS: dashes in the master table, with the stated reason ("measuring our own reconstruction of a mechanism they share would produce a number that is ours, not theirs"). OK.
* Conjugate belief: "Conjugate belief (frozen MF), after Bıyık et al." in Tab. instrdep; "Conjugate belief (frozen MF)" in Tab. master — no rival's name is attached to a reconstruction as if it were their result. OK. One inconsistency: the master-table row has no citation at all while the interview row says "after Bıyık"; harmless, but the two should agree.
* TaNP: "objective-ported implicit adaptation" with ‡∗ and the §3.1 explanation. OK.
* Bibliography spot-checks (author lists against my knowledge):
  * `golbandi2011adaptive` Golbandi, Koren, Lempel (WSDM 2011) — correct.
  * `rashid2008learning` Rashid, Karypis, Riedl (SIGKDD Explorations 2008) — correct.
  * `sun2013learning` Sun, Li, Lee, Zhou, Lebanon, Zha (WSDM 2013) — correct.
  * `lin2021tanp` Lin, Wu, Zhou, Pan, Cao, Wang (WWW 2021) — correct.
  * `ren2020recvae` Shenbin, Alekseev, Tutubalina, Malykh, Nikolenko — correct (key is a misnomer; harmless).
  * `he2023large` He et al., CIKM 2023 — correct.
  * `toroghi2023bcie` "Bayesian Critique-Improve-Explain …", Toroghi & Sanner, SIGIR 2023 — **verify.** My recollection of the SIGIR 2023 Toroghi–Sanner line is a four-author paper with a different title ("Bayesian Knowledge-driven Critiquing with Indirect Evidence"); I am not confident the title/author pair printed here is a real venue entry. Check against the ACM DL before submission.
  * `gharahighehi2025pairwise` — Gharahighehi, Nakano, Vens are real KU Leuven authors; "Cu, Hai" looks like a truncated or garbled name. Verify against arXiv 2510.27342.
  * `xia2026when` — "Xia, Yu and Zhang, Wei and Wang, Jian", arXiv 2607.06765 (July 2026, after my cutoff). Three maximally generic names and no venue: I cannot confirm this entry exists. It is cited for a substantive claim ("attributes dominating the opening turns … items the later ones"). Verify the arXiv id resolves to that title and those authors; if it does not, drop the citation and the clause.
  * `wang2025bdecf` key vs. authors "Cheraghi et al." — key mismatch, harmless.

---

## 5. Style rulings

* **"parity"/"tie":** "frontier-parity" is the requirement's *name* (R1) and Def. 1 does the work; acceptable. Two fixes: §9 "Bridged and unbridged rows": "the *parity* claim rests on the bridge" → "the R1 claim". `\label{def:tie}` is internal only. Nowhere does the body call the tail result a tie or parity; good.
* **"this chapter"/"companion chapter":** none found. Good. (One "that study's job", §8.2 — see claim 8.)
* **CASPER-R/CASPER-U:** none. Good.
* **em-dashes in body prose:** none (`---` occurs only as table dashes). Good.
* **"adaptivity does not pay":** absent; §8.2 says "establishing that adaptivity pays at all … is what an instrument owes a policy study". Good.
* **Performed candour:** "We claim no more than the account supports" (§8.1), "We are precise about which of these is impossible" (§4), "we deliberately do not state" (§8.3), "The same rule applied to ourselves costs something" (§7.2), "including the arms where the number is unflattering" (§9), "we report every item" (§1). Individually fine; together they are a tic. Cut the first, third and last; keep the bridge one (it is a rule with a consequence).
* **Error archaeology / lab notes:**
  * App. A last paragraph: "shared one hyperparameter setting selected by a validation sweep that optimised NDCG@100 on a *previous dataset*, while this article reports NDCG@10" — that is a history of a stale sweep. State the fact ("re-selecting each row's hyperparameters on the reported metric moves RBMF by +0.0601 …") without the archaeology.
  * Tab. battery Controls: "duplicate assertion fired" → "duplicating an answer leaves the ranking unchanged (asserted in code)".
  * §7.3: "EDDI early-stopped at epoch 46 with epoch 31 validation-selected" is a config note; acceptable in a per-row configuration list but "early-stopped at epoch 46" adds nothing — "epoch 31, validation-selected".
  * §6.2 footnote "The operator ablation was run on an earlier tower" — necessary; keep, but it is the one place a reviewer will ask "why not re-run on the certified tower?" (it is cheap). Consider re-running or saying why not.
  * §5.1 "One run was trained; the checkpoint was selected…" — fine in form once the rule is corrected.
* **Undefined internal vocabulary:** "CI-clean" (§8.2, Tab. battery) → "with a confidence interval excluding zero"; "the fixed stack" (§8.2) — undefined; "arm" is used throughout without definition (acceptable in RecSys, but define once); "SEL/VAL band" and "top-SEL concepts" are defined in §6.1; "deployment currency" defined; "existential controls/leakage" — "existential" is project jargon; say "answer-identity controls" or define; "a canary" was removed (good); "no popularity floor" defined in §7.5; "the R3 curves" fine.
* **"first"/"novel":** "the one leg of the conjunction we find nowhere in the literature" (§4), "no published system we are aware of", "Scope of the novelty claim" (§9) — all hedged; defensible. No bare "first".
* **Lab-note residue:** "[repository URL to be added at submission]", "[Affiliation to be completed]" — placeholders, fine. `\acmDOI{XXXXXXX.XXXXXXX}` fine for review.
* **Other:** "Six consumers realise three distinct orderings" (§1) is used before the reader knows what a consumer is; the abstract's "distinct orderings" likewise. Minor.
* **Figure:** Fig. graded panel (a)'s full-profile point plots 0.3489 (the graded-history fold), not the 0.3496 headline; a reader comparing figure to table will see a discrepancy with no explanation. One clause in the caption fixes it.

---

## 6. Verdict

**Not as it stands.** The empirical content is faithful to the chapter and the manuscript is, on the whole, a cleaner and more precise document than the chapter (the corrected "Ours − row" caption, the polar table, the NDCG@100 ordering and the 0.12 figure are all improvements). What blocks submission is the new §5.1 paragraph, which was written from the code and got the selection rule wrong, the curriculum wrong, the loss incomplete, and which introduces a contradiction with the R1 identity claim — precisely the paragraph a TORS reviewer will use to judge whether the checkpoint was chosen honestly. Fix those, resolve the two inherited defects a reviewer will catch (R3 marks; the 0.626 double attribution), and it can go.

### Must-change (ordered)
1. **§5.1 selection rule.** Replace "selected on the validation users by full NDCG@10" with the rule actually pre-registered and applied: the checkpoint maximising, on the validation users, the sum of full-profile NDCG@10 and the mean simulated-interview NDCG@10 at k=2 and k=8 (ep31: 0.3466 + 0.1407). Say it once and plainly; a reviewer who reads the log will otherwise conclude the paper misreports its own model selection.
2. **§5.1 vs R1 identity.** The paragraph says the empty-set belief is a *learned* prior latent; the battery says $|\bmu_0|=0$ and Spearman 1.0000 against the decoder bias. Re-measure both on `t2i26c_ep31.pt`. If z0 is non-zero, the identity claim must be restated as what holds (rank identity, or nothing) and Tab. battery, §4.2, §8.2 and §10 amended; if z0 is zero, say "the prior remains at zero after training (measured)".
3. **§5.1 curriculum and loss.** Name the five regimes as trained (full/dropout-thinned profiles; dense sets of up to eight rated items, all answered; strategy-selected interviews with budgets up to 32 in which unanswerable questions arrive as refusal tokens; empty sets; single-answer sets) and add the dislike term to the objective ("plus a small penalty on the log-probability of the user's held-out dislikes"). Soften "is the frozen substrate exactly" to what the code does (the frozen encoder's mean, scaled by an anchor gate that saturates on a full profile).
4. **R3 column vs gap bullets.** Decide whether ConTS, EDDI and PEBOL hold R3. Either mark them ✓ in Tab. master (and move their gap-map x accordingly) or strike "R3" from their §4 bullets. Then keep or delete the new caption sentence so that it matches the table.
5. **The 0.626 AUC.** Determine which tower produced it. Print the certified fold's like/dislike/other-concept AUCs in Tab. battery; leave the earlier tower's number in §6.2 only if it is genuinely a different measurement that happens to coincide (say so) — otherwise remove one.
6. **§8.3 first-sentence intercept.** Confirm which floor (0.1279/0.0192 or 0.1277/0.0267) the signed-triple-gate arm (+0.0104, 0.1268, 0.1372) was scored against and print that one; the chapter carries both and the manuscript chose without a record.
7. **Restore, in two sentences, the concrete instance behind deployment currency** (§6.4): a fold that passed the sign flip while its deployed answer model never emitted a negative. Without it the section is a principle with no evidence.
8. **Bibliography:** verify `xia2026when`, `gharahighehi2025pairwise` (author "Cu, Hai") and `toroghi2023bcie` against their sources; drop any that do not resolve, together with the clauses that cite them.
9. **Scope the "items win full accuracy off POLAR" claim** (abstract, §1, §8.3, §10) to the cell printed (m=16, POP) unless the chapter's sweep is re-cut; the project's own later record contradicts "decisively at every budget".

### Would-strengthen (ordered)
1. Drop separation (c) "exact conjugacy" from the Bıyık bullet; the instrument maintains no belief.
2. Reword the §7.1 statistics sentence about which claims rest on a single interval (the RQ1 gaps do).
3. Remove "neighbourhood-style … user-based models the heuristics were validated on" from §1 unless checked against Rashid 2008.
4. Re-run the operator ablation on the certified tower (cheap) so the §6.2 footnote can go.
5. Add one clause to Fig. graded's caption explaining why panel (a)'s full-profile point (0.3489) differs from the 0.3496 headline.
6. Restore the TaNP loss/NDCG pair (0.862→0.697 vs 0.0111→0.0066); it is the actual evidence for the anticorrelation sentence.
7. Replace "CI-clean", "fixed stack", "existential", "assertion fired"; define "arm" once.
8. Trim the candour tics (§8.1 "We claim no more than…", §1 "report every item", §8.3 "deliberately").
9. App. A: state the re-selection result without the "previous dataset" archaeology; write "one run of the pre-registered configuration" in §5.1.
10. Make the conjugate-belief row's attribution identical in both tables.
11. "parity claim" → "R1 claim" in §9; "that study's job" → "a policy study's job" in §8.2.
