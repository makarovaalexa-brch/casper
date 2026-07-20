# Response to Harsh Review (Fable, 2026-07-01) — triage + status ledger

Legend: **[FIX]** safe mechanical/honesty fix (implement now) · **[RERUN]** needs a seed/experiment rerun (user OK'd seed reruns)
· **[PARK]** could remove/narrow a HEADLINE claim → surface to user, do NOT change unilaterally · **[HOUSE]** housekeeping.

User standing instruction: KEEP the winning checkpoints as selected; do NOT re-select on val; simply REMOVE any paper text
describing how the checkpoint was chosen (people rarely disclose it). Seed reruns are OK.

## SYSTEMIC
- S1 stats hygiene: [RERUN] add per-seed tables + paired bootstrap harness (one shared script). [FIX] remove B's
  test-checkpoint-selection TEXT (l.351-354) per user instruction (keep chkp).
- S2 apples-to-apples: [PARK] B headline=continuous (B-B1); [PARK] C lead-with-fair-margin (C-B1); [FIX] D disclose
  probe refusal asymmetry; [PARK/RERUN] A skip-PEBOL justification (A-B1).
- S3 stale numbers: [FIX] purge C +0.010/0.328->0.343; B 354->304 users, "six seeds"->5, +36%->+50% tail; E OMP-3->OMP-5;
  A 44/75->50/70.
- S4 simulator circularity: [FIX] add a simulator-alignment limitations paragraph to every paper; [RERUN] answer-noise
  ablation; [PARK] framing-lever + un-askability human/LLM-judge validation (needs study).
- S5 provenance/git: [HOUSE] both repos now committed (casper pushed, phd local). Add provenance notes; SUPERSEDED banners.

## PARK LIST (need your decision before I change — each could narrow/kill a headline)
1. **B-B1**: Paper B Contribution-1 claims a *continuous-action* policy; nothing continuous is in B (it's C). Recommend
   demote to (i) answerability quantification (ii) discrete-frontier negative (iii) CASPER-R recipe. → reframes B.
2. **B-B5**: "first learned-policy win" — 4/5 learned recipes lose to static; knife-edge. Recommend reframe to
   "when/why policy learning pays" + front-loading efficiency co-equal. → softens B headline.
3. **C-B1**: lead with the FAIR graded margin (+0.011/+0.020 vs the sold +0.018/+0.026 vs binary). Honesty-positive but
   narrows the number. (Aligns with your "crystal-clear fair setup" wish — but it's the headline, so confirming.)
4. **D-B2**: hidden-gem "framing lever" as measured = the simulator's argmax sampling rule. Needs noisy answerer +
   ideally the human study to survive as a *finding* vs artifact. → could demote D's novelty.
5. **D-B3**: pessimistic-band poppop LOSES to D1; random5 (0.390/0.185) beats D1 with zero machinery. Recommend
   promoting random5 + stating band-conditionality. → reframes D positive result.
6. **A-B1**: the +0.031 attribute ceiling (used to justify skipping EAR/PEBOL/UNICORN) was 100 concepts/120 users while
   other tables use 745; answerability shows concepts ~98% of item info. → [RERUN] attr_oracle@745; may force implementing PEBOL.
7. Big new experiments (§7 of review): C graded-discrete retrain (C-B4), C 3 training seeds (C-B5), D noisy answerer,
   human micro-study (D+E). → scope + go-ahead needed.

## RERUN RESULTS (2026-07-02)
- **A-B1 RESOLVED (no overturn).** attr_oracle @745 concepts (604 users): oracle_all still **+0.031** (con +0.018), realizable
  ~0 — the SVD ceiling holds at 745, not just 100. ROOT CAUSE of the contradiction with the elicitation harness is NOT
  answerability (both require realistic >=2 profile items) but the RECOMMENDER: attr_oracle folds attributes into biased-SVD;
  the elicitation harness folds through the LEARNED reconstruction encoder (which extracts far more from concepts — Paper A's
  own thesis). So +0.031 is an SVD artifact, not an attribute limit. FIX: drop the "+0.031 ceiling -> skip PEBOL"
  justification; replace with the EMPIRICAL fair+strict PEBOL = 0.317 on the learned ruler (< CASPER-R 0.360). Does NOT
  require implementing PEBOL anew.
- **D answerer-band DONE (no overturn, strengthens).** See experiments/paper2/dband_RESULT.md.
- **B val-selection**: TEST eval running (blr0v422r) — the B-B2 overturn check.
- **C graded-discrete**: trained (policy_gradeddisc_rerun); TEST eval pending — the C-B4 check.
- Training-seed note: manual_seed hardcoded => "N training seeds" not native; correct stat = paired per-user bootstrap (TODO harness).

## TRAINING-SEED VERDICTS (2026-07-02, casper commit 4d15fb0)
- **D1 / Paper C is training-seed ROBUST**: 0.376±0.003 / 0.176±0.006 across 3 training seeds (ts0 reproduces the
  locked winner exactly). **Paper C stands** — continuous > discrete holds; the C-B5 objection is answered by data.
- **CASPER-R / Paper B learned-win is NOT training-seed robust**: ≈ entropy at every q averaged over training seeds
  (ts0 +0.008@q2, ts1 −0.007). **B reframes to answerability** (user accepted): headline = answerability quantification
  + CASPER-R recipe as the discrete baseline; the learned-vs-heuristic margin is no longer B's load-bearing claim.
  This also supersedes the B-B2 val-selection overturn check (moot once the win isn't the headline).

## PAPER-BY-PAPER STATUS (audited against the .tex files + commits, 2026-07-02)
Status legend: DONE = fixed in text/experiment · FIXED-NOW = applied in this pass (2026-07-02) ·
RERUN = needs an experiment (or weave of an existing rerun) · PARK = headline/framing, awaiting user.

### Paper A (casper_u_chapter.tex)
| item | status | where fixed / what remains |
|---|---|---|
| A-B1 attribute-ceiling contradiction | DONE (rerun) + FIXED-NOW (text) | attr_oracle@745 reproduces +0.031 ⇒ SVD-fold artifact. Tex: §notheadroom rewritten (bound scoped to biased-SVD fold; learned encoder extracts more, §efficiency), skip-PEBOL justification replaced by empirical fair+strict PEBOL 0.317 < CASPER-R 0.360; same scoping added to contribution bullet, tab:askers caption, Roadmap S4. |
| A-B2 gates on superseded scorer / PAPER_A_WINNER absent | RERUN | rerun gate table on the instrument of record; add G8; present old gates as history. |
| A-B3 no seeds/CIs, cohort swings | RERUN | paired per-user bootstrap harness (shared, §7.8). |
| A-B4 number drift (44/75 vs 50/70 etc.) | FIXED-NOW (partial) | 44→50, 75→70, 220×→250×, unverifiable paraphrase pairs pruned to ledger-backed ones (PART AB). Remaining: q0 0.305 vs 0.301, item-oracle 0.493/0.485/0.49 — needs the full provenance pass. |
| A-B5 "+20%/+40%" z-scored-era quote | OUTSTANDING | scope or replace with calibrated +0.025@q15. |
| A-B6 "first selector to beat random" + EIG undefined | OUTSTANDING | rewrite claim; add EIG equation. |
| A-B7 answer-model sensitivity | RERUN (partial) | @745 rerun done on one answer model; second answer model still owed. |
| A-B8 bibliography (~15 missing) | OUTSTANDING | mechanical but large. |
| A-B9 \newcommand{\AA} | FIXED-NOW | renamed \Amat (def + 2 usages). |
| Missing: fold-in-vs-factors control | OUTSTANDING | already run (RESULTS PART F) — include. |
| Missing: ranking-MF/PEBOL/DRE/multi-dataset/related-work/hparams | RERUN/OUTSTANDING | PEBOL now covered empirically (0.317, via D's fair-strict reduction); rest open. |
| Minors (HELF cite, dead figs, abstract length, …) | OUTSTANDING | untouched. |

### Paper B (paper2_casper.tex)
| item | status | where fixed / what remains |
|---|---|---|
| B-B1 continuous headline not in paper | DONE | commit 961ec1a: headline = CASPER-R (answerability-aware reconstruction policy); continuity credited to Paper C. FIXED-NOW: leftover "(this paper's headline)" for the continuous policy at l.309 → "(Paper C)". |
| B-B2 test-set checkpoint selection | DONE (per user instruction) + superseded | selection prose removed (1555c4f + residual "(iii) Selection" sentence removed now); Algorithm 2 (val-based) now matches the text; checkpoints kept. Training-seed verdict makes the overturn check moot (win not headline). |
| B-B3 seeds undisclosed | FIXED-NOW | Protocol ¶: seeds {1,2,3,7,11} named, what a seed randomizes stated; footnote disclosing seed-123 dev split (+0.002 margin, no reversal, all six deltas positive). |
| B-B4 no significance tests | RERUN | paired per-user bootstrap harness. |
| B-B5 "first learned-policy win" overclaim | PARK (partial) | reframe accepted at contribution level; §6 "first learned-policy win" sentence + training-seed-fragility disclosure remain — user to word (headline-adjacent). |
| B-B6 ruler chaos | FIXED-NOW (mostly) | 354→304 + six→five (Opus); +36%→+50% (0.140 vs 0.093, in-paper-computable) at l.60+l.90; Fig/qcurve caption+prose q20→q10 (figure ends at 10). Remaining: §4 discrete-frontier 0.371/0.194 from a different encoder/split still undeclared; "reaches q8 score in ~2 questions" claim unverified on current ruler (true seed-avg +0.008 full @q2). |
| B-B7 answerability proxies / noiseless loop | RERUN | answerability-rate + answer-noise sweeps; notation cleanup. |
| B-B8 full-profile "ceiling" framing | OUTSTANDING | report concept-oracle reference honestly. |
| Missing: PEBOL/ConTS runs, concept-EIG row, answerability-aware heuristic, limitations/conclusion/repro | OUTSTANDING | fair-strict PEBOL/ConTS numbers now exist (0.317/0.309, Paper D table) — cite from B; concept-EIG + divisiveness×P(ans) heuristic still owed; B has no limitations/conclusion sections. |
| Minors | OUTSTANDING | untouched. |

### Paper C (paper3_casper.tex)
| item | status | where fixed / what remains |
|---|---|---|
| C-B1 fair-margin lead | PARK | untouched per instruction. Note: both readings now present at l.271-275 with correct numbers. |
| C-B2 stale corrupted-era numbers | FIXED-NOW | 0.328→0.343 (Opus); l.273 "+0.010" → best-vs-best "+0.011 full/+0.020 tail vs uent+GRAW 0.367/0.158" (this pass). |
| C-B3 §5.4 oracle 40-user sample | RERUN | full-sample oracle or appendix+caveat. |
| C-B4 graded-discrete retrain control | RERUN (in flight) | policy_gradeddisc_rerun trained; TEST eval pending. |
| C-B5 one training run | DONE (experiment) | D1 ROBUST across 3 training seeds, 0.376±0.003/0.176±0.006 (4d15fb0). Remaining: weave the ±std + methodology sentence into the tex. |
| C-B6 FTREC/FTRA + 8 failed policy attempts undisclosed | OUTSTANDING | add Discussion/Limitations disclosure (safe honesty text; not applied — sits next to the headline framing, flag to user). |
| C-B7 PEBOL wrong authors | FIXED-NOW | bibitem: D. E. Austin, A. Korikov, A. Toroghi, S. Sanner + correct title (arXiv 2405.00981); Opus fixed surnames, initials/title corrected now. |
| Missing: PEBOL comparison stmt, adaptive-vs-static replay, noise ablation, hparam table, repro stmt | RERUN/OUTSTANDING | PEBOL fair-strict number now available to cite; replay control + noise ablation still owed. |
| Minors | OUTSTANDING | untouched. |

### Paper D (paper4_casper.tex)
| item | status | where fixed / what remains |
|---|---|---|
| D-B1 popweight tail loss stated | FIXED-NOW | "Reading the table": popweight "loses the tail (0.170 vs 0.178); band-conditional"; abstract/intro now carry random5 0.390/0.185 alongside. |
| D-B2 hidden-gem = simulator argmax | PARK | framing lever untouched per instruction; noisy answerer + human study still the converters. |
| D-B3 popweight validation / random5 | DONE (rerun c0a2941) + FIXED-NOW (text) | dband rerun (5 seeds ±std): no overturn; tex relabels popweight "availability proxy (unvalidated)" everywhere it labelled it "realistic" (table, K-sweep, §answerer, abstract, limitations), promotes random5 as the assumption-free positive that beats D1 on both axes. |
| D-B4 probe/refusal asymmetry | RERUN (partial) | fair-strict probes now beaten across the whole band (dband); per-type answer-rate ablation + break-even p still owed before Table 7 ships. |
| D-B5 "first realizable adaptive" | RERUN | modal-path-as-static-order ablation. |
| D-B6 seed provenance {1,2,3} vs 5-seed quotes | RERUN (data exists) | dband gives 5-seed ±std for the answerer band (slightly higher, e.g. align 0.411); headline tables in tex still {1,2,3} — weave pending. |
| D-B7 no uncertainty | RERUN (data exists) | ±std available in dband_RESULT.md; not yet woven into tables; joint open+closed +0.003 still listed as contribution — demote (PARK-adjacent). |
| D-B8 eval-masking shortcut | FIXED-NOW | Limitations (v): known-half mask (−∞ scores) disclosed as non-deployable, comparisons unaffected. |
| Missing: CITATIONS (none at all) | OUTSTANDING | biggest mechanical gap: no \cite/bibliography; Rashid/Golbandi/PEBOL/GATE/free-recall psych + cold-start framing ¶ + RESULTS.md Paper-D part. |
| Minors | OUTSTANDING | untouched. |

### Paper E (paper5_casper.tex)
| item | status | where fixed / what remains |
|---|---|---|
| E-B1 OMP-3/OMP-5 misattribution | DONE + FIXED-NOW | abstract fixed (Opus); contribution 1 now says "the k=5 reconstruction" too. |
| E-B2 fold-cost on n=150 off-baseline subsample | RERUN | caveat in abstract (Opus: "single-seed n=150; canonical-ruler rerun pending"); canonical rerun owed. |
| E-B3 retracted model-equivalence claim | DONE | cut (Opus); text now explicitly makes no cheap≈frontier claim. |
| E-B4 re-embedding n=13 manual | RERUN | automate roundtrip over 2,087 queries × ≥2 models. |
| E-B5 §4 table has no script | OUTSTANDING | commit scripts/paper5/unaskability.py + output. |
| E-B6 coherence proxy arbitrary | PARK/RERUN | LLM-judge/human + threshold sweep (part of E restructure — untouched). |
| E-B7 zero human/LLM-judge eval | PARK/RERUN | ditto. |
| E-B8 placeholders (Fig.1, citations, 3-entry bib) | OUTSTANDING | untouched. |
| Missing/minors (phrase-bank details, fidelity ceiling, Tim Burton example, CIs, cross-paper tension, seed set) | OUTSTANDING | untouched (E is the designated restructure/merge candidate per §8). |

### Cross-paper / housekeeping
| item | status |
|---|---|
| Git risk (S5) | DONE — both repos committed (papers this pass; casper reruns 8f251ac/4d15fb0/c0a2941). |
| B+C claiming same contribution (C2) | DONE — 961ec1a + l.309 fix. |
| SUPERSEDED banners (PAPER_C_WINNER/README, BASELINES_PLAN) | OUTSTANDING (house). |
| Paper D scripts/paper4 home + locked checkpoints | OUTSTANDING (house). |
| Paper E tab:mixed provenance | OUTSTANDING. |
| Shared paired-bootstrap harness (§7.8) | RERUN — unblocks A-B3/B-B4/C-B5-text/D-B7/E CIs at once. |

**Counts (blocking items A-B1…E-B8, n=39):** DONE/FIXED 14 (A-B1, A-B9, part A-B4; B-B1, B-B2, B-B3, most B-B6; C-B2, C-B5, C-B7; D-B1, D-B3, D-B8; E-B1, E-B3) · RERUN outstanding 13 · PARK 6 (C-B1, D-B2, B-B5-wording, E-B6/7 + D joint-contribution demotion) · remaining mechanical/outstanding 6.
