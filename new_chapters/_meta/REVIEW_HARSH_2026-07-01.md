# Harsh Review — 5 Papers + June Notes + Code (2026-07-01, Fable)

Six-way deep review: one adversarial reviewer per paper (A–E) with claims verified against
`casper/RESULTS.md`, `POLICY_RESULTS.md`, experiment logs and code, plus one cross-paper
integrity/consistency audit. This file holds the FULL findings including minor lists.
The chat summary holds only the big picture.

Calibration: the experiments behind the papers are largely real and logged (much better than
typical drafts) — but "90% done" is optimistic. Honest estimate: A/C ≈ 75%, B/D ≈ 65%, E ≈ 40%.
Most fixes are consolidation + reruns of existing scripts, not new research. ~8 genuinely new
experiments needed across the set (listed in §7).

---

## 0. FIVE SYSTEMIC FAILURE MODES (recur across all papers)

S1. **No statistical hygiene anywhere.** No confidence intervals, no paired per-user tests,
    no seed-set discipline. Headline margins of +0.011–0.026 NDCG on ~300 users, reported bare.
    Worse: Paper B explicitly selects checkpoints on the TEST set (paper2:351–354) and dropped
    the worst seed (123, where margin shrinks to +0.002) without disclosure. Fix pattern for
    every paper: paired per-user bootstrap (same users, method vs baseline), all-seed tables
    with per-seed deltas, pre-registered checkpoint-selection rule on a proper val split.

S2. **Headline comparisons are not apples-to-apples.**
    - B: headline "continuous-action policy" is never evaluated in the paper (it's Paper C).
    - C: graded-answer actor headline-compared to binary-answer CASPER-R while the fair
      comparator (uent+GRAW, 0.367/0.158) sits in its own Table 1 — halves the full margin.
    - D: probe baselines penalized for refusals while open recall has zero recall-failure by fiat.
    - A: PEBOL/EAR/UNICORN skipped on the strength of a +0.031 attribute ceiling measured with
      100 concepts, while §answerability shows concepts ≈ 98% of item info with 745 concepts.

S3. **Stale / superseded / retracted numbers survive in the text.**
    - C: l.273 "+0.010" (old actor), l.394 "0.328" (corrupted-cache era; post-fix = 0.343).
    - B: "354 users" (old split; true = 304), "six seeds" (tables say 5), +36% tail (old locked-150 split).
    - E: LLMDROP model-equivalence claim built on a run its own lab notebook RETRACTED
      ("DO NOT put in paper until re-run") and abstract attributes OMP-5's −0.008 fold cost to OMP-3.
    - A: 44%/75% concept-robustness numbers don't match the ledger (50%/70%, PART AB).

S4. **Simulator circularity is under-confronted.** Answers are noiseless geometric functions of
    u* in the same embedding the recommender ranks with; the reconstruction objective is trained
    against the very u* that generates answers (B); Paper D's "hidden-gem framing lever" is, as
    measured, the simulator's own argmax sampling rule; Paper E's coherence proxy uses the
    encoder the paper itself argues is unreliable. Every paper needs: answer-noise ablation +
    an explicit simulator-alignment limitation paragraph; D needs a noisy recall model; the
    framing-lever and un-askability claims need human or at least LLM-judge validation.

S5. **Provenance/git risk.** Entire manuscripts untracked (paper4 tex, all of paper5/), Paper B's
    +394/−220 rewrite uncommitted, Paper D has no locked checkpoints and no scripts/paper4 home,
    Paper E's tab:mixed has NO provenance anywhere, canonical mf_foldin.py untracked in casper/,
    casper/ inner repo has 117 dirty entries including uncommitted continuous_actor.py that
    carries all Paper D code. One careless clean destroys unreproducible work.

---

## 1. PAPER A — casper_u_chapter.tex (CASPER-U instrument/testbed)

### Blocking
- **A-B1 (fatal): attribute-ceiling self-contradiction.** §notheadroom: privileged oracle caps
  attributes at +0.031 (genres +0.025, concepts +0.012; lines 962–996) → used to justify NOT
  implementing EAR/SCPR/UNICORN/PEBOL. §efficiency + §answerability: concept oracle 0.581@4q,
  mixed beats items, fine concepts carry ≈98% of item info (0.479 vs 0.488). Root cause:
  +0.031 bound used only 100 concepts/120 users; other tables use 745 concepts. Never disclosed.
  FIX: rerun attr_oracle.py at 745 concepts, ≥2 answer models, same cohort; reconcile or
  implement PEBOL.
- **A-B2: "passes all seven gates" is claimed for a superseded scorer.** Gates evaluated on the
  z-scored confidence blend, later declared defective (l.874–879, "revealing can hurt"); the
  calibrated recon encoder of record FAILS strict G3 (q2 dip, l.1082–85); the actual guarded
  artifact (PAPER_A_WINNER dual-head rank instrument) appears NOWHERE in the chapter.
  FIX: one final named instrument+checkpoint, rerun full gate table on it, add a G8 no-harm gate,
  present old gate table as history, reconcile with PAPER_A_WINNER.
- **A-B3: zero seeds/CIs/tests**, cohort sizes swing silently (604/150/120/296/60 users) —
  violates the project's own INSTRUMENT_IMPROVEMENTS.md §0 protocol. Headline orderings
  (heuristics≈random; LLM<random on n=60; EIG>Golbandi) rest on 0.005–0.03 gaps.
- **A-B4: number drift** q0 0.305 vs 0.301; item oracle 0.493/0.485/0.49; concept robustness
  44/75 vs ledger 50/70 (and mechanism described ≠ mechanism logged). One provenance pass:
  every number → ledger row + script + date.
- **A-B5: "+20% NDCG/+40% Recall" reframing quote** comes from the artifact-prone z-scored P1
  protocol the chapter itself disowns. Scope it or replace with calibrated +0.025@q15.
- **A-B6: "first selector to beat random and the classical heuristics" is falsified by its own
  tab:select** (Golbandi/HELF also beat random); EIG, the headline winner, is never defined
  in-paper. Rewrite claim; include the one-equation EIG definition + the EIG==LA deployability control.
- **A-B7: answer-model sensitivity** — the attribute ceiling is conditional on one answer model
  (Sen-debiased); logs show answer model flips conclusions. Add sensitivity table.
- **A-B8: bibliography incomplete** — ~15 load-bearing inline citations absent from thebibliography
  (Tag Genome/Vig, TCF, PEPPER, CB2CF, BLaIR, Sen09, GATE, Rashid08-vs-02 HELF confusion, …).
- **A-B9: `\newcommand{\AA}` breaks compilation** (redefines Å). File likely never compiled end-to-end.

### Missing
- Fold-in-vs-factors control (already run! RESULTS PART F: ridge-on-learned-factors 0.294/0.071
  loses → gain is the fold-in) — the exact ablation reviewers will demand; include it.
- Ranking-trained MF under the held-out/tail protocol; entropy/pop×entropy in encoder panel;
  PEBOL; DRE-as-baseline on final protocol (own PART AF lists 9 planned baselines, ~4 shipped).
- Multi-dataset replication of the headroom figure (PART D/E raw material exists).
- Related work: evaluation-rigor lineage (Sun CIKM'20, Elliot, DAISY), CRS user-simulator
  critiques, FacT-CRS; hyperparameter sensitivity (c(m)=m/(m+5), λ, d=64, α=20, pop^{1/2}).
- Honest note that the 745 "open" concepts ARE an enumeration (curated Tag-Genome); only the
  SBERT bolt-on is truly open-vocab, and its honest result is "mixed".

### Minor
- HELF cited as rashid02 (wrong paper — HELF is Rashid 2008); tab:data "UCI-style" for
  California housing (StatLib), test n=250 unexplained; G6 18-genre row breaks equal-budget
  claim; "matches" used for a 7% DRE gap and a 0.015 WRMF deficit under self-chosen 0.02
  tolerance; fig:headroom "best heuristic" label (HELF) contradicts tab:s4panel (popularity);
  figures/ + tables/ contain dead artifacts of the deleted old paper1; abstract ~600 words,
  halve it; mixed prose-vs-\cite styles; empty \author{}.

### Verdict
As-is: reject at any full track; survives as thesis chapter after fixes. After fixes:
RecSys Reproducibility track / TORS / UMAP, led by the oracle-bounded headroom diagnosis +
answerability×information quantification, instrument demoted to "apparatus".

---

## 2. PAPER B — paper2_casper.tex (answerability + CASPER-R)

### Blocking
- **B-B1 (fatal): headline contribution not in the paper.** Intro l.64–65 + Contribution 1
  (l.70–78) claim a continuous-action snap policy; nothing continuous is evaluated — CASPER-R is
  a discrete per-candidate MLP argmax; continuity is deferred to Paper C in the same paper
  (l.200–202). Own logs: the Wolpertinger actor collapsed (POLICY_LADDER rung PC 0.288/0.103).
  FIX: contributions = (i) answerability quantification, (ii) discrete-frontier negative result,
  (iii) CASPER-R recipe. Delete/demote contribution 1.
- **B-B2 (fatal): checkpoint selection on the test set, admitted at l.351–354** and contradicting
  its own Algorithm 2 (best-on-val). With margin +0.012 and per-epoch test spread ~0.008,
  max-over-epochs-on-test can manufacture most of the gap. FIX: 3-way split, pre-registered
  selection rule, plus mean-over-epochs-3..8 robustness row. Contingency: if the win doesn't
  survive, reframe as answerability study + negative result (still publishable).
- **B-B3: "six seeds" (l.62) vs n=5 tables; worst seed 123 (+0.002) dropped undisclosed.**
  On seed 123 fair entropy alone ≈ ties CASPER-R's headline. FIX: report all 6 seeds with
  per-seed deltas (all positive — actually the strongest evidence), footnote the 123 rationale,
  define what a "seed" randomizes.
- **B-B4: zero significance testing** on +0.012 headline; between-seed σ is the wrong unit
  (same ~300 users each seed; the log's "6σ" is invalid). Paired per-user bootstrap for every
  bolded delta.
- **B-B5: narrative contradicted by own table + subsequent work.** Of 5 learned recipes, 4 lose
  to the static heuristic; BC clone (0.137) < heuristic (0.140); Paper C later finds policy ≈
  static entropy across 8 more attempts. "First learned-policy win" overclaims a knife-edge
  result. FIX: reframe §6 as "when/why policy learning pays at all"; make front-loading
  efficiency co-equal; qualify or delete "first".
- **B-B6: ruler chaos.** 354 vs 302 vs 304 test users in one paper; §4 discrete-frontier numbers
  (0.371/0.194) from a different encoder/split, undeclared (own roadmap calls this "Integrity
  blocker"); +36% tail claim from old locked-150 split, not computable from any in-paper table
  (current ruler gives +50%); Fig.1 caption says q=20, figure ends at 10; "reaches heuristic's
  q8 score in ~2 questions" belongs to a different policy vs the crippled-era baseline — the
  true seed-avg is +0.008 full @q2.
- **B-B7: answerability proxies may manufacture the headline.** Item answerable = rated
  (understates; rating ≠ seen), concept answerable = ≥2 rated items (overstates the other way);
  answers noiseless geometric closed loop; train objective (1−cos(u,u*)) aligned with the
  simulator's answer generator; two different u*'s and two answer functions share one symbol
  (§2 threshold form vs Alg.2 sign form) — pin down, check leakage. FIX: answerability-rate
  sweep, answer-noise sweep, notation cleanup, LLM-simulator spot check (already planned in
  PAPER_B_PLAN §4.5, never done).
- **B-B8: "full-profile ceiling" framing hides real headroom.** Own logs: full-profile is NOT
  the ceiling; privileged concept-oracle tail 0.354 vs CASPER-R 0.152 (~1/5 captured). The
  Jun-25 harmonization swapped the oracle out of the tables. Report both references honestly.

### Missing
PEBOL (cite+run or justify), ConTS run (closest prior — inherited idea), concept-EIG row
(conspicuously absent; item-EIG anchors §4), answerability-aware static heuristic
(divisiveness × P(answerable)) — if a 2-line heuristic hits 0.145 the story dies, test it
yourself first; FacT-CRS; NICF fairness (item-restricted by design — label or give it concepts);
Golbandi adaptive < static reversal unexplained (reimplementation red flag); LLM-asker rigor
(versioned IDs, n≥5 generations, non-Anthropic model, adaptive variant excluded as "too noisy" =
selective reporting); limitations/conclusion/hyperparameters/repro statement ALL absent;
single dataset (LastFM weakness known, unmentioned).

### Minor
"scores below greedy EIG below greedy" duplication (l.188); IGCN uncited; MRR only in caption;
decimal inconsistencies (0.75 vs 0.745; 0.126 vs 0.127); "corrected divisiveness" unexplained in
abstract; title acronym differs from historical expansion; head/tail masking definition
underspecified (PART E shows it changes conclusions); Alg.1 "returns metrics"; dead fig assets
(baselines.pdf, concept_landscape.pdf).

### Verdict
As-is: reject (B1+B2 each independently fatal). After fixes: RecSys full borderline; safer as
RecSys short / UMAP / CIKM full with the tail win as small-but-robust + front-loading as the
structural adaptive-vs-static argument. Plan the contingency reframe now.

---

## 3. PAPER C — paper3_casper.tex (continuous actions, D1) — strongest of the five

### Blocking
- **C-B1: headline not answer-fidelity-fair.** Abstract sells +0.018/+0.026 vs binary CASPER-R;
  own Table 1 contains the fair graded comparator uent+GRAW 0.367/0.158 → honest margin
  +0.011/+0.020. l.60–61 "a discrete policy cannot exploit graded" and l.394–395 "every discrete
  policy drops under graded" are contradicted by that row (uent+GRAW GAINS from graded).
  FIX: report both margins, lead fair; narrow the claim to "trained-on-binary drops zero-shot;
  static discrete gains a little; continuous gains most."
- **C-B2: stale corrupted-era numbers in prose.** l.273 "+0.010" (old actor margin); l.394
  CASPER-R graded 0.328 = corrupted-cache COMPARE4 number, post-fix rerun = 0.343 (collapse
  overstated 2×). Purge every pre-fix number.
- **C-B3: §5.4 oracle-headroom hides its sample** — 40 users, seed 1, tail-target, labelled
  "directional" in the log; presented in the same register as 5-seed results; "captures half the
  headroom" compares margins on incommensurate scales; 18% should be 17%. Rerun full-sample or
  move to appendix with caveats.
- **C-B4: answerability confound uncontrolled.** Continuous actor gets 8/8 answerable by
  construction vs CASPER-R's ~5.3/8. Missing decisive control: discrete policy RETRAINED
  natively on graded answers with full answerability over the same 761 concepts. The distill
  ablation is not this. This one experiment makes or breaks "two inseparable continuities".
- **C-B5: one training run.** Seeds are eval-split only around a single checkpoint
  (policy_phase3_d1divw_last.pt); fair headline margin is ~2σ informal. Retrain ≥3 training
  seeds; paired user-level bootstrap; describe the significance methodology.
- **C-B6: material omission — recommender-side lever + failure record.** FTREC 0.385/0.165 and
  FTRA-mix 0.401/0.183 with a STATIC policy beat the headline 0.378/0.178 on the same ruler;
  8 policy-learning attempts lost before D1; continuous actor never validated against the
  improved recommender. Disclose in Discussion/Limitations — the frozen-recommender scoping is
  legitimate but silence will kill it in rebuttal (thesis chapters sit side by side).
- **C-B7: PEBOL citation has the wrong authors** (real: Austin, Korikov, Toroghi, Sanner).

### Missing
PEBOL empirical comparison or explicit method-incompatibility statement (snap-loss is evidence
about YOUR actor snapped, not PEBOL's selector); adaptive-vs-static replay control (own
PAPER_C_DRAFT lists it as a must-ship honesty control; never shipped) — theory (linear-Gaussian
⇒ non-adaptive optimal design) predicts adaptivity worth little here, state the corollary;
answer-noise ablation; hyperparameter table (β, λ=DIVW, τ=DTAU, POS/NEG undefined at l.149);
reproducibility statement (locks/logs exist — cite them); provenance of the §7 tree (0.77 cosine
is the recon variant, headline sits at 0.73).

### Minor
Why te[300:] excluded 300 (one clause); entropy 0.361 vs 0.3608 vs 0.362 rounding; Table 2 q2
rounding drift vs gold CSV; abstract cosines lack "means over 1760 queries"; "measure-zero
subset" rhetoric vacuous — soften; cos vs dot in Alg.1 vs prose; no FULL column in Table 2;
TOC/empty author for submission.

### Verdict
As-is: top-venue reject / workshop weak-accept — but hygiene above average (cache disclosure,
gold-verified headline, honest PCA, snap-loss on headline model all check out). After fixes
1–7: solid RecSys full paper; snap-loss stays the centerpiece (crisp, falsifiable,
competitor-excluding). NeurIPS/ICML-tier additionally needs a second dataset, non-geometric
answerer stress test, training-seed replication.

---

## 4. PAPER D — paper4_casper.tex (open free-recall)

### Blocking
- **D-B1: realistic answerer LOSES tail (−0.008) but text says "level"; abstract quotes only the
  full-metric win.** Same lab called +0.010 tail "significant" in Paper C. State the loss.
- **D-B2: hidden-gem headline (0.382/0.193@2q) uses an info-optimistic simulator** — deterministic
  argmax of u*·Q/log(cnt) over the user's likes; the "framing lever" as measured IS the sampling
  rule. Own log admits "distinct is optimistic". FIX: noisy hidden-gem answerer (softmax /
  below-median-popularity sample), optimistic flags on every distinct/align/rating number
  including abstract, recall-failure probability.
- **D-B3: popweight "realistic" defense never validated**; plan promised a popularity-distribution
  sanity check, absent. Pessimistic bound poppop 0.376/0.142 LOSES to D1 on both axes → headline
  is band-conditional; say so. Note random5 (0.390/0.185) beats D1 on both axes with zero
  machinery — promote it, it's the most assumption-free positive result.
- **D-B4: baseline table unfair.** Probes run STRICT blind+refuse (waste turns); open recall has
  zero recall failure by simulator fiat. Add per-type answer-rate ablation + break-even p vs
  PEBOL/probes. Without it Table 7 should not be published.
- **D-B5: "first realizable adaptive policy to beat a strong fixed schedule" unestablished** —
  comparator is one hand-picked order; learned policy may just be a better static order.
  Ablation: evaluate the learned policy's modal path as a static order.
- **D-B6: provenance mismatches.** tab:closedmix t7/t8 usage ≠ HYBRID_RESULT ranges; Table 5
  mixes single-training-seed and averaged numbers; §2 claims default seeds {1,2,3,7,11} but every
  table uses {1,2,3} while quoted D1 rows are 5-seed numbers. Rerun headline tables on the
  canonical 5 seeds.
- **D-B7: no uncertainty at all** (code prints ±std; paper drops it); joint open+closed +0.003 is
  "near seed noise" per own log yet listed as a contribution — demote.
- **D-B8: non-deployable eval shortcut** (mask user's entire known half, s[seen]=−1e9) known and
  parked in DEPLOYABILITY_TODO, undisclosed in the paper's limitations. Disclose.

### Missing
NO CITATIONS AT ALL (no \cite, no bibliography — PEBOL/GATE/Sanner/ConTS/Cremonesi all
name-dropped bare; verify GATE venue); closest classical prior absent: Rashid 2002/2008
"Getting to Know You" item interviews + onboarding pick-lists + Golbandi trees (cited in Paper A!);
free-recall/self-report psychology (recognition vs recall — naming costs more user effort than a
click, which undercuts "4× more efficient"; availability heuristic for popularity bias);
honest cold-start framing paragraph ("we compare interaction protocols for extracting a profile
the user already possesses"; users filtered to ≥6 rated + non-empty likes — truly cold users out
of scope); learned-tree figure is one seed's tree, opener is seed-dependent — say which;
no Paper-D part in RESULTS.md, evidence scattered in experiments/paper2/*.md.

### Minor
"0.39" rounding vs unrounded comparator; mix row half-uses info-favourable answerer while
labelled realistic; "fully-open realistic ≡ popweight" row is a definition presented as data;
Table 3 upper-band flags missing when quoted in §6/abstract; abstract ~1 page, halve; "4× more
sample-efficient" qualify; Fig.2 caption seed; Alg.1 "held half" wording.

### Verdict
As-is: reject anywhere (no citations, no error bars, headline vs own realistic operating point).
After fixes: credible RecSys full (borderline) or strong RecSys/CIKM short. A 30-person survey
(named-favourite popularity distribution + does "underrated" wording shift it down) converts the
central assumption and the novelty claim from simulation artifact to finding — highest-leverage
single experiment in the whole thesis.

---

## 5. PAPER E — paper5_casper.tex (triangulation / un-askability) — least mature

### Blocking
- **E-B1: abstract misattributes the −0.008 fold cost to OMP-3; it was measured for OMP-5.**
- **E-B2: fold-cost experiment on n=150 single-run subsample whose own baseline is +0.011 off
  canonical — larger than the −0.008 effect claimed.** Rerun on canonical ruler, paired CIs.
- **E-B3: model-tier "indistinguishable faithfulness" claim rests on a run the lab notebook
  RETRACTED** (SBERT back-projection cos≈0.026 → both models scored on near-random folds).
  The notebook header says "DO NOT put in paper until re-run". Cut or re-derive.
- **E-B4: re-embedding-fails claim is n=13 manual renders, one model.** Automate over the 2,087
  dumped queries × ≥2 models (roundtrip.py exists).
- **E-B5: the central §4 un-askability table has NO reproducible script** — commit
  scripts/paper5/unaskability.py + recorded output before any submission.
- **E-B6: coherence proxy = SBERT label-sim > 0.3** — arbitrary threshold using the encoder the
  paper argues is unreliable; own cursed counterexample (−feel-good −animation −erotic IS
  verbalizable as one axis) contradicts the criterion while being used as a success example.
  LLM-judge (and/or 2 humans × 100 blends) + threshold sweep.
- **E-B7: zero human or LLM-judge evaluation of anything language-facing** despite deployment
  claims (rendering quality, safety filter, axis synthesis, select-and-align — no numbers at all
  for select-and-align).
- **E-B8: placeholders** — Fig.1 empty box, "(Full citations to be completed)", 3-entry bibliography.

### Missing
Phrase-bank construction details + the concepts-vs-rich-dict ablation (already run: k3 0.808 vs
0.838); fidelity-ceiling discussion (~0.89 cap = "fundamentally off-manifold", good result,
omitted); the Tim Burton semantic-geometric gap example (cos 0.36, documented, striking);
collinearity-fallback statistics; CIs anywhere (tab:mixed 0.390 vs 0.387 is a coin-flip);
cross-paper tension: l.170 "discrete nearly matches continuous (0.377)" vs Paper C selling the
gap as significant — honest framing: "the continuous edge is real but un-askable, so a closed
turn should spend a discrete concept."

### Minor
Placeholder author block; 1,100 vs 1,114 concepts; three different fidelity rulers never
distinguished (0.842/0.838/0.814); "mean pairwise cos ≈ 0" give the number; price list undated;
"~0.04" → cite C's exact −0.037/−0.040; tab:mixed duplicates Paper D territory; SKELETON's
promised GATE-style LLM-elicitor head-to-head silently vanished; "13 frontier-model renders" —
name the model; hyphenation un-askability/unaskable.

### Verdict
As-is: not submittable. Real decision: standalone vs merge (see §8). The un-askability idea
(faithful ⇒ orthogonal ⇒ incoherent; coherent ⇒ redundant) is the genuinely new thing and per
the deep-research memory the load-bearing differentiator vs PEBOL/GATE.

---

## 6. CROSS-PAPER INTEGRITY AUDIT (key results)

PASSES (worth knowing): CASPER-R 0.360/0.152, entropy 0.361/0.140, D1 0.378/0.178, uent+GRAW,
snap-loss, Paper D open-recall numbers, B→A EIG-frontier quotes — all identical across papers
and matching locked artifacts (paperC_divw_LOCKED MANIFEST, PAPER_C_WINNER/README,
OPENQ/SNAPLOSS logs). Corrupted cache fully resolved on disk (all six live/locked
pool_entavg.npy = 156c072c; bad cache survives only as renamed _crippled.npy).
PAPER_A/B/C_WINNER checkpoints present, shas match memory (instrument 10a7999f, ftrec 5d6406fd).

FAILURES:
- paper2:310 "354 users" vs everything else 304 (C1).
- paper2:62 "six seeds" vs n=5; paper2:64–65 claims continuous policy as headline (C2 — worst
  narrative bug: read together, B and C claim the same contribution).
- Paper E baselines on n=150 subsample, three unreconciled values for the continuous baseline
  (0.388 / 0.377 / 0.378), fixed-order 0.387/0.181 matches nothing in Paper D.
- Paper D: headline tables {1,2,3} vs quoted 5-seed D1 rows (mixed-seed comparisons load-bearing).
- Paper E: no seed set stated anywhere, tail never defined — weakest ruler discipline.
- PAPER_C_WINNER/README still titles the superseded FTREC stack "PAPER C WINNER" — add
  SUPERSEDED banner (likewise BASELINES_PLAN.md).
- Paper D: no scripts/paper4, no locked checkpoints for the learned anytime/joint policies
  (violates save-all-checkpoints rule); all D code in uncommitted continuous_actor.py.
- Paper E tab:mixed: no provenance found anywhere.

GIT RISK (act this week):
- Outer repo: paper2 rewrite (+394/−220) UNCOMMITTED; paper4 tex UNTRACKED; entire paper5/
  UNTRACKED; paper1 deletion + _ARCHIVE uncommitted/untracked; pebol_extract.txt untracked.
- Inner casper/ repo: continuous_actor.py modified-uncommitted (carries Paper D + ftrec code);
  scripts/paper5 + experiments/paper5 + NOREPEAT_RESULT.md + mf_foldin.py untracked;
  uncommitted deletions mixed in → a git clean/checkout is destructive in both directions.

---

## 7. THE ~8 GENUINELY NEW EXPERIMENTS (priority order)

1. **C: retrain a discrete policy natively on graded answers, full answerability, same 761
   concepts** — the decisive control for the thesis's flagship claim.
2. **A: attr_oracle at 745 concepts × ≥2 answer models** — resolves the fatal A-B1 contradiction
   and decides whether PEBOL must be implemented.
3. **B: 3-way-split rerun with pre-registered checkpoint rule + all-6-seed paired bootstrap** —
   decides whether Paper B's headline survives at all (plan the reframe contingency now).
4. **D: noisy hidden-gem answerer + recall-failure/refusal rates + break-even p** — converts the
   framing lever from artifact to bounded claim; rerun headline tables on {1,2,3,7,11} with ±std.
5. **C: ≥3 training-seed retrains of D1 + adaptive-vs-static replay control + full-sample oracle.**
6. **E: canonical-ruler fold-cost rerun (OMP-3 AND OMP-5) + scripted §4 + scaled roundtrip.**
7. **Human micro-study (~30 people), serves D and E at once:** (a) named-favourite popularity
   distribution vs popweight; (b) does "underrated" wording shift named popularity down;
   (c) LLM-judge/human coherence ratings for E's blends. Highest credibility-per-hour in the set.
8. **Everywhere: paired per-user bootstrap harness** (one script, reused by all papers).

Plus mechanical: purge stale numbers (S3 list), fix bibliographies (D has none), one provenance
pass per paper (every number → log row + script), disclose FTREC in C, disclose eval-masking in D.

## 8. ORGANIZATION & VENUES

Recommended shape: **4 submissions + thesis chapters**, not 5.

- **A → RecSys Reproducibility track (2027) or ACM TORS**, framed as testbed + oracle-bounded
  headroom diagnosis + answerability×information quantification ("the apparatus paper").
  Thesis chapter 2. Not a method paper; don't sell it as one.
- **B → RecSys short / UMAP / CIKM full** as "answerability is the realizable lever + when policy
  learning pays at all" (honest framing per B-B5). If the win dies under clean selection: fold
  the answerability study into A (it strengthens A's story) and the negative result into C's
  motivation; B stops being a standalone submission. Decide after experiment #3.
- **C → flagship. RecSys 2027 full paper** (fair-margin headline + snap-loss centerpiece +
  graded-answer conditionality). With a second dataset + noise ablation + training seeds it
  could stretch to SIGIR/WSDM; without, RecSys is the right home. Thesis centerpiece chapter.
- **D + E merged → one paper: "From latent probes to askable questions"** — D's open-recall
  result + E's un-askability theorem-shaped negative + select-and-align fallback is a single
  coherent deployment story (E already reads as D's discussion section; E's tab:mixed already
  duplicates D territory; both need the same human study). Target: **RecSys full or CUI**; CIKM
  as fallback. Alternative if both must stand alone: D → RecSys short; E → CUI (only after the
  human/LLM-judge work, else it's a workshop talk / thesis discussion chapter).
- Thesis arc (5 chapters regardless of publication packaging): A instrument → B what-to-ask
  discrete + answerability → C continuous → D open recall → E deployment limits. The arc is
  genuinely coherent; the pieces mostly need honesty + hygiene, not new ideas.

Novelty strength ranking (defensible): C's snap-loss (competitor-excluding, keep centerpiece) >
E's un-askability (new idea, unvalidated) > D's framing lever (new, currently simulator-artifact)
> A's oracle-bounded headroom method > B's answerability quantification (incremental, known lever).
