# HANDOFF — single source of truth (2026-07-05, consolidates + replaces HANDOFF_I2.md and HANDOFF_REVIEW2_OPUS.md)

Canonical papers: C:\dev\phd\papers\ (edit here; sync casper/papers after, commit+push both repos).
Backups: GitHub casper repo (all code/results/papers), tag pre-instrument-rebuild-2026-07-04, zips C:\dev\phd_BACKUP_2026-07-04.

## 1. THESIS POSITION after the 2026-07-05 adversarial review (the framing that governs all work)

The review (C:\dev\phd\REVIEW_ADVERSARIAL_2026-07-05.md) is largely RIGHT on framing, largely PRE-ANSWERED on
machinery (P4c answer-source ablation; prereg record) — the papers' framing must catch up with what the
experiments already showed. Reframed thesis = ANALYSIS PAPER WITH A LAW, not method-paper-with-a-hero:

- **The fidelity boundary is the discovery**: elicitation value is governed by answer fidelity, not
  action-space continuity; continuous wins iff slider/comparison-grade answers; under the empirically
  fitted human-like channel, answerable discrete + graded real answers + open recall win. Nobody else
  fits their answer channel on real data — that is the novelty.
- **What survives every answer model** (non-circular, foreign-geometry-proof): answerability (4 datasets),
  concept-necessity at scale, open-recall bandwidth, snapping-as-denoising, rank relationships with a
  pre-registered out-of-sample confirmation (Goodreads P3), the measurement layer (protocol distortion,
  instrument-class visibility, gate suite, EASE-tied instrument).
- **Not "tried and failed"**: we have WHY (noise decomposition), WHERE (the boundary), WHAT WINS INSTEAD,
  and HOW TO MEASURE — the failed version of this PhD is the two-weeks-ago one that knew none of this.
  Viva line: "the field believed continuous elicitation worked because everyone graded it with the same
  oracle; we built the honest grader, found the boundary, and mapped both sides."
- Genre/venues: analysis/reproducibility-class papers (Dacrema precedent). A → reproducibility track;
  C → the fidelity-boundary paper; D → fair-comparison bandwidth paper; B → chapter + material in A/C;
  E → merged into D for publication.

## 2. HUMAN STUDY — value either way (cannot sink the REFRAMED thesis)

Under the old framing the study was sink-risk; under the reframe every outcome is publishable:
(a) humans give decent graded fidelity on sliders/pairs → continuous premium partially real in deployment
→ upgrades C; (b) humans give poor fidelity → the realistic-regime side confirmed, boundary paper MORE
relevant, discrete+open-recall recommendation validated; (c) intermediate → places real interfaces on the
boundary map = completes the headline figure. Residual risks, contained: framing-lever fails on real
memories (demotes one D claim to LLM-validated; D has other legs); catastrophic incoherence of all human
answers (would contradict the LLM-study calibration 71.5→85% + 83.6% answerability agreement; if it
happened it is a field-level finding, not a silent failure). Design accordingly: the study MEASURES
fidelity, it does not defend a claim.

## 3. REVIEW-RESPONSE TASKS T1–T9

**STATUS 2026-07-05 experiment batch — all 6 dependency-free experiments DONE (details +
tables in `HANDOFF_PROGRESS_2026-07-05.md`; per-experiment docs cited below):**
- ✅ **R2-noise** (§6/§7 priority): fair noise-adapted static (repeat top axis) 0.3143 BEATS noise-trained
  actor 0.2844, Δ−0.030 p<.001 → adaptivity CLEAN-CHANNEL-ONLY. `SQUEEZE_R2.md`.
- ✅ **R2b** tie-by-construction: actor warmed from the winning schedule TIES the static (0.3177, p=.82) →
  it's an OPTIMIZATION gap not expressiveness; fine-tune from optimum drifts away. → feeds T8 opt-gap note.
- ✅ **T4** bot-play/ABot 4th answer source: denoised-mean answerer keeps continuous actor alive at 0.407
  (collapse was SAMPLING NOISE not model-in-loop); item-8 drops to 0.240. P4C `PART 4`/`V4`.
- ✅ **T5b** popularity-conditioned answerability: concept advantage SURVIVES (tail +0.031 sig every regime).
  `experiments/paper2/T5B_ANSWERABILITY.md`.
- ✅ **T6** Paper D fair comparison: open-recall edge does NOT survive equal known-item access — TIE vs
  PEBOL+profile at equal budget (m=4 Δ+0.003, CIs overlap; PEBOL edges tail by m=6). Value is OPERATIONAL,
  not raw NDCG. `experiments/paper2/T6_PAPERD_FAIR.md`.
- ✅ **T7a** EASE cold-start k=1..8 vs I2: tie is WARM-START-ONLY — I2 wins every k (Δ+0.05–0.06, p=1.0).
  `experiments/instrument2/T7A_EASE_COLDSTART.md`.
- ⏳ **ML-25M I2 port** (backlog): RUNNING (Jul-9 constraint lifted; now fully in-scope).
- ⏳ REMAINING = the TEXT halves (T1, T2, T5a/c, T7b, T8, T9) + T3 kit — numbers now FROZEN, safe to write.
  T2+T8 MUST state: adaptivity is fidelity-conditional ("clean answers only"); the optimization-gap finding
  (§7); Paper D = fair-comparison (open≈PEBOL at equal access).


- **T1** Thesis-level "measurement position" page; condensed into every paper. (text, small, FIRST)
- **T2** Paper C reframe: fidelity boundary = headline; noiseless battery = mechanism study under stated
  oracle channel (label every noiseless table); snap-loss = regime-dependent finding (denoising); pair rung
  = parity + CI only; demote RecVAE-"sharpening" phrasing; retitle/subtitle ("...: Answer Fidelity and the
  Limits of Continuous Preference Elicitation"). (text, medium)
- **T3** Human-study kit for the user: protocol n≈50 (8 rendered Qs: k=3 sliders, 4 pairs, 2 open-recall,
  2 concept y/n; fold via I2; NDCG vs held ratings), materials from existing render artifacts, consent,
  power analysis, analysis script, runsheet. (prep, medium)
- **T4** Resurrect the removed bot-play/ABot learned answerer as a 4th answer source in the P4c framework
  (answers the "suppressed evidence" charge either way). (experiment, small-medium)
- **T5** Paper B: (a) purge residual §5 learned-win (+0.012) presentation — align with §6 + trainseed
  overturn; CIs on every table; (b) popularity-conditioned seen-probability answerability model, re-run
  item-vs-concept, report both regimes; (c) cite LLM answerability agreement (83.6%) at the definition.
  (text small + experiment medium)
- **T6** Paper D fair comparison: access-controlled arms (baselines get matched volunteered known-item
  tokens), PEBOL+profile as primary baseline, oracle-recall rows labeled bounds, realistic answerer as
  headline (tail loss stated), CIs everywhere. (experiment medium + text)
- **T7** Paper A: (a) EASE cold-start k=1..8 curve vs I2 (kills or confirms the "warm-start-only tie"
  charge); (b) scope §12.4 as geometric upper bound + add its empirical-channel counterpart; reword G4;
  cross-ref realizable-EIG to B/C. (experiment small + text)
- **T8** All papers stats pass: PREREGISTERED vs EXPLORATORY labels (link PATH1/GOODREADS/TWOPHASE_E2
  preregs), Holm/FDR within headline families, catalogue-fraction K rule stated once per paper,
  training-seed variance + paired significance on every headline (expect B +0.012, D +0.007, E +0.003 to
  die — say so); state linear-Gaussian predictions BEFORE results; fix E abstract −0.008 → −0.010/−0.012.
- **T9** Paper E: LLM-judge of axis-synthesis renders (n≥100, faithfulness/askability); merged-D+E
  publication carries the constructive arc; E stays a thesis chapter.
Order: T1 → T2+T8 → T4+T7a → T5b+T6a → T3 (parallel) → T9. ≈5–6 Opus days + user-run study.

## 4. OTHER REMAINING WORK (pre-review backlog, still valid)

- ML-25M I2 port (deferred core-cut item; conventions in experiments/instrument2/PHASE2_PORTS.md).
- Squeeze R2-4 pivoted (task #49): noise-robust policies (train-noisy actors, repeat-top-axis probing,
  decoder-metric belief, fix Kalman magnitude shrinkage) + non-peeking decoder-metric EIG selector toward
  the 0.755/0.878 ceilings (SQUEEZE_R01.md).
- Compile passes (deferred; before ECIR).
- Papers-dir consolidation when file lock clears (retire outer copy OR make it canonical permanently —
  decide; currently outer=canonical, casper/papers=synced backup).
- D+E journal merge (post-ECIR). ECIR assembly deadline 2026-10-02 (C+A). User: 1-2 replication datasets.
- User-owned: human study (T3 kit), supervisor sign-off, Overleaf compile.

## 5. STATE SNAPSHOT (what is DONE and where)

I2 arc complete (Jul 4-5): RecVAE-d512 instrument (ties EASE ML-1M 0.554; Goodreads +0.237=78% of EASE,
21.6x V1), 16/16 gates, additive operator, member-bag concepts, P4a flagship battery (noiseless: all claims
sharpen; snap-loss 4x), P4b cross-domain (prereg 3/4 confirmed), P4c fidelity boundary (the pivot finding),
squeeze R0-1 (actor = realizable myopic ceiling clean; Bayesian repeat-probing wins noisy +0.095).
Papers A-E updated through all of the above incl. sexiness pass (titles, worked-example box, figures).
Key docs: casper/experiments/instrument2/*.md, PHASE4A/4B/P4C/SQUEEZE_R01, INSTRUMENT2_PLAN.md,
REVIEW_HARSH_2026-07-01.md + REVIEW_RESPONSE (first review, fully addressed), PUBLICATION_PLAN_2026-07-01.md.


## 6. THE LEARNED-POLICY CLAIM — exact status + THE priority experiment (added 2026-07-05)

What is ours/novel/safe re learned policies (we do NOT show "policy doesn't matter"):
- The LATENT-SPACE ACTOR (D1-recipe on I2 — the same continuous actor whose vs-DISCRETE premium P4c debunks
  under realistic answers). What survives: (a) CLEAN channel: beats the strongest STATIC continuous design
  +0.019 full (paired bootstrap p=0.9998, static-distill control passed) and +0.011 on books (p=0.997);
  (b) NOISY channel: the NOISE-TRAINED actor (0.287) is the best CONTINUOUS method (> Bayesian 0.248 >
  frozen 0.15) while losing to discrete answerable arms (0.325-0.394). Novelty verified: elicitation policy
  in a SOTA latent space has no precedent. Second learned finding: "policies must be trained under the
  deployment answer channel" (0.15->0.287) — nobody else has a fitted channel to show this.
- What we show does NOT matter: reward engineering, endpoint optimization on saturated budgets, per-user
  branching over low-rank menus, clean-channel policy tinkering beyond the myopic ceiling.

**KNOWN GAP (the linchpin): the adaptive-vs-static margin has NEVER been tested under the realistic
channel** — noise-trained actor was compared to clean-era statics, not to a noise-adapted static basis.

**PRIORITY EXPERIMENT (R2-noise, run FIRST from the squeeze handoff):** train actors under the empirical
channel (3 trseeds, val-sel) AND construct the noise-adapted static comparator (greedy-selected basis
evaluated/selected under the same channel; also give the static the repeat-probing schedule option so it is
the STRONGEST fair static); compare with paired bootstrap. Outcome decides the claim's wording: adaptivity
"under clean answers" vs adaptivity, full stop. ~1 agent, hours CPU. Everything else in squeeze R2-4
(EIG selector toward the 0.755/0.878 ceilings, learned belief head, sigma-actor, k-curriculum) comes after.


## 7. R2 VERDICT (Opus ran it) + the "should-at-least-tie" question (2026-07-06)

R2 result: fair noise-adapted static (repeat-probing schedule, 0.3143) BEATS the noise-trained actor
(0.2844) by -0.030 [-0.040,-0.020]. Adaptivity verdict under the realistic channel: DOES NOT HOLD ->
papers use the pre-authorized wording: the clean-channel adaptive margin (+0.019) is FIDELITY-CONDITIONAL;
under noise the optimal realizable strategy is repeat-probing the top informative axis ("ask the best
question five times" - quotable, practical, ours).

WHY below-static is possible despite expressiveness (the actor CAN emit any fixed schedule): the
OPTIMIZATION GAP - (a) the static was found by greedy discrete SEARCH directly on the (val) noisy
objective; SGD through a sampled channel is a weaker optimizer; (b) actor objective is a proxy
(recon+softNDCG); (c) clean-warm-start basin biases toward distinct probes - gradients do not discover
the degenerate repeat strategy. THIS IS A RECURRING, NAMED PHENOMENON in the thesis (BC-clone 0.137 <
teacher 0.140; distill = lossless at best; from-scratch collapses x2; learned tag->z < constructed bags;
FieldActor < plain; Gumbel < REINFORCE < fixed): direct construction/search beats gradient training of a
superset class at these sample sizes. ACTION (T8 addition): state the optimization gap ONCE as a named
methods finding across papers; every surviving positive claim already beat its searched-static control.

NEXT ARM (tie-by-construction, add to R2): warm the actor FROM the winning repeat schedule + val-select ->
ties static on val by construction; any test shortfall then = pure val->test noise (quantifies the gap);
upside = adaptive-repeats might add value on top. Cheap, decisive, makes the "at least tie" property real.


## 8. TWO CORRECTIONS from user review (2026-07-06) — must reach the papers

(a) ANSWERABILITY PRIVILEGE IN THE NOISY-REGIME WINNERS (user caught it): the P4c/R2 "discrete answerable"
arms carry L1 privilege — concept-8 uses the per-user >=2-tagged-items proxy, and item-real (0.394) answers
only items the user actually rated = answerability oracle. Deployable discrete asking must PREDICT
answerability, and B's E2 showed the item-level exposure model captures only ~12% of that ceiling. Nuance:
for CONCEPTS the privilege is mild (answerability is population-stable/predictable — broad tags have high
answer rates); for ITEMS it is severe. CORRECT CONCLUSION WORDING everywhere the fidelity boundary is
stated: under realistic answers the winners are population-answerable CONCEPT questions + VOLUNTEERED items
(open recall — the user self-solves answerability), NOT "item questions" (label item-real rows L1 in every
table). T2/T8 must apply this.

(b) ANSWER-MODEL TAXONOMY for related work (answers "is everyone unclean?"): three traditions, each clean
on one axis, unclean on another — (1) real-ratings-as-answers (Rashid/Golbandi lineage): fidelity-clean,
answerability-ORACLE; (2) simulator-geometry answers (modern CRS/latent incl. EAR/UNICORN-style ground-truth
attribute answers, and our own V1 era): answerability modeled, fidelity-circular; (3) parametric noise
(conjoint/dueling bandits, Canal): noise acknowledged but ASSUMED, not fitted. To our knowledge no prior
work FITS the answer channel from behavioral data and treats fidelity as a measured experimental variable —
that is the fidelity-boundary paper's precise novelty claim (qualify "to our knowledge"; cite test-retest
reliability lit (Amatriain) as the measurement that existed but was never wired into elicitation eval).
