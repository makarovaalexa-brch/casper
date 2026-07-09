# DESIGN SHEET — THE POLICY ARENA (the fair fight, at last)
Status: DRAFT FOR DISCUSSION (author + Fable). Nothing executes until marked.
Prerequisite: fold-v3 gates pass (building now). Contingency: if v3 fails its decisive gates, the
arena runs on fold-v2 and every claim is scoped "explicit-channel only".

## 0. The question, and what changes per outcome
Does ANY deployable adaptive interview policy beat the strongest fair static on the clean
apparatus (v2.1 answerer, fold-v3, full question universe, E1-E7 discipline)?
- WIN -> the thesis anchor result; the coarse-to-fine narrative gets its evidence.
- TIE/LOSS -> an honest bound measured on an apparatus nobody can break; the
  methodology + boundary thesis ships without the anchor. Either outcome is citable; only one
  is the ambition.

## 1. Worlds and the sim-to-real firewall
- TRAINING world: v2.1 answerer served LAZILY over trU population users (162k available; the 300
  study IDs excluded); fold-v3 belief; unlimited simulated interviews.
- DEV evaluation: held-out SYNTHETIC users only. ALL iteration, tuning, model selection happens here.
- HEADLINE evaluation: the 173 real-judged users, touched TWICE maximum — one mid-course sanity
  read (pre-registered date), one final table. Every number on the 173 is pre-registered before
  it is looked at. This is the sim-to-real firewall; it also MEASURES the sim-real gap (dev vs
  headline deltas reported side by side).

## 2. Arena hygiene (every past artifact killed by construction)
- Question universe = ALL judged questions (1,128 concepts + 500 attributes + 800 items). No
  top-N pools, no coverage sampling (the fresh-audit finding), no alphabetical ties (the ladder
  finding). Attributes in a policy's reach for the first time.
- Answerability regime: LENIENT (k>=1 = answerable) as the environment rule — the strict arena is
  retired because its boundary sits in the measured flutter zone. Vividness is NOT a gate; it
  enters as evidence through fold-v3's learned implicit/explicit weights. Refusals (k=0) are real
  and common on the full universe (niche concepts ~47%, entities ~18-34%) — the habitat exists
  without a strict rule.
- Refusal = consumed turn, belief unchanged, user never dropped (E1). Same users, every arm.
- E7 symmetry table printed: every arm's training data, construction cohort, beliefs, and fold are
  listed side by side; statics train on the SAME synthetic world as policies.
- Horizon: full curves to T=24; pre-registered budgets 8/16/24; primary metric endpoint NDCG@50
  at T=24 (room for exploration to pay back) with endpoint@10 and anytime@10 reported alongside;
  MDE printed next to every CI.

## 3. The baseline ladder (all must be beaten or explicitly not-worse)
- b0 cold start (orientation row).
- b1 concept-entropy static — Paper B continuity row.
- b2 THE LEARNED STATIC: greedy question sequence trained on synthetic population users over the
  full universe (the strongest fair static ever constructible in this program; E7-symmetric).
- b3 static+skip (cheap-adaptivity assassin: b2 with refused turns refunded).
- b4 MODEL-BASED MYOPIC GREEDY (the sharpest threat): no learning — each turn, argmax over all
  questions of the 1-step expected gain computed with the SAME beliefs (v2.1 answerer model) and
  fold the policies use. Deployable, adaptive-by-computation. If b4 beats b2, that alone is the
  program's first honest adaptive win — and every learned policy must then beat b4, not b2.
- b5 LLM-INTERVIEWER (author-raised, PENDING SPEND MARK ~<=$8): a cross-family LLM (NOT the judge's
  family — e.g. Haiku vs the GPT judge, to break the think-like-the-judge coupling) chooses each
  next question from the dialogue transcript alone (no grid access); our environment answers.
  Pre-empts the "why not LLM as interviewer" reviewer challenge and connects to GATE/PEBOL
  comparators. Frozen prompt before headline; shared-prior caveat printed. Baseline, not thesis
  policy (black box, no mechanism readout, per-turn API cost at deployment).
- Labelled context rows (never cited as results): true-table router; clairvoyant NDCG-greedy.
- Reference row (out of scope, reviewer-anticipation): open-recall @2 (author-reserved design).

## 4. The adaptive classes (each trained at population scale; each with a tie-by-construction
##    variant anchored on b2; best-on-DEV goes to the headline)
- A. LEARNED SCORER v2 (the CASPER-R lineage at proper scale): outcome-labeled candidate scorer
  over belief-state + candidate features; labels from simulated interviews on the training world;
  2-STEP labels included (attacks the measured myopia gap r-free < r-anchored).
- B. ASK-THE-GRADIENT (snapped-continuous, Papers C+E reconciled): direction = d(ranking)/dz
  through the frozen decoder; score(q) = E_outcomes[ Delta-z(q, outcome) ] . direction, expectation
  over the answerer's outcome distribution with BOTH tokens per outcome (author's two-channel
  correction); snap = argmax within predicted-answerable territory. Little to learn (calibration
  only) — the data-efficient class.
- C. CAT-ROUTER (psychometrics-anchored): maximum-information question selection for the user's
  latent position (taste + knowledge map), IRT test-information machinery; the literature's
  existence proof that adaptive selection beats fixed tests, applied.
- D. (cheap extra) the Golbandi tree at population training scale — the class-matched reference.

## 5. Mechanism deliverables (the "does coarse-to-fine EMERGE" readout — not just deltas)
Per arm: refusal rate per turn; channel mix over turns (attributes/concepts/items trajectory);
granularity trajectory (population answer-rate of asked questions over turns); divergence-from-b2
turn; 3 full user transcripts (what was asked after what answer). If coarse-to-fine is real, it
shows here before it shows in the aggregate.

## 6. Pre-registered verdicts (printed before any headline number is computed)
- PRIMARY: best-on-DEV adaptive vs b2 AND vs b4, endpoint@50 T=24 on the 173, paired CIs + MDE.
  WIN = CI excl 0 vs b2 and >= b4 (if only b4 wins: "adaptivity pays via computation; learning
  adds nothing yet" — stated exactly so).
- SECONDARY: all budgets/metrics reported; no post-hoc metric selection; sim-real gap table.
- The 173 are touched per the §1 protocol only.

## 7. Explicitly out of scope
Open recall as a policy channel (author-reserved; reference row only). Human-transfer claims
(flutter caveat + human study pending). Any LLM calls.

## 8. Compute: label generation + 4-6 training runs + eval — local CPU, ~1-2 days wall, $0.

## SIGN-OFF — SIGNED BY AUTHOR 2026-07-09 ("sounds good, try that")
[x] All sections as drafted. EXECUTION SCOPE GUARD (Fable): the build proceeds through DEV
    evaluation on synthetic users ONLY; the 173 real users are NOT touched by any agent — the two
    permitted headline reads are scheduled explicitly by the author, separately.
