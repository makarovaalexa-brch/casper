# PROGRAM STATE — 2026-07-10 (Fable's synthesis; to be adversarially validated blind)

## 1. THE DESTINATION (constant since day one, author's words distilled)
A deployable adaptive interviewer for cold-start recommendation: blind at turn zero, facing an
independent human-like answerer, free to use all channels (items, concepts, attributes, pairs,
later open recall), whose coarse-to-fine behaviour EMERGES from learning rather than being
hard-coded, and which beats the strongest fair static questionnaire. With defensible novelty and
no leaks, no clairvoyance, no self-authored circularity. Deadline gravity: ECIR 2027 (2 Oct 2026);
~1 year of PhD left; human study planned as the external validator, not the thesis carrier.

## 2. WHERE WE ARE — THE ASSET LEDGER (each with its evidence)

### Instruments (all gated, all committed)
- RecVAE-d512 (certified, ties EASE) — the frozen scorer.
- FOLD-V3 (two-channel belief encoder): implicit consumption tokens + explicit value tokens,
  SURPRISE feature (author's design); GoT gate +0.184, prolific gate +0.316, implicit channel
  load-bearing (+0.020 ablation), clean profile BEATS native (+0.017 where v2 lost 0.135);
  known trades: -0.028 on v2's narrow item-only regime; level-flag semantics inert (no-clue not
  negative — ablation with anti-surprise feature running tonight).
- ANSWERER v2.1 (the distilled LLM judge): values at LLM parity (item corr 0.550 vs LLM's 0.547;
  real ratings ∪ EASE backbone, fitted aggregations for concepts/entities); knowledge structure
  distilled (G1 beats baselines incl. niche); trait dial flutter-corrected via equating; serves
  all 162k users lazily. Gates 13/13 vs corrected targets.
- The 173-user LLM-judged EVAL WORLD: quarantined (touched twice max, by author schedule only);
  values validated vs real held-out ratings; knowledge-level flutter QUANTIFIED (the shuffle probe).
- Beliefs: kmap co-knowledge embeddings; LOUO judged-grid MF; the fitted pmodel (per-arena
  recalibration mandatory).

### Science (findings that survive every audit to date)
- THE FLUTTER: an LLM judge's knowledge-commitment threshold slides with presentation (rate spread
  up to 0.66 same-content; stars stable). Novel methods finding; every LLM-judge study in the
  literature is currently exposed to it silently. Decomposition: item-k2 variance = 34% question /
  ~0.3% user-features / 4.5% real trait / 22% flutter / 39% cell.
- TERRITORY > DIAL: per-user answerability is structured by taste-territory match (era x genre x
  subculture), not by a global knowledgeability constant; the profile-unpredictable residual is
  largely judge noise. (Profile reading + census CV + equating, three converging routes.)
- CONSUMPTION-AS-TASTE (the GoT thesis): engagement direction dominates valence; surprise
  (engagement vs volume-expectation) is the informative quantity — now VALIDATED INSIDE a trained
  instrument (fold-v3 gates), not just argued.
- The RATED-NESS CEILING (~0.73 AUC): which items a user logged is substantially idiosyncratic;
  probing cannot efficiently find rated items — the measured basis for open recall's necessity
  (+0.058/answer premium, re-confirmed under the repaired fold).
- THE EVALUATION SINS, each measured not asserted: survivorship averaging; value-model asymmetry;
  test-fit contaminated baselines (+0.028); fuel-free pools (0.000 in-pool refusal); OOD folds
  flipping verdicts; sample-starved policy learning (86 users vs thousands needed); E1-E7 codified.
- Channel facts: concepts saturate (low rank); items = bandwidth; recognition of famous items has
  ZERO person-trait; the scarcity habitat = know-well level + niche bands.

### In progress TONIGHT (all checkpointed)
- THE ARENA (the decisive experiment): v2.1 world, fold-v3, full 2,428-question universe (first
  arena ever to include attributes; pool bugs designed out), baselines b0-b4 (incl. the learned
  static b2 at population scale and the model-based myopic greedy b4), four adaptive classes
  (learned scorer w/ 2-step labels; ask-the-gradient; CAT-router; Golbandi tree at scale), >=3
  training seeds on leaders, DEV-synthetic evaluation only; the 173 untouched.
- NO-CLUE ABLATION (author hypothesis): fold-v3.1 with anti-surprise on refusals; skip-vs-fold
  decision by measurement.
- BLIND ARENA-CODE AUDIT (pre-verdict): E1-E7 + peek-hunt on arena_*.py before results exist.

## 3. THE HONEST EPISTEMIC STRUCTURE (where a massive logical error would live if it lives)
- Policies train on the DISTILLED world and are selected on SYNTHETIC dev users; the headline runs
  on the 173 REAL-JUDGED users. The distilled world was FIT TO the judge world → DEV verdicts are
  fit-world performance; the 173 is the only read with independent evidential force, and even it
  is judge-mediated (human study = the final de-proxy). ALL claims must be worded "under a
  validated simulated user."
- The synthetic users' ANSWERS are model-mediated but their HELD-OUT TARGETS are real ratings of
  real people — NDCG is grounded in real data everywhere.
- The comparison (policy vs static) is internally fair by construction (E7: same world, same fold,
  same training data rights). The RISK is not unfairness but ECOSYSTEM QUIRKS: a policy could win
  on DEV by exploiting the distilled model's determinism rather than human-like structure. Guards:
  the 173 read + the sim-real gap table + (deferred) cross-family LLM checks.
- b2's quality is load-bearing: if the population-scale greedy static is under-built, adaptive
  "wins" are hollow. Must verify b2 >= all historical statics on comparable footing.
- The fidelity-trust probe (does the fold ACTUALLY trust high-fidelity answers more) is queued but
  unverified — vividness-aware policy claims blocked until it lands.

## 4. THE PAPERS IN THE NEW LIGHT (current files -> proposed final shape)
- PAPER A (casper_u_chapter, instrument): ALREADY extended with the fold-is-part-of-the-instrument
  section + canary gate class. ADD: the reference-ladder figure (cold/static/fold/native) and the
  fold-v3 result (interview encoder beats native full-profile) as the capstone — the instrument
  story is now complete and strong. Status: closest to submission-ready (ECIR reproducibility).
- PAPER B (answerability): the richest upgrade. Now carries: external answerability (sec:external),
  the world-knowledge distillation + anatomy (sec:distill, importance table pending final),
  THE FLUTTER + equating (new section — arguably its headline methods contribution), territory>dial,
  the corrected fuel numbers. The learned-policy-win retraction stands. This is becoming the
  "measurement of what users can answer" paper — coherent and novel (RecQuest = must-cite neighbor).
- PAPER C (continuous/fidelity boundary): reframe holds (fidelity boundary as centerpiece); ADD
  the ask-the-gradient policy class as the CONSTRUCTIVE heir (imagine the nameless question, snap
  to the nearest askable in the user's territory) — C+E reconciliation; arena results will decide
  whether it carries an empirical win or ships as mechanism+boundary.
- PAPER D (open recall): still needs the de-rigging (symmetric baselines); gains the rated-ness
  ceiling + recall premium as its new foundation ("recall is information-theoretically forced").
- PAPER E: merge into C (un-askability as the reason snapping exists) — unchanged verdict.
- PAPER/CHAPTER F (the answerer): now inevitable as a thesis chapter — design, gates, flutter,
  equating, distillation, IRT framing; possibly a standalone resource/benchmark paper (the
  environment + 173-user judged set + discipline E1-E7).
- THESIS SPINE: measurement-first ("the apparatus decides the verdict") -> the validated
  environment -> the boundary science -> the arena verdict (win OR honest bound) -> recall +
  human study as the deployable frontier.

## 5. WHERE WE WANT TO GET (success states, in order of ambition)
S1 (floor, already secured): the methodology + environment + boundary thesis — publishable at ECIR
    as A + B regardless of the arena.
S2 (the working policy): any adaptive class beats b2 AND b4 on DEV with seed-robust CIs -> then
    the one pre-registered read on the 173 -> the anchor result.
S3 (the full ambition): S2 + the mechanism readout showing EMERGENT coarse-to-fine (channel-mix
    trajectories, transcripts) + recall integration designed with the author -> the thesis the
    author described on day one.

## 6. OPEN RISKS (ranked)
R1 b4 beats everything learned (computation suffices) -> still a win for adaptivity, reframes S2.
R2 nothing beats b2 even here -> the honest bound; S1 ships; recall becomes the frontier.
R3 ecosystem-quirk DEV wins that vanish on the 173 -> the sim-real table catches it; wording guard.
R4 clock: 12 weeks to ECIR; consolidation of A+B must start within ~2 weeks regardless of arena.
R5 the flutter finding invites "is your eval world reliable at all" — pre-empted by stars-stability,
   within-call robustness, equating; but the wording must lead with it, not bury it.


# CORRECTIONS (2026-07-10, from the blind validation — experiments/BLIND_VALIDATION.md; owned in full)
The validator confirmed every audited NUMBER (none fabricated, most to-the-digit) and convicted the
SYNTHESIS of three families of status inflation. Corrections of record:
1. CERTIFICATION TRANSFER: "gates 13/13 vs corrected targets" conflated two gates (13/13 was vs the
   OLD flutter-inflated targets; the corrected-target gate was 3 statistics, tolerance-passed,
   driven partly by fixes tuned to the target). Worse: the fold-v3/arena sampler substitutes a
   hand-parameterized sigmoid for the gated v2.1 knowledge models — the arena world was running on
   INHERITED credentials. Fixed tonight (arena fix #7: wire the fitted models + gate the world
   as-built).
2. INDEPENDENCE/BANNERS: the 173 are NOT an independent read (the distillation final-refit on all
   173; quarantine is prospective, adopted 07-09, unenforced in code); the grid is 173/300 unfrozen
   and the directional banner was dropped from the +0.028 contamination, the +0.058 recall premium,
   and elsewhere. Only the human study is independent. All wording downgraded accordingly.
3. ASPIRATIONAL CONTENT AS CURRENT: Paper B's flutter/equating section DOES NOT EXIST (being
   written tonight from LLM_DECOMPOSITION/SHUFFLE_PROBE/DANS v2.1 material; its stale ICC 0.174
   table to be corrected); fold-v3 is NOT "all-gated" (G5 real fail, G6 formal fail, artifact
   sub-fail — decisive gates passed, full suite did not); "serves 162k lazily" is designed, not
   run; "0.550 vs 0.547 parity" compares different targets/cohorts — restated as "the distilled
   value channel recovers LLM-level label agreement," not parity.
S1 is reclassified: NOT a secured floor — a realistic 2-6 week consolidation target (Paper A
near-shippable; Paper B needs the frozen grid, the flutter repeat-study at N, and its headline
section written). The adaptivity thesis has still never beaten a fair static on the judged cohort.
