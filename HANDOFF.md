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
- ✅ **ML-25M I2 port** (backlog, Jul 5): I2 ports to 25M scale — RecVAE-d512 full-profile 0.4998/0.3443
  (+0.248/+0.294 over MOSTPOP, EASE-class, matches ML-1M headroom), monotone k-curve, **8/8 gates PASS**.
  = 2nd replication dataset for A/C. `experiments/instrument2/PHASE2_ML25M_PORT.md`. (Honest scope note:
  concept-ONLY fidelity −0.061 below pop floor = rank-4 genome-centroid concepts, but concepts still ADD
  on items per G5.) Optional leftovers: NSUB=0 full retrain, d-sweep, EASE bar, W1/Phase-4 policy battery.
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
- Squeeze R2 DONE (see §3/§7). **Squeeze R3 QUEUED to run AFTER the ML-25M port (user-requested 2026-07-05),
  tasks #16–20:** (16, FLAGSHIP) non-peeking decoder-metric EIG selector attacking the SELECTION gap
  0.55→0.75 then CONTINUITY gap 0.75→0.88; (17) decoder-metric belief tracker + fix Kalman magnitude
  shrinkage (re-inflate to ||z*||≈eta); (18) learned belief head; (19) sigma-actor (uncertainty-aware,
  needs #17); (20) k-curriculum (anytime/all-lengths reward). Headroom-chasing upside, not review-response.
  Base infra: squeeze_r0/r1/r2 in scripts/instrument2/, SQUEEZE_R01.md compass + recommendations.
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


## 9. THE DEPLOYABILITY-ASSUMPTIONS SECTION (user-elevated: strong, scandalous-adjacent, handle with care)

### 9.1 What to write (a full section, likely in Paper A with a condensed version in C)
Title suggestion: "What simulated users assume: a deployability audit of elicitation answer models."
Structure: a per-lineage table + prose. For EACH lineage: answer source / fidelity axis (clean? circular?
assumed?) / answerability axis (modeled? oracle?) / the un-deployable assumption, stated neutrally /
representative citations (verify each against the actual papers before print):
  (1) Real-ratings-as-answers (Rashid 02/08, Golbandi 11, classic AL): fidelity-clean, answerability-ORACLE
      (asks only rated items -> undeployable at catalogue scale; our Goodreads 0.0003 number quantifies it).
  (2) Ground-truth/simulator-geometry answers (EAR, SCPR, UNICORN, ConTS, latent sims, OUR OWN V1 ERA):
      answerability modeled, fidelity-CIRCULAR (user answers from the evaluator's own representation).
  (3) Parametric-noise choice models (conjoint/polyhedral Toubia04, dueling bandits Yue09, Canal19 pairs):
      noise acknowledged but ASSUMED (logit/Gaussian), rarely/never FITTED to behavioral data in this use.
  (4) LLM-simulated users (2023-26 CRS evals, PEBOL-era): flexible but unvalidated fidelity; popularity/
      sycophancy priors of the LLM = uncontrolled assumption (our LLM study = partial calibration of this).
TONE RULES (this WILL annoy people if done wrong): every tradition solved the axis it cared about;
SELF-IMPLICATE FIRST (our V1 era sits squarely in lineage 2 — say so in the first paragraph; the audit
grew from auditing ourselves); no "flawed/unclean" language — use "carries an assumption that does not
deploy"; reproduce each lineage's setup faithfully; explicitly invite correction ("to our knowledge; we
welcome counterexamples"). The claim, precisely: "we find no prior elicitation evaluation that FITS the
answer channel from behavioral data and treats answer fidelity as a measured experimental variable."
Our contribution framed HUMBLY: a first, imperfect instance (fitted channel + fidelity sweep + boundary),
with its own caveats listed (9.3), completed by the human study (9.2).

### 9.2 HUMAN STUDY — brief design (draft; full kit = T3; ALSO: the paper/chapter must state this study
as designed/forthcoming WITH this summary, so reviewers see the closer coming — add to T2/T8):
n≈50, within-subject. Entry: participants with importable ratings history (Letterboxd/IMDb/MAL) or a
20-seed-film rating phase. Each participant answers ~16 items across channels on THEIR catalogue: 4 slider
screens (k=3 phrase blends), 4 pairwise comparisons, 4 concept yes/no/don't-know, 2 open-recall prompts
(favourite + hidden gem), 2 item ratings (repeat of seed items -> per-person test-retest fidelity).
Measures: per-channel FIDELITY (corr of answers with held ratings / geometric prediction), answerability/
refusal rates, framing-lever shift (gem vs favourite popularity percentile), downstream NDCG (fold answers
through I2, rank held-out titles). Analysis: place each real interface on the fidelity-boundary x-axis;
power ~0.8 to detect channel fidelity differences ~0.15 at n=50 (verify in T3 kit). Output feeds: C
(boundary map gets real-interface markers), B (answerability rates), D (framing lever), E (slider usability).

### 9.3 ADVERSARIAL PRE-MORTEM on the taxonomy claim (write defenses INTO the section):
(i) "Conjoint DID calibrate choice-model noise from data" — partially true in marketing (logit fitted to
holdout choices); defense: qualify claim to ELICITATION-FOR-RECOMMENDATION evaluation + the fidelity-SWEEP
(no one varies fidelity as an experimental axis); ACTION: verify via targeted lit check before print.
(ii) "Test-retest noise was used somewhere" — Amatriain measured it; search for anyone WIRING it into
elicitation eval; if found, cite as closest prior and narrow the claim. (iii) "Your fitted channel is
itself model-dependent" (bins defined by s=cos in OUR space) — TRUE in part; defense already exists:
the rating side is behavioral, and the channel can be refit per s-definition (V1/EASE/I2) — ACTION (cheap
experiment): show the fitted law is stable across the three geometries; add to numbers-review below.
(iv) "Static rating noise != interactive conversational answering" — concede; exactly what the human study
measures; this is why the paper states the study design. (v) "Straw-manning prior work" — the tone rules +
faithful-reproduction table + self-implication are the defense.

### 9.4 NUMBERS REVIEW for the answerability-privilege correction (8a) — Opus task:
(a) verify whether concept-8 (0.325) used per-user lift selection (L1) or population lift (deployable);
if per-user, ADD a population-only deployable-concept arm to the noisy tables; (b) label item-real rows L1
everywhere; (c) re-state the R2/P4c conclusions with the corrected deployable ordering (expected:
population-concepts + open recall lead; verify); (d) propagate to C/B/D texts (T2/T5/T6).


## 10. POSITIONING CORE + STRAIGHT ANSWERS (2026-07-06, user Q&A — governs final framing)

### 10.1 Why the critic hit US on the shared flaw, and the guard
The field uses the flawed channel as BACKGROUND (A-vs-B under one oracle: bias ~cancels). We used it as
the SUBJECT (claims ABOUT answer types) -> circular for us, merely embarrassing for others. Also we armed
the critic (P4c/§5.6 put the refutation in our own text) and 5 papers on 1 substrate = one target.
GUARD (write into T1/T2): claims about CHANNELS are always stated per-channel with the channel's evidence
grade (oracle / fitted / human-pending); we self-implicate (V1 era = lineage 2) and position the audit as
"we made the shared assumption visible, measured its cost, built the first fitted alternative" — being
first-to-surface IS the contribution (Krichene-Rendle precedent: the bar moves when someone makes it salient).

### 10.2 The fitted channel: what it gave us, build status
GAVE US: the boundary result (ordering inversion under realistic answers), the noise decomposition
(linear/quantization/sampling ~0.10 each), channel-robust claim rankings, repeat-probing, train-under-
channel policies — i.e., converted "assume answers" into "measure answers". BUILD STATUS: v1 DONE for
ML-1M item-direction answers (P(rating|s), 809k pairs, corr .516). NOT FINISHED: (a) extrapolation to
continuous q / concepts is assumed (same law applied to cos(z*,q) — flagged); (b) one dataset; (c)
cross-geometry stability refit pending (§9.3iii); (d) static rating noise != interactive answering (human
study closes). State build status honestly in the papers.

### 10.3 THE CLAIM STACK (final): (i) answer fidelity governs elicitation value (boundary law, fitted
channel = the instrument for it); (ii) best DEPLOYABLE method = a HEURISTIC pipeline: open recall
(volunteered items) + population-answerable concept questions (lift/divisiveness-selected) + graded real
answers, on the I2 instrument, with repeat-probing under noise — components have classical ancestry
(Rashid-family selection; recall = classic onboarding), the ASSEMBLY + quantification + instrument +
boundary map + repeat-probing are ours; (iii) learned/continuous adaptivity wins ONLY in high-fidelity
regimes (sliders/comparisons) — fidelity-conditional, honestly labeled; (iv) the measurement layer.
ANSWER to "best method on realistic env: learned, heuristic, ours?": HEURISTIC, assembled-by-us,
components-not-ours, instrument-ours. Learned policies do not make the deployable podium today (pending
§9.4 numbers review for the exact deployable ordering). This is fine: it is an analysis thesis.

### 10.4 Why the squeeze is out of scope (for THIS thesis)
R0-1: clean-channel actor already at the realizable myopic ceiling (nothing to squeeze). R2: under noise
the searched static wins. Remaining rungs = noise-robust policy LEARNING research — a direction the thesis
OPENS (future work, flag planted), not a claim it needs. Budget cut-from-bottom confirmed this. It is in
scope for the field/postdoc, out of scope for the reframed claims.

### 10.5 Title correction (coordinator's own error, acknowledged)
"The Best Question Has No Name" was proposed when the clean-channel flagship was the story; after P4c it
OVERCLAIMS — the best deployable question is highly nameable ("name a hidden gem"). ACTION (T2, firm):
retitle Paper C to the fidelity/boundary framing (e.g. "Answer Fidelity Governs Preference Elicitation:
Mapping the Boundary Between Continuous and Askable Questions"); the poetic phrase may survive ONLY as the
section title of the un-askability/clean-regime mechanism study, where it IS cleanly derived (realization
caps, snap-loss under the stated oracle model).


## 11. WORST-CASE DEPLOYMENT ANSWER (no answerability knowledge, no open questions) — viva prep
(1) Constraint is softer than stated: any trained recommender implies logs implies popularity/coverage =
free population-answerability priors. (2) Measured floor with nothing per-user: ~4 BROAD concept questions
(answerable by semantic breadth, not statistics; saturation at rank ~2-4 = ~4 questions), graded answers
where UI allows, repeat-probing under noise. (3) Beyond the floor: COARSE-TO-FINE ADAPTIVE DESCENT driven
by refusals (each answer/refusal updates taste AND answerability; descend granularity only where user shows
knowledge) — the one adaptivity channel no static design can imitate, and structurally always present.
Evidence status: floor = measured (saturation, concept answerability, repeat-probing, refusals-as-evidence);
the full coarse-to-fine questioner = PREDICTED design, never built — state as the thesis-implied design +
future work, clearly labeled. (Connects: parked channel-failure routing demo; B answerability; rank law.)


## 12. COARSE-TO-FINE DESCENT — NEVER TRIED; priority experiment spec (2026-07-06)
STATUS: every ingredient built separately, the COMPOSITION never run. (two-phase = fixed 4+4 w/ oracle;
refusals-as-evidence = diagnosed missing in Paper B, never implemented on V1 or I2; routing demo #25 parked;
POLOPEN sequenced types but went static, no granularity hierarchy, no knowledge-conditioning.) This is the
deployable-adaptivity result the thesis has circled since day one and the cleanest fair shot at an HONEST
adaptivity win.

SPEC (I2 instrument, ML-1M first, then Goodreads): granularity hierarchy = concept coverage tiers (broad =
high-coverage tags, fine = low-coverage / sub-tags); STRICT wasted-turn regime (no answerability oracle —
a fine concept unanswerable for this user wastes the turn, exactly the per-user variation to exploit);
CRITICAL: FOLD REFUSALS as evidence (the Paper-B-diagnosed missing signal — a "don't know" updates both the
belief and an answerability estimate; benchmark two encodings: (a) refusal as a negative/soft signal on the
concept's member bag, (b) a small running answerability posterior gating candidate granularity). Policy =
coarse-to-fine descent: ask broad; descend into a sub-tree only where the user answered (showed knowledge);
stay coarse elsewhere. Comparators (house discipline): best STATIC mixed-granularity questionnaire (greedy
val-selected), flat-entropy-over-all-concepts, the ~4-broad-concept floor; 3 seeds, paired bootstrap, val-sel.
PREDICTION (pre-register before running): descent > best static, and the margin GROWS with per-user
answerability heterogeneity (measure it). Evaluate under BOTH clean and the P4c fitted channel (this is a
DISCRETE answerable channel -> should be noise-robust, unlike the continuous actor). Also report answered/8.
OUTCOME EITHER WAY: a win = the thesis's deployable adaptive method (learned or heuristic-tree, both cheap);
a tie = strengthens "adaptivity needs the answerability surprise channel, and even that saturates" — both
publishable. Effort: ~1-2 agents. Decision (user): before or after ECIR. If run, it likely becomes a
Paper B/D headline and a genuine adaptive-policy contribution the current draft lacks.
