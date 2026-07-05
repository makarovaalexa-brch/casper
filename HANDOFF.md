# HANDOFF v2 — THE DEPLOYABLE ARENA PLAN (2026-07-06; replaces v1 = HANDOFF_ARCHIVE_2026-07-06.md, kept for detail)

## 0. THE GOAL (user's framing, governs everything)
A recommender INSTRUMENT and an ELICITATION AGENT spanning discrete → continuous → open questions, built to
be FULLY DEPLOYABLE, free of the field-common assumptions that break deployability, compared against strong
SOTA baselines. Working hypothesis (common sense, to be tested fairly at last): a personalized policy that
conditions on received answers beats dumb static heuristics — IN AN ENVIRONMENT THAT CONTAINS WHAT REAL
ENVIRONMENTS CONTAIN.

## 1. THE DIAGNOSIS (what was wrong with everything so far)
Every adaptivity null in this thesis was produced in an arena that had removed the surprises adaptivity
exploits: noiseless answers (no verify value), answerability oracles / answerable-by-construction pools
(no learn-who-can-answer value), single homogeneous channels (no routing value), no turn costs, refusals
discarded from the belief. Static-optimal is a THEOREM in that world — we kept reporting it as a finding
about elicitation when it is a finding about our arenas. Meanwhile per-user heterogeneity in the data is
LARGE (answered-count spreads, answerability variance, cold-cohort gradients) — the fuel adaptivity needs
exists; our arenas withheld it. All prior claims are hereby understood as ARENA-CONDITIONAL; papers must
say so (this also defuses the adversarial review at the root).
We built every COMPONENT of the right system and never the COMPOSITION:
instrument (I2, EASE-tied) ✓ · fitted answer channel ✓ · repeat-probing ✓ · open recall ✓ · pairs ✓ ·
sliders/one-screen ✓ · refusal-as-evidence (specced, never implemented) ✗ · channel routing (parked) ✗ ·
coarse-to-fine descent (specced, never run) ✗ · THE AGENT + THE ARENA (never built) ✗✗.

## 2. PLAN OF ATTACK (phased; pre-register before each run; tie-by-construction discipline throughout)

### A1 — THE DEPLOYABLE ARENA (build first; 1-2 agent-days)
One evaluation environment, ML-1M + Goodreads, with ALL of, simultaneously:
- Fitted answer channels PER QUESTION TYPE: items/directions (done, P4c channel), concepts y/n/dk, pairs,
  sliders (fit from LLM-study data + rating statistics now; refit from human study later). Every channel
  assumption GRADED in a table: fitted / assumed / human-pending.
- NO per-user answerability oracle anywhere: population priors (popularity, tag coverage) only; refusals
  are OBSERVABLE events that cost the turn and are foldable as evidence.
- ALL channels available to any policy: multi-granularity concepts (coverage tiers), items, pairs,
  sliders/one-screen, open recall (with name→factor resolution error rate from the bad-anchor study).
- Turn costs: uniform first; recall>recognition cost variant as sensitivity.
- Metrics: anytime cost-weighted NDCG curve (primary) + endpoint + answered/refused counts; paired
  per-user bootstrap; 3 training × 5 eval seeds; Holm within claim families.

### A2 — BASELINES IN THE ARENA (the bar; 1 agent-day)
Best searched STATIC questionnaire (greedy, val-selected, mixed channels/granularity — the strongest static
constructible); entropy/divisiveness pipeline; the current "heuristic winner" (open-recall-first + broad
concepts + repeat-probing); PEBOL-geometric; cheap LLM-asker; random. All under identical arena physics.

### A3 — THE AGENT (ours; the thesis's promised object; 3-5 agent-days)
One policy that, each turn, chooses (channel, question, granularity) conditioned on the full history
INCLUDING refusals: components = Bayesian belief on I2 (μ,σ) + online answerability posterior (updated by
refusals — the Paper-B-diagnosed missing signal) + coarse-to-fine descent + channel routing + repeat-probing
under noise + open-recall anchoring. DISCIPLINE from our hard lessons: (i) WARM-START from the best heuristic
pipeline = tie-by-construction (can only add on val); (ii) TRAIN UNDER THE ARENA (train-under-channel
lesson); (iii) gate each learned component — keep only what beats its heuristic counterpart (3 seeds,
bootstrap); the agent may end up part-learned part-heuristic — report composition honestly.

### A4 — PRE-REGISTERED PREDICTION (freeze before A3 eval)
Agent ≥ every baseline on the anytime cost-weighted metric on BOTH datasets; the margin GROWS with measured
per-user heterogeneity (answerability variance / knowledge depth); endpoint reported (may tie — saturation).
Honest expectation: the win is efficiency+robustness-shaped (fewer wasted turns, better routing per user),
not a giant endpoint delta. A TIE in THIS arena is, for the first time, a meaningful statement about
elicitation rather than about a simulator — publishable either way, but call the shot first.

### A5 — HUMAN STUDY (user-owned; design in archive §9.2) calibrates the arena channels; arena v1 runs on
fitted+LLM calibration now, refit later. The papers state the study as designed/forthcoming.

### A6 — PAPERS MAPPING
A = instrument + arena + deployability audit (archive §9, tone rules + pre-mortem) → ECIR repro track.
C = mechanisms + fidelity boundary (why channels behave; the clean-regime studies labeled as such;
retitle per archive §10.5) → ECIR full. B/D = component chapters feeding the agent (answerability;
open recall/bandwidth). FINAL CHAPTER / journal paper = THE AGENT IN THE ARENA (A1-A4 result).
ECIR timing: A + C stand on completed work; the agent result targets the journal/next venue if Oct 2 is
tight — do NOT rush A3 into ECIR.

## 3. WHAT SURVIVES AS-IS (claims inventory, one line each; detail in archive)
Fidelity boundary + fitted channel (novel, ours) · answerability 4/4 + concept-necessity (non-circular) ·
open-recall bandwidth + bad-anchor robustness · rank relationships incl. one prereg out-of-sample hit ·
un-askability + one-screen costs · snapping-denoises · repeat-probing under noise · measurement layer
(instrument-class visibility, protocol distortion, gates, EASE-tied I2) · clean-regime mechanism suite
(ladder, inversion, adaptivity isolation) AS mechanism-under-stated-model. DEAD/RESCOPED: unconditional
continuous>discrete; snap-loss as centerpiece; adaptivity under realistic answers (pre-arena); B learned
win; oracle-recall headlines. Answerability-privilege correction (archive §8a) applies to all noisy tables.

## 4. TASK LIST (priority order)
P1 A1 arena build+doc · P2 A2 baselines · P3 A3 agent (descent+refusals first — archive §12 spec is the
core; add routing, then learned gating) · P4 A4 prereg + verdict + weave into final chapter · P5 review-
response residue not superseded by the reframe (archive T1,T2,T5a,T7b,T8,T9 text fixes; §9.4 numbers
review; T4 learned answerer as an arena channel variant) · P6 human study kit (T3) · P7 ECIR assembly
(A+C) + compile · P8 D+E merge, ML-25M port, squeeze leftovers (archive §4) — post-ECIR.

## 5. PRE-MORTEM (attacks on the new plan itself)
"Arena assumptions are still yours" → graded-assumptions table + human refit + sensitivity sweeps; the
arena is at least EXPLICIT where the field's is implicit. "Agent ties" → meaningful now; report; thesis
still stands on §3 inventory. "Too big before Oct 2" → A+C ship regardless; agent = final chapter/journal.
"Tie-by-construction makes the win trivial" → no: warm-start guarantees ≥ static on VAL only; the claim is
the generalized margin + heterogeneity scaling, bootstrapped. "You changed your thesis again" → no: this IS
the original CASPER goal (deployable adaptive elicitation), reached by eliminating what doesn't work —
document the elimination path as the contribution it is.


## 6. LITERATURE CORRECTION (2026-07-06, user challenge: "surely shown before?")
YES — Golbandi/Koren 2011 ternary interview trees (like/dislike/UNKNOWN branch = refusals as evidence)
beat static seed lists on real Netflix ratings: adaptive-beats-static WAS shown, honestly, on the ITEM
channel, 15 years ago. We under-read it (filed as "static seeds"; replicated his curves in June; our own
reimplementation had his tree LOSING to static — flagged, never resolved = an ignored clue our harness
lacked the surprise channels his method feeds on). CONSEQUENCES:
(a) NOVELTY CLAIM (humbled, corrected): first honest adaptivity test across the FULL MODERN CHANNEL SET
(multi-granularity concepts, pairs, sliders, continuous, open recall) on a SOTA instrument with FITTED
answer channels — "the classical era tested adaptivity honestly on one channel; the neural era gained
channels and lost the honest test; we restore it." Cite Golbandi as the ancestor of §12 descent.
(b) A2 MUST include a faithful Golbandi-style ternary tree baseline; the arena owes an explanation of our
June reversed replication (if the arena is right, his tree should beat statics there = validates both).
(c) VERIFY (targeted lit check before print): exact Golbandi margins/protocol; any post-2011 multi-channel
adaptive elicitation under realistic answers (interview/bootstrapping lineage, Sepliarskaia et al?, Elahi
survey descendants) — if something closer exists, narrow the claim again.


## 7. SIM-FIRST REPLAN (2026-07-06, user directive: the calibrated answerer takes CENTER STAGE)
User verdict: ~50% of paper content rests on unrealistic answering; replace/redo with realistic assumptions.
THE CENTERPIECE ARTIFACT = **the Calibrated Answerer** (working name; a released, per-channel,
behaviorally-FITTED user simulator). Deep-research workflow launched 2026-07-06 (run wf_3a5ef549-c13) —
its report (citations, recipes, kill-risks, field-audit quotes) gets appended as §7-ANNEX when it lands;
DO NOT finalize novelty claims before reading it.

### 7.1 CHANNEL TABLE — each channel: grounding data → fitted model → status
S1 ITEM RATINGS: P(rating|affinity)+per-user style — FITTED v1 (P4c, 809k pairs); extend w/ test-retest
   anchor (Amatriain refs via research) + response-style psychometrics. [v1 done]
S2 CONCEPTS/TAGS y/n/dk: **ground in RAW TAG APPLICATIONS** (ML user-item-tag triples = real users
   asserting concepts): per-user tag-vocabulary breadth = answerability model; tag sentiment = polarity;
   genome scores = item-side relevance. NOVEL grounding — verify no prior via research. [to build]
S3 REVIEW-MINED VOCAB: mine the vocabulary users ACTUALLY use (Goodreads/Amazon/IMDb reviews); askable
   concepts = attested phrases; answers grounded in the user's own review text + rated-item phrase
   incidence. Ties to the original language-anchored CASPER vision, done right. [to build]
S4 PAIRWISE: Bradley-Terry FITTED from co-rated pairs (rating differences = revealed choices; real tie/noise
   rates) vs the field's assumed logit. [to build, easy — 1M ratings give millions of co-rated pairs]
S5 SLIDERS/GRADED: quantization + response-style, fitted (extends S1). [half-done]
S6 OPEN RECALL: **ground in FIRST-SESSION RATING ORDER** (a user's earliest ratings ≈ what they volunteer
   first — measure popularity/extremity/recency bias of first-k vs later ratings = an in-dataset revealed-
   recall experiment). NOVEL grounding — verify via research. + availability-lit parameters + LLM persona
   as secondary + human study as final. [to build]
S7 REFUSALS/DON'T-KNOW: from S2 vocabulary breadth + coverage + our LLM-study dk-rates; Golbandi ternary
   protocol as the classical anchor. [to build]
Each channel ships with: fit stats, validation (held-out behavioral prediction), assumption grade
(fitted/assumed/human-pending), and a sensitivity knob.

### 7.2 EXECUTION ORDER (very specific)
STEP 1 (after research lands): read §7-ANNEX; finalize channel recipes + novelty wording; write the
  field-audit section w/ REAL quotes+citations (respectful tone rules of archive §9.1 apply). 
STEP 2: build S4+S2 (cheapest, both from in-hand data) → S6+S7 → S3 (heaviest, needs review corpora).
  Each channel = one Opus agent, one result md, fit+validation numbers, committed.
STEP 3: assemble THE ARENA (v2 §A1) on the calibrated channels; graded-assumptions table auto-generated.
STEP 4: re-run the claim battery on the arena (which prior claims survive calibrated answers — expect the
  fidelity-boundary structure to persist; every paper's numbers rebased or clearly dual-labeled
  oracle-vs-calibrated). THIS is the redo of the "rubbish 50%".
STEP 5: baselines incl. Golbandi ternary tree (§6) + THE AGENT (v2 §A3: descent+refusals+routing,
  tie-by-construction, trained under the arena) + prereg (v2 §A4).
STEP 6: human study (archive §9.2) refits channels; papers state it as forthcoming.

### 7.3 PAPER REMAP (the replace/redo)
NEW PAPER S = **the Calibrated Answerer + field audit** (replaces Paper B's dead policy content; B's
answerability models BECOME sim components S2/S7; the audit of archive §9 lives here with citations).
The great story, if research confirms the kill-risk check: "elicitation has been evaluated against
users that answer like the evaluator's own model; here is the first simulator that answers like the
data says people do — and here is what survives." A = instrument+gates (keeps its role, cites S).
C = mechanisms + fidelity boundary, numbers rebased/dual-labeled. D = open recall on the S6-grounded
channel. E merged into D. FINAL = agent-in-arena. ECIR: A + C if timing holds; S is potentially the
STRONGEST submission — decide venue when S exists.

### 7.4 HALF-FINISHED IDEAS FOLDED IN (nothing lost)
Coarse-to-fine descent (archive §12) = the agent core (STEP 5). Golbandi ternary (§6) = mandatory baseline
+ our June reversed-replication resolution. Learned bot-play answerer (archive T4) = a VALIDATION
comparator for the sim (does a learned answerer match fitted channels?). Cross-geometry channel stability
(archive §9.3iii) = S1 validation. Numbers review (archive §9.4) = subsumed by STEP 4 rebase. LLM-study
phase 2 = sim-validation arm. ML-25M port, squeeze leftovers, D+E merge = post-arena backlog.

### 7.5 NOVELTY-RETENTION CHECKLIST (verify each against §7-ANNEX before claiming)
(1) first behaviorally-calibrated MULTI-CHANNEL elicitation simulator; (2) tag-applications-as-concept-
ground-truth; (3) first-session-order-as-recall-proxy; (4) review-mined askable vocabulary grounded in
users' own text; (5) fitted-BT pairwise from co-rated pairs in elicitation eval; (6) the citation-backed
deployability audit; (7) fidelity boundary (already ours); (8) the restored honest adaptivity test across
modern channels (§6 framing). Each: cite nearest prior, state delta, invite counterexamples.

## 7-ANNEX (PARTIAL, 2026-07-06): sim research — extracted but UNVERIFIED (Fable limit hit mid-run)
STATUS: deep-research run wf_3a5ef549-c13 completed SEARCH+EXTRACT but ALL verification panels failed on
the Fable-5 usage limit (75/104 agents errored). Claims below are EXTRACTED, NOT adversarially verified —
treat as leads, re-verify every citation before print. Full raw: experiments/SIM_RESEARCH_RAW_PARTIAL.txt.
Re-run verification when budget resets: Workflow({scriptPath: the wf_3a5ef549-c13 script, resumeFromRunId:
'wf_3a5ef549-c13'}) — cached search results replay free; only the verifier panels re-run.

STRONG LEADS (S1 item-rating channel — verify then use as the flagship calibration precedent):
- Amatriain et al. UMAP 2009 "Rate it Again / I like it... I like it not": test-retest, 118 users x 100
  Netflix movies x 3 trials (+7-month 4th for 36 users), explicit 'not seen' option. Findings to reuse:
  overall reliability 0.924 (2wk) -> 0.889 (7mo); intra-user RMSE 0.557-0.832; ~40% of users give a rating
  inconsistent with their own prior; noise is VALUE-DEPENDENT (extremes 1/5 stable, mid 2/3 noisiest,
  removing '3' lifts reliability 0.924->0.95); seen/not-seen labels THEMSELVES noisy (refusal-channel
  grounding); order + speed effects. => our fitted channel S1 already matches this LINEAGE; cite it as the
  behavioral target and the "magic barrier" framing.
- Said et al. SIGIR 2012 "magic barrier": irreducible noise floor = expected sq error of the optimal algo,
  estimated via re-rating; B = sqrt(mean (r-o)^2). Directly reusable S1 recipe + the argument that
  evaluation below the barrier is meaningless (extends our fidelity-boundary story with a citable floor).
- Natural-noise-management survey (ACM 2021, TORS?): its critique = prior noise work validated only by
  offline MAE/RMSE, marginal gains, NEVER by downstream task effect -> supports our "no one wired
  measured noise into elicitation eval" novelty angle (verify the exact survey + quote).
EXTRACTED-BUT-UNVERIFIED for other channels (in raw file, re-verify): Tag Genome construction was
SURVEY-BASED per-user tag judgments (Vig et al.) incl. 'unsure' responses (= S2/S7 concept answerability +
don't-know grounding, possibly a stronger ground truth than raw tag apps — CHECK); raw tag-application
sparsity caveats; kill-risk verdict on "has anyone built a calibrated multi-channel elicitation simulator"
= NOT ESTABLISHED (verification never ran) — this is THE claim to confirm before writing Paper S's headline.

ACTION when budget resets: (1) resume-verify wf_3a5ef549-c13; (2) targeted manual checks on the 3 novelty
groundings (tag-apps vs Genome-survey for S2; first-session-order for S6; review-mining for S3) + the
kill-risk; (3) fold verified results into §7 channel table; THEN start building S4+S2.

## 8. DECISIONS 2026-07-06 (user, on Opus) — override earlier points

D1. REPEAT-PROBING IS NOT DEPLOYABLE AS LITERAL REPETITION. You cannot ask a real person the same question
    5x. DELETE "repeat-probing / ask-your-best-question-5-times" as a deployable headline (kills the earlier
    §18b). BUT the underlying insight survives and must be RE-REALIZED deployably: under noisy answers you
    want to RE-MEASURE THE SAME LATENT AXIS via DIFFERENT questions that load on it (axis-redundant
    questioning — several distinct askable questions probing one direction to average answer noise down).
    ACTION: re-run the noise-robust arm with an AXIS-REDUNDANCY constraint (no verbatim repeats; distinct
    questions per turn, allowed to target the same latent direction). This is the deployable, novel form —
    "when answers are noisy, ask several different questions about the same thing." Reframe in Paper C as
    such; the literal-repeat static was only a mechanism probe, label it that way.

D2. SQUEEZE / ADAPTIVITY RESULTS MAY DIFFER UNDER THE PROPER MULTI-CHANNEL ANSWERER. The R0-1/R2 numbers
    were on the P4c single-channel (item-direction) fitted noise. RE-RUN the squeeze battery (actor vs fair
    static, axis-redundancy version per D1) under the calibrated multi-channel answerer once it exists.
    RETRAIN the actor under that answerer. KEEP this analysis in Paper C (the adaptivity-under-realistic-
    answers section) — do not finalize the "fidelity-conditional" wording until the proper answerer is in.

D3. ARENAS = ML-25M + GOODREADS. DROP ML-1M as a primary arena going forward (it stays only as legacy/
    provenance for the V1-era results already written). All new work — calibrated channels, arena, agent,
    claim rebase — targets ML-25M (has raw tag applications for S2, first-session order for S6, scale for
    fitting) + Goodreads (reviews for S3, cross-domain). Update every "ML-1M first" instruction accordingly.
    Note: ML-25M I2 instrument port (was backlog point 35) is now ON the critical path — needed before the
    arena. Goodreads I2 instrument already built.

PROCEED per §7 SIM-FIRST plan with D1-D3 applied.

## 9. EXECUTION RULE (user, 2026-07-06): STOP-ON-BLOCKER
On ANY gate failure, blocker, unexpected/negative result, or kill-risk hit: STOP and surface to the user
for a command. Do NOT improvise workarounds, relax gates, or proceed to the next phase autonomously.
Applies to every agent/phase in the SIM-FIRST plan — especially the verify-first gates (kill-risk check,
the 3 novelty groundings) and each channel-fit validation.

## 10. CHEAP VERIFICATION 2026-07-06 (3 searches; full verify parked to Wed) — RESULTS + BLOCKER
S1 PRECEDENT: VERIFIED. Amatriain "I like it... I like it not" UMAP'09 + "Rate it Again" RecSys'09 (118
users, 3 trials, 100 Netflix movies, mid-ratings noisier) real & correctly cited; "Magic Barrier"
(Said et al., + UMUAI 2018 "Coherence and inconsistencies in rating behavior") real. S1 flagship
calibration story is solid. USE.

S2 NOVELTY — CAVEAT (not a kill, needs wording): the Tag Genome gold standard WAS built from ~50,203
USER-PROVIDED (item,tag) relevance judgments (Vig 2012) — so "user tag judgments" is NOT novel. Our S2
claim must be scoped precisely: we use RAW PER-USER TAG APPLICATIONS as a per-USER ANSWERABILITY/VOCABULARY
model (which concepts THIS user knows/uses), distinct from the Genome's per-(item,tag) RELEVANCE. Verify
Wed nobody did per-user tag-knowledge/answerability modeling; narrow if so.

KILL-RISK — **NOT CLEARED = BLOCKER (stop per §9).** The "user simulator for CRS" space is CROWDED and
ACTIVELY MOVING IN 2026. Near-priors surfaced, unread: (1) ANCHOR "Agentic Noise Creation Framework for
Human Simulation and Denoising Recommendation" (arXiv 2606.05621, 2026) — title alone is close; MUST read
Wed. (2) Zhang & Balog agenda-based simulator (KDD 2020). (3) the LLM-user-sim validity/faithfulness line
(iEvaLM + 2024-26). (4) ICER synthetic-dialogue (2510.02331), Interplay (2603.18573). NONE obviously builds
a per-channel BEHAVIORALLY-FITTED answer-noise simulator — but I cannot clear "first calibrated multi-channel
elicitation answerer" cheaply. => DO NOT write Paper S's headline until ANCHOR + the LLM-sim-validity line
are read (Wed, resume-verify wf_3a5ef549-c13 + targeted fetches). If ANCHOR or another already fits answer
noise from behavioral data per channel, the novelty narrows to the ELICITATION-EVAL application + the
specific groundings (tag-apps/first-session-order/review-mining) — still a paper, smaller claim.
