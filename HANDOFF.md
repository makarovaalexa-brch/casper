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
