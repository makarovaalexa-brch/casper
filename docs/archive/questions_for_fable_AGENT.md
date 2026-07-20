# Fable brief — design the adaptive elicitation AGENT (the flagship crown result)

Self-contained. You designed the answerability GATE (which passed 4/4) and reviewed the study design
(caught 2 study-killers). Now design the AGENT itself. Everything below is DONE and validated; your job is
the agent that runs in this environment.

## WHAT EXISTS (all validated, ML-25M, committed)
- INSTRUMENT: RecVAE-d512, ties EASE (I2 arc). Belief update = additive latent operator z'=z+eta*a*q
  (eta~=16=mean||z*||), z=0 cold seed, graded answer a=cos(z*,q). Concepts = member-bag-encode directions.
- ANSWERABILITY MODEL: fitted P(answerable|features) AUC=0.921 held-out-users, ECE=0.008; features
  [pop, log_ratings_count, decade, genre_match, franchise, is_concept]; genre_match coef +0.63 = the
  user-specific fuel. Artifact .cache/instrument2/answerability_pmodel.{json,npz}. Scores answerability for
  ANY item/concept. (LLM-derived plumbing per Q-D — never the independent witness; the a-priori structural
  rule is the witness.) Also a per-user cached judgment grid (300 users) for exact answerability.
- ANSWER VALUES (non-circular): real ratings for rated items/pairs; data-side member-rating aggregate for
  concepts; LLM-predicted (MAE 0.70 stars, = fidelity sigma) only for never-rated, SENSITIVITY-ONLY.
- FIDELITY: swept knob, sigma~0.70 stars anchored to Amatriain test-retest + our masked-item MAE.
- ORACLE HEADROOM (G2, privileged upper bound): answerability-aware vs blind greedy dNDCG = 0.287
  CI[0.268,0.307]. The realizable agent captures SOME FRACTION of this; that fraction IS the finding.
- CHANNELS available: multi-granularity concepts (coverage tiers), items, pairs, sliders/one-screen,
  open recall, abstract-continuous (ceiling only). Refusals cost a turn and are OBSERVABLE.

## THE THESIS THE AGENT MUST DELIVER (gut-check story)
Coarse->granular ADAPTIVE elicitation: ask broad answerable questions first; DESCEND into finer questions
only where the user's answers (and refusals) reveal they can answer; route across channels; the agent's
edge = discovering per-user answerability IN-CONVERSATION (the one thing static schedules cannot do, and
the one channel — answerability — that is genuinely per-user, now that we have a validated model of it).
Prior sims removed this fuel; we put it back (validated) and test whether adaptivity finally wins.

## HARD LESSONS (do not repeat — all in HANDOFF.md)
- 8+ prior policy attempts TIED the best static; from-scratch RL collapses; direct search/construction
  beats gradient-training a superset class at these sample sizes (the "optimization gap").
- TIE-BY-CONSTRUCTION discipline: warm-start the agent from the best heuristic so it can only ADD on val.
- Clean-channel actor was at the realizable myopic ceiling; adaptivity there was ~0 (linear-Gaussian).
  The NEW hope is that real per-user ANSWERABILITY heterogeneity (which the sims lacked) changes this.
- TRAIN UNDER THE ARENA (the fidelity/answerability environment), not a clean channel.

## THE ASKS
Q1. AGENT ARCHITECTURE: what exactly is the policy? Options to weigh: (a) a learned RL/differentiable-unroll
    actor over (belief z, answerability-posterior, turn) -> (channel, granularity, question); (b) a
    heuristic coarse->granular tree with an online answerability estimate (updated by answers+refusals);
    (c) hybrid (heuristic backbone + learned component, gated). Given the optimization-gap lessons, which,
    and why? Be concrete about state, action space, and the belief+answerability update.
Q2. HOW ADAPTIVITY EARNS ITS KEEP HERE specifically: what user-specific signal does the agent extract that
    a static schedule cannot, and through which mechanism (refusal-folding? answerability-posterior?
    coarse->fine descent gated on answers)? Tie to the +0.63 genre_match coef and the G2=0.287 bound.
Q3. THE BASELINE (must be brutal): the best STATIC schedule constructible with FULL knowledge of the
    population answerability model (not per-user) — greedy val-selected, mixed channel/granularity. The
    agent must beat THIS, not a dumb heuristic. Specify how to build it so the win (if any) is real.
Q4. PRE-REGISTERED SUCCESS: metric (anytime cost-weighted NDCG? endpoint? questions-to-quality?), the
    decision rule, and the honest expectation (efficiency-shaped, not a giant endpoint delta). What Delta,
    at what CI, over which baseline, counts as "adaptive coarse->granular wins"? What's the fallback
    framing if it ties (the static channel-map still ships)?
Q5. THE ABLATION LADDER that isolates WHY it works (each rung = one mechanism): full agent -> minus
    refusal-folding -> minus answerability-posterior -> minus coarse->granular (flat) -> best static.
    Which rungs, in what order, to attribute the gain cleanly?
Q6. COST/RISK: rough compute, the top-2 ways this fails, and the cheapest early GO/NO-GO signal before
    committing to full training (analogous to how the gate de-risked the study for $0.07).
Q7. Does this design, run and won (or honestly tied), deliver the flagship's crown result WITHOUT
    overclaiming? Write the 1-paragraph result statement as you'd put it in the abstract, both branches
    (wins / ties).

Context: HANDOFF.md (full plan), ANSWERABILITY_RESULTS_LOG.md + ANSWERABILITY_WRITEUP.md (this study),
FABLE_ANSWERS_2026-07-06.md (the channel-map reframe), experiments/instrument2/PHASE4A_BATTERY.md +
SQUEEZE_R01/R2 (the policy-ties + oracle-ladder discipline). Answer from THIS brief first.
