# Exploration/belief ablation — the LEVER is item-grounding, NOT exploration (2026-07-01)

Motivation: PEBOL-geo (0.369/0.168) beat our CASPER-R (0.360/0.152) on the TAIL. Hypothesis 1: Thompson sampling helps
the tail. Deep-research (2 agents) reframed: our greedy-EIG = BALD, entropy = max-entropy sampling => we ran the
PARAMETER-uncertainty family => "adaptive ties static" is expected; unexplored axes = optimism(UCB), outcome(EPIG/EVOI),
diversity(DPP).

## TSCASP (abstract-u posterior TS) — FAILS
Bayesian-linear posterior over the abstract taste vector u; sample u_hat; disc=argmax concept / cont=sampled direction.
disc q8 0.311/0.112 ; cont q8 0.276/0.084 — WORSE than CASPER-R. Abstract-belief exploration over 8 turns = uninformed;
continuous random-posterior query << learned D1 query. Honest negative.

## EXPLORE (ITEM-grounded Beta belief, swappable acquisition; our encoder rank) — q8, seed-avg{1,2,3}
| ACQ | FULL | TAIL |
|---|---|---|
| greedy | 0.3702 | 0.1669 |
| ts (Thompson) | 0.3687 | 0.1683 |
| ucb | 0.3695 | 0.1678 |
| dpp (diversity) | 0.3691 | 0.1675 |
| eps (epsilon-greedy) | 0.3644 | 0.1626 |
ALL exploration acquisitions TIE ~0.369/0.167. eps (random) slightly HURTS. DPP-diversity adds nothing.

## ⚠⚠ CRITICAL CORRECTION (Jul 1, user-caught LEAK) — the item-grounding gain was a PEEK, NOT real ⚠⚠
The item-grounded Beta belief used cand=list(half) = the user's KNOWN-HALF ITEM IDENTITIES (per-user) => it PEEKS at
which movies the user has, then asks concepts tuned to them. A real elicitation policy cannot know this. FAIR re-run
(FAIRCAND=1, cand=shared top-500 popular items, no per-user peek, Beta updated only by answers):
| | LEAKY (cand=known half) | FAIR (shared popular) |
|---|---|---|
| greedy q8 | 0.370/0.167 | 0.271/0.083 |
| ts q8 | 0.369/0.168 | 0.272/0.083 |
=> The ENTIRE item-grounded advantage (0.167 tail) was the LEAK. Fair version 0.083 << CASPER-R 0.152.
BOTH LITBASE baselines PEEK: PEBOL-geo (cand=half) AND ConTS (item arms = Qn[j] for j in half). So the Paper-D baseline
table PEBOL-geo 0.369/0.168 + ConTS 0.351/0.151 are LEAKY/INVALID as run. Fair versions ~0.27 (weak; generic-item
acquisition isn't personalized). CORRECTED: our fair CASPER-R (0.360/0.152) was NOT beaten by PEBOL. The "item-grounding
lever" + "exploration helps tail" findings above were CHASING A LEAK ARTIFACT — RETRACTED. Continuous D1 (0.377) and open
recall (0.405) are UNAFFECTED (never peeked). ACTION: re-run PEBOL-geo/ConTS FAIR (general candidate set) for the baseline
table; CASPER-R stands as our discrete method. Repro leak test: EXPLORE=1 [FAIRCAND=1] ACQ=greedy QPTS=8.

## (SUPERSEDED by the leak correction above) VERDICT (clean, honest, ours-to-state)
The TAIL lever is the BELIEF REPRESENTATION (item-grounded P(like known item)), NOT the exploration algorithm. Even GREEDY
over the item-grounded belief gets 0.167 tail (+0.015 vs CASPER-R). Corrects "TS helps tail" -> "item-grounding helps
tail; acquisition irrelevant." EXPLAINS the project's "adaptive ties static": we varied the ACQUISITION (irrelevant) over
an ABSTRACT belief u, never the belief representation. A good item-grounded DISCRETE policy (0.369/0.167) nearly matches
continuous D1 (0.377/0.175) => "discrete ~= continuous; don't snap." OPEN RECALL (0.405/0.195) still beats all of this.
Repro: EXPLORE=1 ACQ=greedy|ts|ucb|dpp|eps QPTS=8 EVALSEEDS=1,2,3 python scripts/paper2/continuous_actor.py

## Deep-research method map (what we missed / to cite)
- Our greedy-EIG = BALD (1112.5745); entropy = max-entropy. Parameter-uncertainty family => modest adaptive gains (linear).
- RUN (research shortlist): LinUCB/Bayes-UCB (1003.0146, mandatory TS contrast — DONE via EXPLORE ucb, ties);
  DPP-diverse (1709.05135/1907.01647 — DONE, ties); IDS (1403.5556, theoretically dominates TS for elicitation — NOT yet);
  EPIG (2304.08151, outcome-oriented EIG reweight to tail — NOT yet, low-cost); SAC (1801.01290, for continuous Paper-C
  actor vs mode-collapse — NOT yet).
- CITE not run: KG/EVOI (EVOI-soft-attributes 2311.02085 must-cite), GP-UCB/EI, GFlowNets, RND/count-based, PSRL,
  BatchBALD, submodular, Decision Transformer, PPO/A2C, calibrated-rec (Steck'18).
- CRS baselines MISSED (cite/compare): UNICORN (2105.09710, standard discrete RL CRS — conspicuous gap), MetaCRS
  (2205.11788, cold-start few-turn RL, independently finds our fixed-policy-fails blocker), Long-Tail CRS (2307.11650,
  only prior CRS on long tail — needed to position our tail headline), ConUCB (1906.01219, bandit PEBOL counterpart),
  ConTS (2005.12979, closest TS prior).
NEXT (optional): IDS + EPIG (do the item-grounded belief + outcome/info-ratio acquisitions beat 0.167 tail?); SAC for
continuous actor. But headline already clean: item-grounding is the lever; open recall wins overall.
