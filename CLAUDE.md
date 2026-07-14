# CASPER — project rules (read before any experiment or agent task)

## HARD RULE #1 — NEVER REDUCE DATA WITHOUT EXPLICIT AUTHOR CONFIRMATION
Do NOT sample, cap, subsample, truncate, top-N-select, coverage-sample, or otherwise DROP any data
— users, items, questions, answers, tokens, features, candidates, rows — without the author's
explicit sign-off, obtained first.

This includes (non-exhaustive):
- token caps / `MAX_*` limits / per-channel caps on a fold or belief encoder,
- top-N candidate pools or question shortlists,
- user cohort subsampling ("use 300/500/1000 for speed"),
- truncating an answer/interview set,
- "for speed" / "memory-safe" reductions of any kind.

WHY: this is a small, expensive-to-label dataset (real LLM-judged answers cost real money), and
silent data reduction has repeatedly corrupted results here — truncated b2 cache, top-N pools that
hid signal, 300-user greedy selection that noise-fit, a token cap that crowded out the strongest
(item) channel. Every one looked like a harmless engineering default and every one changed the
answer.

IF COMPUTE OR MEMORY IS THE CONCERN: use a NON-LOSSY alternative — length-bucketed / micro-batching,
chunked or streaming processing, sparse ops, per-item folding at eval (no padding), caching. Never
drop data to fit. If you genuinely believe a reduction is unavoidable, STOP and ask the author with
the exact tradeoff stated; do not proceed on your own judgment.

This rule applies to Fable (the overseer) AND every subagent. Every agent brief that touches data
must restate it. Use ALL the data unless told otherwise.

## Working rules (from standing author guidance)
- DESIGN SHEETS BEFORE EXECUTION: no experiment runs until the author signs a one-page design sheet
  (question, exact sample Ns, full action space, baseline symmetry, metric+MDE, every shortcut
  flagged with its alternative). Silent shortcuts are the enemy.
- The 173 LLM-judged / 300 study users are QUARANTINED — never train/val/test a fold or policy on
  them; touched only on the author's explicit schedule.
- No LLM API calls without explicit author approval.
- Report faithfully: if a gate fails, say so; never market a failure as a contribution.

## HARD RULE #2 — NEVER AGAIN DISPUTE THE VALUE OF ADAPTIVITY WITH THE "STATIC IS OPTIMAL" ARGUMENT
**Settled empirically on 2026-07-14. See `experiments/ADAPTIVE_PROBE_RESULT.md`.**

On 150,239 users, with NO policy, NO RL and NO optimisation (so it cannot be confounded by a weak static arm):
ask ONE genre question, cluster users by the answer, and give each cluster its own best follow-up item.

    TAIL NDCG@10:  static 0.0414  ->  adaptive 0.0608
    PRIZE = +0.0194   95% CI [+0.0186, +0.0203]   **+47% RELATIVE**   (n = 74,881 eval users)
    (selection and evaluation on DISJOINT halves of users, symmetric for both arms)

**AND THE MECHANISM IS THE HUMAN INTERVIEWER'S:**
- The best question for EVERYONE is a blockbuster: **The Usual Suspects (popularity rank 8/800)**.
- The best question for a CLUSTER is a NICHE film **inside that cluster's own genre** (median rank **471/800**):
  horror -> *Poltergeist*; sci-fi -> *Planet of the Apes*; animation -> *Laputa / Howl's / Totoro* (all Ghibli);
  family -> *Pinocchio*; comedy -> *Wayne's World*; drama -> *Citizen Kane*.
- WHY: once you know someone is a horror fan, "do you like The Usual Suspects?" tells you NOTHING (everyone
  does). What discriminates WITHIN horror fans is *Poltergeist*. **The informative question for a homogeneous
  group is one that POLARISES that group** — always a moderately-obscure, genre-congruent film.
- This is why the effect is in the **TAIL (+47%)** and barely in the head (+6%): a popular probe only re-ranks
  the head, which popularity already gets right. A niche probe locates the user in a NICHE SUBSPACE.

**COARSE-TO-FINE FELL OUT OF THIS, UNPROMPTED:** Q1 = genre (broad, population axis) -> Q2 = a niche film
WITHIN that genre (narrow). And the STATIC Q2 is another blockbuster. **So coarse-to-fine is not merely
emergent — it is only AVAILABLE to an adaptive policy: the "fine" question only exists once the coarse answer
says where to point it.**

**THEREFORE, BINDING:**
1. Do NOT argue "the theorem says a static questionnaire is optimal, so adaptivity is worthless." The
   non-adaptivity theorems (Krause & Guestrin ICML'07; Jedynak 2012) assume a LINEAR-GAUSSIAN belief and a
   VARIANCE-ONLY objective. **We have neither** (learned nonlinear fold-in; NDCG task loss). Krause's own next
   sentence: *"for non-Gaussian models, sequential strategies can strictly outperform a priori designs, even
   with known parameters."*
2. Prior nulls (E0's -0.0003; the eight tied policies) are **ARENA-CONDITIONAL** — discrete pool, binary
   answers, full-NDCG, q=8, population mean. They do NOT generalise to elicitation as such.
3. **Before ANY claim that adaptivity does or does not pay: (a) use ALL the labelled data (153k users, not the
   3k val slice — a 3k run produced the OPPOSITE, and wrong, answer), and (b) report TAIL, not just FULL (the
   effect is 47% on tail and 6% on full).**

## HARD RULE #3 — THE THEOREM IS CLOSED. THE EIGHT FAILED POLICIES ARE DEAD.
1. **NEVER cite, re-derive, or reason from the non-adaptivity theorem again.** It is settled (HARD RULE #2) and
   it BIASES the analysis into a spiral: every time it is invoked, the conclusion drifts back toward "adaptivity
   cannot pay", which is FALSE and is contradicted by our own 150k-user measurement. It does not explain any
   result in this project. Do not use it to predict a result. Do not use it to excuse one.
2. **THE EIGHT FAILED POLICY RUNS ARE DEAD AND BURIED.** They go in **NO paper, in NO form**. They are not a
   finding, not a contribution, not a caveat, not a "negative result". **They lost because THE MODELS WERE WEAK**
   (V1 encoder). The recommender is now a 0.4852 set encoder — a different machine. Do not analyse them, do not
   cite them, do not "explain" them. Forget them.
3. The only live question about the policy is: **does it work on the STRONG model?** Test that.
