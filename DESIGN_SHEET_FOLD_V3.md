# DESIGN SHEET — FOLD-V3: the two-channel belief encoder
Status: DRAFT FOR AUTHOR SIGN-OFF. Nothing executes until marked. (Rule: design-sheets-before-execution.)
Predecessor: fold-v2 (.cache/i25_fold_v2_best.pt) — in-regime gates pass, but (i) treats knowledge
level ONLY as a reliability weight on the liking value (discards consumption-as-taste — the
author's GoT insight), (ii) gives up 0.359-vs-0.494 on clean full profiles (G4 tradeoff), (iii) was
trained on the pre-repair answer distribution.

## 1. The question this answers, and what changes per outcome
Build the belief encoder that hears BOTH things an answer says: the EXPLICIT channel (values E at v)
and the IMPLICIT channel (knows E at level l — consumption/knowledge as first-class taste
evidence), with the blend LEARNED, trained on the v2.1 answerer distribution.
- Gates pass -> the arena instrument for all policy work; policy design sheet unlocks.
- The GoT gate fails -> the two-channel hypothesis itself is challenged on this data; stop, report.

## 2. Token design (author decisions baked in from conversation)
- Every answered question emits up to TWO tokens:
  IMPLICIT token: (channel, entity, knowledge-level in {rough, know_well}) — knowledge as taste
  evidence, strength learned per level. NO hand weights.
  EXPLICIT token: (channel, entity, value in the 4-level scale, fidelity-class in {data, ease,
  llm-style}) — as v2, fidelity feature learned.
- NO-CLUE on an ASKED question emits an implicit-negative token (weak "not in their world"
  evidence; learnable, may learn ~0) — only asked questions emit (no flooding).
- Channels: items, concepts (certified token interface), attributes/entities (decade, genre,
  director/actor/franchise from the IMDb join — NEW vocab entries).
- DECISION D1: two separate tokens (recommended — the blend is genuinely learned) vs one token
  with (level, value) features. REC: separate.

## 3. Training
- Sampler: the v2.1 answerer (real∪EASE values, corrected trait dial, equated knowledge rates) on
  population trU users — the clean distribution, matching what interviews will feed it.
- MIXED CURRICULUM (closes the v2 clean-profile gap): reveals span noisy partial interviews
  (1-24 answers, mixed channels) AND clean full profiles, ratio swept coarsely {30/70, 50/50}.
  DECISION D2: accept the sweep (2 runs) or fix 50/50. REC: sweep once.
- Architecture: Deep-Sets residual (v2 recipe) as primary; ONE Set-Transformer variant if the
  anti-saturation gate (G6) fails on Deep-Sets. DECISION D3: agree ST as contingency only. REC: yes.
- Standard discipline: frozen RecVAE decoder, disjoint val cohort, all checkpoints, deterministic.

## 4. Gates (pre-registered; printed before results)
- G1 canaries per channel AND per token type from cold: one explicit answer helps; one IMPLICIT-ONLY
  token (knows E, value withheld) helps — the direct test that consumption alone is taste evidence.
- G2 THE GoT GATE (author's canary, verbatim): a user who has watched-everything-of-X and rated it
  BADLY must be pulled TOWARD X's region relative to a user who never heard of X (valence refines
  within-region; consumption dominates direction). Constructed probe pairs; CI excl 0.
- G3 dilution: adding vague (rough/llm-style) answers to a vivid set must not reduce NDCG beyond
  noise (the learned blend must down-weight, not drown).
- G4 clean-profile: full-profile through fold-v3 within 0.05 of native RecVAE (v2 gave up 0.135;
  the mixed curriculum should close most of it). DECISION D4: threshold 0.05 (REC) or stricter.
- G5 no-harm vs v2: on noisy interview reveals (the v2 regime), v3 >= v2 - 0.005.
- G6 anti-saturation: NDCG strictly increasing in answer count 1->24 on mixed reveals (the audit's
  "1 answer ~= 16" symptom must not exist); max per-step decline > -0.003.
- G7 implicit-channel ablation: zeroing the implicit tokens must DROP performance (the channel is
  load-bearing, not decorative).
## 5. What this build explicitly does NOT do
No policies, no arena, no statics, no LLM calls, no touching the 173 eval users (gates run on
held-out population users; the GoT probe pairs constructed from population profiles).

## 6. Compute/cost: $0; ~3-5 training runs x ~25 min CPU + gate battery. Opus executes after sign-off;
Fable reviews gates before anything consumes v3.

## SIGN-OFF — SIGNED 2026-07-10 (author amendment + delegated decisions)
[x] AUTHOR AMENDMENT (the prolific correction): implicit evidence = SURPRISE, not raw engagement —
    each implicit token carries a lift feature (engagement with E relative to what the user's
    overall volume/answerability level predicts). NEW GATE G2b (prolific control): a SELECTIVE user
    who watched all of X must be pulled toward X MORE than a PROLIFIC user who watched all of X
    plus everything else. G2+G2b jointly decisive.
[x] D1 separate tokens (Fable, delegated): two entries per answer; pure-knowledge events natural;
    clean implicit-channel ablation.
[x] D2 curriculum: sweep 30/70 and 50/50, keep best-on-val (2 runs).
[x] D3 Deep-Sets primary; Set-Transformer contingency iff G6 anti-saturation fails.
[x] D4 clean-profile threshold 0.05.
[x] Entity embeddings: member-bag aggregates v1; CONTINGENCY = native user-x-entity engagement
    factorization (kmap-style) iff entity canaries weak on member-bag.
