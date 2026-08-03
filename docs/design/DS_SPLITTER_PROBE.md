# Design sheet — k=2 splitter probe (the experiment that decides Paper B's shape)

**Status: DESIGN ONLY. GATED — do not run until BOTH gates below pass.**
Origin: Challenge 2 adjudication in `PAPER_B_DESIGN.md` (author's weak-opener hypothesis, upheld).

## Gates before running
1. **Concept-tail gate (#74):** `concept_cold_check.py --policy niche/uniform` must be run and read
   first. If niche does NOT recover tail (head-alignment is in the genome centroids), this probe is
   scored FULL-ONLY and the tail scope condition is printed; the opener pool is unchanged. If niche
   recovers tail, the opener pool below gets a tail-aware arm.
2. **Dispersion census (item banks) done** — establishes the "≈0 under standard banks" leg so the
   probe has a documented contrast.

## Question (fixed now, before outcomes)
Does an opener CHOSEN TO MAXIMIZE across-user posterior divergence, followed by per-partition
question 2 (fitted on train users only), beat the jointly-optimized static 2-question set by
≥ 0.004 NDCG@10 (paired) at k=2 — and does any gap persist when both arms continue statically to
k=4/8?

## Decision rule (both branches publishable — written before running)
- Gap ≥ 0.004 at k=2, survives controls → adaptivity section gets the positive branch; Paper B
  targets RecSys 2027 / TORS with the validated-diagnostic framing.
- Gap < 0.004 WITH confirmed inter-partition Q2 divergence → the exploitability null: "divergence
  exists and cannot be cashed." Print it; TORS/ECIR framing.
- Gap < 0.004 WITHOUT inter-partition Q2 divergence → the probe failed to create divergence;
  claim NOTHING about adaptivity from it (a weak opener re-finds D3's null and proves nothing —
  this outcome is explicitly anticipated and is not evidence).

## Arms
- **A (splitter):** opener = the top divergence-scoring broad concept (below); partition the test
  users by their behavioral answer (like / dislike / refuse-unknown); per partition, Q2 = best
  single question by greedy search ON TRAIN USERS OF THE SAME PARTITION (never test-fitted);
  then static continuation to k=4/8 using the E3-style per-partition greedy.
- **B (strongest static):** the jointly-optimized static 2-question set built on train users —
  the strongest constructible comparator (comparator-construction scar; non-negotiable).
- Floors/ceilings per cell: prior-mean (k=0), MostPop, random-answerable, clairvoyant oracle @k2.

## Opener selection (train users ONLY)
Candidates: broad concepts (≥1000 members, answer rate ≥ ~80% by the structural rule — D3 failed
precisely because its item-channel root had answer rate 2.18/8 and the partition was degenerate).
Score = Var_u[Δμ_u] — across-user variance of the induced posterior-mean update, estimated from
train users' behavioral answers on frozen ep31. Take top-3 openers; report all three (no
best-of-N selection on test).

## Exact Ns and data
- Instrument: frozen i26c ep31, no-floor primary, floored robustness. Concept fold: cv3 ep7
  (or its successor if #74 replaces it — record the tag).
- Opener scoring + Q2 fitting: the 139,268 train users (or the standing 1500-held-out-train-user
  cold-check protocol for the fold path — whichever, STATE IT and keep it fixed).
- Evaluation: the canonical 10k TEST users, first exposure (they are untouched by this design).
- Metric: NDCG@10 full AND tail, paired bootstrap over users, MDE 0.004.

## Controls (Hard Rule 5)
- **Shuffled-answer partition:** permute opener answers across test users → the A−B gap must
  vanish. If it doesn't, the "gap" is partition-size structure, not information.
- **Popularity-matched opener:** an opener matched on popularity/answer-rate but LOW divergence
  score → its gap must be smaller than the chosen opener's, else the effect is popularity, not
  divergence (M8).
- **MostPop gate** on the regime before interpretation; canonical snap at k=0; leak check that
  no per-partition Q2 fit touched test users.
- Refusal partition is a real partition (i26 has the exposure channel) — report its size and its
  per-partition gap separately; a probe where 40% of users land in "unknown" is D3's failure mode
  and must be visible, not averaged away.

## Shortcuts and non-lossy alternatives
- No sampling of test users, no capping the opener candidate list silently (Hard Rule 1). The
  candidate list is defined by the ≥1000-member + answer-rate rule, stated exactly, applied to all
  1031 genome concepts. If compute forces a reduction, STOP and ask.

## Output
`docs/results/SPLITTER_PROBE_RESULT.md` + same-session update of `PAPER_B_EVIDENCE_POLICY.md`
(this experiment moves from NOT-YET-SEEN to SEEN, with its branch recorded).
