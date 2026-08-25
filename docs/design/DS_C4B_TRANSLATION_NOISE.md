# DS_C4B — how much does translating the direction into English cost? (declared-circular, RELATIVE)

**Status: PRE-REGISTERED 2026-08-25 under the author's same-day ruling** (recorded verbatim in
`PAPER_C_DESIGN.md`): a simulated geometric answer model is ADMITTED, declared circular in the
paper's own voice, chosen consistent with field practice (Info-Synth's geometric response model,
PEBOL's simulated users, our June cans rule — cited beside the declaration), and it may carry a
**RELATIVE claim only**: the translation cost. No absolute "asking it helps" claim routes through
the simulator.

## The question

At a state where the policy has a direction worth asking about, how much of what the DIRECTION
would contribute survives being said in English and read back? **Cost = arm L − arm T**, both arms
under the same simulator, so the circular component is common-mode and the difference is the
verbalisation noise — the quantity the author asked for ("measure how translating back adds noise
to the semantics of the question").

## Arms (per state, the C1 pilot's registered 42; users = the state's own cohort)

| arm | question content | source |
|---|---|---|
| **L** (direct) | the direction's own top-4 tower-space nearest concepts (cos of `d_c` to R_s; ties by id) | pure geometry, no text |
| **T** (translated) | the SBERT top-4 concepts of adapter A's concept-hot render of the blend target | the C4 pilot's banked render |
| **T′** (scrambled render) | same, from twin A′'s render | control |
| **P** (prior text) | same, from the channel-matched prior-only text | control |

Each arm's concepts are asked as questions and answered PER USER by the **declared-circular
geometric rule** (the June cans convention: answer = sign of the user's profile projection on the
concept direction; refusal never simulated), folded through the certified concept fold as answered
questions, and the cohort scored full + tail NDCG@10 through the existing funnel with the learned
decoder bias — no new scorer.

## Registered readings (paired per-user bootstrap within state, 10k, seed 98765; cohort-weighted
summary beside per-state rows; aggregation unit declared: user-rows within state, states nested —
the C1 population caveats travel)

1. **Translation cost = L − T**, full and tail, with CI. This is the headline of the experiment
   and it is a COST, reported as such whichever sign it takes.
2. **T − T′ > 0 with CI excluding zero** — the true render must beat the scrambled twin's render
   through the same simulator (directional, per the C2 lesson).
3. **T − P** — the render must beat channel prior text (directional).
4. Overlap accounting: |concepts(L) ∩ concepts(T)| per state (0–4) reported beside every row — if
   T ≈ L simply because the sets coincide, that is the finding, not a confound.
5. Absolute L and T rows are PRINTED and labelled *"under a declared-circular answer model"*;
   neither is ever headlined or compared to any non-simulated number.

## What may NOT be claimed

Absolute NDCG value of asking anything · anything about the item channel (instrument limit stands)
· anything from emission renders (retracted; C4-final re-renders after the declared rescale) ·
generalisation beyond the 42 nested states.

## Mechanics

`src/instrument/c4b_noise.py`, importing the fold/funnel/bridge machinery (never reimplementing);
commit before run; threads ≤ 4; cost ≈ 42 states × 4 arms × (fold + one funnel pass) — well under
an hour nominal, box-estimate caveat applies. Outputs `experiments/paperC/c4b_noise.json` +
per-state npz; write-up `docs/results/C4B_NOISE.md` (Fable's conclusions). Validation only; no
test users; no LLM API.
