# DESIGN SHEET — LEAK-FREE FOLD/ANSWERER + A/B/C ablation
Status: LOCKED by author 2026-07-10 ("add entropy, proceed with ablations; individual not combos").
Builds on the recovered fold (attention-pool + fixed-prior + content-confidence gate, unified
tokens_for, NO caps). Supersedes the leaky-surprise version.

## 0. Principle (the invariant)
A token the fold sees may contain ONLY: (a) what a human volunteers in the interview — value +
confidence; (b) PUBLIC info — entity popularity p_E, entity entropy H_E; (c) statistics over what
the interview has REVEALED so far (accessible-to-agent set). NEVER the ground-truth profile.

## 1. Tokens (leak-free)
Per answered question, via the ONE unified `tokens_for` (interview and full-profile identical;
only the QUESTION SET differs by regime):
- entity embedding (512-d),
- VALUE: real centered rating (rated) ∪ EASE-predicted sentiment (unrated, EASE on NON-HOLD-OUT
  input only) — user's internal knowledge, volunteerable,
- KNOWLEDGE LEVEL: {no_clue, rough, know_well} (implicit channel; carries consumption direction),
- FIDELITY/confidence,
- PUBLIC: p_E (popularity) AND H_E (population rating-entropy = discriminativeness). Feed BOTH
  (entropy alone is coverage-confounded — the dolphins/antarctica noise; p_E disentangles it).
DROP: profile-derived n_E / V / hand-crafted surprise ratio (the leak; not human-articulable).

## 2. Tightness/spread = the pooling geometry (not a formula)
Consumption "surprise"/clustering is NOT an explicit profile feature. Tight taste = aligned answer
embeddings reinforce under attention-pool → high-norm/coherent belief → sharp recs in cluster;
spread taste = cancel → low-norm → flat. This is the content-confidence gate's coherence term.
Low-info (1-2 answers) is handled by that gate (near prior when sparse) — no fabricated signal.

## 3. THE ABLATION — INDIVIDUAL variants (NO combos/grid; author directive)
Base = §1 leak-free tokens (entropy IN) + §2 emergent coherence + efficiency (§5).
- **A (base):** single belief vector, emergent coherence.
- **B:** + explicit coherence feature (compute set-coherence, feed into the belief so the fold
  doesn't have to rediscover it).
- **C:** multimodal belief (multi-head pool → K region sub-beliefs; item scored by best-match) so
  "loves Nolan AND rom-coms" is representable.
Each is ONE change from base. Run A, B, C separately; pick best. No 6-cell grid.

## 4. Subset (author-authorized data reduction — ABLATION ONLY)
~5,000 train / 1,000 val / 1,000 test population users (study IDs excluded), early-stopped. Fast
(~few min/epoch after §5) but large enough: 1k test → NDCG@10 CI ~±0.006, separates variants; 5k
train ranks them reliably (absolute lower than the 20k final, ORDERING holds). PER HARD RULE #1:
this subset is explicitly authorized for the ablation only; NO headline number comes off it — the
WINNER re-runs on the FULL 20k/2k/2k with the complete gate suite.

## 5. Efficiency (non-lossy — no data dropped)
- Vectorize the answerer across users (stack N rating vectors → one (N×9352)@ease_B matmul).
- Cache the static clean full-profile token sets per user (profile doesn't change across epochs).
- Early-stop on val (drop the fixed 12 epochs).
- No caps (all answerable tokens; length-bucket for memory).

## 6. Gates (on TEST; §7 winner runs the full recovered suite)
Pick-the-winner metric: test NDCG@10 (interview endpoint@8 AND full-profile clean), tie-broken by
GoT/prolific. Plus a DIAGNOSTIC readout: do tight-cluster users get cluster recs regardless of
stars (the author's expectation)?
NEW GATE — **G-no-profile-leak** (the direct falsification): fold an interview reveal → belief;
perturb the user's UNREVEALED profile (swap held/unasked items); rebuild the SAME asked tokens →
belief'. Assert belief == belief'. Invariance to unrevealed items = provably no profile leak.
Honest expectation: GoT/prolific will come DOWN from the leaky version — that's the honest number.

## SIGN-OFF: author 2026-07-10 ("ok, then add entropy to your features and proceed with ablations",
"do not run combos yet, just individual ablations").
