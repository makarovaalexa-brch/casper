# PLAN — Set-input encoder + Multinomial decoder: the unifying architecture (2026-07-13)
Combine the pre-VAE's ARBITRARY-SET input (flexible / continuous / open questions) with the multinomial
recommender's STRONG, SCALING output (~0.5 full-profile NDCG). Design only — no runs yet.

## 0. WHY THE DENSE MODEL IS A DEAD END (the author's call)
Dense u1/SignedAE: one fixed input dim per predefined concept. It CANNOT represent (a) a continuous query
vector (Paper C continuous actor), (b) an open-vocab concept outside the fixed 1,628, (c) an arbitrary
2-step interview without folding the whole catalog as zeros. The thesis needs an ARBITRARY-SET input. But the
pre-VAE set-encoder COLLAPSED at 18k catalog. So: keep the set input, fix the collapse.

## 1. THE KEY INSIGHT — INPUT and OUTPUT are separable
The multinomial STRENGTH lives entirely in the OUTPUT: a Linear decoder z->ni scored by multinomial
log-softmax over the ITEM catalog, with dense per-item gradient every step (Mult-VAE/RecVAE — scales to 41k).
The pre-VAE COLLAPSE was NOT the set-encoder; it was the OBJECTIVE — a listwise softmax over a shared
item+attribute TOKEN vocabulary (tokens competing in the denominator + shared gradients, DIF-SR theorem).
=> Replace the objective, keep the set-encoder input:
   **ARBITRARY-SET ENCODER -> z(512) -> MULTINOMIAL DECODER over items only.**
The decoder never sees concept tokens in its softmax, so no collapse; the encoder can eat any set.

## 2. ARCHITECTURE
- **One shared embedding space for ENTITIES** = items ∪ concepts ∪ (arbitrary continuous queries). Item
  embeddings init from a0c decoder factors (in the scoring geometry). Concept embeddings init member-bag.
  A continuous/open query is just a d-vector token (from the policy / OMP decode) — same space, no new machinery.
- **A token per ANSWERED entity** (arbitrary set; only-asked, no zero-folding):
  `token = MLP( concat[ entity_emb(e), value_emb(v), conf_emb(k), refused_flag ] )`
  - value_emb = Embedding(4) (hated/meh/liked/loved) — LEARNED per level (= one-hot into a linear = embedding).
  - conf_emb  = Embedding(rough/know_well) — LEARNED, SEPARATE from value (NO multiply; the MLP learns the
    value×confidence×entity interaction nonlinearly — the author's requirement).
  - refused = explicit state (asked, no value) — its own signal.
  - Items: value-only at warm-start (real ratings); gain conf+refused in the interview regime.
- **Set encoder = ATTENTION-POOL with a learned query** (June-proven robust on SHORT sequences; NOT a
  set-transformer — V2-ST was better full-profile but WORSE elicitation). Empty set -> z0 (popularity prior =
  intercept, exact). z = z0 + pooled.
- **Decoder = Linear(z -> ni), MULTINOMIAL log-softmax over all 18,430 items.** Loss = multinomial log-lik of
  held-liked. THE strength/scale source. Warm-start from a0c.

## 3. HOW WE REACH ~0.5 ON FULL PROFILES (the hard requirement)
The set-encoder must produce a full-profile z the strong decoder scores at ~0.496. Path:
1. **Warm-start** decoder W + item entity-embeddings from a0c (the scoring geometry is inherited).
2. **z-DISTILLATION to guarantee the 0.5 start**: train the set-encoder so that on FULL PROFILES,
   `set_encoder({item tokens}) ≈ a0c_z(full profile)` (MSE on z, leak-free — teacher z from the same items).
   This transfers a0c's belief into the set-encoder BY CONSTRUCTION, so full-profile NDCG starts ~0.496.
3. **Then train end-to-end** (multinomial) on the full+interview curriculum; the set input gives flexibility
   for free, interviews adapt it, distillation anchor keeps strength.
Evidence it's reachable: V2-ST set-transformer was BETTER full-profile than V1 (set-encoders CAN hit 0.5-class);
June attention-pool encoder beat ridge and accumulated. The risk was elicitation (set-transformer overfits short
seqs) -> mitigated by the single-query attention-pool.

## 4. WHY THIS UNIFIES THE WHOLE PROGRAM
- **Cold-start interview** = a small set of answered tokens -> z -> multinomial rank. Arbitrary length native.
- **Full profile** = the set of all rated item tokens -> z (distilled to a0c) -> 0.5.
- **Continuous / open questions (Paper C)** = the policy emits a continuous query d-vector = one more token in
  the SAME space; no architecture change. This is the payoff the dense model structurally could not give.
- **Confidence learnable, separate, nonlinear** (author's rule): value_emb ⟂ conf_emb, fused by the token MLP;
  the model learns "trust know_well, discount rough, amplify obscure-confident, drown popular-confident."
- **Refusal** = explicit token state, its own signal.

## 5. GATES / RISKS (make-or-break)
- **G-strength [HARD]**: full-profile NDCG@10 >= ~0.486 after distillation + end-to-end (the whole point).
- **G-intercept**: empty set -> z0 -> popularity exactly.
- **G-elicitation**: concept/mixed cold-start beats intercept on TAIL with REALISTIC answerer answers (the
  honest test), confidence ablation (flat vs learned), refusal-as-signal ablation.
- **RISK 1**: attention-pool z can't match a0c's dense-MLP z -> full-profile caps below 0.5. Mitigation:
  distillation; if still short, allow a richer pool (light self-attn block) but watch elicitation overfit.
- **RISK 2**: warm-start mismatch (a0c dense encoder != set encoder) -> can't copy the encoder, only the
  decoder+embeddings; the set-encoder is trained/distilled from scratch. Distillation is the bridge.
- **RISK 3**: token MLP over concat is where value×conf interaction lives; if it underfits, confidence stays
  inert (measure with the confidence ablation).

## 6. OPEN QUESTIONS (for the author / Fable)
1. Distillation (z-MSE to a0c on full profiles) vs pure end-to-end warm-start — is distillation the right way to
   guarantee the 0.5 anchor, or does it over-constrain the set-encoder?
2. Token fusion: concat->MLP (flexible, nonlinear) vs additive entity+value+conf (pre-VAE style, simpler)?
3. Pool: single learned query (robust/short) vs +1 self-attn block (richer/overfit-risk)?
4. Continuous-query token: is the entity_emb of an open query the OMP-decoded blend, and does that stay on the
   manifold the decoder scores?
