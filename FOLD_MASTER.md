# FOLD — MASTER (requirements · past failures · proven models · gates · design)
Deep history review (commits + docs + author notes + this session's 4 Fable passes), 2026-07-11.
Purpose: never re-derive, never re-fail. Design must satisfy every REQUIREMENT, avoid every FAILURE,
and have a GATE for each. Fable reviews check the DESIGN against these lists.

## A. PROVABLY SUCCESSFUL MODELS — NO LEAK (ground truth; commits cited)
- **June `encoder_recon.py`** (biased-SVD): attention-pool reconstruction of UNREVEALED likes, IPS-weighted;
  beat ridge (full 0.338 / tail 0.119); ACCUMULATED. No surprise. Data-side. No leak. (commits 7a64aa8, 853651a)
- **`i25_fold`** (07-07, RecVAE-d512): **Deep-Sets SUM-pool residual** `z = native_z + rho([sum, native_z, log1p(ntok)])`,
  native_z = enc_items(liked items); concepts/entities = member-bag embeddings **through rho (z-space)**.
  Clean 0.4647 (native 0.4787), 5/6 gates, **accumulates** (I2 Phase-1.5 k-curve monotone 0.136→0.290, commit c64ea88).
  **NO surprise. Data-side ("cos(z*,q) nowhere; non-circular from birth"). NO leak.** ← THE proven RecVAE fold.
- **v3** (07-09, commit 58ff224): i25 + two-channel + SURPRISE → G2 GoT +0.184, G2b +0.316, **clean BEATS native +0.017**,
  implicit load-bearing G7 +0.020. Architecture proven. BUT surprise = profile LEAK (see F4) → arch keep, feature drop.
- **RecVAE-d512 scorer** (I2 Phase-2, 60fdb8c): full-profile 0.5541 = tie EASE. Certified ruler.

## B. REQUIREMENTS (full)
- R1 HUMAN SIGNAL IN FULL: use BOTH graded axes — implicit knowledge {no_clue/rough/know_well} AND explicit
  value {hated/meh/liked/loved} — DISTINGUISHED (never collapsed to binary), all channels, two-channel.
- R2 ANY INTERVIEW STRATEGY: random/entropy/EIG/on-profile/off-profile/adversarial/mixed; lengths 1→full;
  refusals; quality mix. Accumulate across every length.
- R3 ACCUMULATE: more distinct answers → sharper belief (monotone-INCREASE, not flat); full profile ≥ native−0.03.
- R4 CLEAN INTERCEPT: empty → prior, fold-independent, exact.
- R5 LEAK-FREE: belief invariant to UNREVEALED profile; no surprise-from-profile; member weights genome×popularity
  ONLY (never user EASE/counts); members never in the score-exclusion mask; data-side answers (no cos(z*,q)).
- R6 CONSUMPTION-AS-TASTE (GoT): know-well pulls TOWARD region regardless of stars — entities AND items (item hole) —
  via a LEAK-FREE carrier (knowledge level, and/or reveal-derived count).
- R7 GRADED EXPLICIT REFINEMENT: loved>liked>meh>hated within a region (requires ordinal loss; likes-only binarizes).
- R8 NO COUNT-CONFOUND: duplicate-invariant; accumulate on DISTINCT evidence.
- R9 ORDER/PERMUTATION INVARIANT.
- R10 NON-LOSSY: no caps; all answerable tokens; length-bucket for memory.
- R11 CONCEPTS IN Z-SPACE: concepts/entities/dislikes enter via LEARNED z-space channel (member-bag embeddings
  through rho), NEVER via RecVAE input member-bags (F1).

## C. PAST FAILURES — do NOT repeat (realised or designed)
- F1 **member-reduction into RecVAE INPUT** (pseudo-item folding, incl. the "member_z=enc_items(bag)" idea): FAILS —
  dilutes below pop prior, nonneg multinomial can't express dislike (I2 Phase-1.6, b67912e). Concepts → z-space.
- F2 additive latent operator z'=z+ηaq: certified on GEOMETRIC answers, BREAKS on REAL (one true answer HURT −0.053).
- F3 **fixed-prior + scalar-gate + softmax-MEAN wrapper** (recovered/leakfree, this session): SATURATION (flat after
  turn 1); mean can't accumulate; `agree` term anti-accumulates (max at 1 token).
- F4 surprise-from-full-profile (v3/v3.1): profile n_E injected in interview = LEAK.
- F5 on_profile=0.7 privileged curriculum (v3.1): OOD to blind interviews.
- F6 fat-tail de-OOD (v4): NaN (q718 corrupt emb) + cold-collapse 0.167.
- F7 Set-Transformer/MHSA: 3× rejected, worse at elicitation (overfits short sequences).
- F8 z-scoring the residual (June): undid shrinkage.
- F9 non-residual plain concat MLP (i25 first try): item folds far below native.
- F10 likes-only + profile-masked LOSS: binarizes explicit (sign(value) sufficient → loved≈liked). No held-disliked
  set exists in cohorts (== June H2 "likes-only → weak polarity").
- F11 knowledge-level weight on a single member bag through enc_items: erased by RecVAE L2-norm (rough≡know_well).
- F12 single member union: within-union dilution (lone know_well washed out in a long interview).
- F13 token/data caps (concept-crowding; any): banned.
- F14 gate suite blind to saturation (G-noQ1drop tolerates flat): BUG 4.
- F17 two-head instrument: COLLAPSE (failed polarity fix, June).
- (context leaks, not fatal, recorded) F15 EASE trained on full matrix (ruling: EASE-from-non-holdout OK);
  F16 RecVAE scorer on full matrix (transductive caveat).

## D. GATES (a gate per key thing; ALL session gates retained; all must PASS on held-out TEST)
- G-intercept (R4): empty ≡ prior, max-dev 0.
- G-no-profile-leak (R5): belief byte-invariant to scrambling UNREVEALED items; + traps (weights genome×pop only;
  members not in exclusion mask).
- G-clean (R3): full-profile fold ≥ native − 0.03. **HARD STOP.**
- G-monotone-increase (R3): span CI — (turn8−turn1) ≥ +0.02 CI>0; (clean−turn8) ≥ +0.05; also on a CONCEPT-ONLY curve.
- G-value-monotone (R1,R7) ×{short,long,full}: sweep one token hated→meh→liked→loved, adjacent-pair CI>0; **+ binarize
  ablation must LOSE** (grading must buy NDCG, not just respond).
- G-know-graded (R1,R6) ×{short(1 answer),long(embedded in 20),full}: absent<rough<know_well adjacent CI>0; **+ collapse
  rough→know_well ablation must LOSE.**
- G-GoT (R6): region-pull(bad-star know-well) − pull(never) > 0, for entities AND items; + in-context (len-20) variant;
  report FRACTION of users with net pull<0.
- G-prolific (R6): selective > diluted.
- G-order (R9): shuffle invariance.
- G-falsify-count (R8): duplicate the token set → belief unchanged.
- G-caplength (R3): elicitation not sacrificed for full-profile.
- G-canaries (R2): one answer per channel×token-type helps from cold.
- G-implicit-ablation (R1): zeroing implicit tokens DROPS NDCG AND does NOT strengthen region-pull (cancel-check).
- G-k2-graded (R7): reproduce i25 k=2 beat-native (+~0.012); vanishes under value-zeroing (proves it rides explicit).
- β/level telemetry: print per-(channel×level) scalars each epoch if used.
- G-token-count-audit (R10, pass-1/2): assert (#tokens emitted − #removed by declared dedup) == #tokens pooled —
  proves no silent token dropping/capping beyond the declared collision dedup.

## E. DESIGN (locked candidate — grounded in A, avoids C, gated by D)
**Base = the PROVEN i25/v3 Deep-Sets SUM-pool residual (NOT this session's wrapper).**
`z = native_z + rho([ SUM-pool(tokens), native_z, log1p(#DISTINCT tokens) ])`
- native_z = RecVAE.enc_items(revealed **LIKED** items only) — item anchor, accumulates natively. **ITEM-HOLE (R6),
  LOCKED default (pass-1 fix):** consumed-but-not-liked items do NOT enter native_z; every item (liked or not) ALSO
  emits implicit+explicit TOKENS, so a consumed item's know-well IMPLICIT token carries item-consumption GoT in
  z-space through rho (same channel as concepts, R11) and the explicit token refines. The multi-hot arm (also feed
  consumed items into native_z) is a PRE-REGISTERED ABLATION, adopted only if it passes G-clean AND G-GoT on val —
  never a build-time fork.
- TOKENS (v3 two-channel, leak-free): IMPLICIT (channel, member-bag emb, knowledge-level {rough/know_well}); EXPLICIT
  (channel, member-bag emb, value {4-level}, fidelity). Concepts/entities = member-bag embeddings **through rho**
  (z-space, R11), member weights genome×pop only (R5). **NO surprise-from-profile (F4).** If GoT needs a count, use
  reveal-derived (over the asked set) count — leak-free (R5,R6).
- rho = 2-layer MLP, **zero-init last layer** → empty → z=native_z=prior, exact intercept (R4).
- **SUM-pool OVER THE DISTINCT TOKEN SET** (pass-1 fix R8: dedup by (channel, entity, kind) BEFORE pooling, so
  folding the same answer twice is a no-op) — accumulates (R3; not mean/F3) + **log1p(#distinct tokens)** (F13 no cap).
  **COLLISION RULE (pass-2 fix):** if the same (channel,entity,kind) is answered with CONFLICTING value/fidelity
  (a re-ask; R2 allows it), keep the **HIGHEST-FIDELITY** answer (tie → highest turn-index). This is a deterministic
  function of the token SET → order-invariant (R9/G-order preserved); "latest-wins" is banned (it needs an order).
  Deployable policies don't re-ask, so this is a safety rule, not a hot path. IMPLICIT tokens (no fidelity field):
  collision keeps the HIGHER knowledge level (know_well > rough), tie → highest turn-index (pass-3 clause).
- **DATA HYGIENE (pass-1 fix F6):** screen ALL member-bag/Qemb embeddings for non-finite rows and zero them (the
  q718 class) BEFORE training; assert finite loss per step.
- LOSS = held-likes reconstruction + **λ · ORDINAL graded term** (R7): hold a STRATIFIED rated sample across all 4 star
  bands (leak-free: held ∉ known/native_z); margin `score(i)−score(j) ≥ m·(star_i−star_j)`. Fixes F10.
  PINNED (pass-1): λ=0.3, m=0.5 star-units as defaults; both swept coarsely on val ({0.1,0.3,1.0}×{0.25,0.5,1.0}),
  best-on-val kept — no silent hyperparameter drift.
- CURRICULUM (R2): any-strategy mixture; **log-uniform lengths 1→full** (fills 24→full gap); 30% clean; natural
  refusals; NO caps (R10, F13); data-side answers (R5).
- SPLIT: 20k train / 2k val / 2k test population users; 173/300 quarantined. All gates on TEST.
- Rollback: this is i25's proven recipe + leak-free tokens + ordinal loss — degrades to i25 (0.4647) if extras add nothing.

## F. THE FIVE-PASS LOOP (this task)
I write/amend DESIGN (§E) → token-light Fable strategic pass (satisfies every R? avoids every F? gate per R?) →
I amend → repeat ≤5 or until Fable 100% happy → train best iteration(s). Fable gets §B/§C/§D/§E on a silver platter,
returns only per-item satisfied/GAP + the single most-dangerous residual risk.
