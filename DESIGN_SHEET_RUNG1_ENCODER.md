# DESIGN SHEET — Rung I: Answer-Native Encoder on a Frozen RecVAE Decoder (2026-07-11)
For author sign-off + Fable 5-pass adversarial review. Context docs: FOLD_MASTER.md (R/F/gates),
INSTRUMENT_REQUIREMENTS.md (R1-R8 + illustrative gates), INSTRUMENT_VARIANTS.md (Fable's hybrid verdict),
EMBEDDING_FOLDING_INDEX.md (lineage/NDCG/leaks). This is the 7th fold done on the RIGHT loss + the one
mechanism (channel-dropout) that explains all six prior failures — NOT a new recommender.

## 0. QUESTION / DECISION
Can a learned answer→z encoder (frozen RecVAE decoder; ordinal loss + channel-dropout + confidence-as-
precision) make the rich channels NON-INERT — value-zeroing MOVES NDCG, polarity flips a region, confidence
scales the pull — while (a) matching the fold's clean NDCG (full-profile ≥ native − 0.03), (b) accumulating
monotonically over interview length, (c) leak-free? 
DECISION RULE: PASS on all HARD gates → this is the instrument for Papers C/D + the adaptivity arena. FAIL by
frozen-decoder CEILING (posterior collapses to prior at short lengths, OR dislike only expresses along the few
probe directions) → escalate to Rung II (joint decoder retrain). FAIL by our own bug (saturation/leak/inertness)
→ fix in place, do not escalate.

## 1. WHY THIS, WHY NOW (one paragraph)
Six folds went inert because they trained on LIKES-ONLY reconstruction: the strong consumption channel
free-rides, so graded value/confidence get no gradient and die (value-zeroing → ΔNDCG 0.000). The frozen-RecVAE
valence probe confirmed the DECODER *can* express dislike (regions move ± symmetrically) — so a target z exists;
what was missing is a TRAINED, non-circular producer of it. The fix has three parts, all absent before:
ORDINAL loss (dislike enters the loss → R2 by construction), CHANNEL-DROPOUT (mask consumption so value must
carry irreducible signal → kills free-riding), CONFIDENCE→PRECISION (graded knowledge sets per-answer posterior
precision → R3, the novelty gap). Decoder frozen → R1 guaranteed, no re-certification.

## 2. ARCHITECTURE
- DECODER: **frozen** RecVAE-d512 (`.cache/instrument2/ml25m_recvae_d512_best.pt`), full-profile 0.5541 = tie
  EASE. Never updated in Rung I. This is where the collaborative geometry lives.
- ENCODER: amortized set-encoder over answer tokens → posterior **q(z | answers) = N(μ, diag(σ²))**.
  - Base = PROVEN i25/v3 **Deep-Sets SUM-pool residual** (NOT softmax-mean, F3), with a **per-token encoder φ
    applied BEFORE the SUM**:
    `μ = native_z + ρ_μ([ SUM_t φ(token_t), native_z, log1p(#distinct SURVIVING tokens) ])`.
  - **φ = the value–entity BINDING fix (pass-2 fix, the single most dangerous risk pass-1 missed):**
    `φ(token) = FiLM(member-bag emb ; value, knowledge, fidelity)` — value & knowledge emit a per-token affine
    (γ,β) that MULTIPLICATIVELY modulates the entity embedding, so polarity BINDS to its entity ("hated Horror"
    ≠ "loved Horror"). Raw concat + SUM would make them commute ⇒ inert. A DISLIKE value flips the sign of that
    entity's contribution — **this is ALSO how a member-bag CONCEPT token carries a SIGN through ρ** (dislike a
    concept). Knowledge/confidence enters φ as a **PRECISION WEIGHT** scaling the token's contribution to the
    pool (know_well > rough > no_clue) ⇒ it moves μ (recommendations), not just σ — else under deterministic-μ
    eval the "precision" claim is unfalsifiable (pass-2 fix). τ_level is a learned per-level scalar carried
    OUTSIDE the RecVAE input L2-norm (applied to φ-output, not to the bag before enc_items) so F11 can't erase it.
    **FiLM is γ-ONLY, γ = SIGNED×POSITIVE factorization (pass-3+4 fix — PIN THIS):** the additive β is ENTITY-
    INDEPENDENT, so Σ_t β survives the SUM as an unbound value-mass pathway that re-creates the commuting failure
    FiLM exists to kill — **β is PERMANENTLY OFF** (not "zero-init + decay"; IG2's specificity leg catches any
    resurrection). `γ = (SIGNED per-dim value/valence factor: unconstrained linear head, identity-init 1) × (τ =
    POSITIVE knowledge-precision scalar: softplus, init 1)`. **γ MUST be signed** — a positivity map (exp/softplus)
    lets dislike only ATTENUATE toward zero ("ignore this entity"), NOT invert below neutral, and R2/G-value-
    NONINERT die the old way. Signed γ does NOT reintroduce commuting: the sign multiplies the entity embedding
    PER TOKEN before the SUM, so (value,entity) stay bound; only the removed additive β commutes. **τ strictly
    positive** so knowledge level can never flip polarity (precision ⊥ sign). **IDENTITY-ANCHOR:** γ(meh)≈1, so
    "value-zeroing to meh" leaves the entity at full weight = value-REMOVAL, not distance-from-arbitrary (consistent
    with §6). Report γ per (value,knowledge) level each epoch — **watch γ(hated) SIGN, not just magnitude.**
  - **native_z = RecVAE.enc_items(revealed LIKED items)** — the item anchor, built from the frozen decoder's
    OWN item embeddings ⇒ the encoder INHERITS collaborative structure (does not re-estimate it). ITEM-HOLE
    (R6): consumed-but-not-liked items do NOT enter native_z; every item ALSO emits implicit+explicit tokens.
    Multi-hot arm (feed consumed items too) = pre-registered ABLATION, adopted only if it passes G-clean+G-GoT.
    **native_z PARTICIPATES IN CHANNEL-DROPOUT (pass-1 fix):** when the consumption channel is masked (train)
    or in the consumption-absent eval regime, native_z falls back to the zero-init empty anchor (=prior), so
    the like/dislike SIGN cannot free-ride through the anchor — value/concept tokens MUST carry it. This closes
    the #1 residual risk (anchor free-ride = failure #7 by the original mechanism).
  - σ head: `log σ = ρ_σ([...])` — the encoder emits a POSTERIOR (R5 adaptivity precondition). Confidence
    contributes here (below). **The reconstruction loss scores from REPARAMETERIZED samples z~q(z|answers)**
    (pass-1 fix) so σ gets a LIKELIHOOD gradient — NOT μ-only, which would leave σ shaped by KL alone and
    DECORATIVE. σ must SHRINK with #answers (G-posterior-sharpens) AND be CALIBRATED to actual per-user z/NDCG
    error (G-posterior-CALIBRATED) — a deterministic count-shrink carrying no information FAILS.
    **EVAL SCORING IS DETERMINISTIC μ (pass-2 fix):** reparam sampling is TRAIN-only; sampling at eval would add
    MC variance to every gate CI. σ is never scored into recommendations at eval — it is exercised ONLY by
    G-posterior-CALIBRATED / G-posterior-sharpens.
  - TOKENS (v3 two-channel, leak-free): IMPLICIT (channel, member-bag emb, knowledge **{no_clue/rough/know_well}**
    — an ANSWERED "no clue" is an explicit token, DISTINCT from ABSENCE = never-asked = no token, so G-know-
    graded's absent<no_clue<rough<know_well legs are all testable, pass-1 fix); EXPLICIT (channel, member-bag
    emb, value {4-level}, fidelity). Concepts/entities/attributes = member-bag
    embeddings **through ρ** (z-space, R11); member weights **genome×popularity ONLY** (R5). NO surprise-from-
    profile (F4).
  - ρ last layer **zero-init** → empty pool → μ=native_z=prior, σ=prior-σ ⇒ exact intercept (R4).
- DEDUP (R8/F13): SUM over the DISTINCT token set (dedup by (channel,entity,kind)); collision → highest-
  fidelity (tie → highest turn-index); implicit collision → higher knowledge level. Order-invariant (R9).

## 3. LOSS (the anti-inertness core — the whole point)
`L = L_ordinal  +  λ_c · L_consume  +  β · KL(q(z|answers) ‖ prior)`  (L_ordinal/L_consume scored on z~q samples)
- **β SCHEDULE (pass-1 fix R5):** β annealed 0→β_max over a warmup + free-bits floor, likelihood-scaled. An
  unannealed KL pulls q toward the prior at ALL lengths ⇒ saturation pressure that fights G-monotone-INCREASE.
- **L_ordinal (R2/R7):** over a STRATIFIED held-out rated sample spanning all star bands INCLUDING hated,
  margin `score(i) − score(j) ≥ m·(star_i − star_j)`. Dislike enters the loss ⇒ down-ranking is trained, not
  bolted on. Held targets DISJOINT from interview inputs (leak-free). Fixes F10 (likes-only binarizes).
  **STRATIFICATION FALLBACK (pass-2 fix, HARD RULE #1):** use whatever bands a user HAS — a user with no hated
  ratings contributes their available bands; NEVER filter/drop users to force a full band set (that would be a
  silent data reduction). The margin is defined pairwise over available bands.
- **CHANNEL-DROPOUT (NEW — the mechanism all six folds lacked):** each training step, with prob p_drop mask an
  ENTIRE channel (item/consumption channel most often; also value channel, concept channel) so the surviving
  channels MUST reconstruct the held ranking. Forces graded value/confidence/concepts to carry irreducible
  signal instead of free-riding on consumption. **Consumption dropout ALSO drops native_z (pass-1 fix)** — else
  the sign free-rides through the anchor. p_drop **swept {0.3,0.5,0.7}; SELECTION RULE (pass-2 fix): max val
  NDCG SUBJECT TO value-zeroing Δ ≥ MDE** (constrained, not lexicographic — forbids the p_drop→0 re-inertify).
- **CHANNEL PARTITION (pass-2 fix — pin it or GoT/free-ride breaks):** three independently-droppable channels —
  (A) consumption anchor native_z; (B) IMPLICIT knowledge tokens; (C) EXPLICIT value tokens. Dropout masks A/B/C
  independently. **G-GoT is tested via B, in a regime where A is dropped but B present ⇒ GoT survives** (does not
  die with consumption). The know_well↔liked free-ride (knowledge predicting value without the value field) is
  broken by (i) TRAINING including know-well-but-hated cases (GoT decorrelates knowledge from value) and (ii)
  G-value-NONINERT requiring C to add NDCG BEYOND A+B. **log1p(#distinct) counts only SURVIVING (non-dropped)
  tokens** (pass-2 fix) so the magnitude pathway matches the actual pooled evidence.
- **CONFIDENCE→PRECISION (R3, novelty):** {no_clue<rough<know_well} maps to per-answer observation precision in
  the posterior update — know_well tightens σ (sharper pull), no_clue barely moves it. Learned scalar per level
  (β/level telemetry printed each epoch), NOT a bag weight (F11: L2-norm erases bag weights).
- λ_c, m, β, p_drop, confidence-precision scalars: coarse val sweep, best-on-val kept, NO silent drift.

## 4. EXPERIMENT DESIGN
### 4.1 Splits — incl. the transductive-RS question you raised
- **RecVAE scorer** was trained on the FULL rating matrix (transductive; F16). The DECODER is FROZEN, so it adds
  no NEW leak during encoder training — BUT "full-profile native ≈ 0.554/0.487" is an OPTIMISTIC ceiling because
  the decoder has seen these items. We keep it as the reference anchor and FLAG it as transductive; the honest
  external-validity claim rides on the held-out-USER encoder eval below, not on native.
- **Encoder split (held-out USERS):** population 20k train / 2k val / 2k test — test users NEVER seen in encoder
  training, so we measure the answer→z MAP on unseen users. 173 LLM-judged / 300 study users QUARANTINED (real-
  LLM eval only; never train/val/test the encoder on them — CLAUDE.md).
- **Within-user leak-free (R5/G-no-profile-leak):** interview INPUT = the "revealed" set only; L_ordinal TARGETS
  = a disjoint stratified held-out set. Belief must be byte-invariant to scrambling UNREVEALED items. Members
  never in the score-exclusion mask; weights genome×pop only.

### 4.2 Rich-data emulation (no new LLM $)
- Value {hated/meh/liked/loved} = the STAR RATING directly (we have all 25M). Confidence + transcript + refusals
  = the **distilled answerer v2.1** (no LLM calls) over rating-derived interviews. The 173/300 REAL-LLM set is
  the transfer check that the distilled confidence channel matches reality. **TRANSFER PASS CRITERION (pass-1
  fix):** on the real-LLM set, (i) all directional gates (G-value-NONINERT sign, IG2/IG4/IG5 monotonicity) hold,
  and (ii) real-LLM NDCG ≥ population-test NDCG − δ (δ pinned pre-run). A directional flip or NDCG collapse on
  real answers = FAIL (distilled confidence didn't transfer), independent of population-test success.

### 4.3 Length curriculum — 1 → full profile (your "datapoints = short interview" point)
- **log-uniform interview lengths 1 → full profile.** Short end = the elicitation regime (8–10, sparse, with
  refusals/no_clue); dense end = the full profile.
- **FULL-PROFILE rich emulation + the F5 trap:** a full profile = distilled answerer over ALL rated items →
  long rich interview. NUANCE (F5): a full profile has NO refusals and few no_clue (they rated everything) — so
  training on full profiles alone is OOD to blind short interviews. MITIGATION: the curriculum is length-mixed
  (log-uniform), refusals injected at short lengths from the natural answerable-rate; NO on_profile=0.7
  privileged curriculum (F5). We report the k-curve so any full-profile overfit shows as a sag at short k.
- **Strategy mix (R2, pass-1 fix — full FOLD_MASTER R2 list):** random / entropy / EIG / **on-profile / off-
  profile / adversarial** / mixed selection over the asked set; the encoder must be strategy-agnostic.
  Adversarial asking (deliberately low-value/off-taste questions) is exactly where a dropout-trained encoder
  can crack, so it is IN the curriculum, not just the eval. Accumulate across every length.

### 4.4 Volumes
- **PILOT (gate before full run):** 2k train users, short curriculum, 1–2 epochs. GO criteria on pilot-val:
  intercept dev 0; no NaN (data-hygiene screen for q718-class non-finite emb, F6); **value-zeroing DROPS val
  NDCG by ≥ MDE** (channel-dropout biting); σ shrinks with length. **PILOT MDE (pass-1 fix):** the value-zeroing
  Δ must exceed a pre-pinned effect size (target ΔNDCG@10 ≥ 0.005, bootstrap CI lower bound > 0 over the 2k val
  users) — "moves" is not enough; noise moves it. If value Δ < MDE on the pilot → STOP, the mechanism failed,
  do not spend the full run.
- **FULL:** training-population size is a **SWEPT knob** (distilled answerer is free — NOT capped at 20k, R10);
  start 20k, scale to 50k/100k/full population if the encoder underfits the map. Many interview samples per
  user (length curriculum). NO token/data caps anywhere (R10/F13); length-bucketed micro-batching for memory.

## 5. FAILURE MODES GUARDED (F1–F17 — the "lot of failure points")
F1 concepts z-space through ρ, never RecVAE input · F3 SUM-pool not softmax-mean · F4 no surprise-from-profile ·
F5 length-mixed, no privileged on_profile curriculum · F6 non-finite-emb screen + assert finite loss · F7 no
Set-Transformer (Deep-Sets) · F8 no z-score on residual · F9 residual not plain-concat · F10 ordinal + dropout
not likes-only · F11 confidence as precision scalar not L2-erased bag weight · F12 distinct-token dedup ·
F13 no caps · F14 gates REQUIRE increase (below) · F17 no two-head collapse (single posterior head).

## 6. GATES (all on held-out TEST unless noted; HARD = stops/blocks the claim)
GATE HIERARCHY (author steer): the PRIMARY verdict is BEHAVIORAL — IG2 polarity-flip+specificity, G-monotone-
INCREASE, G-clean, IG4/IG5 graded sweeps, IG1/IG3 purity/coherence, G-GoT ("look inside, watch it behave").
Aggregate-NDCG numbers (value-NONINERT MDE, regime-(ii)) are SECONDARY quantifications, not the deciders. A few
bars are RESEARCH QUESTIONS (reported, interpreted), not pass/fail — flagged inline.
- **G-clean [HARD]:** full-profile fold ≥ native − 0.03.
- **G-value-NONINERT [HARD, centerpiece — PROTOCOL PINNED, pass-1 fix]:** neutralize the VALUE FIELDS on
  explicit tokens **with native_z membership FROZEN** (the liked-item anchor set is held fixed, so the drop is
  the trained value pathway, NOT anchor-membership churn — an unfrozen membership would let the gate pass
  spuriously through the untrained anchor). Report the Δ in BOTH eval regimes: **(i) consumption-present** and
  **(ii) consumption-absent** (native_z dropped) — value is trained only in dropped regimes, so its contribution
  must show in both. **"value-zeroing" = REPLACE the value field with the in-distribution 'meh'/neutral level
  (pass-2 fix), NOT a zero-vector** — a never-trained zero input is OOD and inflates Δ spuriously. Run ×{short,
  long, full} contexts (pass-2 fix, restores FOLD_MASTER G-value-monotone's three legs). PASS = value-neutralize
  DROPS test NDCG by ≥ MDE (CI>0) in both regimes AND all three contexts; + binarize-value ablation LOSES.
  Regime-(ii) ABSOLUTE NDCG is REPORTED as a research question (author steer — not a hard floor), expected well
  above intercept. (The Δ gate the last 6 folds failed at ΔNDCG 0.000.)
- **G-posterior-CALIBRATED [HARD for the R5 adaptivity claim, pass-1+3 fix]:** per-user σ must correlate with
  actual z-error / NDCG-error **WITHIN LENGTH STRATA** — partial rank-corr controlling for log1p(#answers),
  pooled CI>0. (Pass-3 fix: a plain σ↔error corr is LENGTH-CONFOUNDED — a deterministic count-shrink σ passes it
  because error also shrinks with count, fake-licensing R5.) A σ that only shrinks with count carries no
  information and cannot support info-gain selection — this gate, not G-posterior-sharpens alone, licenses
  "delivers adaptivity signal."
- **G-monotone-INCREASE [HARD]:** (turn8 − turn1) ≥ +0.02 CI>0; (clean − turn8) ≥ +0.05; ALSO on a concept-only
  curve. Catches saturation (F14).
- **G-know-graded:** **rough<know_well** adjacent CI>0 ×{short,embedded-in-20,full} [HARD]; + collapse-to-binary
  ablation loses. The **absent<no_clue** leg is REPORTED, not HARD (pass-2 fix — no_clue ≈ zero info, so
  requiring CI>0 there is a gate built to fail).
- **IG2 POLARITY FLIP (illustrative + CI):** flip like→dislike on a genre → its items move toward the opposite
  end (target 0.90→0.10; raw geometry probe was 0.69→0.31 so the fold must ADD separation). Eyeball both lists.
  **+ SPECIFICITY LEG (pass-3 fix):** UNTOUCHED genres' items must stay put (displacement < ε) — catches the
  entity-independent value-mass leak that a flipped-genre-only check misses.
- **IG4 GRADED VALUE (illustrative + CI):** sweep one item hated→meh→liked→loved → recs shift monotonically.
- **IG5 GRADED CONFIDENCE (illustrative + CI):** rough vs know_well on the same answer → pull scales with
  confidence.
- **IG1 genre purity / IG3 franchise coherence:** top-10 under "like X" are X; one liked film → coherent nbrs.
- **G-GoT (R6):** know-well-but-hated still pulls toward region (consumption dominates), entities AND items.
- **G-posterior-sharpens (R5):** σ shrinks monotonically with #answers (so info-gain selection has signal).
- **G-caplength (R3):** elicitation not sacrificed for full-profile (short-k NDCG not traded away for dense end).
- **G-prolific (R6):** selective > diluted (a focused know-well beats a diffuse many-token bag).
- **G-k2-graded (R7):** reproduce i25 k=2 beat-native (+~0.012); it must VANISH under value-zeroing (proves it
  rides the explicit channel, not consumption).
- **G-intercept / G-no-profile-leak (byte-inv) / G-order / G-falsify-count / G-canaries / G-token-count-audit /
  G-implicit-ablation.**

## 7. ROLLBACK / ABLATION LADDER
Each mechanism is an ablation gated on val, adopted only if it earns NDCG or passes a required gate:
(base i25 residual) → +ordinal loss → +channel-dropout → +confidence-precision → +posterior σ head → [multi-hot
native_z arm]. Degrades to i25 (clean 0.4647) if the extras add nothing — no worse than the proven fold.

## 9. SIGN-OFF THRESHOLDS (author confirms these NUMBERS before the pilot — proposed defaults)
All accept/reject bars in one place (pass-3 requires the two starred ones be valued at sign-off):
- G-clean tolerance: full-profile fold ≥ native − **0.03**.
- G-monotone: (turn8−turn1) ≥ **+0.02** CI>0; (clean−turn8) ≥ **+0.05**.
- G-value-NONINERT MDE [HARD]: value-neutralize ΔNDCG@10 bootstrap CI-lower > 0 AND ≥ **0.01** (raised from
  0.005 — a real channel should move more than a rounding effect). Hard bar = "provably not inert"; the MAGNITUDE
  beyond 0.01 is a reported research finding. Both regimes ×3 contexts. PRIMARY evidence is behavioral (IG2/IG4),
  not this aggregate.
- Regime-(ii) consumption-absent NDCG [RESEARCH QUESTION, NOT a hard gate — author steer]: REPORTED, must be
  well above intercept (0.150), ideally ≥ static (0.226). A low value is a FINDING about how far value+concept
  stand alone without the item anchor — informative, not a STOP.
- ★ Transfer δ (real-LLM 173/300 vs population-test): NDCG@10 ≥ population − **0.03**, directional gates hold.
- IG2 specificity ε: untouched-genre mean displacement < **0.03** percentile-frac.
- Pilot GO: intercept dev 0; no NaN; value Δ ≥ MDE on pilot-val; σ shrinks with length AND within-length
  σ↔error partial-corr directionally > 0; γ(meh)≈identity in telemetry.

## 8. WHAT WE ARE NOT DOING (scope guards)
No decoder retrain (that's Rung II). No new LLM labels. No data caps. No surprise-from-profile. No privileged
curriculum. No claim of external validity from the transductive native ceiling. No policy/adaptivity training
yet — Rung I only delivers a FIT instrument; the adaptivity arena (E0-on-fit-instrument) is a SEPARATE sheet.
