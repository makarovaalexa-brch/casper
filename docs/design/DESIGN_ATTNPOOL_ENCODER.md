# DESIGN — Attention-Pool Belief Encoder on a Frozen RecVAE Decoder (2026-07-11)
For Fable adversarial review, then author sign-off + fast pilot. Supersedes the rung1 FiLM fold (inert).
Context: DESIGN_SHEET_RUNG1_ENCODER.md (what went inert), RUNG1_DIAGNOSIS.md (7-config battery: learned-sign
γ can't move), RUNG1_CHECKS.md (learned residual real at cold-start; frozen-decoder like/dislike selectivity
only +0.05 via a CRUDE hand-built z — a LEARNED pool may exploit more), EMBEDDING_FOLDING_INDEX.md (June
attention-pool encoder = VERIFIED GOOD, accumulated, beat ridge).

## 0. THE IDEA (author's, in one line)
KEEP the frozen RecVAE decoder (the strong scorer). Replace the messy fold (native_z + FiLM value-gate +
member-bags + ρ) with ONE clean mechanism: an ATTENTION POOL over the revealed answers — each a LEARNABLE
item embedding (init from RecVAE) carrying its RATING and CONFIDENCE — pooled into the belief z the frozen
decoder ranks. The learnable part is the ENCODER-side item embeddings + attention; the scorer is untouched.

## 1. WHY THIS, AND WHY IT CAN HEAR DISLIKE WHERE THE FOLD COULDN'T
- The rung1 fold went inert because it asked a FiLM head to LEARN the value sign from a tiny gradient starting
  at identity — unreachable (diagnosis: 7 configs, γ never crossed zero). This design has NO sign to learn if we
  hard-wire it (V-base), or gives it a clean high-capacity path if we learn it (V-feat) — either way, no FiLM.
- The frozen decoder conflates like/dislike NEIGHBORHOODS (checks: selectivity +0.05). BUT that was measured with
  a CRUDE z = native − Σ(disliked directions). A LEARNED attention pool has strictly more freedom: it can use a
  dislike as TASTE-INFORMATION ("hates rom-com + likes Nolan → serious-drama region → rank liked films up"),
  which is NOT crude subtraction and was never tested. Frozen decoder is a headwind (real), not a proven wall.
- Strength is INHERITED by construction: encoder-side item embeddings E are init = RecVAE's own item factors W
  (the decoder is scores = z·Wᵀ+b). At init, pooling the revealed LIKED items' E ≈ enc_items ≈ native_z ≈ 0.52
  full-profile. So we START strong; learning E only has to ADD valence without breaking that.

## 2. ARCHITECTURE (base)
- DECODER: frozen RecVAE-d512, scores = z·Wᵀ + b. Never updated.
- ENCODER-SIDE ITEM EMBEDDINGS E[i] (d=512): **initialized from RecVAE's item factors W[i], made LEARNABLE.**
  This DECOUPLES the folding representation (E, learnable) from the scoring representation (W, frozen) — the key
  lever: E can learn a valence-aware way for an item to enter the belief while the strong scorer stays fixed.
  Concepts/attributes/entities: E_concept = member-bag mean of member items' E (genome×pop weights, leak-free).
- TOKENS: one per revealed answer = (entity id → E[entity], value {hated/meh/liked/loved}, confidence
  {no_clue/rough/know_well}, fidelity). NO surprise-from-profile (F4). Distinct-token dedup (F12).
- ATTENTION POOL (single learned query, June-proven; NOT set-transformer, F7): a learned query q attends over
  the tokens; attention logit per token = f(q, E[entity], value, confidence); softmax → weights a_t. 
- BELIEF: **z = z_prior + Σ_t a_t · g(value_t, conf_t) · E[entity_t]** where z_prior = decoder popularity prior
  (empty interview → z=z_prior EXACT, R4 intercept), and g injects rating+confidence (see variants for g).
- TRAIN END-TO-END on held-out ratings: ranking/ordinal loss over held items INCLUDING dislikes (rank held-liked
  above held-disliked). E and the attention are learned; W (decoder) frozen. Low LR + weight-decay-to-init on E
  so strength is anchored (don't drift E off the strong W init → the weak-June failure).

## 3. VARIANTS (the axes Fable should weigh; base = V-signed + single-query + residual)
**Axis A — how RATING enters g(value,·) (the load-bearing choice):**
- **A1 signed-scalar (BASE, safest):** g = s(value)·c(conf), s = KNOWN centered valence {hated −1, meh 0,
  liked +0.5, loved +1} (hard-wired sign — the diagnosis proved learned sign fails; the probe proved hard-wired
  sign is directionally clean at single-item level), c = learned positive confidence weight. E learnable adapts
  magnitude/direction. Dislike inverts by construction; attention decides which tokens matter.
- **A2 rating-as-feature (learned sign, higher capacity):** value+conf are FEATURES into the attention logit AND
  a learned per-token gain (no hard sign). Risk: could re-inert like FiLM — but it's a clean pool + end-to-end
  ranking, not an identity-init FiLM gate, so the gradient path is direct. Test head-to-head vs A1.
- **A3 dual-embedding (max freedom):** separate learnable E_pos[i], E_neg[i]; value SIGN selects which enters
  (magnitude by |value|). Lets the model learn a fully distinct "disliked-X enters the belief like THIS" — most
  flexible way to exploit the frozen decoder for dislike; 2× item params (init both from W).
**Axis B — attention:** B1 single learned query (BASE, June-proven, robust on short seqs). B2 add 1 self-attn
  block before pooling (richer answer-interactions, e.g. joint "likes Nolan ∧ hates rom-com"; OVERFIT RISK F7 —
  only if B1 underfits).
**Axis C — item-emb init:** C1 RecVAE-init learnable (BASE). C2 frozen-E control (≈ what failed — for ablation).
  C3 from-scratch control (≈ weak June — proves the RecVAE init earns its keep).
**Axis D — output:** D1 residual on prior (BASE, clean intercept). D2 pool-is-z (no prior; intercept via a
  learned empty-token).

## 4. GATES (behavioral = PRIMARY/HARD, per author's standing ruling; aggregates SECONDARY)
MAKE-OR-BREAK (the two the pilot exists to answer):
- **G-strength [HARD]:** full-profile fold ≥ native − 0.03 (~0.52). E-drift must not break the inherited strength.
- **G-dislike [HARD, the whole point]:** IG2 polarity-flip — flip like→dislike on a genre → its items move toward
  the opposite end (target 0.90→0.10; anything clearly >the +0.05 crude-probe selectivity is progress) AND
  untouched genres stay put (specificity leg). PLUS value-neutralize ΔNDCG ≥ MDE on dislike-informative contexts.
BEHAVIORAL (HARD): G-monotone-INCREASE (accumulates); IG1 genre purity; IG4 graded-value sweep; IG5 graded-
  confidence (know_well pulls harder than rough); G-GoT (know-well-but-hated still pulls toward region).
HYGIENE: G-intercept (empty→prior exact — residual-on-prior gives it BY CONSTRUCTION this time); G-no-profile-
  leak (byte-inv to unrevealed); G-order; G-falsify-count; no data caps (R10).
RESEARCH-QUESTIONS (report, not pass/fail): regime-(ii) consumption-absent NDCG; magnitude of the value effect;
  A1-vs-A2-vs-A3 head-to-head; how much E drifts from W.

## 5. EXPERIMENT / PILOT
- Splits: 20k train / 2k val / 2k test held-out USERS; 173/300 quarantined (real-LLM transfer eval only).
- Rich channels: value = star rating; confidence + transcript = distilled answerer v2.1 (no LLM $). No caps.
- Curriculum: log-uniform lengths 1→full, refusals at short lengths, strategy mix incl. adversarial. Signed
  ranking loss (held-liked over held-disliked) — dislikes IN the objective so E is rewarded for valence.
- PILOT (fast kill-switch, 2k users): report G-strength + G-dislike (IG2 flip) + G-monotone + γ/E telemetry.
  GO only if full-profile strength holds AND the dislike flip clearly beats the +0.05 crude-probe selectivity.
  If dislike still won't move with a LEARNED pool + hard-wired sign + learnable E → that is the honest signal the
  FROZEN DECODER is the wall (escalate to decoder retrain), because we've now removed every encoder-side excuse.
- Rollback: at init (E=W, empty pool) the model = native_z fold-in ≈ 0.52 full — degrades gracefully to the
  known-good baseline if learning adds nothing.

## 6. OPEN QUESTIONS FOR FABLE
1. Can a frozen decoder + learned encoder plausibly beat the +0.05 selectivity, or does the checks' result doom
   it regardless of encoder cleverness? (i.e., is the pilot's G-dislike reachable, or is this a dressed-up retry
   of a proven wall?) The single most important question — attack it hard.
2. A1 (hard-wired sign) vs A2 (learned) vs A3 (dual-emb): which best exploits a frozen decoder for dislike, and
   does A3's extra freedom actually help or just overfit?
3. Does weight-decay-to-W-init genuinely protect strength, or will the ranking loss drift E enough to break it?
4. Is "dislike as taste-information" (region-narrowing) real headroom the crude probe missed, or wishful?
5. The single most dangerous risk that makes this the 8th failed fold.
