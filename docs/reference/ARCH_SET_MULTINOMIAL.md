# ARCHITECTURE (detailed) — Set-input encoder + Multinomial decoder (2026-07-13)
The unifying recommender: ARBITRARY-SET input (flexible / continuous / open questions, refusals) + MULTINOMIAL
output (strong, scales, ~0.5 full-profile). Author-approved direction. This is the definitive spec for the
Fable pass. Notation: d=512 latent; ni=18,430 items; C=1,628 concepts (1,128 tags + 500 entities).

## 1. WHAT PROBLEM THE SHAPE SOLVES
- Dense (fixed input dim per concept) CANNOT represent a continuous query, an open-vocab concept, or an
  arbitrary interview without folding the whole catalog. Rejected.
- Pre-VAE set-encoder was flexible but its LISTWISE-over-tokens objective collapsed at 18k catalog.
- Fix: INPUT and OUTPUT are separable. Keep the set-encoder INPUT; make the OUTPUT a multinomial decoder over
  ITEMS ONLY (Mult-VAE/RecVAE objective — dense per-item gradient, scales to 41k). Decoder never sees concept
  tokens in its softmax -> no collapse.

## 2. ENTITY EMBEDDING SPACE (one space for everything)
- `E_item ∈ R^{ni×d}` item embeddings, **init from a0c decoder factors** `W` (so items live in the scoring
  geometry from step 0). Learnable (low-LR).
- `E_concept ∈ R^{C×d}` concept embeddings, **init = DE-MEANED pop-weighted member-bag** of `E_item` (Fable
  change #7: subtract the global item-embedding mean, so a concept starts as a RESIDUAL off the popularity axis,
  not loaded on it — removes the baked-in collinearity that sank the additive operator). Learnable.
- A **continuous / open query** (Paper C) is just an arbitrary `q ∈ R^d` produced by the policy/OMP-decode — a
  token in the SAME space; NO new machinery. (Not used in the base recommender; the architecture must not
  preclude it.)
- Rationale: items, concepts, and queries are all "regions in item-factor space"; sharing the space is what
  makes a concept answer act on items and makes continuous elicitation possible later.

## 3. THE TOKEN — one per ANSWERED entity (arbitrary set; only-asked; NO zero-folding)
Each revealed answer a = (entity e, value level v, confidence level k, refused?) becomes ONE token:
```
token(a) = TokMLP( concat[ E_entity[e] (d),  value_emb[v] (d_v),  conf_emb[k] (d_c),  refused_flag (1) ] )
```
- `value_emb = Embedding(5, d_v)` over {none, hated, meh, liked, loved}; d_v=64. LEARNED per level
  (one-hot-into-linear = a learned embedding; no hand-set scale, sign/ordering learned).
- `conf_emb  = Embedding(3, d_c)` over {refused, rough_idea, know_well}; d_c=32. LEARNED, SEPARATE from value.
- `refused_flag ∈ {0,1}` explicit (asked-but-no-value). A refusal token: value_emb[none], refused_flag=1.
- `TokMLP`: 2-layer MLP (concat_dim -> 2d -> d, swish). This is where value×confidence×entity interact
  NONLINEARLY — the author's rule: value and confidence NEVER multiplied; the MLP learns any combination
  (amplify obscure-confident, discount popular-confident, trust know_well, ...).
- **Items** at warm-start carry value only (real ratings, all implicitly know_well); they gain conf+refused in
  the INTERVIEW regime. **Concepts** carry all three from the start (answerer's know/val).

## 4. SET ENCODER — SELF-ATTENTION over tokens (REVISED: interaction, to match a0c)
DECISION (author, 2026-07-13): a0c's belief is a DEEP JOINTLY-INTERACTING function of the whole item set (fc1
mixes every item into every hidden unit + 5 residual blocks). A single-query attention pool is a WEIGHTED MEAN
(additive) and provably cannot reproduce that interaction -> would cap Phase A below 0.486. So the encoder is
SELF-ATTENTION (interaction), matched to a0c's expressiveness. The "self-attention hurts elicitation" evidence
(V2-ST) was an OVERFIT/generalization gap (rich-profile-trained, sparse-interview-tested), NOT a law -> managed
by regularization + a short-interview-heavy curriculum, not by crippling the model to a mean. THE PROBE IS
RETIRED (it only tested whether the weak additive pool could reach 0.5; moot for an interacting encoder).
```
Given token set {t_1..t_m} (m arbitrary, incl. m=0), prepend a learned CLS token:
H^0 = [CLS ; t_1 ; .. ; t_m]
H^{l+1} = TransformerBlock_l(H^l)      # L=2-3 multi-head self-attention blocks (heads=4, d=512), pre-LN,
                                       #   dropout, weight decay -- REGULARIZED for short-set generalization
z = z0 + H^L[CLS]                      # CLS pools the set; z0 = learned empty-set belief (popularity prior)
```
- **Interaction** = the cross-token attention (learns "loves Nolan ∧ hates rom-com -> serious drama"), the thing
  a mean cannot do; this is what lets Phase A actually reach a0c.
- **Set-size / richness** is captured natively (attention over m tokens; CLS sees all) -- Fable change #2
  (magnitude channel) is subsumed, but keep a scalar log(1+m) feature into the CLS readout as a cheap hedge.
- **Empty set (m=0)** -> H^L[CLS] over CLS-only -> z = z0 -> intercept EXACTLY (R4 intercept). Explicit m=0
  branch (Fable change #8).
- Order-invariant + duplicate-safe (no positional encodings; set attention).
- **Elicitation-generalization plan** (the V2-ST fix): curriculum HEAVY on short interviews (log-uniform lengths
  weighted toward 1..8), dropout in the blocks, weight decay; gate on cold-start early-AUC to catch overfit.

## 5. DECODER + OBJECTIVE — the strength/scale source (unchanged from a0c)
```
scores = z @ W^T + b            # W ∈ R^{ni×d} = a0c decoder (warm-start), b = a0c bias
loss   = multinomial log-softmax NLL of held-liked over ALL ni items   # dense per-item gradient
```
- Decoder warm-started from a0c; the multinomial over items ONLY (never concept tokens) — the anti-collapse.

## 5.0 TOKEN FUSION — DECIDED (2026-07-13, Fable-adjudicated, author-approved)

**SPEC DEVIATION (surfaced, not silent):** §4 says CLS + FULL self-attention over tokens. The running Phase-A
student instead uses INDUCING POINTS (attend into m=32 learned points -> self-attn among them -> pool), because
full self-attention is O(L^2) and profiles reach 14,040 items (it killed the machine). Fable: expressivity-
equivalent for this teacher. Recorded rather than left to ride.

**THE QUESTION:** can ATTENTION ALONE learn the (entity, value) fusion — i.e. can we drop the per-token MLP and
use a plain additive token (item_emb + value_emb), letting attention recombine? (Cheaper: the per-token MLP is
the dominant cost, ~21M tokens/epoch.)

**FABLE'S ADJUDICATION (corrects the overseer):**
- Attention CAN express per-item negation — the sign lives in a HEAD'S VALUE PROJECTION, not the attention
  weight: one head routes likes, a second head with a negated projection routes dislikes; the rating enters the
  attention LOGITS, and softmax gates on its sign. (The overseer's "linear projections => can't flip" was WRONG.)
- BUT the construction is FRAGILE: softmax weights sum to 1 (magnitudes dilute as dislikes accumulate; the
  dislike-head must attend somewhere even with zero dislikes); matching a CONTINUOUS rating needs a basis of
  heads at different temperatures; and SGD must DISCOVER an antisymmetric head-pair routing with no local
  shaping. Capacity yes, inductive bias bad.
- LOCATION MATTERS: the nonlinearity must act PER-TOKEN, BEFORE pooling. After the 32 inducing points, per-item
  sign cannot be unmixed. (Pre-VAE corroborates: additive token WORKED because it kept per-token FF layers —
  the nonlinearity was just downstream-of-token, upstream-of-pool. Its polarity was BINARY; our continuous
  signed rating is a HARDER fusion => argues for MORE per-token capacity, not less.)
- Therefore in the CURRENT model (per-token FFs removed for speed) the TokMLP is the ONLY per-token binding
  site — LOAD-BEARING, not decoration.

**DECIDED TOKEN FUSION (replaces the concat-MLP at the next restart):**
```
token_i = gamma[rating_i] (*) item_emb[i]  +  beta[rating_i]        # (*) = elementwise
gamma, beta = LEARNED lookup tables indexed by the DISCRETE rating (ML ratings are 10 half-star levels)
INIT: gamma = 1, beta = 0   ->  at init the token IS the pure ADDITIVE token (the author's null hypothesis)
```
Why this is the honest choice under the author's rule (NO imposed combination rules; value/confidence/refusal
stay SEPARATE; the model learns any interaction):
- It imposes NOTHING: additive is both the INIT and a valid fixed point. If negation helps, the model LEARNS
  gamma negative; if not it stays additive. We hand it a learned FAMILY, not a formula.
- It CAN express dislike-as-negation (the thing attention can only fake fragilely), and contains the teacher
  exactly (a0c's value*w_i is gamma(v)=v*1, beta=0).
- It is ~100x CHEAPER than the concat-MLP on the dominant cost (lookup + 2 elementwise ops vs a 0.5M-FLOP MLP
  per token) -> the speedup comes free.
- CONFIDENCE and REFUSAL keep their OWN separate slots/tables — NO cross-signal multiplication anywhere.
- Fallback if even a learned gamma is judged "imposing multiplication": bottlenecked concat-MLP (513->128->512),
  ~4x cheaper (the target fusion is essentially bilinear/low-rank).

**DIAGNOSTICS (pre-registered, run after Phase A):**
1. cos( TokMLP(e, +v), TokMLP(e, -v) ) across items. Strongly NEGATIVE => the model LEARNED negation =>
   empirical proof the per-token nonlinearity is load-bearing.
2. Ablation: gamma/beta-table token vs PURE-ADDITIVE token. PRE-REGISTERED PREDICTION: pure-additive degrades
   SPECIFICALLY on DISLIKE-HEAVY users -> slice the eval by dislike count so the MECHANISM is visible, not just
   an aggregate.

**DO NOT change the running Phase A** — the concat-TokMLP has more than adequate capacity; the deficiency is in
the SIMPLIFICATION, not in what is training. Decide on evidence.

## 5.1 VALUE ENCODING per entity type (Fable #4, verified against a0c code)
a0c input = L2-NORMALIZED signed CONTINUOUS value (scale_rating) over items. Therefore:
- **ITEM token value = CONTINUOUS** (real rating scalar -> learned projection), matching a0c so Phase-A
  distillation is not quantization-floored.
- **CONCEPT token value = DISCRETE** value_emb over answerer levels (learned per level).
- Set-size: a0c's L2-norm makes z size-dependent -> CLS attention + a log(1+m) feature carry it.

## 6. TRAINING — STAGED (author plan): Phase A prove-SOTA on FULL, then Phase B interviews with FREEZING
- **Phase A — DISTILLATION to inherit 0.4961 (the hard requirement).** On FULL PROFILES (rated item tokens
  only, **value_emb granularity MATCHED to a0c's actual rating levels** — Fable change #4, learned per level, no
  hand-set scale — so distillation isn't quantization-floored), train the SELF-ATTENTION encoder so
  `z_set(full item set) ≈ z_a0c(full profile)` (MSE on z; teacher = a0c encoder on the same items; leak-free).
  The interacting student CAN represent the interacting teacher (unlike a mean). GATE: full-profile NDCG@10 >=
  0.486 after A -- **test FIRST, before interview/confidence machinery.** If a 2-3 block transformer still can't
  reach it, the a0c belief is un-distillable into a set form and the shape is in question -> STOP and report.
- **Phase B — interview fine-tune, DECODER FROZEN (author plan).** Warm from A. **FREEZE the decoder** (holds the
  0.496 scoring geometry — the V2-ST both-axes-degradation guard; optionally also freeze lower encoder blocks).
  **SUBSET-DISTILLATION anchor (Fable #1, relocated to B): distill the encoder against a0c on random sub-profiles
  of varying length (incl. 1-8), teacher = a0c on the SAME subset** — teaches short-set beliefs from the teacher,
  not invented from the elicitation signal (the set-transformer overfit fix). **NO-JOINT-PEAK IS A REPORTABLE
  NULL (Fable #2): if elicitation-gain and full-profile >= 0.486 never co-occur during B, report it — do NOT
  optimize around it.** Then end-to-end multinomial on the curriculum:
  - FULL / dense profiles (strength retention, item value tokens),
  - INTERVIEWS: strategy-asked sets (random/pop/entropy/on/off-profile), **length curriculum HEAVY on short
    (weighted 1..8)** — the V2-ST elicitation-overfit fix — over items ∪ concepts. **HELD-LIKED ITEMS ARE
    EXCLUDED FROM THE ASKABLE SET (Fable change #3, author-approved)** — a held item asked would be mislabeled
    REFUSED against a LIKED target. Asked-rated-known-item -> value token; **asked-unrated(non-held)-item ->
    REFUSAL token** (informative: strategy asks popular/entropy items, not random obscure); concept -> answerer
    (know,val) incl. refusals. Bounded ask-budget per user (never the whole catalog).
  - Target = held-liked (leak-safe; known/held uid-seeded split matching the answerer tables).
- LR: A modest; B <= A; joint peak (elicitation early-AUC subject to full-profile >= 0.486); save all ckpts.

## 7. DATA (reuse — do NOT recompute)
- Answerer tables `.cache/rich_signal/mm_train_{know,val}.npy` (150k, 2428) + `mm_val_*` (3k) — realistic
  (know,val) for concepts[0:1128]+entities[1128:1628]+bank-items[1628:2428], aligned to E, leak-safe split.
  Refusals already present (know=0). Batched answerer `gen_user_fast` exists for regen.
- Item real ratings from meta (for full-profile value tokens + the rated/unrated split for interview refusals).
- **EASE RETRAIN (Fable change #6): `ease_v21` saw eval-held for non-study users -> retrain EASE excluding ALL
  eval held (batched `gen_user_fast`, cheap) BEFORE ANY GATE READOUT** — not post-hoc — else the leaky answerer
  fakes elicitation gains that a later re-check can't cleanly undo. 173/300 quarantined until the final transfer.

## 8. GATES (make-or-break, in order)
1. **G-distill / strength [HARD]**: full-profile NDCG@10 >= 0.486 after Phase A (the anchor; everything hangs
   on this — test FIRST, before building interview/confidence machinery).
2. **G-intercept**: empty set -> exact popularity prior.
3. **G-elicitation [the contribution]**: concept/mixed cold-start beats intercept on **TAIL** with the
   answerer's REALISTIC answers (not oracle), paired early-AUC; must beat a pop-within-answered-concepts
   control; reproduce in sign on the 173.
4. **G-confidence**: ablate confidence (drop conf_emb) vs keep — does learned confidence help held-liked?
5. **G-refusal**: ablate refusal tokens vs keep — is asked-unrated refusal net-useful?
6. Hygiene: order/duplicate invariance, monotone accumulation, leak asserts (known∩held=∅), no data caps.

## BATCHING (HARD RULE #1 — restated for every build agent): variable-length self-attention -> LENGTH-BUCKETED
micro-batching; NEVER a token cap, MAX_TOKENS, or profile truncation; full profiles always, full catalog, full
held sets. (Historical corruption: a token cap crowded out the item channel. Do not repeat.)

## 9. RISKS
- R1 [PRIMARY, revised]: mixed-regime (interview) fine-tuning DEGRADES full-profile strength -- the V2-ST
  precedent degraded BOTH axes (0.334/0.130 vs 0.361/0.150). Mitigation: FREEZE the decoder in Phase B + subset-
  distillation anchor + the hard >= 0.486 joint-peak constraint. If no joint peak exists, that is a REPORTABLE
  NULL, not a knob (Fable #2).
- R2: warm-start asymmetry — a0c's dense ENCODER can't be copied (different architecture); only decoder +
  item embeddings transfer; the set-encoder is trained/distilled from scratch. Distillation IS the bridge.
- R3: TokMLP is where value×conf interaction must be learned; if it underfits, confidence stays inert (caught
  by G-confidence).
- R4: EASE-in-answerer circularity + eval-held leak (Section 7) — could fake elicitation gains; guarded by
  EASE-retrain-excluding-held + pop-within-filter control + 173 transfer.

## 10. WHAT THIS UNIFIES
Cold-start interview (small set), full profile (all item tokens, distilled to 0.5), continuous/open queries
(a token in the same space), separate learnable confidence, refusal-as-signal, one entity space — all in ONE
model with a single multinomial objective.
**Continuous-query claim scoped (Fable change #8): architecture-COMPATIBLE but UNTESTED** — an arbitrary q∈R^d
is OOD for a TokMLP trained on table embeddings. To validate cheaply, Phase B includes a small fraction of
interpolated/perturbed entity-embedding tokens so the encoder sees off-table vectors; until measured, the paper
says "architecture supports it," not "it works."
