# DESIGN — Reconciled Recommender: full-profile strength + graded-concept elicitation (2026-07-12)
Author-decided sketch (3 key decisions signed). For one low-token Fable pass, then build. Supersedes
rich_signal (which dropped the profile + diluted concepts). Unifies the strong signed-latent instrument
(0.495, item-only) with the rich two-axis concept/knowledge signal — the actual model worth having.

## 0. GOAL
One recommender that is (a) STRONG on full profiles and (b) good COLD-START from a realistic interview,
and whose HEADLINE contribution is understanding **graded concept preference** ("love Sci-Fi / meh romance
/ hate slow films") — the signal pure consumption history cannot express. Cold-start k-curve is the primary
metric; full-profile strength is the secondary anchor.

## 1. AUTHOR DECISIONS (binding)
- **D1 Concepts = DEDICATED CHANNEL — but INITIALIZED FROM MEMBER-BAGS.** Concepts/attributes get their own
  learned embeddings + scoring path (NOT frozen member-bag dilution). CRUCIAL de-risk: each concept embedding
  e_c is *initialized* to its member-bag (mean of member items' embeddings) — the ONLY concept mechanism we've
  actually watched respond (genre sign-flip = −26pt in the item-only model). So the dedicated channel STARTS at
  the proven-responsive point and LEARNS to improve; it does not bet on training a concept embedding from scratch.
  (Note on prior evidence: the −26pt genre flip was a BEHAVIORAL response — the ranking moved — but it did NOT
  raise held-liked NDCG; sign is heard, sign was just not useful for the strength metric. The bet here is that a
  LEARNED graded-concept channel, trained end-to-end against held-liked, converts "heard" into "useful.")
- **D2 TWO-STAGE training.** Stage-1: pretrain on FULL PROFILES ONLY → learn item embeddings, incl. RARE
  items (which only appear in full profiles, not the interview universe). Stage-2: continue on a MIXTURE of
  full profiles + realistic interviews → adapt to cold-start without losing the item geometry.
- **D3 GRADED VALUE INCLUDED — concept-level graded value is the MAIN CONTRIBUTION.** Value {hated/meh/liked
  /loved} carried on BOTH items and concepts. (Item dislike measured inert earlier; concept graded value is
  the point — a loved/meh/hated CONCEPT moves its whole region.)

## 2. ARCHITECTURE
- **Item tower:** deep encoder over item-space channels → latent z; decoder scores ALL 18,430 items.
  - Item input channels: graded value (loved..hated, centered), knowledge (no_clue/rough/know_well), mask.
  - Item universe: full-profile regime = ALL the user's rated items (real values); interview regime =
    the widened ~9,352-item ask-channel (answerer fills via EASE-value + fame-know, NO new LLM).
- **Concept channel (D1, dedicated, member-bag-INITIALIZED):** each concept/attribute/entity has a LEARNED
  embedding e_c, INITIALIZED to its member-bag (mean of member items' embeddings) so it starts at the proven-
  responsive point and learns from there. An answer on concept c carries (graded value v_c, knowledge k_c). The
  concept contributes to the belief z via a NORMALIZED additive map with ONE SHARED W (A5):
  z += (1/√n_c) · Σ_c g(v_c, k_c) · W · e_c — first-class, learned jointly, NOT a FROZEN member-bag average;
  shared W (not per-concept W_c) keeps capacity sane for rare concepts; 1/√n_c so belief magnitude doesn't grow
  with #concept answers. Concepts influence the item ranking through the shared z → decoder.
- **Belief z** fuses item-tower + concept-channel contributions. Decoder ranks 18,430 items from z.
- **Loss:** multinomial log-likelihood of held-liked items (the strength objective; the concept/value
  signal is rewarded only insofar as it improves held-liked ranking — so no circularity credit).

## 3. THREE-PHASE CURRICULUM (D2, author-refined — each warm-starts the last)
- **Phase 1 — item pretrain (DONE = a03b_best.pt, 0.4961).** ACCURATELY (Fable pass-2 A4, verified in
  signed_latent.py): a03b = 40% dropped-profile + 60% STRATEGY-INTERVIEW curriculum (random/pop/entropy/on/off-
  profile/adversarial, real signed values, p_ans=0.85), multinomial-on-held-liked. NO refusal tokens, NO EASE
  (EASE only ever the eval ruler — verified). So Phase 2 imports interview-shape familiarity (helps its premise)
  and the EASE-clean-first ordering rests on VERIFIED provenance. Concept-naive. Phase 2 warm-starts it.
- **Phase 2 — warm-start concepts + CLEAN emulated interviews.** Load Phase-1 encoder+decoder; add concept
  channel (e_c = encoder-faithful init from member-bag; W_shared=I). Training examples = emulated interviews,
  lengths 1→20, built from the user's KNOWN answered items (real signed values) + concepts DERIVED FROM KNOWN
  MEMBERS (real centered-mean value, known-only — A4), at VARYING item:concept proportions. The concept-heavy /
  ITEM-MASKED end (concept revealed, its member items DROPPED from the item channel, held members in target) is
  REQUIRED — it forces concepts to be load-bearing (defeats the redundancy shortcut; at full item coverage
  concepts are correctly inert). Keep a slice of FULL PROFILES for strength. Encoder trains too (learns concepts
  are generalized evidence). NO refusals, NO EASE — genuine preference only; grounds e_c in real signal first.
- **Phase 3 — realistic answerer retrain.** Warm-start Phase 2, LR ≤ 0.1× Phase 2. Distilled answerer drives it:
  items/concepts CHOSEN BY STRATEGY over the WHOLE universe (random/pop/entropy/on/off-profile/adversarial/mixed,
  NOT just known-rated) → REFUSALS (know=0), mehs, EASE-imputed values all appear. Lengths 1→Tmax. ≥30%
  full-profile replay. Joint peak (cold-start early-AUC | full-profile NDCG ≥ 0.4961 − 0.010). EASE-imputation
  enters HERE → guarded by the Ce control + 173 transfer (A2). Phase-2-first ordering blunts "EASE folded twice."

## 4. GATES
- **G-strength** [full profile]: NDCG@10 near the signed-latent 0.495 (Stage-1 target; Stage-2 must not
  degrade it much — the two-stage design exists to protect this).
- **G-coldstart k-curve** [PRIMARY]: NDCG@10 by turn k∈{1,2,4,8,16,full} on realistic interviews; monotone,
  and rises FAST early.
- **G-concept-graded-value** [THE CONTRIBUTION — see A1, gate is USEFULNESS not behavior]: paired per-user
  early-AUC lift on held-liked ATTRIBUTABLE to concept value, via (a) eval KNOCKOUT (zero/scramble v_c) and
  (b) capacity-equalized TWIN trained without concept value; positive-half and negative-half reported
  separately; must BEAT the Ce EASE-direct control (A2) and reproduce in sign on the 173 (A2). The monotone
  sweep (region moves hated→loved) + specificity is a SUPPORTING behavioral check only (it already passed at
  −26pt and that wasn't useful).
- **G-rare-item** [D2 payoff]: rare items (low popularity) are rankable — pretrain gave them real embeddings
  (contrast vs an interview-only model that can't place them).
- Behavioral: IG1 genre purity, IG4 item graded-value sweep. Hygiene: intercept, no-profile-leak, order,
  duplicate-invariance, monotone accumulation. Full-catalog eval, CIs, no data caps.

## 5.1 BUILD SPECIFICS (pinned, single source of truth)
- **Warm-start** item tower (SignedEncoder) + decoder from `.cache/signed_latent/a03b_best.pt` (val_full
  0.4961, alpha=0, full-profile-pretrained = Stage-1-for-items ALREADY DONE). Keys: `encoder.*`, `decoder.*`.
- **Concept universe** = Universe.tagM (1,128 concepts) ∪ Universe.entM (500 entities) = 1,628 concept rows,
  each an e_c ∈ R^512. Item index space is IDENTICAL to signed_latent (both meta.npz keepI) — VERIFIED.
- **e_c init = ENCODER-FAITHFUL (PINNED; supersedes the earlier decoder-mean wording — Fable pass-2 A1).**
  e_c[c] = encoder(memberbag_valuevec_c) − encoder(0) = the belief DELTA the pop-weighted member-bag produces
  through the warm-started encoder. This is the ONLY init that reproduces the −26pt inference-injection probe
  (which fed members THROUGH the encoder); the decoder-row mean lives in a different space/scale and does NOT.
  At empty item input + one LOVED concept (g=+1, W=I): z = z0 + e_c = encoder(memberbag) EXACTLY = "consumed
  the bag" (the proven response, reproduced at t=0). HATED (g=−1) reflects away. Member-bag is pop-weighted
  (logcnt, head-damped) to cut broad-concept dilution. **e_c is LEARNABLE** — init only; training sharpens it.
  **W_shared** = Linear(512→512, bias=False), init = IDENTITY.
- **t=0 REPRODUCTION ASSERT (pre-launch blocker, A1):** before any Phase-2 gradient — (i) no-concept path is
  bit-identical to a03b (z=z_item → full-profile NDCG reproduces 0.4961); (ii) a LOVED concept raises its
  members / HATED lowers them (direction + approx magnitude of the −26pt probe), via the CONCEPT channel. If
  the pinned init fails (ii), the "starts at the proven point" premise is void → STOP and re-derive.
- **g(v_c,k_c)** = s(v)·c(k). Valence s: answerer val {0 hated,1 meh,2 liked,3 loved} → centered
  {−1,−1/3,+1/3,+1}; val=−1 (refusal, know=0) → g=0 (mask flags it). Confidence c: know {0,1,2} → {0,0.5,1.0}.
- **Forward**: z = z_item(value_vec,mask) + (1/√n_c)·W_shared(Σ_c g_c·e_c); scores = decoder(z). n_c = #concept
  answers this example (0 concepts → z = z_item exactly; empty interview → z = z_item(0) = prior, intercept OK).
- **A4 Stage-1 concept derivation** (KNOWN-only, assert): v_c = mean centered rating (star − user-known-mean)
  over the user's KNOWN member items of c → snap to nearest valence level; k_c from #known members
  (0→skip, small→rough(1), many→know_well(2)). Output (v,k) distribution must match the answerer's.
- **Loss**: multinomial-only (A7). **Stage-2 LR ≤ 0.1× Stage-1; ≥30% full-profile replay; joint peak**
  (cold-start early-AUC s.t. full-profile NDCG ≥ 0.4961 − 0.010).
- **SEQUENCING**: Phase-A = concept channel on existing answerer (concepts+entities+800 items) — validates the
  CONTRIBUTION. Phase-B = widen item ask-channel 800→9,352 (answerer EASE-value + fame-know for extra items).

## 5. DATA / SCALE
- Train users: full population (streaming tables — memory solved). 173/300 quarantined for real-LLM transfer.
- Ask-channel widened 800 → ~9,352 (EASE-covered) + concepts + entities. Answerer fills it (EASE+fame, no LLM).
- Scoring/targets: full 18,430 catalog, real held-liked ratings (leak-free known∩held=∅).

## 5.2 QUEUED ABLATIONS (author-approved; run AFTER Phase-2 knockout — do NOT disturb the running model)
Carry-over review of the PRE-VAE DualHeadSetEncoder (synthetic_sanity.py:84; ML-1M instrument, Hit@10
0.433→0.647). Kept 2 ideas, dropped 1:
- **ABL-1 Dedicated channel vs SINGLE channel.** Single = concept answer written into the ITEM value channel
  via its members (member-bag injection), TRAINED, no dedicated e_c / no concept params. The dedicated channel
  (D1) must beat *trained* member-bag injection (a stronger bar than the untrained inference-injection null).
  Decides whether the dedicated channel earns its complexity. Author wants this one.
- **ABL-2 Signed-scalar value vs ONE-HOT valence-level channels.** Replace the single signed-value channel with
  K one-hot level channels (hated/meh/liked/loved [+none]) so the encoder's first layer LEARNS each level's
  effect (valence-as-embedding) while staying in the strong dense AE (NOT the set-encoder, which lost on
  elicitation). Tests whether learned valence beats a linear scalar.
- **DROPPED — answerability/rated head.** The recommender only SCORES given answers; answerability is a
  policy/answerer concern (already modelled by the distilled answerer's know-levels/refusals). No recommender
  need; multi-task regularization too speculative to justify the complexity.
- NOT carried: the Transformer set-encoder backbone (V2-ST tested better full-profile but WORSE elicitation).

## 6. OPEN QUESTIONS FOR FABLE (low-token, ≤1 pass)
1. The concept channel `z += g(v,k)·W·e_c`: is an additive learned contribution the right fusion, or does it
   need attention over concepts (a user gives several concept answers)? Simplest that works?
2. Two-stage (D2): does Stage-2 fine-tuning risk catastrophic-forgetting the rare-item embeddings from
   Stage-1? Guard (freeze item embeddings in Stage-2? low LR? replay full profiles)?
3. Graded CONCEPT value as the contribution: earlier item-dislike was inert and concept member-bag flips did
   nothing — what's DIFFERENT here that makes it work (dedicated channel + learned + full-catalog + strength
   pretrain), and what's the single risk it's still inert? Name the decisive gate.
4. Is the multinomial-positive loss right, or does concept graded value need a term that directly supervises
   the concept→region effect (without re-introducing the circularity)?
5. The single most dangerous flaw that makes this not work.

## 7. FABLE PASS-1 AMENDMENTS (binding; verdict = GO-WITH-AMENDMENTS)

**MOST DANGEROUS FLAW = FALSE POSITIVE (not inertness).** Concept values are a deterministic function of
EASE-predicted taste, so the model can learn a monotone, knockout-passing "graded concept value" effect that
is actually EASE folded in TWICE — the human-stated-preference claim would be unsupported. Inertness gives an
honest null; this gives a fake win. Guarded by A2 (Ce control) + A2 (173 transfer).

**A1. HEADLINE GATE = USEFULNESS, not the monotone sweep.** The sweep (region moves hated→loved) is a
BEHAVIORAL response we already demonstrated (−26pt) and already showed NOT useful — it stays only as a
supporting check. The headline is: paired per-user EARLY-AUC (k∈{1,2,4,8}) lift on held-liked ATTRIBUTABLE to
concept value, via BOTH (a) eval KNOCKOUT (zero/scramble concept v_c on the same paired interviews) and (b) a
CAPACITY-EQUALIZED TWIN trained WITHOUT concept value. Report POSITIVE-half and NEGATIVE-half (hated/meh only)
lift SEPARATELY. Decisive: if knockout costs ~0.000 → heard-not-useful → report the null.

**A2. EASE-circularity guards (wholesale from rich_signal §8-B).** (a) Control **Ce**: identical model fed
EASE-predicted taste DIRECTLY at the asked concept positions (no answerer semantics) — the concept channel
must BEAT Ce or it is EASE plumbing (report null). (b) Pre-register the 173 real-LLM transfer as the arbiter:
the concept-value lift must reproduce IN SIGN on the 173 (caveat n=173 low power). (c) rated-item answers stay
REAL-STAR; concept values labeled "EASE-derived, upper bound" in every table until the 173 confirms.

**A3. Hygiene (carry §8/§9 wholesale).** Loader asserts known∩held=∅ per user AND answerer conditions ONLY on
known (A2 leak). State the ease_B-saw-population-held caveat. Per-channel 1/√nnz scaling (NOT L2-over-concat).
Paired eval interviews + seed repeats. Ceiling-normalized AUC (÷ own full-profile NDCG) to separate SPEED from
ASYMPTOTE. PRIMARY STATISTIC = exactly ONE pre-registered paired one-sided contrast: full − no-concept-value
early-AUC > 0.

**A4. Stage-1 concept derivation must MATCH Stage-2 answerer semantics.** Define the rule explicitly: concept
v_c = mean CENTERED rating over KNOWN member items (known-set only, assert); knowledge k_c from member
coverage. Its (v,k) output distribution must match the answerer's, else Stage-2 sees a distribution shift and
the pretraining is wasted/harmful. Write the rule before build.

**A5. Fusion (Q1): normalized additive + SHARED W.** z += (1/√n_c)·Σ g(v_c,k_c)·W·e_c — normalize (belief
magnitude must not grow with #concept answers, else turn-16 lives off-manifold vs turn-2). ONE SHARED W across
concepts (per-concept W = 18k×d² params rare concepts can't train) + per-concept e_c (member-bag init).
Attention is the ESCALATION, gated: only if order/duplicate/conflict-invariance (loved Sci-Fi + hated Alien)
checks FAIL. Don't buy attention before additive fails a named check.

**A6. Stage-2 (Q2): joint peak, no freezing.** Mixture IS the replay guard — fix full-profile fraction ≥30%
(pre-registered, not incidental). Stage-2 LR ≤ 0.1× Stage-1. Do NOT freeze item embeddings (interview regime
must move them). Guard by MEASUREMENT: track G-strength AND G-rare-item on disjoint val every epoch; preserve
the Stage-1 checkpoint forever; pick Stage-2 peak on a JOINT criterion — cold-start early-AUC subject to
full-profile NDCG ≥ 0.495 − ε (ε=0.010). Can't satisfy both ⇒ reportable TENSION, not a tuning knob.

**A7. Loss stays MULTINOMIAL-ONLY (Q4).** Softmax competition gives the negative half its legitimate gradient
path (suppressing a hated region's logits raises held-liked prob through the denominator) — no direct
concept→region supervision term (that re-introduces circularity + is a suppression objective NDCG@liked can't
reward; the α=0 finding stands). PERMITTED FALLBACK, pre-registered + labeled a MECHANISM CHANGE not a default:
if the A1 knockout gate fails, a small auxiliary latent-repulsion term (S2 lineage) may be tried as a NEW run.
