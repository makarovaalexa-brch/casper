# Step-2 design sheet v2 — belief layer on the certified tower (post-adversarial-review)

> v1's design (i) (rank on the conjugate posterior mean) was REFUTED at review 2026-07-22: the conjugate
> mean is a linear/WLS estimator, rank-singular below d for most users, and provably cannot reproduce a
> nonlinear set-encoder fold (independently corroborated by the un-demoted −0.0074 convex-avg-can't-
> intersect finding). v2 = the pre-registered fallback, promoted: **design (ii), decoupled**.
> Battery: `GATE_BATTERY_INSTRUMENT.md`. Code committed before run (HR10).

## Architecture (design (ii), final)
- **Ranking mean = the frozen tower's own nonlinear fold** enc(S_t). Preserves C1 certification and gives
  G0 as BIT-IDENTITY at full profile (the design-(ii) form). Empty set → enc(∅) = the popularity
  intercept (G0 intercept assert holds by identity; no train-mean prior — v1's F5 repaired).
- **Σ = analytic precision accumulator, decoupled**: Λ_t = Λ0 + Σ_j α_j φ_j φ_jᵀ/σ_j². Σ's job is
  SELECTION (which question next) — and it must EARN it: **G8 runs in selection form** (Σ-greedy question
  choice beats isotropic-cI-greedy on G2 endpoint). R4 is claimed as *selection-grade uncertainty*, not
  scoring-grade; the chapter says so. (v1's A7-strawman motivation deleted — F2.)
- **R5/monotonicity** lives in the fold operator: evidence sets only grow; the demoted "Kalman craters"
  question becomes moot for scoring (the mean never leaves the certified fold); Q4 is retired.

## T2' tower architecture v2 (author-revised 2026-07-22 — FROZEN-GEOMETRY DISTILLATION)
- **Frozen, inherited from T1 RecVAE (canonical-split ckpt):** the decoder Linear(200→n_items) + bias,
  AND the input item embeddings (= decoder rows Wd). The latent space is RecVAE's, permanently.
- **Trainable ONLY:** attention blocks (mab_in→sab→PMA), FiLM γ/β level tables (NLEV=10), any projection.
- **Token:** γ(level)⊙e_i + β(level), e_i frozen — identity frozen, valence learned. **Signed init
  (--sign_prior, default ON):** γ=1; β(level) = κ·v(level)·ū (v = centered valence in [−1,1], ū =
  normalized mean decoder row, κ small) — hated starts mildly repulsive, loved mildly attractive; training
  refines magnitude (RUNG1 hard-wired-sign lesson: sign unreachable from null init).
- **Loss:** λ_z·‖z_set(graded S′) − z_T(binarized S′)‖² + (1−λ_z)·NLL(dec(z_set), held likes), λ_z=0.5;
  teacher z_T = RecVAE encoder fed the binarized SUBSET (subset sizes sampled broadly incl. 1–8 →
  supervises the interview regime directly). EDLAE score-CE demoted to optional hook.
- **Fallback arms:** --unfreeze_emb if val stalls; --no_sign_prior ablation; graded-vs-binarized ablation
  = the G3 canary (teacher is grade-blind — the KD pressure is toward sign-blindness; the NLL must win).
- **Why:** tiny trainable surface (fast, low overfit), certified fixed geometry for everything downstream
  (concept dirs, Arm A shifts, Σ), teacher supervision exactly at interview set sizes.
- **ADVERSARIAL REVIEW REPAIRS (2026-07-22, all adopted):** (1) λ_z gated OFF at k≤8 + annealed →0
  (teacher latent is covariate-shift noise at small subsets — its L2-norm input is 3–9× off-scale there;
  KD = full-profile warm-start only); (2) sign init on γ per-token (γ(hated)=−1 ⇒ −e_i), NO β∝ū (that
  seeded valence on the popularity axis → G3a false-pass + G5 unpassable); (3) input identities =
  normalized Wd rows + norm as scalar feature; (4) held dislikes as down-weighted NLL negatives (else
  dislikes get no gradient); (5) evidence gate g(0)=0 for exact empty-set intercept = softmax(RecVAE's
  TRAINED bias, not log-pop); (6) frozen tensors out of the optimizer + trunk-drift assert; (7) G0 split:
  strength = CI-tie vs RecVAE full-profile; identity = harness bit-pass-through.
- **GATING PRECONDITION for the training night (existential):** the RUNG1-Check-2 like/dislike
  separability probe on the CANONICAL teacher must clear (disliked falls ≥0.10 pctile more than liked;
  dislike-heavy ΔNDCG ≥ −0.005) — if the frozen decoder geometry cannot separate liked/disliked
  neighborhoods, NO training on top passes G3 → escalate to a trainable-decoder tower. Probe:
  `scripts/_verify/probe_dislike_separability.py` (+ teacher cold-curve for the λ_z gate).

## The tower question v2 exposes (feeds Step-1 selection — decision pending tonight's numbers)
The instrument's fold must ingest **(entity, graded value, confidence) tokens** (G3/G4 demand it).
- **RecVAE (tonight's run):** R1-strong but input = binary item vector — graded/signed values are NOT
  native; usable as R1 bar, ruler, and distillation teacher, but awkward as the interview tower.
- **pb2-class set encoder:** graded-native (item,level) tokens, order-invariant — the R2/R3-correct fold.
  ⚠ the existing pb2 ckpt CANNOT be scored on the canonical ruler (its train users overlap the new test
  cohort = leak) → **retrain pb2-class on the canonical split** (1 night) = tower arm T2', with
  EDLAE/RecVAE distillation available to close any R1 gap (T3').
- Working hypothesis: interview tower = graded-native set encoder at best-achievable R1; RecVAE/EASE/EDLAE
  remain the bar it is measured against. Final pick gated on G0' after T2' lands.

## Observation model (for Σ; all analytic — with the review's repairs)
- item: φ_i = raw decoder row Wd[i] in the observation equation; graded target in LOGIT scale
  y_i = g(level)·‖Wd[i]‖ (F3 repair); normalization only in leverage terms.
- concept (Σ direction): φ_c = enc(member-bag) − enc(∅) under the actual tower (the only computable
  definition here — F4); matched non-member set for G5 AUC = non-members popularity/like-count-matched.
- **concept RANKING ingestion (the hole the author caught 2026-07-22): Σ-only concepts cannot move recs.**
  The concept's effect on the MEAN is an explicit three-arm decision, all inference-time on the frozen
  tower, all cheap, decided by G5 + G-collinearity on the clean stack:
  - **Arm A (favoured by the record):** additive score-space residual over the intercept floor —
    `score = base(enc(S)) + β·w_c·⟨Wd, whitened member direction⟩` (IDF w_c, per-user norm; the Jul-17
    operator that worked UNTRAINED: tail +41% with full never dropping — magnitude demoted popb-era,
    structure = strong prior; floor is now the learned-bias base, not popb).
    **NOT a bolt-on — it IS the belief mean update:** since score(i)=⟨Wd[i],z⟩+b_i, the residual is
    identically a latent mean shift z → z + β·w_c·v·d_c — a linear-Gaussian observation update along the
    concept direction. Items update the mean via the certified NONLINEAR fold; concepts via LINEAR latent
    shifts; Σ pools precision from both. One latent, one decoder, one belief. The Jul-17 failure was the
    insertion point (inside the attention pool), not the information; Arm A inserts post-fold. Sign from
    like/dislike (G3 flip applies to concepts too), magnitude from graded value, weight from confidence.
    Cold: mean=enc(∅)=intercept, one concept answer shifts along d_c (G5/G-collinearity measure it).
    Full profile: no concept answers ⇒ mean=enc(S) exactly ⇒ G0 bit-identity preserved.
  - **Arm B (the recorded failure, kept as G5 comparator):** member-bag token injection into the frozen
    attention fold — documented to DISPLACE the prior (0.258→0.226, `z += e_c`); expected to lose.
  - **Arm C (only if A fails G5):** concept tokens trained into T2' (label design from SEL) — a training
    cost and a known-hard path; not default.
  Whichever arm wins carries the R3-concept ranking claim; Σ keeps selection duty regardless.
- confidence: σ ∈ {know-well, vague, refuse} (3-way, matching G4's ordering — F7 nit).
- refusal (AUTHOR CORRECTION 2026-07-22 — refusals are information): "no clue about X" is a CONSUMPTION
  observation, not a null — the user doesn't live in that region (the SEL/film-mute lesson). Refusal
  folds as its own observation type: small shift along the probed direction with separately-fitted
  α_refuse (expected sign: away from region; fitted on val, reported; zero if the data says zero) —
  NEVER a preference-value update. Still burns the turn. G4 ordering unchanged:
  α_refuse < α_vague < α_know-well; G1's exemption becomes "refusal shrinks Σ by at most its small
  fitted α", not "exactly zero".
- α calibration: explicit objective = held-out-item NLL on val increments (PrecAcc's Huber increment loss
  carried forward — F6), grid/convex fit over ≤13 scalars, cost stated in the run plan (F8); pre-fit sign
  proof dL/dα_c < 0 at α=0 asserted before fitting (G1 requirement, reinstated).
- reinstated lineage gates (F10): G-collinearity (concept-only NDCG must clear the popularity intercept)
  and α-crediting report (does the fit give concepts nonzero weight).
- concept-redundancy control (F9): folding item i + the concept whose SEL derives from i must not gain
  over i alone; SEL-on-fold-in labeled as evaluation simulator, not a deployable answerer.

## Battery artifacts now defined (F7)
- **G9 fixed question bank:** frozen, user-independent — top-200-popularity items + the 50 largest
  structural-rule concepts; committed as a JSON before any run.
- **G0 tie/identity:** bit-identity of the mean path (design (ii)); tie-band language deleted (F8) —
  strength questions are Step-1's (tower), not Step-2's.

## Pre-registered questions
- Q1 (G0): bit-identity of full-profile scoring through the belief harness. Cheap, mechanical.
- Q2 (G2): cold curves q∈{0,1,2,4,8,16}, items then +concepts, vs random; endpoint/AUC keyed; 10k test.
- Q3: battery G1, G3–G6, G8-selection, G9 as specced above.
- Answer sources: items = real held ratings; concepts = SEL excluding held-out targets; structural
  answerability rule. No LLM, no retired grids (assert: load_answerer never called).

## Sequencing
1. Tonight: RecVAE finishes (R1 bar + teacher candidate). 2. T2' pb2-class retrain on canonical split
(next night; committed first). 3. Tower pick (G0'). 4. Step-2 code vs this sheet → its own adversarial
review → commit → battery run.
