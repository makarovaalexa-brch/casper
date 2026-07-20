# DESIGN SHEET — VarHead: low-rank Gaussian posterior head on frozen pbC (UNSIGNED DRAFT)
Fable, 2026-07-19. Replaces the Kalman+DKQS line. No run until author signs.

## Question
Can a low-rank Gaussian covariance head, attached to the FROZEN pbC set-encoder and trained with a
partial-ELBO whose likelihood is the REAL multinomial decoder (`z @ Wd.T + bias`), learn a posterior in the
encoder's z-space that (G1) shrinks directionally with evidence and (G2) supports Σ-gated non-degrading folds
and variance-greedy question selection — WITHOUT moving full-profile NDCG (frozen mean = exact 0.4946/0.3372)?

## Model (all existing weights FROZEN; pbC_best.pt trunk + decoder)
- Read r_pool = cat[p, log1p(|set|)] (513-d, the exact input `SetEncoder.head` sees; expose via forward return).
- VarHead: V = W_V(r_pool) reshaped (512, 32); logd = W_d(r_pool) (512). Σ = VVᵀ + diag(softplus(logd)).
  Init: W_V ≈ 0; logd bias set so Σ(empty) ≈ Σ0 (no evidence → prior).
- Prior: N(z0, Σ0); z0 = pbC's frozen empty-set parameter; Σ0 = learned diagonal (512 params), init from the
  empirical variance of paord full-profile z over train users (salvaged bpool2 trick, now in z-space).
- Trainables: W_V, W_d, logΣ0 only (~8.7M params). μ path and decoder bit-frozen (drift-canary assert, as in
  train_concepts_ord.py).

## Objective (partial ELBO, likelihood = the real decoder)
loss = E_ε[ NLL_multinomial(held-liked | z' = μ + Vε₁ + sqrt(D)ε₂) ] + KL( N(μ,Σ) ‖ N(z0,Σ0) )
1 MC sample/step (flagged shortcut; alternative = 2 samples, 2x cost). KL via matrix-determinant lemma (32×32).
μ frozen ⇒ the KL μ-term only trains Σ0; β=1 from the start (no anneal needed in this stage).

## Data (HARD RULE #1: no reduction)
- Train: ALL usable answerer-train users (same cohort/filters as train_concepts_ord.py rows: ≥8 rated,
  ≥1 non-refused concept, ≥1 held-like — pre-existing protocol filters, restated not added).
- Curriculum: set_mn.make_pb_example mix (item-only 0.35 / concept-only 0.35 / mixed 0.30, k~1..32 concepts,
  heavy item dropout) so Σ is trained at EVERY evidence size, both channels.
- Eval: full val answerer cohort (concept_eval.py split, AC.SEED-keyed per-user halves). No user subsampling.

## Gates (pass/fail; ALL report FULL and TAIL, real decoder bias — HARD RULE #4)
- G0 canary (by construction): eval_student on μ = 0.4946/0.3372 bit-identical every epoch; max|drift| = 0.
- G1 posterior usable: (a) tr(Σ) strictly ↓ in k ∈ {0,1,2,4,8,16,32,full}; (b) DIRECTIONAL: after folding
  concept c, var along d_c (whitened member-centroid in decoder space) drops ≥2× more than along random dirs;
  (c) calibration: per-user wᵀΣw vs realized held-like NLL error, Spearman ρ ≥ 0.2 (fail-direction if < 0.05).
- G2 decisive: cold-start per-step curves, q=0..8 truthful answers (answerer tables), arms:
  (A) raw pbC μ fold (the point-shift baseline), (B) Σ-gated fold μ' = μ_old + G(Σ)(μ_new − μ_old),
  (C) question order: variance-greedy argmax wᵀΣw (answerability-masked) vs random.
  PASS = B monotone-nonneg per step where A is not, AND greedy > random by q4 with 95% CI off 0 (tail),
  full never below cold intercept. MDE ~0.005 tail on ~3k val users (report CI regardless).

## Cost
CPU-only, no LLM calls, no new data. Head ~8.7M params on a frozen trunk; each step = existing encoder forward
+ one extra decoder forward. Est. 2–4 h/epoch, 2–3 epochs ⇒ ~1 day wall-clock. Checkpoints every epoch to
.cache/set_mn/vhead_ep*.pt (save-all rule).

## Flagged shortcuts (each with alternative)
1 MC sample (alt: 2); rank r=32 (alt: 64; capacity choice, not a data cap); G1 directional test on 200 probe
concepts (alt: all 1628 — cheap enough, default to ALL unless author says otherwise); Σ-gate functional form
G(Σ)=Σ(Σ+S)⁻¹ with scalar S grid (alt: per-level S from answerer noise).
