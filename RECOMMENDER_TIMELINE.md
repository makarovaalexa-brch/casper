# CASPER recommender build — post-RecVAE timeline (retrace Jul 19 2026)

Canonical purpose of this doc: stop relitigating the recommender arc. All full/tail = NDCG@10.
CANONICAL SCORING = `z @ Wd.T + decoder.BIAS` (learned bias), NOT `popb`=log-count. Report FULL *and* TAIL (HARD RULE #4).

## Timeline
- **Jul 4–5 — RecVAE / Instrument-2.0 rebuild.** V1 (biased-SVD fold-in) was weak: EASE left ~23x headroom on
  Goodreads. Rebuilt to RecVAE-d512 (a VARIATIONAL / belief-distribution recommender): ties EASE ML-1M 0.554,
  reopens Goodreads (+0.237 = 21.6x V1). Flagship actor 0.4907/0.2982. Certified target downstream: ~0.485–0.496 full.
- **Jul 13–14 — the working item recommender (attention set-encoder).** Insight: input/output separable -> keep an
  ARBITRARY-SET encoder (ISAB, interview-native, variable-length folds) but score with a MULTINOMIAL decoder over
  ITEMS ONLY (concepts never enter the 18,430-item softmax -> no collapse). Teacher a0c=0.4961 (dense signed vector,
  NOT set-native). Set-encoders: **paord 0.4859/0.3203**, pb2 0.4852/0.3295. This is the WORKING recommender. Item-only.
- **Jul 15–18 — concept integration (the STUCK saga).** Constraint R1: fold concepts WITHOUT dropping ~0.486 full.
  1. FiLM->fusion token (fusj, xattn): dilutes; full 0.480->0.475.
  2. "concepts aren't directions" -> RETRACTED: they ARE, after stripping the shared popularity component (AUC 0.44->0.94).
  3. OPERATOR was the bug: belief-token fold displaces the popularity prior -> full craters (0.258->0.226). Paper B's
     `popb + <Wd,u>` with a separate popularity FLOOR fixes the full-drop.
  4. UNION vs POINT: additive head scores the UNION of concept dirs -> SATURATES (+0.028 tail, all 470 concepts);
     Paper B folds to a POINT (intersection) -> COMPOUNDS (+0.052). A weaker model beat us on concepts.
  5. Learned FOLD-IN encoder: matched/beat Paper B — all-concepts TAIL +0.056 / FULL +0.029. BUT few-shot HURTS
     (kc=1 full -0.171); fades composed with items past K>=5.
  6. Concepts LOSE to items as an interview router (realizable concept +0.0112 < item +0.0168).
  STUCK POINT: any belief-moving fold that pays FEW-SHOT displaces popularity and drops FULL; any fold that preserves
  FULL (popb floor) only pays on TAIL at MANY concepts (kc>=16). Concepts never became a few-shot interview lever on
  the strong model without losing full or losing to an item question.
- **Jul 18 — Kalman belief pool.** Solved NON-DEGRADATION (NDCG must never drop per truthful answer): every point-shift
  fold could DECREASE NDCG (over-commits). Kalman = closed-form Bayesian update in the FROZEN decoder factor space:
  belief u~N(mu,Sig), score `popb + <mu,Wd>`; each concept answer = linear-Gaussian obs of <u,d_c>, d_c=whitened member
  centroid; mu moves by surprise x uncertainty, Sig ONLY SHRINKS. FIRST monotone-positive-per-question operator
  (oracle +0.090 tail by q8, never drops). Gives Sig for variance-greedy selection (champion +0.0669 tail).
  Answer bottleneck: implicit watch-lift (SEL) R2=0.376 >> LLM ordinal 0.017.
- **Jul 18–19 — learned adaptive policy (DKQS) COLLAPSED** to a static sequence (trainer failure: conditioning-collapse
  / gradient-starvation / soft-NDCG!=metric / BPTT credit-starvation), NOT an adaptivity verdict. Adaptivity is proven
  (Jul-14 tree +47%, E3 +0.0085) but the learned continuous policy didn't capture it.

## ★ THE THROUGH-LINE BUG (Jul 19) + the Kalman verdict
- Belief-pool scored with **popb=log-count** as the head bias, NOT the decoder's LEARNED bias (corr only 0.79).
  popb CRIPPLES full-NDCG (head-dominated); tail is head-masked so it hid. => "full is flat / concepts tail-only" was a
  popb-floor ARTIFACT. FULL is FAR from solved: cold 0.186 -> canonical fold-known 0.393 -> full-profile ~0.486. ~0.20 headroom.
- **Kalman CONFIRMED incompatible with the canonical recommender.** Fold ALL concepts, score with the REAL bias:
  full 0.167 -> **0.097 (CRATERS)**. It only "works" propped by the large popb floor (Wd@mu = gentle re-rank). Its
  non-degradation win lived entirely inside the weaker popb-floored machine. It builds its OWN mu (not the encoder's z)
  and reuses only the decoder weights. => a SEPARATE, WEAKER recommender than the 0.486 one.

## Concept-capable recommenders on disk (canonical scoring)
- **pbC_best.pt** (train_concepts_ord.py): SetEncoder + concept rows, folds items+concepts together, FULL 0.4946 /
  TAIL 0.3372 — concept training did NOT degrade item strength. Eval `concept_eval.py`. **Best concept-capable model.**
- fusion (fusj): ~0.48 full. wmat (W(v)): concept per-value matrix.
- paord/pb2/a0c: item-only base (0.486/0.496).

## OPEN QUESTIONS for the unified redesign
- Discard the Kalman, or salvage it in the encoder's z-space (keep non-degradation, fix the scoring)?
- Retrain the recommender as a BELIEF-DISTRIBUTION model (RecVAE IS variational)? We HAD a belief-distribution on
  items (RecVAE / conjugate pool) "not miles off attention" but chose ATTENTION for interview-native set folding — WHY,
  and could a variational/uncertainty-native encoder unify items+concepts+open at cold-start without losing strength?
- The intuition "concept knowledge SHRINKS belief" is sound (Bayesian), but the current split (attention recommender +
  bolted-on Kalman on a popb floor) is disjointed/broken. Need a UNIFIED multi-channel cold-interview recommender that
  keeps ~0.486 full.
