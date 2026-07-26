# Design sheet — Distillation-trained concept fold (fix the lossy concept operator)

> Pre-registered 2026-07-26. Incorporates the fold-operator deep research
> (`external_literature/findings/concept_fold_distillation.md`) and Fable's design review.
> Companion memory: `fable-concept-strategy-ruling-2026-07-26`. Decomposition:
> `docs/results/ANSWER_DECOMPOSITION_RESULT.md`. Author directive: fix concepts FIRST.

## 0. Question & diagnosis
The concept fold captures only **1–6%** of what folding the concept's own rated MEMBER ITEMS captures
(fine −0.003 vs +0.028; medium −0.001 vs +0.047; broad +0.002 vs +0.102). The information is provably in the
members; the fold loses ~95–99%. Root cause (lit-confirmed, HIGH conf): every prior fold was trained on the
sparse END TASK or hand-set — never against a dense teacher → credit-assignment failure for an upstream
gated module. **Fix (established, "assembled from" not novel):** distill the member-item-fold into the
concept fold (Privileged Graph Distillation, Wang SIGIR'21; cold-start line DropoutNet/Heater/GAR/ALDI;
LUPI Lopez-Paz; FitNets/RKD losses; Prototypical Nets = concept=centroid).

## 1. THE GO/NO-GO PRE-CHECK (Step 0 — run BEFORE any training; Fable §5)
**Open question Fable raised:** is the 95–99% loss credit-assignment (fixable by a dense target) OR an
information ceiling in the answer channel (student sees only (concept, one signed scalar); teacher varies by
*which* members rated)? If the latter, distillation polishes an empty pipe.
**Closed-form tabular-student probe (zero training, ~hours):** bucket train users by (concept, answer level);
set the prediction = the **mean teacher DELTA-score** in each bucket; fold that at eval; measure capture-rate
on val. This is the exact optimum of the distillation loss for this input → the ceiling of the whole plan.
Also report within-cell teacher cos-to-mean (variance diagnostic).
- **capture ≥ ~20–30% → PROCEED** to the trained student (adds smoothing + composition).
- **capture ≤ ~10% → STOP.** The fix is a RICHER STUDENT INPUT (answer + kc-so-far belief context), not a
  better loss. Re-spec before spending a train.
Run this FIRST, time-boxed, before the Step-0b ceiling sweep.

## 1b. Teacher ceiling (Step 0b — the true bar; research's biggest risk)
Measure the member-fold teacher's OWN NDCG ceiling per coarseness tier (fine/medium/broad) and vs full-profile
(~0.42). A perfectly-distilled student inherits the teacher's ceiling; teacher-relative success ≠ item parity.
Pre-register these tier ceilings as the explicit student bars.

## 2. Architecture (UNCHANGED from C-lite; item-safe by construction)
Frozen RecVAE/i25 tower (item path untouched) + gated fold-to-point concept module, gate **zero-init** → empty
concepts = bit-identical item behavior (G0 item-tie is mechanical, not earned). Do NOT co-train concept tokens
into the tower (prior C-full bought nothing). Gated zero-init additive residual = Side-Tuning α-from-0.

## 3. Teacher (Fable §2/§3 rulings)
- **Teacher (PRIVILEGED, train-only, LUPI):** `u_teacher = frozen-tower fold of {(i, r_ui) : i ∈ members(answered concepts), i in user's train-profile}`, signed ratings, SAME fold operator as deployment.
- **Student:** the concept answer(s) ALONE (no member ratings) → deployable for item-cold users.
- **DELTA form (M2):** target = teacher-fold **minus current base belief** at that interview state (else
  L_embed fights the gate). 
- **PRIMARY target = decoder-SCORE match:** KL over the item distribution (temperature τ) between student
  delta-scores and teacher delta-scores — dense (18k items of signal/example = the credit-assignment fix),
  invariant to latent parameterization, aligned with the metric. Raw-coord `‖u_s−u_t‖²` = SECONDARY regularizer
  only (scale-brittle on a frozen tower). Relational/GAR held in reserve; trigger = kc=1 full < 0.
- Full-profile teacher = WRONG (leaks non-concept info). Oracle-concept-belief = secondary arm only.

## 4. kc semantics (Fable M1)
kc = **concepts answered** (the recorded kc=1 crater, full −0.171). Curriculum = sample interview length
kc∈1..32 via member-dropout (DropoutNet). Teacher at kc>1 = the **JOINT** member-fold over the UNION of
answered concepts' rated members → trains multi-concept composition. Pre-register a 2–4-concept composition gate.

## 5. Loss
`L = L_rank (held-like multinomial NLL, keep) + λ · L_score-delta (KL) [+ ε·L_coord-L2 secondary]`.
λ grid **{0.1, 1, 10}**, val-fold selection rule fixed in advance (max concept capture s.t. item G0 tie held).
Answer = SEL signed four-band (Jul-22 audit; NOT LLM-ordinal), IDENTICAL in train and eval.

**Training-schedule arms (author, 2026-07-26 — distill vs distill-then-sharpen):** the teacher is a
coarseness-bounded PROXY; NDCG is the real objective (plain-NDCG C-lite learned a taste-region shift, not
member-reproduction), so we want the student to be ABLE to exceed the teacher.
- **Arm S1 (PRIMARY, safe): JOINT** — L_rank + λ·L_distill throughout (standard Privileged-Graph-Distillation
  form). Distillation regularizes continuously; can't collapse.
- **Arm S2: DISTILL-THEN-SHARPEN** — phase 1 distill to convergence (L_distill dominant), phase 2 finetune on
  NDCG with a RETAINED distillation ANCHOR (λ lowered, NOT zeroed) + **early-stop on the capture-rate gate**.
  Guards the known risk that NDCG's sparse gradient (the original cause of the 1–6% fold) drifts the distilled
  student back down. Monitor capture-rate every phase-2 epoch; stop on erosion.
- **Read:** S2 > S1 ⇒ the concept answer's best use isn't member-reproduction (teacher was a floor, taste-region
  wins); S2 collapses toward 1–6% ⇒ sparse-gradient diagnosis confirmed, S1 is the answer. Both reported.

## 6. Gates (numbers pre-registered; paired per-user bootstrap CI on every delta)
- **G0 item-tie:** bit-identity on full-profile item NDCG (frozen + zero-init) — mechanical assert.
- **CAPTURE-RATE:** student-lift / teacher-lift per tier ≥ **30%** (vs current 1–6%); reported against the
  Step-0b teacher ceiling.
- **NO few-shot harm:** kc=1 full ≥ −MDE (fixes the −0.171 crater).
- **Composition:** 2–4-concept joint fold ≥ single-concept (monotone, not saturating).
- **Deployment:** concepts-only realizable curve rises (no below-intercept dip); coarse-to-fine OPENER
  retained (concept still wins q1).
- **Item cold not degraded:** coldk2/coldk8 within CI of the current signed C-lite.

## 7. Controls (Fable M3; HARD RULE 5)
(a) **same-budget C-lite retrain WITHOUT L_embed** (isolates the distillation claim);
(b) **shuffled-teacher canary** (random user's teacher must NOT help — leakage/gate-opening check);
(c) answer definition fixed SEL, train==eval; canonical-snap of item G0; leak check (known∩held=∅).

## 8. Protocol
Eval = canonical ML-25M Liang (train 140,768 / val 10k / test 10k, seed 98765; tail = top-33%-mass head mask).
Quarantined 173/300 study users excluded. Frozen-tower checksum recorded. ≥3 real seeds (established set)
before any "distillation wins" claim. MDE stated. Commit code before any run (HR10). Certification of the
instrument stays parked (separate track) until the concept operator is settled.

## 9. Build order & stop rule
1. Step 0 tabular probe → **STOP if ≤10% capture** (re-spec to richer student input).
2. Step 0b teacher ceiling per tier.
3. Primary train: score-delta KL + kc-curriculum + zero-init gate. Controls (a)(b) same run.
4. If kc=1 full < 0 after fix → switch to relational/GAR target (pre-registered trigger).
5. Coarse-to-fine hybrid (concept q1–2 → items) consumes the improved fold — separate, later.

## 10. Risks
- **[HIGHEST] Answer-channel information ceiling** — settled cheaply by Step 0 before any train.
- Score-match brittleness on frozen coords → relational/GAR fallback ready.
- Teacher near cos≈0.83 coarseness ceiling → Step 0b makes it the explicit, honest bar (not item parity).
