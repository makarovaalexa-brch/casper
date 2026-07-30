# Design sheet: the interview-native retrain

**Opened 2026-07-30.** Author ruling: *"i actually really dislike all those blends and boltons. can we not
have a native architecture that makes the most of the signal?"* — and that instinct is now supported by
measurement (§1). Design adjudicated by Fable. **Nothing runs until the gates below are agreed.**
Evidence: `INTERVIEW_TABLE.md`, `ARM_N_PHASE1.md`.

---

## 0. The decision rule for failure, stated before anyone is attached to the run

**If G-FULL or G-EMPTY fails after one honest debugging pass, the retrain is ABANDONED**, the certified
`t2final_best.pt` stands, and the chapter ships the scarce-regime framing with no architectural change.
A retrain that *must* succeed is the one configuration under which none of these gates mean anything.

## 1. Why we are not shipping an inference-tier patch — MEASURED, not argued

**B2 (R4-weighted blend toward the population marginal) FAILED.** Weight `w_i = 1 − φᵢ'Σ_t φᵢ / φᵢ'Σ_0 φᵢ`,
unit observation precision, untuned. Against the floored baseline:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| HELF | 0.0000 | −0.0008 | **−0.0094** | **−0.0120** | **−0.0118** |
| entropy0 | +0.0018 | 0.0000 | −0.0024 | **−0.0109** | −0.0099 |
| popularity | **+0.0035** | **+0.0027** | −0.0012 | +0.0031 | −0.0001 |

Tail down almost everywhere. `mean_w` runs 0.004–0.11, so the blend is 90–99% anchor and mostly returns
popularity. **A blend cannot be the answer**, and this is the third independent verdict that Σ is not a
usable weighting signal (G1c dropped it; B2 confirms).

**Anchor-swap control:** anchoring to the LEARNED empty-set decode instead of popularity is catastrophic
(HELF k=16 → 0.1443, entropy0 k=8 → 0.1217), so the mechanism is prior *quality*, not estimator variance.
*Caveat: those numbers degrade monotonically in k, which should not happen; the control may be buggy.
Direction only — do not cite the values.*

## 2. ★ SPAN TEST: the population marginal is NOT reachable — `delta_b` is REQUIRED

Ridge-fit `z0* = argmin ||z0 Wd' + bd − log p_pop||`, then score `decode(z0*)`:
**Spearman 0.836, NDCG@10 = 0.0032** (Most-Popular = 0.1626).

A 200-dim linear decoder tracks popularity loosely across 18,359 items and gets the **top ten**
completely wrong — and NDCG only sees the top ten. **No amount of encoder training can make our
zero-evidence prediction equal the population marginal.** This is a representational limit of the frozen
decoder, not a curriculum gap. Fable's conditional branch fires: **`delta_b` is required**, not optional.

## 3. ★ INTENSITY IS LOAD-BEARING — a new gate follows

Collapsing all like-levels (4.0/4.5/5.0 → one band), dislikes left graded, on the certified checkpoint:

| k | graded | collapsed | Δ |
|---|---|---|---|
| 1 | 0.2002 | 0.1862 | **+0.0140** |
| 2 | 0.2289 | 0.2038 | **+0.0251** |
| 4 | 0.2540 | 0.2168 | **+0.0372** |
| 8 | 0.2770 | 0.2480 | **+0.0290** |

Intensity is worth **+0.014 to +0.037** and does NOT vanish by k=8 (the earlier +0.0147-at-k=2 figure was
arm A with a likes-only fold-in — a different regime). R5 now has BOTH halves evidenced, plus independent
corroboration from Golbandi's `no_sign` ablation (+0.0067…+0.0149 to a 2011 tree). **The retrain must not
damage the γ bands ⇒ G-INTENSITY below.**

## 4. Architecture

```
z = native_z + rho_taste( Σ φ(e_i, gamma(level)) )        [UNCHANGED]
              + rho_expo ( Σ psi(e_i) )                    [NEW: asked-but-unseen items]
score = z @ Wd' + bd + delta_b                             [NEW: per-item bias]
```

- **Exposure branch, not an 11th γ level.** An 11th band forces exposure through the taste pathway as a
  scalar — i.e. encodes "unseen" as weak dislike, the exact category error we identified. A separate
  branch keeps exposure and taste distinct and lets the channel learn the *pattern*.
- `psi` reuses the existing frozen item embeddings with a learned projection — **no new 18k table**.
- `rho_expo` output is **zero-initialised** (the proven i25 trick): the network starts at exactly today's
  certified behaviour and the branch must earn its way in.
- `delta_b`: **zero-init, trainable, per-item** (18,359 params). The RecVAE decoder stays frozen.
- No ask-mask input needed — (answered tokens, unseen tokens) jointly determine it.
- Capacity: +~120–270k ⇒ ~1.0–1.1M trainable. **Do not widen `rho_taste`** unless the smoke run shows a
  train-loss plateau with regime buckets still improving. Chasing a 0.005 gap by 5–10×-ing the head puts
  the 0.3482 floor at risk.

**Adjudication of the "more capacity + more examples" hypothesis: half right, and the wrong half is the
expensive half.** Every measured weakness is a *distribution and channel* failure, not a capacity
failure. Unlimited synthetic interviews are unlimited **masks**, not unlimited **information** — the
information is capped at 140,768 users. Expect gains from distribution match and the new channel; expect
~nothing from scale beyond coverage.

## 5. Curriculum

Train users only (val/test quarantined).

- **Regime mixture:** 35% full-profile/dropout (protects the floor) · 55% interview · 10% zero/near-zero
  (k=0 ≈5%, k=1 ≈5%). **The k=0 bucket IS the prior-anchoring mechanism** — no loss term; NLL on empty
  inputs trains the empty-set decode toward the marginal directly (and `delta_b` gives it somewhere to go).
- **Strategy family, not the six eval arms.** ask-score = α·log-pop + β·entropy + γ·HELF-harmonic + ε·noise,
  with (α,β,γ,ε) sampled per example. The six arms are points in this family. **Hold out the exact HELF
  weights entirely** and gate on them (G-HELDOUT) — that answers the overfit worry with a measurement.
  Simulating public strategy functions on train users is legitimate and symmetric: it is exactly the
  information Golbandi's leaves are built from.
- **Answerability = the transparent structural rule**: answered iff in the user's rated set. Answered →
  level token from the real rating; unanswered → **unseen token**. This one line creates everything the
  model has never seen: unanswerable questions, ask-order/answer correlation, popularity-skewed asks.
- **Budget:** k log-uniform over {1,2,4,8,16,32} (uniform starves k≤2 in effective-answer terms).
- **Masking:** `prof`, NOT `prof|asked` — the Paper B scar. Most likely silent bug; assert it.
- **Volume:** 2–4 passes of full coverage; beyond that it is resampled masks. Spend the compute on gates.

## 6. Objective — ADD NOTHING

Keep exactly: multinomial NLL on held-out likes + `w_neg`·clamped mean-logprob on held dislikes.

- Prior anchoring rides the k=0 bucket through the existing NLL.
- The exposure branch needs no dedicated loss: asked-unseen items are never targets, so NLL already
  teaches both uses — demoting the reported-unseen items and exploiting the pattern for taste inference.
  *If the signal is worth 0.008, NLL will find it; if a bespoke auxiliary loss is needed to make it
  appear, it was an artifact.*
- **Explicitly declined, per the graveyard:** no distillation term (KD null, `lam_glob`=0), no
  σ-calibration term (G1c), no fold-all belief objective (Kalman crater), no sequence/non-myopic term
  (polish dropped).

## 7. Pre-registered gates

Val-side of train users for go/no-go. **Canonical test touched ONCE, at the end.**

**Abort:**
- **G-FULL** — canonical full-profile full@10 ≥ **0.3467** (0.3482 − the 0.0015 paired-CI width). Below
  this, ABORT: R1 dies for a 0.005 interview gain.
- **G-SNAP** — certification snap passes as before.
- **G-EMPTY** — empty-set decode ≥ **0.1626**. Binary; the whole prior story rests on it.
- **G-NOHARM** — random / pure-entropy arms ≥ prior − 0.002 at every budget.
- **G-INTENSITY** *(added from §3)* — like-band collapse must still cost ≥ +0.020 at k=2 and k=4. If the
  retrain flattens γ, we have traded a measured R5 asset for an interview gain.

**Ship:**
- **G-SHORT** — HELF full@10 ≥ Golbandi *no_unknown* at k∈{2,4,8} (0.1714 / 0.1823 / 0.1867) **and** ≥
  ternary at k=8 (0.1948). Stretch, report not gate: ≥ ternary at k∈{2,4}.
- **G-LONG** — entropy0 and popularity at k=16 within 0.003 of the backbone (0.2238 / 0.2182).
- **G-TAIL** — tail@10 at k≤4 not below current by more than the paired CI. *(B2 showed an anchor eats
  tail; this gate exists because of that evidence.)*
- **G-HELDOUT** — held-out strategy config within 0.005 of in-family strategies at matched answered-rate.
- **G-SHUFFLE** — exposure gain must **vanish** under a matched-count random-unseen shuffle. If it
  survives, the branch learned the activity proxy (Spearman 0.65 with profile size) and is disqualified
  **even if NDCG is up**.

## 8. Risk register

| # | risk | precedent | early-warning signal |
|---|---|---|---|
| 1 | full-profile regression | Kalman fold-all crater 0.167→0.097; DAE snap −0.022 | val full-profile every epoch from ep1; abort at >−0.005 by ep2 |
| 2 | exposure branch learns the activity/popularity proxy | LLM answerability popularity-dominated; concept dirs needed whitening | **G-SHUFFLE at the FIRST checkpoint, not the last** |
| 3 | regime competition (interview improves, full profile quietly sacrificed) | — | per-regime NLL logged separately from batch 1; all buckets must descend |
| 4 | `delta_b` popularity double-count | — | `delta_b` norm growth without NDCG gain ⇒ kill it |
| 5 | prior anchor drags tail at low k | B2: tail 0.0274 vs 0.0302 | G-TAIL from the first eval |
| 6 | harness / masking artifact | the void 0.3894 leak; the `prof\|asked` scar | any cell beating ternary Golbandi by >0.01 is a bug until audited |
| 7 | simulator answered-rate ≠ eval answered-rate | — | checked in §9 step 3, **before** training |

## 9. Order of operations

1. ~~**Span diagnostic**~~ **DONE** → not in span, `delta_b` required (§2).
2. ~~**Binary-collapse ablation**~~ **DONE** → intensity is load-bearing (§3), G-INTENSITY added.
3. **Generator + unit tests, NO training.** Simulate the family on train users; verify answered-rate per
   strategy matches eval (HELF 1.6/8, entropy0 3.1/8); verify the masking rule; **commit** (HARD RULE 10).
4. **Smoke run**, 1 epoch on a subset (~1–2 h). Pass: every regime bucket descending, empty-set decode
   moving toward 0.16, full-profile bucket not degrading. Risks 1/3/7 surface here.
5. **Full train** at i25 timescale (expect ep4–13, ≤~3.5 h). Per-epoch val gates. Keep best only.
   `t2final_best.pt` untouched under its own name; the new arm gets a new name.
6. **At best checkpoint: G-SHUFFLE and G-HELDOUT** — before the cascade.
7. **Only after all val gates pass:** canonical test once, then the cascade (certification snap, greedy
   sequences, graded curves, concept-fold re-runs).

## 10. Open questions for the author

1. Is the `delta_b` per-item bias acceptable given §2 shows it is required? It is a popularity sink — a
   reviewer may read it as bolting popularity onto the decoder, which is exactly the aesthetic objected
   to, except now it is trained rather than blended.
2. G-FULL floor at 0.3467 — agreed, or stricter?
3. Hold out HELF (the strategy Golbandi wins) or a different config? Holding out HELF means the headline
   comparison runs on an unseen strategy — honest, but strictly harder.
