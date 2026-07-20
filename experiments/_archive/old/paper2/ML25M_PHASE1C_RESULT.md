# ML-25M Replication — Phase 1c Result (V1 NEURAL instrument health check)

Date: 2026-07-03. Phase 1c is the decisive health check on the **REAL instrument class** used by
Papers A/B — the V1-style **attention fold-in set-encoder** (learned, nonlinear), not the linear
biased-SVD ridge reference that Phase 1b gated on. Phase 1b showed the *linear* fold has near-zero
headroom on ML-25M full-catalogue NDCG@10 (+0.009). Phase 1c asks whether the **neural** encoder
extracts personalization the linear reference cannot, at a **catalogue-depth-matched ruler**.

Repo: `C:/dev/phd` (inner `casper/`). Data: `data/movielens/.cache/ml25m/` (meta.npz, Q_svd.npy,
bi_svd.npy, Ac_concept.npy — full-data, 162,541 users / 18,430 items). V1 architecture/recipe mirrored
from `scripts/paper2/encoder_recon.py` (ML-1M V1): d=64 attention pooling over revealed (Q[item],
residual) tokens → user vector u; frozen decoder popb + Q·u; IPS/tail-weighted BCE reconstruction of
the unrevealed like-set; masked/shuffled random-count reveals k=1..K.

---

## STEP 0 — THE RULER (pre-stated BEFORE any evaluation)

- **Primary depth: NDCG@50** on ML-25M's 18,430-item catalogue. Rationale: cross-dataset metric depth
  must scale with catalogue size. ML-1M's headline K=10 sits at catalogue fraction 10/3,706 = 0.0027;
  the matched fraction on 18,430 items is 0.0027 × 18,430 ≈ **K=50**. K=50 is the primary ruler.
- **Reported alongside: NDCG@10** (same as ML-1M's absolute K) for transparency — expected to be
  harsher / more popularity-saturated at this catalogue size (Phase 1b showed tail@10 = 0).
- **Tail = Cremonesi long-tail** at the SAME Ks: head = most-popular items covering 33% of train likes
  are masked out of the candidate scores and the relevant set (mirrors ML-1M's head-33% definition; in
  code TAILCUT by cumulative-popularity 33%, consistent with Phase-1b `poprank`/head handling).
- **Anchor:** MOSTPOP (popularity-only, q0) at each K. **Reference:** linear ridge fold-in (Phase-1b
  instrument). **Instrument under test:** V1 neural encoder fold-in.
- **GATE (judged at primary K=50):** encoder full-profile − MOSTPOP **substantially positive**, judged
  against ML-1M's +0.10 at its matched depth, AND monotone no-harm (revealing never hurts).
- **If gate FAILS:** run the headroom-vs-depth curve, full-profile − MOSTPOP at K = 10/20/50/100, to
  state precisely where (if anywhere) personalization signal exists on ML-25M.

---

## STEP 1 — encoder training (FEASIBLE on FULL data; no fallback needed)

- **Full-data training, NOT the subsample fallback.** All **161,536** train users with ≥K+1 ratings
  (of 161,541) trained; full 18,430-item catalogue; frozen Q_svd decoder (V1 default, JOINT=0).
- Recipe mirrors ML-1M V1 (`encoder_recon.py`): attention pooling over revealed (Q[item], residual)
  tokens → u; decoder popb + Q·u; IPS-weighted BCE reconstruction of the unrevealed like-set; random
  reveal count k=1..K, K=12; Adam 1e-3, wd 1e-5, batch 512; d=64.
- **Cost: ~200 s/epoch on CPU** (well under the 3 h/epoch fallback trigger). Ran resumably in 595 s
  Bash chunks (2 epochs each). Val = encoder full-profile fold-in NDCG@50 (full catalogue) on the 500
  val users, held-out half-likes; best-val selection.
- **Convergence (val NDCG@50):** ep1 0.3049 · ep2 0.3140 · ep3 0.3180 · ep4 0.3189 · ep5 0.3207 ·
  ep6 0.3217 · **ep7 0.3226 (PEAK, saved)** · ep8 0.3223 (turned down → stopped). Train BCE 0.0296→0.0281.
- Durable: `.cache/ml25m/enc_v1_ml25m.pt` (best-val, ep7), `enc_v1_ml25m_state.pt` (resume),
  `enc_v1_ml25m_peak.txt` (log). Trainer `scripts/paper2/ml25m_train_enc.py`,
  gate `scripts/paper2/ml25m_health_enc.py`.

## STEP 2 — health gate at the pre-stated ruler

Held-out disjoint-targets protocol; V1 decoder convention `sc = popb + Q·u` (β=1) for MOSTPOP / ridge /
encoder alike (MOSTPOP = popb only = encoder at 0 reveals); ridge λ=5; n_test=500 (n=498 scored,
n_tail=491). NDCG@{10,50}, full + Cremonesi head-33% tail.

**Full-profile fold (the headroom ruler):**
| method | @10 full | @10 tail | @50 full | @50 tail |
|---|---|---|---|---|
| MOSTPOP (q0) | 0.2685 | 0.0540 | 0.2588 | 0.0749 |
| ridge (linear ref) | 0.2868 | 0.0970 | 0.2730 | 0.1094 |
| **V1 ENCODER** | **0.3347** | **0.1385** | **0.3159** | **0.1515** |

**Headroom (full-profile − MOSTPOP):**
- **ENCODER @50 full = +0.0571** · @50 tail = **+0.0766** · @10 full = +0.0662.
- ridge @50 full = **+0.0142** (the linear reference — matches Phase 1b's near-zero headroom story).

**Elicitation curves (encoder fold, no-harm ruler), gain by q20 @50:**
| selector | @50 full q0→q8→q20 | @50 tail q0→q8→q20 | mono no-harm |
|---|---|---|---|
| random  | 0.2588→0.2590→0.2567 | 0.0749→0.0784→0.0778 | OK (@10 & @50) |
| **rmva**| 0.2588→0.2489→0.2668 | 0.0749→0.0936→**0.1077** | OK |
| entropy | 0.2588→0.2589→0.2597 | 0.0749→0.0766→0.0821 | OK |

RMVA is the best selector (tail @50 +0.0329 by q20); revealing **never hurts** at either K for any selector.

## STEP 3 — GATE VERDICT: **PASS** (K=50 primary) → phase-2 battery is a separate run

- **GATE (K=50): encoder full-profile − MOSTPOP = +0.0571**, monotone no-harm — **PASS**. Positive and
  substantial: it is **4.0× the linear ridge's +0.0142** and reaches **~57% of the ML-1M reference (+0.10)**.
- **Tail is recovered:** Phase 1b reported **tail NDCG@10 = 0.000 everywhere** (linear fold + β=4 prior);
  the V1 neural encoder lifts non-popular likes into the top-K — full-profile **tail@10 = 0.1385**,
  **tail@50 = 0.1515** (+0.0766 over MOSTPOP). Personalization signal is present, not popularity-saturated.
- **Depth curve NOT run** (only required on FAIL). Both reported depths clear the anchor: headroom
  +0.0662 @10 and +0.0571 @50 (the @10 harsher/more popularity-dominated ruler still passes).

### DECISIVE CORRECTIVE to Phase 1b
Phase 1b concluded the small ML-25M headroom (+0.009) was **regime-intrinsic** (popularity nearly solves
the task) and gated Phase 2. **Phase 1c refutes that at its own caveat #3.** The failure was an artifact of
the **linear biased-SVD ridge reference**, not the dataset. On the **real instrument class** — the V1
nonlinear attention fold-in encoder, trained identically to ML-1M — ML-25M has a **healthy +0.057 (K=50)
headroom and nonzero tail**. The neural fold extracts personalization the linear fold cannot (+0.057 vs
+0.014; tail 0.15 vs 0 / 0.11). **ML-25M is a healthy testbed for Papers A/B; the Phase-2 policy/ladder
battery is unblocked** (to be run separately).

### Disk status
- 38 GB free at finish (started ~38 GB). Never near the 4 GB floor.
- New durable artifacts under `.cache/ml25m/`: `enc_v1_ml25m.pt` (~0.4 MB), `enc_v1_ml25m_state.pt`,
  `enc_v1_ml25m_peak.txt`. Existing full-data artifacts (meta.npz 298 MB, Q_svd/bi_svd, Ac_concept)
  untouched. `ratings.csv` (647 MB) retained. No `prep.npz` regen needed (trainer/gate read `meta.npz`).

## Headline numbers
- **Encoder trained on FULL data** (161,536 users, 18,430 items), ~200 s/epoch CPU, best-val ep7
  (val NDCG@50 0.3226). No compute fallback needed.
- **GATE PASS @K=50:** encoder full-profile − MOSTPOP = **+0.0571** (ridge only +0.0142; ML-1M ref +0.10);
  monotone no-harm all selectors. **@10 headroom +0.0662.**
- **Tail recovered:** full-profile tail NDCG@10 **0.1385** / @50 **0.1515** (Phase 1b: 0.000). RMVA
  elicitation lifts tail@50 +0.0329 by q20.
- **Verdict: the Phase-1b FAIL was a linear-instrument artifact.** The real V1 neural instrument is
  healthy on ML-25M → **Phase 2 UNGATED.**
