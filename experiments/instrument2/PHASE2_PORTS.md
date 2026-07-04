# INSTRUMENT 2.0 — Phase 2 (CORE CUT): port replicated RecVAE to ML-1M + Goodreads

Status: IN PROGRESS (2026-07-04). FOREGROUND, chunked, no commits. Model = the Phase-1
replicated RecVAE recipe (composite prior, per-user beta=gamma*|X_u| gamma=0.005, 3:1
alternating enc:dec with encoder-only denoising dropout 0.5, Adam lr 5e-4, batch 500).
Carry-forward rules honoured: seed empty state with z=0 (never empty encoder input);
belief = amortized encoder pass ONLY (no per-user optimization); concepts/dislikes are Phase-3.

Scripts: `scripts/instrument2/ml1m_arena.py` (shared arena+metric), `ml1m_recvae.py`,
`ml1m_bars.py` (Part A); `gr_recvae.py` (Part B). Durable ckpts `.cache/instrument2/{ml1m,gr}_recvae_d*.pt`
(+ `_best.pt`, `_peak.txt`, `_TEST.json`).

---

## PART A — ML-1M (canonical thesis arena)

### A.0 Protocol mapping (PRE-STATED, before any RecVAE number)

Mirrored VERBATIM from the V1 harness `scripts/paper2/continuous_policy2_st.py` so the
numbers are directly comparable to the V1 papers:

- **Items** = `np.unique(movieId)` over `ml-1m/ratings.dat` -> **ni=3706**, same ordinal ids as V1.
- **Users** = 6040. **like** (implicit positive) = **rating >= 4** (the V1 like definition).
- **User split:** keep users with >=5 likes; `rng(0)` shuffle; **trU = keep[:80%] (4827 users)**,
  **te = keep[90%:]**. te usable (>=6 rated items under the profile-split): **te[:300] = VAL**,
  **te[300:] = TEST (304 users)** — disjoint.
- **Profile-split (held-out disjoint targets):** per user, shuffle ALL rated items, first half =
  profile (fold-in + candidate-exclusion), second half held out; **tlike = held-out items rating>=4**
  = the disjoint targets. FULL-catalogue ranking, profile items excluded, **NDCG@10**; tail =
  Cremonesi head-33% masked. **Seed-averaged over {1,2,3,7,11}** (the V1 paper protocol).
- **RecVAE mapping:** input vocab = ni=3706 (V1 ordinal ids); train matrix = trU x ni binarized
  likes; "full-profile fold" folds the profile-half LIKES (VAE lineage is implicit-positive, so
  profile-half dislikes are dropped; candidate-exclusion still removes ALL profile items, matching V1);
  score = decoder logits; k=0 -> z=prior-mean(0). Early-stop on val full-profile NDCG@10 full.

**Arena FIDELITY CHECK (validates byte-identity to the V1 papers):** seed-avg{1,2,3,7,11} on te[300:]:
- MOSTPOP q0 (popb ranker) = **0.3099 full / 0.0808 tail**  — V1 ref **0.310**. ✓
- V1 encoder full-profile = **0.4074 full / 0.2164 tail** — V1 ref **0.407 / 0.216**. ✓ (exact to 3dp)

### A.1 Honest bars on the identical arena (seed-avg{1,2,3,7,11}, te[300:] test, NDCG@10)

| model | full | tail | headroom over MOSTPOP (full) |
|---|---|---|---|
| MOSTPOP (popb) | 0.3099 | 0.0808 | — |
| V1 encoder (enc_concept, cited) | 0.4074 | 0.2164 | +0.0975 |
| iALS (d=64, Hu2008, fold-in) | 0.4151 | 0.3158 | +0.1052 |
| **EASE (Steck2019, lam=1000 val-sel)** | **0.5549** | **0.3774** | **+0.2450** |
| **RecVAE d=64** | (fill) | (fill) | (fill) |
| **RecVAE d=128** | **0.5361** | **0.3612** | **+0.2261** |
| **RecVAE d=256** | (fill) | (fill) | (fill) |

EASE lam sweep on val (te[:300]): 1->0.325, 10->0.398, 100->0.462, 500->0.488, **1000->0.4886**,
2000->0.487, 5000->0.472 -> val-selected lam=1000 (clean interior peak).

### A.2 k-fold curve at best d (RecVAE d=128; seed-avg{1,2,3,7,11}, NDCG@10)

| k | 0 (z=0) | 1 | 2 | 4 | 8 | full-profile |
|---|---|---|---|---|---|---|
| full | 0.0501 | 0.2891 | 0.3256 | 0.3767 | 0.4340 | 0.5361 |
| tail | 0.0366 | 0.1316 | 0.1634 | 0.2051 | 0.2561 | 0.3612 |

Monotone, healthy, big in-budget gains. NOTE: the k=0 z=0 floor (0.050) sits BELOW MOSTPOP
(0.310) — unlike ML-20M where decode(0)~popularity; on ML-1M the RecVAE decoder bias is not
popularity-shaped, so the cold turn-0 must be seeded but recovers steeply by k=1 (0.289).

### A.3 ACCEPTANCE (Part A): RecVAE >= EASE and >= V1 on full-profile + healthy k-curve
(PENDING best-d — d=128 already >> V1 (+0.129) and iALS, marginally below EASE (-0.019); see verdict.)

---

## PART B — Goodreads composite (the rebuild's raison d'être)

### B.0 Protocol mapping (PRE-STATED)  — see `gr_recvae.py` docstring for the full statement
- Arena `base_comp.npz` (ni=188,867, nu=768,746, 46.1M ratings). like = rating>=4 (drops dislikes;
  Phase-3 handles polarity, consistent with the implicit VAE lineage).
- **RESTRICTED UNIVERSE = top-N=20,000 items by train-like count (76.8% like-mass)** — identical to
  GOODREADS_EASE_DIAGNOSTIC so RecVAE<->EASE is apples-to-apples on the MATCHED universe (the gate arena).
  Full-catalogue RecVAE over 188k items is CPU-infeasible on this box; restricted universe = the gate.
- **Compute-budget deviation (stated):** RecVAE trains on a fixed rng(0) NSUB=80k-train-user subsample
  (mirrors phase-1E's rotating-window V1 training), resumable/chunked. Eval: va=500 val / te=500 test,
  rng(123) held-out disjoint targets, NDCG@{10,510}, profile excluded, fold-in = profile likes in universe.
- **Bars (cited):** MOSTPOP @510 full = 0.2650 (full-cat) / restricted-universe MOSTPOP recomputed here;
  V1-style encoder headroom +0.011; **EASE headroom +0.3037 @510** on the restricted universe (MOSTPOP
  restricted 0.2921 -> EASE 0.5958).
- **THE GATE:** RecVAE full-profile headroom over MOSTPOP (matched universe) **>= 0.8 x EASE headroom
  (+0.3037)** = **+0.243**; band target. Anything **>= +0.15** = rebuild success vs V1's +0.011.

### B.1 Results
(PENDING — training in progress.)

---

## Final verdicts
(PENDING)
