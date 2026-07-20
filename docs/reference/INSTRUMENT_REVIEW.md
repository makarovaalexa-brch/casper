# INSTRUMENT DESIGN REVIEW — strong vs. dislike-expressing (2026-07-11)

Grounded in the actual code + cached result JSONs (not memory). All numbers cited to a file. Read-only.

## COMPARISON TABLE

| Axis | **Paper A — DualHeadSetEncoder ranking** (`instrument_ml1m_rank`) | **Biased-SVD signed** (CASPER-U / calibrated) | **RecVAE** (Instrument 2.0) |
|---|---|---|---|
| Code | `scripts/paper1/train_instrument_rank.py` + `synthetic_sanity.py:84` (`DualHeadSetEncoder`) | `scripts/paper1/instrument_svd.py` (+ `full_breakdown.py`, `mf_foldin.py`) | `scripts/instrument2/recvae.py` (+ `gr_recvae.py`, `ml1m_recvae.py`, `ml25m_recvae.py`) |
| Architecture | Transformer **set-encoder** over (item,polarity) tokens → CLS pool → 2 MLP heads (liked, rated) over all items. `d_model=128, n_heads=4, n_layers=2` (`synthetic_sanity.py:88-112`) | **Biased matrix factorization** (Koren SVD, SGD). Score = `beta*popbias_j + q_j·u`; fold-in `u=(QsᵀQs+λI)⁻¹Qsᵀ(r−μ−b)` (`instrument_svd.py:34-63`) | **Denoising multinomial VAE**. Dense-connected swish encoder (5×600, LayerNorm eps .1) → latent d=200/256/512; decoder = single Linear(d→n_items) softmax. Composite prior (std + prev-posterior + wide-uniform) (`recvae.py:43-124`) |
| **LOSS** | **Listwise softmax RANKING** on liked head (`liked_rank_loss`, `train_instrument_rank.py:58-65`) + BCE on rated/answerability head (`:123`) | **RMSE** on explicit 1–5 ratings via SGD (`instrument_svd.py:46-50`); ridge MAP at fold-in | **Multinomial log-likelihood** `(log_softmax(x_pred)·x).sum()` − β·KL (`recvae.py:116-119`) |
| **TRAINING DATA** | **FULL profiles**, sampled into partial reveal-subsets per step (`build_batch`, `:32-55`); supervised on held-out target likes. NOT elicitation-only. | **FULL** explicit rating matrix of all train users (`:34-53`) | **FULL** binarized like matrix (r≥4), denoised by dropout (`gr_recvae.py:71-91`; `recvae.py:126-140`) |
| **STRENGTH (ML)** | stratified NDCG@10 0.150→**0.272**; **ml1m 0.002→0.003 (worse than random)** (`train_instrument_rank.py:147`) | ~MOSTPOP + tiny residual; captured **~4% of EASE's Goodreads prize** (`INSTRUMENT_REQUIREMENTS.md:11`; `PHASE2_PORTS.md:137,151`) | ML-1M @510 **0.5541** (MOSTPOP 0.310) `ml1m_recvae_d512_TEST.json`; ML-20M NDCG@100 **0.4346** `recvae_ml20m_TEST.json`; ML-25M **0.4998** |
| **STRENGTH (Goodreads)** | not run on GR (weak on ml1m already) | V1/pre-VAE headroom **+0.011 = ~4% of EASE** (`PHASE2_PORTS.md:94,150`) | GR @510 **0.5290**, headroom **+0.2369 = 0.78× EASE = 21.6× V1** (`gr_recvae_d512_TEST.json`; `PHASE2_PORTS.md:132-151`) |
| **DISLIKE / SIGNED INPUT** | **YES but binary** — `pol_emb=Embedding(2,d)`, token = item + polarity; `bp=int(rating≥0.5)` = like/dislike (`synthetic_sanity.py:92,108`; `train_instrument_rank.py:48`). Signed-binary, NOT graded. | **YES, native, graded/continuous** — fold-in target `y = r−μ−b_i` ∈ signed reals; taste-residual sign carries like/dislike; passes G7 polarity gate (`instrument_svd.py:80,104`; `INSTRUMENT_REQUIREMENTS.md:6-9`) | **NO — positive-only** input. `user_ratings` are nonneg counts; drops dislikes at r≥4 binarize (`gr_recvae.py:34,62`). "nonneg multinomial cannot express dislike" (commit b67912e; `PHASE16_MICRO.md:64,141`) |

---

## DIAGNOSES

### A. Why is RecVAE strong and Paper A's ranking instrument weak (esp. large catalogs)?
**It is the OBJECTIVE + CAPACITY, NOT the training-data scope.** The brief's prime suspect (elicitation-only
vs full-matrix training) is **refuted by the code**: all three train on the FULL interaction matrix —
`train_instrument_rank.py:build_batch` samples reveal-subsets *from full profiles*, `instrument_svd.py:34-53`
trains on all train ratings, `gr_recvae.py:71-91` builds the full like matrix. None is elicitation-only.

The real gap:
- **Likelihood.** RecVAE reconstructs the *entire* consumption vector via a multinomial over all items
  (`recvae.py:116`) — dense, every-item gradient per user, the top-N-optimal likelihood (Liang'18). The
  ranking head's listwise softmax puts gradient only on held-out **positive** targets normalized over
  non-revealed items (`train_instrument_rank.py:58-65`) — sparse, noisy supervision that does not scale to
  large vocabularies (ml1m 3706 items → **0.003, worse than random**, `:147`).
- **Bottleneck.** Paper A squeezes everything through one CLS vector → MLP → n_items; RecVAE's high-capacity
  dense-residual encoder + denoising prior learns richer per-item co-occurrence.
- **Linearity for the SVD.** Biased-SVD is linear (pop floor + q·u); on a 188k-item long-tail catalog it
  captures ~4% of EASE's prize (`PHASE2_PORTS.md:150-151`).

**Verdict: strength source = full-matrix MULTINOMIAL-DENOISING reconstruction + capacity; the weak pair fail
on the objective (sparse listwise / linear), not on data access.**

### B. Is RecVAE's strength fundamentally POSITIVE-ONLY, or trainable on a signed objective?
**Partly entangled, not fundamentally.** The multinomial likelihood is *structurally* nonneg: `x` are counts,
`log_softmax·x` is a distribution over consumed items, and `kl_weight = gamma*x.sum()` (`recvae.py:112,116`)
all break under negative "ratings" (a multinomial has no negative counts). So **sign cannot enter through the
input/likelihood** as built. BUT:
- The **architecture** (denoising VAE, capacity, full-matrix) is not inherently positive-only — it could carry
  a Gaussian/ordinal/graded head. The known risk: Gaussian/logistic likelihoods are weaker than multinomial
  for top-N (Liang'18) — flagged in `INSTRUMENT_REQUIREMENTS.md:70-71`.
- The frozen **z-space already contains a usable valence direction**: `recvae_valence_probe.json` verdict
  **YES**, dislike suppresses a genre by **−17.8 pct-pts** (symmetric with +17.7 elevate, 8/8 genres). So the
  *geometry* can down-rank on dislike; what is missing is a **gradient/input path** to drive it.
- When sign IS bolted on as a learned tag→z channel it comes out **INERT**: value-zeroing ΔNDCG = **0.000**,
  flipping loved→hated doesn't move the region (`INSTRUMENT_REQUIREMENTS.md:16-18`; `DESIGN_SHEET_RUNG1_ENCODER.md:19`).

**Verdict: the strength is not metaphysically positive-only, but the specific likelihood that DELIVERS the
strength is; sign must ride a different head, and every bolt-on so far lacked supervision → went inert.**

### C. Did Paper A's ranking instrument express dislike, or was it likes-only?
**It expressed dislike — but only as a BINARY polarity token, not graded.** `pol_emb = Embedding(2, d_model)`
and each reveal token is `item_emb + pol_emb` with `pol = 1` (like) / `0` (dislike) (`synthetic_sanity.py:92,108`;
`train_instrument_rank.py:48`). So it is a **signed-input** model (2 levels), NOT likes-only. The *graded*,
continuously-signed instrument is the biased-SVD (`y=r−μ−b`), which is the one that passes the G7 polarity gate.
Paper A is signed-but-binary **and** weak; the SVD is signed-graded **and** weak; RecVAE is unsigned **and** strong.

### D. Has any repo attempt trained a STRONG full-matrix model with a SIGNED/ranking objective that expresses dislike?
**No — strength and dislike-expression have never robustly coexisted. Two concrete dead ends:**
1. **Explicit/signed weights in strong full-matrix CF HURT** (`RAWDATA_DISLIKE.md` Q3): item-item CF with
   consumed=1 (IMPLICIT) scores **~0.358** held-liked NDCG@10; centered/signed weights (EXPLICIT_CENT)
   **~0.269 (−33%)**. Down-weighting a dislike's neighbors pushes away *liked* thematically-adjacent items,
   because dislikes sit inside the like manifold (a dislike's closest liked item content-sim **0.79**).
2. **RecVAE + bolted value channel** (FOLD-V3, commit 58ff224): the two-channel consumption+value instrument
   passes G7 polarity (+0.020/+0.032) and clean profile beats native RecVAE (+0.017), but the value channel is
   **weak** and the input-space concept/dislike folding **fails** — dislike only works via crude z-space
   subtraction (−15 pts), never through the interaction vector (`PHASE16_MICRO.md:141`). And the aggregate
   arena still could not demonstrate adaptivity on it.

### E. Honest CONSTRAINT MAP
**The tension does NOT live in the data.** `RAWDATA_DISLIKE.md` proves dislike is **separable and generalizes
MORE strongly than like** in raw ML-25M: pooled corr(rating(X), region-avg) = **+0.46**; NEG-tail region
**−0.61** vs POS +0.21; a disliked item's neighbors are 2.6× enriched in the user's other dislikes (48.6% like-
share vs 80% baseline). The signal is real and recoverable (the 0.75–0.78 conflation was a *frozen*-RecVAE
artifact, not a data ceiling).

**The tension lives in the OBJECTIVE:**
- The objectives that make a model STRONG (multinomial top-N likelihood; implicit consumption) are structurally
  **sign-free / nonneg** → sign has no gradient path → any bolt-on goes **inert** (RecVAE, ΔNDCG 0.000).
- The one time sign WAS fed to a strong full-matrix CF, it **hurt** (explicit 0.269 < implicit 0.358), because
  dislikes are geometrically inside the like manifold; naive negative weight in a *like-ranker* is self-defeating.
- Dislike therefore must act as **REPULSION in a taste-structured latent** (where MF/content — not raw
  co-occurrence — reveals it; `RAWDATA_DISLIKE.md` Q2: co-rating cosine hides it ~0, content/MF reveal it),
  not as a negative weight in a positive likelihood.

**So the box (from `INSTRUMENT_REQUIREMENTS.md:27-49`, R1–R8) is:** a nonlinear, continuous-latent model with a
posterior over z, a likelihood strong for top-N that *also* admits sign (ordinal/graded, not multinomial), and
supervision that gives the sign channel gradient so it doesn't inertify. No architecture tried so far occupies
that cell — every one is either strong-and-sign-free or sign-aware-and-weak.

---

**SHARPEST FACT.** The strength-vs-dislike tension is not in the data (dislike is separable and generalizes
*more strongly* than like: pooled r=+0.46, NEG-tail region −0.61 vs POS +0.21) — it is in the **likelihood**:
the multinomial/implicit objective that makes RecVAE and EASE strong is structurally nonneg, so sign has no
gradient path and goes inert (value-zeroing ΔNDCG = 0.000), while the single time sign was fed to a strong
full-matrix CF it *hurt* (explicit 0.269 vs implicit 0.358) because dislikes live inside the like manifold and
must act as repulsion in a taste latent, not as negative weight in a like-ranker.
