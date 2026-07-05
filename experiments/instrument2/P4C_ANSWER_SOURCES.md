# INSTRUMENT 2.0 — Phase 4c: the ANSWER-SOURCE ablation (closing the two non-human holes)

P4a proved the flagship (**continuous + graded + adaptive**) on the certified RecVAE-d512
instrument, but with a **perfect geometric answerer** `a = cos(z*, q)` — a noiseless oracle that
uses the *same* encoder geometry the instrument folds with. Two holes remained, both closeable
without humans:

- **H1 (imperfect answerer):** does the flagship survive a **realistic answer channel** fitted on
  real ML-1M ratings (quantization + noise + user bias)? → **PART 1**
- **H2 (u\* circularity):** do the claims survive answers from a **foreign geometry** (V1 encoder /
  EASE) → **PART 2**, and from **raw observed data** with no model in the answer path (which of two
  items did the user actually rate higher) → **PART 3**?

Protocol: frozen `ml1m_recvae_d512_best.pt`; operator `z' = z + η·a·q`, η=16, z₀=0; `ml1m_arena`
byte-identical V1 splits; eval seeds {1,2,3,7,11}, te[300:] TEST (304 users); reuses the P4a actors
`p4a_actor_s{0,1,2}.pt`. Runner `scripts/instrument2/p4c_answer_sources.py`
(`channel|part1|trainnoisy|part2|part3`), artifacts `.cache/instrument2/p4c_*.json`.

### Scale-matching (load-bearing, honest)
The native geometric answer is *literally* `a = cos(z*,q) = s`. A foreign/empirical channel maps to
a different raw scale (rating→[-1,1]); with a **fixed** operator step η this would collapse the arm
purely from step-size miscalibration, not information loss. We therefore rescale each foreign/
empirical answer stream by **one global scalar** so its cohort RMS matches the native s-RMS
(≡ choosing η for the channel), isolating the answer's **information content** from its raw scale.
We also **center** the empirical answer on the **user's own profile-mean rating** (an observable,
non-circular baseline) — the additive operator needs a *signed* preference, and absolute ML-1M
ratings carry a +0.29 positive offset (mean 3.58) that the operator is not built for. Per-user
centering realizes the fitted **user-bias** law ("did you rate this above/below *your* average").

---

## PART 1a — the fitted empirical answer channel  P(rating | s),  s = cos(z\*, d_i)

Fit on **all TRAIN users' real ratings** (809k user–item pairs, 4,832 users), z\* = enc(user likes),
d_i = unit decoder row. s binned into 12 quantile bins.

| statistic | value |
|---|---|
| **corr(s, rating)** (effective signal) | **0.516** — moderate; s is far from a perfect predictor of rating |
| mean per-bin rating entropy | **1.887 bits** (max log₂5 = 2.322) — given s, the rating is still very uncertain |
| global mean rating | 3.581 (→ +0.29 raw answer offset) |
| per-user mean rating (mean ± sd) | 3.70 ± 0.42 (the user-bias term) |
| per-user dispersion (mean ± sd) | 1.00 ± 0.20 |
| bin mean rating, low-s → high-s (12 bins) | 2.36, 2.87, 3.13, 3.31, 3.46, 3.60, 3.72, 3.84, 3.96, 4.09, 4.22, 4.42 |

**Reading.** The channel is **monotone** (mean rating climbs 2.36→4.42 with s) but **noisy**: each s
bin still spreads ~1.9 bits over the 5 rating levels, and the overall s↔rating correlation is only
0.52. The effective linear response is `a ≈ 1.55·s − 0.36` with channel residual σ_a = 0.48. This is
a genuinely lossy answerer, not the P4a oracle.

---

## PART 1b — key arms under the SAMPLED empirical channel

Full 5-seed TEST, scale-matched, per-user centered, one sampled answer per question (realistic).
`item8_fold` = pure fold-in (no answer channel); `item8_realrating` = 8 item questions answered by
the user's **actual rating** through the operator (fully-real answer path, no sampling — exact).

| arm | P4a noiseless | noise ×0.5 | **noise ×1** | noise ×2 |
|---|---|---|---|---|
| continuous **actor** (seed-avg) | 0.4907 / 0.2982 | 0.2220 / 0.1217 | **0.1502 / 0.0870** | 0.0797 / 0.0494 |
| continuous **SVD-8 static** | 0.4709 / 0.2916 | 0.2307 / 0.1104 | **0.1594 / 0.0779** | 0.0903 / 0.0519 |
| discrete **concept-8 (lift)** | 0.4436 / 0.2349 | 0.3676 / 0.1941 | **0.3250 / 0.1695** | 0.2705 / 0.1425 |
| **item-8 real-rating** (operator, exact) | — | 0.3935 / 0.2383 | **0.3935 / 0.2383** | 0.3935 / 0.2383 |
| **item-8 fold** (pure fold-in) | 0.4625 / 0.2796 | 0.4635 / 0.2731 | **0.4635 / 0.2731** | 0.4635 / 0.2731 |
| *(MOSTPOP anchor)* | 0.3099 | — | 0.3099 | — |

**Noise decomposition on the actor (seed-1, isolating each loss):**
native exact 0.496 → **linear-response** (imperfect s↔a, corr .52) 0.389 → **+ quantization** to 5
levels 0.293 → **+ per-answer sampling noise** (1.9-bit channel) 0.172. Each realistic degradation
step costs the continuous reconstruction ~0.10 NDCG.

**Graded vs binary under the channel (noise ×1):** actor graded 0.150 vs binary 0.145 (**gap gone**);
SVD-8 graded 0.159 vs binary 0.114 (+0.045). The P4a "graded ≫ binary" (−0.374 on the actor) **does
not survive** — once answers are a noisy 5-level reply, the graded advantage on continuous queries
evaporates.

**Snap-loss on the noisy-channel actor (noise ×1):** unsnapped 0.150; concept-snap 0.062
(**−0.088**); **item-snap 0.201 (+0.051 full!)**. Under noise the continuity premium **reverses** —
snapping the continuous query to the nearest real item *helps* on full NDCG (vs P4a where item-snap
cost −0.158).

**Train-noisy vs eval-noisy (the adaptation test).** Retrain seed-0 with the channel's effective law
in the unroll (`a = 1.55s − 0.36 + N(0,0.48)`, noise stop-grad), select on eval-noisy val, evaluate
under the sampled channel:

| actor | eval under sampled empirical channel (noise ×1) |
|---|---|
| BC-warm (trained noiseless), eval-noisy | 0.1486 / 0.0846 |
| **train-noisy (adapted)** | **0.2874 / 0.1038** |

Training under the noisy channel **nearly doubles** full NDCG (+0.139) — the collapse is largely an
off-distribution artifact of the noiseless-trained policy. But even the adapted continuous actor
(0.287) **still loses** to discrete concept-8 (0.325), item-8 real-rating (0.394) and item-8 fold
(0.463) under the same realistic answers.

### PART 1 claim table — does each thesis claim survive a realistic answer channel?

| thesis claim | P4a (perfect answerer) | P4c (empirical channel, noise ×1) | verdict |
|---|---|---|---|
| **continuous > discrete** | actor 0.491 > item-8 0.463 > concept-8 0.444 | actor 0.150 (adapted 0.287) **≪** concept-8 0.325 **≪** item-8 0.463 — ordering **inverts** | **DOES NOT SURVIVE** — continuous is the *most* answer-fragile channel; discrete item/concept win under noise |
| **graded > binary** | −0.374 (actor) | actor +0.005 (gone); SVD-8 +0.045 (small) | **DOES NOT SURVIVE** on the actor; marginal on the static basis |
| **snap-loss (continuity load-bearing)** | −0.147 concept / −0.158 item | −0.088 concept / **+0.051** item | **REVERSES** — under noise, snapping to a real item *helps* |
| **adaptive > static** | +0.019 (actor − SVD-8) | actor 0.150 ≈ SVD-8 0.159 (tie/slightly worse); adapted actor 0.287 ≫ SVD-8 0.159 | **SURVIVES only after retraining under the channel** (adapted +0.128) — not zero-shot |

---

## PART 2 — foreign-geometry answers (H2: break the u\* circularity)

Answers computed from a geometry the I2 instrument **never saw**, while I2 still folds/ranks.
(a) **V1** = the old `enc_concept` 64-d encoder: item answer = cos(u\*_V1, Ql[j]); concept answer =
cos(u\*_V1, V1-concept-dir); continuous arms **item-realized** (snap q → nearest I2 item row, answer
that item in V1). (b) **EASE** = answer is the item's EASE-score percentile for the user (item
questions only). Scale-matched, 5-seed TEST.

| arm | full | tail | note |
|---|---|---|---|
| **V1 item-8** | **0.3916** | 0.2376 | item questions, foreign V1 answers — **works** |
| **V1 concept-8** | **0.3956** | 0.2077 | concept questions, foreign V1 answers — **works** |
| V1 basis-8 (item-realized) | 0.1179 | 0.0668 | static basis → 8 *global* items = non-personalized questionnaire (weak by construction) |
| V1 actor (item-realized) | 0.2386 | 0.1121 | per-user item-realized continuous queries, foreign answers — moderate |
| **EASE item-8** | **0.4393** | **0.2706** | item questions, EASE answers — **strong** (≈ native item-8 fold 0.463) |
| EASE basis-8 (item-realized) | 0.0684 | 0.0636 | global-item artifact (as above) |
| EASE actor (item-realized) | 0.0966 | 0.0722 | continuous item-realized under EASE — collapses |

**Reading.** The **discrete-question channel rankings persist under foreign answers**: item- and
concept-questions answered from a foreign encoder (V1: 0.392/0.396) or from EASE (item 0.439) remain
strong and comparable to native — the instrument's ability to fold a discrete answer is **not
circular**, it works no matter which geometry produced the like/dislike signal. The **continuous /
item-realized** arms stay weak (0.10–0.24), matching Part 1: the continuous advantage is the part
that depends on a high-fidelity same-geometry answer, and it does not transfer.

---

## PART 3 — raw-data PAIR answers (zero circularity, no model in the answer path)

The actor's continuous query q_t is **realized as a pair** (i,j) and answered *only* by the user's
observed ratings: `a = (rating_i − rating_j)/4`. **Profile-restricted**: the pair is chosen among the
user's revealed (profile) items — guarantees both are rated (coverage = 1, no target leak).
**Global**: pair chosen from the whole catalogue via PAIRSNAP top-K (K=20), answerable only if both
items happen to be in the profile.

| arm | full | tail | coverage | note |
|---|---|---|---|---|
| **pair-raw, profile-restricted** | **0.2229** | 0.0927 | 1.000 | continuous interview, **real rating comparisons only** — works (≫ cold floor 0.107) |
| pair-geom, profile-restricted | 0.3290 | 0.1327 | 1.000 | same pairs, geometric answer (upper bound for pair realization) |
| pair-raw, global snap | 0.1086 | 0.0525 | **0.003** | actor's free queries almost never land on two rated items → ≈ cold floor |
| SVD-8 static native (ref) | 0.4709 | 0.2916 | — | noiseless reference |

**Reading.** The pair-realized continuous interview **works with real preference comparisons and no
model in the answer path** (0.223, well above the z₀ cold floor 0.107) — *provided the pair is drawn
from items the user has actually rated* (profile-restricted, coverage 1). Let the actor query freely
over the catalogue and coverage falls to **0.3%** — the continuous policy's off-manifold directions
rarely correspond to two items any given user has rated, so raw pairwise answering does nothing. Raw
pairs also trail their geometric-answer twin (0.223 vs 0.329): a single ordinal comparison carries
less information than a graded cosine.

---

# THREE VERDICTS

**V1 — Empirical channel (H1): the flagship's *continuity* claim does NOT survive realistic answer
noise; robustness lives in discrete item/concept questions.**
Fitting the real ML-1M answer law (corr(s,rating)=0.52, 1.9-bit conditional entropy) and sampling
answers **inverts the P4a ordering**: the continuous actor falls from 0.491 to 0.150 (below MOSTPOP
0.31), while discrete concept-8 (0.325), real-rating item questions (0.394) and item-fold (0.463) are
far more robust. Training the policy *under* the noisy channel recovers the actor to 0.287 (+0.139,
adaptivity survives *conditional on retraining*), but it still loses to the discrete arms. "Graded ≫
binary" collapses to a tie on the actor, and snap-loss **reverses** (item-snap now +0.051). Honest
bottom line: on a strong instrument with a *realistic* answerer, the winning configuration is
**discrete answerable questions (items/concepts) with graded real answers**, not off-manifold
continuous queries.

**V2 — Foreign geometry (H2, circularity): the discrete-question result is NOT circular.**
Item- and concept-questions answered from a foreign encoder (V1: 0.392 / 0.396) or from EASE (item
0.439 ≈ native 0.463) remain strong — the instrument folds a like/dislike signal regardless of which
model produced it. Channel rankings (item ≈ concept, both ≫ continuous-realized) persist. The only
thing that fails to transfer is the continuous/item-realized advantage — the same fragility as V1.

**V3 — Raw-data pairs (zero circularity): the continuous interview works on real comparisons, but
only over answerable (rated) items.**
Pair-realized with real "which did you rate higher" answers, the interview reaches 0.223 (≫ cold
floor 0.107) when restricted to the user's rated items (coverage 1); its geometric-answer twin caps
at 0.329. Unrestricted, coverage is 0.3% and it does nothing — **answerability, not raw geometry, is
the binding constraint**. No model anywhere in the answer path.

### Key numbers
- Channel: corr(s,rating) **0.516**, mean bin entropy **1.887 bits**, response `a≈1.55s−0.36`, σ=0.48.
- Empirical (noise×1): actor **0.150** / SVD-8 **0.159** / concept-8 **0.325** / item-real **0.394** /
  item-fold **0.463**; train-noisy actor **0.287**; graded−binary(actor) **+0.005**; item-snap **+0.051**.
- Foreign: V1 item **0.392**, V1 concept **0.396**, EASE item **0.439**; continuous-realized 0.07–0.24.
- Raw pairs: profile-restricted **0.223** (cov 1.0), geom-pair 0.329, global-raw 0.109 (**cov 0.003**).

### Durable artifacts
`.cache/instrument2/p4c_channel.json` (fitted channel), `.cache/instrument2/p4c_answer_sources.json`
(part1/part2/part3), `.cache/instrument2/p4c_actor_noisy_s0.pt` (train-noisy actor). Script
`scripts/instrument2/p4c_answer_sources.py`.
