# ORACLE-Z HEADROOM — best-possible belief under the FROZEN RecVAE decoder

**Date:** 2026-07-11 · **Decoder:** frozen RecVAE-d512 (`.cache/instrument2/ml25m_recvae_d512_best.pt`),
score = z·Wᵀ+b, W,b FROZEN. **Users:** 1999 held-out TEST users (disjoint from pilot train+val;
study 173/300 quarantined via trU firewall). **Training-free** per-user optimization of a 512-d latent z*
by Adam through the frozen decoder (300 steps, lr 0.05,
reg 0.001, BPR λ 1.0). NO encoder, NO architecture, NO LLM calls.
All ratings used; users batched for compute only (no data reduction). Raw: `.cache/rung1/oracle_z.json`.

Per user: leak-free disjoint KNOWN / HELD half-split. Fit z* to KNOWN under four conditionings; evaluate
on HELD. 1589 users have KNOWN watched-dislikes (r≤2); 1595 have HELD
dislikes (signed-AUC defined).

## Conditionings
1. **likes-only** — positives = KNOWN liked (r≥4), softmax multinomial CE (RecVAE's own objective).
2. **likes + watched-dislikes** — (1) + BPR ranking liked above KNOWN disliked (r≤2, *inside* the cone).
3. **likes + disinterest** — (1) + BPR ranking liked above UNWATCHED items from AVOIDED genres
   (under-represented vs population; *outside* the cone by construction — cannot drag the likes).
- **clairvoyant** — positives = HELD liked (fit directly to the eval target) = upper-bound ceiling.

## PRIMARY — held-LIKED NDCG@10 (the deployment metric)

| conditioning | NDCG@10 | 95% CI |
|---|---|---|
| native z0 (RecVAE-encode known-likes, no opt) | 0.5166 | [0.5045, 0.5288] |
| (1) likes-only oracle z* | 0.4832 | [0.4714, 0.4953] |
| (2) likes + watched-dislikes | 0.4816 | [0.4698, 0.4936] |
| (3) likes + disinterest | 0.4835 | [0.4715, 0.4956] |
| **clairvoyant ceiling** (fit to held-likes) | 0.9943 | [0.9933, 0.9954] |

### Δ vs likes-only (paired, held-LIKED NDCG@10)

| contrast | Δ | 95% CI | verdict |
|---|---|---|---|
| (2) +watched-dislikes — ALL users (n=1999) | -0.0016 | [-0.0029, -0.0003] | WALL (~0) |
| (2) +watched-dislikes — known-dislike subset (n=1589) | -0.0020 | [-0.0037, -0.0004] | — |
| (3) +disinterest — ALL users (n=1999) | +0.0003 | [-0.0011, +0.0017] | WALL (~0) |

## SECONDARY — signed AUC (held-liked ranked above held-disliked)

| conditioning | AUC | 95% CI |
|---|---|---|
| (1) likes-only | 0.8130 | [0.8046, 0.8214] |
| (2) likes + watched-dislikes | 0.8166 | [0.8081, 0.8249] |
| (3) likes + disinterest | 0.8128 | [0.8043, 0.8211] |

Δ signed AUC vs likes-only: (2) watched-dislikes ALL = +0.0035 | [+0.0028, +0.0043];
(3) disinterest ALL = -0.0003 | [-0.0008, +0.0002].

## Robustness to the manifold-anchor `reg` (the only free knob)
`reg` anchors the fit toward the native RecVAE belief; the verdict must not hinge on it. Primary reg =
0.001. Δ's (held-LIKED NDCG@10, ALL users) and signed-AUC Δ's across the sweep:

| reg | likes NDCG | native | clairvoyant | Δ(2) dislike | Δ(3) disinterest | Δ(2) AUC | Δ(3) AUC |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.4832 | 0.5166 | 0.9943 | -0.0016 | +0.0003 | +0.0035 | -0.0003 |
| 0.01 | 0.5075 | 0.5166 | 0.8475 | -0.0031 | -0.0031 | +0.0060 | +0.0006 |
| 0.03 | 0.5136 | 0.5166 | 0.6947 | -0.0036 | -0.0042 | +0.0056 | +0.0012 |

## Scale / ceiling
- clairvoyant − likes-only = **+0.5111** CI[+0.4993, +0.5230] — total headroom that ANY
  belief could capture if it knew the held target. This is how much room exists at all under this decoder.
- oracle-likes − native = -0.0334 | [-0.0389, -0.0281] — the lift from optimizing z past the
  raw RecVAE encoding of the known likes (sanity: the oracle beats the native belief).
- mean ‖z*(2)−z*(1)‖ = 1.205; ‖z*(3)−z*(1)‖ =
  1.163 — the negative signals DID move the belief.

## Verdict
The **clairvoyant ceiling is +0.511** NDCG@10: there is real headroom for a better belief under the
frozen decoder. Against that ceiling:
- **Watched-dislike (inside the cone):** Δ = -0.0016 (ALL). The frozen decoder is the WALL for watched-dislike: even the best-possible z fit with correctly-signed dislike targets does not improve held-liked ranking. Consistent with RUNG1_CHECKS (like/dislike geometry conflated).
- **Disinterest (outside the cone):** Δ = +0.0003 (ALL). Disinterest does not clear the wall either under the frozen decoder.

**Hypothesis (3)>(2)≈0:** NOT confirmed as stated — see the Δ table above; neither negative clears the wall.
