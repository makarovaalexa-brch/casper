# INSTRUMENT 2.0 — Phase 4a: the essential policy battery on the certified instrument (ML-1M)

THE thesis question: does the flagship (**continuous + graded + adaptive** elicitation) survive on a
STRONG instrument (RecVAE-d512, EASE-class, +0.244 headroom over MOSTPOP vs V1's +0.098)?

Foreground, chunked, no commits. Runner `scripts/instrument2/p4a_battery.py`
(stages: `static`, `train --tseed {0,1,2} [--field]`, `eval`). Artifacts `.cache/instrument2/p4a_*.pt`,
`.cache/instrument2/p4a_battery.json`.

Fixed protocol (byte-identical to V1 papers): additive operator `z' = z + eta*a*q` (eta=16, P3 winner),
z=0 cold seed, graded answer `a = cos(z*, q)`, `z* = enc(profile-half likes)`. Arena `ml1m_arena`,
NDCG@10 full + Cremonesi tail, T=8 turns, seed-avg {1,2,3,7,11}, te[:300]=VAL / te[300:]=TEST (304 users).

---

## Item 1 — STATIC BASELINES (no training), seed-avg {1,2,3,7,11}, te[300:] TEST

| baseline | full | tail | note |
|---|---|---|---|
| MOSTPOP (popb ranker) | 0.3099 | 0.0808 | V1 ref 0.310 ✓ (fidelity) |
| z=0 cold floor (decoder bias) | 0.1071 | 0.0531 | P2 ref 0.107 ✓ |
| global-entropy top-8 concepts (graded) | 0.0845 | 0.0415 | **fails below floor** — highest decode-entropy tags are diffuse junk ("sexy","pornography","splatter"); naive entropy-over-concepts without answerability is harmful |
| **lift-concepts-8 (graded)** = uent+GRAW analogue | **0.4436** | **0.2349** | per-user top-8 answerable (lift-selected) genome-tag concepts, member-bag directions |
| lift-concepts-8 (BINARY answers) | 0.4350 | 0.2212 | graded − binary = **+0.009 / +0.014** on concepts |
| PCA-of-train-z* basis-8 (graded) | 0.1688 | 0.1574 | **wrong basis** — population-variance PCA captures a dominant popularity axis, not item-discriminative coords; collapses |
| **decoder-SVD basis-8 (graded)** = P3 W1 static bar | **0.4709** | **0.2916** | P3 ref 0.4688 (3-seed) ✓; the STRONG static bar |
| decoder-SVD basis-8 (BINARY answers) | 0.2149 | 0.1734 | graded − binary = **+0.256 / +0.118** — massive inversion |
| **k=8 item-fold reference** (discrete) | **0.4625** | **0.2796** | P2 A.2 ref 0.4625 ✓ |

### Static-bar readings (thesis claims, static-only)
- **Continuous > discrete-static.** The continuous informative basis (decoder-SVD-8, 0.4709/0.2916)
  BEATS the discrete item-8 fold (0.4625/0.2796) by **+0.0084 full / +0.0120 tail**, and beats the
  discrete concept-8 static (lift-concepts, 0.4436/0.2349) by **+0.027 full / +0.057 tail**. Continuous
  geometric queries along an informative basis reconstruct taste better than folding discrete items or
  concepts at equal turn count — the P3 W1 finding holds on the full 5-seed test.
- **Graded >> binary (answer-model inversion).** On the continuous basis the collapse is enormous
  (0.4709 → 0.2149 full, −0.256); even on the discrete concept channel graded edges binary (+0.009).
  The graded geometric answer is load-bearing; binarizing it destroys most of the signal.
- **The basis matters and is non-trivial.** decoder-weight right-singular vectors (item-discriminative)
  work (0.4709); PCA of population z* (0.169) and global-entropy concept ranking (0.084) both fail.
  The informative static design is a specific, defensible construction — exactly the target a learned
  policy must discover.

---

## Item 2 — D1-RECIPE ACTOR (differentiable unroll in latent space)  [pending eval]

Policy MLP(z_t, turn) → unit q_t (512-d); belief update = additive operator; T=8. Objective =
(1−cos(z_T,z*)) + 0.3·softNDCG(approx, teacher = decode(z*) top-20 vs 108 sampled negs) + optional
divisiveness-field term (qᵀ Cov(z*) q). **From-scratch collapsed** (val full ~0.33, tail ~0.10; cos
reconstruction plateaus ~0.60 — SGD does not discover a diverse orthonormal informative basis). Per the
handoff fallback, actors are **BC-warmed from the static decoder-SVD-8 basis** then unroll-fine-tuned
(lr 3e-4, 20 epochs, val-best on SELVAL tail). This isolates adaptivity as *whatever the fine-tune adds
on top of the static basis*. 3 training seeds from scratch of the BC-warm+finetune recipe.

### Actor results — seed-avg over 3 training seeds, eval seed-avg {1,2,3,7,11}, te[300:] TEST

| actor variant | full | tail |
|---|---|---|
| **D1-recipe actor (graded, adaptive)** — seed-avg | **0.4907** (sd .0038) | **0.2982** (sd .0088) |
| per train-seed s0 / s1 / s2 (full) | 0.4901 / 0.4916 / 0.4905 | 0.2973 / 0.2991 / 0.2982 |
| D1-recipe actor (**BINARY** answers sign(cos)) | **0.1166** | **0.1045** |
| +divisiveness-field variant (s0) | 0.4710 | 0.2915 |

From-scratch (no BC-warm) collapsed: val full ~0.33, tail ~0.10. The **divisiveness-field variant did
not improve over BC-warm** (best epoch = the BC-warm init; rewarding high-population-variance directions
`qᵀCov(z*)q` pulls away from taste-reconstruction-optimal directions — an honest negative in the near-
linear regime). Adopted actor = BC-warm-from-SVD-8 + unroll fine-tune, no field.

**q-curve (primary actor, full / tail), turns 1→8:**
| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| full | 0.116 | 0.142 | 0.161 | 0.191 | 0.235 | 0.399 | 0.468 | 0.490 |
| tail | 0.068 | 0.099 | 0.123 | 0.158 | 0.186 | 0.235 | 0.271 | 0.297 |

Monotone; the informative-basis reconstruction front-loads onto turns 6–8 (the actor completes the
orthonormal span). At 8 turns the adaptive actor (0.490) **exceeds the static SVD-8 basis (0.471) and the
k=8 item-fold (0.463)**.

---

## Item 3 — SNAP-LOSS (post-hoc snap each emitted q_t to nearest bank direction)

| condition | full | tail | Δ full | Δ tail |
|---|---|---|---|---|
| unsnapped actor (continuous) | 0.4900 | 0.2973 | — | — |
| snap → nearest **concept** direction (200 genome tags) | 0.3433 | 0.1640 | **−0.147** | **−0.133** |
| snap → nearest **item-decoder-row** direction (3706 items) | 0.3324 | 0.1539 | **−0.158** | **−0.143** |

Snapping the continuous queries onto the discrete concept/item banks is **catastrophic** on I2
(−0.15 full). The actor's emitted directions (near an informative orthonormal basis) are not
well-approximated by any single concept or item direction; the continuous, off-manifold query is
load-bearing. (V1 concept-snap was only −0.037/−0.040 — the continuity premium is **~4× larger** on the
strong instrument.)

---

## Item 4 — STATIC-8 CONTROL (isolating adaptivity value)

Greedy forward selection of 8 fixed directions maximizing VAL full-NDCG, pool = actor per-turn mean
directions + decoder-SVD-8 + PCA-z*-8, applied statically to all TEST users:

| static construction | full | tail |
|---|---|---|
| greedy-static-8 from actor query distribution (val-selected) | 0.4596 | 0.2653 |
| decoder-SVD-8 basis (strongest single static) | 0.4709 | 0.2916 |
| **adaptive actor** | **0.4907** | **0.2982** |

The strongest realizable **single static** = the SVD-8 basis (0.4709). Greedy-static distilled from the
actor's own query distribution recovers only 0.4596 (< SVD-8, overfits the 300-user val cohort), so the
actor's advantage is **not reducible to a fixed better basis** — the per-turn directions genuinely depend
on the belief z_t. **Adaptive − strongest static = +0.0198 full / +0.0066 tail.**

---

## Item 5 — PAIRED per-user bootstrap (304 test users, per-user seed-averaged over 5 seeds)

| margin | full mean Δ | full 95% CI | full p(Δ>0) | tail mean Δ | tail 95% CI | tail p(Δ>0) |
|---|---|---|---|---|---|---|
| actor − strongest static (SVD-8) | **+0.0192** | [+0.009, +0.029] | **0.9998** | +0.0054 | [−0.006, +0.017] | 0.82 |
| actor − discrete concept-8 (lift, GRAW) | **+0.0464** | [+0.033, +0.060] | **1.00** | **+0.0620** | [+0.047, +0.077] | **1.00** |

Adaptivity beats the strong static basis **significantly on full NDCG**, but is a **tie on tail**. The
continuous-adaptive flagship beats the discrete concept elicitation **significantly on both axes**.

---

# VERDICT — does the flagship survive on the strong instrument I2?

Per-claim, with the honest V1-vs-I2 comparison:

| thesis claim | V1 (weak instrument) | I2 (RecVAE-d512, strong) | verdict on I2 |
|---|---|---|---|
| **continuous > discrete-static** | snap-loss −0.037/−0.040; continuous edge small | static basis-8 **0.4709** > item-8 fold 0.4625 (**+0.008/+0.012**); adaptive **0.4907** > concept-8 lift **0.4436** (**+0.046/+0.062**, both sig) | **SURVIVES** — clear, larger on I2 |
| **graded > binary (answer inversion)** | binary ≈ tie (actor 0.356 ≈ casper); win needs graded | actor graded **0.491** vs binary **0.117** (−0.374); static basis graded **0.471** vs binary **0.215** (−0.256) | **SURVIVES — massively** (I2 amplifies) |
| **snap-loss (continuity is load-bearing)** | −0.037 / −0.040 (concept-snap) | −0.147 / −0.133 (concept), −0.158 / −0.143 (item) | **SURVIVES — ~4× larger** |
| **adaptivity > static** | +0.023 full / +0.032 tail | **+0.019 full (sig, p=.9998)** / +0.005 tail (n.s.) | **SURVIVES on full; SHRINKS to a TIE on tail** |

### Bottom line
On a **certified EASE-class instrument** (2.5× the headroom of the V1 encoder), the flagship
**continuous + graded + adaptive** elicitation **survives**:
- The **continuous** channel is not just intact but *more* decisive — snapping to discrete banks costs
  ~0.15 NDCG (≈4× the V1 penalty), and continuous queries beat both discrete item-folds and discrete
  concept elicitation.
- The **graded** geometric answer is essential and its importance is amplified — binarizing collapses the
  actor from 0.491 to 0.117.
- **Adaptivity** is the one claim that **shrinks**: the differentiable-unroll actor beats the strongest
  static basis **significantly on full NDCG (+0.019)** but only **ties on tail (+0.005, n.s.)**. This is
  consistent with the linear-Gaussian intuition (on a strong, near-linear instrument the optimal design
  is close to non-adaptive) and with V1's "policy ≈ static entropy" history — from-scratch policy search
  collapsed and needed BC-warming from the static basis; the divisiveness field added nothing. The
  adaptive edge is real but modest, and it lives on the head, not the tail.

Honesty notes: the actor is **BC-warmed from the static SVD-8 basis** (from-scratch collapsed — a
documented optimization failure, not a claim of impossibility). The "strongest static" bar is the SVD-8
informative basis (0.4709), the strongest *single* realizable static design; a greedy-static distilled
from the actor's own queries is weaker (0.4596), so the +0.019 is genuine adaptive value, not a hidden
static gain. All numbers seed-avg {1,2,3,7,11} on the 304-user te[300:] TEST cohort, byte-identical V1
arena; per-user bootstrap over 304 users.

### Durable artifacts
`.cache/instrument2/p4a_battery.json` (static + eval), `.cache/instrument2/p4a_bootstrap.json` (clean
paired margins), actors `p4a_actor_s{0,1,2}.pt` + `p4a_actor_s0_field.pt`. Scripts
`scripts/instrument2/p4a_battery.py`, `p4a_bootstrap.py`.
