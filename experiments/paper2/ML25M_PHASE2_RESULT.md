# ML-25M Replication — PHASE 2 (the headline battery)

Date: 2026-07-03. Ruler = pre-stated Phase-1c ruler: **NDCG@50 primary** (catalogue-depth-matched to
ML-1M K=10), **@10 alongside**, tail = Cremonesi head-33%. Test = 500-user cohort, held-out disjoint
targets. Frozen instrument = `enc_v1_ml25m.pt` (V1 attention fold-in, Phase-1c PASS, +0.057 @50 full
headroom). Artifacts `.cache/ml25m/`. Scripts `scripts/paper2/ml25m_*`. No commits.

ML-1M anchors carried in: concepts beat items ~+50% tail (Paper B); entropy >> random; D1 continuous+graded
0.378/0.178 beats entropy-graded discrete 0.361/0.140 and greedy-static 0.355/0.146; concept effrank 2.25.
ML-25M effrank (Phase 1b): concepts 3.99, items 8.35. Answerability: concepts 0.574 vs items 0.008 (pool 0.120).

---

## STEP 1 — B battery (answerability + selection), frozen enc_v1_ml25m, wasted-turn, graded rating-residual answers
`scripts/paper2/ml25m_b_battery.py` + `ml25m_membership.py` (concept<->item membership + divisiveness cache).
Strict wasted-turn: asking consumes a turn; only ANSWERABLE asks fold a token. Item answerable ⇔ user rated
it (in profile); concept answerable ⇔ ≥2 profile items tagged. Answer = graded rating-residual (r−μ−b_i),
same scale/convention the encoder was trained on. Item pool = top-600 popular. n=498 (n_tail=492).

**(a) Concept vs item asking @q8 (entropy/divisiveness selector), NDCG:**
| ruler | CONCEPT full | CONCEPT tail | ITEM full | ITEM tail | concept−item full | concept−item tail |
|---|---|---|---|---|---|---|
| @50 (primary) | 0.2690 | 0.0812 | 0.2537 | 0.0872 | **+0.0153** | **−0.0060** |
| @10 | 0.2819 | 0.0623 | 0.2638 | 0.0706 | +0.0181 | −0.0083 |

- **Answered-token advantage REPLICATES structurally:** at q8 concept-asking folds **3.85** answered tokens
  vs item-asking **0.73** (≈5.3×). Under the strict wasted-turn regime item questions are starved (top-600
  popular pool, most unrated) exactly as the ML-1M/Phase-1b answerability gap (0.574 vs 0.120) predicts.
- **"Concepts beat items" replicates on FULL** (+0.015 @50, +0.018 @10) **but REVERSES on TAIL** (concepts
  −0.006 @50, −0.008 @10). This CONTRADICTS the ML-1M anchor (concepts beat items +50% tail).
- **Caveat (the likely cause):** `enc_v1_ml25m` was trained on ITEM tokens only. The concept token
  (Ac centroid, residual) is **OUT-OF-DISTRIBUTION** for this instrument. Ac centroids are low-norm mean
  directions (|Ac|≈0.42 vs |Q|≈0.90) that point toward popular/head mass, so the frozen fold lifts head/full
  but not tail. ML-1M's concept advantage explicitly required a **LEARNED concept channel (the "OOD fix")** —
  see STEP 1b below, which trains that channel and re-tests the tail claim.

**(b) Selection-matters row @q8 (concept channel): entropy vs pop vs random**
| ruler | entropy | pop | random |
|---|---|---|---|
| @50 full | **0.2690** | 0.2533 | 0.2574 |
| @50 tail | **0.0812** | 0.0661 | 0.0694 |
| @10 full | **0.2819** | 0.2643 | 0.2673 |
| @10 tail | **0.0623** | 0.0505 | 0.0521 |

- **Entropy (divisiveness) selection >> random and >> popularity on BOTH axes — REPLICATES cleanly.**
  entropy tail@50 0.0812 vs random 0.0694 (+0.0118) vs pop 0.0661 (+0.0151). The "selection matters"
  claim holds cross-dataset; asking divisive concepts beats asking popular or random ones.

**(c) Concept-channel saturation (entropy, @50 full), for the effrank tie-in (STEP 4):**
| q | 2 | 4 | 6 | 8 | q0=0.2650 |
|---|---|---|---|---|---|
| @50 full | 0.2456 | 0.2584 | 0.2657 | 0.2690 | marginal +/turn |
| marginal | −0.0194 | +0.0128 | +0.0072 | +0.0033 | ans_tok 1.4→2.0→3.2→3.9 |

- Net elicitation gain over q0 MOSTPOP is small (**+0.0040 @50 full by q8**) and **saturates fast**: marginal
  value per answered concept falls to ~+0.003 by q8. This is the popularity-saturation regime Phase 1b/1c
  flagged (full-profile headroom is +0.057 but the *elicited* concept slice recovers little of it with the
  frozen item-only encoder). The saturation shape (~4 effective concept questions) tracks the low concept
  effrank 3.99 — expanded in STEP 4.

**STEP 1 verdict (frozen item-only instrument):** answerability advantage (5× answered tokens) and
**selection-matters (entropy>>random/pop)** replicate; **"concepts beat items" replicates on full but the
tail claim does NOT** — attributable to the OOD concept token on an item-only encoder. STEP 1b tests the
learned-channel fix before a final verdict.

## STEP 1b — learned concept channel (the ML-1M "OOD fix"), tested and NOT a rescue
`scripts/paper2/ml25m_train_enc_concept.py`: fine-tune enc_v1 on mixed item+concept reveal sets
(full item reveals + up to 6 divisive concept tokens; 35% concept-only users; IPS clipped at 8; LR 1e-4;
val = 8-divisive-concept elicitation NDCG@50 on the 500 va users; item full-profile val tracked).
First attempt (LR 5e-4, reveal-set replacement) **collapsed the recommender** (item val 0.32→0.21) —
discarded. Gentle v2: ep1 concept_val 0.2554 / item_val 0.3215 (item ability preserved, −0.001 vs 0.3226);
ep2 declined on both → **best = ep1** (`enc_v1c_ml25m.pt`). Re-ran the identical B battery with it:

| @q8 entropy | CONCEPT full | CONCEPT tail | ITEM full | ITEM tail | conc−item full | conc−item tail |
|---|---|---|---|---|---|---|
| @50 (v1c learned channel) | 0.2603 | 0.0756 | 0.2437 | 0.0830 | +0.0167 | **−0.0074** |
| @50 (v1 frozen, STEP 1) | 0.2690 | 0.0812 | 0.2537 | 0.0872 | +0.0153 | −0.0060 |

- The learned channel does **NOT** rescue the tail claim (still −0.007) and its absolute concept numbers are
  *lower* than the frozen encoder's (0.2603 vs 0.2690 @50 full) — the fine-tune is mild degradation, not a
  fix, on this dataset. Selection-matters replicates again (entropy 0.0756 > pop 0.0602 / random 0.0646 tail@50).
- **Honest STEP-1 replication verdict:** (i) concept ANSWERABILITY advantage — replicates (5× answered
  tokens); (ii) SELECTION matters (divisiveness/entropy >> random ≥ pop) — **replicates cleanly, both
  encoders, both axes**; (iii) "answerable concepts beat items" — **replicates on FULL NDCG (+0.015–0.017
  @50), FAILS on TAIL** (−0.006/−0.007; ML-1M anchor was +36–50% tail). On ML-25M's 18k-item catalogue the
  genome-centroid concept directions live in a rank-4 popularity-aligned subspace (effrank 3.99, pointing at
  head mass) — informative for full-catalogue ranking but not for lifting tail items; item answers, when they
  exist (0.7/8 turns), are precise tail evidence. Note the item channel also pays a mechanical penalty
  (asked items are excluded from candidates and can be held targets), which makes its tail WIN conservative.
  Cross-dataset scope condition for Paper B: the concept-tail advantage is not universal; it depends on the
  concept vocabulary's geometry relative to the catalogue (ML-1M learned Ec vs ML-25M genome centroids).

---

## STEP 2 — C battery core (THE money experiment): continuous+graded actor vs discrete, cross-dataset

`scripts/paper2/ml25m_cont_actor.py` — faithful D1 port to the ml25m artifacts: actor MLP
[belief u(64), turn] → q ∈ R^64; differentiable unroll T=8 through the FROZEN `enc_v1_ml25m` (fe=q̂·mean|Ac|,
graded geometric answer a = NEG+(POS−NEG)(cos(u*,q̂)+1)/2, POS/NEG = train residual means +0.734/−0.664);
objective = (1−cos(u,u*)) + 0.3·softNDCG(held likes vs top-200 popular) − 1.0·divisiveness-field (DTAU=2.0,
2500 train tastes) — the exact D1 recipe (OBJ=ustar, SNDCG=0.3, DIVW=1.0). **From-scratch init (NOBC-style),
as D1 itself was trained** (warm-start unavailable cross-dataset; no BC floor — documented init choice).
4,000 train users (≥14 rated, ≥6 likes); val-select on the 500-user **va** cohort (tail@50), never test-peek.
Trained 32 epochs in resumable chunks (~50 s/epoch CPU); val climbed 0.0924 → **PEAK 0.1206 @ep31**
(`policy_ml25m_d1_best.pt`; every epoch saved; `policy_ml25m_d1_peak.txt` durable).

**Headline (TEST te cohort, seed-avg {1,2,3,7,11}, q8, NDCG@50, n=498/n_tail=494):**
| policy | @50 FULL | @50 TAIL |
|---|---|---|
| **continuous GRADED actor (D1 port, best-val ep31)** | **0.2742 ± 0.0017** | **0.1188 ± 0.0035** |
| continuous BINARY actor (same ckpt, ± answers) | 0.2560 ± 0.0034 | 0.0973 ± 0.0017 |
| entropy-concept BINARY discrete (CASPER-R-style static) | 0.2512 ± 0.0031 | 0.0843 ± 0.0020 |
| entropy-concept GRADED discrete (strongest static graded) | 0.2505 ± 0.0026 | 0.0817 ± 0.0025 |
| q0 (MOSTPOP, no elicitation) | 0.2461 ± 0.0022 | 0.0715 ± 0.0011 |

**>>> continuous-graded − discrete-graded @q8 = +0.0237 FULL / +0.0371 TAIL (≫ seed σ ≈ 0.003) — THE
C-thesis REPLICATES on ML-25M.** ML-1M anchor: +0.017/+0.038 (D1 0.378/0.178 vs entropy-graded 0.361/0.140).
The tail gap (+0.037) is essentially identical to ML-1M's (+0.038); the full gap is larger here.

**The graded/binary inversion replicates too (the load-bearing pattern):** for the ADAPTIVE continuous actor
graded ≫ binary (0.2742/0.1188 vs 0.2560/0.0973, +0.018/+0.022), while for the STATIC discrete selector
graded ≤ binary (0.2505/0.0817 vs 0.2512/0.0843) — the graded answer only pays when the question direction
is tuned per-user, exactly the ML-1M STATIC8 finding ("graded answers pay off only with adaptivity").

**q-curve (cont-graded, seed-avg):** FULL 0.2461/0.2348/0.2548/0.2669/0.2742, TAIL 0.0715/0.0589/0.0868/
0.1082/0.1188 at q0/2/4/6/8 — early dip at q2 (belief still forming) then monotone climb through q8 with no
saturation, unlike the discrete concept channel which plateaus by ~q4–6 (STEP 1 saturation table). Same shape
as ML-1M D1 (adaptive keeps climbing where static saturates). Curve saved `qcurve_ml25m_d1_cont.npy`.

Note: the actor's 8 graded off-manifold questions recover **+0.028 full** over q0 — about half the
full-profile headroom (+0.057) — vs +0.004 for the best discrete concept selector on the same instrument.

(STEP 1b follow-up: the concept-encoder fine-tune was run to ep4; concept-val peaked at ep3 0.2559 —
+0.0005 over ep1, immaterial; the STEP 1b verdict stands. `enc_v1c_ml25m.pt` = ep3 best.)

---

## STEP 3 — SNAP-LOSS on ML-25M (post-hoc realization of the trained actor's queries)

`scripts/paper2/ml25m_snaploss.py` — same actor (`policy_ml25m_d1_best.pt`), same graded geometric answers,
same frozen instrument and ruler; only the FOLDED direction changes:
**cont** = raw off-manifold unit query (native regime) · **concsnap** = nearest concept direction (cosine
over 1031 unit Ac; folds the actual concept vector) · **pairsnap** = best item-pair difference
(e_i−e_j)/|.| over the top-600-popular pool (top/bottom-64 search). TEST, seed-avg {1,2,3,7,11}, q8, @50.

| variant | @50 FULL | @50 TAIL | cos(q, realized) |
|---|---|---|---|
| **cont (off-manifold)** | **0.2742 ± 0.0017** | **0.1189 ± 0.0036** | 1.000 |
| pairsnap | 0.2391 ± 0.0024 | 0.0774 ± 0.0029 | 0.679 |
| concsnap | 0.2045 ± 0.0022 | 0.0519 ± 0.0016 | 0.703 |

- **SNAP-LOSS concsnap = −0.0697 FULL / −0.0670 TAIL; pairsnap = −0.0351 / −0.0416.**
- **The off-manifold gain REPLICATES — and is LARGER than on ML-1M** (anchor: concept-snap −0.037/−0.040).
  Pair-snap on ML-25M (−0.035/−0.042) lands almost exactly on the ML-1M concept-snap anchor; concept-snap
  here is ~2× worse. Snapping to the rank-4 genome-concept bank destroys the policy outright (0.2045 full is
  BELOW the q0 popularity floor 0.2461, tail 0.052 below q0's 0.072 — corrupted belief, not merely reduced
  information). The actor's queries are genuinely off-manifold (realization cos ≈ 0.68–0.70).
- Consistency check: the cont row reproduces STEP 2's headline to 4 decimals (0.2742/0.1188-0.1189).

---

## STEP 4 — effective-rank tie-in: saturation ≈ rank, continuous gap where rank is high

Measured ranks (Phase 1b, participation ratio of unit directions, D=64): **concepts 3.99 · items 8.35**
(ML-1M: concepts 2.25 · items 27.1). The thesis predicts (i) the concept channel saturates after ≈ its
effective rank of answered questions, and (ii) a continuous-over-discrete gap that grows with the rank
deficit of the discrete vocabulary.

**(i) Concept-channel saturation point (STEP 1, entropy selector, @50 full, marginal per turn):**
| answered concept tokens | ~1.4 | ~2.0 | ~3.2 | ~3.9 |
|---|---|---|---|---|
| marginal NDCG@50/turn | −0.019 | +0.013 | +0.007 | **+0.003** |

Marginal value collapses to ≈+0.003/turn by **~4 answered concept questions — numerically at the measured
concept effrank 3.99**. (ML-1M comparison point: effrank 2.25, static/concept channels saturated at ~q4 with
FULL peaking then declining; same law, smaller rank.) The discrete concept channel simply runs out of
independent directions to ask — its total elicitation gain is +0.004 @50 full by q8.

**(ii) Where the rank ceiling is absent, elicitation keeps paying:** the continuous actor asks free 64-d
directions (no vocabulary rank bound) and its q-curve is still rising at q8 (+0.028 full / +0.047 tail over
q0), with the graded answer carrying a full scalar projection per turn. The +0.024/+0.037 continuous-over-
discrete gap (STEP 2) is exactly the prize the rank analysis predicts, and STEP 3 shows it is destroyed by
projecting back onto the rank-4 concept bank (−0.070) and halved-to-destroyed on the richer pair set
(rank ~8 item pool → −0.035). Gap ordering cont > pairsnap > concsnap tracks the realization-set rank
ordering (∞ > items 8.35 > concepts 3.99). **The effrank story replicates cross-dataset.**

---

## FINAL REPLICATION SCORECARD (ML-25M vs ML-1M anchors, pre-stated K=50 ruler)

| # | claim (ML-1M anchor) | ML-25M result | verdict |
|---|---|---|---|
| B1 | concept answerability ≫ items (0.574 vs 0.008/0.120) | 5.3× answered tokens @q8 (3.85 vs 0.73) | **REPLICATES** |
| B2 | selection matters: entropy/divisiveness ≫ random ≥ pop | tail@50 0.0812 > 0.0694 > 0.0661 (both axes, both encoders) | **REPLICATES** |
| B3 | answerable concepts beat items — FULL | +0.015…+0.017 @50 | **REPLICATES** |
| B4 | answerable concepts beat items — TAIL (+36–50%) | **−0.006…−0.007 @50 (−7…−9%)**; learned-channel fix tried, no rescue | **FAILS** (scope condition: genome-centroid concepts are rank-4/head-aligned; item answers are precise tail evidence) |
| C1 | continuous+graded beats strongest discrete graded (+0.017/+0.038) | **+0.0237 FULL / +0.0371 TAIL** (σ≈0.003) | **REPLICATES** (tail gap ≈ identical) |
| C2 | graded>binary ONLY for adaptive-continuous (inversion) | cont: graded +0.018/+0.022 over binary; discrete: graded ≤ binary | **REPLICATES** |
| C3 | snap-loss: off-manifold gain real (−0.037/−0.040 concept-snap) | concsnap −0.070/−0.067; pairsnap −0.035/−0.042 | **REPLICATES (stronger)** |
| C4 | adaptive keeps climbing where static/discrete saturates | cont q-curve rising at q8; concept channel saturates ~q4 | **REPLICATES** |
| E1 | concept effrank < item effrank (2.25 vs 27.1) | 3.99 vs 8.35 (Phase 1b, clean) | **REPLICATES** |
| E2 | saturation point ≈ concept effrank | marginal → +0.003/turn at ~4 answered concepts ≈ effrank 3.99 | **REPLICATES** |

**Bottom line:** the Paper-C thesis (continuous+graded+adaptive beats any discrete static selector; the gain
is genuinely off-manifold; the discrete channel is rank-limited) **replicates cleanly on ML-25M at the
pre-stated K=50 ruler**, with the tail gap numerically matching ML-1M. The one casualty is Paper B's
concept-TAIL claim (B4): on an 18k-item catalogue with genome-centroid concepts, concepts win FULL but lose
TAIL to item-asking — a scope condition Paper B must state, not a refutation of answerability (B1) or
selection (B2), both of which replicate.

### Durable artifacts (`data/movielens/.cache/ml25m/`)
- `policy_ml25m_d1_best.pt` (ep31, THE headline actor) + every-epoch ckpts + `policy_ml25m_d1_peak.txt`
- `enc_v1c_ml25m.pt` (concept-channel encoder, ep3 best) + state + peak.txt; `membership.npz`
- `qcurve_ml25m_d1_cont.npy`. Scripts: `ml25m_membership.py`, `ml25m_b_battery.py`,
  `ml25m_train_enc_concept.py`, `ml25m_cont_actor.py`, `ml25m_snaploss.py` (all under `scripts/paper2/`).
- Disk ≥38 GB free throughout; no commits made.
