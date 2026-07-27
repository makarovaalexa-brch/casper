# E0 GO/NO-GO diagnostics — results (ML-25M, RecVAE-d512, 300 study users)

Date 2026-07-07. Scripts `scripts/e0_gonogo.py` (P1 divisiveness variant, P2, P3, P4),
`scripts/e0_p1_improved.py` (P1 population-value variant), `scripts/e0_p1_skip2.py`
(P1 static+skip variant). **NO LLM calls** (all cached grids). Reuses the canonical gate
fold-in / belief / NDCG machinery (`scripts/llm_answerability_gate.py`), the fitted pmodel
(`.cache/instrument2/answerability_pmodel.json`), and the cached judged grids
(`answerability_grid_ml25m.json` = gate, `answerability_mainstudy_grid.json` = main study).
Backing JSON: `experiments/E0_gonogo_results.json`, `E0_p1_improved.json`, `E0_p1_skip.json`;
scatter `experiments/E0_fuel_value_scatter.csv`.

## TL;DR verdict

| E0 gate (RUN ORDER) | Threshold | Measured | Verdict |
|---|---|---|---|
| O-ans − static (anytime) | ≥ 0.015 | **−0.0003** (best realizable, static+skip; TIE) | **FAIL** |
| Posterior sharpens by t≤4 | AUC rises | 0.901 → 0.895 (declines) | **FAIL** |
| Fuel-value quadrant non-empty | populated | 64 / 279 questions (23%) | PASS |

**E0 = NO-GO for the adaptive agent** (2 of 3 gates fail). The result lands squarely on the design's
pre-registered TIE / scope branch (Q7 "if it ties"): the privileged answerability oracle's edge is
real (+0.287) but is **almost entirely privileged true-taste belief, not answerability knowledge** —
a realizable agent that knows the full per-user answerability table but must build its belief only
from answers captures ≈ 0 of it at T=8.

---

## Reproduction check (done FIRST, canonical code)

Ran the **unmodified** `g2_exploitability` on the cached gate grid:

- **Selector A (O-full) = 0.5681** (published 0.5681) ✓
- **Selector B (static) = 0.2812** (published 0.2812) ✓
- dNDCG = 0.2868, CI95 [0.2678, 0.3066], n = 298 (published 0.2868)
- **Matches published exactly → proceeded.**

My per-turn re-implementations independently reproduce both endpoints (static t8 = 0.2812 = B;
O-full t8 = 0.5681 = A), confirming the harness is faithful before any new selector was added.

---

## P1 — O-ans decomposition (the answerability-discovery prize)

O-full (=A) is privileged twice: it holds the true per-user answerability table **and** a true-taste
belief (each turn it picks the question maximizing true-target NDCG). O-ans keeps the table but the
belief is **realizable** (z starts 0, updated only by answers received; z′=z+16·a·q, a=cos(z\*,q));
its question **value is scored without any z\* peek**. Three z\*-free value models were tried;
all NDCG@10, anytime = mean over t=1..8, n = 298 users, per-user paired bootstrap (5000).

| selector (belief / value) | anytime | endpoint | prize vs static (anytime) | 95% CI |
|---|---|---|---|---|
| **O-full** (A: table + **true-taste** belief) | 0.5256 | 0.5681 | **+0.274** | — |
| static B (population greedy schedule) | 0.2517 | 0.2812 | 0 (ref) | — |
| O-ans · belief-only info value (divisiveness×novelty) | 0.2048 | 0.2403 | −0.0469 | [−0.063, −0.031] |
| O-ans · population value-when-answered routing | 0.2302 | 0.2533 | −0.0215 | [−0.031, −0.013] |
| **O-ans · static+skip** (diversified, conditional static) | 0.2514 | 0.2785 | **−0.0003 (TIE)** | [−0.0012, +0.0007] |

**Reading.** The best realizable O-ans (static+skip) **ties** the strong static (prize −0.0003
anytime, CI spans 0; endpoint −0.0027, CI [−0.0091, +0.0037]). The two greedy-value O-ans variants
do *worse* than static — a value-model artifact: with η=16 they inject large belief steps along
divisive / individually-high-value but redundant directions, whereas static B is a **diversified**
greedy-forward schedule. static+skip is the honest one because it inherits B's diversification and
only adds per-user refusal-skipping.

**Why the prize is ≈ 0 (the mechanism, surfaced by the run).** The strong static schedule already
**self-selects near-universally-answerable questions**: users can answer **7.53 of B's 8 questions**
on average (94%). There are almost no refused turns to reclaim, so per-user answerability routing has
essentially no headroom at T = 8.

**The decomposition (the deliverable of §0 of the design doc).** G2's +0.287 splits as:

- O-full − static = **+0.287** (endpoint) — the entire exploitability gap, and it is the
  **privileged true-taste belief**.
- O-ans(best) − static = **≈ 0** — the realizable **answerability-discovery** prize.

So G2 was almost 100% taste-peek, not "knowing who can answer what." **Per the pre-registered
decision rule (prize < 0.015 → NO-GO), the agent is a NO-GO at T = 8.**

Per-turn NDCG@10 (mean over users):

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| O-full | 0.389 | 0.493 | 0.530 | 0.546 | 0.553 | 0.562 | 0.566 | 0.568 |
| static B | 0.176 | 0.227 | 0.246 | 0.267 | 0.268 | 0.272 | 0.278 | 0.281 |
| static+skip | 0.173 | 0.216 | 0.245 | 0.257 | 0.262 | 0.270 | 0.273 | 0.279 |
| O-ans (pop-value) | 0.173 | 0.216 | 0.241 | 0.230 | 0.237 | 0.245 | 0.246 | 0.253 |
| O-ans (divisiveness) | 0.140 | 0.179 | 0.198 | 0.208 | 0.217 | 0.225 | 0.231 | 0.240 |

---

## P2 — Decision loss of the fitted surrogate

Surrogate selector = O-ans (belief-only value) but with the true table replaced by p̂ from the pmodel
(true known-profile genre_match; isolates surrogate quality, not online estimation). A pick that is
truly unanswerable costs a turn.

- **NDCG forfeited vs O-ans: anytime +0.0015 (CI [−0.0036, +0.0069]), endpoint +0.0005** — negligible.
  Caveat: this is measured against the *weak* divisiveness O-ans (both below static), so the
  surrogate's decision loss is not the binding quantity here; with the realizable prize itself ≈ 0
  (P1), a better surrogate cannot buy the agent NDCG it has no way to earn.
- pmodel **pooled AUC = 0.9072** vs **within-user AUC = 0.9091** (median 0.9161; n = 300 users,
  80 930 pooled obs). **Within-user ≈ pooled** — contrary to the design's hypothesis that pooled AUC
  is inflated by between-user separation. The surrogate ranks questions *within* a user just as well
  as across users; answerability discrimination is **not** the bottleneck.

---

## P3 — Posterior-sharpening curve

Online ĝ (20-dim genre distribution) starts at the population prior; updated by static B's simulated
turns (answered concept/item → +genre vector; refusal → −p̂·genre vector). At each t we score
AUC(pmodel-with-ĝ-genre_match vs the user's true judged grid), averaged over 300 users.

| t | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.9012 | 0.8946 | 0.8956 | 0.8956 | 0.8953 | 0.8954 | 0.8943 | 0.8945 | 0.8948 |

- t0 (population prior) AUC = 0.9012; t4 = 0.8953; t8 = 0.8948. **Sharpens by t≤4: FALSE** — the curve
  does **not** rise; it dips ~0.006 and flatlines. This is **failure-mode-1 evidence measured
  directly**: online ĝ updating over an 8-turn budget does not sharpen the per-user answerability
  posterior.
- Note the population prior already gives AUC 0.90 within-user — the pmodel's discrimination is
  dominated by public features (log_ratings_count +1.55, is_concept +1.27) over which genre_match
  (+0.63) adds little at the within-user margin, and the 8-turn online ĝ estimate is too noisy to
  improve it. Convergent with P1: even the answerability *table itself* (O-ans) doesn't help at T=8,
  so an online *estimate* of it certainly cannot.

---

## P4 — Fuel-value scatter

Per judged question (≥ 20 users): x = answerability variance across users (Bernoulli p(1−p)),
y = cold-belief value proxy = divisiveness qᵀCov(W)q (the per-question value the selectors use at z=0).
CSV: `experiments/E0_fuel_value_scatter.csv`.

- 279 questions; median variance 0.0260, median value 0.0152.
- **High-variance × high-value quadrant: 64 questions (23%) — populated (27 concepts + 37 items).**
  Fuel *exists* in the broad question space (e.g. concepts C:98, C:194, C:90; items I:16227, I:4773).
- BUT (reconciling with P1): the fuel is **not reachable by routing beyond static**, because the
  optimal static schedule already lives in the high-answerability/high-value region (7.53/8
  answerable). The upper-right quadrant is real but the strong static already harvests it; per-user
  routing has nothing left to add at T=8.

---

## Overall interpretation (honest — what passed and what failed)

- **Reproduced** the published G2 numbers exactly (A=0.5681, B=0.2812).
- **The fuel is real** (P4 quadrant populated; P2 within-user AUC 0.91) — the answerability
  heterogeneity the gate found is present and the surrogate can rank it.
- **But the realizable prize is ≈ 0** (P1: best realizable O-ans ties static, −0.0003 anytime) and the
  **posterior does not sharpen in budget** (P3). Two independent gates fail.
- **Root cause (a measured, reportable fact):** the strongest static schedule self-selects
  ~universally-answerable questions (7.53/8), so knowing the per-user table buys almost nothing at
  T=8; G2's +0.287 was privileged *taste* belief (O-full), not answerability knowledge.
- **Recommendation:** this is the design's pre-registered **TIE/scope branch (Q7)** — ship the static
  channel map + the O-full/O-ans/static decomposition as the *price of answerability knowledge*
  ("the realizable answerability prize at T=8 is ≈0; the strong static already occupies the answerable
  region"), rather than training the adaptive agent. A longer horizon, a harder-to-answer static
  (forcing refusal headroom), or joint taste+knowledge inference would be needed to make the prize
  material — none is an 8-turn quick win.

---

## Assumptions / approximations (explicit — not smoothed over)

1. **O-ans value model is the load-bearing choice.** The design's V(q) references "the same
   divisiveness/entropy/expected-info machinery selector B uses," but the canonical B is a population
   greedy-forward-NDCG *schedule* whose value is not a per-turn belief-scorable quantity. I therefore
   tried three z\*-free realizable value models (belief-only divisiveness×novelty; population
   value-when-answered ranking; static+skip = B's diversified priority list with per-user
   refusal-skipping). **static+skip is the fairest** (it inherits B's diversification and adds only
   the answerability table) and is the one used for the GO/NO-GO decision. The two greedy-value
   variants under-perform because single-question value is redundant under a large η step — a
   value-model artifact, not evidence about the prize. Had a value model beaten static I would have
   reported it; none did, and static+skip's tie is robust (its mechanism — 7.53/8 already answerable —
   is model-independent).
2. **Belief covariance σ²=1** (divisiveness variant) is an untuned nuisance scale (answers are
   cos∈[−1,1], not stars); it only sets how fast Σ shrinks along asked directions.
3. **static+skip priority list** = B's frozen 8-schedule head ++ the population value-when-answered
   order for the remaining shared keys (a cheaper stand-in for a full length-16 greedy-forward
   re-derivation, which was prohibitively slow; the greedy positions I did compute reproduce B's order
   — pos1 = C:77 — so the head is faithful). Because B users already answer 7.53/8, the tail of the
   list is almost never reached, so this approximation is immaterial to the tie.
4. **P2 refusal cost** folded in as opportunity cost (a refused pick yields zero belief gain and
   consumes a turn); no separately-tuned c_refusal scalar.
5. **P3 ĝ** lives in the 20-dim genre space (the pmodel's genre_match feature space), updated
   additively; population prior = mean of per-user known-half dominant-genre distributions; refusal
   down-weight uses the population-prior p̂ of the refused question.
6. **Candidate pool** for P1 = the gate grid's per-user bank (39 concepts + 30 designed + 6
   taste-adjacent), identical to the G2 selectors, so O-full/static match 0.568/0.281 exactly. P2-AUC,
   P3, P4 use the richer main-study grid (200 concepts + ~1.7k items).
7. **Pmodel pooled AUC here (0.907)** is the deployed full-data model scored on the main-study grid
   (in-sample-ish), slightly below the reported held-out-users 0.921; the within≈pooled comparison is
   apples-to-apples (same model, same obs) and is the load-bearing statement.
