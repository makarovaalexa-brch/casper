# E0f — the DECISIVE rerun: descent under the NON-CIRCULAR real-rating item answer model

Date 2026-07-07. Scripts `scripts/e0f_lib.py` (real-rating answer model), `scripts/e0f_control.py`
(mandatory fold-sanity control + baseline reproduction), `scripts/e0f_fold_diag.py` +
`scripts/e0f_fold_diag2.py` (fold-miscalibration diagnosis, run because the control FAILED),
`scripts/e0f_main.py` (main descent arms). Backing JSON: `experiments/E0f_control.json`,
`E0f_fold_diag.json`, `E0f_fold_diag2.json`, `E0f_main.json`. **NO LLM calls** — all cached grids.
Reuses the E0/E0d harness EXACTLY via `import e0_gonogo as E` (RecVAE-d512 fold-in, belief
`z' = z + eta*a*q`, eta=16, NDCG@10, per-user paired bootstrap 5000, ML-25M, 298 eligible users,
pinned seed-123 answerer split).

Motivating context (FABLE_AGENT_DESIGN §"E0f — THE DECISIVE RERUN" + §"E0d/E0e RESULTS" load-bearing
caveat): E0d/E0e answered EVERY question geometrically (`a = cos(z*, q)`); real star ratings never
entered the answer value, so "descent loses" was established only for the leftover circular value
channel. E0f wires in the study-design §3 real-rating item answer and re-runs.

---

## HEADLINE (two sentences)

1. **The mandatory fold-sanity control FAILED**: a single free, true real-rating item answer folded
   from cold **HURTS** NDCG (Δ=−0.053, CI [−0.079, −0.027]); the OLD geometric single item does not
   help either (Δ=−0.011, CI [−0.036, +0.014]). The failure is **not** specific to real ratings — it
   is a **fold miscalibration** for single item tokens, diagnosed below with numbers.
2. Under the corrected understanding, **every REALIZABLE real-rating descent arm loses significantly**
   to static B; only **target-peeking upper bounds** (which rank a user's rated films by their own
   held-out-NDCG improvement) win. **The E0d verdict HOLDS: descent loses under real answers.**

---

## CONTROL RESULT FIRST (the mandated fold-sanity check) — FAILED

From cold `z=0`, fold ONE answer for a random answerable+RATED known item at t=1; compare NDCG@10 vs
asking nothing (`z=0`, base NDCG = 0.1481). n=298 users, paired bootstrap.

| single-token fold from cold | mean Δ vs no-question | 95% CI | HELPS? |
|---|---|---|---|
| **REAL rating answer** (study-design §3) | **−0.0532** | [−0.0788, −0.0272] | **NO (hurts, CI excludes 0)** |
| OLD geometric answer `a=cos(z*,q)` | −0.0106 | [−0.0356, +0.0144] | NO (CI spans 0) |
| REAL, likes only (r≥4) | −0.0265 | [−0.0658, +0.0126] | NO (CI spans 0) |

Per the pre-registered rule ("if the real-rating control HURTS, stop the main runs and investigate the
fold scaling for item tokens; a diagnosis of the fold miscalibration is then the primary deliverable"),
the fold was investigated before the descent verdict was read. Baselines reproduced up front inside the
control: **static B 0.2517/0.2812 ✓, static+skip 0.2516/0.2796 ✓.**

---

## THE FOLD MISCALIBRATION (primary deliverable) — two compounding causes, both with numbers

Rescale: **BETA = 0.3924** RMS-matches the real-rating-centered signal (cohort RMS 0.9083 stars) to
the geometric item answers (cohort RMS 0.3565) over the identical 23,011 (user, known-rated-item)
pairs. So the item fold STEP BUDGET is identical to E0d's; only the answer VALUE changed.

### Cause 1 — eta=16 grossly OVERSHOOTS for a single narrow item token
Base NDCG(z=0)=0.1481. A single **concept** from cold HELPS (+0.0275, CI [+0.008,+0.048]). But the
item token's value is extremely eta-sensitive (mean Δ vs z=0, random known-rated item, n=298):

| eta | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| **geometric** item | **+0.0627** | +0.0312 | +0.0096 | −0.0036 | **−0.0106** | −0.0120 | −0.0128 |
| **real-rating** item | −0.0134 | −0.0238 | −0.0386 | −0.0502 | **−0.0532** | −0.0583 | −0.0614 |

The *privileged* geometric item answer is highly informative at eta≈1 (+0.063, even more than a
concept) but turns HARMFUL by the calibrated eta=16 — the operator's eta was tuned for 8 accumulating
tokens dominated by broad concepts, not a single narrow item spike. Matched-step-norm confirms breadth
matters independently of magnitude: at every fixed step norm s, a concept beats an item
(s=1: concept +0.046 vs item +0.009; s=8: +0.019 vs −0.013). Folding one narrow item trades the
popularity prior (which already captures popular held-out likes at z=0) for a narrow neighbourhood.

### Cause 2 — the fold operator is MIS-SPECIFIED for any non-geometric answer
`z' = z + eta*a*q` is a taste-gradient step ONLY when `a = cos(z*, q)` (literally the coordinate of
true taste along q). A real rating is not that coordinate. Over all known-rated pairs,
`corr(centered rating, cos(z*, Wn[j])) = +0.39` (sign-agreement only 58%). Crucially, the geometric
answer is **positive even for DISLIKED known items** (mean cos when like +0.36, when dislike +0.22 —
both positive, because z* encodes the liked profile and disliked-but-watched films sit in the same
catalogue neighbourhood), whereas the real centered rating is negative for dislikes → the two
**disagree in sign** on dislikes. Hence, even **noiseless**, the real answer barely helps at best-eta
and turns harmful quickly (Δ vs z=0):

| eta | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| real, NOISELESS | +0.0089 | +0.0018 | −0.0116 | −0.0294 | −0.0387 | −0.0431 |
| real, σ=0.70 noise | +0.0035 | −0.0134 | −0.0238 | −0.0386 | −0.0502 | −0.0532 |

The σ=0.70-star fidelity noise (≈44% of the centered-rating energy) erases the sliver of help that
survives at eta≈0.5. **This is the thesis meta-point in its cleanest form: the item channel is
"useful" in the geometric arena only because the geometric answer is a coordinate of true taste;
swap in the honest data-side answer and the same operator, at the same eta, makes the item channel
harmful. THE ANSWER MODEL DECIDES.**

Note the E0d-context check (item folded at t5 AFTER 4 concepts, random known-rated item): geometric
+0.0261 (HELPS — because in-profile items have high cos to z*; E0d's UNRATED bank items did not), but
real −0.0288 (HURTS). E0d's geometric "dilution" was partly an artifact of using off-profile bank
items; the honest real-rating channel hurts even for in-profile items.

---

## MAIN RUNS — descent under real answers (reported under the failed-control caveat)

All arms: n=298 users, T=8, phase-1 = k geometric concepts (identical to E0d), phase-2 = (8−k) known-
rated items answered with the real-rating model, NDCG@10, paired bootstrap vs static B. static B and
static+skip stay fully geometric (references; reproduced 0.2517/0.2812 and 0.2516/0.2796).

### REALIZABLE arms (no target peek) — ALL LOSE

| arm (ranking / eta_item) | anytime | endpoint | Δ anytime vs B | 95% CI |
|---|---|---|---|---|
| static B (ref) | 0.2517 | 0.2812 | 0 | — |
| static+skip | 0.2516 | 0.2796 | −0.0002 | [−0.0011, +0.0008] |
| descent k=4, **\|centered rating\|**, eta=16 | 0.1615 | 0.0956 | **−0.0902** | [−0.1036, −0.0769] |
| descent k=4, \|centered rating\|, **eta=1 (corrected)** | 0.2393 | 0.2454 | **−0.0124** | [−0.0165, −0.0084] |
| descent k=4, **p̂·\|centered rating\|**, eta=16 | 0.1730 | 0.1307 | −0.0787 | [−0.0941, −0.0636] |
| descent k=3, \|centered rating\|, eta=16 | 0.1366 | 0.0946 | −0.1151 | [−0.1320, −0.0983] |
| descent k=5, \|centered rating\|, eta=16 | 0.1859 | 0.0953 | −0.0658 | [−0.0757, −0.0560] |
| emergent (marginal-info × true × div), eta=16 | 0.1865 | 0.2049 | −0.0652 | [−0.0865, −0.0432] |

Correcting the fold (eta_item 16→1) recovers ~0.078 of the 0.090 harm — confirming Cause 1 is most of
the damage — but the realizable descent **still loses** (−0.0124, CI excludes 0): the residual is
Cause 2 (operator mismatch) + noise. **No realizable arm, at any switch point k or corrected eta,
beats static.**

### PER-USER TARGET-PEEK UPPER BOUNDS (NOT realizable — rank rated items by their OWN held-out NDCG)

| arm | anytime | endpoint | Δ anytime vs B | 95% CI |
|---|---|---|---|---|
| UB k=4, real, peek-rank, eta=16 | 0.3270 | 0.4362 | +0.0753 | [+0.0663, +0.0848] |
| UB k=4, real, peek-rank, **eta=1** | 0.2606 | 0.3041 | +0.0089 | [+0.0052, +0.0127] |
| UB k=4, **geometric**, peek-rank (same pool) | 0.3522 | 0.4901 | +0.1004 | [+0.0906, +0.1105] |
| UB all-8 real items, peek-rank | 0.4595 | 0.4292 | +0.2078 | [+0.1872, +0.2294] |

These "win" ONLY because the ranking peeks at held-out targets (each rated item scored by its own
held-out-NDCG improvement — analogous to E0d's population popval, but for user-specific rated items it
degenerates into a per-user target peek). They are upper bounds, not agents: knowing which of your
rated films predict your held-out taste requires knowing the answer. Reported for the record and to
show the channel *contains* signal that no realizable ranker (data-side or p̂) can extract at T=8.

### The per-turn curve around the switch (the E0d diagnostic)

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| static B | 0.176 | 0.227 | 0.246 | 0.267 | 0.268 | 0.272 | 0.278 | 0.281 |
| descent k=4 real datarank (eta16) | 0.176 | 0.227 | 0.246 | 0.267 | **0.106** | 0.087 | 0.088 | 0.096 |
| descent k=4 real datarank (eta1) | 0.176 | 0.227 | 0.246 | 0.267 | **0.254** | 0.251 | 0.249 | 0.245 |
| UB k=4 real peek-rank (eta16) | 0.176 | 0.227 | 0.246 | 0.267 | **0.391** | 0.428 | 0.445 | 0.436 |

Answering the E0d diagnostic question directly ("does the first real-rating item now HELP instead of
dropping 0.267→0.248?"): **NO — under a realizable ranker it drops even harder** (0.267→0.106 at
eta=16; 0.267→0.254 at corrected eta=1, comparable to E0d's geometric 0.267→0.248). It only rises when
the ranker peeks at targets (0.267→0.391).

---

## VERDICT

**The E0d verdict HOLDS: coarse→fine descent LOSES under the non-circular real-rating item answer
model, in every realizable arm, at every switch point, and even after correcting the fold eta.** The
answer-model swap does NOT flip the adaptivity verdict. The mandatory fold-sanity control FAILED, and
its diagnosis is the primary deliverable: (1) the calibrated eta=16 overshoots for single narrow item
tokens, and (2) the belief operator `z'=z+eta·a·q` is a taste-gradient step only for the circular
geometric answer — real ratings are only weakly aligned (corr 0.39) and sign-disagree on dislikes, so
the honest answer channel is mis-specified for this operator. The item channel *does* contain
ranking-relevant signal (visible in the geometric-eta1 fold and the target-peek upper bounds), but no
realizable ranker extracts it at T=8 through this fold. The two-regime / channel-map story proceeds
unchanged, now with the strongest possible footnote: descent was tested under BOTH the geometric and
the real-rating answer models and loses under both — and the recurring lesson (THE ANSWER MODEL
DECIDES) is demonstrated once more, this time in the direction that the honest answer model makes the
fine channel WORSE, not better.

---

## ASSUMPTIONS / judgment calls (explicit)

1. **Rescale choice (STATED, used everywhere): one global scalar BETA = RMS(geometric item answers) /
   RMS(centered real ratings) over all (user, known-rated-item) pairs = 0.3924** (RMS_geo 0.3565,
   RMS_rating 0.9083, n=23,011). This eta-matches the real-rating item channel to the exact energy the
   geometric item channel would have on the same tokens (Paper C fidelity-boundary η-matching), so the
   ONLY thing that changes vs E0d is the item answer VALUE, never the fold step budget. A single global
   BETA (not per-user) was chosen for simplicity and to avoid a per-user free parameter; a per-user
   rescale is an untested alternative.
2. **Sigma handling:** fidelity noise σ=0.70 **stars**, added in star units BEFORE the BETA rescale:
   `a_real = BETA·((r − mean_known_u) + 0.70·ε)`, ε~N(0,1). So σ is rescaled by the SAME BETA as the
   rating (consistent). ε is deterministic, seeded per (user, item): `default_rng((u·100003+j) mod 2^32)`.
3. **Centering:** the real rating is centered on the user's own **known-profile mean rating** (mean over
   ALL known-portion rated items), per the spec.
4. **Item channel = the user's KNOWN-portion RATED items** (trivially answerable per study-design §3
   "Item — RATED → answerability yes"). The E0d bank's designed/taste-adjacent items are almost never
   in a user's rating history, so they cannot carry a real rating; using the known-rated items is the
   faithful realization of "answerable AND rated". This makes the rated-item descent an OPEN-RECALL-
   grade channel (the user supplies a real opinion on a film they have seen — Paper D territory).
   Held-out targets are NEVER used as answers (they live in the other split half); folded known items
   are masked from the NDCG candidate set anyway (can't recommend a known item).
5. **"pop-value" ranking (arm b-i) is ill-defined for user-specific rated items.** E0d's popval was a
   POPULATION mean over the many users who can answer a shared bank item; for user-specific rated items
   it collapses to a single-user statistic. I therefore report: (a) a **realizable** data-side ranking
   `|centered rating|` (no peek) and a **p̂** surrogate ranking, both of which LOSE; and (b) a clearly-
   labelled **per-user target-peek upper bound** (rank rated items by their own held-out-NDCG
   improvement) — the closest analog to E0d's popval, but leakier, so NOT read as a realizable result.
6. **UNRATED-item SENSITIVITY-ONLY arm was SKIPPED** (per the spec's "or skip this arm if time is
   short — say so"): the item channel is defined as the known-rated pool, and unrated-iconic items would
   need an LLM-predicted value in the value channel (reintroducing a model into the answer) — deprioritized
   because the fold-sanity control already failed, making the fold operator, not the value source, the
   binding issue. The geometric bank-item channel (E0d) already stands as the unrated-item reference.
7. **NDCG@10, not @50** (inherited from the E0/E0d/gate harness `G._ndcg_top10`). The spec's main-runs
   bullet says NDCG@50, but the baselines it mandates (static B 0.2517/0.2812) are @10 numbers and the
   whole E0 lineage is @10; @10 was kept to reproduce the baselines exactly and preserve comparability
   with E0d. Switching to @50 would break baseline reproduction and is not adopted.
8. **eta=16 canonical** (ml25m gates G8 value), inherited. The corrected-eta arms (eta_item=1) are
   diagnostic, not the reference. Phase-1 concepts always fold at eta=16 geometric (turn consumed even
   if refused), identical to E0d.
9. **Deterministic** throughout: `np.random.default_rng(0)` for all bootstraps (5000 resamples); the
   per-user control/diagnostic item pick uses `default_rng(1234+u)`; fidelity noise seeded per (u,j).
   RecVAE fold-in, concept bags, and the seed-123 answerer split are cached and deterministic.
