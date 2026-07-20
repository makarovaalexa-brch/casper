# FRESH ADVERSARIAL AUDIT — how the adaptivity experiments could hide a real adaptive advantage

Date 2026-07-09. Independent methods audit (no prior context; interpretive prose in prior reports
deliberately ignored — designs, code, and raw numbers only). Working stipulation: the fuel
measurements (adaptivity_battery_v1_A.json: ICC .05–.14 beyond popularity at know_well, taste-tracking
answerability +0.10–0.15 within tiers, LOUO-AUC lift +0.05–0.13) are real; the question is what in the
experiment designs could mask a real adaptive prize. Two cheap local probes were run (no API, no
training): a grid/pool census and a fold-divergence probe (scripts in scratchpad; outputs quoted
inline). Findings are ranked by (likelihood-of-masking × fixability).

---

## FINDING 1 — the candidate pool structurally excludes the measured fuel (RANK 1)

**The flaw.** Every arena (battery Stage C, repair P3, static-contamination, fair-reruns, tree,
r-learned) draws its action space from the same construction: top-60 questions PER CHANNEL ranked by
**coverage**, where coverage = number of users who answer (k>=1) — `battery_stage_c.py:72-91`,
`repair_probes.py:94-113` (`POOL = 60`, `cov.most_common()`). Ranking by answer-count is ranking by
answerability: the pool is, by construction, the maximally-universally-answerable subset.

**The evidence (probe on `answerer_v1_grid173_WORKING.json`, 173 users):**

| channel | in-pool k>=1 rate | outside k>=1 | scarcity-habitat Qs (rate 0.2–0.7) in-pool | outside |
|---|--:|--:|--:|--:|
| concept | 1.000 | 0.726 | **0/60** | 275/1068 |
| item | 1.000 | 0.991 | 0/60 | 3/740 |
| attr | 0.999 | 0.801 | **0/60** | 111/440 |

- Per-user refusal (no_clue) rate **within the 180-candidate pool: mean 0.000, p90 0.000**.
- Only **12.6%** of a user's know_well LLM cells are reachable in-pool (mean 1083 know_well cells/user).

**How it masks adaptivity.** Stage A itself located the fuel in the *scarcity habitat* (answer rate
0.2–0.7: concept-niche, item-bank-low/mid, attribute-director; ADAPTIVITY_BATTERY_V1.md A3) and in the
taste-validity gap on **concepts (+0.105/+0.148)** — overwhelmingly niche/mid tags. The arena then
hands every policy an action space in which (a) answerability is the constant 1 for every user and
every question — so an answerability-routing belief has *literally zero variance to route on* (in
`plan_rblind`, `y_ev` is 1 for every item probe of every user, so the kmap posterior receives the same
evidence for everyone); and (b) the taste-adjacent niche questions a per-user descent would ask do not
exist as actions. Every "routers tie/lose" verdict is therefore a conditional statement: *given an
action space provably purged of the measured heterogeneity*. The "abundance-regime" framing in the
reports is not a property of the environment (the grid has 275 niche-concept + 111 attribute habitat
questions); it is a property of the pool filter.

**Minimal corrective experiment.** Rebuild the pool stratified by population answer-rate (e.g. per
channel: 20 top-coverage + 40 sampled from the 0.2–0.7 habitat, selected by population statistics only
— no per-user peeking), re-run split-fair s-item/s-mixed vs one knowledge-belief router at T=12.
Refusals now cost turns, so answerability discovery has actual purchase for the first time. All local
compute; the harness (`fair_reruns.py`) needs only the pool builder changed.

---

## FINDING 2 — the routers' value/gain models are trained in a simulator with the fuel surgically removed (RANK 2)

**The flaw.** The E5 "firewall" is implemented not by cross-fitting on grid users but by training all
router beliefs in a *synthetic population world* (`i25_fold_v2.build_reveal_v2`, reused by
`repair_probes.fit_value_model:715-759` and `r_learned_labels.py`). In that world:

1. LLM-item answer values = **population item-mean + N(0, 0.70)** (`i25_fold_v2.py:190`) — zero
   user-specific content by construction.
2. Concept/attr answer values = noisy re-aggregations **of the user's own already-revealed items**
   (`i25_fold_v2.py:193-218`: `agg = (rel[:,ctag]*crv).sum()/mass` over the reveal set) — a redundant
   echo of information the fold already holds; a concept answer can never reveal anything new.
3. know_well is drawn at **population base rates, taste-independent** (`sample_knowledge`,
   admitted in R_LEARNED.md caveats: "a taste-conditional vividness premium is NOT injected").

**How it masks adaptivity.** A gain model (GBM val R² 0.076; r-learned R² 0.076) trained where answers
carry no user-specific signal beyond what is already folded *must* learn that question choice doesn't
pay beyond popularity — the correct answer *in its training world*. It is then deployed in the arena
(the real grid) where the fuel exists, and its tilts are noise. The design guarantees the router
cannot represent the structure whose exploitability is being tested, then reports its failure as
evidence about the arena. The FAIR_RERUNS "routers lose even fairly" (-0.008 at T=12) is exactly what
noise-tilting a tuned schedule produces; it is uninformative about adaptivity.

**Minimal corrective experiment.** Fit the value/gain model by LEAVE-ONE-USER-OUT **on the judged grid
itself** (166-train/7-eval folds preserve the firewall for the eval user while retaining real
answer-taste structure), and re-run the P3/fair-reruns router. If the LOUO-grid-trained V(q|z) still
ties, that is real evidence; the current tie is not.

---

## FINDING 3 — no tested policy conditions on the taste content of answers to choose taste-adjacent questions (RANK 3)

**The flaw.** Inventory of every arm across the seven experiments:

- `r-blind` (Stage C): value = **1/rank in the static schedule** (`battery_stage_c.py:282-320`), tilt
  = kmap LR from answered/refused item events only — which are constant 1 in-pool (Finding 1).
- `r-value-blind` / `r-value+k` (P3, fair-reruns): tilt of the static rank by a GBM whose features are
  `[channel-onehot, fid-onehot, |v|, cos(z, emb), turn]` (`fair_reruns.py:71-77`) — **signed answer
  value discarded (|v|)**, no candidate identity, one scalar of taste, trained per Finding 2.
- `r-vivid` (T3): swap within ±eps value, belief = population k2-rate only (T2 gate failed).
- `u-table`: sorts by the user's own signed answer value — an answer-peek heuristic, not a policy.
- tree (TREE_VS_STATIC): conditions only on answer CLASS (pos/neg/noclue) of popularity-graded probes,
  grown on 86 users (see Finding 6).
- `r-learned`: GBM combiner over the same population features (Finding 2).

**What the fuel numbers say SHOULD work (stated precisely).** The measured structure is: (i) taste-NEAR
questions are more answerable at know_well within popularity tiers (+0.027 item-mid, +0.148 concepts);
(ii) per-user knowledge is predictable from other users' grids (LOUO-MF asymptote 0.938/0.956,
+0.134/+0.054 over popularity). The implied policy: fold the first 2–4 answers → posterior taste
direction; then select **scarcity-habitat questions (rate 0.2–0.7) ranked by LOUO-MF-predicted
P(know_well | taste-so-far) × taste-alignment** — descending into the user's known niche, where a
know_well graded answer on a niche concept moves the latent in a user-specific direction a static
cannot pre-commit to, and where wrong guesses cost refusal-turns (real risk, real signal). No arm in
the program implements any part of this, and Findings 1–2 make it inexpressible/unlearnable anyway.

**Minimal corrective experiment.** The Finding-1 arena + a hand-built (not learned) version of the
above policy (LOUO-MF already exists from Stage B; taste-alignment = cos(z_t, emb)). One run, local.

---

## FINDING 4 — the fold is an unstable, saturating instrument trained on a single static-like answer mix (RANK 4)

**The flaw + evidence.**
- **No accumulation:** G3 (FOLD_V2_REPAIR.md): 1 answer 0.2057 vs 16 answers 0.2070 — the marginal
  value of answers 2–16 is ≈ 0 in aggregate. Stage-C cohort curves *decline* after turn ~6
  (s-item: 0.2169 → peak 0.2256@t6 → 0.2213@t12; s-concept peaks at t4).
- **Ceiling:** G4 FAIL — fold full-profile 0.359 vs native RecVAE 0.494. The interview instrument
  reaches ~0.22 of a 0.49-reachable target; almost all catch-up capacity is discarded by the fold.
- **Choice-instability (fresh probe, v2 fold, arena pool):** within-user latent divergence between two
  different 8-answer sets = **0.668**, vs between-user divergence **0.691** — the latent encodes
  *which questions were asked* nearly as strongly as *who answered*. Per-user NDCG@10 sd across four
  random equal-size 8-sets = **0.061**; best-of-4-random minus mean-of-4 = **+0.077** per user.
- **Training distribution:** the fold was trained on ONE fixed slot mix (`CALIB.slot_mix`,
  `i25_fold_v2.py:55`) — item-data 0.25 / item-llm 0.15 / concept 0.40 / attr 0.20 — i.e. on a
  static-questionnaire-like composition, never on policy-diverse compositions.

**How it masks adaptivity.** The +0.077 random-selection headroom shows per-user question choice moves
NDCG enormously through this fold (consistent with u-clair's +0.040) — the prize is large. But the
response surface is idiosyncratic wobble the fold was never trained to make consistent; any policy
whose answer-set composition drifts off the training mix is OOD for the fold and is charged an
instrument penalty that looks exactly like "adaptive deviation loses". Statics-optimal is partially
self-fulfilling: the instrument was fit to the statics' answer distribution.

**Minimal corrective experiment.** (a) Retrain the fold with randomized compositions (channel-mix
Dirichlet, k=1..16) and re-run one router contrast; (b) cheap diagnostic first: fold the SAME evidence
in permuted order/composition bins and quantify the per-composition penalty (extends my probe, <30 min).

---

## FINDING 5 — the verdict metric (anytime NDCG@10, T=12) cannot pay back exploration and looks where the fuel isn't (RANK 5)

**The flaw.** "anytime" = mean of per-turn NDCG over turns 1..T (`battery_stage_c.py:411-415`,
`SC/eval_*_peruser` + `pt[:, :T].mean(axis=1)` everywhere). With curves saturating by turn 4–6 and
then declining (Finding 4), ~all of the anytime mass is decided in turns 1–4: an adaptive policy that
spends 2–3 early turns probing is charged full price with almost no later room to repay (the
environment stops rewarding information after ~turn 6). The greedy statics are built by
prefix-consistent cohort-mean maximization (`build_greedy`) — they *directly optimize* the anytime
objective the verdicts use; routers do not. Second, NDCG@10 full-catalog under a frozen RecVAE scorer
is a head metric; the fuel lives in niche/tail structure (Stage A habitat), and per-user descent, if
achieved, would first register at deeper cutoffs and on tail items.

**How it masks adaptivity.** Metric asymmetry (baseline optimizes the verdict functional, challenger
doesn't) + insensitivity where per-user differences live.

**Minimal corrective experiment.** Re-read existing per-turn JSONs at endpoint-only and re-run one
contrast with NDCG@50 and tail-stratified NDCG@10 (all local; the per-turn matrices are already
computed).

---

## FINDING 6 — baseline asymmetry and an unreliable greedy comparator (RANK 6)

**The flaw.**
- Even the "fair" statics are constructed on 86–87 **study users' actual grid answers AND their
  held-out NDCG targets** (construction half), while routers are population-firewalled — the
  R_LEARNED E7 table states the asymmetry. "Static beats router" conflates "adaptivity worthless"
  with "study-cohort information beats population information".
- The static family is high-variance cohort fitting: A-built vs B-built schedules share only 4–7/12
  questions (STATIC_CONTAMINATION.md schedule-divergence table); absolute static scores swing 0.21–0.25
  across halves; contamination itself was +0.028 (caught late — all pre-2026-07-09 verdicts were
  against an inflated baseline).
- Greedy is not even reliably the best static: in Stage C the greedy over the FULL mixed pool chose
  all-concepts (0.2166) and *lost to its own subset-pool* s-item (0.2226); in FAIR_RERUNS s-mixed-fair
  beat s-item-fair (+0.012) yet routers were anchored to and judged against s-item.

**How it masks adaptivity.** Router deficits of −0.003..−0.009 are read as "adaptivity loses" while
the comparator family itself wobbles by ±0.03 across cohorts and ±0.012 across families; and the
tie-by-construction anchoring means routers inherit whichever (possibly wrong) family they were
anchored to.

**Minimal corrective experiment.** Anchor the router on s-mixed-fair (the program's own stronger fair
static) — one flag in `fair_reruns.py`; report both anchorings.

---

## FINDING 7 — power: arm-level "ties" are non-answers at the plausible effect size (RANK 7)

**The evidence.** Arm-level contrasts with genuinely different schedules at n=173 have CI half-widths
~0.015–0.017 (R_LEARNED pooled: -0.0089 [-0.0244,+0.0073]); per-estimate n=86 half-widths ~0.03. The
plausible realizable adaptive edge — bounded above by u-clair +0.040 and by the +0.077 selection
headroom, and plausibly ~0.01–0.02 after discovery costs — sits at or below the arm-level MDE. Only
the tie-anchored tilt contrasts reach ±0.005 sensitivity, and those are precisely the arms shackled to
the static (they can only detect deviations of a policy forbidden from deviating much). n=173 users at
per-user answer-set NDCG sd 0.061 (probe) is thin; per-user pairing helps only when arms share most
answers, i.e. only for non-adaptive policies.

**Minimal corrective.** Power analysis written into the pre-registration (MDE at n=173/300 for
schedule-divergent arms); repeated eval over multiple grid-cell resamples per user to shrink
instrument variance; report "tie" only with the MDE alongside.

---

## FINDING 8 — no legitimate ceiling survives, so "prize" statements are unanchored (RANK 8)

u-table sorts by the user's signed answer value (loved-first) — neither an answerability oracle nor an
NDCG ceiling (its −0.043 tells nothing; a different value-ordering confound was even used vs the
static, per the Stage-C retraction header). u-clair is target-peeking (retracted). The program
therefore has **no information-bounded ceiling measurement at all**: nothing bounds what a policy that
knows the user's full GRID (but not the held-out targets) could achieve. The "adaptive prize ≈ 0"
conclusions elsewhere in the program inherit this gap.

**Minimal corrective experiment.** Answer-peek-only oracle: per user, greedy over the pool by 1-step
realized fold-NDCG using the user's own grid values but *selecting* via a model score that never sees
held-out targets... more simply: grid-clairvoyant (knows all answers, not targets) beam search over
12-question sets, evaluated once on held-out. Upper-bounds realizable adaptivity legitimately.

---

## FINDING 9 (minor) — selection-time features peek at the arena answer value

`fair_reruns.make_router.feats` uses `v = rec["val_arr"][cd]` — the user's actual (not-yet-asked)
arena answer — as `|v|` at selection time, while the model was trained on sampler-generated values.
Favors routers slightly (so not masking per se) but confirms label/arena feature mismatch (Finding 2)
and is an E3-privilege inconsistency worth fixing before any positive result is claimed.

---

## THE ONE EXPERIMENT TO RUN FIRST

**Habitat arena, fuel-trained router, both metrics.** Rebuild the 180-candidate pool with a
stratified design (per channel: 20 top-coverage + 40 population-selected from the 0.2–0.7 answer-rate
habitat). Split-construct s-item/s-mixed fair statics on it (existing `SC.build_greedy_sub`). Run ONE
router: LOUO-MF knowledge belief (already built, Stage B) × taste-alignment cos(z_t, emb), warm-started
tie-by-construction, with the value model refit LOUO **on the grid** (Finding 2 fix). Report endpoint
AND anytime NDCG@10 + NDCG@50 at T=12/24, paired bootstrap, both anchor families. Every ingredient
already exists in the repo; it is one local run. This single experiment simultaneously removes the
three highest-ranked masks (pool, fuel-free training world, no-taste-policy) and would either produce
the program's first fair adaptive win or a *meaningful* tie — the current ties are conditional on an
arena where adaptivity was excluded by construction.
