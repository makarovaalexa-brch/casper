# Signed-Latent Recommender (α=0) — Honest Analysis Scorecard

**Model:** `.cache/signed_latent/a0c_best.pt` — SignedAE, latent d=512, signed per-item value input +
consumed-mask channel, `L_pos` only (α=0, no dislike-margin term). Best val_full 0.4961 @ep10.
**Eval:** full-catalog held-liked NDCG@10 (rank each held-liked item vs all 18,430 minus revealed —
no sampled negatives). Arena half-split, seed-avg {1,2,3,7,11}. Cohort = te(500) **minus 300
quarantined study users** (~200 users/seed, ~1000 pooled). Bootstrap 95% CIs (2000 resamples).
EVAL-ONLY — no training. Artifacts: `.cache/signed_latent/analysis.json`, `ruler.json`,
`signflip_a0.json`, `genre_negativity_prevalence.json`.

**Ruler anchors (same harness):** MOSTPOP 0.285 · pre-VAE ~0.33 · **EASE 0.510** · **RecVAE d512 0.526**.
Ruler trustworthy (EASE≈RecVAE, both >0.45).

---

## Whole-battery verdict: **PASS (with two honest nulls)**

The model is **strong** (full-profile 0.495, EASE/RecVAE-competitive), the **cold→full k-curve is
strictly monotone**, **every hygiene gate passes**, and the **attribute (genre) sign channel is real
and specific** (IG2 +0.257 flip gap, 3× the control displacement). The two nulls are stated plainly:
**item-level dislike is inert**, and **folding negatives (disliked OR disinterested genres) as extra
input does NOT buy NDCG — it significantly HURTS it** at every k and weight. Dislike is *expressible*
(the model reads genre sign) but *not useful as an elicitation channel for held-liked ranking*.

---

## 1. STRENGTH  — PASS

| Metric | NDCG@10 | 95% CI |
|---|---|---|
| Full-profile, likes-only, **FULL** | **0.4954** | [0.4782, 0.5124] |
| Full-profile, likes-only, **TAIL** | 0.3110 | [0.2940, 0.3275] |

Confirms the ~0.496 posts. On the ruler this sits just under EASE (0.510) / RecVAE (0.526) and far
above MOSTPOP (0.285) — a genuinely strong instrument that *also* carries a latent and reads signed input.

## 2. k-CURVE BY TURN  — PASS (strictly monotone cold→full)

Curriculum strategy-mix **accumulating** interviews (signed values; k='full' = all profile items).

| k | 0 (cold) | 1 | 2 | 4 | 8 | 16 | 32 | full |
|---|---|---|---|---|---|---|---|---|
| NDCG@10 | 0.2688 | 0.3029 | 0.3371 | 0.3738 | 0.4093 | 0.4442 | 0.4688 | 0.4988 |

Monotone non-decreasing at every step. Cold start (k=0, empty interview) already returns 0.269 — the
popularity prior beats MOSTPOP's 0.285? No: 0.269 < 0.285 (the prior is slightly below raw popularity,
as expected for a learned reconstruction prior). Each turn adds signal; full ≈ likes-only strength.

## 3. REFUSAL ROBUSTNESS  — PASS (graceful degradation)

k-curve under refusal (each revealed answer dropped with prob r):

| refusal | k1 | k2 | k4 | k8 | k16 | k32 | full |
|---|---|---|---|---|---|---|---|
| 0%  | 0.303 | 0.337 | 0.374 | 0.409 | 0.444 | 0.469 | 0.499 |
| 15% | 0.297 | 0.320 | 0.361 | 0.401 | 0.431 | 0.457 | 0.494 |
| 30% | 0.294 | 0.308 | 0.349 | 0.389 | 0.423 | 0.449 | 0.479 |
| 50% | 0.288 | 0.305 | 0.327 | 0.370 | 0.402 | 0.428 | 0.460 |

Degrades smoothly; even at 50% refusal the full interview loses only ~0.04. No collapse.

## 4. DISLIKE GATES (attribute channel)

### 4a. IG2 genre sign-flip  — PASS (real + specific)

Member-bag ± (top-20 popular members revealed as +1 vs −1; measured on 300 held members). Flip gap =
`like_pct − dislike_pct` (percentile of held members), bootstrap CI over held members.

| Genre | like_pct | dislike_pct | **flip gap** | 95% CI | ctrl displ. |
|---|---|---|---|---|---|
| Children | 0.963 | 0.493 | **+0.470** | [+0.443,+0.498] | 0.066 |
| Horror | 0.973 | 0.578 | **+0.394** | [+0.370,+0.418] | 0.050 |
| War | 0.902 | 0.548 | **+0.354** | [+0.326,+0.382] | 0.055 |
| Sci-Fi | 0.980 | 0.761 | **+0.219** | [+0.201,+0.240] | 0.072 |
| Documentary | 0.945 | 0.743 | **+0.202** | [+0.187,+0.217] | 0.120 |
| Romance | 0.964 | 0.811 | **+0.153** | [+0.136,+0.171] | 0.075 |
| Action | 0.978 | 0.829 | **+0.148** | [+0.132,+0.166] | 0.100 |
| Comedy | 0.977 | 0.862 | **+0.116** | [+0.103,+0.130] | 0.104 |

**mean flip gap +0.257** (all CIs exclude 0), **mean control displacement 0.080** — flipping a genre's
sign moves *that* genre's members ~3× more than untouched genres. The model reads genre sign, and does
so specifically. (Reproduces the sign-flip diagnostic's −26pt mean, now with CIs.)

### 4b. DISLIKE-HELPS (fold disliked genres as NEGATIVE input)  — **NULL / significantly NEGATIVE**

Baseline = k random liked items (signed +). Fold = add top-20 popular members of the user's
*actually-disliked* genres (K≥3, mean≤2.5 and ≤user_mean−0.7) as −w input. Weight-swept; paired
bootstrap over **covered** users. Coverage ~15% (disliked genres are rare — 1.45% of user-genre pairs).

| k | coverage | best (least-harmful) lift | 95% CI |
|---|---|---|---|
| 2 | 151/1000 (15.1%) | **−0.122** (w=0.5) | [−0.154, −0.091] |
| 4 | 150/993 (15.1%) | **−0.120** (w=0.5) | [−0.149, −0.091] |
| 8 | 129/903 (14.3%) | **−0.080** (w=0.5) | [−0.110, −0.050] |

Every k and every weight (0.25/0.5/1.0) gives a **significantly negative** lift. Folding disliked
genres in as negative input *hurts* held-liked NDCG. Mechanism: the L2-normalized input dilutes the
positive likes' magnitude, and pushing genre regions down does not help rank the held *likes*.

### 4c. ITEM-level dislike  — **INERT (honest null)**

From `signflip_a0.json`: flipping a single item's sign barely moves its own neighbourhood —
mean nn-percentile drop **+0.0013**, mean top-10 overlap 0.46, mean rank-corr **0.946**. Item-level
dislike is not expressed; the sign channel lives at the **attribute (genre)** level, not the item level.

### 4d. TASTE SEPARATION  — moderate

Encode liked-only vs disliked-only profiles: **cosine 0.515** (not collapsed, not orthogonal),
euclidean margin 13.4 (n=~190). Likes and dislikes land in distinguishable but overlapping latent
regions — consistent with a genre-level (not item-level) dislike signal.

## 5. DISINTEREST-AS-NEGATIVE (author's key test)  — **NULL / significantly NEGATIVE**

Per user, identify **disinterested** genres = systematically under-engaged vs population base rate
(profile share ≤ 0.25× population popularity share, genre share ≥5%, ≥10 profile ratings; top-3 most
under-represented). Fold their popular members in as −w input. Does "I'm not interested in X" buy NDCG?

| k | coverage | best (least-harmful) lift | 95% CI |
|---|---|---|---|
| 2 | 139/1000 (13.9%) | **−0.089** (w=0.5) | [−0.116, −0.063] |
| 4 | 135/993 (13.6%) | **−0.076** (w=0.5) | [−0.100, −0.051] |
| 8 | 96/903 (10.6%) | **−0.049** (w=0.5) | [−0.079, −0.020] |

**Answer: NO.** At every k and weight the lift is significantly negative. Eliciting "not interested in
X" and folding it as negative input does *not* raise held-liked NDCG — it lowers it (same dilution
mechanism as 4b, milder at higher k). Prevalence context (model-free, all 162k users): DISINTERESTED =
33.5% of user-genre pairs (K3), ENGAGED-DISLIKED only 1.45%, ENGAGED-LIKED 16.6%. So disinterest is
abundant in the data — but this α=0 item-level model has no productive way to *consume* it for
held-liked ranking.

## 6. BEHAVIORAL / ILLUSTRATIVE

**IG1 genre purity** (top-10 under a "like genre" bag; excluding the bag):
- **Horror 10/10** members: Misery, Poltergeist, Halloween, Carrie, The Omen…
- **Documentary 10/10** members: Grizzly Man, Enron: Smartest Guys, Bus 174, Devil & Daniel Johnston…
- **Western 7/10** members: Outlaw Josey Wales, High Plains Drifter, High Noon, Magnificent Seven… (3
  leak to war/action neighbours — Saving Private Ryan — a genre-adjacency slip, reported honestly).

**IG3 franchise coherence** (single liked film → neighbours):
- *LOTR: Fellowship* → LOTR: Two Towers, LOTR: Return of the King, Inception, Star Wars I… (coherent).
- *Star Wars IV* → Toy Story, Star Wars V, LOTR, Matrix, Star Wars… (mostly coherent, popularity-tinged).
- *Toy Story* → Sense and Sensibility, Star Wars I, Apollo 13, Beauty and the Beast… (weak — a single
  cold item is dominated by the popularity prior; coherence is partial, reported honestly).

**IG4 graded-value monotone sweep** (Silence of the Lambs value −1→+1, tracking Horror held-member
percentile): 0.826 → 0.839 → 0.873 → 0.887 → 0.896 — **strictly monotone increasing**. The signed value
input is genuinely graded, not just a sign switch.

**IG5 graded-confidence / knowledge-vs-value G-GoT — NOT APPLICABLE.** The model input is signed
per-item VALUE + a *binary* consumed-mask (presence). There is no separate confidence/knowledge
channel, so graded-confidence tests do not apply to this architecture. (Not faked.)

## 7. HYGIENE GATES  — ALL PASS

| Gate | Result | PASS |
|---|---|---|
| G-intercept (empty interview → prior) | top-10 = Shawshank, Star Wars IV, Forrest Gump, Silence, Pulp Fiction… (popularity prior); fixed max-dev prior | ✔ |
| G-order (shuffle input order) | max |Δscore| = 0.0 (n=40) | ✔ |
| G-falsify-count (duplicate revealed answers) | max |Δscore| = 0.0 (n=40) — set-valued input | ✔ |
| G-no-profile-leak (alter unrevealed items) | max |Δscore| = 0.0 (n=40) — output is a pure function of revealed input | ✔ |
| Monotone accumulation (k-curve) | non-decreasing cold→full | ✔ |

---

## Bottom line

A **strong, well-behaved, EASE/RecVAE-competitive** cold-start instrument with a **real, specific
genre-level sign channel** and clean hygiene. Two honest nulls define its limits: **item-level dislike
is inert**, and **negative fold-in (disliked or disinterested genres) does not help — it significantly
hurts held-liked NDCG**. The model can *read* "I dislike/ignore genre X" (IG2), but folding that as
input does not improve ranking of what the user *does* like. Dislike here is expressible, not useful.
