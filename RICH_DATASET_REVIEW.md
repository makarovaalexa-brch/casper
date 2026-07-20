# RICH ANSWERER DATASET — precise spec (for new-recommender input design)

Compiled 2026-07-12 from the actual code. Every claim is cited `file:line`. Where the code is
ambiguous it is FLAGGED as such — nothing here is invented.

Primary sources:
- `scripts/arena_core.py` — the gated v2.1 answerer world + `_gen_user` answer table.
- `scripts/dans_v21.py` — the v2.1 REPAIR bundle (equate / value / dials) that FIT the models.
- `scripts/dans_stages.py` — `know_probs`, `value_probs`, `_sample_cat`, `generate_population`,
  `population_split`.
- `scripts/dans_build.py` — enums (`KLAB/VLAB/VSTAR/KIDX/VIDX`), `study_ids`, `load_173`, split.
- `scripts/i25_fold_v3_sampler.py` — the fold token constants (`TYPE_*`, `LVL_*`, `KIND_*`, `FID_*`,
  `CENTERED_FOLD`).
- `scripts/arena_policies.py` — the interview runner + selection-strategy ladder.

---

## 1. The two graded axes + refusals — EXACT encoding

### Implicit KNOWLEDGE axis — 3 levels
`dans_build.py:48-50`
```
KLAB = ["no_clue", "rough_idea", "know_well"]      # KIDX = {no_clue:0, rough_idea:1, know_well:2}
```
Stored as `int8` in the answer table field `know`, one value per question (`arena_core.py:207-208`,
`_gen_user`). Interpretation of the numeric code:
| code | label | meaning | answered? |
|--|--|--|--|
| 0 | `no_clue` | **REFUSAL** — user can't/won't answer | NO (`answered()` returns `know>=1`, `arena_core.py:290-291`) |
| 1 | `rough_idea` | knows of it vaguely | yes |
| 2 | `know_well` | knows it well | yes |

### Explicit VALUE axis — 4 levels
`dans_build.py:49,51-52`
```
VLAB  = ["hated", "meh", "liked", "loved"]         # VIDX = {hated:0, meh:1, liked:2, loved:3}
VSTAR = [1.5, 3.0, 4.0, 4.75]                       # representative stars per level (bin midpoints)
```
Stored as `int8` in field `val`. **Sentinel: `val = -1`** means "no value" — used for BOTH
`no_clue` (refusal) AND any omitted cell (`generate_population` docstring `dans_stages.py:635`;
init `arena_core.py:252` `val = np.full(self.nQ, -1)`).

**Value is defined ONLY when knowledge ≥ 1.** In `_gen_user` value is sampled only on the
`nz = seg > 0` positions (`arena_core.py:254-259`); every `no_clue` position keeps `val = -1`.
Answer to review Q3: when `knowledge = no_clue` the value is **undefined (`-1`)**, never a neutral
level. There is no (no_clue, someValue) combination.

So the full per-question emission is one of:
```
(no_clue,  -1)         = refusal  [consumes a turn; folded as an implicit-NEGATIVE token]
(rough_idea, v∈{0,1,2,3})
(know_well,  v∈{0,1,2,3})
```

### Centered value used by the belief fold (not the raw index)
`i25_fold_v3_sampler.py:38`
```
CENTERED_FOLD = {"hated": -1.0, "meh": -1/3, "liked": +1/3, "loved": +1.0}
```
`arena_core.py:59-60` maps the 4 value indices → these centered floats (`VBIN_CENTERED`). Rated
items instead carry `crval = star − mean(known stars)` (`arena_core.py:262,270`).

### How a cell becomes fold TOKENS (`tokens_for`, `arena_core.py:306-330`)
Fold token = 8-field tuple `(channel_type, kind, level, surprise, fidelity, value, emb, anti_surprise)`.
Constants (`i25_fold_v3_sampler.py:43-46`):
`KIND_IMPL=0, KIND_EXPL=1`; `LVL_ROUGH=0, LVL_KW=1, LVL_NEG=2`; `FID_DATA=0, FID_EASE=1, FID_LLM=2`;
`TYPE_ITEM=0, TYPE_CONCEPT=1, TYPE_ATTR=2, TYPE_ENTITY=3`.
- **know=0 (refusal)** → ONE token: `(ch, IMPL, LVL_NEG, surp, FID_DATA, 0.0, emb, anti)` where
  `anti = clip(-surprise, 0, 8)` (v3.1 anti-surprise rule; the no-clue token **is folded**, not
  skipped — `arena_core.py:317-319`, and the class docstring 22-24).
- **know≥1** → an implicit token `(ch, IMPL, {LVL_KW if know==2 else LVL_ROUGH}, surp, FID_DATA,
  0, emb, 0)` PLUS, if a value exists, an explicit token `(ch, EXPL, LVL_ROUGH, 0, fid, value, emb,
  0)`. Fidelity of the explicit value: `FID_DATA` for rated bank passthrough (real rating),
  `FID_EASE` when `know_well`, `FID_LLM` when `rough_idea` (`arena_core.py:321-329`).

---

## 2. WHICH SET DO WE TRAIN ON — (a) distilled population, NOT (b) the LLM users

**Decision: (a).** The rich signal is the **fitted v2.1 distilled answerer applied to the population
`trU` users at ZERO LLM cost.** The real-LLM study users are QUARANTINED and never touched by the
world.

Evidence:
- The world's answerer is the fitted gated v2.1 model, "Loaded and RUN, not approximated… NO LLM
  calls" (`arena_core.py:4-14`). `_gen_user` deterministically samples each user's full answer table
  from `know_probs` + `value_probs` (`arena_core.py:201-278`).
- Firewall: "population trU users only (study ids excluded); the 173/300 are NEVER touched"
  (`arena_core.py:30-31`). `make_cohorts` draws only from `trU` with `study_ids()` removed
  (`arena_core.py:433`). `CLAUDE.md`: "The 173 LLM-judged / 300 study users are QUARANTINED".

### Exact user counts
`meta.npz` (verified by loading): `nu = 162541` total users, `ni = 18430` items,
`trU` = **161,541** candidate population users, `24,810,483` interactions. Splits `va`/`te` are
500-item holdout sets (item-level, not user).
- **Quarantined study users = 300** (`study_ids()` returns 300; `dans_build.py:432-441` unions the
  answerability grids). Of these, **173** carry full LLM-judged interview cells — the WORKING grid
  `.cache/instrument2/answerer_v1_grid173_WORKING.json` (`load_173`, `dans_build.py:444-482`;
  verified 173 users). These 173 are what the v2.1 models were FIT on (`load_173(uni)` inside
  `stage_equate`/`stage_value`/`stage_dials`, `dans_v21.py:259,584,745`). The remaining ~127 study
  users are the broader main-study answerability grid; all 300 are excluded from training/eval.

### Train / dev split of the population (the arena cohorts)
`make_cohorts` (`arena_core.py:421-455`), defaults `n_train=1000, n_devval=80, n_devtest=160`,
disjoint, all from `trU` minus the 300 study ids. (The `__main__` smoke uses 30/10/20; the
production sizes are the defaults and are set by the caller — **flag:** the 1000/80/160 are function
defaults, the actual campaign N is whatever the arena driver passes; not hard-coded elsewhere in
these files.) `population_split`/`generate_population` (`dans_stages.py:598-694`) is the bulk path
that can distil the *entire* population into sharded npz — the same recipe, so the training set is
**scalable to all of `trU`** at no LLM cost.

**Summary answer to Q1:** train on (a) the distilled answerer over the population `trU`
(161,541 available; arena cohorts carve train/devval/devtest out of it). Hold (b) the 173 LLM /
300 study users entirely OUT as the quarantined realistic-transfer set. They are never used to
train, val, or test in this world.

---

## 3. Coverage — the askable universe per user

**Every user can be asked EVERY question.** No pools, no coverage sampling (`arena_core.py:15-17`).
The universe is the full deterministic layout `[concept | entity | item]`:
- **1,128 concepts** + **500 attributes/IMDb entities** + **800 bank items** = **2,428 questions**
  (`arena_core.py:132-137,158-159`; the "signed sheet's 2,428").
- Layout offsets: concepts `[0, ntag)`, entities `[ntag, ntag+nent)`, items `[ntag+nent, nQ)`
  (`arena_core.py:134-141`).

### Rated vs unrated items (the disinterest/absence question, Q2)
The value backbone is `t(u,i) = real rating if rated else per-user EASE prediction`
(`arena_core.py:209-251`). So for ANY of the 800 bank items, whether the user rated it or not, the
answerer emits a (knowledge, value):
- **Item the user RATED** (passthrough, `arena_core.py:260-270`, `f["rated_flag"]`): forced
  `know = know_well (2)`; `value` = binned real rating (`3 if star≥4.5, 2 if ≥3.5, 1 if ≥2.5, else
  0`); `crval = star − mean(known)`; explicit fidelity `FID_DATA`. This is ground truth.
- **Item the user did NOT rate**: `knowledge` is **SAMPLED** from the fitted ordinal model on the
  item's fame / co-knowledge / genre-align features (`know_probs` item channel, `dans_stages.py:341-343`);
  it can come out `no_clue`, `rough_idea`, or `know_well`. If it lands `no_clue` → **refusal,
  `val=-1`** (this is how disinterest / absence-of-knowledge is represented). If it lands ≥1 the
  value is sampled from the EASE-predicted taste `t` (`arena_core.py:246-251`).

**So "unrated" is NOT a fixed sentinel.** Absence/disinterest is encoded probabilistically:
the model decides, per unrated item, whether the user has no clue (→ refusal, `val=-1`) or has an
opinion driven by their EASE-predicted taste. There is no separate "not-interested" token; the only
"absence" signal is `know = no_clue` → the implicit-NEGATIVE (`LVL_NEG`) token with anti-surprise
(`arena_core.py:317-319`). Note: item `know≥1` (knows-of-it) is near-saturated (~99% of top-800
items recognized, `dans_v21.py:346-349`), so the discriminating item signal lives at the
`know_well` margin.

---

## 4. Concepts / attributes / entities — how non-item questions work

Non-item questions are asked over a **pop-weighted member bag** of the items in that concept/entity
(`arena_core.py:166-177` `q_emb`; `region_members` `192-198`).

They carry BOTH axes, same encoding as items:
- **Knowledge** (concept/entity channels): sampled from the fitted ordinal knowledge model over
  membership size, pop-weight, taste-align, census + era features (`know_probs`,
  `dans_stages.py:329-340`; entity uses per-cut random effects `ord_prob_percut` 347-359).
- **Value**: sampled from an ordinal model whose feature is the **popularity-weighted mean of member
  t-values** (`ctaste`/`etaste`), plus `has_rated_member`, `log1p(n_rated_member)`, `user_cmean`
  (`arena_core.py:229-245`; fit in `dans_v21.py:620-641`, `stage_value`). Value again defined only
  when knowledge ≥ 1.
- Attributes share the entity channel dial (`i25_fold_v3_sampler.py:101` — `TYPE_ATTR` sigma =
  entity's). In the LLM grid, channel `"attribute"` is folded into `"entity"` (`dans_stages.py:60`,
  `load_173` `dans_build.py:466-467`).

Headline answerability margins differ by channel: concept `k≥1`, entity `k≥1`, item `k≥2`
(`arena_core.py:61`, `dans_v21.py:251`).

---

## 5. Refusals & mehs — representation

- **Refusal** = `know = 0 (no_clue)`, `val = -1`. It is a **distinct knowledge level**, not an
  absence from the table (every question has a `know` entry). In the interview it **consumes a turn
  but the user is never dropped** (E1 rule, `arena_core.py:22-24`, `run_policy` `arena_policies.py:86,
  124-131`). Under the `skip` regime (baseline b3) a refused turn is refunded but its no-clue
  evidence is still folded (`arena_policies.py:129-130`). The `realized_branch` for a refusal is the
  token `"refuse"` (`arena_core.py:297-304`).
- **Meh** = a genuine **VALUE level** (`val = 1`, `VLAB[1]="meh"`, star 3.0). It requires
  `know ≥ 1`. It is not a refusal and not an absence. In the polarity branch,
  `val∈{0,1}` (hated/meh) → `"dislike"`, `val∈{2,3}` → `"like"` (`arena_core.py:302-304`).

---

## 6. The target — held-out liked items (leak-free)

`make_cohorts` (`arena_core.py:436-449`) and `population_split` use the same convention:
- Per user, deterministically permute their rated items with `rng = default_rng(seed*1_000_003 + u)`
  (uid-seeded → independent of cohort size).
- **First half = `known`** (context given to the answerer / EASE); **second half's LIKED items
  (`rating ≥ 4`) = `held`** = the recommendation target (`arena_core.py:445-446`).
- Filters: keep users with `len(known) ≥ 4` and non-empty `held`.
NDCG is scored on `held` with the `known` profile masked out (`ndcg_at_k`, `arena_core.py:65-81`).
EASE training also drops the study users' held-out halves as a leakage guard
(`dans_v21.py:520-534`). **Answer to Q6: yes** — held = the user's own rating≥4 items from the
disjoint second half, never shown during the interview.

---

## 7. Selection strategies + interview / refusal curriculum

The arena is a POLICY arena: "strategies" = the policy ladder that decides WHAT to ask
(`arena_policies.py:14-18`). Interview budget **`Tmax = 24`** turns (default across builders,
`arena_policies.py:239,546,622`; runner `run_policy` iterates `t in 1..Tmax`).

Baseline / strategy ladder:
| tag | strategy | `arena_policies.py` |
|--|--|--|
| b0 | cold (ask nothing) | `154-158` |
| b1 | concept-entropy static (concepts by binary entropy of TRAIN answer rate) | `175-184` |
| b2 | LEARNED static sequence (greedy realized-gain on a construction cohort) | `239` |
| b3 | b2 + skip (refund refusals) + value-ranked tail continuation | `160-172`, `129-130` |
| b4 | BLIND model-based myopic (expected gain in smooth ranking utility `J(z)`) | `66-81` |
| — | ADAPTIVE A (GBM scorer), B (ask-the-gradient), C (CAT/Fisher), D (Golbandi polarity tree) | `16-17` |
| — | CONTEXT arms (labelled, never cited): true-table router, clairvoyant | `15` |

Refusal handling knobs:
- Default regime: refusal = consumed turn, belief unchanged on refusal, user kept
  (`run_policy` docstring `arena_policies.py:86`).
- `skip=True` (b3): the refused turn is refunded (loop continues to pick another) but the no-clue
  token is still folded (`arena_policies.py:129-130`).
- Deployable policies see ONLY the observed dialogue (`asked/answered/branch`), never the ctx table
  (fix #9, `arena_policies.py:11-12,118-119`); labelled/privileged arms get the full ctx.

Interview LENGTH is fixed by `Tmax`; there is no stochastic length sampling in the arena world (the
per-question refusal *rate* is what varies, driven by the sampled `know`). **Flag:** the earlier
sampler world (`i25_fold_v3_sampler.py`, used to TRAIN the fold) DID sample interview length /
refusals via `BASE_ANS`/`BASE_KW` per-channel rates (`i25_fold_v3_sampler.py:50-52`); the fold was
trained there and deployed FIXED on the gated arena (documented distribution shift,
`arena_core.py:18-21`).

---

## 8. Data files (paths + shapes)

| file | what | shape / size |
|--|--|--|
| `data/movielens/.cache/ml25m/meta.npz` | ratings backbone | `uu,ii,rr` = 24,810,483 interactions; `cnt` (18430), `mu` scalar, `ni=18430`, `nu=162541`, `trU` (161,541), `va`/`te` (500 each), `keepI` (18430) |
| `.cache/dans/models_v21.json` | fitted v2.1 knowledge + value ordinal models (equated σ_u) | JSON; per-channel `mu/sd/theta` |
| `.cache/dans/ease_v21.npz` | EASE value backbone | `B` (~9352×9352 f32), `universe`, `mu` |
| `.cache/dans/equate_v21.json` | variance decomposition + corrected fuel ICCs | JSON |
| `.cache/i25_fold_v31_best.pt` | FOLD-V3.1 belief encoder (val 0.4356) | torch ckpt |
| `.cache/instrument2/answerer_v1_grid173_WORKING.json` | 173 LLM-judged users (QUARANTINED) | 173 users, labeled cells + rated data cells |
| answerability grids (union) → `study_ids()` | 300 quarantined study users | 300 ids |
| `.cache/arena/answers2_*.npz` | cached per-cohort answer tables (`know,val,crval,surp,known_hash`) | (n_users × nQ) |

Universe (`DB.Universe`): `ntag=1128` concepts, `nent=500` entities, `nbank=800` bank items → `nQ=2428`.

---

## TL;DR (the load-bearing facts)

- **(knowledge, value, refusal) encoding:** `know ∈ {0 no_clue, 1 rough_idea, 2 know_well}` (int8);
  `val ∈ {0 hated, 1 meh, 2 liked, 3 loved}` (int8), `val = -1` iff `know = 0` or omitted. Value is
  **undefined when know = no_clue**. Refusal = `(know=0, val=-1)`, a real level, folded as an
  implicit-NEGATIVE (`LVL_NEG`) token with anti-surprise; it consumes a turn but never drops the
  user. "meh" is a value level (`val=1`, star 3.0), requiring `know≥1`. Fold value is centered:
  hated −1, meh −1/3, liked +1/3, loved +1. (`dans_build.py:48-52`, `i25_fold_v3_sampler.py:38,43-46`,
  `arena_core.py:252-330`).
- **Which set + coverage:** train on the **distilled v2.1 answerer over the population `trU`
  (161,541 users)** at zero LLM cost — arena cohorts (default 1000/80/160 train/devval/devtest) are
  carved from `trU` minus the 300 quarantined study users. The **173 LLM-judged / 300 study users
  are QUARANTINED** (held out for realistic transfer, never trained/eval'd). Askable universe is the
  **full 2,428 questions for every user** (1128 concept + 500 entity + 800 item) — no pools.
  (`arena_core.py:4-31,132-159,421-455`).
- **Unrated / disinterest:** NOT a fixed sentinel. For a rated item the answer is passthrough
  (`know_well`, real rating). For an **unrated** item the knowledge is **sampled** from the fame/
  co-knowledge model; if it lands `no_clue` → refusal `val=-1` (this IS the disinterest/absence
  signal); otherwise value comes from the user's EASE-predicted taste. (`arena_core.py:209-270`,
  `dans_stages.py:341-343`).
