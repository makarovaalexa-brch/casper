# Answerer v1 — prerequisite builds (P1–P4) — sign-off report

Date: 2026-07-08. Author of builds: prerequisite pass per `casper/ANSWERER_V1_DESIGN_REVIEW.md` §7.
**NO LLM API calls were made.** All builds deterministic, from local ML-25M caches + the free IMDb TSV dumps.
Nothing here is "locked" until the author signs off (§6 R3/R4).

Canonical dense-id space (from `scripts/llm_answerability_gate.load_data`): dense id `j ∈ 0..18429`;
`movieId = keepI[j]` (`keepI` sorted ascending, so concepts alignment is consistent);
`cnt[j]` = train-like ratings count = the popularity used thesis-wide (`D["cnt"]`).

## Deliverables (all writes verified on disk)
| Path | Bytes | Contents |
|------|-------|----------|
| `casper/answerer_schema.json` | 4,878 | P4 machine-readable schema (single source of truth) |
| `casper/answerer_schema.md` | 4,525 | P4 human doc + sign-off checklist |
| `.cache/instrument2/attr_membership.json` | 4.88 MB | P1 IMDb attribute entity universe + franchises + adult flags |
| `.cache/instrument2/item_lists.json` | 47.8 KB | P2 top-800/1000/2000 dense-id lists + stats |
| `.cache/instrument2/tag_questions.json` | 294 KB | P3 all 1,128 tag question templates + membership + priors |
| build scripts | — | `scripts/instrument2/answerer_prereq_items_tags.py`, `answerer_prereq_imdb.py` |

---

## P1 — IMDb JOIN (attributes)  ✅
IMDb dumps downloaded from `datasets.imdbws.com` (2026-07-07 build, ~1.4 GB total; streamed, never
loaded whole into RAM), gzip-integrity verified. Join via `data/movielens/links.csv` (imdbId → `tt%07d`).

- **Join coverage: 18,430 / 18,430 universe items linked to IMDb (100.0%).** `title.basics` matched
  18,387 (43 items have no current basics row — deleted/merged tconsts; they still get crew/cast).
- **Attribute entity universe** (threshold = films in the 18,430 item universe):

  | type | threshold | entities kept | pre-threshold candidates |
  |------|-----------|---------------|--------------------------|
  | director | ≥3 | **2,123** | 8,363 |
  | actor/actress (billed order ≤4) | ≥5 | **2,919** | 25,677 |
  | composer | ≥3 | **1,336** | 6,207 |
  | writer | ≥3 | **3,653** | 18,598 |

  Directors from `title.crew`; cast/composer/writer from `title.principals` (cast filtered to
  `category ∈ {actor,actress}` and `ordering ≤ 4`). Names resolved from `name.basics` (52,087/52,087).
  Each entity carries: `name`, `member_dense_ids`, `member_movieIds`, `n_movies`, per-entity
  `popularity` = Σ member `cnt`.
- **Franchise detection: 1,159 groups** (≥2 items sharing a conservative title-stem: strip year,
  cut at `: / Part / Chapter / Episode / Vol`, strip trailing roman/number, normalize trailing article).
  TMDb collection ids were **not** used — `links.csv` carries `tmdbId` but not a collection id, and
  fetching collections needs the TMDb API (out of scope, no network-API spend). Title-stem only, documented.

### Sanity — join-gap coverage (target <2% missing)
| metric | count | % of universe | verdict |
|--------|-------|---------------|---------|
| no director | 52 | **0.28%** | ✅ <2% |
| no billed cast (order ≤4 actor/actress) | 1,063 | 5.77% | ⚠ see note |
| **no director AND no cast** (truly unattributed) | 49 | **0.27%** | ✅ <2% |

**Note on the 5.77% no-cast:** it is **not a join failure** — 771 of the 1,063 are Documentaries
(no billed fictional cast: narrators/self), plus stand-up/concert Comedy (163) and Animation (49).
Only **7 of the top-1,000** and **31 of the top-2,000** probing-bank items have no billed cast, so
the recognizable item bank is essentially fully cast-covered. The meaningful "no attributes at all"
figure is 0.27%. **Author's eyes: accept the cast filter as-is (documentaries legitimately have no cast)?**

### Sanity — spot-check (eyeballed, all correct)
- Directors: Spielberg (34 films), Tarantino (12: Pulp Fiction, Reservoir Dogs, Jackie Brown…),
  Nolan (10: Memento, Batman Begins…), Scorsese (35: Taxi Driver, Goodfellas…), Ridley Scott,
  Coen, Fincher, Cameron, Zemeckis, Peter Jackson — all filmographies correct.
- Actors: Tom Hanks (53), De Niro (80), Morgan Freeman (57→Shawshank), Harrison Ford (38→Star Wars),
  Brad Pitt, Bruce Willis, Matt Damon, DiCaprio — all correct.
- Franchises: Star Wars (9), LOTR (4), Godfather (3), Terminator (4), Toy Story (4), Back to the
  Future (3), Die Hard (3), Pirates (5), Kill Bill (2) — all correctly grouped. One benign false
  positive class: same-title remakes (e.g. *The Fugitive* 1993 + 1947) group together — conservative, harmless.

---

## P2 — ITEM LIST (probing bank)  ✅
Top-N ML-25M items by ratings count (`cnt`). Primary **N=1000**; N=800 and N=2000 variants emitted.

| list | min ratings-count (recognition floor) | floor percentile | franchise share | dups | missing genome | adult |
|------|----------------------------------------|------------------|-----------------|------|----------------|-------|
| top800 | 3,701 (*Touch of Evil* 1958) | 95.7th | 10.3% | 0 | **0** | 0 |
| **top1000** | 2,879 (*Prometheus* 2012) | 94.6th | 10.6% | 0 | **0** | 0 |
| top2000 | 1,087 (*A Prophet* 2009) | 89.2th | 9.7% | 0 | **0** | 0 |

- **No duplicates; no items missing genome scores in any list** (genome fully covers the popular
  head — the 4,614 universe items lacking genome are all below rank 2000). **No adult titles** in any
  list (the only adult-flagged item in the whole universe is *Sex: The Annabel Chong Story* (1999),
  a documentary, dense id absent from all banks).
- Decade & genre balance (top1000): decades span 1920s→2010s (peak 1990s=350, 2000s=257, 1980s=141);
  all 19 genres present (Drama 495 … Documentary 6). Reasonable, popularity-natural (not designed strata).
- **Absorbable already-judged (user,item) cells** (reuse from cached grids — same judge/split):

  | list | gate_ml25m | mainstudy | arena3 | **total cells** |
  |------|-----------|-----------|--------|-----------------|
  | top800 | 3,368 | 4,392 | 5,950 | **13,710** |
  | **top1000** | 3,682 | 4,903 | 6,800 | **15,385** |
  | top2000 | 4,145 | 6,256 | 10,965 | **21,366** |

  (Total judged item cells available across grids: gate 10,800 / mainstudy 20,918 / arena3 23,800.)

---

## P3 — TAG LIST (all 1,128 genome tags)  ✅
Every genome tag gets a mechanical question template + membership + prior. **No pruning** (author rule:
pruning is empirical, pilot-only); awkward phrasings are flagged, not removed.

- **1,128 tags, 54 flagged awkward** (contains digits / parentheses / ≤2 chars / meta-list keyword
  like "imdb top 250", "criterion", "afi 100", "oscar (best picture)"). Meta tags get a distinct
  template ("Do you specifically seek out films known as '{tag}'?"); adjectives → "Do you like {tag}
  movies?"; nouns/themes → "Do you like movies about {tag}?" (adjective vs noun by a small
  suffix/word rule set). Some non-flagged phrasings are imperfect grammar (e.g. "movies about
  mentor" should be "mentors") — deliberately left for the pilot to judge, per the no-pre-prune rule.
- **Membership threshold = relevance ≥ 0.50**, reused from the project concept machinery
  (`prep_concepts_ml25m.COV_THRESH`), counted over the 18,430 item universe.
- Membership-size distribution: min 0, median 249, mean 545, max 13,747 (tag "mentor"); 10th/50th/90th
  pct = 33/249/1,386; **1 tag has zero members** at the 0.5 threshold.
- `answer_rate_prior` per tag = membership-weighted popularity (Σ member `cnt` / total `cnt`), for
  pilot stratification only (not pruning).

---

## P4 — THE SCHEMA FILE (thesis-wide lock)  ✅ (awaiting confirmation)
`casper/answerer_schema.json` (+ `.md` doc) — the single source of truth the judge prompt, cache
format, and fold interface all import. Encodes Decisions D & E:
- **knowledge ∈ {no_clue, rough_idea, know_well}** — collapses D3's separate `vividness{strong/vague}`
  × `can_answer{yes/no}` into one ordered 3-level field (`no_clue`=can't answer, `rough_idea`=vague,
  `know_well`=strong). `can_answer` is derived = `(knowledge != no_clue)`. **Author's eyes: OK to
  collapse to 3 levels, or keep vividness and can_answer as two separate fields as literally in D3?**
- **value ∈ {hated, meh, liked, loved}**, present iff `knowledge != no_clue`; centered fold
  **−1, −⅓, +⅓, +1**.
- **fold confidence weight = f(knowledge)** = {know_well:1.0, rough_idea:`w_rough`, no_clue:0.0};
  **`w_rough` default 0.5, documented as a fit KNOB** (not a constant).
- **rating→scale**: ≥4.5 loved, 3.5–4 liked, 2.5–3 meh, ≤2 hated (rated ⇒ knowledge=know_well).
- **channels** {item, concept, attribute, pair, recall}; **dual-arena rule** Arena-F (LLM-valued,
  PRIMARY) / Arena-R (rated-only lower bound); fidelity σ = 0.70 stars.

---

## AUTHOR SIGN-OFF SECTION
1. **Item-list N (§7 Decision A).** Recommend **N=1000** primary (floor = 2,879 ratings ≈ 94.6th
   pct — a clean recognition floor; absorbs 15,385 already-judged cells). N=800 (higher floor,
   fewer niche) and N=2000 (floor 1,087 ≈ 89th pct, absorbs 21,366 cells, richer tail) are ready if
   preferred. → **choose N.**
2. **Join gaps.** 100% IMDb-linked. True unattributed items = 0.27% (<2% ✅). The 5.77% "no billed
   cast" is 771 documentaries + concert/animation where no billed cast is *correct*; only 7 of the
   top-1000 bank affected. → **accept cast filter (order≤4 actor/actress) as-is?**
3. **Flagged-awkward tags = 54** of 1,128; nothing pruned (pilot decides). → **confirm no-pre-prune.**
4. **Schema confirmation.** knowledge 3-level collapse of vividness×can_answer; value 4-level;
   `w_rough`=0.5 knob; rating bins; five channels + dual-arena. → **confirm schema, or request the
   two-separate-fields form.**
5. **Attribute thresholds** (director/composer/writer ≥3, actor ≥5) yield 2,123 / 1,336 / 3,653 /
   2,919 entities. The design also mentions capping to ~250 *judged* attribute questions
   (stratified famous→niche) at LLM time — that selection is a downstream (judge-time) step, not
   built here. → **confirm thresholds; the ~250-question stratified sample is built at run time.**

### What still needs the author before any run
- Choice of N (item bank).
- Schema confirmation (or the 2-field variant).
- The **lit pass** on attribute types (EAR/CRM/critiquing/Rashid) is explicitly a to-do for the
  audit phase, run only with author approval — **not** done here (no approval yet).
- All of the above are inputs to the LLM judging run, which is **not** part of this no-LLM prerequisite task.
