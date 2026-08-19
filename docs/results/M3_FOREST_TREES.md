# M3 — the forest: the cold tree regrown, and six opener-conditioned teachers

**Design sheet** `docs/design/DS_M3_FOREST_TREES.md` · **code** `src/instrument/ndcg_tree.py`
(`--cohort_concept`), `run_m3_teachers.sh`, `m3_show_paths.py`, `migrate_v1_tree_to_v2.py` ·
**artifacts** `experiments/battery/{coldtree_v2,m3v2_*}.json`.
**VALIDATION ONLY — the 10,000 canonical TEST users were not built, loaded or named.**
Run 2026-08-18/19. All seven trees exited status 0.

## ★ First: the stopping rule was the shape

The v1 cold tree stopped at **depth 5, 22 nodes, 15 leaves** — and all fifteen stops were min-leaf
stops with positive gain in hand. Nobody knew whether that measured the data or the rule
(memory note `tree-depth-is-the-stopping-rule-not-the-data`). **It measured the rule.**

v1 took `min()` over ALL THREE children, so one small branch vetoed a whole split — in one case an
**empty** branch vetoed a split whose two real children held 334 and 396 users. **Rule v2:** split iff
≥2 children are non-empty and ≥1 reaches `min_leaf`; children below the floor become leaves that
**inherit the parent's tail**, which was fitted on the parent's full cohort, so nothing is ever
fitted below the floor — which is what the 200 threshold's evidence was actually about.

| | v1 | **v2** |
|---|---|---|
| depth | 5 | **11** |
| nodes | 22 | **164** |
| leaves | 15 | **109** |

The v1 tree is a strict PREFIX of the v2 tree — v2's rule is more permissive, and the splitter search
does not depend on the stopping rule, so every node v1 split, v2 splits identically. **Verified:** a
from-scratch v2 run reproduced v1's first four nodes exactly (root concept 114 → 2606/4814/2580;
root.like item 83 → 1010/340/1256; root.dislike concept 283 → 723/3562/529; root.noanswer item 514 →
357/438/1785). 12 of v1's 15 leaves split under v2; 3 genuinely cannot.

**And deeper pays** (validation, cohort-weighted, vs the combined static order):

| K | v1 (depth 5) | **v2 (depth 11)** |
|---|---|---|
| 1 | +0.00093 | +0.00093 — identical |
| 2 | +0.01236 | +0.01236 — identical |
| 4 | +0.01887 | +0.02004 |
| 8 | +0.02780 | **+0.03575** |
| 15 | +0.03144 | **+0.03993** |

K=1 and K=2 are bit-identical (the prefix property), then v2 pulls away: **+0.0085 at K=15, a 27%
relative improvement on the conditioning gain.** The standing instruction to write neither
"conditioning stops paying" nor "deeper would pay" is lifted, in the affirmative.

⚠ **This is an IN-SAMPLE validation number** — the tree was grown on these users. v1's equivalent was
+0.0314 on validation and **+0.01035 on test** (3× shrinkage), and v2 has far more room to overfit:
109 leaves over 10,000 users is ~92 users/leaf against v1's ~667. Expect v2's test number above v1's
but well below +0.0399; how much of the +0.0085 survives is open until the M5 look.

## The six teachers

Cohort = validation users who **STATE** the anchor (endorse it under the env's SEL rule) — the
deployable routing signature. The anchor is folded free at the root and masked from the pool, so each
teacher asks the same 15 questions the cold tree does.

| teacher | n | depth | nodes/leaves | root splitter (answer rate) | K=15 full/tail |
|---|---|---|---|---|---|
| arthouse | 587 | 2 | 7/5 | concept `black and white` (0.93) | +0.033 / +0.032 |
| dialogue | 730 | 2 | 9/6 | concept `pg-13` (1.00) | +0.026 / +0.011 |
| marvel | 734 | 1 | 4/3 | item *Spider-Man (2002)* (0.55) | +0.044 / +0.040 |
| gore | 747 | 3 | 9/6 | concept `shallow` (0.96) | +0.039 / +0.030 |
| teenagers | 786 | 3 | 10/7 | concept `fantasy world` (1.00) | +0.042 / **+0.048** |
| inspiring | 1,822 | **4** | **21/14** | concept `visually appealing` (1.00) | +0.018 / +0.004 |

**Every teacher branches.** Under v1 all of them were depth 0 — flat conditioned orders, exactly the
structure the author rejected in the genre teachers. Under v2 the forest is genuinely adaptive.

⚠ **These numbers are NOT the forest's claim.** The comparator is the global static order **with no
anchor**, so a large part of each gain is the free anchor's own contribution (the seeded sanity gate
priced arthouse's at **+0.056** at K=1). The real test is M5: teacher vs the **cold tree**, anchor
folded into BOTH arms.

## ★ Depth does NOT predict payoff — anchor specificity does

`inspiring` is the deepest teacher (21 nodes, 4 levels) and the **weakest** (+0.018 full, +0.004
tail). It is also the largest and broadest cohort. The narrow anchors pay most with least depth —
`marvel` at depth 1 returns +0.044/+0.040. An earlier "depth predicts persistence" reading is
**withdrawn**: what predicts payoff is how specific the volunteered fact is, not how much the tree
subsequently branches.

## Question paths — semantic coherence, honestly

`m3_show_paths.py` renders each tree as named questions.

- **arthouse** → `black and white` → (no) `fight scenes` → (no) `disturbing`
- **teenagers** → `fantasy world` → (yes) *Harry Potter: Deathly Hallows Pt 2*; (no) `80s`
- **marvel** → *Spider-Man (2002)* → (like) `heartwarming`; (unseen) `alien`
- **dialogue** → `pg-13` → (yes) *Gone Girl*; (no) `afi 100 (movie quotes)` → (no) `bloody`
- **gore** → `shallow` → (yes) *Batman Begins*; (no) `fun movie` → (no) `based on a true story`
- **inspiring** → `visually appealing` → (yes) `disturbing`; (no) `fight scenes`

**Assessment, corrected after author pushback.** An earlier read called all of these "visibly
coherent"; that was **over-read**. `arthouse → black and white` and `teenagers → fantasy world` are
genuinely sensible; the rest are *discriminative* cuts that admit a post-hoc story if you squint —
`shallow` is a strange thing to ask a gore fan, *Batman Begins* is not gore, and `pg-13` is not a
taste question. **The tree optimises separation, not meaning**, and the two coincide only sometimes.

**The comparison that IS fair, and favours the teachers:** the banked genre-conditioned orders
(items-only pool, no branching) ask Drama-lovers about *Clear and Present Danger* first, Comedy-lovers
about *Return of the Jedi*, and Romance-lovers about ***Speed***; LOTR, Terminator, Star Wars and Men
in Black recur across supposedly different genres, because popular films are the informative probes
regardless of what was said. **Neither family writes questions a human would write. The teachers at
least ask about attributes as well as films, and they BRANCH** — `arthouse` asks something different
depending on your answer, while Drama asks *Clear and Present Danger* of everyone.

## Provenance, and four defects fixed along the way

Certified pair `t2i26c_ep31` + `concept_cv3_ep08`; policy-B mask/pool; canonical snap
`0.15686538118354684`; total_L=15 throughout (matching the v1 node table); leak assert clean;
CELF-heap resume verified bit-identical.

Four defects, all caught by the sanity gate before they reached a real run:
1. **Budget accounting** — a seeded teacher's `own` order is `[anchor] + 15`, so the length assert and
   the `own[:K]` budget slice both had to exclude the free anchor.
2. **Inheritance length** — same, at the inheritance site.
3. **Nesting** — the K=1 tie assert is an UNSEEDED-tree property; a seeded teacher carries the free
   anchor, so its K=1 difference is the **anchor's own contribution** and is now recorded as such.
4. **The driver reported success for every run** — `$?` was expanded after a `$(date)` substitution,
   so it reported `date`'s status. **Three earlier teacher runs crashed and were logged as
   successes.** The driver is now fail-fast.

⚠ Also recorded: rebuilding the cold tree from scratch was unnecessary — v2's rule being strictly more
permissive means the v1 tree is a valid prefix, and the migration (re-deciding 15 leaves from their
recorded `child_sizes`) cost minutes where the rebuild cost ~6 h.

## ★ A fifth defect, found on 2026-08-19: the trees have branches their cohort never populated

Applying the forest to TRAIN users crashed immediately: `AssertionError: user 14616: 1 questions,
need 15` on the `dialogue` teacher.

**The cause is a property of the trees, not of the harness.** `dialogue` splits its root on the
concept `pg-13`, whose answer rate on the validation cohort was **1.00** — every one of its 730 users
answered. The `noanswer` branch therefore held zero users, no child node was created, and the
`needs_tail` guard correctly declined to fit a tail there. On train, users **do** fail to answer it,
route into that branch, and the tree has nothing to give them. Four of the seven trees have such
sites: `dialogue` (root), the cold tree (two), `gore` and `inspiring` (one each, both with long
inherited tails that happen to cover the budget).

⚠ **This would have crashed the single TEST look.** M5 routes held-out users through exactly these
trees; the escape is a certainty on 10,000 fresh users, and the failure would have surfaced *while
spending the one look we have*.

**The continuation rule, now stated once in `src/instrument/tree_route.py` and shared by validation,
train and test.** A user routed outside the fitted support receives, for the remainder of the budget:
the nearest **ancestor's fitted tail**, and if that still falls short, the **global static order** —
the unconditioned incumbent, i.e. what a user gets when there is no evidence to condition on —
skipping questions already asked. De-duplication happens **only** inside this fallback; the fitted
path is reproduced verbatim, duplicates included, because the banked M4a numbers were measured with
them present.

**Verified a strict extension:** on all seven validation cohorts the new router returns **bit-identical
sequences** to the old one, with **zero** fallbacks fired — these branches are empty by construction
on the cohort a tree was fitted on. Every banked validation number stands unchanged. The rule can
only ever fire off the fitting cohort, which is precisely where it is needed.

The escape rate on held-out users is itself a reportable quantity (`n_fallback` is returned by the
router and banked in every trajectory npz) — it measures how often a fitted interview meets a user it
has no branch for.

## What M5 receives

Seven trees on one ruler: the cold tree (the incumbent, depth 11) and six opener-conditioned teachers.
M5's arms are pre-registered in `DS_M3_FOREST_TREES.md` — per-anchor teacher vs cold tree with the
anchor folded in BOTH arms, the assembled mix, with/without openers, shuffled-routing control, all at
**K ∈ {1, 2, 4, 8, 15}**, one TEST look, taken once.
