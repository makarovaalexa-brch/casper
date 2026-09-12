# STATE — where the project is now

## ★★★ 2026-09-12 — GOODREADS PORT COMPLETE: the concept channel PORTS, via a different mechanism

**AUTHOR SAID:** *"keep going with all fable suggestions and hypothesis about concept member count, try
to make the channel work for goodreads"*.

**It already worked.** The reported failure was instrumentation, in two compounding layers.

| | result |
|---|---|
| concept fold effect, mean over all 12 epochs, corrected ruler | **ML-25M +0.0015 / Goodreads +0.0016** |
| out-of-sample-selected epoch | ML-25M ep11 +0.0020 / Goodreads ep01 **+0.0035** |
| mechanism (answers' share of the lift at m8) | ML-25M **67%** / Goodreads **11%** — Goodreads works through REFUSALS |
| saturation | peaks at m=16 (fold +0.0034), decays after |
| member-count / ask-set fix | **refuted** on both a weak and a strong checkpoint |

**Two retractions.** (1) A17's "concept answers make Goodreads worse" was eval-time member-drop in the
monitor (`concept_train_v2.py:105-107`). (2) A22's "does not port" was a comparison of `cvg_ep12`
against `cv3_ep04`, two checkpoints BOTH chosen by that bent monitor and both weak.

Results: `docs/results/A24_RESELECTED_DIAGNOSTICS.md` (current) ·
`A22_CONCEPT_MONITOR_CORRECTED.md` (addenda 1–2 superseded as dataset statements) ·
`A17_GOODREADS_PORT.md` (build sound, negative retracted).
Design `docs/design/DS_A22_CONCEPT_MONITOR_AND_ASKSET.md`. Instruments `a22_concept_curve_paired.py`,
`a23_reselect_concept_epoch.py`, `run_a24_reselected_diagnostics.sh`.

### Next, in priority order
1. **Paper B's +0.01430 portability** — the real question, untested. Its harness has no member-drop so
   the monitor defect does not touch it, but the answers-per-question ledger for concepts vs items on
   the deployed Goodreads harness has never been run. Larger job than anything above.
2. **Fix the in-training monitor** in `concept_train_v2` (member-drop at eval, unpaired levels,
   `users[-1500:]` slice). Until then every future run selects on the bent ruler.
3. **Re-certification decision (author's):** `cv3_ep04` is still the certified ML-25M concept
   checkpoint and is one of its weakest epochs on the corrected ruler. Re-certifying moves Paper B's
   artifact set.

---

## ★★★ 2026-09-10 — SECOND DATASET: the instrument is being ported to Goodreads

**AUTHOR SAID:** *"commit everything we have, do not edit the paper or the chapter, and replicate the
process exactly without corner cutting. if you start getting netaive results stop and tell me. Do not
replicate chapter A full battery, just the instrument itself and its gates"* · *"60h is fine"*.

Full design, gates and STOP conditions: **`docs/design/DS_A17_GOODREADS_ARENA.md`**.
Costing that led here: `docs/design/DS_A14_SECOND_DATASET_COST.md`.

**Why Goodreads.** Three candidates measured and rejected first: goodbooks-10k (passed on numbers,
no paper to cite), the Goodreads genre subsets (median **2** ratings per user — a genre slice cuts
each history to one corner), Netflix (formally withdrawn after a privacy lawsuit; no licensed
distribution), Amazon/Yelp (1.6–2.7 reviews per user; an interview collects no answers).
The full UCSD book graph has 104.5M ratings over 816,371 users, mean 128.1 each.

**Two calibrated reductions, both approved, both stated**: arena = the 18,430 most-rated books
matching ML-25M's arena SIZE (keeps 56.7% of rating events); train cohort sampled to 140,768 to match
ML-25M rather than using all 683,636 eligible.

**Done**: arena + graded store + shelf vocabulary; concept bank (2,408 concepts, per-book density
median 37 vs ML-25M's 37); Goodreads emitted in MovieLens RAW shape; the split from the project's own
`liang_split.py --data goodreads` (18,430 items, 140,768/10,000/10,000). Every stage of the
instrument-and-gates chain now follows `CASPER_DATASET`, verified both ways.

**Running**: RecVAE teacher, epoch 17 of up to 50, val 0.3413. Tracking ML-25M's own curve with a
steady ~0.007 offset (ML-25M peaked 0.3512 at ep24, test 0.3540/0.2497).

**Next**: i26 tower (`src/instrument/run_goodreads_tower.ps1`, ~19 h) → concept channel → the Phase-A
gate battery (g3a, g3b, g6 with `--force`, g9, g5) + a fresh `concept_identity_lift` for R2.
Phase B is skipped: G1/G4/G8/G2-formal are moot since the belief layer was deleted 2026-08-02.

⚠ **`warm_lr_scale`**: the i26c watchdog comment says 0.01, the certified run's own training log says
0.1 at every resume. The log is primary evidence and is what the port follows.

⛔ No paper or chapter file is touched by this work.


## ★★★ 2026-08-29 (late) — THE TEST LOOK IS SPENT AND IT REPLICATES

**AUTHOR SAID:** *"continue with todos, inclusing test look"* · *"just no preregistration of anything,
just go ahead and run. and yes, no refitting, as you suggested"*.

### The confirmatory result — ONE pass, 10,000 test users, K=15

| | TEST | VALIDATION |
|---|---|---|
| ALL n=10,000 | **+0.00343\*** (policy 0.26842 / cold 0.26499 / static 0.25691) | +0.00333\* |
| warm n=3,877 | **+0.00946\*** | +0.00892\* |
| cold n=6,123 (do-no-harm) | **−0.00038** | −0.00037 |

`vs static +0.01151` on ALL. Three contrasts, three near-exact reproductions on unseen users.

### How contamination was avoided
- **Material REUSED, never refitted.** `own_item`, `own_concept`, group centroids and half-B pools all
  come from the banked validation artefacts (`m7_tabs_val.npz`). Verified: per-group half-B pools on
  test are **473/58/444/191/242/74**, byte-identical to validation. `--split test` REFUSES to run
  without `--load_tabs`, because `m6d_eval.py:156` would otherwise refit the opener generator on the
  confirmatory split and print plausible, meaningless numbers.
- **Assignment by RULE, verified first.** The own-data argmax was asserted to reproduce the banked
  validation assignment EXACTLY (3,980 warm users) before being applied to test.
- Leak assert `0/10000`; test m=0 `0.15855` vs validation reference `0.15687`.

### Per group on test (route root · policy · tree · cold · vs cold)
```
arthouse   464  0.843  .29621 .30046 .27964  +0.01657*
dialogue   395  0.792  .23299 .22940 .21894  +0.01404*
inspiring 1526  0.855  .25195 .25534 .24338  +0.00857*
gore       391  0.816  .33406 .33208 .32595  +0.00811
teenagers  502  0.741  .24405 .24179 .23712  +0.00692
marvel     599  0.957  .42555 .42534 .41933  +0.00622
```

⚠ **The per-group pattern is NOT stable across splits.** `dialogue` and `inspiring` were flat on
validation and both clear zero on test; `arthouse` fell +0.03077\* → +0.01657\*. At 391–1,526 users
per group this is expected. **Quote the pooled contrast, not the per-group ranking**, and the chapter
claim that the two film-poor groups gain nothing is a VALIDATION-SPECIFIC pattern that must be
corrected.

### Framing (author, 2026-08-29)
The contribution is that the policy is **one extendable artefact**, not that it beats every tree.
★ The tree arm is HANDED its group (`m6d_eval.py:257`); the policy gets no label and must route
itself — so `policy − tree` understates the policy. A deployable forest+router comparator was
considered and **ruled out by the author**.

### NEXT
Chapter: swap test numbers into `sec:onepolicy`, correct the film-poor-groups claim, then W6/W8,
`\PLAN{}` triage, abstract/intro. GreedyExtend analogue (`greedy_static_policyB_items_train10000.json`,
125 min) is a validation-only reference row.


## ★★★ 2026-08-29 — THE POLICY BEATS COLD ON THE DEPLOYMENT MIX. Disjointing the cohorts did it.

**AUTHOR SAID:** *"please go back to our original clusters, original approach, and think how to fix
it to make routing work. why we couldn't just keep the clusters, but switch to clusters from
endorsement cohort?"* · *"ok, do it. and if i don't come back, proceed to tree building, as long as
all looks ok"* · *"continue with this please, proceed to distillation etc"*.

### The fix (`src/instrument/m7b_disjoint.py`)
Two argmaxes make the six endorsement cohorts a **partition**: each user → the group they endorse
most strongly; each film and concept → the group whose centroid lift is highest. Centroids stay on
the OVERLAPPING cohort (as m7b/m6d do) — only the *material* and the *user assignment* are disjoint.
This keeps the author's original cohorts and fixes only what made them unroutable: marvel/inspiring
shared 10 of 12 characteristic words and 31% of opener films sat in more than one pool.

### Results — all trees fitted on TRAIN, all numbers on VALIDATION, K=15. **Test look unspent.**
| | before | after |
|---|---|---|
| probe separability | 0.371 | **0.959** |
| routing, n-weighted | 0.553 (marvel 0.364) | **0.858** (marvel **0.958**) |
| teacher opener−cold, pooled | +0.00760* | **+0.00688 [+0.00250,+0.01128]\*** |
| policy−cold, deployment mix | +0.00085 n.s. | **+0.00333\*** |

- pooled order holds: `static 0.27365 < cold 0.27859 < opener 0.28547`
- one policy, SEVEN teachers, ep5 mean dev top-1 **0.9910**, every bank ≥ 0.982
- mix n=10,000 **+0.00333\*** · warm n=3,980 **+0.00892\*** · cold n=6,020 **−0.00037** (do-no-harm
  holds but is nominally negative — report as measured, do not round to zero)
- half-B, films excluded from every training opener: arthouse **+0.03077\***, gore **+0.01980\***,
  teenagers **+0.01313\***, marvel +0.00643, dialogue +0.00155, inspiring +0.00074
- ★ capture: teachers' weighted warm advantage +0.00877 vs policy +0.00892 ⇒ **~100% inherited**,
  against 30% before. The policy did not get better at recommending — it got better at knowing which
  teacher to consult. **The entire previous shortfall was routing, not fidelity.**

⚠ `dialogue` and `inspiring` are flat; they kept only 58 and 74 half-B films and lean on generic
vocabulary. Predicted in writing before the fit.

### ⛔ A harness bug that printed a plausible catastrophe
`m6v_route_confusion.py` had no `--disjoint` branch (unlike `m6d_eval.py:144`), so it scored the new
policy against the OLD `m6k_<g>` trees and reported routing **0.000 for five of six groups**. It
contradicted 0.9910 dev fidelity an hour earlier, which is what flagged it. Fixed (tree path + banked
assignment + disjoint opener material) and given a **guard asserting the "other" share < 0.5**. The
corrected matrix reproduces m6d's route-root figures exactly.

### NEXT
Writing to-dos W1–W12 in `docs/design/PAPER_B_FINAL_AUDIT_2026-08-25.md`; **W7 (GreedyExtend, the
missing external static anchor) is the only unrun experiment**. Literature gap: MovieLens user
segmentation is essentially absent from `external_literature/`. `docs/design/M7_STATE_CONDITIONED_DESIGN.md`
is SUPERSEDED (banner added) — it records the plan the author rejected.


## ★★★ 2026-08-19 (evening) — FIT THE TREE WHERE YOU MEAN TO SERVE. It works.

**AUTHOR SAID:** *"try building the tree teacher on just one cluster for now, and then compare its
performance with old tree teacher and cold tree and static"* · *"continue with all, going to bed"*.

### The result (M6l, `docs/results/M6L_OPENER_FITTED_TEACHER.md`)
A tree fitted on **varied, group-linked opener states** beats the banked anchor teacher at **all five
budgets** (+0.013..+0.027, every CI excluding zero), static at four (up to +0.033), and the cold tree
at K=15 (**+0.01386★**). At those same states the ANCHOR teacher ran −0.008..−0.014 against cold
(M6a/M6c) — so the diagnosis *"teachers are anchor teachers"* was right and the remedy works.
Controlled A/B: same 786 users, same pool/budget/min_leaf, **only the fitting state differs**.
⚠ One group, depth 1, in-sample; cluster-vs-anchor is the clean cell.

### ⚠ A TERMINOLOGY ERROR, caught by the author and corrected everywhere
"Cluster" was used all session for what is an **ENDORSEMENT COHORT**. **No tree has ever been fitted
on a k-means cluster label** — verified three ways (`--cohort_concept`, `ndcg_tree.py:530`,
`n_cohort`=786/587 matching endorsement counts and no cluster size). The k=20 cluster labels were
**never banked** (M1 merged start_k=40→final_k=4 under a min_size 1200 gate); re-derived and banked
now as `m6/m6p_clusters_k20.npz`. ⚠ **One era projection does not remove era structure: 13 of 20
clusters span under three years.** Endorsement cohorts dodge this because they are semantic.

### Opener synthesis, and it is in the chapter
`sec:opener_synthesis` + `tab:opener_match`. Targets MEASURED on the corpus (film .630 / genre .398 /
tone .214 / coherence +0.224 / popularity 93.3), material chosen by **centroid lift** — which is
validated against Reddit: real named films land at percentile **76–91** of their own group's ranking
vs ~50 random. Three non-matches declared (popularity floor, era tags, unsayable affect words).

### RUNNING OVERNIGHT (launched 21:04)
`run_m6_all.sh`: opener-fitted trees for arthouse → gore → marvel → dialogue → inspiring, each
followed by its four-arm comparison. ~13 h. Cheapest cohorts first so a late failure still leaves the
most groups done; every exit status captured accurately.

### NEXT, PRE-REGISTERED before the data exists (`DS_M6`, M6c/M6d/M6e)
State-only distillation over all groups (no token exists — the opener differs per user), with an
interpretability gate; **generalisation tested on a seeded 50/50 split of each group's diagnostic
pool** so test openers use material never seen in training; then the group mix with a **do-no-harm
cell** on cold users. All three readings declared in advance. **The single TEST look stays unspent.**


## ★★★ 2026-08-19 — M4a RE-RUN, M4b DONE: ONE POLICY HOLDS THE WHOLE FOREST

**AUTHOR SAID:** *"ok, distill and compare all the trees and the model on val data"* · *"please set
the policy to train on train users and then do the write up in parallel"* · *"can you just exclude
those from training?"* · *"can you explain how the question is repeated? how repeated question could
be possibly optimal?"*

### The headline: the forest collapses to ONE deployable object
`docs/results/M4B_FOREST_POLICY.md`. At K=15 on validation, |policy − teacher| ≤ **0.00045** in every
cohort (3 positive, 3 negative). Root agreement **1.0000** in all six; **96.9–99.9% of users get an
interview identical to their teacher's end to end**; teacher-forced validation fidelity ≥ 0.9977
pooled for all seven trees. **The interference falsifier did NOT fire** — cold held at 0.9991 while
the six teachers run 0.9987–1.0000. One 526,542-parameter policy, no hand-written router.
⚠ It claims EQUIVALENCE at one seventh of the artifacts, never superiority (E1: a distilled policy
reproduces its teacher at best).

### ⚠ A ROUTING DEFECT invalidated the first M4a and the first policy
`memory/routing-rebuilt-what-the-artifact-already-stored.md`. The router rebuilt each interview by
accumulating splitters during the descent instead of reading the tree's stored `own = path + steps`.
A below-floor child's `path` EXCLUDES the parent splitter because its `steps` STARTS with it ⇒ the
splitter was supplied twice ⇒ **14 distinct questions on a budget of 15 for 18.5–45.3% of users**
(87 of the cold tree's 164 nodes are such children). **The trees were never at fault:** zero leaves
in any of the seven contain a duplicate in their own `path + steps`.
- **M4a re-run:** 16/30 cells separate (was 17), **30/30 point estimates positive** (was 29),
  `marvel` K=15 loses significance. **No conclusion reverses**; old numbers must not be quoted.
  Superseded artifact kept as `m4a_compare_BUGGY_ROUTER.json`.
- **Caught by the policy itself** — it cannot ask a question twice, and `frac_never_deviates` matched
  **1 − repeat-rate to within 0.002 in all six cohorts**. Two wrong explanations were proposed and
  killed by measurement first.
- ★ The scoring path deduped before folding, so the repeat degraded to a wasted turn and had been
  **written into the results doc as expected behaviour**. Routing now asserts `len(set(seq))==15`.

### Also fixed: a fitted tree escapes its own support on held-out users
`memory/fitted-tree-escapes-its-own-support.md`. `dialogue` splits its root on `pg-13` (answer rate
1.00 on its cohort), so the `noanswer` branch has no child and no fitted tail; train users land there
and get nothing. **This would have crashed the single TEST look.** One continuation rule now lives in
`src/instrument/tree_route.py` (ancestor tail → global static), shared by validation, train and test.
Escape rate on train: 0.13–0.43% for `dialogue`, zero elsewhere; those positions are masked out of
the distillation loss (author's call) but never out of evaluation.

### Chapter B writing done this session
- `\subsection{What volunteered information is worth}` with the 17/30 grid, **and a Scope paragraph
  stating the comparison is BIASED TOWARDS THE TEACHER** (author: teachers are fitted on the cohort
  they are scored on; the cold tree is fitted on all 10,000). Numbers are an upper bound pending M5.
- **W1 done:** `\subsection{What the answer model can and cannot launder}` (`sec:circularity`) in
  Setup — the panel's lead objection was answered in three separate places and never named as a
  defence. Backed by the chapter's OWN shuffle controls (−0.04817 / −0.03934).

### Next
**M5, the single TEST look**, pre-registered in `DS_M3_FOREST_TREES.md`, still unspent. Arm C is now a
legitimate third arm for it. ⚠ The M4a per-budget grid in the chapter still quotes the BUGGY-router
numbers and must be refreshed from `M4A_TEACHER_VS_COLDTREE.md` before the chapter is next compiled
for review.


## ★★★ 2026-08-17 (night, later) — M1b IS RUN AND BANKED; M1 Stage A DONE; Stage A2 RUNNING

**AUTHOR SAID:** *"plz proceed according to plan"* · *"ok, proceed"*.

- **M1b opener census — DONE.** `docs/results/M1B_OPENER_CENSUS.md`, artifacts
  `experiments/battery/m1b_census.json` + `m1b_mapping.json`, code `src/instrument/m1b_census.py`,
  `m1b_map.py`. **63.0% of real requests name a film** (a film is EVIDENCE, not a question);
  **93.6% carry >=2 slots** (bundles, not single facts); bundle cells 35.9% films+signature / 27.1%
  films only / 19.9% signature only / 17.0% neither. Mapping onto our attributes: genre **95.1%**,
  era 70.2%, tone 66.1%; unmapped residual named (`coming of age`, `Christmas`, `wholesome`,
  `uplifting`, `cozy`). **★ Requested vs watched are NEAR-INVERTED: Horror is 27.5% of genre
  mentions but the favourite genre of 51 of 10,000 validation users (Drama 5,481)** — a scope
  condition on any ratings-only elicitation result. **Now written into the chapter** as
  `
ef{sec:opener_census}` (compiles clean, commit `47453a3`), with a `\PLAN` flag: the slot
  labels are model-produced and carry NO gold-set accuracy number, unlike the linking's 0.960.
- **M1 Stage A — DONE.** `experiments/battery/m1/{z_requests,z_full}.npz`, `m1_stage_a.json`.
  **100.0% of the 7,572 OP-named films are inside our 18,359-item vocabulary** (asserted).
  **3,078 of 4,431 requests foldable (69.5%)**: items-only 1,983 · concepts-only 439 · both 656.
  ⚠ First run had a defect — the concept-name lookup returned 0 surface forms, so requests folded
  from FILMS ONLY and the 439 concepts-only requests were mis-classed unfoldable. Fixed (names come
  from `E["sh"]["tags"]`, same construction as `forest_policy.py`) and re-run; an assert now fails
  loudly if the index is empty. Declared choices: anchors at the top liked level (7/8/9 are
  gamma-equivalent under the certified sign init, so the choice cannot move the fold); stated
  concepts asserted at the cohort's own max positive value (1.0000), refusal flag 0.
- **★ AUTHOR CORRECTION, binding:** `z_full` **defines the taste partition and is NEVER an interview
  state** — a satiated profile has nothing left to ask. The IDEAL family therefore starts from **one
  perfect volunteered fact** that places the user in their cluster (Stage A2), making both families
  cold + a little evidence. It is an **ORACLE ceiling** and must never be presented as deployable.
- **Stage A2 — RUNNING** (`src/instrument/m1_ideal.py`, log `experiments/battery/m1_ideal.out`).
  Ideal fact by **identification, not distance** (a 1-fact fold and a 60-item fold differ more in
  norm than in taste); min 50 staters per fact; k sweep {4,6,8,10,12}. Watch **purity vs prior**.

**Next:** Stage B (cluster request states), the alignment cross-tab (opener-state vs full-taste
state), then Stage C twin matching and the M2 screen.

---

## ★★★ 2026-08-19 01:25 — THE FOREST IS BUILT. M3 COMPLETE. NOTHING IS RUNNING.

Full write-up: **`docs/results/M3_FOREST_TREES.md`**. Seven trees banked, all status 0:
`coldtree_v2.json` + `m3v2_{arthouse,dialogue,marvel,gore,teenagers,inspiring}.json`.

**★ The cold tree's depth-5 stop measured the STOPPING RULE, not the data.** v1 took `min()` over
all three children, so one small — sometimes EMPTY — branch vetoed a whole split. Rule v2 splits iff
≥2 children are non-empty and ≥1 reaches min_leaf; children below the floor inherit the parent's tail
(fitted on the parent's full cohort), so nothing is ever fitted below the floor. **v1 22 nodes / 15
leaves / depth 5 → v2 164 nodes / 109 leaves / depth 11**, and the v1 tree is a strict PREFIX
(verified: a from-scratch run reproduced its first four nodes exactly).

**★ Deeper PAYS** (validation, cohort-weighted, vs combined static): K=1 and K=2 bit-identical, then
K=4 +0.02004 · K=8 **+0.03575** · K=15 **+0.03993** against v1's +0.03144 — **+0.0085, 27% relative.**
The standing "write neither 'conditioning stops paying' NOR 'deeper would pay'" is lifted, affirmative.
⚠ IN-SAMPLE: v1 was +0.0314 val → +0.01035 test (3× shrinkage) and v2 has ~92 users/leaf vs v1's ~667.

**★ All six teachers BRANCH** (under v1 all were depth 0 — the flat-order structure the author
rejected): arthouse d2 · dialogue d2 · marvel d1 · gore d3 · teenagers d3 · inspiring d4 (21 nodes).

**★ DEPTH DOES NOT PREDICT PAYOFF — ANCHOR SPECIFICITY DOES.** `inspiring` is deepest (21 nodes) and
weakest (+0.018 full / +0.004 tail at K=15) on the broadest cohort; `marvel` at depth 1 is strongest
(+0.044/+0.040). An earlier "depth predicts persistence" reading is WITHDRAWN.
⚠ Teacher numbers are vs the static order WITH NO ANCHOR, so much of each gain is the free anchor
itself (arthouse's priced at **+0.056** at K=1). M5 is the real test: anchor folded into BOTH arms.

**Question paths** (`m3_show_paths.py`): only `arthouse → black and white` and
`teenagers → fantasy world` are genuinely sensible; the rest are DISCRIMINATIVE cuts (an earlier
"visibly coherent" read was over-read and is corrected in the results doc). The fair comparison
favours us anyway: the banked genre orders ask Romance-lovers about *Speed* and never branch.

**Four defects fixed, all caught by the sanity gate:** seeded-teacher budget accounting; inheritance
length; the K=1 nesting assert (an unseeded-tree property — a teacher's K=1 diff is the ANCHOR's
contribution); and **the driver reported status 0 for every run** (`$?` after `$(date)`), so three
earlier teacher runs crashed and were logged as successes. Driver is now fail-fast.

**NEXT:** M4 (distil the seven trees into one policy) · the volunteered-information section + figure
for Paper B · the M2 screen's outstanding cold-state control · **M5 = the single TEST look,
pre-registered in `DS_M3_FOREST_TREES.md`, UNSPENT and the author's call.**
Paper C: design settled and Fable-ruled (`PAPER_C_DESIGN.md`), nothing running.

---

## ★★★ 2026-08-17 (night) — THE PLAN IS SETTLED: Part M "COVERAGE PROGRAMME" of `docs/design/PAPER_B_POLICY_DESIGN.md` (M + M.1–M.4). NOTHING IS RUNNING; AUTHOR SAID "do not run yet".

**Read Part M + amendments M.1–M.4 before doing anything — they ARE the plan for the rest of the
chapter.** In one line: spine = the elicitation audit (banked); closing act = two opener families
(DREAM = taste-clustered ML users, REALISTIC = Reddit-slot-mined, mapped to ML attributes — never
SBERT cohorts) → the M2 SCREEN vs the tree at +0.004 (the front-loaded bet, ~1 day; softened by the
banked genre-vs-static test number +0.00817 but the tree may absorb the opener's signal — that IS
the screen's question) → teachers for clearing clusters → ONE policy (cold tree kept verbatim + opener-routed teachers) → the Reddit-weighted MIXED-OPENER BENCHMARK (data-driven weights fixed from
the census BEFORE any arm runs; hard baseline = tree+FOLD; TEST look spent here) → qual trajectory
panel. Contribution statement + reviewer-risk register in M.3. Reddit: 4,431 adjudicated threads
with slots ALREADY EXTRACTED suffice — no further processing, no new LLM calls; the data pipeline is
a CLAIMED CONTRIBUTION (M.4 §5). M5a (real-text demo) is optional, cuttable, and the only linking
anywhere. Writing todos W1–W5 are schedulable now with zero compute — W1 (simulator-circularity
section) is urgent: `chapterB_v2.tex` has ONE circularity mention in 1,647 lines and the field's
lead objection is unanswered in our own chapter.

---

## ★★★ 2026-08-17 (evening) — superseded by Part M above; kept for the axis reasoning. NOTHING IS RUNNING.

**AUTHOR SAID (verbatim):** *"i think building teacher forest is the first step here."* · *"genre is a
heuristic, i do not like that … we don't know if genre is a good / reasonable opener at all. maybe
it's mood? or decade? perhaps we should mine for an opener … or maybe cluster user profiles after
folding in their rating? there must be some statistical ways to cover multiple bases."* · **binding on
the chapter:** *"this chapter does not need a limitation. either we show it working or we omit it."*

Memory note: `forest-axes-must-be-mined-not-guessed`. Fable brief written and **UNSENT**:
`docs/design/FABLE_BRIEF_PAPERB_VISION.md` (session subagent cap was exhausted; cap now raised to 400
via `CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION`). **Send it first thing.**

### Two proposals were withdrawn this evening — do not resurrect either

- **The 17-genre forest is DEAD.** Those teachers are **static orders after the genre is stated, not
  trees**, and L2a measured them losing to the cold tree at their own states (−0.026/−0.033/−0.031;
  rebuild over the tree's own pool lost on holdout, 0 of 4 groups won). Distilling them teaches the
  policy to be worse where it is used. Separately, **genre was never shown to be a good opener.**
- **Scaling L1 to k=2/4/8 is NOT decisive** (author caught this). Facts drawn from the tree's own 22
  nodes are a rigged manifold; facts from anywhere else already have a banked answer and it is
  negative (−0.047 random concepts; −0.022/−0.016/−0.011 on R's items). Either rigged or known.

### The three claims the chapter must deliver, and where each stands

| | claim | status |
|---|---|---|
| 1 | policy beats static AND the cold tree **when volunteered info exists** | **true but tiny** — L1's saved turn +0.00167 [+0.00096, +0.00238]. **The crux; the forest exists to enlarge it** |
| 2 | question order genuinely changes with the opener (count trajectories) | R2 changes 9–13 of 16 questions **but it costs** — changing ≠ conditioning |
| 3 | the recommender consumes volunteered info and beats popularity | **BANKED** — R's fold is +0.0206 full / +0.0353 tail at k=8 with questions held identical. Needs writing up as a result, plus the author's upgrade: consume **out-of-vocabulary request text** via the existing SBERT → genome-concept → concept-fold path |

### How teacher axes get found (Fable to rule)

- **(A) Cluster the folded evidence** → one **TREE** per cluster (adaptive, not a static order).
  Signature computed at ask time ⇒ admissible; no heuristic; k controls cohort size (10,000/20 ≈ 500).
- **(B) Mine the opener distribution from Reddit** — secondary, per the author: it shows what people
  volunteer in one context, not what differentiates users.
- **Each teacher is gated by the L2 condition: it must BEAT THE INCUMBENT at its own states.**

### Reddit splits in two, and only one half needs linking

1. **Opener-type census** (exemplar / genre / mood / era / constraint + frequencies) — needs text
   classified, **zero items linked**. Feeds the forest axes; equivalent evidence is then constructed
   from ML attributes we already hold.
2. **Real-text consumption demo** (claim 3's upgrade) — needs the mapping; **feasibility gate first**,
   recorded in `docs/design/DS_REDDIT_LINKING.md` (transplant variant: request evidence → matched
   ML-25M profiles → their held-out targets; gate = cohort size ≥ ~500 per request).

Paper C's emission arc is specced but unrun: `docs/design/PAPER_C_EMISSION_DESIGN.md` — two blockers
(the one-hot target forbids in-between emission; the snap makes novelty invisible to any reward), two
fixes in order (blend targets, then an answerer accepting arbitrary directions), and a free geometry
check on E1's existing checkpoint that decides whether the arc has a foundation.

---

## ★★★ 2026-08-17 — PART L IS RUN: L1 (as-if-asked encoding) and L2a (the two-teacher forest). VALIDATION ONLY, NO TEST LOOK.

**AUTHOR SAID (verbatim, 2026-08-17):** *"plz run l1 and l2."*

Write-ups: **`docs/results/L1_ASIF_ASKED.md`**, **`docs/results/L2_FOREST.md`**. Code
`src/instrument/{policy_asif,forest_policy,group_greedy_signature}.py`. Design sheet Part L.

| | claim | falsifier | verdict |
|---|---|---|---|
| **L1** as-if-asked encoding | a volunteered fact is an answered question and must enter as one | cell 1 does not reproduce the tree's post-root branch | **DID NOT FIRE.** 100.00% of 10,000 users branch exactly as the tree's post-root arm; per-user score identical to banked E1; redundancy 0 asserted |
| **L2a** two-teacher forest | one policy holds many cohort teachers | fidelity to T1 costs fidelity to T0 | **DID NOT FIRE.** T0 1.000000 → 0.999969 (**1 state of 32,000**) while T1 went 0.000369 → 0.999963; spanning per-user IDENTICAL on 2,000/2,000 |

### L1 — the encoding was the bug, and it is exact where a teacher has an opinion
History token (`phi_q` ⊗ ternary outcome) + fold into `z` + **mask from re-asking**. R did the fold
alone, so the policy read "nothing asked" and was inert; under the new encoding it is exact. The
saved turn is worth **+0.00167 full [+0.00096, +0.00238]** against the budget-matched tree.
**★ But encoding is necessary, not sufficient:** where no teacher covers the volunteered fact the
policy improvises and loses — **−0.04663** (two random answerable concepts), **−0.02231 / −0.01643 /
−0.01070** (R's item draw at k = 2 / 4 / 8). R's inertness was worth +0.047 there. Chain checks: R's
banked cells reproduce to 1e−12 at every k; cell 1's tree arm reproduces banked E1 per user.

### L2a — one policy, two teachers, no interference; and the second teacher is not worth following
T0 = the cold tree, T1 = the four stated-genre orders clearing the 500-user floor (Drama 5,481,
Action 1,582, Comedy 1,475, Adventure 534 — 90.7% of the cohort). Warm-started from E1, 6.7 minutes,
FIT 8,000 / HOLDOUT 2,000. All five pre-registered gates pass, including the shuffle control
(permuted teacher assignments: T1 fidelity **0.000000** against the true orders, worse in every
group). **★ The result that matters: at genre-seeded states the ordering is tree > forest (= T1
exactly) > E1 improvising** — forest − tree = **−0.02648 / −0.03290 / −0.03075** (Drama / Action /
Comedy, all excluding zero) and −0.00587 (Adventure, null). **Admissibility is not sufficiency: a
teacher earns its place only if it beats the teacher already covering those states, on the same
action space and the same budget.** T1's orders were searched over 500 ITEMS ONLY; the tree spends
most of its budget on CONCEPTS.

**★ THE DECISION EXPERIMENT IS RUN AND THE FOREST STOPS AT ONE TEACHER.** T1'
(`group_greedy_signature.py`, ~4.7 h) rebuilt the stated-genre teacher over the tree's own combined
pool, FIT users only. It does use concepts (5–9 of 15) where the item-only teacher had none — and on
HOLDOUT it **still loses to the tree in every group: 0 won, 3 lost, 1 null** (−0.01650 / −0.04789 /
−0.03734 / −0.02806 vs item-only −0.02648 / −0.03290 / −0.03075 / −0.00587). The item-only column is
the optimistic one — those orders were searched on all 10,000 validation users — and the clean column
still loses. **The action space was not the explanation; a second admissible teacher was built twice
and neither was worth following.** T2's harness (`group_greedy_bands.py`, per-user-prefix CELF) is
committed and UNRUN; whether to spend its search time is the author's call. ⚠ T3 as specified is NOT
admissible: a cold user's fold-in tail mass is not observable at ask time.

**Superseded note (kept for the record):** `group_greedy_signature.py` rebuilds the
stated-genre teacher over the tree's own 1,000-candidate combined pool, FIT users only, CELF with a
persisted heap. Beats the tree inside the group ⇒ T1 failed for its action space and the T2/T3
bands are worth their search time; does not ⇒ the forest stops at one teacher. Either way it is
reported.


## ★★★ 2026-08-16 — THE GENERALISATION PAIR (Part I) IS COMPLETE ON VALIDATION. BOTH FALSIFIERS FIRED. NO TEST LOOK.

**AUTHOR SAID (verbatim, 2026-08-15):** *"please run."* Her goal, verbatim: *"i want to show
generalisability working here and my recommender support ood questions etc. i need a policy capable
of understanding that and emitting a questions."*

Write-ups: **`docs/results/P1_HELDOUT_ACTIONS.md`**, **`docs/results/P1B_NEVER_TAUGHT.md`**,
**`docs/results/P2_EVIDENCE_SEEDED.md`**. Code `src/instrument/{policy_p1,policy_p1b,policy_p2}.py`
plus `--heldout_json` in `policy_net.py`. Design sheet Part I, with **I.2's state count corrected
upward from 138 to 209**.

| | claim | falsifier | verdict |
|---|---|---|---|
| **P1** held-out label questions | "emits questions it was never taught" | held-out top-5 emission ≤ the label-shuffled floor | **FIRED.** top-1 and top-5 both **0.0000**; shuffled floor ranks the question BETTER (median 465 vs 338); popularity beats both (top-5 1.45%) |
| **P1b** never-taught questions, undamaged policy | "it ranks by similarity" | never-taught ranking ≤ popularity | **FIRED.** null in the full-pool group, significantly WORSE in the sampled group (+68 ranks [+9, +126]; ρ −0.061 [−0.109, −0.014]) |
| **P2** evidence-seeded fine-tune | "taught to condition, deviations stop costing" | deviations still cost, CI excluding zero from below | **FIRED at every k: −0.02825 / −0.02054 / −0.01434**, all excluding zero — worse than R2's untrained −0.01246 / −0.01234 / −0.00860 |

### P1 — the supervised label vocabulary is load-bearing; similarity does not substitute
36 of 172 label questions held out by identity (channel × label-mass quartile, seed 98765; the draw
took **the tree's ROOT**), E1 retrained from scratch with them still emittable. Val full
**0.28355 → 0.23185**, now LOSING to the static order (−0.02376 [−0.02661, −0.02087]) and −0.05170
vs the tree. Teacher-forced top-1 falls to **0.7604 = exactly the share of states whose label was not
held out** — ~100% right where supervised, ~0% where not, nothing in between. Emission at the 38,336
held-out states: **0.0000 top-1, 0.0000 top-5, 0.0000 neighbour credit.** Control (also removed from
the training candidate table) is a null against main (+0.00230 [−0.00090, +0.00552]) — nothing was
learned-then-unused. Forcing the root back recovers **66%** of the gap (0.26616) and the policy beats
static again, but still agrees with the tree at only 37% of later positions.

### P1b — the non-crippling version, with the ON-MANIFOLD control that makes it readable
The held-out set already existed: the tree uses only **172 of the 1,000 pool questions**, so 828 were
never a training target. Scored the **unmodified E1** at 42 full-pool below-floor states, 138
random-prefix states, and **7 tree-internal nodes** (exact 1,000-candidate gains).

> **E1 ranks the measured-best question 1st at 7 of 7 states it was TRAINED on, and 208th–374th of
> 1,000 at states it was not.** On taught questions it beats popularity by 253 ranks [−376, −130] and
> hits top-1 0.24; on never-taught ones it is indistinguishable from its own label-shuffled control.

The 7/7 also **verifies the harness** (its ranking agrees with the policy's own argmax). ⚠ The
never-taught cells sit at unseen STATES too, so P1b **confounds unseen question with unseen state**
and cannot separate them; no banked artifact can.

### P2 — taught to condition on volunteered evidence, it moves, and every move costs
Corrected census: **209 off-manifold states** carrying measured ACTION labels had never trained
anything (42 below-floor + 138 random-prefix + 31 static-order; 216 more are the tree's own states =
the replay). **210 used**, 500,000 examples, **125 label questions of which 77 the tree never asks**.
Fine-tuned from E1 at lr 2e-4, 7/8 replay + 1/8 seeded, each seeded state presented in BOTH encodings
(as-asked and as-volunteered — R2 showed the second is the one the evaluation presents).

Every process gate passed: **spanning gate 0.28354971825495506 = E1 to every digit**, seeded top-1
0 → 0.663 (as-asked) / 0.358 (as-volunteered), and inertness gone — **89.9 / 90.8 / 93.6% of users
now get a changed interview** against R's 1.0 / 2.5 / 5.1%. Evaluated on a **fresh draw (seed 424242,
disjoint from R/R2's 98765)** through R2's own harness.

**And it loses at every k.** vs its own frozen order: −0.02825 / −0.02054 / −0.01434, all excluding
zero; vs the tree the same; it now also loses to the static order. Among deviators only: −0.03141 /
−0.02261 / −0.01533. The shuffled-label control costs MORE (−0.03394 / −0.02301 / −0.01797), so the
measured labels do carry signal — just not enough to reach zero.

**★ The one thing that flipped, and it is real:** redundancy. R and R2 found no detectable avoidance;
the fine-tuned policy re-asks a volunteered film at **roughly half the tree's rate** (0.794% vs
1.439%, 1.532% vs 2.873%, 3.523% vs 5.408% per item question), every paired interval excluding zero.
**It paid for it in answerability:** concept share 0.5457 → 0.29–0.38 and interview answer rate
**0.579 → 0.379**. It asks more films, gets answers to fewer, and burns turns.

⚠ **Teacher honesty, on every P2 number:** the 138 random-prefix states — most of the training mass —
carry an argmax over ~22 in-pool candidates of 1,000, not the pool argmax (the 42 below-floor and 30
static-order labels are exact). The full-pool upgrade (12–23 h) is **not indicated**: I.2's rule was
to upgrade only if the direction was right and the magnitude shy, and the direction is wrong.

### Q — the descriptor objection, settled by ablation (2026-08-16)
The author noted `build_phi` hands concepts their **train answer rate** and items their
**log1p(train count)**, while Q's fair comparator got popularity only as a decile — so part of the
margin was the head reading a feature we gave it. Zeroing those two columns and refitting the
standardisation: **0.98008 vs 0.98167, i.e. 91.6% of the margin survives**, and the
**interview-accrued portion is untouched (+0.0172 of +0.0175)**. What shrinks is the position-0
static read, +0.0047 → +0.0033. `HELDOUT_QUESTIONS.md` now **leads with the decomposition** and
claims only the interview-accrued part.

### What the chapter can now say about generalisation
**The reaction head generalises to unseen questions; the action policy does not.** Predicting how a
person will react to a question nobody has asked them is learnable from the descriptor plus the
interview (Q). Deciding that an unseen question is the right one to ask is not — the policy's ranking
is first-rate exactly where it was supervised and no better than popularity outside it (P1, P1b), and
teaching it to condition on volunteered evidence makes it move without making it right (P2).
**Never write "the policy emits questions it was never taught" or "the policy adapts to evidence it
did not choose and benefits".** Both are measured negatives with pre-registered falsifiers fired.

**No test look was taken anywhere in Part I. `test_touched=false` on every artifact.**

---

## ★★★ 2026-08-15 — THE CAPABILITIES PROGRAMME (Part H) IS COMPLETE ON VALIDATION. NO TEST LOOK.

**AUTHOR SAID (verbatim):** *"plz run the experiments."* And, earlier, on why: *"policy is supposed
to eb scalable, generalisable (it knows similarity between concepts and items, tree doesn't), could
support new vocab, can support new states, most improtantly i guess. so i am ok not to beat tree on
these 5 levels or whatnot, but i wanna show how this architecture in general is better."*

Three experiments, one per claimed property, each with a pre-registered falsifier. Design sheet
`docs/design/PAPER_B_POLICY_DESIGN.md` **Part H**. Write-ups: **`docs/results/RESUMPTION.md`**,
**`docs/results/HELDOUT_QUESTIONS.md`**, **`docs/results/VOCAB_GUARD.md`**. Code
`src/instrument/{policy_resume,outcome_q,policy_guard}.py`. System under test: distilled policy (E1,
frozen) + reaction head (frozen). **The value model stays deleted.**

| | claim | falsifier | verdict |
|---|---|---|---|
| **R** resumption | "supports new states" | policy adapts and still loses to the tree | **did not fire — because the policy never adapted.** Capability STRUCTURAL, not behavioural |
| **Q** held-out questions | "it knows similarity" | held-out AUC ≤ the popularity backoff | **did not fire.** 0.98167 vs 0.96285, and it also beats the *rigged* per-question prior 0.96915 |
| **G** open-vocab guard | "knows what NOT to ask" | recovery needs out-of-pool → ~0% | **did not fire** (57% recovered at 13% out-of-pool) — **but the DUMB guard beats the learned one, 78% vs 57%** |

### R — the policy does not condition on evidence it did not choose
Seed each validation user with k volunteered items (popularity-weighted from their fold-in half,
seed 98765), then 16 questions. At k=2/4/8 the policy's interview is **identical to its unseeded one
for 98.99 / 97.52 / 94.89% of users**; policy − tree is **+0.00002 / −0.00031 / −0.00026**, every CI
including zero. **Redundancy parity is the mechanism made visible:** re-asks of a volunteered item,
per item question asked, are 5.4% for the policy against 5.5% for the tree and 5.2% for the static
order — the arm that CAN see the volunteered set avoids it at the rate of arms that cannot. Among the
1–5% of users where it does deviate it is **worse** (−0.01232 at k=4, n=248). Volunteered evidence
does pay, but through the **fold**: seeded tree − unseeded E1 = **+0.0206 full / +0.0353 tail at k=8
with identical questions**. The interview still adds +0.0734 / +0.0529 / +0.0307 over the seed alone.
**Writing rule: "supports new states" is an architectural property, never a demonstrated behaviour.**

### R2 (2026-08-15, author: "ok, go ahead") — the BELIEF-ONLY policy adapts, badly, and the falsifier FIRES
Hypothesis: volunteered evidence moves `z_t` but leaves the ternary asked-history EMPTY — a pairing
that never occurs in E1's training — so the both-inputs model falls back to its empty-interview
behaviour. The `z`-only ablation cannot make that distinction by construction, so it should adapt.
**It does, hugely: 87.66 / 94.59 / 98.60% of users change at k=2/4/8, 9.3 / 11.4 / 12.9 of 16
questions, scaling with the amount of evidence.** So the asked-history feature IS what freezes the
both-inputs policy. **But the deviations COST** — z-only minus its OWN frozen question order (same
fold, same evidence, only the questions differ) is **−0.01246 / −0.01234 / −0.00860**, every interval
excluding zero, and it loses to the tree at every k. **R's pre-registered falsifier fires for this
arm and is reported dead.** ★ And the redundancy diagnostic kills the "it is conditioning" reading:
re-asks per item question asked are **1.442 / 2.857 / 5.429%** against the tree's **1.417 / 2.900 /
5.459%** — no arm separates from any other. It changes 9–13 of 16 questions and re-asks volunteered
films at exactly the tree's rate. **Sensitive to the volunteered evidence, not informed by it.**
Neither policy has a behavioural "supports new states" capability; the fine-tune case is now exact,
and the fine-tune was NOT run.

### Q — the reaction head generalises to questions it has never seen
20% of the 19,390 questions held out by identity, stratified by channel × popularity decile; the fair
comparator is the **(channel, decile) backoff prior**, because a held-out question has no per-question
row. Held-out-question AUC macro **0.98167** vs backoff **0.96285** and vs the rigged per-question
prior **0.96915**, over 620,606,659 scored cells on held-out questions × held-out users. **The curve
saturates early:** 10% of the training questions (1,551) buys **69.6%** of the margin, 25% buys 92.8%.
Seen/unseen gap **0.0011**. The margin decomposes: position-0 gap **+0.0047** and the shuffle
control's residual **+0.0040** independently price the DESCRIPTOR's own contribution at ~¼, and the
interview supplies the other ¾ (position curve 0.9676 → 0.9851; shuffle flat 0.9672 → 0.9665).
⚠ Composition caveat: the held-out set is tail-dominated (median question answerable by ~1 fit user
in 1,000), so the claim is **"who can answer an unseen question and what they say when they can"**.

### G — the guard works, and a table of answer rates works better
Opening to 19,390 costs **−0.02034**. Learned guard (hard mask on predicted P(no-answer), swept, whole
curve reported): best **τ=0.98 recovers 57.0%**, +0.01159 [+0.00976, +0.01345], keeping 12.98% of
picks out of pool answered at 25.7% (unguarded: 9.1%). **Budget-matched dumb guard** (state-free
per-question empirical answer rate, same number of candidates alive) recovers **78.1%**; head-to-head
**learned − dumb = −0.00429 [−0.00545, −0.00312]**, and the learned guard loses at 3 of the 4 budgets
where guarding pays. It wins only at tight budgets (+0.0111 / +0.0125 / +0.0090 at 987 / 744 / 499
alive), where a fixed list collapses. Pushed to τ=0.50 the policy answers 83.8% of questions and
scores 0.047 below the ceiling — **answerability is a constraint, not the objective**. And the 499
most-answerable candidates in the catalogue score **0.056 below the 1,000-candidate pool**, so "the
pool was already right" is too strong: the pool is right for reasons a guard cannot see.

### The sentence the chapter now has to carry
**The bank beats conditioning, three times over:** +0.01430 > +0.01035 (static bank vs the tree on
test), the mean value vector beat the learned lift head, and now the marginal answer rate beats the
state-conditioned reaction head as a guard. Q is the one place conditioning demonstrably wins —
and it wins on PREDICTION, not on question selection.

### Not run, and awaiting the author's decision
**The evidence-seeded fine-tune** (the 138 random-prefix states from 2026-08-14, measured labels,
zero new cost) — R's designated fix. It is a new training arm rather than part of the authorised
three, so it goes to the author as its own decision.

### Cost note (three misses in one day, recorded)
R estimated 3–5 h, ran in **12 min**. Q estimated 4–6 h, ran in **2.6 h**. G estimated 2–3 h, ran in
**36 min**. The scoring funnel does a full 10,000-user cohort in ~15 s; estimates built from
candidate-sweep intuitions do not transfer to one-evaluation-per-cell experiments.

---

## ★★★ 2026-08-13 — WORKSTREAM E1 (THE PARAMETRIC POLICY) IS COMPLETE ON VALIDATION. NO TEST LOOK.

**Author, verbatim:** *"plz start on policy and tell me eta"* · *"i want it to scale not only by
interview length, but also by vocab … continuous embedding emitted and snapped to closest vocab
question is my preferred design"* · *"so we train policy on train dataset, but we do not regrow the
tree, and we select model variant or chkp on val. is that right?"* — confirmed.

Full write-up: **`docs/results/E1_POLICY_IMITATION.md`**. Design sheet
`docs/design/PAPER_B_POLICY_DESIGN.md` (D.6b binding). Code `src/instrument/train_split.py`,
`policy_traj.py`, `policy_net.py`.

**Every E1 gate passes, and the result is that the policy is an EXACT functional replica of the tree
on the 10,000 validation users:** top-1 agreement **1.0000 at every one of the 16 positions**; its own
rollout (it picks its own questions) reproduces the tree's 15 realised interviews; its gain over
`greedy_static_policyB_combined` is **+0.027939557 [+0.02493, +0.03099] full** and **+0.01055
[+0.00749, +0.01373] tail** against the tree's banked **+0.027939556** — agreement to 1e-9. The
shuffle control lands at **−0.01489 [−0.01803, −0.01172]** and fails every gate.

**E1 therefore buys NO gain over the tree and was never designed to** (design sheet A.0). It is the
instrument check that licenses E2.

### The two things E1 measured that change what E2 must do

1. **`frac_never_deviates = 1.0000`.** A perfect imitator never leaves the teacher's trajectory
   manifold, so **E2's DAgger step is the only mechanism in the design that can produce an off-tree
   state at all** — harvesting "the states the trained policy visits" would harvest the tree's own
   states back. The labeller must be driven to deliberately chosen states (below the floor stops).
2. **Open vocabulary: the head scales, the VALUE is not priced.** 19,390 candidates in one forward
   pass with no retraining, **29.99% of picks out-of-pool, 189 distinct never-trained questions, 82.3%
   of users get one** — the mechanism claim is demonstrated. But those picks are answered **9.1%** of
   the time against **55.2%** in-pool, and full NDCG drops **−0.02034 [−0.02236, −0.01837]**. Weak
   method, not a verdict on OOV (HARD RULE 6).

Ablations: `z` alone recovers 99.5% of the decisions and 97% of the gain (very nearly a sufficient
statistic, not exactly); dropping the position/budget feature changes nothing (+0.02796), which
**closes** the A.4 asymmetry rather than merely declaring it.

### Next decision points for the author

- E2 (labels below the tree's floor stops + the DAgger loop) — the payoff step, and the only one that
  can produce a gain. Pre-registered no-go stands: **do not take a test look unless E2's validation
  gain over the tree is ≥ +0.011 full**.
- The E1 test look is a **separate author decision** and has not been taken.
- ⚠ **Disk:** `C:` has ~1.1 GB free; the E1 state cache is 1.80 GB. E2 will need space freed first.

---

## ★★★ 2026-08-12 — NDCG-OBJECTIVE INTERVIEW TREE: TEST LOOK RUNNING; POLICY-B COMPARATOR REBUILD **PAUSED**

**Author, verbatim:** *"i do want 16 or consistency"* · *"actually, i think adding 16th q and then
running on test would be right, wdyt?"* · *"add 16th step, run on test. we are changing comparator, we
are NOT changing this tree, don't see a problem"* · *"i also need total ndcg for the tree. we will
compare to current static with assumption that current static is weak."*

Design sheet `docs/design/DS_NDCG_TREE.md`. Code `src/instrument/ndcg_tree.py` (grower),
`ndcg_tree_extend16.py` (comparator rebuild), `ndcg_tree_test16.py` (extend + test look).

**The tree is FROZEN and must not be regrown:** `experiments/battery/ndcgtree.json` — depth 5, 22
nodes, 15 leaves, grown 10.33 h on the 10,000 validation users, combined pool (500 pop items + 500
answerable concepts), min-leaf 200 as the only stopping rule, every question charged and folded.

### PAUSED JOB — the policy-B static combined comparator (RESUME THIS)

`greedy_static_i26c.json`'s `combined` arm was **selected under policy A** (confirmed:
`docs/results/MASK_POLICY_B_MIGRATION.md` line 86 records its `intercept_eval` 0.13139/0.02343 as the
policy-A bridge; policy B is 0.15855/0.02819). The rebuild resumes the CELF heap the tree's root node
already built — that heap **is** the search's 1,000-candidate seed pass.

**Killed cleanly at 15:07 on 2026-08-12 with 7 of 16 picks banked** (pick 8 in progress was lost,
~12 min). Banked sidecar:
`experiments/battery/ndcgtree16_heaps/greedy_static_policyB_combined_heap.json` — 7 picks, nev 1497,
993 heap entries, cur 0.23253419940036288.

Picks so far: `[[1,114],[1,173],[0,19],[1,748],[1,785],[0,83],[0,583]]`

**EXACT RESUME COMMAND** (do not delete the heap sidecar; it is the only thing that makes this cheap):

```
schtasks /Run /TN casper_ndcgtree16
```
or directly:
```
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8 & set OPENBLAS_NUM_THREADS=8 & set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument
dcg_tree_extend16.py --full_threads
```
It re-resumes from the banked heap automatically (contract-signed) and continues at pick 8. Remaining
cost ≈ 9 picks × ~30–40 min ≈ **4.5–6 h** at the observed rate (~41 min/pick late in the search).

### RUNNING NOW — the test look (`ndcgtree16test`)

Extends the 15 leaf tails 15→16 from their banked heaps (prefix-unchanged asserted), re-measures on
validation, then routes the 10,000 canonical TEST users through the frozen tree and scores K ∈
{1,2,4,8,16}. Comparator is the **policy-A-selected** `combined` order, labelled as such everywhere.
Paired percentile bootstrap, 10,000 resamples, seed 98765. Absolute NDCG for both arms plus the m=0
intercept on test are reported, not only differences.

**Validation result at 15 questions (in-sample, against the policy-A comparator):** cohort-weighted
+0.0314 full / +0.0226 tail at K=15; +0.0278 at K=8. **Spearman(n_users, gain) = −0.55 (p=0.035)** —
small leaves gain most, the overfitting signature. Leaves ≥700 users +0.0223, leaves <700 +0.0464.


> The single current-status doc. **Overwrite as things change.** Pointed to from `MEMORY.md`.
> End-state is in `VISION.md`. Last updated: **2026-08-11 (genre-split conditioned greedy MEASURED ON TEST: FALLBACK-200 +0.00817 [+0.00539,+0.01097] full NDCG@10 at K=15, but Drama — 55% of users — is a null and all 17 groups shrank val→test)**.

## ★★★ 2026-08-11 — GENRE-SPLIT CONDITIONED GREEDY, MEASURED ON TEST (`groupgreedy_genre15`)

**Author instructions, verbatim:** *"can you not fold in the genre question? but instead partition by
genre, evaluate the static order vs adaptive order found per group? so 16 vs 16, would just show,
hopefully, that adaptive order is indeed optimal"* · *"only result on test users is interesting"* ·
*"ok, do 15 for now, that is fine. add missing groups and kill long run"*.

Design sheet `docs/design/DS_GROUP_GREEDY_GENRE.md` (stage-2 plan **pre-registered and committed
before any test number existed**). Code `src/instrument/group_greedy_genre.py` +
`group_greedy_genre_test.py`. Artifacts `experiments/battery/groupgreedy_genre15{,_TEST}.json`
(+ `_per_user.npz`). **All tasks deleted; nothing is running.**

### THE HEADLINE (TEST, 10,000 users, paired bootstrap 10k resamples, seed 98765)

**Primary endpoint — cohort-weighted group-order minus global-order, full NDCG@10 at K=15:**

| arm | diff full | 95% CI | diff tail | 95% CI |
|---|---|---|---|---|
| **ALL-GROUPS** | **+0.00955** | [+0.00668, +0.01243] | +0.01071 | [+0.00770, +0.01373] |
| **FALLBACK-200** (deployable) | **+0.00817** | [+0.00539, +0.01097] | +0.00891 | [+0.00598, +0.01185] |

Global at K=15 = 0.24356 full. So the conditioned orders are **~+3.9% relative** — real, separated
from zero, and **an upper bound**: the genre answer is charged nothing (see below).

**It only appears at long budgets.** Both arms are a NULL at K=1,2,4 (all CIs cross zero) and turn
on at K=8: ALL-GROUPS +0.00144 / +0.00121 / +0.00239 / **+0.00814** / **+0.00955**. Conditioning
does not buy a better *first* question; it buys a better *tail* of the interview.

### THE FINDING UNDERNEATH: it is not the big groups

| group | val n | test n | val gain | **test gain** | shrinkage | overlap@K15 |
|---|---|---|---|---|---|---|
| Drama | 5,481 | 5,493 | +0.00697 | **−0.00111** [−0.00461,+0.00245] | −0.00808 | 6/15 |
| Action | 1,582 | 1,545 | +0.02594 | **+0.01686** | −0.00908 | 2/15 |
| Comedy | 1,475 | 1,445 | +0.02270 | **+0.02212** | −0.00059 | 4/15 |
| Adventure | 534 | 565 | +0.05565 | **+0.03244** | −0.02321 | 2/15 |
| Thriller | 451 | 499 | +0.03524 | **+0.02291** | −0.01233 | 3/15 |
| Sci-Fi | 161 | 142 | +0.10498 | **+0.08955** | −0.01543 | 2/15 |
| Romance | 88 | 90 | +0.06414 | +0.02655 (ns) | −0.03759 | 1/15 |
| Crime | 79 | 80 | +0.08940 | **−0.04331** | −0.13271 | 0/15 |
| Horror | 51 | 56 | +0.06517 | −0.00131 (ns) | −0.06648 | 1/15 |
| Animation | 44 | 25 | +0.13971 | +0.03805 (ns) | −0.10166 | 0/15 |
| Children | 29 | 32 | +0.12155 | **+0.09427** | −0.02729 | 2/15 |
| Fantasy | 11 | 15 | +0.12734 | **−0.11562** | −0.24296 | 2/15 |
| Musical 3·Documentary 3·Mystery 3·War 2·Film-Noir 1 | | 4·3·2·1·3 | +0.024…+0.311 | ~0 / −0.036 / +0.085 | −0.024…−0.303 | 1–2/15 |
| **Western** | **2** | **0** | +0.00000 | **empty on test** — reported, not merged | | 2/15 |

* **Drama is a NULL on test** — and it is **54.9% of the cohort**. The headline is carried entirely by
  Action, Comedy, Adventure, Thriller and Sci-Fi.
* **EVERY ONE OF THE 17 GROUPS SHRANK from validation to test. Not one held its gain.** That is the
  overfitting measurement, and it is unanimous.
* **Shrinkage tracks group size almost perfectly.** Comedy (n=1,475) lost 0.0006; Fantasy (n=11) lost
  0.243; Mystery (n=3) lost its entire +0.303. The validation table's beautiful gain-vs-size gradient
  (+0.007 at n=5,481 rising to +0.311 at n=2) was **almost all noise**, exactly as pre-registered.
* **The conditioned orders are genuinely different** — 0–6 of 15 items shared with the global order,
  and no group reproduced its positions. So this is not "there was nothing to condition on"; the
  orders diverge hard and *still* mostly fail to generalise.

### Which arm is the headline

**FALLBACK-200 (+0.00817).** It is the only one that describes a shippable policy: you cannot deploy
an order fitted on 3 users, and ALL-GROUPS gets part of its number from groups (Children n=29,
War n=1) whose test cells are noise in both directions. The two arms agree closely (+0.00955 vs
+0.00817) precisely because the sub-200 groups are only 4.53% of test users — which is itself the
point: **the small groups are loud in the per-group table and nearly irrelevant in the cohort mean.**

### The caveats that must travel with this number

1. **The genre answer is FREE.** No turn is charged and it is not folded into the recommender — the
   comparison is 15 items vs 15 items. A real system pays a turn for it. **+0.00817 is an upper
   bound**, and at K=1–4 the gain is indistinguishable from zero, so a charged turn would very
   plausibly erase it at short budgets.
2. **Validation `sanity_pass` is meaningless as evidence** — at 15-vs-15 on the selection cohort the
   group order is guaranteed to win. All 18 passed; that only says the harness is correct.
3. Validation cohort-weighted gain was **+0.02046** at K=15 vs **+0.00955** on test: **the selection
   cohort overstates the effect by ~2.1×.**

### Provenance

* Four orders (Drama, Action, Comedy, Adventure = 9,072 of 10,000 val users) were **imported** from
  the abandoned 15-step run after an **11-field contract assertion** (tower, concept ckpt, objective,
  bank, pool size, branch rule, split, cohort size, L, sanity flag, full group-size map) — **all
  matched**, logged field by field into `contract_assertion`. The other 14 groups were searched here
  (19.2 min). Curves/overlap/per-user vectors were recomputed for all 18 by the same code.
* Grouping honesty asserted on **both** cohorts: 0/10,000 validation and 0/10,000 test users have a
  held-out target inside the grouping lookup. Test grouping used test users' own fold-in only, via
  the *imported* `partition_by_genre`.
* Genre distributions are stable val→test (Drama 54.81→54.93%, Action 15.82→15.45%).
* **Defect found and fixed:** the stage-1 summary emitted keys named `@16` while the values were
  computed at K=15. Values were always right; keys renamed in place to `@K15`, `key_name_correction`
  recorded in the artifact, and the source now derives the key from K.

### Next

Nothing is running. Open question for the author: this is the *upper bound* — the obvious next cut is
to charge the genre turn (14 items + 1 genre question vs 15 items), which the K=1–8 curve suggests
would eat most of it at short budgets.

## ✅ ABANDONED 2026-08-11 19:20 — 15-step/charged-turn variant (`groupgreedy_genre`)

The answerable-branch successor to `groupgreedy_d1` below. That run branched on ONE popular ITEM and
**70.45% of the cohort could not answer it**; the conditioning gain in exactly that 70% was **+0.0019**
against **+0.0129** (like) and **+0.0237** (dislike). So depth-1 was limited by ANSWERABILITY, not by
conditioning. This run swaps the branch for an **OPEN "what is your favourite genre?"**, answered
faithfully from the user's own **LIKED FOLD-IN** items. Design sheet
`docs/design/DS_GROUP_GREEDY_GENRE.md`; code `src/instrument/group_greedy_genre.py`.

* **Branch rule:** most frequent MovieLens genre over liked fold-in items (level ≥ 7 = ≥ 4.0 stars),
  **ties alphabetical**; no liked/genred item → explicit `none` group (**empty in practice: 0 users**).
  Genres from `data/movielens/movies.csv` × `data/ml-25m/proc/unique_sid.txt`; **18,235/18,359
  (99.32%)** catalogue items carry ≥1 genre; sid map asserted equal to `ctx.show2id`, 0 mismatches.
* **Honesty asserted, not assumed:** `_leak_assert` confirms 0/10,000 users have a held-out target
  inside the grouping lookup.
* **18 groups over 10,000 VALIDATION users** (nothing merged, nothing dropped — HARD RULE 1):
  Drama 5,481 · Action 1,582 · Comedy 1,475 · Adventure 534 · Thriller 451 · Sci-Fi 161 · Romance 88
  · Crime 79 · Horror 51 · Animation 44 · Children 29 · Fantasy 11 · Documentary 3 · Musical 3 ·
  Mystery 3 · War 2 · Western 2 · Film-Noir 1. **13/18 are under 200 users and are flagged
  `small_group: true` / statistically untrustworthy.** Merging is an author decision, not a runtime one.
* **The genre answer conditions the SELECTOR only** — it is not folded into the recommender — and it
  **COSTS A TURN**. Two comparisons are therefore stored side by side:
  * `curve_total_budget[T]` (**headline**): group's first **T−1 items** vs global's first **T items**,
    T ∈ {1,2,4,8,16}.
  * `curve_item_matched[K]` (**containment**): K items vs K items. **`sanity_pass` gates on this one
    only** — the "global order is inside the search space" argument is true at matched item count and
    FALSE at matched total budget, where a shortfall is a finding, not a bug.
* **Sanity mode PASSED** (Thriller, 451 users, L=2, 13-item pool incl. the comparator's whole head):
  item-matched own `[583,123]` **0.22593** ≥ global `[583,123]`-prefix **0.22392** (**+0.00201**);
  total-budget @3 turns **0.22593** vs global 3 items **0.22654** (**−0.00061**) — i.e. at a 3-turn
  budget the genre turn does not pay for itself. 0.891 s/eval at 451 users.
* **VALIDATION ONLY.** The test cohort is not built, not loaded, not touched.
* Outputs (unique label, shared with nothing): `experiments/battery/groupgreedy_genre.log`,
  `groupgreedy_genre_partial.json` (rewritten after EVERY pick), `groupgreedy_genre.json` at the end.
* **Launched 19:05, task Status=Running, PID 22740.** Liveness confirmed on the ARTIFACT, not a PID:
  the log grew across samples 19:07:32 → 19:11:59 → 19:16:13 (>3 min apart) and process CPU time
  climbed 45.3 s → 55.1 s → 92.4 s over the same window.
* **Measured s/eval: 0.891 at n=451, 10.42 at n=5,481** → almost exactly proportional,
  `s/eval ≈ 1.90e-3·n + 0.04`. (d1's points 1.62@856, 4.85@2,099, 12.78@7,045, 13.54@10,000 sit on
  the same line except at 10k, where it flattens.) **ETA ≈ 3.9 h → ~23:00**, dominated by Drama
  (≈2.0 h); the 13 sub-200 groups together cost under 20 min.

**Driven by a Windows scheduled task.** Delete it with:
`schtasks /Delete /TN casper_groupgreedy_genre /F`
It needs `DisallowStartIfOnBatteries=$false` + `StopIfGoingOnBatteries=$false` — the box is on
battery, and with the `schtasks /Create` default the task sits at `Status: Queued` reporting
`Last Result: 0` while doing nothing.

## ✅ FINISHED 2026-08-11 03:12 — depth-1 group-conditioned greedy (`groupgreedy_d1`)

**Result (validation, item-matched at 16):** like (n=2,099) **+0.0129** · dislike (n=856) **+0.0237**
· did-not-answer (n=7,045) **+0.0019**. All three sanity gates passed. s/eval 4.853 / 1.621 / 12.784.
Task deleted. Superseded as the live job by `groupgreedy_genre` above.

The one-branch-point control for Act 1: same objective, same optimiser (`greedy_certified`), same
top-500-popularity item pool, same cohort — the ONLY thing that varies is whether the 15 steps after
q1 were chosen knowing the answer to q1. Design sheet `docs/design/DS_GROUP_GREEDY_D1.md`; code
`src/instrument/group_greedy_d1.py` (committed `795fcd3`, fix `d8966bf`) — `greedy_certified` gained
`prefix=`/`on_pick=`, bit-identical with `prefix=()`.

* **q1 = (channel 0, item 83)**, READ from `experiments/battery/greedy_static_i26c.json` arm
  `items-only` step 0, never re-derived.
* **Partition of the 10,000 VALIDATION users** on q1 by the harness's own behavioural rule:
  **like 2,099 · dislike 856 · did-not-answer 7,045**. Exact partition, nothing dropped.
* **VALIDATION ONLY.** The test cohort is not built, not loaded, not touched.
* Outputs (unique label, shared with nothing): `experiments/battery/groupgreedy_d1.log`,
  `groupgreedy_d1_partial.json` (rewritten after EVERY pick), `groupgreedy_d1.json` at the end.
* Sanity mode PASSED end to end first: on the 856-user dislike group, own `[83,21,19]` **0.20441** ≥
  global `[83,583,19]` **0.19836** (+0.00605), 1.637 s/eval.
* **Cost finding (this was the point of the abandoned "cost smoke"): per-eval scales with cohort
  size.** 1.637 s/eval at 856 users vs the recorded **13.543 s/eval at 10,000**. Fits
  `≈0.5 + 1.3e-3·n`, so ETA ≈ **3.5 h** total (≈03:00), not 3 × 3.2 h.

**Driven by a Windows scheduled task.** Delete it with:
`schtasks /Delete /TN casper_groupgreedy_d1 /F`
It needed `DisallowStartIfOnBatteries=$false` — the box is **on battery**, and with the schtasks
default the task sat at `Status: Queued` reporting `Last Result: 0` while doing nothing. If a future
scheduled task "runs" and produces no output file, check the battery flag first.

**Also removed:** the leftover one-time task `casper_act2pv_score`, still queued for 23:59 tonight
even though the section below records it as deleted. It would have started the ~11 h Act-2 scorer
alongside this run (HARD RULE 11d). Resume is unchanged and documented below.

## ★★★ 2026-08-10 — TODO 1.2 SETTLED: the ACTION SPACE was not the constraint. **Mechanism hypothesis RETRACTED.**

`docs/results/ACT2_CONCEPT_BANK_PV.md` + `.json`, design sheet `docs/design/DS_ACT2_CONCEPT_BANK_PV.md`,
from ONE `adaptive_harness` run (label **act2pvc**, 28 cells, 43 min) analysed by the SAME
`scripts/act2_pv_quartiles.py` that produced the parent table — one script, two tables, so they are
commensurable by construction. Arms: `conlinucb_bs_static`, `conlinucb_mcr`, `conucb`, `dpe_concept`
vs `fixedorder:entropy0` / `greedy_concepts` / `greedy_items`, k ∈ {1,2,4,8}, all 10,000 TEST users,
certified pair, NO-FLOOR, policy-B pool, behavioural answers on both channels, **`--concept_neg_cap 0`**
(the certified-era convention cv3 was trained under) and **`--keyterm_repr uniform_member_mean`** (the
value all four carried-forward Act-1 concept-asker cells used — read off their run JSONs, not guessed).

**The question.** Act 2's one mechanism hypothesis was: the loss-optimal question is a CONCEPT for
~100% of users at every budget, and every adaptive selector in that battery shops in the ITEM bank.
Give an adaptive selector the concept bank and it should convert PV.

**It does not.** Of **96** Q4−Q1 trend cells, **77 separate from zero and 75 are negative**; the shuffle
control separates **1 of 96**. The two positive cells are `conlinucb_mcr` vs `greedy_concepts` at k=2,
where channel accounting shows that arm asked **zero** concepts. Against `greedy_concepts` at k=8 the
trends run **−0.038 to −0.080**, *deeper* than the item-bank battery's −0.021…−0.035 against
`greedy_items`. **The concept askers fail harder, not less.**

**Why this is not an unfired-schedule artefact (control 5).** `conlinucb_bs_static` and `dpe_concept`
ask concepts on **100% of turns at every budget** and both are negative against `entropy0` everywhere.
`conlinucb_mcr`/`conucb` ask **0% at k=1,2 and 25% at k=4,8** — `b(t)=5⌊log t⌋` fires at turn 3 and turn
8 only — so their k≤2 rows are reported as **uninformative, not as a null**. Restricted to the two arms
that genuinely condition the next concept on previous answers, at k=4/8: **21 of 24 trend cells
separate, all negative**, −0.0111…−0.0665.

**Answerability is not the escape.** The fixed concept order is answered on **90/89/90/79%** of turns at
k=1/2/4/8, vs **44/43/41/39%** for `entropy0` and **30/28/32/30%** for `greedy_items`. Roughly 3× the
answers per turn, the criterion says a concept is optimal for ~100% of users, and adaptive still loses.

**The concept bank's best use is a FIXED ORDER.** `fixedorder:greedy_concepts` is the best arm in the
table at every budget on both axes — **0.2362 / 0.1022 at k=8** vs `greedy_items` 0.2247/0.0776 and
`entropy0` 0.2053/0.0792. Every concept-*asking* adaptive arm sits far below all three static orders
(best of them `conlinucb_mcr` 0.1902/0.0714 at k=8).

**Caveat, measured not argued.** `conlinucb_bs_static` conducts **1 distinct interview** over 10,000
users — the barycentric spanner is user-independent by the paper's own construction — so its row is a
**bank** result (the spanner is a bad bank: −0.0540 full vs `greedy_concepts` at k=8, refusal rate
0.129→0.581), not an adaptivity result. `dpe_concept` branches 10,000 ways but its published
initialisation makes that a seeded random permutation.

**CONTROLS — all passed, all printed in the doc.**
* **entropy0 REPRODUCTION GATE — PASS.** Interview payload byte-identical between `act2pv` and `act2pvc`
  for `fixedorder:entropy0` **and** `fixedorder:greedy_items` at k=1,2,4,8, and the **scored per-user
  vectors agree at 0.0e+00** on all 16 (spec, budget, axis) cells. This is what licenses reading PV off
  the banked `act2pv` criterion cache; nothing on the PV side was re-scored or re-cut.
  (`scripts/act2pvc_gates.py --gate`; two fields are excluded and *printed* — `info`, i.e. wall-clock
  timings, and `_contract`, which carries the `concept_neg_cap` 1→0 the design sheet requires.)
* **Self-difference** over 224 (arm, k, axis, quartile) cells: **0.0e+00 exactly**.
* **Shuffle**: 1 of 96.
* **Answer rate per quartile** printed; ρ(PV, answer rate) **+0.758 at k=2** is stated as a "did you
  answer" split — **k=4/8 carry the result**.
* **Channel accounting** (control 5): section 1 of the doc.
* **Free cross-run control**: before its schedule fires (k=1,2) `conlinucb_mcr` reproduces `glm_ucb`'s
  per-user vectors from the *parent* run at **0.0e+00** on both axes — same LinUCB index over the same
  `GaussianBelief` while `M_t` has no key-term row — and diverges at k=4/8 where key-term rows enter.
  The identity breaks exactly where the suspected cause appears.

**REGRESSION GATE (HARD RULE 8).** `scripts/act2_pv_quartiles.py` was EXTENDED, never forked, so both
tables share the quartile cut, the paired bootstrap, the trend bootstrap and the shuffle. Re-running
`--analyze` with no new flags reproduces `ACT2_PV_QUARTILES.json` byte-identically over all 121,089
bytes once the two structurally volatile keys are removed, and the `.md` differs on **2 of 420 lines** —
both halves of the `Run <timestamp> ... (sha256 ...)` header, neither of which can be constant across
any two invocations. The published files were restored from git afterwards.

**CONSEQUENCE — the retraction is made, in place.** The mechanism paragraph in
`docs/results/ACT2_PV_QUARTILES.md` is **withdrawn inside that document** with the numbers that killed
it. The claim hardens to: **adaptive selection fails to convert personalisation value even when the
optimal action type is available to it.** Next candidate, per the design sheet, is the `_pick_mixed`
extension of `glm_ucb` to the unified action space — the question has narrowed from "can they reach the
right action type?" (they can; it does not help) to "does ANY acquisition rule convert PV?". A null is
not citable on its own (author ruling); this one's job was to say which chapter sentence is true.

## ★★★ 2026-08-10 — ACT 2 MEASURED ON THE SAME USERS: the gap tracks personalisation value, BACKWARDS

`docs/results/ACT2_PV_QUARTILES.md` + `.json`, from `scripts/act2_pv_quartiles.py`. Per-user **PV(u) = U(q*_u|z_u) − U(q*_pop|z_u)** on the leg-1 loss-coupled criterion, all
10,000 TEST users at k ∈ {1,2,4,8}, certified pair (tower `t2i26c_ep31` + concept `cv3_ep08`), NO-FLOOR,
behavioural answers, comparators differenced IN-RUN from one `adaptive_harness` process
(`adaptive_table_act2pv.json`).

**The finding.** Of 48 Q4−Q1 trend cells, **38 separate from zero and ALL 38 are negative; none is
positive**; under the PV shuffle **0 of 48** separate. The adaptive selectors lose *hardest* where
personalising is worth most (Thompson vs entropy0 at k=8: +0.0004 ns in Q1 → −0.0285 in Q4). This is
the **"methods cannot find it"** branch, not "nothing to adapt to" — and it redirects Act 4.

**Second finding.** `golbandi_tree` — the only arm that beats the static order on our consumer
(+0.0064 full / +0.0052 tail at k=8) — earns *all* of it in the PV=0 quartile (+0.0243 → −0.0003 across
Q1→Q4). Its crossover is a better static opener for the un-personalisable, not exploitation of PV.

**Confound, stated plainly.** ρ(PV, answer rate) = **+0.76 at k=2** (so the k=2 cut is largely "did you
answer"), +0.35/+0.41 at k=4/8; ρ(PV, profile length) only +0.10…+0.30, and mean profile length is
NON-MONOTONE and *lowest* in Q4 at k=4/8 — profile size is not doing the work at the budgets that carry
the result.

**Branching ratio (§2).** `entropy0`: **1** interview → 5,767 distinct states at k=8 → only **78**
distinct loss-optimal questions, modal question optimal for 25% of users. Thompson at k=1 spreads **760
distinct interviews across 25 distinct loss-optimal questions** — 0.033 optimal questions per interview.

**FRONTIER STOPPED at 12,019 / 56,231 states — 11 of 20 §6 cells scored. This was a decision, not a
failure.** Scored: the whole PV arm (`fixedorder:entropy0`, k=1/2/4/8 — everything §§1–4 need),
`golbandi_tree` / `glm_ucb` / `fixedorder:greedy_items` at k=1,2, and `thompson` k=1. The remaining 12
cells are the q\* columns at k≥4 and are marked *not scored*, never estimated; their interview and state
counts are exact regardless, because those need no criterion evaluation.

*Why stopped:* ~11 h of box time for a sharper version of an argument §5 already makes **exactly** — a
deterministic selector conditioning on a fixed order has at most 112 distinguishable inputs at k=2, which
bounds it without any criterion scoring — plus Thompson k=1, which already carries the ratio. Under HARD
RULE 11(d) it was blocking the concept-bank test (TODO 1.2), the direct test of the chapter's mechanism
hypothesis. **Fully reversible:** `python scripts/act2_pv_quartiles.py --score` resumes from the
contract-signed sidecar in `.cache/act2_pv/`, then `--analyze` refreshes §6 in place (~9 min).

*Scheduled task removed.* The sweep had been driven by `scripts/act2_pv_score_daemon.cmd` under a Windows
scheduled task `casper_act2pv_score`; **that task is deleted and no scorer is live.** Five launches were
killed mid-run — the `.err` file holds exactly two bytes, `^C`, so these were console interrupts delivered
to the process group (which is why `cmd.exe` died alongside `python` and no traceback ever appeared), not
the job failing. If ever resumed, drive it from a real console, not a background launcher.

*Defect found and fixed in the same pass:* the §6 scope sentence counted `b.get('status')` — a key that
does not exist in the branching dict, so it was always falsy and every cell counted as scored ("20 of
20") beside a table showing 9 rows *not scored*. Now counts `b['scored']`; reads 11 of 20, which matches
an independently derived cell list.

## ★★★ 2026-08-10 — PAPER A CLOSED. Author read-through pending; not submitting yet. Focus moves to Paper B.

**State: 41pp, compiles clean, 165 automated checks resolving every numeric claim against its own source
file, 0 mismatches — including assertions that the checkpoint is the certified `t2i26c_ep31.pt`.** No
measurement work outstanding. Two Fable reviews closed: every point either actioned or declined with the
reason recorded (`docs/PAPER_A_FABLE_REVIEW_2026-08-09.md`).

**Do not reopen without cause.** Declined deliberately: a second dataset (limitation now stated in the
text instead); ordering-stability across checkpoints (available checkpoints are one training run); a
defensive paragraph on why our model is the fixed consumer (the argument is made once already, and a
longer version was judged to weaken it).

### What changed this week — the headline claim was rewritten

*Every ratings-native baseline gets worse with its ratings* is **dead**. Replaced by: **centring and
configuration did the damage, gradedness itself does roughly nothing at full profile.**
iALS-as-confidence **+0.0194**; RBMF **−0.0045 full / +0.0046 tail** (both separated, opposite
directions); user-kNN **−0.0934** (centring); ours +0.0144 at k=2, ~0 at full profile.
RBMF's old −0.070 decomposes as centring −0.0981, values-erased support −0.0499, values restored +0.0454.

A **weight-vs-estimand mechanism was proposed and FALSIFIED** by its own within-model test (estimand rule
scored .3580 vs .3459 for weight). Paragraph deleted. **Do not revive it.**

**The inversion now has a mechanism** (the last review item, ~30 min not the estimated 4 h): at k=8, HELF
returns **1.59** graded answers of eight against **3.10** (entropy0) and **3.05** (popularity). HELF's 1.59
reproduces a 1.6 published independently in a footnote. Fold-in consumers are paid per graded answer and
starve; Golbandi's tree is paid in partition, so an unanswerable question still routes a user. Stated
limit: E0 and P differ by 0.05 answers and still swap, so this explains why HELF divides the table, not
the rest.

### Verdict on file
ACM TORS regular paper. **70–75% after one revision round** (second review, with the mechanism paragraph
in). Avoid SIGIR/TOIS — "the method loses to RecVAE by 1%" triggers a reflexive reject there regardless of
the methodological contribution.

### Next: Paper B
Live queue: **#52** (GO/NO-GO on across-user dispersion of top-K(mu) — decides the whole selector line),
#53, #55, #56, #68, #70, #76. **Ready artefact:** `experiments/baselines/interview/adaptive_table_bald_aopt_v1.json`
completed 2026-08-08 — BALD / A-opt / GLM-UCB / entropy0 / greedy-items × budgets 1–16 on the certified
tower, 10k users, per-cell `answered` counts included. Nothing has been read from it yet.

## ★★★ TOP OF THE QUEUE — THE ORGANIC OPEN REQUEST (task #68, author idea 2026-07-31, CORRECTED SAME DAY)
Memory: `reddit-mined-open-request-opener.md`. Full lit record:
`external_literature/findings/organic_open_request_novelty.md`. Design: `docs/design/PAPER_B_DESIGN.md`.

**The idea.** The open opener need not be a question *we* ask — it can be the **proactive request the user
arrives with** (*"something like Inception but less confusing"*), **mined from real Reddit threads**.

**THE MORNING FRAMING WAS WRONG IN THREE PLACES.** Deep research returned the same day.

**Dead — the data contribution.** It is not "ReDial with extra steps", it is **narrative-driven
recommendation with extra steps**: a decade-old published programme with a CLEF shared task.
**Bogers & Koolen** (RecSys'17, KaRS'18, INEX/CLEF Social Book Search 2011–16) define it verbatim —
*"a narrative description of the aspects of items desired by the users (≥1 sentence)… The narrative must
describe **a open request for recommendations** as opposed to locating a specific item"* — over 115,899
LibraryThing threads, and **they have the outcome signal we lack** (SBS grades: suggested=1, already
owned=0, **subsequently added to the requester's catalogue=8**; `<request>`/`<examples>`/`<catalog>` XML;
94,656 profiles; nDCG@10; 46 runs from 10 institutions in 2016). **Eberhard et al. IUI'19** already built
the exact r/MovieSuggestions→IMDb artifact. **He et al. CIKM'23** already published our anti-circularity
argument in print. Cold-start-as-differentiator is also already in Eberhard. The only surviving field-wide
statement: **no public dataset in the MOVIE domain has both an organic open request at turn 1 and an
out-of-dialogue consumption outcome.**

**Wrong — "it cannot be circular".** The *opener* can't; **the ground truth can.** Reply titles come from
the same self-selected community, and He et al. document the leak we would inherit: **>15% of INSPIRED
ground-truth items are already named earlier in the conversation, a trivial copy-the-history baseline beats
most CRS models, and stripping repeats drops #HIT@1 by >60%.**

**Wrong — "votes rank the replies". DO NOT USE VOTES AS GRADED LABELS.** GuessTheKarma (CSCW'18, 20,674
influence-free judgments): higher-scored item preferred only **68%** of the time. Glenski & Weninger
(HT'15, N≈93,019): one artificial upvote → **+11.02%**, one downvote → **−5.15%**, and **the corrective
effect does not replicate on Reddit**. **73% of votes are cast without viewing the content.** 51.52% of
eventually-popular links were previously submitted and ignored (Gilbert CSCW'13). IR/CQA calls these
**silver** labels. **Neither Bogers & Koolen nor He et al. use votes** — He ships an `upvotes` column and
the paper has zero occurrences of "upvote"/"vote". Use votes only as **a coarse binary inclusion filter**.

**HARD RULE 1 applies to the pipeline.** Of Bogers & Koolen's 1,457 annotated requests, **483 (33.2%) are
known-item needs, not recommendation needs**, and ~25% of the rest give neither examples nor context — an
entity-anchored pipeline silently discards **~42%** of genuine requests, biased toward users who already
name titles. That is a truncation and it is not neutral.

**WHAT SURVIVES — one cell, and it is the strongest posture available.** *The adaptivity gap measured
DOWNSTREAM of a non-circular organic opener, cold-start, with a ranking outcome.* The three pieces exist
and never co-occur: organic request + ranking with **zero** follow-up (Eberhard IUI'19/WWW'25, He CIKM'23,
Mysore RecSys'23, OCG-Agent EMNLP'25 — zero occurrences of "follow-up"/"multi-turn"); adaptive multi-turn +
ranking but curated request and role-played answerer (Qulac, ClariQ, ConvSim); real request + real signal
but one question and no ranking (Zamani WWW'20: *"In this work, we do not study multi-turn interactions."*).
In recommendation the openers are **target-derived** — EAR: *"we randomly choose an attribute from the
oracle set as the user's initialization"*, scored by Success Rate@t, not ranking. Negative searches:
`"narrative" AND "clarifying" AND recommend` → **0**; `"r/MovieSuggestions"` → **0**.

> **REFRAME: "what does an adaptive interview add once a real request has already been given?"**
> Eberhard/He become the **ruler**; the mined opener becomes **given evidence**, not the contribution.
> That converts our two most dangerous prior works into our baseline table.

**Usable now.** Aliannejadi's best case is an **open** question (*"Open questions are very hard to
formulate… however, it is more likely to get useful feedback"*) — independent support for the opener-type
hypothesis. **Krasakis ICTIR'20: on Qulac, negative answers DEGRADE ranking** (NDCG@20 0.130→0.106,
−18.5%) — a published contrast for our sign line (+0.0229 at k=2). Calibration target: **ConvSim +16%
nDCG@3 after one feedback round, +35% after three**, but with a curated opener and an LLM simulator.
One point in Reddit's favour: top-popular items are ~2% of ReDial ground truth but **<0.3% on Reddit**.

**FIRST DEBT before committing:** Koolen, Kamps & Kazai, **CIKM 2012**, "Social book search: comparing
topical relevance judgements and book suggestions for evaluation" — closed access, and literally the
validity study for the ground truth we would adopt.

**⚠ 2 of 9 sub-agents fabricated primary-source content before self-correcting** (retracted: BK-VAE
internals, Burke/Chen&Pu quotes, LangPTune mechanism, a fabricated "NEO arXiv:2603.17533", Eberhard's
field list). Anything not tied to an extracted quote is provisional.

## ★★★ PAPER A IS IN FINALISATION — A0–A6 plan, A1/A2 done
Chapter: `new_chapters/chapterA_v2/chapterA_v2.tex`. Compiles clean via Tectonic, **0 undefined, 0 TODO,
0 REDO markers** (all status macros removed 2026-07-28). Full review record:
`docs/results/PAPER_A_NUMBER_AUDIT.md`.

### The A-plan (tasks #59–#64, #58)
- **A0 (#64) ★ DECISION RULE, applies to all of A1–A5.** For every self-critical sentence ask: *does the
  paper read better with this in or out?* If out, cut. Evidence it matters: nearly every hit in the harsh
  review was a quote of something WE volunteered (our own 2.5/4 self-score, our own "not yet fitted", our
  own "illustrative rather than representative", our own myopia explanation). The reviewer discovered
  almost nothing; we handed it over. BOUNDARY: keep anything whose omission makes a claim false,
  unreproducible or unfairly attributed (protocol, ruler, seeds, baseline substitutions, and the failed
  snaps that changed what we report — both reviewers named those as the credibility engine). Cut process,
  hedging, self-scoring, pre-emptive defence.
- **A1 (#59) ✅** — contribution reframed onto the artifact that produced the numbers.
- **A2 (#60) ✅** — belief-MF recast; no longer printed under Bıyık's/ConTS's names.
- **A3 (#61) ✅ 2026-07-29** — every A3 item closed (list below, with what each turned out to be).
- **A4 (#62) ✅ 2026-07-29** — gap-map self-score removed, best-static table numbered + captioned with the
  CI scale, graded caption's missing "(a)" added, G7 explained in the results caption too.
- **A5 (#63) ✅ 2026-07-29** — defensiveness and archaeology out (commit `0ef5c4c`). NOT done: halving the
  lineage prose. Deliberate — a 30pp thesis chapter and a venue submission are different artifacts;
  length reduction belongs to the submission pass, not here.
- **A6 (#58) ✅ 2026-07-29** — cold pass done, whole chapter read start to finish. It earned its keep;
  five defects no grep would catch:
  1. **R3's formal statement contradicted the method section** ("the operator is identical for every
     channel" vs. "these are distinct operators") — the harsh review's one-fold-vs-three-mechanisms hit,
     still alive inside the requirement after being fixed everywhere else. R3 now constrains the
     *observation*, not the number of code paths.
  2. Two consecutive sentences contradicting each other on whether the belief layer preserves the G0 tie.
  3. A protocol paragraph promising a mixed-channel comparison that is both already done and partly
     impossible by construction.
  4. The iALS rule quoted a sentence we had already deleted ("below even iALS… withdrawn as evidence").
  5. "The belief layer is **specified** here" — the exact word that became the harsh review's
     "contradicts itself on whether its central component exists".
  Also removed: **C3 / the human study** (author ruling) — the battery is now **C1–C2** and no table,
  abstract or limitation promises an unrun study; the answer-model limitation is stated as fact instead.
  Status vocabulary no longer contains "queued": every unmeasured row reads **"not run here"**.

### A3, resolved — what each discrepancy actually was
- **sign-flip's three numbers were three different arms.** −0.2012 = the concept operator (`clite_gates`);
  −0.2985 = the full-profile item arm (`g3a.json`, 0.4279→0.1294). The graded figure's vague "up to −0.29"
  now cites G3a explicitly. All three now name their arm.
- **G3a/G3b reference values are full-profile** (whole history folded, no 80/20 holdout), which is why the
  G3b tail is 0.2999 against a 0.2497 frontier tail. Caption now says so and states that the gate is the
  *drop within* an arm, never a comparison across arms.
- **concepts@8 was on a SUPERSEDED fold.** 0.2091/0.1222 and 0.2219 came from the pre-fix fold; the current
  `cd_s1_l10_best.pt` gives **0.1759/0.0841 at m=8** (up from 0.1322/0.0503 at one) and redundancy
  **0.1572→0.1685** under correlated top-SEL. Paragraph rewritten on the current numbers. The additive-union
  crater (0.1349→0.0991) is fold-independent (identical in both ledger runs) and stands.
- **0.1717 was the ONE-ANSWER value, not the cold point.** The cold point is and stays 0.1279/0.0192.
- **0.1372 vs 0.1379**: retrain-gate arm vs the ledger's paired deployment sweep. Each now names its harness.
- **The double standard is gone.** Both differences are the same magnitude, so the paper now says the top of
  the ruler is one narrow band and RecVAE is *the bar* because it is the conservative choice — not because
  it is separated from EASE.
- **C2 in Limitations** now matches the battery: closed by construction, with C3 (human study) as the real
  outstanding item.
- **G2's canary is now reported**: one answer from cold lifts both channels on full and tail
  (items 0.1310/0.0219, concepts 0.1436/0.0323).

## ★★ TWO REVIEWS OF THE PAPER ALONE (2026-07-28, fresh sessions, tex + figures only)
**Balanced:** *"strong thesis chapter, solid-but-not-flashy publication."* **Harsh:** *reject as it stands*,
ceiling *weak accept*. Both agree the contribution type is methodological infrastructure — battery +
bridge + certified tie — with no accuracy win and no policy win (both deliberately out of scope).

**Venues (both converge):** TORS regular paper = best fit (major revision → accept); RecSys/ECIR
**reproducibility track** = strong; ECIR full paper = plausible and matches the 2027 timeline; SIGIR main
= no; UMAP = no.

**Both said to PROTECT in editing:** the self-disqualification paragraph (applying the snap rule to our own
iALS) — "the paper's credibility engine"; the false-positive taxonomy (sign-blind / circular / decorative);
C2-as-construction; the deployment-currency lesson; the answerability paragraph + its scholarship;
Eq.(region) and the mean/covariance asymmetry; the free-text illustration including its displayed failure.

**Harsh review's two structural hits, both now resolved:**
1. *"One fold" vs three mechanisms.* We claimed a single set-valued fold while shipping the T2′ set encoder
   (items), the gated fold-to-point (concepts) and the conjugate layer alongside. **Resolved (author chose
   option B):** claim the unification of the REPRESENTATION — one latent, one form of observation
   (direction, value) — not of the code path. **Key supporting fact now in the paper:** ranking on the
   conjugate posterior mean IS `belief_mf.py` and measures **0.2019/0.1200** vs the learned fold's
   **0.3482/0.2462** — uniformity costs ~0.15 full NDCG, which is WHY the belief sits beside the ranking
   path. That turns the most attackable number into the architecture's justification.
2. *belief-MF straw man.* Fixed: Bıyık/ConTS rows now carry a dash with the verified fact that neither
   reports full-catalogue accuracy (Bıyık's NDCG = top-|S| agreement over **16 users, slates of 5**).

## ★ OVERNIGHT 2026-07-28/29 — mixed
| job | result |
|---|---|
| **Mult-VAE** on the ruler | ✅ **0.3202 / 0.2194** (@100 0.4303) — below EASE/RecVAE, frontier claim holds |
| **RBMF seed** | ✅ **0.2534 / 0.1764** (@100 0.3607) |
| **TurboCF** | ❌ 0.0000 everywhere → **DIAGNOSED AND FIXED** (see below); re-run queued |
| **SASRec** | ❌ died 23:41 on `val_tr.csv`; the Liang split writes `validation_tr.csv`. Fixed + restarted. |
| **greedy polish, items-only** | ✅ **0.2098 → 0.2118** (1146 evals, converged) |
| **greedy polish, combined** | 🔄 running (was queued LAST by mistake; now runs in parallel via `--arms`) |

**Two self-inflicted nulls this round — watch for the pattern:** (1) `greedy_polish` built its swap pool
from the sequence itself, so every candidate was skipped and it reported "0 evals / no improvement" — a
confident-looking null that was a bug; (2) the earlier greedy ran on the SUPERSEDED fold because that was
the script default. Both were caught by inspection, not by the runs failing.

## ✅ Turbo-CF MEASURED: **0.2622 / 0.1621** (@100 0.3569) — `docs/results/TURBOCF_SWEEP_RESULT.md`
Val-selected $\alpha{=}0.5$, $s{=}1$, 2nd-order (val 0.2555). Mid-pack in the accuracy corner: above RBMF
(0.2534) and iALS (0.2442) on full, below Golbandi (0.3065), well below EASE (0.3476); its tail 0.1621 is
below iALS's 0.1853. In `tab:master` as its own row (R1 = partial); **GF-CF/BSPM split out as "not run
here"** rather than sharing a measured row. Flagged best-effort tuned — no published ML-20M/25M number.

**The val grid is the proof of the diagnosis.** ρ(P̄) = **1.000 exactly** at α=0.5, s=1.0 — symmetric
normalisation, no Hadamard power — and that is the ONLY cell where the 2nd-order filter helps
(0.2313 → 0.2555). Every ρ>1 cell collapses under it: 0.0001 / 0.0086 / 0.0001. At α=1.0, s=0.6, ρ=269 and
F reaches −1.9e3 with 99.9% negative entries. The winner is the public repo's own default; the paper's
tuned region (α=0.7, s=0.6) does not transfer to an 18k catalogue.

## TurboCF: a degenerate configuration, not a weak baseline (fixed 2026-07-29, commit `3f25c7f`)
Verified against the public source (`jindeok/Turbo-CF/main.py`): the code does **not** re-normalise
$\bar P$ after the Hadamard power, so the polynomial filters are low-pass only while $\rho(\bar P)\le1$.
Our default — the paper's tuned region $\alpha{=}0.7$, $s{=}0.6$ with the 2nd-order filter — pushes $\rho$
far past 1 at 18,359 items, the $-P^2$ term dominates, and the ranking **inverts**. Harmless on the
130-item smoke (which is exactly why the smoke passed), fatal on the real split.
Fixes: defaults revert to the repo-stable $(\alpha{=}0.5, s{=}1, \text{linear})$; `build_filter` estimates
$\rho$ by power iteration and warns loudly outside the stable region, logs F stats and rejects non-finite F;
polynomials computed row-blocked (~2.7 GB peak, not ~4 GB); `fit_sweep` selects on **our val** and records
the whole grid, since no published ML-20M/25M number exists to snap to. `scripts/queue_turbocf.ps1` fires
the run when RAM frees. **Lesson: a smoke test on toy data cannot catch a scale-dependent numerical
degeneracy — the guard has to be a property check on the real matrix.**

## ★ THE POLISH IS DROPPED (author ruling, 2026-07-29) — greedy is the right object
The endpoint polish optimises q=16 and **nothing else, and nobody set the budget at 16**. On items-only it
buys **+0.002 at q16** and pays **−0.029 at q2**, −0.012 at q4, −0.004 at q8; its curve is not even
monotone (q2 0.1234 < q1 0.1310). For a cold-start interview, where the budget is a deployment choice and
short interviews are the realistic case, that is a bad trade in every column that matters.

**Greedy is prefix-optimal — the best its bank can do at every budget at once — which is exactly the
property an interview needs.** The paper now says so, and the old "read the combined−items gap only at
q≤4 and on the tail" reading rule (what the harsh review called a moved goalpost) is GONE, replaced by:
containment binds at the *optimum*, not along the *path*, so q16 is the one column where a bank can be
behind. No goalpost, no apology, and the section no longer waits on a run.

Polish numbers for the record (not in the paper): items-only 0.2098→0.2118 (1146 evals); concepts-only
**unchanged** (640 evals, nothing worth swapping); combined killed part-way at 0.2024→0.2067 on build —
it was indeed unwinding toward items-only, consistent with the myopia account but not needed.

## Paper A: what is settled
- **G0 CERTIFIED**: T2′ TEST **0.3482 / 0.2462** (@100 0.4486); tie vs RecVAE 0.3540/0.2497
  (diff −0.0057, CI 0.0069); empty-interview identity EXACT (|z|=0, Spearman 1.0000).
- **C1 bridge**: EASE 0.4203/0.420 PASS · RecVAE 0.4425/0.442 PASS · Mult-VAE 0.4223/0.426 near-snap ·
  **Mult-DAE 0.3969/0.419 FAIL → DROPPED** · iALS 0.358 vs WMF 0.386 → **uncertified floor**, and the
  "below even iALS" comparison is WITHDRAWN.
- **C2 = satisfied BY CONSTRUCTION** (no experiment): item answers are the user's own ratings; concept
  answers are log watch-lift `log2((n_c+.5)/(e_c+.5))` over raw counts and item marginals. No
  recommender-derived quantity enters an answer. Contrast is with PUBLISHED practice (PEBOL's simulator
  answers from the ground-truth description; UNICORN assumes users hold clear preferences over everything)
  — NOT with our own earlier drafts.
- **G1c DROPPED**: a popularity lookup predicts held-out NLL ~3× better than the belief σ (|ρ| 0.70 vs
  0.24). The gate measured target difficulty, not covariance quality. `docs/results/G1C_CONTROL_RESULT.md`.
- **G4 = refusals are inert, architectural**; the confidence-ordering arc is dropped (MovieLens has no
  within-channel confidence signal).
- **Graded curve on the CERTIFIED tower**: +0.0147 @k2 [+.0127,+.0167], +0.0083 @k4, ties from k8.
- **Greedy best-static** (fixed fold, build 10k val / eval 10k TEST): items .1405/.0275→.2098/.0679;
  concepts .1436/.0323→.1808/.0758; combined .1436/.0323→.2049/.0776. Order `cccciiiiciiiiiii`.
  **Answerability: items 2.18 of 8 answered, concepts 6.02 of 8 — 2.8×.**
- **Open-vocabulary illustration**: held-out concepts predicted from name alone recover the top of their
  own ranking >300× chance; four screened free-text phrases with their nearest concept shown
  (`docs/results/SBERT_OPEN_CONCEPTS_RESULT.md`). Framed as an enabled research direction, not a claim.

## Literature verified this round (primary text)
- **Göpfert 2022 / Bıyık 2023 BOTH FREEZE the CF model** — our "frozen vs co-trained" wedge was FALSE and
  is removed. CAVs are per-concept supervised probes over a CLOSED 164-tag vocabulary, no text-encoder path
  for unseen attributes. Differentiator SURVIVES. `external_literature/findings/text_to_cf_projection.md`.
- **Balog SIGIR 2021 §5.2** is the real label-free precedent (BM25 phrase → centroid of retrieved items'
  embeddings). Cite; do not claim label-free attribute directions as new.
- **Sun et al. WSDM 2013** — entire contribution is an answerability fix (unknown branch >80% of users per
  split). Engaged head-on. **Gharahighehi 2025 (arXiv:2510.27342)** and **Xia 2026 (COPE)** both report our
  concept-early/item-late crossover; neither measures answer rate. Our contribution is the MECHANISM.
  `external_literature/findings/answerability_in_elicitation.md`.
- Rank-targeted acquisition (Paper B): our criterion = greedy **transductive V-optimal / A-GOODE**; cite
  Yu-Bi-Tresp ICML 2006 + Attia 2018. Best alternative = **Knowledge Gradient** (Frazier 2009).
  `external_literature/findings/rank_targeted_acquisition.md`.

## ★★★ OPEN, TOP OF QUEUE (#66): sign-aware evaluation — scope R5, then optionally probe
**Adjudicated twice by Fable + one deep-research pass, evening of 2026-07-29. NOT a blocker.**
**★ FULL WRITE-UP → `docs/results/PROTOCOL_DISLIKE_DISCARD.md`** (why the Liang protocol is canonical, the
whole argument, both verdicts, the rejected fix, the 4-arm design, open items). Citations →
`external_literature/findings/sign_aware_evaluation.md`. Plan → task #66.

**The question.** The Liang protocol discards every sub-3.5 interaction (55.7% of a test user's rated
history). Our tower is sign-aware; item-indicator models cannot represent a dislike at all. So can a parity
claim rest on a ruler that forbids the capability?

**The verdict.** The protocol is **silent** on sign, not structurally incapable of crediting it. The
inflation objection is right but scoped only to *observed* dislikes — demoting them is a free lift, since
targets ⊂ likes makes them guaranteed non-targets. **Comprehension is different and is measurable here:** a
model that understands a dislike re-ranks *unseen* items, judged by held-out likes it never saw. Do **not**
change the primary protocol: masking all rated items would break the C1 bridge, hand every model an oracle
the models were denied, and delete the advantage it means to credit. Never write "rubbish" about our own
bridge.

**Owed (writing only).** Scope R5's evidence to the scarce-evidence regime — intensity-within-likes is
**−0.0002** at full profile, the premium is **+0.0147 at k=2** and gone by k=8. Frame as a finding: *sign
carries information when evidence is scarce and washes out once 60+ likes are on the table*, which is the
argument for an interview instrument. The limitation paragraph is already in, with the intensity/sign
distinction corrected; the chapter-wide R5 scoping pass is not done.

**Optional 4-arm probe** (A likes-only / B +trivial dislike filter / **B′ dislikes as unsigned positives** /
C dislikes signed natively). The estimand is **gap(C,B′)**, not gap(C,B) — without B′ the comparison credits
sign-awareness for merely having 55.7% more data. That was the confound I missed. Mask fixed at `te_tr`, and
the `-inf` exclusion of observed dislikes applied in B, B′ *and* C.

**★ The metric already exists — cite, do not christen.** Frolov & Oseledets RecSys 2016 **nDCL**
(arXiv:1607.04228) uses a negativity threshold of **3.5, the same boundary as Liang**, admits irrelevant
items to the holdout, and even ran a **negative-only cold-start** arm. Sánchez & Bellogín RecSys 2018
**anti-relevance**, extended in Mena-Maldonado TOIS 2021. *We already had Frolov flagged in our own 24-Jul
notes (`graded_inputs_for_ranking.md` line 67) and missed it — HARD RULE 4 exists for this.*

**Constraints:** avoidance must be a separate lower-is-better number, never negative gains inside NDCG
(Gienapp CIKM 2020 — implementations clamp to 0); report it popularity-stratified (Mena-Maldonado — observed
dislikes skew popular); full-rank probe, no sampled negatives (Krichene & Rendle 2020).

**What is ours (verified from full text):** the sign-aware *model* line evaluates positives-only — SIGformer
(SIGIR 2024) and SiReN report Recall/NDCG only, and *no metric measures whether disliked items rank low*.
Also ours: the conjunction with the Liang protocol, the anti-gaming rationale, frozen-tower purpose, and —
searched, not found — no protocol measures the marginal value of a *"no"* answer separately from a *"yes"*.

## ★ VOID: the graded-baseline reruns (2026-07-29 evening)
Both regimes were artifacts of one coupling resolved in opposite directions; `metrics.evaluate` uses the
fold-in matrix as **both** model input and `-inf` mask. Regime 1 over-masked the graded arm (Golbandi
"0.3894", above RecVAE — a candidate-pool result, not a model result); regime 2 under-masked it (RBMF 0.1377,
*at* the Most-Popular floor). Seven variables moved at once. **The canonical rows are simply correct**
(Golbandi 0.3065, RBMF 0.2534, TaNP 0.2645) and numbers from different information contracts are not
ordered. Nothing from this goes in the paper. → memory `surprise-means-debug-the-harness`.

## Closed WITHOUT running (author ruling, 2026-07-29) — do not reopen
- **#30 Wave-2 rest** (GF-CF, BSPM, BERT4Rec, EDDI): not implemented, and none can snap to published
  numbers on our ruler (no Gowalla/Yelp/Amazon data, different metrics). They stay dashes in `tab:master`
  with a stated disposition. Turbo-CF and SASRec/TaNP are the exceptions — those runs exist.
- **#36 G10**: never run and unreferenced anywhere in the text. Dropped, not run.
- **#34 certified-tower interview re-runs**: not needed. Both towers freeze the same RecVAE decoder and
  the graded curve replicated with the same shape.
- **#43 full PEBOL-style open-vocabulary interview**: the illustration is in and framed as an enabled
  direction; the full experiment is Paper C territory.
- **#46 the polish**: see the ruling above.

## Still open
- **PAPER B BLOCK (#56)** — belief-driven selection; do not start until A is closed. Σ-greedy is provably
  static-equivalent (Λ's update ignores answers); escapes are answerability and NDCG-coupling.
- **SASRec / TaNP** — running until ~14:00. SASRec's early dip was a training dip, not a ceiling:
  val 0.1174 → 0.1158 → 0.1013 → 0.1151 → 0.1313 → **0.1376** (ep6) → 0.1362. It is now above
  Most-Popular (0.1345), so the earlier concern about withholding the row does not apply. TaNP follows.
- **The polish sentence** — author deferred: we hold the numbers that pre-empt "you only tried a myopic
  optimiser" (+0.002 at q16 for −0.029 at q2) and are keeping them out until all results are in.
  Revisit before submission; it is evidence FOR our criterion, so A0 argues for it, not against.

## ECIR 2027 submission (added 2026-09-08)
- Draft v2 at `new_chapters/paperB_ecir/` (12 body pages, LNCS). Review panel + summary in `reviews/`, `REVIEW_SUMMARY.md`;
  ordered to-do in `TODO.md`. Abstract due **21 Sep 2026**, paper **5 Oct 2026**.
- WARNING: the paper's code footnote points at https://github.com/makarovaalexa-brch/casper . Before submission replace it
  with an anonymised mirror (double-blind). Before publication clean the repository: remove `.bak_*`, stray `.out/.err`
  logs and untracked experiment dumps, add a README that maps the paper's tables to scripts and artifacts.
