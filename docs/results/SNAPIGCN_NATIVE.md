# SNAP-IGCN: Rashid'08 IGCN — native snap PASSES, and the published form is a static order at k≤4

Run 2026-08-08 (task #19). Target: Rashid, Karypis & Riedl, *"Learning Preferences of New Users in
Recommender Systems: An Information Theoretic Approach"*, **SIGKDD Explorations 10(2) 2008, pp.
90–100**, DOI 10.1145/1540276.1540302 — the **IGCN** algorithm.
Code: `src/baselines/igcn_selector.py`, `scripts/baselines/igcn_ml100k_snap.py`,
`scripts/baselines/igcn_selftest.py`. Artifacts: `experiments/results/snapIGCN/`.

**Nothing is committed.** Overseer reviews first.

## Why this row was missing

`interview_strategies.CITES` already credits this paper for `entropy0` and `helf`, and that credit is
correct — the paper's own words are *"All of the heuristics except the Igcn can be computed prior to
the simulation"*, so a static precomputed order **is** the faithful form of those two. But IGCN, the
paper's own headline method and the **one adaptive member of the family**, had never been
implemented; a repo-wide grep found prose references only. The 2025 attribute-tree literature still
benchmarks against it (`findings/adaptive_elicitation_baselines.md` §1.5).

## STEP 0 — the published target [V]

Source of record: the repo's own full-text pass,
`external_literature/findings/rating_elicitation_family_deep.md` §2.3. **The publisher PDF is CLOSED
everywhere reachable** — Semantic Scholar reports `openAccessPdf: {"url": "", "status": "CLOSED"}`
for the DOI; kdd.org mirrors 404, ACM 403, `karypis.github.io` 404, `papers.karypis.org` 404. This is
the same situation SNAP-5 recorded for Golbandi (HTTP 410), and it is why the stored extraction, not
a re-fetch, is cited. **A live re-verification of the PDF was attempted and failed; the [V] marking
is inherited from the earlier full-text pass, not re-earned today.**

* **Criterion.** `IG(a) = H(C) − Σ_r (|C_a^r|/|C|) H(C_a^r)` over **user-cluster labels** C, with the
  missing-value-weighted variant `IG(a;W) = H(C) − Σ_r [w_r |C_a^r| H(C_a^r)] / E(C;W)`. Asking about
  an item is worth what it tells you about *which kind of user* you are talking to.
* **Clusters:** bisecting k-means, **c = 300**.
* **Adaptivity, verbatim:** *"One issue with information theoretic measures such as entropy and its
  variants discussed so far is that they are **not adaptive to a user's rating history**."*
  Algorithm 3.1 is two-phase — a non-personalized step on global IG until the user has rated ≥ *i*
  items, then a personalized step that finds the best *l* neighbours and **recomputes IG from those
  l users' ratings alone**, repeating as the neighbour set moves.
* **Parameters (c, n, i, l) = (300 clusters, 15 items/screen, switch after 5 ratings, 150
  neighbours).**
* **Contract:** MovieLens ~11,000 u / 9,000 i / ~3M r; budgets 15/30/45/60/75 (pages of 15); metrics
  **#seen, MAE, Expected Utility**; plus a 20-day online study, 381 valid subjects.
* **Reported numbers** — online MAE with their user-kNN: **IGCN 0.67 (best), Popularity 0.73,
  Entropy0 0.73, HELF 0.84 (worst)**; online pages-to-finish Entropy0 3.0 / Popularity 3.6 / HELF
  4.6 / IGCN 5.5. Verbatim: *"Overall, the Popularity measure performed the worst, and the Igcn
  measure performed the best."*

## STEP 1 — native snap: **PASS** (ordering snap, ML-100k)

**A numeric snap is impossible and was not attempted.** Their corpus is the live MovieLens site of
2008 — not a released dataset — and the headline numbers come from **human subjects**, so they are
irreproducible by construction. Per the standing protocol for a paper that reports only a study, the
**ORDERING** is snapped and labelled as such.

Setup: ML-100k, 200 held-out cold users × 3 seeds, each cold user's ratings split **70/30 into a
fold-in side (the only source of answers) and a disjoint eval side** — no answer can come from an
item the MAE is scored on. Budgets are the paper's own. An asked-but-unrated item **burns** the
question. Consumer = user-kNN, the paper's own recommender family.

**Parameter rescaling, reported not hidden.** (c, l) = (300, 150) is sized for ~11,000 users; on 743
training users that is 2.5 users per cluster and a neighbourhood of 20% of the corpus. The **primary**
row keeps the paper's *ratios* → **c=20, l=10**, exactly as SNAP-5 rescaled Golbandi's ξ. The literal
(300, 150) is run too, because "the constants do not transfer" is a finding.

**MAE / #seen, mean over 3 seeds. Lower MAE better.**

| arm | k=15 | k=30 | k=45 | k=60 | k=75 |
|---|---|---|---|---|---|
| **igcn (rescaled c=20,l=10)** | **.8770**/3.0 | **.8647**/5.6 | .8537/8.1 | .8478/10.5 | .8396/12.9 |
| igcn (literal c=300,l=150) | .8819/4.2 | .8678/7.8 | **.8536**/11.4 | **.8476**/14.9 | **.8359**/18.2 |
| entropy₀ | .8806/4.9 | .8700/8.7 | .8622/11.9 | .8606/14.8 | .8528/17.6 |
| HELF | .8830/3.9 | .8720/7.2 | .8663/9.8 | .8559/12.5 | .8525/14.8 |
| popularity | .8815/5.0 | .8716/8.7 | .8635/12.0 | .8580/14.9 | .8536/17.6 |
| random | .8666/0.8 | .8668/1.4 | .8633/2.1 | .8598/2.8 | .8544/3.5 |
| **`none` (control, 0 questions)** | .8350/0.0 | .8350/0.0 | .8350/0.0 | .8350/0.0 | .8350/0.0 |

**Gates, declared before running: PASS on both.**
* **G1 — IGCN beats HELF at every budget:** yes. Paired over users (10,000 resamples): −.0211, −.0162,
  −.0164, −.0092, −.0113 at k=15…75; **separated at 4 of 5** (k=60 is ns, [−.0208,+.0022]).
* **G2 — IGCN best of the four at every budget:** yes. Against entropy₀ **−.0163 to −.0195, separated
  at every budget**; against popularity **−.0168 to −.0210, separated at every budget**.
* Against `random`: ns at k=15, then separated negative from k=30 on.

**Where the published ordering reproduces and where it does not — state both.** "IGCN best"
reproduces at **every** budget, separated. **"HELF worst" reproduces only at k ≤ 45**; at k=60 and
k=75 HELF is the *best* of the three statics and the IGCN−HELF gap loses separation at k=60. The
paper's own Popularity≈Entropy₀ tie reproduces (they sit within .001–.003 of each other throughout).

## Two controls, and the first one caught a real bug

1. **`none` (ask nothing).** The first version of the consumer centred each cold user on **the mean of
   their own answers**. Every elicitation arm asks popular, well-liked items, so that mean is biased
   upward *by an amount that depends on the arm* — and the control caught it: `none` beat **every**
   elicitation arm at every budget, i.e. the arms were being ranked on how little they corrupted the
   centring, and `random` got monotonically **worse** with more questions. Replacing it with the
   textbook **baseline-predictor kNN** (`r̂ = μ + b_u + b_i + kNN(residual)`, b_u estimated against the
   *item* baselines, β_i=25 / β_u=10, applied identically to every arm and to the control) removes the
   item-selection bias by construction. Every arm then became monotone in budget and the ordering
   above appeared. **This was a consumer repair, not arm tuning** — no per-arm constant exists.
2. **Full-profile sanity.** `none` still beats every arm at k ≤ 75, so a second control was needed to
   separate "the interview is too short for this consumer" from "this consumer ignores information".
   Fed the cold user's **whole** fold-in profile (~73 ratings) the same kNN reaches **MAE .7778/.7694/
   .7722** vs the .8350 floor. The consumer responds to information; ~13–18 elicited ratings on a
   743-user training set is simply below its crossover. **Report the snap as an ORDERING result only** —
   the absolute MAEs sit above the no-interview floor and must never be quoted as elicitation gains.

## The implementation is checked against a naive reference, not just against itself

`scripts/baselines/igcn_selftest.py` (PASSES) proves the three things that could be silently wrong:
the vectorised IG equals a slow textbook implementation to **8.9e-16 (global)** and **1.1e-15
(personalised, 5 neighbour sets)**; the **unrated branch derived by subtraction** matches the
directly-gathered one; and the claim that lets the sweep be affordable — *an item no neighbour rated
has IG identically 0* — is asserted against the naive scores rather than assumed. Bisecting k-means
is verified deterministic and a true partition.

## STEP 2 — the port: built, smoked, **not launched** (box busy)

Wired into `adaptive_harness.py` as three specs. Harness `--regression` **PASSES: 30 (arm,budget)
cells byte-identical**, so the existing arms are untouched.

| spec | what | why it is a separate row |
|---|---|---|
| `igcn` | the paper's parameters (c=300, l=150, i=5), re-rank every turn | the published method |
| `igcn_i1` | **labelled deviation**: switch after **1** answer | see below — the published form is static at our budgets |
| `igcn_page15` | paper-**literal** re-rank period (one page of n=15) | at k≤16 it can fire at most once |

★ **The deviation is forced by our budget and must be stated in any table.** The paper switches to the
personalized step after **5 ratings**, and our answer rate is ~0.3/question. Measured on a 300-user
smoke (**SMOKE, NOT CITABLE**):

| spec | k | distinct ask-prefixes | **personalised ask rate** |
|---|--:|--:|--:|
| `igcn` | 8 | 13 / 300 | **1.9%** (46/2400) |
| `igcn` | 16 | 101 / 300 | **12.9%** (621/4800) |
| `igcn_i1` | 8 | 147 / 300 | **48.5%** (1164/2400) |
| `igcn_i1` | 16 | 236 / 300 | **63.3%** (3039/4800) |
| `entropy0-static` | 8 | **1** / 300 | — |

So at **k ≤ 4 the published `igcn` row IS the global-IG static order, exactly and by construction**,
and even at k=8 it is 98% static. `igcn_personalised_rate` now lands in **every cell** so this is
readable off the table rather than inferred. `igcn_i1` is the row that exercises the personalized step
at our budgets; it is a **different row, never a repair of the first**.

### Two harness fixes made on the way

* **`fixedorder:<name>` selector spec (new).** The greedy items-only comparator can now be walked
  *inside the same run*, so its per-user vectors land in the same npz under the same contract and the
  paired difference is exact **by construction** — rather than re-scored in a separate file and
  admitted after a 0.0e+00 check, as `gtport_greedy_rescore.py` had to. Verified: `fixedorder:entropy0`
  vs the independent `entropy0-static` matches **per user at 0.0e+00** on both axes.
* **Pre-existing latent bug fixed.** Interview checkpoint filenames were built from the raw spec
  name, and `:` is illegal in a Windows path — so every `treestatic:` / `fixedorder:` arm crashed with
  `WinError 87` at its *first* checkpoint write. The persistence rule was silently a crash for exactly
  the comparator arms. The spec still goes in the checkpoint *contract*, so sanitising the filename
  cannot make two specs share a sidecar.

### Not launched, and why

**HARD RULE 11(d).** At every check the box held a live heavy twin (`run_ials_conf_decompose.py`, then
`ials_ladder_paired.py`, 1.3 GB RSS) with **2.7–4.0 GB free**. Build and smoke only, per brief.
IGCN asks **items only** — it never touches the concept action space, so `--concept_neg_cap` is
irrelevant to this arm and no concept convention applies to any of its cells.

**Launch command (single run: IGCN + both comparators, one contract, one per-user npz):**

```
python src/baselines/adaptive_harness.py \
  --selectors igcn,igcn_i1,igcn_page15,entropy0-static,fixedorder:greedy_items \
  --budgets 1,2,4,8,16 --only ours --arch i26 \
  --snapshot .cache/instrument/t2i26c_ep31.pt --no_floor --per_user \
  --igcn_cluster_cache .cache/baselines/igcn_clusters_ml25m_c300.npz \
  --label igcn_port01
```

Clusters are already banked at `.cache/baselines/igcn_clusters_ml25m_c300.npz` (300 clusters over
140,768 train users, contract-signed), so the run starts on the interview immediately. Expect well
under an hour on the smoke's scaling (cost is per distinct interview *state*, not per user). Every
cell writes on completion and every interview checkpoints per turn, so a crash costs one turn.

## Owed before this row is cited

* **The port cells do not exist yet.** Everything above the STEP-2 heading is the native snap; the
  smoke numbers are stamped `SMOKE_NOT_CITABLE` and are 300 users.
* **Every IGCN cell must be published with its `igcn_personalised_rate`.** A row at rate 0 is
  `entropy0`'s cousin, not an adaptive method, and the k≤4 column is exactly that.
* The snap is an **ordering** snap on **ML-100k** with **MAE**, against a paper whose numbers come
  from a **user study**. It licenses "our IGCN reproduces the published ordering", never a number.
* "HELF worst" **inverts at k ≥ 60** in our setup; do not quote the paper's ranking as if it held
  throughout.
