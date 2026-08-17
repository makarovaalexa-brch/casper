# Design sheet — M1 / M1b: the two opener families (DREAM and REALISTIC)

**Author go: 2026-08-17, "plz proceed according to plan".** Governing plan: `PAPER_B_POLICY_DESIGN.md`
Part M + M.1–M.4. Nothing here touches TEST. Validation/train only.

## The question each stage answers

- **M1b (REALISTIC family):** what do people *actually* open with, at what frequencies, and which of
  those opener types can we form measurable ML cohorts for?
- **M1 (DREAM family):** if we could choose the opener freely, which partition of users by taste
  separates them best — and what observable descriptor names each partition?
- Both produce the same object: **a list of (opener, cohort) pairs** for the M2 screen, tagged by
  family. M2 then asks whether any of them beats the tree's own first question.

## Fixed contract (both stages)

Certified pair `t2i26c_ep31.pt` + `concept_cv3_ep08.pt`; policy-B mask/pool; canonical m=0 snap
`0.15686538118354684`; seed 98765; NDCG@10 full AND tail; paired percentile bootstrap 10,000
resamples. Volunteered-evidence draw follows R exactly: popularity-weighted from the user's own
fold-in half, **k=8** (the strongest evidence condition; k=2/4 are secondary).

---

## M1b — REALISTIC openers, mined from the Reddit slot corpus

**Input, already on disk, nothing new processed:** `data/reddit_he/slots/full_out_*.jsonl`, 4,431
records (one per ruler thread), fields `anchors_pos`, `anchors_neg`, `genres_pos`, `genres_neg`,
`era`, `tone`, `constraints`, `context`.

**Step 1 — the census (pure groupby, no model).** Report per slot type: number and share of threads
that use it, co-occurrence matrix, and the value distribution within each type (which genres, which
tones, which eras). Report the **opener** specifically as the highest-priority slot present, under a
fixed priority order declared here: `anchors_pos > genres_pos > tone > era > constraints`. Also
report the marginal ("how many threads carry each slot at all"), because priority ordering is a
modelling choice and the marginal is not.

**Step 2 — mapping to ML attributes.** Each opener VALUE maps to an ML cohort signature or it does
not:
- `genres_pos` → MovieLens genre (direct, 18,235/18,359 items carry ≥1 genre).
- `tone` → genome concept from the 1,031-concept bank, by exact/alias match on the tag vocabulary.
- `era` → release-year band from `movies.csv`.
- `anchors_pos` → the anchor film's own genre/concept neighbourhood (a named film is too granular
  for a cohort; it enters as *evidence*, not as a signature).
- **Unmappable values are COUNTED AND REPORTED, never dropped silently** (HARD RULE 1). The
  unmapped share is a headline number of this stage: it prices what our attribute vocabulary cannot
  express about real requests.

**Step 3 — cohorts.** For each mapped opener value, the ML cohort is the validation users whose
**fold-in** satisfies the signature (liked-genre majority for genres, ≥k rated members of the concept
for tones, year-band majority for eras — the same behavioural rules already used by
`group_greedy_genre.py`). **Gate: cohort ≥ 1,200 validation users** (Fable's floor; 500 sits in the
−0.724 noise regime). Cohorts below it are reported with their size and excluded from M2.

**Deliverable:** `docs/results/M1B_OPENER_CENSUS.md` + `experiments/battery/m1b_census.json` —
frequencies, the mapping table with its unmapped share, and the surviving (opener, cohort) list.

**Cost:** hours; CPU-light; no contention with anything.

---

## M1 — DREAM openers, from taste clustering

**Step 1 — the representation.** For 10,000 validation users (and TRAIN users where teachers will
later be fit), draw k=8 volunteered items and fold them to get `z_evidence`. Cluster **that** space —
not the full-profile space — because the signature must be computable from what the user has
volunteered at ask time.

**Step 2 — clustering.** k-means, **8 clusters** (10,000/8 = 1,250 mean, at Fable's floor of ~1,200),
seed 98765, on standardised `z_evidence`. Report cluster sizes; any cluster under 1,200 is reported
and excluded from M2 rather than merged.

**Step 3 — NAMING (this is the stage's real content).** For each cluster, find the observable
descriptor that best separates it from the rest of the cohort. Candidate descriptors: 20 MovieLens
genres, the 1,031 genome concepts, release-year bands. Separation metric: **AUC of the descriptor's
per-user affinity score in predicting cluster membership**, with the descriptor's own base rate
reported alongside. Declared in advance:
- **A cluster is NAMED if its best descriptor reaches AUC ≥ 0.70**; below that it is reported as
  **unnameable** and excluded from M2 — it cannot be routed to, because no question a user can answer
  identifies it. *This is the direct test of Fable's objection that clusters track variance rather
  than decision-relevance, and it is allowed to fail.*
- The naming is reported in full: top-5 descriptors per cluster with AUCs, so "drama is useless but
  dark is useful" is visible as data rather than asserted.

**Control:** the same naming procedure run against **shuffled** cluster labels must produce no
descriptor at AUC ≥ 0.70. If it does, the metric is broken, not the clusters.

**Deliverable:** `docs/results/M1_TASTE_CLUSTERS.md` + `experiments/battery/m1_clusters.json` —
sizes, top descriptors with AUCs, named/unnameable verdicts, the shuffle control, and the surviving
(opener, cohort) list.

**Cost:** ≤ 1 day. The fold and draw harness exist (`volunteered_cohorts.py`, R's draw).

---

## What M2 receives

One combined list of (opener, cohort, family) with every cohort ≥ 1,200 validation users and every
opener stated as a question a user could answer. M2 screens all of them under the same +0.004 bar,
and the DREAM-vs-REALISTIC contrast — how much separation and screen gain natural openers lose
against statistically optimal ones — is read off that single table.

## Pre-registered failure modes (declared before running)

- **M1b:** if the unmapped share is large, the chapter reports it as the measured price of our
  attribute vocabulary and the realistic family runs on what maps. It is not a reason to widen the
  vocabulary mid-stage.
- **M1:** if no cluster reaches AUC 0.70, the dream family is empty and M2 runs on the realistic
  family alone — which is the deployable one anyway.
- **Both empty ⇒ M2 cannot run**, and the fork is decided without it: the chapter reverts to audit +
  tree + mechanism record, and nothing is marketed in its place.
