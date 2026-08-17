# Design sheet — M1 (revised): STATE-SPACE opener clustering and ML twin matching

**Supersedes the "descriptor naming" half of `DS_M1_OPENER_FAMILIES.md`** (the AUC ≥ 0.70 naming gate
is withdrawn). M1b steps 1–2 stand and are banked (`docs/results/M1B_OPENER_CENSUS.md`).
Author's design, 2026-08-17: *"fold in those openers, get the state and cluster by state … not from a
question, but from actual embedding"*, with matching by **near-exact twins**, not sampling.

## The idea, and why it is better than what it replaces

A real request is a **bundle** — 93.6% of threads carry ≥2 slots. Bundles cannot be routed by a single
stated descriptor, and naming a cluster by a descriptor was an extra gate that could fail for reasons
unrelated to the science. Instead: **fold whatever the user volunteered into the recommender, and let
the resulting STATE be the signature.** Routing is then "which centroid is your folded state nearest",
computed at ask time from evidence alone, with no naming step and no stated-word cohorts.

The chapter argument this supports: **real users arrive in a continuum of folded states, not at the 22
discrete nodes a ternary tree can reach — so the system must learn to ask from a state, not follow a
fixed trajectory.**

## Facts established before this sheet (all measured, none assumed)

- 4,431 ruler threads; **2,639 (59.6%) name ≥1 film**, mean 2.87 films, median 2.
- **100.0% of the 7,572 OP-named films are inside our 18,359-item training vocabulary** — every
  foldable request is fully expressible in our action space. No new linking is required.
- Genre values map to MovieLens genres at 95.1%; tone → genome tag 66.1%; era → decade 70.2%.
- Bundle composition: 35.9% films+signature · 27.1% films only · 19.9% signature only · 17.0% neither.

## ★ THREE OBJECTS, ONE SPACE (correction, 2026-08-17 — author: *"you start from reddit clusters, not ideal ml clusters?"*)

An earlier draft of this sheet clustered Reddit states only, which silently dropped the DREAM family
and the dream-vs-realistic comparison that is the point of having two. All three objects are built,
in the same latent space:

1. **IDEAL clusters** — ML users clustered by their **full fold-in state** `z_full`. The best
   partition of taste: what you would route on if you could see everything. **Upper bound.**

   ★ **CORRECTION (author, 2026-08-17): `z_full` DEFINES THE PARTITION, IT IS NEVER AN INTERVIEW
   STATE.** A fully folded profile is *satiated* — the user sits near the full-profile ceiling
   (~0.486), nothing is left to ask, and a teacher grown at that state measures nothing. The author's
   framing: *"the ideal group has satiated stated at the point when full profile is folded — nothing
   to ask. but full profile shows their tastes."* So the ideal family's interview starts from **one
   perfect volunteered fact that places the user in their cluster** (Stage A2), which makes it
   comparable to the realistic family: both are cold + a little evidence (dream = 1 optimal fact,
   realistic = the actual bundle, ~2.9 films). **The gap between them is the price of natural
   openers.**
2. **REALISTIC clusters** — the 2,639 folded Reddit **request** states `z_r`. Where real users
   actually arrive, having said one sentence. **Deployable.**
3. **THE ALIGNMENT** between them — how well an opener-state places a person relative to where their
   full taste sits. This IS the dream-vs-realistic loss, measured rather than asserted.

Teachers are grown on BOTH clusterings, and the gap between their M2 screen gains prices what natural
openers cost against statistically optimal ones. Scope condition to state plainly: full-taste
clustering partitions people by *who they are* and is **not observable at ask time**, so an ideal
cluster is a ceiling, not a deployable route, unless the alignment cross-tab shows something
observable reaches it.

## Stage A — fold BOTH populations into states

Both populations are folded through the **same operator from the same starting point** (cold, no
profile), which is what makes them comparable: an ML user in the cold-start interview also starts cold
and receives volunteered evidence. Entry point `fold_items_frozen(ctx.enc, [(ids, levels)])` — it takes
explicit ids and levels, so no user row is needed.

Declared modelling choices (each is a choice, not a measurement, and each is reported):
- **Anchor films enter at a LIKE level.** `anchors_pos` asserts the poster likes them; the level used
  is the fixed "liked" level already used by the interview harness, applied uniformly. `anchors_neg`
  enters at the dislike level.
- **Mapped genre/tone concepts enter as asserted-positive** (value +1, refusal 0). A stated preference
  is a maximally confident answer; there is no per-user `V` row for a Reddit poster.
- Requests with neither a linked film nor a mapped concept are **not foldable** and are counted, not
  imputed.

Output: `z_r` for every foldable request + the count and composition of the unfoldable remainder.

## Stage B — cluster the request states

k-means on `z_r`, seed 98765. **k chosen by the cohort floor, not by an elbow**: the binding constraint
is that each cluster must eventually carry ≥1,200 matched ML users, so k is swept over {4,6,8,10,12} and
reported with per-cluster request counts; the operating k is the largest whose clusters all survive
Stage D. Shuffle control: cluster labels permuted must destroy any structure reported downstream.

## Stage C — twin matching (the part the author corrected)

**Not sampling.** For each foldable request R with anchors `a_1..a_m` and mapped concepts `C`:

1. **Concepts match exactly** — shared vocabulary. A candidate user must behaviourally endorse each
   concept in `C` under the existing SEL rule (`E["foldable"]`, `E["V"]`). No substitution.
2. **Films match by per-anchor substitution.** For each `a_i` take its neighbourhood `N(a_i)` = the
   top-`nn` items by tower item-space similarity. A candidate user must have a **liked fold-in item in
   every** `N(a_i)`; their evidence set takes, for each `a_i`, their liked item closest to it. The twin
   therefore volunteers **their own films**, one standing in for each of the request's, preserving the
   bundle's size and structure. Every volunteered item is one they actually rated — the behavioural
   firewall is untouched, and nothing consults held-out targets.
3. **Verify in state space.** Fold the twin's constructed evidence, keep the pair only if
   `‖z_u − z_r‖ ≤ ε`. The substitution proposes; the state distance disposes.

Implementation is retrieval, not search: an inverted index item→users-who-liked-it, intersected over
the anchors' neighbourhoods, then ≤ m greedy picks per candidate from a shortlist of ~m·nn items.

**Two knobs, reported as a curve rather than a chosen point:** neighbourhood size `nn` and threshold
`ε`. Deliverable includes twins-per-request as a function of both.

## Stage D — selection, and the constraint that drives it

**Every ML user appears at most once.** The inference protocol is a paired bootstrap over per-user
difference vectors; a user appearing as a twin for three requests would contribute three rows, inflate
n, and corrupt every CI.

- **Fixed quota T per request** (T ≈ 4 fills ~10,000 users from 2,639 requests). Equal quota ⇒ equal
  weight per request, so the benchmark is weighted by *observed request frequency* rather than by how
  many twins a request happens to attract.
- Requests processed in seeded random order, each taking its closest unused twins.
- **Unfilled quotas are the coverage measurement**, reported per cluster and per genre. Expected to
  concentrate in Horror (27.5% of genre openers, 51 of 10,000 ML users by fold-in genre).

Teachers do **not** need twinning — they need a cohort at a region, so cluster membership suffices.
Twinning is required only for the final benchmark population.

## Stage E — the two cross-tabs (free outputs, both are findings)

1. **Opener-state vs full-taste-state.** Cluster ML users' *full* fold-in states separately; cross-tab
   against their opener-state cluster. Anything other than a diagonal says openers systematically
   mislocate people relative to their full taste — which is precisely what an adaptive interview
   exists to correct.
2. **Reddit state distribution vs ML reachable state distribution.** Where our corpus cannot follow
   real requests, named per region.

## Gates, controls, failure modes (pre-registered)

- Canonical snap `0.15686538118354684` reproduced before any number is trusted; leak assert (no
  answered item is a held-out target) is already inside `genv_eval_multi` and must fire clean.
- Shuffle control at Stage B; twin-matching control: matching against **permuted requests** must give
  materially worse state distances than true matching, else the matcher is not doing anything.
- **If < 4 clusters carry ≥1,200 twins**, the benchmark population is too thin: report coverage as the
  finding and fall back to cluster-level cohorts without request weighting.
- **If twins are dense at any ε** (i.e. matching is trivially easy), report it — that would say ML
  users already span the request space, which strengthens the transplant but weakens "the tree cannot
  reach these states".

## What M2 receives

Cluster centroids with their matched ML cohorts (≥1,200 each). The screen is then run **from the
centroid state**, not from a stated descriptor: does any question beat the tree's own first question
by ≥ +0.004 at that state, paired, identical fold in both arms.

## Cost

Stage A hours · Stage B minutes · Stage C the only real cost (inverted index + per-request
intersection over 2,639 requests) — bench one request before estimating · Stages D/E minutes.
Validation users only. No TEST. No LLM. No new Reddit processing.
