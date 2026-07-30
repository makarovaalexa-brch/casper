# The dislike-discard problem: why our ruler cannot credit sign, and what follows

**Opened by the author, 2026-07-29.** *"We just simply discard all info on dislikes. The rules are not fair.
We cannot claim a SOTA model if we just throw away perfectly valid signal."*
Adjudicated twice by Fable plus one deep-research pass. **Verdict: a limitation to scope, not a blocker.**
Companion files: `external_literature/findings/sign_aware_evaluation.md` (citations),
`BASELINE_INPUT_REGIME_AUDIT.md` (the void reruns), task #66 (the plan).

---

## 1. Why the Liang protocol is canonical, and why that matters here

Liang, Krishnan, Hoffman & Jebara, *Variational Autoencoders for Collaborative Filtering*, WWW 2018
(`liang2018variational`) introduced Mult-VAE **and** released the split recipe and code (`dawenl/vae_cf`).
That recipe — **strong generalization**, i.e. held-out *users* rather than held-out interactions, with a
per-user 80/20 fold-in/target split — became the de facto standard for top-N evaluation in this literature.

This is precisely why our certification bridge has value. EASE at NDCG@100 0.420, Mult-VAE at 0.426, RecVAE
at 0.442 are comparable to each other **only because they are all reported on this one protocol**, and our
reproductions (EASE 0.4203, RecVAE 0.4425) are meaningful for the same reason. Both blind reviews called
that bridge the chapter's strongest asset.

**The corollary that matters:** the protocol's `r > 3.5` filter is not a choice we made about our baselines.
It is inherited by everyone who uses the split. The discard is a property of the field's standard instrument,
which is why the problem below is general rather than ours.

## 2. The problem, precisely

The recipe filters to `r > 3.5` and **discards** every sub-threshold interaction before the matrices are
built. It does not flatten a dislike to zero — the dislike is *absent*.

For the 10,000 ML-25M test users: the binary fold-in holds 626k interactions; their full rated history minus
targets holds 1.41M. **55.7% of rated history is discarded.**

Two consequences:
1. No model may use dislikes, however capable. R5 (answers carry sign and intensity) names a capability the
   protocol removes the input for.
2. The candidate pool still *contains* the user's disliked items, which can never be targets, and which no
   model can identify.

## 3. The author's proposed fix, and why it fails

*Mask every rated item out of the ranking, for all rating values, for all models.*

Technical note: it cannot be literally every rated item, since the held-out targets are themselves rated
(they are likes). The mask becomes `te_tr ∪ sub-3.5`. Applied uniformly it is internally consistent, and
arguably closer to deployment — one would not recommend a film the user rated 2 stars.

**Rejected on three grounds, the third decisive:**
1. It breaks the C1 bridge. `te_tr ∪ sub-3.5` is not Liang's protocol, so 0.4203/0.420 and 0.4425/0.442 stop
   being comparable to any published number — trading the chapter's most-praised asset for a private ruler.
2. It hands every model a **dislike-avoidance oracle at evaluation time**, using information the models were
   denied. Scores inflate, and differentially: dislikes skew popular under MNAR, so full-NDCG rises more than
   tail, muddying the decomposition.
3. **It erases the advantage it seeks to credit.** Once the evaluator removes disliked items for everyone, a
   dislike-aware model gains nothing. The fix deletes the phenomenon.

Author's own observation, which is correct and settles it: *"sure, it should raise everyone"* — a uniform
lift changes no comparison, so it buys no fairness; and the part that is not uniform cuts the wrong way.

## 4. Is the protocol therefore "rubbish"? No — silent, not incapable

The strongest form of the objection: *dislikes carry signal; some models use it, some cannot; admitting
dislikes to the input breaks the eval logic because demoting guaranteed non-targets is a free lift; therefore
the protocol cannot distinguish comprehension from filtering.*

**Half right. Step 3 is correct but its scope is only the *observed* dislikes.** A model that genuinely
comprehends a dislike does something a filter cannot: it shifts the latent and re-ranks **unseen** items —
demoting the neighbourhood of the disliked thing and reallocating mass. Whether that helps is decided by the
held-out likes, which the model never saw and which can perfectly well sit near the dislikes (people like
*some* films in genres they mostly hate). That effect **is** measurable here and is not inflationary: if
crude neighbourhood-demotion hurts genuine likes, NDCG punishes it.

The objection conflates "the benefit of demoting observed dislikes" with "the entire benefit of dislike
input". The protocol is **silent on sign and locally contaminated**, not structurally incapable.

**Never write "rubbish" about our own certification bridge.**

## 5. A correction we had to make twice

The `−0.0002` full-profile number bounds **intensity within likes** (4.0 / 4.5 / 5.0 vs all-1 on the same
support). It says nothing about **sign**, because a fold-in filtered to `r > 3.5` contains no dislike to
fold. The chapter briefly claimed it bounded "the channel the protocol forbids" — wrong, and corrected.
R5 has two halves and only one of them is measurable on this ruler.

## 6. What the literature already has — cite, do not christen

Full detail and verification levels in `external_literature/findings/sign_aware_evaluation.md`.

- **Frolov & Oseledets, RecSys 2016, "Fifty Shades of Ratings"** (arXiv:1607.04228) — **nDCL**, normalized
  Discounted Cumulative *Loss* over held-out items below a negativity threshold of **3.5, the same boundary
  Liang uses**. Their holdout explicitly admits relevant *and* irrelevant items, and they ran a
  **negative-only cold-start** condition (1–3 disliked items) — the published template for "what a dislike
  answer buys". They pre-empt the motivation almost verbatim.
  **This was already flagged in our own 24-Jul notes (`graded_inputs_for_ranking.md`, line 67) and missed.**
- **Sánchez & Bellogín, RecSys 2018, "Measuring Anti-Relevance"**, extended by **Mena-Maldonado et al.,
  TOIS 2021** — anti-precision / fallout / anti-nDCG on held-out negatively-rated items.

**Design constraints, all from the literature:**
- Avoidance must be a **separate lower-is-better number**, never negative gains inside NDCG — negative gains
  make NDCG unbounded and implementations silently clamp to 0 (Gienapp et al., CIKM 2020).
- Report it **popularity-stratified / tail**: false-positive metrics carry their own popularity bias because
  observed dislikes concentrate on popular items (Mena-Maldonado, TOIS 2021).
- **Full-rank probe, no sampled negatives** (Krichene & Rendle, KDD 2020).

**Contrast worth citing:** Google's production dislike metric (RecSys 2023, arXiv:2308.12256) is post-dislike
**responsiveness** — 60.8% / 64.1% fewer similar recommendations — precisely the *filtering* behaviour an
ungameable probe refuses to credit.

## 7. What is defensibly ours

Verified from primary text: **the sign-aware model line evaluates positives-only.** SIGformer (SIGIR 2024)
and SiReN report Recall/NDCG only — *no metric measures whether disliked items rank low*. Signed-feedback
models are published and their evaluations never test the capability.

Also ours: attaching the probe to the Liang binarised protocol beside frontier NDCG@10; the anti-gaming
rationale (observed dislikes are guaranteed non-targets ⇒ free demotion lift); the frozen-tower purpose; and
— searched directly, not found — **no established protocol measures the marginal value of a "no" answer
separately from a "yes"**.

## 8. The decision

**Primary: Liang verbatim, untouched.** The bridge is what lets a reader trust every other number.

**R5 is scoped to the scarce-evidence regime**, and this is the better and truer story: intensity is worth
`−0.0002` at full profile, `+0.0147` at k=2, gone by k=8. *Sign carries information precisely when evidence
is scarce and washes out once 60+ likes are on the table* — which is itself the argument for why an
**interview** instrument, not a full-profile recommender, is the right place to exercise R5. A finding, not
a confession.

**Claim wording:** parity on a stated protocol, never unqualified "SOTA". Definition 1 has been renamed
*"Parity on a shared protocol"* and all "ties SOTA" phrasings removed.

## 9. The 4-arm probe — SUPERSEDED by §11 (2026-07-30)

*Kept for the record of how the design evolved. The author cut it to two arms: the claim is a systems
comparison ("who converts the available signal best"), not a mechanism study, so the decomposition arms
(B, B′) are unnecessary. The final design is §11.*

Mask fixed at `te_tr` throughout; same targets, same pool; the `-inf` exclusion of observed dislikes applied
in **B, B′ and C** so C's residual edge is purely off-support generalisation.

| arm | input | who can run it |
|---|---|---|
| **A** | canonical, likes only | everyone (current numbers) |
| **B** | A + trivial re-rank demoting the user's own observed sub-3.5 items | everyone, near-free |
| **B′** | dislikes folded as **unsigned** positives | everyone |
| **C** | dislikes folded natively as **signed** observations | sign-capable models only |

- `gap(B,A)` = what the free filter is worth
- `gap(B′,A)` = what the extra exposure is worth
- **`gap(C,B′)` = sign-awareness** — *not* `gap(C,B)`, which credits sign for merely having 55.7% more data.
  **B′ was the confound originally missed.**

B is *not* contaminated: fold-in data belongs to the held-out user and is disjoint from targets by the
filtering.

**Gate before believing any arm** (→ memory `surprise-means-debug-the-harness`): push Most-Popular and EASE
through it first. Everyone should rise; what matters is whether they rise *unequally* and reorder the
frontier.

## 10. Open items (see §11 for the superseding experiment spec)

1. The chapter-wide **R5 scoping pass** (the limitation paragraph is in; the rest is not).
2. Optionally run A/B/B′/C with an nDCL-style avoidance number reported separately and popularity-stratified.
3. **Bib debt:** `frolov2016fifty` exists only in the findings file, not in any `.bib`; plus
   `sanchez2018antirelevance`, `menamaldonado2021tois`, `gienapp2020cikm`, `wang2023dislike`,
   `krichene2020sampled`. Author lists for RecSys 2025 and the SiReN/SIGformer orders need verification —
   the ACM pages returned 403.

---

## 11. FINAL DESIGN (author decision, 2026-07-30) — the two-arm A/N comparison

**Supersedes the 4-arm probe of §9.** The claim is a *systems* comparison — "given the same dataset, users,
split, and no cheating, which model converts the available signal into the best ranking?" — not a mechanism
study. So no decomposition arms, no forcing dislikes into binary models, no unsigned-positive control.
**Every model runs on its published input contract.**

### 11.1 The two arms

**Arm A — canonical anchor.** Liang recipe verbatim, unchanged, exactly the current G0 numbers. Its sole
job is certification: it ties every implementation (ours and baselines) to recognisable published numbers.
Nothing is re-run for A.

**Arm N — native regime.** Same 10k test users, same held-out targets, full-rank NDCG@10 (full + tail),
no sampled negatives. Two changes relative to A:

1. **Input = each model's published contract.**
   - RecVAE / Mult-VAE / EASE / TurboCF / iALS / SASRec / Most-Popular: binary likes — *identical to their
     A input*. Their N rows need only re-evaluation under the new mask.
   - Our tower: signed/graded fold-in, natively (what the belief update was built for). Re-eval only.
   - Golbandi / RBMF / TaNP: ratings-native, per their papers — like `r > 3.5`, dislike `r ≤ 3.5`,
     unknown = unrated, drawn from the user's full rated fold-in-side history. **These three retrain**,
     because the discard happens before training matrices are built: Golbandi's tree must be regrown on
     graded data (its dislike branches are dead on binarised data), RBMF's ridge refit on **centred** graded
     targets (the 1-star-as-weak-positive scar), TaNP meta-trained on ratings.
2. **Uniform mask: ALL rated fold-in-side items** (likes ∪ dislikes) are excluded (`-inf`) from every
   model's ranking, identically, regardless of what the model consumed. Targets remain scoreable (they are
   held-out likes, never in the fold-in). Rationale: (a) deployment-realistic — a served system filters
   everything the user has rated; (b) ungameable — a signed model cannot collect a free lift by demoting
   the observed dislikes it was shown, because they are already out of everyone's pool.

The A-vs-N delta per model is itself a finding: it shows exactly which models the canonical protocol
punishes and by how much (Golbandi is the predicted headline case — on A it is routed down the
"never-seen-it" branch for films the user hated).

### 11.2 Non-negotiable implementation rules

1. **Decouple input from mask.** `metrics.evaluate` currently derives the `-inf` mask from the model's
   input matrix. That coupling produced the fake Golbandi 0.3894 (richer fold-in silently deleted 2.26×
   more candidates). The N harness must take the mask as an **explicit argument**, fixed per user at
   all-rated, independent of what the model was fed. This is the single most dangerous line of the build.
2. **Canary gate before any number is believed** (memory: `surprise-means-debug-the-harness`). Push
   Most-Popular and EASE through the N harness first. Their input is unchanged, so their N numbers must
   differ from A **only** through the mask — expect a small, near-uniform lift. If they reorder, jump, or
   drop, the harness is broken, not the ranking.
3. **One variable at a time.** The three retrains change training data AND fold-in AND mask relative to
   their A rows. Stage it: (i) canary re-evals, (ii) binary models under N mask, (iii) our tower signed
   fold-in, (iv) the three retrains — each compared against the previous stage, never straight to A.
4. **Data already exists and is verified:** `src/baselines/graded_data.py` builds the graded matrices via
   `reproduce_partition()` — 1,411,761 nnz, checked against the instrument's all-bands reconstruction,
   zero target overlap. Threshold 3.5 throughout (Liang's own boundary, also Frolov's).
5. **HARD RULE 10:** commit the harness code before launching any run. Seeds and configs logged per row.
6. Prior void-rerun code (`run_graded_native.py`, `evaluate_decoupled`) may be salvaged for parts, but
   every result from the 29-Jul session is void and must not be compared against.

### 11.3 What this changes in the papers

- **Paper A is untouched** except the already-planned R5 scoping pass (§8). Arm A remains the chapter's
  ruler; the limitation paragraph stands.
- **Arm N is the designated arena for comparative elicitation claims** (baseline-ordering across
  recommenders, Golbandi with its dislike branch restored, sign-aware vs binary) — Paper B territory,
  reported side-by-side with the A column for transparency.
- Claim wording stays "parity on a stated protocol"; never unqualified SOTA on either arm.
