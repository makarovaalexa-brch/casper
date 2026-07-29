# STATE — where the project is now

> The single current-status doc. **Overwrite as things change.** Pointed to from `MEMORY.md`.
> End-state is in `VISION.md`. Last updated: **2026-07-29 (Paper A finalisation round)**.

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
- **A1 (#59) ✅ DONE** — contribution reframed onto the artifact that produced the numbers.
- **A2 (#60) ✅ DONE** — belief-MF recast; no longer printed under Bıyık's/ConTS's names.
- **A3 (#61) NEXT** — number reconciliation (details below).
- **A4 (#62)** — figures/tables: gap-map caption + self-scored axis, taxonomy-table redundancy,
  best-static table unnumbered/no CIs, graded caption's phantom "(a)", the G7 hole.
- **A5 (#63)** — writing: cut defensiveness ("aware of none" ×3, "this is worth more than it looks"),
  compress failure archaeology to one paragraph, halve lineage prose (~30pp is too long).
- **A6 (#58)** — final compile, number sweep, human-speed read-through. MUST grep for scripted-edit
  damage: this round introduced a truncated sentence and a corrupted `\ref` via automated edits.

### A3's open list (each checkable in-document; some need the source JSON)
- **sign-flip appears as THREE numbers**: −0.201, −0.2985 (CI [−0.3047,−0.2922]), "up to −0.29"
- **concepts@8 answers appears twice**: 0.2091/0.1222 vs 0.1630/0.0639; cold point 0.1717 vs 0.1279/0.0192
- **signed curve @q16**: 0.1372 vs 0.1379
- **G3b tail 0.2999** exceeds every other tail in the paper (frontier tail is 0.2497) — arm unstated
- **statistical double standard**: RecVAE's +0.0064 over EASE = "nominal lead, not settled"; T2′'s −0.0057
  vs RecVAE = "a tie… pays no measurable price". Same magnitude, opposite treatment, both flattering us.
- **C2 goalpost**: Limitations still describes C2 as a measurement while the battery discharges it "by
  construction". The construction argument is right — fix Limitations.
- **G2's "out-of-envelope canary"** is in the stated criterion but never reported.

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
| **TurboCF** | ❌ **0.0000 everywhere — BROKEN.** Must not enter the paper. Needs debugging. |
| **SASRec** | ❌ died 23:41 on `val_tr.csv`; the Liang split writes `validation_tr.csv`. Fixed + restarted. |
| **greedy polish, items-only** | ✅ **0.2098 → 0.2118** (1146 evals, converged) |
| **greedy polish, combined** | 🔄 running (was queued LAST by mistake; now runs in parallel via `--arms`) |

**Two self-inflicted nulls this round — watch for the pattern:** (1) `greedy_polish` built its swap pool
from the sequence itself, so every candidate was skipped and it reported "0 evals / no improvement" — a
confident-looking null that was a bug; (2) the earlier greedy ran on the SUPERSEDED fold because that was
the script default. Both were caught by inspection, not by the runs failing.

## What the polish decides
Combined trails items-only at q16 (0.2049 vs 0.2098) despite the combined bank CONTAINING the item bank.
That is possible because greedy picks one question at a time by immediate gain: concepts genuinely win the
opening, so it commits 4 concept slots and cannot undo them. The sequence space contains the items-only
solution; the greedy path does not reach it. Polish swaps positions scoring the FULL 16-sequence, and its
pool for the combined arm is items-first, so it CAN walk toward items-only.
**Note items-only improved to 0.2118, so combined must now clear a higher bar.** If combined still trails,
check HOW MANY concept slots were swapped out before concluding anything — a bounded search failing to
escape is evidence about the search, not about concepts.

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

## Parked / pending
- **PAPER B BLOCK (#56)** — belief-driven selection; do not start until A is closed. Σ-greedy is provably
  static-equivalent (Λ's update ignores answers); escapes are answerability and NDCG-coupling.
- **#30 Wave-2 baselines**: GF-CF/BSPM/BERT4Rec/EDDI not implemented; TurboCF broken. None can snap to
  published numbers (no Gowalla/Yelp/Amazon data, different metrics) — they are best-effort rows only.
- **#36** — G10 never run and unreferenced in the text; recommend DROPPING rather than running.
- **#34** — certified-tower interview re-runs: NOT needed. Author ruling: the non-certified run does not
  exist; keep numbers and say nothing about source, or re-run. Nothing is expected to change (both towers
  freeze the same RecVAE decoder; the graded curve replicated with the same shape).
