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
- **A1 (#59) ✅** — contribution reframed onto the artifact that produced the numbers.
- **A2 (#60) ✅** — belief-MF recast; no longer printed under Bıyık's/ConTS's names.
- **A3 (#61) ✅ 2026-07-29** — every A3 item closed (list below, with what each turned out to be).
- **A4 (#62) ✅ 2026-07-29** — gap-map self-score removed, best-static table numbered + captioned with the
  CI scale, graded caption's missing "(a)" added, G7 explained in the results caption too.
- **A5 (#63) ✅ 2026-07-29** — defensiveness and archaeology out (commit `0ef5c4c`). NOT done: halving the
  lineage prose. Deliberate — a 30pp thesis chapter and a venue submission are different artifacts;
  length reduction belongs to the submission pass, not here.
- **A6 (#58)** — final read-through. Scripted-edit damage scan is CLEAN (no mid-clause sentences, no stray
  `ef{`, no `\REDO`/`\TODO` uses — only the macro definitions remain).

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
- **#58 A6** — one human-speed read-through. Everything mechanical is clean.
