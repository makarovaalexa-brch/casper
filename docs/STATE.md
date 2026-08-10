# STATE — where the project is now

> The single current-status doc. **Overwrite as things change.** Pointed to from `MEMORY.md`.
> End-state is in `VISION.md`. Last updated: **2026-08-10 (TODO 1.2 concept-bank PV test DONE — the mechanism hypothesis is dead; Paper A closed pending author read)**.

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
