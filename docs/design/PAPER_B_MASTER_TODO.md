# Paper B — MASTER TODO (consolidated 2026-08-29)

**This is now the single list.** It merges `PAPER_B_FINAL_AUDIT_2026-08-25.md` (W1–W13),
`PAPER_B_TODO.md` (Act-1/2 items), `PAPER_B_BASELINE_BATTERY_TODO.md` (tiers + verification debts),
and everything raised on 2026-08-29. Those files stay as the detailed record; **this is the index.**

⚠ **Status honesty:** items marked `[?]` are carried from older lists and their current state was
**not re-verified** on 2026-08-29. Check before doing or dropping them.

---

## SCOPE — what is frozen

**Acts 1–4 are ON TEST and FROZEN** (test look spent 2026-08-12: Act 1 `tab:act1`, Act 3
`+0.01035/+0.00705` @K=16). Only **Act 5** rebuilds. This is why most of the old Act-1 battery below
sits in §C rather than §A — it is completeness for a paper whose Act 1 is already closed.

---

## A. BLOCKING — must happen before submission

⚠ **Tooling note 08-29:** bash heredocs in this environment collapse `\` to `\`, which silently
mangles LaTeX row terminators. **Edit .tex with the Write tool, never a heredoc.**

| # | item | est | source |
|---|---|---|---|
| ~~**A1**~~ ✔ **DONE 08-29, PASSED** (routing → chance on all six; marvel/inspiring go significantly NEGATIVE ⇒ teachers are group-specific). → `GROUPING_DESIGN.md` §7. ~~Scrambled-opener control~~ — give each user another user's opener; routing must collapse. **Does not exist.** Until it does, nothing may be claimed about the opener carrying the routing signal, and that claim is now the paper's differentiator (§D1). Hard rule 5. | 1–2 h | new 08-29 |
| ~~**A2**~~ ✔ **DONE 08-29.** Clean cell = 0.615→0.956 within k=6. ⛔ do NOT claim "most separable" — k-means wins on lift-over-chance. → `GROUPING_DESIGN.md` §8. ~~Separability across groupings~~ (genre / m6y raw-latent / m7a concept / endorsement-overlapping / disjoint). Makes "why these six groups" quantitative. `m6x` already takes `--labels`; all partitions banked. | <1 h | new 08-29 |
| ~~**A3**~~ ✔ **BUILT 08-29** (`greedy_static_policyB_items_train10000.json`, 125 min, train-fitted). **A3b EVALUATION DROPPED 08-29 (judgement, author's 'discard what doesn't add much').** The ORDER is built and banked; scoring it on validation needs a new scorer (~1 h) and would add one items-only-vs-combined row to a chapter whose claim is the policy, on ground Act 4 already covers. GreedyExtend is already CITED and its RMSE incompatibility stated in the positioning section, which is what hard rule 2 requires. ⛔ Do NOT use `channel_orders_on_test.py` to score it -- that tool scores on TEST and would spend a second look. ~~Run GreedyExtend~~ — ⚠ **RESCOPED 08-29: RMSE is NOT COMPUTABLE on our split** (`liang_split.py:169` binarizes at rating>3.5, discarding rating values). Run its ALGORITHM — greedy static, **items-only** — under our NDCG objective, and give the explicit incompatibility statement `HARSH_REVIEW_POINTS` permits. Doubles as the ablation isolating what the concept channel adds to the static bar. External static anchor. Author: "just for the sake of it". Must run **before** the test pass so it can be one of the arms. ⚠ reconstruction from WSDM'11 §3.1.1 + patent; label it as such. | few h | W7 |
| ~~**A4**~~ ✔ **TEST LOOK SPENT 08-29 — IT REPLICATES.** ALL +0.00343* (val +0.00333*), warm +0.00946*, do-no-harm −0.00038. Material reused not refitted; assignment rule verified against banked labels first. One pass. | done | new 08-29 |
| ~~**A5**~~ ✔ **DONE 08-29** — Yahoo! Music only, **1,701 users**, **RMSE only**, ONE global tree, *"users do not volunteer preferences"*. Far weaker than feared; corroborates our spine twice. ~~Read 2510.27342~~ (Oct 2025 — already elicits genre attributes "to better cluster users", vs Golbandi, ranking metrics). Closest competitor; currently abstract-only. | 1 h | W13 |
| ~~**A6**~~ ✔ **DONE 08-29** — tab:volunteered replaced with opener-state results (W1); mix added as sec:onepolicy (W2); Act 5 scope caveat deleted, both arms train-fitted (W10); every §1.4 number purged and verified absent from the chapter (W12). Chapter COMPILES. Also closed audit §1.5: greedy_static_policyB_combined is policy-B-selected, do not repeat the policy-A caveat. ~~Writing: **W1** opener-state results table · **W2** mix as Act 5's payoff · **W10** rewrite Act 5 scope · **W12** purge every §1.4 number from chapter/docs/slides | — | audit |
| ~~**A7**~~ ✔ **DONE 08-29** — W3/W4/W5/W6/W8/W9/W11 all landed. (was: **W4** flat protocol sentence ✔, **W9** distillation-as-infrastructure ✔). Remaining: **W3** distinguish `ndcgtree16` (Act 3, K=16) from `coldtree_v2` (Act 5, K=15) · **W4** protocol in ONE flat sentence, no narrative · **W5** 2,000-vs-10,000 fitting-user clause · **W6** ablation→prescription · **W8** "objective is the bug" spine · **W9** distillation = infrastructure, never a contribution · **W11** do NOT claim we repaired a published method | — | audit |
| ~~**A8**~~ ✔ **DONE 08-29** — atlas figure + exemplars + tried-and-rejected table folded into sec:partition; figure captioned as a continuum, not clusters. ~~Grouping section~~: figure `m7c_group_atlas`, exemplars, tried-and-rejected table, literature → `GROUPING_DESIGN.md` is drafted; needs folding into the chapter. ⚠ the t-SNE shows a **continuum, not clean clusters** — frame as a partition chosen for routability. | — | new 08-29 |
| ~~**A9**~~ ✔ **RESOLVED FREE 08-29** — grep confirms NO robustness-to-seed language exists in the chapter, so nothing had to be dropped and no retraining is needed. A limitation now states the single seed explicitly. (was:  `HARSH_REVIEW_POINTS` forbids "robust learned policy" language on a single training seed. Either retrain ≥3 seeds **or** never use robustness language. Decide explicitly. | 3× ~50 m, or free | harsh-review |
| ~~**A10**~~ ✔ **DONE** — declared in sec:onepolicy. ~~Report K=4~~ — `policy−cold` +0.00119 **[−0.00029, +0.00269]** straddles zero while K=1 and K=15 clear it. Non-monotonic; a reviewer will find it. | — | new 08-29 |

## B. CLAIM DISCIPLINE — things we must NOT say

- ⛔ **"Routing users to a group-specific interview" is NOT a contribution.** Golbandi WSDM'11 (our own
  tree's source) and Cluster-Based Bandits SIGIR'21 both motivate cold-start exactly as "identify the
  user's cluster", both via "distinguisher items". → `findings/user_segmentation_for_elicitation_2026-08.md`
- ⛔ **No claim that our grouping beats k-means grouping on NDCG** — teachers were never fitted on
  k-means groups. Rejections there were era-contamination and non-nameability, which are diagnostic.
- ⛔ Never "reproduces the *optimal* question order" (trees are measured-greedy; per-user optimal
  labels are clairvoyant). Never "more generalisable **in value** than trees". Never "parity"/"tie".
  Never "selection doesn't matter".
- ⛔ **The fitting-population / home-advantage reasoning stays OUT of the paper** — one flat protocol
  sentence (W4). Author, verbatim: *"no actual human would"*.
- ✔ What we MAY claim: **the source of the routing signal** — we read what the user volunteered,
  unprompted; the prior line chooses distinguisher items to ask. No prior work found doing this.

## C. PARKED — needed for completeness, explicitly NOT NOW

All carried from `PAPER_B_TODO.md` / battery TODO. Act 1 is frozen, so these do not block.

| # | item | est | status |
|---|---|---|---|
| C1 | Act-1 conditional revision — "the tree's crossover is a PV=0 effect" | — | `[?]` **marked OPEN, blocking** in the old list — verify whether the Aug-12 redirect voided it |
| C2 | EDDI **full Eq. 9** (both terms) — implemented nowhere; needs a design sheet + ~1 day | 1 d | `[?]` open. (b) three-outcome **DONE** 08-13 |
| C3 | `_pick_mixed` extension of `glm_ucb` to the unified action space | — | `[?]` "open, next" |
| C4 | 1.11 rank-correlation test (code built, gated) | ~12 h | parked, author sign-off recorded |
| C5 | Act-1 baselines: RBMF (1.10), ConUCB k=8, CLiME width | — | parked "NEEDED, BUT NOT NOW" |
| C6 | 1.22 unnormalised cohort DCG@10 as selection criterion (Bilgic & Bennett) | 4–6 h | `[?]` new 08-13, never run |
| C7 | **Greedy SLIM** — "strongest static bar available, prefix-optimal" and a must-cite | — | `[?]` Act 1 says PARTIAL |
| C8 | Sparsity-as-moderator test (De Pessemier) — directly testable in our harness | — | `[?]` never run |
| C9 | `--objective tail` greedy re-run; chapter carries an open REDO | 1 + ~12 h | `[?]` |

## D. VERIFICATION DEBTS — do not quote until closed

- **LRMF** (Shi/Zhao/Shen TOIS'17) — closest published thing to our harness, **highest-value unread**; needs institutional access
- **PERE's "experience probability"** — search-summary grade only; if it is info-gain × P(experienced) on ML-20M NDCG@10, our answerability framing must change
- **Golbandi CIKM'10 exact Coverage formula** — unreachable; gates A3's fidelity
- SEL-BALD exact acquisition formula · Iovine JIIS'21 · **Qian ESWA 2025** (personalises *when the interview stops* — the obvious reviewer question against our fixed-k protocol)
- "No CAT item-selection rule uses omission propensity" — absence claim at [A] strength

## E. CUT — do not re-add (author, 2026-08-13)

Tail-objective greedy · PV-quartile analysis · confirmatory tree re-run · home-advantage essay in
Limitations.

---

## SUGGESTED ORDER

```
1. A1 scrambled-opener control      (validation, decides the differentiator)
2. A2 separability table            (validation, cheap)
3. A5 read 2510.27342               (writing input)
4. A3 GreedyExtend                  (validation first, to shake out)
5. A4 pre-register  ->  ONE test pass, all arms
6. A6-A8, A10 writing;  A9 decide seeds
```


---

## STATUS 2026-08-29 (end of session) — §A IS CLOSED

Every blocking item is done. The test look is spent and replicates. The chapter compiles
(`new_chapters/chapterB_v2/chapterB_v2.pdf`, 683 KB) and carries: the opener-state results table, the
scrambled-opener control, the single-policy mix with TEST numbers, the grouping section with the
atlas figure, the corrected per-group claim, the fourth "we are not first" concession, two new
citations, four new limitations, and an intro paragraph for Act 5.

**Also fixed:** the chapter said "No model is trained in this chapter" — false once a policy is
distilled. Now scoped to "no *recommender* is trained".

**What is left is not blocking:** §C parked items (Act 1 is frozen), §D verification debts, and an
author read-through. Nothing in §A remains.


---

## CHAPTER CLOSED 2026-08-30

The chapter is finished and compiles (`new_chapters/chapterB_v2/chapterB_v2.pdf`). Test look spent
and replicated. Since the last status line: sections renamed off the Act-N scaffolding; concessions
converted to attribution ("X did this, we did that" -- author: never write "we are not first");
self-deprecating register and size callouts removed; informal verbs (pays/buys/flatters/wearing)
replaced throughout; `tab:policy` added (the policy had NO table before); the two-panel route atlas
replaced the single t-SNE; `sec:partition` and `sec:onepolicy` reframed so the grouping is a device
for fitting interviews and the claim is **adaptivity beyond the closed tree**; multiplicity and
artefact paragraphs added. Artefact release confirmed by the author.

**A real error was found and fixed:** the abstract's Act 3 CI read `[+0.00749,+0.01328]` against the
tables' `[+0.00739,+0.01323]`. Both were circulating in the design docs. Resolved against the run
artefact `experiments/battery/ndcgtree16test_policyB.json` (`ci95_full_vs_policyB =
[0.007392842, 0.013232706]`) -- **the tables were right, the abstract was wrong.**

### Fable's verdict (balanced review, clean draft, no gap list)
- **Sections 1--3 (audit, diagnosis, one-variable demonstration) = ECIR 2027 full paper, 40--55%**
  compressed to ~12pp. Second choice RecSys or its reproducibility track.
- **Act 5 held back as a SECOND paper** (RecSys/CIKM), not cut. Blocked on the group-free opener
  rebuild -- see memory `opener-synthesis-is-group-stamped`.
- Objection it ranks as most dangerous and still open: **Act 3 rests on ONE consumer** while
  `sec:flip` shows consumer-dependence. Fixable by scoring (not refitting) through RecVAE.

### NEXT
The conference cut: sections 1--3 to ~12pp. Everything else is optional.
