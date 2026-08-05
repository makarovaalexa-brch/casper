# What should be masked out of the candidate list when evaluating an interview recommender?

> Opened by the author, 2026-08-05: *"asking about an item might remove it from the recommendable set and
> thereby penalise a method for asking well."* Internal audit + literature pass. Verification marks:
> **[V]** full text · **[A]** abstract only · **[S]** search-result summary only.
> Companion: `docs/results/PROTOCOL_DISLIKE_DISCARD.md` §11 (the arm-N decision),
> `findings/rating_elicitation_family_deep.md` (Greedy SLIM entry — **corrected below**),
> `findings/sign_aware_evaluation.md`.

---

## 0. The three policies, named

| | policy | what is `-inf`'d out of the ranking | strategy-dependent? |
|---|---|---|---|
| **A** | **Fold-in exclusion** (Liang strong-generalization, the field default) | the user's fold-in *training* items only — under the Liang recipe that means their `r>3.5` fold-in likes. Sub-threshold ratings were **discarded from the data**, so they sit in the pool as permanent unmaskable non-targets | **no** |
| **B** | **Rated exclusion** (our arm N) | every item the user rated on the fold-in side, **likes ∪ dislikes**. Held-out targets are never in the fold-in, so they stay rankable | **no** |
| **C** | **Asked exclusion** | B (or A) **plus** every item *asked* during the interview, whether or not the user answered | **YES** |

C splits into two sub-cases that behave completely differently and must never be conflated:

- **C-answered** — mask asked items the user *did* answer. **Under policy B this is a no-op**: an answered
  item is by definition a rated fold-in-side item and is already out of the pool. C-answered only bites on
  top of policy A.
- **C-unanswered** — mask asked items the user did *not* answer. These are items the user never revealed
  anything about. Crucially, under our answer model an asked item that is a **held-out target** looks
  exactly like "unanswered" (the level lookup is built with `te` excluded, so the simulator must not reveal
  it). So **C-unanswered deletes held-out targets from the candidate list.** That is the author's fear,
  and it is real — it is implemented in one of our own code paths.

---

## 1. INTERNAL AUDIT — what each code path actually masks

Six distinct conventions are live in the repo. They do **not** agree.

| # | code path | mask expression (quoted) | policy | cohort | strategy-dependent? |
|---|---|---|---|---|---|
| 1 | `src/baselines/metrics.py::evaluate` | `X_pred[Xm.nonzero()] = -np.inf` with `mask_X` defaulting to `data_tr` | **A** by default, **B** when `mask_X` is passed | any | no |
| 2 | `src/baselines/arm_n.py::build_pool` → `run_interview_table.py::score_arm` | `mask_X=D["pool"]`, pool `= te_tr ∪ g_te_tr` | **B** | test 10k | **no** (by design) |
| 3 | `src/instrument/run_battery_phaseA.py::gate_g9` | `mask = ctx.va_tr + rows_to_csr(toks, …)` — comment `# tower parity + credit-neutral` | **A + C-answered** | val | **yes, weakly** |
| 4 | `src/instrument/strategy_channel_suite.py::genv_eval` (the greedy path) | `mask = ctx.va_tr[ch] + ans[cj]` — comment `# tower cold parity + answered items` | **A + C-answered** | val | **yes, weakly** |
| 5 | `src/instrument/strategy_ladder.py::eval_budget` | `mask = ctx.va_tr[chunk] + extra` where `extra` covers **every** `i in asked[r][:q]`, **plus** `te[j, i] = 0` for every asked item | **A + full C (answered *and* unanswered), with targets deleted from the ground truth too** | val | **YES, strongly** |
| 6 | `src/instrument/concept_cold_check.py::score` (used by `bank_sweep.py` — the concept-vs-item headline) | `sc[r, e[6]] = -np.inf  # never recommend back an answered item`; no fold-in mask (zero-history cold users) | **C-answered only** | 1,500 cold users | **yes** |
| 7 | the graded-vs-binary premium curve, `experiments/battery/premium_vs_k_t2final.json` | artifact records `"mask": "PARITY (liked fold-in only)"` | **A**, nothing asked-dependent | val | no |

### 1a. The author's suspicion: VERIFIED, with a correction

**Verified.** The greedy path (#4) does deviate from the arm-N decision. `run_interview_table.py`'s own
docstring names the deviation explicitly and rules against C:

> *"PROTOCOL. Candidate pool = the arm-N protocol pool (every rated fold-in-side item, likes AND dislikes),
> FIXED and strategy-independent. This deliberately departs from strategy_ladder's credit-neutral
> convention of masking every ASKED item: if the pool depended on what a strategy asked, a strategy that
> asks about popular items would delete those items from its own candidate pool, systematically
> penalising the popularity arm — a confound landing on exactly the arm the literature ranks third.
> An asked-but-unrated item still BURNS the question (the lit protocol); it simply stays rankable."*
> — `src/baselines/run_interview_table.py`, docstring

**But the correction matters, and it is the opposite of what the phrasing implies.** The greedy path is
**not** doing full C. `_env_item_state` masks only items with `lv < UNSEEN_LEVEL` — i.e. answered — and the
code comments the reasoning out loud:

> *"An asked-but-unrated item is never revealed to the user, so it must not leave the ranking; an ANSWERED
> item can never be a held-out target (lvl_lookup is built with te excluded). Assert it."*
> `assert ans.multiply(ctx.va_te[act]).nnz == 0, "an answered item is also a held-out target"`
> — `strategy_channel_suite.py:499-501`

So the greedy path already refuses C-unanswered and asserts no target is ever deleted. Its actual gap
against arm N is different and smaller: **it is on policy A, not policy B.** Because it never masks the
user's *unasked* dislikes, the only items the ask-set removes that would otherwise stay rankable are the
**asked items the user rated ≤ 3.5**. Under arm N those would be masked anyway, for everyone, regardless of
what was asked — which is exactly the strategy-independence arm N was built to buy.

**The genuinely non-compliant path is #5, `strategy_ladder.py`.** Its docstring:

> *"CREDIT-NEUTRAL masking of every ASKED item (answered or not: it was shown) — masked from candidates
> AND dropped from the IDCG denominator"*

and the implementation deletes the asked item from the ground truth as well (`te[j, i] = 0`). This is the
full C policy, and the IDCG-drop is an admission that it deletes real held-out targets. It makes the
measured task itself arm-dependent: two arms are scored against **different target sets**, which voids the
paired-bootstrap logic used everywhere else in the project.

**Blast radius check:** `strategy_ladder`'s outputs (`experiments/battery/strategy_ladder.json`) are **not
cited anywhere in `docs/` or `new_chapters/`** (grep-verified), and they were produced on the pre-certification
snapshot `t2i25_EP4_SNAP.pt`. No published number depends on the full-C path, and its numbers are stale on
two independent grounds. The fix is a re-run, not a retraction.

### 1b. How big is the strategy-dependent part? (measured, 2026-08-05)

The only strategy-dependent cells in paths #3/#4/#6 are asked items rated ≤ 3.5 (asked likes are already in
`va_tr`). Measured on the arm-N graded matrices, 10,000 test users, 16-question budget:

```
users=10000  graded nnz/user=141.2  (<=3.5: 78.6, >3.5: 62.6)  te_tr likes fold-in/user=62.6

bank                      answered/user      liked (already masked)   DISLIKED = extra mask   users w/ >=1
top-16 popular            5.42  (33.9%)      4.00                     1.43                    57.3%
rank 200-216 (mid-pop)    1.77  (11.1%)      0.89                     0.89                    43.0%
random 16                 0.08  ( 0.5%)      0.03                     0.06                    4.7%
```
(script: scratchpad `mask_delta.py`; loader `arm_n.load_arm_n`)

So at m=16 a **popularity** arm silently deletes ~1.4 guaranteed-non-target items per user from its own
candidate list that a **random** arm does not, and that a **concept** arm deletes *zero* of. On path #6
(the concept-vs-item headline) the item arm additionally masks the ~4.0 asked-and-liked items, which a
cold model would rank at or near the top — a much larger free lift.

**Direction of the bias is the reverse of the author's fear.** Masking an answered item removes a
*guaranteed non-target* that the just-informed model would have ranked very high, freeing top-10 slots for
real targets. C-answered therefore **rewards**, not penalises, an arm that asks answerable items. Under C
in general:

- **C-answered ⇒ free lift**, proportional to answer rate × how highly the model ranks the answered item.
- **C-unanswered ⇒ real penalty**, because unanswered ⊇ held-out targets under our answer model.

Path #5 does both; paths #3/#4/#6 do only the first.

### 1c. What Paper A's published numbers were computed under

- **Full-profile G0 table (0.3496/0.2491, EASE 0.3476, RecVAE 0.3540, …): policy A, Liang verbatim.**
  The chapter states it and defends it: *"the same binarised fold-in, the same candidate pool, the same
  binary targets"* (§ master table); and in the limitations, *"the candidate pool contains rated-and-disliked
  items that can never be targets … masking the disliked items out of the candidate pool would not be a
  repair: it would supply every model with a dislike-avoidance oracle the models themselves were denied."*
  Paper A therefore **explicitly rejects policy B for its headline table**.
- **The R3‡ interview curve printed in Paper A** (`0.1574 → 0.2609` full, `0.0505 → 0.1062` tail, fixed-bank
  rise `+0.1035` CI `[0.1000, 0.1071]`, random bank `+0.0051`) traces to `experiments/battery/g9.json`,
  i.e. code path **#3 = policy A + C-answered**. Verified by reproducing the exact values from the artifact.
- **The graded-vs-likes curve** (`+0.0147` at k=2, Figure `fig:gradedcurve`) is on **pure policy A** — its
  artifact `experiments/battery/premium_vs_k_t2final.json` records `"mask": "PARITY (liked fold-in only)"`,
  with no asked-item term at all. So even *within Paper A* the two interview figures use different pools.

### 1c-bis. How big is the A→B pool difference? **+0.027 NDCG at the cold state.**

Already measured and recorded in `docs/design/PAPER_B_BASELINE_BATTERY_TODO.md`, task 0.3: the greedy
(policy-A) table and the battery (policy-B) table *"are on different masks (141.2 vs 62.6 items/user ⇒
+0.027 at the cold state)."* That is an order of magnitude larger than most of the method deltas the
battery is trying to resolve (the concept-vs-item tail peak is +0.0310; the tower-vs-RecVAE full-profile
gap is ~0.004). **The pool choice is not a second-order detail — it dominates.** Any table mixing the two is
reporting the mask, not the method.

**Consequence for Paper B: the two papers are already on different pools.** Paper A prints A (+C-answered
on the interview curve); the Paper B battery (`run_interview_table` / `adaptive_harness`) prints B. An
A-vs-B delta is *not* a method delta — this is the same failure mode that produced the void
"Golbandi 0.3894" (memory: `surprise-means-debug-the-harness`). Paper B must print the pool rule in its
protocol paragraph and must never place an A-pool number next to a B-pool number in one table.

### 1d. Corrections to existing files

`findings/rating_elicitation_family_deep.md` line 395 says of Greedy SLIM: *"questionnaire items are
excluded from the recommendations (**same discipline as our harness**)."* The parenthetical is **wrong** and
should be struck. Greedy SLIM excludes **asked** items (policy C, answered or not); our battery harness
(`run_interview_table`) excludes **rated** items (policy B) and deliberately does not exclude asked items.
The two are different policies with opposite strategy-dependence properties.

Two housekeeping items in `external_literature/INDEX.md`: **`proissl2024greedyslim` appears twice** (two
different rows, differing in detail) — merge them; and **Sepliarskaia et al. RecSys 2018 has no row**
despite being cited in the prose of two addenda. Added below.

---

## 2. LITERATURE

### 2.1 The strong-generalization lineage (Liang → RecVAE → EASE → EDLAE): policy A

Liang et al., WWW 2018, *Variational Autoencoders for Collaborative Filtering* (`liang2018variational`),
and the released code `dawenl/vae_cf`, define the protocol every one of these papers reports on. Two
separate things happen, and conflating them is the standard error:

1. **Data construction discards** all interactions with `r ≤ 3.5` *before* any matrix is built. Dislikes
   are not masked — they are **absent**. (Measured on our split: 55.7% of the test users' rated history.)
2. **Evaluation masks** the fold-in items only: `X_pred[X.nonzero()] = -np.inf` in the vae_cf
   "Evaluate the model" cells — ported verbatim into our `metrics.py` (whose module docstring records the
   provenance line-by-line).

Verbatim, verified this pass [V paper + V code]:

- **Liang, paper:** *"For each held-out user, we randomly choose 80% of the click history as the 'fold-in'
  set to learn the necessary user-level representation and report metrics on the remaining 20% of the click
  history."* and *"We binarize the explicit data by keeping ratings of four or higher and interpret them as
  implicit feedback."* **The paper never states that fold-in items are excluded from the ranking — only the
  code does.** `VAE_ML20M_WWW2018.ipynb`:
  ```python
  X = test_data_tr[idxlist_test[st_idx:end_idx]]
  pred_val = sess.run(logits_var, feed_dict={dae.input_ph: X})
  pred_val[X.nonzero()] = -np.inf
  ```
  `X` is `test_data_tr` — **fold-in only**; nothing else is removed. The discard is upstream:
  `raw_data = raw_data[raw_data['rating'] > 3.5]`, applied *before* `filter_triplets` and before the user
  split. **Consequence, confirmed:** a film a test user rated 1 star is neither masked nor a target — it
  never enters the matrix, stays fully rankable, and simply counts as a miss if recommended.
- **RecVAE (Shenbin et al., WSDM 2020)** [V]: *"sort the items in descending order of the likelihood
  predicted by the decoder, exclude items from the training set"*; code `run.py`:
  `if not (data_in is data_out): ratings_pred[batch.get_ratings().nonzero()] = -np.inf`, where
  `get_ratings()` defaults to the **input** (fold-in) matrix. Identical semantics to Liang.
- **EASE (Steck, WWW 2019)** [V]: states only that the evaluation follows Liang's strong generalization.
  **No statement of its own** about candidate-list construction — it defers wholly to Liang.
- **EDLAE (Steck, NeurIPS 2020)** [V] — the most explicit in the lineage, and it is load-bearing for his
  central claim: *"the set of items j in a test-user's interaction-history (which serves as input to the AE)
  **is disjoint from** the test-user's set of test-items i to be predicted (see [21]). Consequently, each
  item i has to be predicted based on the other items j≠i, so that prediction accuracy immediately suffers
  in these experiments if the AE overfits to the identity function."*

So the lineage is unambiguously **policy A**, and it is a code convention that only one of the four papers
writes down. **Nothing in this lineage has any notion of an "asked" item** — there is no interview.

### 2.2 The elicitation/interview literature

**Greedy SLIM — Proissl, Vatic & Waldschmidt, arXiv:2406.06061 (2024), *"Greedy SLIM: A SLIM-Based Approach
For Preference Elicitation"*. [V] full HTML text, v1.**

Our deep-research claim is **CONFIRMED on both counts**, verbatim:

> §3.3.2: *"Let I_Q be the set of items asked by the questionnaire and let I_R be the set of recommended
> items. **We always require I_Q ∩ I_R = ∅ to avoid trivial recommendations.**"*

> §3.3.1: *"Let I_R ⊆ I be the item subset from which the recommender system is allowed to draw the
> recommendations. This may not be the complete item set I as we sometimes exclude items."*

This is **policy C in its strongest form: the exclusion is on the ASK, not on the answer.** An item asked of
a test user who never rated it copies a `0` into the fold-in row (no information) *and* is burned out of the
candidate pool. Their own encoding confirms unknown = 0: *"the values x_ui of X are either 0 if user u has
not rated item i yet or x_ui = r_ui."* In their user study only **34.4%** of Q_GSLIM and **48.7%** of
Q_Bandit questions received a non-zero rating — so the majority of asks were pure candidate-pool deletions.

And the load-bearing attribution, §6.5:

> *"The NDCG of Q_Bandit converges or even decreases after ten questions. **We think that the main reason
> for the decreasing NDCG is that the questionnaires are not allowed to recommend items that they contain
> as questions. Therefore, using popular items as questions may have a negative impact on the
> recommendation performance** if the questionnaire is unable to generalize the gained information."*

Same mechanism stated earlier for their own all-items setting, §4:

> *"Considering all items (left) we can observe that, surprisingly, the recommendation accuracy decreases
> with the number of questions. The questionnaires mainly consist of popular items that would have been
> good candidates for the recommendation. **As we restrict the recommendation list to items not included in
> the questionnaire, asking questions about popular items may have a negative effect on the recommendation
> performance.**"*

**How we must now cite it.** The "adaptive degrades past k≈10" curve in that paper is, *by the authors' own
diagnosis*, a **protocol artefact** — and more precisely a **popularity-of-asked-items × exclusion-policy**
interaction, not a property of adaptivity. Q_Bandit (Christakopoulou et al. 2016) asks popular items;
Q_GSLIM does not. Note also Q_Bandit still **wins at short budgets** (*"Q_Bandit clearly outperforms
Q_GSLIM during the first ten questions"*). Never cite this paper as evidence that adaptive selection stops
paying at long budgets. Cite it as **published evidence that policy C penalises whichever arm asks the most
recommendable items.** Other details [V]: ML-25M unfiltered (162,541 u / 59,047 i) + Netflix; 90/10 random
user split; explicit 1–5 ratings, not binarised; NDCG@10 primary, all-items **and** long-tail (Cremonesi
33% head); budgets 5/10/15/20; *"we do not apply any kind of item sampling to avoid the problems reported
in Krichene and Rendle"*.

#### The rest of the elicitation family: everyone excludes asked items, almost nobody says so — and three different mechanisms are at work

**The one-paragraph answer.** *Every* paper in this family excludes the asked/elicited items from the
evaluation; none leaves them rankable. But they arrive there three different ways, and only two of the
three are a *choice*:

**(a) Split-by-construction — the RMSE/tree lineage. The question never arises.** Interview answers come
from the train side, evaluation from a disjoint held-out side of the same user, and the candidate "list" is
**only the user's own held-out rated items**, not the catalogue.

- **Rashid et al., IUI 2002** [V]: *"we withheld all of that user's ratings from the system. As we presented
  the movies, users 'rated' the movies they had 'seen' … we used these as training data for that user and
  made predictions for **all of the other movies the user had rated** in the original dataset."* MAE.
- **Rashid et al., SIGKDD Expl. 2008 (IGCN)** [V]: 80/20 per-user rating split; *"Predictions are computed
  using the entire training data and the new profile, after u_t's old rating information is discarded; that
  is, using Dtrn − D^{u_t}_trn + D^{u_t}_seen."* MAE + Expected Utility.
- **Golbandi, Koren & Lempel, WSDM 2011** [V]: *"For each resulting seed set, we predict all ratings in the
  test set, while accounting only for train set ratings on items belonging to the seed set."* **The test set
  is FIXED and independent of the seed set** — exclusion is an artefact of the Netflix per-rating split and
  **the paper never remarks on it.** RMSE. Both signs retained (like/dislike/unknown around ~3.6 stars).
- **Zhou, Yang & Zha, SIGIR 2011 (fMF)** [V]: each cold user's ratings split 75/25 into an *"answer set"*
  and an *"evaluation set"*; asked items are excluded by disjointness. Sun et al., WSDM 2013, uses identical
  wording and calls it *"the standard settings"* — so this **is** the convention in the tree/RMSE lineage.

  **Note what this means for us: none of these papers rank a catalogue at all.** Their candidate set is a
  handful of the user's own rated items. Their protocol does not transfer to a full-catalogue NDCG@10 setting
  and cannot be cited as precedent for either policy A, B or C.

**(b) Explicit set-difference at ranking time — the RBMF/DRE lineage. A deliberate choice, stated.**

- **Liu et al., RecSys 2011 (RBMF)** [V] — the clearest statement in the whole literature: *"for the users in
  the test set, we make predictions only based on their ratings on the representative items. The
  effectiveness of the representative items are then evaluated based on **how well they help identify which
  of nonrepresentative items a user also likes**."* Candidate list = full item set **minus representatives**,
  with sub-threshold *and* missing entries kept in as explicit negatives (*"a bad recommendation if the
  rating is either lower than 4 or unknown"*, NMAR-motivated). Pre@K, MAP, AUC. **Because RBMF's seed set is
  a single global set, the exclusion is identical for every user and does not vary across the compared
  methods** — the confound is latent but unexercised.
- **Fonarev et al., ICDM 2016 (RMVA)** [V]: inherits RBMF's pipeline explicitly (*"similar to the one
  introduced in [8]"*) but **does not state whether seed items are dropped from the ranking.** Unverified.
- **Kweon et al., WWW 2020 (DRE)** [V paper + V code] — the exact analogue of our `strategy_ladder`:
  *"We define seed itemset as a set of items that we ask new users to rate … **We also define candidate items
  as all items except the seed items**"*, with ground truth defined against the reduced set (*"V(u) as the set
  of **candidate items** that user u interacted with (i.e., the ground-truth items)"*). The code
  (`WonbinKweon/DRE_WWW2020`) deletes the columns from **both** sides:
  ```python
  mask = torch.ones(x.size()[1]);  mask[seed] = 0
  x = x[:, mask==True];  x_recon = x_recon[:, mask==True]
  ```
  **So DRE is policy C with the target set shrunk too — precisely what `strategy_ladder.py` does.** It is
  survivable there only because DRE's seed set is static and shared; the moment two selectors with
  *different* seed sets are compared under this rule, they are scored against different ground truths.
  Binarised at >3.5, so sub-threshold ratings become 0 and stay in the candidate list as negatives.
  P@N, NDCG@N.

**(c) A global item-level firewall — Sepliarskaia et al., RecSys 2018 (SPQ)** [V]. The strictest and, for our
purposes, the most interesting design. Algorithm 3 splits the **item set**, not the ratings:
> *"divide I into two subsets of equal size randomly: I_test, I_train … create a questionnaire, consisting of
> questions about items from **I_train** … choose test questions about items from **I_test**."*

Askable and evaluable items are globally disjoint, fixed in advance, and **independent of the arm** — which
gets policy C's "you can never recommend back what you asked" property *without* any strategy-dependence.
This is the one published design that solves the problem head-on. Its cost is that half the catalogue is
unrecommendable and the tail metric is computed on half a catalogue, which for us would break the tail
head-mask definition and the certification bridge. Precision@10.

**Elahi, Ricci & Rubens, Computer Science Review 2016 (the survey)** [V] — **states no normative
convention.** It describes the mechanic (*"The system elicits some ratings of this new user from the unknown
set … Finally, the prediction of the system on test ratings (white boxes) is evaluated"*) and flags the
offline/online mismatch (*"in an online interaction … a candidate item … can be any item of the catalogue,
while in offline experiments only ratings that have been already expressed by users can be requested"*), but
never rules on what to exclude. It also confirms the metric problem: *"the majority of the surveyed works …
measured … MAE and RMSE reduction. Hence, we cannot compare many of the surveyed strategies also with
respect to … NDCG or MAP."*

**PERE: NOT IDENTIFIED.** dblp returns zero hits for `PERE preference elicitation`. Our INDEX carries a
"PERE" entry (Nguyen et al., UAI 2024, arXiv:2406.00973) — that is the paper meant, and its candidate-set
rule was **not** established this pass. Open item.



### 2.3 Is there a methodological paper on this exact choice?

**No paper is about "what to exclude from the candidate list in interview/AL evaluation."** Verified by
sweeping the **complete forward-citation set of Parapar & Radlinski (28 works)** — not one is a
candidate-set methodology paper, and **nobody has built on `AvailablePreferences` at all**. Three adjacent
literatures exist; none rules on it.

#### (i) Candidate-set construction in offline top-N — Bellogín, Castells & Cantador [V]

*"Precision-oriented evaluation of recommender systems: an algorithmic comparison"*, RecSys 2011, extended
as *"Statistical biases in IR metrics for recommender systems"*, Inf. Retrieval J. 20(6) 2017. The taxonomy
has **five** members, not three, and the framing sentence is the one we want:

> *"The main substantial difference between how this general evaluation scheme has been brought to practice
> by different authors lies on **how the set of target items is formed**"* … *"(note that **items already
> present in the user's profile are excluded in the evaluation**)."*

- **TestRatings** — only the user's own test-rated items. *"significantly overestimated"*; the outlier that
  agrees with RMSE and nothing else.
- **TestItems** — all items with a test rating by *some* user, *"and no training rating by the target user,
  **since it makes no sense to predict known training ratings**"*. *"all users are tested on the same set of
  target items."*
- **TrainingItems** — every item rated by anyone, minus the target user's.
- **AllItems** — the whole catalogue minus the target user's rated items.
- **One-Plus-Random** — one relevant item + N random (N=1000 in Cremonesi RecSys'10).

> §2.6: *"TestRatings … shows a heavy bias towards known relevant items … **TrainingItems and AllItems
> represent the opposite end of the spectrum** … **TestItems and One-Plus-Random appear to be somewhat
> fairer**."*

**This is the single most directly usable citation for our ruling, and it argues for policy B.** The only
explicit exclusion rule the taxonomy states is *"items already present in the user's profile are excluded"*
— a **profile** rule, not an **ask** rule. Under an interview, an asked-and-answered item **is** a profile
item, so the standard protocol already implies excluding it *because it is now rated*, not because it was
asked. That is exactly policy B's logic, published in 2011. **Nobody, however, states this for the interview
case or measures the consequence.**

#### (ii) Parapar & Radlinski, WSDM 2021 — **OUR PRIOR SUMMARY WAS HALF WRONG; correct it** [V full text]

Free full text: `https://storage.googleapis.com/gweb-research2023-media/pubtools/6086.pdf`.

**Correction 1 — `AvailablePreferences` is a JUDGMENT-set change, not a candidate-set change.** Verbatim:

> *"many items in the training split are never used … If the system does not choose to ask the user about
> any particular item, the rating for that item is, in effect, wasted data — we term such unused rated items
> **leftovers**."* … *"we propose to compute the results … by **including the leftover preferences as
> relevance judgments**, together with those in the test set used by TestItems. We also remark that this
> does not affect the recommendation process: **the exact same system output is produced under both
> evaluation methodologies**."*

So it is **orthogonal to this file's question**. It changes the qrels, not the mask. It can be applied on
top of policy B at zero cost and **without re-running the recommender**.

**Correction 2 — the mechanism is POPULARITY, not entropy, and they run NO entropy baseline.** Their arms
are `Greedy`, `Greedy+MMR`, `DPE-ILD`, `DPE-Hellinger`, `DPE-Oracle` — that is all. The measured result:

> §5.1: *"**for every cut-off and metric, the Greedy baseline performs much worse than DPE** … Greedy tries
> to maximize recommendation performance, and RSs are known to be biased to popularity … **When evaluating
> on a small proportion of the relevance judgments, it performs well because popular items would also be
> more frequent in the test data.**"*
> §6: *"reduce the undesired effect of missing ratings, **which tends to favor methods that over-weight
> popularity**."*

The entropy sentence our INDEX quotes is a **single summary-section inference from the literature, not a
measured result**:

> §5.3: *"previous entropy reduction approaches to preference elicitation are favored by the sensitivity of
> the TestItems offline approach to the missing ratings problem"*

— and §2.1 makes clear it means *popularity-blended* entropy (Rashid's `Entropy0`/HELF): *"popular items
have a higher chance of having been rated in offline test data used to evaluate elicitation algorithms."*

**What this means for us, and it is sharper than the version we had logged.** The threat is
**popularity-shaped, not entropy-shaped** — which is *exactly* what our own bank sweep already found
independently (`CONCEPT_VS_ITEM_BANK_SWEEP.md`: *"`entropy` is degenerate — a popularity criterion in
disguise"*, top-16 at the 99.9th popularity percentile). So Parapar & Radlinski supply the **published
mechanism for our own measured result**, and the correct statement of the threat is:

> *if a bank wins because its criterion correlates with popularity, and the judgments come from a sparse
> held-out split, the win may be an artefact of missing ratings rather than of information gain.*

By the same token, the **`polar` bank (pure discrimination, popularity-agnostic) is the arm least exposed
to this threat**, and it is the arm our headline rests on — which is a defence, not a coincidence, and
should be written that way. **Task 0.4 in `PAPER_B_BASELINE_BATTERY_TODO.md` must be re-worded**: it
currently says the threat lands on "entropy arms"; it lands on *popularity-correlated* arms, and the
remedy (fold leftovers into the qrels) is cheap, orthogonal to the pool ruling, and does not require
re-running the recommender. **This file does not discharge task 0.4.**

Also note the apparent contradiction and its resolution: Bellogín treats TestItems as the **bias-reducing**
choice (vs AllItems), Parapar treats it as the **bias-introducing** choice for elicitation. Both are right —
TI reduces the popularity advantage of a naive *recommender* while amplifying the missing-ratings advantage
of a popularity-seeking *elicitor*. Have that sentence ready for a reviewer.

#### (iii) Sampled-metric critiques — Krichene & Rendle, KDD 2020 [V via arXiv:1912.02263 companion]

> *"**sampled metrics are inconsistent with their exact version. Sampled metrics do not persist relative
> statements, e.g., algorithm A is better than B, not even in expectation.**"*

**Checked directly: it does NOT bear on filtering.** The full text has nothing on `filter`/`candidate`/
`popular`; the only exclusion is mechanical (*"the pool … excludes the truly relevant item"*), and the whole
analysis assumes **uniform random** sampling of irrelevant items. Structured filtering is a strictly
stronger perturbation — non-uniform and correlated with the estimated quantity — so **nothing here licenses
the assumption that filtering is harmless.** Cite it for the moral (*the candidate set is part of the
metric's definition, and changing it can reverse rankings even in expectation*), never as if it analysed
filtering. Greedy SLIM cites it as its reason to avoid item *sampling* — and then filters the candidate set
per-arm anyway, which is the same error in a different coat.

#### (iv) Cañamares & Castells — the closest thing to a paper on our question [V for the IRJ companion]

- **Cañamares, Castells & Moffat, *"Offline evaluation options for recommender systems"*, Inf. Retrieval J.
  23(4) 2020** [V, author preprint]. Their **Decision C1 "Condensed or Full"** is precisely the
  candidate-filtering axis, and their verdict is the quotable one for our protocol paragraph:
  > *"varying the split … or changing the evaluation metric, **or how target items are selected** … can give
  > rise to comparisons that are vulnerable to misinterpretation, and **may lead to different or even
  > opposite outcomes**."*
  > *"[condensing] means that **many – perhaps the great majority – of the (u,i,s) tuples generated are
  > discarded. Significant distortion to the metric score might be introduced as a result, since the task in
  > the experiment no longer represents the task the algorithm serves in production**."*
  > *"**if experimental outcomes can be affected by the methodological decisions made in these regards just
  > as much as they are by the innate quality of the RS systems being compared, then great care is
  > required.** The very first step should be a detailed statement in regard to experimental settings."*

  They also demonstrate the flip empirically: condensing removes a training-ratio artefact entirely and
  makes the kNN-vs-MF ordering insensitive to the split on Netflix. **That is the general-form evidence for
  our +0.027 A-vs-B number.** Their honest closing — *"Neither approach seems truly satisfactory, because
  the advantage of each is the Achilles' heel of the other"* — is worth quoting when we justify reporting
  both columns.
- **Cañamares & Castells, SIGIR 2018 [A abstract only]** — *"contradictions between the accuracy that would
  be measured in a common biased offline experimental setting, and the actual accuracy."* Closed access,
  confirmed no OA copy.
- ⚠ **Cañamares & Castells, RecSys 2020, *"On Target Item Sampling in Offline Recommender System
  Evaluation"* (DOI 10.1145/3383313.3412259) — NOT OBTAINED, closed access.** Its *title* is the closest
  match to our question in the entire literature. **Pull it through institutional access before we write any
  novelty claim** — a reviewer will ask.

#### (v) Selection bias in elicitation

Gupta et al., arXiv:2405.00554 (2024): *which items you ask about* is itself a biased sample.
Already logged in `findings/adaptive_elicitation_baselines.md` §5 with the note that our fixed
strategy-independent candidate pool is the guard. That note is correct; this file is its justification.

#### Verdict on the gap

**Genuine, and narrow.** No paper states a convention for what to exclude from the candidate list in
interview evaluation, argues for one, or measures the bias of the alternatives. Parapar & Radlinski is the
only paper that changes an evaluation setup *because of elicitation*, and they explicitly leave the
candidate set alone. Consequences: (a) **state our rule explicitly in the protocol paragraph** — there is no
citation to lean on; (b) the sensitivity analysis in Q5.6 is a small, cheap, genuinely publishable
methodological contribution, and we already have its headline number, **+0.027 NDCG at the cold state
between A and B** (§1c-bis), larger than most method effects the battery reports; (c) **two verification
debts before any novelty claim**: the RecSys 2020 target-sampling paper, and the Elahi/Ricci/Rubens AL
survey's evaluation section (ScienceDirect 403 this pass) — the one remaining place a boilerplate exclusion
rule might already be written down.

---

## 3. VERDICT — the five questions

### Q1. Which policy is cleanest?

**Policy B — mask every rated fold-in-side item, likes ∪ dislikes, fixed per user before the interview
starts.** For our protocol specifically (asked-but-unanswered burns the budget; answers are real held-out
ratings), B is the only policy that is simultaneously:

- **strategy-independent.** The pool is a property of the *user*, not of the ask-list. Two arms are scored
  on identical candidate sets and identical target sets, which is what makes the paired bootstrap valid.
- **deployment-realistic.** A served system filters everything the user has already rated.
- **ungameable.** It removes the free lift a signed model could collect by demoting the observed dislikes
  it was shown (`PROTOCOL_DISLIKE_DISCARD.md` §11.1), and it removes the free lift an item-asking arm
  collects by deleting the items it just asked about.
- **a superset that makes C-answered vacuous.** Under B there is nothing left for "mask the asked items" to
  do except delete unanswered ones — which is exactly the harmful half.

Policy A is the right choice for the *full-profile certification table* and must stay there (it is the
bridge to EASE/RecVAE/EDLAE published numbers, and Paper A defends it explicitly). It is the wrong choice
for the interview battery, because it leaves the strategy-dependent residue of §1b in play.

Policy C should not be used at all *in the form we would use it* (per-arm ask-lists over a shared
catalogue). Greedy SLIM is the cautionary case: a published paper whose headline "adaptive degrades past
k≈10" result is, by its own authors' diagnosis, produced by C.

**The one respectable version of C is Sepliarskaia's (RecSys 2018): split the *item set* globally and in
advance into an askable half and an evaluable half.** That buys C's "never recommend back what you asked"
property with zero strategy-dependence, because the exclusion is a property of the catalogue, not of the
arm. We should not adopt it — it halves the catalogue, breaks the top-33%-mass head mask, and severs the
certification bridge — but it is the design to cite if a reviewer asks why we did not exclude asked items.

Note also that the field's apparent unanimity on "asked items are excluded" is mostly an **artefact of a
protocol we do not share**: in the RMSE/tree lineage (Rashid, Golbandi, Zhou, Sun) the candidate set is only
the user's own held-out *rated* items, so an asked item is out by disjointness and the choice is never made.
Only RBMF, DRE and Sepliarskaia make it a choice, and of those only Sepliarskaia makes it arm-independent.
**There is no precedent that binds us**, and the one lineage whose protocol we *do* share — Liang/RecVAE/
EASE/EDLAE — has no asked-item concept at all.

### Q2. Is masking asked-but-UNANSWERED items defensible? **No.**

**For (the honest case).** (i) The item was *shown* to the user on screen; a deployed UI would not put it
straight back in the recommendation slate one turn later, so C is arguably more deployment-faithful about
screen real estate. (ii) It is *credit-neutral in intent*: neither arm can profit from asking about an item
it would have recommended, so no arm can "cheat" by asking about its own top prediction. (iii) It makes the
budget cost of a question uniform across arms — every question costs one candidate, answered or not.

**Against (decisive).** (i) **It deletes held-out targets.** Under our answer model, an asked item that is a
held-out target is indistinguishable from "unanswered" (the level lookup excludes `te` so the simulator
cannot reveal it). Masking unanswered items therefore removes real positives from the ranking — the exact
harm the author suspected. `strategy_ladder.py` handles this by *also* deleting them from the ground truth,
which is worse: the arms are then scored against **different target sets**, so their NDCGs are not on a
common ruler and paired CIs are meaningless. (ii) **The user revealed nothing.** No information was
transferred, so no information-based justification exists for changing the candidate set. The item remains
a perfectly legitimate recommendation — indeed for an unanswered *popular* item it is often the *best*
remaining recommendation. (iii) **It is a strategy-dependent tax proportional to answer rate.** An arm with
a 0.5% answer rate (random catalogue) pays almost the full 16-question tax; an arm with a 34% answer rate
pays two-thirds of it. That is a confound on precisely the axis (answerability) our concept-vs-item claim
is about. (iv) **The "it was shown" argument confuses UI policy with evaluation.** Screen-repetition is a
presentation constraint, not a relevance judgement; if it matters it should be a separate reported number,
not folded into NDCG — the same discipline `sign_aware_evaluation.md` imposes on the avoidance metric.

Note the asymmetry the "credit-neutral" framing hides: **a concept question masks nothing**. Any
asked-item exclusion is therefore automatically a *channel*-dependent rule, not just a strategy-dependent
one, in a paper whose headline compares channels.

### Q3. Is masking asked-AND-ANSWERED items defensible? **Yes, but do it via policy B, not via C.**

**For.** Recommending back an item the user just told you they have seen is trivially non-useful, and under
our protocol it can never be a target anyway (targets are excluded from the answer lookup, and
`strategy_channel_suite.py:501` asserts it). Leaving it rankable lets a model bank NDCG-neutral but
deployment-absurd behaviour and, more practically, lets a just-informed model park its own fold-in item at
rank 1 — which is a *pure* non-target occupying a top-10 slot for everyone.

**Against.** It is a **free lift proportional to how well the arm asks**, which is the author's worry in its
correct form: masking removes guaranteed non-targets that the model would have ranked high, which *promotes
targets*. Measured on our data (§1b) a popularity arm at m=16 removes ~5.4 answered items per user (~4.0 of
them likes the model has just been told about, which a cold model ranks near the top) while a random arm
removes ~0.08 and a concept arm removes **zero**. So C-answered does not penalise good asking — it
**rewards** it, and it rewards the *item* channel over the *concept* channel by construction.

**The resolution is that policy B makes this moot.** Under B, the answered items are already out of the pool
*for every arm and every channel*, because they are rated — whether or not anybody asked about them. The
"reward for asking well" disappears because the exclusion no longer depends on the ask.

**Consequence for the current headline, and it is the good direction:** the concept-vs-item sweep
(`bank_sweep.py`, code path #6) runs under C-answered with no fold-in mask, i.e. the *item* arm receives a
free lift the *concept* arm does not. Concepts win both axes at every budget under the polar bank **despite**
that handicap. The result is therefore **conservative**, and re-running it under policy B should, if
anything, widen the concept margin. That is a robustness check worth running and reporting, not a threat.

### Q4. Does the choice favour or penalise ADAPTIVE selectors?

**Yes, and the sign flips depending on which half of C you use.**

- **C-unanswered penalises adaptive selectors.** An adaptive selector conditions on the answers so far and
  asks increasingly *user-specific* items — items it believes this user is likely to have seen and liked.
  Those are exactly the items most likely to be, or to sit near, held-out targets. Every such ask is a
  candidate deleted (and, under our simulator, sometimes a target deleted). The penalty compounds with
  budget, which is precisely the shape Greedy SLIM observed: **Q_Bandit ties or wins at k≤10, then declines
  monotonically to k=20** (ML-25M full 0.3382→0.3456→0.3221→0.2916 at k=5/10/15/20). The authors attribute
  it to the exclusion rule in as many words. This is a published instance of the mechanism, not a
  hypothetical.
- **C-answered rewards whichever selector achieves the higher answer rate.** Adaptive selectors typically
  achieve higher answer rates (that is much of what adaptivity buys), so C-answered hands them a lift.
- **Under policy B, neither effect exists.** The pool depends only on the user.

**Why the field has largely not noticed.** RBMF and DRE both use policy C, but both use a **single static
seed set shared by every user and every compared method**, so the exclusion is a constant and cancels in
every comparison they run. The confound is latent, not exercised. It becomes live the moment you compare
two selectors that ask *different* items — which is exactly Act 1 (static vs adaptive) and exactly the
concept-vs-item comparison. Greedy SLIM is the first paper in this family to compare a static and an
adaptive questionnaire under C, and it is also the first to trip over it.

There is a second-order asymmetry worth naming: even under B, adaptive selectors change *which* items are
folded in, not which are masked — so their advantage is measured purely as ranking improvement, which is
what we want. This directly protects the Paper B Act-1 claim (static vs adaptive at matched budget) from
the single most likely reviewer objection, and it means **our "no consensus that adaptive beats static at
k≤16" reading of the family must now be re-stated with a caveat**: one of the two headline
static-beats-adaptive results in the literature (Greedy SLIM) is protocol-contaminated in the direction of
its own conclusion. Do not cite it as clean evidence for static. Cite SPQ/Sepliarskaia 2018 for that, and
cite Greedy SLIM for the protocol lesson.

### Q5. What must we do so every arm in our battery is comparable?

**Recommendation — adopt policy B as the single battery-wide rule, and print it.** Policy B is already
implemented, already canary-gated, already asserted leak-free (`arm_n.py`), and already the author's
standing decision (`PROTOCOL_DISLIKE_DISCARD.md` §11, 2026-07-30). The work is migration, not design.

1. **One pool rule for the whole interview battery: policy B (`arm_n.build_pool`), fixed per user,
   computed before any question is asked, identical for every arm, every channel, every recommender, every
   budget.** `run_interview_table.py` / `adaptive_harness.py` / `interview_paired.py` already do this. They
   are the reference implementation.
2. **Migrate the three non-compliant instrument paths** to take the pool as an explicit argument, exactly
   as `metrics.evaluate(mask_X=…)` does:
   - `strategy_ladder.py::eval_budget` — **delete the asked-item mask and the `te[j,i]=0` target deletion.**
     This is the only path that deletes held-out targets and the only one whose ground truth varies by arm.
     Every number it has produced is arm-dependent and must be re-run before being cited.
   - `strategy_channel_suite.py::genv_eval` — replace `ctx.va_tr[ch] + ans[cj]` with the fixed rated pool
     for the val cohort. The `ans` term then becomes redundant, which is the point.
   - `concept_cold_check.py::score` / `bank_sweep.py` — replace `sc[r, e[6]] = -inf` with the user's full
     rated-item mask (available from the same `u["items"]` structure that builds the interview). This
     removes the item-vs-concept channel asymmetry the doc currently flags in its own protocol paragraph.
   - `run_battery_phaseA.py::gate_g9` — the Paper A gate. **Do not silently change a printed number.** Add
     policy B as a second column, keep the published A+C-answered column, and print both until the chapter
     is re-cut.
3. **Add a harness assertion**, in the same spirit as the existing `pool.multiply(te_te).nnz == 0` leak
   check: `assert mask is identical across arms within a budget`. A one-line invariant catches every future
   regression of this class. Add a second assertion that the target set (`te`) is never modified by an arm.
4. **State the rule in the Paper B protocol paragraph, verbatim**, together with the fact that Paper A's
   full-profile table is on policy A and its interview curve on A+C-answered. A number from one pool must
   never be tabulated next to a number from another (memory: `surprise-means-debug-the-harness`; the void
   "Golbandi 0.3894").
5. **Run the canary.** Push Most-Popular and EASE through any changed path first. Their input is unchanged,
   so any movement is the mask alone: expect a small near-uniform lift going A→B and no reordering.
6a. **This is already a blocking item.** `docs/design/PAPER_B_BASELINE_BATTERY_TODO.md` task 0.3 exists
   precisely because the greedy and battery tables sit on different masks (+0.027 at the cold state). This
   findings file supplies the ruling that task was waiting for: **the joint re-run goes on policy B.**

6. **Report the sensitivity once, as a robustness paragraph.** Re-run the concept-vs-item headline under
   both A+C-answered (current) and B, and print both. The prediction is that the concept margin widens under
   B; if it does not, we have learned something important before a reviewer does.

**One sentence for the paper, which is also the answer to the author's original unease:** *we never remove
an item from the candidate list because it was asked about; the pool is fixed per user from their rated
history before the interview begins, so no arm can shrink or enlarge its own haystack, and no method is
either rewarded or punished for asking well.*
