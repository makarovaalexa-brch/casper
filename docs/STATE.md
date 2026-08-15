# STATE — where the project is now

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
