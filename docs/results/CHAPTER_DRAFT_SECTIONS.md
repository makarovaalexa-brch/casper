# CHAPTER B — DRAFT SECTIONS (new and revised, week of 2026-08-11 → 2026-08-17)

**Status:** markdown draft only. **`new_chapters/chapterB_v2/chapterB_v2.tex` was NOT edited.**
Integration is a supervised step.

**How to read this file.** Each section is chapter prose written in the voice of the current
`chapterB_v2.tex` (bold lead sentences, intervals attached to every claim, live limitations kept
in-line). Beneath each section is a **provenance block**: every number, the absolute artifact path,
and the exact JSON field it was read from. Every number below was read out of its artifact for this
draft; the ones that did *not* reproduce are listed in **§0 — NUMBERS THAT DID NOT REPRODUCE** and
are **not used in any prose**.

Unless stated otherwise, every interval is a paired percentile bootstrap over per-user differences,
10,000 resamples, seed 98765 (`src/instrument/policy_net.py:630`, `paired_bootstrap`). **No
`1.96·√2·SE` construction was found anywhere in the artifacts behind this draft**; the repository's
only occurrences of `1.96` in `src/` are comments forbidding it. Full-NDCG comparisons carry
n = 10,000; tail comparisons carry n = 9,217 on validation and n = 9,281 on test, because users with
no tail target are dropped from the tail bootstrap.

---

## §0 — NUMBERS THAT DID NOT REPRODUCE (read this first)

These are the numbers in the brief, the design sheet or a results doc that could **not** be verified
against an artifact. **None of them appears in the prose below.**

| number | where it is asserted | what the artifact actually holds |
|---|---|---|
| **per-user oracle `+0.229`** | `docs/design/PAPER_B_POLICY_DESIGN.md` K.6 | No JSON field holds it. Recomputed from the raw label matrix `experiments/battery/pu_labels/pu_gains.f32`: mean per-user max full-gain is **0.19366** over all 1,000 pilot users, and **0.22918** over the 845 users with any strictly positive gain. `+0.229` is the second, and the restriction to 845 users is stated nowhere. **Draft uses 0.19366 over all 1,000 users.** |
| **best cohort-wide question `+0.0115`** | K.6 | Banked value is `gain_summary.best_candidate_mean_gain_full` = **0.009663939**. Rescaled to the 845-user subset it is 0.011437; re-argmaxed on those 845 it is 0.012665. **`+0.0115` matches no computable variant exactly.** Draft uses **+0.00966**, the banked field. |
| **trained cold arm `+0.0153` (null)** | K.6 | **Not locatable in any artifact.** The cold arm that exists (`pu_main_cold0.03.pt`) has banked behavioural cells in `pu_gates_coldspan.json` of +0.00422 (k=2, null), +0.00074 (k=4, null), +0.00771 (k=8, **separates**) — against the policy's own frozen order, not against the static pick. **Draft quotes the banked cells and never quotes +0.0153.** Note also `pu_gates_coldspan.json` has been overwritten across checkpoints and the 0.003-tau cold arm has no gates artifact at all. |
| "argmax over noisy estimates selects the over-estimates; answer rate 0.58 → 0.37" | the brief's framing of the value line | **Two unrelated results conflated, and the mechanism is wrong.** 0.5789 → 0.3746 is the *warm direct regressor walking off its E1 warm start* (`frac_never_deviates` = 0.0, 14.20 of 16 positions changed). The ceiling sweep's answer rate moves the *other* way — 0.6188 at ρ=0.50 down to 0.5650 at ρ=1.00, i.e. **falling as accuracy rises** — and its documented mechanism is *redundancy concentration*, not a winner's curse. The sweep's construction rules a winner's curse out: perturbed values are mapped back onto `sorted(t)`, so the multiset of values is preserved exactly and only the order is degraded. **Draft writes the two real mechanisms separately.** |
| "the value surface is rough across states (~+0.29 inter-state correlation)" | the brief | +0.2956 is **not** an inter-state correlation. It is the mean leave-one-state-out held-out Spearman of a *state-blind mean-vector predictor* over 187 evaluation states — a predictor score. It is also **log-only** (`experiments/battery/value_train.out:35`); the run was killed before writing JSON. The nearest JSON-backed figure is `value_dataset_meanvec.json → arms[0].mean_spearman` = **0.25645** on a different selection (49 complete states). **Draft uses +0.4163 (7 spine states) vs +0.25645 (49 complete states, JSON) and cites +0.2956 explicitly as log-only.** |
| `dumb@499` "0% out-of-pool … every one of which is inside the training pool" | `docs/results/VOCAB_GUARD.md` RESULT 3 | `arms.dumb_m499.out_of_pool_pick_rate` = 6.25e-06, `n_distinct_out_of_pool_questions` = 1. One question, one user. The rounded table cell is fine; the prose sentence is literally false. **Draft says "effectively zero".** |
| "paired over the same 10,000 test users" for the bank's **tail** contrast | `docs/results/CONDITIONED_ARMS_ON_TEST.md` §3 | `contrasts."combined_minus_items@K16".paired_ci_tail.n` = **9,281**. Full is 10,000. Wording defect, not a numerical one. **Draft states both n.** |
| `PHASE3_DEEPER.md` | — | The doc's `## RESULTS — PENDING` block is still an unchecked checklist; **it contains no results**. Every Phase-3 number in §4 below was read directly out of the artifacts, which are dated 2026-08-14/15. The doc is stale relative to its own run. |
| `VOLUNTEERED_COHORTS.md` | — | RESULT 1/2/3 and WHAT THIS SETTLES are literally `PENDING`. The run has since finished and `experiments/battery/volcohorts/volcohorts_report_K15.json` holds numbers, but **no prose is drafted for it** — see §6. |
| `E1_POLICY_IMITATION.md` §0.5 "31.3 minutes wall"; `experiments/battery/policy_traj/step0_probe.json` | E1 doc | The log says `all 15 chunks banked in 17.8m`; the named `step0_probe.json` **does not exist on disk**. Cosmetic — no drafted number depends on either. |
| E1 §0.1 "root's three children sit at 0.374 / 0.320 / 0.087" | E1 doc | Not sourceable as stated. **The chapter already uses these three numbers** (`chapterB_v2.tex`, "Reach, which conditioning does not touch") as the *baselines of the root's did-not-answer branch's three children*, which is a different object from the root's own children (`preflight.best_balanced.concept.sizes` = 2606/4814/2580, shares .261/.481/.258). See §7. |

**One more, structural rather than numerical.** Experiment **Q reports no confidence intervals at
all.** `heldout_questions.json` and `heldout_questions_strippop.json` contain no `lo`/`hi`/
`excludes_zero` anywhere. 0.98167 vs 0.96285, the 69.6% and the 91.6% are uncertainty-free point
estimates over 620 million *highly non-independent* cells (10,000 users × 16 correlated positions ×
3,880 questions), so the "N is huge" defence does not hold. A user-clustered or question-clustered
bootstrap is the missing analysis. Q is the chapter's one positive conditioning result, so this is
the single largest statistical gap in the new material. **The draft prose below states it as a live
limitation rather than omitting it.**

---

## §1 — THE POLICY: distillation is exact, and it is the optimum available

> Intended placement: a new section after Act 3 (`sec:act3`), before Act 4. The tree of Act 3 is its
> premise; this section is what the tree turns out to be *for*.

### 1.1 The tree is an existence proof; the question is whether it is a method

Act 3's tree is a lookup: twenty-two nodes, a fixed splitter at each, a greedy tail inside each of
fifteen leaves. It answers the chapter's question — put the reported metric in the split and the sign
of the conditioning gain flips — and it answers nothing about whether that behaviour can be carried
by a model rather than a table. A table cannot score a question outside its pool, cannot resume from
a state it did not itself create, and cannot say what it thinks of a question it has never been
asked. Each of those is a property a deployed interviewer needs and the tree structurally lacks.

So we distil the tree into a parametric policy: a permutation-equivariant scorer over questions,
$526{,}542$ parameters, reading the user's current belief state and the asked-and-answered history,
trained by cross-entropy against the tree's own next question at $2{,}252{,}288$ banked
$(\text{state}, \text{question})$ pairs from $140{,}768$ training users. The tree's label is a
function of the policy's input, so the object being fitted exists.

### 1.2 The replica is exact

**On $10{,}000$ held-out validation users the policy reproduces the tree's choice at every position
of every interview.** Top-1 agreement is $1.0000$, and $1.0000$ separately at each of the sixteen
positions; not one user in ten thousand ever deviates. The behavioural consequence is that the two
arms are the same arm: the policy's gain over the static comparator is $+0.02794$
$[+0.02493, +0.03099]$ full and $+0.01055$ $[+0.00749, +0.01373]$ tail, against the tree's banked
$+0.027939556$ — the two agree to $9.5\times10^{-10}$, the residue of float32 banked vectors scored
against float64 fresh ones. Both arms realise the same $15$ distinct interviews, the same
$0.66370$ pairwise divergence rate over $68{,}571{,}148$ user pairs, and the same $54.57\%$ concept
share of asked questions.

The controls say this is imitation and not an artefact of the harness. Retraining on **shuffled
labels** collapses top-1 to $0.2356$ and turns the gain negative, $-0.01489$
$[-0.01803, -0.01172]$; its position-1 agreement of $0.4814$ is exactly the share of the cohort in
the root's dislike branch, which is what a policy that has learned the marginal and nothing else must
score. Removing the asked-history from the input and leaving only the belief state costs almost
nothing on fidelity — top-1 $0.9951$, gain $+0.02723$ $[+0.02424, +0.03031]$ — but the resulting
policy deviates from the tree for $6.19\%$ of users, and, as §3.1 reports, its deviations cost.

**What exactness buys is a licence to ask the questions the table cannot be asked.** Sections §3
and §5 spend it.

### 1.3 The one thing the replica inherits is the tree's vocabulary

Released onto the full $19{,}390$-question inference catalogue rather than the $1{,}000$-candidate
training pool, the policy picks a never-trained question $29.99\%$ of the time, drawing on $189$
distinct out-of-pool questions, and $82.31\%$ of users receive at least one. It costs $-0.02034$
$[-0.02236, -0.01837]$. The mechanism is answerability, not relevance: the interview's answer rate
falls from $0.5789$ to $0.4139$, because in-pool picks are answered $55.2\%$ of the time and
out-of-pool picks $9.1\%$. **The policy generalises its ranking to unseen questions and does not
generalise the constraint that a question must be answerable to be worth asking**, which is the
subject of §3.3.

### 1.4 Why the tree is the right teacher — and why the per-user optimum is not

The obvious objection to distillation is that it caps the student at the teacher. If we can measure,
for each user, which single question would most improve *that user's* held-out ranking, why not train
on that instead?

**Because the two labels are different kinds of object.** The tree's next question is computed from
the asked-and-answered history alone: it is a function of the policy's input, and a function that
exists can be learned — §1.2 learned it exactly. The per-user optimal question is computed by scoring
candidates against **that user's held-out targets**. It is not a function of the policy's input.
Two users with identical interview histories have different best questions precisely because their
hidden tastes differ, and no quantity of training data supplies a mapping that is not there.

That is not an argument from principle only; it is measured. Over $1{,}000$ pilot users, each at a
sampled interview state, the **per-user oracle** — the best single question for that user, chosen
using their held-out targets — is worth $+0.19366$ in full NDCG@10 over the user's own current state.
The **best single cohort-wide question**, chosen to maximise the mean gain over the same users, is
worth $+0.00966$. **The cohort-wide choice captures $5.0\%$ of the per-user prize.** The remaining
$95\%$ is carried by information the interview has not yet revealed — which is not a modelling
failure but the reason interviews exist.

And what "enough examples of state and optimal question" converges to, in the limit, is the best
state-conditioned answer: the question that is best *on average* for users at that state. That object
is the greedy criterion, and at coarse states it is the tree. **The per-user route, taken to its
limit, converges back to the thing distillation already reproduces exactly.**

The empirical attempt is consistent with this and adds one caution of its own. A policy trained on
soft per-user targets is a null against its own frozen order at two and four questions
($+0.00422$ $[-0.00080, +0.00917]$; $+0.00074$ $[-0.00443, +0.00608]$) and separates in its favour at
eight ($+0.00771$ $[+0.00225, +0.01302]$) — but it fails the spanning requirement badly, scoring
$0.22483$ against the distilled policy's $0.29280$ on cold states, a difference of $-0.06798$
$[-0.07659, -0.05948]$. **It buys a small, inconsistent gain at one budget by giving up the
behaviour the distilled policy has exactly.** We report the arm; we do not build on it.

**Distillation is not the fallback. It is the optimum given the information available.**

> **Live limitations for this section.** (i) The per-user comparison is a validation-cohort pilot of
> $1{,}000$ users at one sampled state each, not a $K{=}16$ interview, and the oracle is a
> clairvoyant quantity by construction — it is a ceiling, never a score. (ii) The distilled policy is
> selected on the same $10{,}000$ validation users it is scored on here; its held-out claim is the
> tree's test number, which it reproduces by construction rather than the other way round.

**Provenance — §1**

| number | artifact | field |
|---|---|---|
| 526,542 parameters | `C:\dev\phd\casper\experiments\battery\policy_main.log` line 30 | log line `arch=equivariant state=both pos=True params=526,542` (**log only, no JSON**) |
| 2,252,288 pairs; 140,768 train users | `C:\dev\phd\casper\experiments\battery\policy_traj\manifest.json` | `Σ done.*.stats.n_users` = 140,768; × 16 |
| top-1 = 1.0000 | `C:\dev\phd\casper\experiments\battery\policy_main_equivariant_both_gates.json` | `gates.G-E1a_fidelity.top1` = 1.0 |
| 1.0000 at all 16 positions | same | `gates.G-E1a_fidelity.by_position` = `[1.0]×16` |
| never deviates | same | `gates.covariate_shift.frac_never_deviates` = 1.0; `deviation_position_histogram` = `[0]×16 + [10000]` |
| +0.02794 [+0.02493, +0.03099] | same | `gates.G-E1b_gain_vs_static.full.{mean,lo,hi,n}` = 0.02793955711893247 / 0.024929262440067067 / 0.03098967446070921 / 10000 |
| +0.01055 [+0.00749, +0.01373] | same | `gates.G-E1b_gain_vs_static.tail.{mean,lo,hi,n}` = 0.010548198280223806 / 0.007486113262661875 / 0.013727399707256175 / **9217** |
| tree's banked gain 0.027939556; agreement 9.5e-10 | same | `gates.G-E1b_gain_vs_static.tree_val_gain_full` = 0.027939556166529655 (difference 9.524e-10; **a description, not a gate — no 1e-9 tolerance exists in the artifact**) |
| 15 distinct interviews | same | `gates.G-E1c_anti_collapse.n_distinct_interviews` = 15 |
| divergence 0.66370, 68,571,148 pairs | same | `gates.G-E1c_anti_collapse.pep_divergence_policy.divergence_rate` = `…pep_divergence_tree.divergence_rate` = 0.6636982656320702; `.n_pairs` = 68571148 |
| concept share 54.57% | same | `gates.channel_mix.policy_concept_share` = `.tree_concept_share` = 0.54570625 |
| shuffle top-1 0.2356; −0.01489 [−0.01803, −0.01172]; position-1 0.4814 | `C:\dev\phd\casper\experiments\battery\policy_shuffle_equivariant_both_gates.json` | `gates.G-E1a_fidelity.top1` = 0.23561875; `gates.G-E1b_gain_vs_static.full.{mean,lo,hi}` = −0.014889058591762424 / −0.018030121411188066 / −0.011721256071975654; `gates.G-E1a_fidelity.by_position[1]` = 0.4814 |
| z-only: 0.9951, +0.02723 [+0.02424, +0.03031], 6.19% deviate | `C:\dev\phd\casper\experiments\battery\policy_main_equivariant_z_gates.json` | `gates.G-E1a_fidelity.top1` = 0.995125; `gates.G-E1b_gain_vs_static.full.{mean,lo,hi}` = 0.02723202512709504 / 0.02424180631762067 / 0.030307207778476335; `gates.covariate_shift.frac_never_deviates` = 0.9381 |
| 19,390 candidates; 29.99%; 189; 82.31% | `C:\dev\phd\casper\experiments\battery\policy_scalability.json` | `n_inference_catalogue` = 19390; `out_of_pool_pick_rate` = 0.299925; `n_distinct_out_of_pool_questions` = 189; `users_with_any_out_of_pool_pick` = 0.8231 |
| −0.02034 [−0.02236, −0.01837] | same | `paired_full_catalogue_minus_1000.{mean,lo,hi,n}` = −0.02034050445662675 / −0.02236287169390005 / −0.018371441835480597 / 10000 |
| 0.5789 → 0.4139; in-pool 0.5523, out-of-pool 0.0909 | same | `answer_rate_1000_pool` = 0.57889375; `answer_rate_full_catalogue` = 0.41394375; `answer_rate_of_in_pool_picks` = 0.5523336785344427; `answer_rate_of_out_of_pool_picks` = 0.09091856297407686 |
| per-user oracle +0.19366 (1,000 users) | `C:\dev\phd\casper\experiments\battery\pu_labels\pu_gains.f32` (+ `pu_labels.json` `contract`, `base_full` = 0.2144452898323591) | **derived from the raw label matrix — no JSON field.** Mean over users of the per-user max full-gain = 0.19366 (all 1,000); 0.22918 restricted to the 845 users with any positive gain |
| best cohort-wide question +0.00966 | `C:\dev\phd\casper\experiments\battery\pu_labels\pu_labels.json` | `gain_summary.best_candidate_mean_gain_full` = 0.009663939 |
| 5.0% ratio | derived | 0.009663939 / 0.19366 = 0.0499 |
| per-user arm k=2/4/8 cells | `C:\dev\phd\casper\experiments\battery\pu_labels\pu_gates_coldspan.json` | `gates."G-K-behavioural".by_k.2` = {mean 0.004216016469022044, lo −0.0008003930873317043, hi 0.00917129180368464, n 2000, excludes_zero false}; `.4` = {0.0007420599021290169, −0.004428303444190394, 0.006078225981373024, false}; `.8` = {0.007713639239831721, 0.0022528990253146306, 0.013024035847797832, **true**} |
| spanning failure 0.22483 vs 0.29280, −0.06798 [−0.07659, −0.05948] | same | `gates."G-K-spanning"` (`pass` = false) |
| pilot contract: 1,000 users, validation, test untouched | `pu_labels.json` | `contract.split` = "validation", `contract.n_users` = 1000, `test_touched` = false, `contract.mixture_weights` = tree 1/3, vol 1/3, rand 1/3 |

---

## §2 — THE VALUE LINE: five ways to score a question by what it is worth, and five losses

> Intended placement: immediately after §1. It is the negative that makes §1's "distillation is the
> optimum available" a measured claim rather than a preference.

### 2.1 The design, and the bar it has to clear

A distilled policy inherits its teacher's question set. The obvious way off that leash is to stop
imitating a choice and start estimating a quantity: score each candidate by the expected value of
asking it, $\sum_o P(o \mid q, s)\,V(o \mid q, s)$, and take the argmax. The outcome head $P$ is
independently good (§3.2). Everything below concerns $V$.

The bar is not the tree. **It is the bank**: on the same validation cohort, at the same budget, with
no conditioning of any kind, replacing the item-only static order with the item-plus-concept static
order is worth $+0.01280$ $[+0.00972, +0.01580]$. Any value-driven policy that does not clear that
is not worth its parameters.

### 2.2 Five arms, five losses

All five are scored at $K{=}16$ on the same $10{,}000$ validation users, against the same combined
static order (full NDCG@10 $0.25561$), paired.

| arm | full@10 | gain vs the static bank |
|---|---|---|
| flat value table (mean of banked gain vectors) $\times$ outcome head | $0.22650$ | $-0.02911$ $[-0.03231, -0.02596]$ |
| learned two-head, $V = \text{base} + \delta$ | $0.19721$ | $-0.05840$ $[-0.06180, -0.05495]$ |
| the same flat table at perfect rank fidelity ($\rho = 1.00$) | $0.22650$ | $-0.02911$ |
| nearest-state conditional lookup (oracle) $\times$ outcome head | $0.23583$ | $-0.01978$ $[-0.02274, -0.01684]$ |
| direct gain regression, warm-started from the distilled policy | $0.22002$ | $-0.03559$ $[-0.03918, -0.03197]$ |
| *the bank change, for reference* | | $\mathbf{+0.01280}$ $[+0.00972, +0.01580]$ |

**Every arm loses, and the best of them loses by more than twice what the bank change wins.** Three
mechanisms account for it, and they are not the same mechanism.

### 2.3 Mechanism one: a marginal value cannot express redundancy

The natural response to the first two rows is that the estimator is not good enough. It is not the
estimator. We degraded the value vector in normal-score space and mapped it back onto its own sorted
values — so the simulated estimator holds **exactly the same multiset of values** and is wrong only
about the order, by a controlled amount — and swept the realised rank correlation from $0.50$ to
$1.00$, three noise draws per level, the full policy rerun on all $10{,}000$ users at each.

**Across nineteen runs the policy's gain moves within a total spread of $0.0075$ — smaller than the
seed-to-seed spread inside a single accuracy level — and the correlation between estimator accuracy
and policy score is $-0.187$.** Doubling the estimator's rank correlation buys nothing, and the
undegraded table is the worst point of the sweep.

The reason is legible in the arm's own behaviour. A marginal value is what a question is worth *on
its own* at a state; the policy takes the top sixteen by that score, so it will happily ask sixteen
near-duplicates of one highly-valued question. The greedy static comparator does the one thing this
design structurally cannot: pick, fold, re-score everything, pick again. **As the estimator gets more
accurate the interview's answer rate falls monotonically, $0.6188$ at $\rho = 0.50$ to $0.5650$ at
$\rho = 1.00$** — greater accuracy concentrates the picks, and the noise was helping only by
scattering them.

**This licenses one claim and not another.** The sweep degraded a single value vector reused at every
state, so every row of it — $\rho = 1.00$ included — is a *state-marginal* value. What is established
is that **a state-marginal value fails at any accuracy**. What is *not* established is that a
state-conditioned value fails, and the evidence runs the other way: replacing the flat vector with a
crude $245$-way lookup on the nearest banked state is worth $+0.00933$ $[+0.00659, +0.01216]$
paired — more than twice what the outcome head itself contributes — and moves the policy's answer
rate from $0.5650$ to $0.6448$ and its concept share from $0.3931$ to $0.5484$, both toward the
tree's $0.5789$ and $0.5457$. **That lookup is an oracle**: the gains it reads were measured on the
same users it scores, so it is a ceiling probe showing the conditioned line is live, and never a
deployable number. It is also still $0.0326$ short of the bank.

### 2.4 Mechanism two: substituting a regressor for a policy destroys the policy

The fifth arm was warm-started from the distilled policy — sixteen of sixteen tensors identical at
initialisation, maximum absolute difference $0.00\mathrm{e}{+}00$, verified rather than asserted —
and then fitted to measured gains at seven states.

**It moved off the exact replica for every single user, at $14.20$ of $16$ positions on average, and
downward by $-0.06353$ $[-0.06697, -0.06011]$.** The interview's answer rate collapsed from
$0.5789$ to $0.3746$. Three seeds land within $0.0008$ of each other. The regressor is not badly
behaved at the seven states it was fitted on — at three of them it picks the single best available
question — but a policy visits thousands of states, and at every state off that set the fit is on
its own.

**The lesson is narrow and it is a design constraint, not a defect of the fit:** on this label
budget, a value model must be *added* to a working policy as a correction, never *substituted* for
it. The arm was also fitted on the cohort it is scored on, which only strengthens the negative — a
fit with every advantage still ended $0.064$ below the policy it started from.

### 2.5 Mechanism three: action-learning needs one bit per state; value-learning needs the surface

The two failures compose, and the composition is the point. The value estimator we could *build*
well is state-blind, and a state-blind estimator is structurally incapable of the one thing that
matters (§2.3). The value estimator we could *learn* is state-conditioned and was weak: a
parameter-free mean of six banked gain vectors beats the learned lift head at rank fidelity,
$+0.4163$ against $+0.0623$, on the seven shallow states where both were scored.

And that parameter-free mean does not travel. Off those seven same-spine states — on deeper,
smaller, off-spine states — the same state-blind estimator degrades to $+0.2564$ over $49$ complete
states, which is what a state-blind estimator should do when the surface it is flattening actually
varies with the state. **So the value line is caught between a surface that varies enough to matter
and an estimator budget that cannot resolve it.**

Distillation does not face that problem, and the asymmetry is the reason. **To imitate an action a
policy needs to learn one discrete choice per state; to imitate a value it needs the whole surface
over a thousand candidates at every state it will ever visit.** The first is what our label budget
buys exactly. The second it does not buy at all.

> **Live limitations for this section.** (i) None of the five arms was measured on test; the value
> line is a validation-cohort result throughout, and no test user was built or loaded. (ii) The
> $+0.4163$ / $+0.2564$ contrast compares two *different selections of states*, not a single
> estimator on a single cohort, and should be read as a direction rather than a magnitude. (iii)
> Every measured gain we hold is for a question in the top-500-popular $\cup$ top-500-answerable
> pool: no model here has seen what a *typical* catalogue question is worth, which for open
> vocabulary is plausibly a larger problem than the state count.

**Provenance — §2**

| number | artifact | field |
|---|---|---|
| bank-alone bar +0.01280 [+0.00972, +0.01580] (validation) | `C:\dev\phd\casper\experiments\battery\meanvec_value.json` | `bank_alone_gain_full` = 0.012802034569302992, with its `{lo,hi}` = 0.009718…, 0.015798… (same value in `policy_deeper_tree_control.json` `bank_alone_val.combined_minus_items_full` = 0.012802034569302928) |
| static combined full 0.25561 | same, and `greedy_static_policyB_combined.json` | `static_combined.full` = 0.25561016113602253; `curve_policyB.16.full@10` identical |
| flat table 0.22650, −0.02911 [−0.03231, −0.02596] | `C:\dev\phd\casper\experiments\battery\meanvec_value.json` | `arms.meanvec_PxV.full` = 0.22649800822270308; `arms.meanvec_PxV.gain_vs_static_combined_full.{mean,lo,hi}` = −0.02911215291331945 / −0.0323063609778107 / −0.02595788712691123 |
| learned two-head 0.19721, −0.05840 [−0.06180, −0.05495] | `C:\dev\phd\casper\experiments\battery\lift_head_stable.json` | `policy.pool1000.full` = 0.19720727080593728; `policy.pool1000.gain_vs_static_combined_full.{mean,lo,hi}` = −0.05840289033008527 / −0.06180344504491106 / −0.054949219095765556; arm identified by `v_offset` = true |
| ρ=1.00 row = −0.02911 | `meanvec_value.json` | `ceiling[18]` (`rho_target` = 1.0, `rho_realised` = 1.0) is **field-for-field identical to `arms.meanvec_PxV`** — same run, recorded twice, not an independent measurement. `ceiling_note` states ρ=1.0 is the best a *static* estimator can be, not the true per-state optimum. **Prose therefore says "the same flat table at perfect rank fidelity", never "perfect measured values".** |
| 19 runs, spread 0.0075, corr −0.187 | same | recomputed over `ceiling[]` (19 entries): min −0.029242396979642524, max −0.02171878268686878, spread 0.0075236, corr(ρ, gain) = −0.18668 |
| answer rate 0.6188 → 0.5650 | same | mean of `ceiling[k].answer_rate` at `rho_target` = 0.5 over three seeds = 0.618775; `ceiling[18].answer_rate` = 0.56495625 |
| nearest-state 0.23583, −0.01978 [−0.02274, −0.01684] | `C:\dev\phd\casper\experiments\battery\meanvec_value_cond.json` | `conditional_nearest_state.full` = 0.23583125067253163; `.gain_vs_static_combined_full.{mean,lo,hi}` = −0.01977891046349091 / −0.022744472993332245 / −0.016841753745449717 |
| +0.00933 [+0.00659, +0.01216] | same | `conditional_nearest_state.vs_flat_mean_full.{mean,lo,hi}` = 0.009333242449828541 / 0.006589212901631943 / 0.012159998707742004 |
| answer rate 0.5650 → 0.6448; concept share 0.3931 → 0.5484 | `meanvec_value.json` / `meanvec_value_cond.json` | `arms.meanvec_PxV.answer_rate` = 0.56495625; `conditional_nearest_state.answer_rate` = 0.64483125; concept shares in the same blocks |
| outcome head's own contribution +0.00392 [+0.00105, +0.00676] | `meanvec_value.json` | paired block, `mean` = 0.0039192701994113545, `lo` = 0.0010524755279456394, `hi` = 0.006763… |
| warm regression 0.22002, −0.03559 [−0.03918, −0.03197], vs E1 −0.06353 [−0.06697, −0.06011] | `C:\dev\phd\casper\experiments\battery\direct_gain_policy.json` | `arms[0].full` = 0.2200153809715377; `arms[0].gain_vs_static_combined_full.{mean,lo,hi}` = −0.03559478016448486 / −0.03918082228103214 / −0.031967530313693; `arms[0].vs_E1_full` = −0.06353433728341734 (`arms[0].seed` = 98765; seeds 99765/100765 at −0.03620 / −0.03639) |
| deviates for every user, 14.20 of 16, answer rate 0.3746 | same | `arms[0].frac_never_deviates` = 0.0; `mean_positions_differing_from_tree` = 14.1951; `arms[0].answer_rate` = 0.3745625 |
| tree answer rate 0.5789 | `C:\dev\phd\casper\experiments\battery\g_main.log` line 30 (**log only**); corroborated by `policy_scalability.json` `answer_rate_1000_pool` = 0.57889375 | — |
| ρ +0.4163 vs +0.0623 (7 states) | `lift_head_stable.json` | LOSO Spearman blocks; in-sample values e.g. `in_sample_spearman[].spearman` = 0.16652836652836653 |
| +0.2564 over 49 complete states | `C:\dev\phd\casper\experiments\battery\value_dataset_meanvec.json` | `arms[0].arm` = "unbiased_only(complete states)", `arms[0].mean_spearman` = 0.25645269895815304, `n_eval_states` = 49 |
| (**not used**) +0.2956 over 187 states | `C:\dev\phd\casper\experiments\battery\value_train.out` line 35 | **stdout only; the run was killed before writing JSON.** Cited in the draft only as a log-provenance caveat |
| warm start bit-identical at init | `direct_gain_policy.json` | init check, max abs diff 0.0 over 16/16 tensors |

---

## §3 — CAPABILITIES: what a model can do that a table cannot

> Intended placement: after §2. Three experiments, one positive.

### 3.1 R — resumption: the capability is structural, not behavioural

A table cannot be started from a state it did not build. A model can. So each of $10{,}000$
validation users volunteers $k \in \{2,4,8\}$ items before the interview begins; the items are folded
as ordinary evidence, and each arm then conducts its full sixteen-question interview from that seeded
state.

**The distilled policy does not use the evidence.** At $k = 2, 4, 8$ the share of users whose
interview changes *at all* is $1.01\%$, $2.48\%$ and $5.11\%$, and the mean number of questions
changed is $0.13$, $0.30$ and $0.60$ of sixteen. For ninety-five to ninety-nine per cent of users the
policy conducts exactly the interview it would have conducted had the person volunteered nothing —
which is exactly the interview the tree conducts. It does not lose for it either: policy minus tree
is $+0.00002$ $[-0.00019, +0.00023]$, $-0.00031$ $[-0.00067, +0.00004]$ and $-0.00026$
$[-0.00073, +0.00020]$, all three straddling zero.

The mechanism is visible in a count that needs no model at all. Asking a user about an item they have
already volunteered is a wasted turn, and at $k = 8$ the re-ask rate per item question is $5.4\%$ for
the policy against $5.5\%$ for the tree and $5.2\%$ for the fixed order — **parity with two arms that
cannot see the evidence at all.** The policy asks $7.24$ item questions of sixteen, the tree $7.27$,
the static order $6.0$.

**The honest sentence is that one-hot distillation of a tree teaches the policy the tree's function
and nothing outside it, including how to use evidence the tree cannot see.** "Supports new states"
is available as an architectural property — the state is an input, and the wiring is demonstrably
live on the one to five per cent of users who do move — and never as a demonstrated behaviour. Where
the policy does change its mind it is not better for it: among deviators the paired difference is
$+0.00241$ ($n = 101$) at $k = 2$, $-0.01232$ ($n = 248$) at $k = 4$ and $-0.00500$ ($n = 511$) at
$k = 8$. Two of three are losses and the largest effect is a loss.

The belief-only ablation of §1.2 completes the picture, and it is the falsifier firing rather than a
second null. Removing the asked-history from the input forces the policy to read the seeded state
through the latent, and it does adapt — $6.19\%$ of users deviate — and its deviations cost at every
$k$. **Adaptation to volunteered evidence is available in this architecture and, as distilled, it is
not worth having.**

> **Live limitation.** Under the weak-method rule this is a result about *one-hot distillation from a
> tree teacher*, not about the architecture. A fine-tune on evidence-seeded states is designed and
> its labels already exist; it was not run.

### 3.2 Q — held-out questions: the one place conditioning demonstrably wins

The capability a table most conspicuously lacks is an opinion about a question nobody has asked it.
We split the question vocabulary, not the users: $3{,}880$ questions ($3{,}670$ items, $210$
concepts) are held out from training entirely and $15{,}510$ are kept, stratified so that each
popularity decile contributes exactly $367$ items and $21$ concepts. The reaction head then predicts
a person's three-way reaction — like, dislike, could-not-answer — to a question **no gradient ever
touched**, from that question's frozen descriptor and the person's interview state, over
$620{,}606{,}659$ scored $(\text{state}, \text{question}, \text{outcome})$ cells with nothing
subsampled.

The comparator must be constructed rather than borrowed, because a held-out question has no
per-question row. It backs off to the mean outcome distribution of the question's
(channel, popularity-decile) stratum, computed on training questions and fitting users only. That
backoff is state-free by construction, and it is the bar.

**The head scores $0.98167$ macro AUC against the backoff's $0.96285$.** Two facts have to be
conceded before that margin can be read as knowledge of a question. First, the descriptor literally
hands a concept its training answer rate and an item its $\log(1{+}\text{rating count})$, while the
backoff sees popularity only as a decile — so the head holds strictly more of the same information.
Second, the class distribution is dominated by *could-not-answer* at $96.6\%$ of cells, so aggregate
AUC is largely a statement about answerability.

**Both concessions are survivable, and the decomposition is the claim.** Zeroing the two
answerability columns and refitting the standardisation costs almost nothing: AUC $0.98008$, which
is $91.6\%$ of the margin retained. And the shape of the curve cannot be produced by descriptor
reading at all. The backoff is flat at $0.9629$ across interview positions by construction; a
label-shuffle control is flat at $0.967$; **the head climbs from $0.9676$ at position zero to
$0.9851$ at position fifteen, and $+0.0172$ of that $+0.0175$ climb survives the ablation.** At
position zero, where there is no state to read, the head's advantage over the backoff is only
$+0.0047$ with the full descriptor and $+0.0033$ without the two columns — a third of the static
part *was* the conceded columns, and the interview-accrued part is untouched.

The claim is therefore narrow and it is real: **given a question it has never seen, the head predicts
who will be able to answer it and what they will say when they can, and its prediction improves as it
hears the person's answers to other questions.** That improvement is not the descriptor's popularity
features and does not survive label permutation. It is not, and is nowhere claimed to be, taste
prediction on unseen films.

It is also sample-efficient in the axis that matters. Training on a nested stratified tenth of the
question vocabulary — $1{,}551$ questions — already buys $69.6\%$ of the full margin; a quarter buys
$92.8\%$ and a half $97.3\%$.

**This is the single place in the chapter where conditioning on the interview state demonstrably
wins.** It wins at *prediction*, not at question *selection*: §2 is the record of what happens when
a comparably good predictor is put in charge of choosing.

> **Live limitations, and they are load-bearing here.** (i) **No confidence interval exists for any
> number in this experiment.** The margins are point estimates over cells that are massively
> non-independent across users, positions and questions; a user- or question-clustered bootstrap is
> owed and has not been run. (ii) The evaluation cohort is $10{,}000$ users held out from the *head*,
> drawn from the training population; the tower that produces the state latent was fitted on them.
> The user axis is held out from the predictor and not from the representation, which is defensible
> only because no NDCG is measured anywhere in this experiment — an observed answer outcome is not a
> model output — and it is stated rather than left implicit. (iii) The held-out questions' answer
> rates are extremely bimodal (quantiles from $7.6\times10^{-5}$ to $0.99998$), so a macro AUC over
> them is an average across two nearly disjoint regimes.

### 3.3 G — the guard: a state-free table beats the state-conditioned one

§1.3 left an open cost: released onto the full catalogue the policy loses $-0.02034$
$[-0.02236, -0.01837]$, because it picks questions almost nobody can answer. The capability claim
would be that the model knows which unseen questions *not* to ask — a state-conditioned guard using
the head's own predicted probability of getting an answer, refusing to ask below a threshold.

**It works, and a table with no state in it works better.** The best learned guard recovers $57.0\%$
of the open-vocabulary deficit, $+0.01159$ $[+0.00976, +0.01345]$. A parameter-free popularity cutoff
— keep the $1{,}996$ most-answerable questions, consult nothing else — recovers $78.1\%$,
$+0.01588$ $[+0.01401, +0.01776]$. Head to head at matched budget the learned guard **loses** by
$-0.00429$ $[-0.00545, -0.00312]$.

The guard is not doing nothing. It cuts out-of-pool picks to $13.0\%$ and nearly triples their answer
rate, $9.1\%$ to $25.7\%$. It simply costs more than it recovers, because the questions it is willing
to release ($279$ distinct never-trained questions, against the cutoff's $54$) are on average worse
than the ones the flat cutoff would have kept. **And neither closes the gap**: the better of the two
is still $-0.00446$ $[-0.00550, -0.00343]$ below the policy restricted to its training pool. Driving
out-of-pool picks to effectively zero by shrinking the cutoff to $499$ questions is worse still,
$0.056$ below the pool ceiling — the deficit is not repaired by refusing to leave the pool.

**Provenance — §3**

| number | artifact | field |
|---|---|---|
| 1.01 / 2.48 / 5.11% changed | `C:\dev\phd\casper\experiments\battery\resume\policy_resume.json` | `by_k.{2,4,8}.adaptation.frac_never_deviates_from_its_own_unseeded_rollout` = 0.9899 / 0.9752 / 0.9489 |
| policy − tree nulls | same | `by_k.{2,4,8}.comparisons.policy_minus_tree.full.{mean,lo,hi}` = 2.4322953219138488e-05 / −0.0001937232127081164 / 0.00022969491860658065; −0.00030562690120903955 / −0.0006660321262378592 / 4.406094517569549e-05; −0.0002556259793152266 / −0.0007267675625183244 / 0.0002031902329666236 |
| redundancy 5.4 / 5.5 / 5.2% (**k=8**) | same | `by_k.8.redundancy_per_item_question.{policy,tree,static}.reasks_per_item_question` = 0.053548244523631945 / 0.05459022933949675 / 0.051949999999999996 |
| 7.24 / 7.27 / 6.0 item questions (**k=8**) | same | `by_k.8.redundancy_per_item_question.{policy,tree,static}.mean_item_questions_of_16` = 7.2402 / 7.2687 / 6.0 |
| deviators +0.00241 (101) / −0.01232 (248) / −0.00500 (511) | same | `by_k.{2,4,8}.do_the_deviations_pay.policy_vs_tree.{mean_among_deviators,n_deviators}` |
| z-only deviates 6.19% and costs | `policy_main_equivariant_z_gates.json` + RESUMPTION addendum | `gates.covariate_shift.frac_never_deviates` = 0.9381 |
| Q head 0.98167 | `C:\dev\phd\casper\experiments\battery\heldout_q\heldout_questions.json` | `runs[3].heldout_question_results.head.auc_macro` = 0.981666516056691 |
| backoff 0.96285 | same | `runs[3].heldout_question_results.backoff.auc_macro` = 0.962850815110623 |
| shuffle 0.96682 | same | `runs[4].heldout_question_results.head.auc_macro` = 0.9668220151168376 (`runs[4].arm` = "shuffle") |
| 10 / 25 / 50% → 0.97595 / 0.98032 / 0.98116 | same | `runs[{0,1,2}].heldout_question_results.head.auc_macro` |
| 69.6 / 92.8 / 97.3% of margin | derived (**no stored field**) | (0.9759511279672299 − 0.962850815110623) / (0.981666516056691 − 0.962850815110623) = 0.69625 |
| ablation 0.98008 = 91.6% | `C:\dev\phd\casper\experiments\battery\heldout_q\heldout_questions_strippop.json` | `runs[0].heldout_question_results.head.auc_macro` = 0.9800798632192481; margin share derived = 0.91553. `strip_pop.zeroed_columns` = [200, 201] |
| position curve 0.9676 → 0.9851; stripped 0.9662 → 0.9834; backoff/shuffle flat | both files | `runs[3].heldout_question_results.head.auc_by_position[0]` = 0.9675507511379489, `[15]` = 0.985103145388983; stripped `[0]` = 0.9662045759425674, `[15]` = 0.9833992803911137 |
| 620,606,659 cells | `heldout_questions.json` | `runs[*].heldout_question_results.*.n_scored` = 620606659.0 |
| 3,880 / 15,510 / 3,670 / 210 | same | `split_composition.{n_heldout_questions,n_train_questions,heldout_n_items,heldout_n_concepts}` |
| answer-rate quantiles 7.6e-05 … 0.99998 | same | `split_composition.heldout_answer_rate_quantiles` |
| Q has no CIs; no NDCG measured; train-user split | same | no `lo`/`hi`/`excludes_zero` anywhere in the file; `ndcg_measured` = false; `n_fit_users` = 130768, `n_heldout_users` = 10000 |
| open-vocab deficit −0.02034 | `C:\dev\phd\casper\experiments\battery\vocab_guard\vocab_guard.json` | `arms.open_unguarded.vs_pool_ceiling.full.{mean,lo,hi}` |
| guard +0.01159 [+0.00976, +0.01345] = 57.0% | same | `arms."guard_tau0.98".vs_open_unguarded.full.{mean,lo,hi}` = 0.0115946431251847 / 0.009756042783939758 / 0.013449890721867778; **the 57.0% is derived** (ratio to `banked.E1_open_deficit_full`), no `recovered` field exists |
| table +0.01588 [+0.01401, +0.01776] = 78.1% | same | `arms.dumb_m1996.vs_open_unguarded.full.{mean,lo,hi}` = 0.01588166149464348 / 0.01400933489517865 / 0.01776435275402121; 78.1% derived |
| head-to-head −0.00429 [−0.00545, −0.00312] | same | `learned_minus_dumb_at_matched_budget."guard_tau0.98_vs_dumb_m1996".full.{mean,lo,hi}` |
| 13.0% out-of-pool; 25.7% vs 9.1% answered | same | `arms."guard_tau0.98".out_of_pool_pick_rate` = 0.12978125; `.answer_rate_of_out_of_pool_picks` = 0.25706718035155307; `arms.open_unguarded.answer_rate_of_out_of_pool_picks` = 0.09091856297407686 |
| 279 vs 54 distinct never-trained questions | same | `arms."guard_tau0.98".n_distinct_out_of_pool_questions` = 279; `arms.dumb_m1996.…` = 54 |
| best arm still −0.00446 [−0.00550, −0.00343] below pool | same | `arms.dumb_m1996.vs_pool_ceiling.full` |
| dumb@499 is 0.056 below pool | same | `arms.dumb_m499.vs_pool_ceiling.full.mean` = −0.05600122834654386; `out_of_pool_pick_rate` = 6.25e-06 (**"effectively zero", not zero**) |
| R and G touched no test data | `resume\`, `vocab_guard\` | 84 occurrences of `"test_touched": false`, zero of `true`; all `"split"` fields read `"validation"` |

---

## §4 — PHASE 3: a deeper tree carries real signal and no value

> Intended placement: as a subsection of Act 3, immediately after "Why the tree stopped at depth
> five" (`sec:depth`). It is that section's empirical follow-through.

Section~\ref{sec:depth} established that the tree's realised depth measures its stopping rule and
not the point where conditioning stops paying, and it explicitly declined to claim in either
direction. **This is the measurement that settles it, and it settles it against extrapolation.**

We took the frozen tree's fifteen leaves, dropped the minimum-leaf floor, and swept the full
$1{,}000$-candidate pool for the exact measured criterion at all $42$ resulting states —
$41{,}834$ exact held-out NDCG@10 evaluations — then grew one more level and rescored the whole
interview at $K = 16$ against the same static comparator.

**A sixth level is worth nothing: $-0.00059$ $[-0.00144, +0.00028]$ full and $+0.00055$
$[-0.00039, +0.00154]$ tail against the frozen tree.** Both straddle zero, and the full mean is
slightly the wrong side of it.

**The signal is nonetheless real, which is what makes the null informative rather than merely
absent.** A control that shuffles the branch labels — same states, same candidate sweeps, same
budget, the measured labels assigned to the wrong branches — lands $-0.00436$
$[-0.00545, -0.00328]$ below the frozen tree, and the true deep labels beat their own shuffle by
$+0.00377$ $[+0.00263, +0.00493]$. **The deeper split is choosing genuinely better questions for
genuinely identified subgroups; the users simply do not end up better ranked for it.**

The decomposition explains why, and it is arithmetic rather than interpretation. **For $5{,}130$ of
the $10{,}000$ users the sixth level changes which questions are asked but not which set** — the two
arms are an exact null on them by construction, $6.1\times10^{-6}$
$[-1.0\times10^{-4}, +1.3\times10^{-4}]$. The whole effect lives in the other $4{,}870$, where it is
$-0.00121$ $[-0.00296, +0.00058]$ and still does not separate.

The exposure was pre-registered and it is the honest reading of the null. **Twenty-seven of the
$42$ states sit under the $200$-user floor the frozen tree respected, covering $2{,}571$ users
($25.7\%$ of the cohort), and ten hold fewer than $50$ users.** A criterion measured on $9$ to $55$
users is measuring noise, and the cohort-size discipline that produced the frozen tree's gain is
exactly what a sixth level has to abandon.

Fine-tuning the distilled policy on the deep labels reproduces the same verdict from the other side.
It clears the static comparator, $+0.02059$ $[+0.01760, +0.02357]$, and it is below the deeper tree
it was trained to imitate, $-0.00676$ $[-0.00820, -0.00541]$; a label shuffle takes it to $-0.02357$.
**It also fails its pre-registered spanning gate** — the requirement that a later stage not degrade
what an earlier one proved — by $-0.00735$ on cold-state behaviour. The finer labels cost the exact
replica and bought nothing that replaces it.

**What this licenses.** Depth is not the lever. Conditioning finer than the frozen tree's twenty-two
nodes is real and it is small, and it is bounded by the cohort size any honest measurement of it
requires.

> **Live limitation.** This is a validation-cohort result, and its cohort is the one the frozen tree
> was grown on. No test look was taken at depth six.

**Provenance — §4**

*All from `C:\dev\phd\casper\experiments\battery\policy_deeper_tree_control.json` unless noted.
`docs/results/PHASE3_DEEPER.md` still reads `RESULTS — PENDING` and contains none of these; they were
read from the artifacts directly.*

| number | field |
|---|---|
| deeper − tree full −0.00059 [−0.00144, +0.00028] | `deeper_minus_tree.full.{mean,lo,hi,n,excludes_zero}` = −0.0005882277860560665 / −0.001438929432236357 / 0.0002830663288328936 / 10000 / **false** |
| deeper − tree tail +0.00055 [−0.00039, +0.00154] | `deeper_minus_tree.tail.{mean,lo,hi,n,excludes_zero}` = 0.0005521338784334732 / −0.0003925517009422527 / 0.0015387798468303284 / 9217 / **false** |
| shuffled − tree −0.00436 [−0.00545, −0.00328] | `shuffled_minus_tree.full.{mean,lo,hi}` = −0.004361516539690867 / −0.005449097465131278 / −0.003283417539023536, `excludes_zero` true |
| deeper − shuffled +0.00377 [+0.00263, +0.00493] | `deeper_minus_shuffled.full.{mean,lo,hi}` = 0.003773288753634801 / 0.002629380006489876 / 0.004930405532365119, `excludes_zero` true (**tail is a null: 0.00075 [−0.00053, +0.00203]** — the draft claims the shuffle separation on full only) |
| same-set null, n=5,130 | `SAME_SET_DECOMPOSITION.same_set_deeper_minus_tree_full.{mean,lo,hi,n}` = 6.102440229200473e-06 / −0.00010353639848409692 / 0.00012820857315544513 / 5130 |
| different-set, n=4,870 | `SAME_SET_DECOMPOSITION.different_set_deeper_minus_tree_full.{mean,lo,hi,n}` = −0.0012142881681594378 / −0.002961047944531857 / 0.0005792840920318013 / 4870 |
| deeper val 0.28296 vs tree 0.28355 | `deeper_val.full` = 0.28296149046889896; tree = 0.283549718… |
| 42 states, 27 under floor, 2,571 users (25.7%), 10 under 50 users | `PREREGISTERED_overfitting_exposure.{n_states, n_states_under_the_200_floor, users_under_the_200_floor, frac_cohort_under_the_200_floor, n_states_under_50_users}` = 42 / 27 / 2571 / 0.2571 / 10; state sizes min 9, p25 55, median 154, p75 311, max 1438 |
| 41,834 exact gains | `C:\dev\phd\casper\experiments\battery\policy_deep_labels.json` → `complete_level0_vectors.n_exact_gains` = 41834 |
| fine-tuned policy +0.02059 [+0.01760, +0.02357] | `C:\dev\phd\casper\experiments\battery\policy_deep_equivariant_both_phase3_gates.json` → `gates.G-E1b_gain_vs_static.full.{mean,lo,hi}` = 0.020586763710051897 / 0.017599958616372133 / 0.023571573624333684 |
| vs deeper tree −0.00676 [−0.00820, −0.00541] | same → `gates.gain_vs_deeper_tree.full.{mean,lo,hi}` |
| spanning gate fails by −0.00735 | same → `gates.G-B-regression_spanning.{delta,pass}` = −0.007352793153925508 / false |
| deep-label shuffle −0.02357 | `C:\dev\phd\casper\experiments\battery\policy_deepshuffle_equivariant_both_phase3_gates.json` → `gates.gain_vs_tree.full.mean` = −0.023568711802012796 |
| pool contract fingerprint identical to E1's | `policy_deeper_tree_control.json` → `pool_contract.fp` = `c282962b49a2d67ef47ff07544519f20` |

---

## §5 — THE SYNTHESIS: the bank beats conditioning, three times

> Intended placement: replaces and extends `sec:bankvsconditioning`. The existing table stays; this
> is the frame around it.

The chapter's demonstration is that putting the reported metric in the split flips the sign of the
conditioning gain. Its accounting is that the flip is small, and the honest way to say how small is
to price it against the other thing one could change. **The pattern that emerged over this
workstream is that at every level of the system, the parameter-free, state-free option beat the
conditioned one.**

**First, in the interview itself.** On $10{,}000$ held-out test users at sixteen questions, adaptive
conditioning through the tree is worth $+0.01035$ $[+0.00739, +0.01323]$ full and $+0.00705$
$[+0.00394, +0.01012]$ tail over the static order built by the same search on the same objective.
Changing nothing about *how* questions are chosen, and only *what they are about* — replacing the
item-only bank with an item-plus-concept bank at equal budget — is worth $+0.01430$
$[+0.01124, +0.01739]$ full and $+0.02199$ $[+0.01872, +0.02516]$ tail. On the tail the ordering is
not close.

**Second, inside the value model.** A parameter-free mean of six banked gain vectors beat the learned
state-conditioned lift head on rank fidelity, $+0.4163$ against $+0.0623$, and the policy built on
the parameter-free mean beat the policy built on the learned head by nearly three points of NDCG
($-0.02911$ against $-0.05840$, both losing to the bank). The estimator we did not fit outscored the
one we did.

**Third, in the open-vocabulary guard.** A state-conditioned refusal rule driven by the model's own
predicted answer probability recovers $57.0\%$ of the open-vocabulary deficit; a fixed popularity
cutoff that consults no state at all recovers $78.1\%$, and beats the learned guard head-to-head at
matched budget by $-0.00429$ $[-0.00545, -0.00312]$.

**The exception is the fourth level, and it is instructive.** Given a question it has never seen, a
state-conditioned reaction head beats the best state-free predictor available to it, $0.98167$
against $0.96285$ macro AUC, and — the part no state-free predictor can have at all — its advantage
*grows* as the interview proceeds, $+0.0175$ across sixteen positions against a flat comparator,
$+0.0172$ of which survives deleting the answerability features from the descriptor.

**So conditioning on the interview state is demonstrably valuable at prediction and demonstrably
expensive at selection.** The head knows things about the user that the state-free baseline does not;
every attempt in this chapter to convert that knowledge into a better *choice of question* has lost
to a fixed list. That is a statement about the objective and the label budget, not about adaptivity:
a fixed order is an adaptive policy that ignores its inputs, so no measurement can put adaptivity
below staticness — only particular methods. **What the accounting supports is a priority. If one
thing can be changed, change what the system asks about, not how it decides what to ask.**

**Provenance — §5**

| number | artifact | field |
|---|---|---|
| tree on test +0.01035 [+0.00739, +0.01323] | `C:\dev\phd\casper\experiments\battery\ndcgtree16test_policyB.json` | `cohort_weighted_test."16".diff_full@10_vs_policyB` = 0.010352646841853858; `.ci95_full_vs_policyB` = [0.007392842153497041, 0.01323270571999252]; n = 10,000 |
| tree tail +0.00705 [+0.00394, +0.01012] | same | `cohort_weighted_test."16".diff_tail@10_vs_policyB` = 0.007045511512352518; `.ci95_tail_vs_policyB` = [0.003935908509544605, 0.010116235760241576] |
| bank +0.01430 [+0.01124, +0.01739] full | `C:\dev\phd\casper\experiments\battery\channel_orders_policyB_test.json` | `contrasts."combined_minus_items@K16".diff_full@10` = 0.014302307171995562; `.paired_ci_full.{lo,hi}` = 0.011239260595086118 / 0.017389655715623356; n = 10,000; `"split"` = "test" |
| bank +0.02199 [+0.01872, +0.02516] tail | same | `contrasts."combined_minus_items@K16".diff_tail@10` = 0.021990577987644942; `.paired_ci_tail.{lo,hi}` = 0.018715717991948565 / 0.025162537170831053; **`.paired_ci_tail.n` = 9,281, not 10,000** |
| bootstrap kind | `ndcgtree16test_policyB.json` | `bootstrap` = {"kind": "paired percentile over per-user differences", "n_resamples": 10000, "seed": 98765} |
| +0.4163 vs +0.0623; −0.02911 vs −0.05840 | see §2 provenance | — |
| 57.0% vs 78.1%; −0.00429 [−0.00545, −0.00312] | see §3 provenance | — |
| 0.98167 vs 0.96285; +0.0175 / +0.0172 | see §3 provenance | — |

---

## §6 — PENDING: no prose drafted

- **L1 (as-if-asked encoding, R3)** and **L2a (forest distillation, cold tree + stated-genre
  teachers)** — spec only, not run (`docs/design/PAPER_B_POLICY_DESIGN.md` Part L). **PENDING.**
- **`VOLUNTEERED_COHORTS.md`** — the doc's RESULT 1/2/3 and WHAT THIS SETTLES are still literally
  `PENDING`, and the tie-break census with them. The run has finished and
  `C:\dev\phd\casper\experiments\battery\volcohorts\volcohorts_report_K15.json` holds numbers
  (`volcohorts_k4_volunteered.all_groups.full` = +0.00549 [+0.00290, +0.00798] on validation;
  fallback-200 arm +0.00216 [−0.00031, +0.00454], a null; tail null on both arms;
  `paired_contrasts."V_minus_genre|all_groups".full` = −0.01282 [−0.01624, −0.00944], i.e. the
  volunteered-cohort arm is significantly **worse** than the asked-genre arm). The doc declares
  **20** groups, not 17. Its own §"THE LIMITATION" calls the arm an in-sample ceiling, and the
  placebo arm the design sheet calls mandatory is not in the report JSON — only `v_placebo.log`
  exists. **No prose drafted; the doc must be written up before anything here is cited.** Note the
  "17 groups / +0.018 val / +0.010 test" figures in the brief are the *stated-genre* arm
  (`groupgreedy_genre15_*`), a different experiment, and those do verify:
  `groupgreedy_genre15_VAL_policyB.json` `n_groups` = 17, `cohort_weighted."15".all_groups_full_vs_policyB`
  = 0.01831040985524879; `groupgreedy_genre15_TEST_policyB.json` `primary_endpoint.all_groups` =
  0.010152854396385682 [0.007325019734769477, 0.012960526896982849].
- **Part K's full run** — closed at the pilot (`PART K IS CLOSED`, design sheet K.6). Nothing beyond
  the pilot cells quoted in §1.4 exists.

---

## §7 — WHAT THE CURRENT TEX CLAIMS THAT THE RESULTS DO NOT SUPPORT

Three items, in descending order of consequence. **None is edited here; this is the integration
list.**

1. **`sec:depth`'s closing sentence is now under-stated in one direction and needs the Phase-3
   result attached.** The tex currently writes a negative in both directions: "This run supports no
   claim that conditioning stops paying at depth five, and no claim that a deeper tree would pay."
   The first half stands. **The second half is now measured**, and §4 supplies it: a sixth level is
   a null against the frozen tree ($-0.00059$ $[-0.00144, +0.00028]$) while separating against its
   own shuffle. The sentence should become "no claim that conditioning stops paying at depth five —
   and a sixth level, measured, does not pay."

2. **The "Reach" paragraph's `0.374 / 0.320 / 0.087` could not be sourced.** The tex reads: "The
   three children of the root's did-not-answer branch have baselines $0.374/0.320/0.087$ against a
   parent at $0.119$", and derives from it the load-bearing claim that answering *at all* is worth
   roughly $+0.25$ against a largest question-*choice* effect of $+0.048$. I could not locate those
   four baselines in any artifact; the root's own child sizes are banked
   (`ndcgtree_PREFLIGHT.json` `preflight.best_balanced.concept.sizes` = 2606/4814/2580) but the
   per-child baseline NDCGs are not. **This is the one number in the existing chapter that a reader
   could not check and that carries a headline sentence.** It should be re-derived from the tree
   artefact and banked as a field before submission, or the sentence rewritten around the
   $1{,}785$/$1{,}438$ reach counts, which do verify.

3. **The bank table's caption says "$10{,}000$ test users" for a row whose tail interval has
   $n = 9{,}281$.** `sec:bankvsconditioning` and `CONDITIONED_ARMS_ON_TEST.md` §3 both say "paired
   over the same $10{,}000$ test users"; the full contrast is $10{,}000$ and the tail contrast is
   $9{,}281$, since users with no tail target are dropped from the bootstrap. One clause fixes it,
   and the same fix is owed on every tail interval in the chapter (validation tail n is $9{,}217$).

Two smaller ones, for completeness: `E1_POLICY_IMITATION.md`'s gate table still lists the shuffle
control as *running* two sections above the section that reports its result; and the E1 doc names an
artifact (`policy_traj/step0_probe.json`) that does not exist on disk. Neither affects a chapter
number.

---

## §8 — DISCIPLINE CHECKS ON THIS DRAFT

- **Routing:** the word does not appear as a claim anywhere above. §1 says the policy *replicates*
  the tree; §4 says the deeper split *identifies subgroups and chooses different questions for
  them*, which is the divergence result, not congruence. The tex's "partitions and diverges, is not
  shown to route" stands unamended.
- **"Adaptivity is large":** not written. Every adaptive sentence in §5 carries the bank number in
  the same paragraph, and §1.4 and §4 both close on limits rather than promise.
- **"Parity" / "tie":** not written. R's three policy-minus-tree cells are described as straddling
  zero, never as parity, and the redundancy sentence says "parity with two arms that cannot see the
  evidence", which is a *statement about redundancy counts*, not about scores — flagged here in case
  integration wants it reworded to "matches, at $5.4\%$ against $5.5\%$ and $5.2\%$".
- **Strategic / non-myopic:** the policy is never so described. §1 calls it a replica and §2 calls
  the value arms selectors.
- **Q's claim:** stated once, in the licensed form — *who can answer an unseen question and what they
  say when they can*. The phrase "taste prediction" appears nowhere.
- **Every interval above is a paired percentile bootstrap.** The one experiment with no intervals
  (Q) is labelled as such in its own prose, not only here.
