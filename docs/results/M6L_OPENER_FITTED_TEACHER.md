# M6l — a tree fitted on realistic opener states beats the anchor teacher, the cold tree and static

**Code** `src/instrument/m6k_cluster_tree.py` (fit) · `m6l_compare.py` (four arms) ·
`opener_gen.py` (openers) · **artifacts** `experiments/battery/m6/{m6k_teenagers.json,
m6l_teenagers.json}` · run 2026-08-19. **VALIDATION ONLY — the TEST look remains unspent.**

## The question

M6a/M6c established that the banked teachers are **anchor teachers**: their advantage over the cold
tree exists only at the one state they were fitted at (their anchor concept folded), and at realistic
openers it vanishes or goes negative. The author's remedy: **fit the tree on the varied states
instead**. This is that fit, on one group.

## The design — a controlled A/B on ONE variable

Same cohort (the 786 validation users who endorse `teenagers`, bit-identical to the banked teacher's),
same 1,000-candidate pool, same 15-question budget, same `min_leaf`, same env. **Only the fitting
state differs:**

| | state each user is in when the tree is fitted |
|---|---|
| banked **anchor** teacher | concept 922 folded — one shared state for all 786 |
| this **opener-fitted** tree | that user's own cluster-linked opener — 786 different states |

Openers come from `opener_gen`, matched to measured Reddit targets (film 0.580 / genre 0.363 /
tone 0.314 for this group; coherence and popularity per `M6E`), selected by **centroid lift** and
drawn only from the user's own fold-in evidence. Seeded, so a user wears the identical opener at fit
time and at scoring time. Depth 1: root splitter plus a fitted order per branch.

## ★ THE RESULT

All four arms scored at the same opener states, opener folded free and masked from being asked in
every arm. Paired percentile bootstrap, 10,000 resamples, seed 98765; ★ = CI excludes zero.

| K | opener-fitted | anchor | cold | static | vs cold | vs anchor | vs static |
|---|---|---|---|---|---|---|---|
| 1 | 0.21117 | 0.19071 | 0.20407 | 0.20407 | +0.00710 | **+0.02046★** | +0.00710 |
| 2 | 0.22892 | 0.20230 | 0.21488 | 0.20964 | **+0.01405★** | **+0.02662★** | **+0.01928★** |
| 4 | 0.24942 | 0.23091 | 0.24376 | 0.23126 | +0.00567 | **+0.01851★** | **+0.01816★** |
| 8 | 0.27713 | 0.26242 | 0.26762 | 0.24591 | +0.00951 | **+0.01471★** | **+0.03122★** |
| 15 | **0.30010** | 0.28708 | 0.28624 | 0.26701 | **+0.01386★** | **+0.01301★** | **+0.03308★** |

**Beats the anchor teacher at all five budgets** (+0.013 to +0.027, every interval excluding zero),
static at four of five (up to +0.033), and the cold tree at the headline K=15 (+0.01386), positive at
all five.

**The size of the swing.** At realistic opener states the *anchor* teacher ran −0.008 to −0.014
against the cold tree (M6a/M6c). Refit on those states, the same cohort's tree is **+0.0139**. The
diagnosis "teachers are anchor teachers" was right; fitting where you mean to serve is the fix.

## Which cells are clean

**★ cluster vs anchor is the controlled cell.** Both trees fitted on the *same 786 users*, differing
only in fitting state. Nothing else varies. That margin is the result.

⚠ **cluster vs cold carries an in-sample advantage.** This tree was fitted on exactly these users at
exactly these states; the cold tree was fitted on all 10,000 at cold states. +0.01386 is an upper
bound, as M4a's numbers were. **The static arm has no such advantage and shows the largest margin**
(+0.033 at K=15), which is the reassuring direction.

## What the fitted tree chose, and why it differs

Root splitter: **item *Harry Potter and the Deathly Hallows: Part 1*** (gain +0.02370, answer rate
**0.299**, sizes 198/37/551). The banked anchor teacher's root is the **concept `fantasy world`** at
answer rate **1.00**. Fitted where users have already volunteered something, the tree reaches for a
sharply discriminative *film* that most users cannot answer, rather than a universally-answerable
attribute — because the attribute's information is largely already in the opener.

`like` (198) and `dislike` (37) fall below the floor and inherit the parent tail; only `noanswer`
(551) earns its own fitted order. 4 nodes, 136 minutes.

## ★ ALL SIX GROUPS (complete, 2026-08-20 03:03)

Same design in every row: same cohort as the banked anchor teacher, same pool, same budget, same
min_leaf, **only the fitting state differs**. K=15, ★ = 95% paired bootstrap interval excludes zero.

| group | n | opener-fitted | anchor | cold | static | vs cold | vs anchor | vs static |
|---|---|---|---|---|---|---|---|---|
| arthouse | 587 | .30757 | .27555 | .28779 | .25808 | **+0.01978★** | **+0.03202★** | **+0.04949★** |
| marvel | 734 | .44193 | .41501 | .42312 | .39876 | **+0.01881★** | **+0.02692★** | **+0.04318★** |
| dialogue | 730 | .28527 | .26519 | .26835 | .24431 | **+0.01692★** | **+0.02008★** | **+0.04097★** |
| teenagers | 786 | .30010 | .28708 | .28624 | .26701 | **+0.01386★** | **+0.01301★** | **+0.03308★** |
| gore | 747 | .37246 | .35459 | .36472 | .33600 | +0.00774 | **+0.01788★** | **+0.03646★** |
| inspiring | 1,822 | .27182 | .26596 | .26913 | .24928 | +0.00269 | +0.00586 | **+0.02254★** |

**vs static 6/6 · vs anchor 5/6 · vs cold 4/6. All 18 point estimates positive; none negative.**

### `inspiring` is the exception, and it is the predicted one

It fails to separate on two of three arms — and it has been the weakest group in every prior
measurement (M4a: separating only at K=1, flat at every later budget). The structural reason is now
known: at 1,822 of 10,000 users it is **18% of the population**, so lift against the global rate is
≈1 for nearly everything and almost nothing is characteristic of it. Under frequency lift it had
**53 qualifying films** against 1,059–1,464 for the other five, and its most "distinctive" film was
*The Shawshank Redemption* — the popularity artifact M1 removed the era axis to avoid.

**A group with no distinctive material cannot have a group-linked opener**, so there is nothing for
an opener-fitted tree to exploit. This is the same conclusion M4a reached by a different route:
**specificity predicts payoff; breadth does not.** Reported, not excluded.

## ⚠ Scope

All six groups, depth 1, validation, **in-sample**: each tree is fitted and scored on the same users
at the same states. The `vs anchor` column is the controlled cell (both trees fitted on the same
users, differing only in fitting state); `vs cold` carries an in-sample advantage the cold arm does
not have, so those margins are upper bounds. **The static arm carries no such advantage and is the
only 6/6 column**, which is the reassuring direction.

## Cost, measured

136 min for 786 users at 1.68 s/eval — about **1.6× the banked anchor teacher's per-eval cost**
(1.051) on the identical cohort, because per-user prefixes destroy the shared item-state fold that
the concept block reuses. An earlier probe suggested no slowdown; it timed isolated calls and did not
reproduce the sweep's access pattern.
