# Concept-fold key-term REPRESENTATION: the three-way, with paired CIs

**Question.** The key-term feature `xtilde_k` — the f=200 vector that stands for a concept inside the
conversational-bandit normal equations — can be built three ways. Which one is best, and is the
difference real or noise?

| construction | provenance |
|---|---|
| `uniform_member_mean` | ConUCB Eq. 4's own definition (the weight-normalised member mean) |
| `representative` | Hier-UCB (arXiv:2209.06129) — the `n_rep=20` most representative members |
| `whitened_centroid` | **ours** — the member centroid in the whitened basis |

**Contract.** Canonical Liang ML-25M, 10,000 test users, NO-FLOOR, tower `t2i26c_ep31.pt` +
concept channel `concept_cv3_ep08.pt`, key-term schedule `b(t)=5⌊log t⌋`, selector
`conlinucb_mcr`, and — the thing that makes the three comparable — **one answer convention**
(`--concept_neg_cap 0`, CERTIFIED-ERA). Verified equal across all three runs before any test:
`concept_answer_convention`, `concept_neg_cap`, `snapshot`, `concept_ckpt`, `n_test_users`,
`no_floor`, `ucb_alpha`, `conv_b_schedule`, `n_concepts`.

Analysis: `src/baselines/cfold_paired.py` → `experiments/baselines/interview/cfold_paired.json`.

## ★ These are three SEPARATE runs — pairing was PROVEN, not assumed

A paired bootstrap over three separate runs is only legitimate if the same users were scored in the
same order. That was not taken on trust, and it did not need to be, because the schedule supplies a
free oracle: under `b(t)=5⌊log t⌋` **the first key-term turn is t=3**, so at k=1 and k=2 the key-term
representation is never consulted and the three runs must be **bit-identical per user**.

| budget | uniform vs repr | uniform vs whitened | repr vs whitened |
|---|---|---|---|
| k=1 full / tail | 0.000e+00 (n=10000 / 9281) | 0.000e+00 | 0.000e+00 |
| k=2 full / tail | 0.000e+00 | 0.000e+00 | 0.000e+00 |

All twelve comparisons identical; divergence begins at k=4, exactly when key terms first fire. That
can only hold if the runs are row-aligned, so pairing is valid. The arm-vs-itself self-test passes
exactly (Δ 0.0000, CI [0.0000, 0.0000]) at every budget on both axes.

**One contract-recording trap, resolved.** `adaptive_table_cfold_representative_PARTIAL.json` carries
a **stale top-level `keyterm_repr: uniform_member_mean`** — a run-level default written before the
per-cell field existed. Its own `_contract.keyterm_repr` says `representative`, and its run log
confirms it (`[conv_bandit] key-term features: mode=representative`). `_contract` is authoritative;
the top-level field on that one file must be ignored. (The two newer runs record
`keyterm_repr_by_selector` per cell, which is the fix.)

## Point estimates (conlinucb_mcr, full / tail NDCG@10)

| repr | k=4 | k=8 | k=16 |
|---|---|---|---|
| uniform_member_mean | .1744 / .0586 | .1902 / .0714 | .2168 / .0877 |
| representative | .1753 / .0571 | .1899 / .0694 | .2135 / .0864 |
| **whitened_centroid** | **.1858 / .0647** | **.1935 / .0754** | **.2211 / .0942** |

## ★ Result: whitened_centroid separates from BOTH rivals on BOTH axes at EVERY budget

Paired percentile bootstrap over the per-user difference, 10,000 resamples, seed 98765.

**A. whitened − uniform**

| k | full | tail |
|---|---|---|
| 4 | **+0.0113** [+0.0094, +0.0133] | **+0.0061** [+0.0036, +0.0086] |
| 8 | **+0.0033** [+0.0014, +0.0051] | **+0.0039** [+0.0019, +0.0059] |
| 16 | **+0.0043** [+0.0022, +0.0064] | **+0.0065** [+0.0043, +0.0088] |

**B. whitened − representative**

| k | full | tail |
|---|---|---|
| 4 | **+0.0105** [+0.0085, +0.0124] | **+0.0076** [+0.0051, +0.0101] |
| 8 | **+0.0035** [+0.0017, +0.0054] | **+0.0059** [+0.0039, +0.0080] |
| 16 | **+0.0076** [+0.0056, +0.0097] | **+0.0078** [+0.0056, +0.0100] |

All twelve intervals exclude zero. This is the cleanest form the claim can take: **six of six
comparisons separate positive, on both axes, at every budget tested.**

**C. uniform − representative** (completes the three-way; fixes the order of the two losers)

| k | full | tail |
|---|---|---|
| 4 | −0.0008 [−0.0016, −0.0001] SEP | +0.0015 [+0.0006, +0.0024] SEP |
| 8 | +0.0003 [−0.0007, +0.0012] **ns** | +0.0020 [+0.0010, +0.0030] SEP |
| 16 | +0.0034 [+0.0018, +0.0049] SEP | +0.0013 [−0.0002, +0.0028] **ns** |

The two published constructions are **not reliably distinguishable from each other**: the signs
disagree across axes at k=4 and two of six cells are ns. Do not claim an ordering between
`uniform_member_mean` and `representative`.

## What may and may not be said

* **Supportable:** the whitened centroid beats both published key-term constructions, on both axes,
  at every budget from k=4, with paired intervals that exclude zero in all twelve cells. The margin
  is largest at k=4 on full (+.0113) and largest at k=16 on tail (+.0065 vs uniform, +.0078 vs
  representative) — i.e. it does not decay into the long interview.
* **Not supportable:** any ordering between `uniform_member_mean` and `representative`.
* **Scope:** one selector (`conlinucb_mcr`). The representation is a property of the concept feature,
  not of that selector, but nothing here measures the other arms.
* The margins are small in absolute NDCG (+.003 to +.011). Report them as separated-but-small; the
  interval, not the point estimate, is the claim.
