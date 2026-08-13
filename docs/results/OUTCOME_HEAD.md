# PART F — THE TWO HEADS: outcome `P(o | state, q)` and lift `V(o | state, q)`

**Status: RUNNING (2026-08-13). Numbers below are filled in as each stage lands; every table states
its own cohort. Validation and TRAIN only — no test look, no NDCG on a train user.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` — **Part F is governing**.

> **AUTHOR SAID (2026-08-13):** *"nah, plz train on all and see if it even works. after than we can
> build curves and stuff. how long to train one version?"*
> **AUTHOR SAID (2026-08-13):** *"not following. how you not running 2nd head?"*
> **AUTHOR SAID (2026-08-13):** *"maybe having 2 heads, answerability and ndcg lift would be good?"*

---

## What this run is, and what it deliberately is not

Two heads, both fitted on data that already exists, neither costing a single new NDCG evaluation:

| | what it predicts | fitted on | cost of the labels |
|---|---|---|---|
| **outcome head** | `P(like / dislike / no-answer \| state, question)` | 140,768 TRAIN users × 16 interview positions × **all 19,390 catalogue questions** | **zero** — the outcome is the structural answer rule, evaluated |
| **lift head** | `V(like / dislike / no-answer \| state, question)` | the exact NDCG gains already banked in the tree's CELF heap sidecars | **zero** — already on disk |

Combined, they score a question the way Part F.1 specifies: `Σ_o P(o | q, s) · V(o | q, s)` — the
decomposition of the target quantity itself, with the information-theoretic value of EDDI's identical
decomposition replaced by **measured NDCG**.

**Deferred by the author's instruction, and not built:** the held-out-QUESTION generalisation test and
the 10 / 25 / 50 / 100% sample-efficiency curve of F.5. This run trains on all questions and all train
users. *"train on all and see if it even works"* is the question being answered.

**Three outcomes, never two.** The root's three children sit at 0.374 / 0.320 / 0.087 and ARM N
measured a "no" worth +0.023 at k=2. A binary answered/not encoding throws away the like/dislike
distinction exactly where it pays, and would not match the tree's own ternary branching.

## The firewall

- The 10,000 canonical TEST users are **never built, never loaded, never touched**. Every artifact
  stamps `test_touched=false`; neither file's parser accepts `test`.
- **No NDCG is measured anywhere in the outcome head.** That is what makes fitting on the 140,768
  TRAIN users legitimate: the tower memorised their ratings, so NDCG there would be meaningless — but
  an *observed answer outcome* is not a model output. Train users supply interview STATES and
  OBSERVED OUTCOMES, both functions of the user's own fold-in half.
- The certified pair (`t2i26c_ep31.pt`, `concept_cv3_ep08.pt`) is frozen; the tree is never regrown.
- Every NDCG number in the lift-head half is on the **10,000 validation users**, through the existing
  scoring funnel, after the canonical m=0 snap `0.15686538` reproduced exactly.

## ★ The no-profile guard, and why it is structural rather than declarative

Part F.5 is explicit: **never any profile feature**, because answerability is a structural rule partly
computable from the profile, so a head that sees profile-like input learns an arithmetic identity and
the generalisation win is hollow. The head sees the ternary answer history so far, the tower latent
`z_t`, depth/remaining budget, and the scored question's frozen descriptor. It never sees the user's
rating of the question it is scoring.

Reading the feature list is not proof. The guard asserted in code is:

> **At position 0 the interview has revealed nothing, so the state must be BIT-IDENTICAL across every
> user.** Anything that distinguishes two users at an empty interview is a profile feature.

`guard_no_profile` asserts max |z₀(u) − z₀(u′)| **== 0.0** over 20,000 user-states before a single
gradient step. Any leak — the user's ratings, their popularity profile, their own latent — breaks that
equality exactly.

## Coverage: no sampling anywhere

Every state scores **all 19,390 candidates** (18,359 items + 1,031 concepts) and the loss is taken
over all of them — one matmul, so full coverage is affordable. Per-user outcome labels are stored
**sparsely** (answered items as a CSR, concepts dense) rather than as a 140,768 × 19,390 dense block,
which is the difference between 0.25 GB and 2.7 GB. Nothing is subsampled: not users, not positions,
not candidates. Already-asked candidates are excluded from both the loss and every metric — at most 15
of 19,390 cells — because their outcome is literally in the input and scoring them would be measuring
a copy.

## Where the labels come from, and the gates they passed

`policy_traj.py --mode gen` applies the **frozen** tree to all 140,768 train users and emits, per
chunk: the 16 interview states, and the three-way outcome of every catalogue question.

| gate | what it asserts | result |
|---|---|---|
| **canonical snap** | validation m=0 intercept == `0.15686538` exactly | ✅ `0.15686538118354684` |
| **G-T1** | the re-applied tree reproduces every internal node's banked ternary `child_sizes` and `answer_rate` exactly | ✅ 7/7 |
| **G-O1** | the harvested outcome table reproduces `branch_of` — the tree's own answer rule — exactly, on questions from both channels | ✅ 24 questions × every user, every chunk |
| pool identity | the train env's 1,000-candidate pool is byte-identical to the validation env's | ✅ |
| leak | 0 users have a held-out target inside the routing lookup | ✅ |

---

## RESULTS

*(filled in as stages land)*

---

## Appendix — a blocker worth recording

The harvest died twice with **exit 1 and no traceback**. Cause: `train_tower_t2.load_recvae_teacher`
hard-codes `.cache/baselines/recvae_ml25m_liang.pt`, and yesterday's disk-recovery commit (`49b3586`)
deleted it as an MD5-verified duplicate of `models/baselines_ml25m_liang/recvae_ml25m_liang.pt`. It is
a live dependency of the certified tower's build path, not a dead copy. `raise SystemExit` prints only
its message, which is why it read as a benign warning twice before being diagnosed. Restored as a
**hard link** (zero additional disk), MD5 re-verified equal on both paths;
`models/DELETED_CACHE_CHECKPOINTS.md` corrected.

**The lesson, and it generalises:** byte-identity to a surviving copy proves a file is redundant as
*data*, not that its *path* is unreferenced. Grep for the path before deleting a "duplicate".
