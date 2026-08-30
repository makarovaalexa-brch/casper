# C4D — the synthesised-interview transcript: what the capstone figure shows, and what it may not claim

**Status: SPEC ONLY, written before the run (2026-08-30).** Machinery
`src/instrument/c4d_transcript.py`; outputs `experiments/paperC/c4d/`. Predecessors, all imported:
`docs/results/C1_FINAL.md` (the policy, and the 57 states it emits at), `C4C_DECODING.md`
(best-of-8 by LM likelihood is the decoding of record), `C4_PILOT.md` (R_s and the fold-back path),
`docs/design/DS_C3B_V2_QUESTIONS.md` (the demo is conditional on that run's reading 1 passing).
No number in this file yet; it exists so the caption writes itself and so nothing can be claimed
after the fact that was not admissible before it.

## The figure in one sentence

One cold-start interview, walked down Paper B's cold tree over the validation cohort, in which every
question asked at a state the C1-final policy was trained on is **the policy's own off-catalogue
direction, said in English**, printed beside **the catalogue question the tree would have asked at
that same state** — with the belief, the top-5 recommendations and full/tail NDCG@10 updating turn
by turn.

## What each row contains

| column | what it is |
|---|---|
| state | the tree node the cohort stands at, and how many users are there |
| synthesised question | the adapter's rendering of the emitted direction, **best-of-8 by LM likelihood** (the decoding of record), with the **greedy** render printed beside it because C4C ruling 2 requires the variance to be visible |
| the direction itself | the five bank phrases nearest the emitted vector, and its cosine to the state's own measured-gain blend target |
| the catalogue comparator | `question_text` of the node's own splitter — what the tree actually asks here |
| the answer | the cohort's answer to each concept the render reads back to, **under a declared-circular answer model** |
| the fold | the cumulative concept set folded so far, through the certified `ConceptFoldNet(cv3)` path |
| the consequence | top-5 recommendations after the turn (display only), and full/tail NDCG@10 from `genv_eval` under the certified contract, against that cohort's own m=0 base |

## ★ The honesty rule — the most important feature of the figure

C1-final emits at **57 trained states and nowhere else**; its held-out-state emissions are both
off-target (cos −0.119 / −0.011) and out of scale (norm ratio 3.480 against 1.045), and
`C4_PILOT.md` ruling 2 binds them. So:

> If the walk reaches a state outside the 57, the row prints the tree's own question and is marked
> **`[no emission — outside the fitted set]`**. No question is invented, interpolated, or borrowed
> from a neighbouring state.

The whole `root.like.*` subtree is held out in C1-final's primary cell. A walk that turns `like` at
the root therefore falls off the fitted set immediately — and that is a row of the figure, not a bug
in it. The figure's claim is *"here is what a synthesised interview looks like, and here is exactly
where the method stops"*, which is a stronger and more defensible claim than a full column of
plausible-looking text would be.

## The three declared choices, stated before the run

1. **Branching is behavioural; folding and scoring are declared-circular.** The walk descends the
   tree on `ndcg_tree.branch_of` — the tree's own three-way rule on real answers — decided *before*
   the circular answer model is installed, so the path down the tree is the tree's real path. The
   fold and the NDCG then run under the declared-circular model (`answer(r,c) = sign(u*_r · d_c)`,
   `c4b_noise.geometric_answers`), because an off-catalogue synthesised question has no entry in the
   behavioural answer table at all and refusal is never simulated there. Both columns receive the
   identical treatment, so the choice cannot move the comparison — but it is an inconsistency and it
   is declared here, in the JSON, and in the caption.
2. **The fold is cumulative.** Turn *t* folds everything asked up to *t* (concepts ascending by id
   within a turn, `c4b_noise`'s arm-symmetric ask order), and every turn is scored on the cohort
   standing at that node against that same cohort's own m=0 base. A transcript whose belief did not
   accumulate would not be a transcript.
3. **The top-5 lists are a display object**, the cohort mean of the decoder score, unmasked. Every
   NDCG number comes from `genv_eval` under the certified contract; none comes from the display path.

## ★ The one place the artifacts forced a choice: which space the emission is rendered in

The C4 pilot rendered points from **E1's** `k(phi)` space, which is the space the C3B adapter was
trained on. C1-final changed warm start — its policy is Paper B's `m7dj_policy.pt`, so its emissions
live in **m7dj's** `k(phi)` space. Both are 128-dimensional over the *same* frozen `phi` (the item
counts, decoder rows and concept embeddings all come from `train.csv` and do not move with the
cohort), but they are two different learned read-outs and nothing guarantees a shared basis.
C1-final banks no policy checkpoint — only the per-state emitted vectors — so this cannot be
sidestepped by re-emitting.

The script therefore treats the space as a **declared, measured flag**, not an assumption. It builds
and renders all three constructions and banks a snap diagnostic for each:

| `--space` | what it feeds the adapter | what the row renders |
|---|---|---|
| `direct` (default) | the emission's own vector, standardised by m7dj's own bank mean and global scale | the policy's emission, **on the assumption the two bases coincide** |
| `lstsq` | the emission's score profile over the 19,390-entry bank (the quantity the policy's own argmax reads) re-expressed in the adapter's space by ridge-stabilised least squares, then rescaled | the policy's emission, re-expressed; report R² |
| `target` | the state's m=5 measured-gain blend rebuilt out of the adapter's own bank rows | the state's **blend target**, not the emission — labelled in every row |

**The diagnostic that decides it:** for each construction, does the fed point's nearest bank entry
agree with the emission's nearest bank entry in its own space? Near-zero agreement means the two
read-outs do not share a basis, in which case a `direct` render is a render of a *different*
direction and must not be shown. That is a finding to report, not a number to work around.

**The gate that makes the m7dj rebuild trustworthy:** the m=5 blend targets rebuilt inside the
validation env are asserted **bit-identical** to the 72 rows `c1_final_per_state.npz` banked from the
train env. If `phi` were not split-invariant, or the pool / norm / checkpoint were not m7dj's, the
run stops there.

## What this figure may NOT claim

- **No value claim of any kind.** Every NDCG row runs under a declared-circular answer model and
  carries at most a *relative* reading (AUTHOR RULING 2026-08-25). "Asking the synthesised question
  helps" is not available from this artifact, and neither is "it beats the tree".
- **Nothing at states outside C1-final's 57.** Not a hedged version, not an illustration.
- **Nothing about the item channel** (the C4 pilot disqualified it as an instrument: its prior floor
  is not a floor).
- **Not that the questions are good because they read well.** C3B found exact content recovery at
  0.0000 with above-ceiling re-encode — fluency is not evidence.
- **Not that a human would answer them.** There is no human study; the cohort's answers are
  simulated and declared circular everywhere they appear.
- **No test-set claim.** The 10,000 canonical TEST users are not built, not loaded, not touched.
- The population caveats travel: the cold tree's states are nested and structurally selected, the
  cohort is the validation split, and the blend's five ingredients are not claimed to be the right
  five.

## The caption this is meant to make writable

> One cold-start interview. At every state where the policy was trained to emit, the question shown
> is the policy's own off-catalogue direction rendered into English (best-of-8 by LM likelihood);
> beside it is the catalogue question the tree would have asked at that state. Where the policy was
> not trained to emit, no question is invented and the row says so. Answers, folds and both NDCG
> columns run under a declared-circular answer model and carry no claim about the value of asking.
