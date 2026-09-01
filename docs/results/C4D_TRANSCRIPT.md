# C4D — the synthesised interview, end to end

**Status: COMPLETE (validation-side; TEST untouched). 2026-09-01.** Spec
`docs/results/C4D_TRANSCRIPT_SPEC.md`; code `src/instrument/c4d_transcript.py`; artifacts
`experiments/paperC/c4d/` (commits `c771be7`, `6956162`, `2728af9`, `cab1156`, `4dfbba8`).
Conclusions Fable's own. Renderer: the v2 **question** adapter, best-of-8 by LM likelihood, greedy
beside it. Cohort: all 10,000 validation users, `--path majority` (the neutral largest-child walk —
no cohort was shopped for). Answers, folds and NDCG **under a declared-circular answer model**.

**One-line verdict: the loop runs end to end and reads as an interview — six turns, six distinct
synthesised questions, four of them clean and topically distinct from what the tree would have
asked. Two of six show the templating failure C3B-v2 predicted, at a rate far worse than C3B-v2's
5.3%, and that gap is itself the finding: the question-form rate measured on bank questions
OVERSTATES what blends produce.**

## The transcript

| turn | state | n | synthesised question | the tree's question |
|---|---|---|---|---|
| 1 | `root` | 10,000 | **Are you in the mood for something action films?** | Do you like better than expected films? |
| 2 | `root.dislike` | 4,814 | **Do you like it when a film is suspense?** | Do you like dark humor films? |
| 3 | `root.dislike.dislike` | 3,815 | **How do you feel about sci-fi movies?** | Do you like alien films? |
| 4 | `…dislike³` | 2,409 | **Do you like it when a film is movie?** | Have you seen *Star Wars: Episode V*…? |
| 5 | `…noanswer` | 1,335 | **Do you like it when a film is detective drama?** | Have you seen *Crimson Tide (1995)*…? |
| 6 | `…noanswer²` | 1,108 | **How do you feel about children films?** | Have you seen *Star Wars: Episode VI*…? |

Turns 3, 5 and 6 are the figure's argument: the emitter asks about **sci-fi** where the tree asks
about "alien films", about **detective drama** where the tree asks an item question about *Crimson
Tide*, about **children films** where the tree asks about *Star Wars VI*. Same neighbourhood,
different — and more askable — object.

## ★ The templating failure is worse on blends than on bank questions

Turns 2 and 4 — *"Do you like it when a film is suspense?"*, *"Do you like it when a film is
movie?"* — are the boilerplate wrapped round a bare phrase with no agreement, and turn 4 is nearly
contentless (its blend is a diffuse *movie business / oscar (best cinematography)* direction).

**2 of 6 is 33%, against C3B-v2's measured 5.3% malformed rate.** The explanation is structural and
must be written into the limitation: C3B-v2 measured question-form on **held-out bank questions**,
each a single clean entity; the transcript renders **blends**, which are diffuse by construction
(median top-1 weight 0.225). **A template family fitted to single phrases degrades on mixtures, so
the 94.7% figure does not transfer to the deployed object.** Six turns is far too few to estimate
the blend-side rate — it is a warning, not a number — but the direction is unambiguous and the
mechanism is clear.

## ⛔ The NDCG column in the artifact may NOT be quoted

The synthesised arm scores above the tree's arm at 5 of 6 turns. **That is not a result and must
never be reported as one.** It is (i) under the declared-circular answer model, whose only licensed
use is the *relative* translation cost with both arms sharing the simulator; (ii) an unregistered
comparison invented after the fact; (iii) n=6 turns on one path; and (iv) display output from a
figure script, not an experiment. The author's 2026-08-25 ruling forbids exactly this claim. The
column exists so the figure can show a belief updating, and for no other purpose.

## The honesty rule did not fire — and that needs saying in the caption

All six turns emitted; there is no `[no emission — outside the fitted set]` row. The reason is
structural, verified by enumerating the tree: C1-final's primary cell holds out **only the
`root.like.*` subtree**, so every held-out state begins `root.like`, and the majority cohort turns
`dislike` at the root. Fully-emitting paths run to **depth 12**; any path avoiding `like` at the root
stays inside the fitted set for its whole length.

Two consequences, and both belong in the paper:
1. **Good news for deployability**: the fitted region is not a scattering of isolated states — it is
   a large connected part of the tree, gated by a single binary choice at the root. "Emits at 72 of
   144 nodes" understates how walkable it is.
2. **A limitation of this figure**: because the boundary never appears, *the figure cannot
   demonstrate the method's limits*. It looks better than the method deserves. If the `[no emission]`
   row is wanted — and the spec argues it is the most important feature — it needs a second figure
   on a `root.like…` path, which falls outside at turn 2 and stays out.

## The space question, settled empirically — and my diagnostic was the wrong one

The registered top-1 arm (does the fed point's nearest bank entry equal the emission's?) returns
**0.000 for all three constructions, including `target`, which is correct by definition.** An exact
index match over a 19,390-entry bank cannot separate anything. The arm that does is the top-5
neighbourhood overlap (chance 0.00026):

| space | top-5 overlap | verdict |
|---|---|---|
| `direct` | **0.00 / 5** | renders an unrelated direction — the amendment's suspicion, now measured |
| `lstsq` | **0.00 / 5** | collapses onto one obscure film at 5 of 6 states; R²=0.671 is **not quotable** |
| **`target`** | **2.17 / 5** | used |

Same neighbourhoods, different index: *cgi/visuals/plot* → *visuals/plot/cgi*; *Star Wars IV/space
opera* → *space opera/Star Wars V*. **The lesson for the record: a diagnostic that returns the same
value for the construction that is right by definition is not a diagnostic.** Mine was; the
neighbourhood arm replaces it.

## Gates

m7dj rebuild bit-identical to C1-final's 72 targets (max|diff| 0) · 72/72 sweeps argmax == splitter,
max gain error 0 · `tree_route` match 1.0000 on all 1,108 users on the walked path · canonical
validation m=0 snap exact · leak check 0/10,000 · emissions traced to C1-final rev `a52c9640`,
57 trained states, median cos to own target 0.96958 · HARD RULE 10 clean · 6/6 distinct generations,
0 empty · TEST untouched · no LLM API.

## Open item

The neighbourhood numbers are reproducible from the banked npz but land *inside* the JSON only on
the next run of the committed code (~6 min on a clear box). Deferred rather than risk a second OOM
against a live job.
