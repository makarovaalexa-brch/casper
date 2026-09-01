# Paper C — what is actually wrong, and the ranked menu of fixes
**Fable, 2026-09-01, at the author's request ("think of anything at all that can be improved…
i think the policy model itself and starting embedding?"). Companion to
`PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md`, which stays the publication-strategy read.
Author's steer of 2026-08-30 binds: fun capstone, not an audit paper.**

The author's two guesses are both right, and they are the same problem seen from two ends. This
sheet states the mechanism, then ranks every improvement by value-per-hour with its decisive test.

---

## 1. THE DIAGNOSIS

### 1.1 We built the emitter in the shape this project has already measured to fail

Paper B ran the cleanest possible experiment on which learned shapes generalise here, and got a
split verdict that we then ignored:

| shape | experiment | result |
|---|---|---|
| **reaction** — predict a per-question quantity from the state | `HELDOUT_QUESTIONS.md` (Q) | **generalises**: AUC 0.98167 on questions held out by identity, beating the fair backoff 0.96285 |
| **action** — choose which question is right | `P1_HELDOUT_ACTIONS.md`, `P1B_NEVER_TAUGHT.md` | **does not**: 0.0000 top-5 emission on held-out labels; ranks never-taught questions no better than popularity |

C1-final's objective — regress `ẑ` onto a 128-d blend direction — is **action-shaped**. One
supervised vector per state, 72 states, and the mapping must invent the whole geometry of the gain
landscape from the state alone. It fit (0.970) and did not extrapolate (−0.119 held-out, with the
untrained warm start matching the own-vs-other margin). That is exactly the P1/P1b result again,
one level up. **We should have expected it, and the fix is to change the shape, not the
hyperparameters.**

### 1.2 The target space is a classification read-out, not a semantic space

C0 measured the bank's mean pairwise cosine at **0.9646 raw against 0.0018 centred**. `k(φ)` is
trained by cross-entropy, which constrains only score *differences*, so it is free to carry an
enormous common mode — and it does. Consequences we have been paying for all month:

- an arbitrary point sits at cos ≈ 0.93 to *something* in the bank, so "close to a bank entry"
  means almost nothing (C1 pilot Stage 0);
- blends are dominated by the shared component, so the state-specific signal is a small residual;
- the emission's raw-space geometry was misread three separate times before the mean-removed
  diagnostic was run (C0's cos₁, the split-half stability, the C4 emission scale);
- the renderer and the policy ended up in **different** `k(φ)` read-outs (E1's vs m7dj's) — the
  space mismatch the transcript build caught.

The **tower space** (centred, popularity-stripped decoder rows for items; `build_concept_dirs`
directions for concepts) has none of these problems: it is the space the recommender ranks in, the
space the evaluation bridge measures in, and it is semantically organised rather than
score-organised. There is no good reason the emission target lives anywhere else.

### 1.3 The input may not determine the target at all

This is the possibility that would make every architecture change pointless, and it had never been
tested. The gain of a question at state `s` depends on the **residual disagreement** in the cohort
still consistent with the answers so far. The policy input is the belief state — a summary of what
is *known*. The target depends on what is *unknown*. Two states can share a belief vector and differ
in how much spread remains.

`c5_diagnostics.py` (launched today) settles it: correlation between state-similarity and
profile-similarity, plus a **ridge linear probe** from state → target on the same 57/15 split. If
ridge also fails, the inputs are insufficient and no architecture fixes it; if ridge beats the
trained policy, the policy is the problem. **Run this before building anything.**

---

## 2. THE MENU, RANKED BY VALUE PER HOUR

### ★★★ A. Re-parameterise: predict the gain PROFILE, construct the blend (`DS_C5`, code launched)

Instead of regressing onto the blend, predict `s(q | state)` over candidates — which the existing
head already computes — with a listwise loss toward the measured gains, then emit
`ẑ = Σ_j p_j e_{q_j}` over the softmax-weighted top-m.

Why this is the right object, not merely a different one:
- **Supervision density**: ~1,000 numbers per state instead of 128. 72 states become ~72,000
  constraints.
- **It is reaction-shaped** — the shape Paper B measured as generalising here (§1.1).
- **The pathologies become impossible by construction**: the emission is a convex combination of
  real bank entries, so it cannot leave the manifold and its norm cannot drift. That kills the 38×
  scale bug *and* retires the scale-term hack C1-final needed.
- **It nests the discrete policy exactly**: τ→0 recovers the tree's argmax, so "continuous ⊇
  discrete" becomes a proved property rather than an aspiration — and τ_t→0 recovers C1-final's own
  one-hot objective, so C5 nests the incumbent too.
- **Better evaluation**: profile-level readings (Spearman/NDCG@5 of predicted vs measured gains at
  held-out states) are far more informative than a single cosine, and have honest baselines
  (popularity prior, global-mean profile).

Cost: one training run, hours not days. Risk: if §1.3's probe says the inputs are insufficient,
this improves the fit but not the extrapolation — and that is still worth knowing.

### ★★★ B. The closed-form emitter: the principal axis of residual disagreement (`c6`, launched)

If the measured-best direction at a state is approximately the top eigenvector of the cohort's
residual covariance, then the emitter needs **no training at all**, works at **every** state
including the 872 with no gain table, and comes with an interpretation a reader can hold:
*ask about the axis along which the people who answered like this still disagree most.*

This is the single highest-upside item on the list. It would dissolve the generalisation problem
rather than mitigate it, and it turns the contribution from "a network memorised 72 directions"
into "here is where the informative directions live, in closed form".

Two ways it can fail, both informative and both measured today:
- the axis is not where the gains point (clean negative, reported, and the learned route stands);
- the axis is *unanswerable* — it points where no real question sits. That gap is the project's
  Baum & Lang problem quantified, and is publishable in itself: **how far answerability bends the
  ideal direction**.

⚠ Positioning risk being checked in parallel: Info-Synth synthesises along the posterior's principal
eigenvector. If our construction coincides with theirs, we cite and reposition as the *rendered,
catalogue-free, answerability-analysed* variant — we do not claim the geometry.

### ★★ C. Move the emission space to the tower (`--space tower`, in C5's implementation)

Removes §1.2's whole class of problems at once: no common-mode domination, no E1-vs-m7dj mismatch,
native evaluation (the bridge already measures there), and the renderer's target becomes the same
object as the policy's output. Cost: retrain the adapter on tower vectors — the same overnight cost
as the v2 run we just did. Test first with the cheap version: is the *target set* better conditioned
in tower space (D1's pairwise-diversity statistic computed both ways)?

### ★★ D. Give the policy the input the target actually depends on

If §1.3 says inputs are insufficient, the principled additions are cohort **dispersion** statistics:
size, covariance trace, top-3 eigenvalues, and the top eigenvector itself (which by §B may be nearly
the answer). Note the elegance: item B's closed-form direction becomes item D's input feature, so
the two are the same insight used two ways — and if B works, D is how a learned model gets to use it
while still learning the answerability correction B cannot.

### ★★ E. More measured states (the inheritor sweep, ~4.4 h detached)

Doubles 72 → ~144. Straightforwardly more data for A/D, and more states where the transcript can
show an emission rather than `[no emission — outside the fitted set]`. Cheap, unattended, no new
ideas required. Worth launching the moment the box is free — but *after* the diagnostics, because
if the inputs are insufficient this buys fit, not generalisation.

### ★ F. Renderer robustness (deferred, and correctly so)

Two-stage adapter→LoRA (ELM needed it to converge; our targets are far shorter, which is why
adapter-only worked), a second training seed per arm to license the word "robust", and the item
channel's instrument limit. None of these block the claim; all are cheap; none change the story.

---

## 3. WHAT I WOULD NOT DO

- **Chase generalisation as a headline.** Under the capstone steer, "the policy emits where the
  landscape was measured" is an honest scope line, and a *closed-form* emitter (B) is a better
  answer than a bigger network anyway.
- **A larger LM.** The claim is faithfulness, not fluency, and our own machine-only-synonym result
  is the argument that fluency is the wrong axis.
- **Any human study.** Same reason, now with our own numbers behind it.
- **More controls for their own sake.** Three separate controls have already decided headlines here
  (C3B's directionless C2, C4B's reversed twin, C1-final's warm-start null). The lesson is
  registered; adding a fourth control to a 4–6 page paper is not the marginal win.

---

## 4. THE ORDER OF OPERATIONS

1. **Diagnostics** (`c5_diagnostics`) — decides model-vs-inputs. Minutes. *Running.*
2. **Eigenvector test** (`c6_eigen_emitter`) — decides whether B dissolves the problem. Minutes.
   *Running.*
3. **Lit check** — decides what B may claim. *Running.*
4. Then, informed by 1–3: launch **C5 training** (A, code ready), and/or the **tower-space** variant
   (C), and/or the **inheritor sweep** (E) overnight.
5. Transcript figure and chapter reconciliation proceed in parallel regardless. *Running.*

**The honest framing for whatever comes back:** Paper C's current claim survives all of this
untouched — the mechanism result, the loop, the catalogue-anti-alignment hook and the translation
cost are banked and do not depend on the emitter improving. Everything on this menu is upside on top
of a paper that already has enough to be written.

---

## 5. FIRST DIAGNOSTIC RESULT (D1, banked `experiments/paperC/c5_diagnostics.json`, 2026-09-01)

**Pairwise cosine among the 72 blend targets** (2,556 pairs), and the comparison that matters:

| set | raw | mean-removed |
|---|---|---|
| the 72 **blend targets** | median **0.920** | median **0.425** [IQR 0.247, 0.562] |
| their own **top-1 candidates** | — | median **0.247** [IQR 0.083, 0.407] |
| random bank entries | — | mean **0.002** |

### ★ Blending itself destroys state-specificity — and that is a design defect, not a curiosity

The top-1 measured candidates at different states differ from each other at 0.247. **The 5-blends of
those same candidates differ at 0.425** — averaging five vectors pulls every state toward the pool's
common direction, so inter-state similarity nearly doubles. *The operation that buys us an
off-catalogue point costs us the thing that makes the point state-specific.*

### It also EXPLAINS C1-final's null

C1-final's reading 1a — "the emission is closer to its own target than to a random other state's" —
cleared zero (+0.059) but was matched by the **untrained** warm start (+0.084), and I recorded that
as "neighbouring states carry correlated targets". D1 quantifies it: the targets sit in a cone with
median pairwise cosine 0.425, so **any** vector pointing into that cone earns a positive own-vs-other
margin without knowing anything about the state. The null was not a fluke of the pairing; it is a
direct consequence of the target construction.

### The fix, and it is cheap: RESIDUALISE THE TARGET

Train the emitter on `target − mean(targets over training states)` — the state-specific deviation —
and add the common component back at inference. Consequences:
- the learner spends its capacity on the part that actually varies, instead of re-deriving a shared
  cone 72 times;
- identification becomes a meaningful test rather than a cone-membership test;
- the reported quantity gains an interpretation we currently lack: **what fraction of the
  measured-best direction is common to all states, and what fraction is state-specific?** That number
  belongs in the paper whatever it turns out to be.

⚠ Registered honestly in advance: if the state-specific part is small, the correct sentence is *"the
measured-best directions share a dominant common component; conditioning moves a modest residual"* —
which is **the same fact Paper B reports as "the bank beats conditioning"**, now visible in the
geometry of the question space rather than in an NDCG table. Two chapters, one finding, arrived at
independently. That is a better result than a bigger margin would have been.

**Implementation note:** residualisation is a one-line change to the target in either C1-final's or
C5's loss (`t ← t − t̄`, with `t̄` computed on TRAINING states only and banked in the manifest), and
it composes with the profile parameterisation rather than competing with it.
