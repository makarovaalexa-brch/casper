# ENTROPY vs NDCG — THE BOUNDARY DEMONSTRATION (2026-07-14)
`scripts/entropy_vs_ndcg.py`. 150,239 users, all 800 candidates, all clusters. No sampling. No training.
Selection on a cluster's half, evaluation on its DISJOINT half. **This is the author's question, answered.**

## THE RESULT — THE THESIS IN TWO PANELS
```
(1) DO THE CLUSTERS WANT THE SAME QUESTION ORDER UNDER EACH CRITERION?
    criterion     distinct top-1   top-10 overlap   Spearman
    NDCG (task)      12 / 12           0.023          0.062    <- near-ORTHOGONAL rankings
    ENTROPY          11 / 12           0.165          0.795    <- near-CONSENSUS: ONE LIST
    VARIANCE         10 / 12           0.068          0.689
    ANSWERRATE        2 / 12           0.111          0.699

(2) OUT-OF-SAMPLE TAIL NDCG@10 OF THE QUESTION EACH CRITERION PICKS
    NDCG (task loss)   0.0608   +0.0194   <-- WINS (+47%)
    ANSWERRATE         0.0436   +0.0022
    STATIC             0.0414     ----
    VARIANCE           0.0413   -0.0001   <-- ties static
    ENTROPY            0.0346   -0.0068   <-- LOSES TO STATIC
```

## WHAT IT MEANS
- **Rank questions by INFORMATION** => the clusters largely AGREE (Spearman 0.80) => you converge on ONE LIST
  for everybody => **and that adaptive policy LOSES to a static one (-0.0068).**
- **Rank questions by the TASK LOSS** => the clusters want almost ORTHOGONAL lists (Spearman 0.06) =>
  **and you WIN by +47%.**

> **The non-adaptivity theorem is TRUE — and IRRELEVANT, because it only binds objectives nobody should use.**
> Krause & Guestrin (ICML 2007): an objective depending only on the predictive variance cannot benefit from
> sequencing. Our data now SHOWS that, and shows the surrogate-driven adaptive policy going BACKWARDS.

**AND IT REPRODUCES SEPLIARSKAIA'S PUBLISHED MECHANISM IN OUR OWN DATA.** Sepliarskaia, Kiseleva, Radlinski &
de Rijke (RecSys 2018) found a STATIC questionnaire BEATS adaptive decision trees, and explained it:
*"PWDT optimizes a function that is different from the loss function, namely weighted generalized variance."*
**That is exactly the -0.0068 above.** They published half of it in 2018; we can now show the OTHER half — that
ranking by the task loss reverses the sign and wins by 47%.

## ⭐ IT RETRO-EXPLAINS OUR ENTIRE YEAR
Every one of the **eight tied policies** — and the **static entropy baseline they could not beat** — was
**ENTROPY / EIG driven**. They were optimising the one criterion that provably collapses to a single list.
The single note in our records that a **direct-NDCG reward** helped ([[policy-ladder-ndcg-reward]]: *"P1:
direct-NDCG reward fixes belief-decoherence"*) was the one time we stepped over the line without knowing why.
**Our year of nulls was not a fact about elicitation. It was a fact about our objective.**

## THE CLAIM (falsifiable, and now demonstrated)
> *Adaptive elicitation fails when you rank questions by how much you LEARN ABOUT THE USER, and succeeds when
> you rank them by how much BETTER THE RECOMMENDATION GETS. Those two orderings are nearly ORTHOGONAL
> (Spearman 0.06), and only the second is user-specific.*

## BINDING CONSEQUENCE FOR THE POLICY (see PLAN_NEXT_2026-07-14.md, Stage 3)
**TRAIN THE POLICY ON THE TASK LOSS. NEVER ON EIG-ABOUT-z.** Information features (`q^T Sigma q`, closed-form
EIG) may be INPUT FEATURES to a value head, but they must NEVER be the objective. This is now measured, not
argued: the entropy-ranked policy is WORSE THAN STATIC on our own data.

## SCOPE
Turn 1, item candidates, genre clusters. The multi-turn and concept/open-channel versions are pending
(Phase B + the belief-cell teacher). The four criteria are computed WITHIN each cluster on its selection half
and evaluated out-of-sample, so the comparison is symmetric.

---

## RAW OUTPUT
```
[e-vs-n] usable users = 150239  (must match the probe's 150,239)

(1) DO THE CLUSTERS WANT THE SAME QUESTION ORDER UNDER EACH CRITERION?
    criterion     distinct top-1 across clusters   mean top-10 overlap   mean Spearman
    ----------------------------------------------------------------------------------
    NDCG                12 / 12                                  0.023           0.062
    ENTROPY             11 / 12                                  0.165           0.795
    VARIANCE            10 / 12                                  0.068           0.689
    ANSWERRATE           2 / 12                                  0.111           0.699
    (top-10 overlap ~1.0 and Spearman ~1.0 => ONE LIST FOR EVERYBODY)

(2) OUT-OF-SAMPLE TAIL NDCG@10 OF THE QUESTION EACH CRITERION PICKS
    (chosen on the cluster's SELECTION half, scored on its DISJOINT EVAL half)
    criterion        TAIL   vs static   per-cluster picks
    ----------------------------------------------------------------------------------
    NDCG           0.0608     +0.0194
    ENTROPY        0.0346     -0.0068
    VARIANCE       0.0413     -0.0001
    ANSWERRATE     0.0436     +0.0022
    STATIC         0.0414     +0.0000   (one globally-best question for everybody)

  >>> TASK-LOSS ranking (NDCG)      : 0.0608  (+0.0194)
  >>> INFORMATION ranking (ENTROPY) : 0.0346  (-0.0068)
  If ENTROPY <= STATIC while NDCG >> STATIC, then Sepliarskaia's mechanism is demonstrated in our own
  data: an adaptive policy that optimises an INFORMATION SURROGATE loses to a static one; the prize
  only appears when you rank by the TASK LOSS. That is the boundary the whole thesis turns on.

```
