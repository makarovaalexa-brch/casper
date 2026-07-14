# ADAPTIVE-PROBE — the author's test (2026-07-14)

**THE QUESTION (author, verbatim):** *"If you ask people their favourite genre, cluster them by it, and rank
probe questions by entropy within those groups, will you get the same order of questions?"*

**WHY IT IS DECISIVE:** no policy, no RL, no optimisation => **it cannot be confounded by a weak static arm**
(the flaw that undermines STATIC8). It measures the one thing that matters, directly: **does the answer to Q1
change which Q2 is best?** (= Naghshvar & Javidi, JSTSP 2013, Cor. 8: adaptivity gain > 0 IFF the most
informative question depends on which hypothesis is true.)

**SETUP.** `scripts/adaptive_probe.py`. Q1 = favourite genre, **IMPLICIT** (the genre the user WATCHES MOST in
their known half — the author's correction; the answerer's 4-level ordinal ties across genres and its argmax is
noise). Q2 = each of the **800 bank items**, folded COLD into **a0c** (our strongest recommender, full-profile
NDCG@10 = 0.4961), scored NDCG@10 vs held-liked. **3,000 val users, all 800 candidates, all 18 genres — no
sampling (HARD RULE #1).** Refusals burn the turn (fold = cold). Ordinal->star map DERIVED FROM THE DATA
(hated 1.54 / meh 2.91 / liked 3.86 / loved 4.82), not invented.
**HONESTY GUARD:** the best Q2 is selected on a **DISJOINT half** of users and evaluated on the other half,
**symmetrically for both arms**. Without this the adaptive arm wins by overfitting its own selection.

---

## RESULT 1 — THE ORDER IS **NOT** THE SAME. (The author's question: answered NO.)
| cluster | n_eval | its best Q2 | NDCG(own Q2) | NDCG(global Q2) | gain |
|---|---|---|---|---|---|
| drama | 650 | 468 | 0.2082 | 0.2063 | +0.0019 |
| action | 487 | 216 | 0.1935 | 0.1925 | +0.0010 |
| comedy | 151 | 439 | 0.1335 | 0.1374 | **-0.0040** |
| adventure | 118 | 263 | 0.1298 | 0.1344 | **-0.0046** |
| family | 44 | 429 | 0.1562 | 0.0791 | **+0.0771** |

- **5 clusters -> 5 DISTINCT best questions.**
- **top-10 overlap between clusters: mean 0.030, min 0.000** — their top-10 lists are essentially DISJOINT.
- Spearman rank agreement: mean 0.54, min 0.07.
=> **The linear-Gaussian prediction ("one list for everybody") is FALSE in our system.** Measured directly, no
policy to blame. Consistent with our fold-in being NONLINEAR.

## RESULT 2 — ...AND IT BUYS NOTHING.
```
PAIRED BOOTSTRAP (n=1450 eval users, 5000 resamples)
  adaptive - static = +0.0027   95% CI [-0.0025, +0.0082]   NOT SIGNIFICANT
  P(adaptive > static) per user = 0.406
```
(Cluster-weighted point estimate +0.0067; the bootstrap is the honest number.)

## RESULT 3 — WHY: THE OBJECTIVE IS FLAT NEAR THE TOP
Population-mean NDCG by question rank:
| rank | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 800 |
|---|---|---|---|---|---|---|---|---|
| NDCG | 0.1823 | 0.1811 | 0.1792 | 0.1772 | **0.1755** | 0.1700 | 0.1627 | 0.0325 |
| % of best | 100 | 99.3 | 98.3 | 97.2 | **96.3** | 93.2 | 89.3 | 17.8 |

**The top ~20 questions are NEAR-SUBSTITUTES (96% of the best).** So the clusters are swapping between members
of a large equivalence class of near-optimal questions. **RANK DISAGREEMENT DOES NOT IMPLY VALUE.**
*Every argument in this debate — mine and the literature's — conflated those two.*

## RESULT 4 ⭐ — THE CEILING IS ENORMOUS, AND IT IS CLAIRVOYANT
```
PER-USER ORACLE (each user asked THEIR OWN best Q2):  0.5173
GLOBAL-BEST-FOR-ALL (the static questionnaire):        0.1823
CEILING                                               +0.3350
users for whom the global question is optimal:          1.2%
```
**One perfectly-chosen question beats the FULL PROFILE (0.4961).** The static questionnaire is optimal for
**1.2% of users**. So adaptivity's VALUE is not small — **it is vast.**
**BUT THE CEILING IS TAINTED:** `max_q NDCG[u,q]` picks the question that *happens* to rank THIS user's
held-out favourites highest — **it knows the outcome.** This is **TASTE-PEEK**, and it independently reproduces
E0's finding on completely different machinery (E0: *"G2's 0.287 = all taste-peek"*; here: +0.335).

---

## THE HONEST CONCLUSION
> **The clusters want different questions (the order is genuinely user-dependent). But realizable
> genre-conditioned adaptivity at turn 2 buys +0.0027 (CI spans zero), while the clairvoyant ceiling is
> +0.335. The gap between those two numbers IS the problem.**

**This reframes the question.** It is not *"does adaptivity have value"* (it has enormous value) and not
*"is a fixed list optimal"* (it is optimal for 1.2% of users). It is:
> **WHAT OBSERVABLE SIGNAL ACTUALLY LOCATES A USER'S BEST QUESTION?**
**Genre does not.** That is a MEASUREMENT problem, not a theorem — and it is a far better research question
than the one we have been arguing about.

## SCOPE / CAVEATS (do not over-read)
- **Turn 2 only, ITEM questions only, GENRE clusters, a0c.** It does **NOT** contradict STATIC8 (8 turns,
  CONTINUOUS queries, GRADED answers, different ruler). Different regime.
- Only 5 clusters had >=20 users in BOTH halves (genre clusters are coarse and drama/action dominate: 76%).
- comedy/adventure show NEGATIVE gains => real selection noise at n~150. family's +0.077 at n=44 is inside it.
- MDE: the CI half-width is ~0.005, so effects below ~0.005 are undetectable here. STATIC8's +0.038 would have
  been detected easily.

## WHAT TO DO NEXT (in order of value)
1. **Richer conditioning signal than genre.** Genre is a 18-way coarse cut and it captures ~0 of a +0.335
   ceiling. Try: the user's *belief state after Q1* (continuous), k-means clusters in the taste space, or the
   answer to a *better* Q1. **The question is which observable signal locates the user's best question.**
2. **Repeat at deeper turns** (the ceiling may be reachable only after the belief sharpens).
3. **Repeat with CONCEPT and CONTINUOUS Q2** (items may be the wrong channel: the top-20 flatness may be an
   ITEM-channel property).
4. Re-run the flatness curve on the *tail* metric — the family cluster hints the action is on atypical users.
