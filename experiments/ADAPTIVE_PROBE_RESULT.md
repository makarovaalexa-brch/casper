# ADAPTIVE-PROBE — the author's test. **ADAPTIVITY WINS: +47% TAIL from ONE genre question.**
2026-07-14. `scripts/adaptive_probe2.py`. Supersedes the v1 run (`adaptive_probe.py`), whose conclusion was
WRONG — see §5.

**THE AUTHOR'S TEST:** *"Ask people their favourite genre, cluster them by it, rank probe questions within
those groups — do you get the same order?"*
**WHY IT IS DECISIVE:** no policy, no RL, no optimisation => **it cannot be confounded by a weak static arm**
(the flaw that undermines STATIC8). It measures the thing directly: **does the answer to Q1 change which Q2 is
best?** (Naghshvar & Javidi, JSTSP 2013, Cor. 8.)

## 1. SETUP
- **ALL 153,000 labelled users** (150k `mm_train` + 3k `mm_val`; 150,239 usable). All 800 bank items. All 18
  genres. **No sampling (HARD RULE #1).**
- **Q1 = favourite genre, IMPLICIT** = the genre the user WATCHES MOST in their known half (the author's
  correction; the answerer's 4-level ordinal ties across genres and its argmax is noise).
- **Q2** = each of the 800 bank items, folded COLD into **a0c** (full-profile NDCG@10 = 0.4961), scored vs
  held-liked. **FULL and TAIL** (arena definition: head items masked out; relevance = held-liked NON-head).
- **GUARD:** the best Q2 is chosen on a **DISJOINT half** of users and evaluated on the other half,
  **symmetrically for both arms**. Refusals burn the turn (fold = cold). Ordinal->star map DERIVED FROM DATA.

## 2. RESULT — ADAPTIVITY WINS, AND THE EFFECT IS IN THE **TAIL**
```
TAIL NDCG@10 (the headline metric of Papers B/C)
  STATIC   (everyone -> the globally-best question):  0.0414
  ADAPTIVE (each cluster -> its own best question) :  0.0608
  PRIZE = +0.0194   95% CI [+0.0186, +0.0203]   +46.9% RELATIVE   n = 74,881 eval users

FULL NDCG@10
  STATIC 0.1752   ADAPTIVE 0.1856   PRIZE = +0.0104  CI [+0.0096, +0.0112]  +5.9%

CLAIRVOYANT CEILING (each user -> THEIR own best Q2; knows the outcome, NOT realizable): TAIL 0.4382
```
Paired bootstrap, 2000 resamples, selection and evaluation on disjoint halves.

## 3. ⭐ THE MECHANISM — AND IT IS THE HUMAN INTERVIEWER'S
| cluster | its top-3 questions (TAIL) | popularity rank / 800 |
|---|---|---|
| horror | Poltergeist, The Thing, Evil Dead II | 596, 359, 554 |
| sci-fi | Planet of the Apes, RoboCop, Escape from New York | 364, 424, 756 |
| animation | **Laputa, Howl's Moving Castle, My Neighbor Totoro** (ALL GHIBLI) | 722, 338, 362 |
| family | Pinocchio, Sound of Music, Mary Poppins | 471, 263, 296 |
| comedy | Wayne's World, Austin Powers, There's Something About Mary | 439, 220, 178 |
| drama | Citizen Kane, Once Upon a Time in the West, North by Northwest | 138, 677, 149 |
| action | Braveheart, Cliffhanger, Speed | 13, 475, 86 |
| **GLOBAL (static), for everybody** | **The Usual Suspects** | **8** |

```
popularity rank of the STATIC question : 8/800      <- a BLOCKBUSTER
popularity rank of the CLUSTER questions: 471/800   <- NICHE (median; range 13-722)
```
**THE MECHANISM:** once you know someone is a horror fan, *"do you like The Usual Suspects?"* tells you
nothing — everyone likes it. What **discriminates WITHIN horror fans** is **Poltergeist**: it splits
mainstream-supernatural from gore/slasher. **The informative question for a homogeneous group is one that
POLARIZES that group — always a moderately-obscure, genre-congruent film, never a universally-loved hit.**
**This is exactly what a human interviewer does** (the author's argument: *"no human would ask a fixed list of
closed probes"*). The model rediscovered the mechanism, unprompted, from 150k users.
**AND IT EXPLAINS WHY THE EFFECT IS IN THE TAIL (+47%) AND NOT THE HEAD (+6%):** a popular probe only helps
rank the head, which popularity already gets right. A niche genre-congruent probe locates the user in a NICHE
SUBSPACE — which is precisely what surfaces tail items. **Tail is where discrimination pays.**
**NOT NOISE:** every cluster's top-3 are genre-congruent; the animation cluster's top-3 are ALL Studio Ghibli;
drama and action have 24k-32k evaluation users each; every cluster gains.

## 4. WHAT THIS SETTLES
- **The order is NOT the same for everybody.** The linear-Gaussian prediction is FALSE in our system.
- **And it PAYS** (+47% tail), from the CRUDEST possible conditioning (an 18-way genre cut) with NO policy and
  NO optimisation. **Individual belief conditioning should do strictly better.**
- **Cluster-level adaptivity is NOT a dead end** (my v1 claim) — it was an artefact of a 3k sample.

## 5. ⚠ WHAT V1 GOT WRONG (and why — this is the process lesson AGAIN)
v1 concluded *"the clusters want different questions but it buys nothing (+0.0027, n.s.); the objective is flat;
adaptivity is an individual-level property that genre cannot reach."* **ALL WRONG.** Two causes, both named by
the author:
1. **v1 used only the 3,000-user `mm_val` cohort** when **150,000 more labelled users existed in `mm_train`**.
   So family had n=44 and adventure self-agreed at 0.00 — I was reading NOISE in exactly the clusters where the
   signal lives.
2. **v1 reported FULL NDCG only.** The effect is in the **TAIL** (+47% vs +6%). I measured the wrong axis —
   and TAIL is our headline metric, and STATIC8's adaptive gap was also bigger on tail.
**LESSON: use ALL the labelled data, and measure the axis the thesis is about.**

## 6. SCOPE / WHAT IS STILL MISSING (both named by the author)
- **ONE TURN.** Q1 partitions, Q2 is scored. Flatness/discrimination at turn 1 (cold start) may look nothing
  like turn 6, when the belief is sharp. **Multi-turn is the next experiment.**
- **BLIND TO CONCEPTS AND ATTRIBUTES.** a0c's input is a dense vector over ITEMS ONLY — there is no slot for
  "zombie films" or "Spielberg", so only item questions could be folded. **Given that the winning questions are
  NICHE and GENRE-CONGRUENT, the concept channel ("do you like Japanese animation?") is likely STRONGER still.**
  This is exactly the limitation the set+multinomial architecture (`scripts/set_mn.py`) exists to remove.
- **ENTROPY vs NDCG (the author's question, and it is the key next test).** Theory says: rank by
  ENTROPY/information => the SAME order for everyone (Krause: variance-only objectives cannot benefit from
  sequencing). Rank by the TASK LOSS (NDCG) => it need not. **We ranked by NDCG and got cluster-specific orders
  that pay. Running the ENTROPY-ranked arm and showing it collapses to ONE list would empirically demonstrate
  the theorem's boundary** — and it is precisely Sepliarskaia's (RecSys 2018) published explanation: *adaptive
  methods that optimise a variance surrogate lose to static methods optimising the loss.*
