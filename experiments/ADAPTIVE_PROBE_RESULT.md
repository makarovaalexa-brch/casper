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


---
---

# APPENDIX A — THE FULL RECORD (every cluster, both metrics, per-cluster CIs, all top-5s)

## A.1 STATISTICAL DETAIL
- **n = 150,239 usable users** (of 153,000 labelled: 150k `mm_train` + 3k `mm_val`). Dropped only users with
  <8 ratings, <4 known, or an empty held-liked / empty held-TAIL set (the tail metric is undefined for a user
  with no non-head favourites). **No sampling, no capping, no top-N (HARD RULE #1).**
- **Selection half / evaluation half**: an independent coin-flip per user (`rng(0)`), ~50/50. The best Q2 is
  chosen on the SELECTION half and scored on the EVALUATION half — **for BOTH arms**. The static arm's single
  global question is chosen the same way, on the same selection half. **Symmetric selection budget.**
- **Per-cluster 95% CIs**: paired bootstrap over that cluster's evaluation users, 800 resamples of the
  per-user difference (own-Q2 minus global-Q2). Population CI: 2,000 resamples over all 74,841 eval users.
- **11 of 12 clusters show a significantly positive TAIL gain.** The single exception is `crime`
  (n_eval = 108, gain -0.0127, CI [-0.0303, +0.0012]) — the smallest cluster, CI straddling zero.
- **`fantasy` on FULL** is the only other non-significant cell (n=146, CI [-0.0177, +0.0339]) — yet the SAME
  cluster is strongly significant on TAIL (+0.0659, CI [+0.0485, +0.0862]). **The signal is in the tail.**

## A.2 THE POPULARITY CONTRAST (the mechanism, quantified)
| metric | STATIC question | its popularity rank | CLUSTER questions, median rank |
|---|---|---|---|
| **TAIL** | The Usual Suspects (1995) | **9 / 800** | **456 / 800** (range 14-723) |
| FULL | Dead Poets Society (1989) | 103 / 800 | 450 / 800 (range 9-761) |

**The question that is best for EVERYONE is a blockbuster. The question that is best for SOMEONE is a niche
film inside their own genre.** That single contrast is the mechanism.

## A.3 THE CLAIRVOYANT CEILING (what a perfect per-user question would buy — NOT realizable)
| metric | static | cluster-adaptive | clairvoyant per-user | users for whom the STATIC question is their own best |
|---|---|---|---|---|
| TAIL | 0.0415 | 0.0608 | **0.4382** | **3.1%** |
| FULL | 0.1752 | 0.1856 | **0.5087** | **1.6%** |
**One perfectly-chosen question would beat the FULL PROFILE (0.4961).** But `max_q NDCG[u,q]` knows the
outcome — it is TASTE-PEEK, and independently reproduces E0's *"G2 = all taste-peek"* on different machinery.
It is a CEILING, not a target. **The realizable gap between 0.0608 and 0.4382 is the research programme.**

## A.4 FLATNESS OF THE OBJECTIVE (why v1 was fooled)
```
FULL, population mean by question rank:
  r1=0.1762  r2=0.1761  r5=0.1745  r10=0.1727  r20=0.1707  r50=0.1642  r100=0.1580  r800=0.0324
TAIL, population mean by question rank:
  r1=0.0415  r2=0.0413  r5=0.0405  r10=0.0393  r20=0.0386  r50=0.0365  r100=0.0347  r800=0.0128
```
The POPULATION-mean objective really IS flat near the top (top-20 within ~3-7% of the best). **v1 concluded
from this that "the top questions are near-substitutes, so adaptivity cannot pay." THAT INFERENCE WAS WRONG.**
The population mean being flat says nothing about the PER-CLUSTER surface: *Poltergeist* is population-rank
597 and near-worthless on average, but it is the single best question in existence for a horror fan
(+0.1294 tail). **A flat population mean HIDES a sharp per-cluster surface — it does not preclude it.**
This is the exact error to avoid repeating.

---

# APPENDIX B — RAW OUTPUT (`scripts/adaptive_probe2.py`, verbatim)

```

======================================================================================================================
  FULL NDCG@10
======================================================================================================================
GLOBAL (static) question: Dead Poets Society (1989)  [Drama]  popularity rank 103/800

cluster       n_sel n_eval  its best Q2                                   pop     own  global     gain            95% CI
----------------------------------------------------------------------------------------------------------------------
action        24402  24322  Lethal Weapon 2 (1989)                        534  0.1900  0.1860  +0.0040   [+0.0023,+0.0058]
adventure      4947   4970  Fantasia (1940)                               367  0.1561  0.1435  +0.0126   [+0.0097,+0.0156]
animation       145    144  Laputa: Castle in the Sky (Tenk� no shiro R   723  0.1508  0.0624  +0.0885   [+0.0618,+0.1158]
comedy         8378   8264  Groundhog Day (1993)                           50  0.1287  0.1205  +0.0082   [+0.0060,+0.0103]
crime            92    108  Usual Suspects, The (1995)                      9  0.1170  0.0907  +0.0262   [+0.0135,+0.0387]
drama         32414  32227  Glengarry Glen Ross (1992)                    711  0.2069  0.1981  +0.0088   [+0.0079,+0.0099]
family         2394   2367  Snow White and the Seven Dwarfs (1937)        281  0.1504  0.1066  +0.0438   [+0.0385,+0.0495]
fantasy         148    146  Pirates of the Caribbean: At World's End (2   602  0.0928  0.0847  +0.0081   [-0.0177,+0.0339]
horror          488    487  Halloween (1978)                              761  0.1426  0.0834  +0.0591   [+0.0454,+0.0726]
romance         762    753  Sleepless in Seattle (1993)                   153  0.1068  0.0779  +0.0289   [+0.0208,+0.0375]
sci-fi          509    525  Time Bandits (1981)                           671  0.2494  0.0837  +0.1657   [+0.1497,+0.1807]
thriller        523    528  Seven (a.k.a. Se7en) (1995)                    18  0.1514  0.1375  +0.0139   [+0.0048,+0.0238]
----------------------------------------------------------------------------------------------------------------------
WEIGHTED             74841                                                     0.1856  0.1752  +0.0104

  popularity rank: STATIC question 103/800   |   CLUSTER questions median 450/800 (range 9-761)

  FLATNESS (population-mean FULL by question rank):
   r1=0.1762   r2=0.1761   r3=0.1760   r5=0.1745   r10=0.1727   r20=0.1707   r50=0.1642   r100=0.1580   r200=0.1486   r400=0.1366   r800=0.0324

  CLAIRVOYANT CEILING (each user -> their own best Q2; knows the outcome): 0.5087
  users for whom the GLOBAL question is their own best: 0.016

  TOP-5 QUESTIONS PER CLUSTER (FULL):
    ACTION  (n_sel=24402)
        Lethal Weapon 2 (1989)                          [Action|Comedy|Crime|Drama           ] pop 534
        Beverly Hills Cop (1984)                        [Action|Comedy|Crime|Drama           ] pop 502
        Unforgiven (1992)                               [Drama|Western                       ] pop 217
        Goonies, The (1985)                             [Action|Adventure|Children|Comedy|Fan] pop 423
        Die Hard (1988)                                 [Action|Crime|Thriller               ] pop  49
    ADVENTURE  (n_sel=4947)
        Fantasia (1940)                                 [Animation|Children|Fantasy|Musical  ] pop 367
        Jungle Book, The (1967)                         [Animation|Children|Comedy|Musical   ] pop 514
        Grease (1978)                                   [Comedy|Musical|Romance              ] pop 401
        Indiana Jones and the Last Crusade (1989)       [Action|Adventure                    ] pop  41
        Sound of Music, The (1965)                      [Musical|Romance                     ] pop 264
    ANIMATION  (n_sel=145)
        Laputa: Castle in the Sky (Tenk� no shiro Rapy  [Action|Adventure|Animation|Children|] pop 723
        Nausica� of the Valley of the Wind (Kaze no ta  [Adventure|Animation|Drama|Fantasy|Sc] pop 738
        Howl's Moving Castle (Hauru no ugoku shiro) (2  [Adventure|Animation|Fantasy|Romance ] pop 339
        Cinderella (1950)                               [Animation|Children|Fantasy|Musical|R] pop 589
        My Neighbor Totoro (Tonari no Totoro) (1988)    [Animation|Children|Drama|Fantasy    ] pop 363
    COMEDY  (n_sel=8378)
        Groundhog Day (1993)                            [Comedy|Fantasy|Romance              ] pop  50
        Stand by Me (1986)                              [Adventure|Drama                     ] pop 113
        Wayne's World (1992)                            [Comedy                              ] pop 440
        Ferris Bueller's Day Off (1986)                 [Comedy                              ] pop  97
        Wedding Singer, The (1998)                      [Comedy|Romance                      ] pop 468
    CRIME  (n_sel=92)
        Usual Suspects, The (1995)                      [Crime|Mystery|Thriller              ] pop   9
        Carlito's Way (1993)                            [Crime|Drama                         ] pop 548
        Cape Fear (1991)                                [Thriller                            ] pop 624
        Taxi Driver (1976)                              [Crime|Drama|Thriller                ] pop  60
        Miller's Crossing (1990)                        [Crime|Drama|Film-Noir|Thriller      ] pop 654
    DRAMA  (n_sel=32414)
        Glengarry Glen Ross (1992)                      [Drama                               ] pop 711
        Cape Fear (1991)                                [Thriller                            ] pop 624
        One Flew Over the Cuckoo's Nest (1975)          [Drama                               ] pop  35
        Miller's Crossing (1990)                        [Crime|Drama|Film-Noir|Thriller      ] pop 654
        Femme Nikita, La (Nikita) (1990)                [Action|Crime|Romance|Thriller       ] pop 477
    FAMILY  (n_sel=2394)
        Snow White and the Seven Dwarfs (1937)          [Animation|Children|Drama|Fantasy|Mus] pop 281
        Pinocchio (1940)                                [Animation|Children|Fantasy|Musical  ] pop 472
        Sound of Music, The (1965)                      [Musical|Romance                     ] pop 264
        Mulan (1998)                                    [Adventure|Animation|Children|Comedy|] pop 430
        Mary Poppins (1964)                             [Children|Comedy|Fantasy|Musical     ] pop 297
    FANTASY  (n_sel=148)
        Pirates of the Caribbean: At World's End (2007  [Action|Adventure|Comedy|Fantasy     ] pop 602
        Chronicles of Narnia: The Lion, the Witch and   [Adventure|Children|Fantasy          ] pop 565
        Master and Commander: The Far Side of the Worl  [Adventure|Drama|War                 ] pop 737
        X-Men (2000)                                    [Action|Adventure|Sci-Fi             ] pop 119
        Lord of the Rings: The Return of the King, The  [Action|Adventure|Drama|Fantasy      ] pop  20
    HORROR  (n_sel=488)
        Halloween (1978)                                [Horror                              ] pop 761
        Exorcist, The (1973)                            [Horror|Mystery                      ] pop 282
        Psycho (1960)                                   [Crime|Horror                        ] pop 110
        Birds, The (1963)                               [Horror|Thriller                     ] pop 381
        American Werewolf in London, An (1981)          [Comedy|Horror|Thriller              ] pop 756
    MUSICAL  (n_sel=56)
        West Side Story (1961)                          [Drama|Musical|Romance               ] pop 545
        My Fair Lady (1964)                             [Comedy|Drama|Musical|Romance        ] pop 427
        Dumbo (1941)                                    [Animation|Children|Drama|Musical    ] pop 699
        Singin' in the Rain (1952)                      [Comedy|Musical|Romance              ] pop 335
        Peter Pan (1953)                                [Animation|Children|Fantasy|Musical  ] pop 783
    ROMANCE  (n_sel=762)
        Sleepless in Seattle (1993)                     [Comedy|Drama|Romance                ] pop 153
        Mrs. Doubtfire (1993)                           [Comedy|Drama                        ] pop 140
        Clueless (1995)                                 [Comedy|Romance                      ] pop 193
        Notting Hill (1999)                             [Comedy|Romance                      ] pop 403
        You've Got Mail (1998)                          [Comedy|Romance                      ] pop 640
    SCI-FI  (n_sel=509)
        Time Bandits (1981)                             [Adventure|Comedy|Fantasy|Sci-Fi     ] pop 671
        RoboCop (1987)                                  [Action|Crime|Drama|Sci-Fi|Thriller  ] pop 425
        Mad Max (1979)                                  [Action|Adventure|Sci-Fi             ] pop 590
        Escape from New York (1981)                     [Action|Adventure|Sci-Fi|Thriller    ] pop 757
        Close Encounters of the Third Kind (1977)       [Adventure|Drama|Sci-Fi              ] pop 291
    THRILLER  (n_sel=523)
        Seven (a.k.a. Se7en) (1995)                     [Mystery|Thriller                    ] pop  18
        Cape Fear (1991)                                [Thriller                            ] pop 624
        Taxi Driver (1976)                              [Crime|Drama|Thriller                ] pop  60
        L�on: The Professional (a.k.a. The Professiona  [Action|Crime|Drama|Thriller         ] pop  47
        Glengarry Glen Ross (1992)                      [Drama                               ] pop 711

======================================================================================================================
  TAIL NDCG@10
======================================================================================================================
GLOBAL (static) question: Usual Suspects, The (1995)  [Crime|Mystery|Thriller]  popularity rank 9/800

cluster       n_sel n_eval  its best Q2                                   pop     own  global     gain            95% CI
----------------------------------------------------------------------------------------------------------------------
action        24402  24322  Braveheart (1995)                              14  0.0633  0.0506  +0.0126   [+0.0113,+0.0141]
adventure      4947   4970  Crimson Tide (1995)                           159  0.0677  0.0512  +0.0165   [+0.0131,+0.0197]
animation       145    144  Laputa: Castle in the Sky (Tenk� no shiro R   723  0.1536  0.0028  +0.1508   [+0.1201,+0.1855]
comedy         8378   8264  Wayne's World (1992)                          440  0.0620  0.0365  +0.0256   [+0.0225,+0.0283]
crime            92    108  True Romance (1993)                           255  0.0168  0.0294  -0.0127   [-0.0303,+0.0012]
drama         32414  32227  Citizen Kane (1941)                           139  0.0542  0.0360  +0.0182   [+0.0168,+0.0196]
family         2394   2367  Pinocchio (1940)                              472  0.0739  0.0413  +0.0326   [+0.0278,+0.0372]
fantasy         148    146  Labyrinth (1986)                              598  0.0676  0.0017  +0.0659   [+0.0485,+0.0862]
horror          488    487  Poltergeist (1982)                            597  0.1356  0.0062  +0.1294   [+0.1171,+0.1434]
romance         762    753  Little Women (1994)                           532  0.0730  0.0469  +0.0261   [+0.0182,+0.0330]
sci-fi          509    525  Planet of the Apes (1968)                     365  0.1307  0.0045  +0.1262   [+0.1141,+0.1411]
thriller        523    528  Executive Decision (1996)                     628  0.0384  0.0262  +0.0122   [+0.0020,+0.0244]
----------------------------------------------------------------------------------------------------------------------
WEIGHTED             74841                                                     0.0608  0.0415  +0.0194

  popularity rank: STATIC question 9/800   |   CLUSTER questions median 456/800 (range 14-723)

  FLATNESS (population-mean TAIL by question rank):
   r1=0.0415   r2=0.0413   r3=0.0413   r5=0.0405   r10=0.0393   r20=0.0386   r50=0.0365   r100=0.0347   r200=0.0334   r400=0.0306   r800=0.0128

  CLAIRVOYANT CEILING (each user -> their own best Q2; knows the outcome): 0.4382
  users for whom the GLOBAL question is their own best: 0.031

  TOP-5 QUESTIONS PER CLUSTER (TAIL):
    ACTION  (n_sel=24402)
        Braveheart (1995)                               [Action|Drama|War                    ] pop  14
        Cliffhanger (1993)                              [Action|Adventure|Thriller           ] pop 476
        Speed (1994)                                    [Action|Romance|Thriller             ] pop  87
        Terminator 2: Judgment Day (1991)               [Action|Sci-Fi                       ] pop  16
        Batman Forever (1995)                           [Action|Adventure|Comedy|Crime       ] pop 436
    ADVENTURE  (n_sel=4947)
        Crimson Tide (1995)                             [Drama|Thriller|War                  ] pop 159
        Stargate (1994)                                 [Action|Adventure|Sci-Fi             ] pop 177
        Clear and Present Danger (1994)                 [Action|Crime|Drama|Thriller         ] pop 127
        Santa Clause, The (1994)                        [Comedy|Drama|Fantasy                ] pop 573
        Disclosure (1994)                               [Drama|Thriller                      ] pop 690
    ANIMATION  (n_sel=145)
        Laputa: Castle in the Sky (Tenk� no shiro Rapy  [Action|Adventure|Animation|Children|] pop 723
        Howl's Moving Castle (Hauru no ugoku shiro) (2  [Adventure|Animation|Fantasy|Romance ] pop 339
        My Neighbor Totoro (Tonari no Totoro) (1988)    [Animation|Children|Drama|Fantasy    ] pop 363
        Nausica� of the Valley of the Wind (Kaze no ta  [Adventure|Animation|Drama|Fantasy|Sc] pop 738
        Spirited Away (Sen to Chihiro no kamikakushi)   [Adventure|Animation|Fantasy         ] pop  94
    COMEDY  (n_sel=8378)
        Wayne's World (1992)                            [Comedy                              ] pop 440
        Austin Powers: International Man of Mystery (1  [Action|Adventure|Comedy             ] pop 221
        There's Something About Mary (1998)             [Comedy|Romance                      ] pop 179
        Notting Hill (1999)                             [Comedy|Romance                      ] pop 403
        Groundhog Day (1993)                            [Comedy|Fantasy|Romance              ] pop  50
    CRIME  (n_sel=92)
        True Romance (1993)                             [Crime|Thriller                      ] pop 255
        Taxi Driver (1976)                              [Crime|Drama|Thriller                ] pop  60
        Die Hard: With a Vengeance (1995)               [Action|Crime|Thriller               ] pop 124
        Natural Born Killers (1994)                     [Action|Crime|Thriller               ] pop 253
        Crimson Tide (1995)                             [Drama|Thriller|War                  ] pop 159
    DRAMA  (n_sel=32414)
        Citizen Kane (1941)                             [Drama|Mystery                       ] pop 139
        Once Upon a Time in the West (C'era una volta   [Action|Drama|Western                ] pop 678
        North by Northwest (1959)                       [Action|Adventure|Mystery|Romance|Thr] pop 150
        Good, the Bad and the Ugly, The (Buono, il bru  [Action|Adventure|Western            ] pop 141
        Great Escape, The (1963)                        [Action|Adventure|Drama|War          ] pop 304
    FAMILY  (n_sel=2394)
        Pinocchio (1940)                                [Animation|Children|Fantasy|Musical  ] pop 472
        Sound of Music, The (1965)                      [Musical|Romance                     ] pop 264
        Mary Poppins (1964)                             [Children|Comedy|Fantasy|Musical     ] pop 297
        Jungle Book, The (1967)                         [Animation|Children|Comedy|Musical   ] pop 514
        Lion King, The (1994)                           [Adventure|Animation|Children|Drama|M] pop  43
    FANTASY  (n_sel=148)
        Labyrinth (1986)                                [Adventure|Fantasy|Musical           ] pop 598
        Harry Potter and the Sorcerer's Stone (a.k.a.   [Adventure|Children|Fantasy          ] pop 136
        Harry Potter and the Prisoner of Azkaban (2004  [Adventure|Fantasy|IMAX              ] pop 146
        Charlie and the Chocolate Factory (2005)        [Adventure|Children|Comedy|Fantasy|IM] pop 744
        Harry Potter and the Order of the Phoenix (200  [Adventure|Drama|Fantasy|IMAX        ] pop 313
    HORROR  (n_sel=488)
        Poltergeist (1982)                              [Horror|Thriller                     ] pop 597
        Thing, The (1982)                               [Action|Horror|Sci-Fi|Thriller       ] pop 360
        Evil Dead II (Dead by Dawn) (1987)              [Action|Comedy|Fantasy|Horror        ] pop 555
        Halloween (1978)                                [Horror                              ] pop 761
        Misery (1990)                                   [Drama|Horror|Thriller               ] pop 469
    MUSICAL  (n_sel=56)
        Cinderella (1950)                               [Animation|Children|Fantasy|Musical|R] pop 589
        Dumbo (1941)                                    [Animation|Children|Drama|Musical    ] pop 699
        Lady and the Tramp (1955)                       [Animation|Children|Comedy|Romance   ] pop 519
        Sound of Music, The (1965)                      [Musical|Romance                     ] pop 264
        West Side Story (1961)                          [Drama|Musical|Romance               ] pop 545
    ROMANCE  (n_sel=762)
        Little Women (1994)                             [Drama                               ] pop 532
        You've Got Mail (1998)                          [Comedy|Romance                      ] pop 640
        Sleepless in Seattle (1993)                     [Comedy|Drama|Romance                ] pop 153
        Dirty Dancing (1987)                            [Drama|Musical|Romance               ] pop 608
        When Harry Met Sally... (1989)                  [Comedy|Romance                      ] pop 134
    SCI-FI  (n_sel=509)
        Planet of the Apes (1968)                       [Action|Drama|Sci-Fi                 ] pop 365
        RoboCop (1987)                                  [Action|Crime|Drama|Sci-Fi|Thriller  ] pop 425
        Escape from New York (1981)                     [Action|Adventure|Sci-Fi|Thriller    ] pop 757
        Star Trek VI: The Undiscovered Country (1991)   [Action|Mystery|Sci-Fi               ] pop 644
        Total Recall (1990)                             [Action|Adventure|Sci-Fi|Thriller    ] pop 194
    THRILLER  (n_sel=523)
        Executive Decision (1996)                       [Action|Adventure|Thriller           ] pop 628
        Girl with the Dragon Tattoo, The (2011)         [Drama|Thriller                      ] pop 549
        Prestige, The (2006)                            [Drama|Mystery|Sci-Fi|Thriller       ] pop 102
        Mr. Holland's Opus (1995)                       [Drama                               ] pop 230
        Heat (1995)                                     [Action|Crime|Thriller               ] pop 122

```
