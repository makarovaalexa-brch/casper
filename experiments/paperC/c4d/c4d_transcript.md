# C4D -- the synthesised-interview transcript

*ONE cold-start interview walked down Paper B's cold tree, in which every question at a C1-final EMITTING state is the policy's own off-catalogue direction rendered into English, shown beside the catalogue question the tree would have asked at that state.*

- adapter: `v2` (questions targets), decoding **bestofk_lm** (greedy shown beside it)
- emission space: **target** -- the state's BLEND TARGET, not the policy's emission (labelled in every row)
- cohort: 10,000 validation users, path mode `majority`; stop: --max_turns 6 reached
- answers, folds and every NDCG below are **under a declared-circular answer model** and carry no value claim

> ⚠ SPACE PROBE: exact top-1 agreement between the fed point's nearest bank entry and the emission's own nearest entry in m7dj's space is **0.000**, and it is 0.000 for all three constructions -- including `target`, which is the right direction in the adapter's own bank by construction -- so exact-index equality over a 19,390-entry bank is too strict to separate them. Re-run to bank the neighbourhood arm that does.

| turn | state | n | synthesised question (policy's own direction, in English) | the catalogue question the tree would ask | folded | full@10 | tail@10 | top-5 after this turn |
|---|---|---|---|---|---|---|---|---|
| 1 | `root` | 10,000 | **Are you in the mood for something action films?**<br/>*greedy:* Do you enjoy action films?<br/>read back: action, action packed, feel good movie, fun movie | Do you like better than expected films? | action, action packed, feel good movie, fun movie | 0.16797 | 0.04822 | Shawshank Redemption, The (1994); Forrest Gump (1994); Pulp Fiction (1994); Schindler's List (1993); Silence of the Lambs, The (1991) |
| 2 | `root.dislike` | 4,814 | **Do you like it when a film is suspense?**<br/>*greedy:* Are you in the mood for something dark?<br/>read back: cinematography, great cinematography, suspense, suspenseful | Do you like dark humor films? | action, action packed, feel good movie, fun movie, cinematography, great cinematography, suspense, suspenseful | 0.20178 | 0.05977 | Shawshank Redemption, The (1994); Forrest Gump (1994); Schindler's List (1993); Pulp Fiction (1994); Silence of the Lambs, The (1991) |
| 3 | `root.dislike.dislike` | 3,815 | **How do you feel about sci-fi movies?**<br/>*greedy:* Do you enjoy sci-fi films?<br/>read back: intelligent sci-fi, sci fi, sci-fi, scifi | Do you like alien films? | action, action packed, feel good movie, fun movie, cinematography, great cinematography, suspense, suspenseful, intelligent sci-fi, sci fi, sci-fi, scifi | 0.20698 | 0.05470 | Shawshank Redemption, The (1994); Matrix, The (1999); Forrest Gump (1994); Star Wars: Episode IV - A New Hope (1977); Pulp Fiction (1994) |
| 4 | `root.dislike.dislike.dislike` | 2,409 | **Do you like it when a film is movie?**<br/>*greedy:* Are you in the mood for something romantic?<br/>read back: feel good movie, independent film, movie business, oscar (best cinematography) | Have you seen Star Wars: Episode V - The Empire Strikes Back (1980), and did you like it? | action, action packed, feel good movie, fun movie, cinematography, great cinematography, suspense, suspenseful, intelligent sci-fi, sci fi, sci-fi, scifi, independent film, movie business, oscar (best cinematography) | 0.23399 | 0.06310 | Shawshank Redemption, The (1994); Matrix, The (1999); Schindler's List (1993); Pulp Fiction (1994); Forrest Gump (1994) |
| 5 | `root.dislike.dislike.dislike.noanswer` | 1,335 | **Do you like it when a film is detective drama?**<br/>*greedy:* Are you in the mood for something crime movies?<br/>read back: detective, film noir, great cinematography, private detective | Have you seen Crimson Tide (1995), and did you like it? | action, action packed, feel good movie, fun movie, cinematography, great cinematography, suspense, suspenseful, intelligent sci-fi, sci fi, sci-fi, scifi, independent film, movie business, oscar (best cinematography), detective, film noir, private detective | 0.16765 | 0.04970 | Shawshank Redemption, The (1994); Forrest Gump (1994); Schindler's List (1993); Pulp Fiction (1994); Matrix, The (1999) |
| 6 | `root.dislike.dislike.dislike.noanswer.noanswer` | 1,108 | **How do you feel about children films?**<br/>*greedy:* Are you in the mood for something film?<br/>read back: childhood, children, feel good movie, kids | Have you seen Star Wars: Episode VI - Return of the Jedi (1983), and did you like it? | action, action packed, feel good movie, fun movie, cinematography, great cinematography, suspense, suspenseful, intelligent sci-fi, sci fi, sci-fi, scifi, independent film, movie business, oscar (best cinematography), detective, film noir, private detective, childhood, children, kids | 0.15306 | 0.05064 | Shawshank Redemption, The (1994); Forrest Gump (1994); Star Wars: Episode IV - A New Hope (1977); Schindler's List (1993); Matrix, The (1999) |

### the same turns, the TREE's own column (the honest comparator)

| turn | state | tree question | full@10 | tail@10 | base (m=0) full |
|---|---|---|---|---|---|
| 1 | `root` | Do you like better than expected films? | 0.16336 | 0.04444 | 0.15687 |
| 2 | `root.dislike` | Do you like dark humor films? | 0.18535 | 0.04368 | 0.18024 |
| 3 | `root.dislike.dislike` | Do you like alien films? | 0.19026 | 0.05363 | 0.18692 |
| 4 | `root.dislike.dislike.dislike` | Have you seen Star Wars: Episode V - The Empire Strikes Back (1980), and did you like it? | 0.21926 | 0.06131 | 0.22111 |
| 5 | `root.dislike.dislike.dislike.noanswer` | Have you seen Crimson Tide (1995), and did you like it? | 0.17108 | 0.05956 | 0.16048 |
| 6 | `root.dislike.dislike.dislike.noanswer.noanswer` | Have you seen Star Wars: Episode VI - Return of the Jedi (1983), and did you like it? | 0.13962 | 0.04077 | 0.13637 |

---

**What this may not claim.** NO VALUE CLAIM: every NDCG row runs under a declared-circular answer model and carries at most a RELATIVE reading (AUTHOR RULING 2026-08-25) - nothing at states outside C1-final's 57 - nothing about the item channel - not that the questions are good because they read well - not that a human would answer them.
