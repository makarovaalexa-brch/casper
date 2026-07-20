# Greedy elicitation TREE on pb2 — 150k users, depth 6 (2026-07-15)

**What:** a greedy decision tree on the FROZEN pb2 set-encoder recommender (0.4852 full / 0.3295 tail).
= Golbandi/Koren (WSDM'11) but (1) node predictor = the 0.485 self-attention recommender, not an item mean;
(2) graded, BUNDLED answers {disliked 1-2*, meh 3*, liked 4*, loved 5*}; (3) split criterion = TAIL NDCG@10,
a RANKING loss (the extension Koren's own sec:8 flagged as the open hard part). Cheap because within a node
every user shares the belief -> one fold per (question, answer-bucket), shared across all users in the node.
`scripts/tree_grow.py`. ALL 150,239 users, ALL 800 bank candidates per node, credit-neutral masking (Paper B).

## THE NUMBERS (tail NDCG@10, credit-neutral, disjoint per-user targets)
```
turn   ADAPTIVE   STATIC     prize (adaptive-static, paired 95% CI)
base   0.0350     --
1      0.0408     0.0408     (Q1 identical: cold state is shared -> Q1 is static for everyone)
2      0.0422     0.0408     +0.0014 +/- 0.0004     <- static Q2 adds NOTHING
3      0.0456     0.0421     +0.0035 +/- 0.0005     <- significant; prize GROWS with depth
4      0.0479     (frozen)   adaptive-only past here
5      0.0513     (frozen)
6      0.0555     (frozen)   +0.0205 over base = +59% RELATIVE after 6 questions
```
Q1 = **Braveheart** (cat 107). Stopped at depth 6 by the 4h wall-clock (still climbing, not plateaued).
Saved: `.cache/tree_grow_result.npz` (per-user adaptive/static, depth_stats). 150,231 users w/ >=1 tail target.

## TWO CLEAN FINDINGS
1. **Adaptivity COMPOUNDS with depth.** Adaptive climbs every turn (0.0408->0.0555); static STALLS
   (0.0408->0.0408->0.0421 — a 2nd and 3rd FIXED question barely move it). The matched-depth prize widens
   +0.0014 -> +0.0035 as the tree descends. This is the "adaptivity accumulates" argument, measured.
2. **The niche-polariser mechanism reproduces per-node** (= the Jul-14 probe, now on the real model):
   - the huge homogeneous "liked Braveheart" bucket (115k users, 77%) is always probed with a NICHE film
     (Tangled rank 668/800, then Lost in Translation 174, ...) — a blockbuster wouldn't split it;
   - EXTREME answers are more informative: loving/disliking Braveheart yields per-node value ~0.06-0.07 vs
     the default "liked" ~0.04 (those groups are rare and distinctive);
   - coarse-to-fine is unforced: every path runs blockbuster-anchor -> mid-catalog (asked ranks 21-712, almost
     never top-20).

### Example coarse-to-fine chains (Q1 = Braveheart)
- loved -> Clear and Present Danger -> liked -> The Crow -> liked -> The Social Network -> liked -> Birdman
- liked -> Tangled -> liked -> Lost in Translation -> liked -> Star Trek Into Darkness -> liked -> Terminator
- meh   -> Fargo -> liked -> American President ;  disliked -> Dr. Strangelove (val 0.053)

## ⚠ HONESTY CAVEAT — the matched-depth prize is +0.0035 (depth 3), NOT +0.0205
To go deep in 4h I froze the STATIC arm at depth 3 (computing a full static frontier every depth would have
halved the reachable depth). So:
- **The clean, matched-depth, significant claim is: prize = +0.0035 +/- 0.0005 at depth 3.**
- The +0.0205-over-base at depth 6 is ADAPTIVE lift, NOT a prize — there is no depth-6 static counterfactual.
  Static was near-flat through depth 3 (+0.0013 over two turns), so a deep static would likely stay low and the
  true deep prize is probably much larger than +0.0035 — but that MUST be measured, not extrapolated.
- **TODO before any paper number: run a STATIC-only arm to depth 6** (single fixed questionnaire, cheap) so the
  prize is reported at matched depth all the way down.

## WHERE THIS UNDERPERFORMS, AND WHY THAT MOTIVATES PART B
The item-Q1 tree's prize is modest because an ITEM rating is a WEAK splitter: bundling puts 77% of users in one
"liked" bucket, so the top split barely raises homogeneity. The Jul-14 probe got +0.0194 tail because Q1 was a
GENRE question (18 semantic, ~even clusters). **=> the first question should be a CONCEPT, not an item.** That
is Part B (`set_mn.py pb`, training the concept channel on pb2). Then re-run this tree with concept Q1
(concept-Q1 tree) — expected to recover the large prize. See [[policy-is-golbandi-tree-on-pb2]].

## POSITIONING vs GOLBANDI (author decision Jul 15: run it as a baseline)
Golbandi is our ANCESTOR, not a rival to hide from. Implement his squared-error ternary tree + node-mean
predictor as a baseline arm; our tree = his structure with a strong recommender and a ranking-loss split (the
efficient-ranking-tree Koren named as open). `BASELINES_TO_RUN.md` has the verified Golbandi details.
