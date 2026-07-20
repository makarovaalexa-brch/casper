# Policy Optimization Log — beat conc_pop in cold-start (one change at a time)

TARGET to beat (locked ruler: ML-1M, concept-aware enc, ANSWER=geom, realistic cold-start, q8):
- **conc_pop**: FULL 0.315 / TAIL 0.116  (most-frequent concepts; the baseline OURS must match then beat)
- pop_item: 0.307 / 0.085 ; q0: 0.293 / 0.088
- ceilings: conc_gprof (profile) 0.330 / 0.143 ; conc_oracle (test) 0.469 / 0.325

KEY MECHANISM (PART AG/AG2): concept belief saturates at cos~0.83 with u*; ITEMS converge to ~1.0;
**BOTH (items+concepts) breaks the ceiling -> cos 0.887**. So OURS must use the item+concept COMBINATION,
and the right training objective is to **RECONSTRUCT u*** (user embedding), not just held-out item BCE.

## Optimizations tried (one variable each)
| # | change (vs previous) | FULL q8 | TAIL q8 | ans/8 | verdict |
|---|---|---|---|---|---|
| O1 | emit+snap actor, soft-snap, recon-BCE | 0.288 | 0.102 | 6.4 | FAIL (below conc_pop; train/eval mismatch) |
| O2 | + answerability prior (POOL_PRIOR) bias | 0.281 | 0.111 | 7.6 | ~conc_pop (prior-dominated) |
| O3 | scorer + popularity feature | 0.288 | 0.102 | 6.4 | FAIL (below) |
| O4 | + population info-gain feature | 0.288 | 0.102 | 6.4 | FAIL (below) |
| O5 | straight-through (train==eval) | 0.293 | 0.085 | 1.4 | FAIL (collapsed to items) |
| O6 | BC-to-conc_pop floor + BCE finetune | RUN | RUN | | (running) |
| O7 | objective = RECONSTRUCT u* (MSE) | DIVERGED | | | FAIL (MSE explodes w/ straight-through) |
| O7b | RECONSTRUCT u* (1-cos, bounded) from BC floor | ~0.31 | ~0.11 | | flat at concept ceiling (cos stuck 0.81~=conc_pop); finetune didn't improve |
| **conc_mix** | non-learned: interleave popular ITEM + frequent CONCEPT (the COMBINATION) | **0.318** | **0.116** (Rec 0.159) | 5.0 | **>= conc_pop** (full-NDCG +0.003, tail-Rec +0.006) with FEWER answers -> combination validated |

## Queue (to try next, one at a time)
- O8: BC-to-conc_pop floor + **u\*-reconstruction** finetune (combine O6 floor + O7 objective)
- O9: confirm COMBINATION: run OURS with pool = concepts-only vs items-only vs BOTH (show both > each)
- O10: cos(u,u*) reward weighting / add magnitude term; tune TAU
- O11: explore schedule; longer finetune; per-turn intermediate u* reward (dense)
- O12: stronger answerable-item inclusion (so the few fine items get folded)

## Notes
- "match conc_pop must be trivial" (user): O6 BC floor guarantees it; finetune must not regress below.
- gate each: must be >= conc_pop on BOTH full+tail; log every run here.

| O8 | BC floor + u*-cos, LR 5e-4, EXPL 0.3 | ~0.31 | ~0.11 | | stuck at floor |
| O9 | BC floor + REINFORCE (cos reward) | ~conc_pop | | | cos saturates (concept ceiling) |
| O10 | BC floor + REINFORCE (coverage reward) | ~0.31 | ~0.10 | | gameable (inflation) |
| O11 | BC floor + dense REINFORCE (Dcov - PEN*unans) | ~conc_pop | | | flat (BC trap) |
| O12 | **NO-BC** + dense coverage REINFORCE | 0.309 | 0.105 | 0i/all-c | LEARNS (return rises -> BC WAS trapping) but gameable coverage; eval ~conc_pop |
| O13 | NO-BC + RANK-AWARE reward (likes - non-likes) + answerability penalty | 0.291 | 0.075 | 27i/1037c | picks some items, return rises, but TEST WORSE (tail declines) |

## O12 DEEP-DIVE (instrumented, checkpoint policy_o12.pt, EP25) — answers the user's 5 questions
FULL: pop_item .293/.287/.288/.307 | conc_pop .293/.306/.312/.315 | **policy .293/.305/.309/.309**
TAIL: conc_pop .088/.105/.109/.116 | **policy .088/.104/.104/.107**
1. **q-curve monotone?** policy rises q0->q2 then PLATEAUS (.305->.309->.309); sits just BELOW conc_pop at every q.
2. **answered/8** 7.8 (vs conc_pop 8.0) — not wasting budget, asks answerable.
3. **item vs concept picks** = **0 items / 1170 concepts** — even with items in the pool, coverage-REINFORCE NEVER asks an item (items=unanswerable=no coverage gain). The item+concept COMBINATION never emerges from this reward.
4. **adaptive?** distinct-first-pick=1/150 (opening fixed, expected). REFINED via full-sequence probe: distinct-pick/turn [1,2,3,4,5,5,4,4]; top-pick-share/turn ~0.53; **distinct-traj 5/150; total-vocab 18 concepts**. => WEAKLY adaptive (branches after opening) but COSMETIC: a ~5-leaf mini-tree over an 18-concept popular shortlist, one default concept per turn (~53%). After turn ~2 it recycles the same shortlist => explains the plateau + cos-degradation. It IMITATES conc_pop with a thin branch, not out-asks it.
5. **cos(u,u*)** policy 0/.731/.712/.710 vs conc_pop 0/.773/.805/.825 — policy belief PEAKS at q2 then DEGRADES; conc_pop keeps rising. The coverage reward is MIS-ALIGNED with belief fidelity: the policy's later picks add noise.
=> O12 is a genuine learner (no BC-trap: moves off the .293 popularity floor) but converges to concept-only, fixed-opening, and lands just UNDER conc_pop. Root cause = coverage reward (gameable, decoheres belief), NOT optimization. Confirms: beating conc_pop needs a belief-faithful reward/objective + a mechanism that actually USES items, not reward tweaks alone.

## HEURISTIC item:concept MIX SWEEP (non-learned; mixK = K popular items + (8-K) frequent concepts, interleaved) — THE COMBINATION WORKS
FULL q8 NDCG: conc_pop .315 | mix0 .315(==sanity) | mix1 .309 | mix2 .307 | mix3 **.315** | mix4 **.318** | mix6 **.322**
TAIL q8 NDCG: conc_pop .116 | mix2 **.122** | mix3 **.120** | mix4 .116 | mix6 .096
- **mix3 (3 items / 5 concepts) weakly DOMINATES conc_pop**: FULL-NDCG .315(=), FULL-Rec .246(+.005), TAIL-NDCG .120(+.004), TAIL-Rec .160(+.007).
- FULL likes MORE items (mix6 .322, +.007); TAIL likes a MODERATE mix (mix2 .122, +.006), too many items HURTS tail (mix6 .096) — items are head/popular, sharpen full ranking but don't help tail discovery.
- combination > pure: beats both conc_pop (concepts-only .315/.116) AND pop_item (items-only .307/.085). **Validates "items+concepts together > each alone."**
- **This is exactly the gain O12 left on the table** (O12 picked 0 items). New policy target = beat **mix3**, not just conc_pop. (margins small, single seed — confirm with seeds.)

## RS-VALIDITY (user Q: are our cold-start numbers weak RS or lit-aligned?) — VALID, lit-aligned; low abs = PROTOCOL + ML-1M pop-dominance
- WRMF replication (scripts/paper1/mf_foldin.py): MOSTPOP NDCG@10 **0.4134** (pub 0.3921) ; RMVA-maxvol **0.5011** (pub 0.5387) ; REPRESENTATIVE 0.4816 ; WARM-half MF 0.239. => data+protocol+model FAITHFUL to lit.
- PROTOCOL reconciliation (scripts/paper2/protocol_check.py, SAME MostPop scorer, locked 150 users): PROT-A (pub: ALL likes, no excl, R@10) **0.4200** ; PROT-C (OURS: held-half, exclude-known, R@50) **0.2929** == our q0 0.293 EXACTLY. The 0.42->0.29 gap is 100% the held-half+exclude-known protocol, NOT RS weakness.
- ML-1M full-cat NDCG@10 is POPULARITY-DOMINATED: best published cold-start (RMVA/DRE) beat MOSTPOP by only ~0.09-0.16; pure MF fold-in (no pop) is WORSE than MostPop (0.239<0.413). => small elicitation-over-popularity gains are EXPECTED/lit-consistent; the TAIL metric (pop neutralized) is where elicitation legitimately shows value (entropy +15%).
- RMSE: q0 baseline 0.976 (mu+bi); biased-SVD MF RMSE replicates published ~0.94 (memory).
- METRIC-DEPENDENT WINNER (RMSE added to all): entropy wins NDCG (ranking) but is WORST on RMSE (>q0); tree/pop/greedyext win RMSE (tree=RMSE-optimal). conc_pop best RMSE 0.961, tree 0.958, entropy 0.997.

## SEED-AVG (n=6 splits, q8 mean±std) — CANONICAL; corrects single-seed noise
FULL-NDCG: entropy .331±.011 ~ tree .329±.008 ~ greedyext .326 ~ conc_pop .325±.008 (ALL TIED within ~1σ — no robust full winner; single-seed "entropy/mix beat pop on full" was NOISE)
TAIL-NDCG: **entropy .127±.006 > conc_pop .115±.007 (+.012, ~2σ ROBUST)** ; tree .101 / igcn .096 (robustly WORST)
RMSE(full): tree .958 < greedyext .963 < conc_pop .960 << entropy .995 (stable; entropy worst — discrimination vs calibration tradeoff)
=> NARROWED CLAIM: in answerable concept-space, divisiveness (entropy) buys a robust ~10% TAIL-NDCG gain over popularity at an RMSE cost; FULL is popularity-saturated/tied. O14 target = match ~.33 full cluster + BEAT entropy .127 tail.
IGCN robustly worst -> needs item-level validation/diagnosis (suspected train/test answer-distribution mismatch in my concept port).

## (single-seed, superseded by seed-avg above) LIT BASELINE LEADERBOARD (locked 150-split, q8 NDCG@10 FULL/TAIL)
pop_item .307/.085 | random .297/.106 | **conc_pop .315/.116** | igcn .301/.104(WORSE) | popent .306/.120 | logpopent=helf .311/.127 | greedyext .316/.124 | tree .318/.106(full-only) | **entropy .320/.134 (TAIL KING +.018)** | mix6 .322/.096(FULL KING) | mix3 .315/.120
- **ENTROPY (Rashid 2002 divisiveness) BEATS conc_pop**: tail +.018 (.134 vs .116, +15%!), full +.005 — and does it with only 5.1/8 answered (few but maximally-discriminating concepts). INFORMATION > FREQUENCY for concept elicitation.
- **Our learned O12 policy (.309/.107) LOSES to entropy, greedyext, logpopent/helf, the mixes — it's BELOW conc_pop.** A one-line entropy heuristic out-asks the RL policy. (coverage reward chases popularity = anti-entropy.)
- Adaptive/personalized LIT (tree full-only, igcn WORST non-random) DON'T help => adaptivity per se isn't the win; question INFORMATION is.
- pop×ent/HELF (down-weight divisive by popularity) do WORSE than raw entropy => don't pull back toward popular. Candidate next heuristic = ENTROPY-among-answerable (divisive AND answerable).
- **NEW TARGET: beat ENTROPY (.320/.134), not conc_pop.** All single-seed; confirm with seeds.

## VERDICT (after 13 variants)
Pure cold-start q8: NO learned policy beats conc_pop. Each user-identified bug was real & fixed (crippled EIG, BC trap,
gameable coverage) and it STILL doesn't beat conc_pop -> training reward improves while TEST NDCG doesn't = no realizable
signal left, NOT a bug. Lines up with the MEASURED ceiling (PART AG/AG2): concepts saturate cos 0.83, answerable items
sparse (1.9/8) -> item+concept gain marginal at q8 (conc_mix +0.003). conc_pop is the realizable cold frontier = a CEILING
OF THE SETTING. To beat it needs item-level fineness: (a) WARM/returning user (conc_gprof +23%), (b) higher budget, (c)
SPARSE/large catalogue (frequency weak). NOT more policy tweaking.

## INVESTIGATION QUEUE (user, stop declaring frontier)
1. Best SIMPLE mixed heuristic (item+concept ratio sweep) -> achievable threshold.
2. RUN the actual LIT baselines (NOT yet run!): entropy, pop x entropy, HELF (Rashid), GreedyExtend (Golbandi 2010), Golbandi ternary tree (2011), IGCN. Be true to lit. (conc_pop/conc_eig/random were the ONLY ones run.)
3. Pretrained policies didn't budge from BC floor -> can their ideas port onto from-scratch? 
4. Distill the UNIFIED item+concept ORACLE (not just conc_gprof/items).
5. DEEP-DIVE O12 (no-BC, ~matched conc_pop from scratch): per-turn NDCG monotone? answerable count? adaptive vs conc_pop? what signal/reward? -> tweak from it.
6. ENSEMBLE RL(O12) + conc_pop if they learned different things.
7. Known-profile info: quantify how much it carries for NDCG (warm fold-in headroom); brainstorm why reconstructing it (reward/pretrain) adds no value yet.
8. [BIG, separate family] OPEN free-text questions: "what do you like? favourite genre? mood?" -> user gives a SHARP answer -> fold -> then nuanced follow-ups. The real CASPER open-concept elicitation.
## TODO (later)
- Debug UI: user sees questions, answers free-text+rating, recs render & recompute in real time.
