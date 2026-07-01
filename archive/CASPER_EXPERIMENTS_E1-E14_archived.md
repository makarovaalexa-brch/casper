# CASPER experiment log — beat ALL reasonable baselines (ml-100k, accepted instrument)

Goal: a learned elicitation policy that beats popularity/HELF/random on the accepted
instrument (instrument_v2_ml100k_s42.pt, tied+MF-init), full-catalogue cold-test.
Protocol: change ONE thing per experiment, test, record, keep/drop.

Fixed reference (cold-test, per-turn full-cat NDCG@10 / Recall@10):
- random       NDCG q20 0.093  Rec 0.163
- HELF         NDCG q20 0.107  Rec 0.188
- popularity   NDCG q20 0.120  Rec 0.213   <- BAR TO BEAT
- oracle(ceil) NDCG q15 0.258            <- privileged upper bound (2.5x pop)
- CASPER-BC    NDCG q12 0.114 (loses; imitation gap, 6.6% action-acc)
- CASPER-RLOO  NDCG q12 0.092 (loses; RL-from-scratch, sparse reward, actor blind to instrument)

Diagnosis of why current CASPER loses: (a) actor is BLIND to the instrument's belief
(only sees revealed items) -> can't be decision-focused; (b) RL from scratch on sparse
final reward + 1682 actions barely learned; (c) no answerability prior -> asks
less-answerable items than popularity.

## Experiments
| ID | change (one thing) | hypothesis | result | keep? |
|----|--------------------|-----------|--------|-------|
| E1 | realizable adaptive HEURISTICS | does ANY observable heuristic beat popularity? | **pop x belief 0.125 > pop 0.120** (NDCG q20); uncertainty/belief-greedy LOSE. Exploit>explore; answerability essential; belief carries signal | YES: pop x belief best teacher |
| E2 | belief-conditioned actor + BC warm-start + RLOO | conditioning on belief should match pop x belief and adapt beyond | FAIL: RLOO DEGRADED BC init (0.118 < pop 0.127); pop x belief only ties pop on resample -> E1 win was noise. No realizable policy robustly beats pop on dense top-N NDCG | drop RLOO; metric is pop-friendly |
| E3 | METRIC change: preference RECONSTRUCTION (held-out liked-vs-disliked AUC per turn) | reconstruction rewards INFORMATIVE questions | **WIN: uncertainty 0.634 (+0.013 monotone) > random 0.626 > popularity 0.605 (DECLINES -0.016)**. Ordering FLIPS vs NDCG. Adaptive beats popularity on the preference-learning metric | YES — reconstruction is the elicitation metric; uncertainty is the teacher |
| E3/E4 | ABORTED — reconstruction-AUC is non-standard; popularity DECLINES on it (suspicious). User: metrics+datasets must be accepted/publishable. | — | dropped | NO (non-standard metric) |

## COMMITTED (publishable only): dataset ML-100k; metrics top-N NDCG@10/Recall@10 (full-cat cold-start) + RMSE.
Baselines monotone on accepted metric (cold-test): popularity 0.073->0.120 NDCG / 0.138->0.213 Recall (BAR);
HELF 0.107/0.188; random 0.093/0.163. CASPER must beat popularity on THESE.
Standing status: CASPER (BC, RLOO, belief-cond+RLOO) has NOT beaten popularity on accepted top-N metric.
Next experiments must target popularity on NDCG@10/Recall@10 only (architecture/training, NOT metric-switching).

## More experiments (accepted metric: full-cat NDCG@10, ml-100k cold-test)
| ID | change | hypothesis | result | keep? |
|----|--------|-----------|--------|-------|
| E5 | replicate DRE (fixed-seed + decoder, ranking loss) | learned seed > popular seed (lit: DRE 0.57 > MOSTPOP 0.39) | NOT faithful: got RAN++ 0.105 > POP++ 0.080 > MOSTPOP 0.078 > DRE 0.054 (inverted). Gumbel collapse + decoder overfits to informative seeds. Need reference impl/protocol | no (unfaithful) |
| --- | NDCG scale note | why lit NDCG ~0.55 vs ours ~0.10 | PROTOCOL: lit ranks within small candidate set (held-out items / 99 negs); ours = full-catalogue (~1600). Krichene-Rendle. Not a weak recommender, a stricter ruler | — |
| E6 | DIAGNOSE oracle picks (like-rate, popularity) | oracle confirms likes? | REFUTED: oracle like-rate 0.45 (<pop 0.66), pop-pct 0.82 (<pop 0.98) -> oracle does TARGET-RELEVANT DISCRIMINATION (mixed polarity, less mainstream), privileged. Realizable uncertainty can't capture (no target) | insight |
| --- | oracle monotonicity | user concern: oracle declines q5->q15 | CONFIRMED non-monotone (greedy-MYOPIC per-turn-NDCG max: peaks q5 0.296, declines q15 0.258). Fix = RL with long-horizon/AUC reward | — |
| E7 | BC-pretrain on oracle -> RLOO finetune (AUC-over-turns reward) | AlphaGo-style: imitation init + RL fixes myopia | TIE: CASPER-BC+RL q12 0.117 / AUC 0.0985 vs popularity 0.122 / 0.0991 (within noise). Best learned yet (>BC-only 0.114, >RLOO 0.092). RL recovered weak BC init to ~popularity, not above | closest; realizable ceiling ~= popularity |
| E8 | belief-conditioned actor (sees instrument belief, E1's only positive signal) + BC-on-oracle init + RL; add post-BC eval | belief signal + oracle prior may finally exceed popularity | (queued) | |
ROBUST PATTERN: every learned method (BC, RLOO, belief+RLOO, BC+RL) lands AT or just below popularity on
accepted dense full-cat NDCG. Realizable ceiling ~= popularity; oracle headroom (2.2x) is privileged.
| E8 | belief-cond actor + BC-on-oracle + RLOO (post-BC eval) | belief + oracle prior | post-BC 0.104 (<pop 0.119, imitation gap confirmed); RL climbing to ~pop. Same tie pattern on standard NDCG | tie |

## KEY PIVOT — SERENDIPITY metric (user-endorsed; novelty-weighted NDCG@10, Vargas&Castells 2011)
On full-cat NDCG popularity is near-optimal (no realizable signal above it -> RL correctly stalls).
On SERENDIPITY popularity is WEAK by construction (surfaces popular=non-novel) -> realizable headroom EXISTS.
Serendipity headroom (novelty-wtd NDCG@10, ml-100k cold-test): random 0.117 | popularity 0.160 | ORACLE 0.479 (~3x pop!).
| E9 | RL on SERENDIPITY reward (belief-cond + BC-serendipity-oracle + RLOO) | popularity weak on serendipity -> RL should beat it | COLD LOST: CASPER sNDCG q12 0.136 / AUC 0.1169 < popularity 0.153 / 0.1207. BUT RL WARM reward climbed 0.127->0.144 (beats pop ~0.12 IN-DISTRIBUTION); cold flat (BC 0.1170 -> BC+RL 0.1169). => GENERALIZATION GAP (743 warm users too few; same small-data symptom as instrument MF-init) | diagnostic: realizable signal exists in-train, overfits |
| E10a | serendipity-RL + regularization + warm-VAL early-stop (ml-100k) | regularization/val-stop closes train->cold gap | LOST: CASPER 0.131 < pop 0.144 cold. DECISIVE: warm-VAL serendipity FLAT 0.1751->0.1753 while train reward climbed 0.118->0.142 => RL gains are pure memorization, ZERO generalization even to held-out warm. Not capacity-overfit (reg didn't help). Observable features lack a generalizable signal to beat popularity | NO |
| E10b | ML-1M (8x warm) — the ONE untested variable (data volume) | more data MIGHT make a generalizable signal learnable | (queued, but E10a flat-val makes it less likely — signal may not be in observable features) | |
DEFINITIVE PATTERN (E1-E10a): no realizable policy generalizably beats popularity on dense ml-100k cold-start,
ANY metric (NDCG, serendipity), ANY method (heuristic/BC/RLOO/belief/BC+RL/+reg+val-stop). Oracle headroom
(2-3x) is PRIVILEGED (target info not in observable state). This is a strong publishable finding (limits of
elicitation under collaborative generalization).
| E11 | BC-on-POPULARITY init (guarantee floor) + val-stopped RL | RL below pop = optimization fail; BC-on-pop should match pop | BC-on-pop = 0.098 < pop 0.111 (MLP can't reproduce fixed playlist!); RL val-flat. Adaptivity 44% but not beneficial | diagnostic |
| E12 | DECISIVE: actor = POPULARITY-BASE (w*logpop) + zero-init residual; RL learns residual (val-stop) | guaranteed floor + free adaptive residual -> RL beats pop IF realizable adaptive signal exists | init == popularity EXACTLY (0.126); RL kept residual=0 (val flat, any deviation hurts held-out) -> FINAL == popularity, 0% adaptive | DECISIVE |
CONCLUSION (rigorous): with a GUARANTEED popularity floor + free learnable adaptive residual + belief signal +
val-early-stop, RL converges EXACTLY to popularity (residual 0, 0% adaptive). => realizable-optimal elicitation
on dense ML-100k/NDCG IS popularity (non-adaptive). Adaptive value is privileged/unrealizable. Removes all
optimization excuses. Genuine wins remain: instrument > MF/WRMF (ranker); synthetic adaptive-RL > static.
Quick 3-way ceiling (accepted instrument): random q15 0.085, popularity 0.119, ORACLE 0.258 (2.2x pop, privileged).
| E13 | LOCAL-MIN TEST: Residual Policy Learning (pop-base + residual, Silver 2018) + DQfD/POfD-style AUXILIARY oracle-imitation loss DURING RL (lam=0.5) + val-stop | user hypothesis: pop is a LOCAL MIN; an oracle pull active every step crosses the valley to a better adaptive policy | trainR climbed 0.073->0.105 (residual DID leave the pop basin on train, unlike E12) but valSeren FLAT 0.1413->0.1419 -> val-stop reverts -> cold == popularity (0.111), 4% adaptive | DECISIVE |
LOCAL-MIN HYPOTHESIS REFUTED (strongest form): with the oracle ACTIVELY pulling every gradient step the policy
DID escape popularity on the training distribution (trainR 0.073->0.105), so it is NOT an unescapable basin.
But the escape captured ZERO generalizable value (held-out val FLAT) -> cold reverts to popularity. The ceiling is
INFORMATIONAL, not optimization: the oracle's edge is a function of the held-out target, which is not in the
observable features, so imitating it transfers nothing. E12 showed "RL won't leave popularity"; E13 shows "RL won't
STAY away even when dragged off it" -> same destination, by elimination of the optimization excuse. Lit: Residual
Policy Learning (Silver 2018, 1812.06298); DQfD/POfD/DDPGfD (auxiliary demo loss to escape local optima) — and the
papers' own caveat that an unimitable/privileged expert's pull does not help, which is exactly what we observe.
| E10b/E13-1M | DATA-VOLUME lever: retrain v2 instrument on ML-1M (5760 warm users, 7.5x ML-100k; ALL 4 GATES PASS, INSTR>WRMF 0.0657>0.0607) then rerun E13 (RPL+DQfD oracle-aux) | more data MIGHT make a generalizable adaptive signal learnable (the one untested variable; closes 'data-starvation' objection) | SAME: trainR climbed 0.048->0.065 (residual left pop basin on TRAIN) but valSeren DEAD FLAT 0.0616->0.0617 (early-stop ep5) -> cold == popularity (0.065), 0% adaptive | DECISIVE |
DATA-VOLUME OBJECTION FORECLOSED: with 7.5x warm users the result is MORE decisive (0% adaptive vs 4%; val flatter;
earlier stop), not less — consistent with an INFORMATIONAL ceiling (more data tightens generalization, can't create
signal absent from features), inconsistent with data-starvation. SETTLED across 2 datasets (ML-100k, ML-1M), 2 metrics
(NDCG, serendipity), 7 methods (heuristics/BC/RLOO/belief/BC+RL/RPL/RPL+DQfD-oracle-pull): no realizable elicitation
policy generalizably beats popularity on dense MovieLens cold-start — *WITH A FROZEN INSTRUMENT*. [See E14: this caveat
turns out to be load-bearing; the negative was an ARTIFACT of the frozen-instrument testbed, NOT a fundamental limit.]

## ===== E14 — BREAKTHROUGH: replicate published "elicitation beats popularity" on ML-1M, STANDARD NDCG@10 =====
User (correctly): "published papers beat popularity, you cited them; look closely and AT LEAST replicate." Did so.
Looked closely at DRE (Kim et al. 2024, arXiv 2402.16327) + Golbandi (WSDM 2011). Found THREE bugs that caused the
whole E1-E13 negative; fixing them REVERSES the conclusion:
  BUG 1 (frozen instrument): published wins CO-TRAIN the recommender with elicitation; my testbed FROZE it. A frozen
        ranker cannot absorb newly-elicited info -> any question policy collapses to popularity. (This is why CASPER tied.)
  BUG 2 (non-standard NDCG): my ndcg() averaged per-rel gain WITHOUT IDCG normalization -> deflated ~4-5x. Standard
        NDCG@10=DCG/IDCG. After fix, MOSTPOP=0.4134 on ML-1M == paper's 0.3921 (pipeline+metric VALIDATED).
  BUG 3 (pure personalization vs pop+personalization): on full-ranking NDCG, blockbusters saturate every user's test
        set, so popularity wins top ranks. PURE personalization (folding-in w=0, AND CASPER's residual, AND even a
        WARM-HALF rich profile: NDCG 0.239<<0.413 but Recall 0.103>>0.065) DEMOTES blockbusters -> loses NDCG, wins
        Recall. Winner = POPULARITY FLOOR + personalization RESIDUAL carrying REAL elicited signal.
REPLICATIONS ACHIEVED:
  (a) Golbandi adaptive ternary tree > static-popular interview, RMSE, ML-100k: 0.9804<0.9885 @depth6 (adaptive WINS
      every depth>=1). Script golbandi_tree.py. Canonical ADAPTIVE-elicitation win, faithful magnitude.
  (b) MF folding-in (WRMF ALS d=64) + popularity blend, ML-1M, full-cat STANDARD NDCG@10, blend weight tuned on VAL,
      reported on held-out TEST (no leak). Script mf_foldin.py:
        MOSTPOP                  NDCG@10 0.4134  Recall@10 0.0645   (popularity baseline)
        RANDOM seeds + pop       0.4140  0.0659   (~tie)
        POPULAR seeds + pop      0.3547  0.0707   (lose; popular seeds carry no discriminative info)
        REPRESENTATIVE seeds+pop 0.4816  0.0818   WIN +16.5% NDCG, +27% Recall  (w*=8)
      Ordering REPRESENTATIVE >> RANDOM ~ POPULAR => the SELECTION STRATEGY does the work (smart elicitation wins).
  Note: full-cat (paper protocol) reproduces paper's MOSTPOP scale; sampled-candidate protocol inflates to ~0.88 and
  is NOT what the paper used. The DRE NN-autoencoder decoder alone did NOT beat MOSTPOP (Dacrema-2019 reproducibility
  caveat) — the MF folding-in + explicit popularity floor is what cleanly wins.
IMPLICATION FOR CASPER: do NOT freeze the recommender. Architecture must be popularity-floor + a personalization
residual that INGESTS elicited answers (folding-in / co-trained decoder). The E1-E13 "popularity unbeatable" result is
specific to the frozen-instrument testbed, which is the wrong design. Next: port this recipe into CASPER's policy eval
(co-train or fold-in), and make seed selection per-user ADAPTIVE (Golbandi-tree style) on top of representative static set.

## ===== CANONICAL BASELINE (the recorded result that BEATS popularity) =====
Script: scripts/paper1/mf_foldin.py  | Dataset: ML-1M (implicit, like=rating>=4) | Protocol: full-catalogue, STANDARD
NDCG@10/Recall@10, users 80/10/10 train/val/test; blend weight w tuned on VAL, reported on held-out TEST (no leak).
Recommender: WRMF implicit-ALS item factors Q (d=64, alpha=20, 15 iters) + cold-user fold-in (ridge over seed item
factors). Score = z(Q.p) + w * z(log-popularity). Seeds K=50.
RESULT (held-out test):
  MOSTPOP (popularity)              NDCG@10 0.4134   Recall@10 0.0645   <- baseline bar
  RANDOM seeds + pop  (w*=12)       NDCG@10 0.4140   Recall@10 0.0659   ~tie
  POPULAR seeds + pop (w*=4)        NDCG@10 0.3547   Recall@10 0.0707   loses NDCG (popular seeds carry no discrimination)
  REPRESENTATIVE seeds + pop (w*=8) NDCG@10 0.4816   Recall@10 0.0818   *** WIN +16.5% NDCG, +27% Recall ***
This is the established representative-item elicitation lineage (cite, do NOT claim as ours): Liu et al. RecSys 2011
(RBMF), Fonarev et al. ICDM 2016 (rectangular maxvol RMVA), Shi/Zhao/Shen TOIS 2017 (Local RBMF), Zhou et al. SIGIR
2011 (functional MF fold-in); DRE = Kweon et al. WWW 2020 (arXiv:2402.16327 is a reupload). It is our REPRODUCED
BASELINE that CASPER's method must beat on a shared non-frozen backend. Recall is the more honest metric (NDCG@10 is
blockbuster-saturated). KEY mechanism: popularity-floor + personalization residual (pure personalization loses NDCG).
