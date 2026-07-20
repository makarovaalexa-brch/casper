---
name: continuous-ndcg-policy-research
description: "Deep-research menu for a LEARNED CONTINUOUS-action selection policy (belief->query direction, trained toward tail-NDCG) to replace the discrete NDCG-tree and enable the open/LLM-verbalised channel. Recommends amortized-BED (Huang NeurIPS'24 decision-aware) + continuous-action RL (Blau ICML'22 + Wolpertinger snap) warm-started by tree distillation; NeuralNDCG differentiable aux."
metadata:
  node_type: memory
  type: reference
  originSessionId: a2a12990-a215-4a69-8853-6a98488dcf50
---

Jul 18 2026. The discrete NDCG-tree ([[ndcg-voi-adaptive-selection-design]]) works but can't emit CONTINUOUS or
OPEN (LLM-verbalised) questions -> need a LEARNED continuous-action policy: state=belief(mu,Sig)/history,
action=query DIRECTION d in the frozen 512-d CF space, reward=tail-NDCG after the Kalman fold, compatible with
snap-to-askable / LLM-verbalise / open-free-text-fold ([[three-channel-continuous-elicitation]]). Deep-research
menu (all works verified real):

**FAMILY 1 - Amortized sequential Bayesian experimental design (BED) = closest formalism.** Policy net
history->next continuous design, trained upfront. ★ **Huang, Guo, Acerbi, Kaski "Amortized BED for
Decision-Making" NeurIPS 2024 (arXiv:2411.02064)** = THE closest paper: Transformer Neural Decision Process
trained on DOWNSTREAM DECISION UTILITY not EIG -> substitute utility:=NDCG@10 = near-blueprint. Also DAD (Foster
ICML'21, contrastive-EIG policy net, needs closed-form likelihood + conditionally-independent experiments), iDAD
(Ivanova NeurIPS'21, implicit/simulator-only), Step-DAD (semi-amortized, refit mid-experiment). RISK: native
objective=EIG; our repeated-probe-of-same-u-through-Kalman is CORRELATED obs (not cond-independent -> Blau MDP
framing fits better).

**FAMILY 2 - Continuous-action RL over the embedding action space = BEST FIT for NDCG-as-reward.** MDP:
state=(mu,Sig) suff stats, action=direction d in R^512, reward=ΔNDCG (black-box, no differentiability needed).
★ **Blau, Bonilla, Chades, Dezfouli "Optimizing Sequential Experimental Design with Deep RL" ICML 2022
(arXiv:2202.00821)** = reduces amortized BED to an MDP solved by deep RL, handles continuous+discrete design AND
black-box models = the template for policy->continuous-direction, reward=NDCG. + **Wolpertinger (Dulac-Arnold
2015, arXiv:1512.07679)**: actor emits continuous proto-action -> kNN-snap to nearest valid discrete -> critic
re-rank = EXACTLY our propose-direction->snap-to-askable pattern (degrades to open by skipping snap). NB our own
scar "Wolpertinger k=1 FAILS" -> use k>1 + critic re-rank. UNICORN/ConTS = convrec RL baselines but DISCRETE
actions. RISK: RL sample-efficiency + sparse terminal-reward SNR at 2-8 steps (our June-review flaw) -> dense
per-step ΔNDCG shaping + tree warm-start.

**FAMILY 3 - Policy-based active feature acquisition ("learning to ask") = reward-shaping + baseline donor.**
GSMRL (Li&Oliva ICML'21, generative surrogate supplies INTERMEDIATE rewards -> fixes RL SNR), EDDI (Ma ICML'19,
partial-VAE greedy info-gain = the myopic discrete baseline our tree generalises), Shim NeurIPS'18 (set-encoding
of acquired features). RISK: info-gain-targeted -> reproduces myopic-info-gain-loses-to-popularity if used as the
objective. Value = GSMRL intermediate-reward trick + EDDI ablation.

**FAMILY 4 - Differentiable ranking as training signal (gradient not RL).** Kalman update is differentiable in d;
only the final SORT isn't. ★ NeuralNDCG (Pobrotyn&Bialobrzeski 2021, arXiv:2102.07831, NeuralSort soft-permutation
-> differentiable NDCG) chained after the Kalman update = one differentiable graph d->answer->Kalman->scores->
softNDCG. LambdaLoss (Wang CIKM'18) lighter alt. Fit: great for the CLOSED/continuous channel (low-variance
grads, fast on 150k). RISK: snap + LLM-verbalise are NON-differentiable (soft-sort fixes the metric not the
discretisation); must differentiate through EXPECTED answer (reparam/marginalise under belief); end-to-end
diff-NDCG-after-elicitation has NO direct precedent (novel, unproven).

**FAMILY 5 - Distil the discrete NDCG-tree into a continuous policy (DAgger warm-start).** Tree=NDCG-optimal
discrete teacher -> BC/DAgger (Ross AISTATS'11) into a continuous actor -> RL-finetune. RISK: teacher lives on a
FINITE concept pool -> distillation inherits the vocabulary ceiling; need RL-finetune/diff-exploration to invent
off-pool continuous/open directions. NO direct precedent for distilling a discrete design-tree into a continuous
policy (a genuine gap).

**★★ AMORTIZED-E3 RESULT (Jul 18, `scripts/bpool_amorte3.py`, refusal cache 2500, disjoint halves, static q1=
concept128, DKQS = MLP mu1->softmax over 1317-bank->blend dir->exact-Kalman-fold->soft-listwise-NDCG loss,
differentiable, NO RL). Eval half q1-only 0.0661:** static-q2 +0.0171 | var-greedy(champion) +0.0186 | **LEARNED
DKQS +0.0197 (peak +0.0211 @ep40)** | peeking-oracle +0.2406 (selection-max-INFLATED, not a real target).
**DKQS BEATS var-greedy + static.** Controls CLEAN: frozen-mu twin +0.0142 -> **answer-contingent prize +0.0054**
(policy genuinely reads the belief); answer-permutation null -0.0053 ~= q1-only (vanishes). Snap-agreement w/
static-q2 = 0.00 (picks per-user concepts), mean max-weight 0.67 (sparse 1-2 blend). **MILESTONE: a differentiable
learned continuous-action selector trains cleanly, beats the champion, has a real answer-contingent prize, passes
the null = the proof-of-concept the open/continuous vision needed. NOVELTY WEDGE CONFIRMED: exact conjugacy makes
elicitation differentiable in the query (Biyik can't).** CAVEATS: margin over var-greedy THIN (+0.0011) at q2 (one
adaptive Q = hardest case; full T=8 unroll is where it compounds, cf tree +0.0198 over static); overfits after
ep40 (need val-checkpoint/early-stop); 2500-probe (all-users for headline). Honesty gate PASSED first
(`bpool_honesty.py`: 2/3-sparse blend answer corr(blend,true)~0.44 = single-concept level, not degraded).
Design sheet `casper/DESIGN_AMORTIZED_E3.md`. NEXT: full T=8 differentiable unroll + sparsemax + early-stop,
then all-users + sign-off.

**★★ FULL T=10 NON-MYOPIC DKQS + ITEMS + ALL-USERS DESIGN (Jul 18, Fable efficiency pass). Author directive:
"10 steps, non-myopic, include items." Physics ALL EXACT on CPU (no accuracy compromise); only 2 RISKY modeling
assumptions, both gated by controls run FIRST:**
- **Sigma across 10 turns = EXACT rank-1 V-downdate:** Sig_t@d = Sig0@d - V(V^T d), V is 512xt<=10 per user;
  bit-identical to dense fold, ~0.3 MFLOP/user/turn, differentiable. Shared-Sigma + diagonal-Sigma REJECTED
  (a learned policy diverges directions per-user; diagonal breaks the collinearity-discount invariant). No approx.
- **150k cache = sparse WATCH-COUNTING job (NOT encoder):** us only used for calibration (done on 15k) + honesty.
  SEL/VAL for all 150k via 2 sparse matmuls vs Mbin + vectorized searchsorted. ~20-60 min. Quarantined 300/173
  study users EXCLUDED. Minibatch SGD sees ALL users across epochs = NOT a data reduction (HARD RULE #1 OK).
- **Unified action+target = FULL 18,430 items + 1,317 concepts, NO shortlist** (restricting to head items would
  amputate the Jul-14 niche-drill signal). Dense softmax over 19,747 affordable on CPU. Item answer = singleton-
  concept SEL (refusal for popular=confident-neg), calibrated in POPULARITY BINS (control: per-item vs binned on
  top-1000-rated). NO Wolpertinger/Gumbel at train (train on soft blend, exact grad); SNAP is a deploy/eval arm.
- **Loss = full exact softmax-DCG (NO sampling; would bias the tail). Eval always exact NDCG@10.**
- **float32 train / float64 eval** + one-batch f32-vs-f64 divergence check (f64 fallback -> can't silently
  change a conclusion). **State = [mu, diag(Sig_t), turn-emb]** (diag nearly free, differentiable -> drilling
  expressible). **Objective = late-weighted ramp lam_t~t^2 PRIMARY (non-myopic) + terminal-only ablation.** sig2
  FROZEN (policy never emits confidence = anti-gaming, carries from E3 sheet).
- **Wall-clock: ~1 day** (cache ~1h; train ~8-20h float32 overnight, ~2x f64; baselines ~1-3h). Checkpoint-to-peak.
- **★ 2 RISKY rows (only real accuracy risks):** (5) blend-answer with ITEMS compounded 10 turns -> re-run honesty
  gate with item components + 10-deep blends BEFORE training = GO/NO-GO. (6) soft-blend-train vs hard-snap-deploy
  -> snapped-top-1 eval arm is the REPORTABLE number if it diverges from soft. Both NON-NEGOTIABLE, run first.
- **BUILD ORDER:** 1) 150k vectorized refusal cache + quarantine assert + split manifest. 2) item-answer
  calibration + honesty gate (GO/NO-GO). 3) batched V-downdate unroll + exactness self-test + f32/f64 check.
  4) train DKQS T=10 (ramp + terminal ablation), val-checkpoint. 5) eval by-turn full+tail vs random/static/
  var-greedy/VoI/tree + frozen-mu twin + perm null + SNAPPED-top1 arm. Bar: beat var-greedy (+0.0563 tail@q8
  probe) by-turn, positive twin-prize, vanishing null. CONTINUOUS FALLS OUT of the sparsemax blend (not separate);
  LLM-verbalise + open-text-input = thin deploy adapters, no retrain. TRAINING RUN HELD FOR AUTHOR GO post-gate.

**★★ UNIFIED POLICY CURVES (Jul 18, `scripts/bpool_curves.py`, 2500 cache, same eval half, T=8, tail floor
0.0433 full 0.1860). TAIL@q8: static 0.1187 > tree(+vg) 0.1153 > var-greedy 0.1118 > voi 0.0962 > random 0.0819
> dkqs-MYOPIC 0.0786.** KEY: (1) **MYOPIC DKQS is BEST at q1 (0.0727, beats all) then COLLAPSES** (flat ~0.078
q2-q8, LAST) -- no repeat-mask + no sequence training -> re-picks the same top concept, exact fold makes repeats
useless, turns 2-8 WASTED = clean proof the NON-MYOPIC T=10 unroll is NECESSARY not optional. (2) **The real bar
is the STATIC greedy questionnaire (0.1187 tail@q8), which BEATS var-greedy + tree at long horizon** -- a strong
fixed questionnaire is hard to beat in concept-only; non-myopic DKQS must clear static@q8=0.1187. (3) ADAPTIVITY
edge is EARLY (var-greedy best thru q3-4: 0.0996@q3 fastest climb, then static overtakes ~q5) = sample-efficiency,
NOT long-horizon -- concept-only adaptivity headroom is THIN late. (4) FULL barely moves (~0.19->0.20 all) =
concepts are a TAIL instrument. IMPLICATION for all-users run: target sharpened = beat static 0.1187 tail@q8;
DKQS strongest case = early-Q efficiency + the ITEM action space giving something to DRILL (concepts alone thin
late). Myopic collapse = exactly the failure the full unroll fixes (terminal obj + turn-state + repeat-mask).

**★★ NON-MYOPIC DKQS PROBE (Jul 18-19, `scripts/bpool_dkqs2.py`, 2500 cache, T=10, concept-only, repeat-mask +
Sigma-aware state [mu,diagSig,turn] via exact rank-1 V-downdate + late-weighted lam~t^2 terminal obj).**
- **The non-myopic fix WORKS: collapse CURED.** Myopic DKQS was stuck 0.078 tail@q10; non-myopic peaks **0.1111**
  (ep5, best-ckpt; overfits after). Mask + Sigma-state + terminal credit fix the re-ask-same-concept failure.
- **BUT concept-only DKQS still LOSES to strong static:** tail@q10 DKQS 0.1111 < var-greedy 0.1160 < **static
  0.1244**; full@q10 DKQS ~0.2027 < static 0.2059. On BOTH metrics, concept-only adaptivity does NOT beat a
  well-tuned greedy-static questionnaire at 10 questions. **Concept-only adaptivity is genuinely THIN.**
- (FULL-objective arm hung ~8.5h after ep35 -> KILLED; tail arm already gives both metrics. Static/vg cached at
  `.cache/set_mn/dkqs2_static.npz` -> rerun skips the ~70min build.)
- **VERDICT: the non-myopic machinery is sound but has nothing to work with in concept-only. ITEMS (fine drill
  probes, where Jul-14 signal lived) are now the LOAD-BEARING test, not an add-on.** If items also don't beat
  strong-static, the honest paper story = "a strong static questionnaire is hard to beat; adaptivity helps only
  EARLY / in specific regimes" and novelty shifts to the differentiable-continuous machinery + OPEN channels
  rather than "adaptive beats static". NEXT = item cache rebuild (per-user known-half item ratings) + item
  singleton-SEL calibration (popularity-binned) + item-inclusive honesty gate (GO/NO-GO) + items in the DKQS
  action space; then re-run this probe. HELD for author direction (concept-only result reframes the strategy).

**★★★ DKQS DIAGNOSTIC (Jul 19, `scripts/bpool_dkqs2b.py`, ckpt `dkqs2b_ckpt.pt`, concept-only, 16 ep, per-epoch
best-ckpt). DECISIVE: the learned policy COLLAPSED TO A STATIC SEQUENCE, and a WORSE one than greedy-static.**
- **ROUTES = NOT ADAPTIVE:** across 800 eval users, EVERY turn ~ALL users get the SAME concept (q1->553, q2->1092,
  q3->74, ... modal share 1.00); **only 3 distinct full sequences / 800 users** (1=static, 800=personalised).
  The actor IGNORES the belief and emits one fixed sequence for everyone -> a static questionnaire.
- **And a SUBOPTIMAL static:** dkqs tail@q10 0.1132 < greedy-static 0.1244 (gap -0.011). Soft-NDCG-surrogate
  optimisation finds a WORSE fixed sequence than exact-greedy search.
- **NOT overfitting -> UNDERFITTING:** train tail@10 0.1112 ~= eval 0.1132 (eval slightly higher). Generalises
  fine; stuck at a mediocre static optimum. (Revises the earlier "overfit/small-data" read: more users alone may
  not fix a policy that keeps collapsing to static.)
- SEEDING NOTE: fully deterministic (ep0/ep5 bit-identical across runs); the earlier 0.1111-vs-0.1132 difference
  was CHECKPOINT GRANULARITY (run1 evaluated every 5 ep -> caught ep5; run2 every ep -> found ep12=0.1132).
- **RESOLVES author Q1 ("shouldn't it discover static if optimal?"): it DID collapse to A static seq, just a
  worse one. Two separable gaps: (a) NOT adapting (concept answers too weak to route on -> rational collapse),
  (b) even as static it UNDERperforms greedy-static (surrogate/optimisation gap ~0.011).**
- **★ CORRECTED VERDICT (author pushback + HARD RULE #2/#3): this is a TRAINER FAILURE, NOT an adaptivity verdict.
  Do NOT conclude "concept adaptivity is thin/dead" — FORBIDDEN and contradicted by our OWN proof (Jul-14 tree
  +47% different-drill-per-cluster on 150k [[adaptivity-bounded-by-model-uncertainty]]; E3 +0.0085).** The policy
  CAN represent greedy-static (emit fixed picks, ignore state) yet FAILS to reach it -> the OPTIMIZER/ARCHITECTURE
  is broken, so this run says NOTHING about whether adaptivity pays. train=eval is the SIGNATURE of static
  collapse (a static policy can't overfit), reassures nothing. ALSO: eval used the diffuse soft-BLEND fold, NOT
  the snapped clean concept (Fable's flagged control, initially skipped) -> part of the 0.113-vs-0.124 gap may be
  blend-dilution (`scripts/bpool_snap.py` running to separate). **NEXT (author directive): FIX THE MODEL FIRST.
  deep-research on why a differentiable/amortized policy collapses to an input-independent near-static solution +
  fails to reach a greedy baseline it can represent (running, agent), THEN Fable passes on OUR specifics (input
  features [mu,diagSig,turn] sufficiency, capacity, softmax-blend action, soft-NDCG surrogate, BPTT credit).
  Candidate fixes: tree/greedy DISTILLATION warm-start (guarantee >= baseline), sparser action (sparsemax/Gumbel-
  ST), entropy/exploration, richer history-encoder state (DeepSets over asked tokens, cf DAD), dense per-step
  reward. Items/scale ON HOLD until the trainer at least MATCHES greedy-static and shows genuine per-user routing.**

**★★ WHY DKQS COLLAPSED — deep-research diagnosis (Jul 19, verified-lit). SNAP=BLEND (0.1123~0.1132) RULES OUT
blend-dilution; it's a degenerate OPTIMIZATION attractor. FOUR compounding causes (ranked):**
1. **CONDITIONING-COLLAPSE** (posterior-collapse analog, CVAE lit): population-marginal sequence gets most of the
   soft-NDCG -> net learns a mapping CONSTANT in [mu,diagSig] -> input-independent (modal share 1.0).
2. **GRADIENT STARVATION** (Pezeshki NeurIPS'21 arXiv:2011.09468) + softmax winner-take-all + Adam large-logit
   moves -> freezes to a fixed SEQUENCE (repeat-mask then reveals the next fixed pick).
3. **RELAXATION-OPTIMUM != HARD-OPTIMUM**: soft-BLEND (avg of 1317 dirs) rewards NON-COMMITMENT (mean-field) +
   soft-NDCG temperature bias (NeuralNDCG SIGIR'21: surrogate=NDCG only as tau->0) displaces the optimum from the
   SHARP greedy picks -> can't reach greedy-static even though representable.
4. **BPTT CREDIT STARVATION**: late-weight lam~t^2 gradient through 10 CONTRACTIVE (Sig-shrinking) folds vanishes
   to early turns -> early Qs stuck at init=population-mean=the collapse solution.
(Amortization gap, Cremer ICML'18 = background; our failure is WORSE = can't reach amortized-representable greedy.)
**FIX MENU (ranked): (1) DAgger/BC DISTILLATION from the tree/greedy teacher = THE first fix (guarantees
>=greedy-static, seeds OUT of the collapse basin; we HAVE the teacher = bpool_tree/E3 per-cluster-best-concept);
(2) straight-through / GUMBEL-SOFTMAX TOP-1 action (commit to one askable dir, kills non-commitment optimum,
aligns w/ snap-on-askability); (3) DENSE per-step reward (gradient to early turns, Blau ICML'22 RL-BED); (4)
entropy reg + temp anneal (band-aid, use WITH 1-3); (5) richer history encoder LOWER prio -- [mu,Sig] IS the
Bayesian sufficient stat -> OPTIMIZATION not REPRESENTATION; cheap idea: OFF-DIAGONAL/low-rank Sig (diag discards
cross-cov routing may need); (6) curriculum short-horizon-first.** BED field concurs: DAD/iDAD use CONTRASTIVE
bounds (not loose pointwise soft-metric) + perm-invariant arch; Blau adds per-step reward+exploration; Huang'24
decision-aware utility (soft-NDCG decoupled from real tail-NDCG = our cause 3). ★ HIGHEST-LEVERAGE FIRST MOVE =
BC+DAgger from the tree teacher + fine-tune w/ straight-through Gumbel top-1. NOTE the q2 ONE-STEP actor
(bpool_amorte3) DID read the belief (twin-prize +0.0054) -> a 1-step policy adapts; the T=10 UNROLL collapses ->
confirms causes 1+4 (multi-turn credit/collapse), not a fundamental can't-adapt. Fable pass on our specifics running.

**★ BOTTOM LINE:** start with **Family 2 (continuous-action RL, Blau MDP + Wolpertinger snap) WARM-STARTED by
Family 5 (distil the NDCG-tree)** -- direct route to a learned continuous-direction policy on the TRUE
non-diff tail-NDCG reward, clean snap-to-askable (closed) + raw-vector pass-through (open LLM), tree-distill
solves RL cold-start/SNR, RL-finetune escapes the tree vocabulary. Run **Family 4 (NeuralNDCG-through-Kalman)**
in parallel as a low-variance trainer/pretrainer for the closed channel. OBJECTIVE = RL reward (non-diff NDCG)
PRIMARY (the open snap/verbalise is non-diff anyway) + NeuralNDCG gradient AUX. FIRST READ: Huang NeurIPS'24
(2411.02064), then Blau ICML'22. NOVELTY: decision-aware amortized-BED is moving fast (Huang'24 + 2025 goal-
driven/constrained BOED preprints) -> our defensible combination = NDCG/ranking utility + frozen-CF Kalman belief
+ LLM-verbalised OPEN direction channel (no BED paper touches recsys ranking or an open free-text action;
continuous-direction elicitation policy for recommendation appears UNOCCUPIED). Fable architecture pass running.
