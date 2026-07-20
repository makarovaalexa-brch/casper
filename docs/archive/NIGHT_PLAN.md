# Overnight rescue plan — capturing the adaptive edge (2026-06-14)

## The problem (precisely)
- Held-out metric (excludes directly-asked targets; the honest scoring). On stratified MovieLens (300 movies across popularity bands + 59 attrs), 300 test users:
  - adaptive heuristics: scpr 0.7210, greedy_infogain 0.7157, greedy_answerability 0.7130 (all BRANCH)
  - PPO honest-reward: 0.7192 but STATIC (2 distinct sequences, no branching)
  - popularity (static): 0.7006 ; random 0.6811 ; thompson 0.6523
  - clairvoyant attr-only routing ORACLE: ~0.796  (target-probing oracle 0.836 but unrealizable)
- So: adaptive > static is real but small (+0.02); the ORACLE shows +0.10 of routing value that NO realizable method captures; RL collapses to static.

## Reframe that saves the thesis
"Personalised elicitation has large LATENT value (oracle ~0.80 vs static ~0.70); current CRS methods (greedy heuristics AND naive PPO) capture only ~1/5 of it; CASPER's new method captures more." Stronger than "RL beats random."

## Root-cause hypotheses (PROVE these)
- H1 reward SNR: per-step adaptive advantage (~0.02) buried in per-user reward variance -> PG converges to best FIXED order. (On synthetic, large edge -> PPO DOES branch -> supports SNR not capacity.)
- H2 wrong signal: greedy IS one-step-optimal adaptive; RL's only value-add is non-myopic lookahead, unfindable by noisy MC return. Fix = amortized EIG objective (DAD/iDAD) or privileged teacher.
- H3 actor architecture: flat MLP->N logits has no per-candidate inductive bias; need permutation-equivariant per-item scorer (deep-sets/pointer actor).
- H4 realizability limit: oracle actions may be unpredictable from observable belief state -> edge information-limited -> impossibility proof (also a strong result).

## Candidate methods (stay in actor-critic / policy-gradient family)
1. **Privileged-oracle distillation** (asymmetric imitation): train student policy on realizable belief state to imitate clairvoyant oracle. Prototype = scripts/paper2/oracle_distill.py. Decides H4.
2. **Amortized sequential EIG policy** (DAD/iDAD-style): policy net maximizes total expected information gain via sPCE/InfoNCE bound (lower variance + non-myopic than MC reward). Lit agent 1.
3. **Equivariant actor** + asymmetric critic (privileged value fn at train time): combine 1/2 with better actor architecture.

## Experiments
- RUNNING: oracle_distill.py on stratified (b8ol71t1t) -> teacher AUAC, student AUAC, BC oracle-action top1, student branching. DECISION: student near 0.796 => edge realizable, distillation is the method; student ~0.72 => info-limited => impossibility proof.
- 4 background research agents: (1) amortized BOED/DAD, (2) privileged distillation/asymmetric AC, (3) PG collapse/SNR fixes, (4) CRS acquisition + novelty.
- TODO next: SNR diagnostic (gradient SNR, advantage variance within/between state); behavior-clone greedy into actor (capacity test); k-step belief-lookahead vs greedy (non-myopic headroom); instrument calibration.

## Key files
- scripts/paper1/oracle_ceiling.py (clairvoyant ceiling: full 0.836, attr-only 0.796)
- scripts/paper2/oracle_distill.py (privileged distillation prototype)
- scripts/paper1/testbed.py (held-out metric: _movie_metrics exclude=, accuracy_heldout, summarize auac_heldout)
- scripts/paper2/env.py (held-out reward: CASPER_HELDOUT, _excl(), meas_prev consistent-denominator reward)
- experiments/paper1/benchmark_ml_stratified.json (held-out numbers)

## Infra note
Background training jobs die at ~42min (~2500s) — likely wall-clock kill. Size training <= ~8k PPO episodes per job; bench partials from saved best-val checkpoints.
