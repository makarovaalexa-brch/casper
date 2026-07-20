# Policy Method-Search Ladder — disciplined, one-variable-at-a-time (anti-thrash)

## CONVERGENCE (verified): P4@40ep tail 0.122 ≈ P4@25ep 0.125 (no gain from more training) => 25 ep is CONVERGED, numbers trustworthy (±~0.006 single-seed). Earlier delta@15→@25 rise was just under-training at 15.

## ORACLE DISTILLATION — RESULT (ep25, single-seed test-150): tail-leaning small beat; CEILING is huge
| method | FULL | TAIL |
|---|---|---|
| conc_pop | 0.315 | 0.116 |
| entropy | 0.320 | 0.127 |
| **oracle-distilled+tail-FT (ep25)** | 0.309 | **0.130** | (tail +.014 vs pop, +.003 vs entropy; FULL loses to pop −.006; 21-traj/22-vocab, monotone tail curve 0.088→0.112→0.126→0.130) |
| **conc_oracle CEILING (privileged)** | 0.332 | **0.354** | (3x entropy — selection headroom is MASSIVE; realizable captures ~1/6) |
KEY: checkpoints policy_poracle_bc.pt (BC-floor 0.291/0.121) + policy_poracle.pt (finetuned 0.309/0.130, backed up to policy_poracle_ep25.pt). TEACHER cached oracle_demos_tail.npz. RL noisy + non-monotone (dip ep4-9 to 0.119 then climbs to 0.132 by ep22) => NOT converged at 25 => RESUMED +50.
## +50 RESUME RESULT (Jun 2026): **STALL CONFIRMED.** 3 epochs oscillating tail 0.127/0.129/0.125, train-return FLAT 0.0801/0.0802/0.0819 => vanilla REINFORCE is PINNED at the entropy level (0.127); more epochs won't escape. The +ep1 dip 0.130->0.127 was RL noise + Adam optimizer-state reset on resume (momentum restarts), NOT overfitting (recovered to 0.129 @ +ep2). KILLED the resume (user-approved) to pivot.
## P-REG (Jun 2026, RUNNING bhssn026q): **regularize to escape the plateau** + **best-checkpoint/early-stop**. (1) ENTROPY BONUS loss-=beta*mean_t H(pi), ENT_COEF=0.003 — keeps policy exploratory to escape the ~22-vocab local optimum. (2) WD env = L2 weight-decay (0 this run; one-variable). (3) best-val checkpoint saved on a SEPARATE val te[150:400] (disjoint from locked test-150 -> NO leakage from selection/early-stop), USEBEST=1 evals the best. Resumes from BC floor (=0.130 run's start), EP=40, SELVAL=tail. One variable vs the 0.130 run: +entropy reg. Code: continuous_policy2.py ENT_COEF/WD/SELVAL/USEBEST + _bestvt save.
## ★ P-REG RESULT (Jun 2026) = CURRENT WINNER: best-ckpt @ep2 → **test-150 FULL 0.317 / TAIL 0.130** (best BALANCED learned policy; dominates no-reg 0.309/0.130 + conc_pop 0.315/0.116; edges entropy tail .130 vs .127). DECOMPOSITION: the ENTROPY BONUS was a WASH (val peaked @ep2 then bled reward −26% over 13ep, no climb — exploration is NOT the lever; plateau = imitation gap, signal/representation limit). The VALUE came from **EARLY-STOP/best-checkpoint** (caught the early peak; heavy tail-finetuning erodes full 0.317→0.309). → KEEP best-checkpoint+early-stop as standard; DROP entropy bonus. Caveat: tail beat over entropy within noise (±.006) — NOT yet the decisive beat; single-seed. Checkpoint policy_poracle_reg_best.pt (=policy_poracle_pk.pt). Recorded in POLICY_RESULTS.md. NEXT LEVER = attention state (P6) for real headroom vs the imitation gap. conc_oracle/teacher objective-aware (REWTAIL=tail else full); FULL-NDCG version queued after.

## ORACLE DISTILLATION (orig plan)
Rationale: policy ALREADY has pop+answerability+entropy+infogain as features yet only MATCHES entropy => it's a SELECTION gap, not a feature gap. Headroom is real (full-profile 0.185 tail; privileged concept-oracle ~0.32). FIX = teach the per-user selection: BC the feature-rich policy (P4 feats) to imitate the CLAIRVOYANT GREEDY oracle (per user, pick concept maximizing TAIL-NDCG on held items), then finetune. Code: BCTGT=oracle + conc_oracle ceiling eval. If distilled policy clears entropy 0.127 => combine-and-beat confirmed; if not => cold-start genuinely can't support per-user selection at 8 Q (honest ceiling) => pivot to open free-text questions.

## PC2 — ANCHORED CONTINUOUS elicitation (fix the PC collapse; the genuine "continuous beats discrete") [TODO]
PC (cont_elicit.py) emitted novel decodable middle-grounds but COLLAPSED (actor ignored u, emitted one niche point). NOT Wolpertinger: snapping back to existing concepts ≈ the discrete scorer (we already exhaustively score the 1361 pool, so snap gives no gain; snap also kills the novel-middle-ground that's the whole point). FIX = anchored/critic-regularized continuous emission, warm-started from the best DISCRETE model:
  1. WARM-START actor to emit the embedding of the discrete scorer's top pick (avoid cold-collapse).
  2. REGULARIZE emitted e: frozen discrete scorer as a CRITIC (penalize low-rated points) + on-manifold term (keep e near the concept cloud) + within-session DIVERSITY term (stop single-point collapse). Reuse pretrained scorer as anchor/critic, NOT backbone (per-candidate vs per-state I/O differ → no shared layer to freeze).
  3. per-user CONDITIONING so emission uses the belief u.
Decode e->nearest concepts for NL. Win condition = beat discrete tail (≈0.13) WITH novel answerable middle-grounds.

## TARGET (what we must beat — seed-avg n=6, the robust numbers)
- **TAIL NDCG@10: entropy 0.127±0.006** (the one ROBUST heuristic win, ~2σ over conc_pop 0.115). PRIMARY target.
- FULL NDCG@10: ~0.33 cluster (entropy .331 / tree .329 / greedyext .326 / conc_pop .325 — TIED within ~1σ). Just don't fall below it.
- Also report RMSE (tree .958 best; entropy .995 worst — discrimination-vs-calibration tradeoff).
- Headroom for context: full-profile warm fold 0.380 (realistic ceiling); oracle 0.587 (privileged).

## RULER (fixed, never changes mid-search)
ML-1M, concept-aware encoder (enc_concept), GEOMETRIC answers, unified item+concept pool (PITEMS 600 + 761 concepts),
realistic cold-start (known-half profile), full-catalogue, NDCG@10 + Recall@50 + RMSE, head-33%/tail.
- Ladder eval = **locked 150 split (seed 123)** for speed.
- Winners RE-CONFIRMED on **seed-avg (seeds 123,1,2,3,7,11)** before they count.

## PROTOCOL (the anti-thrash rules — STRICT)
1. **One variable per rung.** Change exactly one thing vs the current best.
2. **Train seeded, SAVE every checkpoint** to `.cache/policy_<rung>.pt` (TAG=<rung>). LOAD=1 to re-eval (no retrain).
3. **Log every run** in the table below: rung, change, FULL/TAIL/RMSE @q8, picks(i/c), distinct-traj, cos, verdict.
4. **GATE to KEEP a change:** must beat current-best **TAIL by > 0.012 (~2σ)** without dropping FULL below the ~0.33 cluster.
   - Within-noise (<2σ) = NOT a win; revert. Combine ONLY rungs that individually passed the gate.
5. **Catch collapse early** (instrumentation already in eval): ans/8, picks i/c, distinct-1st, distinct-traj, total-vocab, cos(u,u*) per turn. Flag if concept-only / fixed-shortlist / cos-degrading.
6. No moving the ruler, no inventing metrics, report BOTH full+tail, no overclaiming, single-seed margins are provisional until seed-avg.

## BASE = P0 (FIXED REFERENCE)
**O12** — no-BC REINFORCE, scorer over unified pool, features [emb, align(u·E), pop/answerability-prior, pop-info-gain, belief-strength, turn], COVERAGE reward (Δ held-out like-coverage), straight-through.
Checkpoint `policy_o12.pt`. **FULL 0.309 / TAIL 0.107 / RMSE 0.973.** Concept-only (0i/1170c), 5-traj/18-vocab shortlist, cos peaks q2 then degrades.

## IDEA INVENTORY (everything from this project + lit, grouped)
**Reward/objective** (REINFORCE reward value, or training loss):
- coverage of held-out likes (O12, base) — gameable/anti-entropy [DONE]
- rank-aware likes−nonlikes (O13) — worse tail [DONE, lost]
- **direct held-out NDCG@10** (O14, coded REW=ndcg) — the actual target
- u*-reconstruction 1−cos (O7) — stalled [DONE, lost]
- **held-out PROFILE reconstruction** (BCE on held-out likes, NOT interview items) — user's idea; recon() exists
- penalize-unanswered PEN term [in base]
- joint NDCG + reconstruction
**Per-candidate features** (the policy's "smart" inputs):
- emb, belief-alignment u·E, belief-strength, turn [base]
- popularity / answerability prior [base; SEPARATE pop vs answerability untested]
- population coverage-info-gain POOL_IG [base]
- **ENTROPY / divisiveness** (the signal entropy-heuristic wins with — MISSING from base!) (coded FEATS=ext)
- **avg-rating** per candidate (coded FEATS=ext)
- explicit answerability (split from popularity)
**Architecture:**
- per-candidate MLP scorer on frozen-encoder belief [base]
- deeper / wider scorer
- learned ATTENTION state over interaction history (vs frozen belief) — NICF-style
- separate item vs concept heads
**RL family / optimisation:**
- REINFORCE (base)
- actor-critic / A2C baseline (variance reduction)
- PPO (clipped, stable)
- DQN value-based (NICF — failed, asked items, under-trained: revisit done-right)
- straight-through supervised (recon) vs policy-gradient
**Init / pretraining:**
- no-BC (base) — learns but stalls at shortlist
- BC-to-conc_pop (froze — BC-trap) [DONE, lost]
- warm-start from entropy as FEATURE-init (NOT BC target) — anti-freeze
**Action space:**
- unified item+concept [base]
- concept-only / item-only ablation (mechanism check)
- Wolpertinger continuous action (the original S5)
- non-myopic look-ahead

## LADDER (prioritized rungs; run one at a time, gate each)
| rung | change vs current best | FULL | TAIL | RMSE | picks/adapt | verdict |
|---|---|---|---|---|---|---|
| P0 | BASE = O12 (coverage reward) | 0.309 | 0.107 | 0.973 | 0i/1170c, 5-traj | reference |
| P1 | reward → direct NDCG@10, ABSOLUTE (prev=0) | 0.313 | 0.113 | 0.973 | 0i/1113c, 14-traj, cos↑.782 | cos RISES + 14 traj (O12: degraded/5) — but reward absorbed q0 baseline (turn-0 bias). SUPERSEDED by P1b. |
| P1b | reward → NDCG DELTA-OVER-q0 (prev=q0), EP=15 | 0.306 | 0.107 | 0.980 | cos .720, 14-traj | undertrained (delta has smaller advantage magnitude → slower learning, user diagnosis) |
| P1b@25 | same, resumed +10 ep → 25 | 0.312 | 0.111 | 0.974 | cos .755, 12-traj | ≈ P1-abs ≈ conc_pop. EP confound confirmed: delta just trains slower; at 25ep it catches up. Reward def NOT the lever. |
| P1s | delta-over-q0 + REWSCALE=20 | — | — | — | — | KILLED: Adam is scale-invariant → REWSCALE is a NO-OP (traces matched unscaled ÷20 exactly). Real speed knob = LR, not reward scale. |
| **P2** | + **entropy+avg-rating features** (FEATS=ext), abs-NDCG, EP=25 | 0.313 | **0.120** | 0.984 | 17-traj, cos↓.707, RMSE↑ | **PASS (keep feat): TAIL 0.113→0.120, first learned policy to EXCEED conc_pop (0.116) on tail; moved INTO entropy regime (divisive picks). Still <entropy 0.127 + within ~1σ of conc_pop → seed-confirm. ckpt policy_p2feat.pt** |
| P3 | + held-out reconstruction aux (λ=0.5 collapsed; λ=0.1) on P2 | 0.299 | 0.099 | 1.014 | 200i/367c, 66-traj, ans 3.8 | **REJECT**: recon wants item-level signal → picks ITEMS (unanswerable) → tail tanks. Confirms answerability thesis. P2 stays best. |
| P3 | + held-out **reconstruction** auxiliary loss | | | | | |
| **P4** | + explicit **answerability** feature (ANSF, on P2) | 0.311 | **0.125** | 0.980 | 17-traj, cos .745, 0i | **PASS (keep): TAIL 0.120→0.125, NEW BEST, beats conc_pop +0.009, ≈ entropy 0.127 (within noise). Divisiveness+answerability features STACK. ckpt policy_p4ans.pt** |
| P5 | deeper/wider scorer (HID=256) on P4 | 0.313 | 0.117 | 0.976 | cos↑.768 | **REJECT**: tail 0.125→0.117 (back to conc_pop); more capacity drifted to calibration (cos/RMSE up) away from tail. P4 stays best. |
| P6 | **attention state** over history (re-distilled oracle+attn) | | | | | **RUNNING** b01r4cegm |
| P7 | RL: **actor-critic / PPO** (baseline) | | | | | |
| P8 | init: entropy **warm-start** (anti-freeze) | | | | | |
| P9 | action-space: concept-only ablation; Wolpertinger | | | | | |
| **PC** | CONTINUOUS elicitation (actor emits embedding, geometric answer, decode→words) | 0.288 | 0.103 | — | 1200/1200 middle-grounds, decodable | **MECHANISM SHOWN (emits novel answerable middle-grounds, decodes to NL blends e.g. musicians/glbt) but ACTOR COLLAPSED to a fixed niche direction (ignores u) → underperforms. Needs entropy-reg/TD3 continuous RL. script cont_elicit.py, ckpt actor_cont.pt** |

Rationale for order: P1 (right objective) + P2 (the missing winning signal = entropy) are the highest-expected-value, cheapest changes → do first. Architecture/RL-family (P5-P7) are bigger lifts, only if P1-P4 plateau below entropy. Combine cumulative winners only.
