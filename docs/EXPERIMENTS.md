# EXPERIMENTS — durable index of what was run

> One row per experiment: what it tested · the result (faithful to the source) · the source doc.
> This is the record that lets old design/plan docs be retired. Grouped by paper. Sources in
> `docs/reference/` (promoted) or `docs/archive/` (not yet reviewed). Add rows as experiments run.
> All chapters filled (A–E + recommender-core + apparatus).

## Paper A — recommender / instrument
| experiment | tested | result | source |
|---|---|---|---|
| Lit replications R1/R2/R3/Golbandi/DRE | reproduce cold-start elicitation baselines | R1 Repr-MF +21% vs MOSTPOP; R2 EAR +47%; R3 EDDI active RMSE 0.816<0.894; Golbandi tree beats static every depth; **DRE did NOT reproduce (honest negative)** | `reference/RESULTS.md` |
| S2 learned set-encoder instrument | encoder vs MOSTPOP, full profile | FULL 0.4863 vs 0.4134 (+17.6%), 4 gates pass, matches WRMF 0.5011 | `reference/RESULTS.md` |
| Root-cause: weak polarity / blockbuster attractor | why instrument under-uses dislike | popularity floor ~48× personalization spread; polarity under-trained by likes-only | `reference/RESULTS.md` |
| Polarity/MNAR fixes (EXPO / CONTRA / two-head) | fix polarity through a collapse gate | **EXPO canonical win** 0.362/0.159; +CONTRA polarity gap +10→+42pp; two-head REJECTED (collapse) | `reference/RESULTS.md` |
| Cheap LLM asker (gpt-4o-mini) | world-knowledge asker on the calibrated instrument | LLM 0.304→0.274 (**worse than random**); world-knowledge ≠ collaborative informativeness | `reference/RESULTS.md` |
| RS-validity / protocol reconciliation | are cold numbers weak, or the protocol? | 100% the held-half+exclude-known protocol (PROT-A 0.4200 vs PROT-C 0.2929), not RS weakness | `reference/POLICY_LOG.md` |

## Recommender-core (belief / fold / dislike — feeds A & B)
| experiment | tested | result | source |
|---|---|---|---|
| RecVAE-d512 instrument rebuild (I2) | strong-CF ruler vs V1/EASE | ML-1M 0.5541 (ties EASE); Goodreads 21.6× V1; ML-25M 0.4998/0.3443, 8/8 gates | `reference/INSTRUMENT_REVIEW.md` |
| RecVAE valence probe + bolt-on channel | is a dislike direction real / usable | z-space carries dislike (−17.8pp, 8/8 genres) BUT a bolted sign channel is **INERT** (ΔNDCG 0.000) | `reference/INSTRUMENT_REVIEW.md` |
| Oracle-Z headroom (frozen RecVAE) | best-possible belief; do dislikes help | likes-only 0.483 → clairvoyant ceiling 0.994 (+0.51); watched-dislikes −0.0016 = the **WALL** | `reference/ORACLE_Z.md` |
| Fold-model lineage (recon/v3/v4/recovered/leakfree) | which fold accumulates without leaking | recon > ridge; v3 clean but LEAKS; v4 cold-collapse 0.167; recovered under-accumulates; leakfree SATURATES | `reference/EMBEDDING_FOLDING_INDEX.md`, `reference/FOLD_MASTER.md` |
| Raw-data dislike anatomy (model-free ML-25M) | is dislike separable / generalizable | corr +0.46; implicit 0.407 > explicit-sign 0.398 > explicit-cent 0.270; disinterest region inert | `reference/RAWDATA_DISLIKE.md` |
| SignedAE (a0c) scorecard | is a signed/dislike channel real & useful | full 0.4954 (val-peak 0.4961), k-curve monotone; genre sign-flip gap +0.257 (3× control); folding disliked genres as neg **HURTS** −0.12@k2 | `reference/SIGNED_LATENT_ANALYSIS.md` |
| Rung-I answer-native encoder (FiLM value gate) | make the value channel non-inert on a frozen decoder | **STOP — value INERT** (ΔNDCG +0.0002 vs 0.005 bar); a Rung-II (decoder-geometry) ceiling, not a loss bug | `reference/RUNG1_CHECKS.md` |
| Early two-tower / LSTM / DDPG (Jan 2025) | first recommender + RL attempts | two-tower 0.389 (68% like/dislike overlap); LSTM+Attn 0.428 (100% collapse); all 3 DDPG runs FAILED (user-agnostic policy) | `reference/RECOMMENDER_TASKS.md` |

## Paper B — concepts / set-encoder reconstruction / policy
> **★ LEAD = uncertainty-shrinkage / belief-distribution.** Design lineage in `docs/design/`
> (`PLAN_BELIEF_HEAD` → `PLAN_BELIEF_SET` → `DESIGN_ATTNPOOL_ENCODER` → `DESIGN_SHEET_VARHEAD` → `DESIGN_PRECACC`).

| experiment | tested | result | source |
|---|---|---|---|
| **★ Belief-pool elicitation invariant** | does a closed-form belief update keep NDCG monotone per question | **first monotone-positive-per-step operator**: Kalman update (frozen decoder, no training), Σ **only shrinks**; oracle greedy +0.090 / random +0.068 tail by q8, **never drops** | memory `belief-pool-satisfies-elicitation-invariant`; `scripts/bpool2.py` |
| **★ Kalman-on-canonical → PrecAcc pivot** | can the Kalman pool ride the strong recommender | pure Kalman **craters** full 0.167→0.097 (only survives on a popb floor) → redesign = PrecAcc (frozen encoder mean + analytic covariance) | memory `kalman-incompatible-unified-redesign`; `design/DESIGN_PRECACC.md` |
| Oracle-imitation / RL / Wolpertinger item-selection | realizable item-selection value (dense ML-1M) | ALL flat ≈ random vs oracle +0.186; no realizable item-selection value; Wolpertinger validated k=1→k=N | `reference/RESULTS.md` |
| Tail / debiased regime | full-cat NDCG hid value | tail elicitation DOUBLES NDCG 0.039→0.093; realizable ceiling OPEN on tail | `reference/RESULTS.md` |
| Reconstruction encoder (attention fold-in) | learned fold-in vs ridge | encoder +21%@q8 tail, FULL +0.047 vs +0.014; ceiling oracle best-subset 0.508/0.333 | `reference/RESULTS.md`, `reference/ALGO_UNIFIED_CONCEPT_SPACE.md` |
| Adaptive EIG selector | realizable adaptive item selection | infogain-EIG +0.072 full/+0.111 tail, ~40% of oracle; **greedy EIG = realizable frontier** | `reference/RESULTS.md`, `reference/POLICY_LOG.md` |
| Answerability × info by question type | items vs genres vs concepts | answer-rate items 2% / genres 63% / concepts 57%; info items≈concepts≫genres → **concepts = sweet spot** | `reference/RESULTS.md` |
| Concept-coarseness ceiling | can concept belief reach item-level | concept belief saturates cos~0.83 with u* (items ~1.0); graded answers don't raise it — fundamental ceiling | `reference/RESULTS.md` |
| **entdistill / CASPER-R winner** | entropy-BC floor + reconstruction refinement | seed-avg TAIL 0.152±0.002 vs fair entropy 0.141 (+0.011, ~6σ); beats HELF/Golbandi/NICF/LLM; ckpt `policy_entdistill_ep4` | `reference/POLICY_RESULTS.md`, `reference/OVERNIGHT_CAMPAIGN_2026-06-24.md` |
| Policy ladder O1–O13 / crippled-baseline fix | learned policy vs conc_pop | no learned policy cleanly beats conc_pop; earlier "beats both axes" was a crippled-baseline artifact (fair entropy 0.355/0.150) | `reference/POLICY_LOG.md`, `reference/POLICY_LADDER.md` |
| Full-NDCG reward campaign | reward-variable gating | de-biased-full reward KEEP 0.347/0.144; cos-belief/AUC discarded | `reference/CAMPAIGN_LOG.md` |
| Attr/concept answer recipe (ML-1M) | make popular-attribute elicitation monotone | graded-lift + additive channel + bounded coarse-weight ~0.4 → genre 0.320→0.393 monotone; unbounded DECLINES | `reference/ATTR_ANSWER_LOG.md` |
| Hand-set concept re-rank operator | `score=popb+⟨Wd,u⟩` whitened+IDF | β≈120: TAIL +41%, FULL never drops; belief-token fold = WRONG operator (−0.037) | `reference/DESIGN_SHEET_V4_RERANK.md` |
| LLM static-asker capability gradient | opus/sonnet/haiku blind static | TAIL opus 0.130 > sonnet 0.111 > haiku 0.092; all < ours 0.152 | `reference/POLICY_RESULTS.md` |
## Paper C — continuous-action elicitation
| experiment | tested | result | source |
|---|---|---|---|
| Continuity vs discrete (headroom) | does a continuous actor beat discrete under graded answers | actor 0.366/0.162 > CASPER-R (+0.006 full/+0.010 tail); needs GRADED answers (binary ties); privileged continuous oracle > discrete +0.029/+0.049 | memory `paperC-continuity-headroom-confirmed` |
| R2-noise squeeze | noise-adapted static vs noise-trained actor | static repeat schedule 0.3143/0.1167 **beats** actor 0.2844 (Δ−0.0299, CI[−0.040,−0.020]) → **adaptivity is clean-channel-only** | `reference/HANDOFF_PROGRESS_2026-07-05.md`; memory `squeeze-r2-adaptivity-clean-only` |
| R2b tie-by-construction | actor warm-started from the winning schedule | ties static (Δ+0.0035, p=0.82) → defeat = optimization gap, not expressiveness | `reference/HANDOFF_PROGRESS_2026-07-05.md` |
| Dislike-signal GO/NO-GO probe | do oracle dislikes add short-k NDCG | **GO**: signed-EASE +0.0583 at k=2 (+17.5%), decays with k; over-repulsion α≥1 hurts | `reference/DISLIKE_SIGNAL_PROBE.md` |
| Bot-play answer-source (T4) | denoised-mean vs sampled answerer | continuous actor 0.407/0.222 vs sampled 0.150 — the collapse was sampling noise | `reference/HANDOFF_PROGRESS_2026-07-05.md` |
| SNAP-K defense (OPEN) | is Paper C's snap-loss a Wolpertinger k=1 artifact? | open question / paper defense — no in-doc result; confirm a SNAP-K run exists before closing | `reference/DESIGN_SHEET_E2_SNAPK.md` |
## Paper D — open-vocabulary free-recall
| experiment | tested | result | source |
|---|---|---|---|
| T6 Paper-D fair comparison @m=4 | open-recall vs PEBOL at a matched budget | OPEN 0.3926/0.1776 ≈ PEBOL+profile 0.3899 ≈ ConTS 0.3796 (oracle 0.4047) → tie at m=4; edge is operational | `reference/HANDOFF_PROGRESS_2026-07-05.md`; memory `paperD-open-recall-beats-continuous` |
| Open free-recall vs D1 / hidden-gem framing | question-framing as a head/tail lever | open free-recall 0.378/0.178 beats D1; hidden-gem@K=2 0.382/0.193 (~2 Qs); favourite=head, hidden-gem=tail | memory `paperD-open-recall-beats-continuous` |
| Favourite = high-affinity-region sample (idea) | how a volunteered favourite should fold | a named favourite is drawn from the tight high-affinity cluster, not a random highly-rated item → fold as a high-affinity anchor (untested on the 173) | `reference/ANSWERER_TODOS.md` |

## Paper E — deployment / LLM verbalisation
| experiment | tested | result | source |
|---|---|---|---|
| Triangulation → sparse phrase blend | continuous query → signed phrase blend for rendering | OMP k=3 fidelity 0.84; LLM must synthesise the AXIS, not enumerate | memory `triangulation-llm-rendering` |
| Scoop-risk + PROBE-V falsifier (methodology) | is an LLM-verbaliser paper defensible | ELM (same Google group) is one merge away; **random-direction baselines match trained SAEs** (an LLM names a random direction fluently → zero evidence); PROBE-V: LLM must beat q_random AND the k=3 OMP blend (0.84) | `reference/FABLE_VERDICT_2026-07-14.md`, `reference/PLAN_LLM_VERBALIZER.md` |

## Apparatus — answerer / answerability
> Frozen specs consulted by any model: `reference/answerer_schema.md`, `ANSWERER_V1_DESIGN_REVIEW.md`,
> `DESIGN_SHEET_DISTILLED_ANSWERER.md`, `RICH_DATASET_REVIEW.md`.

| experiment | tested | result | source |
|---|---|---|---|
| Answerability GATE (ML-25M, LLM-judged) | does answerability vary per-user & track taste | PASS 4/4: ICC 0.174 (p=.003), taste OR 2.71/sd, dAUC 0.106, G2 dNDCG 0.287; cost $1.29/300 users | `reference/ANSWERABILITY_RESULTS_LOG.md` |
| Cross-checks (EASE value + Haiku judge) | is the signal a single-model / shared-prior artifact | EASE MAE 0.740 (beats item-mean); GPT-vs-Haiku agreement 83.2%, κ 0.668 | `reference/ANSWERABILITY_RESULTS_LOG.md` |
| Fitted P(answerable) surrogate | scalable answerability scorer | AUC held-out 0.921, CV 0.916, ECE 0.008; top coefs log_ratings +1.55, is_concept +1.27, genre_match +0.63 | `reference/ANSWERABILITY_RESULTS_LOG.md` |
| Answerer v2.1 flutter decomposition | distilled LLM-judge simulator fidelity | item value corr 0.550 ≈ LLM 0.547; k2 variance = 34% question / 4.5% real trait / 22% flutter | `reference/DESIGN_SHEET_DISTILLED_ANSWERER.md` |
