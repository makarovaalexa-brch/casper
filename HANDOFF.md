# CASPER — Project Handoff (state as of 2026-06-13)

Continuation guide for the CASPER conversational-elicitation research
programme. Three outputs: **Paper 1** (testbed, near-complete),
**Paper 2a** (continuous-action method, code-ready), **Paper 2b**
(verbal bot-play, planned). Read this before resuming.

---

## 0. The one-paragraph status

A controlled testbed (Paper 1) measures *strategic preference
elicitation* in isolation: a fixed, audited recommender "instrument"
scores how much a questioning policy learns per turn, against verifiable
simulated MovieLens users, with every policy choosing from a shared
entity slate. Twelve policies benchmarked across two real slates + a
synthetic world. Headline findings are in §3. Paper 2a (continuous
semantic action space) and 2b (LLM prompt-optimised questioner) build on
the same testbed.

---

## 1. Repo geography

```
casper/
  scripts/paper1/        # testbed, instruments, policies, benchmarks
    test_instrument_lib.py     # instrument loaders + wrappers (single & dual head)
    testbed.py                 # ElicitationEnv profiles, simulator, episode runner, metrics
    policies.py                # ALL policies (baselines, LLM, RL adapters)
    run_benchmark.py           # slate-1 benchmark CLI
    run_benchmark_slate2.py    # slate-2 benchmark CLI
    train_instrument_v5.py     # accepted slate-1 instrument (single head)
    train_instrument_s1_dual.py / _slate2_dual.py  # dual-head (answerability) instruments
    train_botplay_dual.py      # bot-play REINFORCE, world+dual parameterised
    train_dqn_policy.py        # UNICORN-style dueling DQN, world parameterised
    synthetic_sanity.py        # synthetic "indicator world" + dual instrument
    synthetic_rl_improved.py   # REINFORCE-v3 + PPO on synthetic
    synthetic_baselines.py     # popularity/SCPR/Thompson/DQN on synthetic
    recompute_metrics.py       # NDCG@10/Hit@10/AUAC@k by replaying episode logs
    make_figures.py            # all 11 figures from episode logs
  scripts/paper2/
    env.py                     # SHARED RL env (world='slate1'|'slate2', state_mode='liked'|'dual')
    train_discrete_ppo.py      # PPO comparator (CASPER_WORLD, CASPER_DUAL env vars)
    train_continuous_policy.py # CASPER-CA: TD3/Wolpertinger -- WRITTEN, NEVER RUN
  experiments/paper1/          # benchmark_results.json, benchmark_slate2.json,
                               # synthetic_sanity.json, episodes_*.jsonl, *.pt, llm_cache.jsonl
  experiments/paper2/          # discrete_ppo*.pt (4), NO casper_ca.pt yet
papers/paper1_casper/          # paper1_casper.tex (~7.2k words), figures/, tables/, references.bib
lit_review/overleaf_report/chapters/literature_review_2026.tex  # ~7.7k-word chapter
PAPERS_PLAN.md                 # titles, abstracts, plan
CASPER_Expert_Review_2026-06.md, LitReview_Verification_*.md     # rationale docs
```

Run everything from the `casper/` root via `poetry run python scripts/...`.
Machine: i7 4c/8t, 16GB, CPU torch 2.2.2. No GPU needed.

---

## 2. NON-NEGOTIABLE methodology rules (learned the hard way)

1. **Audit behaviour before any leaderboard claim.** Point metrics lied
   repeatedly. Run the adaptivity audit (turn-1 concentration, unique
   entities, answer-conditional branching from `episodes_*.jsonl`) on
   every learned policy. A "winning" policy that is a fixed playlist is a
   finding, not a win.
2. **Instruments must pass acceptance gates before use** (`test_instrument_v2.py`):
   genre-DiD polarity >=80%, franchise-flip DiD, monotonicity rho>0.9,
   liked/disliked overlap <30%, SNR reported. Five instrument generations
   failed differently before v5/dual passed.
3. **Canonical label scheme = absolute, attr support >=3**
   (`build_profiles(..., attr_min_support=3, taste_margin=None)`). Must
   match between instrument training and simulator. A mismatch silently
   corrupted a whole benchmark once.
4. **LLM calls: alway via the cache + validated parser.** `llm_cache.jsonl`
   makes re-runs free; the parser is offline-validated to ~99.5% on cached
   responses. Failure counters are thread-aggregated. Never claim LLM
   results without checking `llm_parse_failures / llm_calls` (a parser bug
   once made 93% of "LLM decisions" random fallbacks).
5. **Measurement instrument stays fixed per slate** (v5 single-head on
   slate 1). Dual-head models are policy-side *decision aids* only, so
   accuracy stays comparable across rows.

---

## 3. Findings to date (defensible, evidence-backed)

- **Instrument science:** 5 generations; failures = OOD-on-partial-reveals,
  rater-harshness collapse, label-sparsity, etc. Final set-encoder passes
  all gates. The *answerability discovery*: policies need P(rated) beliefs
  (dual head), not just P(liked), to express adaptive routing.
- **Slate 1 (top-100):** top tier statistically tied — ppo_dual 0.7524,
  bot-play 0.7516, greedy 0.7472 AUAC@15. All learned policies are
  **static playlists** (audited). Blockbuster slate => static near-optimal.
- **Slate 2 (mid-pop, ranks 101-400 + genome tags):** adaptive heuristics
  separate from static (scpr 0.7053 vs popularity 0.6942, p=0.001) but
  best static learned (ppo_dual 0.7077) still ties best adaptive
  (p=0.23). Dual beliefs help marginally, still static.
- **Synthetic "indicator world":** known analytic optimum. PPO **learns
  genuine adaptive routing** (branches, beats hand-built oracle on AUAC);
  REINFORCE saturates the best static policy; greedy is myopia-bound.
  Proof that adaptivity is learnable *when it pays*, and that the
  optimiser (PPO vs REINFORCE) is the boundary.
- **LLMs (gpt-4o-mini, provisional):** all 3 prompt styles below greedy;
  needs clean re-run on fresh quota (see §5).

---

## 4. Per-paper status & remaining work

### Paper 1 — ~80%, near-submittable
DONE: testbed, instruments, 12 policies, slate-1/slate-2/synthetic
results, ~7.2k-word draft (1 RESULTS + 1 TODO placeholder left),
11 figures, lit-review chapter.
REMAINING:
- Fold dual-belief + DQN + synthetic-baseline rows into tex tables.
- Clean LLM rows: slate-2 + slate-1 strategist (tomorrow's API quota).
- Paired-bootstrap CI table for all rows; refresh figures.
- A7 (MF reference instrument) — reviewer-proofing, optional.
- Final prose polish of results/discussion.

### Paper 2a — ~25%, code-ready, unrun
DONE: shared env, TD3/Wolpertinger trainer (`train_continuous_policy.py`,
complete with critic re-ranking + executed-entity replay), discrete-PPO
comparator (run, 4 checkpoints), synthetic motivating result.
REMAINING:
- Smoke-test + train `train_continuous_policy.py` (NEVER RUN). Output
  `experiments/paper2/casper_ca.pt`.
- Continuous-vs-discrete ablation (CASPER-CA vs discrete PPO) on
  synthetic + slate 2.
- Write the draft (none exists; abstract in PAPERS_PLAN.md).
- HONEST RISK: discrete PPO already goes static on real slates; CASPER-CA
  may tie it. A null/"when does continuous help" result is still
  publishable but is not the originally-hoped positive headline.

### Paper 2b — ~5%, plan only
DONE: title + abstract paragraph in PAPERS_PLAN.md. NO code, NO results.
REMAINING: everything — design the verbal/prompt-optimisation loop
(Reflexion/OPRO over the strategy prompt, scored by the testbed reward),
implement, run, ablate, write. Most LLM-heavy of the three (see §6).

---

## 5. Queued / pending jobs

- RUNNING (CPU, $0): DQN-slate2 -> slate2 dqn row -> synthetic
  popularity/SCPR/Thompson/DQN (task `b4bk1kz8t`).
- DEFERRED to fresh API quota (gpt-4o-mini 10k/day): slate-2 LLM rows
  (vanilla/gate/strategist) + slate-1 strategist clean re-run. Cache-aware;
  parser validated. Use `run_benchmark*.py --llm --confirm-llm-cost`.
- OPTIONAL backlog: A7 MF instrument, B1 free-form LLM arm, B3 second
  dataset (LastFM/Yelp), multi-model LLM ladder (needs work API key).

---

## 6. Paper 2b cost & LLM-intensity (the user's question)

2b is the ONLY LLM-in-the-training-loop paper. Rough call budget for a
thorough study (both slates, ablations, held-out eval): ~30k–60k QBot
calls (each ~1.4k input / ~100 output tokens), most cacheable.

- Dev + optimisation on gpt-4o-mini: ~$20–40 (cache makes re-runs ~free).
- Headline results on ONE frontier model (e.g. GPT-5.4 @ $2.50/$15):
  ~$150–250 for the distinct runs; GPT-5.5 ~2x that.
- All-in publishable estimate: **~$200–500**, trimmable to ~$150 by
  optimising on the cheap model and reserving the frontier model for the
  final eval only. Requires the work API key.

---

## 7. Decision rules
- 2a vs 2b priority: if continuous-vs-discrete ablation (2a) is null,
  pivot energy to 2b (genuinely novel, LLM-native, matches "LLM-heavy"
  goal). If 2a shows a real continuous advantage anywhere, it's the safer
  paper.
- Any new slate/world: train instrument -> pass acceptance gates ->
  benchmark -> audit adaptivity -> only then report.

---

## 8. Paper 2a — full design notes (don't lose these)

**Title:** *Learning What to Ask: Continuous Semantic Action Spaces for
Preference Elicitation in Conversational Recommendation.* System name in
paper: **CASPER-CA**. It is the original CASPER continuous-action idea
(the thing the user struggled with for years), rebuilt correctly.

**Mechanism (what `train_continuous_policy.py` implements):**
- Wolpertinger architecture (Dulac-Arnold et al. 2015): a DDPG/TD3 actor
  emits a continuous *proto-action* in the SBERT entity-embedding space;
  the k nearest unasked entities are retrieved (FAISS/cosine), the critic
  re-ranks them, argmax is executed. k=20 currently.
- TD3 stabilisation: twin critics, delayed actor updates, target-policy
  smoothing — with the smoothed target action **also grounded to a real
  entity** so target-Q stays on the embedding manifold.
- Reward = instrument BCE-loss reduction (the IJCNN bot-play reward).
- Free episodes from the verifiable simulator (no LLM in training loop).

**The five expert-review fixes vs the original failed CASPER (V1–V3) — these
ARE the methodological contribution, state them explicitly:**
1. Critic re-ranking with k>1 (original used k=1 hard snap → action
   aliasing → ill-posed critic). THE key fix.
2. Replay stores the **executed entity's embedding**, never the
   proto-action (original stored proto-actions → Q-targets decoupled from
   realised actions).
3. TD3 twin-critic/delayed-update/target-smoothing (original plain DDPG
   was unstable).
4. Rule-based simulator → 10^5 free episodes (original LLM-in-loop capped
   at ~1k transitions, a 100x sample deficit).
5. Accuracy/BCE-delta reward with SNR>1 (original NDCG-delta reward had
   SNR≈0.025, literally unlearnable — see CASPER_Expert_Review).

**Novelty framing (verified via deep-research, June 2026 — see
LitReview_Verification doc & memory). CLAIM SURVIVES ONLY AS A COMBINATION:**
- MUST cite **Wolpertinger (Dulac-Arnold et al. 2015, arXiv:1512.07679)**
  as the direct mechanism ancestor — it even has a recommender demo. Do
  not claim the continuous-action+kNN mechanism as novel.
- Also cite **ECoC (arXiv:2408.08047)** — continuous-action RL for
  sequential recommendation (mechanism prior art, no dialogue).
- Nearest RL-policy+LLM-verbaliser analogues: **RSO (arXiv:2509.26093,
  2025)** — RL planner over 13 *discrete* strategies + frozen LLM; and
  **PPDPP (Deng et al., ICLR 2024)**. Differentiate: CASPER-CA selects
  *entity-level question content in a continuous semantic space*, not a
  closed strategy set.
- Defensible claim: "first to apply Wolpertinger-style continuous
  semantic-embedding actions to elicitation question selection in CRS,
  with LLM verbalisation."
- Foundation: the user's own **IJCNN 2024 paper (Makarova et al.)** —
  REINFORCE bot-play. 2a = "same framework, modern optimiser + continuous
  action space."

**Opening experiment (already run):** PPO cracks the synthetic indicator
world that REINFORCE cannot — proves adaptivity is learnable and that the
*optimiser* is the boundary. This motivates CASPER-CA before MovieLens.

**Ablations to run:** CASPER-CA vs discrete-PPO (RQ3: does continuous
help?); with vs without answerability/dual beliefs in the actor state;
**contextual-bandit (myopic) version** — because episode return =
final − baseline telescopes, the sequential aspect may matter less than
assumed; "when does lookahead help elicitation?" is a clean sub-question.

**GOTCHAS:**
- `env.py` `entity_emb` (SBERT action geometry) is loaded ONLY for slate1
  (`concept_embeddings_paper_config.npy`). For slate2 it is currently
  `None` — CASPER-CA on slate2 needs SBERT embeddings of the 380 slate2
  entities generated first.
- The continuous trainer has NEVER been run — smoke-test small first.
- Expect it may tie discrete PPO on real slates (both go static). The
  reframe ("when does a continuous semantic action space help?") is the
  fallback and is publishable.

## 9. Paper 2b — full design notes

**Title:** *Verbal Bot-Play: Improving LLM Elicitation Strategies without
Gradient Training.* System name: **CASPER-V**. The most LLM-native and
most clearly-novel of the three; nothing in the verified literature does
this for elicitation strategy.

**Mechanism:** keep the bot-play loop EXACTLY (QBot asks, ABot answers
from profile, reward = recommender BCE-loss reduction — identical to
IJCNN/Paper 1), but replace DDPG/gradient policy learning with
**prompt-space optimisation**: the QBot's *strategy system-prompt* is
iteratively rewritten by an LLM based on which past episodes/trajectories
scored highest (Reflexion / OPRO-style verbal reinforcement). No PyTorch
training at all — "learning" happens in natural-language prompt space.

**Why it's interesting:** (a) zero gradient training, pure LLM; (b) the
learned artefact is a human-readable strategy prompt (interpretable);
(c) directly tests "can an LLM be *taught* to ask strategically" vs the
Paper 1 finding that prompt-only LLMs ask poorly.

**Reuse from existing code:** the LLM cache + validated parser + transcript
saving + throttle in `policies.py` already support it; the testbed reward
and simulator are ready. What's missing: the optimisation meta-loop
(generate strategy → run N episodes → score → LLM critiques & rewrites →
keep best; archive of prompts + best-of selection).

**Decision rule (from Paper 1 leaderboard):** if greedy >> prompt-only LLM
→ 2a is the lead story (learned policy chases the greedy ceiling). If
prompt-only LLM ≈ greedy → 2b is the lead story (cheap optimisation of an
already-strong asker). Current provisional LLM rows are below greedy, but
that's pending the clean LLM re-run.

**Cost:** see §6 (~$200–500 publishable, ~$150 trimmed; needs work API key).

## 10. Cross-cutting ideas worth preserving

- **Answerability beliefs (dual head) may be the strongest cross-paper
  contribution.** "Elicitation instruments must expose P(answerable),
  not just P(liked), for adaptive questioning to be expressible." Spans
  Paper 1 (discovery), 2a (actor state), 2b (could inform prompt).
- **Free-form LLM arm (B1)** = the bridge between Paper 1 and 2a/2b: LLM
  asks free-form, answer SBERT-mapped to nearest slate entity. Also
  defuses the reviewer objection "you tested LLMs only on your menu."
- **Reward telescoping** (return = final − baseline) → try the
  contextual-bandit/myopic version everywhere before assuming the full
  sequential MDP is needed.
- **The static-playlist finding is a feature, not a bug** for the whole
  programme: it explains why prior "learned elicitation" systems were
  effectively playlists nobody audited. Lead with it.
- **Venues:** 2a → RecSys / UMAP / ECIR / CIKM. 2b → same, or a more
  LLM/agent-flavoured venue. Paper 1 = the resource/testbed paper that
  both cite.
- **LLM transcripts** (`llm_transcripts_*.jsonl`) now saved — use for
  qualitative analysis in both Paper 1 (what LLMs ask) and 2b.
- Reference docs with full rationale: `CASPER_Expert_Review_2026-06.md`,
  `LitReview_Verification_and_Paper_Plan_2026-06.md`, and the memory files
  under `~/.claude/projects/C--dev-phd/memory/`.
