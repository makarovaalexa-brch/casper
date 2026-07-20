# VISION — end-state of the project

> The north star: what the finished system **is**. Changes rarely.
> No status (→ `STATE.md`), no project rules (→ `CLAUDE.md`), no conference/paper plans (→ `PUBLICATION_PLAN.md`).
> Where a requirement is already established by evidence, a `→ file` pointer is given.

## Thesis in one line
A **cold-start conversational recommender** that runs a short **adaptive interview** over **any channel**
and, from a handful of answers, ranks a large catalog nearly as well as if it knew the user's full
history — with the questions eventually **rendered in natural language by an LLM**.

Priority tags: **[MUST]** ship-critical · **[WANT]** strong goal · **[EXP]** experimental / stretch.

---

## 1. The recommender — a channel-agnostic belief machine · **[MUST]**
The load-bearing foundation. One encoder that is:
- **SOTA on full profiles** — ties the strong CF baselines; never traded away.
- **Channel-agnostic** — folds items, concepts, entities, and open text through the *same* interface.
- **Arbitrary-embedding** — accepts any query/answer embedding, not a fixed vocabulary.
- **Any-length & cold-start** — variable-length evidence sets, from 0 (fully cold) upward.
- **Uncertainty-native** — carries a confidence / covariance over the belief, not just a point.
- **Monotone** — a truthful answer never *lowers* ranking quality.

Acceptance gates every build must pass:
- **G0** — item strength preserved (eval on the belief mean = the base recommender).
- **G1** — usable posterior (uncertainty strictly shrinks per question, and directionally).
- **G2** — cold-start per-question curves rise, on full **and** tail, vs random.

## 2. Discrete elicitation policy — adaptive & coarse-to-fine · **[MUST]**
A policy over closed probes (items / concepts / entities) that:
- **beats the non-adaptive baselines** (static questionnaire, popularity, greedy info-gain);
- is **adaptive** — the next question depends on the answers so far;
- shows **coarse-to-fine** — a broad axis first, then a niche discriminating probe inside it.

*→ adaptivity + coarse-to-fine established: `experiments/ADAPTIVE_PROBE_RESULT.md`.*

## 3. Continuous elicitation + LLM verbalisation · **[WANT]**
The policy emits a **continuous** query in embedding space (not a pick from a fixed bank); an LLM
**verbalises** it into a natural question — inside the *same* belief loop, not a separate system.

## 4. Open questions — woven into the interview · **[MUST]**
Free-form questions and answers ("tell me a film you love") as a **first-class part of the same
interview**, interleaved with closed probes — a volunteered film folds as an item token; open text is
encoded and folded like any other channel. **Woven in, not a bolt-on.**

## 5. Test UI · **[WANT]**
An interface that drives a **real interview end-to-end** — render question → take answer → update belief
→ re-rank — for demos and the human validation study. *→ `simulator_ui/`.*

## 6. LLM interpretation / simulator · **[EXP]**
Highest-risk / lowest-commitment: (a) an LLM that **interprets** free answers into belief updates; (b) an
LLM **user-simulator** for training/eval. A stretch, never a dependency.

---

## Constraints on the end-state
- **One unified loop** — every channel folds through the same belief; no per-channel bolt-ons.
- **The LLM renders / interprets; it never scores or ranks.**
- **Validated by ≥1 human round-trip study**, not simulator-only.

## Near-term sequencing
Closed probes **first** (pillars 1–2) → continuous (3) → weave open questions (4); UI (5) alongside for
validation; LLM interpretation (6) last, and only if it clears an existential control.
