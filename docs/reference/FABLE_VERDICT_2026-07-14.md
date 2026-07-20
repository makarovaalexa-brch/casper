# FABLE ADVERSARIAL PASS — the LLM verbalizer + the baseline plan (2026-07-14)

## THE THREE DECISIONS FABLE ASKS FOR
1. **SNAP-K and PROBE-V this week, BEFORE any LLM work.** Both ≤2 days. **Either can kill the centrepiece for
   free.**
2. **CUT gSASRec. PROMOTE E2-HARD-STATIC and Greedy SLIM into the minimum set.** Net effort ~unchanged; risk
   profile enormously better.
3. **LLM verbalizer = NO-GO as the ECIR centrepiece. GO as a post-policy Section 6 and as Paper 2.**

---

# PART 1 — THE LLM VERBALIZER

## 1. Stage 1 is not a fantasy. It is worse: it is ALREADY PUBLISHED, by our nearest rival.
**ELM (Tennenholtz, Kumar, Boutilier, Meshi et al., ICLR 2024, arXiv 2310.04475)** does Stage 1, ON OUR DATASET:
an MLP adapter maps **ML-25M WALS matrix-factorisation embeddings** into PaLM 2-XS, and the model *"does not
receive any information about the movie (including its title), apart from its ... embedding vector."* It
generates fluent, behaviourally-consistent text **including for interpolated points off the item manifold**
(semantic consistency 0.86–0.96 cos; behavioural 0.76–0.96 Spearman).

**RecInterpreter (arXiv 2310.20487, now TOIS)** does the identity half with a **single linear layer**, both
models frozen. Honest numbers: on ML-**100K**, only ~35% of sequences recover >5 items, and **28–49% of
generated item mentions DO NOT EXIST in the dataset.**

**=> "hand it a film's CF vector; does it NAME the film?" is a REPLICATION, not a milestone.** Budget it as a
2-day sanity check, never a paper section.

**=> AND IT IS THE WRONG TEST.** Titles are memorised by any 7B model. A projector needs ~14 bits of routing to
make the LLM say "Blade Runner" — it can pass that test **while transmitting NO taste geometry at all.** The
geometry test is: *given q and two held-out films, which is more +q?*

**⚠ SCOOP RISK IS ACUTE.** ELM + arXiv 2510.12015 (*Asking Clarifying Questions for Preference Elicitation with
LLMs*) + Martin/Boutilier IJCAI-24 are **THE SAME GOOGLE GROUP**. They hold the latent→LLM adapter, the LLM
question generator, and the elicitation formalism, **in three separate papers. Our idea is one merge away, and
they are holding both halves.**

**AND: DO NOT PUT AN LLM ANYWHERE NEAR THE RECOMMENDER.** arXiv 2408.14238 — retrain SASRec/FMLP with
full-softmax cross-entropy and they beat P5, POD, LlamaRec and E4SRec by 10–26% NDCG. **The LLM-recsys wins
were a LOSS-FUNCTION ARTIFACT.** The LLM is a RENDERER. Nothing else.

## 2. ☠ THE KILLER OBJECTION — and the literature answers it AGAINST US
**Nameable directions are a sparse, privileged subset. A dense q is generically a mixture and should be expected
to have NO NAME.**
- **Rohe & Zeng, JRSS-B 2023:** varimax recovers latent factors **only when they are sparse/leptokurtic**. That
  is the formal statement of "there is a privileged nameable basis, and a generic direction is not in it."
- **Cunningham et al., ICLR 2024, §3.2:** sparse dictionary features are far more interpretable than **random
  directions, PCA and ICA**. Random dense directions are the baseline that LOSES.
- The name→direction literature (Bolukbasi; SemAxis ACL'18; POLAR WWW'20; TCAV ICML'18; **Göpfert/Boutilier
  soft-attribute CAVs 2202.02830, in a recommender CF space**) ALL runs **name → axis**. **Nobody runs
  axis → name and validates it.**
- Koren & Bell's own text says MF factors range from "comedy vs drama" to **"completely uninterpretable"**. (The
  famous "serious vs escapist" axes are HYPOTHETICAL ILLUSTRATIONS, not discovered factors. Do not mis-cite.)

**☠☠ AND THE FINDING THAT SHOULD TERRIFY US:**
> **arXiv 2501.17727 (Heap et al.):** *"SAEs trained on RANDOMLY INITIALIZED transformers produce
> auto-interpretability scores ... similar to those from trained models."*
> **Sanity Checks for SAEs:** random-direction baselines **MATCH** trained SAEs on interpretability (0.87 vs
> 0.90), sparse probing (0.69 vs 0.72), causal editing (0.73 vs 0.72).

**TRANSLATION: AN LLM WILL PRODUCE A FLUENT, PLAUSIBLE, CONFIDENT NAME FOR A RANDOM DIRECTION.** So *"the LLM
wrote a beautiful question about q"* is **ZERO EVIDENCE. Any demo we run will look like it works.**

## 3. ⭐ THE ESCAPE, AND IT REFRAMES THE WHOLE DESIGN
> **The deployable unit is NOT a question. It is a (question, READ-BACK) pair.**
> A phrase from the bank comes with its embedding for free — that is why snapping works. **An LLM question has
> NO embedding.** If we cannot map the user's answer back to a direction q̂ in the latent, **we cannot fold the
> answer into the recommender, and the question is worthless however beautiful.**

## 4. PROBE-V — the 1-day, no-GPU, no-projector test that decides everything
For each direction q from the actor's own rollouts (ALL of them; no sampling):
1. Render q **with NO projector**: the top-m positive- and top-m negative-projection item titles along q. (This
   is the autointerp protocol, and **it is the projector's competitor.**)
2. Off-the-shelf LLM, zero-shot: name the contrast axis; write one elicitation question separating the poles.
3. **READ-BACK, FROZEN AND INDEPENDENT:** a model mapping question text → q̂, trained **ONLY on the phrase bank**
   (text → its known embedding), **never shown q or the generator.** Score **fidelity = cos(q̂, q)**.
4. **THE THREE MANDATORY CONTROLS:**
   - **q_random** — same pipeline on a uniform random unit direction. **If fidelity(q) ≈ fidelity(q_random), the
     LLM is naming NOISE and Stage 2 is DEAD.** This single control is the falsifier and it costs nothing.
   - **q_shuffled** — poles swapped. Does the LLM read the SIGN?
   - **q_nearest_phrase** — the OMP/snap incumbent, **fidelity 0.84 at k=3. The LLM MUST BEAT 0.84 or it has no
     reason to exist.**

**PRE-REGISTERED DECISION RULE:** LLM fidelity must exceed BOTH the random control AND the k=3 OMP blend. If it
beats random but not OMP: *directions ARE verbalizable, but a phrase blend already verbalizes them, and there is
no LLM paper here.* **An honest, publishable sentence that saves six weeks.**

**FREE GIFT: even a NEGATIVE PROBE-V is publishable inside the existing story** — *"the best question has no
name — we measured it."* The sharpest possible defence of the continuity claim, for one day of work.

## 5. ⚠ SNAP-K IS A PRECONDITION AND IT CAN DISSOLVE THE PROBLEM
Our own `DESIGN_SHEET_E2_SNAPK.md` says the −0.037/−0.040 snap-loss may be a **k=1 retrieval artifact** — our
snap is `argmax(q·POOL)` with **no critic re-rank**, *the exact configuration Wolpertinger reports as failing.*
**If snap-loss collapses under proper k-NN + re-rank, the LLM verbalizer HAS NO PROBLEM TO SOLVE.** The whole
motivation ("the continuous policy is undeployable") evaporates. **RUN IT FIRST.**

## 6. IF IT SURVIVES — the three-arm test, and the three ways it fools us
| arm | what | status |
|---|---|---|
| **A** unsnapped q | fold q directly | **oracle upper bound, NOT deployable** |
| **B** SNAP-K + critic re-rank | the REPAIRED phrase-bank incumbent (not SNAP-1) | **THE NUMBER TO BEAT** |
| **C** LLM question → answerer → read-back q̂ → fold q̂ | the candidate | |
**C loses to B ⇒ the verbalizer is worse than the phrase bank. Written down BEFORE the run.**

1. **Fluent-but-wrong-axis** → report cos(q̂,q) and corr(answer, u*·q) ALONGSIDE NDCG. NDCG with fidelity≈0 is a
   popularity gain.
2. **Text-prior leakage** → add **C-notoken**: the LLM writes a question with NO q conditioning ("what films do
   you love?"). If C ≈ C-notoken, we have re-invented a generic chatbot.
3. **☠ ANSWERER OOD — the one that will actually bite.** Our answerer was distilled **on the 2,428-phrase bank.**
   Free-form LLM questions are **OUT OF DISTRIBUTION for it**, and arm C's entire result flows through it.
   Noisier on novel questions ⇒ C loses for a reason unrelated to verbalization. Cleaner ⇒ C wins spuriously.
   **THIS IS WHAT THE QUARANTINED 173/300 USERS ARE FOR. Until that validation runs, arm C is
   UNINTERPRETABLE.**

## 7. TRAINING — cycle-consistency is MANDATORY, and its trap is COLLUSION
`q → question → answer → q̂` **IS the deployable system** (no read-back ⇒ no fold). So the reward IS the cycle;
rejection sampling **on the cycle reward** is the cheap version of it.
**☠ THE TRAP: if the generator and the read-back are trained JOINTLY they invent a PRIVATE CODE** — fluent
questions carrying information only the read-back decodes, which a HUMAN would answer completely differently.
The classic emergent-language failure. It produces a beautiful, entirely fake result.
**CLOSURE (non-negotiable): the read-back is FROZEN, trained only on the phrase bank, and NEVER receives
generator gradient.**

**LADDER:** 0. PROBE-V (zero training, kills or greenlights) → 1. **best-of-N + SFT on winners** (reward =
frozen read-back fidelity + realized NDCG) — *do this and stop* → 2. DPO (free: the same N samples give
preference pairs at zero extra sampling cost) → 3. GRPO (only with a spare GPU-week).
**PROJECTOR + LoRA LAST**, and only if PROBE-V shows the top-±items TEXT rendering is insufficient. **Demand the
ablation: soft-token vs 2m item titles in the prompt.** LLaRA's own ablation (embedding-only < hybrid-with-
titles) and SAILRec's attention measurements say **the text rendering probably wins and the projector is an
expensive way to do what a list of titles does for free.** If so: lose the projector novelty, KEEP the working
verbalizer. A fine trade.

## 8. NOVELTY — survives, narrowly, in exactly one phrasing
Everything adjacent is taken: **ELM** = latent→description; **arXiv 2510.12015** = LLM→clarifying question but
conditioned on a **TEXT profile**, evaluated by **BLEU/ROUGE, no recommender metric**; **PEBOL** = question
verbalized from an item's **text description**, 100-item shortlist; **GATE** = questions from a text task
description; **RecInterpreter** = latent→item identity; **EAGLE** = RL-steers LLM *content generation*.
> *Prior work either generates elicitation questions from **text** (GATE; PEBOL; Montazeralghaem et al. 2025)
> or decodes recommender latents into **descriptions** (ELM) or **item identities** (RecInterpreter). We are
> aware of no work that generates a preference-elicitation question conditioned on a **continuous recommender
> latent** — a direction that names no item.*

---

# PART 2 — THE BASELINE PLAN

## ⭐ MISSING #1 — THE HARD STATIC. The most important baseline in the paper, and it was NOT on the list.
C2′ makes the confound load-bearing (*"either the static arm wins, or the adaptive win is not attributable"*).
**If we then beat a WEAK static, we have written the very paper we accuse the field of writing.** And our own
design sheet CONFESSES it: *"STATIC8's static arm could not be optimised (it collapsed; the 'best static' is an
UNTRAINED bank)."*
**BUILD E2-HARD-STATIC the way DAD does: jointly optimise all q questions on the downstream NDCG objective, from
an actor-independent pool, with THE SAME CHECKPOINT-SELECTION BUDGET the actor got.** Multiplicity asymmetry
(best-of-many actor vs one-shot static) **will manufacture a fake adaptive win all by itself.**
**=> PROMOTE TO BASELINE #1. Above popularity. Above Golbandi.**

## ⭐ MISSING #2 — ANSWER-MODEL PARITY. Not in our trap list, and FATAL.
Every baseline faces the SAME answer channel (graded ordinal + refusals + noise) — **and each baseline's
SELECTION CRITERION must be REFIT under that channel.** Golbandi's tree grown on clean ratings vs our policy
trained against a refusing, noisy answerer ⇒ **we win by robustness to a channel they never saw. That is not a
contribution, it is an unfair comparator. WE HAVE MADE THIS EXACT MISTAKE BEFORE** (Squeeze R2: *"P4C's
'adaptivity recovers under retraining' was an UNFAIR-COMPARATOR artifact"*). **Do not make it twice.**

## ⭐ MISSING #3 — RECOMMENDER PARITY, ENFORCED AS A RULE.
Golbandi's win is unattributable precisely because his tree predicted with **node means** while the baseline used
a seed-restricted item-item model. **RULE: every baseline is a SELECTOR ONLY. Its chosen questions are answered
by OUR answerer and folded into OUR recommender.** Run his native node-mean predictor as a separate, clearly
labelled row. **This symmetric attributable test IS the paper C2′ promises.**

## ⭐ MISSING #4 — GREEDY SLIM'S STATIC ORDERING IS NOT OPTIONAL.
It is (a) the ONLY prior ML-25M NDCG@10 elicitation result in existence, and (b) the reason our **cheapest
novelty** exists (its authors tried and FAILED to reproduce Golbandi AND Sepliarskaia, and excluded both). **The
"first head-to-head of tree vs SPQ vs Greedy SLIM on one ruler" contribution REQUIRES ALL THREE ARMS.**

## ✂ CUT — gSASRec / BERT4Rec. And cut the "REQUIRED EXPERIMENT" label in NOVELTY_CLAIMS.
The reasoning was sound GIVEN a SOTA claim — **but we already dropped it** (RecVAE 0.4998 > our 0.4852;
"SOTA-class" survives). Once we concede we are not SOTA, the sequential comparison buys **nothing**: 2–3 days
plus a protocol port we would then have to DEFEND as fair — **a new attack surface, invented voluntarily.**
Nothing in C1–C5 depends on it. **KEEP EASE** (<1 day, closed form; Dacrema means a reviewer WILL ask). Handle
SASRec in one honest limitations sentence. **That is a STRONGER position than a hand-rolled port.**

## ⚠ RE-LABEL — UpsRec is a CEILING, not a baseline.
Keep it (2 days), but **it must NEVER be a row in the baselines table** — a privileged oracle in a baselines
table reads as *"they lost to UpsRec."* Separate **"upper bounds"** block, beside the clairvoyant ceiling.

## PORTING MARTIN ET AL. — as specified we are building a STRAWMAN. The steelman, and its risk.
None of the gap (PMF d=50, 1,000 items, synthetic users, expected utility) is THEIR METHOD'S FAULT. Six ways we
would hand ourselves an unearned win:
1. **Architecture** — DeepSets f_v vs our Set-Transformer. ⇒ **same trunk for both**, or run the 2×2 and report
   the interaction.
2. **☠ TRAINING-DISTRIBUTION — the big one.** Their episodes come from a **uniformly random query policy**. If
   ours trains on probe-derived or on-policy data, **we win on the DATA DISTRIBUTION, not the method.** ⇒ same
   behaviour policy, same episodes, same budget, same answer channel.
3. **Action space** ⇒ same 2,428 actions for both.
4. **Utility mismatch** ⇒ **GIVE THEM NDCG@10 as their realized utility.** A favour to them; required for
   comparability. Say so in the paper.
5. **Response model** ⇒ fit their f_r on **our real graded+refusal answerer.** This gives them a calibrated
   answer distribution **we never build** — a genuine advantage, honestly granted. **That is what makes a win
   mean something.**
6. **Planner** ⇒ depth-0 is their WEAKEST. Running only depth-0 is cherry-picking. Run it, and if compute
   forbids MCTS, SAY SO WITH THE REASON. Do not pretend depth-0 is their method.

**☠ THE RISK THIS EXPOSES.** With the six closures, the comparison reduces to exactly **ENUMERATE vs REGRESS** —
and:
- **The compute argument for amortization is THIN.** Enumerating 4 ordinal levels + refusal = **5 forward passes
  per candidate. That is 5×, not 1000×.** A reviewer will say "5× is free, and f_r gives calibration you don't
  have."
- **Marginalising might simply be BETTER than regressing.** Their EVOI integrates over a CALIBRATED answer
  distribution; we regress a conditional mean and **eat the answer variance. THIS RUN CAN KILL C5's LAST
  SURVIVING DELTA.** Better we find out than a reviewer.
**⇒ FRAME THE DELTA CORRECTLY NOW:** *no response model to misspecify, and one forward pass over the whole
action space at selection time.* **NOT "faster".** And **PRE-REGISTER that enumerate-vs-regress may go against
us** — if it does, C2/C3 still stand and the honest report is a contribution.

## FOUR TRAPS TO ADD (7–10)
7. **ANSWER-MODEL PARITY** — every baseline's criterion REFIT under our answer channel.
8. **RECOMMENDER PARITY** — baselines are SELECTORS; answers fold into OUR recommender.
9. **SELECTION-MULTIPLICITY PARITY** — the actor is a best-of-many checkpoint; a one-shot static loses to
   selection noise ALONE. **Same val-selection budget for both arms, or no comparison.**
10. **BUDGET PARITY** — same q for every arm; report the FULL CURVE, not one q.

---

# ORDER OF WORK — 11 weeks to 2 Oct
**TRACK 2 STARTS NOW, IN PARALLEL** (needs only the frozen recommender + answerer; CPU/closed-form; contends for
nothing): EASE (<1d) · static seed battery incl. Entropy0/HELF (1–2d) · Greedy SLIM static ordering (2d) ·
Sepliarskaia SPQ (3–5d, code live).
**TRACK 1 (critical path):** Phase B concepts → the value head → **BEAT E2-HARD-STATIC**.
**TRACK 3 (only after the recommender is FROZEN — they must fold into the FINAL encoder):** Golbandi tree
(4–7d) · Martin port STEELMANNED (5–7d, not 3–5) · UpsRec ceiling (2d).
**Slot SNAP-K and PROBE-V into Track 2's slack.**

## ☠ THE HARD DATE — AND THE FALLBACK, WRITTEN DOWN NOW SO THE DEADLINE CANNOT FORCE AN OVERCLAIM
**If the policy is not beating E2-HARD-STATIC by MID-AUGUST: C5 dies, C2 dies with it, the LLM centrepiece dies,
and the paper becomes:**
> *the first head-to-head of Golbandi vs SPQ vs Greedy SLIM on one ruler, plus the entropy-vs-value boundary
> (C3), plus the honest negative.*
**THAT IS A REAL ECIR PAPER.**
