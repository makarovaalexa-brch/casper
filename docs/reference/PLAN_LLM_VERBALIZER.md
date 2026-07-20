# PLAN — THE LLM VERBALIZER: from a continuous query to a question a human can answer
2026-07-14. Deep lit sweep (5 agents) + the author's brief. **DESIGN ONLY — nothing runs without a design sheet.**

---

## 0. THE PROBLEM IT SOLVES (not decoration — the missing half of Paper C)
Our policy emits a continuous direction `q` in the recommender's 512-d latent — the taste axis it most wants to
probe. **You cannot ask a human a vector.** Today we SNAP `q` to the nearest of 2,428 pre-written phrases and
LOSE **-0.037 full / -0.040 tail NDCG**. So the continuous policy is **UN-DEPLOYABLE**, and the snap-loss is a
confession in the limitations section.
**If an LLM can verbalize `q` directly and close that gap, the continuous line becomes real and the limitation
becomes the result.**

---

## 1. ☠ THE PAPER THAT NEARLY KILLS IT — CITE IT IN THE FIRST PARAGRAPH
> **ELM — Tennenholtz, Chow, Hsu, Jeong, Shani, Tulepbergenov, Ramachandran, Mladenov, Boutilier.
> "Demystifying Embedding Spaces using Large Language Models." ICLR 2024. arXiv:2310.04475.**

**ELM ALREADY DOES:** MovieLens **CF latent -> a 2-layer MLP adapter -> LLM token space -> fluent, faithful
natural language**, **including for INTERPOLATED / HYPOTHETICAL points that correspond to NO REAL ITEM**, with
a **semantic-consistency metric of 0.87-0.96** proving the mapping is not superficial.
**EVERYTHING IN OUR PROPOSAL EXCEPT THE SPEECH ACT AND ITS OBJECTIVE IS IN ELM.**

### ⭐ OUR SURVIVING DELTA — SAY EXACTLY THIS, AND ONLY THIS
> **ELM decodes a latent to EXPLAIN it. We decode a latent to ASK A QUESTION whose answer is designed to MOVE
> THE RECOMMENDER'S POSTERIOR.**

### CLAIMS WE MUST **NOT** MAKE (all occupied)
- first to inject a CF latent into an LLM — **CoLLM / LLaRA / E4SRec (2023)**
- first to inject a USER latent as a soft prompt — **User-LLM, A-LLMRec**
- first to render an OFF-CATALOG latent point as language — **ELM, EAGLE**
- first to show a CF latent is verbalizable — **ELM, FACE**
- first to train an LLM to ask sequential elicitation questions — **Montazeralghaem (2510.12015), GATE, PEBOL**

### ☠☠ SCOOP RISK IS CONCRETE AND URGENT
**Tennenholtz and Boutilier are co-authors on ELM (vector->language), on EAGLE (NeurIPS 2024 — an RL policy
optimising IN A RECOMMENDER LATENT, realised as LLM text), AND on the Oct-2025 clarifying-question paper
(arXiv 2510.12015).** **ONE GOOGLE TEAM HOLDS BOTH HALVES OF OUR CONTRIBUTION AND HAS NOT YET JOINED THEM.**
(Same Boutilier as the IJCAI-24 "Model-Free Preference Elicitation" paper — he is everywhere in this space.)
**Also live: LatentCRS (KDD 2026, arXiv 2503.10703)** — a strong team looked at exactly this design space and
**CHOSE TO DISCRETIZE** (categorical intent prototypes). **Our paper must say why they were wrong.**

---

## 2. ☠ THE CONTROL THAT DECIDES WHETHER WE HAVE A PAPER AT ALL — RUN IT FIRST
> **"Are EEG-to-Text Models Working?" (arXiv 2405.06459): *"Model performance on NOISE data can be comparable
> to that on EEG data."***

An entire subfield published fluent, plausible decodings from a **non-linguistic latent -> projector ->
pretrained LM decoder** — **OUR EXACT ARCHITECTURE** — and the fluency turned out to be **the LANGUAGE MODEL'S
PRIOR, not the latent.** Reinforced by **Tan et al., NeurIPS 2024** (removing the LLM does NOT degrade
time-series forecasting).
=> **"Non-text modality + projector + LLM" papers are PRESUMED GUILTY until an ablation proves the modality is
load-bearing.**

### **E0-SHUFFLE (EXISTENTIAL, ~1 day, run BEFORE anything else)**
Generate the question from **ANOTHER USER'S `q`** (and from a random unit vector), everything else fixed.
**If shuffled-q performs like real-q on the downstream ladder, WE ARE THE EEG PAPER and the idea is DEAD.**

---

## 3. ⭐ THE SURVIVAL ROUTE: THE ITEM BRIDGE (and it is the ONLY one)
**We are NOT doing embedding inversion.** Inversion recovers a SPECIFIC SOURCE SENTENCE and needs (i)
per-encoder training, (ii) encoder query access, (iii) **GROUND-TRUTH TEXT — which we do not have at all.**
**BUT: items have TITLES, GENRES and TAGS.** So `item-embedding <-> item-text` gives us **exactly the paired
corpus that vec2text needs and EEG never had**: **18,430 items + 1,628 named concepts.**
**Our existing OMP k=3 fidelity of 0.84 IS THAT BRIDGE ALREADY WORKING.**
=> **Frame the method as "SUPERVISED VIA THE ITEM BRIDGE", NEVER as "inverting a non-text embedding".**
⚠ **Replicate ELM's construction exactly**: build targets from the ground-truth textual handle, then WITHHOLD
the handle and hand the model only `q`. Otherwise the synthetic corpus is CIRCULAR and the result unfalsifiable.

---

## 4. ARCHITECTURE (ranked; single workstation, 7-8B, LoRA)
1. ⭐ **MLP projector -> k=4 soft tokens + LoRA on the LLM** (the LLaVA/ELM pattern). ~35M trainable params,
   hours not days on 24GB. **TWO-STAGE TRAINING IS NON-NEGOTIABLE** — ELM reports single-stage *"failed
   completely"* while two-stage converged in <1000 iterations. Add **STRUCTURE's (arXiv 2506.16895)
   neighborhood-preserving regulariser** — the only paper aligning modalities in our data regime.
2. **Discrete quantisation / semantic-ID tokens** (TIGER, LC-Rec, FACE) — cheapest and most robust, **but it
   RE-INTRODUCES EXACTLY THE SNAP-LOSS OUR THESIS IS ABOUT.** The honest fallback, not the thesis.
3. **Q-Former** — **do NOT start here.** It compresses MANY tokens; we have ONE vector. Converges slower for
   want of data (EMNLP 2024 connector study).
4. **Pure soft-prompt / prefix-tuning, no LoRA** — Lester (EMNLP 2021): prompt tuning only closes the gap at
   **>=10B**; we are on the wrong side. *Universality and Limitations of Prompt Tuning* (NeurIPS 2023) proves
   prompts of ANY length cannot express some functions a low-rank update can. **LoRA IS REQUIRED.**
**Cheap go/no-go (an afternoon): compute CKA between our q-space and the LLM's input-embedding space.** High
CKA predicts low alignment loss (arXiv 2409.19425).

---

## 5. TRAINING (cheapest defensible first)
**Rung 0 — teacher-distilled LoRA SFT** (~2 GPU-h). Mandatory: it is the reference policy every later rung
anchors to (KL).
**Rung 1 — REJECTION-SAMPLING FT ON THE GEOMETRIC REWARD. ⭐ RECOMMENDED.** Sample N questions per `q`, run
each through OUR ANSWERER, score by recovery of `u* . q`, SFT on the winners, repeat x2. ~20-60 GPU-h, one
card, **cannot diverge**.
> **NEAR-LINE-FOR-LINE PRECEDENT: STaR-GATE (Andukuri, Fränken, Gerstenberg, Goodman, arXiv 2403.19154)** —
> Mistral-7B questioner, N=10 samples, K=3 turns, **scalar oracle reward, 2 iterations, 72% win rate.** Our
> loop with `u*.q` swapped for their log-prob oracle. **A reviewer cannot call this exotic.**
**Rung 2 — DPO on reward-ranked pairs** (~5 GPU-h, reuses Rung-1 rollouts). *"How do you DPO from a SCALAR
reward?"* has a named answer: **RS-DPO (Khaki et al., Findings of NAACL 2024)** — sample n, score, build
synthetic pairs, threshold on the reward GAP. Add Iterative RPO's NLL-on-winner term (NeurIPS 2024).
**Rung 3 — GRPO** — ablation only. **TRAP: TRL's GRPO `beta=0.0` default DISABLES the KL anchor** — the exact
opposite of what the drift literature demands.
**Rung 4 — Gumbel-softmax: DO NOT.** Structural: relaxation removes the non-differentiability of SAMPLING, not
of our REWARD. We would still have to backprop through the answerer.

### ☠ THE TWO NAMED FAILURE MODES (pre-register against both)
1. **LANGUAGE DRIFT.** **Lee, Cho & Kiela (EMNLP 2019)** [NOTE: **Kiela**, not Weston]: agents finetuned on a
   NON-LINGUISTIC reward **drift away from natural language — task score goes UP while language quality goes
   DOWN**, and an **LM-likelihood constraint ALONE was INSUFFICIENT** (grounding was also needed).
   **MITIGATION: Rung 1 gives us the fix FOR FREE** — re-distilling each iteration IS seeded iterated learning
   (Lu et al., ICML 2020). Plus KL-to-reference, and monitor true-reward-vs-sqrt(KL) (Gao et al., ICML 2023) —
   **stop at the peak.**
2. **THE STEGANOGRAPHIC QUESTION.** Our reward passes through a **LEARNED** answerer, so it is **HACKABLE**:
   the degenerate optimum is a prompt-shaped string that steers the answerer into a high-projection answer and
   **reads as gibberish**. Detect with the human study (§6.3) and a fluency/perplexity gate.

---

## 6. EVALUATION (and what falsifies the whole thing)
**6.1 E0-SHUFFLE (existential, §2).** Shuffled-q / random-q vs real-q. Flat => we are the EEG paper.
**6.2 OFF-MANIFOLD FIDELITY CURVE.** Round-trip cosine (`q -> question -> answer -> re-embed -> q_hat`) **as a
function of distance from `q` to the phrase-bank manifold**, for the q's our policy ACTUALLY emits.
⚠ **This decides whether LaRL (NAACL 2019) kills us** — LaRL already ran continuous-vs-discrete latent actions
into a language decoder and found **CONTINUOUS LOSES**, because *"RL exploration in the continuous space may
lead to areas in the manifold that are not covered in supervised training, which causes undefined decoder
behavior."*
**6.3 ⭐ HUMAN ANSWERABILITY STUDY (N~40-60), STRATIFIED BY OFF-MANIFOLD DISTANCE.** Both the contribution AND
the only break in the circularity (our answerer is a model; a human is not).
> **THE KILLER RESULT, IF IT EXISTS: ANSWERABILITY DOES NOT DEGRADE AS `q` MOVES OFF THE PHRASE MANIFOLD —
> "the best question has no name, AND HUMANS CAN STILL ANSWER IT."**
**6.4 CROSS-FAMILY ANSWERER MATRIX + PROFILE-LEAKAGE ABLATION.** Defuses **Zhu et al. (WWW 2024 Companion)**
(LLM user simulators LEAK and inflate CRS results — demonstrated ON iEvaLM) and **Panickssery et al. (NeurIPS
2024)** (LLM judges recognise and FAVOUR their own generations). **If the ladder only holds for ONE answerer
family, we learned a code, not a question.**
**DO NOT USE BLEU/ROUGE.** Nema & Khapra (EMNLP 2018): n-gram metrics *"do not always correlate well with human
judgments about ANSWERABILITY"* — **which is a gift: answerability is an established, publishable evaluation
axis, and it is our thesis.** (Montazeralghaem reports only BLEU/ROUGE against a ground-truth profile — a soft
target we outrank by reporting downstream NDCG per turn.)

### WHAT FALSIFIES IT
If **6.1** comes back flat, or **6.2** shows fidelity collapsing off-manifold, the honest paper becomes
**"SNAP — and here is the bank, and here is why the bank is ENOUGH."** Given our OMP k=3 fidelity of 0.84 and
the field's revealed preference for discretisation (LatentCRS, FACE, TIGER, LC-Rec **all quantise**), that is a
perfectly publishable ECIR paper. **Do NOT sell it as a negative result — sell it as the BOUNDARY MAP**, which
is exactly the two-regime move Paper C already makes.

---

## 7. CITATION CORRECTIONS (load-bearing; some are in our own memory)
- **"Countering Language Drift via Visual Grounding" is Lee, Cho & KIELA** (EMNLP-IJCNLP 2019) — **not Weston**.
- **"Multi-Agent Cooperation and the Emergence of (Natural) Language" is Lazaridou, Peysakhovich, Baroni
  (ICLR 2017)** — not Havrylov & Titov (whose paper is "Emergence of Language with Multi-Agent Games",
  NeurIPS 2017).
- **GATE is ICLR 2025**, not 2024. **PPO has no peer-reviewed venue** — cite as arXiv. **"Item2Text" does not
  exist.** **Alpaca is not a paper** — cite Self-Instruct (ACL 2023).
- ⚠ **GEIA's "53-63% token-F1" (IN OUR MEMORY) IS NOT IN GEIA'S ABSTRACT.** It is corroborated by an
  independent **SIGIR 2025 reproduction (arXiv 2504.16609)**: SimCSE-BERT F1 **63.22**, Sentence-T5 63.04,
  MPNet 57.07, Sentence-RoBERTa 52.78 **on PersonaChat** — but it **COLLAPSES to F1 32-36 on QNLI**. **So the
  number is right FOR CHAT-LIKE TEXT ONLY. Cite the reproduction, not GEIA, and state the dataset.**
- ⚠ **ZSInvert (arXiv 2504.00147): agents DISAGREE on whether it exists. FETCH IT YOURSELF before citing.**

## 8. STAGING (nothing runs without a signed design sheet)
0. **CKA probe** (an afternoon) — is our q-space alignable with the LLM's token space at all?
1. **E0-SHUFFLE** (~1 day) — **EXISTENTIAL. If flat, STOP.**
2. Rung 0 (teacher SFT) + the **item bridge** grounding stage. **Milestone: hand it a film's CF vector — DOES
   IT NAME THE FILM?** (ELM's construction; withhold the handle.)
3. Rung 1 (rejection sampling on `u*.q` recovery) — STaR-GATE's loop.
4. Off-manifold fidelity curve (**the LaRL test**).
5. Human answerability study (**the killer result, and the only break in the circularity**).
