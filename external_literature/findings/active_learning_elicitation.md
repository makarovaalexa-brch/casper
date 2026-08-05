# Active Learning / Bayesian Experimental Design / Information-Theoretic Acquisition — and the answerability question

*Deep research 2026-08-05, Paper B baseline battery. Family: AL / BED / info-theoretic acquisition applied to
preference elicitation and recommendation. Verification: **[V]** primary text read, **[A]** abstract/landing page,
**[S]** search summary / third-party.*

**Read WITH, not instead of:**
- `adaptive_elicitation_baselines.md` — the recsys-side baseline battery (DRE, PERE, NICF, ICF bandit, Golbandi).
  That file already settles which *elicitation* methods to run; this one covers the *acquisition-theory* family
  behind them and does not repeat its verdicts.
- `rank_targeted_acquisition.md` — already settles goal-oriented / transductive design (A-GOODE, Yu-Bi-Tresp TED,
  Knowledge Gradient, Chajewska–Koller–Parr, Viappiani–Boutilier, adaptive-submodularity theory). **Not re-derived
  here.** Anything about *what to target* lives there; this file is about *how the acquisition prices the answer*.
- `answerability_in_elicitation.md` — the recsys-side answerability audit (Rashid, Golbandi, Sun WSDM'13, Elahi).
  This file adds the **non-recsys** answerability prior art, which is stronger and more formal than the recsys side.

---

## 0. THE HEADLINE — the answerability-aware acquisition question, answered

**YES. It is formalized, it has been formalized at least five independent times, and one of them is a granted
patent that writes our exact formula. We must cite, not christen.** The roll call, strongest first:

### 0.1 The literal formula, in a granted patent **[V]**
**US 11,775,493 B2, "Information retrieval system"** — Microsoft Technology Licensing; inventors **Matteo Venanzi,
Katja Hofmann, Filip Radlinski, Brian Brost**; priority **2016-12-06**, granted **2023-10-03**. Verbatim:

> "Expected reward s_i = IG_i × A_i" — "the expected reward of question S being answered by user i is equal to the
> information gain for user i times an estimate A of the probability that the question S can be answered by user i."

and the estimator:

> "The estimate A of the probability that the question S can be answered by user i is given by the number of times
> user i has answered the question divided by the total number of times the question has been presented."

It even ships the motivating example we would have written ourselves: a question about whether an item was created
at an odd or even second has **high information gain and near-zero answerability**. Note **Radlinski is already in
our INDEX** (`radlinski2019coached`, CCPE-M) — a reviewer from that community will know this work. Treating
`EIG × P(answer)` as our invention is a live desk-reject risk.

### 0.2 The *principled* version — you do not need a heuristic weight **[V]**
**Bıyık, Palan, Landolfi, Losey & Sadigh, "Asking Easy Questions: A User-Friendly Approach to Active Reward
Learning", CoRL 2019** (arXiv:1910.04365; code https://github.com/Stanford-ILIAD/easy-active-learning).
Abstract verbatim: *"Existing approaches attempt to choose questions where the robot is most uncertain about the
human's response; however, they do not consider how easy it will be for the human to answer! In this paper we
explore an information gain formulation for optimally selecting questions that naturally account for the human's
ability to answer."* Body verbatim: *"just because a question is optimal does not mean that the human can correctly
answer it!"*

Mechanism (this is the load-bearing part):

```
Q* = argmax I(ω ; q | Q)  =  H(q | Q)  −  E_ω [ H(q | ω, Q) ]
```

The **first** term is the system's uncertainty about the answer; the **second** is the *human's own answer entropy
given their true preference*. Maximising the difference automatically **penalises questions the human answers
noisily or cannot answer** — the answerability discount falls out of computing the mutual information with an
honest response model, rather than being bolted on as a multiplicative weight. They additionally add an explicit
third response — **"About Equal" (Υ)** — governed by a minimum-perceivable-difference parameter δ ≥ 0, with
`P(q=Υ|Q,ω) = (exp(2δ)−1)·P(q=ξ₁|Q,ω)·P(q=ξ₂|Q,ω)` (their Eq. 8). Domains: linear dynamical system, Driver,
Tosser, Fetch. Baselines: volume removal, random. Metric: alignment to ground-truth reward. User studies n=15 and
n=12, including Likert *easiness* ratings.

**This is the single most important citation in this whole file**, for two reasons: (a) it is the exact sentence
we would otherwise write as our contribution; (b) it tells us *why our EDDI arm is starving* — see §0.6.

### 0.3 The active-learning-theory version, 2024 **[V-abstract + S-body]**
**Gao, Saar-Tsechansky & Yin, "SEL-BALD: Deep Bayesian Active Learning with Selective Labels", NeurIPS 2024**
(Best-Paper Runner-Up WITS 2024; code **https://github.com/ruijiang81/SEL-BALD**). Defines the
**Active Learning with Instance Rejection (ALIR)** problem: the human annotator *may decline to label* the instance
the model selected ("regulation constraint, domain knowledge, or algorithmic aversion"). The acquisition is
**BALD weighted by the modelled probability the human actually labels the instance**, i.e.
`a(x) = P(label obtained | x) · BALD(x)` **[S — body formula not read from primary text; verification debt]**, and
crucially the paper also argues you must **acquire information about the human-discretion model itself**, not just
about the task model — i.e. explore on the answerability model. Variants cover known vs learned rejection
probabilities.

**This is our problem, stated in AL's own vocabulary, two years ago.** The mapping is exact:
their "human declines to label" = our "user has not seen the film". Our budget-burn framing ("unanswerable
questions BURN the budget") is their sample-inefficiency motivation verbatim.

### 0.4 Inside recsys — already on record, plus one more **[V]/[S]**
Already settled in `answerability_in_elicitation.md`, restated here so this file is self-contained:
- **Rashid et al., IUI 2002 — Pop×Ent**: *"popularity is the probability that this user will rate this item"*, with
  an explicit Bayes framing. This is `EIG × P(answer)` in 2002, 14 years before the patent. **[V, prior finding]**
- **Rashid et al., SIGKDD Expl. 2008 — Entropy0 / HELF**: folds unanswerability *into* the information measure
  (missing treated as a rating category); HELF = harmonic mean of entropy and log rating frequency. **[V, prior]**
- **Elahi, Ricci & Rubens, ACM TIST 2014 — "Binary Prediction"**: MF on the binarised observation mask = a *learned*
  P(user can rate item). Used as a ranking heuristic, never as a calibrated term in an objective. **[V, prior]**
- **PERE, Nguyen et al., UAI 2024**: information-gain item selection over a frozen embedding space, **weighted by an
  estimated "experience probability"**. **[S — the weighting is reported in `adaptive_elicitation_baselines.md`
  §1.2 from the arXiv HTML; the exact formula is NOT verified. Standing debt.]**
- **Golbandi et al., WSDM 2011**: `w_U = 0.02` vs `w_L = 5` — a *price* on an unanswered question. **[V, prior]**

### 0.5 What is therefore DEAD, and what survives
**DEAD — never write these:**
- "No acquisition function accounts for whether the user can answer." False five times over.
- "We introduce answerability-aware acquisition / askability-weighted information gain." Pre-empted by the patent
  (literal formula), by Rashid 2002 (same formula, recsys), by Bıyık 2019 (the principled version), by SEL-BALD
  (the AL-theoretic version), and arguably by PERE 2024 (the recsys version, same year we would publish).
- Coining a new word for it. The field's words are **rateable / acquisition probability / success ratio**
  (Elahi et al.), **selective labels / instance rejection** (SEL-BALD), **easy questions** (Bıyık),
  **answerability** (the patent — so even *our* word is taken).

**SURVIVES — the narrow, defensible cells (in decreasing confidence):**
1. **Answerability as a reason to switch the QUESTION CHANNEL, not just to re-rank within one.** Every method above
   prices answerability *inside a fixed question space* — items (Rashid, PERE, Elahi), trajectory pairs (Bıyık),
   instances (SEL-BALD), search facets (the patent). None selects across *heterogeneous channels* whose
   answerability regimes differ structurally (our 23–27% items vs 69–75% concepts). An acquisition function that
   is commensurable across channels *because* both terms are modelled per channel is unoccupied.
2. **"Cannot answer because I have never encountered this entity" as a distinct outcome from "no preference" /
   "indifferent" / "decline".** Bıyık's Υ is perceptual indistinguishability; SEL-BALD's rejection is annotator
   discretion; the patent's A is a raw frequency; Montazeralghaem 2025's "I don't know" is profile-absence.
   Entity-unfamiliarity is a different latent and nobody separates it.
3. **The C-decomposition** (already claimed in `answerability_in_elicitation.md`) — hold informativeness fixed,
   vary answerability, attribute the gain. Still empty, and this pass did not find a challenger.
4. **Held-out ranking metric.** Nothing above is evaluated on post-interview full/tail NDCG@10 on an 18k catalogue.
   Weak on its own; useful as the contract statement.

### 0.6 ⚠ CORRECTION TO OUR OWN FRAMING — the EDDI 3.84/16 number is a LIKELIHOOD bug, not an indictment of EIG
The brief states our EDDI implementation collects **3.84 answers per 16 questions vs 5.52 for a static order**, and
frames this as "answerability-blind criteria waste budget". **That framing is half wrong and a reviewer will say so.**

EDDI's acquisition is `argmax_i I(x_i ; x_φ | x_O)` under the partial VAE — a *correct* expected-information-gain
criterion. It is answerability-blind **only because our observation model is**: in binary implicit data,
"has not seen" and "dislikes" are both the value 0 (this is exactly the DRE indictment already on record in
`answerability_in_elicitation.md`). With a two-outcome likelihood the criterion *cannot represent* the event
"no answer arrives", so it cannot price it. Add a third outcome — `p(a ∈ {unanswered, like, dislike} | z)` — and the
mutual information **prices answerability automatically**, because an `unanswered` outcome carries ~zero information
about z and drags the expectation down. That is precisely Bıyık's `H(q|Q) − E_ω H(q|ω,Q)` decomposition and
precisely Golbandi's ternary branch.

**Consequences (binding):**
- Do **not** write "information-theoretic acquisition is answerability-blind". Write "*a two-outcome response model
  makes any information criterion answerability-blind; the fix is the likelihood, not the criterion.*"
- Before EDDI appears in any Act-1 table, run it **twice**: as-implemented (2-outcome) and with the 3-outcome
  response model. If the 3-outcome version recovers the answer rate, that is a *better* paper result than the
  current one — it is a mechanism, not a gripe. Cost ≈ 4–8 agent-hours, `src/baselines/eddi_pvae.py` already exists.
- HARD-RULE 6 applies: the current 3.84/16 is a **weak-method** result, not evidence about the idea.

---

## 1. The EDDI lineage (partial-VAE / active feature acquisition)

### 1.1 EDDI — Ma, Tschiatschek, Palla, Hernández-Lobato, Nowozin, Zhang, ICML 2019 **[V-prior/A]**
`ma2019eddi`, PMLR 97:4234–4243. Already indexed and **already implemented** (`src/baselines/eddi_pvae.py`).
- **Mechanism.** *Partial VAE*: a permutation-invariant set encoder over an arbitrary observed subset x_O gives
  q(z|x_O); acquisition maximises **expected information gain on a set of TARGET variables x_φ**, i.e.
  `R(i) = E_{x_i~p(x_i|x_O)} KL[ p(x_φ | x_i, x_O) ‖ p(x_φ | x_O) ]`, estimated by the "chain-rule" trick
  (target-oriented, not parameter-oriented).
- **Why it matters for us beyond being a baseline:** it is the *only* method in this family whose acquisition is
  already **goal-oriented** (targets x_φ, not the latent) — that is a free half of the `rank_targeted_acquisition.md`
  story. Our x_φ should be the held-out target items, not z.
- **Verdict: MUST-INCLUDE (already paid for), but re-run with the 3-outcome likelihood — see §0.6.** Effort: 4–8 h.

### 1.2 Icebreaker — Gong, Tschiatschek, Nowozin, Turner, Hernández-Lobato, Zhang, **NeurIPS 2019** **[A]**
arXiv:1908.04537; code **https://github.com/microsoft/Icebreaker**. NB: **2019, not 2020** — cite correctly.
- **Mechanism.** The *ice-start* problem: no training data at all, and acquiring each **element** (a single
  user×item cell) costs. Full-Bayesian **Deep Latent Gaussian Model (BELGAM)** with amortised inference +
  SG-MCMC over the *weights*, so the acquisition can use **epistemic uncertainty in the model**, not just in z.
  Two acquisition objectives: one for training-set construction (element-wise), one EDDI-style at test time.
- **Fit to us.** Its headline experiment family is **MovieLens-style matrix element acquisition**, which is our
  contract's shape. But it targets *model training economy* (which cells to buy to train the recommender), not
  cold-user interview quality — a different question from Act 1. It is however the correct citation for
  "uncertainty over the model, not just the latent" if we are ever asked why our tower is frozen.
- **Verdict: NICE (cite; run only if a reviewer demands a partial-VAE variant beyond EDDI).** Effort: M, 12–20 h
  (SG-MCMC over weights is the expensive part). Contract mismatch: element-acquisition, not per-user interview.

### 1.3 Active Feature Acquisition (AFA) more broadly **[S]**
Saar-Tsechansky & Provost lineage; Melville et al.; recent AFABench (arXiv:2508.14734) and
Learning-to-Measure (arXiv:2510.12624). The AFA framing is *ours exactly* (which feature to buy under budget), and
**Saar-Tsechansky is also a SEL-BALD author** — the "acquisition may fail" idea entered AL through this group.
**Verdict: CITE the framing (one sentence, AFA = our problem class); do not run.**

---

## 2. BALD and the mutual-information family

### 2.1 BALD — Houlsby, Huszár, Ghahramani, Lengyel, 2011 **[S]**
arXiv:1112.5745, "Bayesian Active Learning for Classification and Preference Learning".
- **Mechanism.** `BALD(x) = H[y|x,D] − E_{θ~p(θ|D)} H[y|x,θ]`. Picks points where the *predictive* entropy is high
  but each individual model is *confident* — i.e. where models disagree. Note the structural identity with Bıyık's
  criterion in §0.2: **BALD's second term is exactly "how noisily will the respondent answer"**.
- **It is already a preference-learning paper**, not just a classification one: §"preference learning" recasts
  pairwise preference as binary classification over a difference kernel in a GP, and applies BALD to pick pairs.
  So the "BALD for preference elicitation" cell is occupied at the source. **[S — get the primary text.]**
- **Fit to us.** Directly runnable over our tower posterior (`belief_mf.py` already has a Gaussian over u): with a
  probit/logistic answer model, BALD over candidate question directions is ~40 lines and is the *named*
  information-theoretic arm an AL-literate reviewer looks for after EDDI.
- **Verdict: MUST-INCLUDE.** Effort: **S, 4–6 agent-hours.** Same posterior, same answers, same metric.

### 2.2 BatchBALD (Kirsch, van Amersfoort, Gal, NeurIPS 2019) **[S]**
Joint mutual information over a *batch* of acquisitions, fixing BALD's redundancy when the top-k scorers are
near-duplicates. **Directly relevant**: our budget-k interview asked as a single screen is a batch acquisition, and
BALD-top-k will pick k near-identical popular films. Also the theoretical justification for Sun et al. WSDM'13's
multi-question screens (`answerability_in_elicitation.md` §3), which currently has none.
**Verdict: NICE-HIGH — cheap (it is a submodular greedy over the same scores), and it is the principled answer to
"why does your static order beat your greedy one".** Effort: **S–M, 6–10 h.**

### 2.3 SEL-BALD (NeurIPS 2024) — see §0.3
**Verdict: MUST-INCLUDE as the answerability arm.** We already have the structural answerability rule
(rated ∨ pop>τ; ≥k rated members) that can serve as the discretion model, so we need no new LLM or model.
Effort: **M, 6–10 h** on top of the BALD arm. This is the single highest-leverage new baseline in this file:
it converts our answerability finding from a *complaint about the literature* into a *named method we ran*.

### 2.4 Prediction-oriented / decision-oriented BAL **[S]**
Bickford Smith et al., "Prediction-Oriented Bayesian Active Learning" (EPIG, AISTATS 2023, arXiv:2304.08151);
Huang/Bickford Smith/Rainforth "Loss-Driven BAL" (arXiv:2604.11995, already in `rank_targeted_acquisition.md`).
EPIG = expected predictive information gain, targets predictions on a *test distribution* rather than parameters —
the BALD-family cousin of transductive design. **Verdict: CITE (it is the same move as EDDI's target variables);
do not run separately.**

---

## 3. Amortized / sequential Bayesian experimental design (Foster–Rainforth line)

### 3.1 DAD — Foster, Ivanova, Malik, Rainforth, ICML 2021 **[V-abstract/S]**
arXiv:2103.02438, PMLR 139. Trains a **design policy network** π(history) → next design offline against a
**sequential contrastive information bound** (sPCE/sNMC), so at deployment the next design is one forward pass
(milliseconds) and the policy is **non-myopic** by construction.
### 3.2 iDAD — Ivanova, Foster, Kleinegesse, Gutmann, Rainforth, NeurIPS 2021 **[S]**
arXiv:2111.02329. Implicit-likelihood version (no explicit likelihood needed).
### 3.3 Step-DAD (arXiv:2507.14057) and TNDP / decision-aware amortized BED (Huang et al., NeurIPS 2024,
arXiv:2411.02064) **[S]** — semi-amortized (re-optimise the policy as data arrives) and decision-utility-targeted
variants respectively. TNDP overlaps `rank_targeted_acquisition.md` §"decision-focused BED".

**Do they apply to an 18,359-item discrete question space? — NO, and this is a clean, citable UNFIT.** Three
independent reasons, in the order a reviewer will accept them:
1. **The design space.** DAD's policy emits a design vector and is trained by **gradients through the design**.
   Published applications are low-dimensional and *continuous* (location finding, pharmacokinetics, death process,
   hyperbolic temporal discounting). A categorical design space of 18,359 requires either a softmax head over 18k
   classes trained by REINFORCE/Gumbel (high variance, and at that point it is DRE's Gumbel head, which we are
   already running as the *static* bar) or a continuous proto-design + kNN snap (which is Wolpertinger,
   `dulacarnold2015deep`, already our INDEX's continuous-action ancestor). **Neither is DAD any more.**
2. **The likelihood.** sPCE needs a tractable `p(y|θ,ξ)` to contrast against. Our answer generator is *real
   held-out ratings*, i.e. an empirical, non-parametric, MNAR process. iDAD relaxes this to implicit likelihoods
   but then needs a *simulator* — which our behavioural firewall forbids (`answer-simulator-circularity-ruling`).
3. **Amortization buys nothing here.** DAD's headline benefit is *real-time* design at deployment. Our greedy
   selector over a frozen tower is already closed-form and instant; we have no latency problem to amortize away.
**Verdict: UNFIT (all four). Cite DAD + iDAD in one sentence as the amortized-BED state of the art, state reason
(1) — categorical 18k design space with no likelihood — and move on.** Effort if attempted: **2+ weeks, and it
would be an unforced error.**

### 3.4 BED-LLM — Choudhury, Williamson et al. (Apple / Oxford / CityU), arXiv:2508.21184, 2025 **[S]**
Sequential BED where an LLM's predictive distribution supplies the probabilistic model; EIG estimated
Rao-Blackwellised over filtered posteriors. Evaluated on 20-Questions **and a movie preference-elicitation
scenario**, beating naive and entropy-based questioning by turn ~3.
**Verdict: UNFIT to run** (LLM API calls forbidden without author approval; LLM-simulated preferences =
the circularity our firewall exists to prevent) — **but MUST-CITE**: it is the 2025 statement that sequential BED
is the right frame for multi-turn preference elicitation, and it is the paper a 2026 reviewer will name.

### 3.5 InfoBAX — Neiswanger, Wang & Ermon, ICML 2021 **[S]**
arXiv:2104.09460, PMLR 139:8005–8015; code https://github.com/willieneis/bayesian-algorithm-execution.
Acquisition = **mutual information between the next query and the OUTPUT OF AN ALGORITHM** run on the black-box
function (via its execution path), rather than with the function or its optimum. Up to 500× fewer queries.
**Why it is interesting for us and not just a curiosity:** set the "algorithm" = *top-10 retrieval over the
catalogue*. Then InfoBAX's criterion is literally "information about the identity of the top-10 list", which is a
principled, published alternative to both EVOI and our L-optimal/rank-discounted criterion — and it is
**posterior-mean-dependent by construction**, i.e. it fails the static-optimality theorem's hypothesis in exactly
the way `rank_targeted_acquisition.md` §3 wants. It is the closest published thing to "target the ranking, through
a posterior."
**Verdict: NICE-HIGH (theory), NICE (baseline).** Effort: **M, 12–20 h** — the execution-path MI needs posterior
samples of the score vector (cheap under our Gaussian) and a top-K "algorithm" whose execution path is just the
argsort. Worth it if Paper B needs a second principled acquisition beyond EDDI/BALD.

---

## 4. Goal-oriented / transductive design — SETTLED ELSEWHERE, do not redo
`rank_targeted_acquisition.md` already covers, with verdicts: **Yu, Bi & Tresp, TED (ICML 2006)**;
**A-GOODE (Attia, Alexanderian & Saibaba 2018)**; Neuberger et al. 2024; Hübotter et al. NeurIPS 2024;
**Knowledge Gradient** (Frazier, Powell & Dayanik 2009); OCBA/OCBA-m; **Chajewska, Koller & Parr AAAI 2000**;
**Viappiani & Boutilier** (RecSys'09 / NeurIPS'10 / AIJ 2020); Lin et al. WWW 2023 greedy-NDCG selector;
Jedynak–Frazier–Sznitman; Krause–Guestrin; Golovin–Krause.
**One addition from this pass:** TED is **static/batch by construction** (it selects a set minimising predictive
variance over the target pool before any label arrives) — that makes it a legitimate *static* arm for Act 1,
sitting alongside DRE, and it is ~50 lines over our item factor matrix. **Verdict: NICE, effort S (4 h).**

---

## 5. Classic active learning for collaborative filtering

| Method | Cite | Mechanism | Adaptive? | Verdict |
|---|---|---|---|---|
| **Jin & Si, UAI 2004** | arXiv:1207.4146 **[V-abstract]** | Expected loss under the **posterior distribution over the model**, not the point estimate; "previous work … only considers the expected loss based on the estimated model, which can be misleading when the estimated model is inaccurate" | Adaptive | **NICE** — the Bayesian-AL-for-CF anchor; the paper that says "use the posterior" 20 years before we do. Loss is prediction (RMSE-type), not ranking. Effort **S, 6 h** over `belief_mf.py`. Cite regardless of running. |
| **Rubens & Sugiyama, RecSys 2007** | Influence-based collaborative AL **[S]** | Select items that are uncertain **and influential** — reduce uncertainty over *all other items* | Adaptive | **NICE** — the closest pre-existing recsys relative of a targeted criterion; already flagged in `rank_targeted_acquisition.md`. Effort **S, 4 h**. |
| **Elahi, Ricci & Rubens, CSR 2016 survey** (`elahi2016survey`) | already indexed **[V-prior]** | The taxonomy; **"acquisition probability based"** category; "success ratio = #acquired/#requested" | — | **MUST-CITE** — supplies the field's vocabulary for answerability. Never run. |
| **Elahi, Ricci & Rubens, ACM TIST 2014** | Binary Prediction **[V-prior]** | MF on the binarised observation mask ⇒ learned P(user rated i) | Static order | **NICE** — the recsys-native learned answerability model; our natural `A_i` estimator if we run the patent formula. Effort **S, 3 h**. |
| **Karimi et al., ICTAI 2011 / RecSys 2012** | **[S]** | MF-specific AL incl. non-myopic; "optimal AL selects queries that directly optimize expected error for test data" | Both | **CHECK-THEN-CITE** — standing debt from `rank_targeted_acquisition.md`; if they do transductive design in a CF latent, part of our framing shrinks. |
| **Sutherland, Póczos & Schneider, KDD 2013** | AL & search on low-rank matrices **[S]** | Posterior-variance querying in an MF latent | Adaptive-in-name | **CITE only** — the max-variance baseline we already know is static-equivalent. |

**Decision-theoretic elicitation (Boutilier lineage), all CITE-ONLY, all already partly in
`rank_targeted_acquisition.md`:** Boutilier, *A POMDP formulation of preference elicitation problems*, **AAAI 2002**
(the formally-correct-but-intractable statement of our problem; the reason everyone else is myopic);
Chajewska, Koller & Parr, AAAI 2000 (VOI-driven adaptive utility elicitation);
Braziunas & Boutilier, *Local utility elicitation in GAI models* (UAI 2005), *Minimax regret-based elicitation of
generalized additive utilities* (UAI 2005), *Elicitation of factored utilities* (**AI Magazine 29(4):79–92, 2008** —
the readable survey); Braziunas, *Computational approaches to preference elicitation*, UofT TR 2006 (the survey to
cite once for the whole field). **[S]** for all. **None of these is answerability-aware**; all assume the user can
answer any utility query. Their contribution to us is the POMDP framing and the myopic-VOI justification, both of
which we should cite in one sentence and not attempt to run — the POMDP is intractable at |A| = 18,359.

---

## 6. Active preference learning from pairwise comparisons (Bıyık / Sadigh, incl. robotics origins)

The robotics origin matters: this line was invented for **reward learning**, where the query is a pair of robot
trajectories, and it is *far* more careful about the human than the recsys line is.

| Paper | Mechanism | Answerability? | Verdict |
|---|---|---|---|
| **Sadigh, Dragan, Sastry & Seshia, RSS 2017** — *Active preference-based learning of reward functions* **[S]** | The origin: synthesise a trajectory pair, update a Bayesian belief over reward weights ω, select the query maximising **volume removal** in the belief | No | **CITE** — the ancestor. Volume removal is the criterion Bıyık 2019 shows is *degenerate* (it is maximised by queries the human cannot distinguish!) — a beautiful, quotable failure mode of an answerability-blind criterion. |
| **Bıyık & Sadigh, CoRL 2018** — *Batch active preference-based learning* **[S]** | Batch query selection with diversity; ~2–3 s query generation | No | CITE (batch = our screen). |
| **Bıyık, Palan, Landolfi, Losey & Sadigh, CoRL 2019** — *Asking Easy Questions* **[V]** | See §0.2. Info gain + "About Equal" + δ | **YES — the reference implementation of answerability-aware acquisition** | **MUST-CITE (do not run — pairwise trajectory queries, no catalogue, no ranking metric).** Effort to port the *criterion* to our 3-outcome item/concept answers: **M, 8–12 h**, and it would be a strong Act-1 arm. |
| **Bıyık, Huynh, Kochenderfer & Sadigh, IJRR 2024** — *Active preference-based GP regression* **[S]** | Non-parametric GP reward, active queries | No | CITE. |
| **Bıyık et al., THRI 2024 / arXiv:2402.15757** — *Batch AL of reward functions* **[S]** | Consolidated batch methods | No | CITE. |
| **APReL library** (arXiv:2108.07259, code public) **[S]** | Implements the whole family (volume removal, info gain, batch, easy-questions) | — | **Useful**: if we port the criterion, APReL is the reference implementation to check against. |
| **Bıyık et al. 2023 — Soft Attributes** (`biyik2023soft`) | Gaussian belief, items + CAV directions, EVOI | **NO — and explicitly names the gap**: *"In lieu of a detailed familiarity model for u…"* (§3.1) | Already ruled **NICE** in `adaptive_elicitation_baselines.md`. **The quoted admission is our single best "the nearest rival concedes this cell" sentence.** |

**Verdict for the family: MUST-CITE, RUN NOTHING AS-PUBLISHED** (trajectory-comparison queries, alignment metric,
no catalogue, no held-out ranking — a contract violation of the same class as NICF). **Port one criterion**
(the §0.2 information gain with a three-outcome answer) as an Act-1 arm.

---

## 7. Computerized Adaptive Testing — the portability verdict

### 7.1 What CAT actually does
Three components (Deng/Chang et al.): a **response model** (IRT/CDM), an **item-selection rule**, and
**exposure control**. **[S/V]**
- **Maximum Fisher Information (MFI)** — select `argmax_j I_j(θ̂)`; for 2PL `I_j(θ) = a_j² P_j(θ)(1−P_j(θ))`. The
  structural consequence is **b-matching**: an item is maximally informative when its **difficulty b_j ≈ the
  person's ability θ**. Known failure: unstable at the start (θ̂ is garbage) and **grossly uneven item exposure** —
  a few high-`a` items are administered constantly, most are never used.
- **a-stratification** — **Chang & Ying, *Applied Psychological Measurement* 23(3):211–222, 1999**. Partition the
  bank by discrimination `a`; administer **low-a items EARLY and high-a items LATE**, because early θ̂ is
  imprecise and spending your most discriminating items against a bad estimate wastes them. Variants: AST
  (a-stratification with b-blocking), MIS (adds the guessing parameter c).
- **Exposure control** — Sympson & Hetter (1985) probabilistic exposure caps; randomesque (pick randomly from the
  top-n informative); Stocking–Lewis; fade-away. **Purpose: test SECURITY and item-bank economics.**
- **Modern/RL CAT** — **NCAT** (attention DQN item selection, exposure controlled by Boltzmann sampling),
  **GMOCAT** (KDD 2023, graph-enhanced multi-objective: accuracy + coverage + **exposure**), BOBCAT,
  MAAT (active-learning-based, model-agnostic); toolkit **EduCAT** (https://github.com/bigdata-ustc/EduCAT). **[S]**

### 7.2 Portability, rule by rule — the verdict
| CAT rule | Portable? | Why |
|---|---|---|
| **Maximum Fisher Information** | **YES, in form; already ours in substance.** | MFI over a logistic response model at the current posterior mean is the same object as our discrimination-ranked bank. Our certified concept channel is ranked by **pure discrimination** and that is CAT's `a` parameter under a different name (the Aug-4 finding that ranking the bank by pure discrimination flips which channel wins is, in CAT terms, an `a`-ranked bank). **Running a real 2PL/MIRT fit on our data to get calibrated (a, b) per item is a legitimate, cheap arm.** |
| **b-matching (difficulty ≈ ability)** | **PARTIALLY — and it is the interesting one.** | Our latent is a taste vector, not a unidimensional ability, so "difficulty" has no direct analogue. But MIRT gives a multidimensional version, and MIRT+CAT has been fitted **on ML-1M**, where a 3-dimensional MIRT beat MF with up to 20 dimensions (Behaviormetrika 2024) **[S]**. |
| **a-stratification (low-a first, high-a late)** | **YES, and it is a cheap TESTABLE ARM that contradicts what we currently do.** | Our ranked bank is greedy-by-discrimination from question 1. Chang & Ying's argument — do not spend high-discrimination items on a bad ability estimate — is a direct, published prediction that a **greedy discrimination order is suboptimal at small k**. Given the certified result that the bank ranking decides which channel wins, this is worth one run. Effort: **S, 4–6 h** (it is a re-ordering of an existing bank). |
| **Exposure control (Sympson–Hetter, randomesque, Stocking–Lewis)** | **NO — do not present this as answerability.** | It caps how often an item is shown **across the population**, for test security and bank economics. The recsys analogue is popularity bias / question-bank diversity, **not** the user's ability to answer. Misrepresenting exposure control as an answerability mechanism is the kind of error a psychometrics-literate reviewer will catch instantly. |
| **The whole CAT response model** | **NO for our failure mode.** | **CAT assumes universal response**: every examinee responds to every administered item. A *wrong* answer is still informative — that is what the guessing parameter `c` is for. Our unanswerable question yields **zero** information and burns budget. There is no term in MFI for "no response arrives." |

### 7.3 The one genuinely empty CAT-side cell — and it is a good one
IRT **does** have a model of non-response, and it is exactly the right shape:
**Holman & Glas, *BJMSP* 2005, "Modelling non-ignorable missing-data mechanisms with item response theory models"**
**[S]** — estimate an IRT model for the observed responses **concurrently** with a second IRT model for the
**propensity to omit**, i.e. *"the joint probability of scoring/non-scoring on and not missing/missing an item as a
function of the item features, the examinee's ability and his propensity to omit an item."* Extensions:
Glas & Pimentel (not-reached items), Rose/von Davier/Xu, Liu et al. *BJMSP* 2017 (examinee-selected items),
response-time-based treatments. This is a **two-dimensional latent: (ability, omission propensity)** — precisely a
principled `P(answerable | u, i)` with item-side and person-side parameters.

**But: this pass found NO work that puts a non-ignorable-missingness IRT model inside a CAT ITEM-SELECTION RULE.**
The missingness literature is about *calibration and scoring*; the item-selection literature assumes response.
Searches for nonresponse-aware Fisher information / omission-propensity item selection returned nothing.
**[S — absence of evidence at this search depth, i.e. [A]-strength, not [V]. Do not write "nobody has" without one
more targeted pass through *Applied Psychological Measurement* / *Journal of Educational Measurement*.]**

**This is the best framing available to us on the CAT side:** *psychometrics has a rigorous latent model of the
propensity to omit, and a rigorous theory of adaptive item selection, and has not joined them; we join them, in a
recommender.* It is narrower and more defensible than any "the literature ignores answerability" claim.

### 7.4 Has anyone bridged CAT and recsys elicitation?
**Thinly, and never on our contract. [S]**
- **MIRT + CAT on MovieLens-1M** (*Behaviormetrika*, 2024, "Recommendation with item response theory") — 3-dim MIRT
  beat MF up to 20 dims; explicitly frames CAT as preference *exploration* with an interpretable
  explore/exploit knob. **The closest existing bridge. Get the primary text.**
- **"What We Evaluate When We Evaluate Recommender Systems" (RecSys 2023)** — uses IRT to analyse *recommender*
  performance (items = users, ability = system quality). Different direction; cite only if we discuss IRT at all.
- **IRT rating-scale models to correct CF ratings** (older, [S]).
- **Active Matrix Factorization for Surveys** (arXiv:1902.07634) — CAT-style adaptive item selection for survey
  batteries via MF; "shorten a battery by 40%". Adjacent and quotable.
- **NCAT / GMOCAT / BOBCAT** — modern CAT, education only; no ranking metric, no catalogue.
**Verdict: the bridge is UNOCCUPIED for our contract (cold-start, 18k catalogue, held-out full/tail NDCG@10).
Cite the MIRT-on-ML-1M paper as the nearest bridge and do not overclaim novelty on "IRT for movies".**

---

## 8. Per-method verdict table

Effort in **agent-hours**. Contract = ML-25M, binary implicit, cold user, k∈{1,2,4,8,16}, item or genome-concept
questions, answers from real held-out ratings, unanswerable questions burn budget, full+tail NDCG@10, no LLM,
no simulator, 18,359 items.

| # | Method | Cite | Mechanism (reimplementable) | Adapt? | Code | Their data / metric | Fit | Effort | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **EDDI** | Ma et al., ICML 2019 | Partial-VAE q(z\|x_O); `argmax_i E_{x_i} KL[p(x_φ\|x_i,x_O) ‖ p(x_φ\|x_O)]` over **target** vars | Y | ours + MS repo | UCI, MIMIC / RMSE-AUIC | ours already | **4–8** (3-outcome rerun) | **MUST-INCLUDE** — and rerun with a 3-outcome likelihood (§0.6) |
| 2 | **BALD** | Houlsby et al. 2011 | `H[y\|x,D] − E_θ H[y\|x,θ]` over the tower posterior + probit answer model | Y | trivial | GP classification / preference pairs | perfect | **4–6** | **MUST-INCLUDE** — the named info-theoretic arm |
| 3 | **SEL-BALD** | Gao, Saar-Tsechansky & Yin, NeurIPS 2024 | `P(label obtained\|x) · BALD(x)`; also acquires info about the discretion model | Y | ruijiang81/SEL-BALD | tabular high-stakes / AUC | our structural answerability rule = their discretion model | **6–10** | **MUST-INCLUDE** — the answerability arm; turns our complaint into a method we ran |
| 4 | **Bıyık "easy questions" info gain** | CoRL 2019 | `H(q\|Q) − E_ω H(q\|ω,Q)` with a 3rd "About Equal" outcome (δ) | Y | Stanford-ILIAD/easy-active-learning | Driver/Tosser/Fetch / alignment | criterion ports; query type does not | **8–12** (port only) | **MUST-CITE; NICE-HIGH to port** — the principled answerability criterion |
| 5 | **Patent IG×A** | US 11,775,493 B2 (Microsoft; Venanzi, Hofmann, Radlinski, Brost) | `s_i = IG_i × A_i`, `A` = empirical answer frequency | Y | — | search / — | it IS our formula | **2** (≈ Pop×Ent with a learned A) | **MUST-CITE (blocking prior art); NICE to run as an arm** |
| 6 | **BatchBALD** | Kirsch, van Amersfoort & Gal, NeurIPS 2019 | Joint MI over a batch; greedy submodular | Y (batch) | public | image AL / accuracy | our k-question screen | **6–10** | **NICE-HIGH** — the principled account of why a static order can beat greedy |
| 7 | **InfoBAX** | Neiswanger, Wang & Ermon, ICML 2021 | MI between query and an **algorithm's execution path**; set algorithm = top-K retrieval | Y | willieneis/bayesian-algorithm-execution | synthetic/BO / — | reframes acquisition as "info about the top-K list" | **12–20** | **NICE-HIGH (theory) / NICE (baseline)** |
| 8 | **TED** | Yu, Bi & Tresp, ICML 2006 | Minimise predictive variance over a *target pool*; batch/static | **N** | ≈50 lines | regression / MSE | perfect; a legit STATIC arm next to DRE | **4** | **NICE** — cheap static bar |
| 9 | **Jin & Si** | UAI 2004 | Expected loss under the **posterior** over the model, not the point estimate | Y | none | CF / MAE | prediction loss ≠ ranking | **6** | **NICE** — the Bayesian-AL-for-CF anchor; cite regardless |
| 10 | **Rubens & Sugiyama** | RecSys 2007 | Select items uncertain **and influential** on other items' predictions | Y | none | CF / MAE | untargeted influence | **4** | **NICE** |
| 11 | **Elahi Binary Prediction** | ACM TIST 2014 | MF on the binarised mask ⇒ learned P(user rated i) | static order | none | CF / #ratings+MAE | our `A_i` estimator | **3** | **NICE** — pairs with #5 |
| 12 | **MFI / a-stratification (CAT)** | Chang & Ying, APM 23(3):211–222, 1999 | Fit 2PL/MIRT ⇒ (a,b); MFI = `argmax a²PQ`; a-strat = low-a early, high-a late | Y | mirt/EduCAT | education / θ-recovery, exposure | needs an IRT fit of our data; **contradicts our greedy-a bank** | **8–12** (incl. IRT fit) | **NICE-HIGH** — cheap, current, and it is a *falsifiable prediction against our current bank order* |
| 13 | **Non-ignorable-missingness IRT** | Holman & Glas, BJMSP 2005 | Joint IRT for response + IRT for **omission propensity**; 2-D latent (ability, propensity) | (estimation) | mirt-adjacent | education / calibration | the right psychometric model of answerability; **never used inside item selection** | **12–16** | **NICE-HIGH (conceptual)** — the best CAT-side framing; run only if we build a learned answerability model |
| 14 | **Icebreaker** | Gong et al., NeurIPS 2019 | BELGAM (Bayesian weights, SG-MCMC) + element-wise acquisition | Y | microsoft/Icebreaker | MovieLens-shaped element acquisition / RMSE | acquires cells to TRAIN, not to interview | **12–20** | **NICE (cite; run only on demand)** |
| 15 | **BED-LLM** | arXiv:2508.21184, 2025 (Apple/Oxford) | Sequential BED with LLM predictive model; Rao-Blackwellised EIG; 20-Q + movie PE | Y | ? | LLM-simulated / success rate, rating | LLM forbidden; simulated preferences | — | **UNFIT to run / MUST-CITE** |
| 16 | **DAD** | Foster et al., ICML 2021 | Amortized design **policy** trained on sPCE contrastive bound; non-myopic, ms-latency | Y | public | location finding, PK, death process / EIG | ✘ 18k **categorical** design space, no tractable likelihood, no latency problem to solve | 2+ wk | **UNFIT (reason: design space + likelihood)** — cite in one sentence |
| 17 | **iDAD** | Ivanova et al., NeurIPS 2021 | Implicit-likelihood DAD | Y | public | same | ✘ same, plus needs a simulator (firewall) | 2+ wk | **UNFIT** |
| 18 | **Step-DAD / TNDP (decision-aware amortized BED)** | arXiv:2507.14057; Huang et al. NeurIPS 2024 | Semi-amortized policy; transformer neural decision process jointly amortizing design+action | Y | public | synthetic / decision utility | ✘ same as DAD | 2+ wk | **UNFIT — cite with the goal-oriented line in `rank_targeted_acquisition.md`** |
| 19 | **Boutilier POMDP elicitation** | AAAI 2002 | Elicitation as a POMDP over utility space | Y (optimal) | none | — | intractable at \|A\|=18,359; the reason everyone is myopic | — | **CITE-ONLY (framing)** |
| 20 | **Chajewska–Koller–Parr; Braziunas & Boutilier** | AAAI 2000; UAI 2005 ×2; AI Mag. 29(4):79–92, 2008 | VOI-driven adaptive utility elicitation; GAI/minimax-regret elicitation | Y | none | decision analysis / regret | no answerability, no catalogue, no ranking | — | **CITE-ONLY** (already partly in `rank_targeted_acquisition.md`) |
| 21 | **Sadigh RSS'17 volume removal** | RSS 2017 | Belief-volume-removal query selection | Y | APReL | robotics / alignment | ✘ | — | **CITE** — its degeneracy (maximised by indistinguishable queries) is the textbook answerability-blind failure |
| 22 | **MIRT+CAT on ML-1M** | Behaviormetrika 2024 | 3-dim MIRT + CAT item selection on MovieLens | Y | ? | ML-1M / rating fit | the nearest CAT↔recsys bridge; no held-out ranking | **8** to replicate | **NICE (cite; the bridge exists, do not claim it doesn't)** |
| 23 | **PERE experience-probability weighting** | Nguyen et al., UAI 2024 | Region-cut info gain **× estimated experience probability** | Y | none | ML-10M/20M, NDCG@10, 100-item budget | already MUST-INCLUDE per `adaptive_elicitation_baselines.md` | (there) | **MUST-INCLUDE — and it is also an answerability-aware acquisition; verify the formula** |

---

## 9. What to actually do (shortlist, ranked by leverage per hour)

1. **Fix the EDDI likelihood (3-outcome) and rerun.** 4–8 h. Converts a soft complaint into a mechanism, and
   removes the strongest objection to our current framing. §0.6.
2. **BALD + SEL-BALD over the tower posterior.** 10–16 h combined. Gives us the named information-theoretic arm
   *and* the named answerability-aware arm, both 2011/2024 citations, both on our own contract.
3. **The patent formula `IG × A` with Elahi's Binary-Prediction `A`.** 5 h. It is Rashid's Pop×Ent with a learned
   answerability term, and running it lets us say we compared against the literal prior art.
4. **a-stratified bank order (Chang & Ying 1999).** 4–6 h. A falsifiable published prediction *against* our current
   greedy-discrimination bank, at small k. Cheap and it could change a certified result.
5. **BatchBALD.** 6–10 h. The principled account of "why does a static screen beat greedy selection", which is
   currently an unexplained finding.
Everything else: cite.

**Explicitly park (with the reason to write in the paper):** DAD/iDAD/Step-DAD/TNDP (18k categorical design space,
no tractable likelihood); BED-LLM (LLM + simulated preferences); the whole Bıyık/Sadigh robotics evaluation
(trajectory queries, alignment metric); Boutilier's POMDP (intractable); Icebreaker (acquires cells to train, not
to interview).

---

## 10. Verification debts (do not quote these without closing them)
- **[HIGH] SEL-BALD's exact acquisition formula.** I have the setting and the multiplicative structure at **[S]**
  (search summary + a low-fidelity PDF extraction that OpenReview blocked). Read the NeurIPS 2024 camera-ready
  before writing `P(label)·BALD` as *their* formula, and get their datasets/metrics.
- **[HIGH] PERE's experience-probability term.** Carried at **[S]** through two files now. If PERE really does
  `info-gain × P(experienced)` on ML-20M NDCG@10, it is a 2024 peer-reviewed pre-emption of our *recsys*
  answerability contribution and the framing must change. **Read the UAI/PMLR v244 paper.**
- **[HIGH] Houlsby et al. 2011 preference-learning section.** The claim "BALD-for-preference-elicitation is
  occupied at the source" is **[S]**. Read arXiv:1112.5745 §preference learning.
- **[MED] "No CAT item-selection rule uses omission propensity."** This is **[A]**-strength (absence at this search
  depth). One targeted pass through *Applied Psychological Measurement* and *Journal of Educational Measurement*
  before the claim goes in a paper.
- **[MED] Bıyık CoRL 2019 Eq. numbers.** The formulas in §0.2 came from an ar5iv HTML extraction **[V]**, but the
  equation numbering (Eq. 5, 7, 8) should be checked against the CoRL camera-ready before citing by number.
- **[MED] MIRT+CAT on ML-1M (Behaviormetrika 2024)** — abstract-level **[S]**; get authors, exact metric, and
  whether any held-out ranking metric is reported.
- **[LOW] Icebreaker's exact acquisition objectives** — **[A]**.
- **[LOW] US 11,775,493 B2 claim scope.** The quotes in §0.1 are **[V]** from the granted specification via Google
  Patents. Whether the *claims* (as opposed to the description) cover `IG × A` matters not at all for citation, but
  do not describe it as "patented" — describe it as "disclosed in".

---

## 11. New sources — owed to `../INDEX.md` + `references.bib`

| suggested key | citation | why |
|---|---|---|
| `venanzi2023patent` | Venanzi, Hofmann, Radlinski & Brost. *Information retrieval system*. US Patent 11,775,493 B2 (priority 2016, granted 2023) | **The literal `EIG × P(answerable)` formula.** Blocking prior art for any answerability-weighted acquisition claim |
| `biyik2019asking` | Bıyık, Palan, Landolfi, Losey & Sadigh. Asking Easy Questions: A User-Friendly Approach to Active Reward Learning. CoRL 2019 | The principled answerability-aware information gain; explicit "the human may not be able to answer" |
| `gao2024selbald` | Gao, Saar-Tsechansky & Yin. SEL-BALD: Deep Bayesian Active Learning with Selective Labels. NeurIPS 2024 | Active Learning with Instance Rejection = our problem in AL's vocabulary |
| `houlsby2011bald` | Houlsby, Huszár, Ghahramani & Lengyel. Bayesian Active Learning for Classification and Preference Learning. arXiv:1112.5745, 2011 | BALD; the MI decomposition every answerability-aware criterion is an instance of |
| `kirsch2019batchbald` | Kirsch, van Amersfoort & Gal. BatchBALD. NeurIPS 2019 | Batch/screen acquisition; why greedy top-k is redundant |
| `gong2019icebreaker` | Gong, Tschiatschek, Nowozin, Turner, Hernández-Lobato & Zhang. Icebreaker. NeurIPS **2019** | EDDI's successor; ice-start element acquisition |
| `foster2021dad` | Foster, Ivanova, Malik & Rainforth. Deep Adaptive Design. ICML 2021 | The amortized-BED SOTA we cite-and-exclude |
| `ivanova2021idad` | Ivanova, Foster, Kleinegesse, Gutmann & Rainforth. Implicit Deep Adaptive Design. NeurIPS 2021 | ditto, implicit-likelihood |
| `neiswanger2021infobax` | Neiswanger, Wang & Ermon. Bayesian Algorithm Execution. ICML 2021 | MI with an algorithm's output = "information about the top-K list" |
| `bedllm2025` | Choudhury, Williamson et al. BED-LLM. arXiv:2508.21184, 2025 | 2025 statement that sequential BED is the frame for multi-turn preference elicitation |
| `chang1999astratified` | Chang & Ying. a-Stratified Multistage Computerized Adaptive Testing. *Applied Psychological Measurement* 23(3):211–222, 1999 | CAT's exposure-aware selection; low-a-first is a falsifiable prediction against our greedy bank |
| `sympson1985exposure` | Sympson & Hetter. Controlling item-exposure rates in CAT. Proc. 27th Ann. Meeting Military Testing Assoc., 1985 | The canonical exposure-control cite (**for security, NOT answerability**) |
| `holman2005nonignorable` | Holman & Glas. Modelling non-ignorable missing-data mechanisms with item response theory models. *BJMSP* 58:1–17, 2005 | Joint IRT for response + omission propensity — the psychometric answerability model |
| `boutilier2002pomdp` | Boutilier. A POMDP formulation of preference elicitation problems. AAAI 2002 | The intractable-but-correct statement of our problem |
| `braziunas2008factored` | Braziunas & Boutilier. Elicitation of factored utilities. *AI Magazine* 29(4):79–92, 2008 | The readable survey of the decision-theoretic elicitation line |
| `sadigh2017active` | Sadigh, Dragan, Sastry & Seshia. Active preference-based learning of reward functions. RSS 2017 | Volume removal; the answerability-blind criterion Bıyık falsifies |
| `jin2004bayesianal` | Jin & Si. A Bayesian approach toward active learning for CF. UAI 2004 | *(already owed by `rank_targeted_acquisition.md`)* |
| `rubens2007influence` | Rubens & Sugiyama. Influence-based collaborative active learning. RecSys 2007 | *(already owed)* |
| `yu2006ted` | Yu, Bi & Tresp. Active learning via transductive experimental design. ICML 2006 | *(already owed)* — also a legit STATIC arm |
| `elahi2014active` | Elahi, Ricci & Rubens. Active learning strategies for rating elicitation in CF. ACM TIST 5(1), 2014 | *(already owed by `answerability_in_elicitation.md`)* — Binary Prediction = learned `A_i` |
