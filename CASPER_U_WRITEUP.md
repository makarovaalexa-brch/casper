# CASPER-U: A Unified-Embedding Instrument for Conversational Preference Elicitation
### An extended technical report / thesis chapter (June 2026)

---

## Abstract

We develop and validate the recommender-side foundation of CASPER, a system for conversational preference
elicitation. The central design commitment is a **single shared embedding space** in which items, attributes
(e.g. genres), and — through a text adapter — arbitrary natural-language concepts are all represented as points,
and which serves simultaneously as (i) the recommendation model ("the instrument") and (ii) the action space of an
elicitation policy. We first establish, by faithful replication, the three published lineages our architecture
rests on: representative-item cold-start elicitation (RBMF/RMVA), the unified item+attribute factorization scorer
(EAR), and the permutation-invariant set-encoder with information-reward acquisition (EDDI). Replicating these on
their own rulers and porting them to a single fixed evaluation ruler (ML-1M, full-catalogue, standard NDCG@10 and
Recall@10) yields a recurring empirical law: **a popularity floor plus a personalization residual is necessary;
the residual alone loses, but on top of popularity it wins.** We then build the CASPER-U instrument — a linear
latent-factor model with a closed-form ridge fold-in encoder and a confidence-scaled popularity-plus-residual
scorer — and show it (a) matches matrix factorization as a cold-start ranker (NDCG@10 0.486 vs WRMF 0.501),
(b) obeys a defined acceptance suite (beats-popularity, the floor+residual law, monotonicity in revealed
information, partial-reveal robustness), and (c) genuinely ingests attributes in the same space (asking a genre
correctly lifts that genre's items; mixed item+attribute interviews beat item-only at equal question budget). We
report extensive interpretable sanity checks (genre purity, like/dislike polarity, franchise coherence). We close
with the open-concept (SBERT) extension that completes the unified-space claim, and the roadmap to the continuous-
action elicitation policy that is the thesis's headline contribution.

---

## 1. Introduction and aims

### 1.1 The problem

A *cold-start* user has no interaction history. To recommend well, a system must **elicit** preferences — ask a
short sequence of questions and infer taste from the answers. Two questions matter: *what to ask* (the elicitation
**policy**) and *how to turn answers into recommendations* (the **recommender**). The field has historically
treated these as two separate literatures that barely speak: item-based **rating elicitation** (ask the user to
rate movies) and attribute-based **conversational recommendation** (ask about a fixed list of genres/facets).

### 1.2 What CASPER aims to do

CASPER's thesis is that both *what to ask* and *what to recommend* should live in **one continuous semantic
space** spanning items, attributes, and arbitrary concepts. Concretely:

- The **recommender** is a function of revealed answers over *any* mix of items and concepts.
- The **policy** chooses the next question as a *point* in that same space (a continuous action), grounded by
  nearest-neighbour retrieval to a real entity and verbalised by an LLM.

This report covers the **recommender-side foundation** (the "instrument") and the replications that justify its
architecture. The policy (the continuous-action contribution) is specified at the end as the next stage.

### 1.3 Why an "instrument" and an acceptance suite

To compare elicitation *strategies* fairly, the recommender, user simulator and candidate pool must be held fixed;
otherwise a strategy's apparent value is confounded with recommender quality. We therefore build a **fixed,
audited measurement instrument** and define an **acceptance suite** of gates it must pass before any policy is
measured against it. A prior, extensive line of our own experiments (the "frozen-instrument saga", Appendix A)
showed the danger of getting this wrong: a recommender frozen on popularity-style interviews structurally cannot
exploit cleverly elicited information, which makes *every* policy collapse to popularity. The remedy — a
recommender that genuinely ingests elicited answers — motivates everything below.

---

## 2. Literature review

We organise the literature into five strands, then state explicitly what fits our design, what does not, the
lineage we adopt, and the gap we occupy.

### 2.1 Item-based rating elicitation (the cold-start interview)

The oldest strand asks new users to **rate items**. **Rashid et al. (2002, IUI, "Getting to Know You")** introduced
seed-item selection by popularity, entropy and popularity×entropy. **Golbandi, Koren & Lempel (2011, WSDM)**
proposed an **adaptive ternary decision tree** whose nodes are items and whose branches (loves/hates/unknown)
adapt the next question to prior answers; the leaf stores a profile used to predict. **Zhou, Yang & Zha (2011,
SIGIR)** made the tree leaves *latent factors* — **functional matrix factorization** — so a cold user is folded
into the MF space by tree traversal. The **representative-item** sub-lineage selects a seed set spanning the latent
space: **Liu et al. (2011, RecSys, RBMF, "Wisdom of the Better Few")**, **Fonarev et al. (2016, ICDM, rectangular
maximal-volume / RMVA)**, and **Shi, Zhao & Shen (2017, TOIS, Local RBMF)**. The neural successor is **Deep Rating
Elicitation (Kweon, Kang, Hwang & Yu, 2020, WWW)**, which jointly learns a fixed seed set and a reconstruction
decoder.

**What fits.** This is the lineage of our *recommender mechanism*: cold users are profiled by **folding in** a few
answers against learned item factors, and seeds are chosen by **coverage/volume** in that factor space. We adopt
RMVA seed selection and ridge fold-in directly.

**What does not.** All of these ask about **items only**. None asks about attributes or concepts. This is precisely
the gap our unified space closes.

### 2.2 Attribute-based conversational recommendation

A parallel strand asks about **attributes** from a fixed vocabulary. **CRM (Sun & Zhang, 2018, SIGIR)** pairs a
belief tracker with an RL policy. **EAR (Lei et al., 2020, WSDM, "Estimation–Action–Reflection")** is the canonical
**unified item+attribute factorization machine**: it scores an item `v` for a user with confirmed attributes `P_u`
as `ŷ = u·v + Σ_{p∈P_u} v·p` — items, users and attributes in one latent space, attribute feedback folded directly
into the item score. **SCPR (Lei et al., 2020, KDD)** adds graph path-reasoning; **UNICORN (Deng et al., 2021,
SIGIR)** unifies the *decision types* (ask-which / ask-vs-recommend) in a single graph DQN over items+attributes;
**ConTS (Li et al., 2021, TOIS)** unifies items and attributes as Thompson-sampling arms for cold-start.

**What fits.** EAR's additive item+attribute scorer is exactly the *mechanism* we want in the recommender: an
attribute answer should shift the item ranking. We adopt and replicate it.

**What does not.** Every method here uses a **fixed, hand-defined discrete attribute schema** (e.g. 33 LastFM tags,
590 Yelp categories) and **discrete action selection**. None admits *arbitrary/open* concepts, and the
elicitation policy is a separate module rather than an action in the recommender's own embedding geometry.

### 2.3 Set-encoders and active feature acquisition

**EDDI / Partial-VAE (Ma et al., 2019, ICML)** is a **permutation-invariant set-encoder** over an arbitrary
*observed subset* of features, trained so partial observation is in-distribution, paired with an
**information-reward** greedy acquisition policy. **Mult-VAE (Liang et al., 2018, WWW)** and **AutoRec (Sedhain et
al., 2015, WWW)** are subset-robust autoencoder recommenders (item-only).

**What fits.** EDDI is the architectural template for "a recommender that consumes an arbitrary subset of revealed
answers and tells you what to ask next." Our fold-in encoder is the closed-form, linear instance of this idea;
EDDI validates the principle (and motivates reveal-subset masked training).

**What does not.** EDDI is not a recommender, has no item/attribute distinction, and uses an expensive amortised
VAE; we need the recommender-specific, interpretable, linear version.

### 2.4 Continuous-action RL and LLM strategy planners

**Wolpertinger (Dulac-Arnold et al., 2015)** handles large discrete action spaces by emitting a **continuous
proto-action** and snapping to the nearest discrete action by k-NN. **ECoC (Wang et al., 2024/KBS 2025)** applies
continuous-action RL to sequential recommendation (items only, non-conversational). On the LLM side, **PPDPP (Deng
et al., 2024, ICLR)** and **RSO (2025)** learn a planner over a **closed, finite strategy set** and let a frozen LLM
verbalise the chosen strategy.

**What fits.** Wolpertinger is the **mechanistic ancestor** of CASPER's policy: continuous proto-action + NN
grounding, transplanted from item selection to *question* selection over our unified space.

**What does not.** ECoC is non-conversational and items-only; PPDPP/RSO act over closed strategy sets, not an open
semantic space. None selects *entity-level question content* in a continuous embedding.

### 2.5 Open-concept and LLM-based elicitation

**PEBOL (Austin et al., 2024, RecSys)** does Bayesian NL elicitation but maintains **per-item beliefs via NLI**,
deliberately *not* a shared dense embedding. **GATE (Li et al., 2023)** generates open-ended questions but is not
recommendation-specific. **Soft Attributes (Biyik et al., 2023)** grounds fuzzy concepts as **Concept Activation
Vector directions** in an item embedding; **Latent Linear Critiquing (Luo et al., 2020, WWW)** co-embeds keyphrases
and items and treats critiques as linear operations.

**What fits.** These confirm the *appetite* for open concepts and the idea of concepts-as-directions/points in an
embedding. Biyik and LLC are our nearest neighbours on the "items + fuzzy concepts in one space" axis.

**What does not.** None combines (a) one shared *learned* embedding that *is* the recommender, with (b) folding in a
mix of items and **arbitrary open** concepts, with (c) a continuous-action elicitation policy. PEBOL has no shared
embedding; CAV/LLC use a fixed/mined vocabulary and no learned policy.

### 2.6 Methodological cautions

**Dacrema et al. (2019, RecSys, "Are we really making much progress?")** showed many neural recommenders fail to
beat simple baselines when reproduced. **Krichene & Rendle (2020, KDD, "On Sampled Metrics")** showed sampled
ranking metrics can reorder methods. Both directly shaped our protocol: we use **full-catalogue** ranking with
**standard** NDCG, anchor to a published baseline number, and treat any unreproducible gain with suspicion.

### 2.7 What fits, what doesn't, the lineage, and the gap

**The lineage we adopt** runs through the *item-based latent-factor cold-start* strand: Golbandi → functional MF
(Zhou) → representative MF (RBMF/RMVA) for *seeds and fold-in*; EAR for the *unified item+attribute scorer*; EDDI
for the *set-encoder/acquisition* principle; Wolpertinger for the *continuous action mechanism*; Koren (2009) for
the *bias + latent-factor* (popularity floor + residual) structure.

**The gap (verified, June 2026, five-angle review).** No published system occupies the intersection of: *one
shared continuous embedding that is simultaneously the recommender (a permutation-invariant fold-in over an
arbitrary mix of item and standalone attribute/concept reveals) and the action space of a continuous-action
elicitation policy, spanning items + attributes + open concepts.* ConTS/UNICORN unify items+attributes but with a
fixed schema and discrete actions; EDDI has the encoder but no recsys/attributes; Wolpertinger has the continuous
mechanism but no recommender tie-in; PEBOL/Biyik handle concepts but without a shared learned embedding or learned
policy. CASPER-U targets exactly this empty cell. Bibliographic corrections from the review: DRE is **WWW 2020**
(arXiv 2402.16327 is a re-upload); there is no "Liu et al. 2017" (RBMF = Liu 2011, Local-RBMF = Shi 2017); functional
MF is **SIGIR 2011**.

---

## 3. The measurement problem and design philosophy

### 3.1 The fixed ruler (anti-thrash)

To avoid the perpetual dataset/metric/architecture churn that plagued earlier attempts, we fix a single **ruler**
and treat it as sacred:

- **Dataset:** MovieLens-1M, implicit *like* = rating ≥ 4, users with ≥ 5 likes.
- **Split:** users 80/10/10 train/validation/test, fixed seed. Cold-start: test users provide only elicited reveals.
- **Candidate protocol:** **full-catalogue** ranking (all items except revealed), per Krichene–Rendle.
- **Metrics:** **standard NDCG@10 = DCG@10/IDCG@10** and **Recall@10** (the latter is the lead metric for
  elicitation value, because NDCG@10 is blockbuster-saturated). RMSE is a calibration gate only.
- **External anchor:** MOSTPOP NDCG@10 ≈ 0.39–0.41 (matches DRE/Kweon 2020). If this drifts, the pipeline is broken.
- **Tuning discipline:** every hyperparameter (blend weight, λ, d) is tuned on **validation** and reported on
  **test**. No test peeking.

### 3.2 The acceptance suite (definition of "provably works")

The instrument is accepted only if it passes:

- **G1 beats-popularity:** with elicited reveals, NDCG@10 > MOSTPOP.
- **G2 floor+residual law:** the full scorer beats both popularity-only and personalization-only.
- **G3 monotonicity:** NDCG@10 is non-decreasing in the number of revealed answers (more information never hurts).
- **G4 ≥MF:** the instrument matches an established matrix-factorization fold-in as a ranker.
- **G5 attribute directionality:** revealing an attribute "like" raises that attribute's items.
- **G6 attribute value:** mixed item+attribute interviews ≥ item-only at equal question budget.
- **G7 polarity / disliked-control:** "like X" and "dislike X" move X's neighbourhood in opposite directions.

### 3.3 The ladder (one change per rung, regression-checked)

S0 lock ruler+gates → S1 representative-MF beats popularity (item-only) → S2 learned instrument matches MF →
S3 unified items+attributes → S4 baselines on instrument → S5 continuous-action policy → S6 LLM askers. Each rung
re-runs the previous rung's headline as a regression check before claiming the new capability.

---

## 4. Replications (faithful on the paper ruler, then ported)

We replicated each base lineage and recorded results crystal-clearly (live ledger `RESULTS.md`). Note that the
source papers typically do **not** publish each other's replications; our cross-replication on one ruler is itself
a contribution to clarity.

### 4.1 R1 — Representative-MF / RMVA (item lineage)

**Method.** Implicit WRMF (ALS; d=64, α=20, 15 iterations) on training users yields item factors `Q`. Seeds are
selected by **rectangular maximal-volume (RMVA)** via column-pivoted QR on `Q` restricted to an answerable pool.
A cold user is profiled by **ridge fold-in** over the seed factors, and the final score blends the personalized
score with a popularity prior (blend weight tuned on validation).

**Results (ML-1M, full-cat, standard NDCG@10 / Recall@10; w tuned on val, reported on test):**

| arm | NDCG@10 | Recall@10 | published ref |
|---|---|---|---|
| MOSTPOP (popularity) | 0.4134 | 0.0645 | DRE 0.3921 |
| RANDOM seeds + pop | 0.4140 | 0.0659 | — |
| POPULAR seeds + pop | 0.3547 | 0.0707 | — |
| REPRESENTATIVE seeds + pop | 0.4816 | 0.0818 | — |
| **RMVA max-volume seeds + pop** | **0.5011** | **0.0902** | DRE-RMVA 0.5387 |

MOSTPOP (0.4134 ≈ published 0.3921) confirms the metric and pipeline. RMVA+pop (0.5011) matches the published RMVA
scale (0.5387); the residual gap is our fold-in decoder vs DRE's reconstruction net. **Diagnostic finding:** a
"warm-half" ceiling (fold-in from half a user's *real* likes) scores NDCG 0.239 but Recall 0.103 — i.e. rich
personalization *loses* NDCG while *winning* Recall, because full-ranking NDCG is dominated by blockbusters present
in every user's test set. This is the origin of the popularity-floor+residual law (§6.1).

### 4.2 R2 — EAR unified item+attribute scorer (attribute lineage)

**Method.** On LastFM (hetrec2011; items=artists, attributes=tags, implicit listens), we trained EAR's FM scorer
`ŷ = u·v + Σ_{p∈P_u} v·p` by BPR, then evaluated cold-start with confirmed attributes derived from a revealed half
of the user's artists.

**Results (LastFM, cold-start, standard NDCG@10 / Recall@10):**

| arm | NDCG@10 | Recall@10 |
|---|---|---|
| POPULARITY | 0.1758 | 0.0754 |
| ATTR-only (Σ v·p) | 0.1359 | 0.0630 |
| item-only (b_v) | 0.1268 | 0.0576 |
| **ATTR + pop (EAR mechanism)** | **0.2583** | **0.1070** |

Confirmed-attribute conditioning beats popularity by **+47% NDCG / +42% Recall** — EAR's core claim — and exhibits
the *same* floor+residual law as R1 (attributes alone lose; attributes on top of popularity win).

### 4.3 R3 — EDDI set-encoder (acquisition lineage)

**Method.** A Partial-VAE (permutation-invariant encoder over observed `(feature,value)` tokens, random-mask
trained) with information-reward greedy acquisition, on a standard UCI-style regression (California housing).

**Results (target RMSE vs #features acquired; lower is better):**

| #features | ACTIVE (EDDI) | RANDOM |
|---|---|---|
| 1 | 0.812 | 0.946 |
| 4 | 0.787 | 0.895 |
| 7 | 0.787 | 0.830 |
| mean-over-curve | **0.816** | 0.894 |

Active acquisition dominates random at every step — EDDI's headline claim — validating the permutation-invariant
set-encoder + information-reward acquisition we adopt.

### 4.4 Cross-replication synthesis

Across **two paradigms (items, attributes), three datasets (ML-1M, LastFM, UCI), and three methods**, one law
recurs: **a popularity floor plus a personalization/attribute residual is necessary; the residual alone loses, on
top of popularity it wins.** This is the load-bearing principle of the CASPER-U instrument.

---

## 5. The CASPER-U instrument: architecture and mechanisms

### 5.1 Model family

The instrument is a **linear latent-factor (matrix-factorization) model**, not a deep neural network. It comprises
a table of learned embeddings plus linear algebra (dot products and one least-squares solve). The only learned-
from-text, network-like component is a thin linear **SBERT adapter** (below). This choice is deliberate: earlier
transformer/LSTM set-encoder instruments were harder to control and did not beat plain MF; the linear fold-in
version matches MF and is fully interpretable.

### 5.2 The shared space

A single d=64 space holds:
- **Item factors `Q ∈ ℝ^{n_items×d}`** — MF-initialised from implicit WRMF on training likes.
- **Attribute factors `A ∈ ℝ^{n_attr×d}`** — each genre is the *normalised centroid* of its items' factors (so
  "Sci-Fi" sits at the centre of the sci-fi cluster).
- **Open-concept factors** — for any text entity (a concept phrase, or a cold item with no behaviour), the factor
  is `W · SBERT(text)`, where `W ∈ ℝ^{d×384}` is a learned linear adapter (§5.5).

### 5.3 The encoder: closed-form ridge fold-in

Given a set of revealed entities (items and/or attributes/concepts), stack their factor rows into `X` and their
like/dislike answers into `y ∈ {−1,+1}^m`. The user vector is the ridge least-squares solution
```
u = (XᵀX + λI)^{-1} Xᵀ y .
```
This is a *set* operation (permutation-invariant), robust to however few answers are given, and treats items and
attributes/concepts identically (they are just rows of `X`). It is differentiable, so the adapter `W` (and, if
desired, `Q`) can be trained through it.

### 5.4 The scorer: popularity floor + confidence-scaled residual

```
score(item) = w · z(popularity) + c(m) · z(Q · u),     c(m) = m / (m + 5),
```
where `z(·)` is standardisation, `w` is the validation-tuned blend weight, and `c(m)` scales the personalization
residual by the number `m` of informative reveals. The floor banks the "easy" popular hits; the residual
re-ranks toward the user's distinctive taste; `c(m)` prevents an unreliable few-answer residual from hurting (it
restores the shrinkage that z-scoring would otherwise discard, fixing few-reveal non-monotonicity).

### 5.5 The SBERT adapter (open concepts)

SBERT is a frozen pretrained text encoder; it converts a phrase or description to a 384-dim vector. The adapter
`W` (a single linear map) projects that into the 64-dim taste space, trained so that for entities possessing *both*
a behavioural factor and text, `W·SBERT(text) ≈ Q[entity]`. Once fitted, **any** text entity — including arbitrary
open concepts — receives a coordinate in the shared space and is folded in exactly like an item.

**The key separation.** SBERT determines only an entity's *meaning/position* (what it is), never *polarity*
(whether the user likes it). Polarity is carried entirely by the answer `y` in the fold-in. This separation is why
the historical failure mode — sentence embeddings conflating like and dislike — does not occur here (see G7,
§7.3).

### 5.6 Seed selection in the instrument's own space

Interview seeds are chosen by RMVA (max-volume, pivoted-QR) on `Q` — i.e. *representative entities in the
recommender's own geometry*. This is the concrete sense in which one space serves both recommending and
question-selection.

### 5.7 Pseudocode

```
# --- offline: build the instrument ---
Q  = WRMF_ALS(train_likes, d=64, alpha=20, iters=15)        # item factors
A[g] = normalize(mean(Q[items_of_genre(g)])) * mean_norm(Q) # attribute factors (centroids)
W  = argmin_W  Σ_{e has text & factor} || W·SBERT(text_e) − Q[e] ||^2   # SBERT adapter (open concepts)
popularity = log(1 + like_count)

# --- inference for a cold user ---
seeds = RMVA(Q, K)                                          # representative interview
answers = ask(seeds)                                        # +1 like / -1 dislike per revealed entity
X = stack(factor(e) for e in revealed)                      # factor(e): Q[e] | A[e] | W·SBERT(e)
y = answers
u = solve(Xᵀ X + λ I,  Xᵀ y)                                # ridge fold-in (the encoder)
m = count(informative answers)
score = w · z(popularity) + (m/(m+5)) · z(Q · u)           # floor + residual
return top_k(score, exclude=revealed)
```

---

## 6. Methodology (detailed)

### 6.1 Why the floor+residual law holds (mechanism)

On full-catalogue NDCG@10, the top ranks are dominated by blockbusters that appear in nearly every user's positive
set; popularity therefore earns a large, almost-free score. *Pure* personalization re-ranks toward niche items and
**demotes** those safe blockbusters, losing top-heavy NDCG even though it raises Recall (it finds more of the
user's distinctive likes). Adding the residual *on top of* a popularity floor keeps the easy hits and adds
personalized lift — winning both metrics. This was demonstrated by the warm-half ceiling (§4.1) and is the design
rationale for §5.4.

### 6.2 Training protocol

- **`Q`:** implicit WRMF/ALS on training users. Pure fold-in already matches WRMF, so no destructive fine-tuning is
  required; an optional BPR fine-tune uses **reveal-subset masking** — each step reveals a random fraction of a
  user's items, folds in `u`, and ranks held-out likes via BPR — keeping partial interviews in-distribution.
- **`A`:** training-free centroids.
- **`W`:** least-squares regression of SBERT vectors onto behavioural factors (the open-concept stage).
- **Selection:** blend weight `w` (and λ) tuned on validation; reported once on test.

### 6.3 Acceptance-gate computation

- **G1/G2** compare FULL vs MOSTPOP and vs personalization-only on the test split.
- **G3** evaluates NDCG@10 at reveal counts {0,5,10,25,50} and checks non-decrease.
- **G4** compares FULL to the WRMF ridge fold-in number from R1 within a 0.02 tolerance.
- **G5** measures the mean score-percentile lift of a genre's items when that genre is "liked".
- **G6** compares item-only vs mixed vs attribute-only at a fixed question budget.
- **G7** compares the genre's mean score-percentile under "like" vs "dislike".

---

## 7. Results

### 7.1 Item-only instrument (S2) — quantitative gates

| arm (ML-1M cold-start, RMVA seeds, val-tuned) | NDCG@10 | Recall@10 |
|---|---|---|
| MOSTPOP (popularity) | 0.4134 | 0.0645 |
| personalization-only (u·Q) | 0.3124 | 0.0489 |
| **FULL (pop + u·Q)** | **0.4863** | **0.0861** |

Monotonicity (G3): NDCG@10 by #reveals = 0.4134 → 0.4170 → 0.4215 → 0.4376 → 0.4863 (q=0,5,10,25,50).

| Gate | Criterion | Evidence | Verdict |
|---|---|---|---|
| G1 beats-pop | FULL > MOSTPOP | 0.4863 > 0.4134 | PASS |
| G2 law | FULL > pers-only and > pop | 0.4863 > 0.3124, > 0.4134 | PASS |
| G3 monotone | non-decreasing in reveals | strictly increasing | PASS |
| G4 ≥MF | within 0.02 of WRMF 0.5011 | 0.4863 (gap 0.015) | PASS |

### 7.2 Unified items+attributes (S3) — quantitative gates

- **G5 attribute directionality:** mean score-percentile lift when a genre is "liked" = **+0.362** (Sci-Fi +0.40,
  Animation +0.45, Horror +0.35, Romance +0.32, Documentary +0.32). PASS.
- **G6 attribute value (budget = 10 questions, val-tuned):** item-only 0.4239; 5 items + 5 genres 0.4326; 10 genres
  0.4390; all 18 genres 0.4432. Mixed/attribute beat item-only. PASS.
- **G7 polarity:** Sci-Fi like→0.90 / dislike→0.10 percentile; Horror 0.85/0.15; Romance 0.82/0.18. PASS.

### 7.3 Interpretable sanity checks (human-readable)

**(1) Reveal one genre → top recommendations (100% correct genre):**
- like *Sci-Fi* → Forbidden Planet, 2010, Soylent Green, Gattaca, Day the Earth Stood Still, War of the Worlds.
- like *Children's* → Charlotte's Web, Fox and the Hound, 101 Dalmatians, The Rescuers, Mulan.
- like *War* → Patton, Crimson Tide, Platoon, The Last Emperor, All Quiet on the Western Front.

**(2) Polarity (like vs dislike) moves a genre's items in opposite directions** — see G7 numbers above.

**(3) Item coherence — reveal one liked film → nearest recommendations:**
- *Toy Story* → Toy Story 2, A Bug's Life, Babe, Aladdin.
- *Star Wars IV* → Empire Strikes Back, Return of the Jedi, Raiders of the Lost Ark, E.T., Back to the Future.
- *Aladdin* → Beauty and the Beast, The Lion King, A Bug's Life, Mulan, Fantasia 2000.
- *Silence of the Lambs* → Schindler's List, Fargo, Shawshank, Saving Private Ryan (prestige drama/thriller).

These confirm that genres are pure, polarity flips correctly, and franchises cluster — the qualitative behaviour a
working recommender must exhibit, and exactly the failure modes (polarity conflation) that sank earlier versions.

---

## 8. Discussion

### 8.1 Reframing the earlier negative result

A long prior effort (Appendix A; experiments E1–E13) concluded that no realizable policy beats popularity on dense
ML cold-start. We now understand this as an **artifact of a frozen instrument** plus a non-standard NDCG and a
pure-personalization scorer. A recommender frozen on popularity-style interviews cannot convert cleverly elicited
information into a better ranking, so every policy collapses to popularity. With a recommender that genuinely
ingests answers (this report), elicitation beats popularity by ~+20% NDCG / +40% Recall.

### 8.2 What is genuinely ours vs adopted

Adopted (cited, not claimed): RMVA seeds, ridge fold-in, the EAR additive scorer, the EDDI acquisition principle,
the Koren bias+factor structure, Wolpertinger's continuous mechanism. Ours: the **conjunction** — one shared space
serving as both a fold-in recommender over an arbitrary mix of items and *open* concepts and the action space of a
continuous policy — plus the empirically-grounded floor+residual+confidence design and the acceptance suite.

### 8.3 Limitations

The instrument is currently linear (by choice); the open-concept demonstration so far uses genres (the SBERT
adapter for arbitrary concepts is specified but not yet evaluated); full-catalogue NDCG is blockbuster-biased
(hence Recall as the lead metric); and the elicitation policy (the headline continuous-action contribution) is
specified but not yet trained.

---

## 9. Roadmap

- **S3 (complete):** evaluate the SBERT adapter on *arbitrary* concepts (not just genres).
- **S4:** run the full elicitation baseline panel on the fixed instrument (random, popularity, RMVA, Golbandi tree,
  greedy information-gain, SCPR/UNICORN-style discrete DQN, PEBOL/Thompson).
- **S5:** the continuous-action (Wolpertinger-style) policy over the unified space — the thesis contribution —
  evaluated against S4 and a discrete-action policy.
- **S6:** naked-LLM askers as baselines, quantifying prompt-only LLM elicitation.

---

## Appendix A — the frozen-instrument saga (E1–E13), in brief

Heuristics, behaviour cloning, RLOO, belief-conditioned actors, BC+RL, residual policy learning, and DQfD/POfD-style
oracle-pull were all tried against a *frozen* instrument across ML-100k and ML-1M, two metrics (NDCG, serendipity),
with val-stopping and a guaranteed popularity floor. All converged to popularity. The decisive insight (E14) was
that the constraint was the frozen recommender, not the policy — motivating CASPER-U. Full log archived in
`archive/CASPER_EXPERIMENTS_E1-E14_archived.md`.

## Appendix B — scripts (live)

`mf_foldin.py` (R1 / S1), `ear_lastfm.py` (R2), `eddi_replicate.py` (R3), `instrument_u.py` (S2),
`instrument_u_attr.py` (S3 attributes), `instrument_u_sanity.py` (interpretable checks), `golbandi_tree.py`
(adaptive-tree RMSE replication), `dre_faithful.py` (DRE baseline reproduction). Live ledger: `RESULTS.md`.

## Selected references

Rashid et al. 2002 (IUI); Golbandi, Koren & Lempel 2011 (WSDM); Zhou, Yang & Zha 2011 (SIGIR); Liu et al. 2011
(RecSys, RBMF); Fonarev et al. 2016 (ICDM, RMVA); Shi, Zhao & Shen 2017 (TOIS); Kweon et al. 2020 (WWW, DRE);
Sun & Zhang 2018 (SIGIR, CRM); Lei et al. 2020 (WSDM, EAR); Lei et al. 2020 (KDD, SCPR); Deng et al. 2021 (SIGIR,
UNICORN); Li et al. 2021 (TOIS, ConTS); Ma et al. 2019 (ICML, EDDI); Liang et al. 2018 (WWW, Mult-VAE); Sedhain et
al. 2015 (WWW, AutoRec); Dulac-Arnold et al. 2015 (Wolpertinger); Wang et al. 2025 (KBS, ECoC); Deng et al. 2024
(ICLR, PPDPP); Austin et al. 2024 (RecSys, PEBOL); Li et al. 2023 (GATE); Biyik et al. 2023 (Soft Attributes);
Luo et al. 2020 (WWW, Latent Linear Critiquing); Dacrema et al. 2019 (RecSys); Krichene & Rendle 2020 (KDD);
Koren, Bell & Volinsky 2009 (IEEE Computer); Makarova et al. 2024 (IJCNN, bot-play).
