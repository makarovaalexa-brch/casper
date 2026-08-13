# Our position against the modern conversational-recommender literature

*Written 2026-08-12 in answer to the author's question, verbatim: "the whole new deep neural, llm family
etc, are we still missing the strongest baselines? are we comparing wrong? … but should we borrow ideas
at least? i am a bit scared to be alone in this?" and the follow-up: "based on that survey, write up our
position in a very clear detailed way, but also make a clear note which methods to try and part-replicate
and borrow for our crown jewel, the parametric policy."*

**Evidence and per-system verification:** `external_literature/findings/modern_crs_positioning.md`.
**Feeds:** PAPER_B_TODO **1.21** (positioning + audit rows) and **1.16** (the parametric policy).
**This document is prose for the chapter's Positioning section. It does not touch `chapterB_v2.tex`.**

**How to read it.** The **lead finding** is the one thing to read if you read nothing else — it is a
concession, not a win, and it is the single item most likely to be raised by a reviewer who knows this
literature. **Part I** (§1–§5) is the position: our cell, then each neighbouring family in turn, then
the residual exposure, the contradictions, and a run/don't-run recommendation. **Part II** is the borrow
list for the parametric policy, ranked, each item with a mandatory "what must NOT come with it" clause,
plus a table of rejected mechanisms and the reason each was rejected. **Part III** documents how hard we
looked and what we could not check, so every negative in Part I can be judged rather than trusted.

Verification marks are carried through from the source files: **[V]** read from primary full text,
**[S]** abstract/index/third-party only, **[A]** absence at a stated search depth — which is evidence
about our search, not about the world.

---

## ★ Lead finding — the answer to "has anyone used our comparator?" is *yes, once, in 2011, and adaptivity won*

The comparator this chapter builds — a **fixed question order constructed by maximising the metric the
paper reports** — is not unprecedented. It exists exactly once in the literature we can find, and it is
the ancestor of our own greedy:

**GreedyExtend** (Golbandi, Koren & Lempel, CIKM 2010) builds a static seed set by incrementally adding
the item that most reduces error *with the recommender refit on seed-restricted data*:
`i = argmin_{i'∉k} F([k, i'])`, where `F` is the **RMSE** of a factorised item-item model. That is a
wrapper around a held-out evaluation — structurally our greedy, with RMSE where we put NDCG@10. And
RMSE **was** the reported metric of its era, so GreedyExtend is a genuinely metric-optimised static
order. *(Mechanism reconstructed at [A/S]: the CIKM'10 full text is closed on every route tried; the
reconstruction comes from the same authors' WSDM'11 §3.1.1 [V], the Yahoo patent US8510248B2 [V], and
RMVA's description [V].)*

The following year, the same authors' **adaptive ternary interview tree** (WSDM 2011) beat it, and beat
it convincingly: a depth-6 tree reaches RMSE 0.97172 after **six** questions, where GreedyExtend needs
**more than twenty** shown items to reach 0.971, and Var and HELF need more than thirty
[V, via WSDM'11 Fig. 3].

**This matters in three ways and the chapter must state all three.**

1. **We cannot claim nobody ever compared adaptivity against a metric-optimised static order.** They did,
   in the founding paper of the line we are extending. Any "first" phrased that way is false and a
   reviewer who knows Golbandi will kill it.
2. **The claim that survives is sharper and is about the field's evaluation practice, not about us.**
   In 2010–11 the split criterion (within-node squared loss) essentially *was* the reported metric
   (RMSE), so the adaptive method and its static comparator were optimising the same thing and the
   comparison was fair. The field then moved its reported metric to top-*k* ranking — and **kept the
   rating-era criteria**. Nobody rebuilt the static comparator on the new metric. Our audit of
   twenty-one distinct selection rules (`BASELINE_OBJECTIVE_AUDIT.md`) finds exactly one that optimises
   the reported quantity, and it is ours. So the correct sentence is: *since the field's reported metric
   moved from rating error to truncated, discounted top-k ranking, no adaptive selector has been tested
   against a static order built by maximising that metric.*
3. **It predicts our own result, and it did.** When we rebuilt our static comparator properly under
   policy B (workstream A), Golbandi's tail-at-eight win over the previous static order
   (+0.0068 [+0.0039, +0.0098]) **evaporated** to −0.00159 [−0.00489, +0.00167], ns
   (`BATTERY_TABLE_POLICYB_ROWS.md`). Comparator strength is exactly the axis the 2011 result says it is.

Everything else below is the survey that establishes the second sentence.

---

## 1. What our cell is, stated precisely

Six conditions, all of which must hold simultaneously. The value of the survey is that no other paper
satisfies more than four of them.

1. **The selection criterion is the reported metric.** Each candidate question is scored by folding its
   answers through a frozen, certified recommender and measuring **cohort-mean NDCG@10 on held-out
   targets**. It is *truncated* (top-10) and *position-discounted*. Both properties are load-bearing:
   an untruncated, undiscounted pairwise loss collapses analytically to an A-optimality trace criterion
   (Sepliarskaia et al., RecSys 2018, Theorem 1, via Anava et al., WWW 2015) — NDCG@10 does not collapse,
   which is why it cannot be replaced by a variance surrogate.
   ⚠ **Scope this correctly.** A truncated, position-discounted ranking metric *has* been used as an
   acquisition function — in IR active learning, at @10, in SIGIR 2012 (§2.6). What is unoccupied is its
   use as the criterion for a **question asked of a user**, and as an **expected improvement from
   acquiring** rather than an expected regret under current uncertainty.
2. **No ground truth is read at selection time.** The criterion is a cohort-level quantity computed on a
   disjoint validation cohort; it never reads the *evaluated* user's labels. (This is precisely where
   UpsRec, the only published rule that names NDCG as its selection objective, fails — see §2.4.)
3. **Answers are non-circular.** Item answers come from the user's real held-out ratings; concept answers
   come from behavioural attribute affinity (NPMI watch-lift plus a shrunk residual rating), computed
   from consumption logs and **independent of the recommender's embedding**. No LLM, no target item, no
   simulator sharing the recommender's latent geometry.
4. **Questions are charged.** A question the user cannot answer consumes a turn and returns nothing.
   ~14% of our cohort answer nothing at all across sixteen questions. This is a first-class part of the
   contract, and it is the property that most of the modern literature has structurally deleted.
5. **The catalogue is fully ranked.** 18,359 items, no negative sampling, no target-seeded candidate
   pool, no pre-filtering. Reported on **full and long-tail NDCG@10** under the Liang
   strong-generalisation split (140,768 train / 10,000 validation / 10,000 test users, seed 98765).
6. **The comparator is strong.** The static order against which every adaptive arm is measured is built
   by the same greedy maximisation of the same metric on the same cohort. This is the condition the
   literature does not meet, and it is the reason our Act 1 result exists.

---

## 2. The neighbouring families, one at a time

### 2.1 The interview-tree / rating-elicitation lineage — our own line, and the strongest prior art

**Shared:** the object. A cold user, a budget of closed questions, a tree or a fixed order, an explicit
unanswerable branch, and a wrapper-style greedy search over candidates.

**Golbandi, Koren & Lempel (CIKM 2010, WSDM 2011).** The ternary Lovers/Haters/Unknowns tree with
`Err_t(i) = e²(t_L) + e²(t_H) + e²(t_U)`, argmin over all items at every node [V]. Two things must be
conceded to it explicitly rather than discovered by a reviewer. First, *"optimise the task loss at the
split, not an information heuristic"* is **Koren's 2011 principle, not ours** — verbatim, they "deviate
from common error metrics typical to classification tree construction (like entropy or Gini impurity),
but rather rely on the squared loss metric" [V]. Second, Koren wrote our contribution into his own
future work: *"we would like to experiment with other cost functions, especially ones related to the
quality of the top-K item ranking"* **[S — this quote is carried second-hand and is still owed a check
against the PDF before it is printed; it is load-bearing for the "the field named its own gap" line]**.
Our delta is the metric, the honesty of measuring it rather than approximating it, and the comparator —
never the principle.

Golbandi is also, by its own admission, an *instrument* rather than a recommender: *"a decision-tree
recommender delivers a quite high RMSE. This makes decision-trees undesirable as a traditional
recommendation system"* [V]. That is exactly our framing of the tree as an existence proof rather than
the method.

**Hu & Yu (RecSys 2013), "Interview Process Learning for Top-N Recommendation" (EIP).** The nearest prior
construction, and the one we replicate and ablate (TODO 1.12 / plan B1). A ternary interview tree in the
fMF lineage; the split is chosen by optimising a general loss which the paper states "can be square loss,
logistic loss, **pairwise rank loss**, etc." [V]. So **we do not claim to be first to admit a ranking
objective into an interview tree's split.** Four separations, all re-verified from the full text:
- *Nothing they optimise or report is truncated or position-discounted.* The framework's ranking option
  is a smooth pairwise surrogate over all pairs; the loss their ranking experiments **actually ran** is a
  pointwise **logistic** loss ("since fMF cannot handle the logistic loss", §3.2.1 [V]) — pairwise rank
  loss appears exactly once, in a list attached to Eq. (1), and is never instantiated, tuned, ablated or
  evaluated; and the reported metric is **untruncated, undiscounted MAP** (Eq. 13) [V].
- *They optimise a second-order approximation of a training loss over the tree fit, not the value of
  asking.* Their split score is a Taylor expansion with the Hessian replaced by its diagonal, of how far
  the training loss on a node's own users would fall — used, in their words, "only to find the interview
  question" [V]. Ours folds answers through a frozen recommender and ranks held-out targets.
- *Ten sampled negatives per positive, in training **and** test data* [V], against our full catalogue.
  Sampled-negative ranking metrics are known to reorder methods (Krichene & Rendle, KDD 2020).
- *Their static comparator is top-k popularity* — "FixQ … select top-k popular items as static interview
  questions" [V]. Their other baselines are Warm-MF (an upper bound, not a cold-start method) and fMF.
  **Nothing in that paper tests an adaptive selector against a metric-optimised static order.**

Two further facts from the re-read, recorded because they bear on how much weight the paper can carry:
Warm-MF **beats** EIP on two of three datasets, and on the third the text's claim is contradicted by the
paper's own Figure 2(c) [V]; and their tree stops paying at depth 6 on Yahoo by their own admission
("the MAP begins to decrease, which means our model becomes overfitting") [V]. Their budgets are 3–10;
ours is 16.

**And the idea went nowhere.** A citation sweep run for this survey found **seven citing works in
thirteen years** — OpenAlex and Semantic Scholar independently return the *same* seven (2026-08-12;
Crossref's 3 is a subset artefact; Google Scholar could not be used, its records were mismatched and
must not be quoted). **Not one is an interview-tree method paper**, and **nobody has ever instantiated
the pairwise-rank-loss variant.** Even the nearest neighbour among the seven — Sepliarskaia's SPQ —
reaches the lineage through Golbandi rather than through Hu & Yu's general-loss equation. Four of the
seven were verified from abstract or full text; the three unverified are a TV-domain literature review, a
cross-site user-modelling paper and a software-crowdsourcing paper, none topically an interview method.
This matters for the chapter in two ways: it is a *fair* way to say the ranking-loss-in-a-tree idea is
prior art that was never taken up, and it is the third independent instance of the same dormancy pattern
we have measured elsewhere (UpsRec: published, cited twenty times, adopted by nobody; DRE/PERE/Greedy
SLIM likewise). **A published idea in this subfield does not get picked up — which is an argument for
running the ablation ourselves rather than citing a number.**

**The FacT (Tao, Jia, Wang & Wang, SIGIR 2019) must also be conceded.** It was an unverified lead in our
records and has now been read: its split criterion **does** contain a pairwise **BPR** ranking term
alongside the fit loss and an L2 penalty, over three children (left / right / **unknown**), and it is
**explicitly framed as an interview** — "the rules learnt in the user tree naturally serve as a set of
interview questions to solicit user preferences when a new user comes to the system", with nodes asking
"How do you like this [feature]?". It reports **NDCG@10/20/50/100** — reported, never optimised. Its
successor **FacT-CRS (CIKM 2022)** carries the same BPR term into a conversational tree and is evaluated
on the target-item CRS protocol (SR@T, Average Turns). Both compute the ranking term on **observed
training interactions**, undiscounted and untruncated. So: *a ranking loss inside an interview tree's
split is 2019 prior art, not ours.* ⚠ *One caveat we should not hide: The FacT gives no numeric value
for the weight `λ_b` on its ranking term in the method section, so how load-bearing that term is remains
unverified. Both papers were read through a fetch summariser over ar5iv HTML, not raw text; re-read
before printing any equation or dataset size.*

**fMF (Zhou, Yang & Zha, SIGIR 2011)**, **Sun et al. (WSDM 2013)** multi-question trees, **RBMF**
(Liu et al., RecSys 2011), **RMVA** (Fonarev et al., ICDM 2016), **DRE** (Kweon et al., WWW 2020),
**PERE** (UAI 2024), **SPQ** (Sepliarskaia et al., RecSys 2018). All confirmed surrogate-objective:
post-split squared error, soft-membership squared loss, maxvol/volume criteria, reconstruction MSE,
half-space-cut information gain, A-optimality. Two of these deserve individual sentences.

*SPQ is the most instructive near-miss in the whole literature.* It set out to do exactly what we do —
derive the selection rule from the loss the system is judged on — and *proved* the answer is A-optimality
`argmin_B tr[(V_B V_Bᵀ + εE)⁻¹]`, because the loss it chose (a flat pairwise squared error over all item
pairs) is a quadratic form in the parameter error. Row 5 of our own audit table (`aopt`) **is**
Sepliarskaia's rule, and the chapter should say so rather than double-count it. The sharpened statement
of our contribution follows directly: *the equivalence "task loss ⇒ A-optimal design" holds for a flat
pairwise loss and breaks for a rank-discounted, truncated one. The positional discount is not decoration;
it is the thing that makes the objective irreducible to a variance criterion.*

And SPQ contains the single most useful sentence found in this entire survey [V, §5.2.2]:
> *"if the method directly optimizes the loss function, then it works better than others. Moreover, the
> difference becomes noticeable if the questionnaire is long enough: SPQ and FG achieve better performance
> than PWDT, despite the fact that they are static, while PWDT is dynamic. A possible explanation is that
> PWDT optimizes a function that is different from the loss function, namely weighted generalized
> variance."*

That is Act 1's result, in a peer-reviewed venue, in 2018, used there to explain a *static* questionnaire
beating a *dynamic* one. **Our finding should be framed as a replication and generalisation of a known
result, not as a discovery.** That framing is both more defensible and, for a reviewer, more interesting.

*Greedy SLIM (Proissl, Vatić & Waldschmidt, arXiv:2406.06061, 2024) is the closest thing to a protocol
sibling that exists.* Greedy forward selection of a question order, prefix-optimal, one fit covering
every k; evaluated on **unfiltered ML-25M** (162,541 users / 59,047 items), **no item sampling**
("we do not apply any kind of item sampling to avoid the problems reported in Krichene and Rendle" [V]),
**NDCG@10 at k ∈ {5,10,15,20}**, and a long-tail split defined exactly as ours (Cremonesi 33%-of-ratings
short head) [V]. It is our protocol in all but two respects: its selection objective is a **SLIM
elastic-net squared loss** (NDCG@10 is reported, never optimised), and it is a CoRR-only preprint that
was never peer-reviewed (DBLP-verified). It therefore *strengthens* Act 1 rather than threatening it —
it is the nearest anyone has come to our comparator and it misses on exactly the axis we are testing.

Three things in that paper are gifts we should use. Its adaptive foil, QBandit (Christakopoulou et al.
KDD'16 over PureSVD), has **NDCG that decreases past k ≈ 10 on both full and tail** (ALL 0.3456 → 0.2916
from k=10 to k=20) [V] — an independent external replication of the bank-quality effect our own sweep
measures. Its authors report that *"For Golbandi et al. [2011] and Sepliarskaia et al. [2018] we,
unfortunately, obtained poor results. We therefore decided to not include them"* and *"We had to
implement all the baselines we wanted to consider on our own"* [V] — a citable statement of exactly the
hole our baseline battery fills. And its limitations paragraph **identifies unanswerability as the threat
to its own offline protocol** and goes to a user study to address it, without ever pricing it [V]. That
is the cleanest available motivation for our charged-question contract, written by someone else.

**Why the separation is principled, not convenient.** This family and we are measuring the same object
under the same information contract. The only difference is which quantity the selection step maximises.
That is the smallest possible difference, which is why the comparison is fair and why the result means
something. We are not excluding these systems; we run them.

### 2.2 The bandit and conversational-bandit line — the family whose policies genuinely port

**ICF (Zhao, Zhang & Wang, CIKM 2013)** is the structural ancestor of our architecture: PMF backbone,
item factors frozen at MAP, a Gaussian posterior over the user vector only, and four policies —
Thompson, LinUCB, GLM-UCB (identity and sigmoid links), ε-greedy [V]. We run its policies. Its metric,
however, is **cumulative** precision/recall/nDCG over the *recommended sequence*, with
`θ_hit = 1 if the rating is no less than 4, and 0 otherwise` [V]. That definition is worth quoting in the
chapter because of what it implies: **an item the user never saw scores 0, identically to an item the
user disliked.** No method in this lineage can price a burned question, because under its metric burning
a turn and being wrong are the same event. That is a clean, verifiable statement of why the
answerability construct cannot exist in this family, sourced from the founding paper's own metric.

**Conversational bandits — ConUCB (WWW 2020) → ConLinUCB-BS/-MCR (AAAI 2023) → Hier-UCB (2022) →
CLiSK/CLiME (KDD 2025) → FedConPE (2024)** — are, by our own criteria, the strongest modern family, and
the one a reviewer is most entitled to raise. They ask about a **key-term** (an attribute, a genre, a
tag) *or* about an item, in one algorithm, with an explicit model of how attribute feedback moves the
item-level estimate — **that is our item-vs-concept channel, formalised, with regret bounds, and with no
target item at all.** ConUCB's Eq. 4 states verbatim that a key-term's expected reward is the
membership-weighted average of its member items' rewards [V]: our concept-as-member-aggregate assumption,
published in 2020. ConLinUCB's one-step estimator puts a key-term answer into the *same normal equations*
as an item answer [V]: our signed concept fold, in bandit clothing. And CLiME (KDD 2025) runs on
**MovieLens-25M with genres as key-terms** [V], so this cannot be dismissed as a 2011 straw man.

**We ran them.** `conucb`, `conlinucb_mcr`, `conlinucb_bs`, `conlinucb_bs_static` and `clime` are rows in
our battery, ported policy-faithfully and re-scored under our contract. They lose
(`ACT2_CONCEPT_BANK_PV.md`), and two of them are reported **UNINFORMATIVE at k ≤ 2** rather than as
nulls, because ConUCB's conversation-frequency schedule `b(t) = 5⌊log t⌋` fires no key-term question at
all before turn 3 — a fact we state with the schedule attached rather than reporting a null on an
unfired mechanism.

**What separates them, and why it is principled.** Their metric is **cumulative regret over T rounds
against the best arm**, not post-interview held-out ranking; and their evaluation scale is truncated —
CLiME's "ML-25M" is **5,000 arms and 200 users at T = 6,000 rounds** [V]. Regret prices the probes
themselves; we price what the system does after the probing stops. Neither is wrong; they are different
quantities, and a number in one is not orderable against a number in the other. This is the sentence to
put in the chapter, along with the corollary: **because their numbers cannot be matched, the acceptance
gate for a ported policy is not "reproduce their published number" — it is (i) reproduce the selection
rule exactly as written, (ii) beat random and popularity on our metric, (iii) label the row "policy
ported, protocol replaced."** Say it before a reviewer says it for us.

One honest caveat about the whole bandit line: **COFIBA and CLUB use "cold start" to mean *early in the
stream*, not *a user with zero history***. The terminological trap is theirs, not ours, but we should
name it, because a reviewer from that community will otherwise think we ignored their work.

### 2.3 The deep-RL attribute-asking CRS family — the one the author is worried about

This is CRM (SIGIR 2018) → EAR (WSDM 2020) → SCPR (SIGIR 2020) → UNICORN (SIGIR 2021) → ConTS (TOIS
2021), plus a dense 2022–2026 successor layer: CRIF (SIGIR'22), MCMIPL (WWW'22), HutCRS (EMNLP'23),
MHCPL (SIGIR'23), DAHCR (IJCAI'23), CRSIRL (NeurIPS'23), CoCHPL (DASFAA'24), TCRS (CIKM'24), VPMCR
(TOIS 2025), SAPIENT (2025), MINICORN, MSMCR, FedCRS.

**What it shares with us.** It is adaptive; it asks about attributes as well as items; and — the point
that should be conceded generously — **it optimises the metric it reports.** The RL reward *is*
target-hitting and the metric *is* Success Rate. On the axis this chapter is about, this family is not
guilty of the objective mismatch we diagnose in the elicitation line. That must be said plainly, because
it is true and because saying it is what makes the rest of the paragraph credible.

**What separates it, and why the separation is principled.**

*First, the simulator.* The user's answer is a deterministic lookup on the ground-truth **target item's**
attribute set. The chain is verbatim and inherited, each paper citing the previous as justification:
CRM's AMT workers are "presented with the facet values of the target restaurant on the side, so that she
can correctly answer the questions"; EAR "treat[s] the v as the ground truth item to seek for and its
attributes P_v as the oracle set of attributes preferred by the user"; SCPR's user "will only confirm he
likes it if this attribute is included by item v", explicitly collapsing "does not care" into "reject";
UNICORN axiomatises that "the user preserves clear preferences towards **all** the attributes and items.
Thus, the user will respond accordingly"; ConTS gives positive feedback "only to item v and attributes
P_v in current session" — all [V].

**Be precise about what kind of circularity this is.** It is **label-based, not representational**: the
simulator answers from the target item's ground-truth attribute labels, not from the recommender's own
learned embedding. That is a different and in one respect milder defect than the geometric self-play our
own firewall was built against, and we must not conflate them. What disqualifies it for us is the
consequence rather than the mechanism: **every question is answerable by construction, so an
unanswerable question costs exactly zero.** The quantity this chapter measures — that ~14% of cold users
answer nothing, and that answering *at all* is worth roughly +0.25 against +0.048 for the best question
*choice* — cannot exist inside their protocol. ConTS's own future work concedes the gap: "we can consider
how to handle 'don't know' [or] 'don't care' responses" [V].

*Second, the candidate pool is seeded by the answer key.* The conversation is initialised with an
attribute `p₀` **of the target item**, and the candidate set is the items carrying it. MCMIPL: "𝒱ₚ₀ is
the item set in which all items are associated to attribute instance p₀ which initializes the
conversation session"; DAHCR: "Candidate item set V_cand is formed with the items associated with p₀";
CoCHPL and VPMCR the same — all [V]. So the reported top-10 is a re-rank *inside a pool the target's own
attribute created*. **And nobody reports the pool size**: MCMIPL and VPMCR both give **zero**
candidate-set-size statistics [V], VPMCR despite over-filtering being its entire motivation. We
therefore cannot even estimate the ratio to our 18,430-item catalogue. That absence is itself the
finding, and it is the fair way to say "these numbers are not orderable against ours" — not by asserting
their evaluation is invalid, but by observing that the quantity needed to compare is not published.

*Third, the metric is Success Rate and Average Turns.* And it is worth being exact here, because the
exactness helps us: **CRM, EAR, SCPR and ConTS report no rank-sensitive metric whatsoever.** The string
"hDCG" does not occur in EAR at all [V]; hDCG@(T,K) is introduced in **UNICORN** §5.2.3 and inherited by
its successors, and UNICORN explains why it is the metric it is: *"Since there is only one target item
for each session in MCR, we simply use hDCG@(T,K) for evaluation"* [V]. One labelled item per session
means an NDCG over held-out consumption is **not definable** in their protocol. **Not one paper in this
family reports a post-interview held-out ranking metric over a full, unsampled catalogue.** We tried to
falsify this and could not.

*One honest caveat on that last sentence, which must be in the chapter.* The **ReDial-style dialogue
branch** (GRSU, HyperCRS/HyFairCRS, DareCRS) *does* report Recall@k / NDCG@k / MRR@k over an item
vocabulary of a few thousand. That is closer to a ranking metric over an unsampled item set. It still
fails our definition on three counts — the label is the single item **named next in the dialogue**, i.e.
a known target, not held-out consumption; the metric is measured at a dialogue turn, not after a
controlled interview budget; and whether they rank the full vocabulary or a pre-filtered list **could
not be verified**. So the claim to make is the precise one: *no paper in this line reports a
post-interview held-out ranking metric over a full unsampled catalogue with no known target item.*

*Fourth, the catalogue is actively shrunk, and by the papers' own descriptions.* SCPR keeps only "the
items that directly link to all attributes in the updated [set]"; **UNICORN prunes the action space
itself** — *"Instead of enumerating the whole candidate item and attribute set, we adopt preference-based
item selection and weighted entropy-based attribute selection strategies to only consider potentially
important actions"* [V], top-`K_v` items and top-`K_p` attributes; HyperCRS (2025) does the same
"following Qian et al. (2023)". So the action space is negative-sampled as well as the candidate set.

*Fifth, the benchmark is saturated.* CoCHPL reports SR@15 = **0.991** on LastFM at 3.75 average turns;
DAHCR **0.992** on Yelp at 5.16 [V]. A protocol at its ceiling cannot adjudicate between methods. This
is a stronger and more useful sentence than "their numbers are high", and it explains why the successor
layer competes on turns rather than on success.

*Seventh — and this is the observation that most changes how the family should be read — the RL apparatus
does not appear to be doing the work.* **MINICORN** (arXiv:2206.14468) uses **no RL at all**:
Monte-Carlo-dropout epistemic uncertainty over attribute beliefs, ask the highest-uncertainty attribute,
one hand-written ask-vs-recommend rule. It reports beating SCPR by "67.40% and 13.39% in recommendation
success rate and conversation turns" on LastFM [V]. If a harmonic mean of two uncertainty heuristics
beats the RL line by that margin, then the contribution of the policy machinery over a good acquisition
rule is unestablished within the family's own protocol. This is the citation to use when a reviewer asks
why we did not build a deep RL policy for Act 1 — and it is also, quietly, an argument *for* our
workstream E, which is a policy trained to imitate a measured argmax rather than to maximise a
target-hitting reward.

*Eighth, and decisive for the comparator question: no paper in this family compares against a static
question order optimised for the metric it reports.* We went through the baseline tables one by one.
EAR's full baseline list is exactly three — Max Entropy, Abs Greedy, CRM — and it opens with the
disarming admission "As our multi-round conversational recommendation scenario is new, there are few
suitable baselines" [V]. SCPR adds EAR; UNICORN adds SCPR; HyperCRS in 2025 lists "AbsGreedy,
MaxEntropy, CRM, EAR, SCPR, UNICORN, MCMIPL, MHCPL, HutCRS" [V]. **The comparator set is the chain
itself plus two heuristics, and Max Entropy and Abs Greedy both re-derive the question each turn from
the live candidate set, so neither is even a fixed order.**

The single near-miss is **CRM**, and it is instructive rather than threatening. Sun & Zhang run
fixed-length variants and then *select among them by the reported metric* [V]: *"We name the baseline
'MaxEnt Full' for the one that asking all the facets and 'MaxEnt@K' for ones that asking exactly K
facets"* … *"the reward increases as [K] increases … it seems [MaxEnt Full] is the best choice, thus we
use it as the baseline for follow up experiments."* **That is metric-driven selection over a budget, not
over an order** — the facet at each turn is still chosen by max-entropy on the live belief. It is the
closest anything in this literature comes and it is not the object.

*Ninth: the 2025–2026 successor layer has not changed any of this.* **HyperCRS** (Findings of ACL 2025)
is a multi-grained hypergraph DQN reporting SR@t / AT / hDCG@(15,10) against that same baseline chain —
**the MCR ruler is unchanged in 2025**. The most methodologically distinct new acquisition rule found is
**CCR** ("A Causal-Based Attribute Selection Strategy for CRS", TKDE 2025), which selects "the attribute
with the highest **Average Treatment Effect** per round" with stratification to de-confound — genuinely
novel, and **[S], abstract only: the IEEE full text is paywalled with no OA copy, so its metric,
simulator and catalogue are unverified.** Almost all other 2025–26 CRS output has migrated to the
ReDial/LLM-dialogue branch, which is a different task. *(One withdrawal to note and never cite:
**CT-CRS, arXiv:2508.13209, was withdrawn by its authors** in September 2025 — "The tables … need to be
modified" — and search engines still surface its numbers.)*

**And the fairness clause, which must be in the chapter.** The honest sentence is not "their evaluation
is invalid" but "their evaluation prices a different quantity, and its own community has published the
leakage critique." Two sources carry that clause and both should be cited: **PEPPER** (Kim et al.,
EMNLP 2025 Findings, arXiv:2411.16160) — "Stop Playing the Guessing Game!" — which shows that even
*attribute-only* priming "can still shortcut the recommendation process by implicitly narrowing the
candidate space" [V]; and **Interplay** (Ramos et al., ECIR 2026, arXiv:2603.18573), which states that
existing work "feed[s] target items to simulators in advance. This causes: (1) data leakage (the
simulator knows the answer beforehand) and (2) rigid dialogues" [V]. Note also, and say it, that
**none of the three published critiques comes from inside the attribute-asking line itself** — they all
come from the LLM/dialogue branch. That is a reportable observation about the family, not a slur.

**One successor is worth naming as the closest anyone gets to our firewall:** **TCRS** (CIKM 2024) builds
its conversational user model from **logged data only**, trains a PPO policy against it, and then
evaluates against an **independent black-box simulator** [V]. The train/eval separation is exactly the
right instinct. The target item, however, stays.

### 2.4 The LLM elicitation line — PEBOL and after

**What it shares with us.** Several of these systems have a real acquisition rule, and — the finding that
matters — **the good ones put the LLM outside the selection loop**. PEBOL's stage 1 is a closed-form
bandit rule over independent Beta posteriors (Thompson / UCB / max-variance / greedy); the LLM only
extracts a ≤3-word aspect and phrases the yes/no query [V]. OPEN's acquisition is expected information
gain over a Bradley–Terry particle filter; the LM supplies only the verbalisation and featurisation maps
[V]. BED-LLM maintains an explicit probabilistic model and asks the argmax-EIG question [V]. So this
family is not "LLMs instead of principles"; the principle is intact and the LLM is a phrasing layer.
That is also why an LLM-free ablation of PEBOL's acquisition is legitimate under our contract, and it is
the one row from this family we should run.

**What separates it, and why the separation is principled.** The measurement contract, in every case:
- **PEBOL** (RecSys 2024): MRR@10 over a **100-item** catalogue — and the cap is a pure baseline artefact,
  present only "for fair comparison to the MonoLLM baseline", which must fit every item description into
  a GPT-3.5 context window [V]. **Our catalogue is 184× larger.** Its users are 100 GPT-3.5 simulations
  each of which "likes a single item" and is handed that item's description; the NLI scorer then entails
  against **the same description string** — the simulated user and the measurement instrument share
  their text.
- **CLARINET**: 1,877 documents, retrieval success, and a GPT-3 simulator that "has access to the true
  item" [V].
- **OPEN**: 30 New York Times articles, TIDA, pairwise queries — but with ~40 **real** users per
  condition, which is more than most of this literature manages.
- **GATE** (ICLR 2025): **388 real preregistered participants**, no learned selection, no catalogue, and
  the metric is p(correct) integrated against elicitation *time*. It threatens a framing claim about
  question format; it does not threaten a selection claim.
- **BED-LLM** (ICLR 2026): six turns, 200 profiles, and the film-task metric is the **mean rating of ten
  recommendations** — not a ranking metric.
- **Montazeralghaem et al. (2025)**: free-form generation scored by **BLEU/ROUGE against a reconstructed
  profile** [V]. This is the cleanest available evidence that the family does not measure recommendation
  quality at all.
- **COPE** (2026): selects one of **three strategy types**, not a question; Recall@k against a target
  named in the dialogue.

**Two of these are gifts rather than threats, and the chapter should treat them as such.**
CLARINET's selection objective — *simulate the answer and measure the rank improvement of the true item
under the current retriever* — **is our greedy prefix-optimal objective in different notation**, and it
**beats EIG and KL acquisition by 17% relative** on retrieval success [V]. That is independent published
pre-support for our prefix-optimal-greedy result; its disqualifier is the target-biased simulator, not
the criterion. And COPE reports, from an entirely different method, dataset and metric, that
*"attribute-based strategies are more effective for early-stage preference exploration, whereas
item-based strategies become more suitable once users have formed clearer and more specific
preferences"* [V] — an independent, 2026, dialogue-side statement of our concept-beats-item-at-small-
budget finding. Convergent evidence from a different contract is worth more than another row in our own
table. It must be cited, not discovered by a reviewer, and our differentiator stated: the non-circular
behavioural protocol, the quantitative budget sweep, and the answerability *mechanism* (COPE attributes
the effect to preference specificity and explicitly not to answerability).

**And the strongest single fact about this family: there is no benchmark.** PEBOL runs on 100 items,
CLARINET on 1,877, OPEN on 30, GATE on no catalogue at all. A reviewer cannot demand "the standard
benchmark" because none exists — and PEBOL's own authors make the argument for us, declining ReDial
because it is "made up of pre-recorded conversation histories and cannot be used to evaluate active,
cold-start NL-PE systems" [V]. What a reviewer *can* demand is that we justify our protocol, and the
contract argument (Liang strong generalisation, full and tail NDCG@10, behavioural answers, twenty-one
named selection rules ported) is an argument we are better placed to win than any paper in this table.

**The one genuinely adjacent LLM-line item on our own dataset:** ICER / MD-DICER (Ryu, Hsu, Chow,
Ghavamzadeh & Boutilier, GENNEXT@SIGIR'25, arXiv:2510.02331) generates 100K synthetic CRS dialogues on
**MovieLens-25M** with EVOI-style item *and* attribute queries — and still declines to report
post-interview catalogue NDCG, evaluating instead on dialogue quality and NDCG **on item pairs**, with a
learned response model that has access to ground-truth CF user embeddings [V]. A Google-scale team, on
our exact dataset, with our action space, did not report our number. That is worth one sentence.

### 2.5 Meta-learning cold start — a category error, stated as such

MeLU, MAMO, TaNP, CDRNP and the 2023–26 successors (ColdNAS, MetaKG, M2EU, MetaHIN, PAML) are
**answer consumers**, not selectors: the support set is *given*, by recency or random partition, and
nothing is asked. TaNP's `N_S ∈ {10,15,20}` sweep looks like a budget sweep but is a given support size,
so TaNP is only a fair comparison against a **random-bank** arm. Exactly one paper in the line selects —
**MeLU** (KDD 2019), with a single **static, global** popularity × gradient-norm ranking, validated on
**39 human subjects in the treatment arm** — and in doing so it independently reproduced our
answerability collapse: *"only one-quarter of our evidence candidates were selected"* against "one-half
of the popularity-based" ones [V]. It reported this as an aside and shipped. That is a citable human
precedent for the concept-vs-item argument and should appear in the chapter next to our own answer-rate
numbers.

### 2.6 ★ Active learning for ranking, in IR — the family that owns half our criterion, and the one we must concede

This family is not a conversational recommender at all, and it is exactly the one an IR-literate reviewer
will raise. **It has to be in the chapter, because it falsifies the loosest version of our claim.**

**ELO-DCG (Long, Chapelle, Zhang, Chang, Zheng & Tseng, SIGIR 2010) [V, full text read].** Its loss is
`ℓ(π,Y) = max_{π'} DCG(π',Y) − DCG(π,Y)` with the standard `1/log₂(1+rank)` discount, and its acquisition
is the expected value of that loss under a bootstrap-ensemble posterior over the labels — argmax. So a
**position-discounted ranking quantity has been inside an acquisition function since 2010**. Two
properties keep it out of our cell: the DCG in the criterion **sums over all documents of the query**
(the @10 truncation appears only in the evaluation), and the quantity is **regret, not improvement** — it
measures how much the current model stands to lose if it is wrong, and *never models the effect of
acquiring the label*: no retraining, no counterfactual, no post-acquisition expectation. And what it
selects are queries and documents to **label for training a ranker**; no user is asked anything.

It also, however, states our thesis, in IR, eight years before Sepliarskaia [V]:
> *"ELO-DCG optimizes the expected DCG loss that is directly related to the objective function DCG-10
> used to evaluate the ranking quality; on the other hand, VR reduces the score variance that is not
> directly related to the objective function. In fact, VR performs even worse than random document
> selection when the size of the selected example is small."*

**Cite that sentence beside Sepliarskaia's.** Two independent fields, twelve years apart, measuring that
optimising the surrogate loses to optimising the metric, is a far stronger opening for Act 2 than either
alone.

**Bilgic & Bennett (SIGIR 2012) is the real falsifier and we should name it ourselves [V, full text].**
Their Eq. 1 is the same ELO regret shape with the measure `M` **instantiated as DCG@10 or NDCG@10**;
their arms are literally named `dcgELOq` and `ndcgELOq`. **So a truncated *and* position-discounted
ranking metric has been an acquisition function, at @10, in SIGIR, in 2012.** Any sentence of the form
"nobody has used a truncated discounted ranking metric as a selection criterion" is false and must be
deleted from the chapter. Four separations survive, and they are clean:
1. **improvement, not regret** — we score what the metric *becomes after the answer arrives*; they score
   what the current model could lose;
2. **a held-out cohort-mean target metric**, not posterior regret on the candidate itself;
3. **a question asked of a user**, not a training label for a ranker;
4. **the full 18,359-item catalogue**, not a per-query candidate list.

Two smaller neighbours complete the picture. **Donmez & Carbonell (ICML 2008)** selects by *estimated
loss differential* where the loss is the **ranker's own training objective** (RankSVM hinge, RankBoost
exponential) — untruncated, undiscounted, pairwise, NDCG only in evaluation [V]. **Carterette, Allan &
Sitaraman (SIGIR 2006)** greedily selects *judgments* to reach 95% confidence about which of two systems
is better; AP is rank-weighted but untruncated, and the objective is **metric estimation, not model
improvement** [V].

**And one more concession, from a different corner of IR: greedy forward selection with an NDCG wrapper
criterion is an ordinary technique in learning-to-rank feature selection.** "For each state, i.e. a
subset of features … we train the model learnt with only the features that are included in the feature
subset at hand. We then measure the performance of the learnt model using NDCG and MAP" (arXiv:2310.13269,
LambdaMART on LETOR 3.0 and MSLR-WEB10K) [V] — with forward feature selection named as a standard wrapper
search. That is our greedy machine with "feature" swapped for "question". It is not an acquisition (there
is no unknown being acquired, no user, no answer model, no cohort, no prefix over a question sequence),
but it means **the technique is not the contribution.** Claim the conjunction, never the technique — the
same demotion GreedyExtend and Greedy SLIM already force.

**One negative that survived a deliberate attempt to break it:** no Bayesian-experimental-design,
Bayesian-optimisation or ranking-and-selection acquisition we could find carries a **positional
discount**. Top-k Ranking Bayesian Optimization (AAAI 2021) is a false friend — its acquisition is
multinomial predictive entropy search, and "top-k ranking" describes the *observation model*, not the
utility [S]. And smooth NDCG surrogates (SoftRank, ApproxNDCG, LambdaRank, NeuralNDCG) appear only as
training losses, never as acquisitions [A]. Both negatives are arXiv-abstract-index evidence with no
free-text web search available, so they are **[A], absence at a stated depth** — state the instrument
alongside them and never write "the literature contains none".

---

## 3. The honest residual exposure

Stated plainly rather than defended, because a reviewer will find it anyway.

**The comparator is our own construction, and no published number validates it.** The whole chapter turns
on a static order built by greedily maximising cohort-mean held-out NDCG@10. Nobody has published that
object, so there is no external number to snap it against — HARD RULE 2's usual gate does not apply, and
we should say so rather than pretend it does. Three mitigations, all of which are true and checkable:

1. **It beats every published selector we ran**, across thirty comparisons, against all three rebuilt
   policy-B comparators (30/30 each, `BATTERY_TABLE_POLICYB_ROWS.md`). A comparator that beats
   twenty-one published rules is not weak by construction.
2. **The recipe is fully specified and the search is deterministic and reproducible** — greedy forward
   selection with lazy CELF bounds over a stated 1,000-candidate pool, on a stated cohort, with the
   heaps persisted. Anyone can rebuild it.
3. **We measured what comparator weakness costs and reported it against ourselves.** The policy-A→B
   rebuild moved the argmax at *every* position (0 of 16 positions shared with the order it replaced),
   cut the conditioning gain at K=16 from +0.01371 to +0.01035 on full and from +0.02092 to +0.00705 on
   tail, and **retracted Golbandi's tail-at-eight win entirely**. A chapter that criticises weak
   comparators and then demonstrates the effect on its own headline numbers is in a much stronger
   position than one that merely asserts the principle.

**Second exposure: the size of the effect.** The conditioning gain is ~10% of the interview's own lift,
and it is *smaller* than the gain from changing the question bank with no conditioning at all
(+0.01430 [+0.01124, +0.01739] at equal budget vs +0.01035). "Adaptivity pays" is the claim;
"adaptivity is large" is not, and never travels without its size qualifier.

**Third exposure: the validation→test shrinkage is real and asymmetric.** 2.70× on full and 1.50× on
tail. The earlier "tail does not shrink" figure was a comparator artefact and is retracted.

**Fourth exposure: we do not run the deep-RL family at all.** §5 states exactly what running one would
cost and where the circularity would have to be broken. The answer is uncomfortable and should be given
directly: breaking it deletes the method.

---

## 4. Where a neighbouring family's result contradicts ours

Three, and none is fatal, but all three must be in the chapter rather than discovered.

**(1) Golbandi WSDM'11 beats a metric-aligned static comparator.** See the lead finding. An adaptive
interview tree beat GreedyExtend, a static order built by maximising the era's reported metric, by a
wide margin (6 questions vs >20). Our Act 1 says published adaptive selectors *lose* to a
metric-optimised static order. **The reconciliation — and it is the chapter's central diagnosis, not a
patch — is that in 2011 the adaptive method's split criterion and the reported metric were the same
quantity, and today they are not.** Golbandi's variance split essentially *was* RMSE. When we restore
that alignment by putting the reported metric back into the split, adaptivity wins again, from K=4 on
both axes. The 2011 result is therefore evidence *for* our diagnosis, not against it — but only if we
present it ourselves.

**(2) Greedy SLIM's crossover runs the other way at very short budgets.** Their own summary: "QBandit
seems to perform slightly better than QGSLIM if the questionnaire consists of very few questions …
QGSLIM is able to profit from every additional question" [V]. So the *adaptive* bandit wins at k=5 on
the tail and the *static* questionnaire wins from k=10 — a published, budget-dependent crossover on our
dataset with our tail definition. Our own tree is exactly 0.0 at K=1 by construction and separates from
K=2–4 onward, so the shapes are compatible; but we should not write "adaptive loses at short budgets and
wins at long" as though it were a law, because a 2024 preprint on our dataset measures the opposite
ordering between a different pair of methods.

**(3) Gharahighehi et al. (2025) find genres win early and items win from iteration 13** in an
attribute-aware interview tree — directionally the same shape as our channel result, but on a different
dataset, in RMSE space, attributed to clustering efficiency rather than answerability. It is months old
and it is the closest live threat to the *channel* claim. Our differentiator has to be the answerability
mechanism and the tail-NDCG ruler, not the direction of the effect.

*(A fourth, weaker one, recorded for completeness: ICF's own founding table (CIKM'13) contains `AL` and
`Interview` baselines, and ICF beats `Interview` by only 1.6% (45.745 → 46.47) [V]. **The identity of
those two baselines is UNVERIFIED** — if `Interview` is a Golbandi-style tree, that row matters. This is
a standing verification debt.)*

---

## 5. Should we RUN any of them? — the recommendation

**Do not run the deep-RL attribute-asking family. Here is precisely why, in terms of what would have to
be rebuilt.** To put CRM/EAR/SCPR/UNICORN on our contract you must break four things at once:

1. **The answerer.** Replace the oracle attribute lookup on the target item with our behavioural answer
   model. Cost: small — our answer model already exists. Consequence: their state encoding, which is
   binary desire/disprefer per attribute, must gain a third "no answer" state that no published version
   has.
2. **The candidate pool.** Delete the `p₀` target-seeded initialisation and rank the full 18,359-item
   catalogue. Cost: moderate. Consequence: their graph-reasoning machinery (SCPR's paths, UNICORN's
   unified graph) is built around a shrinking candidate set; with the full catalogue there is nothing to
   shrink and the mechanism's stated purpose disappears.
3. **The metric.** Replace SR@15 / Average Turns with post-interview full and tail NDCG@10 at fixed
   k ∈ {1,2,4,8,16}. Cost: small — our harness does this. Consequence: "average turns" is undefined under
   a fixed budget, so half of every reported result in that literature has no analogue here.
4. **The reward — and this is where it fails.** Their RL reward *is* target-hitting. There is no target
   under our contract. Replacing it with measured held-out NDCG@10 is not a modification of their method;
   **it is our method (workstream E) with their state encoder.** The policy is the only thing that
   survives, and the reward that trained it no longer exists.

**That is the concrete answer to "where would the circularity have to be broken": it is in the reward,
and breaking it deletes the method.** Estimated cost of doing it anyway: 40–60 agent-hours per system
plus a dead PyTorch stack (CRIF's repo pins torch 1.4.0, last commit 2022-09-28 [V]), to produce a row
that is honestly labelled "policy ported, reward replaced, protocol replaced" — i.e. a row whose content
is our own reward. **Cite, do not run.** This is TODO 1.21's ruling and this survey confirms it.

**What is worth running, ranked by protection-per-hour:**

| # | What | Why | Cost |
|---|---|---|---|
| 1 | **UpsRec (WWW'23) as a declared privileged-oracle ceiling row** | The only published rule that names NDCG as its selection objective. It reads the user's own validation labels at ask time, so it cannot be run honestly — but run *as a labelled oracle* it makes our headroom visible. Mechanism is one refiner pass plus an NDCG recompute per candidate: **cheaper than our own criterion**. Must be labelled an oracle in the table caption. | **~4–6 h** (my estimate; the mechanism is already in our harness in another form) |
| 2 | **Greedy SLIM's QGSLIM as a static item bar, with QBandit as its adaptive foil** | The only published protocol sibling — our dataset, our tail definition, NDCG@10, no sampling. Its QBandit-declines-past-k=10 curve is external corroboration of our bank result. Also settles a live design question: they **exclude asked items from the recommendation pool** and attribute QBandit's decline to it — we should check what `run_interview_table.py` does, because it changes the sign of the popularity arm's late-budget behaviour. | **12–20 h** (store estimate) + ~5–6 h compute for k ≤ 16 |
| 3 | **FacT-CRS's BPR-in-the-split as a fourth criterion arm in the B1 ablation** | We are already running variance / pairwise / logistic / measured-NDCG splits in one harness (`split_criteria.py`). Adding a BPR split answers "did anyone try a ranking criterion in a tree?" with a run rather than a citation, at near-zero marginal cost. ⚠ **Gated on actually reading arXiv:2208.14614 and arXiv:1906.02037** — our record of its criterion is [S], second-hand. | **~6–8 h** on top of B1 |
| 4 | **PEBOL's acquisition, LLM-free, as one ablation row** | Legitimate precisely because PEBOL's stage 1 is a closed-form Beta-posterior bandit rule with the LLM outside the loop. Gives the LLM family a row without an LLM API call. | **~4–6 h** |
| 5 | *(optional)* **HutCRS-style coarse→fine asking over the tag taxonomy** | The only baseline in the deep-RL family whose stated motivation — users cannot answer attribute questions they have no knowledge of — is *our* motivation, and it adds a "don't know" feedback class. **But** Act 4's routing claim is settled negative by 1.20, so this cannot be sold as a curriculum result. | **20–30 h** |
| — | **Do NOT run** | PTS, COFIBA/CLUB, factorUCB, full IGCF, CRIF, DAHCR, MHCPL, FedCRS, SAPIENT's MCTS (unless a reviewer challenges prefix-optimality), anything LLM-warm-started, anything requiring an unfrozen tower or a social graph. | — |

---

# PART II — THE BORROW LIST FOR THE PARAMETRIC POLICY (TODO 1.16)

*"our crown jewel" — the author. This section is meant to be actionable, not a reading list.*

Staged design being served (plan §6): **E1** imitation of the NDCG tree (labels = the tree's cohort-level
exact-NDCG argmax at each node, plus the leaf greedy tails); **E2** extend past the tree's depth using
kernel-weighted cohort labels at states the tree cannot reach; **E3** RLOO finetune with exact NDCG
reward, only if E2's supervised ceiling binds.

Every entry carries a mandatory **"must NOT come with it"** clause. The firewall is: **no
model-generated answers, no simulator-derived reward, no training on the test cohort, no reward that is
a function of a target item.**

## B.1 — Take these first (do them in E1)

### ★ 1. Dueling decomposition of the per-candidate score — from MCMIPL (WWW 2022)

**Mechanism.** MCMIPL's Q-network factorises as `Q = max_k (f_V(q_k) + f_A(q_k, a))` [V] — a
state-value term plus a per-action advantage term, scored over a candidate set.

**Why it is the highest-value borrow in this list.** The measured reason the June 2026 policy-gradient
attempts collapsed to a best fixed order was **reward SNR: the routing signal is 2.1% of cross-user
reward variance.** A dueling head is the textbook fix for exactly that pathology: the shared state-value
term absorbs the 98% of variance that is "this user is easy/hard", leaving the advantage head to learn
the 2% that is "this question suits *this* state". It is a head change, not an architecture change, and
it composes with the equivariant per-candidate scorer already specified in E1.

**Part-replicate or borrow conceptually?** Borrow the *decomposition* only. Do not port their network.

**Cost.** ~4–8 h on top of E1's scorer.

**Must NOT come with it:** their reward (target-hitting), their action space (ask-vs-recommend — our
budget is fixed at k, so there is no recommend action to decide), and their multi-choice question format
unless separately motivated.

### ★ 2. Distil an expensive exact criterion into a one-forward-pass policy — from CLARINET (2024)

**Mechanism.** CLARINET generates 50 candidate questions, scores each by *simulating the answer and
measuring the rank improvement of the true item*, keeps only those that put the target in the top-10 or
improve its rank by ≥10 as **training data**, then finetunes a small model (Flan-T5-base, Fusion-in-
Decoder) to emit the rank-maximising question directly — so inference is **one forward pass** [V].

**Why it matters.** This is E1's exact architecture, published, in a different domain, and it **beats
EIG and KL acquisition by 17% relative** [V]. It converts E1 from "an idea we had" into "the move that
worked when someone else made it". It also validates the specific choice to distil a *rank-gain* label
rather than an information-theoretic one — which is the same choice our own
`prefix-optimal-greedy-is-the-right-object` result made, and the same one Sepliarskaia's 2018 sentence
predicts.

**Part-replicate?** No code exists. Borrow the *training recipe shape*: expensive exact scoring offline →
filtered label set → supervised distillation → cheap inference. We already have the expensive scorer
(the NDCG tree and its CELF heap sidecars) and the labels are already banked.

**Cost.** ~0 extra — it is a citation plus a design confirmation for E1.

**Must NOT come with it:** their user simulator (a GPT-3 model that "has access to the true item" [V]),
their target-conditioned rank objective, and their label filter (keeping only questions that help
*this* user's target is a per-user oracle; our labels are cohort-level argmaxes on a validation cohort).

### ★ 3. Amortised experimental design as the framing — from the DAD/iDAD line and BED-LLM's successor

**Mechanism.** Sequential Bayesian experimental design has an established sub-literature whose entire
content is "replace per-step optimisation of the design with a **policy trained offline that maps
history to the next design**" — Foster, Ivanova, Malik & Rainforth (DAD, ICML 2021), iDAD (NeurIPS 2021),
Step-DAD, and decision-aware amortised BED (Huang et al., NeurIPS 2024) [store level: S/V-abstract].
The 2026 LLM-side instance RL-finetunes the questioner with an EIG reward so the design is amortised
into the weights (arXiv:2607.03426, FoGen@ICML 2026 workshop, non-archival) [A].

**Why it matters.** It gives 1.16 a *name and a home*. "We amortise the design" is a recognised move with
a citation trail; "we trained a network to pick questions" is not. It also pre-empts the reviewer
question "why not just run the greedy at test time?" — the amortisation literature has already answered
it (per-step optimisation does not scale with depth and cannot reach states the offline search never
visited), which is exactly E2's argument.

**Part-replicate?** No. Adopt the vocabulary and cite.

**Cost.** ~0, writing only.

**Must NOT come with it:** the EIG reward. Our reward is the measured metric, and the whole point of the
chapter is that information-theoretic surrogates lose to it. Borrowing the amortisation without the
objective is the correct half.

### ★ 4. The adaptivity diagnostic — from Pep (arXiv:2602.15012)

**Mechanism.** *When two users give different answers to the same question, how often does the system's
follow-up differ?* Pep reports **39–62%** for its method against **0–28%** for RL baselines [A, abstract
only].

**Why it matters.** It is a cheap, contract-free, immediately-runnable measure of whether a policy is
*genuinely adaptive* or has silently collapsed to a fixed order — which is the exact failure mode the
June 2026 policy-gradient runs exhibited and the exact failure mode E1 could reproduce without anyone
noticing, because a policy that imitates the tree perfectly on tree-visited states still has to *diverge*
off them. **Recommendation: make it a gate on E1 and E2, alongside the "reproduce the tree's test gain
within CI" gate.** We already have the analogous measurement machinery from D2 (1.20), which measured
leaf-tail overlap and matched-chance controls on the tree.

**Cost.** ~2–4 h, reusing D2's code.

**Must NOT come with it:** Pep's domain (reasoning tasks, no item catalogue), its numbers (abstract-level
only — do not quote them as a bar we must clear), and any inference that a high divergence rate implies
a *useful* divergence. D2 already taught us that lesson: our tree **diverges without routing** — 80/105
leaf pairs share no free pick, and answer-congruence is a null (p=0.82 matched-availability, p=0.61
shuffle). The diagnostic detects collapse; it does not license a routing claim.

### ★ 5. The candidate-scoring amortisation trick — from Hu & Yu (RecSys 2013), as a PROPOSAL only

**Mechanism.** Their split score expands the child loss to second order about `Δu = 0` and replaces the
Hessian by its diagonal, giving `L̃(0) − ½ gᵀD⁻¹g` [V]. Because `Σ_C L̃(0)` is candidate-independent and
`g`, `H` are sums over a child's users of quantities that **do not depend on the candidate**, they are
computed once per node and **every candidate is then a pair of masked sums** — structurally an XGBoost
split gain. Our exact objective pays a tower fold per candidate (10.3 h for one tree); theirs pays none.
They are explicit that the approximation only *chooses* the question: "since there are approximations in
the derivations, we only use it to find the interview question. After the question is found, we use
coordinate descent to obtain optimal Δu" [V].

**Why it matters for 1.16.** E2's whole problem is *label cost at depth*: to train the policy on states
the tree never reached, we need many exact-NDCG argmaxes at many states, and each one currently costs a
fold per candidate. The Hu & Yu trick is the published route to that scale — **but only as a proposal
distribution.** Use the cheap Taylor gain to rank the 1,000 candidates, take the top-m, and **re-score
those m exactly**. The label remains the exact argmax over the shortlisted set; the surrogate never
enters the objective.

**And we have the gate for it already.** C1 (TODO 1.11) measures criterion-vs-measured-NDCG rank
correlation and regret over all 1,000 candidates at every node of the real tree, read out of the banked
CELF heap sidecars for free. **That measurement tells us directly whether a surrogate proposal is safe
and at what m.** Do not implement the proposal shortlist before C1 lands; if the rank correlation is
poor, the shortlist is a HARD RULE 1 truncation with no justification.

**Part-replicate?** Yes — and we already are: `src/instrument/split_criteria.py` implements their
gradients and diagonal Hessians, verified against central finite differences, with the Taylor gain
verified against a brute-force minimisation.

**Cost.** ~6–10 h to wire the shortlist path, *after* C1.

**Must NOT come with it:** their objective (a training loss, not the reported metric), their sampled
negatives (10 per positive, in training data too), and any use of the Taylor gain as the *label*. And
note HARD RULE 1: shortlisting the candidate pool is a truncation and needs the author's sign-off with
the C1 correlation in hand, not a hand-wave.

## B.2 — Optional, take if E1/E2 stall

### ★ 5b. Test an UNNORMALISED DCG@10 selection criterion against our normalised one — from Bilgic & Bennett (SIGIR 2012)

**Mechanism and finding.** Bilgic & Bennett run the same ELO acquisition with `M = DCG@10` and with
`M = NDCG@10`, and report that **using the DCG loss for *selection* gives better performance whether DCG
or NDCG is the final evaluation metric** [V]. Their explanation: NDCG's per-query normalisation pulls
selection toward queries that have few relevant documents, because a small absolute gain there looks
large after dividing by a small ideal DCG.

**Why this is directly actionable for us and nobody has connected it.** Our criterion is **cohort-mean
NDCG@10** — normalised *per user*. The same pathology applies verbatim with "user" for "query": a user
with two held-out targets has a tiny IDCG, so a question that moves one of them one position looks
enormous, and the cohort mean will chase such users. **An unnormalised cohort-mean DCG@10 criterion is a
one-line change to the scorer and would test this directly.** It also bears on a live puzzle: 4/15 of
our tree's leaves are negative on test and ~14% of users answer nothing — if the criterion is being
dragged by low-IDCG users, that is a candidate mechanism.

**Part-replicate?** No — take the *finding* and run the ablation on our own criterion. **We would still
report NDCG@10**; only the selection criterion changes, which is exactly the single-factor design B1
already uses.

**Cost.** ~4–6 h to add a `dcg_raw` criterion to the greedy/tree scorer and rebuild one static order at
k=16 for comparison; the CELF machinery and cohorts are unchanged. **Cheap, and it is a real scientific
question rather than a table row.**

**Must NOT come with it:** their regret framing (theirs is a loss under label uncertainty, ours is a
measured post-answer value), their query/document label-acquisition setting, and any change to the
*reported* metric — the ruler is fixed (HARD RULE 8) and stays full-and-tail NDCG@10.

### 6. Representative-item propagation for the concept fold — from Hier-UCB (arXiv:2209.06129)

Published claim, from their own survey plus real data, that key-term rewards "are mainly affected by
rewards of **representative items**", not by a uniform member average [V abstract]. That directly
contradicts ConUCB Eq. 4 (uniform weighted average) and is a third answer against our whitened centroid.
Testable, and it is about the *concept fold* rather than the policy, so it is orthogonal to E1/E2.
**Cost ~6–10 h. Must NOT come with it:** the ask-vs-recommend decision (no analogue at fixed k), or
their regret framing.

### 7. Non-myopic lookahead at fixed k — from SAPIENT (arXiv:2410.09580v3)

MCTS over action types with N=20 simulations, maintained repo [V]. It is the only real alternative to a
greedy prefix at fixed budget, and it would answer a prefix-optimality challenge with a run. **But** we
already measured that endpoint polish is +0.002 at q16 and −0.029 at q2, i.e. greedy is prefix-optimal
in the regime that matters. **Only build this if a reviewer challenges prefix-optimality. Cost 25–35 h.
Must NOT come with it:** their simulator (a template answerer that accepts only lists containing the
target [V]) or their reward constants.

### 8. Bank ranking by discrimination × popularity — from MeLU (KDD 2019)

`score(item) = norm(mean local-update gradient magnitude) × norm(popularity)`, top-k, k=20 [V]. A named,
published instance of *discrimination × popularity* — the closest published relative of our `expinfo`
family, reimplementable on the certified tower in an afternoon with no retraining. **Cite it primarily
for the answerability corroboration** (one-quarter vs one-half selection rates in humans). **Must NOT
come with it:** any presentation of MeLU as adaptive — its selection is global, static and one-shot, and
its own paper says so.

### 9. a-stratification — from computerised adaptive testing (Chang & Ying 1999)

Administer **low-discrimination items early and high-discrimination items late**, because early ability
estimates are imprecise and spending your most discriminating items against a bad estimate wastes them.
This is a direct, published, 27-year-old prediction that **our greedy-by-discrimination order is
suboptimal at small k** — and given the certified finding that bank ranking decides which channel wins,
it is worth one run. **Cost 4–6 h (a re-ordering of an existing bank). Must NOT come with it:** exposure
control, which is about test security and item-bank economics and is **not** an answerability mechanism —
misrepresenting it as one is the kind of error a psychometrics-literate reviewer catches instantly.

## B.3 — Rejected, with the reason (the author asked to see these)

| Mechanism | Source | Why rejected |
|---|---|---|
| **Hindsight reward shaping** `r̃ = r + γw(s_{t+1},v) − w(s_t,v)`, `w = log(ρ(s_t,v)+1)` | CRSIRL (NeurIPS 2023) [V] | The shaping potential `w` is **an explicit function of the target item `v`**. Under our contract there is no target. This is the cleanest exhibit in the literature that the line optimises directly against the answer key — cite it as such, never implement it. |
| **Inverse RL to learn the reward** | CRIF (SIGIR 2022) [A/S] | The learned reward is *defined by* target-hitting; porting it means re-deriving the premise. Also a dead stack (torch 1.4.0, last commit 2022-09-28 [V]). |
| **Learned world-model / environment simulator** | FeedRec (KDD'19) S-Network; Pseudo Dyna-Q (WSDM'20) | Falls squarely under the simulator disqualification. Also: **Pseudo Dyna-Q could not be verified to exist on arXiv and its DOI 403'd — do not cite a DOI for it.** |
| **Online learning of item factors** | factorUCB (AAAI'17), PTS (NeurIPS'15), IGCF | Our certification freezes the tower. PTS's extra power *is* learning `V` online; with `V` frozen it collapses to ICF Thompson, which we already run. |
| **Cross-user streaming / clustering** | COFIBA, CLUB | Their mechanism is sharing statistics across users who already have interactions. At t=0 a cold user is a singleton component and COFIBA **degenerates to LinUCB**. |
| **Social-graph views** | MHCPL (SIGIR'23) | We do not have a social graph. |
| **Federated axis** | FedCRS, FedConPE | Orthogonal to every question this chapter asks. |
| **UpsRec's greedy-NDCG selector as a deployable policy** | WWW 2023 [V] | It reads the user's held-out validation labels at ask time and scores every candidate under an assumed YES. **Run it as a declared oracle ceiling or not at all.** |
| **Boroomand & Wright's expected-improvement-in-ranking-utility acquisition** | arXiv:2508.14911 [V] | Genuinely the right idea and honestly done (expectation under the model's own Plackett–Luce posterior, no label leakage) — but the query space is **pairs**: at 18,359 items that is ~1.7×10⁸ candidate queries, each needing a Monte-Carlo permutation estimate **and a model retrain per hypothetical outcome**. Infeasible at our scale. Cite as the concurrent mechanism; do not port. |
| **CAT exposure control** | Sympson–Hetter, randomesque, Stocking–Lewis | Population-level item-exposure caps for test security. Not answerability. See B.2 §9. |
| **Causal / ATE attribute selection** | CCR, TKDE 2025 [S] | Genuinely the most novel new acquisition rule in the 2025 sweep — "the attribute with the highest **Average Treatment Effect** per round", with stratification to de-confound. **Not rejected on merit: rejected because we cannot read it.** IEEE paywall, no OA copy per OpenAlex, so its metric, simulator and catalogue are all unverified. **Chase it before dismissing it** — an ATE-based selector is the one idea in this family that is not obviously a surrogate. |
| **Any LLM-in-the-selection-loop design** | GATE, Montazeralghaem, RSO, PPDPP | Project rule: no LLM API calls. Also unnecessary — the good LLM elicitation systems (PEBOL, OPEN, BED-LLM) already put the acquisition *outside* the LLM, which is the part worth having. |

## B.4 — Part-replicate vs cite: the summary the author asked for

**PART-REPLICATE (build enough of it to learn from or compare against):**
- **Hu & Yu 2013** — in progress (B1/1.12): faithful snap on their protocol + the four-criterion split
  ablation in our own harness. Harness built and committed (`7d3e302`); grow/measure run queued.
- **Greedy SLIM (QGSLIM + QBandit)** — 12–20 h. The only published protocol sibling.
- **UpsRec** — 4–6 h, as a labelled privileged oracle.
- **FacT-CRS's BPR split** — 6–8 h as a fourth arm in B1, **gated on reading the paper** (our record of
  its criterion is second-hand).
- **PEBOL's Beta-posterior acquisition, LLM-free** — 4–6 h.
- *(already done)* ConUCB / ConLinUCB-BS / -MCR / CLiME / ICF's four policies — ported and run.

**CITE ONLY:** CRM, EAR, SCPR, UNICORN, ConTS and every 2022–26 successor; CLARINET; OPEN; BED-LLM;
GATE; COPE; Montazeralghaem; ICER/MD-DICER; Bıyık et al.; MeLU/MAMO/TaNP/CDRNP; PTS; COFIBA/CLUB;
factorUCB; IGCF; MINICORN; SAPIENT; TCRS; PEPPER; Interplay; Anava 2015; SPQ (already in the battery as
`aopt` — say so rather than double-counting).

---

---

# PART III — HOW HARD WE LOOKED, AND WHAT WE COULD NOT CHECK

## C.1 The falsification attempt, and its instrument

The author's question was whether we are missing the strongest baselines. The honest way to answer it is
to show the search, not to assert the conclusion. Four independent passes were run on 2026-08-12, and
**WebSearch was unavailable for all of them** (session budget exhausted before the first query), so
everything is structured-API evidence — stronger per source, weaker in recall over venue-only
publications. The cells worth putting in the paper:

| Instrument | Query | Result |
|---|---|---|
| arXiv abstract index, all years | `abs:"preference elicitation" AND abs:"NDCG"` | **1 hit** — UpsRec (2023), already audited |
| arXiv, all years | `abs:"expected NDCG"` · `"expected NDCG" "question selection"` | **0** · **0** |
| arXiv, all years | `"experimental design" AND "NDCG"` · `"acquisition function" AND "NDCG"` · `"expected improvement" AND "NDCG"` · `"best arm identification" AND "NDCG"` | **0 each** |
| DBLP | `cold-start interview recommendation` | **0** — the Golbandi "interview" vocabulary is dead in indexed titles (partly a query-form artefact: DBLP ANDs prefixes) |
| arXiv, 2026-02→08, cs.IR | `abs:"cold-start" AND abs:"interactive"` (301 hits, first 35 triaged) | **all item-side cold start** — multimodal embeddings, graphs, diffusion, generative retrieval, semantic IDs. **Zero user-interview papers.** |
| OpenAlex | forward citations of EAR / SCPR / UNICORN from 2025-01 | successor layer enumerated; nothing new occupies the cell |
| Semantic Scholar | `/citations` on PEBOL (arXiv:2405.00981) | **26**; not one 2025–26 citer reports NDCG over a full catalogue |
| OpenAlex ∩ Semantic Scholar | citations of Hu & Yu (RecSys 2013) | **7**, identical sets; none is an interview-tree method paper |

**The structural finding behind the negative, and it is more interesting than the negative itself: the
field has bifurcated.** Elicitation work went LLM-in-the-loop and now evaluates with target-item accuracy
or user studies; cold-start work went item-side and evaluates with offline ranking. **Nobody is doing
budgeted user-side questioning evaluated by held-out full-catalogue NDCG.** The two cells to quote, each
with its instrument named, are the arXiv `"preference elicitation" AND "NDCG"` count (**one**, from 2023,
already audited) and the DBLP `cold-start interview recommendation` count (**zero**).

**Nearest miss in the 2025–26 sweep, and worth a sentence in related work:** Gharahighehi, Nakano, Yang,
Cu & Vens, "Pairwise and Attribute-Aware Decision Tree-Based Preference Elicitation for Cold-Start
Recommendation" (arXiv:2510.27342, CoRR only). Golbandi-lineage tree; asks **attributes (genres) as well
as items**; answers from real logs. **It fails only because the reported metric is RMSE** — splits pick
"the item with the lowest prediction error", so there is no ranking metric at all and the full-catalogue
question is moot. Yahoo! Music, 1,701 users / 4,793 artists / 218 genres. *(Read through a fetch
summariser; re-read `arxiv.org/html/2510.27342v1` directly before citing its numbers.)*

---

## C.2 Verification status and standing debts

Everything above is sourced from `external_literature/findings/` files whose per-claim verification
marks are reproduced here. **Closed this pass:** ELO-DCG's `[HIGH]` debt (read in full — its integrand is
expected DCG **regret**, exactly as we had guessed, so the existing safe-phrasing paragraph stands but
must now also name Bilgic & Bennett); The FacT's UNVERIFIED-LEAD status (read; it *is* a ranking-split
interview tree); FacT-CRS's second-hand criterion (confirmed); Top-k Ranking BO (a false friend);
Hu & Yu's citation graph (7, identical across two indexes).

**What is NOT verified, and why:**

- **Koren's WSDM'11 future-work quote** ("cost functions … related to the quality of the top-K item
  ranking") is **[S]**, carried second-hand from an internal raw source. It is load-bearing for the
  "the field named its own gap" line. **Read it off the PDF before it is printed.**
- **GreedyExtend's own full text (CIKM 2010)** is **[A/S]** — closed on every route tried (ACM 403,
  S2 `openAccessPdf: CLOSED`). The mechanism is reconstructed from WSDM'11 §3.1.1 [V], patent
  US8510248B2 [V] and RMVA's description [V]. The lead finding of this document rests on that
  reconstruction plus WSDM'11 Fig. 3 [V]. **This should be upgraded via institutional access before the
  lead finding goes into the chapter as stated.**
- **`λ_b`, the weight on The FacT's BPR term** — not stated numerically in the method section. If it is
  tiny, its "ranking split" is nominal. **Open, and it matters for how much we must concede.**
- **The FacT and FacT-CRS equations and dataset sizes** were read through a fetch summariser over ar5iv
  HTML, not raw text (the PDFs would not extract). Structure trustworthy; **re-read before printing any
  equation.** Same caveat for arXiv:2510.27342.
- **ICF's `AL` and `Interview` baselines** — identity UNVERIFIED; matters because ICF beats `Interview`
  by only 1.6%.
- **Q&R (KDD 2018) offline protocol and metrics** — PDF would not extract.
- **RLT4Rec** datasets, metrics, session length — HTML 404s, PDF would not extract.
- **"Towards Efficient Conversational Recommendations: EVOI Meets Bandit Learning", WWW 2025.**
  ⚠ **Authorship correction: it is NOT Wang et al. It is Zhuohua Li, Maoli Liu, Xiangxiang Dai and
  John C. S. Lui (CUHK)** [V, OpenAlex + S2; DBLP `conf/www/0001LDL25`]. It is **gold open access under
  CC-BY and still unreachable** — ACM DL 403'd landing page and PDF, Unpaywall lists only the ACM copy,
  no arXiv entry, author pages 404. From the abstract only [S]: a **gradient-based EVOI** acquisition
  replacing conventional EVOI's Bayesian updates with SGD, plus "smoothed key term contexts", giving
  ConTS-EVOI and ConUCB-EVOI; **the only quantity named is regret.** Rejected on metric, but the
  rejection rests on an abstract. Cheapest route to close it: an institutional proxy or an email to the
  first author.
- **CCR (TKDE 2025)**, the causal/ATE attribute selector — abstract only, IEEE paywall, no OA copy.
- **Geng, Liu, Qin & Li, "Feature Selection for Ranking", SIGIR 2007** — the key paper for the
  wrapper/NDCG concession. ACM 403, DuckDuckGo CAPTCHA, S2 429.
- **CAT/IRT negative** — arXiv-only instrument; *Psychometrika*, *Applied Psychological Measurement* and
  *JEM* are not indexed there. **[A], not [V].** Do not upgrade it.
- **RecSys 2025/2026 were not enumerated venue-by-venue** — reached by keyword only. A "we swept RecSys
  exhaustively" claim would need a separate proceedings walk.
- **Pep's numbers** are abstract-level only.
- **The K values in CRM's MaxEnt@K sweep** — rendered as MathML, lost on extraction. Not guessed.

**Bibliographic corrections this survey forces:**
1. **SCPR is KDD 2020, not SIGIR 2020** — DOI 10.1145/3394486.3403258 is the KDD '20 proceedings
   [V via OpenAlex]. The bib key `lei2020interactive` carries the wrong venue.
2. **PEBOL's DOI in our store is wrong.** `10.1145/3640457.3688151` resolves to a conversion-rate
   prediction paper [V]. The correct DOI is **`10.1145/3640457.3688142`** [V, S2 + DBLP
   `conf/recsys/AustinKTS24`].
3. **hDCG must not be attributed to CRM/EAR/SCPR/ConTS** — it is UNICORN-onward.
4. **"The simulator is built from the recommender's embeddings" must not be written** of any of the five
   base systems — the circularity is label-based.

**Do-not-cite list (binding):** CT-CRS (arXiv:2508.13209, **withdrawn by its authors**, numbers still
surfaced by search engines); any Pseudo Dyna-Q DOI (not on arXiv, DOI 403); any Google Scholar citation
count from this run (records were mismatched); the "+9.7 NDCG@10" figure attributed to arXiv:2605.14236
(does not appear in the abstract and could not be verified).
