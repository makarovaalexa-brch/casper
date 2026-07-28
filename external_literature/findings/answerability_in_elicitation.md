# Findings: Answerability in Elicitation — who models it, who reports it, who measures it

*Can the user actually answer the question? Audit of the cold-start-interview and conversational-recommendation
literature on four axes (A models / B reports / C quantifies / D compares channels).*
Compiled 2026-07-28 from PRIMARY TEXT (PDF/HTML extractions) except where marked. Companion to
`answerability_and_channels.md` (which covers the channel *result*); this file covers the *literature position*.
Bib-keys traceable to `../INDEX.md`; new sources flagged at the bottom for INDEX+bib insertion.

**Our measurement being positioned:** on ML-25M with a fixed static ITEM question bank an average user answers
**2.18/8 (27%)** and **3.63/16 (23%)**; with an out-of-catalogue CONCEPT bank, **6.02/8 (75%)** and **11.04/16 (69%)**.

---

## What we concluded

1. **The hypothesis is CONFIRMED for the modern line and REFUTED for the classic line.** The claim "the literature
   ignores answerability" is false and must never be written. Golbandi (WSDM'11) makes *unknown* a first-class branch;
   Rashid (IUI'02, SIGKDD Expl.'08) makes answerability the **explicit motivating tension** of the whole item-selection
   problem, reports answer-rate curves, and runs live-user studies on it. The defensible claim is much narrower and
   much sharper (see 7–8 below).

2. **The classic line does A and B properly, and does a WEAK form of C.** This is the correction to our prior
   assumption. Two concrete pre-emptions of "nobody quantifies":
   - **Rashid 2002 measures both axes on the same runs and argues the causal link.** Offline (Fig. 2/3): "*The poor
     performance of Pure Entropy in both metrics is directly related... Since popularity directly relates to the chance
     that a new user has seen a movie, this strategy presents movies that users are less likely to have seen, resulting
     in poor performance in the movies-seen metric. Moreover, with fewer rated movies to base predictions on, the MAE
     for Pure Entropy also suffered.*" [VERIFIED full text, §Results]
   - **Golbandi 2011 puts a NUMBER on the value of an unanswered question.** Blending weights optimised by exhaustive
     search: `w_L = 5, w_H = 1, w_U = 0.02` — "*Thus, an unknown vote was found to be relatively insignificant.*"
     [VERIFIED full text, §7 Eq. 8]. An "unknown" answer is worth ~1/250 of a "like" — a direct, citable
     quantification of the cost of unanswerability, expressed as prediction weight rather than as attributed gain.

   Also note the **answer-rate reporting REGRESSED over time**: Rashid 2008 measured familiarity as a first-class
   metric; RBMF 2011 and RMVA 2016 report a weak set-level Coverage; DRE (WWW 2020) reports nothing at all and drops
   the coverage analysis its own two baselines had. That trajectory is a clean, citable narrative for the paper.

   What NOBODY does is the **decomposition**: hold informativeness fixed, vary answerability, and report how much of
   the achievable interview gain is answerability vs informativeness. Rashid's Item-Item result is in fact a *published
   dissociation* proving the two are separable — it "*trounced the competition*" on movies-seen yet lost on MAE
   ("*It was hard to believe that the Random strategy could get an error rate with eight ratings as training data
   comparable to the item-item personalized strategy with 57 ratings*") — but they never turn that into an attribution.

3. **The single most dangerous paper for us is Sun et al., WSDM 2013 — by FMF's own authors, and its entire
   contribution is an answerability fix.** §1, verbatim [VERIFIED full text, re-checked by me]: "*most users do not
   have the luxury to go through the entire item set… the constructed decision trees are often extremely unbalanced
   with **the unknown branch capturing more than 80% of the users at each split** (Figure 1, upper). Many users have to
   repeatedly select unknown multiple times before locating any item they know about. In this case, not only did the
   system gain little valuable information on users' interests, but the users also easily get bored and may opt out of
   the system prematurely.*" Their fix is explicitly an answer-rate mechanism: "*we advocate designs focusing on asking
   multiple questions at each trial. **Displaying multiple questions on one screen increases the chance that a user
   knows at least one of them** and thus allowing for solicitation of more valuable information.*" We must engage this
   head-on. Our surviving delta: they raise the answer rate *within the item channel* by widening the screen, and never
   decompose the resulting gain; they do not compare channels; and 80%-unknown-per-split is a per-node statistic, not a
   per-user interview answer rate.

4. **A formal P(user can answer) model exists and is 24 years old.** Rashid 2002 §Balanced strategies: "*Rank items by
   the product of popularity and entropy. Entropy is the number of bits of information if the user rates the item, and
   **popularity is the probability that this user will rate this item**. Using Bayes' theorem in this way assumes that
   popularity and entropy are [independent].*" [VERIFIED full text]. Rashid 2008 goes further and folds
   unanswerability into the information measure itself — **Entropy0** treats "*the missing evaluations as a separate
   category of evaluation, for example, a rating value of 0*" so "*the frequency of the new rating category (0) of an
   item indicates how (un)popular the item is*" [VERIFIED full text, §3.4]; **HELF** = harmonic mean of entropy and
   log rating-frequency (Eq. 2). We must never present a popularity/answerability-weighted acquisition score as new.

5. **The field's own vocabulary is "ratable" / "acquisition probability" / "familiar" — "answerability" is our
   coinage.** Say so explicitly and bridge to their terms, or reviewers will read the new word as an attempt to hide
   the prior art. Elahi/Ricci/Rubens (CSR 2016) devote a whole taxonomic category to it: "***Acquisition probability
   based.** The strategies in this group focus on improving the performance of the system by maximizing the probability
   that the selected items are familiar to the user, **hence, are rateable**.*" §7.1 states the failure mode we are
   re-measuring: "*strategies that only focus on the informativeness of the items **may fail to actually acquire
   ratings, by selecting obscure items that users do not know and cannot rate**.*" And their Table 1 "# Rating" column
   marks FMF, Golbandi's tree, TreeU, GreedyExtend and RBMF as "–" (not reported) — **a neutral survey's own evidence
   that the MF-interview line does not report answer rates.**

6. **The modern attribute/LLM CRS line ASSUMES universal answerability, and the mechanism is the user simulator.**
   This is the strongest, most quotable thing in the audit. The assumption is *inherited*, not independently made:
   CRM (SIGIR'18) → EAR (WSDM'20) → SCPR / ConTS (2020–21) → UNICORN (SIGIR'21), each citing the prior as
   justification. See "The simulator mechanism" below for the verbatim chain.

7. **Nobody does C in the modern line and nobody at all does D.** No paper in either line reports answer rate
   *as a property of the question channel*, and no paper contrasts item questions vs attribute/concept questions
   **on answerability**. The one paper that compares those channels in a Golbandi tree (Gharahighehi et al. 2025)
   measures only RMSE and attributes its (directionally identical) result to clustering efficiency, not answerability.

8. **The regression framing is the honest, defensible pitch.** The 2011 literature had a ternary answer space and an
   answerability-aware acquisition score; the 2018–2026 literature replaced them with a forced-binary answer space and
   an omniscient simulator, and lost the construct. Our contribution is (i) putting the number back, per channel,
   on a modern catalogue and a certified ruler, and (ii) the C/D decomposition nobody has ever run.

9. **Anchoring caveat that must be stated whenever we cite the classic answer rates.** Rashid's rates are measured on
   *power users*: the 2002 offline study "*eliminated users who had fewer than 200 ratings*"; the 2008 offline
   simulation "*only used users who have at least 80 ratings*" and admits "*Selecting users with 80 or more ratings
   may create a bias in that our findings may apply only to users with many ratings*". Their ~1/3 answer rate is
   therefore an **upper bound** on a genuinely cold population — which makes our 27% item rate on ML-25M consistent
   with, not contradictory to, theirs.

---

## The audit table (ready to adapt for the paper)

**A** = explicit unknown/unanswerable branch, or answer-rate-aware question selection.
**B** = reports an answer rate / coverage / how often users could respond.
**C** = measures how much of the achievable gain is attributable to answerability vs informativeness.
**D** = contrasts items vs attributes/concepts **on answerability** (not on accuracy).

| System | A models | B reports | C quantifies | D compares channels | Verification |
|---|---|---|---|---|---|
| **Golbandi, Koren & Lempel, WSDM 2011** (adaptive tree) | **YES** — ternary tree, "like"/"dislike"/"unknown"; unrated ⇒ unknown; seed sets should be "familiar items, since asking users to rate obscure items is mostly futile" | **partial** — states 99% sparsity ⇒ "the Unknowns users group much larger... than the Lovers and Haters groups"; no answer-rate metric | **partial** — `w_L=5, w_H=1, w_U=0.02`: "an unknown vote was found to be relatively insignificant" | **NO** — item-only; genres named as future work: "user feedback not only to basic items, but also to broader attributes such as genres, which are particularly important for a quick bootstrapping process" | full text |
| **Rashid et al., IUI 2002** (Getting to know you) | **YES** — Pop*Ent: "popularity is the probability that this user will rate this item"; Item-Item personalized selects items "the user is likely to have seen" | **YES, strongest in the corpus** — Fig. 2 movies-seen vs presented; live study Fig. 4/5, mean pages-of-10 to reach 10 ratings: Popularity 1.9, Item-Item 2.3, logPop*Ent 4.0, Classique 7.0 → ≈53% / 43% / 25% / 14% answer rate (**our derivation**, 10 ratings ÷ 10·pages); 49/351 dropouts | **partial** — both axes on the same runs + causal argument; Item-Item = published dissociation (wins effort, loses MAE); no decomposition | **NO** — all five strategies are item-level | full text |
| **Rashid, Karypis & Riedl, SIGKDD Expl. 2008** (HELF/IGCN) | **YES** — Entropy0 makes "missing" a rating category (0); HELF = harmonic mean of entropy and log rating-frequency | **YES** — Fig. 5 "#movies seen vs #presented" per strategy; "users are able to rate at least one third of the presented movies by each approach"; live study (381 subjects) Table 5 pages-to-finish 20 ratings: Entropy0 3.0, Popularity 3.6, HELF 4.6, IGCN 5.5 (page size stated only as "10/15" — do **not** convert to a percentage) | **partial** — same dissociation ("the confusing results... are from Popularity and Helf"); no decomposition | **NO** — item-level only; zero genre/attribute arm | full text |
| **Zhou, Yang & Zha, SIGIR 2011** (FMF) | **YES** — inherits Golbandi's ternary tree; unknown = "not in the training log" (structural branch, not an answerability model) | **NO** | **NO** | **NO** — item-level interview only | **secondary only — primary PDF unobtainable (Unpaywall `oa_status: closed`; ACM 403; all Georgia Tech mirrors 404). Evidence chain: Sun WSDM'13 §1–2 (3 shared co-authors), Chen ICML'13, Elahi survey Table 1. Do NOT cite a section number for FMF's node structure.** |
| **🚩 Sun et al., WSDM 2013** (multi-question trees; FMF's own successor) | **YES** — split over a *set* of items; unknown group is the residual "only when the user answers none of the questions" | **YES — the number** — "the unknown branch capturing **more than 80% of the users at each split**"; live study n=76 on interview *time* | **partial, and the strongest in the corpus** — RMSE vs screens / questions / **user-time** curves separate informativeness from interaction cost; but the multi-question fix is *both* more answerable and more informative and the confound is never resolved | **NO** — genre/attributes only in future work | full text (key quote re-verified by me) |
| **Liu et al., RecSys 2011** (RBMF) | **partial** — "activeness scores… to increase the success rate of obtaining ratings"; **but activeness is never defined by an equation** (mentioned twice, no formula). De facto popularity bias via missing⇒0 + max-norm column selection | **YES** — defines **Coverage** = proportion of users who have rated the representative set; Netflix 0.853 / ML 0.918 / Last.FM 0.960 | **NO** — coverage (§4.3) and accuracy (§4.4) reported in separate experiments, never linked | **NO** | full text (agent) |
| **Fonarev et al., ICDM 2016** (RMVA) | **NO — the cleanest bad quote in the literature**: "if user u was asked to rate item i with unknown rating, then, according to PureSVD model, r_ui is set to 0" | YES (coverage/diversity fig, borrowed from RBMF) | **NO** | **NO** | full text (agent) |
| **Kweon et al., WWW 2020** (DRE) | **NO — worst case**: response is a slice of the binary interaction matrix, so "can't answer" and "dislikes" are **the same value, 0**; no mask, no third state | **NO** — and it *drops* the coverage analysis its own two baselines (RBMF, RMVA) reported | **NO** | **NO** | full text (agent) |
| **Sepliarskaia et al., RecSys 2018** (SPQ) | **partial, by restriction not by model** — Real condition asks only about items the user rated: "However, optimal items may not be available"; Simulated condition generates every answer from the LFM | **NO** | **NO** | **NO** | full text (agent) |
| **Elahi, Ricci & Rubens, CSR 2016** (AL survey) | **YES — names the concept**: whole taxonomic category "**Acquisition probability based**… maximizing the probability that the selected items are familiar to the user, hence, are **rateable**" | **YES** — Table 1 has a "# Rating" column; §7.1 names "**success ratio (#acquired_ratings/#requested_items)**" | **partial** (survey-level, no decomposition) | **NO** | full text (agent) |
| **Bıyık et al. 2023** (soft attributes, CAV, EVOI) | **NO** — response models are ρ∈{+1,−1} (attribute) and ρ=i∈S (item choice); probability mass sums to 1 over answers, no abstention. **But names the gap**: "this would assume an unrealistic level of familiarity with available items I by u... **In lieu of a detailed familiarity model for u**, we capture some familiarity with I by constraining her target item" (§3.1) | **NO** | **NO** | **NO** — items vs attributes compared on elicitation efficiency only | full text |
| **Sun & Zhang, SIGIR 2018** (CRM) | **NO** — three user behaviours enumerated, none is "cannot answer"; AMT workers "presented with the facet values of the target restaurant on the side, so that she can correctly answer the questions" | **NO** | **NO** | **NO** | full text |
| **Lei et al., WSDM 2020** (EAR) | **NO** — state encodes only 0 = "attribute u disprefers" / 1 = "attribute u desires"; oracle attribute set P_v of the target item | **NO** | **NO** | **NO** | full text |
| **Lei et al., SIGIR 2020** (SCPR) | **NO** — "when a system asks for an attribute, he will only confirm he likes it if this attribute is included by item v"; explicitly **collapses** "does not care" into "reject" | **NO** | **NO** | **NO** | full text |
| **Deng et al., SIGIR 2021** (UNICORN) | **NO** — axiom: "the user preserves clear preferences towards **all** the attributes and items. Thus, the user will respond accordingly"; no limitations section | **NO** | **NO** | **NO** | full text |
| **Li et al., TOIS 2021** (ConTS) | **NO** in the model; **only acknowledgement in the corpus**, in future work: "we can consider how to handle 'don't know' of [or] 'don't care' responses" | **NO** | **NO** | **NO** — unifies items+attributes as "undifferentiated arms", which presupposes equal answerability | full text |
| **Austin et al., RecSys 2024** (PEBOL) | **NO** — simulator prompt: "*You must reply yes or no to the query based on the description of your preferred item*"; the simulated user is handed the ground-truth item description | **NO** | **NO** | compares item vs aspect queries **on MRR/NDCG only** | full text + repo prompt template |
| **Li et al. 2023** (GATE) | **partial, wrong axis** — real humans told they are "free to avoid answering any questions that are overly broad or uncomfortable" (comfort ≠ knowledge); free-text so refusal is *possible* | **NO** — never counted or categorised | **NO** | **NO** | full text |
| **Handa et al. 2024** (OPEN) | **NO** — forced binary pairwise A/B, no indifference option; burden measured as *mental demand*, not ability | **NO** | **NO** | **NO** | full text |
| **Montazeralghaem et al. 2025** (clarifying-Qs, arXiv:2510.12015) | **YES** — "we instruct the LLM to respond with 'I don't know' in cases where the answers are not present in the user profile"; Alg. 2: else set A ← "No Preference" | **YES** — Fig. 3(b) "Percentage of unanswered questions for models" (plot only, no numbers in text) | **NO** — used as a *simulator-fidelity diagnostic*, not a channel property | **NO** | full text |
| **Kostric, Balog & Radlinski, TORS 2024** (usage questions) | **premise only** — "ordinary users often do not possess this kind of attribute understanding, which might require extensive domain-specific knowledge" | **NO** | **NO** | **argued, never measured** — usage vs attribute questions evaluated on question quality, not answer rate | full text |
| **Gharahighehi et al. 2025** (pairwise + attribute-aware tree, arXiv:2510.27342) | **YES** (inherited Golbandi unknowns branch) | **NO** | **NO** | **NO on answerability** — compares item-only vs item+genre trees **on RMSE**; finds genres win early, items win from iteration 13 | full text |
| **Xia, Zhang & Wang 2026** (COPE, arXiv:2607.06765) | **NO** | **NO** | **NO** | **nearest miss** — finds attribute elicitation dominates early turns, item elicitation later, but attributes it to *preference specificity*, explicitly not answerability | full text (via agent) |

---

## The simulator mechanism (the sharpest thing we can say)

The modern CRS benchmark makes answerability **structurally invisible**: the simulated user's answer is a
deterministic lookup on the ground-truth target item's attribute set, so every question is answerable by construction
and unanswerability has exactly zero cost. Verbatim chain, all VERIFIED full text:

- **CRM (2018), §4.2** — origin. Three enumerated user behaviours: answer the question / find the target in the list /
  leave. "Answering the agent's question" is unconditional. Its *human* AMT study: "*The worker is presented with the
  facet values of the target restaurant on the side, so that she can correctly answer the questions.*"
- **EAR (2020), §4.1.2** — "*given an observed user–item interaction (u, v), we treat the v as the ground truth item to
  seek for and its attributes P_v as the oracle set of attributes preferred by the user in this session.*"
- **SCPR (2020), §5.2.2** — "*when a system asks for an attribute, he will only confirm he likes it if this attribute is
  included by item v. There is no denying that such simulation has many limitation, but it is the most practical and
  realistic at current stage.*" The limitation it then names is false-negative exposure, **not** answerability.
- **UNICORN (2021), §3** — "*the user preserves clear preferences towards **all** the attributes and items. Thus, the
  user will respond accordingly, either accepting or rejecting the asked attributes or the recommended items.*"
- **ConTS (2021), §5.2.2** — "*the user will give positive feedback only to item v and attributes P_v in current
  session*" — everything else is a negative, including attributes a cold user has never encountered.
- **PEBOL (2024)** — the LLM-era version. Paper §5.2.1: the simulated user "*is given item description x_i and
  instructed to provide only 'yes' or 'no' responses*". Production prompt (`templates/user_simulation_movie.jinja2`,
  authors' repo): "*You are a user who is being asked yes or no queries to elicit your preferences. **You must reply
  yes or no** to the query based on the description of your preferred item.*" The simulated user therefore has perfect
  knowledge of the target item — a reading-comprehension agent, not a person with a viewing history.

**Useful secondary rhetoric:** EAR calls itself "*an upper bound study of real applications as we do not include the
errors for language understanding and generation*". It names NLU error as the bound; answerability is a second,
unnamed source of upper-bounding in the same evaluation.

**Do not overreach on the "omniscient simulator" critique.** A 2025–26 simulator-realism strand exists (RecUserSim /
CSHI / "Interplay", arXiv:2603.18573, 2605.05250 — *abstract-level only, unverified*) but it attacks **target-item
leakage into the dialogue policy** and unrealistically high **acceptance** rates, not the user's ability to answer a
question about an entity they have never encountered. Different construct; cite as adjacent, not as support.

---

## What is NOVEL vs pre-empted

**PRE-EMPTED — cite, never claim:**
- *"Users can't rate obscure items, so ask about familiar ones."* — Golbandi §3.1.1 verbatim; Rashid 2002 throughout.
- *An unknown/haven't-seen branch in an interview.* — Golbandi's ternary tree, 2011. First-class, load-bearing.
- *Popularity as an answerability proxy / P(user can answer) × informativeness.* — Rashid 2002 Pop*Ent (Bayes framing),
  Rashid 2008 Entropy0 + HELF.
- *Reporting an answer-rate curve per selection strategy.* — Rashid 2002 Fig. 2, 2008 Fig. 5, plus two live studies.
- *Item-vs-attribute channel comparison in a cold-start interview tree.* — Gharahighehi et al. 2025 (RMSE), and
  Golbandi 2011 flags genres as future work. Our contribution is the **answerability metric**, not the comparison.
- *An "I don't know" branch in an LLM elicitation loop.* — Montazeralghaem et al. 2025 (but as *no preference*
  = profile-absence, not *entity-unfamiliarity*, and only as a simulator diagnostic).
- *Diagnosing the unanswerable-question pathology in an interview tree and engineering around it.* — **Sun et al.
  WSDM 2013** (>80% unknown per split; multi-question screens as the fix). The single strongest pre-emption.
- *Naming answerability as a design axis and a survey category.* — Elahi/Ricci/Rubens CSR 2016 ("acquisition
  probability based", "rateable", "success ratio"). Use their vocabulary; do not present the concept as new.
- *A personalized P(user has experienced item) model.* — **Binary Prediction** (Elahi, Ricci, Rubens, ACM TIST 2014):
  MF on the binarised observation mask, i.e. predict *whether* the user rated an item regardless of value, to
  "maximize the chances that the user can provide the requested rating". Closest existing thing to a learned
  answerability model — but used as a ranking heuristic, never as a calibrated probability inside an objective.

**Still genuinely empty (agent-verified across the classic MF line):** no work writes a joint objective of the form
`argmax_i P(answerable | u,i) x Info(i)` with **both** terms modelled. HELF's harmonic mean of normalised-log-frequency
and normalised-entropy is the field's high-water mark, and it is non-personalised and uncalibrated.

**NOVEL / defensible (in decreasing confidence):**
- **D — answer rate as a per-channel property, measured on the same users and the same budget.** No paper in either
  line reports "items 27% vs concepts 75% at 8 questions". This is the cleanest empty cell in the table.
- **C — the decomposition.** Nobody separates "the user could answer" from "the answer was informative" as a share of
  the achievable interview gain. Rashid's Item-Item dissociation proves the two are separable and is the strongest
  near-miss; it must be cited *as* the near-miss, and our claim stated as the attribution they did not perform.
- **Answerability measured on a modern, long-tailed catalogue at the actual cold-start population** rather than on
  users pre-filtered to ≥80 or ≥200 ratings. Modest but real: the classic numbers are power-user upper bounds.
- **An unanswerable branch meaning "I don't know this entity"** as distinct from "I have no preference" — the
  distinction the 2025 clarifying-questions work does not make and ConTS lists as open future work.

---

## Open questions / caveats

- **Do not claim the classic line ignored this.** Any reviewer who has read Rashid or Golbandi will kill the paper on
  that sentence. Lead with the *regression* framing (§What we concluded, 8) and the *decomposition* (C/D).
- **Gharahighehi et al. 2025 (arXiv:2510.27342) is the closest live threat and is only months old.** Its crossover
  (genres win at low budget, items win from ~13 questions) is directionally the same shape as our result on a
  different dataset with a different metric. Read it against our D1 curve before writing the claim; if it is an
  RMSE-space replication of our qualitative finding, our differentiator has to be the answerability *mechanism* and
  the tail-NDCG ruler, not the direction.
- **Rashid's "1/3 answer rate" and our 27% are not directly comparable** — different catalogue size (4k/9k vs 18,430),
  different user filters, different question banks. Report both with their filters attached; do not tabulate as a
  like-for-like row.
- **Sun et al. WSDM 2013 must be read in full before the claim is drafted.** Their Fig. 3/6 RMSE-vs-user-time curves
  are the closest anyone gets to C, and their confound (4 questions/screen is simultaneously more answerable AND more
  informative) is exactly the confound our decomposition is supposed to break. If we can show our C-decomposition
  resolves *their* stated confound, that is a much stronger framing than "nobody measured this".
- **The FMF (Zhou 2011) node structure is UNVERIFIED by primary text** — closed access everywhere (Unpaywall
  `oa_status: closed`). "FMF splits like/dislike/unknown" is safe via Sun WSDM'13 §1–2 and Chen ICML'13;
  "FMF never reports the unknown rate" is safe via the Elahi survey's Table 1 "–". Do **not** cite a section number.
  Get the ACM PDF through the institution before the thesis goes out.
- **Verification levels.** Rows marked "full text" were read from extracted PDF/HTML text; those marked "(agent)" were
  read by a subagent and spot-checked by me only where flagged. arXiv IDs 2510.12015 / 2510.27342 / 2607.06765 /
  2603.18573 all verified to resolve; 2605.05250 not checked. The RecUserSim/CSHI simulator-realism strand is
  **abstract-level only**.
- **Bonus finding for `elicitation_and_belief_pool.md` open questions:** Bıyık's EVOI integrand is confirmed
  **expected UTILITY**, not a rank-discounted ranking metric — Eq. 13 `EVOI(q|H) = PEU(q|H) − EU*`, with
  `EU*(P_U(u)) = max_i E[φ_u^T φ_I(i)]` (Eq. 14). The "VoI integrand IS the ranking metric" wedge SURVIVES.
  [VERIFIED full text, §5.1]

---

## New sources — owed to `../INDEX.md` + `references.bib`

| suggested key | citation | why |
|---|---|---|
| `montazeralghaem2025asking` | Montazeralghaem, Tennenholtz, Boutilier, Meshi. Asking Clarifying Questions for Preference Elicitation With LLMs. arXiv:2510.12015 | only LLM system with an "I don't know" branch **and** an unanswered-rate figure |
| `gharahighehi2025pairwise` | Gharahighehi, Nakano, Yang, Cu, Vens. Pairwise and Attribute-Aware Decision Tree-Based Preference Elicitation for Cold-Start Recommendation. arXiv:2510.27342 | closest published item-vs-attribute interview-tree comparison; live threat |
| `sun2018conversational` | Sun & Zhang. Conversational Recommender System. SIGIR 2018 | origin of the omniscient user simulator |
| `sun2013learning` | Sun, Li, Lee, Zhou, Lebanon, Zha. Learning Multiple-Question Decision Trees for Cold-Start Recommendation. WSDM 2013 | **the strongest pre-emption** — >80%-unknown-per-split diagnosis + multi-question fix |
| `elahi2016survey` | Elahi, Ricci, Rubens. A Survey of Active Learning in Collaborative Filtering Recommender Systems. Computer Science Review 20:29–50, 2016 | the field's vocabulary ("rateable", "acquisition probability", "success ratio"); Table 1 is our evidence of non-reporting |
| `elahi2014active` | Elahi, Ricci, Rubens. Active learning strategies for rating elicitation in collaborative filtering: a system-wide perspective. ACM TIST 2014 | **Binary Prediction** = MF on the binarised mask, the closest existing learned P(user can answer) |
| `fonarev2016rectangular` (already in bib) | Fonarev et al. RMVA, ICDM 2016 | quotable: unknown rating "is set to 0" |
| `kweon2020deep` (already in bib) | Kweon, Kang, Yu. DRE, WWW 2020 | quotable: can't-answer and dislike are the same 0 |
| `handa2024bayesian` | Handa et al. Bayesian Preference Elicitation with Language Models (OPEN). arXiv:2403.05534 | forced-binary answer space, real humans |
| `xia2026when` | Xia, Zhang & Wang. When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation (COPE / InPE). arXiv:2607.06765 | nearest miss on axis D (attributes early, items late — attributed to specificity) |

*(All four 2025–26 arXiv IDs above verified to resolve: 2510.12015, 2510.27342, 2607.06765, 2603.18573.)*
