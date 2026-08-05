# Paper B — comprehensive baseline battery + TODO (assembled 2026-08-05)

Sources: four deep-research passes (`external_literature/findings/rating_elicitation_family_deep.md`,
`active_learning_elicitation.md`, `llm_elicitation_and_metalearning.md`, `bandit_rl_crs_baselines.md`)
plus a full internal audit of Paper A's baselines and the repo's existing machinery.
Effort is in **agent-hours** (we code with LLM agents). Everything runs on the certified pair
(tower `i26c ep31` + concept channel `cv3 ep08`), canonical Liang ML-25M split, NO-FLOOR primary,
full+tail NDCG@10, behavioural answers.

---

## 0. THE FINDING THAT REFRAMES ACT 1

**Greedy SLIM (arXiv:2406.06061) already publishes our Act-1 result on our dataset, our metric and
our tail definition.** It runs ML-25M, reports NDCG@10, and defines the tail as the items outside
the most-popular 33% of ratings — verbatim our head mask. Static Greedy SLIM vs adaptive QBandit
(Christakopoulou KDD'16), full NDCG@10 at k=5/10/15/20: **0.3390/0.3594/0.3709/0.3752 (static) vs
0.3382/0.3456/0.3221/0.2916 (adaptive)** — the adaptive arm ties early and then *degrades
monotonically*. Tail: 0.1163/0.1543/0.1674/0.1779 vs 0.1183/0.1465/0.1488/0.1466 (adaptive wins
only the k=5 tail cell). On **Netflix it goes the other way at k≤10** — report that too.

**⚠ CORRECTED 2026-08-05 — this comparison is a PROTOCOL ARTEFACT by its own authors' diagnosis.**
Greedy SLIM §3.3.2 requires `I_Q ∩ I_R = ∅` — asked items are excluded from the recommendable set
**on the ask, not the answer** (only 34–49% of their questions drew a rating) — and §6.5 states
plainly: *"the main reason for the decreasing NDCG is that the questionnaires are not allowed to
recommend items that they contain as questions."* An adaptive selector asks more user-specific,
target-adjacent items, so this tax compounds with budget and falls hardest on the adaptive arm.
**Greedy SLIM can no longer be cited as clean evidence that static beats adaptive.** It remains a
must-run baseline (its static questionnaire is real) and a must-cite for protocol design, but the
QBandit decline is theirs, not nature's. See `findings/candidate_masking_protocol.md`.

Consequences, all of which the chapter must absorb:
1. We are **not alone and not first**; the honest frame is "an independent measurement on a
   certified instrument, with a mechanism" — not "we discovered adaptivity does not pay", and not
   "a third confirmation" now that one of the two prior pillars is protocol-confounded.
2. It is a **must-run baseline** (it is the strongest static bar available: prefix-optimal, so one
   fit gives every budget) AND a must-cite.
3. Two prior comparisons of best-static vs best-adaptive both go to static (SPQ/Sepliarskaia 2018,
   Greedy SLIM 2024). The field's "static seed sets are bad" claim traces to an **unsupported
   assertion** in the 2011 Handbook chapter, repeated verbatim by PERE 2024 as motivation.

**The real moderator may be SPARSITY, not adaptivity** (two De Pessemier papers: adaptivity pays on
dense data — Jester, ML-1M — and vanishes or reverses on sparse; on Publiq MostPop beats the
adaptive tree on both precision and NDCG). This is directly testable in our harness and is a
better-posed question than "does adaptivity work".

---

## 1. TIER 0 — corrections that gate every claim (do first)

| # | Task | Effort | Why blocking |
|---|---|---|---|
| 0.1 | **Paired bootstrap CIs on every Act-1 cell** (same 10k users; never the `1.96·√2·SE` unpaired form) | 3 | Every Act-1 number is a point estimate. The unpaired form already killed one claim in this project. |
| 0.2 | **Re-run EDDI with a 3-outcome response model** | 4–8 | Two-outcome likelihood cannot represent "no answer arrives", so our answerability mechanism claim is currently unsupported. If the answer rate recovers this is a *better* result. |
| 0.3 | **Re-run the three selectors inside the JOINT environment against the greedy sequence** | 6–10 | Act 1's null is currently measured against entropy₀, not against the prefix-optimal comparator the chapter builds. The two tables are on different masks (141.2 vs 62.6 items/user ⇒ +0.027 at the cold state). |
| 0.4 | **RE-WORDED 2026-08-05.** Parapar & Radlinski's `AvailablePreferences` is a **judgment-set** change, not a candidate-set change, and the bias it documents favours **popularity-seeking** elicitation (they run no entropy baseline). Apply it as a robustness row on top of policy B — zero cost, no recommender re-run | 2 | Not the threat we thought. It is the **published mechanism** for our own finding that the 4-way entropy bank is popularity in disguise, and our popularity-agnostic `polar` bank is the least-exposed arm. |
| 0.7 | **★ ADOPT CANDIDATE-MASK POLICY B BATTERY-WIDE** (mask every rated fold-in item; `arm_n.build_pool` is the reference). Seven disagreeing conventions are live in the repo. Fixes: strip the asked-mask and `te[j,i]=0` from `strategy_ladder`; fixed pool in `genv_eval`; full rated mask in `concept_cold_check`; add B as a *second column* to `gate_g9` (never silently change a Paper A number). Two asserts: mask identical across arms within a budget, `te` never modified by an arm. Canary MostPop + EASE first | 8 | **A→B is worth +0.027 NDCG at the cold state** — an order of magnitude above the effects the battery resolves. Until this lands, cells from different scripts are not orderable. Supersedes and unblocks 0.3. |
| 0.5 | **Scope statement from Elahi (TIST'14)**: per-user evaluation isolated from natural rating acquisition can mislead | 1 | Our protocol is exactly that configuration. Cite and scope, don't wait to be told. |
| 0.6 | **Floored arm of Act 1** (drop `--no_floor`) | 1 | Never produced. TaNP sits *below* MostPop on zero-answer arms — precisely what the floor exists to expose. |

## 2. TIER 1 — free wins (implemented, never run on the certified stack)

| # | Task | Effort | Note |
|---|---|---|---|
| 1.1 | Wire **EASE, Mult-VAE, EDLAE, iALS, item-kNN, Turbo-CF** as interview consumers | 1 each | All already satisfy `predict(X_csr)→dense`. Spans .2428→.3540 full — the accuracy range the instrument-dependence claim needs. EASE is the one every reviewer asks for. |
| 1.2 | Add **RBMF** to the Act-1 run | 0.2 | It is `RECS[3]` and a row in Paper A's table; omitted from our run by an incomplete `--only`. |
| 1.3 | Promote **belief_mf** from silent JSON row to reported consumer | 0.2 | It is inert (Spearman 0.995 vs pure prior) — that is a finding, not a reason to hide it. |
| 1.4 | Re-point **`strategy_portability.py`** at `greedy_static_i26c.json` | 1 | Gives the compression / ordering-flip result for free; currently consumes the superseded sequence. |
| 1.5 | **`--objective tail` greedy** re-run | 1 + ~12h compute | Flagged "not optional" for any tail claim; the chapter carries an open REDO. |
| 1.6 | **Golbandi routing ablation** (ternary / no_unknown / no_sign) on the certified stack | 2 | Implemented; measured only on the old tower. Second architecture corroborating the sign result. |
| 1.7 | **Pep's adaptivity diagnostic**: how often does the follow-up change when the answer changes (Pep 39–62%, RL 0–28%) | 2 | Published comparison scale for what our distinct-prefix counter already measures. |

## 3. TIER 2 — the battery proper

### 3a. STATIC bars (the comparator must be unimpeachable)

| Method | Cite | Effort | Verdict |
|---|---|---|---|
| **Greedy SLIM** | arXiv:2406.06061 (CoRR only) | 6–10 + 6h compute | **MUST — strongest static bar.** Our dataset/metric/tail. Provenance caveat: arXiv-only, user study n=103 "many linked to the authors". |
| **DRE** | Kweon et al. WWW'20 | 3–5 | **MUST — the peer-reviewed static bar.** Public PyTorch. Binarises >3.5 like us. Retrain per k. Cheap criticism available: no duplicate-seed guard (k independent categoricals + per-row argmax). |
| **RBMF (maxvol selector)** | Liu/Meng/Liu/Yang RecSys'11 | 2–4 | **MUST.** We have the fold-in, not the selector. Its budgets 5/10/20 are ours. |
| **RMVA** | Fonarev ICDM'16 | 3–6 | **MUST** — named baseline of both DRE and PERE. |
| **GreedyExtend** | Golbandi CIKM'10 | 4–8 | **NICE-HIGH** — the direct ancestor of our prefix-optimal greedy; cite as prior art for our own static bank. |
| **MeLU evidence-candidate score as a bank ranker** | Lee KDD'19 | 4 | `norm(mean local-update gradient) × norm(popularity)`. The only published prior instance of the object our bank sweep studies. **Never present as adaptive** (one-shot, global). |
| **A-optimal design** | Anava WWW'15 / Yu ICML'06 | 4–8 | **MUST — near-free**: `tr(A_t⁻¹)` is one line on `bandit_selectors.py`'s existing precision matrix. |

### 3b. ADAPTIVE bars

**★ TOP NOMINATION — CONVERSATIONAL BANDITS, a family the earlier passes missed entirely.**
These ask about a **key-term (attribute/genre/tag) OR an item in one algorithm**, model explicitly
how key-term feedback moves the item-level estimate, carry regret bounds, and have **no target item
and no simulator** — the only published adaptive line whose action space is ours and which passes
the behavioural firewall unmodified.

| Method | Cite | Effort | Verdict |
|---|---|---|---|
| **ConUCB** | Zhang/Xie/Li/Lui **WWW'20** (arXiv:1906.01219) | — | **MUST (policy).** Item-level ridge shrunk toward a key-term-level fit; key-term value = uniform member average (Eq. 4). |
| **★ ConLinUCB-MCR / -BS** | Wang/Liu/Li/Lui **AAAI'23** (arXiv:2303.00315) | **4–8** | **★ MUST — the single highest-value new arm.** One-step `θ=M⁻¹b` where a key-term answer enters *the same normal equations* as an item answer; `M_t` **is** our posterior precision and MCR's `argmax_c α_t‖d_c‖_{M⁻¹}` **is the width term `bandit_selectors.py` already computes**. Public numpy code. Compare 40+ h for NICF/PERE. |
| **CLiSK / CLiME** | Liu/Li/Dai/Lui **KDD'25** (arXiv:2505.21393) | +3 | **MUST — the 2025 anchor** (runs ML-25M with genres as key-terms). Diagnoses ConUCB as "no better than not conversing"; CLiME = `argmax_k|x̃_kᵀv_i|` on small-eigenvalue directions. |
| **Hier-UCB** | Zuo et al. arXiv:2209.06129 | — | **NICE-HIGH** — finds key-term rewards are driven by a few *representative items*, not a uniform member average. **Directly challenges our whitened-centroid concept fold.** |

**Why this is a scientific test and not just a table row:** ConUCB says a concept's value is the
uniform member average; Hier-UCB's data says a few representative items dominate; our whitened
centroid is a third answer. Running it adjudicates the concept-fold question on our own metric.

| Method | Cite | Effort | Verdict |
|---|---|---|---|
| **ICF's four policies** (TS, LinUCB, GLM-Lin, GLM-Sig) | Zhao/Zhang/Wang **CIKM'13** | 3–5 | **MUST** — upgrades "we tried a bandit" to "we ran the Interactive-CF paper's own policy set". NB in their own experiment **TS loses to LinUCB/GLM** (43.7 vs ~46.4), contrary to how it is usually cited. |
| **Golbandi ternary tree — the SPLIT SEARCH** | WSDM'11 | 6–12 | **MUST — strongest classical adaptive bar and the largest hole in our harness.** We have the leaf/consumer half only. No method in fifteen years has a clean, contract-compatible, replicated win over it at k≤16. |
| **IGCN (genuine form)** | Rashid SIGKDD-Expl'08 | 4–7 | **MUST** — must do the 150-NN re-computation per 15-item batch or it is not IGCN. |
| **fMF** | Zhou SIGIR'11 | 6–12 | **MUST** — "Golbandi with a real latent model". NB **paper bug**: Algorithm 2 prints `argmax` where Eq. 4 says `argmin`; do not code from the pseudocode. |
| **DPE (topic bandit)** | Parapar & Radlinski WSDM'21 | 2–3 | **MUST — our only named concept-channel rival.** Genome gives the item→topic map free. |
| **BALD** | Houlsby arXiv:1112.5745 | 1–2 | **NICE-HIGH — cheapest unclaimed arm**; needs only (μ,σ²) which `belief_mf` already emits. |
| **SEL-BALD** | NeurIPS'24 | 6–10 | **MUST — the named answerability arm** (`P(answer)·BALD`). Our structural rule = their discretion model. |
| **Patent `IG × A`** | US 11,775,493 B2 (Radlinski et al.) | 2 + 3 for Elahi's learned `A` | **MUST-CITE (blocking), NICE to run** — it is literally our formula. |
| **PEBOL acquisition, behavioural answers** | Austin RecSys'24 | 4 | **MUST (one row).** Its selection stage is LLM-free by construction, so this is honest not crippled. Beta-Bernoulli per item, four rules (TS/UCB/ER/Greedy). |
| **PERE** | Nguyen UAI'24 | 10–20 | **MUST but DEMOTED.** No public code. Real margins are +0.59…+3.94% over *its own* c-DPP construction; the "3–10%" figure is **not in the paper**; its DRE column is ~10× below DRE's own numbers (broken harness). |
| **a-stratification** (CAT) | Chang & Ying 1999 | 4–6 | **NICE-HIGH** — a published prediction that our discrimination-ranked bank is *suboptimal at small k*. Falsifiable against our own setup. |
| **BatchBALD** | NeurIPS'19 | 6–10 | **NICE-HIGH** — may explain why a static screen beats greedy. |
| **Karimi non-myopic** | ICTAI'11 | 3–5 | NICE (~30 lines over MF). |

### 3c. EXCLUDED, each with a stated reason (never table their numbers)
NICF / RLT4Rec / MCTS-CBB (cumulative reward during probing, or group-identification accuracy with
an oracle-informed planner) · EAR / SCPR / UNICORN / ConTS / CRM (omniscient target-item simulator,
SuccessRate@turn, closed schema) · BED-LLM (its EIG posterior *is* the LLM — not legitimately
ablatable) · Bıyık soft attributes (arXiv-only, 16 simulated test users, "NDCG" = agreement with the
simulator's own embedding) · MAMO/TaNP/CDRNP/ColdNAS/MetaKG/M2EU as *elicitation* baselines
(category error — they select nothing) · DAD/iDAD (gradients through an 18k-way categorical design;
sPCE needs a tractable likelihood, ours is empirical MNAR) · 2025 attribute trees' Yahoo-RMSE numbers
· Sun WSDM'13 as a row (no accuracy table exists).

---

## 4. TIER 3 — paper-level work

| # | Task | Effort |
|---|---|---|
| 3.1 | **Instrument-dependence table on the certified stack** under ONE stated contract (Paper A's `tab:instrdep` and `INTERVIEW_TABLE.md` currently disagree on floor and on which consumers) | 3 |
| 3.2 | **`--conc_bank polar` in the greedy suite** — port the `H(bands\|answered)` estimator from `concept_cold_check` (`std\|v\|` is a different quantity) | 4 |
| 3.3 | **Sparsity moderator experiment** (De Pessemier's claim): does the adaptive-static gap open on sparse users? Stratify our 10k by profile length | 4 |
| 3.4 | **Face-validity exhibits**: hand-picked trajectories + genre-congruence vs matched-divisiveness control | 6 |
| 3.5 | **Act 3**: Reddit-mined openers on ML-25M, DiD with paired CIs, dispersion validation gate first | 20+ |
| 3.6 | Rewrite Act 1 around the Greedy SLIM / SPQ / QBandit precedent (see §0) | 4 |
| 3.7 | `adaptive_concept_arms.py` ported to cv3 ep8 — the only adaptive *concept* selector we own | 8 |

---

## 5. POSITIONING CORRECTIONS (things we must stop claiming)

1. **Coarse-to-fine is twice published** — Gharahighehi 2025 (*"Starting from iteration 13, the
   decision tree-based model that elicits preferences only on artists outperforms the hybrid
   approach"*) and COPE SIGIR'26. Worse: they report attribute-early/**item-late**, so our
   "concepts win at EVERY budget under a pure-discrimination bank" is a **contrast with the field,
   not a confirmation**. Claim the mechanism (it emerges from loss-optimal selection, unspecified),
   not the phenomenon.
2. **Answerability-weighted acquisition is published ≥5×** (Rashid'02 pop×entropy; US patent
   11,775,493 B2 writing `IG × A`; Bıyık CoRL'19; SEL-BALD NeurIPS'24; possibly PERE). Ours is
   narrower: answerability as a reason to switch **channel**.
3. **Rashid IUI'02 published the answerability-vs-discrimination tension in 2002** (*"Pure Entropy
   is shockingly poor"*; pilot users had to view several hundred films to find ten to rate). Cite as
   the problem statement our concept channel answers.
4. **Golbandi WSDM'11 already published our sign asymmetry** — optimised leaf weights
   `w_L=5, w_H=1, w_U=0.02` with the remark that a like is far more useful than a dislike.
5. **The behavioural firewall is not a novel mechanism** — published as *"counterfactual
   validation"* (Balog & Boutilier, EACL'26). Rigour we followed, not an invention.
6. **The classical lineage is RMSE/MAE only** (Rashid'02/'08, Golbandi'10/'11, fMF, Sun'13, Karimi,
   the 2025 trees); ranking metrics start at RBMF 2011. Say it before a reviewer does.
7. **Sepliarskaia does not support what we cited it for**: Golbandi is cited but never compared
   against (the only tree is PWDT), the stated cause is the *objective* not staticness, and at k≤4
   it reports *no difference* (p>0.1). Authors include Kiseleva (missing from our note).
8. **De Pessemier Sci Rep 2025 is the best external cover the concept channel has**: an adaptive
   tree whose offline win collapsed to non-significance with 50 real users, diagnosed by the authors
   as *"Real users are not always able to rate the item selected by the active learning
   algorithm."*
9. **PEBOL's catalogue is 100 items** (a GPT-3.5 context artefact, by their own admission). Ours is
   **184× larger**. Their authors also argue our protocol case: ReDial-style corpora are
   *"pre-recorded conversation histories [that] cannot be used to evaluate active, cold-start NL-PE
   systems."* There is no consensus benchmark.
10. **The continuous-policy + LLM-verbalization bet is partially taken** — ELM decodes hypothetical
    embeddings; **LDPP (AAAI'25)** is continuous-action + frozen-LLM rendering; **Info-Synth (2026)**
    still snaps to dataset items. Open ground: closing the loop *inside one latent space*.

---

## 5b. TWO ARGUMENTS THE BANDIT PASS HANDS US (both sourced, both new)

1. **The founding bandit paper's own metric cannot see a burned question.** ICF (CIKM'13) defines
   `θ_hit = 1 if the rating is no less than 4, and 0 otherwise` — so an *unseen* item scores exactly
   what a *disliked* item scores. **No method in this lineage can price unanswerability, by its own
   metric definition.** That is a clean novelty claim sourced from the family's founding paper
   rather than asserted by us.
2. **Greedy SLIM independently replicates the bank-quality effect**: its adaptive QBandit curve
   *decreases* past k≈10 (full 0.3456→0.2916) while the static questionnaire keeps climbing. A
   published precedent for the static-wins-at-long-budgets shape of Act 1. ⚠ Their numbers use
   graded gain `2^r−1` and a 90/10 user split — same shape, different ruler, **never tabled beside
   0.3496/0.2491**.

**Immediate harness check this raises:** Greedy SLIM **excludes asked items from the recommendation
pool** and identifies that as the cause of QBandit's decline. Verify what `run_interview_table.py`
does; if we differ, our popularity arm's late-budget behaviour differs from theirs for a reason we
must state. (Our greedy env masks only *answered* items — see `GREEDY_BEST_STATIC_I26C.md`.)

**Prior-art risks from the CRS line, to read before writing:** HutCRS (EMNLP'23) does **coarse
"aspect" → fine attribute asking with an explicit "don't know" class** — the same motivation as our
budget-burn and our coarse-to-fine; CoCHPL (DASFAA'24) builds a **diverse non-redundant question
chain**, which is prior art adjacent to our greedy-prefix claim. MINICORN is the useful counter-cite
(no RL at all — MC-dropout uncertainty and one rule — yet beats SCPR by +67% success rate), i.e.
evidence the RL apparatus is not doing the work.

**Stop carrying the leakage critique on our own authority:** it is published — PEPPER (EMNLP'25
Findings) and Interplay (ECIR'26) state verbatim that existing work feeds target items to simulators
in advance, causing data leakage. Conversely Bernard & Balog find success-rate metrics correlate
positively with self-reported satisfaction, so the honest phrasing is **"prices a different
quantity"**, never "invalid".

**Acceptance gate for any ported policy** (HARD RULE 2 corollary — none of these can be
number-matched): (i) reproduce the selection rule exactly, (ii) beat random-catalogue and popularity
on our metric, (iii) label the row **"policy ported, protocol replaced"**. Write that before a
reviewer writes it for us.

**DO-NOT-CITE (binding):** Filippi's exact confidence radius (an OFUL-style formula was confabulated
in an earlier pass and rejected) · Pseudo Dyna-Q (not on arXiv, DOI 403s) · **CT-CRS arXiv:2508.13209
— WITHDRAWN by its authors yet still surfacing in search** · IGCF's numbers/venue · factorUCB's
alleged ranking metrics · Q&R's offline metric · RLT4Rec's datasets.

## 6. VERIFICATION DEBTS (do not quote until closed)

* **LRMF (Shi/Zhao/Shen TOIS'17)** — the RBMF↔Golbandi hybrid, closest published thing to our
  harness; ACM 403 and the S2 abstract is elided. **Highest-value unread method.** Needs
  institutional access.
* **PERE's "experience probability" term** — carried at search-summary grade across two files; if it
  really is info-gain × P(experienced) on ML-20M NDCG@10, our answerability framing must change.
* **SEL-BALD's exact acquisition formula** (OpenReview blocked; read the camera-ready).
* **Golbandi CIKM'10's exact Coverage formula** — not in any reachable secondary source.
* **"No CAT item-selection rule uses omission propensity"** — absence claim at [A] strength; one
  targeted *Applied Psych. Measurement* / *J. Educ. Measurement* pass owed.
* **Iovine JIIS'21** (219-user study reportedly favouring non-personalized); **Qian ESWA 2025**
  (personalises *when the interview stops* — the obvious reviewer question against our fixed-k
  protocol).
* Bıyık's method section; Balog & Zhai monograph full text (use arXiv:2501.04410 instead).

---

## 7. SUGGESTED ORDER

**Week 1 (gates + free wins):** 0.1 → 0.2 → 0.6 → 1.1–1.4 → 0.3 → 0.4.
**Week 2 (the bars that make the null falsifiable):** **ConLinUCB-MCR + CLiME (4–8 h, top value)** →
ICF's four policies → DRE → Greedy SLIM → Golbandi split search → BALD + A-optimal → IGCN.
**Week 3 (breadth + concept-channel rivals):** DPE → SEL-BALD → PEBOL-acquisition → RBMF/RMVA
selectors → MeLU bank ranker → a-stratification → Hier-UCB (challenges our centroid).
**Then:** tail-objective greedy, polar bank, sparsity moderator, instrument-dependence re-run,
face validity, and only then Act 3.

**Deferred / not built:** PERE (10–20h, no code, demoted numbers) unless a reviewer demands a
2024 adaptive name; fMF (paper bug, RMSE lineage) unless the tree line needs a latent control.
