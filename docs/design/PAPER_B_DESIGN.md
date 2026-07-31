# Paper B design (Fable, 2026-07-31)

Commissioned by the author as 'the very key paper'. Built from three independent sub-reviews:
a web-verified novelty audit, blind alternative-design generation, and an adversarial DoE critique.
Inputs: Paper A's claims and its six printed deferrals; the full failure corpus (adaptivity, RL,
coarse-to-fine, continuous actors); the arm-N and interview-table findings; the i26 retrain in flight.

---

All three agents are in. The novelty verification materially resolves task #55 — partly in our favor, partly against — and the three reports triangulate cleanly. Here is the design document.

---

# PAPER B DESIGN DOCUMENT

**Working title:** *The Anatomy of a Cold-Start Interview: Answer Contracts, Consuming Models, and the Limits of Question Selection*

**Prepared by:** Fable (senior strategic review), with three independent sub-reviews: (V) web-based novelty verification, (G) blind independent design generation, (X) adversarial DoE-literate critique. Where I cite a numbered attack (F1, M4, …) it is from X; verification verdicts are from V.

---

## 0. THE VERIFICATION RESULTS THAT DECIDE EVERYTHING (task #55 — debts now paid)

Before the design, the facts that constrain it, because two of them overturn the plan of record:

**0.1 — The Shen/Dong/Huan quote is real, verbatim, and useless to us.** V extracted the CMAME 444 (2025) PDF directly: *"integrating goal-oriented objectives into sOED has yet to be developed"* — but that sentence is their **motivation**, and the same paper then develops it (vsOED, goal-oriented prediction QoIs). The QoI stays fixed across the sequence, so a sliver of "posterior-dependent goal operator" survives *within the inverse-problems OED literature* — but:

**0.2 — Lin, Zhu, Wang, Caverlee WWW 2023 (UpsRec) substantially pre-empts the criterion.** Algorithm 2 is literally a *"Greedy NDCG Attribute Selector"*: re-derive the slate from the current user representation each turn, compute expected NDCG gain per candidate attribute, pick the argmax, at inference time. And **InfoBAX (ICML 2021)** already published sequential EIG about a posterior-dependent algorithm output *with top-k as a headline example*. What survives for us is a two-condition conjunction — *expectation under a maintained posterior* + *rank discount* + elicitation (Lin has no posterior; PERE UAI 2024 has a belief but a parameter-space criterion; nobody has both). That is machinery, not a headline. **The rank-targeted criterion is hereby demoted from contribution to instrument.** Task #53's increment (1) does not fully collapse, but it shrinks to something a paper cannot stand on.

**0.3 — Two published factorials ran strategy × recommender and found NO interaction** (Rashid et al. SIGKDD Expl. 2008: same ordering on user-kNN and item-kNN; De Pessemier et al. ORSUM 2021: "no major differences" across three final recommenders). Meanwhile Pagano/Quadrana/Elahi/Cremonesi 2017 explicitly named the cross-model experiment as future work, Greedy SLIM 2024 asserts *"the answer may depend on the chosen recommender"* without testing it, and Kweon et al. WWW 2020 state our mechanism in one sentence without generalizing. **So our inversion finding is unclaimed — but it must be framed as a refutation of an existing empirical null: the field's "no interaction" result was real within the kNN family and breaks when consumer families diverge (2011 conditional-popularity tree vs. modern VAE-class).** That framing is stronger than "we ran a factorial," and it is mandatory, because a reviewer holding Rashid 2008 will otherwise say the factorial was run in 2008.

**0.4 — The answer-alphabet claim is the safest ground we own.** Golbandi WSDM 2011 fitted w_L=5, w_H=1, w_U=0.02, found "like considerably more useful than dislike… quite interesting," **never ran the ablation**, and wrote *"our choice might be suboptimal"* about declining graded splits. A 2025 Scientific Reports paper **drops the dislike branch by assertion** ("we don't see this as a serious limitation"). Nobody has measured the marginal per-question *ranking* value of admitting "dislike," anything at k≤4, or graded-vs-binary at small budgets. The field guesses at the exact thing we have measured.

**0.5 — Knowledge Gradient has never been applied to cold-start elicitation** (nearest: Wu & Gardner, Jan 2026, preferential BayesOpt). Available as an uncontested, closed-form, decision-dependent baseline — and V confirmed the Viappiani–Boutilier equivalence is scoped to expected-utility-of-slate under noiseless/constant-noise *choice* responses: **no rank discount, no rating-type answers**. That is a clean citable boundary we stand on rather than fight.

---

## 1. THE CLAIM

**One falsifiable sentence:**

> *In cold-start interviews at canonical scale, what a user is permitted to say and which model consumes it determine the interview's value — collapsing sign or intensity in the answer costs 0.014–0.037 NDCG@10 at budgets ≤8, and the published ordering of question-selection strategies fails to transfer across consumer families (difference-in-differences ≠ 0 under matched users, answers, and pools) — while answer-adaptive branching, sandwiched between a clairvoyant upper bound and a Knowledge-Gradient lower bound, contributes less than the pre-registered 0.004 MDE.*

Three independently falsifiable components: (a) contract ablations exceed the MDE with paired CIs and survive noise/popularity controls; (b) at least one strategy-ordering inversion survives a DiD CI across ≥4 consumers; (c) the branching gap over the strongest constructible static arm stays below the MDE.

**Is this a negative-result paper?** No — and this is where I disagree with the framing in the brief. The brief asks whether Paper B "should be" a measurement/negative-result paper as a fallback. It should be a measurement paper **as a first choice**, because the measurement results are the *large* ones. Every adaptivity effect in the corpus is ≤0.0026; every contract effect is 0.007–0.037, i.e. 2–9× the MDE, corroborated across two unrelated architectures (our instrument and a 2011 decision tree), and sitting on unclaimed ground (0.4). The adaptivity null is not the apology at the end — it is the *mechanism section* explaining why the field spent twenty years optimizing the small factor. What elevates it above an apology (G's criterion, which I adopt): a prior prediction measured before outcomes (the dispersion diagnostic, genuinely held-out regimes), a normative rule that changes practice (comparator construction), a shipped artifact (the strongest static interview + the harness), and a demonstrated power budget (MDE 0.004 against an available oracle gap of 0.029 — we could detect 1/7 of the ceiling and found nothing).

**The thesis contract.** Paper A printed "closing the gap is the method chapter's job." Paper B's resolution: **the gap is closed — by the contract and the consumer, not by selection — and the paper proves that attribution instead of assuming it.** Paper A's "which is asked matters far less than that they are answerable" is *refined*, not contradicted: B splits "selection" into two factors Paper A's sentence conflated (X's M4, the single most important framing fix): the question **set/order** (matters, and interacts with the consumer — the inversion) versus answer-adaptive **branching** (below MDE). Both chapters' sentences survive under that split; examiners must see it made explicit.

---

## 2. THE BETS (ranked by payoff × probability / cost)

| # | Bet | Role | P(survives) | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | **Answer contract in the scarce regime** (sign +0.0229@k2→0@k8; intensity +0.014–0.037 persisting; the unknown/exposure channel ±) | **Headline 1** | ~0.85 (in hand; controls pending) | Low | **BET** |
| 2 | **Non-transfer of strategy orderings across consumer families** (refutes the Rashid-2008/De-Pessemier null) | **Headline 2** | ~0.6 (DiD CIs + controls pending) | Medium | **BET** |
| 3 | **Strongest-static artifact + adaptivity sandwich** (oracle above, KG below, branching < MDE) | Spine / product | ~0.9 for the null; the artifact is unconditional | Medium | **BET** (as method + artifact, not as surprise) |
| 4 | Dispersion GO/NO-GO as pre-registered predictor | Supporting diagnostic | ~0.5 | Low | Support only (X's M10/M11: tautological at extremes; only the off-diagonal cell is science) |
| 5 | C3 information-vs-task-loss boundary | Mechanism, appendix-gated | ~0.4 | Medium | Support; must pass the "surprising number = harness bug" gate first (one experiment, sign-flip history) |
| 6 | Answerability as dominant lever | One factor in the factorial | ~0.9 | Free | Support; it is Paper A's sentence, quantified — never re-headline it |
| 7 | Rank-targeted posterior-dependent criterion | Machinery only | — | — | **DO NOT CLAIM** (0.2). Cite A-GOODE, Yu 2006, Lin 2023, InfoBAX ourselves, first, prominently |
| 8 | Myopic-vs-sequential reframe | One paragraph in §Adaptivity | high | Free | Support (see §5 — the 1-step edge is itself sub-MDE, which sharpens the null) |

Bets 1+2+3 together are the paper. If bet 2's inversions die under DiD CIs (real risk — X's M5: interaction variance ≈ sum of four correlated marginal variances), the paper survives on 1+3 as "The Answer Contract" with non-transfer demoted to an observation; pre-commit to that demotion path in writing.

---

## 3. KEY EXPERIMENTS

Every experiment: paired bootstrap CIs (never the unpaired 0.0069 — the Part-4 lesson is codified as a stated methods rule), Most-Popular gate on every new regime before interpretation, shuffle control, canonical snap, leak check. Per-cell floors and ceilings reported everywhere: prior-mean (no interview), random-answerable, Most-Popular, own full-profile ceiling (X's N17 — effects without dynamic range are uninterpretable).

**E0 — Gates (run first, cheap).**
(i) Freeze the instrument: i26 at end of curriculum, certified against three gates — prior ≈ MostPop (±0.005), one-answer-beats-zero monotonicity, full-profile ≥ 0.335. Pass → i26 is primary, certified tower is the robustness pair. Fail → certified tower primary with the prior-floor caveat printed. (ii) Dispersion diagnostic functional form + numeric threshold τ frozen in a timestamped commit **before** E4/E5 outcomes exist, with two regimes (noisy channel, second dataset) declared held-out and untouched (X's M10 — you cannot pre-register on regimes already seen; in-sample vs held-out prediction reported separately).

**E1 — The factorial, done as a factorial (fixes F1).**
Common reference cell: random answerable questions × binary-positive-only contract × each consumer. Factors: **contract** (A0 binary-positive → A1 +unknown/exposure → A2 +sign → A3 +intensity bands), **question set** (random, popularity, entropy, entropy0, HELF, item-item), **consumer** (≥5 families: Golbandi node model, i26, certified tower, EASE fold-in, Mult-VAE; MostPop as floor), **budget** as a curve k∈{1,2,4,8,16} with a functional-form test, not 16 independent tests (M7). Deliverables: (a) variance decomposition — partial η² or Shapley over {contract, consumer, strategy, budget} + two-way interactions, with bootstrap CIs on each component; **no cross-factor "5–20×" ratio anywhere** — if a ratio is quoted it is best-vs-null for *both* factors at a stated budget, as a curve over budget (M12); (b) the inversion set, pre-registered, each scored by a paired-bootstrap **DiD CI**; any inversion crossing zero is demoted to "no transfer evidence" in print (M5); (c) the Bonferroni arithmetic stated in one sentence — at ~1,200 comparisons the corrected threshold (~0.0031) sits *below* our MDE, so every headline effect survives full correction and the +0.0015 adaptivity number fails even uncorrected (M7 — do the arithmetic so multiplicity stops being an attack surface).

**E2 — Contract ablations with teeth.**
Matched-question design (identical question sequences across contract arms) plus: (i) **popularity-matched question controls** — the k=2 sign effect must survive when early questions are matched on popularity, else it is a popularity-information effect (M8); (ii) **ε-noise curves** — sign-flip rate and intensity-noise sweeps; report the ε that erases each effect, including if it is small; (iii) the exposure/unknown channel framed explicitly as MNAR with ExpoMF/Liang-Hofmann cited as prior art — the +0.0081/−0.0193 unknown-branch result is partly a MNAR re-derivation and must say so (M8); (iv) **the human reliability check: spend the 300 quarantined study users** on intensity/sign response reliability (they are quarantined from training, which is exactly what makes them legal here — this needs the author's explicit sign-off, and any LLM-judging component needs separate approval per project rules); (v) G's **iso-cost frontier** as a sensitivity band, not a claim: k graded answers vs c·k binary answers, exchange rate reported with the break-even cost c₂, even if unfavourable; (vi) one sentence defending sign-valued answers under a protocol whose holdout discards dislikes (N19), citing the PROTOCOL_DISLIKE_DISCARD analysis — the primary protocol does not change.

**E3 — The strongest static interview (the shipped artifact; the never-run decisive fix, finally run).**
Jointly optimize an actor-independent static question set per (consumer × budget) from a common pool; also report random-order, popularity-order, train-fitted-greedy, and oracle-static (best fixed order selected on test — a legitimate upper bound *for the static class*). Then G's **comparator-construction table**: one fixed adaptive policy scored against all four static arms, spanning from a headline-worthy positive to a negative. The normative rule it licenses — *report the gap against the strongest constructible static arm or it is not a gap* — is a citable contribution, and it retroactively re-scores our own demoted +47% and the RL-BOED/DAD 6.7-nat comparator gap. This section is what converts "we failed at RL" into methodology.

**E4 — The adaptivity sandwich (replaces "KG as the correct estimand" — the brief's framing is wrong here, see §5).**
Upper bound: clairvoyant oracle (.2237/.1305 at q16). Lower bounds (each a *policy*, hence lower): KG (correlated-normal, closed-form), the 1-step amortized selector, the D3 one-level tree, all against E3's strongest static, all paired. The width is named the **unrealized band** and reported per budget/channel. Supporting evidence for KG's premise: a posterior-calibration experiment (coverage/PIT of predicted answer distributions per turn), plus KG under a deliberately degraded posterior, so a small gap is attributable (F2's fix — Paper A certified the point estimate, never the second moment; without this, "adaptivity is worthless" and "the covariance is miscalibrated" are confounded). Decision rule: if every realizable policy sits within the MDE of E3's static arm, print the null with the band; no policy result below the MDE is ever narrated as a win.

**E5 — Dispersion diagnostic.** Computed on the **loss-optimal** argmax, not the information argmax (M11 — dispersion of the wrong ranking measures nothing; C3 predicts info-argmax dispersion is degenerate). Falsification conditions stated up front: the diagnostic FAILS if any regime shows dispersion > τ with gap < MDE, or the dispersion–gap relation is non-monotone across the pre-registered regime set; commit to publishing the failure.

**E6 — Second dataset, reduced grid (least negotiable addition for a measurement paper — F3).**
Netflix Prize (graded ratings, comparable Liang-style protocol; Amazon Books as stretch): 3 strategies × 3 consumers × k∈{2,4,8,16} × the contract ladder. Claims are pre-declared to be about **orderings and interactions, not magnitudes** — magnitude portability is not claimed.

**E7 — C3 replication, gated.** Rerun under the harness-bug protocol (7-variables lesson) on ≥2 consumers. Survives → mechanism subsection linking the inversion to "information rankings converge, task-loss rankings don't." Fails → cut without mourning.

---

## 4. BASELINES

- **Golbandi 2011 node model** — the load-bearing baseline: it *beats* our certified instrument on HELF at k≤8, which is the strongest possible answer to "your modern model is just weak" (X's M9); run **with and without** the unknown branch and, per M9, under the ε-noise control — a leaf-lookup under deterministic answers is near-neighbour retrieval of the test user, and the inversion must survive noise to be believed. Its 3^k exhaustion at k=8 is reported as a finding about the tree class, not hidden.
- **EASE fold-in and Mult-VAE** as third/fourth consumer families; a capacity-matched consumer pair (similar full-profile NDCG, different family) as the cleanest inversion demonstration, so inversions can't be saturating-curve level artifacts (M9c).
- **belief-MF / ConTS-class** — one row, one sentence (inert, Spearman 0.995 vs own prior; verified not a harness bug). It is the published Bayesian-elicitation representative and its inertness at interview budgets is itself evidence for the consumer-matters headline.
- **KG** (E4), **RecVAE backbone raw** (collapses at k=2 below the prior, wins at k=16 — both ends reported; its k=16 win over our instrument is printed, not buried), **MostPop / prior-mean / random-answerable** floors everywhere.
- NOT baselines: the 13 RL variants (one seed-averaged appendix table, zero narrative), LLM answer models (banned by firewall), the geometric answer model (cited only as the circularity cautionary example).

---

## 5. THE ADAPTIVITY QUESTION

**Is there an experiment separating "no headroom" from "wrong criterion/trainer"? Partially — and the honest axis is myopic vs. sequential, with one correction to the brief.** The brief proposes KG-adaptive vs KG-frozen as "the CORRECT adaptivity estimand." It is not: KG is a policy, and a policy's gap is a **lower** bound on adaptivity value; writing "we bound realizable adaptivity above by KG" in a TORS abstract is a reject when the AE notices the inequality points the other way (F2). The correct object is the sandwich of E4, and the honest sentence is: *"the value of conditioning on answers lies in a band [best realizable policy, clairvoyant oracle]; every policy we or the literature can construct sits at the bottom of it."*

The myopic/sequential evidence: the 1-step amortized selector trains cleanly and passes the permutation null; the 10-step unroll collapses for diagnosed trainer reasons; KG is one-step lookahead; Viappiani–Boutilier says myopic query selection is (near-)optimal in their scoped setting. But note the arithmetic the brief glosses: the 1-step selector's edge over static is +0.0197 − 0.0171 = **+0.0026, itself below the 0.004 MDE**. So the defensible claim is not "myopic adaptivity works, sequential fails" — it is: *"myopic selection is realizable (trains, passes nulls) and its edge is below our MDE; sequential planning has no demonstrated headroom above myopic, and the per-step routing signal (2.1% of reward variance) explains why policy-gradient methods correctly collapse to fixed orders."* One optional cheap experiment sharpens the trainer-vs-headroom split: re-run the 10-step unroll with the three diagnosed defects fixed (terminal critic, no full BPTT, on-manifold constraint); if it then merely ties the 1-step selector, the ceiling is informational, not optimizational — a clean sentence either way (G's E4b). Cap it at one engineering-week; it is a paragraph, not a section.

**We cannot fully separate the hypotheses, and the paper should say so**: the SNR argument, the theorem for Σ-greedy, the oracle's 14% predictability, and the sandwich width together make "no realizable headroom at this scale/SNR" the best-supported reading, with dataset-scope stated. That is a boundary statement, not a universal impossibility claim — never write "adaptivity doesn't work."

---

## 6. IMPACT OF THE i26 RETRAIN

**Run Paper B on i26 if it passes E0's gates; keep the certified tower as the robustness pair.** Reasons: (a) an interview paper whose own instrument fails "one answer beats zero answers" (prior 0.1279 < MostPop 0.1626) is indefensible; i26 exists to fix exactly that; (b) X's M14 is decisive on one point — **the exposure/unknown result is currently measured against a consumer that structurally cannot consume exposure**; pricing the exposure channel requires i26 (and Golbandi's unknown branch as the second exposure-capable consumer). The exposure headline is hostage to i26 either way; better to hold the hostage ourselves.

**"Strategy ranking is recommender-dependent" gets STRONGER under i26, not weaker** — provided both arms are reported. The epoch-3 signal (entropy0 and popularity cells improve, HELF *worsens*, under the same architecture with a different training curriculum) is a gift: it shows the interaction attaches to the consumer's inductive bias, not merely to architecture family, giving a within-family inversion alongside the cross-family one. Pre-declare (M14): the paper's claims are about **orderings and interactions**; magnitudes are reported per-instrument; any headline cell is run on both i25 and i26 and must be ordering-stable to be claimed. Discipline: i26 must be **frozen and certified before any Paper B table is produced** — no mid-training point estimates ever appear in the paper (the current 0.1986-beats-0.1948 has no interval and is exactly the kind of number the harness-bug rule exists for).

---

## 7. NOVELTY AND RISK

**Verification state:** the two #55 debts are paid (§0.1, §0.2) — but the answer flipped the plan: the criterion is demoted, and three new must-handle citations arrived (Lin WWW 2023, InfoBAX ICML 2021, Huang et al. NeurIPS 2024 amortized decision-BED), plus PERE UAI 2024, vsOED/GO-CBED, Wu & Gardner 2026, Pagano 2017, Rashid 2008, De Pessemier 2021, Kluver & Konstan 2014, Kweon 2020, the SciRep 2025 assertion paper, COPE (arXiv 2607.06765 — our direct opponent: varies strategy, never the recommender; a reviewer will hand it to us if we overclaim "selection doesn't matter"), and the Frolov/Sánchez-Bellogín/Mena-Maldonado line cited in the introduction, not related work. **Residual debts:** Elahi TIST 2014 §5 (paywalled — verify from the library before printing any "single predictor" characterization); Priyogi WSDM 2019 (403'd, unverified); and V flagged that one fetch summarizer *fabricated* a supporting quote for Anava WWW 2015 — every direct quote in the related-work section gets re-verified against a PDF before submission, no exceptions.

**Biggest rejection risk: simulator circularity aimed at the contract headline.** "Your simulator is told the intensity; you are measuring the value of a signal nobody has shown users can supply" (F3/M8). This is the field-wide attack that hit all three papers in the blind calibration panel, and our behavioural firewall (answers only from actually-rated items) is necessary but not sufficient here. Pre-empts, all four: popularity-matched controls, ε-noise curves with the kill-threshold reported, the MNAR framing for exposure, and the 300-user human reliability measurement — the last converts the paper's weakest flank into a unique strength no competing elicitation paper has. **Second risk:** "benchmark paper without a method" — pre-empted by shipping the harness with a documented answer-contract API, the 480-cell grid as a reusable benchmark, and the strongest-static question sets as a deployable artifact (N15). **Third:** "trivial — richer feedback obviously helps" — pre-empted by the surprise being *non-transfer* and the *k-dependence* (sign expires at k=8, intensity doesn't; neither derivable from "dislikes are informative"), and by Golbandi's own "our choice might be suboptimal" doing the motivating for us.

---

## 8. SCOPE AND VENUE

**In:** E0–E6; the Σ-greedy static-equivalence proposition as a theorem with α_refuse = 0.0007 as the measured escape size; the comparator-construction table; the RL corpus as one seed-averaged appendix table; C3 if it survives E7's gate.
**Out / Paper C:** continuous answers and continuous actors (already Paper C's territory per the roadmap), the noisy-channel campaign beyond one robustness row, coarse-to-fine (absorbed into Paper A as geometry; the granularity reversal is artifact-shaped without a mechanism — G and I agree: drop), the concept-channel story, any new architecture, anything LLM.
**Too small to print as findings** (G's list, adopted): Σ-greedy vs random +0.0036; belief-MF inertness beyond one sentence; the 1-step actor as a standalone claim; α_refuse as an "escape."

**Venue: TORS regular paper, primary.** The blind calibration says TORS accepts work at exactly this level; TORS wants artifacts and has room for theorem + factorial + human study; and it keeps A and B as sibling chapters under one editorial culture. **Fallback: ECIR 2027 full paper** (deadline 2 Oct 2026 — tight against E6 and the human study; if targeting ECIR, cut E6 to one dataset-2 contract-ladder table and pre-register the rest). Not SIGIR main, not RecSys main with this framing.

---

## 9. WHERE I DISAGREE WITH THE BRIEF, EXPLICITLY

1. **"KG-adaptive vs KG-frozen is the CORRECT adaptivity estimand"** — no; it is a lower bound and the abstract-level inequality error would sink the paper (§5). The sandwich is the correct object.
2. **Task #53's increment (1) is weaker than the plan of record believes** even though the Shen quote verified — InfoBAX and Lin close most of the gap. The criterion is machinery. Any paragraph presenting it must cite A-GOODE, Yu, Lin, and InfoBAX in its first two sentences.
3. **"Consider whether Paper B should be a negative-result paper"** understates it: the measurement framing is not the consolation prize, it is the larger contribution. The honest small version — "The Answer Contract" alone, if the inversions die under DiD — is still a publishable RecSys/ECIR paper on V's safest-ground verdict. The full version, with non-transfer surviving, is a field-level claim: *twenty years of elicitation-strategy comparisons were conditional on consumers nobody deploys anymore, and the transferable lever was the answer contract all along.* That is bigger than any selector Paper B was ever going to ship — and unlike the selector, every number behind it clears the MDE.

Paper B is smaller than "the very key paper" as a **method** paper, and larger than it as a **measurement** paper. Design accordingly.

---

# ADDENDUM: author challenges adjudicated (2026-07-31)

The author challenged four points. Fable's scoreboard: **the author beat the design document on 1 and 2.**

# ADJUDICATION OF THE FOUR CHALLENGES + PUBLICATION STRATEGY

## CHALLENGE 1 — Sign novelty. **The author wins. Bet 1 survives, restated.**

The author is right and my design document priced this wrong: "sign matters" is not news about methods — Golbandi routes on dislike, EAR/SCPR take no's, twenty years of elicitation systems consume sign. What is news is that **the canonical evaluation stack cannot see it**: Liang's binarisation deletes the negative half of the input contract before any model is scored, so the *price* of sign has never been measurable on the ruler the field actually uses — and when priced, it is the largest per-question lever ever measured on that data in the scarce regime.

Does that collapse to "we fixed an evaluation artifact"? No, for one reason, and it is the asset the author identified: **the protocol demonstrably mis-ranks a published method.** Golbandi loses 0.0067–0.0149 when his dislike branch is merged into unknown — which is precisely the surgery the canonical protocol performs on every method it evaluates. A protocol critique without consequences is a fix; a protocol critique that shows a published method's ranking *inverts* under the canonical harness is a finding, and it is the same genre as bet 2. So the two bets fuse into one headline:

> **The canonical cold-start evaluation stack — binarised protocol plus an assumed consumer — systematically misprices elicitation design choices: it erases the answer contract (mis-ranking Golbandi by up to 0.015) and its strategy orderings do not transfer across consumer families.**

Three supporting facts make this robust: (i) the field is *acting* on the unpriced assumption (the 2025 Scientific Reports paper drops the dislike branch by assertion; Golbandi himself wrote "our choice might be suboptimal" and never ablated); (ii) the sign-aware evaluation line (Frolov, Sánchez-Bellogín, Mena-Maldonado) covers the *target/holdout* side only — the *input-contract* side at interview budgets is unclaimed (verified in the novelty audit); (iii) the k-dependence explains the blind spot mechanistically — sign's value expires by k=8, so full-profile evaluation, which is what the protocol was built for, could never have noticed. Lead the section with the Golbandi mis-ranking exhibit, not with the ablation table.

## CHALLENGE 2 — The adaptivity null as weak-question artifact. **The author is substantially right, and this beats my design.**

I demoted the dispersion diagnostic to "supporting" — that was the wrong call, and I'll say why plainly: my design read the corpus's null as more unconditional than the evidence licenses.

**(a) Is the mechanism plausible given Σ is answer-independent?** Yes — in fact the answer-independence of Σ makes it *sharper*. Since Σ_t is identical for all users, divergence can enter **only through μ_t**. Adaptivity value therefore requires three things simultaneously: answers that move μ differently across users, a criterion that reads μ (task-coupled, not Σ-only), and a bank whose early questions *create* that μ-divergence. Every failed experiment in the corpus lacks at least one leg. The corroborating evidence is already in hand: C3 shows that when you rank by realized task loss (a μ-reader), per-cluster lists go near-orthogonal (ρ=0.062) *and adaptive wins* (+0.0194 tail) — divergence exists and is exploitable when a criterion looks for it. The broad-concept result (+0.0019/+0.0074, oracle-selected concepts median 1320 members) says exactly the splitter-shaped questions still work on the new stack; it was *fine* concepts that reversed. And ADAPTIVE_PROBE's demotion was for its answer model, not its design — the design (partition first, condition second) was never retested behaviorally.

One piece of counter-evidence must be stated honestly, because it constrains the experiment: **D3 already is a one-level partitioning tree and got only +0.0015.** But D3's root was item-channel — items are answered 2.18/8, so most users fall into "unknown" and the partition is degenerate — and its root was greedy-chosen, not divergence-chosen. The untested cell is precisely: *broad-concept opener (answer rate ~6/8), selected to maximise across-user posterior divergence, behavioral answers.* If the cheap experiment below re-runs D3's conditions, it will re-find D3's null and prove nothing.

**(b) Promotion?** Yes — from supporting diagnostic to **load-bearing conditional structure of the adaptivity section**. The claim becomes: *adaptivity pays iff early questions create posterior divergence; under standard banks measured dispersion ≈ 0, which explains the field's nulls and ours; under a divergence-maximising opener, dispersion is X and the gap is Y.* Both branches are publishable: Y ≥ MDE gives Paper B an unexpected positive method result; Y < MDE despite confirmed dispersion is the off-diagonal cell my adversarial reviewer called the only scientifically interesting one — "divergence without exploitability" hardens the null from "we never created divergence" to "divergence exists and cannot be cashed."

**(c) Cheapest separating experiment (≈ days on the existing harness, fully behavioral):**
1. **Dispersion census on existing banks** (the literal task #52, on the *loss-optimal* argmax): establishes the "≈0 under standard banks" leg. Nearly free.
2. **Splitter probe at k=2.** Candidate openers: broad concept questions (genre-scale, ≥1000 members, high answer rate), scored by across-user variance of the induced posterior-mean update, Var_u[Δμ_u], estimated on *training* users' behavioral answers. For the top openers: partition the 10k test users by behavioral answer; per partition, choose the best Q2 by greedy search **on training users in the same partition** (never test-fitted); measure (i) do partitions choose different Q2s (the dispersion leg), (ii) paired gap of [splitter + per-partition Q2] vs the jointly-optimized static 2-question set from E3 — the strongest comparator, non-negotiable given the comparator-construction scar — then continue both arms statically to k=4/8 for persistence.
3. **Controls:** shuffled-answer partition (gap must vanish), popularity-matched openers, MostPop gate, paired CIs. Decision rule pre-registered: gap ≥ 0.004 at k=2 → promote to headline branch; gap < 0.004 with confirmed inter-partition Q2 divergence → print the exploitability null.

**(d) Is the splitter the unclaimed contribution?** Half of it. Be careful here: "ask a partitioning question first" is *not* new as a heuristic — the entire decision-tree line (Golbandi, IGCN, fMF) is partition-then-condition, and Golbandi's root already splits three ways. What is unclaimed is (i) the objective stated as **maximising across-user posterior divergence in a maintained Bayesian belief** — explicitly *not* information (Σ-only, theorem says static) and *not* immediate NDCG (Lin's territory) — and (ii) the **diagnostic law**: dispersion of the induced argmax predicts the adaptivity gap, with both sides measured. Frame the splitter as the *intervention that validates the diagnostic*, not as a flagship new criterion — that keeps us out of the criterion arms race we just lost on novelty grounds (Challenge 4), and it replaces the rank-targeted criterion as the paper's "ours" object far more defensibly.

## CHALLENGE 3 — Answerability placement. **Split the ruling: reiterate in B, do not move from A.**

Moving it out of A orphans A's own printed sentence ("…matters far less than that they are answerable"), which must keep its evidence in-chapter, and guts A's bank-construction sections; A's five-requirement conjunction technically stands (answerability is not one of R1–R5) but the chapter thins badly and the "answerability" coinage loses its home. The author is right, however, that B cannot treat it as foreign territory — it is an elicitation property and it enters B structurally in three places: as a factor in the factorial (answer rate is *why* item and concept banks behave differently), as the mechanism of the exposure/unknown channel, and as a constraint on the splitter (a partitioning opener only partitions if most users can answer it — that is exactly why D3's item-channel root failed). **Ruling: A keeps the definition, the structural rule, and the 2.18-vs-6.02 facts; B owns all *new* quantification — answerability × strategy × consumer interactions, the exposure pricing, and the answerability constraint on splitters — restating the definition with a citation since B must be self-contained anyway.** No contradiction between chapters; B refines, cites, and prices.

## CHALLENGE 4 — The gloss. **Close, but two corrections before the author carries it into writing.**

The proposed gloss — "reduces uncertainty along the directions *separating* the current top-K" — is subtly wrong: "separating" implies score *differences* (contrast directions φ_i − φ_j), but the criterion as derived is L-optimality on the *scores themselves* of the top-K, rank-discount-weighted. A DoE reviewer will parse that word. Second, "stripped of the discount this is A-GOODE" misplaces the increment: A-GOODE already allows arbitrary weights, so the discount alone is just a weight choice; what A-GOODE does *not* have is the re-derivation of the target set from the posterior each turn. Corrected gloss:

> *"Ask the question that most reduces posterior variance of the predicted scores of the currently top-ranked items, weighted by rank discount, with the item set and weights re-derived from the current posterior mean each turn. With the target set held fixed, this is exactly transductive experimental design (Yu et al. 2006) / Bayesian weighted L-optimality (A-GOODE 2018), K=1 being c-optimality; the re-derive-each-turn idea appears without a posterior in Lin et al. (WWW 2023) and without ranking in InfoBAX (ICML 2021); our conjunction — maintained posterior + rank discount + per-turn re-derivation — is the machinery, not the contribution."*

---

# PUBLICATION STRATEGY

**Two papers, not a bundle.** A and B have different reviewers, different failure modes, and different genres (instrument-conjunction vs. measurement); bundling couples their risks, exceeds even TORS length norms, and wastes A's role as B's citable foundation. What *is* shared: the harness/benchmark artifact, released once, cited by both.

**Order and venues:**
1. **Paper A → TORS regular, submit first, arXiv immediately.** Already calibrated (the blind panel put it at the level of two TORS-accepted papers), plan of record stands.
2. **Paper B → branch on the Challenge-2 probe, which should run *before* the writing plan is fixed** (it is days of compute and it decides the paper's shape):
   - **Splitter positive (gap ≥ MDE):** B = protocol-mispricing headline + divergence diagnostic + splitter as validated intervention. That version has a positive method result and belongs at **RecSys 2027 full paper or TORS regular** — it is the noticed-paper version.
   - **Splitter null:** B = protocol-mispricing (Golbandi mis-ranking + contract pricing) + non-transfer + strongest-static artifact + the *conditional* adaptivity null with the dispersion evidence. Still publishable at **TORS regular or ECIR 2027 full** (deadline 2 Oct 2026 — feasible only if the second-dataset grid is trimmed to one contract-ladder table).
3. **Thesis ≠ papers, and they should diverge:** the thesis chapter B keeps everything the paper cuts — the RL archaeology, the full boundary material, the comparator-construction history — because examiners reward exactly the archaeology reviewers punish. The paper presents the working version; the chapter presents the campaign.

**Ceiling:** A at TORS + B at RecSys/TORS with the positive splitter branch, the Golbandi mis-ranking exhibit, and the 300-user human reliability study — a pair that methodologists cite ("the field's ruler misprices elicitation; here is the diagnostic, the price list, and the fix"), and a thesis with a clean instrument→measurement→boundary spine. Realistic, not a moonshot.

**Floor:** bet 2's inversions die under DiD CIs *and* the splitter is null. What remains: contract mispricing with the Golbandi exhibit, the strongest-static artifact, and a hardened exploitability null with both diagnostic legs measured. That is an honest ECIR/RecSys-reproducibility-track paper plus a strong thesis chapter — smaller than hoped, but nothing about it is an apology, and it is *unconditionally reachable from results already in hand*.

**Scoreboard on the challenges:** the author beat the design document on 1 and 2 (restated headline; diagnostic promoted and the probe now gates the writing plan); split ruling on 3; gloss corrected on 4. The revised design is better than the one I delivered, and the single action that most reduces uncertainty about what Paper B is — before any writing — is the k=2 splitter probe.