# Bandits / RL / attribute-asking CRS — what this family actually prices, and which policies port

*Added 2026-08-05. Question asked: "Paper B baseline battery — audit the BANDIT / RL / CONVERSATIONAL-RS
(attribute-asking) family. For each method: (a) what is its reward/metric really measuring, (b) does it
need a known target item or an omniscient simulator, (c) does it report ANY post-interview held-out
ranking metric, (d) is its POLICY portable to our contract even though its NUMBERS are not?"*

**Verification levels: [V]** primary full text/HTML extracted · **[A]** abstract/landing page only ·
**[S]** search summary / a third paper's table. Nothing is quoted unless it came from [V].
Prior agent passes on this project have fabricated citations; every row here carries its level, and the
"do not cite unchecked" list at the end is binding.

**Read this WITH** `adaptive_elicitation_baselines.md` (the rating-elicitation family + DRE/PERE),
`rating_elicitation_family_deep.md` (the parallel 2026-08-05 deep pass on the RATING-ELICITATION family,
which independently found Greedy SLIM and carries the fuller treatment of it — **read its §0 before citing
anything from the elicitation side; this file covers Greedy SLIM only from the bandit angle, because its
adaptive foil "QBandit" is Christakopoulou KDD'16**),
`answerability_in_elicitation.md` (the omniscient-simulator audit table + the CRM/EAR/SCPR/UNICORN/ConTS
verbatim chain), `simulator_circularity_conv_rec.md`, `rank_targeted_acquisition.md`.
Those files settled the *base* CRS-RL papers and the rating-elicitation line. **This file adds three
things they did not have: the ICF/bandit lineage in depth, the CONVERSATIONAL-BANDIT (key-term) family
which was entirely missing, and the 2022–2026 EAR/SCPR/UNICORN successors.**

---

## 0. The headline — three findings, in order of usefulness

**1. There is a whole family we had missed: CONVERSATIONAL BANDITS (key-term bandits).**
ConUCB (WWW'20) → ConLinUCB (AAAI'23) → Hier-UCB (2022) → CLiSK/CLiME (KDD'25) → FedConPE (2024).
These bandits **ask about a key-term (= an attribute / a genre / a tag) OR about an item**, and propagate
key-term feedback into the item-level parameter estimate. That is *literally our item-vs-concept question
channel*, formalised, with regret bounds, and **without a known target item** — the reward comes from
replayed real ratings, not from a target-conditioned oracle. Their metric (cumulative regret) is still not
ours, but this is the family whose **policies port most cleanly and most cheaply**, and one of them
(CLiSK/CLiME, KDD'25) **runs on MovieLens-25M with genres as key-terms**. This family was not in any prior
findings file. It is the single biggest correction of this pass.

**2. The metric incompatibility is real, but the *disqualifying* quotes are better than we had.**
The prior file said "the interactive-RL family prices cumulative reward during the interaction". True, and
now verifiable in the papers' own words on both sides of the family:
- **Bandit side** — PTS (NeurIPS'15) restricts recommendations "to items rated by this user in the dataset"
  [V]; COFIBA (SIGIR'16) "we simulated random choices by the system" with candidate sets of 10–20 items
  [V]. Their regret is only *computable* because the user's best item is known. That is a
  known-target-**equivalent** requirement, self-reported.
- **CRS side** — the candidate item set is not merely "shrinking", it is **seeded by an attribute of the
  target**: MCMIPL "𝒱ₚ₀ is the item set in which all items are associated to attribute instance p₀ which
  initializes the conversation session" [V]; DAHCR "Candidate item set V_cand is formed with the items
  associated with p₀" [V]; CoCHPL the user "indicates a preferred attribute p₀ ∈ 𝒫ᵥ to the CRS" [V];
  VPMCR the same [V]. The reported top-10 is a re-rank *inside a pool the target's own attribute created*.
- **And nobody reports the pool size.** MCMIPL and VPMCR both report **zero** candidate-set-size statistics
  [V] — VPMCR despite over-filtering being its entire motivation. So we cannot even estimate the ratio to
  our 18,430-item catalogue. That absence is the finding.

**3. The strongest new *argument* is that ICF's own metric cannot see an unanswerable question.**
ICF (CIKM'13) defines `θ_hit = 1 if the rating is no less than 4, and 0 otherwise` [V verbatim]. An item
the user never saw scores 0 — **identical to an item the user disliked**. No method in this lineage can
price a burned question, because burning and being wrong are the same event under their metric. That is a
clean, verifiable novelty claim for our "unanswerable questions BURN the budget" contract, sourced from the
founding paper's own metric definition rather than from our reading of its intent.

**4. And the answer to "does ANYONE in 2023–2026 use a post-interview held-out ranking without target
leakage?" is: exactly one paper — GREEDY SLIM (arXiv:2406.06061, §3.4), which we had missed.** It runs on
**unfiltered ML-25M (162,541 users / 59,047 items)**, cold user = a zero row, answers copied from real
held-out ratings, **no item sampling**, **NDCG@10 at k ∈ {5,10,15,20}**, and — the part that matters most —
**a long-tail split defined exactly as ours (Cremonesi 33%-of-ratings short head)**. Its adaptive baseline
is Christakopoulou'16 ("QBandit"), whose NDCG **decreases past k≈10 on both full and tail** while the
*static* Greedy-SLIM questionnaire keeps climbing. That is an independent external replication of the
bank-quality effect our own sweep has been chasing, and a published precedent for the *shape* of Act 1's
result. It is an unpublished preprint (CoRR only, DBLP-verified) — cite it as such.
A ~93-query arXiv+DBLP sweep found nothing else; the cleanest supporting citation is Gupta, Oosterhuis &
de Rijke's own admission that **"no publicly available recommendation dataset that has preference
elicitation interactions"** exists [V].

**Bonus, and it is load-bearing for how we describe the CRS-RL line:** **MINICORN** (arXiv:2206.14468) [V]
uses **no RL at all** — Monte-Carlo-dropout epistemic uncertainty over attribute beliefs, query the
highest-uncertainty attribute, one hand-written rule for ask-vs-recommend — and reports beating SCPR by
"67.40% and 13.39% in recommendation success rate and conversation turns" on LastFM [V]. If a
harmonic-mean-of-two-uncertainty-heuristics beats the RL line by that margin, the RL apparatus is not doing
the work. Cite this whenever we are asked why we did not build an RL policy.

---

## 1. CONVERSATIONAL BANDITS — the family we had missed (key-term = concept questions)

This is the family to engage. All of it: **no known target item**, reward replayed from real ratings,
selection rule = a closed-form index over a ridge/Gaussian posterior, key-terms = attributes.

### 1.1 ConUCB — Zhang, Xie, Li, Lui, **WWW 2020** (arXiv:1906.01219) **[V]**
The origin. Two coupled ridge regressions over a *shared* parameter:

- Arm-level: `θ_t = argmin_θ λ Σ(θᵀx_{a_τ,τ} − r_{a_τ,τ})² + (1−λ)‖θ − θ̃_t‖²` — i.e. the item-level fit is
  **shrunk toward the key-term-level estimate** `θ̃_t`, with λ∈[0,1] the trust weight [V].
- Key-term-level: `θ̃_t = argmin_θ̃ ΣΣ(Σ_a w_{a,k} θ̃ᵀx_{a,τ} / Σ_a w_{a,k} − r̃_{k,τ})² + λ̃‖θ̃‖²` [V].
- **Propagation model** (Eq. 4, verbatim): `E[r̃_{k,t}] = Σ_{a∈A} (w_{a,k} / Σ_{a'} w_{a',k}) E[r_{a,t}]`,
  with `Σ_k w_{a,k} = 1` per arm [V]. *A key-term's reward is the weighted average of its member items'
  rewards* — this is exactly our concept-as-member-aggregate assumption, published in 2020.
- **Key-term selection** (Eq. 8, verbatim):
  `k = argmax_{k'} ‖X_t M_t⁻¹ M̃_{t−1}⁻¹ x̃_{k',t}‖₂² / (1 + x̃_{k',t}ᵀ M̃_{t−1}⁻¹ x̃_{k',t})` [V].
- **Conversation frequency** `b(t)` decides *when* to ask a key-term instead of recommending; tested
  `b(t)=5⌊log t⌋` and `b(t)=5⌊t/50⌋` [V]. (A fixed schedule — the thing every successor attacks.)
- (a) **Cumulative regret**, `R(T) = Σ_t (E[r_{a*_t,t}] − E[r_{a_t,t}])` [V]. (b) **No target item**,
  arms re-drawn per round [V]. (c) **No** held-out ranking. (d) **Policy ports directly.**
- Data [V]: synthetic (5,000 arms / 500 key-terms / d=50 / 200 users); **Yelp** 1,998 users, 25,160
  restaurants, 1,004 categories as key-terms; **Toutiao** 2,000 users, 1,746,335 articles, 2,384 keywords.
- Code: **none in the paper** [V].
- **Verdict: MUST-INCLUDE (policy). The single best-matched adaptive baseline for our concept channel.**

### 1.2 ConLinUCB (-BS, -MCR) — Wang, Liu, Li, Lui, **AAAI 2023** pp. 10288–10295 (arXiv:2303.00315) **[V]**
Fixes ConUCB's two-stage estimation with a **one-step** estimator — and this is the version to implement:

- `θ_t = M_t⁻¹ b_t` with `M_t = Σ_τ x_{a_τ}x_{a_τ}ᵀ + Σ_τ x̃_k x̃_kᵀ + βI`,
  `b_t = Σ_τ x_{a_τ} r_{a_τ,τ} + Σ_τ x̃_k r̃_{k,τ}` [V].
  **In plain terms: a key-term answer enters the SAME normal equations as an item answer, with the
  key-term's feature vector in place of the item's.** That is our signed concept fold, in bandit clothing.
- **ConLinUCB-MCR** selection: `k ∈ argmax_k α_t ‖x̃_k‖_{M_t⁻¹}` [V] — a **pure posterior-width criterion
  over concept directions**. This is a ~20-line addition to our existing `GLMUCBSelector`.
- **ConLinUCB-BS**: draw key-terms uniformly from a precomputed **barycentric spanner** of the key-term
  feature set (a fixed, user-independent exploration basis) [V]. Note this is a *static* concept order
  chosen for spanning — a genuinely different static arm from popularity/entropy, and cheap.
- Regret `O(d√(T log T))`, better than ConUCB's `O(d√T log T)` [V].
- Data [V]: synthetic (5,000/500/d=50/200 users); **Last.FM** 2,000 arms / 2,726 key-terms / 500 users;
  **MovieLens** 2,000 arms / 5,585 key-terms / 500 users. Metric: cumulative regret.
- Code: **https://github.com/ZhiyongWangWzy/ConLinUCB** [V].
- (a) regret · (b) no target · (c) no held-out ranking · (d) **ports directly**.
- **Verdict: MUST-INCLUDE — the cleanest and cheapest concept-arm bandit to port.**

### 1.3 CLiSK / CLiME / CLiSK-ME — Liu, Li, Dai, Lui, **KDD 2025** (arXiv:2505.21393) **[V]**
The current state of the art, and **it runs on ML-25M**.

- Diagnosis: ConUCB's `O(d√T log T)` is **no better than not conversing at all** [V]. Two fixes:
  **CLiSK** = *smoothed key-term contexts*, add Gaussian noise to key-term features then
  `k = argmax_k x̃̃_kᵀ θ_t`; **CLiME** = *adaptive conversation trigger*, ask when preference uncertainty
  is high, selecting key-terms aligned with **small-eigenvalue eigendirections** of the design matrix:
  `k = argmax_k |x̃_kᵀ v_i|` [V].
- Regret: CLiSK `O(√(dT log T) + d)`; CLiME / CLiSK-ME `O(√(dT log T))` [V].
- **ML-25M setup** [V]: `|A| = 5,000` arms (highest rating/tag count) and `N = 200` users (most active);
  **key-terms = movie GENRES**; `d = 50` from truncated SVD of the feedback matrix; binary reward
  "a user's feedback for a movie is 1 if the user's rating is higher than 3, otherwise 0"; `T = 6,000`
  rounds; metric cumulative regret; baselines LinUCB, Arm-Con, ConUCB, ConLinUCB-BS, ConLinUCB-MCR [V].
- **Note the truncation**: 5,000 items / 200 users out of ML-25M. Their numbers are not orderable against
  ours for that reason alone, quite apart from the metric.
- Code: **https://doi.org/10.5281/zenodo.15490021** [V].
- (a) regret · (b) no target · (c) no · (d) **ports**.
- **Verdict: MUST-INCLUDE as the 2025 anchor** ("the current conversational-bandit SOTA is a key-term
  selection rule over a linear posterior; we run it as a question selector"). Implement CLiME's
  eigendirection rule as an optional second concept arm; CLiSK's Gaussian smoothing is the weaker idea.

### 1.4 Hier-UCB / Hier-LinUCB — Zuo, Hu, Yu, Li, Zhao, Joe-Wong, arXiv:2209.06129 (2022) **[V abstract]**
Attacks exactly the thing that hurts us: ConUCB's **pre-defined conversation frequency**. Lets the
algorithm choose *at each round* whether to ask a key-term or recommend an item, "explicitly models the
rewards of these actions", and exploits the empirical finding — from their own survey + real data — that
"key-term rewards are mainly affected by rewards of **representative items**", not by a uniform average
over members [V abstract]. Regret's dependence on the item count is reduced.
**Directly relevant to our concept fold**: it is a published claim that the *uniform member-average*
propagation model (ConUCB Eq. 4, and our centroid) is wrong, and that a few representative items dominate.
Worth testing against our whitened-centroid.
(a) regret · (b) no target · (c) no · (d) ports (the ask-vs-recommend decision does not exist in our fixed-k
contract, but the hierarchical key-term→representative-item reward model does).
Code: not stated [V]. **Verdict: NICE-HIGH — read for the representative-item propagation claim.**

### 1.5 The rest of the family, for completeness
- **Arm-Con** — the conversational-bandit ancestor, from `christakopoulou2016towards` (KDD'16, already in
  our bib): LinUCB extended to initiate conversations directly over the arm set [S].
- **RelativeConUCB** (comparison-based / relative bandit feedback) and **ConDuel** — named in the CLiSK
  and FedConPE lineage lists [S]; **not verified, do not cite without checking.**
- **FedConPE** — Li, Liu, Lui, arXiv:2405.02881 (2024) [V]. Federated conversational bandits, phase
  elimination, per-client **G-optimal design**, server picks key-terms to fill low-eigenvalue directions;
  regret `Õ(√(dMT log(KM log T/δ)))`; synthetic + **ML-25M** + Last.fm + Yelp. Names the lineage as
  Christakopoulou'16 → Sun&Zhang'18 → Zhang'20 (ConUCB) → Wang'23 (ConLinUCB) [V].
  **Verdict: UNFIT (federated axis is orthogonal) — cite once as evidence the family is active in 2024–26.**

---

## 2. THE ICF / interactive-CF BANDIT LINEAGE (Zhao CIKM'13 and descendants)

### 2.1 Zhao, Zhang & Wang — Interactive Collaborative Filtering, **CIKM 2013** **[V]**
`zhao2013interactive`, already in our bib as a one-line row. It is **four policies, not one**, and it is the
direct structural ancestor of our setup: PMF backbone, item factors frozen at MAP `ν_i`, Gaussian posterior
over the user vector only.

- Posterior: `Σ_u = σ²(D_u'D_u + λ_p I)⁻¹`, `μ_u = (D_u'D_u + λ_p I)⁻¹ D_u' r_u` (`D_u` stacks the item
  vectors already rated) [V].
- **Alg. 1 Thompson**: sample `p̃_u ~ N(μ_u, Σ_u)`, `q̃_i ~ N(ν_i, Ψ_i)`, pick `argmax_i p̃_u'q̃_i` [V].
- **Alg. 2 LinUCB**: `argmax_i (μ_u'ν_i + α√(ν_i'Σ_u ν_i))`, `α = 1 + √(ln(2/ζ)/2)` [V].
- **Alg. 3 GLM-UCB**, two links — **GLM-Lin** `ρ(x)=x` and **GLM-Sig** `ρ(x)=1/(1+e⁻ˣ)`;
  `argmax_i (ρ(p̂_u'ν_i) + c√(log t)·‖ν_i‖_{Σ_u})`, `p̂_u` from the quasi-likelihood root
  `Σ_τ (r_{u,i(τ)} − ρ(p̂_u'ν_{i(τ)}))ν_{i(τ)} = 0` [V]. **Alg. 4 ε-greedy.**
- Global rule: "the same item should not be repeatedly recommended" [V].
- (a) **Cumulative** precision@T / recall@T / nDCG over the *recommended sequence*, T∈{10,20,40,120} cold /
  {60,80,100,120} warm; `θ_hit = 1 if the rating is no less than 4, and 0 otherwise` [V].
- (b) No single target item, but it needs ground-truth ratings for whatever it recommends, and **the paper
  never states how that is guaranteed** [V — the text is silent]. In practice the pool is the rated set.
  **That silence is itself citable: the lineage's founding paper never specifies its answerability contract.**
- (c) **No** post-interview held-out ranking.
- (d) **Ports trivially** — this IS our architecture (frozen item factors, Gaussian user posterior, cold
  user, budgeted rounds). **MUST-INCLUDE despite unusable numbers; say so in the paper.**
- Data [V]: MovieLens 943u/1,683i/100k; EachMovie 72,916u/1,648i/2.81M; Netflix 480,189u/17,770i/100.5M.
  ML cold-start cumulative precision@120: Random 8.420 · Pop 42.170 · SVD 43.155 · **AL 41.785** ·
  **Interview 45.745** · TS 43.745 · ε-greedy 43.180 · **LinUCB 46.455 · GLM-Lin 46.470 · GLM-Sig 46.445**.
  EachMovie SVD 31.435 → GLM-Lin 35.045 (+11.5%); Netflix SVD 48.900 → LinUCB 56.490 (+15.5%) [V].
- **Two things worth carrying.** (i) **TS LOSES to LinUCB/GLM in the lineage's own founding experiment**
  (43.745 vs ~46.4) — contrary to how it is usually cited, and relevant because our roster has both.
  (ii) The table already contains **`AL` and `Interview` baselines**, and ICF beats `Interview` by only
  **1.6%** (45.745 → 46.47). If `Interview` is a Golbandi-style tree that row matters to Paper A.
  **UNVERIFIED: the identity of ICF's `AL` and `Interview` baselines — chase this.**
- Code: no official repo. Only `karapostK/Interactive-Collaborative-Filtering-` (2 stars, last push
  2021-12-13) [V] — a study repo, treat as no code.
- Effort: **3–5 agent-hours** for all four policies over `belief_mf.py` (+1–2h for GLM-Sig's Newton solve).

### 2.2 Filippi, Cappé, Garivier & Szepesvári — GLM-UCB, **NeurIPS 2010** **[V partial]**
Model `E[r_t|m_t] = μ(m_tᵀθ*)`; estimator solves `Σ_s (r_s − μ(m_sᵀθ̂_t))m_s = 0` [V verbatim];
design matrix `M_t = Σ_s m_s m_sᵀ`; index `argmax_m [μ(mᵀθ̂_t) + β_t‖m‖_{M_t⁻¹}]`; constants
`c_μ = inf_{θ∈Θ,‖a‖≤1} μ̇(aᵀθ) > 0` and `k_μ` the Lipschitz constant of `μ` [V via Russac et al.
arXiv:2011.00819]; a **projection step** returns the MLE to the admissible set when it escapes [V] — the
part most reimplementations silently drop. Regret Õ(d√T) [A].
**⚠ UNVERIFIED — do NOT cite:** the exact closed form of `β_t`/`ρ(t)`. An OFUL-style
`√(d log(1+t/λ)) + √λ‖θ*‖` surfaced during extraction and is believed **confabulated**; it was rejected.
Also UNVERIFIED: Filippi's own experiments/datasets. **Practical consequence: none — the form the recsys
lineage actually uses is ICF's `c√(log t)·‖ν_i‖_{Σ_u}`, which IS verified.**
(a) cumulative regret, theory only · (b) N/A · (c) no · (d) ports only *through* ICF.
**Verdict: NICE — cite as provenance for ICF's GLM arm. UNFIT standalone.**
*(Note: our `bandit_selectors.py` header already states this lineage correctly and honestly names the
identity link. Nothing to fix.)*

### 2.3 Li, Chu, Langford & Schapire — LinUCB, **WWW 2010** (arXiv:1003.0146) **[V]**
`A_a ← A_a + x x'`, `b_a ← b_a + r x`, `θ̂_a = A_a⁻¹b_a`,
`a_t = argmax_a (x_{t,a}'θ̂_a + α√(x_{t,a}'A_a⁻¹x_{t,a}))`, `α = 1 + √(ln(2/δ)/2)` [V].
(a) CTR vs a random bucket. (b) **Requires logged uniformly-random traffic** — the `Policy_Evaluator`
replayer's unbiasedness (Thm 1) depends on it [V]. **We have no logged bandit traffic, so their protocol is
unavailable to us in principle, not just in practice.** (c) no. (d) ports — but note the **structural
transposition**: in WWW'10 the user is context and each arm owns a `θ_a`; in ours there is one frozen item
factor per arm and one posterior over the user. **ICF Alg. 2 already performs that transposition — port
ICF's LinUCB and cite WWW'10 as origin.**
Data [V]: Yahoo! Front Page, >33M events, 1–9 May 2009. Hybrid LinUCB **+12.5% click lift over ε-greedy**,
and the margin *grows* as data is subsampled to 1–30% — a usable citation for the cold-start framing.
Effort 1–2 agent-hours (subsumed). **Verdict: MUST-INCLUDE as citation + the ICF-Alg-2 policy; UNFIT as an
experimental comparison in its own protocol.**

### 2.4 Chapelle & Li — Empirical Evaluation of Thompson Sampling, **NeurIPS 2011** **[V citation, A content]**
Beta–Bernoulli TS for binary rewards; logistic TS with a Laplace-approximated Gaussian posterior [A].
Uses the replay method → same logged-randomisation requirement. No held-out ranking.
**Verdict: NICE — cite as *why* TS is a mandatory baseline (the paper's own thesis); do not implement
separately, ICF Alg. 1 is it.** (UNVERIFIED: the delayed-feedback section's numbers.)

### 2.5 PTS / PTS-B — Kawale, Bui, Kveton, Tran-Thanh, Chawla, **NeurIPS 2015** **[V]**
Rao-Blackwellised particle filter over `D` particles each holding `(V, σ_U)`; `U` marginalised analytically;
per round sample a particle → sample `Ũ_i` → `argmax_j Ũ_i'Ṽ_j` → reweight `w_d ∝ Pr(R_ij=r|·)` → resample
→ Gibbs-update `U`,`V` inside the particle. Cost `O(((N̂+M̂)K² + K³)D)` [V]. PTS-B puts `Γ(α,β)` priors on
the precisions.
- (a) Expected **cumulative regret** + offline MSE [V]. (b) **YES, and this is the money quote:** the
  protocol constrains recommendations "**to items rated by this user in the dataset**" [V] — regret is only
  computable because the user's best item `j*(i)` is known. **A known-target-equivalent requirement, and it
  also means PTS never faces an unanswerable question: by construction every arm is answerable.**
  (c) no held-out ranking; offline MSE only — ML-1M PTS **0.7925** vs PMF **0.7718**, i.e. *worse* than
  batch PMF at reconstruction [V]. (d) Policy portable but **pointless for us**: its extra power is
  learning `V` online, which our frozen certified tower forbids; with `V` frozen PTS collapses to ICF TS.
- Data [V]: ML-100k, ML-1M, Yahoo Music (15,400/1,000/311,704), Book-Crossing, EachMovie. Beats Random,
  MostPopular, **ICF-20, ICF-50**, SGD-Eps on cumulative regret. Theorem 1 is rank-1 + discretised only.
- Code: **UNVERIFIED** (no official repo found).
- **Verdict: MUST-INCLUDE as a CITATION (the strongest self-reported disqualification exhibit in the
  lineage); UNFIT to implement (collapses to ICF TS under a frozen tower).**

### 2.6 COFIBA (SIGIR'16) / CLUB (ICML'14) — Li, Karatzoglou, Gentile; Gentile, Li, Zappella **[V]**
arXiv:1502.03473 and arXiv:1401.8257. Multiple user-side graphs (one per item cluster) + an item-side
graph; clusters = connected components; ridge updates per served user; cluster-aggregate
`w̄ = M̄⁻¹b̄`; select `argmax_k (w̄'x + α√(x'M̄⁻¹x·log(t+1)))`; delete edge `(i,j)` when
`|w_i'x̄ − w_j'x̄| > cb_i(x̄) + cb_j(x̄)` [V]. CLUB = the same, user-side only.
- (a) streamed **CTR** (regret is theoretical only, never reported empirically) [V].
- (b) **YES on 2 of 3 datasets** [V]: on Telefónica and Avazu "we simulated random choices by the system",
  candidate sets of 10–20 random items plus the served item. Only Yahoo! had genuine random logging.
- (c) no. (d) **NOT portable in the way that matters**: these are *warm, multi-user, streaming* methods
  whose whole mechanism is sharing statistics across users who already have interactions. At `t=0` a cold
  user is a singleton component and COFIBA **degenerates to LinUCB**.
- **Terminological trap worth naming in the paper:** COFIBA's "cold-start" means *early in the stream*, not
  *a user with zero history*. The bandit literature routinely uses the first sense.
- **Verdict: UNFIT to implement; NICE to cite** for the cold-start-terminology point and the self-reported
  simulated candidate sets.

### 2.7 factorUCB — Wang, Wu & Wang, **AAAI 2017** **[V partial]**
`r_{a,u} = x_a'Θw_u + v_a'Θw_u + η`, i.e. observable item features **plus** online-learned latent item
factors, with a user-influence graph `W`; ALS updates `vec(Θ̂_t) = A_t⁻¹b_t`, `v̂_{a,t} = C_{a,t}⁻¹d_{a,t}`;
UCB bonus on **both** the user- and item-factor uncertainty [V].
(a) normalised CTR + regret bound. (b) synthetic simulator + Yahoo logged data. (c) **UNVERIFIED** — one
extraction claimed precision/recall on ranking tasks; **low confidence, do not claim it.** (d) partly, but
its contribution (learning item factors online) is exactly what our certification forbids, and its
user-dependency graph is inapplicable to a cold user with no graph position.
Data [V]: Yahoo clickstream 45.8M visits / 10 candidate articles / 6-dim features; LastFM 1,892u/17,632i.
Code UNVERIFIED. **Verdict: NICE — cite as the "learn item factors online" branch our scope closes off.**

### 2.8 IGCF — Zhang, Lian, Xie, Li, Chen, arXiv:2309.01453 (2023) **[V mechanism, numbers UNVERIFIED]**
LightGCN embeddings → Gaussian priors on user/item embeddings → **variational inference** with
diagonal-Gaussian posteriors (reparameterisation + MC sampling; handles binary feedback) → **Bayesian
meta-learned prior**: the initialisation distribution for a new user is a *meta-distribution* whose mean and
covariance come from the existing user population → **Bayesian LinUCB** selection [V].
- **The meta-learned prior is the single most transferable idea in this whole file.** Instead of `N(0,σ²I)`
  for a cold user, initialise at the **empirical mean and covariance of the trained user population**.
  Our `belief_mf.GaussianBelief` **already does exactly this** (`mu_0, Sigma_0 = mean/cov of TRAIN user
  factors`, per the module header) — so this is not a to-do, it is a **published provenance for a choice we
  already made**, and it should be cited as such rather than re-implemented.
- (a) cumulative reward + Bayesian regret. (b) not discussed [V] — assume the lineage's rated-item
  restriction, UNVERIFIED. (c) no. (d) ports.
- **UNVERIFIED: datasets, baselines, all numbers; and whether a SIGIR'23 version exists — the arXiv metadata
  says journal "Artificial Intelligence". Do not cite an IGCF number or venue.**
- **Verdict: NICE-cite (as provenance for our population prior + the most recent ICF successor). Do not
  implement the full method — it requires unfreezing the tower.**

### 2.9 Other ICF successors — brief
- **Wang et al., "Online ICF Using MAB with Dependent Arms"**, arXiv:1708.03058, IEEE TKDE 2018 [V]. Item
  dependencies via topic modelling + particle learning. **NICE-cite** (the "arms are not independent"
  branch — our concepts already supply that structure). Numbers UNVERIFIED.
- **MV-ICTR**, Lentini & Thayasivam, arXiv:2305.18306 (2023) [V]. Multi-view ICF, explicitly **not**
  frozen-item-factor. NICE-cite, low priority.
- **FeedRec**, Zou et al., arXiv:1902.05570, KDD'19 [V]. Hierarchical-LSTM Q-network **plus an S-Network
  that simulates the environment** — a learned user simulator. **UNFIT**, falls under our existing
  simulator disqualification; cite alongside EAR/SCPR.
- **Pseudo Dyna-Q (WSDM'20)** — **NOT VERIFIED. Not on arXiv** (all 32 arXiv entries for Lixin Zou were
  enumerated and it is absent [V]); DOI fetch returned 403. **Do not cite a DOI for it.** From the title it
  is a Dyna-style world-model method (learns a simulator) — same disqualification, but that is inference.
- **Unaudited gaps:** "Efficient online recommendation via low-rank ensemble sampling" (RecSys'18);
  "ICF with knowledge graphs" (found nothing under that name — treat as *not found*, not *does not exist*).

---

## 3. THE QUESTION-BASED RS LINE

### 3.1 Qrec — Zou, Chen & Kanoulas, **SIGIR 2020** (arXiv:2005.14255) **[V]**
Entities extracted from item descriptions/reviews via TAGME (p>0.1); question template *"Are you seeking
for a [entity] related item?"*; selection by **Generalized Binary Search** — the entity minimising
`|Σ_d (2·1{e_d=1} − 1)·π*(d)|`, i.e. split the belief mass in half; belief = Dirichlet over items with `α`
initialised from an offline ranker; a **QMF** matrix-factorization updated online by closed-form ALS mixing
historical ratings with QA feedback weighted by γ [V].
- (a) Recall@5 / AP@5 / NDCG / MRR **against the known target item**. (b) **YES, omniscient, verbatim [V]:
  "We assume that the user will respond to the questions with full knowledge of whether an entity is
  present or not in the target item. Hence, we assume that the user will respond with 'yes' if an entity is
  contained in the target item documents and 'no' if an entity is absent."** (c) No held-out ranking —
  the ranking IS of the target. (d) **The GBS half-splitting acquisition ports** (it is a belief-entropy
  criterion, and a legitimate ablation next to our greedy); the protocol does not.
- Data [V]: Amazon "Home and Kitchen" 9,124u / **557 items** / 10,108 ratings; "Pet Supplies" 2,248u /
  **2,475 items**. **557 items.** At that catalogue size, 10 questions Recall@5 = 0.943 [V] vs NeuMF 0.222.
  Do not put that number anywhere near an 18,430-item NDCG@10.
- Code: https://github.com/JieZouIR/Qrec [V].
- **Verdict: UNFIT (omniscient answerer + toy catalogue). Cite the GBS acquisition; optionally run
  "GBS acquisition, behavioural answers" as one ablation row, exactly as we scoped PEBOL.**

### 3.2 Q&R — Christakopoulou, Beutel, Li, Jain, Chi, **KDD 2018** **[A]**
Two-stage: an RNN generates a **topic** the user might be interested in; the user's chosen topic is fed to
an extended RNN video recommender. Evaluated **on live YouTube traffic** (User Onboarding, Homepage,
Notifications), ">1% increase in video notifications opened"; deployed in production [A].
This is the **closest published thing to a one-round concept question followed by a ranking**, and it is
industrial. **VERIFICATION DEBT: the offline protocol and metrics are UNVERIFIED — the ACM/Google PDF would
not extract (a wrong PDF came back on one attempt).** Do not state its offline metric.
**Verdict: NICE (cite as the industrial precedent for topic/concept questions); UNFIT to run (no data).**

### 3.3 ★ Greedy SLIM — Proissl, Vatić & Waldschmidt, arXiv:2406.06061 (11 Jun 2024) **[V, full text]**
**The single most contract-matching paper found in this entire pass, and it was missed by the Aug-4 sweep.**
It is a *questionnaire* paper that reports **NDCG@10 on ML-25M after a k-question interview**, which is
almost exactly our protocol — and it uses **Christakopoulou et al. 2016 (`christakopoulou2016towards`,
already in our bib) as its main baseline under the name "QBandit"**, which is why it belongs in this file
rather than the elicitation one.

- **Mechanism.** SLIM (`min ‖X − XW‖²_F + λ_F‖W‖²_F + λ₁‖W‖₁`) is unsuitable for elicitation as normally
  trained; they propose **Greedy SLIM**, which computes the SLIM matrix **row by row, greedily choosing the
  next row/item to minimise the SLIM loss** [V]. The questionnaire `QGSLIM` = the first k rows chosen.
  **The questionnaire is STATIC** ("As our questionnaire is static, we can compute the SLIM matrix in a
  preprocessing step") [V] — so this is a *static* bar, alongside DRE.
- **Protocol [V], and it is ours in all but the split.** Random 90/10 user split; for each test user, a zero
  row is added to `X_train`, the questionnaire is called iteratively, answers are copied in, then the
  recommender ranks. **NDCG@N / Precision@N / Recall@N**, gain `2^{x_ui} − 1`. `I_R ⊆ I` is the pool the
  recommender may draw from — **and it EXCLUDES the asked items**, which they identify as the reason
  QBandit's NDCG *decreases* after ~10 questions ("questionnaires are not allowed to recommend items that
  they contain as questions… using popular items as questions may have a negative impact") [V].
  **This is a live design decision for us: do we exclude asked items from the candidate pool?** They do; it
  penalises popularity-heavy questionnaires. Worth checking what `run_interview_table.py` does.
- **No sampling, full catalogue [V]:** "we do not apply any kind of item sampling to avoid the problems
  reported in Krichene and Rendle"; candidates are all items with `x_ui = 0`.
- **Datasets [V]: MovieLens-25M and Netflix, UNFILTERED** — "We do not filter these sets to ensure that we
  do not accidentally introduce any biases." **ML-25M at full size: 162,541 users / 59,047 items /
  25,000,095 ratings**; Netflix 480,189 / 17,770 / 100,480,507. Questionnaire size up to **20**; budgets
  reported at **k ∈ {5, 10, 15, 20}**.
- **★ AND IT REPORTS A TAIL SPLIT, defined the way we define ours [V]:** "Following Cremonesi et al. (2010),
  we refer to the set of most popular items that represent **33% of the ratings** as short-head items. The
  remaining items are called long-tail items… the questionnaires are still allowed to contain short-head
  items, of course, but the **top-N recommendation is restricted to long-tail items**." That is our
  full-and-tail pair, on ML-25M, from an independent group.
- **Numbers [V]** — ML-25M NDCG@10, QGSLIM vs QBandit, ALL items and LONG-TAIL:
  | k | ALL: QGSLIM | ALL: QBandit | TAIL: QGSLIM | TAIL: QBandit |
  |---|---|---|---|---|
  | 5 | **0.3390** | 0.3382 | 0.1163 | **0.1183** |
  | 10 | **0.3594** | 0.3456 | **0.1543** | 0.1465 |
  | 15 | **0.3709** | 0.3221 | **0.1674** | 0.1488 |
  | 20 | **0.3752** | 0.2916 | **0.1779** | 0.1466 |
  (Netflix NDCG@10 at k=5: QGSLIM 0.2859 vs QBandit 0.3030; at k=10: 0.3042 vs 0.3251 — QBandit wins there.)
- **★ The most useful single fact in this paper: QBandit's NDCG *DECREASES* past k≈10 on BOTH axes**
  (ALL 0.3456 → 0.2916; TAIL 0.1465 → 0.1466 flat then down), and the authors attribute it to popular
  question items being barred from the recommendation pool [V]. **That is an independent external
  replication of the bank-quality effect our own concept-vs-item bank sweep has been chasing** — a
  popularity-heavy question bank stops paying and then costs. Cite it.
  Their own summary [V]: "QBandit seems to perform slightly better than QGSLIM if the questionnaire consists
  of very few questions… QGSLIM is able to profit from every additional question, which leads to favorable
  results if the questionnaire consists of ten or more questions." **A published budget-dependent crossover
  between two elicitation strategies on ML-25M — directly relevant to our k∈{1,2,4,8,16} sweep, and note the
  direction: the STATIC method wins at long budgets and the ADAPTIVE bandit wins only at very short ones.**
  ⚠ **Do NOT put these numbers beside ours.** Their NDCG uses graded gain `2^r − 1` on raw 1–5 ratings (we
  are binary implicit), their split is a random 90/10 user split with *everything unasked* as the target set
  (not Liang's 80/20 fold-in/eval), and their candidate pool excludes asked items. Same *shape* of number
  and the same tail definition — **different information contract, so not ordered against 0.3496/0.2491.**
- **Baselines** [V]: QBandit (= Christakopoulou 2016, with **PureSVD** as the LFM), plus their own QPop
  (most-rated), QVar (Golbandi entropy×popularity trade-off), RGain, QGreedy.
- **A reproducibility datapoint we should record and can cite:** "For Golbandi et al. [2011] and
  Sepliarskaia et al. [2018] we, unfortunately, **obtained poor results**. We therefore decided to not
  include them in our official experiments" [V] — an independent 2024 group failed to reproduce Golbandi
  usefully. Also: "the **reproducibility issue** within the recommender systems community is also an issue
  when it comes to questionnaires. **We had to implement all the baselines we wanted to consider on our
  own**" [V]. That is a *citable* statement of exactly the hole our baseline battery fills.
- **User study** [V]: 10 movies rated on 1–5 **with an explicit "don't know" zero-rating option**, then 10
  recommendations judged (bad / good / very good / don't know). Small, authors say "might not be
  representative" and explicitly ask for help repeating it at scale. **Their zero-rating IS our
  unanswerable branch**, in a human study, in 2024.
- **Cost caveat [V]:** 19 minutes *per SLIM row* on ML-25M (9 min on Netflix); 20 rows = hours. They scan
  every item as a candidate for each row. For k≤16 this is ~5 hours of compute, which is affordable.
- **Code: OSF anonymous view-only link in the paper —
  `https://osf.io/myh2q/?view_only=a9a747dee8704203ae594ae0a8cc88f8`** [V link present; **not opened,
  language/framework UNVERIFIED**]. **Venue: CoRR/arXiv ONLY** — DBLP has exactly one record,
  `journals/corr/abs-2406-06061`, venue "CoRR", 2024; **no conference or journal version** [V, DBLP API].
  **It is not peer-reviewed — say so in any citation.**
- **Answerability: implied by construction, never stated offline.** "copy the user-item interaction `r_ui`
  from `X_test`" means an unrated asked item copies nothing while consuming a turn, and QGSLIM is a static
  top-k list asked of everyone — so the burn is structurally forced. But the paper never analyses it.
  Its limitations paragraph is the closest [V]: "the information contained in `X_test` may be incomplete
  because the users do not rate all the items they know. Second, a real new user most likely knows less
  items than the users contained in `X_test`. We found that these points **might have a severe impact on the
  results**, which is why we additionally conducted an online user study."
  **Frame it precisely: they IDENTIFY unanswerability as the threat to their own offline protocol and go
  online to address it; they do not price it. That is a gift for our answerability argument, not a rival.**
- (a) **NDCG@10 AFTER the interview** — the only method in this whole file that prices our quantity.
  (b) **No target item, no simulator** — answers are copied from the real interaction matrix. (c) **YES.**
  (d) Ports fully.
- **Verdict: MUST-INCLUDE.** It is a static ML-25M NDCG@10 questionnaire bar that we did not know existed,
  it names Christakopoulou'16 as the bandit questionnaire to beat, and its budget-dependent crossover is a
  published precedent for the shape of result Act 1 is measuring. Effort: **~12–20 agent-hours** (greedy
  row-wise SLIM + the questionnaire; the loss is a plain elastic-net least-squares per row).

### 3.4 NICF / RLT4Rec — already audited in `adaptive_elicitation_baselines.md`
One correction to record: **RLT4Rec's authorship is now [V] — Dilina Chandika Rajapakse and Douglas Leith,
arXiv:2412.07403, 10 Dec 2024.** Abstract verified verbatim: transformer over the user's (item,rating)
history → next item, "no need to input a state observation or estimate". **Datasets, metrics, session
length and code remain UNVERIFIED** (the HTML build 404s and the PDF would not extract).

---

### 3.5 THE HARD SEARCH — "do ANY 2023–2026 CRS/elicitation papers use a post-interview held-out ranking with no target leakage?"

**Answer: essentially no. Exactly ONE paper satisfies the full contract — Greedy SLIM (§3.3) — and it is an
unpublished preprint whose own authors distrusted it enough to run a user study to check.**

Coverage behind that negative (so it can be judged): ~93 distinct searches — **43 arXiv API abstract
queries** (≥2022-06, sorted by date) and **50 DBLP publication-API queries**, plus DBLP TOC fetches for
RecSys 2025/2024 and SIGIR 2025. **WebSearch was unavailable (session budget 200/200 exhausted) and ACM DL
returns 403 to automated fetches**, so SIGIR 2024/2026, WSDM, WWW, TORS, UMAP and CIKM were reached only by
DBLP keyword search, not by reading TOCs end to end. **Known hole:** `"cold start" + "interview" +
"recommender"` returns **zero** arXiv abstracts since 2023 — the Golbandi "interview" vocabulary is dead, so
a 2025 paper doing our contract under another name could be missed.

**The claim to make in the paper:** across those searches, the number of 2023–2026 papers that (i) end the
interaction, (ii) rank the full unsampled catalogue, (iii) score NDCG on *held-out* items, (iv) with answers
from real ratings and no known target, and (v) report a tail split, is **one**. Every other 2023–2026
elicitation paper substitutes one of: a small filtered catalogue (≤5k items), a target-aware LLM simulator,
a success-rate/regret/coverage metric, a text-reconstruction metric, or a user study.

**The best supporting citation is someone else's admission, not ours** [V]: Gupta, Oosterhuis & de Rijke
(`gupta2024pebias`, CONSEQUENCES@RecSys 2023 / arXiv:2405.00554) had to *simulate* elicitation because
**"no publicly available recommendation dataset that has preference elicitation interactions"** exists.
That is the gap, stated by a UvA/Radboud group, and it is the cleanest justification for why our contract
had to be built rather than adopted.

**Near-miss worth a paragraph of related work — ICER / MD-DICER (Google DeepMind), on OUR dataset:**
Ryu, Hsu, Chow, Ghavamzadeh & Boutilier, "Synthetic Dialogue Generation for Interactive Conversational
Elicitation & Recommendation", **GENNEXT@SIGIR'25 workshop, arXiv:2510.02331** [V]. 100K CRS dialogues built
on **MovieLens-25M**, restricted to **2,109 movies with ≥1,000 ratings** and 3,650 users with ≥50 ratings;
7-turn budget; EVOI-style item *and* attribute queries; dataset public on Kaggle (`googleai/md-dicer`).
**But: no full-catalogue held-out ranking.** Evaluation is human raters on eight dialogue-quality dimensions
plus an LM ranking **two-item pairs** (accuracy and NDCG *on pairs*). The answer model is a learned
choice/response model with access to ground-truth CF user embeddings — a simulator [V].
**Verdict: UNFIT as a comparator; NICE as evidence that a Google-scale team, on our exact dataset, with an
EVOI item-and-attribute query design, still declined to report post-interview catalogue NDCG.**

**PEPPER venue correction:** `kim2024targetfree` is **EMNLP 2025 Findings** (arXiv:2411.16160v4), not an
arXiv-only preprint. It *does* report Recall/NDCG/MRR@{20,50} alongside Preference Coverage, but on
ReDial/OpenDialKG-scale catalogues (~3.3k users) with gpt-4o-mini as simulator and GPT-4o as judge — so it
stays **MUST-CITE for the leakage argument, UNFIT as a comparator.** The practical consequence: *stop
carrying the "target-biased simulator is a guessing game" claim on our own authority — it is now published.*

**Other 2023–2026 candidates checked and rejected, with the reason:** Montazeralghaem et al.
arXiv:2510.12015 (MovieLens, 10 questions — but scored with **ROUGE/BLEU on reconstructed profiles**, and
the LLM simulator is given the ground-truth profile); Hsu et al. arXiv:2409.17436 (proprietary YouTube-Music
logs, metric is agreement with live A/B, no NDCG); Harne et al. arXiv:2606.24194 (simulator "accesses the
full product description and profile" of the target; ~2,000 items pre-filtered to top-200);
Tran et al. arXiv:2603.11399 (150 LLM personas holding explicit ground-truth preferences);
Boroomand & Wright arXiv:2508.14911 (**pretrained with 55 comparisons per user — not cold**);
Bose et al. ICLR 2026 arXiv:2602.15012 (T=5 questions but MedQA/AIME domain, LLM-judge rubric score);
Wang et al. WWW'25 Companion arXiv:2412.06949 (Reddit-ML32M, 22,014 items, full-catalogue NDCG@k — but
ground truth is items **named inside the conversation**, i.e. exactly the He CIKM'23 circularity we already
documented). **[S]-level, unfetched, ACM-DL-blocked:** Wang et al. "Towards Efficient Conversational
Recommendations: EVOI Meets Bandit Learning", **WWW 2025, 10.1145/3696410.3714773** — by title and venue it
completes the EVOI×bandit thread and **should be fetched manually**; UMAP 2026 "Investigating Preference
Elicitation Strategies in CRS" (10.1145/3774935.3803059); CIKM'24 SAQRec; SIGIR'25 "Action First";
WSDM'23 "Meta Policy Learning for Cold-Start CRS"; CHIIR'26 "Expanded Tag Genomes".

---

## 4. THE ATTRIBUTE-ASKING CRS LINE — 2022–2026 successors

Base papers (CRM'18, EAR'20, SCPR'20, UNICORN'21, ConTS'21) are audited in
`answerability_in_elicitation.md` with verbatim simulator quotes. **Not one successor drops the known
target item.** The innovation budget went entirely into the policy; the contract never changed.

| Method | Cite | Mechanism (1 line) | (a) metric | (b) target/omniscient? | (c) held-out rank? | (d) policy ports? | Code / stack | Verdict |
|---|---|---|---|---|---|---|---|---|
| **CRIF** | Hu et al., SIGIR 2022, 10.1145/3477495.3531844 [A] | Infers *implicit* preference from attribute↔item feedback; **inverse RL** learns the reward | SR@15 / AT [S] | Yes (oracle attr. set `P_v`) [V via FedCRS] | No | **No** — IRL learns a reward *defined by target-hitting*; porting means re-deriving the premise | `Studenthc/CRIF`, **torch 1.4.0**, last commit 2022-09-28 [V] — dead stack | **UNFIT** |
| **MCMIPL** | Zhang et al., WWW 2022, arXiv:2112.11775 [V] | Dueling-Q with multi-interest heads `Q = max_k(f_V(q_k)+f_A(q_k,a))`; **multi-choice** attribute questions; **union** candidate filter | SR@T, AT, hDCG@(T,K) | **Yes, verbatim [V]:** "We regard each item vᵢ∈𝒱u as the ground-truth target item" | No (hDCG ranks *within the recommended list*) | Policy no; **the multi-choice question FORMAT ports** | `ZYM6-6/MCMIPL`, last commit 2022-05-08 [V] | **NICE** (format only) |
| **HutCRS** | Qian et al., EMNLP 2023, 10.18653/v1/2023.emnlp-main.635 [V] | 4-layer interest tree root→**aspect**→attribute→item; ask aspects first, then attributes *within accepted aspects*; adds a **"don't know"** feedback class | SR@15 / AT / hDCG | Yes [V] | No | **YES — best in this slice.** Coarse-then-fine asking over a static taxonomy needs no target and maps onto tag-genome structure | `xinle1129/HutCRS`, py3.9 + **torch 1.12.1** + DGL 1.0.1 [V] — portable | **MUST-INCLUDE-candidate** |
| **DAHCR** | Zhao et al., IJCAI 2023, arXiv:2305.02575 [V] | Director(option: ask/rec) → Actor(primitive); dynamic **hypergraph**; option *sampled* not argmax; intrinsic reward for ask-when-uncertain | SR@15 / AT | **Yes, verbatim [V]:** "a user u initializes the conversation session with a target item v* and an attribute that belongs to the target item p₀ ∈ P_{v*}" | No | No — our budget k is fixed, there is no ask-vs-recommend decision | `Snnzhao/DAHCR`, last commit 2023-09-19 [V] | **UNFIT** |
| **CoCHPL** | Fan et al., DASFAA 2024, arXiv:2310.17922 [V] | Hierarchical options + a **chain** of attributes per round optimised for **diversity + interdependence**; reverts to **intersection** candidate filtering | SR@15 / AT / hDCG | Yes [V] | No | **The chain objective ports** — and it is **prior-art risk on our greedy-prefix framing** | `AlexFanw/CoCHPL`, last commit 2024-04-04 [V] | **NICE — read before claiming novelty on "mutually-informative question set"** |
| **CRSIRL** | Chu, Wang, Wang, **NeurIPS 2023**, arXiv:2310.20109 [V] | Bi-level: inner trains policy on a learned intrinsic reward, outer tunes it to trade SR against turns; **hindsight reward shaping** `r̃ = r + γw(s_{t+1},v) − w(s_t,v)`, `w = log(ρ(s_t,v)+1)` | SR / AT | **Yes, and worst-case [V]:** the shaping potential `w` is *an explicit function of the target `v`* | No | **No** | No code URL [V] | **UNFIT — but the cleanest exhibit that this line optimises directly against the target** |
| **VPPL / VPMCR** | Zhang et al., **TOIS 2025**, 10.1145/3717831, arXiv:2306.04487 [V] | *Vague preference* MCR: attribute types split clear/vague; **soft estimation gives every candidate item non-zero confidence and never filters on clicks** | SR / AT | Yes, still p₀-initialised [V] | No | Partially | No code URL [V] | **NICE — the ONLY paper structurally attacking the shrinking candidate set; the right cite for our "not orderable against full-catalogue NDCG" argument.** Honest hole: motivates on over-filtering, reports **zero** over-filtering statistics [V] |
| **SAPIENT** | Du, Peng, Ning, arXiv:2410.09580v3 (May 2025) [V] | Conversational agent + **MCTS** over action types (ask/rec), N=20 simulations, policy net for types + Q-net for actions | SR / AT / hDCG, T_max=15 | Yes [V] — template simulator accepts only lists containing the target | No | **YES — non-myopic lookahead is a real alternative to our greedy prefix at fixed k** | `ninglab/SAPIENT`, last commit 2025-01-22, maintained [V] | **NICE-HIGH** |
| **TCRS** | Zhang et al., **CIKM 2024**, arXiv:2408.06809 [V] | Tri-phase: (1) conversational user model from **logged data only**, (2) PPO against that model, (3) evaluation against an **independent black-box simulator** | SR@T / AT | Yes [V] — simulator initialised with `(u, V_his, V_tar)` | No | The **train/eval simulator separation** is the transferable idea | No code URL [V] | **NICE — closest anyone in this line comes to our behavioural firewall, but the target stays** |
| **MHCPL** | Zhao et al., SIGIR 2023, arXiv:2307.14024 [V] | Multi-view hypergraph (Like/Dislike/**Social**) + cross-view contrastive | SR / AT | Yes [V] | No | No — needs a social graph we don't have | `Snnzhao/MHCPL` [V] | **UNFIT** |
| **MSMCR / MSCAA** | Ji et al., arXiv:2310.13365 [V] — **venue UNVERIFIED** | Multi-**subsession** (user doesn't quit on success); "activation attributes"; reward includes **`ask_unknown = −0.1`** | SR@10, AT, AR@10, hN@(10,10) | Yes [V] | No | — | "will be released", no URL [V] | **NICE-marginal — the nearest published analogue to our "unanswerable question burns the budget"** |
| **MINICORN** | Zhang, Li, Liu, Yuan, Miao, arXiv:2206.14468 [V] | **No RL.** Epistemic uncertainty = harmonic mean of (MC-dropout variance over N=10 passes) and (midpoint proximity `|p−0.5|`); ask the max-uncertainty attribute; one hand-written ask-vs-rec rule | SR@T / AT | **Yes, verbatim [V]:** "The user answers affirmatively to all attributes associated with the target item v and negatively to all other attributes" | No | The **uncertainty criterion ports** and is trivial | No public repo [V] | **NICE — and the load-bearing citation that the RL apparatus is not doing the work.** LastFM +67.40% SR / +13.39% AT vs SCPR [V]; ML +7.93% SR; Yelp +5.81% SR. Baselines EAR, EAR-FPAN, SCPR |
| **FedCRS** | Lin et al., arXiv:2503.00999 (2025) [V] | Federated version of the same protocol | SR / AT | Yes [V] | No | — | — | **UNFIT** (orthogonal axis); useful only as the verified source of CRIF's baseline row |
| **CT-CRS** | arXiv:2508.13209 (2025) | — | — | — | — | — | — | **WITHDRAWN by the authors [V]** ("The tables … need to be modified"). Search engines still surface its numbers. **DO NOT CITE.** |

**Verified reward constants (useful if we ever re-implement one):** MCMIPL rec_suc 1.0 / rec_fail −0.1 /
ask_suc 0.01 / quit −0.3 [V] (ask_fail sign UNVERIFIED). HutCRS rec 1 / −0.1, ask_**aspect** 0.02 / −0.2,
ask_**attribute** 0.01 / −0.1, quit −0.3 [V] — note aspects are weighted 2× attributes. DAHCR 1.0 / −0.1 /
0.01 / −0.1 / −0.3 plus intrinsic +1 for ask-when-uncertain and +1 for recommend-when-certain [V].
CoCHPL 1 / −1e−4 / 1e−2 / −1e−4 [V]. CRSIRL +1 / −0.1 / +0.1 / −0.1 / −0.3 [V]. SAPIENT 1.0 / 0.01 /
−0.1 / −0.1 / −0.3 [V].

**The benchmarks are saturated.** CoCHPL SR@15 = **0.991** on LastFM at AT 3.75; DAHCR **0.992** on Yelp at
AT 5.16 [V]. A protocol whose ceiling is hit cannot adjudicate anything. Say this rather than "their
numbers are high".

**And a skeptical detail worth keeping:** HutCRS beats MCMIPL on SR everywhere but **LOSES on hDCG on
LastFM (0.348 vs 0.396) and MovieLens (0.475 vs 0.599) and loses on AT on both** [V] — it buys success rate
with turns and with ranking quality, and the abstract does not say so.

---

## 5. EVALUATION CRITIQUES — the citable ones

Already recorded elsewhere: PEPPER (`kim2024targetfree`), Bernard & Balog, iEvaLM (`wang2023rethinking`),
RecSim/RecSim NG, Hsu 2024, He CIKM'23, Panickssery NeurIPS'24. **New in this pass:**

- **"Stop Playing the Guessing Game!" is the published NAME for our objection.** PEPPER's phrase, Nov 2024.
  Its replacement metrics are **Preference Coverage (PC)** and **Preference Coverage Increase Rate
  (PCIR)** — cumulative discovery of distinct user preferences across turns [V per agent]. Code
  `happysnail06/PEPPER`. **Use their name; do not christen the defect ourselves** (same lesson as nDCL).
- **Zhu, Huang & Sang, "How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based
  User Simulators for Conversational Recommendation", arXiv:2403.16416 (2024) [V abstract].** Three named
  limitations, quotable: "**Data leakage, which occurs in conversational history and the user simulator's
  replies, results in inflated evaluation results**"; "The success of CRS recommendations depends more on
  the availability and quality of conversational history than on the responses from user simulators";
  output control via a single prompt template is unreliable. Proposes SimpleUserSim. *Datasets named only
  as "two widely-used datasets" on the abs page — UNVERIFIED.*
- **Ramos, Xia, Wang, Chatterjee, Fu, Rahmani, Lipani — "Interplay: Training Independent Simulators for
  Reference-Free Conversational Recommendation", arXiv:2603.18573, ECIR 2026 [V].** Verbatim: existing work
  "**feed target items to simulators in advance. This causes: (1) data leakage (the simulator knows the
  answer beforehand) and (2) rigid dialogues (exploration is limited, reducing realism)**"; and "simulated
  users reject strong recommendations simply because they do not match a preset title". Two independently
  fine-tuned simulators, neither seeing the target. PEARL corpus, 52.9k conversations, 2,069 unseen-target
  test conversations; SR 93–95% for their user simulator vs 36% (Llama-70B) / 77% (Qwen-32B). Code
  `jeromeramos70/interplay-sim`. **This is the cleanest 2026 statement of the leakage critique — cite it
  and PEPPER together.**
- **`arXiv:2510.05624` is the arXiv version of Bernard & Balog (SIGIR-AP 2025)** — reconcile with the
  existing INDEX row. It frames Recall@N and SR as instances of a general Reward/Cost framework and reports
  SR/SRRR/RDL correlating positively with self-reported satisfaction [S] — **i.e. it partly DEFENDS the
  protocol. Not verified from full text; do not quote it.**
- **Nobody inside the attribute-asking line has challenged its own evaluation.** All three critiques above
  come from the LLM/ReDial branch. That is itself the reportable finding.

**Fairness note for the paper:** the honest sentence is not "their evaluation is invalid" but "their
evaluation prices a different quantity, and its own community has published the leakage critique". We cite
PEPPER and Interplay for the second clause.

---

## 6. THE CONTRACT AUDIT — the (a)–(d) summary

| Family | (a) what the metric prices | (b) target/omniscient? | (c) post-interview held-out ranking? | (d) policy portable? |
|---|---|---|---|---|
| **Conversational bandits** (ConUCB, ConLinUCB, Hier-UCB, CLiSK/CLiME, FedConPE) | Cumulative **regret** over T rounds vs the best arm | **NO target.** Reward replayed from real ratings | **No** | **YES — most cleanly of anything in this file.** Key-term arms = our concept channel |
| **ICF bandits** (Zhao'13, LinUCB, TS, GLM-UCB, PTS, factorUCB, IGCF) | Cumulative precision/recall/nDCG **over the recommended sequence**; regret | No explicit target, but **the candidate pool is the user's rated set** (PTS/COFIBA say so verbatim) | **No** | **YES** — frozen item factors + Gaussian user posterior IS our architecture |
| **Contextual bandits in production** (LinUCB WWW'10, Chapelle&Li) | CTR vs a random bucket, via replay | Needs **logged uniformly-random traffic** — unavailable to us in principle | **No** | Yes, via ICF's transposition |
| **Question-based RS** (Qrec, NICF, RLT4Rec) | Cumulative Precision/Recall/α-NDCG@T; Recall/MRR **against the target** | **YES, omniscient** (Qrec's answerer has "full knowledge of … the target item") | **No** | Acquisition rules port (GBS half-split; NICF's Q-network as a selector) |
| **Attribute-asking CRS** (CRM→EAR→SCPR→UNICORN→ConTS→CRIF/MCMIPL/HutCRS/DAHCR/CoCHPL/CRSIRL/VPMCR/SAPIENT/TCRS) | **Success Rate@turn, Average Turns**, hDCG *within the recommended list* | **YES, always**, and the candidate pool is **seeded by an attribute of the target** (`p₀`) | **No — not one paper** | **Mostly no** (rewards are target-defined). Portable *ideas*: HutCRS coarse→fine asking, MCMIPL multi-choice format, CoCHPL diverse question chain, SAPIENT MCTS lookahead, MINICORN uncertainty criterion |
| **Rating elicitation** (DRE, PERE, trees) — see the other file | NDCG@N **after** the interview | No | **YES** | Yes |
| **Greedy SLIM** (arXiv:2406.06061) — §3.3 | **NDCG@10 after the interview, full catalogue AND long tail, ML-25M** | **No** — answers copied from real held-out ratings | **YES — the only one** | Yes |

**Conclusion, stated for the paper:** *outside the rating-elicitation family, we found **no** method in the
bandit/RL/CRS literature that reports a post-interview held-out ranking metric over a full catalogue*
— with a single exception, an unpublished 2024 preprint (Greedy SLIM) that also happens to use our dataset
and our tail definition.
The family scores the probes themselves (bandits) or hitting a known target (CRS). Their numbers are
therefore not orderable against ours — **but several of their policies are portable, and a policy whose
numbers are unusable is still a must-include baseline once re-scored under our contract.**

**HARD RULE 2 corollary:** because none of these can be number-matched, the acceptance gate for a ported
policy is *not* "reproduce their published number" — it is (i) reproduce their *selection rule* exactly as
written, (ii) show it beats random-catalogue and popularity on our metric, (iii) report it labelled
"policy ported, protocol replaced". Say that in the paper before a reviewer says it for us.

---

## 7. What to actually build — ranked by (value to Paper B) ÷ (agent-hours)

Checked against the repo on 2026-08-05: `src/baselines/bandit_selectors.py` already implements
**GLM-UCB and Thompson as question selectors** over `belief_mf`'s Gaussian posterior (population mean/cov
prior, signed values, full 18,359-item argmax, no shortlist), and `adaptive_harness.py` registers
`glm_ucb`, `thompson`, `eddi_ig` plus the six static arms. `src/instrument/adaptive_concept_arms.py`
already has a greedy adaptive **concept** selector (max expected `|signed response|`). So items 1–2 below
are *additions to a working rig*, not new infrastructure.

1. **★ ConLinUCB-MCR as a CONCEPT-arm selector — `argmax_c α_t ‖d_c‖_{M_t⁻¹}` over the whitened concept
   directions, with the concept answer entering the SAME normal equations as an item answer.**
   **~4–8 agent-hours.** This is the nomination (see §8). It gives the concept channel a *named, published,
   target-free, regret-bounded* adaptive baseline, it is one selection rule away from code we already ship,
   and it directly tests ConUCB Eq. 4's uniform-member-average propagation against our whitened centroid.
2. **ICF's LinUCB / GLM-Lin / GLM-Sig item selectors** to complete the ICF row (we have GLM-Lin ≡ our
   GLM-UCB and TS; GLM-Sig's sigmoid link + Newton solve is the missing one). **~3–5 agent-hours.**
   Turns "we tried a bandit" into "we ran the ICF paper's own four policies".
3. **ConLinUCB-BS static concept arm** — barycentric spanner over the concept feature set. A *principled
   static* concept order that is neither popularity nor entropy. **~4 agent-hours.** Cheap insurance against
   "your static concept bar is a heuristic".
4. **CLiME's low-eigendirection rule** `argmax_c |d_cᵀv_i|` over the small-eigenvalue eigenvectors of `M_t`.
   **~3 agent-hours** once (1) exists. The 2025 SOTA criterion; nearly free as a second arm.
4b. **★ Greedy SLIM's `QGSLIM` questionnaire as a STATIC item bar, and `QBandit` (Christakopoulou'16 over
   PureSVD) as its adaptive foil.** **~12–20 agent-hours** (greedy row-wise SLIM is an elastic-net
   least-squares per row; their own implementation takes 19 min/row on ML-25M, so ~5–6 h compute for k≤16).
   This is the *only* published protocol sibling we have, on our dataset, with our tail definition — and its
   QBandit-declines-past-k=10 curve is external corroboration of our bank result. Also check whether
   `run_interview_table.py` excludes asked items from the candidate pool, because they do and it changes the
   sign of the popularity arm's late-budget behaviour.
5. **HutCRS-style coarse→fine concept asking** (genome tag-cluster level first, then tag within accepted
   clusters). **~20–30 agent-hours.** Real value for the concept arm and the only baseline whose stated
   motivation — users cannot answer attribute questions they have no knowledge of — is *our* motivation.
6. **SAPIENT-style MCTS lookahead at fixed k** as a non-myopic control on our greedy prefix.
   **~25–35 agent-hours.** Only worth it if a reviewer challenges prefix-optimality.
7. **Do NOT build:** PTS, COFIBA/CLUB, factorUCB, full IGCF, CRIF, DAHCR, MHCPL, FedCRS, anything
   LLM-warm-started. Each either collapses to something we already have under a frozen tower, needs
   cross-user streaming or a social graph we do not have, or is forbidden by project rules.

---

## 8. Nomination — the strongest ADAPTIVE baseline in this family whose POLICY we can honestly port

**ConUCB / ConLinUCB (Zhang et al. WWW 2020 → Wang et al. AAAI 2023), instantiated as ConLinUCB-MCR over
our frozen tower's Gaussian posterior, with tag-genome concepts as the key-terms.**

Why it beats the alternatives:

- **It is the only published adaptive method whose ACTION SPACE is ours** — ask about an item *or* ask about
  an attribute, in one algorithm, with an explicit model of how attribute feedback moves the item-level
  estimate. ConTS unified items and attributes as arms but under a target-conditioned oracle; ConUCB does it
  **with no target item at all**.
- **It has no omniscient anything.** Reward is replayed real feedback; there is no `P_v` oracle attribute
  set, no `p₀` target-seeded candidate pool, no LLM. It passes the behavioural firewall as-is — which is
  true of essentially nothing else in this file.
- **Its selection rule is four lines over machinery we already ship.** `M_t` is our posterior precision;
  ConLinUCB's one-step estimator is our conjugate update with the concept direction substituted for the
  item vector; MCR's `argmax_k α_t‖x̃_k‖_{M_t⁻¹}` is our existing UCB width term evaluated on concepts.
  ~4–8 agent-hours, against 40+ for NICF or PERE.
- **It is current and it is on our dataset.** The line runs 2020 → AAAI'23 → KDD'25, and the KDD'25 member
  (CLiSK/CLiME) evaluates on **MovieLens-25M with genres as key-terms**. A reviewer cannot call this a
  2011 straw man.
- **It gives us a real scientific test, not just a table row.** ConUCB Eq. 4 asserts a concept's value is
  the *uniform weighted average of its member items'* values; Hier-UCB's survey evidence says it is
  dominated by a few *representative* items instead. Our whitened centroid is a third answer. Running
  ConLinUCB against our fold adjudicates that directly, on our metric.

**State the contract mismatch explicitly wherever it appears:** ConUCB/ConLinUCB/CLiSK publish *cumulative
regret over 6,000 rounds on 5,000 arms and 200 users*; we publish *full and tail NDCG@10 over 18,430 items
on 10,000 held-out users after k ≤ 16 questions*. **The policy is ported; the number is not compared.**

**Runner-up, and cheap enough to do both:** ICF's four policies (Zhao CIKM'13) as item-question selectors —
the item-channel counterpart of the same argument, ~3–5 agent-hours, and it lets us write "we ran the
Interactive-CF paper's own policy set" instead of "we tried a bandit".

---

## 9. Verification debts and DO-NOT-CITE list (binding)

**Do not put these in the bibliography or a claim until checked:**
- Filippi et al.'s exact confidence radius `β_t`/`ρ(t)` — an OFUL-style formula surfaced and is believed
  **confabulated**; rejected. Also Filippi's experiments/datasets.
- **Pseudo Dyna-Q (WSDM'20)** — not on arXiv, DOI 403. No DOI may be cited for it.
- **CT-CRS (arXiv:2508.13209)** — **WITHDRAWN**. Its numbers still appear in search results. Never cite.
- IGCF's datasets, baselines, numbers, and venue (arXiv metadata says journal "Artificial Intelligence";
  a SIGIR'23 version is UNVERIFIED).
- factorUCB reporting precision/recall on ranking tasks — low confidence.
- The identity of ICF Table 2's **`AL` and `Interview`** baselines — *chase this, it matters to Paper A*
  (ICF beats `Interview` by only 1.6%).
- Q&R's offline protocol and metrics — the PDF would not extract.
- RLT4Rec's datasets, metrics, session length, code.
- MCMIPL's `ask_fail` reward sign (extracted as "0.1"; every sibling uses −0.1).
- MSMCR/MSCAA (arXiv:2310.13365) venue.
- RelativeConUCB and ConDuel — named only in third-party lineage lists.
- The `juanfacabian/InPE` repo attributed to Xia/Zhang/Wang (arXiv:2607.06765) — the org name does not match
  the author list; verify the repo exists before citing it.
- **Greedy SLIM's OSF repo** (`osf.io/myh2q/?view_only=…`) — link present in the paper but **not opened**;
  language/framework/completeness UNVERIFIED. Also: it is **CoRR-only, not peer-reviewed** (DBLP-verified) —
  never present it as a published bar.
- **Bıyık et al. 2023 soft attributes — its EVALUATION PROTOCOL is still unverified** (arXiv HTML 404s;
  abstract only). Given that Bıyık is the standing threat to Paper A, **this is the verification debt to
  close first** — fetch the TeX source at `arxiv.org/src/2311.02085`. Assert nothing about its metric until
  then. (This debt is also standing in `text_to_cf_projection.md` and `adaptive_elicitation_baselines.md`.)
- **Wang et al., "Towards Efficient Conversational Recommendations: EVOI Meets Bandit Learning", WWW 2025,
  10.1145/3696410.3714773** — ACM DL 403'd every automated fetch. By title it joins EVOI to the bandit
  family, i.e. it sits exactly between our belief line and this file's family. **Fetch manually.** [S]
- **Unaudited (search budget exhausted at 200/200):** TUT4CRS (MM'24, 10.1145/3664647.3681259); graph-based
  dynamic attribute clipping (Discover Computing 2024, 10.1007/s10791-024-09437-6); Confident Action
  Decision via Hierarchical Policy Learning (WWW'23, 10.1145/3543507.3583536); the TORS fuzzy-feedback MCR
  simulator (10.1145/3616379); CRIF's full mechanism; RecSys'18 low-rank ensemble sampling; "ICF with
  knowledge graphs"; whether anyone initialises bandit priors from a trained offline recommender's user
  distribution *as a named contribution* (found nothing beyond IGCF's meta-prior — treat as **not found**,
  not *does not exist*).

**Also flagged (a live prior-art risk, not a debt):** **"When and How to Ask: Dynamic Preference Elicitation
Strategies for Conversational Recommendation", Xia, Zhang & Wang, arXiv:2607.06765 (Jul 2026; SIGIR 2026
per [S] only)** — claims verbatim [V] that "attribute-based strategies are more effective for early-stage
preference exploration, whereas item-based strategies become more suitable once users have formed clearer
and more specific preferences", on an annotated INSPIRED corpus (InPE, 6,719 turn-level samples), reporting
Recall@{1,10,50} (COPE R@10 = 0.314) plus a 328-participant user study. **This is a published, qualitative
version of our concept-vs-item budget crossover.** Our differentiator is the non-circular behavioural
protocol and the quantitative m-sweep — but the claim must be cited, not discovered by a reviewer.
(It is also already logged as the "nearest miss" row in `answerability_in_elicitation.md`.)
