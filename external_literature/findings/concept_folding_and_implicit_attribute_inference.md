# Findings: Concept/Attribute Folding (encoder-side, continuous-OOD, signed) + Implicit Attribute-Preference Inference

*Deep research 2026-07-25 (web only, no LLM calls). Two questions:*
*Q1 — how published work FOLDS attribute/concept preferences into a CF recommender at the ENCODER side, with a continuous OOD-capable attribute embedding + signed feedback.*
*Q2 — the established estimator for signed attribute affinity (incl. DISLIKE) from item-level IMPLICIT consumption, volume/popularity-corrected with a support measure.*
*Feeds Arm C-lite / SEL. Read this + `answerability_and_channels.md` + `unified_embedding_architecture.md` before re-searching. Bib-keys in `../INDEX.md`.*

---

## Q1 — Encoder-side attribute folding: per-paper records

### `biyik2023soft` — Preference Elicitation with Soft Attributes (arXiv 2311.02085) — THE central near-miss [full-text ar5iv, high confidence]
- **Where/operator:** Attribute enters as a **Concept Activation Vector (CAV)** direction ϕ_g in the item-embedding space of a two-tower CF model. Fold = **Bayesian posterior update** over a distribution on the user embedding ϕ_u: `P(ϕ_u | ρ,q) ∝ P(ρ | q,ϕ_u) · P(ϕ_u)`. Item queries AND attribute queries update **ONE shared belief** `P_U(u|H)` — the unified-belief property CASPER wants.
- **Signed feedback:** YES, first-class. Response ρ ∈ {+1,−1} ("more of g" / "less of g"); likelihood is a **probit** on the projection of the belief onto the CAV: `P(ρ=+1) = Φ(c_g(·)/σ_g)`. Sign is the sign of the directional score → clean negation, not a bolt-on.
- **Continuous + OOD:** Direction is continuous, BUT **NOT OOD**. Each attribute needs its own CAV trained by **logistic regression on tag-labeled items** ("for a tag g we attempt to find a CAV ϕ_g"). New/unseen attribute ⇒ retrain a probe. No text→direction map. This is the exact wedge: we want a continuous encoder that maps an *arbitrary* text attribute to a direction with no per-attribute fit.
- **Saturation:** With few taggable attributes (5 in RecSim-NG) attribute-only PE "quickly saturates with more queries"; their fix is Item-plus-Attribute (IpA) queries to keep learning. So attribute-only channels saturate fast — matches our concepts-are-a-tail-instrument finding.
- **Data/numbers:** synthetic (1k items/10 tags); RecSim-NG (25k users/10k items/5 attrs); ML-20M (138k users/27k movies/164 tags analyzed). Metrics are belief-recovery / recommendation-quality curves vs #queries (no NDCG@10 headline).
- **Verdict-for-us:** Closest prior art to C-lite's signed-attribute fold. Has R2 (belief)+R3(direction)+R4(signed) but is co-trained (fails R1 frozen-SOTA tower) and per-attribute (fails OOD). Cite first-page; our wedge = **frozen certified tower + a continuous text→direction encoder (no per-tag probe) + exact conjugacy**.

### `gopfert2022discovering` — Discovering Personalized Semantics for Soft Attributes via CAVs (WWW 2022, arXiv 2202.02830) [search + citation, medium]
- The CAV machinery Biyik builds on: per-user CAV directions for subjective attributes ("scary", "artsy") discovered from the item embedding space, relating attribute semantics to interaction preference. Still **per-attribute probe training**, not OOD. Verdict: cite as the CAV-semantics substrate; same OOD limitation.

### `antognini2021mmvae` — M&Ms-VAE / Fast Multi-Step Critiquing (RecSys 2021, arXiv 2105.00774) [full-text ar5iv, high]
- **Where/operator:** Two inference networks give unimodal Gaussian posteriors over the user latent z_u — one from ratings q_Φr(z|r), one from keyphrases q_Φk(z|k). They combine by a **(mixture/product)-of-Gaussian-experts**: `α·q_Φr + (1−α)·q_Φk`, α set by which modality is observed (0.5 both, else 1/0). A **GRU-based gating blend** `z̃ = ξ(z⁰, zᵗ)` folds each successive critique into the running latent (update/reset gates, tanh) — learned adaptive blend, not plain averaging.
- **Signed:** Critiques are *disliked* keyphrases; negation handled in the **loss** (`r̂_i⁺,1 < r̂_i⁺,0` — score of items carrying the critiqued keyphrase must drop). No independent positive/negative axis (that's the ++ paper).
- **Continuous + OOD:** NO. Keyphrases are **one-hot** over a fixed vocab K; no continuous embedding, no unseen-keyphrase handling.
- **Data/numbers:** BeerAdvocate, Amazon CD&Vinyl, Yelp, HotelRec. "~26× faster" multi-step critiquing than CE-VAE; ~9% recommendation improvement over CE-VAE.
- **Verdict-for-us:** The PoE combination of a rating-posterior and an attribute-posterior over one latent is a clean, adoptable **fold-as-posterior-fusion** pattern (conceptually a Kalman/PoG merge, matching our belief-pool). But its attribute channel is closed-vocab. Adopt the *PoE-fusion shape*, replace the one-hot channel with a continuous text→direction encoder.

### `antognini2022posneg` — Positive AND Negative Critiquing for VAE Recommenders (M&Ms-VAE++, arXiv 2204.02162) [full-text ar5iv, high]
- **Where/operator:** Same VAE + GRU-blend backbone, but **two distinct blending modules** sharing weights with separate trainable **sign embeddings e⁺ / e⁻** conditioning the gate: `ξ⁺(z, z_c⁺) = GRU([e⁺; z; z_c⁺])`, `ξ⁻(z, z_c⁻) = GRU([e⁻; z; z_c⁻])`. This is the cleanest published treatment of an *independent* signed critique axis — the sign is a learned conditioning token, not just a loss.
- **Continuous+OOD:** still NO (one-hot keyphrase vocab, retrain for unseen).
- **Saturation:** **Negative** critiquing gains **plateau after ~5 turns**; positive keeps improving. Concrete published saturation number for signed attribute folding.
- **Data/numbers:** Yelp (9.8k users/4.7k items/234 keyphrases), HotelRec (7k/4.9k/141). NDCG@10 explanation 0.2797 Yelp / 0.3591 Hotel.
- **Verdict-for-us:** The **sign-as-conditioning-token** idea (e⁺/e⁻ gating) is directly adoptable for C-lite's γ-per-token signed fold and matches our fusion-token design (`fusion-token-context-dependent-value`). Cite for "negative critique saturates ~5 turns."

### `luo2021bkvae` — Bayesian Critiquing with Keyphrase Activation Vectors, BK-VAE (SIGIR 2021, DOI 10.1145/3404835.3463108) [search-summary only, dl.acm 403, medium]
- **Mechanism (as reported):** A **Keyphrase Activation Vector (KAV)** — explicitly *CAV-inspired* — measures alignment of an item's keyphrase properties with the latent user preference inside VAE-CF; critiques update the user latent by a **Bayesian critiquing** step over KAV directions. Matches or beats CE-VAE on both recommendation and multi-step critiquing.
- **Continuous+OOD/signed:** KAV is a continuous latent direction (CAV-style) → structurally the *most OOD-adjacent* of the critiquing-VAE line, but per-keyphrase (still needs the keyphrase in vocab); positive+negative critiquing supported.
- **Verdict-for-us:** The KAV = "attribute as a latent direction folded by Bayesian update" is the same skeleton as Biyik and as our whitened-concept-centroid fold. Confirms the direction-fold idea is well-trodden; novelty must be the *frozen SOTA tower + OOD text→direction*, not the direction-fold itself. (Fetch full text later if a mechanism detail becomes load-bearing.)

### `luo2019deepcritiquing` — CE-VAE / Deep Language-based Critiquing (RecSys 2019) [our prior notes + search, medium]
- **Mechanism:** Adds a **second VAE head** (keyphrase reconstruction) to a Mult-VAE; critiquing = perturb the predicted keyphrase vector, then run an **inverse decode→re-encode loop** to get an updated z. Keyphrase one-hot, closed vocab.
- **Known drawbacks (from the M&Ms line):** multi-objective head trades off recommendation accuracy; the inverse decode-encode loop is slow and degrades over steps.
- **Verdict-for-us:** The cautionary baseline — a bolt-on keyphrase head hurts the recommender (documented Mult-VAE dislike-blindness cost). Argues FOR keeping the tower frozen and folding at inference, AGAINST co-trained critique heads.

### `perez2018film` — FiLM: Feature-wise Linear Modulation (AAAI 2018) [full-text/search, high]
- **Operator:** conditioning signal → per-feature **affine (γ scale, β shift)**: `FiLM(h) = γ(c)⊙h + β(c)`. The generic, parameter-cheap conditioning operator. In recsys it appears as timestep/label conditioning (FiLM-DiffRec) and as "user-latent z → γ,β modulating an incoming token embedding" for adaptive latent conditioning.
- **Continuous+OOD/signed:** γ,β are produced by an MLP from *any* continuous conditioner → **natively continuous and OOD-capable** if the conditioner is a text/attribute embedding. Signed feedback is expressible (a −1 answer flips the shift). BUT: a single global (γ,β) applies the SAME modulation to every token — cannot simultaneously encode "item-hate = weak positive" and "concept-hate = below-prior" (exactly the failure our fusion-token replaced FiLM for; see `fusion-token-context-dependent-value`).
- **Verdict-for-us:** FiLM is the continuous-OOD conditioning primitive, but per-token context-dependent value needs attention-over-{emb,val,conf} (our fusion token), not global FiLM. Cite as the conditioning-operator ancestor we generalize.

### Continuous-OOD text-attribute steering (Q1a gap check) [search, medium]
- LLM-embedding recommenders (RLMRec `ren2024representation`, TextGCN zero-shot, P5) put items/attributes/profiles in ONE text-derived space → a text attribute is a continuous, OOD vector. BUT none fold a *signed* attribute answer into a **belief over a user in a frozen SOTA CF tower**; they either re-embed a whole NL profile or do prompt-level ranking. **No hit** for "signed continuous text-attribute → conjugate belief fold on a frozen RecVAE-class tower." The OOD-continuous + signed + frozen-tower + belief conjunction remains unoccupied ("we are aware of none").

---

## Q2 — Signed attribute affinity from implicit consumption: per-paper records

### `liang2016expomf` — Modeling User Exposure in Recommendation, ExpoMF (WWW 2016) [PDF binary; search + prior knowledge, high]
- **The principled confound fix.** Latent per-cell **exposure** a_ui ∈ {0,1} (did the user even *see* item i) is separated from **preference** y_ui. Non-consumption is modeled as EITHER not-exposed OR exposed-and-disliked — exactly the "everyone under-consumes almost everything" confound. EM: the E-step responsibility that a zero is a true negative is downweighted when exposure is unlikely; preference then re-enters as a **confidence weight** (recovers Hu-Koren-Volinsky as a special case). The **exposure prior μ_i can be conditioned on item content/popularity** (content-ExpoMF: μ_i = σ(text/popularity features)) → popularity enters the *exposure* model, not the *preference* model. Known limitation: biased toward popular items for rare items; EM is slow.
- **Verdict-for-us:** The generative gold-standard for "a non-watch is not a dislike until you account for exposure." For attribute affinity: aggregate the exposure-corrected preference over an attribute's member items → a popularity-corrected signed affinity with a natural confidence (posterior exposure mass). Heavier than a moment estimator but the principled anchor to cite.

### `hu2008collaborative` — CF for Implicit Feedback, WMF (ICDM 2008) [in INDEX, high]
- Template: raw count → **preference p=1[r>0]** × **confidence c=1+αr**. Value re-enters ONLY as a loss weight, never as signed direction. **Institutionalizes "value = confidence, not sign"** — the assumption our signed attribute affinity breaks. Cite as the baseline our estimator departs from; also the confidence-weight primitive to reuse for the support measure.

### `steck2010training` — Recommenders on Data Missing-Not-At-Random (KDD 2010) [in INDEX, high]
- The *absence* of a rating is informative (MNAR); AllRank imputes missing as a low value with a weight. Theoretical root of "non-consumption carries signal." Popularity/user-activity are the MNAR mechanism. Verdict: the theory license for reading dislike out of non-consumption — but it imputes a scalar, doesn't give a per-attribute signed estimator.

### `sen2009tagommenders` — Tagommenders (WWW 2009) [in INDEX + search, high]
- Infers a user's **preference for a tag** from item-level signals (ratings, clicks) then aggregates tag-prefs back to items. Best variants weight tags **TF-IDF-style** (down-weight globally common tags = popularity correction) to build a normalized **TagProfile** vector; matches/beats latent-factor RMSE on ML. Tag-preference *ratings* survey (118k) is the eval ground truth. Verdict: the canonical "item signal → attribute preference → back to items" channel and the TF-IDF popularity correction. Mostly **unsigned/positive**; dislike is thin.

### PMI / NPMI as the signed popularity-corrected estimator [search, high on math]
- **log-lift = PMI:** `PMI(u,g) = log [ P(u consumes a g-member) / P(u consumes ·)·P(g-member) ]`. Positive = affinity, negative = aversion; the denominator's `P(g-member)` **divides out attribute popularity**, and using per-user consumption *rates* divides out **user volume**. **NPMI ∈ [−1,+1]** normalizes and tames PMI's low-frequency inflation. This is precisely the "**watch-lift, popularity-corrected**" SEL signal we already found carries ~40% of concept-affinity variance (`answerability_and_channels.md`). Caveat: PMI over-rewards rare attributes → pair with a support/shrinkage term.
- **Verdict-for-us:** PMI/NPMI IS the closed-form estimator behind SEL; cite it to give SEL a principled name and a confidence handle (NPMI + support count). Signed by construction.

### `qin2020attributepropensity` — Attribute-based Propensity for Unbiased Learning (Google, WSDM 2020) [search, medium]
- Estimates **exposure propensity as a function of item ATTRIBUTES**, then IPS-reweights implicit feedback. Directly relevant: propensity-per-attribute is the debiasing layer for turning attribute-level consumption into an unbiased affinity. Verdict: the IPS route to volume/popularity correction; heavier than PMI, unbiased under correct propensities but variance-prone.

### `steck2018calibrated` — Calibrated Recommendations (RecSys 2018) [search, high]
- Builds the user's **genre distribution p(g|u)** from consumed items (optionally weight-normalized so heavy users don't dominate) and calibrates recommendations to match it. Verdict: the standard **normalized genre-propensity profile** — but a *distribution* (unsigned, sums to 1), so it encodes relative emphasis, NOT dislike. Use for the positive-affinity profile; it cannot express aversion (need PMI<0 or ExpoMF for that).

### `toroghi2023bcie` — BCIE: Bayesian Critique-Improve-Explain (SIGIR 2023) [in INDEX, medium]
- Conjugate-Gaussian fold of critiques into a FROZEN factorization space — the Q1 fold mechanism (conjugate update over frozen items) closest to our belief pool, applied to critiques. Cross-links Q1↔Q2: the estimator (Q2) produces the pseudo-observation the conjugate fold (Q1) ingests.

---

## SYNTHESIS — most adoptable recipes

### (a) Encoder-side continuous-OOD concept folding with SIGNED feedback
1. **Signed CAV/KAV direction + Bayesian posterior fold** (`biyik2023soft`, `luo2021bkvae`, `toroghi2023bcie`). Attribute = a direction in the frozen item space; a signed answer is a probit/Gaussian pseudo-observation projected onto that direction; update ONE shared user belief conjugately. → *Onto C-lite/SEL:* this is exactly our whitened-concept-centroid + conjugate belief-pool fold; the citable skeleton. Our wedge over all three = **the direction comes from a continuous text→direction encoder (no per-attribute probe = OOD) over a FROZEN certified SOTA tower**.
2. **PoE / posterior-fusion of a rating-posterior and an attribute-posterior** (`antognini2021mmvae`). Treat the attribute answer as a second Gaussian expert over the same latent and multiply. → *Onto us:* the Kalman/belief-pool merge already does this; adopt the PoE shape, keep the attribute channel continuous+OOD instead of one-hot.
3. **Sign-as-conditioning-token gate** (`antognini2022posneg` e⁺/e⁻; generalize `perez2018film` FiLM). Encode valence as a learned token that gates a per-token affine value, so "concept-hate = below prior" ≠ "item-hate = weak positive." → *Onto us:* this is our fusion-token γ-per-token design; cite ++ and FiLM as ancestors, note the ~5-turn negative-critique saturation as the expected ceiling.
4. **Frozen-tower + fold-at-inference (NOT a co-trained critique head)** (cautionary `luo2019deepcritiquing`). CE-VAE's second-head accuracy cost is the documented reason to freeze the recommender and fold answers at inference. → *Onto us:* validates the frozen-tower + additive-popularity-floor fold; do not co-train a concept head into the tower.

### (b) Volume/popularity-corrected SIGNED attribute affinity from implicit data
1. **Moment estimator = PMI/NPMI log-lift** (`sen2009tagommenders` TF-IDF flavor; PMI/NPMI math). `NPMI(u,g)` of consuming g-members vs base rate: signed, popularity-corrected (÷ attribute base rate), volume-corrected (per-user rate), bounded [−1,1]. → *Onto us:* **names and licenses the SEL watch-lift signal** (already ~40% of concept variance, ~24× the LLM ordinal). Cheap, closed-form, per-user. Pair with a support count for confidence.
2. **Generative estimator = ExpoMF** (`liang2016expomf`). Separate exposure (popularity/content-conditioned prior) from preference so a non-watch of a niche-attribute member isn't scored as dislike; aggregate exposure-corrected preference over member items → signed affinity + posterior-exposure confidence. → *Onto us:* the principled anchor when the confound (unaware vs disliked) is load-bearing; heavier than PMI, cite as the gold standard SEL approximates.
3. **IPS / attribute-propensity reweighting** (`qin2020attributepropensity`, `hu2008collaborative` confidence). Down-weight consumption by attribute-exposure propensity; confidence = HKV `1+αr` or Beta(support). → *Onto us:* the debiasing layer + the support/confidence measure for SEL.
4. **Normalized genre-propensity profile** (`steck2018calibrated`) for the POSITIVE side only — a distribution, cannot express dislike; use PMI<0 or ExpoMF for aversion.

**One-line takeaway:** the *fold* (direction + conjugate/PoE belief update + signed gate over a frozen tower) and the *estimator* (NPMI log-lift ≈ SEL, ExpoMF as its principled parent) are both well-established individually; CASPER's unoccupied conjunction is **continuous OOD text→direction folding over a FROZEN certified SOTA tower, driven by an exposure/popularity-corrected signed implicit affinity** — none of the surveyed systems hold all of it at once.

## Caveats
- BK-VAE (`luo2021bkvae`) and ExpoMF (`liang2016expomf`) equations are from search summaries + prior knowledge (dl.acm 403 / arXiv-PDF binary), not full-text this pass — medium confidence on exact update forms; fetch full text before quoting an equation.
- "No hit" for the frozen-tower + continuous-OOD-signed-text-attribute + belief conjunction is a search-pass negative (medium confidence on absence) — say "we are aware of none," not "there are none."
- PMI/NPMI popularity correction is well-established math but its *dislike* (negative) tail is noisier than its positive tail (low support on unconsumed niche attributes) — ExpoMF is the correction when the negative tail matters.
