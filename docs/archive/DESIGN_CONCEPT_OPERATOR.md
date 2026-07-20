# DESIGN — Concept Operator: how to add concepts to a strong recommender CORRECTLY (2026-07-12)
For Fable adversarial criticism, then a converged plan. Grounded in a 5-stream deep-research pass
(operators-in-CF, region-narrowing-in-CRS, add-without-forgetting, CASPER-history, attribute-only-cold-start)
and the honest failure of the additive channel. Author directions baked in.

## 0. THE FINDING THAT FORCED THIS
The additive-to-latent operator `z += e_c` DRAGS cold-start NDCG BELOW the popularity intercept (0.258→0.226),
and a0c member-bag injection is catastrophic (0.10). Our own ledger: EVERY concept/dislike signal bolted onto a
sign-free positive top-N scorer went INERT or HURT (member-bag −0.08..−0.12; FiLM γ never crossed 0, ΔNDCG≈0;
z-repulsion +0.05 selectivity ceiling / −0.165 dislike-heavy; additive e_c below intercept). Root cause
(unanimous across streams): adding a concept vector moves the query OFF the manifold the tower scores and BLURS
toward the genre-popularity centroid instead of NARROWING. Every NET-USEFUL method in the literature instead
touches items through THEIR OWN attribute evidence (per-item re-weight / filter / region), never a free vector
added to a belief.

## 0.1 THE DUALITY (answerer ↔ recommender)
The distilled ANSWERER already deduces a concept value as pop-weighted member taste (ctaste = Σ pr·t over the
concept's members / Σ pr) → ordinal model. The correct recommender operator is the DUAL: place/score the concept
via its member items in the SAME item geometry. Use the same item↔concept structure both directions.

## 0.2 THE HONEST BAR (reframed by the attribute-only stream)
Broad genres ≈ popularity is an INFORMATION-THEORETIC CEILING, not only our bug. So: (a) value lives at FINE
tags/entities (directors, franchises, genome tags — which our bank HAS) and on the TAIL; (b) the honest control
is beating POPULARITY-WITHIN-THE-ATTRIBUTE-FILTER on TAIL NDCG, NOT global popularity on full NDCG. Every
variant below is judged on that bar. Existence proofs the signal is real: DualHeadSetEncoder (dislike native in
a strong ranker, Hit@10 0.433→0.647); signed-EASE repulsion in centered-rating taste space (+17.5% @k=2).

## 0.3 NOVELTY CELLS (open, for Papers B/D)
No published method fuses GRADED concept answers into a STRONG pretrained CF tower via re-ranking; none shows
NEGATIVE attribute preferences improving full-catalog TAIL NDCG in cold-start. Both are defensible novelty.

---

## PART 1 — CONCEPT-OPERATOR VARIANTS (every approach = a runnable variant, judged on the honest bar)
All share: frozen strong item tower s_base (a0c 0.4961) unless noted; graded g_c∈[−1,+1]; eval on the honest bar
(pop-within-filter, tail); full catalog; no caps; knockout + Ce + 173 as headline gates.

### GROUP A — score-space re-weighting (item-side; safe, likely net-useful)
- **V0  additive-e_c (CURRENT, the null to beat).** z += (1/n_c)Σ g_c W e_c. Documented to go below intercept.
- **V1  member-bag input injection (historical null).** concept → members into item value channel. Documented
  to HURT. Kept only as the known-bad floor.
- **V2  hard faceted filter (pure narrowing baseline).** keep items matching loved concepts, prune hated;
  rank survivors by s_base. Binary, no grade, but the interpretable safe floor every learned variant must beat.
- **V3  per-item signed graded re-weight (PEBOL/soft-SCPR), hand-set β.**
  `score(i) = s_base(i) + Σ_c g_c·β·align(c,i)`, align(c,i) = item i's own affinity to concept c (genome
  relevance / membership / content cosine). Dislike = negative coefficient (penalize matching items). β small =
  monotone safety floor. NO sign to learn (g_c given) → sidesteps the FiLM-inertness trap.
- **V4  LLC learned-weight (ranking-margin).** learn per-concept g_c·β (or a small map) by minimizing ranking
  violations on interview logs (known-liked out-rank known-disliked given the answers). Calibrates magnitudes.
- **V5  multiplicative soft-filter (PEBOL-faithful).** `score(i) = s_base(i)·Π_c (1 + g_c·β·align(c,i))`.

### GROUP B — belief-modulation (gated, narrows not translates; identity/zero-init)
- **V6  FiLM / critiquing GRU-gate (M&Ms-VAE++), SEPARATE like/dislike, identity-init.**
  γ,β = MLP([pol_emb(sign g); g; e_c]); z' = (1+tanh γ)⊙z + β. Per-dim γ<1 suppresses a region, γ>1 sharpens.
  Identity-init ⇒ unanswered concept = exact no-op (0.4961 preserved). Separate params for +/− (M&Ms-VAE++
  found symmetric underperforms). Low-rank γ = I+UVᵀ if diagonal underfits.  ⚠ our FiLM went INERT before —
  the fix is NO-sign-to-learn (hard-wire g_c) + concept-forcing curriculum + item-side redundancy removed.
- **V7  NOVA non-invasive attention.** concept answers modulate ATTENTION weights only; item embeddings stay
  the untouched value signal. Recsys-native frozen-CF-preserving fusion.
- **V8  signed attention-pointer residual.** score_i = s_base(i) + λ·Σ_c g_c·⟨q(e_c), attr(i)⟩. Cannot lose CF
  (additive residual over pool); dislike = negative logits.

### GROUP C — geometric / region (thesis-connected; dislike sharpens)
- **V9  constraint-region / PERE half-spaces.** each answer = half-space in item-tower space; user = Chebyshev
  center of the feasible region; dislike SHARPENS (not cancels). +16% NDCG@10 in PERE. Composes as prior+
  likelihood; connects to the continuous-actor / off-manifold thesis (Paper C).

### GROUP D — content-prior infra (build well-placed e_c; not a standalone operator)
- **V10 tag-genome profile cosine.** taste vector = signed sum of genome rows of endorsed tags; cosine score.
  Content baseline; also the graded/dense representation for align(c,i).
- **V11 cold-start content map (DropoutNet/Heater).** learn concept embedding INTO CF space so a concept vector
  is even meaningful. Infra for e_c in V3/V6/V9, not itself the belief operator.

---

## PART 2 — PROPOSED NOVEL ARCHITECTURE: UNIFIED ITEM+CONCEPT SPACE (author's vision)
Vision: concepts share the SAME embedding space as items; the model is AGNOSTIC to whether an input token is an
item or an attribute; one unified space that enables CONTINUOUS experiments (a query is a continuous point in
the shared space — the continuous-actor thesis). Directly generalizes the PRE-VAE DualHeadSetEncoder (tokens =
(entity, polarity), entity ∈ items ∪ genres ∪ tags; dual head; strong on ML-1M) to GRADED value + a STRONG
large-catalog scorer. Brainstormed versions (Fable to weigh):

- **U1  Unified-INPUT dense encoder (cheapest; reuses the 0.4961 SignedAE).** Input = concat[ item_values(ni) ;
  concept_values(C) ] over one (ni+C) vector → SignedAE encoder → z → decoder(ni). Items AND concepts are
  first-class input dims; the encoder's nonlinearity learns how each concept reshapes z JOINTLY with items (not
  a fixed additive centroid — this is why it can narrow where V0 blurs). Dislike = signed concept value.
  Concepts are not literally "item embeddings" but share the encoder's first layer. Fast to try; strong base.
- **U2  Unified ENTITY-EMBEDDING set model (truest to the vision; pre-VAE reborn, graded+strong).** ONE embedding
  table E over entities = items ∪ concepts. Belief = pooled bag of revealed entity embeddings, each weighted by
  graded value g and knowledge (attention-pool, June-proven; NOT set-transformer). Score item i by z·E_item(i).
  Items and concepts LITERALLY share embeddings; the model is agnostic; a query is a continuous vector in E-space
  (enables Paper-C continuous elicitation). ⚠ risk: pre-VAE's listwise-softmax COLLAPSED on large catalogs
  (0.002 on 3706 items) — must pair with the strong multinomial/EASE-class objective + the dense-AE lessons.
- **U3  Shared DECODER-FACTOR space.** concepts are embedded as points in the decoder's item-factor W-space
  (init member-centroid, learned); belief z is compared to items AND concept regions in W-space. Unifies via the
  scorer geometry; continuous query = a point in W-space. Closest to the current model but concepts made
  first-class in the SAME space as the scoring factors.
- **U4  Unified GEOMETRIC / region model (thesis-max).** items = points, concepts = REGIONS (member span /
  half-spaces) in one space; scoring by geometry; dislike = half-space (V9). The continuous, off-manifold,
  region-elicitation realization of the unified vision. Most ambitious; unifies U2 + V9.

LEAD CANDIDATES: **U1** (cheap decisive first test of unified-input) and **U2** (truest to "same embeddings /
agnostic / continuous"). U3/U4 are the thesis-max escalations.

---

## PART 3 — TRAINING SCHEDULE (author's gut)
1. **Items-first pretrain (DONE = a0c 0.4961).**
2. **FREEZE the decoder (and item tower); train the concept path on FULL-PROFILE, CONCEPTS-ONLY signal.**
   Derive ALL concepts from the full profile (rich, real values — the answerer duality), MASK the items, train
   the model to recover held-liked from CONCEPTS ALONE. Frozen decoder ⇒ item strength preserved by construction;
   concepts-only + full-profile ⇒ forces the concept path to be load-bearing (defeats redundancy). Zero-init gate
   so t=0 = the 0.4961 model exactly.
3. **Compose at eval:** item channel + concept channel; as item history grows the concept term gracefully cedes
   to CF (documented decay = a feature). Optional Stage-4 minimal co-adapt (LoRA on decoder final projection +
   item-only REPLAY) ONLY if the frozen ceiling leaves headroom — dislike is the axis most likely to force it
   (positive decoder may lack a push-away direction → the taste-structured-latent escalation).

## PART 4 — GATES (every variant, same ruler)
Honest bar: pop-WITHIN-attribute-filter, TAIL NDCG. Plus: knockout (concepts on/off, pos-half/neg-half),
Ce EASE-direct control, 173 transfer, strength floor (full ≥ 0.4961−ε via frozen decoder), monotone sweep
(supporting), hygiene. Report concept-only curve AND items+concepts. Leaderboard across V0–V11 + U1–U4.

## PART 5 — OPEN QUESTIONS FOR FABLE
1. Rank the operator families for OUR setting (frozen strong tower, graded+dislike, tail bar). Is score-space
   re-weight (V3/V4) the right FIRST build, or does the unified-input encoder (U1) dominate it (learns align)?
2. The unified vision (U2) vs the safe re-rank (V3): is "same-embedding agnostic space" worth the large-catalog
   collapse risk the pre-VAE hit, and how to avoid it (objective, init, capacity)?
3. Does frozen-decoder + concepts-only-full-profile training actually make concepts NET-USEFUL, or does a frozen
   positive decoder structurally lack the geometry for graded DISLIKE (forcing the taste-structured-latent
   escalation)? Name the decisive early gate.
4. Is "align(concept,item)" (item-side affinity) leak-safe and non-circular given the answerer computes values
   the same way (item-side member taste)? Where's the circularity risk?
5. The single most dangerous flaw that makes the WHOLE unified/re-rank direction fail — and the cheapest
   experiment that kills or confirms it fastest.
