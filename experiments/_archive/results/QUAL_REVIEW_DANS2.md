# QUAL REVIEW — D-ANS iteration-2 (Fable, 2026-07-09)

## Matched-pair profile reading (features matched, outcomes divergent)
Pair A: users 2988 (rate .32) vs 42642 (.93) — both ~34 ratings, medPR .99, all 1989-1995.
  42642 = pure mega-canon (Schindler, Gump, Braveheart, Lion King). 2988 = mid-list 90s + light
  arthouse (Three Colors: Red, Farewell My Concubine, Sliver, The Ref). If anything 2988 reads as
  MORE of a movie person; the LLM said the opposite.
Pair B: 48632 (.28) vs 76811 (.83) — both broad-era mainstream geeks. 76811 = prestige canon
  (Nolan/Scorsese/war epics + standup + TV); 48632 = franchise/comedy/animation geek.
READINGS: (1) candidate REAL missed feature: CANON-CORE SHARE — mass in the absolute all-time
  canon (top-50/top-250-equivalent) vs merely-top-800-popular; percentile features SATURATE at .98+
  and cannot see this. Prestige-orientation (canonical dramas over franchises/comedy) may be what
  the judge reads as "serious film person -> knows everything". (2) Beyond that, the pairs look
  partly ARBITRARY — consistent with census features failing user-level CV. Implication: part of
  the residual dial models JUDGE idiosyncrasy, not human trait. Fine for reproducing the judged
  world; a CAVEAT for human-transfer claims (the human study measures the true human ICC).

## TODOS RECORDED
1. IRT DEEP-DIVE (author-flagged): upgrade fits toward 2PL (per-question DISCRIMINATION, not just
   difficulty); person-fit statistics to flag aberrant users; TEST-INFORMATION functions.
   And the big one: CAT — computerized adaptive testing — 40 years of proof that ADAPTIVE question
   selection beats fixed tests when measuring latent person traits; max-information item selection
   is the canonical machinery for the knowledge-map estimation half of our interview. Cite as the
   existence proof + adopt the selection math. If statics win against a CAT-informed policy in a
   world where person traits gate information, THAT is a finding demanding explanation.
2. FEATURE: canon-core share (top-50/250 mass; de-saturate the popularity percentile top end).
3. NEW GATE G5 — VALUE-SIGNAL PRESERVATION: the LLM's value answers carry real taste
   (masked corr .547 vs held-out ratings); the DISTILLED value channel carries corr .177 — the
   synthetic explicit channel is heavily DILUTED. G3 checked only "not too clean"; nothing gated
   "carries enough". Before any policy trains: upgrade the value model (stronger CF/affinity
   features, e.g. neighbor-based prediction) + gate at corr >= ~0.4. Without this the explicit
   channel undertrains policies.
4. Judge-noise caveat (above) for any human-transfer wording.

## Answers of record (author questions)
- Fuel systematic? G2 proves the WORLD CONTAINS the measured heterogeneity; it cannot prove a
  policy converts it. Cold-start note: at t=0 the policy sees NO profile — territory AND dial are
  discoverable only in-conversation; dial fastest (answer rates), territory from answer locations.
  Every previously-identified masking mechanism (pool, fold, baselines, scale, sample) is now
  removed; CAT provides the theoretical prior that adaptivity wins when person traits gate
  information flow. Statics may still win at short horizons — that is THE open experiment, now
  finally measurable.
- Taste signal in LLM data: YES (corr .547) — but see G5: the distilled world currently transmits
  only a third of it; fix before policy training.
