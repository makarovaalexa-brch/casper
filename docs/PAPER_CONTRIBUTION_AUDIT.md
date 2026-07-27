# PAPER CONTRIBUTION AUDIT — what's a contribution vs. our own mistake

> Written 2026-07-27 from a four-corpus sweep (current results, historical `experiments/*.md`,
> the 64-note memory ledger, baselines + paper drafts). Purpose: for each experiment/ablation,
> decide **is it a genuine contribution** (novel + validated against a reasonable *published*
> baseline), **a fix/diagnosis worth showing as lineage**, or **our own mistake/dead-end** (take
> the final working version only, or omit). Governing rule (CLAUDE.md HARD RULE 6): never market
> a negative as a contribution.

## One-paragraph verdict
Only **two** things in the whole record are validated against a *reproduced published baseline*:
(1) the **C1 reproduction bridge** — EASE, RecVAE and Mult-VAE snap to the literature on the faithful
ML-20M Liang split (DAE does **not** — it fails its snap); and (2) the **G0 non-inferiority tie** of our
frozen tower to RecVAE on the canonical ML-25M ruler (a *tie*, not a beat). Everything else is measured
against *internal* comparators (our own pop-item-ask, our clip-up convention, oracle upper bounds, or
best-effort reconstructions of Golbandi/Bıyık that were never snapped). The genuinely novel, still-standing
work is narrow and honest: the **signed fold-to-point concept operator** (fixed by a training recipe, beats
the item channel on the tail), the **i25 recommender recipe**, a **small-but-significant "adaptivity pays on
items"** result, and a strong set of **methodological lessons** (the circularity firewall, the certification
battery, the vacuous-gate and item-mask diagnoses). The distillation teacher, richer answer imputers, the
Kalman/PrecAcc line, the continuous-policy wins, "open-recall beats probing", and the entire LLM-answerability
apparatus are all null or dead-ended **by our own later rulings** — one-line ablations or lineage at most,
never claims.

---

## The anchor: BASELINE LEDGER (the only published-baseline validation we have)
Two-stage: reproduce the literature's NDCG@100 on ML-20M (`experiments/baselines/snap_ml20m/`), then re-run
the same code on our canonical ML-25M Liang ruler (`experiments/baselines/ml25m_liang/`), full/tail NDCG@10.

| Baseline | Published (ML-20M @100) | Our ML-20M snap | Snap | On our ML-25M ruler — full/tail @10 (@100) |
|---|---|---|---|---|
| **EASE** (Steck'19) | 0.420 | 0.4203 | **PASS** +0.0003 | **0.3476 / 0.2441** (0.4375) |
| **RecVAE** (Shenbin'20) | 0.442 | 0.4425 | **PASS** +0.0005 | **0.3540 / 0.2497** (0.4537) ← frontier/bar |
| **Mult-VAE** (Liang'18) | 0.426 | 0.4223 | **PASS** −0.0037 | not yet on ruler (TODO) |
| **Mult-DAE** (Liang'18) | 0.419 | 0.3969 | **FAIL** −0.0221 | not run (do not cite) |
| EDLAE (reimpl.) | ~0.427 approx | 0.4167 | approx −0.010 | 0.3433 / 0.2395 (0.4330) |
| iALS / Item-kNN / Pop | approx / floor | below approx | advisory | 0.244 / 0.243 / 0.135 (landscape only) |
| Golbandi node-rec | Netflix RMSE only | not snapped | best-effort | 0.3065 / 0.1895 |
| belief-MF (Bıyık/ConTS core) | own task | not snapped | best-effort | 0.2019 / 0.1200 |
| **Tower T2′** (ours) | — | — | — | **0.3487 / 0.2471** → **ties RecVAE** (Δ−0.0053, CI ±0.0070) |

**RECORD CORRECTIONS (verified against the committed JSONs):**
- The **canonical-ruler bar is RecVAE 0.354 / EASE 0.348 / EDLAE 0.343**, NOT the `0.508/0.523` figures that
  float in some notes — those are the **retired 500/500 arena** and must never be cited as the Liang ruler.
  (Memory `ml25m-canonical-liang-split` carried the wrong number; corrected 2026-07-27.)
- **DAE is not a reproduced baseline** — it failed its snap (−0.022). The chapter's "queued @100 0.419" is a
  *target*, not an achieved number.

---

## Disposition table (grouped by verdict)

### CONTRIBUTION — novel + on the canonical stack (but validation still mostly internal; see caveats)
| item | numbers | published anchor | paper |
|---|---|---|---|
| **Signed fold-to-point concept operator** (fixed by the recipe: signed four-band SEL + kc/member-drop curriculum + NDCG-NLL) | capture 1–6%→~25%; concepts beat items on **tail every budget** (k8 .121 vs .084, +44%), lead full thru k4; G0 item bit-tie held | vs pop-item-ask **(internal)** — NOT yet Golbandi/EDDI/PEBOL | SHOW-FINAL-ONLY |
| **i25 recommender recipe** (native RecVAE z + zero-init sum-pool residual) | ties the frozen tower to RecVAE at G0 (0.3487/0.2471 vs 0.3540/0.2497, tie) | RecVAE (snapped) | SHOW-FINAL-ONLY |
| **Adaptivity pays on items** (1-level tree, out-of-sample) | Δfull +0.0015 CI[.0007,.0024], Δtail +0.0016 CI[.0008,.0024], sig | vs static schedule (internal); frames vs Golbandi/Sepliarskaia | SHOW-FINAL-ONLY / ABLATION-LINE (small) |
| **C1 reproduction bridge + canonical ruler** | EASE/RecVAE/Mult-VAE snap to literature | EASE, RecVAE, Mult-VAE | SHOW-FINAL-ONLY (the G0 table) |

> **The one gap that blocks the headline:** "concepts beat items" and "adaptivity pays" are measured against
> our *own* pop-item-ask and static schedules — **no published elicitation baseline underneath them yet**.
> Until they are run against the **Golbandi cold-start interview tree** (and ideally EDDI/PEBOL), they are
> strong internal results, not certified contributions. This is the single most important thing to add.

### LINEAGE-FIX — a bug/wrong design we fixed, where the fix or diagnosis is itself the finding (SHOW the journey)
| item | the lesson | paper |
|---|---|---|
| **Answer-model circularity firewall** (geometric answer = shares the recommender's latent → circular; behavioral SEL is the non-circular fix; CKA(G,u\*) 0.406 > SEL 0.351; weak/strong bracketing 14× confirms it) | *the answer model is the load-bearing assumption in any concept-elicitation claim* — reframes a sub-literature's eval | **SHOW-LINEAGE** (strongest methods contribution) |
| **Vacuous capability gate** (a sign-flip gate PASSED on a channel that, as deployed, never emitted a negative answer — positive-only clip; only the deployment-currency gate caught it) | *capability gates are vacuous unless the deployed model exercises the capability* | **SHOW-LINEAGE** |
| **Certification battery G0–G9 + C1–C3** (G0–G2 outcome gates pass 9 distinct false-positive constructions) | a reusable instrument-certification protocol; semantics need G3–G9 | **SHOW-LINEAGE** |
| **Item-mask replication bug** (`excl=prof|asked` deleted held-out popular targets → fake item "crater"; fix `excl=prof`) | apples-to-apples cautionary tale; the corrected small positive is the citable number (+0.0054 full/+0.0039 tail behavioral) | SHOW-LINEAGE (bug) + final number |
| **Seed-hacking in training not just eval** (a learned-policy win was a favourable *training* seed; died on train-seed average) | statistical-hygiene lesson | SHOW-LINEAGE / OMIT the dropped win |
| **Fold-to-point vs additive-union** (union saturates on redundant tags; point/intersection compounds) | the mechanism that justifies the winning operator | SHOW-LINEAGE |
| **Fold/answer-model-decides diagnosis** (E0f: additive `z+ηaq` is a taste-gradient step only for geometric answers → honest real-rating fold-in HURTS; motivates the learned fold) | why the additive fold was mis-specified | SHOW-LINEAGE (methods spine) |

### NEGATIVE-RESULT — real but null (≤1 ablation sentence; must NOT be sold as a win)
- **Distillation teacher is NULL** — distilled 25.0–25.7% ≈ no-distill 25.5% ≈ shuffle-canary 25.9%; a random
  teacher matches. The *recipe*, not the KL teacher, fixes the fold. (Pre-empts "why not distill?".)
- **No non-circular imputer beats behavioral SEL** — SEL⁺/content tie within ~0.003; ExpoMF/PITF worse. SEL is
  the answer ceiling → *the answer model was never the lever; selection is*.
- **The whole blind adaptive-prober ladder (a2→a6, K-map in-loop)** — six increasingly clever blind probers,
  none beats a popular-item static; the prize exists only with privileged info (Branch B). Honest one-liner.
- **Adaptivity is a clean-channel-only effect** — under a noisy answer channel a noise-adapted static beats the
  trained actor (optimization gap, not expressiveness).
- **Concept granularity / NDCG-VoI-at-cold-start / capacity–length tradeoff** — diagnostic nulls; ablation lines.

### OPEN FAILURES — real failing controls that MUST appear as limitations (not smoothed over)
- **Volume-leak gate FAILS** — ridge R²(signed fold latent → log user-volume) 0.025→**0.249** vs the
  pre-registered ≤0.05 bar. Counterfeit-clean waiver attached, but the gate is failing. Author ruling pending.
- **G5 attribute-semantics does not hold** — trained concept operators learn "what X-likers watch," not
  "members of X" (member-AUC 0.630; disc Spearman 0.948 vs pop / 0.030 vs member-ness). Fine for
  recommendation, a gap for any *stated-attribute-semantics* claim → deferred to the C3 human round-trip.

### MISTAKE / DEAD-END — omit, or keep only the final corrected version (never narrate the journey)
- **+36% "answerable concepts beat item-asking"** — circular geometric answer × weak-ruler artifact. Honest
  version is the small behavioral positive above.
- **Kalman belief-pool → VarHead → PrecAcc** — incompatible with the strong recommender (craters full
  0.167→0.097); the whole amortization/decoupling arc is design churn. Keep only the "don't put Bayes in the
  pooling" lesson.
- **Paper-C continuous-policy wins + FTREC/unified-recommender numbers** — retired arena + circular answers;
  the ~10 policy-learning nulls are the only durable (negative) content.
- **"Open-recall beats the continuous probe ~4×"** — overturned by our own equal-budget test → a wash.
- **LLM-answerability apparatus** (gates "passed" → audit found popularity-domination, per-call flutter,
  cross-family FAIL) — retired. Keep only "answerability = structural rule, not an LLM-validated estimand."
- **"Concepts are not directions in CF space"** (retracted next day by whitening) · **"concepts lose to items
  as a router"** · additive/learned-emb concept heads · fusion-token line — all superseded by the recipe.
- **pbC / paord 0.4946/0.3372 anchors** — irreproducible (training forward never committed); never cite.
  Cite pb2 0.4917 or the on-ruler EASE/RecVAE instead. (→ HARD RULE 10.)
- **Pre-Jul-22 interview magnitudes** (+36% tail, +47% adaptivity, all B/C/D numbers) — retired LLM answer
  environment; directions survive as priors, magnitudes do not.
- **Early-arc artifacts** — non-standard NDCG deflation, frozen-instrument nulls, popb-vs-decoder-bias metric
  bug, survivorship-mean bug, static-contamination "routers would win" (retracted), no-clue fold (n.s.).
  Internal methods hygiene; omit as results.

---

## Where the paper shows LINEAGE vs takes the FINAL version (the author's question, answered)

**Show the lineage (the journey is the point):**
1. The **circularity firewall** — the retraction of the old "+36%" and the CKA/oracle-B guard *is* the
   methods contribution. This is the strongest thing we have; write it as a discovery, not a footnote.
2. The **vacuous-gate** and **item-mask** diagnoses — short, quotable "how we avoid fake wins" stories.
3. The **fold-is-lossy → recipe-fixes-it** arc, but **only** the diagnosis + the fix — *not* the distillation
   detour (that's the ablation line "a dense KL target added nothing over the rank loss").
4. The **certification battery** as a reusable artifact (with the 9 false-positive constructions it catches).

**Take the final version only (never narrate how we got there):**
- The **signed fold-to-point concept operator** (λ=1.0) — show the fixed fold; omit the 1–6% broken clip and C-full crater.
- The **i25 recommender recipe** — show z = native + zero-init residual; omit the Kalman/VarHead/PrecAcc churn.
- The **corrected item-vs-concept table** — show the flat item arm; omit the masking crater.
- The **G0 baseline table** — show EASE/RecVAE/Mult-VAE snapped; omit DAE (it failed).

**Omit entirely:** everything in the MISTAKE/DEAD-END list; all pre-Jul-22 magnitudes; the arena numbers.

---

## Actionable next steps this audit surfaces (all await author decision)
1. **Add a published elicitation baseline** (Golbandi cold-start tree; ideally EDDI/PEBOL) under the
   concept-vs-item and adaptivity claims — without it they are internal-only, not contributions.
2. **Finish the G0 table** — Mult-VAE on the ruler; drop DAE or label it a non-reproducing baseline.
3. **Rule on the volume-leak gate** (fallback grid `C_NEG∈{1,2,4}` / orthogonalised fold) — it is failing.
4. **Correct the ruler-number conflation** everywhere it appears (done in memory; check the chapter tables).
