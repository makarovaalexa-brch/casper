# PAPER A CONTRIBUTION AUDIT — the instrument, and our own method pieces

> Rewritten 2026-07-27 after author scope correction. **Paper A is the INSTRUMENT paper.** Its
> contribution is *the recommender that makes policy measurement possible* — a frozen certified
> shallow-SOTA tower + a conjugate-Gaussian belief layer + a channel-agnostic fold-in interface.
> Elicitation is **evidence the instrument works** (it must move NDCG on *both* channels), not a
> headline. **Out of scope:** elicitation policy (Paper B), continuous queries (Paper C),
> adaptivity-as-a-claim (the 1-level tree is a private diagnostic, not a gate).
>
> **The lens for everything we built** (author's framing): each method piece is one of
> - **(P) PICKED** — we compared published options and chose/justified the appropriate one (due
>   diligence: *we didn't reinvent the wheel*; show the comparison, claim nothing novel).
> - **(B) BUILT-ON-TOP** — we extended a published method into a channel its base architecture
>   cannot express, and it works.
> - **(A) ARCHIVED** — we tried something, it didn't work; **take the working version, archive the
>   rest, never show the journey externally.** An internal bug we fixed is not a contribution.

## Scope correction vs. the first draft of this audit
The first version elevated internal course-corrections into "contributions." That was wrong:
- The **circular geometric-answer episode** is an internal bug we caught and fixed. Nobody outside saw
  the old draft. We simply *use* the correct (behavioral SEL) answer model. **Nothing to retract, nothing
  to show** — archive it.
- **"Concepts beat items / concepts own the short interview"** is Paper-B elicitation material, and its
  "items win the long interview" half was a broken-fold artifact anyway. For Paper A it collapses to one
  evidence sentence: *elicitation moves NDCG on both channels*.
- **Adaptivity** is not a Paper-A gate. The tree probe was for the author.

---

## Method-piece ledger (what WE built, vs the published anchor)

| # | Method piece (Paper-A requirement) | Published anchor we picked/built on | Verdict | Status / evidence it works |
|---|---|---|---|---|
| 1 | **Recommender substrate** (R1: full-profile SOTA) | EASE (Steck'19), RecVAE (Shenbin'20), Mult-VAE (Liang'18) | **P** — compared the shallow-SOTA family, reproduced them, froze RecVAE as backbone | C1 snap: EASE/RecVAE/Mult-VAE reproduce literature; RecVAE frontier **0.3540/0.2497** is the bar |
| 2 | **Set-encoder fold-in** (R2: any-length, accepts SETS) | permutation-invariant set encoders — EDDI/Partial-VAE (Ma'19), TaNP (Lin'21); DeepSets | **B** — zero-init residual set encoder on the *frozen* RecVAE encoder; folds any-length set; empty set = frozen bias by construction | G0 identity holds (empty=bias, z-norm 0); tower **ties native RecVAE** at full (0.3487/0.2471, Δ−0.005 within CI) |
| 3 | **Graded/"star" value channel** (R1/R6/R7: signed graded values) | binary implicit feedback — Mult-VAE/RecVAE are **positive-only, cannot express dislike**; ordinal MF | **B** — graded (item, level) tokens carrying hated…loved | Tower ties RecVAE. **⚠ GAP: full-profile graded premium ≈ +0.004 only; the k=2..full graded-vs-binary interview-length curve was DESIGNED (GATE_BATTERY:34) but NOT run — see below.** |
| 4 | **Out-of-catalogue concept channel** (R3) | Bıyık soft-attributes (co-trained) `biyik2023soft`; Li'21 unified item+attribute; tag-genome | **B** — signed fold-to-point concept tokens on the frozen tower; zero-init gate → items bit-safe | Concept fold moves NDCG off the 0.128 intercept (both-channel evidence). Archive: broken clip, C-full, additive-union, learned-emb — take final λ=1.0 fold-to-point |
| 5 | **Continuous direction tokens** (R3) | novel leg (interpretability via sparse coding/OMP) | **B** — direction-token interface (same fold operator) | Interface exists; deep eval is future work, not a Paper-A gate |
| 6 | **Belief layer** (R4: uncertainty-native) | conjugate-Gaussian belief — Bıyık, BCIE/Toroghi'23, Li'21; ConTS Thompson; **Kalman** | **P + A** — tried Kalman/conjugate pooling → it **craters full NDCG** on the strong frozen decoder (0.167→0.097): *archived*. Use the precision-accumulator; wedge = **frozen tower + exact conjugacy** vs their co-trained/approximate | Working belief head; the Kalman incompatibility is an archived detour, not a result |
| 7 | **Concept ANSWER model** | ExpoMF (Liang'16), PITF/TagMF (Rendle'10), content-projection, SEL⁺/BM25-EB, behavioral SEL (Hu-Koren-Volinsky implicit) | **P** — ran the imputer panel; **behavioral SEL is the ceiling**, the fancier published imputers don't beat it (ExpoMF/PITF worse) | The answerer-contrast table **is** the contribution here: due diligence, "picked the appropriate published estimator." Claim nothing novel. |
| 8 | **Fold distillation** | Privileged Graph Distillation (Wang'21), DropoutNet, Heater, GAR, ALDI | **P** — tried the published cold-start distillation lever to fix the concept fold; **teacher is NULL** (distill ≈ no-distill ≈ random teacher); the plain recipe suffices | One honest comparison line: *we tried the obvious published lever; it wasn't needed*. Not a failure to hide — it's due diligence. |
| 9 | **Acceptance battery** (G0–G9, C1–C3) | outcome gates + our own false-positive constructions | **B** — the reusable certificate | This is the instrument-validation apparatus; the paper proposes it as portable |

**Reading of the ledger:** the three genuine "built-on-top" novelty legs are exactly what the chapter already
claims — **graded values, out-of-catalogue concepts, continuous tokens** (impossible in the item-indicator
basis of every R1-bar model). The substrate, set-encoder and belief layer are correctly framed as
*picked/extended established prior art* (Bıyık/BCIE/Li/EDDI/TaNP, all cited). The answer-model panel and the
distillation comparison are the two "we compared published approaches and picked/justified" pieces — show them
as comparison tables, claim nothing novel. Everything else is archived.

---

## What shows the instrument works (evidence — this is what elicitation is FOR here)
- **G0 strength:** the frozen tower ties RecVAE (published, snapped) at full profile.
- **C1 bridge:** EASE / RecVAE / Mult-VAE snap to the literature. **Drop DAE** — it failed its snap
  (target 0.419, got 0.397, −0.022); it is not a reproduced baseline.
- **Both channels elicit:** item fold-in and concept fold-in each raise NDCG from the cold intercept
  (0.128/0.019) across k — the "works for both channels" evidence the instrument needs.
- **G0 identity / no-harm:** empty evidence set = frozen bias (nothing folded, nothing broken).

---

## GAPS to close for Paper A (concrete, cheap, high value)
1. **The star/graded interview-length curve on the TOWER** — `graded vs binarized, k=2..full`. It was
   *designed* as a gate (`docs/design/GATE_BATTERY_INSTRUMENT.md:34`, "gradedness must win") but **never
   run on the neural tower**; only a full-profile premium (~+0.004) and a k-indexed *classic-baseline*
   comparison exist (`experiments/baselines/ml25m_liang/rating_baselines_val.json`). Short interviews are
   exactly where the extra bits per rating should pay; the full-profile +0.004 is the worst case for
   gradedness. **This is the highest-value missing instrument evidence.** (Author's "star curve" question.)
2. **Mult-VAE on the ruler** — finish the G0 table (it snapped on ML-20M but has no ML-25M number); drop or
   explicitly label DAE as non-reproducing.
3. *(Optional, instrument-validation only, not a "beats" claim):* a published elicitation baseline (the
   Golbandi cold-start tree) under the both-channels curve, to anchor "elicitation works" against prior art.

---

## ARCHIVE — take the working version, never show externally (not Paper A)
- The **circular geometric-answer** episode (internal bug → fixed; use behavioral SEL).
- The **broken positive-only clip, C-full, additive-union, learned-emb** concept attempts → show only the
  final signed fold-to-point (λ=1.0).
- The **Kalman → VarHead → PrecAcc** churn → show only the working belief layer + the one-line "Kalman
  pooling is incompatible with a strong frozen decoder" justification for the design choice.
- The **LLM-answerability apparatus**, all **pre-Jul-22 interview magnitudes**, the **pbC 0.4946 anchor**
  (irreproducible), the retired **500/500 arena** numbers.
- **All Paper B (elicitation policy) and Paper C (continuous) material**, incl. "concepts beat items",
  the adaptivity tree, and "concepts own the short interview" as a headline.

## Record fix (verified vs the committed JSONs)
Canonical Liang ruler = **EASE 0.3476/0.2441, RecVAE 0.3540/0.2497, EDLAE 0.3433/0.2395**. The `0.508/0.523`
figures are the **retired 500/500 arena** — never cite as on-ruler. (Fixed the MEMORY.md hook line.)
