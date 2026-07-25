# DESIGN — Faithful replication of OLD (pre-RecVAE) Paper B onto the NEW canonical ruler

## ★ ADDENDUM 2026-07-25 — SINGLE-RUN faithful version (NOT the §3 2×2)
Author directive supersedes the §7 "blocked" stop: run the **one faithful cell** = the old ML-1M
reconstruction-encoder method, method held byte-faithful, moved onto OUR canonical ML-25M Liang ruler.
No 2×2, no ML-1M rebuild, no recommender-strength decomposition — just the new-ruler lift curve
side-by-side with the OLD recorded numbers, to answer: **did the strong short-interview lift survive?**

Faithful stack (all recipes verbatim from `scripts/_archive/paper2_old/`): biased-SVD D=64 LAMF=0.05 LR=0.01
EP=15 (`ml25m_build_svd_full.py`) + concept-aware attention reconstruction encoder with LEARNED enc+Qp+Ec
(`freeze_concept_encoder.py`; genre tokens dropped per the ML-25M port) + genome concepts relevance>0.5,
≥30 members (`ml25m_concepts_full.py`) + GEOMETRIC answer (no LLM, no SEL) + item & concept short-interview
T=8 with static-pop AND EIG selection (`answerability_concept.py` infogain_items/infogain_concepts).
Decoder = the method's own `score = popb + Qp·u` (this IS the weak old recommender — the point of the test).

Deltas vs the archived code = dataset + split ONLY. The canonical proc/*.csv are binary (rating>3.5);
the method is explicit-rating based, so `pb_prep.py` reproduces the exact Liang identity maps (seed 98765,
asserts vocab == proc/unique_sid.txt byte-for-byte) and recovers ALL-BAND explicit ratings from raw ML-25M
restricted to the 18,359 vocab — exactly as the archived ML-25M port trained its biased-SVD. Fold-in
(test_tr) / targets (test_te) / head mask are the canonical ruler, untouched. All 10k test users, all items,
all concepts (HARD RULE 1). Code: `scripts/paper2_repl/pb_{prep,train_enc,interview}.py`. Controls:
canonical-snap (full-profile fold), answer-permutation existential, disjoint fold-in/target leak check.


> Pre-registered per HARD RULE 10 / "design sheet before any run". Question: **was the old
> pre-RecVAE Paper B short-interview lift a genuinely stronger METHOD, or an artifact of the OLD
> RULER (dataset + recommender strength + metric/split)?**
> Status: **archaeology complete; faithful fresh run BLOCKED (see §7). This sheet pre-registers the
> replication AND documents the blocker + closest-faithful-proxy.** No run launched.

---

## 1. What "old pre-RecVAE Paper B" EXACTLY was (archaeology, cited)

"Pre-RecVAE" pins the era to **June 2026**: the RecVAE instrument rebuild landed ~Jul 4
(`instrument2-rebuild-arc`), and the LLM answerer (mm_* grids, answerer v2.1) landed ~Jul 9
(`answerer-v2-flutter-and-repair`) — both AFTER RecVAE. So old Paper B = the **June ML-1M
reconstruction-encoder work**. Correcting the task brief's assumption: **its answers were NOT
LLM-derived** — the mm_* / answerability-pmodel environment is the *later* (post-RecVAE) line.

| element | old Paper B (June 2026) | source |
|---|---|---|
| **Dataset** | **ML-1M** (6,040 users / 3,706 items, dense ~166 ratings/user), like = r≥4 | `encoder_recon.py:11-18`, `answerability_concept.py:8-15` |
| **Recommender** | **frozen biased-SVD** `Q_svd`/`bi_svd` (D=64) + **attention reconstruction encoder** (attn-pool over revealed (item-factor, residual) tokens → user vec u); decoder FIXED = `popb + Q·u`. Trained on masked/shuffled reveal subsets k=1..12, IPS/inverse-pop weighted, to reconstruct unrevealed likes | `encoder_recon.py`; mem `paperB-reconstruction-encoder-breakthrough` |
| **Concept definition** | **761 genome tags** mapped to ML-1M movies via `genome-scores.csv` (genome score **>0.5** = member); a concept is answerable iff user rated **≥2** member items; learned concept channel `Ec` (`freeze_concept_encoder.py`, fixes the OOD bug where enc saw only items+18 genres) | `answerability_concept.py:30-45`; mem `answerability-concept-channel-works` |
| **Answer model** | **GEOMETRIC** (`ANSWER=geom`): like/dislike = whichever fold moves belief toward the user's true taste `u*` = fold(known-half profile); state-independent (POS if `u*·Ec > per-user-mean`). **NO LLM.** ('mean'-residual was the step-1 variant) | `answerability_concept.py:1-6` |
| **Answerability (structural)** | item answerable iff user rated it; concept iff ≥2 rated members. NOT a learned/LLM model | `answerability_concept.py`; mem `answerability-concept-channel-works` |
| **Interview protocol** | cold-start empty, **fixed T=8** budget, full-catalogue; selectors: static (pop/random/entropy/HELF) and **EIG** (deployable greedy: rank candidates by expected like-coverage over rest-of-profile using belief only, then fold the true received answer) | `encoder_adaptive.py`, `encoder_realizable.py`; mem note |
| **Metric** | **full-catalogue NDCG@10** + **Cremonesi head-33% tail** NDCG@10; cold q0 ≈ **0.284 full / 0.064 tail** (some scripts 0.293/0.305) | `RESULTS.md` PART A4, S4-CAL2 |

### Recorded OLD short-interview numbers (the "felt so much stronger" gains)
- **Reconstruction-encoder + EIG@8** (canonical instrument): FULL 0.284→**0.364 (+0.080)**, full-profile 0.338, oracle 0.514; TAIL 0.064→**0.191 (+0.127)**, full-profile 0.162, oracle 0.334. EIG@8 **beats folding the whole profile**. `RESULTS.md:261`.
- **Answerable-concept vs item-asking** (T=8, q0=0.293, 150-user split): conc_pop **TAIL 0.116 vs item 0.085 (+36%)**; full 0.315 vs 0.307. mem `answerability-concept-channel-works`.
- **Concept-EIG** tail **+0.084** (~85% of item-EIG +0.099), profile-restricted. `RESULTS.md` PART T.
- Answerability motivation: item cold-answerable **1.9/8**; answer-rate items 2.1% / concepts 57% / genres 63%.

### The confound (why the OLD ruler could inflate the lift) — three bundled axes
1. **Dataset**: ML-1M dense small-catalog (3,706 items) vs new ML-25M sparse large-catalog (18,359 items).
2. **Recommender strength**: weak biased-SVD+attn encoder (full-profile 0.338 full) vs strong RecVAE/EDLAE-class (**EASE 0.5078/0.3394, EDLAE 0.5230/0.3464**, oracle-Z 0.994). A weak recommender leaves huge cold→oracle headroom (0.284→0.514) for questions to fill; a strong one has already absorbed it (`overnight-adaptivity-study`: *"collaborative generalisation substitutes for adaptivity on real catalogs"*).
3. **Metric/split/protocol**: full-cat NDCG@10 + head-33% tail + q0≈0.284 vs Liang 80/20 fold-in/target, tail=190-head-mask, cold full ≈0.19.

---

## 2. The NEW canonical ruler (fixed, from CLAUDE.md)
Liang-recipe strong-generalization split on ML-25M, catalog-restricted to the 18,430 arena movies
(≥20 ratings; final vocab **18,359**); ratings >3.5, min_uc=5; **train 140,768 / val 10,000 / test
10,000 users**; per-user 80/20 fold-in/target; **seed 98765**. Data `data/ml-25m/proc/`; runner
`src/baselines/run_ml25m_liang.py`. Metric: **full AND tail NDCG@10** (tail = top-33%-mass head mask,
190 head items), scored with the recommender's **learned decoder bias** (never a popb floor). Old
500/500 arena split is elicitation-continuity only.

---

## 3. Pre-registered design — separating RULER-effect from METHOD-effect

Hold the **METHOD fixed** (concept def genome>0.5 / ≥2 members, geometric answer, static-pop + EIG
selection, T=8) and vary the ruler along the two structural axes in a **2×2 (dataset × recommender)**:

| cell | dataset | recommender | isolates | expected if pure-ruler-artifact |
|---|---|---|---|---|
| **A** (OLD, faithful re-run) | ML-1M | weak biased-SVD+attn encoder | reproduces the recorded number | big lift (+0.08/+0.127; +36%) |
| **B** | ML-1M | strong (RecVAE/EASE on ML-1M) | **recommender strength**, dataset held | lift shrinks → strength is the driver |
| **C** | ML-25M | weak biased-SVD+attn encoder | **dataset**, method held | lift persists → dataset is *not* the driver |
| **D** (NEW ruler) | ML-25M | strong canonical | the canonical stack — the number to compare | small lift |

**Read-out:** if `lift(A) ≫ lift(D)` and `lift(B) ≈ lift(D)` while `lift(C) ≈ lift(A)` → the driver is
**recommender strength** (the new recommender absorbs the signal — *not* a genuinely stronger old
method). If `lift(C) ≈ lift(D)` while `lift(B) ≈ lift(A)` → the driver is **dataset**. The two-axis
design attributes the shrink rather than leaving it tangled. Both arms use item AND concept questions.

**Report:** the short-interview lift **curve q1..8** (full + tail NDCG@10, over cold q0) for each cell,
side-by-side with the OLD recorded curve. Headline = `lift(D)` vs `lift(A)` with the B/C decomposition.

### Ns, action space, baseline symmetry, metric, MDE
- **Ns**: new-ruler cells = all **10,000 test users**, all **18,359** items, all genome concepts (NO subsampling — HARD RULE 1). ML-1M cells = the old ~604/150 test-user protocol (faithful).
- **Action space**: item questions (structural answerability: rated ∨ pop>τ) + concept questions (genome>0.5 membership, ≥2 rated members). Same space for every arm.
- **Baseline symmetry (static arm beyond reproach)**: static-pop selector, **identical** T=8 budget, identical candidate pool, selection on a disjoint profile half, eval on the other half — the exact symmetry the old EIG arm used. Every adaptive arm compared to this static arm at equal budget.
- **Metric + MDE**: full+tail NDCG@10. At 10k test users, paired-bootstrap CI half-width ≈ ±0.002 → **MDE ≈ 0.004 (2σ)**; pre-register it. The old +36%/+0.08–0.127 lifts are 20–60× MDE, so a real shrink to near-MDE is detectable. ML-1M cells (150–604 users) are noisier — report CIs.
- **Controls on every quantitative arm (HARD RULE 5)**: (i) existential/shuffle — permute answers, lift must vanish; (ii) canonical-snap — q0 and full-profile must reproduce the ruler's recorded cold/full NDCG before any curve is trusted; (iii) leak check — disjoint select/eval halves, train-only vocab, no held-target peek in EIG (belief-only).

### Shortcuts + non-lossy alternatives
| shortcut | non-lossy alternative / ruling |
|---|---|
| Answer model | **Use the GEOMETRIC answer (the OLD faithful convention).** This is faithful and needs NO LLM. **Do NOT substitute the clip[0.25,1] SEL convention** (post-Jul-24, drops the dislike half) — that would destroy the comparison. If SEL is shown at all it is a **separate labeled arm**, never "the replication". |
| Concept membership | genome>0.5 (faithful to `answerability_concept.py`); full genome table, no tag subsampling. |
| Recommender weights | none reusable (§7) — must retrain from committed code, not reload a checkpoint. |
| User/item sampling | **none** (HARD RULE 1): all 10k users, all items, all concepts. |

---

## 4. Gates (must pass before any cell's curve is reported)
- **G-snap**: cell's cold q0 and full-profile NDCG reproduce that ruler's recorded anchors (ML-1M q0≈0.284/0.064, full-profile 0.338/0.162; ML-25M cold ≈0.19, full ≈0.49–0.52) within noise.
- **G-shuffle**: answer-permutation control lift ≈ 0.
- **G-symmetry**: static and adaptive arms share budget, pool, split — verified in code.

---

## 5. Prior evidence already bearing on the answer (strong prior, NOT a substitute for the run)
The July line effectively already ran the old method's concept channel on the NEW strong stack:
- **Concepts lose to items as a few-shot router** on the strong model: honest best CONCEPT +0.0112 tail vs best ITEM +0.0168 (concept "wins" were selection-max noise). mem `concepts-lose-to-items-as-interview-router`.
- **Concept escalation** (Jul 24): concepts-only additive saturates/craters; trained C-lite reaches item-parity only at m=2 then hits the **coarseness ceiling cos≈0.83** (fundamental, dataset-independent). mem `concept-channel-escalation-2026-07-24`.
- **The +36% answerability edge depended on items being cold-unanswerable (1.9/8) on ML-1M.** On the new ruler the strong recommender + popularity floor folds items fine, erasing that edge.
- Mechanism: `overnight-adaptivity-study` — collaborative generalization on real catalogs substitutes for what elicitation captured on weak ML-1M.

→ **Strong prior verdict: the old strong short-interview lift was largely a RULER artifact** (weak
recommender × dense small catalog × huge cold→oracle headroom × item cold-unanswerability), NOT a
genuinely stronger method we lost. This sheet's run would convert that prior into a fresh side-by-side.

---

## 6. Old-split bridge control
The cleanest bridge (same method on the OLD split, to re-confirm the OLD number under our own re-run)
= **Cell A**. But the old ML-1M split, biased-SVD factors, and encoder weights are **all gone** (§7),
and **ML-1M raw is absent from the repo**. So the bridge is reproducible only by a from-scratch rebuild
after re-downloading ML-1M — not a reload. This is flagged, not silently skipped.

---

## 7. ★ BLOCKER — faithful fresh run is BLOCKED this session
- **Old recommender weights GONE.** `data/.cache/` is **empty**; `.cache/` is gitignored (`.gitignore:98`) and the artifacts (`Q_svd.npy`, `bi_svd.npy`, `enc_concept.pt`, `Ql_concept.npy`, `Ec_concept.npy`, `ctags_concept.npy`) were **never tracked in git** — unrecoverable. Same class as the `anchor-provenance-hole-pbc-recvae` / HARD RULE 10 provenance holes.
- **ML-1M raw ABSENT.** No `ratings.dat`, no `ml-1m/` dir on disk; never committed (`git log` empty). `data/movielens/` is actually **ML-25M**. The old scripts read `data/movielens/ml-1m/ratings.dat`.
- **Consequence**: every cell needs a **from-scratch multi-hour build** — re-download ML-1M, rebuild biased-SVD, retrain the reconstruction + concept encoders, then port the whole method to a from-scratch ML-25M training (140k users × 18,359 items) — of *uncertain* reproduction fidelity (precedent: pbC/paord "NO-SNAP under any git pin"). This exceeds a scoped faithful run and needs author sign-off (design-before-run + Fable-oversight).
- **NOT a blocker**: the answer model. It is geometric/structural, so **no LLM and no retired-grid dependency** — the "no new LLM calls" constraint is satisfied. (The mm_* grids belong to the later environment and are not needed here.)

### Closest faithful proxy + what it can / cannot tell us
- **Cell D on the existing strong stack** (i25 ep4 tower / EASE on the Liang split) + geometric answer + genome>0.5 concept def + static-pop/EIG, item + concept curves. **CAN**: give the new-ruler lift magnitude for a side-by-side vs the OLD recorded numbers (the headline comparison). **CANNOT** alone separate dataset from recommender-strength (needs cells B & C) — and much of it overlaps the already-run July escalation/router arena.
- **Do NOT** run a version that swaps in the SEL clip[0.25,1] answer, subsamples users/concepts, or reloads a non-faithful recommender and calls it "the replication."

---

## 8. Decision
Faithful fresh replication is **blocked** (weights + ML-1M raw gone; full rebuild = multi-hour, uncertain
fidelity, needs sign-off). Per the task's STOP condition, we **stop at this committed sheet + the §5
strong-prior verdict + the §7 proxy analysis** and surface the fork to the author:
**(a)** authorize the multi-hour 2×2 rebuild for a clean fresh side-by-side, or **(b)** accept the §5
evidenced-prior verdict (ruler artifact) + the closest-faithful Cell-D proxy, or **(c)** run only the
Cell-D proxy now (clearly labeled, recommender-strength confounded).
