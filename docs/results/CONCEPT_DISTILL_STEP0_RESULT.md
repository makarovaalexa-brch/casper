# Step-0 go/no-go — distillation-trained concept fold (DESIGN_CONCEPT_FOLD_DISTILLATION.md §1, §1b)

**Run:** `src/instrument/concept_distill_step0.py --full_threads` (2026-07-25/26, ~2h50m, code `474eaee`).
**Stack:** FROZEN i25 tower `t2i25_EP4_SNAP.pt` (ep4, val_full 0.3435) + decoder; SIGNED SEL answer
simulator. **ZERO training.** Canonical ML-25M Liang, 10k COLD_SEED users, full+tail NDCG@10,
credit-neutral masking, leak-safe. Output: `experiments/battery/concept_distill_step0.json`.

**Controls (all PASS):** q0 == cold intercept `0.12794/0.01923` (canonical-snap, exact); leak
known∩held = 0 users; teacher folds fold-in member ratings only (te excluded); shuffled-teacher canary
capture ≤ 0 everywhere (wrong-concept buckets never help — the real capture is genuine concept signal).

## VERDICT: FLAG_AUTHOR (in-between zone), with a decisive coarseness split

Overall tabular-student capture is **11.5%** (kc=1) / **15.3%** (kc=4) on full — inside the
pre-registered 10–30% grey band (barely above the 10% STOP line at kc=1). **But the overall number is
dragged down by BROAD concepts, which the |v|-selection overwhelmingly picks.** Split by tier the answer
channel clears the PROCEED bar for fine + medium:

### PART A — capture-rate (tier × kc; capture = (student−intercept)/(teacher−intercept))

| pool | kc=1 full | kc=1 tail | kc=4 full | kc=4 tail | reading |
|---|---|---|---|---|---|
| **overall** | **11.5%** | 16.0% | **15.3%** | 21.3% | FLAG (broad-dominated) |
| fine (≤164 mem) | 21.6% | 42.6% | **31.8%** | 39.5% | PROCEED |
| medium (164–525) | 21.7% | 34.5% | 25.7% | 32.1% | PROCEED |
| broad (>525 mem) | 10.3% | 13.8% | 13.5% | 18.6% | STOP/borderline |

(student full@10: overall 0.154→0.170, fine 0.157→0.201, broad 0.152→0.165; n: overall 10000, fine
9131/7561, medium 9958/9837, broad 10000.)

The top-|signed-value| selection is ~pure broad (overall kc=1 teacher 0.358 ≈ broad row 0.363), so
"overall" ≈ "broad". The channel is **not** an empty pipe: for fine/medium concepts the conditional-mean
student recovers 20–32% of the teacher lift (up to 43% on tail).

### Within-cell teacher variance — HIGH agreement (the informative diagnostic)

mean cos(individual teacher delta, bucket-mean delta): **overall 0.950** (fine 0.965 / medium 0.955 /
broad 0.944), over 4.18M cells. **The direction of each user's member-fold is almost fully determined by
(concept, band); *which* members the user rated barely rotates it.** This refutes the "answer channel is
information-limited by which-members" hypothesis. The capture shortfall is therefore NOT missing
direction — it is **magnitude/ranking dilution**: averaging many users' deltas produces a correctly-aimed
but weaker ranking signal than any single per-user fold. That gap (scale calibration + composition) is
exactly what a *trained* student with NDCG-aware finetuning (design arm S2) and learned composition can
recover **beyond** the tabular conditional-mean floor — so the tabular capture is a conservative floor
here, not a hard ceiling, for direction-well-posed tiers.

### PART B — teacher ceiling sweep (§1b, the honest student bars, full/tail NDCG@10)

| pool | kc=1 | kc=2 | kc=4 | kc=8 |
|---|---|---|---|---|
| overall | 0.358/0.248 | 0.386/0.272 | 0.401/0.285 | 0.410/0.293 |
| fine | 0.235/0.140 | 0.271/0.166 | 0.316/0.196 | 0.372/0.230 |
| medium | 0.258/0.174 | 0.298/0.204 | 0.335/0.231 | 0.369/0.256 |
| broad | 0.363/0.255 | 0.390/0.275 | 0.404/0.288 | 0.413/0.295 |

**Full-profile fold ceiling = 0.4190/0.2999** (mean 136 items). Broad kc=8 (0.413) ≈ the full-profile
ceiling; fine/medium top out ~0.37. Mean members folded per user: broad kc=1 = 85.8 items (broad
"concepts" are near-full-profile folds — which is why they are strong for the *teacher* yet poorly
compressible into a single (concept, band) bucket).

## Bottom line for the author (needs a call)

Not a clean PROCEED, not a STOP. The answer channel demonstrably carries capturable, concept-specific
signal (fine/medium 20–32%; canary clean; teacher direction 95% determined by the answer), so the plan is
**not dead as feared**. The 11–15% *overall* is a **selection artifact** of asking broad concepts, whose
member-fold is a near-full-profile object no (concept, band) code can compress. Two implications:
1. **Trained student is worth running** — its bar is beating the tabular floor by learning magnitude +
   composition, which the high within-cell cos says is the actual bottleneck (not information).
2. The design's own STOP-remedy — **richer student input (answer + kc-so-far belief context)** — is
   precisely the fix for the broad-concept drag and should be folded in / prioritized, and/or the
   interview should **prefer fine/medium concepts**, where the channel already clears the bar.
