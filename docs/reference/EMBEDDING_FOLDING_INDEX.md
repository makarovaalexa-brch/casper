# EMBEDDING-FOLDING INDEX — lineage, what worked, NDCG, leaks (2026-07-10)
Aggregated to re-establish ground truth after a session of churn. The point: separate VERIFIED-GOOD
models from this-session inventions, and pin where accumulation broke.

## THE LINEAGE (oldest→newest)
| tag | date | scorer | aggregation / arch | code | ckpt | status |
|---|---|---|---|---|---|---|
| **June encoder (WORKING)** | Jun | biased-SVD | **attention-pool → user vector** (direct; NO fixed-prior/gate; NO surprise) | scripts/paper2/encoder_recon.py (+ _adaptive/_panel/_realizable/_ceiling/_sanity) | data/.cache/enc_unified.pt / enc_concept.pt | **VERIFIED GOOD — accumulated, beat ridge** |
| i25_fold | 07-07 | RecVAE-d512 | **Deep-Sets SUM-pool residual** z=native_z+ρ([sum,native_z,log ntok]) | scripts/i25_fold.py | .cache/i25_fold_best.pt | 5/6 gates; clean 0.4647 vs native 0.4787 |
| i25_fold_v2 | 07-08 | RecVAE | Deep-Sets residual, retrained | scripts/? | i25_fold_v2_best.pt | interim |
| **v3 / v3.1** | 07-09/10 | RecVAE | Deep-Sets SUM-pool residual + two-channel + **SURPRISE** | scripts/i25_fold_v3.py / v31.py | i25_fold_v31_best.pt | val 0.4356; **clean 0.4994 > native 0.482 (accumulated)**; GoT +0.184, prolific +0.316; G5/G6 fail; **on_profile=0.7 OOD** |
| v3_st | 07-09 | RecVAE | Set-Transformer (SAB+PMA) | (contingency) | i25_fold_v3_st_best.pt | REJECTED (worse) |
| v4 | 07-10 | RecVAE | de-OOD mixture, fat tails | scripts/i25_fold_v4.py | i25_fold_v4_best.pt | NaN(q718) → cold-collapse 0.167 |
| v5 | 07-10 | RecVAE | CNP mean-pool | scripts/i25_fold_v5.py | i25_fold_v5_best.pt | moot |
| **recovered** | 07-10 | RecVAE | **attention-pool (softmax) + FIXED-PRIOR + content-confidence GATE** + SURPRISE + unified builder + no caps | scripts/i25_fold_recovered.py | i25_fold_recovered_best.pt | intercept 0.0, no-Q1-drop; **clean 0.32 vs native 0.487 (gap −0.167, accumulated a bit)**; this-session invention |
| **leakfree A/B/C** | 07-10 | RecVAE | recovered arch, **NO surprise**, +entropy/p_E, emergent(A)/explicit-coh(B)/multimodal(C) | scripts/i25_fold_leakfree.py | i25_fold_leakfree_{A,B,C}_best.pt | **SATURATES: cold 0.165 → turn1 0.241 → FLAT; full-profile==interview==0.24; GoT −0.13; tight-cluster 6.4%.** leak-gate PASS (0.0). this-session invention |

## WHAT WORKED (verified)
- **June encoder_recon.py: attention-pool over reveal tokens → user vector, trained by masked-reveal
  reconstruction of UNREVEALED likes, IPS-weighted. NO surprise. NO fixed-prior/gate.** Beat ridge:
  full q8 **0.338** vs ridge 0.305 vs pop 0.291; tail **0.119** vs 0.098. EIG selector +0.079 full /
  +0.127 tail. Oracle ceiling ~0.51. **It ACCUMULATED** (full-profile clearly > few-answer).
- v3.1 (RecVAE, sum-pool residual + surprise) also accumulated: clean 0.4994 > native 0.482. But had
  the on_profile=0.7 OOD + count-confound + cold-intercept issues.

## NDCG QUICK REFERENCE (cold → capability)
- Cold-start (prior) ≈ 0.15-0.17 (cohort-dependent). Full-profile NATIVE RecVAE ≈ 0.487.
- June biased-SVD: q0 pop 0.284 → fold-full 0.334 → EIG@8 0.355 → oracle 0.508.
- v3.1 RecVAE: clean 0.4994 (> native). recovered: clean 0.32. **leakfree: clean 0.24 == interview (SATURATED).**

## LEAKS DETECTED
1. **SURPRISE from full profile** (v3/v3.1/recovered): n_E = members of E in the FULL known profile,
   injected per interview token → the agent "knows" consumption before the interview surfaced it.
   Caught 07-10. (leakfree removed it → G-no-profile-leak PASS 0.0, but that removal also killed
   accumulation → see SATURATION.)
2. **EASE trained on full matrix** (structural): ease_v21.npz B fit on all ratings incl. held items;
   values via EASE reflect held-out → mild collaborative leak. Author verdict: EASE-from-non-holdout
   OK (user's internal knowledge, volunteerable); exclude HOLD-OUT input.
3. **RecVAE scorer trained on full matrix** incl. held items — standard transductive fold-in caveat;
   separate, bigger question (strong-generalization / held-out users).

## THE AGREED-vs-IMPLEMENTED DIVERGENCE (what the author flagged)
Agreed (from the trace ~2h ago): recover the WORKING **attention-pool that ACCUMULATES** (June-style),
drop hand-crafted surprise, unify token builder, no caps, keep a clean cold intercept.
Implemented instead: a NEW **fixed-prior + gated-residual + softmax-attention** wrapper (this session's
invention, never verified good). Softmax weights sum to 1 → convex combination → **bounded magnitude →
cannot accumulate**. June's direct attention-pool had NO such bound and DID accumulate WITHOUT surprise.
=> the saturation is the RecVAE-era ARCH WRAPPER (fixed-prior/gate/softmax), NOT the missing surprise.

## THE BUG TO FIND (for the scan)
Why does leakfree saturate (cold 0.165 → turn1 0.241 → FLAT; full-profile==interview==turn-1) when
June's attention-pool accumulated WITHOUT surprise? Suspects: softmax-convex-combination bounding the
pool; the content-confidence gate w saturating after 1 token; the residual ρ collapsing; a
mask/normalization bug making N tokens ≈ 1 token. Compare leakfree pooling/gate/residual (scripts/
i25_fold_leakfree.py forward, ~line 204-222) against the WORKING encoder_recon.py aggregation.
