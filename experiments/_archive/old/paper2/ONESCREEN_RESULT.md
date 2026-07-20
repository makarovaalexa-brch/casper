# Paper E — ONE-SCREEN k=3 deployability (closes review item E-B2) (2026-07-03)

Can one SCREEN (the k=3 signed-phrase OMP decomposition of a D1 continuous query, answered as 3 sliders)
count as ONE turn without giving up the continuous win? Machinery: OMP over the rich 6724-phrase bank
(phrasebank.npz); D1 headline actor policy_phase3_d1divw_last.pt. CANONICAL ruler this time (the earlier
OMP-5 −0.008 was n=150, single run): NDCG@10 full/tail, te[300:] (304 users), q8, seed-avg {1,2,3,7,11}.
Reference (a) reproduces canonical D1 EXACTLY: 0.3780/0.1782.

| variant (q8) | FULL | TAIL | cost vs (a) |
|---|---|---|---|
| (a) CONT: fold true q (unsnapped D1) | 0.3780±0.0032 | 0.1782±0.0065 | — |
| (b) OMP-3 → ONE composite token, a_hat=Σ w_k(u*·e_k) | 0.3600±0.0037 | 0.1582±0.0055 | **−0.018 / −0.020** |
| (e) OMP-5 → ONE composite token (same formula) | 0.3681±0.0027 | 0.1665±0.0053 | **−0.010 / −0.012** |
| (c) 3 raw (e_k, u*·e_k) tokens/screen, 8 screens (24 toks) | 0.3051±0.0037 | 0.1015±0.0020 | −0.073 / −0.077 |
| (d) = (c) at 8-TURN budget (2 screens + 2 phrases) | 0.2962±0.0046 | 0.1053±0.0026 | −0.082 / −0.073 |

(d) full curve, 3 turns/screen (FULL/TAIL): s=1 0.3025/0.1029, s=2 0.2953/0.1048, s=3 0.2950/0.1043,
s=4 0.2951/0.1007, s=5 0.2977/0.0997, s=6 0.3013/0.0996, s=7 0.3032/0.1013, s=8 0.3051/0.1015 — FLAT.

Blend fidelity cos(q, q_hat): OMP-3 mean 0.827 (p10 0.720), OMP-5 mean 0.891 (p10 0.822)
(matches TRIANGULATE: 0.842 / 0.896 on unperturbed rollouts).

## FINDINGS
1. **One screen = one turn WORKS, via the DECOMPOSITION FORMULA (b)**: show the 3 phrases as sliders, score
   the composite a_hat = Σ_k w_k·(u*·e_k), fold the OMP reconstruction as ONE token. Cost −0.018/−0.020 vs
   the unsnapped continuous policy — vs −0.037/−0.040 for the single-phrase snap (Paper C snap-loss), i.e.
   the k=3 screen HALVES the naming cost while staying fully verbalized. A 5-slider screen (e) nearly
   halves it again: −0.010/−0.012.
2. **The OMP weights are LOAD-BEARING: naive multi-token folding (c) is catastrophic** (−0.073/−0.077, worse
   than the k=1 snap). The frozen set-encoder weighs tokens ~equally, so folding the 3 phrases as separate
   unweighted tokens scrambles the belief (24 on-manifold phrase tokens ≠ 8 off-manifold queries). The
   composition must happen OUTSIDE the encoder (weights → one reconstructed token), which is exactly what
   the formula in (b) does — and it needs no training.
3. **Budget accounting of (c)/(d) is moot**: the curve is flat ~0.30 from screen 1 to screen 8 — the failure
   is the fold FORMAT, not the turn budget; even 24 tokens never recovers. Do not deploy per-phrase folds.
4. **Continuity with the earlier claim**: earlier single-run n=150 OMP-5 fold was −0.008/−0.007; canonical
   5-seed OMP-5 composite = −0.010/−0.012. The "nearly lossless at k=5" claim survives the canonical ruler
   (small extra shrinkage here because a_hat = u*·recon scales by ||recon||≤1).

## Honest caveats
- a_hat is computed from the same geometric answers u*·e_k a real user would give per slider; no privileged
  information (|a_hat| ≤ ||recon|| ≤ 1 by construction, no clipping needed).
- The policy emits q from the belief built on FOLDED (reconstructed) tokens — i.e. (b)/(e) are honest
  closed-loop rollouts, not post-hoc re-scoring.
- The screen is still simulated geometrically; whether a human answers 3 phrase-sliders as u*·e_k is the
  human-study question (same status as all Paper-E folds; SBERT round-trip is banned per the retraction).

## Code / repro
ONESCREEN block in continuous_actor.py (uses the TRIANGULATE OMP + phrasebank.npz).
`NOBC=1 EP=0 CONTMODE=cont ONESCREEN=1 EVALSEEDS=1,2,3,7,11 python scripts/paper2/continuous_actor.py` (~4 min).
