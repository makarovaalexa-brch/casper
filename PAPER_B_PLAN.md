# Paper B — Reconstruction-based elicitation (encoder/decoder framing)

## Why we pivoted here (the diagnosis)
On dense full-catalogue NDCG@10, popularity is near-ceiling and every realizable elicitation method ties random/HELF
(6 convergent lines). That was the WRONG LENS: full-profile fold-in barely moves full-cat NDCG@10 (+0.035) but clearly
improves RMSE (−0.07) — personalization is real, just hidden because the top-10 is blockbuster-saturated. On the
**long-tail task** (Cremonesi/Koren/Turrin RecSys2010; exclude the popular head from candidates) elicitation DOUBLES
NDCG (0.039→0.093 realizable) and the oracle is huge (+0.27, robust across head cutoffs 20/33/50% mass). IPS caveat:
reweighting the full-catalogue metric alone is flat — value appears only when ranking IN tail space. So the regime is
the **long-tail recommendation task**.

## The core idea (user's insight; grounded in EDDI R3 + user's IEEE referential-dialogue work)
The current elicitation reward = held-out NDCG: a FEW random targets → sparse, high-variance, top-10-saturated →
"signal washes out" (RL playlist-collapses; flat). DECOUPLE elicitation from prediction and make the signal DENSE:
- **Elicitation = active reconstruction.** Ask the questions that best RECONSTRUCT the user's WHOLE preference vector
  (dense target over all items), i.e. maximize information-gain about the user — not a sparse held-out sample.
- **Prediction = decoder.** A (fixed or learned) decoder maps the user representation to item rankings.
This is EDDI (Partial-VAE + info-reward acquisition, replicated in R3) and the user's image-via-dialogue framing
(profile = "image"; questions = turns; goal = reconstruct in fewest turns).

## Architecture (Phase A→C)
- **A. Learned sequential reconstruction encoder** (replaces ridge fold-in). Ingest revealed (item-factor, rating)
  tokens via attention/DeepSets → user vector u. Train with MASKED, SHUFFLED reveals (variable count 1..K), predicting
  the user's FULL preference vector at every step (masked-autoencoder / Partial-VAE recipe). This fixes the
  full-profile→partial-profile distribution shift (user's RNN-attention point) and learns item+attribute weights (Q1).
  **INVERSE-POPULARITY (IPS) weighting** on the reconstruction loss so "reconstruct the user" = get their NICHE/tail
  tastes right (else the loss is dominated by blockbusters → re-learns popularity).
- **B. Elicitation policy from reconstruction info-gain** (EIG over the encoder's predicted profile) or a small policy
  trained on the DENSE reconstruction reward. NO held-out-NDCG reward (that is the decoupling).
- **C. Evaluate on the Cremonesi long-tail task** vs ridge+HELF baselines and the oracle (+0.27 headroom).

## HOLES / risks (watch these)
1. Reconstruction is still prediction — but DENSE (all items) not sparse (held-out), and the policy optimizes
   info-gain-about-user, not a noisy target. That density is the win.
2. Reconstruction target is popularity-biased → MUST IPS/tail-weight the loss, else it re-learns popularity.
3. Reconstruction ≠ top-N: must CHECK it translates to tail-NDCG (Phase C gate).
4. Encoder vs ridge was tried before (instrument_u) and ridge won — but on DENSE; revisit on TAIL with recon objective.

## GATES (make-or-break; if A fails, reconstruction makes zero sense)
Decoder fixed = β·popb + Q·u (frozen Q_svd) so we isolate the ENCODER. Compare LEARNED-ENCODER vs RIDGE-FOLDIN vs
POPULARITY, same reveals, on the TAIL objective (Cremonesi head-33%). Metrics: tail-NDCG@10, tail-Recall@50, recon AUC.
- **GATE A1 (beats ridge):** learned encoder > ridge fold-in on tail-NDCG@10 at matched reveals (q=4 and q=8), and > popularity floor.
- **GATE A2 (low-reveal robustness):** advantage holds at FEW reveals (q=1,2) — the deployment condition (shuffled/masked training).
- **GATE A3 (monotone / no-harm):** tail-NDCG non-decreasing in #reveals (revealing never hurts).
- **GATE A4 (recon ≈ ranking):** better reconstruction (held-out recon AUC) correlates with better tail-NDCG (so recon is the right proxy).
IF A1 fails (encoder can't beat ridge at reconstructing the tail profile) → the reconstruction-elicitation premise is
dead; STOP and reconsider. Only if A1–A4 pass do we proceed to B (policy) and C (long-tail eval vs oracle).

Scripts: scripts/paper2/encoder_recon.py (Phase A). Frozen instrument factors Q_svd/bi_svd. Ledger: RESULTS.md PART F.
