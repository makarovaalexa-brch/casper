# INSTRUMENT SOLUTION — strong + dislike-expressing, developed from INSTRUMENT_REVIEW.md (2026-07-11)
For Fable adversarial review. Grounded in INSTRUMENT_REVIEW.md (code-cited diagnosis) + RAWDATA_DISLIKE.md
(dislike generalizes −0.61, hidden in raw co-consumption ~0) + the valence probe (z-space HAS a symmetric
valence direction) + the fold saga (every bolt-on sign channel went inert, ΔNDCG=0.000).

## THE DIAGNOSIS (settled)
Strength-vs-dislike tension is in the LIKELIHOOD, not the data:
- Strong top-N likelihoods (multinomial/implicit, RecVAE/EASE) are structurally NON-NEGATIVE → sign has no
  gradient path → INERT.
- Naive signed weight in a strong like-ranker HURTS (−33%, RAWDATA Q3) because disliked items sit INSIDE the
  like manifold (content-sim 0.79) → a negative weight drags liked neighbors.
- Dislike info is REAL and separable — but only in a TASTE-structured space (MF/content; −0.61), NOT in raw
  co-consumption (~0), which is where RecVAE lives.
CONCLUSION: dislike must be **REPULSION in a taste-structured latent**, and strength must come from an implicit
ranking backbone that is SIGN-FREE by itself. The two must live in ONE nonlinear continuous latent.

## REQUIREMENTS (the "open cell" from review §E)
R1 STRONG top-N (near EASE/RecVAE) via an implicit ranking backbone.
R2 TASTE-STRUCTURED latent: liked and disliked regions genuinely separated (not the consumption manifold where
   they overlap). Dislike ⇒ move the belief AWAY (repulsion/distance), never a negative in a nonneg likelihood.
R3 SIGN STAYS LIVE: the dislike channel must have a direct, strong gradient (contrastive/metric), not a weak
   bolt-on head (which went inert every time).
R4 NONLINEAR + CONTINUOUS LATENT + POSTERIOR (Paper C; adaptivity headroom).
R5 COLD-START FOLD-IN from a sparse signed interview (attention over (entity, value, confidence) tokens).
R6 DISLIKE HELPS RANKING as region-narrowing (eliminate wrong region ⇒ held-liked NDCG up), the mechanism the
   author named — NOT suppression (which NDCG@liked can't reward and RecVAE can't do).

## CANDIDATE ARCHITECTURES (for Fable to weigh / break)
**S1 — Signed Collaborative Metric Learning (metric latent; dislike = distance).**
User u and items in a nonlinear metric space; score = −‖u − v_i‖. Loss = 3-level metric ranking:
d(u, liked) < d(u, unrated) < d(u, disliked) (margins). Liked pulled close, disliked pushed FAR (beyond
unrated) ⇒ taste-structured BY CONSTRUCTION; dislike is literally repulsion (increases distance). Implicit
part (liked<unrated) carries top-N strength (CML is competitive). Fold-in = attention over signed tokens → u.
Posterior via probabilistic/ensemble metric. ✓R1(if CML strong) ✓R2 ✓R3(margins=direct sign gradient) ✓R4 ✓R5.
Risk: is metric-learning top-N-competitive with EASE at 25M scale? disliked-beyond-unrated margin may distort
the like ranking.

**S2 — Dual-objective latent VAE (strong multinomial + contrastive taste-structuring).**
Keep RecVAE's multinomial reconstruction (R1 strength, positives) BUT add an auxiliary CONTRASTIVE term that
REPELS disliked items from u in the latent (taste-structuring), trained jointly so the latent separates
like/dislike. Dislike enters via the contrastive term (repulsion in latent), never the multinomial. Sign stays
live because the contrastive term is a direct strong gradient (R3). ✓R1(multinomial) ✓R4 ✓R5.
Risk: do the multinomial (positive reconstruction) and the contrastive (repulsion) COHERE in one latent or
fight? Does the multinomial keep pulling disliked back into the like manifold, cancelling the repulsion?

**S3 — Taste-latent from rating-MF, strong nonlinear ranker on top.**
Build the ITEM latent from the TASTE signal where separation is revealed (MF on the graded/signed rating
matrix, or content), NOT co-consumption. Then a strong nonlinear fold-in + implicit ranking objective on that
taste latent. Dislike-repulsion works because the latent is taste-structured from birth. ✓R2 ✓R4.
Risk: taste/rating-MF is weaker for top-N (Liang'18); needs the implicit backbone bolted on — may reinherit
the biased-SVD weakness (this is closest to "just make biased-SVD nonlinear", which could still be weak).

**S4 — (control) implicit-strong + z-space repulsion at inference only.**
Keep frozen RecVAE for positives; apply dislike as a z-space repulsion at fold-in only (no retrain). This is
what the checks already tried (hard-wired signed z) — dislike-heavy HURT (−0.165) because RecVAE's latent
isn't taste-structured. INCLUDED ONLY as the baseline that proves R2 (taste-structured latent) is necessary.

## GATES (make-or-break, measured in a fast pilot)
G-strength [HARD]: full-profile NDCG near EASE/RecVAE (the bar biased-SVD & Paper A failed).
G-taste-separation [HARD]: in the learned latent, a user's disliked items are FAR from their liked items
  (measure the like/dislike margin; the +0.05 selectivity must become large). This is the R2 test.
G-dislike-helps [HARD, the author's mechanism]: in a cold-start interview, disliking a region RAISES held-liked
  NDCG (region-narrowing), and IG2 flip moves (like→high, dislike→low) WITHOUT dragging liked neighbors
  (specificity leg: untouched & liked items stay put).
G-sign-live: value/dislike-zeroing DROPS NDCG by ≥ MDE (not 0.000) — the anti-inert gate.
Plus: monotone accumulation, intercept, leak-free, no caps.

## OPEN QUESTIONS FOR FABLE
1. Which of S1/S2/S3 best delivers a TASTE-STRUCTURED + TOP-N-STRONG latent where dislike is repulsion? Rank them.
2. S2's core risk: can a multinomial-positive and a contrastive-repulsion share one latent, or does the
   positive objective keep collapsing disliked items back into the like manifold? Is there a clean separation
   (e.g. two sub-spaces: a positive-ranking subspace + an orthogonal taste/valence subspace)?
3. Is metric learning (S1) actually top-N-competitive with EASE/RecVAE at 25M, or is that a fatal strength gap
   (reinheriting Paper A's failure in a new coat)?
4. Is there a KNOWN architecture we're missing that already achieves strong top-N + signed/repulsion latent
   (e.g. CML variants, EASE with a signed Gram matrix, two-subspace VAEs, ordinal-VAE with a ranking head)?
5. The single most dangerous risk that makes this the 9th failed instrument.
