# FOLD FIX PROPOSAL — un-cap the belief, keep the rich data (2026-07-11)
For adversarial review BEFORE implementation. Context: EMBEDDING_FOLDING_INDEX.md (lineage / what
worked / NDCG / leaks), the Fable bug-scan (BUGS 1-6), the working reference scripts/paper2/encoder_recon.py,
the broken current code scripts/i25_fold_leakfree.py.

## The problem (one line)
The leak-free fold's belief is FLAT after the first answer: cold 0.165 → turn-1 0.241 → … → full
profile 0.241 (native RecVAE ~0.487). It does not accumulate evidence.

## Root cause (from the bug-scan, confirmed against working models)
`z = prior + w·ρ(pool)` where (BUG 1) `pool` is a softmax convex combination (bounded, can't grow
with N) and `w` is a sigmoid of COUNT-FREE means; (BUG 2) the `agree` term inside `w` is MAXIMAL at
one token and only decreases with more evidence → confidence peaks at turn 1. So the reachable belief
is {prior + bounded·ρ(convex hull)} — nothing in it grows with evidence. NOT the softmax alone:
encoder_recon.py uses softmax pooling and accumulated (feeds the pooled vector straight to a linear
decoder, no prior/gate wrapper); v3.1 used a SUM pool and accumulated. The saturation is the
fixed-prior + scalar-gate WRAPPER (this-session invention), applied to tokens stripped of count.

## The changes (KEEP the good decisions; fix the one broken piece)

### CHANGE 1 — remove the scalar gate + agree/agg_conf machinery (fixes BUG 1, BUG 2)
Reasoning: the gate is the cap and it anti-accumulates. June had no gate; the pooled vector was the
belief. We keep a residual-on-prior (for a clean cold intercept) but drop `w`, `agree`, `agg_conf`,
`wnet`. rho's last layer stays zero-initialised so empty pool → delta 0 → z = prior (intercept holds).

CURRENT (i25_fold_leakfree.py:204-231, paraphrased):
```python
alpha = softmax(att_logit, dim=1)                 # (B,T,Kh) sum to 1
pool  = einsum("btk,btd->bkd", alpha, h) * has    # convex combination, bounded
agg_conf = (alpha * conf).sum(1)                   # count-free mean
agree    = pool.norm(dim=-1) / hnorm_mean          # =1.0 at one token, decreases after
w        = sigmoid(self.wnet([agg_conf, agree]))   # scalar, count-free, bounded
delta    = self.rho(pool)                          # bounded
z        = prior_z + w * delta                     # ONE bounded step, forever
```
PROPOSED:
```python
att   = self.att(h).masked_fill(~mask.bool().unsqueeze(-1), -1e9)  # attention CHOOSES tokens
alpha = torch.softmax(att, dim=1)                                  # (B,T,1)
pool  = (alpha * h).sum(1)                                         # (B,d) attention-pooled
delta = self.rho(pool)                                            # rho[-1] zero-init -> rho(0)=0
z     = prior_z + delta                                           # NO scalar gate
return z
```

### CHANGE 2 — signed/unsigned is ATTENTION'S job (concede BUG 3; do NOT hand-fix)
Reasoning (author): with the belief capped, attention never had a reason to learn which tokens
matter. Once the cap is gone, attention should down-weight the value=0 "I-know-it-no-opinion" tokens.
We do NOT bolt on a manual signed/unsigned rule. We VERIFY after training whether dilution remains;
only if softmax provably can't concentrate over hundreds of tokens do we address that specifically.

### CHANGE 3 — the gates that actually catch this (author-directed)
Reasoning: BUG 4 — the old sanity gates pass a FLAT model (±0.003 band on "no-Q1-drop"). Two fixes:
- **G-clean = hard stop.** Full-profile fold must be within a SMALL degradation of RecVAE's own
  full-profile NDCG (native). Small is acceptable, large is not. Its FAILURE stops the run (it was
  failing at −0.167 / −0.25 and I wrongly pressed on).
- **G-monotone-REQUIRES-INCREASE.** Not "doesn't drop" — each added answer (or the turn-1→turn-8
  span) must move NDCG UP by a real margin. A saturating curve FAILS immediately.
"Full profile beats one answer" is scrapped (ridiculous / trivial).

## KEEP (unchanged — the good decisions we agreed)
Leak-free tokens (drop profile-`surprise`); one unified `tokens_for` (only the question set differs
by regime); NO caps (all answerable tokens); public p_E + entropy features; G-no-profile-leak
(passed 0.0); G-intercept; the tight-cluster diagnostic; ALL the rich multi-channel LLM-distilled
tokens (items/concepts/attributes/entities, values, knowledge levels). We are NOT reverting to
June's thin items-only encoder — only borrowing its accumulating aggregation.

## OPEN RISKS for the reviewer to attack (I am not sure of these)
1. **Does a softmax-MEAN pool accumulate ENOUGH?** June accumulated but its full-profile fold was
   only ~0.334; v3.1 hit 0.50 using a SUM pool (grows with count). If mean-direction-cleaning is
   insufficient to approach native ~0.487, we may need a count/magnitude pathway — but a raw SUM
   reintroduces the cardinality confound (duplicate test). Is there a duplicate-invariant way to let
   magnitude grow with DISTINCT evidence (e.g., scale by count of DISTINCT high-attention tokens)?
2. **Does removing `w` actually restore accumulation, or is `ρ(bounded pool)` still bounded?**
   z = prior + ρ(mean) — ρ of a bounded input is bounded. Does direction-cleaning alone clear the
   G-clean bar, or is this fix cosmetic?
3. **Attention concentration:** softmax over hundreds of full-profile tokens — can it really put
   enough mass on the ~dozens of signed ones, or does BUG 3 dilution survive Change 1?
4. Anything in the token features / masking / training loss that would still bound accumulation.
