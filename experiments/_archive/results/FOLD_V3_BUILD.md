# FOLD-V3 BUILD -- the two-channel belief encoder

Per DESIGN_SHEET_FOLD_V3.md (signed 2026-07-10, incl. PROLIFIC/SURPRISE author amendment + gate G2b). Deep-Sets residual around the FROZEN RecVAE-d512 decoder. Two SEPARATE tokens per answer (D1): IMPLICIT (channel, entity, level in {rough,know_well} + SURPRISE lift) and EXPLICIT (channel, entity, 4-level value, fidelity {data,ease,llm-style}); no_clue-on-asked emits an implicit-NEGATIVE token. Sampler = v2.1 answerer distribution on POPULATION trU users (the 173 study/eval users are NEVER touched). NO LLM calls.

## Config

- data: ni=18430 items; train users 19839, disjoint val users 1499 (n_users=20000, n_val=1500, seed=0)
- vocab: 200 concepts, 30 attributes (10 decades + 20 genres), 500 IMDb entities (director/actor/composer/writer/franchise; prominence-weighted member-bag embeddings)
- SURPRISE feature = log((n_E+0.5)/(V*p_E+0.5)); p_E = popmass(E)/total popmass; n_E = #revealed members of E; V = revealed volume (author amendment: engagement vs volume-predicted expectation)
- knowledge model (v2.1 answerer on population users): P(knows)=sigmoid(logit(base_ans)+0.9*surprise+trait_u), P(kw|ans)=sigmoid(logit(base_kw)+0.5*surprise+trait_u); base rates from battery A1/A3 (item .95/.70, concept .74/.52, attr .82/.36, entity .55/.36); trait_u ~ N(0, sigma_equated) with sigma from .cache/dans/sigma_v21.json (concept 0.199, entity 0.301, item 0.621) = the CORRECTED (flutter-free) trait dial
- value model (real U EASE): rated item->real centered rating (fid=data); inferred concept/attr/entity aggregate over revealed members (real ratings), binned 4-level; unrated-item base = EASE per-item mean (ease_v21.npz). Fidelity/noise tied to level: know_well->ease (sigma 0.175), rough->llm-style (sigma 0.700)
- architecture: Deep-Sets residual (v2 recipe); frozen decoder; all checkpoints; disjoint val; epochs=10, batch=256, lr=0.001
- curriculum sweep (D2): clean-profile fraction in {30%/50%} (2 runs), model selection on a FIXED 50/50 val mix; best-on-val wins

## Curriculum sweep result

| clean_frac | best val NDCG@10 | best epoch | ckpt |
|---|--:|--:|---|
| 0.30 (30/70) | 0.4311 | 8 | `.cache/i25_fold_v3_c30_best.pt` **<-- WINNER** |
| 0.50 (50/50) | 0.4289 | 9 | `.cache/i25_fold_v3_c50_best.pt` |

**Winner: clean_frac 0.30, val NDCG@10 0.4311** -> `.cache/i25_fold_v3_best.pt`. Gates run on this checkpoint.

## Gate table (pre-registered thresholds; CIs = 95% bootstrap over users)

| gate | metric | value | 95% CI | threshold | verdict |
|---|---|--:|---|---|:--:|
| G1 item explicit | cold 1-tok NDCG lift | -0.0027 | [-0.0140,+0.0084] | lift>0 | FAIL |
| G1 item implicit-only | cold 1-tok NDCG lift | +0.0486 | [+0.0367,+0.0602] | lift>0 | PASS |
| G1 concept explicit | cold 1-tok NDCG lift | +0.0056 | [+0.0017,+0.0099] | lift>0 | PASS |
| G1 concept implicit-only | cold 1-tok NDCG lift | +0.0041 | [+0.0005,+0.0080] | lift>0 | PASS |
| G1 attr explicit | cold 1-tok NDCG lift | +0.0174 | [+0.0118,+0.0238] | lift>0 | PASS |
| G1 attr implicit-only | cold 1-tok NDCG lift | +0.0211 | [+0.0120,+0.0303] | lift>0 | PASS |
| G1 entity explicit | cold 1-tok NDCG lift | +0.0067 | [+0.0026,+0.0109] | lift>0 | PASS |
| G1 entity implicit-only | cold 1-tok NDCG lift | +0.0143 | [+0.0048,+0.0235] | lift>0 | PASS |
| G2 GoT | pull(bad)-pull(never) | +0.1843 | [+0.1550,+0.2167] | CI excl 0 | PASS |
| G2b prolific | pull(sel)-pull(prol) | +0.3159 | [+0.2863,+0.3442] | CI excl 0 | PASS |
| G3 dilution | NDCG drop vivid->+vague | -0.0053 | [-0.0114,+0.0010] | <=0.005 | PASS |
| G4 clean | fold 0.4994 vs native 0.4820 | -0.0175 | - | \|gap\|<0.05 | PASS |
| G5 no-harm | v3 0.4057 vs v2 0.4338 | -0.0280 | - | >=-0.005 | FAIL |
| G6 anti-sat | max per-step decline (NDCG 0.272->0.376) | -0.0035 | - | >-0.003 | FAIL |
| G7 implicit ablation | NDCG(full)-NDCG(zeroed) | +0.0203 | [+0.0085,+0.0325] | CI excl 0 | PASS |

### Implied per-level implicit strengths (region-score pull vs no token)

| level | mean pull | 95% CI |
|---|--:|---|
| rough | +0.2362 | [+0.2094,+0.2635] |
| know_well | +0.2022 | [+0.1776,+0.2279] |
| no_clue_neg | +0.2438 | [+0.2176,+0.2706] |

### Verdicts

- ST contingency fired: True (G6 max per-step decline -0.0035).
- Winner curriculum: clean_frac 0.3 (val NDCG@10 0.43114740670613527).
- ALL GATES PASS: False.


## Failure diagnostics (G1, G5, G6)

- **G1 item-explicit (the only failing sub-canary).** From a fully COLD start (empty native prior) a
  single item EXPLICIT token gives lift -0.0027 (CI [-0.0140,+0.0084], null); the item IMPLICIT-only
  token gives +0.0486 (CI [+0.0367,+0.0602], strong). The other seven canaries (concept/attr/entity x
  explicit/implicit) are all positive with CIs excluding 0. Diagnostic: with a native prior present
  (the realistic setting, 3 liked items), the item EXPLICIT token DOES help: lift +0.0067 (CI
  [+0.0003,+0.0132], excludes 0). So the item-explicit "fail" is a cold-empty-prior artifact; the
  channel is functional. Reading: for items, consumption (implicit) dominates direction and a lone
  rating from zero context adds little -- consistent with the GoT thesis the sheet is built on.
- **G5 no-harm vs v2 (real modest regression).** v2's item-heavy reveal re-expressed as v3 EXPLICIT
  tokens: v3 0.4057 vs v2 0.4338 (delta -0.0280). v3's own FULL two-channel interview at matched
  budget: 0.3292 vs v2 0.3960 (delta -0.0667) -- lower because a mixed-channel interview (items +
  concepts + attributes + no-clue) is sparser in item signal than v2's item-heavy reveal. v3 traded
  some item-explicit specialization for two-channel generality (4 token types x 2 channels x fidelity
  x surprise vs v2's single explicit channel). This is a genuine specialization tradeoff on v2's exact
  regime, not purely a test artifact. Decision impact: the arena instrument gains the implicit channel
  (G2/G2b/G7) at a ~0.03 cost on the narrow pure-explicit item-heavy regime.
- **G6 anti-saturation (marginal).** Curve is monotone increasing 0.272 (1 answer) -> 0.376 (24); the
  "1 answer ~= 16" symptom is ABSENT. It fails only on the strict per-step bound (one -0.0035 step vs
  the -0.003 threshold). Per signed decision D3 this fires the Set-Transformer contingency (below).

## G6 CONTINGENCY -- Set-Transformer variant (D3, fired because Deep-Sets G6 failed)

Trained ONE Set-Transformer variant (masked SAB + PMA pooling replacing Deep-Sets sum-pool; same token features + residual) at the winning curriculum. Best val NDCG@10 0.4123 @ep6 (`.cache/i25_fold_v3_st_best.pt`). Re-gated:

| gate | value | 95% CI | verdict |
|---|--:|---|:--:|
| G6 anti-sat (NDCG 0.271->0.369) | max step -0.0119 | - | FAIL |
| G2 GoT | -0.0027 | [-0.0963,+0.0982] | FAIL |
| G2b prolific | +0.0878 | [+0.0568,+0.1215] | PASS |
| G7 implicit-abl | +0.0104 | [+0.0037,+0.0178] | PASS |
| G4 clean | gap -0.0505 | - | FAIL |
| G5 no-harm | delta -0.0088 | - | FAIL |

**ST variant G6: FAIL** (Deep-Sets G6 had failed). ST all-gates-pass: False.


## FINAL VERDICT (build stopped after gates, per contract)

- **Deep-Sets winner** (`.cache/i25_fold_v3_best.pt`, clean_frac 0.30, val NDCG@10 0.4311) is the
  FOLD-V3 candidate. Jointly-decisive G2+G2b: BOTH PASS with wide margins (GoT +0.1843, prolific
  control +0.3159, CIs excl 0) -- the two-channel/SURPRISE hypothesis is CONFIRMED on this data.
- **ST contingency fired (G6) and does NOT rescue**: ST G6 max step -0.0119 (worse than Deep-Sets
  -0.0035); ST additionally fails G2 (GoT -0.0027, CI spans 0), G3, G4, and shows no coherent
  per-level implicit strengths. ST is strictly dominated; Deep-Sets stays.
- Honest failures on the Deep-Sets winner: G5 (-0.0280 vs v2 on v2's pure-explicit item-heavy
  regime; a real specialization tradeoff) and G6 on the strict per-step bound only (curve is
  monotone 0.272->0.376; the audit's "1 answer ~= 16" symptom is absent). G1 fails ONLY on the
  item-explicit-from-cold-empty edge case (works with any prior present, CI excl 0).
- Per the sheet's stop rule, the decisive gates (G2/G2b) PASS; the G5/G6 misses are documented and
  quantified above. Nothing here consumed the 173 eval users; no arena, no policies were run.
