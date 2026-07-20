# Paper D — BAD-ANCHOR robustness of open recall (2026-07-03)

Recall-error ablation on the OPENQ harness (open free-recall: user names K=8 favourites, their real 64-d
(Q,resid) tokens are folded). With probability p each named anchor is CORRUPTED:
- **mid** (MID-LOVE): user names an item they rated 3/5 instead of a loved one (sampled from their mid-rated
  known-half items; if none, corruption skipped). Folded at loved strength POS (the system believes "favourite").
- **midresid**: same mid item but folded at its TRUE residual (self-correcting bound — isolates wrong-ITEM from
  wrong-STRENGTH).
- **wrong** (WRONG-TITLE): grounding resolves to a popularity-weighted random catalog item NOT in their profile,
  folded at POS (a fully wrong 64-d token).

Canonical ruler: NDCG@10 full/tail, te[300:] (304 users), q8, seed-avg {1,2,3,7,11}. Refs: D1 scalar-probe
0.378/0.178 ; half-fold ceiling ~0.41. Baselines reproduce canonical (popweight 0.386/0.172 ~= 0.387/0.170).

## popweight (REALISTIC popularity-biased recall)
| error | p | FULL | TAIL | dFULL | dTAIL |
|---|---|---|---|---|---|
| none | 0.0 | 0.3860±0.0040 | 0.1715±0.0067 | — | — |
| mid | 0.1 | 0.3906±0.0079 | 0.1702±0.0065 | +0.005 | −0.001 |
| mid | 0.2 | 0.3906±0.0055 | 0.1702±0.0080 | +0.005 | −0.001 |
| mid | 0.4 | 0.3888±0.0044 | 0.1593±0.0066 | +0.003 | −0.012 |
| midresid | 0.1 | 0.3873±0.0046 | 0.1657±0.0084 | +0.001 | −0.006 |
| midresid | 0.2 | 0.3874±0.0031 | 0.1706±0.0038 | +0.001 | −0.001 |
| midresid | 0.4 | 0.3811±0.0082 | 0.1574±0.0059 | −0.005 | −0.014 |
| wrong | 0.1 | 0.3752±0.0063 | 0.1596±0.0064 | −0.011 | −0.012 |
| wrong | 0.2 | 0.3621±0.0047 | 0.1508±0.0110 | −0.024 | −0.021 |
| wrong | 0.4 | 0.3296±0.0058 | 0.1181±0.0063 | −0.056 | −0.053 |

## distinct (hidden-gem framing, tail-info upper band)
| error | p | FULL | TAIL | dFULL | dTAIL |
|---|---|---|---|---|---|
| none | 0.0 | 0.4065±0.0033 | 0.2115±0.0036 | — | — |
| mid | 0.1 | 0.4064±0.0015 | 0.2107±0.0038 | −0.000 | −0.001 |
| mid | 0.2 | 0.4060±0.0047 | 0.2071±0.0065 | −0.001 | −0.004 |
| mid | 0.4 | 0.4073±0.0026 | 0.1993±0.0064 | +0.001 | −0.012 |
| midresid | 0.1 | 0.4054±0.0012 | 0.2107±0.0044 | −0.001 | −0.001 |
| midresid | 0.2 | 0.4060±0.0061 | 0.2104±0.0057 | −0.001 | −0.001 |
| midresid | 0.4 | 0.4018±0.0037 | 0.2022±0.0057 | −0.005 | −0.009 |
| wrong | 0.1 | 0.3995±0.0008 | 0.2026±0.0025 | −0.007 | −0.009 |
| wrong | 0.2 | 0.3927±0.0024 | 0.1964±0.0040 | −0.014 | −0.015 |
| wrong | 0.4 | 0.3666±0.0077 | 0.1696±0.0066 | −0.040 | −0.042 |

## FINDINGS
1. **Open recall is ROBUST, not fragile.** No cliff anywhere: degradation is smooth and ~linear in p. The
   feared "one bad anchor = one whole corrupted 64-d token" does NOT materialize as catastrophe because the
   set-encoder fold AVERAGES 8 tokens — a bad anchor is diluted to ~1/8 of the belief, exactly the mechanism
   that makes a bad scalar answer (1 dim of D1's probes) survivable too.
2. **MID-LOVE errors are essentially FREE on FULL** (within noise, popweight even +0.005: a mid-rated item is
   still an item from the user's taste region — a weaker but not wrong anchor). Only the TAIL pays, and only
   at heavy corruption (p=0.4: −0.012). midresid ~= mid ⇒ the wrong fold STRENGTH is not the problem either;
   the anchor's direction carries the information.
3. **WRONG-TITLE (grounding failure) is the real hazard**: ~ −0.014 FULL / −0.013 TAIL per +10% error rate
   (popweight; distinct ~ −0.010/−0.011 per 10%). At p=0.4 open recall collapses well below D1
   (0.330/0.118 vs 0.378/0.178).
4. **Crossing vs the D1 scalar-probe reference**: realistic popweight stays ~level with D1 FULL up to
   p≈0.1 wrong-title (0.375 vs 0.378) and loses beyond; its tail is below D1 from the start. The gem/distinct
   framing keeps open recall ABOVE D1 on BOTH axes through p=0.2 wrong-title (0.393/0.196) and only falls to
   ~D1 at p=0.4 (0.367/0.170). So the Paper-D headline survives realistic recall-error rates (~10–20%)
   provided grounding is decent; entity-resolution quality, not user memory slippage, is the deployment risk.

## Honest caveats
- Corrupted anchors fold at POS (system's belief "favourite"); genuine anchors fold at their true resid
  (canonical OPENQ convention). midresid bounds the strength-convention effect (small).
- 'wrong' is popularity-weighted over the whole catalog = a plausible mis-grounding model; adversarial or
  near-title confusions (similar-titled films are often geometrically close) would likely be MILDER.
- mid corruption skipped when the user has no mid-rated known-half item (rare; corrupted counts ~= p·N).

## Code / repro
BADANCHOR sweep inside the OPENQ block, continuous_actor.py.
`NOBC=1 EP=0 CONTMODE=cont OPENQ=1 HEUR=popweight OPENK=8 BADANCHOR=mid,midresid,wrong BADP=0.1,0.2,0.4
EVALSEEDS=1,2,3,7,11 python scripts/paper2/continuous_actor.py` (and HEUR=distinct).
