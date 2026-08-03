# Concept vs item at matched budget, across four static question banks

> Run 2026-08-03. `src/instrument/bank_sweep.py --ckpt .cache/instrument/cv3_ep11.pt`
> Tower: `models/i26c_ep31/t2i26c_ep31.pt` (certified, frozen). Concept net: cv3 ep11.
> Raw: `experiments/instrument/bank_sweep.json`.

## Protocol

1,500 held-out users, a third of each user's likes hidden as targets, **zero items of history**, m ∈
{0,1,2,4,8,16} questions from a user-independent ranked bank. NDCG@10, full and tail (tail = head items
deleted from the candidate list, canonical). An unanswerable question **burns the budget** in both arms —
concepts arrive as a refusal, items as an exposure token (`UNSEEN_LEVEL`). Answered items are masked from
the candidate list; the concept arm has no items to mask.

**Arms-matched canary (asserted in code, passes for every bank):** at m=0 both arms fold nothing and score
bit-identically, **0.1277 / 0.0267**.

## The four banks

Each is a different answer to "which question is worth asking?", applied identically to both channels.

| bank | criterion | what it turns out to be |
|---|---|---|
| `pop` | most answerable | play count / member mass |
| `entropy` | H over {like, meh, dislike, **unanswerable**} | **DEGENERATE — a popularity criterion in disguise** |
| `polar` | H(bands \| answered) | pure discrimination, ignores who can answer |
| `expinfo` | P(answered) × H(bands \| answered) | the defensible middle |

**`entropy` is degenerate and must not be reported as a discrimination criterion.** On a sparse catalogue
almost every item's 4-way distribution is ~100% "unseen", so only ultra-popular items score at all.
Measured: its top-16 sits at the **99.9th popularity percentile**, overlaps `items-pop` **12/16**, and burns
at *exactly* the same rate (59.8%). An item cannot be both obscure and high-entropy under this definition.

## Results (concept net cv3 ep11), full / tail

| bank | m | concept | item | tail Δ |
|---|---|---|---|---|
| pop | 1 | .1281/.0308 | **.1431**/.0335 | −.0026 |
| | 2 | .1307/**.0428** | **.1553**/.0420 | +.0008 |
| | 4 | .1368/**.0513** | **.1673**/.0477 | +.0036 |
| | 8 | .1415/**.0617** | **.1930**/.0587 | +.0031 |
| | 16 | .1441/.0697 | **.2211**/.0697 | −.0000 |
| entropy | 1 | .1318/**.0402** | **.1368**/.0334 | +.0069 |
| | 2 | .1354/**.0461** | **.1427**/.0388 | +.0073 |
| | 4 | .1367/**.0487** | **.1522**/.0480 | +.0007 |
| | 8 | .1379/.0514 | **.1700**/**.0599** | −.0085 |
| | 16 | .1428/.0629 | **.2133**/**.0748** | −.0119 |
| **polar** | 1 | **.1308**/**.0402** | .1281/.0302 | +.0100 |
| | 2 | **.1331**/**.0437** | .1282/.0313 | +.0124 |
| | 4 | **.1356**/**.0495** | .1312/.0329 | +.0166 |
| | 8 | **.1411**/**.0619** | .1303/.0346 | +.0273 |
| | 16 | **.1451**/**.0704** | .1352/.0409 | +.0295 |
| expinfo | 1 | .1308/**.0402** | **.1333**/.0324 | +.0079 |
| | 2 | .1331/**.0437** | **.1427**/.0388 | +.0049 |
| | 4 | .1356/**.0495** | **.1522**/.0480 | +.0015 |
| | 8 | .1411/.0619 | **.1693**/**.0678** | −.0059 |
| | 16 | .1451/.0704 | **.1973**/**.0760** | −.0056 |

## THE FINDING — concepts decouple discrimination from answerability, items cannot

Answer rates on the eval users, top-16 of each bank:

| bank | item arm | concept arm |
|---|---|---|
| `polar` (pure discrimination) | **4.8%** | **97.2%** |
| `expinfo` | 37.5% | 97.2% |
| `pop` | 40.2% | — |

A maximally divisive **item** is, by construction, one that few people have seen — asking it burns the
budget 95% of the time, and the item arm duly collapses to .1352/.0409 at m=16, barely above the
.1277/.0267 floor. A maximally divisive **concept** aggregates hundreds of films, so it stays answerable by
97% of users *while still splitting them*. This is structural, not a tuning artifact.

Consequence: **under `polar` the concept channel wins BOTH axes at every budget from m=2** — the only bank
where it beats items on full. Under every other bank items win full comfortably.

## Secondary findings

1. **Concepts win TAIL at short budgets under 3 of 4 banks** (`pop` m2–m8, `entropy` m1–m4, `expinfo`
   m1–m4), losing at long budgets. Consistent direction, and it revives the "concepts own the short
   interview" claim under a matched protocol. **Mechanism (Fable, plausible not tested):** high
   information *rate*, low information *ceiling* — one concept promotes a whole region at once, but the
   independent taste axes run out after ~4 answers, while items are lower-rate and compositional, so they
   keep compounding. High-slope/low-ceiling vs low-slope/high-ceiling curves must cross.
2. **The bank changes which CHANNEL wins.** That is the same structural claim Paper A makes about the
   recommender — a strategy comparison is meaningless without a stated, reproducible selection criterion.
3. **A static ranking decays with budget**: concept-`entropy` beats random draws on tail at m=1
   (.0402 vs .0336) but loses by m=16 (.0629 vs .0752) — top-ranked questions are individually
   discriminative but mutually redundant. Argues for greedy de-correlation of the bank *before* any
   adaptivity claim, so adaptivity is credited only for the gap above a de-correlated static bank.

## NOT YET ESTABLISHED — do not quote as significant

- **No paired bootstrap on any cell.** Per-user tail SE ≈ .003–.004 at n=1500, so the m=4 crossings
  (+.0007, +.0015) are noise; the `polar` deltas (+.0100 … +.0295) and the m=8/16 item wins are comfortably
  outside it, but nothing here has a CI yet. Given the 2026-07-30 unpaired-CI scar, only paired intervals
  may be quoted.
- The crossover *budget* is not identified — only "concepts ahead at short, items ahead at long".
- Mechanism claims in Secondary 1 are reasoning, not measurement.

## Provenance / traps

- Two arms must share a target RNG stream. They originally did not, and the m=0 rows disagreed
  (.1277 vs .1267) where they are required to be identical. The m=0 equality is now asserted per bank.
- The concept arm was originally scored on RANDOM draws while the item arm used a ranked bank — not
  comparable. Both are ranked now; `--bank random` is retained but is concept-arm-only and non-comparable.
- Tail must be scored on a head-deleted candidate list. Scoring it globally pins the no-evidence prior at
  exactly 0.0000 and produced the phantom "tail collapse" earlier the same day.
