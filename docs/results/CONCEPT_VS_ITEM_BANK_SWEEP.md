# Concept vs item at matched budget, across four static question banks

> **Re-cut 2026-08-12 onto the POLICY-B artifacts.** Every headline number below now comes from
> `experiments/instrument/bank_sweep_policyB.json` (2026-08-05 15:34) and
> `experiments/instrument/concept_vs_item_paired_policyB.json` (2026-08-05 15:58), certified pair
> tower `models/i26c_ep31/t2i26c_ep31.pt` + concept `models/i26c_ep31/concept_cv3_ep08.pt`.
> Until this re-cut the document printed `experiments/instrument/bank_sweep.json`, whose numbers are on
> a **third, retired contract** — answered-asked-items masked, **no fold-in mask at all** — and stated
> that retired rule as current fact. Found by `docs/results/POLICY_A_PROVENANCE_AUDIT.md`; the data had
> been on disk unused since 2026-08-05 (`docs/results/MASK_POLICY_B_MIGRATION.md` §3).
>
> **What changed and what did not:** the load-bearing `polar` result is unchanged in direction and
> **roughly twice as large** — all ten cells still separated positive, full margins 1.7–2.5×. The
> *secondary* claim did **not** survive intact: on the full axis at short budgets the other three banks
> flip or lose significance (see "What the re-cut changed", below). Retired numbers are kept where they
> are the only measurement of something, each behind an explicit contract label — never deleted.

## Protocol

1,500 held-out users, a third of each user's likes hidden as targets, **zero items of history**, m ∈
{0,1,2,4,8,16} questions from a user-independent ranked bank. NDCG@10, full and tail (tail = head items
deleted from the candidate list, canonical). An unanswerable question **burns the budget** in both arms —
concepts arrive as a refusal, items as an exposure token (`UNSEEN_LEVEL`).

**Candidate pool — POLICY B** (2026-08-05 ruling,
`external_literature/findings/candidate_masking_protocol.md`): every rated fold-in-side item of the
user, likes **and** dislikes, targets excluded, is `-inf` — **identically for the item arm and the
concept arm at every budget**, and independent of what was asked.
`bank_sweep_policyB.json` records this as its `_pool_policy` field.

> ⛔ **The rule this document used to print is retired and was never a fair comparison.** It masked
> only the ANSWERED asked items and applied no fold-in mask at all, so the item arm collected a free
> lift — the ~4 items it had just revealed were deleted from its own candidate list — that the concept
> arm cannot receive by construction, because a concept question masks nothing. That is why every
> margin below grows when the pool is made a property of the protocol instead of the arm.

**Arms-matched canary (asserted in code, passes for every bank):** at m=0 both arms fold nothing and score
bit-identically, **0.2154 / 0.0354** (policy B). On the retired contract this floor read `.1277 / .0267`
— the single fastest way to tell which pool a number in this line of work is on.

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

## Results — POLICY B, certified concept net cv3 ep8, full / tail

Raw: `experiments/instrument/bank_sweep_policyB.json`. Floor (m=0, both arms, every bank)
**.2154/.0354**. Bold = the winning channel on that axis.

| bank | m | concept | item | full Δ | tail Δ |
|---|---|---|---|---|---|
| pop | 1 | .2121/.0410 | **.2163**/**.0471** | −.0042 | −.0061 |
| | 2 | **.2242**/**.0601** | .2234/.0593 | +.0007 | +.0008 |
| | 4 | **.2345**/.0672 | .2307/**.0717** | +.0039 | −.0046 |
| | 8 | .2465/.0827 | **.2595**/**.0896** | −.0130 | −.0069 |
| | 16 | .2624/.1021 | **.2931**/**.1054** | −.0307 | −.0033 |
| entropy | 1 | **.2309**/**.0609** | .2196/.0501 | +.0113 | +.0108 |
| | 2 | **.2380**/**.0693** | .2256/.0591 | +.0124 | +.0102 |
| | 4 | **.2434**/**.0760** | .2366/.0719 | +.0068 | +.0040 |
| | 8 | .2419/.0786 | **.2660**/**.0972** | −.0241 | −.0186 |
| | 16 | .2532/.0920 | **.3000**/**.1191** | −.0468 | −.0270 |
| **polar** | 1 | **.2240**/**.0548** | .2141/.0442 | +.0099 | +.0106 |
| | 2 | **.2305**/**.0631** | .2139/.0461 | +.0166 | +.0171 |
| | 4 | **.2358**/**.0709** | .2180/.0455 | +.0178 | +.0254 |
| | 8 | **.2562**/**.0910** | .2186/.0503 | +.0376 | +.0407 |
| | 16 | **.2632**/**.0974** | .2274/.0589 | +.0358 | +.0385 |
| expinfo | 1 | **.2240**/**.0548** | .2226/.0489 | +.0014 | +.0059 |
| | 2 | **.2305**/**.0631** | .2256/.0591 | +.0049 | +.0040 |
| | 4 | .2358/.0709 | **.2366**/**.0719** | −.0008 | −.0010 |
| | 8 | .2562/.0910 | **.2673**/**.1041** | −.0111 | −.0131 |
| | 16 | .2632/.0974 | **.2972**/**.1216** | −.0340 | −.0242 |

<details>
<summary>⚠ Superseded: the same table on the retired contract, concept net cv3 <b>ep11</b> (kept, not deleted)</summary>

Floor `.1277/.0267`. These are the only measurement of **ep11** we have, which is why they are kept —
but they are on the retired pool AND on a superseded concept checkpoint, so they may not be placed
beside anything above. **Their raw file no longer exists:** `experiments/instrument/bank_sweep.json`
was later overwritten by the cv3-ep8 run (checked 2026-08-12 — its `polar` concept m=1 reads
`.1327/.0412`, the ep8 value, not the `.1308/.0402` printed here; the item arm agrees because it does
not depend on the concept checkpoint).

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

</details>

## THE FINDING — concepts decouple discrimination from answerability, items cannot

Answer rates on the eval users, top-16 of each bank:

| bank | item arm | concept arm |
|---|---|---|
| `polar` (pure discrimination) | **4.8%** | **97.2%** |
| `expinfo` | 37.5% | 97.2% |
| `pop` | 40.2% | — |

A maximally divisive **item** is, by construction, one that few people have seen — asking it burns the
budget 95% of the time. Under policy B the `polar` item arm reaches .2274/.0589 at m=16, i.e. sixteen
questions buy it **+.0120 full / +.0235 tail** over the .2154/.0354 floor, while the concept arm reaches
.2632/.0974 — **+.0478 / +.0620** from the same sixteen questions. A maximally divisive **concept**
aggregates hundreds of films, so it stays answerable by 97% of users *while still splitting them*. This is
structural, not a tuning artifact, and it is a pool-invariant fact about the banks: the answer rates above
do not depend on the mask at all.

Consequence: **under `polar` the concept channel wins BOTH axes at every budget, m=1 through m=16** — the
bank where it beats items on full by the widest margin, and (see the CIs below) the only bank where it is
separated positive on both axes at every budget.

## What the re-cut changed (2026-08-12)

Checked cell by cell against both artifacts, not taken from the migration doc's summary.

1. **The load-bearing `polar` claim survives and roughly doubles.** All ten paired cells remain separated
   positive. Full Δ: `+.0045 → +.0099`, `+.0075 → +.0166`, `+.0103 → +.0178`, `+.0189 → +.0376`,
   `+.0146 → +.0358` (1.7×–2.5×). Tail Δ: `+.0151 → +.0171`, `+.0192 → +.0254`, `+.0310 → +.0407`,
   `+.0265 → +.0385` at m=2…16, and **essentially unchanged at m=1** (`+.0109 → +.0106`) — the one cell
   that does not grow, so "margins roughly double" is a statement about the full axis and about tail at
   m ≥ 2, not about every cell.
2. **The secondary claim does NOT survive as written.** *"Off that bank, items win full decisively"* was
   true on the retired contract and is false under policy B at short budgets. On the FULL axis:
   `entropy` m=1 flips from **separated −.0040 to separated +.0113**, and m=2 from **separated −.0081 to
   separated +.0124**; `entropy` m=4 goes from separated −.0155 to **ns** +.0068; `pop` m=2 and m=4 go
   from separated −.0232 / −.0299 to **ns**; `expinfo` m=2 and m=4 go from separated −.0070 / −.0107 to
   **ns**. Six cells change verdict, all in the concept channel's favour. The item channel's full-axis
   dominance is now confined to **m ≥ 8** (plus `pop` at m=1).
3. **Two tail cells move the other way**, which is why this was checked rather than assumed: `expinfo`
   tail m=1 and m=2 go from separated positive (`+.0088`, `+.0075`) to **ns** (`+.0059`, `+.0040`), and
   `expinfo` tail m=8/m=16 go from ns/marginal to **separated negative** (`−.0131`, `−.0242`).
4. So the audit's summary — *"the direction survives and the printed numbers understate the result"* — is
   **correct for the headline and incomplete for the rest**: the retired contract did not merely shrink
   the concept channel's margin, it reversed the sign of the full-axis verdict on three banks at short
   budgets. The defensible statement of the whole result is now: **concepts win the short interview on
   both axes under any discriminative bank, and items take over on full from m=8.**

## Secondary findings

> ⚠ Findings 1 and 3 below are stated on the RETIRED contract and are corrected by the CI section
> further down; finding 2 is a statement about the banks and is unaffected by the pool.

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

---

# Structural diagnostics (2026-08-03, `src/instrument/concept_diagnostics.py`, cv3 ep11)

Three free measurements (no training) proposed by Fable to separate "weak fold" from "thin channel".
**All three suspects CLEARED.**

| diagnostic | result | verdict |
|---|---|---|
| D1 norm-cap binding | clipped 18.9% (m1) → 50.4% (m4) → **83.2% (m16)**; ‖U‖ mean 7.15 vs cap 3.61 | binds hard — but see below |
| D2 gate self-throttling | per-step ‖g·delta‖ 1.607 → 0.721, decay **2.23×** (threshold 3×) | recursion OK, not throttling |
| D3 oracle-z decoder ceiling | **0.9844 full / 0.9831 tail** | frozen decoder EXONERATED |

**D1 follow-up — the cap is load-bearing, not a ceiling.** Raising it at test time (no retrain):

| cap | m=8 full/tail | m=16 full/tail |
|---|---|---|
| 3.61 (trained) | **.1411**/.0619 | **.1451**/.0704 |
| 7.22 (×2) | .1276/.0607 | .1303/**.0716** |
| 14.44 (×4) | .1225/.0592 | .1196/.0695 |
| uncapped | .1225/.0592 | .1196/.0695 |

Full craters −.0148 at m=16 while tail gains +.0012. The net was trained *with* the cap, so its updates are
calibrated to being clipped; removing it lets overshoot through. **83% clipping is the mechanism working,
not a defect.** A small full/tail tradeoff lives in the cap; there is no free headroom.

**Scoping consequence of D3.** The whole gap from .1451 to a ~0.98 attainable ceiling is input + fold —
**none** of it is attributable to the frozen tower. That does not say the gap is recoverable (much of it is
the intrinsic thinness of 16 genre-scale answers), but nothing downstream is structurally blocked.

**Status of the architecture: CONVERGED under the current configuration; capacity, curriculum mixture and
answer granularity UNABLATED.** Not "optimal" — nobody has evidence for that. Fable retracted its earlier
"don't spend a run on capacity": flat NLL under a fresh-resampled curriculum proves *converged*, not
*capacity-saturated*.

---

# cv5 (hidden 1024, graded meh, 30 epochs) vs cv3 — MATCHED BANKS, 2026-08-04

## CORRECTION to what was reported at ~23:40

I told the author cv5 ep24 beat cv3 ep11 by **+0.0148 full / +0.0083 tail at m=16**. That was
**contract mixing and is retracted.** `concept_cold_check.py --bank expinfo` had no `expinfo` branch in
its ranking chain, so it fell through to `crank=None` — random popularity-weighted draws — while still
logging `bank=expinfo`. cv5's number was a RANDOM-DRAW score; cv3's was a genuine ranked-bank score.
Fixed in `26fe795` (total dispatch + assert: every named bank must resolve to a ranking).

## The matched numbers (both from `bank_sweep.py`, identical protocol, canary passing per bank)

delta = cv5 ep24 − cv3 ep11, full/tail:

| bank | m=1 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| pop | +.0004/−.0012 | +.0014/+.0018 | −.0015/+.0041 | −.0037/−.0003 | +.0050/+.0028 |
| entropy | +.0010/+.0013 | −.0003/−.0006 | −.0003/+.0002 | −.0009/−.0004 | +.0016/−.0019 |
| polar | +.0015/+.0020 | +.0020/+.0018 | +.0016/+.0013 | +.0038/+.0015 | +.0016/+.0001 |
| expinfo | +.0015/+.0020 | +.0020/+.0018 | +.0016/+.0013 | +.0038/+.0015 | +.0016/+.0001 |

**VERDICT: cv5 ≈ cv3.** Every delta is inside the ±.003–.004 per-user noise band at n=1500, and several
are negative. Under the two discriminative banks the sign is consistently positive but tiny; under `pop`
and `entropy` it is mixed. **The combined config change — hidden 256→1024, graded meh instead of zeroed,
kc0 dropped, zero-item share raised, 12→30 epochs — bought nothing measurable.**

That is a real negative result and it is worth more than it looks:

1. **Capacity was not binding**, which is what Fable's prior said and its retraction left untested. A 4×
   wider network trained 2.5× longer moved nothing. The flat-NLL reading was right after all.
2. **The remaining headroom is not in the network.** It is in the question bank (the `polar` result: which
   questions you ask changes which CHANNEL wins) and in the answer channel's own information content.
3. It cost ~11 hours of compute to learn, because all four levers moved at once — the trade the author
   explicitly accepted ("just one best bet at architecture, we don't need any analysis"). Worth restating
   the cost, not to relitigate the call, but because a single-lever run would have cost the same and
   answered which lever mattered.

**Which checkpoint the paper uses:** cv5 ep24, per the author's ruling to select on best full under the
pre-registered rule. Since cv5 ≈ cv3 this is a free choice; cv5 is preferred only because it is the run
whose config is fully recorded in the checkpoint (`hidden`, `keep_meh`, `p_item_zero`).

**cv5 ep24 absolute numbers, `expinfo` bank, floor .1277/.0267:**
m1 .1323/.0422 · m2 .1351/.0455 · m4 .1372/.0508 · m8 .1449/.0634 · m16 .1467/.0705.

---

# PAIRED CIs on concept-minus-item — POLICY B, certified cv3 ep8

Raw: `experiments/instrument/concept_vs_item_paired_policyB.json` (2026-08-05 15:58, `_pool_policy`
= `"B (see concept_cold_check.apply_pool_mask; 2026-08-05 ruling)"`). 10k-resample paired bootstrap,
n=1500 full / n=1448 tail (tail excludes users with no tail target). Both arms score identical users with
identical held-out targets; m=0 arms-matched canary asserted. `SEP` = the 95% interval excludes zero.

**`polar` — the load-bearing result, SEPARATED ON BOTH AXES AT EVERY BUDGET** (retired-contract value in
grey for scale):

| m | full Δ [CI] | (retired) | tail Δ [CI] | (retired) |
|---|---|---|---|---|
| 1 | **+.0099** [+.0054,+.0146] | +.0045 | **+.0106** [+.0047,+.0167] | +.0109 |
| 2 | **+.0166** [+.0115,+.0217] | +.0075 | **+.0171** [+.0104,+.0240] | +.0151 |
| 4 | **+.0178** [+.0118,+.0238] | +.0103 | **+.0254** [+.0181,+.0327] | +.0192 |
| 8 | **+.0376** [+.0308,+.0446] | +.0189 | **+.0407** [+.0325,+.0487] | +.0310 |
| 16 | **+.0358** [+.0287,+.0432] | +.0146 | **+.0385** [+.0310,+.0462] | +.0265 |

**The other three banks — the item channel's full-axis win is now confined to m ≥ 8.** FULL axis:

| bank | m=1 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| pop | −.0042 SEP | +.0007 ns | +.0039 ns | −.0130 SEP | −.0307 SEP |
| entropy | **+.0113 SEP** | **+.0124 SEP** | +.0068 ns | −.0241 SEP | −.0468 SEP |
| expinfo | +.0014 ns | +.0049 ns | −.0008 ns | −.0111 SEP | −.0340 SEP |

TAIL axis:

| bank | m=1 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| pop | −.0061 SEP | +.0008 ns | −.0046 ns | −.0069 ns | −.0033 ns |
| entropy | +.0108 SEP | +.0102 SEP | +.0040 ns | −.0186 SEP | −.0270 SEP |
| expinfo | +.0059 ns | +.0040 ns | −.0010 ns | −.0131 SEP | −.0242 SEP |

## The defensible claims, restated on policy B

1. **Under a pure-discrimination bank (`polar`) the concept channel beats the item channel on both axes at
   every budget** — all ten cells separated, and by roughly twice the margin the retired contract showed.
   This is the claim the chapter carries. Mechanism, recorded above and pool-invariant: a divisive item is
   one few people have seen (4.8% answered), while a divisive concept stays 97.2% answerable.
2. **Off `polar`, the item channel wins FULL from m=8 onward — not "decisively at every budget".** At
   m ≤ 4 the full axis is either a wash (`pop`, `expinfo`: five ns cells) or separated in the *concept*
   channel's favour (`entropy` m=1, m=2). The retired contract's blanket item win at short budgets was an
   artifact of masking the items the item arm had just revealed.
3. **The crossover is now identified to a budget band rather than only in direction:** concepts lead on
   both axes at m ≤ 4 under both discriminative banks, and items take over on full at m=8. The exact
   crossover point between m=4 and m=8 is still not measured.
4. `entropy` remains degenerate as a *criterion* (see above) whatever the pool: its top-16 sits at the
   99.9th popularity percentile. Its cells are reported, not interpreted as discrimination.

<details>
<summary>⚠ Superseded: the same CIs on the retired contract, concept net cv5 ep24 (kept, not deleted)</summary>

10k-resample paired bootstrap, n=1500 full / n=1448 tail. Retired pool; cv5, not the certified cv3 ep8.

`polar`, separated on both axes at every budget:

| m | full | tail |
|---|---|---|
| 1 | +.0042 [+.0016,+.0067] | +.0120 [+.0071,+.0171] |
| 2 | +.0069 [+.0029,+.0108] | +.0143 [+.0086,+.0201] |
| 4 | +.0060 [+.0007,+.0112] | +.0179 [+.0120,+.0236] |
| 8 | +.0147 [+.0086,+.0208] | +.0288 [+.0219,+.0354] |
| 16 | +.0115 [+.0054,+.0177] | +.0297 [+.0227,+.0366] |

The other three banks, tail axis:

| bank | tail m=1 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| pop | −.0038 ns | +.0026 ns | +.0077 SEP | +.0027 ns | +.0028 ns |
| entropy | +.0081 SEP | +.0067 SEP | +.0009 ns | **−.0089 SEP** | **−.0138 SEP** |
| expinfo | −.0010 ns | +.0068 SEP | +.0027 ns | −.0044 ns | −.0054 ns |

Full was separated NEGATIVE almost everywhere off `polar` (e.g. `pop` m=16 −.0720 [−.0798,−.0641]) —
the reading that item 2 above corrects.

</details>

---

# CERTIFIED CHANNEL IS cv3 ep8 (2026-08-04, author ruling: prefer the simpler architecture)

`models/i26c_ep31/concept_cv3_ep08.pt` — hidden 256, keep_meh FALSE, **464k params** vs cv5's 1.24M.

Re-selected under the FIXED ranked bank (the earlier cv3 pick of ep7 was made under the random-draw bug).
cv3 ep8 is not merely indistinguishable from cv5 ep24, it is **better**: SEL **0.1411** vs 0.1381, and
m16 **.1498/.0673** vs .1467/.0705. The simpler model wins outright, so the eleven hours cv5 cost bought a
negative result twice over — no gain, and the wider net was slightly worse.

**cv3 ep8, `polar` bank, all ten cells SEPARATED (paired, n=1500 full / 1448 tail):**

| m | concept | item | full Δ | tail Δ |
|---|---|---|---|---|
| 1 | .1327/.0412 | .1281/.0302 | +.0045 [+.0017,+.0073] | +.0109 [+.0062,+.0159] |
| 2 | .1357/.0463 | .1282/.0313 | +.0075 [+.0041,+.0108] | +.0151 [+.0097,+.0205] |
| 4 | .1415/.0521 | .1312/.0329 | +.0103 [+.0059,+.0147] | +.0192 [+.0138,+.0245] |
| 8 | **.1492/.0656** | .1303/.0346 | **+.0189** [+.0140,+.0238] | **+.0310** [+.0243,+.0374] |
| 16 | .1498/.0673 | .1352/.0409 | +.0146 [+.0094,+.0197] | +.0265 [+.0204,+.0325] |

Full-profile margins are **larger than cv5's at every budget**. The other three banks behave as before:
items win full decisively, tail mostly not separated.

**Everything above about cv5 stands as the reason we reverted** — the null result is what licensed
choosing the smaller model, and it should be kept, not deleted.
