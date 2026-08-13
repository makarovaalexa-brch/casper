# EDDI's three-outcome response model, at contract scale (TODO 1.3(b))

Run 2026-08-13. **Discharges TODO 1.3(b) and nothing else** — see §5 for what remains open.

The only prior artifact for `eddi_ig3` was `SMOKE_adaptive_table_SNAP3_smoke.json`: 200 users, stamped
`"SMOKE_NOT_CITABLE": true`, budgets ≤ 8, one consumer, no intervals, and its two sub-variants
disagreed on the tail. This is the same two arms on the chapter's contract.

**Contract, identical to the rest of `tab:battery`:** all **10,000** canonical Liang ML-25M TEST
users (no sampling of users, items or candidates anywhere), consumer = certified pair
(`t2i26c_ep31.pt` + `concept_cv3_ep08.pt`), **policy-B** candidate pool, **NO-FLOOR**, behavioural
answers, learned decoder bias, full **and** tail NDCG@10. Inference: paired percentile bootstrap over
per-user differences, **10,000 resamples, seed 98765** — never the unpaired `1.96*sqrt(2)*SE` form.

| | |
|---|---|
| launcher | `scripts/run_eddi3_policyB.ps1` (commit `609f029`) |
| raw cells | `experiments/baselines/interview/adaptive_table_eddi3_policyB.json` (+ `_per_user.npz`) |
| analysis | `src/baselines/a9_battery_policyB.py --label a9_battery_policyB_eddi3` (extended, not forked) |
| intervals | `experiments/baselines/interview/a9_battery_policyB_eddi3.json` |
| cost | interviews 11:48–15:11 (the `signed` k=16 cell alone is 75.5 min / 3,662 states); bootstrap 49.9 min |

---

## 0. Gates — all passed; no number below is reportable without them

| gate | check | result |
|---|---|---|
| **G3 — the banked contract check** | this run also re-computed `eddi_ig`, the **two-outcome row already in `tab:battery`**. It must reproduce `act1_per_user_vectors.npz` **per user** | **max per-user \|d\| = `0.0e+00`** on all 5 budgets × both axes. **Deviation is exactly zero.** Answered@16 `3.844` and 1,993 distinct ask-prefixes also reproduce the banked run and `tab:prefixes` exactly |
| **G2 — cross-run** | this run's `entropy0-static` vs banked act1, per user | **`0.0e+00`** (and the four pre-existing sources still `0.0e+00`) |
| **banked static targets** | the three rebuilt policy-B orders' K=16 test figures | combined `.25900/.12370`, items `.24470/.10171`, concepts `.24467/.11607` — exact to 5 dp |
| **G1 — cross-family** | instrument vs arm-N harness, per user, on their one shared arm | `0.0e+00` after the instrument's float32 round-trip, both axes |
| **bootstrap equivalence** | chunked vs the `gtport_paired` reference | IDENTICAL |
| **self-test** | an arm minus itself | delta and both CI ends exactly `0.0` |
| **leak check (test side)** | `arm_n` ruler line, this run's own log | `18359 items, 10000 test users, head=190; pool excludes 141.2 items/user vs 62.6 in arm A; **target leak 0**`; derived pool fingerprint `nnz=1411761 per_user=141.1761 fp=cc002b4eb984ed9880346c98fc0a2970` |

Because G3 holds at 0.0e+00, the two-outcome vs three-outcome comparison below is **in-run** — same
process, same seeds, same pool — not a cross-run difference.

---

## 1. The curves, all 10,000 test users

Consumer `ours`, full/tail NDCG@10. `entropy0-static` and `latent-only info-gain` are re-derived
here and are bit-identical to the banked table; they are shown as anchors.

| arm | k=1 | k=2 | k=4 | k=8 | k=16 | answered@16 |
|---|---|---|---|---|---|---|
| latent-only info-gain (two-outcome, `eddi_ig`) | .1537/.0336 | .1539/.0373 | .1618/.0432 | .1747/.0611 | .1917/.0749 | **3.84** |
| **EDDI three-outcome, `zero`** | **.1607/.0363** | .1597/.0388 | **.1732/.0497** | **.1869/.0660** | .2041/**.0834** | **2.86** |
| EDDI three-outcome, `signed` | **.1607/.0363** | **.1607**/.0383 | .1655/.0442 | .1852/.0649 | **.2041**/.0815 | 3.16 |
| *(anchor)* entropy₀ static | .1635/.0433 | .1706/.0528 | .1831/.0636 | .2053/.0792 | .2303/.0924 | 5.53 |

Unrounded, for anyone checking:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| two-outcome | .15372/.03363 | .15394/.03726 | .16181/.04320 | .17470/.06111 | .19168/.07493 |
| three-outcome `zero` | .16075/.03626 | .15969/.03875 | .17320/.04966 | .18690/.06600 | .20408/.08342 |
| three-outcome `signed` | .16075/.03626 | .16068/.03825 | .16547/.04423 | .18522/.06489 | .20413/.08149 |

Answered counts and interview branching, per budget:

| arm | answered @1/2/4/8/16 | distinct ask-prefixes @1/2/4/8/16 |
|---|---|---|
| two-outcome | 0.26 / 0.42 / 0.79 / 1.80 / **3.84** | 1 / 2 / 7 / 99 / **1,993** |
| three-outcome `zero` | 0.18 / 0.36 / 0.75 / 1.51 / **2.86** | 1 / 2 / 8 / 112 / **1,217** |
| three-outcome `signed` | 0.18 / 0.36 / 0.73 / 1.62 / **3.16** | 1 / 3 / 27 / 817 / **3,662** |

---

## 2. The mechanism claim: CONFIRMED on the full cohort, in-run and paired

The chapter's claim is that the three-outcome model collects **fewer** answers and still ranks
**better**. Both halves hold at 10,000 users, and the ranking half separates almost everywhere.

**Answers.** `zero` collects **2.86** answers in sixteen questions against the two-outcome arm's
**3.84** — **26% fewer**. (`signed` collects 3.16, 18% fewer.)

**Ranking — three-outcome minus two-outcome, paired, in-run:**

| variant | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| `zero` | full | **+.0070** [+.0056,+.0084] | **+.0057** [+.0041,+.0074] | **+.0114** [+.0090,+.0138] | **+.0122** [+.0098,+.0146] | **+.0124** [+.0096,+.0151] |
| `zero` | tail | **+.0026** [+.0011,+.0042] | +.0015 [−.0001,+.0031] ns | **+.0065** [+.0042,+.0088] | **+.0049** [+.0024,+.0073] | **+.0085** [+.0057,+.0113] |
| `signed` | full | **+.0070** [+.0056,+.0084] | **+.0067** [+.0050,+.0085] | **+.0037** [+.0017,+.0057] | **+.0105** [+.0081,+.0130] | **+.0124** [+.0096,+.0153] |
| `signed` | tail | **+.0026** [+.0011,+.0042] | +.0010 [−.0006,+.0026] ns | +.0010 [−.0009,+.0029] ns | **+.0038** [+.0013,+.0062] | **+.0066** [+.0038,+.0094] |

**10 of 10 full-axis cells separate positively.** On the tail, `zero` separates in 4 of 5 (the k=2
cell is a null); `signed` separates in 3 of 5. So the mechanism sentence is now a measurement, not a
status — but it should be stated with the tail qualification, not as a flat "better rankings".

---

## 3. Which variant the chapter should carry: **`zero`** — and the reason is not the number

**Carry `eddi_ig3_zero`.** Two reasons, in this order:

1. **It is the faithful instantiation; `signed` is a documented approximation.** The frozen
   Partial-VAE was trained on binary LIKES and has **no learned token for a disliked observation** —
   there is no `Phi^(0)`. `zero` therefore lets the third outcome enter only through the
   answerability weight `a_i`, which is exactly what the Partial-VAE premise licenses (an unobserved
   variable is an *absence*, not a value). `signed` conditions on `c_O − Phi_i`, borrowing arm N's
   negative-evidence convention — a deviation `eddi_selector.py`'s own header labels "A DOCUMENTED
   APPROXIMATION, not the paper". Under a chapter that relabels arms the moment they stop being the
   published method, the approximation should not be the row that carries EDDI's name.
2. **The data does not force a trade-off, so the principled pick is not a flattering pick.** `zero`
   is **never worse than `signed` at any budget on either axis**: identical at k=1 and within 0.00005
   on full at k=16, ahead on full at k=4 (+.0077) and k=8 (+.0017), and ahead on **tail at every
   budget from k=2** (+.0005, +.0054, +.0011, +.0019). It also collects **fewer** answers (2.86 vs
   3.16), which sharpens the mechanism claim rather than softening it.

**The smoke's disagreement does not survive.** On 200 users the two variants split hard on tail at
k=8 — `zero` .0682 against `signed` .0517, straddling the two-outcome's .0600. At 10,000 users they
are .0660 and .0649, **both above** the two-outcome's .0611, and the gap between them is 0.0011. The
smoke's apparent contradiction was a 200-user artifact, and no conclusion should have rested on it.

One structural difference worth a line: `signed` keys its cache on `(likes, dislikes)` **and**
conditions on both, so it branches nearly 3× harder — 3,662 distinct interview states at k=16 versus
`zero`'s 1,217 — for no gain. That is 75 minutes of CPU against 26, buying a strictly weaker arm.

---

## 4. Against the static comparators — Act 1's claim does not move

Differences are **arm minus static**, so negative = the static order wins. `items-only` is the
like-for-like comparator (EDDI is an item-channel method); the combined and concepts orders are
reported as **bounds** and are labelled as such — they include a channel the EDDI arms cannot ask.

**vs `greedy items-only` (policy-B rebuild — THE like-for-like comparison):**

| arm | axis | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| three-outcome `zero` | full | −.0152 | −.0287 | −.0306 | −.0366 | −.0406 |
| three-outcome `zero` | tail | −.0072 | −.0175 | −.0180 | −.0200 | −.0183 |
| three-outcome `signed` | full | −.0152 | −.0277 | −.0383 | −.0383 | −.0406 |
| three-outcome `signed` | tail | −.0072 | −.0180 | −.0234 | −.0211 | −.0202 |
| *(two-outcome, for scale)* | full | −.0223 | −.0344 | −.0420 | −.0488 | −.0530 |
| *(two-outcome, for scale)* | tail | −.0098 | −.0190 | −.0245 | −.0249 | −.0268 |

**Every one of these 20 cells separates**, all in the static order's favour.

**vs `greedy combined` and `greedy concepts` (BOUNDS, not like-for-like** — those orders may spend
turns on the concept channel, which is unavailable to an item-only acquisition): `zero` runs from
−.0200 to −.0549 on full and −.0175 to −.0403 on tail; `signed` from −.0200 to −.0549 and −.0175 to
−.0422. All separate. Quote these only as an upper bound on how far the arm is behind.

**What this changes in Act 1: the size of a gap, not the sign of anything.** Fixing the response
model recovers roughly **a quarter of the two-outcome arm's deficit** against the items order on full
(−.0530 → −.0406 at k=16) and about **a third on tail** (−.0268 → −.0183), and it does so while
asking fewer questions. It does not come close to closing it. The narrative sentence re-verified over
**all** adaptive rows including the two new ones: **138/138 cells** lead-and-separate for the static
order against the combined and the concepts orders, **134/138** against items — and the four
exceptions are the **pre-existing** Golbandi k=4/k=8 tail and IGCN k=2/k=4 tail cells, none of which
separates in either direction. **No new exception is introduced by either EDDI variant, and no
published adaptive selector in the battery beats the strong static order on either axis at any
budget.** Act 1's claim is unchanged; it is now defended against a stronger version of EDDI.

---

## 5. What this run does NOT settle — state this plainly wherever the row is used

This run fixes the **response model** only. Two of the three defects recorded in
`docs/results/EDDI_IMPLEMENTATION_STATUS.md` are untouched, and TODO 1.3 stays open on both:

1. **Full Eq. 9 is still not implemented — anywhere.** Our acquisition is the **first term only**
   (the latent-movement / BALD-style gain). The stated leakage justification for dropping the second
   term was **retracted in commit `b4a4783`** — Eq. 9's second term samples its hypothetical answers
   from the model and leaks nothing, so the term was dropped for a reason that does not hold, and
   App. C.3 argues *against* the first term alone. The native snap still stands: `eddi_latent`
   pooled average rank **3.402 vs random 3.236** on Ma et al.'s own Fig-3 datasets. **The row must
   keep the label "latent-only info-gain", and the reported snap failure must stay.** (TODO 1.3(a),
   needs a design sheet on what plays `x_phi` in cold-start ranking.)
2. **The answerability weight is still not the paper's marginal.** In the two-outcome arm `p_i` is a
   Mult-VAE^PR **softmax across items** — a relative next-draw probability, not the per-variable
   marginal `p(x_i | x_O)` that Ma et al.'s factorised likelihood (their Eq. 2) requires; the
   docstring records `corr(log p, log R) = 0.965` at the empty state, i.e. close to a
   most-probable-item selector. The three-outcome arm replaces that weight with `a_i` and `q_i`,
   both proper per-item quantities in [0,1] — **which is very likely why the numbers move** — but
   **the 7.37 / 1.67 / 0.965 diagnostic itself still has no reproducible artifact** (docstring and
   commit body only) and must be re-derived before it is quoted as a measured number.

So: the three-outcome arm is a **better-instantiated response model over the same reduced
acquisition**. It is not EDDI.

---

## 6. Operational note (HARD RULE 11)

The first launch died silently at 12:46 during the `signed` k=16 cell — no traceback, empty stderr,
python gone (consistent with the detached child being reaped, or an OS kill; free RAM was 6.3 GB at
launch and 4.1 GB after). **Per-cell persistence did its job: 19 of 20 cells were already on disk,
and the per-user npz with them.** The relaunch matched the results contract, printed
`RESUMING … 19 scored cell(s) already banked`, skipped all 19 without repeating either the interview
or the scoring, and computed only `eddi_ig3` k=16. Nothing was recomputed and nothing was lost —
which is the whole point of the rule, and the second time this month it has paid for itself.

The `signed` variant is also the memory/compute risk in this family: keying on `(likes, dislikes)`
and running two KL sweeps per state gives it 3,662 cached 18,359-wide float64 vectors at k=16. If it
is ever re-run, bound that cache first (HARD RULE 11c) — and note that a bound changes which
questions get asked, so it makes a different arm and must be labelled as one.
