# Design sheet: Act 2 — make the mechanism a MEASURED relationship, not a correlation

Author discussion 2026-08-10. Act 2 currently shows low dispersion where adaptivity fails
(10,000 ML users) and high dispersion where placement exists (4,431 Reddit requesters). Those are
**different populations**, so the mechanism is argued, not measured. Two designs fix that, plus one
reframing the author identified that may matter more than either.

---

## 1. THE QUARTILE STRATIFICATION (author-approved, do first)

**The problem with "dispersion" as a per-user quantity.** Dispersion as defined is a *population*
statistic — the spread of argmaxes across users — so it cannot be assigned to an individual. What
*can* be assigned is:

> **Personalisation value** `PV(u) = U(q*_u | z_u) − U(q*_pop | z_u)`
>
> where `q*_u` is that user's own best next question, `q*_pop` is the population's modal best
> question, and both are scored **on that user's own state**. It is exactly what a perfect
> personaliser would gain over a one-size-fits-all order, for that user, at that moment.

Worked example, to keep the definition honest:
* **User A**, no answers yet. Their best question is the same as everyone's (`Shawshank?`).
  `PV = 0.000` — personalising is worth nothing to them.
* **User B**, has said yes to a horror concept. Their best question is now `Poltergeist?` while the
  population pick is still `Shawshank?`. `PV = 0.008` — that is what a one-size-fits-all order costs
  them.

**Design.** Compute `PV(u)` for all 10,000 test users at k ∈ {1,2,4,8}; sort; cut into quartiles.
For each quartile report the **adaptive-minus-static gap** (per-user paired, same discipline as
everywhere else: in-run `fixedorder:` comparators, paired percentile bootstrap, 10k resamples,
seed 98765).

**The pre-registered reading, both branches publishable:**
* **Gap grows with PV** → the mechanism is measured on the same users. "Adaptivity fails because
  there is nothing to adapt to" becomes a demonstrated relationship rather than a story.
* **Gap flat or absent even in the top quartile** → the story is *not* "nothing to adapt to" but
  **"methods cannot find it"** — a different and equally publishable finding, and one that would
  redirect Act 4 toward why.

**Cost:** low. Per-user vectors and per-user states are already on disk; `PV` reuses the leg-1
criterion machinery.

**Controls:** shuffle `PV` across users (the gap-vs-quartile trend must vanish); report quartile
answer rates, since `PV` may correlate with profile size and that must not be doing the work.

---

## 2. LINK IT TO THE BRANCHING DIAGNOSTIC (cheap, same run)

Act 1 measured that Thompson gives all 10,000 users a *distinct interview* by k=4, while Act 2 says
only ~5 questions are genuinely worth asking. Those are two separate observations of one thing.
Quantify it directly: **of the distinct interviews a selector actually conducts, how many distinct
loss-optimal questions do they correspond to?** That single ratio turns "divergence without
exploitability" into a number.

---

## 3. ★ THE REFRAMING (author, 2026-08-10) — answerability has a bootstrapping problem

The author's objection, which the chapter must absorb: *"we could reliably predict what they can
answer only after we know something about them, and in a cold interview without open questions we
cannot, for pretty much all of that interview."*

This is right, and it has three consequences.

**(a) Popularity IS the answerability prior.** With no knowledge of a user, the best available
predictor of "can they answer this?" is "have many people seen it?". This is a candidate
explanation for several standing results: why popularity is a stubborn baseline; why entropy₀
(which blends popularity) is the strongest heuristic; and why the coarse channel wins early — a
concept is answerable by ~97% of users *regardless of who they are*, so it requires no personal
knowledge at all. **Testable:** correlate each selector's answer rate against the population
popularity of what it asks, per turn.

**(b) The oracle's +.084 is NOT all capturable headroom.** `dpe_oracle` reads the user's own
ratings at selection time; part of its advantage is information **that does not exist at interview
time**. The honest decomposition is:

> oracle gap = (what is predictable from answers so far) + (what is unknowable at ask time)

Only the first term is a target. Reporting the whole gap as "headroom methods fail to capture"
overstates it — a correction the chapter should make itself rather than wait for a reviewer.

**(c) The decomposition is measurable.** Build a selector that *predicts answerability* from the
answers so far — P(user has rated item i | answers so far), trained on train users, no test leakage
— and use it to weight an otherwise standard criterion. How much of the oracle gap does it recover?
That number separates "adaptive methods are failing at something achievable" from "adaptive methods
are chasing something unknowable", and it is the cleanest remaining test of the chapter's own
mechanism claim.

---

## 4. WHAT BELONGS TO ACT 3 INSTEAD (clarified with the author)

Inferring a user's favourite genre or favourite film and folding it in as an opener is **not** the
Act-2 stratification — it is the **Act-3 intervention**. The distinction:

| | Act 2 stratification | Act 3 open opener |
|---|---|---|
| operation | **observational** — partitions users by a property they already have | **interventional** — changes the state |
| what it shows | that the gap tracks the constraint | that relieving the constraint changes the outcome |
| Reddit's role | none (ML users only) | supplies what real open answers look like |

Act 2 measures the constraint; Act 3 relieves it. The genre/film emulation, the pre-registered
answer rule and its sensitivity row all stay in Act 3 (`sec:act3`), with the 9-point threat model
already recorded in `PAPER_B_DESIGN.md`.

---

## Order

1. `PV` quartile stratification (+ shuffle control, + answer-rate-by-quartile).
2. The branching/optimal-question ratio (same run, nearly free).
3. The answerability-prediction selector and the oracle-gap decomposition.
4. Act 3 proper.
