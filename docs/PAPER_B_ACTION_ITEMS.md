# Paper B — action items (as of 2026-08-20)

Consolidated from the M4/M6 runs, two Fable verdicts and the author's rulings. Ordered by priority
within each block. **The single TEST look remains unspent and nothing below spends it.**

---

## ⚠ STATUS 2026-08-20 22:00 — what is running and what was retracted

**RUNNING (~40 h):** all six groups refitting on 2,000 TRAIN users, `min_leaf` 200, **no depth cap**
(author's ruling — match the cold tree's rule). One at a time. `run_m6_all_train2k.sh`.

**RETRACTED:** the +0.02162 held-out figure for a 2,000-user `arthouse` tree. The number is valid;
the claim attached to it was not — the fit used a corrupted group centroid (train indices against a
validation-keyed table, 597/2,000 matched silently). See
`memory/cross-population-index-match-is-silent.md`. Guards added; artifacts quarantined.

**STILL TRUE and unaffected** (validation-fitted trees, correct centroids throughout):
M6l in-sample (vs static 6/6, vs anchor 5/6, vs cold 4/6) and M6m held-out on train users
(vs cold 3/6, vs anchor 4/6, pooled +0.00278) — **the stop rule tripped on that evidence**, and the
running refit is the attempt to clear it with more data and depth.

**OPEN CHECK, cheap, do when the box is free:** is concept-side centroid lift genuinely
group-specific? The corrupted and correct centroids produced the same six genres and six tones.

---

## A. TREES — the payoff act. Highest priority: these decide what the paper claims.

| # | item | cost | why |
|---|---|---|---|
| **A1** | **Per-user deterministic openers** — seed the generator from the user id, not globally | minutes | The anchor approach had no draw to overfit to; ours re-draws per seed. This removes the fit/eval state mismatch that produced the apparent shrinkage (+0.0134 → +0.0076). Population still follows the census distribution. |
| **A2** | **Evaluate the six existing trees on TRAIN users as a held-out cohort** | ~10 min/group, **no refitting** | ~11,000 users per group instead of ~400, cutting the SE ~5× and answering Fable's "the A/B spread is one noise-width". Genuinely out-of-sample in users. ⚠ Needs D.6b amended (below). |
| **A3** | **Refit on half the cohort, score the held-out half** (Fable's gated experiment) | ~30–40 min/group | Closes the user in-sample axis directly. With A1 done, the 10-draw averaging Fable wanted is unnecessary. |
| **A4** | **Report uplift against headroom** (cold-arm NDCG at the opener state vs the tree's gain) | minutes | Tests the "a strong opener leaves less for the interview to add" mechanism. Turns a possible weakness into a finding. |
| **A5** | **Depth-2 trees — only for groups that survive A2/A3** | ~3 h/group | Depth-1 is a ceiling (3–4 nodes vs the cold tree's 164). Do not deepen a tree whose depth-1 effect is noise. |

**STOP RULE (pre-declared, Fable):** if the held-out tree−cold effect fails to exclude zero in ≥4 of 6
groups, or the pooled mean is below ~+0.004, the opener line becomes a characterised thesis negative,
Act 5 goes descriptive, and the paper ships on the ablation.

---

## B. POLICY / DISTILLATION — free overnight compute; the paper does NOT depend on it

Fable's position, correctly stated: the current failure is the **protocol, not the paradigm**, and
fixing it is the highest-EV single move. What it advises against is letting the *submission* depend on
it landing in six weeks. Promote back into the paper only if B1–B3 make the mix cell non-negative
before the test look.

| # | item | why |
|---|---|---|
| **B1** | **One group per train user**, assigned by nearest centroid — the *same rule* evaluation already uses | 27.8% of warm users currently appear in several banks with conflicting labels for near-identical latents. Irreducible noise; also a train/eval distribution mismatch. |
| **B2** | **★ Distil from the banked GAIN VECTORS, not the argmax label** | Verified available: 22 sweeps in `m6/`, 238 in the M4 battery, each 999 candidates. A user near two groups then gets two gain *profiles* whose mixture is a coherent target, where two argmaxes are pure noise. Costs nothing new; targets stay measured NDCG. |
| **B3** | **Train to convergence**, not a fixed 3 epochs | Fidelity was still climbing when we stopped (0.9564 → 0.9717 → 0.9722). |
| **B4** | Re-run generalisation (half-B films) and the mix | The existing null is **not interpretable** — the policy loses to its own teachers, so it never cleared the fidelity gate. |
| **B5** | Optional: a small DAgger dose, 20–50 oracle labels on high-traffic divergence states | Viable for the first time — E1's `frac_never_deviates=1.0` starved it; the policy now genuinely diverges. |

---

## C. PAPER — writing todos (W-series in `PAPER_B_POLICY_DESIGN.md`)

| # | item |
|---|---|
| **W6** | Elevate the ablation from analysis to **prescription** in the abstract and intro: *optimise the reported metric, and only that.* |
| **W7** | **Run GreedyExtend** (`golbandi2010bootstrapping`) — the missing external *static* anchor. Their tree is already run and loses; only their static is uncited-unrun. Hours of work; kills the "your comparator is self-built" objection. |
| **W8** | Adopt the **"the objective is the bug"** spine; the ablation is load-bearing, the volunteered line is the payoff act. |
| **W9** | Demote distillation to **infrastructure**, never a contribution. |
| **W10** | Fix the **test-look budget** and retreat lines before the run. Pre-register the **value** criterion, not routing. |
| **W11** | State the ablation's **held-fixed design** plainly and early ("only the scalar changes"); do NOT claim we "repaired a published method". |
| **W12** | If the policy is deferred: **two sentences in the discussion**, not silence — representable (fidelity ≥0.999, no interference), routes unseen films above chance, does not convert to ranking gains, characterised as a negative. |
| ✔ done | `sec:circularity` (answer-model defence consolidated); `sec:opener_synthesis` + `tab:opener_match` (corpus-matched opener methodology). |

---

## D. DECISIONS NEEDED FROM THE AUTHOR

1. **Amend D.6b?** It forbids NDCG on train users for two reasons: "no fold-in/target split" (no longer
   true — `train_tr_te` now exists, 2,114,192 target rows) and "the tower memorised them" (still true —
   levels inflated, but the inflation is common to both arms in a contrast). A2 depends on this.
2. **M5 test look:** is arm B reported only on the 28.74% single-anchor users? Is K=15 the pre-registered
   primary budget?
3. **Ethics application** for the undergraduate answerability study — Fable says start it now but treat it
   as a Paper A / thesis asset, not a Paper B deliverable (latency can't clear 2 Oct, and a negative
   arriving mid-review would be radioactive).

---

## E. EXPLICITLY NOT DOING

- **Policy gradient / bandit optimisation against measured NDCG.** Converts the firewalled ruler into a
  training signal on the same 10,000 users; our own record (held-out exposing RL gaming) says how that ends.
- **Oracle labels at broadly sampled states.** No amortisation — weeks of compute for a training set with
  the count-not-composition pathology.
- **A second value/gain head as a label source.** The first lost to a parameter-free constant
  (+0.0623 vs +0.4163). Only defensible form is proxy-as-*pruner*, and that is not on the critical path.
- **A human study for Paper B.** Ethics latency; the behavioural-answer design is the substitute and
  should be argued as such.
