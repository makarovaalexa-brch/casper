# P2 — GENERALISATION ACROSS STATES: the evidence-seeded fine-tune

**Status: COMPLETE, 2026-08-15. Validation only — the 10,000 canonical TEST users were never built,
never loaded, never named. `test_touched=false` on every artifact.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **Part I.2 is governing**, with its state count
corrected upward (§"the census" below). Code: `src/instrument/policy_p2.py`, committed before every
run (HARD RULE 10); the evaluation calls R2's own harness functions
(`policy_resume.draw_volunteered / rollout_seeded / build_seqs`) rather than a lookalike.

> **AUTHOR SAID (2026-08-15):** *"please run."* — and, on why: *"i want to show generalisability
> working here and my recommender support ood questions etc."*

---

## ★ THE VERDICT, FIRST — THE PRE-REGISTERED FALSIFIER FIRED

The falsifier was: *if, trained ON evidence-seeded states, deviations on fresh seeded states still
cost (CI excludes zero from below), then this action space does not reward state-conditioning beyond
the tree's even when taught. That is the strongest honest closure available, and it is reported as
the finding, not buried.*

**It fired at every k, and the deviations cost MORE after training than they did before it.**

| k volunteered items | ★ fine-tuned − ITS OWN frozen order | R2's untrained belief-only policy | shuffled-label control |
|---|---|---|---|
| 2 | **−0.02825 [−0.03113, −0.02529]** | −0.01246 | −0.03394 [−0.03690, −0.03105] |
| 4 | **−0.02054 [−0.02337, −0.01771]** | −0.01234 | −0.02301 [−0.02583, −0.02025] |
| 8 | **−0.01434 [−0.01702, −0.01174]** | −0.00860 | −0.01797 [−0.02053, −0.01537] |

The bar was **≥ 0 with the interval excluding a loss**. Every interval excludes zero **from below**.

> **The sentence the chapter gets: taught on 210 states where evidence was already on the table, with
> measured labels and a working anti-forgetting replay, the policy learned to move — and every way it
> moved was worse than not moving. This action space does not reward state-conditioning beyond the
> tree's, even when the conditioning is explicitly taught.**

**The fine-tune is not the failure — it did its job.** It cleared every process gate:

- **spanning gate PASS**: unseeded K=16 validation rollout `0.28354971825495506` — E1's value to
  every digit, 0.03% of users deviating, 16 distinct interviews against the tree's 15. The replay
  protected the anchor exactly as designed.
- **it learned the seeded states**: top-1 on the seeded labels rose from ~0 to **0.663** in the
  as-asked encoding and **0.358** in the as-volunteered encoding across 4 epochs.
- **it stopped being inert**: on fresh seeded states it now changes the interview for **89.9 / 90.8 /
  93.6%** of users and **14.2 / 14.3 / 14.7 of 16 positions**, against R's **1.0 / 2.5 / 5.1%** and
  0.13 / 0.30 / 0.60 positions. Distinct realised interviews: 4,031 / 5,772 / 7,291 against 15.

So P2 converts R's *"it does not respond"* into *"it responds, and responding is what costs it"*.
Combined with R2 (*sensitive but not informed*), the three experiments now bracket the question
completely: the policy can be made inert, sensitive-by-accident, or sensitive-by-training, and only
the inert version scores well.

---

## ★ THE ONE THING THAT DID FLIP — and it is a real behavioural finding

R and R2 both reported **no detectable avoidance**: every arm re-asked a volunteered film at the rate
of an arm that could not see it. **The fine-tuned policy is the first arm that avoids them**, and now
with an interval:

| k | tree | static | **fine-tuned** | shuffled control | fine-tuned − tree, re-asks per user (paired) |
|---|---|---|---|---|---|
| 2 | 1.439% | 1.370% | **0.794%** | 0.876% | **−0.0150 [−0.0228, −0.0074]** |
| 4 | 2.873% | 2.658% | **1.532%** | 2.089% | **−0.0447 [−0.0552, −0.0344]** |
| 8 | 5.408% | 5.098% | **3.523%** | 4.037% | **−0.0416 [−0.0562, −0.0276]** |

(re-asks **per item question asked**; the paired interval is on re-asks per user, against the tree.)

**The fine-tuned policy re-asks a volunteered film at roughly half the tree's rate, and every
interval excludes zero.** That is the mechanism the design said would be the clean tell — *a policy
that reads the volunteered set has no reason to ask about a film whose answer it already holds* — and
it is the first time in this workstream it has fired.

⚠ **And it must be reported with its confound, which is large.** The shuffled-label control *also*
reduces re-asking (0.876 / 2.089 / 4.037%), and so does the static order. Part of the drop is
therefore not "it read your list" but "it asks a different mix of questions". Given H's standing
correction that a 1.4–5.5% base rate is too small to carry a claim either way, the honest sentence
is: **avoidance is now measurable and interval-backed, it is roughly half the tree's rate, and a
label-shuffled arm gets a good part of the way there.**

---

## ★ WHY THE DEVIATIONS COST — the mechanism, measured

| k | arm | concept share of asks | **interview answer rate** | full NDCG@10 |
|---|---|---|---|---|
| — | tree / E1 (unseeded) | 0.5457 | 0.5789 | 0.28355 |
| 2 | tree, seeded | 0.5457 | 0.579 | 0.28610 |
| 2 | **fine-tuned** | **0.2943** | **0.3792** | **0.25785** |
| 4 | **fine-tuned** | 0.3305 | **0.4038** | 0.27180 |
| 8 | **fine-tuned** | 0.3764 | **0.4375** | 0.28858 |

**The fine-tuned policy swings hard toward the item channel and its answer rate collapses from 0.579
to 0.379.** It asks about eleven films per interview instead of the tree's seven, and a third to a
half of those questions get no answer at all. A burned turn buys nothing, and sixteen turns with a
0.38 answer rate cannot beat sixteen turns with a 0.58 answer rate no matter how well-targeted they
are. **The avoidance it learned is real; it paid for it in answerability, and answerability is the
thing this whole chapter keeps measuring as the dominant term.**

This is the same mechanism as E1's open-vocabulary result (out-of-pool picks answered 9.1% of the
time, −0.02034 NDCG) arriving through a different door: **whenever this policy is pushed off the
distribution its supervision came from, it spends turns on questions people cannot answer.**

---

## The census — Part I.2's state count was WRONG, and is corrected upward here

Part I.2 specified *"the 138 evidence-seeded states from the 2026-08-14 random-prefix sampling"*.
That undercounted. The 425-state / 72,717-triple bank was built for the **value** model and has only
ever been used for value regression — but **every one of its states also carries a measured ACTION
label, and none of them had ever trained anything.** Counted by source rather than assumed:

| source | states | on E1's manifold? | teacher quality of the action label |
|---|---|---|---|
| `ndcgtree16_heaps` + `ndcgtree_heaps` | **216** | **YES** — the tree's own greedy-tail states | (this is the replay, not new material — **excluded**) |
| `policy_labels_deep_partial` | **42** | no (below the tree's floor stops) | **exact** — ~996 of 1,000 pool candidates measured |
| random-prefix sweeps | **138** | no | **weak** — argmax over the ~22 in-pool of 50 sampled candidates |
| `greedy_static_policyB_{items,concepts}_heaps` | **31** | no (static-order states) | **exact** — the banked `pick` is greedy's own chosen question |
| | **209 off-manifold** | | |

One of the 31 static-order states is the **empty state**, which E1 already handles with the tree's
root; relabelling it would disturb behaviour the policy already has — the exact phase-3 failure mode
Part I.0 warns about — so it was dropped and the drop is counted. **210 states were used**
(138 + 42 + 30). Part I.2's "138" should read **209 available / 210 used** and the correction is
recorded in the design sheet.

⚠⚠ **Teacher honesty, and it is not uniform.** The 138 random-prefix states — the majority of the
training mass — carry the **weakest** label in the mixture: the argmax over ~22 in-pool candidates of
1,000, not the pool argmax. The 42 below-floor and 30 static-order states carry exact labels. The
full-pool upgrade for the 138 is priced at 12–23 h and **was not run**; Part I.2's rule was to run
the cheap version first and upgrade only if the direction was right and the magnitude shy. **The
direction is wrong, so the upgrade is not indicated** — a better teacher for a fifth of the states
does not turn −0.028 into ≥ 0.

⚠ **And the labels are measured on the cohort they are scored on.** The sweeps measured NDCG on the
10,000 validation users, so this fine-tune fits labels derived from validation NDCG and is then
evaluated on validation. That is the standing protocol of the whole chapter (the tree itself was
grown this way; the single test look is reserved and unspent), and it means the falsifier fired under
conditions that were, if anything, *favourable* to the fine-tune.

## The training set, printed before the time was spent

| | |
|---|---|
| states | **210** |
| (user, state) examples | **500,000** |
| from a measured answer path / prefix marginal / static-order / full-pool below-floor | 86,520 / 103,480 / 300,000 / 10,000 |
| distinct label questions | **125** (56 items / 69 concepts) |
| **label questions the TREE NEVER ASKS** | **77 of 125** |
| concept share of labels, example-weighted | 0.5446 (the tree's 0.5457) |
| prefix lengths | 1 … 15 |
| states whose argmax over all 50 was catalogue-wide (in-pool argmax used instead) | 2 of 210 |

**77 of the 125 label questions are ones the tree never asks**, so this fine-tune genuinely widened
the action vocabulary the policy is supervised on — it is not a re-run of the tree's own labels at
new states.

**★ The encoding decision, declared before the run.** A random-prefix state and a volunteered-evidence
state are the same *belief* with different *bookkeeping*: in the first the asked-history is populated,
in the second `z_t` has moved and the history is empty. R2 measured that this pairing — moved `z`,
empty history — is exactly what E1 never sees and what pins it to its opening. Training only the
as-asked form would have taught nothing the evaluation could see, and the falsifier would have fired
for a bookkeeping reason. **So every seeded state was presented in both forms, 50/50 inside the seeded
eighth.** The measured label is a property of the evidence, not of how it arrived. The two forms
learned at different rates (as-asked top-1 0.663, as-volunteered 0.358), which is itself worth
recording: the empty-history encoding is genuinely harder for this architecture.

## The mixture and the schedule

7/8 replay (E1's tree-state labels) + **1/8 seeded**, declared constant, not tuned post hoc: batch
512 = 448 replay + 64 seeded. Fine-tuned from the E1 anchor at **lr 2e-4** (the phase-3 rate),
4 epochs × 2,000 steps. **Checkpoint selection was declared before the run**: the last epoch clearing
the spanning gate. Epochs 1 and 2 failed it (by 3.6e-4 and **3.0e-6** — the second is 2 decisions in
160,000, a knife-edge that is recorded rather than rounded away); epochs 3 and 4 passed; **epoch 4 was
selected**, and no epoch was chosen on anything the evaluation can see.

---

## The evaluation

A **fresh volunteered draw, seed 424242**, disjoint by construction from R/R2's 98765, same
popularity-weighted recipe, k ∈ {2, 4, 8}, through R2's own harness functions.

| k | ask-nothing-more | static + 16 | tree + 16 | its own frozen order | **fine-tuned + 16** | shuffled control |
|---|---|---|---|---|---|---|
| 2 | 0.21162 | 0.26876 | 0.28610 | 0.28610 | **0.25785** | 0.25215 |
| 4 | 0.24242 | 0.27866 | 0.29233 | 0.29233 | **0.27180** | 0.26933 |
| 8 | 0.27252 | 0.29155 | 0.30293 | 0.30293 | **0.28858** | 0.28496 |

Against the tree the fine-tuned policy loses at every k (**−0.02825 / −0.02054 / −0.01434**, all
excluding zero) and it now also loses to the **static order** (−0.01091 / −0.00686 / −0.00296). It
still beats ask-nothing-more comfortably (+0.04623 / +0.02938 / +0.01606), so the interview is worth
having; it is just worse than the interview it replaced.

**The frozen-order arm is numerically identical to the tree here**, and that was verified rather than
assumed: the fine-tuned policy's unseeded rollout differs from the tree's on 3 of 10,000 users, and
those 3 users score exactly 0.0 under both orders, so the per-user vectors coincide.

**Among the deviators only** (the users whose interview actually changed): −0.03141 [−0.03464,
−0.02821] at k=2, −0.02261 at k=4, −0.01533 at k=8. The loss is not diluted by non-deviators; it is
concentrated in exactly the behaviour the fine-tune created.

**Controls.** The shuffled-label arm (a state gets another state's measured label; the replay
untouched) also clears the spanning gate and also adapts — on 96.5–98.2% of users — and costs *more*
at every k. **The measured labels therefore do carry real signal relative to permuted ones** (+0.0057
/ +0.0025 / +0.0036 in the fine-tune's favour); it is simply not enough to reach zero. Anti-collapse:
4,031–7,291 distinct interviews, Pep divergence 0.482–0.636 against the tree's 0.664, no channel
collapse (both channels present at every k).

**Gates before any number was trusted:** canonical m=0 snap `0.15686538118354684` exact;
`greedy_static_policyB_combined` rescored to `0.25561016113602253` exactly; pool identity; E1's banked
per-user vector re-asserted equal to the tree's realised interview; `branch_of_per_user == branch_of`;
leak check 0/10,000; both checkpoints asserted to carry this pool and this descriptor standardisation.

---

## WHAT THIS SETTLES

1. **The falsifier fired at every k, with the intervals excluding zero from below.** Trained on
   evidence-seeded states, deviations on fresh seeded states still cost — **more** than they cost the
   untrained belief-only policy. **This action space does not reward state-conditioning beyond the
   tree's, even when taught.** That is the strongest honest closure available and it is the finding.
2. **The training worked; the behaviour it produced does not pay.** Spanning gate passed to every
   digit, seeded top-1 0→0.66, inertness gone (1% → 90% of users adapting). This is not a failed
   optimisation being read as a failed idea.
3. **Redundancy avoidance is real and interval-backed for the first time** — roughly half the tree's
   re-ask rate at every k — and it is bought with a channel swing and an answer-rate collapse from
   0.579 to 0.379 that costs far more than the avoidance saves.
4. **The corrected census is itself a contribution to the record**: 209 off-manifold states with
   measured action labels existed and had never trained anything, and 77 of the 125 label questions
   they carry are questions the tree never asks.
5. **The upgrade is not indicated.** The weakest teacher covers most of the training mass, but the
   direction is wrong, and Part I.2's own rule ("upgrade only if the direction is right and the
   magnitude shy") therefore says stop.

**Read with Part I.0's expectation-setting:** even full success was only ever worth a small score
movement, and the prize was the capability demonstration. The capability is now demonstrated in the
narrow sense that matters — **the policy can be taught to condition on evidence it did not choose,
and it does** — and the demonstration comes with a measured price tag rather than a benefit.

**Nothing about test.** No test user was built, loaded or touched.

---

## Cost — a property of this box

| step | measured |
|---|---|
| build the 210-state training set (19 prefix folds + 72 extra states, 500k latents) | **2.9 min** |
| fine-tune, main arm (4 epochs × 2,000 steps, with a full rollout + agreement pass per epoch) | **21.6 min** |
| fine-tune, shuffled control | 21.9 min |
| evaluation, 3 k × 7 arms on a fresh draw, all CIs | **17.3 min** |
| **total** | **≈ 1.1 h** |

The Part I.2 estimate was 3–6 h. It came in at 1.1 h.

## Artifacts

| what | where |
|---|---|
| the 210-state training set + composition | `experiments/battery/p2_seeded/p2_seeded_states.{json,npz}` |
| fine-tune histories (per-epoch gate, seeded top-1 in both encodings) | `experiments/battery/p2_seeded/p2_{main,shuffle}_history.json` |
| checkpoints (gate-passing; `_last` is the diagnostic copy) | `experiments/battery/p2_seeded/p2_{main,shuffle}.pt` |
| evaluation: all k, all arms, all CIs, adaptation + redundancy | `experiments/battery/p2_seeded/p2_eval.json` (+ `p2_eval_k{2,4,8}_per_user.npz`) |
| logs | `experiments/battery/p2_build.log`, `p2_seeded/train_*.log`, `eval3.log` |
| code | `src/instrument/policy_p2.py` |

**Reproduce:**
`python src/instrument/policy_p2.py --mode build --threads 6 --i_have_author_goahead`, then
`--mode train --arm main --epochs 4 --steps 2000 --lr 2e-4` (and `--arm shuffle`), then
`--mode eval --ks 2,4,8 --draw_seed 424242 --ckpts p2ft=...,p2shuf=...`
