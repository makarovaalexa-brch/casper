# A24 — the diagnostics re-run on the re-selected epochs: the channel ports, but by a DIFFERENT MECHANISM (2026-09-12)

Closes the A22/A23 line. Design `docs/design/DS_A22_CONCEPT_MONITOR_AND_ASKSET.md`; prior results
`docs/results/A22_CONCEPT_MONITOR_CORRECTED.md` (whose Addenda 1–2 this supersedes as *dataset*
statements) and `docs/results/A17_GOODREADS_PORT.md` (retracted).

A23 showed every A22 diagnostic had run on `cvg_ep12` (fold +0.0014) and `cv3_ep04` (+0.0006) — both
chosen by the bent monitor, both weak. A24 repeats them on `cvg_ep01` (+0.0028/+0.0035) and `cv3_ep11`
(+0.0024/+0.0020), on seeds **777/1777/2777**, independent of the 4242/5242 set that made the selection.
Runner `src/instrument/run_a24_reselected_diagnostics.sh`.

## 1. The channel ports — confirmed, and the earlier "does not port" is dead

Mean paired fold effect over m in (1,4,8), all 12 epochs, no selection: **ML-25M +0.0015, Goodreads
+0.0016.** At the out-of-sample-selected epoch, Goodreads is ahead (+0.0035 vs +0.0020).

## 2. ⭐ The MECHANISM differs, and this survives re-selection

Fold effect at m1 / m4 / m8, with and without the refusal tokens:

| | refusals fed | answers only | share carried by answers |
|---|---|---|---|
| **Goodreads ep01** | +0.0036 / +0.0027 / **+0.0045** | −0.0001 / −0.0015 / **+0.0005** | **11%** |
| **ML-25M ep11** | +0.0028 / +0.0024 / **+0.0036** | +0.0016 / +0.0013 / **+0.0024** | **67%** |

This is the one Addendum-1 finding that **survives** the re-selection, and it is now a positive result
rather than a symptom of failure. On ML-25M the concept channel works through what the user *says*; on
Goodreads it works through what the user *cannot* say. With 73% of shelf questions refused, the refusal
pattern localises a reader within the catalogue more sharply than the answers do.

Both are real channels. They are simply not the same channel.

## 3. Saturation is real but far out, not immediate

Equal-ANSWER curve on Goodreads ep01 (broad policy, 0.273 answers per ask):

| m asked | answers | NDCG | fold |
|---|---|---|---|
| 0 | 0.0 | 0.1846 | — |
| 8 | 2.2 | 0.1874 | +0.0029 |
| **16** | **4.4** | **0.1880** | **+0.0034** |
| 22 | 6.0 | 0.1860 | +0.0015 |
| 32 | 8.7 | 0.1846 | +0.0001 |

It rises to a peak near **16 questions** and decays after. On the weak ep12 this same curve read
−0.0001 at m22 and was the basis for declaring the answer-rate story refuted and the channel dead —
**that reading is withdrawn.** The decay beyond the peak is real and worth reporting, but it is
saturation at a sensible horizon, not absence of signal.

## 4. The member-count hypothesis stays refuted — now robustly

Ask-set arm on the re-selected Goodreads epoch, fold at m1 / m4 / m8:

| ask set | fold |
|---|---|
| broad (shipped) | +0.0036 / +0.0027 / **+0.0045** |
| topN:61 | +0.0026 / +0.0015 / **−0.0012** |
| band:61–200 | +0.0042 / +0.0031 / +0.0017 |

The shipped member-weighted sampler wins, and it wins by more at the good epoch than at the bad one.
Restricting the ask set to broader concepts is worse at every depth past m1. This now holds on both a
weak and a strong checkpoint, so it is a property of the bank, not of a checkpoint: **`ask_pool`'s
`clip(member_count, 30, 1000)` (`concept_curriculum_v2.py:119`) is doing real work.**

The underlying measurements stand unchanged: Goodreads concepts are narrower (median 80 members vs 291,
A19 having matched per-*item* density and never per-*concept* size), broad-policy refusal is 0.727 vs
0.252, and forcing the answer rate up costs more than it buys.

## Where the whole line lands

1. The A17 negative result was an artifact of eval-time member-drop. **Retracted.**
2. A22's "does not port" was an artifact of comparing two bent-ruler-selected checkpoints.
   **Retracted.**
3. The concept channel **ports to Goodreads at full strength**, +0.0016 vs +0.0015.
4. It ports **by a different mechanism**: refusals carry 89% of the Goodreads lift and 33% of ML-25M's.
5. Concept narrowness is real and is **not** the constraint to relieve; the shipped ask policy is
   already the right one.

## Still open

- **Paper B's +0.01430 portability.** Untested. Its harness (`strategy_channel_suite.py:459-461`) has
  no member-drop so it is unaffected by the monitor defect, but the answers-per-question ledger for
  concepts vs items on the deployed Goodreads harness has never been run. That is the real portability
  question and it is a larger piece of work than anything above.
- **The in-training monitor is still bent.** `a22`/`a23` are standalone; `concept_train_v2`'s own
  `build_val_interviews` still applies member-drop at eval, still redraws targets per level, and still
  takes `users[-1500:]`. Any future run selects on the bent ruler unless it is fixed.
- **cv3_ep04 remains the certified ML-25M concept checkpoint** and is one of its weakest epochs on the
  corrected ruler. Whether to re-certify is an author decision — it would move Paper B's artifact set.

## ⚠ CORRECTION (same day) — "bent-ruler selection" overstated the criterion's fault

Checking the shipped SEL against the corrected fold effect, averaged over BOTH seed sets:

| | Pearson | Spearman |
|---|---|---|
| ML-25M | +0.752 | +0.664 |
| Goodreads | +0.745 | +0.685 |

**The shipped criterion was a mediocre ranker, not a broken one**, and the claim that both certified
checkpoints were victims of it is wrong. The two datasets fail differently:

- **Goodreads — this was a HANDLING error, not a criterion error.** The shipped SEL's best epoch is
  **ep10**, and ep10 *is* the genuinely best epoch (+0.0032). The criterion got it right. A17 evaluated
  **ep12** because it took the run's last epoch, and even recorded "Best SEL 0.1830 at ep10" without
  acting on it. ep12 is +0.0019, so roughly 40% of the channel's benefit was discarded by scoring the
  wrong file. **Every A22/A24 diagnostic inherited that choice.**
- **ML-25M — this one is the criterion.** It picked ep4 (+0.0016); the best is ep12 (+0.0029).

Also correcting a number used earlier in this line: `cv3_ep04` was described as "one of ML-25M's weakest
epochs, +0.0006". That came from a **single** seed set. Averaged over both it is **+0.0016** — middling,
6th of 12. The epoch spread is small enough that one seed set reorders the ranking, which is the same
caution the A23 method lesson already states.

| | criterion picked | its value | actual best | best value |
|---|---|---|---|---|
| ML-25M | ep4 | +0.0016 | ep12 | +0.0029 |
| Goodreads | **ep10** | **+0.0032** | **ep10** | **+0.0032** |

**Nothing in the headline changes**: the channel ports (+0.0015 ML-25M / +0.0016 Goodreads averaged
over all 12 epochs), the mechanism difference stands, and the member-count fix stays refuted. What
changes is the explanation of how we came to run on a weak Goodreads checkpoint — it was not the
selection rule, it was not following it.

## ⭐ CERTIFIED GOODREADS CONCEPT CHECKPOINT: `cvg_ep10`

Author ruling 2026-09-12: **ML-25M is left as-is** (`cv3_ep04` stays certified — not enough time to
redo Paper B's artifact set). Goodreads proceeds on **`cvg_ep10`**, which is what the project's own
pre-registered selection rule picks when it is actually applied. No post-hoc selection is involved.

### Certified numbers for `cvg_ep10` (seeds 777/1777/2777, corrected protocol)

| m | NDCG@10 | fold | answers only | scrambled |
|---|---|---|---|---|
| 0 | 0.1846 | — | — | — |
| 1 | 0.1882 | +0.0036 | −0.0005 | **+0.0040** |
| 4 | 0.1875 | +0.0029 | −0.0013 | +0.0020 |
| 8 | 0.1886 | **+0.0040** | −0.0002 | −0.0011 |

Equal-answer curve: m8 +0.0020, **m16 +0.0024**, m22 +0.0003, m32 −0.0028 — same peak near 16 asks.

Mean fold over m in (1,4,8): **Goodreads `cvg_ep10` +0.0035** against ML-25M's certified
`cv3_ep08` **+0.0015**. On the corrected ruler the Goodreads concept channel is the *stronger* of the
two certified pairs.

**⚠ Caveat on the control, stated rather than buried.** G3 fires at m8 (+0.0040 → −0.0011) but **not at
m1**, where a scrambled concept does as well as the real one (+0.0040 vs +0.0036). With 73% refusal a
single question is almost always a refusal, so at depth 1 the model is learning *"this is a light
reader"* — a generic sparsity cue — not anything about taste. Concept-identity sensitivity emerges only
with depth. Any claim that one Goodreads concept question carries preference information is therefore
**not supported**; the claim that eight of them do, is.

### Correction to the certified ML-25M artifact

`cv3_ep04` was repeatedly described in this line as "the certified ML-25M concept checkpoint". It is
not. The certified artifact is **`models/i26c_ep31/concept_cv3_ep08.pt`** (verified byte-identical to
`.cache/instrument/cv3_ep08.pt`); ep04 was merely the epoch whose validation curve A17 quoted. On the
corrected ruler ep08 scores **+0.0015**, exactly the all-epoch mean — average, not weak. **Nothing on
the ML-25M side needs re-certifying**, which is consistent with the author's ruling to leave it.
