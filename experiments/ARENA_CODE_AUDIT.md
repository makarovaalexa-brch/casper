# ARENA CODE AUDIT (adversarial, pre-results) — 2026-07-10

Scope: `scripts/arena_core.py`, `arena_policies.py`, `arena_eval.py`, `arena_validate.py`
(+ interfaces of `i25_fold_v3.py`, `i25_fold_v3_sampler.py`, `i25_lib.py`).
Contract: `DESIGN_SHEET_POLICY_ARENA.md` (signed 2026-07-09). Discipline: E1–E7
(`STATE_2026-07-08.md`). Auditor had no prior project context (deliberate). No training, no API
calls; probes limited to reading code, the cache directory, and `.cache/arena/run.log`.

Evidence anchors used below:
- `.cache/arena/run.log`: `universe nQ=1530 (200 concept / 30 attr / 500 entity / 800 item)`;
  `[b2] candidate search cap=1030: 730 non-item + 300 popular items`.
- `.cache/arena/b2_seq_T24_K50_n50.json`: contains **10 picks** (of 24) — an interrupted build.
- `.cache/arena/`: `answers_devtest160.npz`, `answers_devtest200.npz`, `answers_devtest40.npz`,
  `answers_devval30/80/100.npz`, `answers_train60/1000.npz` — multiple cohort configs have run.

---

## FINDINGS (ranked by severity)

### 1. CRITICAL — b4 "deployable myopic greedy" scores candidates against the EVAL user's held-out likes (unlabelled privilege; E3/E7 violation)
**Where:** `arena_policies.py:280-282` (`start` caches `self.held` per eval user),
`:294` (`base = ndcg_at_k(..., held, ...)`), `:305-309` (candidate NDCG@K computed with
`[held]*len(tl)`, then `exp_gain = pa * (nd - base)`, argmax).
Mislabel: `arena_eval.py:125` E7 row says b4 `privileged? no`; header text `:102-112` calls b4 blind.

**What's wrong:** the *tokens* are honestly blind (belief-forecast value, model answerability), but
the *objective* is the realized test metric on this user's held-out likes. Each turn b4 picks the
question whose hypothetical belief-shift maximizes NDCG **on the eval targets** — a per-turn peek at
the ruler. A cold-belief b4 will preferentially ask about regions containing the user's held likes
with zero observed evidence. This is exactly the class of asymmetry E7 was written for, and it is
the same shape as the STATIC_CONTAMINATION leak (fit-to-the-ruler), now on the adaptive side.

**Which comparison it poisons:** (a) the pre-registered "b4 vs b2 computation-suffices" verdict —
a b4 WIN would be spurious; (b) the rule "if b4 wins, every learned policy must beat b4" — raises
the bar for A–D with a privileged comparator, potentially killing a legitimate adaptive win;
(c) mechanism readouts for b4 (its channel trajectory reflects target-chasing, not adaptivity).

**Minimal fix (pick one, pre-register which):**
- Replace the objective with a blind surrogate: expected gain in the smooth ranking utility
  `J(z) = tau*logsumexp(decode(z)/tau)` (the machinery already exists in `AskGradient._direction`),
  i.e. `exp_gain = pa * (J(z') - J(z))`. Fully deployable, same fold, same expectation structure.
- OR relabel b4 as a CONTEXT arm (myopic-with-oracle-objective), move it out of the results rows,
  and add the blind-surrogate b4' as the deployable baseline. Do NOT keep the current b4 in the
  E7 table as `privileged? no`.
Note: do not "fix" it by scoring against belief-predicted likes — that is the circular-belief
failure mode in the ledger.

### 2. CRITICAL — `build_b2` accepts a PARTIAL cached sequence as complete (b2 can silently be a 10-question static)
**Where:** `arena_policies.py:206-210` (unconditional load if file exists, no length check) and
`:267` (`json.dump` INSIDE the greedy loop — incremental checkpointing).
**Evidence:** `.cache/arena/b2_seq_T24_K50_n50.json` currently holds 10/24 picks (interrupted run).

**What's wrong:** the checkpoint-per-step is good discipline, but the loader treats any existing
file as done. If `eval` runs against today's cache, b2 = 10 questions; `StaticSeq.pick_static`
returns None from turn 11, the belief freezes for turns 11–24, and **every arm beats "the strongest
fair static"** at the pre-registered T=24 endpoint. The b2-anchored variants (`B2Anchored`), b3, and
the Golbandi tail all inherit the truncated sequence, so the corruption is correlated across the
whole ladder and would read as a clean adaptivity win.

**Minimal fix:** on load, `if len(blob["seq"]) < Tmax: resume greedy from the partial prefix`
(recompute tokens for the cached picks, continue the loop); assert `len(seq) == Tmax` before
returning. Also assert in `stage_eval` that `len(b2_seq) >= Tmax`.

### 3. HIGH — answer-memo `.npz` cache keys don't encode the cohort config or universe; stale caches inject a DIFFERENT known/held split (held-out leakage into beliefs)
**Where:** `arena_core.py:279-311` (`prefill_answers`, path = `answers_{tag}.npz`, tag =
`train{n}` / `devval{n}` / `devtest{n}` from `arena_eval.py:56-58,179-181`), interacting with
`arena_core.py:315-329` (`make_cohorts`).

**What's wrong:** a user's known/held split is NOT a function of the uid alone — `make_cohorts`
draws `load_train_profiles(D, total+2000, seed)` where `total = n_train+n_devval+n_devtest`, and
`prep_users` consumes ONE shared rng across users, so changing ANY cohort size changes the sampled
user set and every user's known/held split. But the cache tag encodes only the cohort's own size.
The cache dir shows runs with `devtest160`, `devtest200`, `devtest40`, `devval30/80/100` — all of
which share tag `train1000` for the train cohort. Whichever config ran second loaded train answers
computed under the OTHER config's splits. A stale answer can then contain a real-rating explicit
token (`FID_DATA`, value = old `cr[j]`) for an item that is NOW in the user's held set → the
held-out target's true rating flows into b2/Golbandi/scorer-A construction beliefs, and (for stale
devval/devtest tags) into evaluation itself. Silent — `_answer_raw` cache-hits never recompute.

**Which comparison it poisons:** primarily the constructions (b2, D, A labels) — differentially,
since each is built on a different train slice; secondarily any eval run whose dev npz predates a
config change. It also breaks the "same cell, same answer, all arms" pairing guarantee across
build/eval invocations with different flags.

**Minimal fix:** put a config hash in the tag — `sha1(SEED, n_train, n_devval, n_devtest, n_item,
nQ)` — AND store per-user `sorted(known.keys())` hashes in the npz, verified against the current
ctx on load (mismatch = recompute). Cheap and decisive.

### 4. HIGH — construction-budget asymmetry: the "strongest fair static" is built on 50 users while scorer-A gets 1000 users / 3500 labels (E7 asymmetry, tilts toward an adaptivity win)
**Where:** `arena_eval.py:437-441` defaults: `--n_b2 50`, `--n_gol 200`, `--n_scorer 1000`,
`--n_labels 3500`; `build_b2` at `arena_policies.py:201-269`.

**What's wrong:** b2 greedily argmaxes over ~1030 candidates × 24 steps on a mean over **50 users'
fixed answer realizations**. That is a winner's-curse regime: per-step gains of ~0.003–0.03
(run.log) selected from 1030 candidates on n=50 will partly be selection noise that does not
transfer to DEV-TEST — b2 arrives at the arena weaker than "the strongest fair static ever
constructible" (sheet §3), while A trains on 20x the users. Any adaptive win over b2 is then
confounded with construction-sample asymmetry — the exact shape of the a3/a4 value-model confound
the author caught before (STATE §Author-Q&A-2). The E7 table *documents* the cohorts but the sheet
sells b2 as population-scale.

**Minimal fix:** raise `n_b2` until b2's DEV-VAL score plateaus (report the b2-vs-n curve — 50 vs
200 vs 500; the greedy is the long pole, so cap candidates harder before capping users), or at
minimum pre-register "b2 built on n=50" as a stated limitation next to every b2 contrast. A cheap
robustness check: rebuild b2 on a disjoint 50 and report seq overlap + DEV-VAL delta (if the two
b2s differ materially, n=50 is under the noise floor and the arena is not fair as configured).

### 5. MEDIUM — b3 (static+skip) exhausts its 24-question list and silently forfeits remaining turns (understates the cheap-adaptivity assassin)
**Where:** `arena_policies.py:126-132` (skip refund loop; `qi is None -> break`), b2_seq length =
Tmax by construction (`build_b2`), `StaticSeq.pick_static` `:170-174`.

**What's wrong:** with refusals refunded, b3 needs more than 24 questions to fill 24 answered
turns (universe refusal rates: niche concepts ~47%, entities ~18–34% per the sheet). Once the
24-item seq is spent, `pick_static` returns None and the user's remaining turns pass with the
belief frozen — b3 is quietly capped at b2's answered-count plus whatever refunds landed early.
b3 exists to kill "adaptivity = mere refusal-avoidance"; weakening it makes the adaptive arms'
margin look more meaningful than it is.

**Minimal fix:** build the static longer than Tmax (e.g. greedy to 40–48; picks beyond 24 only
serve b3), or give b3 a pre-registered fallback order (remaining universe by population
answerability) when the seq is exhausted.

### 6. MEDIUM — the coded universe is NOT the pre-registered universe (200 concepts vs the sheet's 1,128)
**Where:** `arena_core.py:119-139` (`_build_universe` = sampler vocabulary);
run.log: `nQ=1530 (200 concept / 30 attr / 500 entity / 800 item)`. Sheet §2: "ALL judged questions
(1,128 concepts + 500 attributes + 800 items). No top-N pools."

**What's wrong:** the sampler's concept vocabulary (`item_tag` columns) is 200, not the judged
1,128; attributes are 30 (decades+genres), entities 500. Whatever the right universe is, the signed
contract and the code disagree BEFORE results exist — and this program's history says channel
composition (concept-vs-item balance) decides verdicts. An 800-item block inside nQ=1530 is also a
much more item-heavy bank than the sheet implies; combined with finding 7 it shapes every arm's
opening book.

**Minimal fix:** amend the design sheet (author-signed) to the actual counts with a one-line
rationale, or rebuild the universe to spec. Do not let the first results memo be the place the
discrepancy surfaces.

### 7. MEDIUM — candidate-cap asymmetries across arms are individually documented but not symmetrized (class comparisons confounded)
**Where:** b2 search cap 1030 (drops 500 tail items; `arena_policies.py:221-231`); b4 `M=80`
(`arena_eval.py:443`); A/B/C `cand_M=100` (`:442`); D's `cand_pool` = top-250 by TRAIN answer-rate
(`arena_policies.py:499-503`); clairvoyant `M=200`; ttab `M=80`.

**What's wrong:** at cold start `top_answerable` ranks by channel prior (item .95 > attr .82 >
concept .74 > entity .55) with stable ties → every belief-based arm's turn-1 candidate set is
exactly the 80–100 most popular items, while b2 searched a different (1030-wide, concept-heavy)
pool and D a third (answer-rate-ranked) pool. None of these is a leak, but arms differ in TWO ways
(policy AND reachable set), which is the tie-by-construction sin the ledger names: class deltas
(A vs C vs b4) are partly cap artifacts. The B2Anchored wrapper mitigates vs b2 only.

**Minimal fix:** one shared `cand_M` for b4/A/B/C (single flag), and a printed line in the E7
table listing each arm's reachable candidate set per turn. Report turn-1 candidate-set overlap
across arms as a symmetry check.

### 8. LOW — cached answers are float32, lazy answers float64: results depend on whether the prefill npz existed
**Where:** `arena_core.py:288-289` (load: `float(val[a,qi])` from float32) vs `:234-242` (lazy:
float64 `CENTERED_FOLD` values, e.g. 1/3). ~1e-8 token-value differences through the fold; no
directional bias (same cache state serves all arms), but it breaks bit-exact reproducibility and
run.log-vs-rerun diffs. Fix: store `val` as float64, or round-trip lazy values through float32.

### 9. LOW — every "blind" policy receives the user's full private `ctx` (ratings, trait) in `pick(...)`
**Where:** `arena_policies.py:124` (runner call), `:155` (interface). No current deployable arm
reads it (verified: b4/A/B/C never touch `ctx`; D only via `_branch_static` on already-asked
questions = observed dialogue) — but the interface hands the answerer's internals to the agent,
one refactor away from a silent leak that no gate would catch. Fix: pass an observed-dialogue view
(asked/ansf/tokens/natives/z) and route `ctx` only to the world and the LABELLED context arms.

### 10. LOW — context arms run on `dt[:50]` but share the results table with full-cohort rows
**Where:** `arena_eval.py:236-238` (`ctx_n=50`), `_write_results` `:268-283`. Different n, unpaired
against the other rows; fine for labelled context, but annotate `(n=50)` in the table so a reader
never computes a delta against them. Also `Clairvoyant` caps at M=200 most-answerable answered
candidates — the "ceiling" is an approximation; keep the existing "approximate" docstring wording
in the md output too.

### 11. LOW — cohort shortfall is silent
**Where:** `arena_core.py:319-328`. If `prep_users` filters below `total`, devtest is quietly
short. Add `assert len(recs) >= total`.

### 12. LOW — constructed-artifact caches don't encode enough config
**Where:** `b2_seq_T24_K50_n{n}.json` (no `cand_cap`, no `n_item`, no fold-ckpt hash),
`golbandi_d{d}_n{n}.json` (no `cand_cap`, no b2 tail id), `scorerA_n{n}.joblib` (no `n_samples`,
no `M2`). Consequence: change `--n_labels` or retrain the fold checkpoint and the eval silently
reuses artifacts built under the old world — an E7 violation (arms must share one fold) that no
symmetry table would reveal. Fix: fold-state hash + the relevant flags into every cache filename
(one helper function).

---

## WHAT IS RIGHT (so it doesn't get broken while fixing the above)
- **Pairing/determinism (angle E/G): sound.** Answers are seeded per `(uid, ch, key)` via sha1
  (`arena_core.py:228`), independent of arm, order, and cache state — all arms face the identical
  user, and the memo is a pure speedup. Traits are uid-seeded. All deploy-time policies are
  RNG-free. Paired bootstrap on per-user deltas is valid.
- **E1: sound.** Same cohort every arm; refusal = consumed turn, empty token list, belief
  unchanged, user retained (`run_policy` + `tokens_for`); no drop paths; `nanmean`/None guards are
  dead safety nets (held is non-empty by `prep_users` construction).
- **Metric plumbing (angle F): sound.** NDCG@50/@10 over the full catalogue, profile (known-half)
  masked, held likes as relevance; the answerer never reads held ratings (`region_value` uses `cr`
  = known half only; held item probes get the EASE population mean); turn indexing (cold at t=0,
  endpoint at t=24) is correct; greedy b2's prefix-consistency makes T=8/16 endpoints legitimate
  without per-budget rebuilds.
- **A/B/C deploy-time blindness verified** feature-by-feature (`_feat`, `AskGradient.pick`,
  `CATRouter.pick`): belief + population priors only.
- Pre-registered verdicts and the E7 table are written to the md BEFORE results; the 173 are
  untouched (`load_train_profiles` samples trU only).

---

## VERDICTS (is the comparison trustworthy as coded?)
- **Harness (runner, world, metric, pairing): PASS** (findings 3, 8, 11 are cache/hygiene, fixable
  without design change).
- **Statics b1/b2/b3: BLOCK** — finding 2 (partial b2 cache is live in `.cache/arena/` right now),
  finding 4 (n=50 construction), finding 5 (b3 exhaustion). No verdict involving b2 is meaningful
  until these land.
- **b4 myopic greedy: BLOCK** — finding 1. As coded it is a privileged arm mislabelled deployable;
  the two pre-registered b4 contrasts are invalid until the objective is blind or the arm is
  relabelled context.
- **Adaptive A (scorer): PASS with conditions** — deploy-blind as coded; its verdict is blocked
  only by its comparators (1, 2, 4) and label-cache keying (3, 12).
- **Adaptive B (gradient) / C (CAT): PASS with conditions** — blind and deterministic; symmetrize
  candidate caps (7) before citing class-vs-class deltas.
- **Adaptive D (Golbandi): PASS with conditions** — branches on observed answers only; inherits
  the b2 tail, so blocked transitively by finding 2.
- **Context arms (ttab, clairvoyant): PASS** — privileged, labelled, never cited; annotate n.

**Overall: BLOCK the eval stage until findings 1–4 are fixed** (1 and 2 are each individually
sufficient to manufacture a false "adaptivity wins" headline; 3 and 4 bias the same direction).
Findings 5–7 should land before the DEV verdict is treated as directional; 8–12 before the
headline run.
