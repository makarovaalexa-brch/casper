r"""adaptive_harness.py -- the SEQUENTIAL interview loop: a selector sees the answers so far and
chooses the next question, per user, per turn.

WHY THIS FILE EXISTS (Paper B, Act 1; harness gap named in PAPER_B_DESIGN addendum 3). Until now every
arm in `run_interview_table.py` was a STATIC GLOBAL ORDER: `interview_strategies.ask()` walks a fixed
item list and the only per-user variation is which of those items the user happens to have rated.
`golbandi_leaf` is a CONSUMER of answers, not a selector. So the harness could not express the object
Act 1 has to measure -- "does conditioning the NEXT question on the PREVIOUS answers buy anything over
the strongest static order?" -- because it had no way to ask a different second question of two users
who answered the first one differently.

WHAT IS AND IS NOT NEW HERE. Only the ASK side is new. The answer rule, the answer->input contracts,
the Most-Popular floor, the fixed arm-N candidate pool and the evaluate() cursor discipline are all
IMPORTED from run_interview_table (`score_arm`, `build_recommenders`, `mostpop_row`), not re-written.
That was a deliberate call: the regression assert below can only prove the ask side agrees, so the
score side must be literally the same code or the two paths will drift where no test is looking.

THE REGRESSION CONTRACT (requirement (a)). A static arm driven through the sequential loop must
produce a BYTE-IDENTICAL (asked, answered) to `interview_strategies.ask()`:

    python src/baselines/adaptive_harness.py --regression

asserts exact list equality for all six arms x k in {1,2,4,8,16}, and prints the entropy0/k=8 cell
by name because that is the cell the overseer specified. If it fails, no adaptive number produced by
this file means anything, so it is an assert and not a warning.

THE ANSWER RULE (requirement (b)) is `interview_strategies.level_lookup(g_te_tr)`: the test user's own
half-star level for the asked item, from their fold-in-side ratings with the held-out targets already
removed. An asked-but-unrated item BURNS the question and yields no answer -- the literature's
protocol, and identical to the static path. No simulator, no imputation, no LLM. A selector NEVER sees
`lvl`; it sees only what the user actually answered.

SELECTOR INTERFACE (requirement, and the API Paper B ships):

    class MySelector(Selector):
        name = "my_selector"
        def reset(self, ctx):                 # ctx: n_users, n_items, q_max, D, glob, bank, log
            ...                               # called once per (selector, budget) run
        def select(self, turn, users, asked, answered) -> np.ndarray[int64]  (len == len(users))
            # asked[u]    : list of item ids already asked of user u, in order
            # answered[u] : list of (item_id, half_star_level) the user could answer, in order
            # MUST return an item NOT already in asked[u]; the harness asserts this.

`PerUserSelector` gives a scalar `select_one(turn, u, asked_u, answered_u)` for selectors that cannot
vectorise. Selectors may be stateful per user (keep your own dict keyed on u); the harness passes the
full histories every turn anyway, so statelessness is the easier and safer default.

BATCHING WITHOUT SUBSAMPLING (requirement (c)). The loop hands the selector ALL users of the cohort at
once by default (`--sel_batch 0`); users are never sampled, capped or dropped. Batching is the
selector's business: `bandit_selectors` and `eddi_selector` group users by identical interview STATE
and compute each state's acquisition once, which is exact (the posterior is a function of the state)
and is what makes a 10k-user x 18,359-candidate exact sweep affordable. `--smoke_users N` exists ONLY
for the code-path smoke, refuses to write a citable filename, and stamps SMOKE_NOT_CITABLE into the
JSON.

  python src/baselines/adaptive_harness.py --regression
  python src/baselines/adaptive_harness.py --selectors entropy0-static,glm_ucb --budgets 1,2 \
      --only ours --arch i26 --snapshot .cache/instrument/t2i26c_ep31.pt --no_floor --smoke_users 200
"""
import os
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import interview_strategies as ST
from arm_n import load_arm_n
from arm_n_tower import SNAP_DEFAULT
from golbandi_leaf import MAX_DEPTH
from run_interview_table import (OUT, RECS, build_recommenders, score_arm, mostpop_row, logln)

STATIC_SUFFIX = "-static"


# ===================================================================== selector interface
class Selector:
    """Base class. Subclass and implement `select`; see the module header for the contract."""
    name = "selector"

    def reset(self, ctx):
        self.ctx = ctx

    def select(self, turn, users, asked, answered):
        raise NotImplementedError

    def stats(self):
        """Optional per-run diagnostics, merged into the output JSON cell."""
        return {}


class PerUserSelector(Selector):
    """For selectors with no vectorised form: implement `select_one` and pay the Python loop."""

    def select_one(self, turn, u, asked_u, answered_u):
        raise NotImplementedError

    def select(self, turn, users, asked, answered):
        return np.asarray([self.select_one(turn, int(u), asked[u], answered[u]) for u in users],
                          dtype=np.int64)


class StaticArmSelector(Selector):
    """The six lit-anchored static arms, expressed as selectors.

    This is the regression harness for the whole sequential path: it must reproduce
    `interview_strategies.ask()` EXACTLY, so the order construction below is a verbatim copy of that
    function's branch (same SEED, same per-user RNG stream, same `q_max*4` draw for random_catalog --
    which is why q_max is a constructor argument: the random_catalog order depends on it).

    A static arm ignores `answered` entirely. That is the null hypothesis of Act 1, made explicit in
    the type system: the adaptive selectors have exactly one extra argument available to them."""

    def __init__(self, arm, glob, bank, q_max):
        assert arm in ST.ARMS, f"unknown arm {arm}"
        self.arm, self.glob, self.bank, self.q_max = arm, glob, bank, int(q_max)
        self.name = arm + STATIC_SUFFIX
        self._orders = None

    def _order(self, u, n_items):
        if self.arm == "random_catalog":
            rng = np.random.default_rng(ST.SEED * 1000 + u)
            return rng.choice(n_items, size=min(self.q_max * 4, n_items), replace=False)
        if self.arm == "random_bank":
            rng = np.random.default_rng(ST.SEED * 1000 + u)
            return rng.permutation(self.bank)
        return self.glob[self.arm]

    def reset(self, ctx):
        super().reset(ctx)
        n, ni = ctx["n_users"], ctx["n_items"]
        if self.arm in ("random_catalog", "random_bank"):
            # Materialised once per run: rebuilding the per-user RNG stream every turn would be both
            # slower and a place for the two paths to disagree.
            self._orders = [self._order(u, ni) for u in range(n)]
        else:
            self._orders = None
        probe = self._orders[0] if self._orders is not None else self.glob[self.arm]
        assert len(probe) >= self.q_max, (
            f"{self.arm}: order length {len(probe)} < budget {self.q_max}; ST.ask would return a "
            f"SHORT interview here and the sequential path cannot reproduce it")

    def select(self, turn, users, asked, answered):
        if self._orders is None:
            return np.full(len(users), int(self.glob[self.arm][turn]), dtype=np.int64)
        return np.asarray([int(self._orders[int(u)][turn]) for u in users], dtype=np.int64)


# ===================================================================== the sequential loop
def run_interview(selector, k, n_users, n_items, lvl, ctx=None, sel_batch=0, log=logln,
                  collect_dispersion=True):
    """Ask `k` questions sequentially, re-invoking `selector` after every answer.

    Returns (asked, answered, info):
      asked[u]    list of k item ids, in ask order
      answered[u] list of (item_id, level) for the subset the user could actually answer
      info        per-turn wall time and the number of DISTINCT ask-prefixes (the dispersion
                  diagnostic Act 2 wants; free to collect here, so we do)

    NO USER IS EVER DROPPED, SAMPLED OR CAPPED: the loop covers range(n_users) every turn.
    """
    asked = [[] for _ in range(n_users)]
    answered = [[] for _ in range(n_users)]
    base_ctx = {"n_users": n_users, "n_items": n_items, "q_max": int(k), "log": log}
    base_ctx.update(ctx or {})
    selector.reset(base_ctx)

    all_users = np.arange(n_users, dtype=np.int64)
    step = n_users if sel_batch in (0, None) else int(sel_batch)
    info = {"turn_seconds": [], "distinct_prefixes": [], "answer_rate": []}
    for t in range(k):
        t0 = time.time()
        for s in range(0, n_users, step):
            users = all_users[s:s + step]
            q = np.asarray(selector.select(t, users, asked, answered), dtype=np.int64)
            assert q.shape == (len(users),), f"selector {selector.name} returned {q.shape}"
            for u, i in zip(users.tolist(), q.tolist()):
                assert 0 <= i < n_items, f"selector {selector.name} returned item {i} out of range"
                assert i not in asked[u], (
                    f"selector {selector.name} re-asked item {i} to user {u} at turn {t}; a repeat "
                    f"burns a question for zero information and silently shortens the interview")
                asked[u].append(i)
                d = lvl[u]
                if i in d:                                   # identical test to interview_strategies.ask
                    answered[u].append((i, d[i]))
        dt = time.time() - t0
        info["turn_seconds"].append(round(dt, 2))
        info["answer_rate"].append(round(float(np.mean([len(a) for a in answered])), 4))
        if collect_dispersion:
            info["distinct_prefixes"].append(len({tuple(a) for a in asked}))
        log(f"[adaptive] {selector.name:16s} turn {t + 1}/{k}: {dt:.1f}s  "
            f"answered so far {info['answer_rate'][-1]:.2f}  "
            f"distinct ask-prefixes {info['distinct_prefixes'][-1] if collect_dispersion else '-'}")
    info.update(selector.stats())
    return asked, answered, info


def common_asked(asked):
    """asked[0] if EVERY user was asked the same items in the same order, else None.

    This is exactly run_interview_table's `asked[0] if arm in GLOBAL_ARMS else None` rule, generalised
    from the arm's name to the realised ask-lists: it is what lets GolbandiLeaf build its partition
    once. For a static global arm it returns asked[0] (same as today); for an adaptive selector it
    returns asked[0] at k=1 (everyone gets the same opener) and None once the policy branches."""
    first = asked[0]
    for a in asked:
        if a != first:
            return None
    return list(first)


# ===================================================================== regression check
def regression_check(D=None, budgets=(1, 2, 4, 8, 16), arms=None, log=logln):
    """REQUIREMENT (a): a static arm through the sequential path == the batch path, exactly.

    HOW IT WORKS. For each (arm, k) we build the batch answer set with the untouched
    `interview_strategies.ask()` and the sequential answer set with `StaticArmSelector` driven by
    `run_interview`, over the FULL 10,000-user test cohort, and assert Python list equality of BOTH
    returned structures. List equality is the strongest available statement: it compares ask ORDER,
    ask IDENTITY, which questions were answerable, and the half-star LEVEL attached to each answer --
    i.e. every input any consumer downstream can see. It is an assert, not a report: a mismatch means
    every adaptive number from this file is uninterpretable.

    `entropy0` at k=8 is asserted and printed by name because that is the named acceptance cell."""
    arms_req = list(arms or ST.ARMS)
    assert "entropy0" in arms_req and 8 in list(budgets), (
        "the NAMED acceptance cell entropy0/k=8 must be inside the regression grid; got "
        f"arms={arms_req} budgets={list(budgets)}")
    D = D if D is not None else load_arm_n(log=log)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=log)
    lvl = ST.level_lookup(D["g_te_tr"])
    arms = arms_req
    ok = 0
    for arm in arms:
        for k in budgets:
            a_batch, t_batch = ST.ask(arm, glob, bank, n, ni, lvl, k)
            sel = StaticArmSelector(arm, glob, bank, q_max=k)
            a_seq, t_seq, _info = run_interview(sel, k, n, ni, lvl, log=lambda *_: None,
                                                collect_dispersion=False)
            assert a_seq == a_batch, f"ASK MISMATCH arm={arm} k={k}"
            assert t_seq == t_batch, f"ANSWER MISMATCH arm={arm} k={k}"
            n_ans = float(np.mean([len(x) for x in t_seq]))
            log(f"[regression] {arm:14s} k={k:2d}  IDENTICAL  "
                f"(users={n} answered={n_ans:.3f}/{k})")
            ok += 1
    log(f"[regression] PASS: {ok} (arm,budget) cells byte-identical through the sequential path; "
        f"the named cell entropy0 k=8 is among them")
    return True


# ===================================================================== selector construction
def build_selector(spec, D, glob, bank, k, args, cache, log=logln):
    """spec -> Selector. `cache` holds the expensive shared artifacts (the frozen iALS belief, the
    Partial-VAE) so they are fitted/loaded ONCE per process, not once per budget."""
    if spec.endswith(STATIC_SUFFIX):
        return StaticArmSelector(spec[:-len(STATIC_SUFFIX)], glob, bank, q_max=k)
    if spec in ("glm_ucb", "thompson"):
        import bandit_selectors as B
        if "belief" not in cache:
            import belief_mf
            cache["belief"] = belief_mf.fit(D["train"], D["n_items"], log=log).belief
        if spec == "glm_ucb":
            return B.GLMUCBSelector(cache["belief"], alpha=args.ucb_alpha)
        return B.ThompsonSelector(cache["belief"], seed=args.ts_seed)
    if spec == "eddi_ig":
        import eddi_selector as E
        if "eddi" not in cache:
            import eddi_pvae
            eck = os.path.join(_ROOT, ".cache", "baselines", "eddi_ml25m_liang.pt")
            cache["eddi"] = eddi_pvae.load_trained(eck, D["n_items"], log=log).model
        return E.EDDIInfoGainSelector(cache["eddi"], cand_topm=args.eddi_topm)
    raise SystemExit(f"unknown selector spec '{spec}'. Known: <arm>{STATIC_SUFFIX} for arm in "
                     f"{ST.ARMS}, plus glm_ucb, thompson, eddi_ig")


def _smoke_slice(D, n_smoke, log=logln):
    """SMOKE ONLY. Keeps the FIRST n users of the test cohort and nothing else changes. This IS a
    truncation of the evaluation cohort, so it is loud, explicit, opt-in, and its output is stamped
    non-citable. It exists to exercise the code path in minutes, never to produce a number."""
    log("=" * 100)
    log(f"[adaptive] *** SMOKE: TRUNCATING THE TEST COHORT TO THE FIRST {n_smoke} USERS ***")
    log("[adaptive] *** every number from this run is a CODE-PATH CHECK and is NOT CITABLE ***")
    log("=" * 100)
    Ds = dict(D)
    for key in ("te_tr", "te_te", "g_te_tr", "pool"):
        Ds[key] = D[key][:n_smoke]
    return Ds


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--selectors", default="entropy0-static,glm_ucb,thompson,eddi_ig",
                    help="comma list; '<arm>-static' for any of " + ",".join(ST.ARMS))
    ap.add_argument("--only", default=None, help="comma list of consumers (default: all of RECS)")
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--arch", default="i25", choices=["i25", "i26"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--no_floor", action="store_true",
                    help="NO-FLOOR contract (the Act-1 primary). See run_interview_table for the rule.")
    ap.add_argument("--sel_batch", type=int, default=0,
                    help="users handed to the selector per call; 0 = the whole cohort at once "
                         "(default, and what lets state-grouping selectors vectorise). This is a "
                         "CHUNKING knob only -- every user is always processed.")
    ap.add_argument("--ucb_alpha", type=float, default=1.0,
                    help="GLM-UCB exploration constant. NOT fitted on test; see bandit_selectors.")
    ap.add_argument("--ts_seed", type=int, default=98765)
    ap.add_argument("--eddi_topm", type=int, default=0,
                    help="0 = score the FULL 18,359-item candidate pool (the default and the only "
                         "citable setting). >0 truncates the candidate pool and requires sign-off.")
    ap.add_argument("--regression", action="store_true",
                    help="run the static-arm equivalence assert and exit")
    ap.add_argument("--smoke_users", type=int, default=0,
                    help="SMOKE ONLY: keep the first N test users. Output is stamped non-citable.")
    a = ap.parse_args()

    budgets = [int(x) for x in a.budgets.split(",")]
    specs = [s.strip() for s in a.selectors.split(",") if s.strip()]
    recs = RECS if not a.only else [r for r in RECS if r in set(a.only.split(","))]

    D = load_arm_n(log=logln)
    if a.regression:
        regression_check(D, budgets=budgets, log=logln)
        return

    ni = D["n_items"]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    smoke = a.smoke_users > 0
    if smoke:
        D = _smoke_slice(D, a.smoke_users, log=logln)
    n = D["te_tr"].shape[0]
    lvl = ST.level_lookup(D["g_te_tr"])

    built = build_recommenders(recs, D, arch=a.arch, snapshot=a.snapshot, log=logln)
    pop_row, prior = mostpop_row(D, log=logln)

    lbl = a.label or a.arch
    outp = os.path.join(OUT, f"{'SMOKE_' if smoke else ''}adaptive_table_{lbl}.json")
    results = {"mostpop": {"_prior": prior}}
    meta = {"selectors": specs, "budgets": budgets, "consumers": recs, "arch": a.arch,
            "snapshot": os.path.basename(a.snapshot), "no_floor": bool(a.no_floor),
            "n_test_users": n, "ucb_alpha": a.ucb_alpha, "ts_seed": a.ts_seed,
            "eddi_cand_topm": a.eddi_topm}
    if smoke:
        meta["SMOKE_NOT_CITABLE"] = True
        meta["smoke_users"] = a.smoke_users
    if a.eddi_topm:
        meta["CANDIDATE_POOL_TRUNCATED"] = True
    logln(f"[adaptive] {len(recs)} consumers x {len(specs)} selectors x {len(budgets)} budgets "
          f"over {n} users; floor={'OFF' if a.no_floor else 'ON'}")

    cache = {}
    interviews = {}
    for spec in specs:
        for k in budgets:
            t_int = time.time()
            sel = build_selector(spec, D, glob, bank, k, a, cache, log=logln)
            asked, answered, info = run_interview(sel, k, n, ni, lvl,
                                                  ctx={"D": D, "glob": glob, "bank": bank},
                                                  sel_batch=a.sel_batch, log=logln)
            info["interview_minutes"] = round((time.time() - t_int) / 60.0, 2)
            interviews[(spec, k)] = (asked, answered, info)
            logln(f"[adaptive] INTERVIEW {spec:16s} k={k:2d} done in "
                  f"{info['interview_minutes']:.2f}m; answered={info['answer_rate'][-1]:.2f}/{k}; "
                  f"distinct ask-prefixes={info['distinct_prefixes'][-1]}")

    for name in recs:
        mode, obj = built[name]
        results[name] = {}
        for spec in specs:
            results[name][spec] = {}
            for k in budgets:
                if name == "golbandi_leaf" and k > MAX_DEPTH:
                    logln(f"[adaptive] SKIP golbandi_leaf k={k} > {MAX_DEPTH} (3^k leaf collapse)")
                    continue
                asked, answered, info = interviews[(spec, k)]
                ts = time.time()
                cell = score_arm(name, mode, obj, asked, answered, D, pop_row,
                                 no_floor=a.no_floor, global_asked=common_asked(asked))
                cell["score_minutes"] = round((time.time() - ts) / 60.0, 2)
                cell["interview_minutes"] = info["interview_minutes"]
                cell["distinct_prefixes"] = info["distinct_prefixes"][-1]
                results[name][spec][f"k{k}"] = cell
                logln(f"[adaptive] {name:14s} {spec:16s} k={k:2d} "
                      f"answered={cell['answered']:4.1f}/{k} floored={cell['floored']:5d} "
                      f"full={cell['full']:.4f} tail={cell['tail']:.4f} "
                      f"(interview {cell['interview_minutes']:.2f}m + score "
                      f"{cell['score_minutes']:.2f}m)")
                json.dump({"meta": meta, "results": results}, open(outp, "w"), indent=2)
    logln(f"[adaptive] done -> {outp}")


if __name__ == "__main__":
    main()
