r"""act2_surrogate_rank_corr.py -- ACT 2 / workstream C1 / TODO 1.11, PHASE 1.

Design sheet: docs/design/DS_ACT2_SURROGATE_RANK_CORR.md. Read it before touching this file; every
choice below (which states, which pool, which controls, which gates) is pre-registered there and this
script is not allowed to differ from it silently.

THE QUESTION. Act 2 claims published selectors lose because they optimise a SURROGATE rather than the
quantity anyone reports. `BASELINE_OBJECTIVE_AUDIT.md` establishes that by reading the code: the
surrogates are different OBJECTS. It does not establish that they are UNINFORMATIVE about the
reported object -- a different objective can still rank questions almost the way the metric does, and
if it does, objective mismatch is not the mechanism and Act 2 is wrong. So: at matched user states,
rank every candidate question two ways -- (a) by each method's own acquisition score, (b) by the
ACTUAL realised NDCG@10 gain of asking it -- and report the rank agreement.

WHAT THIS FILE DOES, AND WHAT IT DELIBERATELY DOES NOT DO. It computes side (b) ONLY: the realised
per-user NDCG@10 gain of every candidate at every state, for every budget. That is the entire cost of
the measurement (~12 h) and it needs NO selector code whatsoever, so it can be launched and left
alone while the cheap surrogate side (phase 2, `--surrogates`) and the statistics (phase 3,
`--analyze`) are built against its output. Splitting it this way is also what keeps the expensive
artifact reusable: the per-user x per-candidate gain matrix this run banks is exactly the label set
1.14, 1.15 and E1 need, and none of them will have to re-score it.

THE STATES. `fixedorder:entropy0`, the reference static interview, read out of the harness's OWN
checkpoints via `act2_pv_quartiles.read_interview` -- imported, never re-derived, so these are the
same states the PV strata are cut on and the two Act-2 exhibits sit on one ruler. That cell asks ONE
fixed item order of everybody and spends zero concept turns, so a state is exactly (asked items, the
user's own half-star levels, with the UNSEEN sentinel where they could not answer) and the population
occupies 11 / 112 / 1,463 / 5,767 distinct states at k = 1 / 2 / 4 / 8. Those four counts are
ASSERTED against the published table, not trusted (`EXPECTED_STATES`).

A CONSEQUENCE WORTH STATING, BECAUSE IT IS WHY THIS IS CHEAP. Every user is asked the SAME questions
at a given budget; only the ANSWERS differ. So the realised gain of candidate q at every state at
once is ONE call to `genv_eval(E, prefix + [q], all 10,000 rows)` -- the tree's own per-candidate
exact-NDCG path, unmodified, with the per-user answers it already looks up. States are recovered
afterwards by grouping users on the state id. There is no new scoring code on this path at all,
which is the strongest form of "same ruler as the tree" available.
(`strategy_channel_suite.genv_eval_multi`, added alongside this work, is for PHASE 1b -- each arm's
unrestricted argmax differs BY STATE, so there the users genuinely carry different candidates.)

THE POOL. The tree's COMBINED action space, built by `ndcg_tree.build_pool`: top-500 items by train
popularity + top-500 concepts by the `answerable` bank, items first. This is a declared restriction
of the published arms' native action spaces (they argmax over 18,359 items / 1,031 concepts); the
non-lossy alternative is 19,390 candidates at ~230 h and is NOT run. See design sheet section 3,
including phase 1b, which recovers a pool-FREE regret statistic for 1.3% extra cost.

PERSISTENCE (HARD RULE 11). One directory per budget; the gains live in a float32 memmap of shape
(n_pool, n_users) flushed after EVERY candidate, with a per-candidate `done` mask and a
contract-signed sidecar. The blast radius of a kill is one candidate (~8 s). A resume against a
different contract is refused loudly and nothing is merged. Budgets are scored in priority order
(1, 2, 0, 4, 8) and each is a complete, self-contained cell, so a stop after any budget leaves a
valid partial table rather than nothing.

  python scripts/act2_surrogate_rank_corr.py --selftest       # no env, no model: states + memmaps
  python scripts/act2_surrogate_rank_corr.py --smoke          # env + ALL gates + measured s/eval
  python scripts/act2_surrogate_rank_corr.py --score          # phase 1 (heavy, ~12 h, resumable)
"""
from __future__ import annotations

import os
import sys

_FULL = "--full_threads" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ.setdefault("OMP_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines"))
sys.path.insert(0, str(ROOT / "src" / "instrument"))
sys.path.insert(0, str(ROOT / "scripts" / "reddit"))
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / ".cache" / "act2_surrogate"
IVIEW = ROOT / "experiments" / "baselines" / "interview"
PV_LABEL = "act2pv"
CKDIR = IVIEW / f"adaptive_table_{PV_LABEL}_interviews"
PERUSER = IVIEW / f"adaptive_table_{PV_LABEL}_per_user.npz"
TREE_TEST_JSON = ROOT / "experiments" / "battery" / "ndcgtree16test.json"
TREE_TEST_NPZ = ROOT / "experiments" / "battery" / "ndcgtree16test_per_user.npz"

PV_SPEC = "fixedorder:entropy0"
BUDGET_ORDER = (1, 2, 0, 4, 8)                 # priority order; each budget is a complete cell
EXPECTED_STATES = {0: 1, 1: 11, 2: 112, 4: 1463, 8: 5767}   # published, ACT2_PV_QUARTILES.md sec 5
N_ITEMS_POOL = 500
N_CONCEPTS_POOL = 500
N_COHORT = 10000
G_A_LEAVES = 3                                 # how many banked tree leaves the contract gate replays


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(o):
    return hashlib.sha1(json.dumps(o, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


# ============================================================== the states (banked, never re-derived)
def build_states(k):
    """(prefix, state_id per user, n_states) for `fixedorder:entropy0` at budget k.

    The asked list, the answered levels and the concept tokens all come from the harness's own
    checkpoint through `act2_pv_quartiles.read_interview`; the state IDENTITY comes from
    `dispersion_leg1.States`, the same keying `act2_pv_quartiles` uses (exact half-star levels, item
    half sorted by item id, UNSEEN sentinel for asked-but-unrated). Nothing here re-implements either.

    Two properties of this cell are ASSERTED rather than assumed, because the whole cheap design
    rests on them: every user is asked the SAME items (so one `genv_eval` call scores every state at
    once), and no turn is spent on a concept (so the state is an item state).
    """
    import dispersion_leg1 as L1
    from act2_pv_quartiles import read_interview, states_of

    if k == 0:
        return [], np.zeros(N_COHORT, np.int64), 1
    asked, answered, toks = read_interview(PV_SPEC, k, CKDIR)
    assert len(asked) == N_COHORT, f"{PV_SPEC}/k{k}: {len(asked)} users != {N_COHORT}"
    prefix = [int(q) for q in asked[0]]
    assert all([int(q) for q in a] == prefix for a in asked), \
        f"{PV_SPEC}/k{k} is not a fixed order -- the one-call-per-candidate design does not hold"
    assert all(len(t) == 0 for t in toks), \
        f"{PV_SPEC}/k{k} spends a turn on a concept; this builder only handles item states"
    ST = L1.States()
    sid = np.array([ST.add(s, l, c)[0] for s, l, c in states_of(asked, answered, toks)], np.int64)
    ns = int(len(set(sid.tolist())))
    assert ns == len(ST), "state ids are not dense"
    assert ns == EXPECTED_STATES[k], \
        (f"STATE COUNT MISMATCH at k={k}: rebuilt {ns}, ACT2_PV_QUARTILES.md section 5 publishes "
         f"{EXPECTED_STATES[k]}. The banked states have moved -- stop and find out why.")
    return prefix, sid, ns


# ============================================================================= the scoring env
def build_env(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool, leak_assert
    E = build_greedy_env(args.snapshot or CERT_TOWER, "test", args.concept_ckpt or CERT_CONC,
                         "act2surr(test)")
    rows = E["rows"]
    assert len(rows) == N_COHORT and rows == list(range(N_COHORT)), \
        (f"cohort mismatch: build_greedy_env gave {len(rows)} rows starting {rows[:3]}; the banked "
         f"interviews are indexed by user 0..{N_COHORT - 1} and the two must be the same object")
    contract_selftest(E, rows)
    leak_assert(E, rows, "test")                          # GATE G-E
    pool, item_pool, conc_pool = build_pool(E, N_ITEMS_POOL, N_CONCEPTS_POOL)
    log(f"[pool] COMBINED {len(item_pool)} items (bank 'pop') + {len(conc_pool)} concepts "
        f"(bank 'answerable') = {len(pool)} candidates")
    return E, pool


def contract_of(E, pool, k, prefix, ns):
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    return {"v": 1, "analysis": "act2_surrogate_rank_corr:phase1_realised_ndcg_gain",
            "tower": os.path.basename(CERT_TOWER), "concept_ckpt": os.path.basename(CERT_CONC),
            "split": "test", "n_users": N_COHORT, "mask_policy": E["pool_policy"],
            "state_source": f"{PV_LABEL}/{PV_SPEC}", "budget": int(k),
            "prefix": [int(q) for q in prefix], "n_states": int(ns),
            "n_pool": len(pool), "n_items_pool": N_ITEMS_POOL, "n_concepts_pool": N_CONCEPTS_POOL,
            "pool_sha1": sha([[int(a), int(b)] for a, b in pool]),
            "objective": "per-user full AND tail NDCG@10 gain vs the k-question baseline"}


# ============================================================================= gates
def gate_A_tree_contract(E):
    """G-A. The exact-NDCG path here must reproduce, BIT FOR BIT, per-user values the NDCG tree has
    already banked on this same TEST cohort. Free: the tree's leaves are on disk with their own
    sequences and row sets. If this fails, nothing downstream is on the tree's ruler."""
    from strategy_channel_suite import set_active_rows, genv_eval
    assert TREE_TEST_JSON.exists() and TREE_TEST_NPZ.exists(), "banked tree TEST outputs missing"
    blob = json.load(open(TREE_TEST_JSON))
    z = np.load(TREE_TEST_NPZ)
    leaves = sorted(blob["leaf_table"], key=lambda r: -r["n_test"])[:G_A_LEAVES]
    worst = 0.0
    n_checked = 0
    for lf in leaves:
        nid = lf["nid"]
        rk = f"test|{nid}|rows"
        if rk not in z.files:
            log(f"[G-A] leaf {nid} has no banked test rows -- skipped")
            continue
        rows = [int(r) for r in z[rk]]
        seq = [(int(a), int(b)) for a, b in lf["own_sequence"]]
        set_active_rows(E, rows)
        for K in blob["budgets"]:
            key = f"test|{nid}|{K}|own_full"
            if key not in z.files:
                continue
            f, _t = genv_eval(E, seq[:K], rows)
            d = float(np.nanmax(np.abs(f - z[key].astype(np.float64))))
            worst = max(worst, d)
            n_checked += 1
    set_active_rows(E, E["rows"])
    assert n_checked > 0, "G-A checked nothing -- the banked tree npz does not carry test row sets"
    log(f"[G-A] replayed {n_checked} banked (leaf, K) cells of the NDCG tree on TEST: "
        f"max |diff| = {worst:.3e}")
    assert worst == 0.0, (f"G-A FAILED: this harness does not reproduce the tree's banked per-user "
                          f"NDCG (max |diff| {worst:.3e}). Two different rulers -- stop.")


def gate_B_state_reconstruction(E, k, prefix, base_full):
    """G-B. Our reconstruction of the entropy0 state must reproduce the interview harness's OWN
    scored per-user NDCG at that budget. This is the check that the state builder, the answer lookup
    and the fold contract all agree with the run the states were taken from."""
    assert PERUSER.exists(), f"missing {PERUSER}"
    z = np.load(PERUSER)
    key = f"ours|{PV_SPEC}|k{k}|full"
    if key not in z.files:
        log(f"[G-B] k={k}: no banked harness vector ({key}) -- gate not applicable")
        return None
    b = z[key].astype(np.float64)
    assert len(b) == len(base_full), f"G-B length mismatch {len(b)} vs {len(base_full)}"
    d = float(np.nanmax(np.abs(base_full - b)))
    log(f"[G-B] k={k}: baseline vs the harness's own `{key}`: max |diff| = {d:.3e} "
        f"(means {np.nanmean(base_full):.8f} vs {np.nanmean(b):.8f})")
    assert d <= 1e-6, (f"G-B FAILED at k={k}: max |diff| {d:.3e} > 1e-6. Our state reconstruction "
                       f"and the harness that produced the states disagree -- explain it before "
                       f"reporting anything.")
    return d


def gate_D_partition(base_full, sid, ns):
    """G-D. The cohort mean must equal the state-size-weighted mean of the state means. Algebraic
    identity for a per-user quantity over a partition; it is here because it catches a wrong
    state->user map, which nothing else would notice."""
    ok = np.isfinite(base_full)
    tot = float(base_full[ok].mean())
    w = 0.0
    for s in range(ns):
        m = ok & (sid == s)
        if m.any():
            w += m.sum() / ok.sum() * float(base_full[m].mean())
    err = abs(w - tot)
    log(f"[G-D] partition decomposition err = {err:.3e} over {ns} states")
    assert err < 1e-9, f"G-D FAILED: state partition does not reconstruct the cohort mean ({err:.3e})"


def gate_C_multistate_identity(E, prefix, pool):
    """G-C. `genv_eval_multi` (phase 1b's engine) must equal `genv_eval` bit for bit when every user
    carries the same sequence. Run on a small row slice so the gate is cheap; the equality it checks
    is exact, not statistical."""
    from strategy_channel_suite import set_active_rows, genv_eval, genv_eval_multi
    rows = list(range(600))
    set_active_rows(E, rows)
    seq = list(dict.fromkeys([(0, int(i)) for i in prefix] + [pool[0], pool[-1]]))
    f1, t1 = genv_eval(E, seq, rows)
    isq = [[int(i) for c, i in seq if c == 0]] * len(rows)
    csq = [[int(i) for c, i in seq if c == 1]] * len(rows)
    f2, t2 = genv_eval_multi(E, rows, isq, csq)
    df = float(np.nanmax(np.abs(f1 - f2))); dt = float(np.nanmax(np.abs(t1 - t2)))
    set_active_rows(E, E["rows"])
    log(f"[G-C] genv_eval_multi vs genv_eval on {len(rows)} rows, seq {seq}: "
        f"max |dfull| = {df:.3e}, max |dtail| = {dt:.3e}")
    assert df == 0.0 and dt == 0.0, "G-C FAILED: the heterogeneous evaluator is on a different ruler"


# ============================================================================= phase 1
def open_cell(cdir, contract, n_pool, resume=True):
    """Memmaps + contract sidecar for one budget. A banked cell whose contract differs is REFUSED,
    never merged (HARD RULE 11b)."""
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / "contract.json"
    if cpath.exists():
        old = json.load(open(cpath))
        if old != contract:
            raise SystemExit(f"[resume] REFUSING: {cdir.name} carries a DIFFERENT contract.\n"
                             f"  banked={old}\n  wanted={contract}\n  delete {cdir} to start clean.")
        if not resume:
            raise SystemExit(f"[resume] {cdir} exists and --no_resume was given")
    json.dump(contract, open(cpath, "w"), indent=1)

    def mm(name, shape, dtype):
        p = cdir / name
        mode = "r+" if p.exists() else "w+"
        a = np.lib.format.open_memmap(p, mode=mode, dtype=dtype, shape=shape)
        return a
    return {"gain_full": mm("gain_full.npy", (n_pool, N_COHORT), np.float32),
            "gain_tail": mm("gain_tail.npy", (n_pool, N_COHORT), np.float32),
            "done": mm("done.npy", (n_pool,), np.bool_)}


def score_budget(E, pool, k, cdir, args):
    from strategy_channel_suite import genv_eval

    prefix_items, sid, ns = build_states(k)
    prefix = [(0, int(i)) for i in prefix_items]
    ctr = contract_of(E, pool, k, prefix_items, ns)
    C = open_cell(cdir, ctr, len(pool), resume=not args.no_resume)
    np.save(cdir / "state_id.npy", sid)

    rows = E["rows"]
    t0 = time.time()
    bf, bt = genv_eval(E, prefix, rows)
    np.save(cdir / "base_full.npy", bf.astype(np.float32))
    np.save(cdir / "base_tail.npy", bt.astype(np.float32))
    t_base = time.time() - t0
    log(f"[k={k}] baseline over {len(rows)} users in {t_base:.2f}s: full {np.nanmean(bf):.8f} "
        f"tail {np.nanmean(bt):.8f} | {ns} states | prefix {prefix_items}")
    gate_B_state_reconstruction(E, k, prefix_items, bf)
    gate_D_partition(bf, sid, ns)

    inpref = set(prefix)
    todo = [j for j in range(len(pool)) if not C["done"][j] and pool[j] not in inpref]
    skipped = [j for j in range(len(pool)) if pool[j] in inpref]
    for j in skipped:                                     # asked already: never a candidate
        C["gain_full"][j, :] = np.nan
        C["gain_tail"][j, :] = np.nan
        C["done"][j] = True
    log(f"[k={k}] {len(todo)} candidates to score ({int(C['done'].sum())} already done, "
        f"{len(skipped)} excluded because they are in the prefix)")
    if args.max_candidates:
        todo = todo[:args.max_candidates]
        log(f"[k={k}] --max_candidates {args.max_candidates}: SMOKE ONLY, never citable")

    t0 = time.time()
    for n, j in enumerate(todo, 1):
        q = pool[j]
        f, t = genv_eval(E, prefix + [q], rows)
        C["gain_full"][j, :] = (f - bf).astype(np.float32)
        C["gain_tail"][j, :] = (t - bt).astype(np.float32)
        C["done"][j] = True
        C["gain_full"].flush(); C["gain_tail"].flush(); C["done"].flush()
        if n % 25 == 0 or n < 3:
            el = time.time() - t0
            log(f"[k={k}] {n}/{len(todo)} | {el / n:.2f}s/candidate | "
                f"eta {(len(todo) - n) * el / n / 3600:.2f} h | last {'concept' if q[0] else 'item'} "
                f"{q[1]} mean gain {float(np.nanmean(f - bf)):+.6f}")
    el = time.time() - t0
    sec = el / max(len(todo), 1)
    log(f"[k={k}] DONE {len(todo)} candidates in {el / 60:.1f} min ({sec:.2f}s/candidate)")
    json.dump({"contract": ctr, "n_scored": int(C["done"].sum()), "n_excluded": len(skipped),
               "sec_per_candidate": round(sec, 3), "baseline_seconds": round(t_base, 3),
               "base_full_mean": float(np.nanmean(bf)), "base_tail_mean": float(np.nanmean(bt)),
               "citable": not bool(args.max_candidates)},
              open(cdir / "meta.json", "w"), indent=1)
    return sec


# ============================================================================= modes
def selftest():
    """Everything that can be verified WITHOUT the tower, the concept net or the ratings frame --
    so it costs no RAM and cannot disturb another job on the box."""
    log("=" * 100)
    log("SELFTEST: state reconstruction + persistence. No model, no env, nothing citable.")
    for k in (0, 1, 2, 4, 8):
        prefix, sid, ns = build_states(k)
        sizes = np.bincount(sid, minlength=ns)
        log(f"  k={k}: prefix {prefix} | {ns} states (published {EXPECTED_STATES[k]}) | "
            f"largest {sizes.max()} users, {int((sizes == 1).sum())} singleton states, "
            f"median {int(np.median(sizes))}")
    tmp = CACHE / "_selftest"
    if tmp.exists():
        shutil.rmtree(tmp)
    ctr = {"v": 1, "probe": True}
    C = open_cell(tmp, ctr, 4)
    C["gain_full"][2, :5] = np.arange(5, dtype=np.float32)
    C["done"][2] = True
    C["gain_full"].flush(); C["done"].flush()
    del C
    C2 = open_cell(tmp, ctr, 4)
    assert bool(C2["done"][2]) and float(C2["gain_full"][2, 3]) == 3.0, "memmap resume is broken"
    log("  memmap resume: a flushed candidate survives a reopen")
    try:
        open_cell(tmp, {"v": 1, "probe": False}, 4)
    except SystemExit:
        log("  contract guard: a cell with a DIFFERENT contract is REFUSED, not merged")
    else:
        raise AssertionError("contract guard did not fire")
    del C2
    shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST PASSED")
    log("=" * 100)


def smoke(args):
    """Build the env, run EVERY gate, and measure s/candidate on one budget. Writes only under
    .cache/act2_surrogate/_smoke and is never citable."""
    log("#" * 100)
    log("# SMOKE: gates + a measured per-candidate time. NOT CITABLE.")
    log("#" * 100)
    E, pool = build_env(args)
    gate_A_tree_contract(E)
    prefix_items, _sid, _ns = build_states(1)
    gate_C_multistate_identity(E, prefix_items, pool)
    sdir = CACHE / "_smoke" / "k1"
    if sdir.exists():
        shutil.rmtree(sdir)
    args.max_candidates = args.max_candidates or 4
    sec = score_budget(E, pool, 1, sdir, args)
    n_eff = len(pool) + 1
    log("=" * 100)
    log(f"[cost] MEASURED {sec:.2f} s/candidate at n={N_COHORT} -> {n_eff * sec / 3600:.2f} h per "
        f"budget, {5 * n_eff * sec / 3600:.1f} h for k in {BUDGET_ORDER}")
    log("=" * 100)
    return sec


def phase_score(args):
    budgets = ([int(x) for x in args.budgets.split(",")] if args.budgets else list(BUDGET_ORDER))
    for k in budgets:
        assert k in EXPECTED_STATES, f"budget {k} has no banked entropy0 state set"
    E, pool = build_env(args)
    # Every gate runs inside the real job too -- the smoke is a convenience, never the only place a
    # contract is checked. G-B and G-D fire per budget inside `score_budget`.
    gate_A_tree_contract(E)
    gate_C_multistate_identity(E, build_states(budgets[0])[0], pool)
    log(f"[plan] budgets in priority order {budgets}; each is a complete, self-contained cell")
    for k in budgets:
        score_budget(E, pool, k, CACHE / f"k{k}", args)
    log("[done] phase 1 complete for budgets " + ",".join(str(b) for b in budgets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no env: states + persistence only")
    ap.add_argument("--smoke", action="store_true", help="env + every gate + measured s/candidate")
    ap.add_argument("--score", action="store_true", help="PHASE 1 (heavy, resumable)")
    ap.add_argument("--budgets", default="", help="comma list; default is the priority order")
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--concept_ckpt", default="")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--max_candidates", type=int, default=0,
                    help="SMOKE ONLY -- stop after this many candidates; marks the cell not citable")
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed at IMPORT by strategy_channel_suite (it reads sys.argv); declared "
                         "here only so argparse accepts it. The BLAS vars must also be exported by "
                         "the launcher before python starts.")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.smoke:
        return smoke(args)
    if args.score:
        return phase_score(args)
    ap.error("choose one of --selftest / --smoke / --score")


if __name__ == "__main__":
    main()
