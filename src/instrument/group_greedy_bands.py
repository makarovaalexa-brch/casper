r"""T2 -- THE EVIDENCE-COUNT BAND TEACHER: a greedy order per band, with PER-USER prefixes.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part L.2 and L.3. VALIDATION ONLY; no `--split`;
`test_touched=false`. **NOTHING HERE HAS BEEN RUN** at the time of writing (HARD RULE 10: committed
before any run), and `--mode selftest` is the gate that must pass before `--mode search` is allowed.

WHY THIS FILE EXISTS AND WHY IT IS NOT `group_greedy_signature.py`
------------------------------------------------------------------
T2's signature is **how many items the user has volunteered** (none / 1-2 / 3-8). That is observable
at ask time by construction, and its bands are >= 1,000 users by construction, so it satisfies both
conditions of the design law. But its states carry **per-user prefixes** -- every user volunteered
DIFFERENT items -- and `strategy_channel_suite.greedy_certified` searches with one prefix shared by
the whole cohort. So the CELF loop is rebuilt here over `genv_eval_multi`, which is the function that
scores heterogeneous per-user sequences in one pass.

★ T3 (taste-popularity band) IS NOT IMPLEMENTED HERE, DELIBERATELY. A cold user's fold-in tail mass is
not observable at ask time -- it is computed from the profile the system does not yet have, which is
the Part K.6 error in a new costume. If T3 is ever run it must be a band estimated from the evidence
the user actually volunteered, i.e. a refinement INSIDE a T2 band, and its cohorts counted there.

THE GATE (`--mode selftest`, and it is not optional)
----------------------------------------------------
Hand every user the SAME prefix and this loop must reproduce `greedy_certified` **exactly** -- same
picks, same gains, same evaluation count. A per-user-prefix search that cannot reproduce the shared-
prefix search on shared-prefix input is not a generalisation of it, it is a different algorithm.

PERSISTENCE (HARD RULE 11): every pick is written the moment it lands, and the CELF heap is persisted
under a contract signature that includes the band definition and the draw.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import heapq
import argparse
import numpy as np

from policy_net import OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL, log

OUT_DIR = os.path.join(OUTDIR, "forest")
BANDS = (("none", 0, 0), ("few", 1, 2), ("many", 3, 8))    # declared, not tuned


def split_channels(qs):
    return ([int(i) for c, i in qs if int(c) == 0], [int(i) for c, i in qs if int(c) == 1])


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def mean_score(E, rows, seeds, seq, objective="full"):
    """Cohort-mean NDCG@10 when every user carries their own volunteered prefix and the SAME `seq`.

    A question already volunteered by a user is folded once for that user -- the turn is burned and no
    new evidence arrives, which is the honest semantics and the same one `policy_resume` measured.
    """
    from strategy_channel_suite import genv_eval_multi
    iseqs, cseqs = [], []
    for sd in seeds:
        it, cc = split_channels(list(sd) + list(seq))
        iseqs.append(dedup(it)); cseqs.append(dedup(cc))
    f, t = genv_eval_multi(E, rows, iseqs, cseqs)
    return float(np.nanmean(f if objective == "full" else t))


def celf_per_user_prefix(E, rows, seeds, pool, L, objective, tag, on_pick=None, heap_path=None,
                         sig=None, log_every=25):
    """Lazy-greedy over a candidate pool where each user carries their own already-known evidence.

    Structurally identical to `strategy_channel_suite.greedy_certified`: seed every candidate's gain
    once, then pop the best, and re-evaluate stale entries against the current sequence. The only
    difference is that the score comes from `mean_score` (per-user prefixes) instead of `genv_mean`
    (one shared prefix), and with equal prefixes the two agree exactly -- that is `--mode selftest`.
    """
    t0 = time.time()

    def save(heap, seq, gains, base, cur, nev):
        if not heap_path:
            return
        blob = {"sig": sig, "tag": tag, "base": float(base), "cur": float(cur), "nev": int(nev),
                "seq": [[int(a), int(b)] for a, b in seq], "gains": [float(g) for g in gains],
                "heap": [[float(a), [int(q[0]), int(q[1])], int(u)] for a, q, u in heap]}
        os.makedirs(os.path.dirname(heap_path) or ".", exist_ok=True)
        with open(heap_path + ".tmp", "w") as fh:
            json.dump(blob, fh)
        os.replace(heap_path + ".tmp", heap_path)

    banked = None
    if heap_path and os.path.exists(heap_path):
        try:
            b = json.load(open(heap_path))
        except Exception as e:
            b = None
            log(f"[bands {tag}] heap sidecar unreadable ({e}) -- ignored, seeding cold")
        if b is not None and b.get("sig") == sig:
            banked = b
        elif b is not None:
            log(f"[bands {tag}] heap sidecar carries a DIFFERENT contract -- IGNORED, seeding cold")

    if banked is not None:
        base = float(banked["base"]); cur = float(banked["cur"]); nev = int(banked["nev"])
        seq = [(int(a), int(b)) for a, b in banked["seq"]]
        gains = [float(g) for g in banked["gains"]]
        heap = [(float(a), (int(q[0]), int(q[1])), int(u)) for a, q, u in banked["heap"]]
        heapq.heapify(heap)
        log(f"[bands {tag}] RESUMED: {len(seq)} pick(s), {len(heap)} heap entries, {nev} evals paid")
        if len(seq) >= L:
            return seq[:L], gains[:L], nev
    else:
        base = mean_score(E, rows, seeds, [], objective)
        heap = []; nev = 0
        for q in pool:
            v = mean_score(E, rows, seeds, [q], objective); nev += 1
            heapq.heappush(heap, (-(v - base), tuple(q), 0))
            if nev % log_every == 0:
                el = time.time() - t0
                log(f"[bands {tag}] seed {nev}/{len(pool)} ({el / 60:.1f}m, {el / nev:.2f}s/eval)")
        seq = []; gains = []; cur = base
        save(heap, seq, gains, base, cur, nev)
    while len(seq) < L and heap:
        neg, q, upd = heapq.heappop(heap)
        if upd == len(seq):
            seq.append(q); gains.append(-neg); cur += -neg
            log(f"[bands {tag}] pick {len(seq)}/{L} = {'concept' if q[0] else 'item'} {q[1]} "
                f"gain {-neg:+.5f} -> {cur:.5f} (nev={nev}, {(time.time() - t0) / 60:.1f}m)")
            save(heap, seq, gains, base, cur, nev)
            if on_pick is not None:
                on_pick(list(seq), list(gains), nev)
        else:
            v = mean_score(E, rows, seeds, seq + [q], objective); nev += 1
            heapq.heappush(heap, (-(v - cur), q, len(seq)))
    return seq, gains, nev


def open_env(tag):
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, tag)
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], f"{tag}/val")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")
    pool, _i, _c = build_pool(E, 500, 500)
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, _ = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: {float(np.nanmean(sf))!r}")
    return E, rows, pool


def mode_selftest(args):
    """THE GATE: with one shared prefix, this loop must equal `greedy_certified` exactly."""
    from strategy_channel_suite import greedy_certified
    E, rows, pool = open_env("bands/selftest")
    sub = rows[:args.n_users]
    cand = pool[:args.n_candidates]
    pre = [(1, 114)]                                   # the tree's root, shared by every user
    cand = [q for q in cand if tuple(q) != tuple(pre[0])]
    seeds = [list(pre) for _ in sub]
    log(f"[selftest] {len(sub)} users x {len(cand)} candidates, L={args.L}, shared prefix {pre}")
    a_seq, a_gain, a_nev = greedy_certified(E, sub, cand, args.L, "full", "bands/ref")
    b_seq, b_gain, b_nev = celf_per_user_prefix(E, sub, seeds, cand, args.L, "full", "bands/new")
    assert [tuple(q) for q in a_seq] == [tuple(q) for q in b_seq], (
        f"PICKS DIFFER\n  greedy_certified {a_seq}\n  per-user-prefix  {b_seq}")
    assert np.allclose(a_gain, b_gain, atol=1e-12), (
        f"GAINS DIFFER\n  {a_gain}\n  {b_gain}")
    assert a_nev == b_nev, f"EVAL COUNTS DIFFER: {a_nev} vs {b_nev}"
    log(f"[selftest] PASS: identical picks, gains to 1e-12, and {a_nev} evaluations on both paths")


def mode_search(args):
    assert args.i_have_author_goahead, "REFUSING TO SEARCH without --i_have_author_goahead"
    import pu_labels as PL
    from policy_resume import draw_volunteered
    t00 = time.time()
    E, rows, pool = open_env("bands/search")
    fit, hold, fw = PL.fit_holdout_split(np.asarray(rows, np.int64))
    fs = set(int(x) for x in fit.tolist())
    log(f"[firewall] the search sees the {len(fit):,} FIT users only")

    # THE DRAW, DECLARED (HARD RULE 1): the band is how many items the user has volunteered. Users are
    # assigned to bands by a single rng stream over the FIT cohort in canonical row order, then each
    # user's items are drawn by R's own popularity-weighted rule so the two experiments are comparable.
    rng = np.random.default_rng(args.draw_seed)
    frows = [int(r) for r in rows if int(r) in fs]
    band_of = rng.integers(0, len(BANDS), size=len(frows))
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    for bi, (name, lo, hi) in enumerate(BANDS):
        brows = [r for r, b in zip(frows, band_of) if int(b) == bi]
        ks = rng.integers(lo, hi + 1, size=len(brows)) if hi > lo else np.full(len(brows), lo)
        seeds = []
        for r, k in zip(brows, ks):
            if int(k) == 0:
                seeds.append([]); continue
            v, _info = draw_volunteered(E, [r], int(k), seed=args.draw_seed + int(r))
            seeds.append([(0, int(i)) for i in v[0]])
        n_seed = float(np.mean([len(s) for s in seeds]))
        log(f"[composition] band {name}: {len(brows):,} FIT users, mean {n_seed:.2f} volunteered "
            f"items (range {lo}-{hi})")
        assert len(brows) >= args.min_band, (
            f"band {name} has {len(brows)} users, below the declared floor {args.min_band}")
        L = TOTAL_L
        stem = os.path.join(OUT_DIR, f"t2band_{name}")
        sig = {"tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
               "pool_contract": E["pool_contract"], "band": [name, lo, hi],
               "draw_seed": int(args.draw_seed), "L": L, "objective": args.objective,
               "n_users": len(brows)}

        def on_pick(seq, gains, nev, _n=name, _stem=stem, _b=brows):
            with open(_stem + "_partial.json", "w") as fh:
                json.dump({"analysis": "t2band_partial", "band": _n, "test_touched": False,
                           "split": "validation", "cohort": "FIT", "n_users": len(_b),
                           "order": [[int(a), int(c)] for a, c in seq],
                           "gains": [float(x) for x in gains], "n_evals": int(nev)}, fh, indent=1)

        seq, gains, nev = celf_per_user_prefix(E, brows, seeds, pool, L, args.objective,
                                               f"t2/{name}", on_pick=on_pick,
                                               heap_path=stem + "_heap.json", sig=sig)
        full = mean_score(E, brows, seeds, seq, "full")
        tail = mean_score(E, brows, seeds, seq, "tail")
        blob = {"analysis": "t2_evidence_count_band_order", "band": name, "citable": True,
                "design_sheet": "docs/design/PAPER_B_POLICY_DESIGN.md L.2/L.3",
                "test_touched": False, "split": "validation", "cohort": "FIT", "contract": sig,
                "n_users": len(brows), "mean_volunteered_items": n_seed,
                "order": [[int(a), int(c)] for a, c in seq], "gains": [float(x) for x in gains],
                "n_evals": int(nev), "n_concepts_in_order": int(sum(1 for a, _c in seq if a == 1)),
                "in_sample_full": full, "in_sample_tail": tail,
                "in_sample_warning": "SELECTION-cohort numbers; the citable comparison is HOLDOUT",
                "firewall": fw, "minutes": round((time.time() - t00) / 60, 2)}
        with open(stem + ".json", "w") as fh:
            json.dump(blob, fh, indent=2, default=float)
        summary[name] = {"full": full, "tail": tail, "n_evals": nev}
        log(f"[{name}] DONE: in-sample full {full:.5f} tail {tail:.5f} ({nev} evals)")
    log(f"[done] {json.dumps(summary, default=float)}")


def main():
    ap = argparse.ArgumentParser(description="T2: evidence-count band orders, per-user prefixes")
    ap.add_argument("--mode", required=True, choices=("selftest", "search"))
    ap.add_argument("--L", type=int, default=3, help="selftest only: how many picks to compare")
    ap.add_argument("--n_users", type=int, default=1000, help="selftest only")
    ap.add_argument("--n_candidates", type=int, default=40, help="selftest only")
    ap.add_argument("--objective", default="full", choices=("full", "tail"))
    ap.add_argument("--draw_seed", type=int, default=20260817)
    ap.add_argument("--min_band", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert "--split" not in sys.argv, "there is no --split here: this file is validation-only"
    {"selftest": mode_selftest, "search": mode_search}[args.mode](args)


if __name__ == "__main__":
    main()
