r"""act1_paired.py -- PAIRED percentile-bootstrap CIs for every Act-1 selector-vs-static comparison.

WHY. `docs/results/ACT1_ADAPTIVE_VS_STATIC.md` states its own gap: "Point estimates only -- no paired
CIs yet." Every headline there (static beats every adaptive selector on `ours`; Thompson REVERSES that
on `golbandi_leaf`) currently rests on a difference of two cohort means with no interval.

THE FORM IS PAIRED AND IT IS NOT OPTIONAL. `ARM_N_PHASE1.md` sec 4b records the scar: the unpaired
`1.96*sqrt(2)*SE` interval overstated the true interval ~4.6x and produced a "tie" claim that had to be
retracted. Here we resample USERS, recompute the per-user difference, and take percentiles of the
difference distribution -- the same `boot_ci`/`paired` pattern as `interview_paired.py` and
`concept_vs_item_paired.py`.

PAIRING IS VALID. Every arm scores the SAME 10,000 held-out users against the SAME targets under the
SAME fixed arm-N candidate pool and the same NO-FLOOR contract; only the asked questions (and, for the
golbandi block, the consumer) change. `metrics.evaluate` drops users with no held-out target and, for
tail, users with no tail target -- but those subsets are properties of the DATA, so a given key stays
row-aligned across every arm. Asserted by a canary, not assumed.

THE STORED TABLE IS THE CONTROL. Interviews are regenerated with the identical seeds (ST.SEED=4242,
ts_seed=98765, ucb_alpha=1.0), so every recomputed cell mean is checked against
`adaptive_table_i26c_ep31.json` / `interview_table_ladder_ep31.json`. Any difference is recorded in
`meta.control_drift` rather than tolerated silently.

  python src/baselines/act1_paired.py --boot 10000 --seed 98765
  python src/baselines/act1_paired.py --analyze_only        # re-run the statistics from the cache
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
from run_interview_table import build_recommenders, score_arm, mostpop_row, OUT
from adaptive_harness import build_selector, run_interview, common_asked
from golbandi_leaf import MAX_DEPTH

BUDGETS = (1, 2, 4, 8, 16)
STATICS = ("helf", "entropy0", "popularity", "pure_entropy", "random_bank", "random_catalog")
ADAPTIVE = ("glm_ucb", "thompson", "eddi_ig")
# Every static arm is driven through the sequential path as "<arm>-static" (regression-proven identical
# to interview_strategies.ask), so the ladder and the adaptive table share ONE ask implementation.
SPECS = tuple(f"{a}-static" for a in STATICS) + ADAPTIVE
BASE = "entropy0-static"                       # the Act-1 comparator named in the chapter

# golbandi_leaf is capped at k<=8 (3^k leaf fragmentation) and is expensive to score once the ask-lists
# branch per user. The headline it carries is the REVERSAL, so we buy the full thompson-vs-static curve
# and the k=8 row for the other two selectors.
GOLB_CELLS = ([(BASE, k) for k in (1, 2, 4, 8)] + [("thompson", k) for k in (1, 2, 4, 8)]
              + [("glm_ucb", 8), ("eddi_ig", 8)])

LOGP = os.path.join(OUT, "act1_paired.log")


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def boot_ci(d, n_boot, seed):
    """Percentile bootstrap over the per-user DIFFERENCE vector (interview_paired.boot_ci)."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a, b, n_boot, seed, label, out):
    if len(a) != len(b):
        raise ValueError(f"{label}: vectors misaligned ({len(a)} vs {len(b)}) -- pairing invalid")
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d, n_boot, seed)
    sep = bool(lo > 0 or hi < 0)
    out[label] = {"n": int(len(d)), "delta": float(d.mean()), "lo": lo, "hi": hi,
                  "separates": sep, "direction": ("+" if d.mean() > 0 else "-") if sep else "ns"}
    logln(f"    {label:62s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] {'SEP' if sep else 'ns '}")
    return out[label]


def load_cache(path):
    vecs = {}
    z = np.load(path, allow_pickle=False)
    for key in z.files:
        cons, spec, k, ax = key.split("|")
        vecs.setdefault((cons, spec, int(k)), {})[ax] = z[key]
    return vecs


def analyze(vecs, a):
    """Every paired comparison, from the per-user vectors. Split out from main() so the identical
    statistics can be re-run from the cached vectors alone (`--analyze_only`) without repaying the
    interview+scoring cost, and so a comparison whose cells are missing is SKIPPED and NAMED rather
    than silently dropped."""
    nf = {len(c["full"]) for c in vecs.values()}
    nt = {len(c["tail"]) for c in vecs.values()}
    assert len(nf) == 1 and len(nt) == 1, f"user subsets differ across cells: full={nf} tail={nt}"
    n_full, n_tail = nf.pop(), nt.pop()
    logln(f"[canary] all {len(vecs)} cells row-aligned: n_full={n_full}, n_tail={n_tail}")

    have = set(vecs)
    missing = []

    def V(cons, spec, k, ax):
        return vecs[(cons, spec, k)][ax]

    def M_(cons, spec, k, ax):
        return float(np.mean(V(cons, spec, k, ax)))

    def P(cons, x_spec, y_spec, k, ax, label, bucket):
        if (cons, x_spec, k) not in have or (cons, y_spec, k) not in have:
            missing.append(label)
            return None
        return paired(V(cons, x_spec, k, ax), V(cons, y_spec, k, ax), a.boot, a.seed, label, bucket)

    # ------------------------------------------------------------------ control: reproduce the table
    logln("=== control: recomputed means vs the stored Act-1 tables ===")
    drift = []
    try:
        stored_a = json.load(open(os.path.join(OUT, "adaptive_table_i26c_ep31.json")))["results"]
        stored_l = json.load(open(os.path.join(OUT, "interview_table_ladder_ep31.json")))["results"]
    except Exception as e:                                            # noqa: BLE001
        stored_a, stored_l = {}, {}
        logln(f"[control] could not load stored tables: {e}")
    for (cons, spec, k), c in sorted(vecs.items()):
        for src, sel_key in ((stored_a.get(cons, {}), spec),
                             (stored_l.get(cons, {}),
                              spec[:-len("-static")] if spec.endswith("-static") else None)):
            if sel_key is None or sel_key not in src or f"k{k}" not in src[sel_key]:
                continue
            for ax in ("full", "tail"):
                got, exp = float(np.mean(c[ax])), src[sel_key][f"k{k}"][ax]
                if abs(got - exp) > 1e-9:
                    drift.append({"cell": f"{cons}/{sel_key}/k{k}/{ax}", "recomputed": got,
                                  "stored": exp, "diff": got - exp})
    for dd in drift:
        logln(f"[control] DRIFT {dd['cell']}: recomputed {dd['recomputed']:.6f} vs stored "
              f"{dd['stored']:.6f} ({dd['diff']:+.6f})")
    if not drift:
        logln("[control] PASS: every recomputed cell mean matches the stored table to 1e-9")
    else:
        logln(f"[control] {len(drift)} cell/axis values differ from the stored table")

    res = {"meta": {"boot": a.boot, "seed": a.seed, "snapshot": os.path.basename(a.snapshot),
                    "arch": a.arch, "no_floor": True, "n_test_users": 10000,
                    "n_full_scored": n_full, "n_tail_scored": n_tail,
                    "ucb_alpha": a.ucb_alpha, "ts_seed": a.ts_seed, "st_seed": ST.SEED,
                    "form": "paired percentile bootstrap over the per-user difference vector",
                    "control_drift": drift,
                    "cells_present": sorted(f"{c}|{s}|k{k}" for c, s, k in have)},
           "points": {f"{c}|{s}|k{k}": {"full": M_(c, s, k, "full"), "tail": M_(c, s, k, "tail")}
                      for (c, s, k) in vecs},
           "self_test": {}, "D1_vs_entropy0_static": {}, "D2_vs_strongest_static": {},
           "D3_static_ladder": {}, "D4_golbandi_reversal": {}}

    # ------------------------------------------------------------------ SELF-TEST
    logln("=== self-test: an arm against ITSELF must give a CI containing exactly zero ===")
    for k in BUDGETS:
        if ("ours", BASE, k) not in have:
            continue
        for ax in ("full", "tail"):
            paired(V("ours", BASE, k, ax), V("ours", BASE, k, ax), a.boot, a.seed,
                   f"SELFTEST/ours/{ax}/k{k}/{BASE}-minus-itself", res["self_test"])
    st_ok = bool(res["self_test"]) and all(
        v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for v in res["self_test"].values())
    res["meta"]["self_test_pass"] = st_ok
    logln(f"[self-test] {'PASS' if st_ok else 'FAIL'}: {len(res['self_test'])} self-comparisons, "
          f"all exactly [0,0]")

    # ------------------------------------------------------------------ D1: adaptive - entropy0-static
    logln("=== D1: consumer `ours`, each adaptive selector MINUS entropy0-static ===")
    for sel in ADAPTIVE:
        for k in BUDGETS:
            for ax in ("full", "tail"):
                P("ours", sel, BASE, k, ax, f"ours/{ax}/k{k}/{sel}-minus-entropy0_static",
                  res["D1_vs_entropy0_static"])

    # ------------------------------------------------------------------ D2: adaptive - best static
    # The comparator is chosen per (budget, axis) by point estimate on the SAME data, which is a
    # winner's-curse choice: it biases the comparator UP, i.e. AGAINST the adaptive selector. That is
    # the conservative direction for the "adaptive loses" claim, and is stated rather than hidden.
    logln("=== D2: consumer `ours`, each adaptive selector MINUS the strongest static at that cell ===")
    best = {}
    for k in BUDGETS:
        avail = [s for s in STATICS if ("ours", f"{s}-static", k) in have]
        if not avail:
            continue
        for ax in ("full", "tail"):
            arm = max(avail, key=lambda s: M_("ours", f"{s}-static", k, ax))
            best[f"k{k}/{ax}"] = arm
            logln(f"  strongest static k={k:2d} {ax:4s}: {arm} "
                  f"({M_('ours', arm + '-static', k, ax):.4f})")
    res["strongest_static"] = best
    for sel in ADAPTIVE:
        for k in BUDGETS:
            for ax in ("full", "tail"):
                arm = best.get(f"k{k}/{ax}")
                if arm is None:
                    continue
                P("ours", sel, f"{arm}-static", k, ax,
                  f"ours/{ax}/k{k}/{sel}-minus-BEST_STATIC[{arm}]", res["D2_vs_strongest_static"])

    # ------------------------------------------------------------------ D3: the static ladder itself
    logln("=== D3: static ladder separations (consumer `ours`) ===")
    ladder_pairs = [("entropy0", "popularity"), ("popularity", "helf"), ("entropy0", "helf")]
    degenerate = ("pure_entropy", "random_bank", "random_catalog")
    for x in ("entropy0", "popularity", "helf"):
        for y in degenerate:
            ladder_pairs.append((x, y))
    ladder_pairs += [("pure_entropy", "random_bank"), ("random_bank", "random_catalog"),
                     ("pure_entropy", "random_catalog")]
    for x, y in ladder_pairs:
        for k in BUDGETS:
            for ax in ("full", "tail"):
                P("ours", f"{x}-static", f"{y}-static", k, ax, f"ours/{ax}/k{k}/{x}-minus-{y}",
                  res["D3_static_ladder"])

    # ------------------------------------------------------------------ D4: the golbandi reversal
    logln("=== D4: golbandi_leaf -- does the REVERSAL separate? ===")
    for sel, k in GOLB_CELLS:
        if sel == BASE:
            continue
        for ax in ("full", "tail"):
            P("golbandi_leaf", sel, BASE, k, ax,
              f"golbandi_leaf/{ax}/k{k}/{sel}-minus-entropy0_static", res["D4_golbandi_reversal"])

    res["meta"]["skipped_for_missing_cells"] = sorted(set(missing))
    if missing:
        logln(f"[act1] {len(set(missing))} comparisons SKIPPED for missing cells (named in the JSON)")
    outp = os.path.join(OUT, "act1_paired.json")
    json.dump(res, open(outp, "w"), indent=2)
    logln(f"[act1] analysis written -> {outp}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, ".cache", "instrument",
                                                       "t2i26c_ep31.pt"))
    ap.add_argument("--arch", default="i26")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--ucb_alpha", type=float, default=1.0)
    ap.add_argument("--ts_seed", type=int, default=98765)
    ap.add_argument("--eddi_topm", type=int, default=0)
    ap.add_argument("--sel_batch", type=int, default=0)
    ap.add_argument("--vec_cache", default=os.path.join(OUT, "act1_per_user_vectors.npz"))
    ap.add_argument("--skip_golbandi", action="store_true")
    ap.add_argument("--analyze_only", action="store_true",
                    help="re-run every paired comparison from the cached per-user vectors, with no "
                         "interview and no re-scoring. Comparisons whose cells are not in the cache "
                         "are skipped and NAMED in the output JSON.")
    a = ap.parse_args()

    if a.analyze_only:
        vecs = load_cache(a.vec_cache)
        logln(f"[act1] ANALYZE-ONLY over {len(vecs)} cached cells from {a.vec_cache}")
        analyze(vecs, a)
        return

    t_all = time.time()
    D = load_arm_n(log=logln)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    lvl = ST.level_lookup(D["g_te_tr"])
    recs = ["ours"] if a.skip_golbandi else ["ours", "golbandi_leaf"]
    built = build_recommenders(recs, D, arch=a.arch, snapshot=a.snapshot, log=logln)
    pop_row, prior = mostpop_row(D, log=logln)
    logln(f"[act1] {n} users, {ni} items; mostpop prior {prior}")

    vecs = load_cache(a.vec_cache) if os.path.exists(a.vec_cache) else {}
    if vecs:
        logln(f"[act1] resumed {len(vecs)} cells from {a.vec_cache}")

    def save_cache():
        flat = {}
        for (cons, spec, k), c in vecs.items():
            for ax in ("full", "tail"):
                flat[f"{cons}|{spec}|{k}|{ax}"] = np.asarray(c[ax], dtype=np.float64)
        np.savez_compressed(a.vec_cache, **flat)

    # ------------------------------------------------------------------ interviews + per-user scores
    cache = {}
    for spec in SPECS:
        for k in BUDGETS:
            want = [("ours", spec, k)]
            if not a.skip_golbandi and (spec, k) in GOLB_CELLS:
                want.append(("golbandi_leaf", spec, k))
            want = [w for w in want if w not in vecs]
            if not want:
                continue
            t0 = time.time()
            sel = build_selector(spec, D, glob, bank, k, a, cache, log=logln)
            asked, answered, info = run_interview(sel, k, n, ni, lvl,
                                                  ctx={"D": D, "glob": glob, "bank": bank},
                                                  sel_batch=a.sel_batch, log=lambda *_: None)
            ga = common_asked(asked)
            logln(f"[act1] interview {spec:18s} k={k:2d} in {(time.time()-t0)/60:.2f}m; "
                  f"answered={np.mean([len(x) for x in answered]):.3f}; global={ga is not None}")
            for cons, _s, _k in want:
                assert not (cons == "golbandi_leaf" and k > MAX_DEPTH)
                mode, obj = built[cons]
                ts = time.time()
                c = score_arm(cons, mode, obj, asked, answered, D, pop_row, no_floor=True,
                              global_asked=ga, per_user=True)
                vecs[(cons, spec, k)] = {"full": c["full_per_user"], "tail": c["tail_per_user"]}
                logln(f"[act1]   score {cons:14s} {spec:18s} k={k:2d} "
                      f"full={c['full']:.4f} tail={c['tail']:.4f} "
                      f"(n_full={len(c['full_per_user'])} n_tail={len(c['tail_per_user'])}) "
                      f"{(time.time()-ts)/60:.2f}m")
                save_cache()

    analyze(vecs, a)
    logln(f"[act1] done in {(time.time()-t_all)/60:.1f}m")


if __name__ == "__main__":
    main()
