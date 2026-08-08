r"""run_userknn_proper.py -- the three conditions that make the user-kNN native row citable (#88).

The row on record is .2815/.1396 against a binary .3756/.2377, and it is NOT citable, for three reasons
found on 2026-08-07/08. This runs all three in one sequential job.

(1) LAMBDA WAS PINNED AT THE GRID EDGE. run_native_userknn.py selected lam=400.0 = LAM_GRID[-1]. It logs
    a warning for exactly this and the warning was ignored. Validation NDCG was still climbing
    (.2503 .2663 .2713 .2737 .2754 .2763), so the recorded row is not the method's best. Grid extended to
    3200; if the optimum is still at the edge, the row stays unquotable and the log says so.

(2) THE ZERO-EVIDENCE FALLBACK WAS ARGUED, NOT MEASURED. golbandi_native allocates np.zeros and leaves a
    cell with no rating neighbour at 0 -- published behaviour (an unsupported item falls back to the user
    mean, which is zero in centred space), but the consequence is that a zero-evidence item outranks any
    item whose neighbours disliked it, ties breaking by item id. The last run recorded 89.9% of CELLS at
    exactly 0. The cell fraction is the wrong statistic: what matters is how much of the REALISED TOP-10
    is fallback, and what the tail number becomes with those items excluded. Both measured here, on all
    10,000 test users rather than a 2,000-user diagnostic slice.

(3) THE BINARY TWIN WAS VOID. golbandi_native_binctrl.json recorded ndcg@10 = 7.8e-05 because the control
    inherited centre=True, and row-centring a binary matrix subtracts a row mean of 1 from values of 1 --
    every entry becomes 0. Deleted; centring is now forced off for the control and fit() raises on an
    annihilated input. The twin is re-run here through the SAME estimator code as the graded arm, so a
    result near the .3756 binary row validates the code path and a result far from it indicts it.

    Its lambda is tuned on its OWN validation sweep. Carrying the graded arm's lambda into the binary twin
    would be the same carried-constant-across-a-contract-change failure that produced the void control in
    the first place, five times over on this project now.

COST ~3.5 h sequential (two 9-point validation sweeps plus four full test evaluations). Per HARD RULE
11(a) every cell is written to disk the moment it completes, so a crash loses one cell, not the run.

  python src/baselines/run_userknn_proper.py
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

import metrics as M
import pop
import golbandi_native
from arm_n import load_arm_n
from graded_data import load_graded_val

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
OUTP = os.path.join(OUTDIR, "userknn_proper_armn.json")
LAM_GRID = [8.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]
# Batch 250, not the usual 500. golbandi_native.predict materialises a DENSE (batch x n_train_users)
# similarity block -- 500 x 140,768 float32 = 281 MB per batch. This run was launched, at the author's
# explicit instruction, alongside a live adaptive_harness job with only ~4.4 GB free, so the batch is
# halved to cap that block at ~141 MB. Batching does not affect any reported number.
BATCH = 250
STATE = {}


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "userknn_proper.log"), "a") as f:
        f.write(line + "\n")


def persist():
    """HARD RULE 11(a): write after every cell, never one dump at the end."""
    json.dump(STATE, open(OUTP, "w"), indent=2)


def sweep(name, train, va_in, va_te, hm, k, centre):
    """Validation sweep. Returns (lam, val_full, at_edge)."""
    best = (None, -1.0)
    STATE.setdefault("sweeps", {})[name] = []
    for lam in LAM_GRID:
        t0 = time.time()
        prd = golbandi_native.fit(train, train.shape[1],
                                  args=argparse.Namespace(k=k, lam=lam, centre=centre),
                                  log=lambda *_: None)
        m = M.evaluate(prd, va_in, va_te, batch_size=BATCH, head_mask=hm)
        logln(f"  [{name}] lam={lam:7.1f} val full={m['ndcg@10']:.4f} tail={m['tail_ndcg@10']:.4f} "
              f"({(time.time()-t0)/60:.1f}m)")
        STATE["sweeps"][name].append({"lam": lam, "val_full": m["ndcg@10"],
                                      "val_tail": m["tail_ndcg@10"]})
        persist()
        if m["ndcg@10"] > best[1]:
            best = (lam, m["ndcg@10"])
    lam, val = best
    edge = lam in (LAM_GRID[0], LAM_GRID[-1])
    logln(f"[{name}] val-selected lam={lam} (val {val:.4f})"
          + ("   *** STILL AT THE GRID EDGE -- row remains unquotable ***" if edge else ""))
    return lam, val, edge


def fallback_diagnostics(predict, te_in, pool, te_te, hm, n_users):
    """How much of the REALISED top-10 is the zero-evidence fallback, and what the tail is without it.

    Measured on all test users. Two numbers the cell-level 89.9% cannot give: occupancy of the ranked
    list, and the tail metric with fallback items removed from contention entirely.
    """
    occ, n_zero_top1, seen = [], 0, 0
    for st in range(0, n_users, BATCH):
        S = predict(te_in[st:st + BATCH])
        P = pool[st:st + BATCH].tocsr()
        S[P.nonzero()] = -np.inf
        top = np.argpartition(-S, 10, axis=1)[:, :10]
        vals = np.take_along_axis(S, top, axis=1)
        occ.append((vals == 0.0).mean(axis=1))
        best1 = S.argmax(axis=1)
        n_zero_top1 += int((S[np.arange(S.shape[0]), best1] == 0.0).sum())
        seen += S.shape[0]
    occ = np.concatenate(occ)
    logln(f"[G3] realised top-10 that is zero-evidence fallback: {occ.mean():.2%} of slots; "
          f"users whose RANK-1 item has no neighbour evidence: {n_zero_top1}/{seen} "
          f"({n_zero_top1/max(seen,1):.2%})")

    def predict_no_fallback(X):
        S = predict(X)
        S[S == 0.0] = -np.inf          # remove zero-evidence items from contention entirely
        return S
    r = M.evaluate(predict_no_fallback, te_in, te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    logln(f"[G3] with fallback items EXCLUDED: full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")
    return {"top10_fallback_occupancy": float(occ.mean()),
            "rank1_no_evidence_frac": n_zero_top1 / max(seen, 1),
            "excl_full": r["ndcg@10"], "excl_tail": r["tail_ndcg@10"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    a0 = ap.parse_args()

    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]
    n_users = D["te_tr"].shape[0]

    pr_pop = pop.fit(D["train"], ni, log=logln)
    rp = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    canon = json.load(open(os.path.join(OUTDIR, "pop.json")))["N"]
    ok = abs(rp["ndcg@10"] - canon["ndcg@10"]) <= 1e-4
    logln(f"[G1] MostPop this path {rp['ndcg@10']:.4f} vs arm-N {canon['ndcg@10']:.4f} -> "
          f"{'PASS' if ok else '*** FAIL ***'}")
    if not ok:
        raise SystemExit("[G1] eval path does not reproduce MostPop; fix wiring first")
    STATE["gates"] = {"mostpop": bool(ok)}
    persist()

    va_tr_bin, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)

    # ---- ARM 1: the published contract, mean-centred ratings, lambda extended -----------------------
    logln("--- arm 1/2: mean-centred ratings (published contract), lambda grid extended to 3200 ---")
    lam_g, val_g, edge_g = sweep("graded", D["g_train"], g_va, va_te, hm, a0.k, centre=True)
    prd = golbandi_native.fit(D["g_train"], ni, args=argparse.Namespace(k=a0.k, lam=lam_g, centre=True),
                              log=logln)
    r_g = M.evaluate(prd, D["g_te_tr"], te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    logln(f"[row] user-kNN native (mean-centred ratings): full={r_g['ndcg@10']:.4f} "
          f"tail={r_g['tail_ndcg@10']:.4f}")
    STATE["graded"] = {"lam": lam_g, "val_full": val_g, "lam_at_edge": edge_g,
                       "ndcg@10": r_g["ndcg@10"], "tail_ndcg@10": r_g["tail_ndcg@10"],
                       "ndcg@100": r_g["ndcg@100"]}
    persist()

    STATE["graded"]["fallback"] = fallback_diagnostics(prd, D["g_te_tr"], pool, te_te, hm, n_users)
    persist()

    # ---- ARM 2: the binary twin, same estimator code, centring OFF ----------------------------------
    logln("--- arm 2/2: BINARY TWIN through the same estimator code, uncentred ---")
    lam_b, val_b, edge_b = sweep("binary", D["train"], va_tr_bin, va_te, hm, a0.k, centre=False)
    prb = golbandi_native.fit(D["train"], ni, args=argparse.Namespace(k=a0.k, lam=lam_b, centre=False),
                              log=logln)
    r_b = M.evaluate(prb, D["te_tr"], te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    ref = json.load(open(os.path.join(OUTDIR, "golbandi_node.json")))["N"]
    gap = r_b["ndcg@10"] - ref["ndcg@10"]
    logln(f"[twin] binary through the native path: full={r_b['ndcg@10']:.4f} tail={r_b['tail_ndcg@10']:.4f}")
    logln(f"[twin] vs the node recommender's binary row {ref['ndcg@10']:.4f}: {gap:+.4f} -> "
          + ("code path VALIDATED (within 0.005)" if abs(gap) <= 0.005 else
             "OUTSIDE 0.005 -- the two implementations differ by more than noise; say so, do not "
             "present them as the same estimator on different inputs"))
    STATE["binary_twin"] = {"lam": lam_b, "val_full": val_b, "lam_at_edge": edge_b,
                            "ndcg@10": r_b["ndcg@10"], "tail_ndcg@10": r_b["tail_ndcg@10"],
                            "node_reference": {"ndcg@10": ref["ndcg@10"],
                                               "tail_ndcg@10": ref["tail_ndcg@10"]},
                            "gap_vs_node": gap, "code_path_validated": bool(abs(gap) <= 0.005)}

    d_full = STATE["graded"]["ndcg@10"] - r_b["ndcg@10"]
    logln("=" * 78)
    logln(f"[verdict] graded minus binary THROUGH ONE CODE PATH: {d_full:+.4f} full, "
          f"{STATE['graded']['tail_ndcg@10'] - r_b['tail_ndcg@10']:+.4f} tail")
    logln("[verdict] citable? " + ("YES" if not (edge_g or edge_b) else
                                   "NO -- a lambda is still at the grid edge"))
    logln("=" * 78)
    STATE["graded_minus_binary_one_codepath"] = {"full": d_full,
                                                 "tail": STATE["graded"]["tail_ndcg@10"] - r_b["tail_ndcg@10"]}
    STATE["citable"] = not (edge_g or edge_b)
    persist()
    logln(f"done -> {OUTP}")


if __name__ == "__main__":
    main()
