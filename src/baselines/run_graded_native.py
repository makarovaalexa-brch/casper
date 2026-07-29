r"""run_graded_native.py -- re-run the ratings-native baselines faithful to their published design.

Golbandi's interview tree, RBMF / Functional-MF and TaNP are ratings-native by publication. The canonical
split keeps only r>3.5, so on it they receive binary support and 55.7% of the ratings their designs consume
are absent. This re-runs them on the graded catalogue ratings (graded_data.py) over the IDENTICAL user and
item sets, with held-out targets excluded, and scores them with the same metrics.evaluate against the same
binary targets -- so only the model INPUT changes and the numbers stay comparable to every other row.

  golbandi_node   value-agnostic: the node score is a mean over neighbour rows, which becomes a mean
                  RATING, and the user-user cosine is computed on graded rows. Fully faithful.
  rbmf_seed       value-agnostic in the seed step: the per-user ridge LS now regresses the user's actual
                  ratings on the frozen item factors instead of ones. SUBSTITUTION STATED: the frozen Y
                  still comes from the implicit iALS fit (as the existing row does); only the elicitation
                  seed step -- the mechanism the interview exercises -- becomes graded.
  tanp            needs its rating channel un-pinned and a retrain; run separately with --tanp.

  python src/baselines/run_graded_native.py            # golbandi + rbmf (closed form, minutes)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import metrics as M
import golbandi_node, rbmf_seed
from graded_data import load_graded
from run_ml25m_liang import compute_head_mask

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="golbandi_node_graded,rbmf_seed_graded")
    args = ap.parse_args()
    want = set(args.only.split(","))
    t0 = time.time()

    meta = M.load_meta(PROC); n_items = meta["n_items"]
    bin_train = M.load_train(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    hm, _ = compute_head_mask(bin_train, n_items)
    log(f"ruler: {n_items} items, {te_tr.shape[0]} test users, head={int(hm.sum())}")

    g_train, g_te_tr = load_graded(te_te, log=log)
    assert g_te_tr.multiply(te_te).nnz == 0, "graded fold-in overlaps held-out targets"
    log(f"graded fold-in carries {g_te_tr.nnz / te_tr.nnz:.2f}x the binary fold-in's interactions; "
        f"{float((g_te_tr.data <= 3.5).mean()):.1%} of them are sub-3.5 and absent from the split")

    out = {}
    if "golbandi_node_graded" in want:
        log("--- golbandi_node on graded ratings (fully faithful: node mean = mean rating) ---")
        pr = golbandi_node.fit(g_train, n_items, log=log)
        r = M.evaluate(pr, g_te_tr, te_te, batch_size=500, head_mask=hm)
        r["input_regime"] = "graded catalogue ratings, all bands, targets excluded"
        out["golbandi_node_graded"] = r
        log(f"[done] golbandi_node_graded: full@10={r['ndcg@10']:.4f} tail@10={r['tail_ndcg@10']:.4f} "
            f"ndcg@100={r['ndcg@100']:.4f}  (binary row was 0.3065/0.1895)")
        json.dump(r, open(os.path.join(OUTDIR, "golbandi_node_graded.json"), "w"), indent=2)

    if "rbmf_seed_graded" in want:
        log("--- rbmf_seed on graded ratings (seed step regresses real ratings; frozen Y still iALS) ---")
        pr = rbmf_seed.fit(bin_train, n_items, log=log)     # Y from the certified implicit fit
        r = M.evaluate(pr, g_te_tr, te_te, batch_size=500, head_mask=hm)
        r["input_regime"] = ("graded catalogue ratings in the per-user LS seed; frozen Y from the "
                             "implicit iALS fit (stated substitution)")
        out["rbmf_seed_graded"] = r
        log(f"[done] rbmf_seed_graded: full@10={r['ndcg@10']:.4f} tail@10={r['tail_ndcg@10']:.4f} "
            f"ndcg@100={r['ndcg@100']:.4f}  (binary row was 0.2534/0.1764)")
        json.dump(r, open(os.path.join(OUTDIR, "rbmf_seed_graded.json"), "w"), indent=2)

    log(f"complete in {(time.time() - t0)/60:.1f}m")


if __name__ == "__main__":
    main()
