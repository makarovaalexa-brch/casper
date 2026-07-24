"""premium_vs_k.py -- GRADED PREMIUM AS A FUNCTION OF EVIDENCE SIZE (the likely headline figure).

On the frozen ep4 tower snapshot, VAL cohort, PARITY MASK throughout (only liked fold-in items masked,
every arm): at k in {2, 4, 8, 16, full}, with FIXED random k-subsets sampled from the ALL-BANDS token
set (truncate_graded; one draw shared by every arm at that k):
  arm (i)   true graded tokens                         (sids, real levels)
  arm (ii)  membership-only: same items, levels := 4 stars (values ERASED)
  arm (iii) likes-only subset of the SAME k draw       (non-liked tokens dropped; effective size logged)
  arm (iv)  flip: levels mirrored around 3 stars       (the sign-value curve)
Per k: full+tail NDCG@10 per arm; paired per-user deltas with bootstrap CI95:
  values premium  = (i) - (ii)      [what the LEVELS add on the same reveal set]
  reveal premium  = (ii) - (iii)    [what the extra non-like ITEMS add, values erased]
  flip delta      = (iv) - (i)      [sign fidelity as a function of evidence size]
Output: experiments/battery/premium_vs_k.json + printed table. Scoring only; thread-cap OMP=6.
Subset seeds (documented): k2=COLD_SEED, k8=COLD_SEED+1 (matching every earlier cold protocol),
k4=COLD_SEED+2, k16=COLD_SEED+3.
"""
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)
from run_battery_phaseA import build_real_ctx, eval_tokens, bootstrap_ci, rows_to_csr, SNAP_DEFAULT
from train_tower_t2 import truncate_graded, COLD_SEED, NLEV, LIKE_MIN_LEVEL, log

OUTP = os.path.join(_ROOT, "experiments", "battery", "premium_vs_k.json")
KSEEDS = {2: COLD_SEED, 4: COLD_SEED + 2, 8: COLD_SEED + 1, 16: COLD_SEED + 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    args = ap.parse_args()
    ctx = build_real_ctx(args.snapshot)
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    # all-bands carrier (levels+1) for subset sampling
    from scipy import sparse
    rr, cc, dd = [], [], []
    for r in range(ctx.n):
        s, l = ctx.allb[r]
        rr.extend([r] * len(s)); cc.extend(s.tolist()); dd.extend((l + 1.0).tolist())
    Lv = sparse.csr_matrix((np.asarray(dd, np.float32), (rr, cc)), shape=(ctx.n, ctx.ni))

    def tokens_from_carrier(L):
        return [(L[i].indices.astype(np.int64), (L[i].data - 1).astype(np.int64))
                for i in range(ctx.n)]

    def arms_from_tokens(toks):
        true_t = toks
        memb_t = [(s, np.full(len(l), 7, np.int64)) for s, l in toks]            # 4 stars
        like_t = [(s[l >= LIKE_MIN_LEVEL], l[l >= LIKE_MIN_LEVEL]) for s, l in toks]
        flip_t = [(s, np.clip(10 - l, 0, NLEV - 1)) for s, l in toks]
        return {"true": true_t, "memb4": memb_t, "likes": like_t, "flip": flip_t}

    conds = {}
    for k, seed in sorted(KSEEDS.items()):
        conds[str(k)] = arms_from_tokens(tokens_from_carrier(truncate_graded(Lv, k, seed)))
    conds["full"] = arms_from_tokens(ctx.allb)

    out = {"snapshot": os.path.basename(args.snapshot), "mask": "PARITY (liked fold-in only)",
           "seeds": {str(k): s for k, s in KSEEDS.items()}, "per_k": {}}
    for kname, arms in conds.items():
        t0 = time.time(); row = {"arms": {}, "deltas": {}}
        vecs = {}
        for aname, toks in arms.items():
            f, t = eval_tokens(ctx, toks, ctx.va_tr, rows=rows)
            vecs[aname] = f
            row["arms"][aname] = {"full@10": float(np.nanmean(f[rows])),
                                  "tail@10": float(np.nanmean(t[rows]))}
        eff = np.array([len(arms["likes"][r][0]) for r in rows], float)
        row["likes_effective_size"] = {"mean": float(eff.mean()),
                                       "frac_zero": float((eff == 0).mean())}
        for dname, (a, b) in (("values_premium(true-memb4)", ("true", "memb4")),
                              ("reveal_premium(memb4-likes)", ("memb4", "likes")),
                              ("flip_delta(flip-true)", ("flip", "true"))):
            m, ci, nn_ = bootstrap_ci(vecs[a][rows] - vecs[b][rows])
            row["deltas"][dname] = {"mean": m, "ci95": ci, "n": nn_}
        row["seconds"] = round(time.time() - t0, 1)
        out["per_k"][kname] = row
        os.makedirs(os.path.dirname(OUTP), exist_ok=True)
        json.dump(out, open(OUTP, "w"), indent=2)
        log(f"[k={kname}] " + " ".join(f"{a}={row['arms'][a]['full@10']:.4f}" for a in arms) +
            f" | values={row['deltas']['values_premium(true-memb4)']['mean']:+.4f}"
            f" reveal={row['deltas']['reveal_premium(memb4-likes)']['mean']:+.4f}"
            f" flip={row['deltas']['flip_delta(flip-true)']['mean']:+.4f}"
            f" ({row['seconds']/60:.1f}m)")

    print("\n=== GRADED PREMIUM vs EVIDENCE SIZE (tower ep4, VAL, parity mask; full@10) ===")
    print(f"{'k':>5} {'true':>8} {'memb4':>8} {'likes':>8} {'flip':>8} "
          f"{'values_prem':>12} {'reveal_prem':>12} {'flip_delta':>11} {'likes_eff':>9}")
    for kname in list(map(str, sorted(KSEEDS))) + ["full"]:
        r = out["per_k"][kname]; a = r["arms"]; d = r["deltas"]
        print(f"{kname:>5} {a['true']['full@10']:8.4f} {a['memb4']['full@10']:8.4f} "
              f"{a['likes']['full@10']:8.4f} {a['flip']['full@10']:8.4f} "
              f"{d['values_premium(true-memb4)']['mean']:+12.4f} "
              f"{d['reveal_premium(memb4-likes)']['mean']:+12.4f} "
              f"{d['flip_delta(flip-true)']['mean']:+11.4f} "
              f"{r['likes_effective_size']['mean']:9.1f}")
    print(f"JSON -> {OUTP}")


if __name__ == "__main__":
    main()
