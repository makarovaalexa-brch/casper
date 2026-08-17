"""M1 diagnostic -- how much of the taste-state geometry is RELEASE ERA?

The A5 film check showed three of four train clusters defined by a window of years (1995-96,
2009-14, 2013-15). If the leading directions of the state space encode WHEN a user was active
rather than WHAT they like, the forest would route on join date -- which nobody can volunteer.

Method: decode each principal direction of the 140,768 train states through the SAME decoder the
recommender uses, then correlate the resulting per-item scores with the item's RELEASE YEAR. A
direction that encodes era shows a large |rho|. Read-only; nothing is scored, no NDCG, no targets.
"""
import os, sys, csv, re, json
_H = os.path.dirname(os.path.abspath(__file__)); _R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)
import numpy as np
from scipy import stats

def main():
    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    M1 = os.path.join(_R, "experiments", "battery", "m1")
    Z = np.concatenate([np.load(os.path.join(M1, "train", f))["z"]
                        for f in sorted(os.listdir(os.path.join(M1, "train")),
                                        key=lambda x: int(x.split("_")[1]) if x.startswith("slice_") else -1)
                        if f.startswith("slice_")], 0)
    print(f"[in] Z {Z.shape}")
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1era")
    Wd = np.asarray(E["ctx"].Wd)
    with open(os.path.join(_R, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    year = {}
    with open(os.path.join(_R, "data", "movielens", "movies.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"\((\d{4})\)\s*$", r["title"].strip())
            if m: year[int(r["movieId"])] = int(m.group(1))
    yr = np.array([year.get(s, np.nan) for s in sids], float)
    ok = ~np.isnan(yr)
    print(f"[years] {ok.sum()} of {len(sids)} items carry a release year")

    Zc = Z - Z.mean(0)
    # top principal directions of the state space
    _u, s, vt = np.linalg.svd(Zc[np.random.default_rng(98765).permutation(len(Zc))[:20000]],
                              full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    print("[pca] variance explained by the first 6 directions: "
          + ", ".join(f"{v:.3f}" for v in var[:6]))
    out = []
    for i in range(6):
        sc = vt[i] @ Wd.T
        rho, p = stats.spearmanr(sc[ok], yr[ok])
        out.append({"pc": i, "var": float(var[i]), "rho_year": float(rho), "p": float(p)})
        print(f"[pc{i}] var {var[i]:.3f}  rho(item score, release year) = {rho:+.3f}  (p={p:.1e})")
    # and the projection of each user's state on PC0 vs their cluster
    lab = np.load(os.path.join(M1, "m1_train_labels.npz"))["labels"]
    proj = Zc @ vt[0]
    print("[cluster] mean PC0 projection by cluster:")
    for c in range(int(lab.max()) + 1):
        print(f"   c{c} n={int((lab==c).sum())}  PC0 mean {proj[lab==c].mean():+.3f}")
    json.dump({"analysis": "m1_era_probe", "pcs": out}, open(os.path.join(M1, "m1_era_probe.json"), "w"), indent=1)
    print("[done]")

if __name__ == "__main__":
    main()
