r"""M1 -- remove the RELEASE-ERA axis from the taste states, then re-cluster.

The era probe (`m1_era_probe.json`) measured PC1 at rho=+0.709 with item release year and the top
three directions (49% of variance) all era-loaded, while PC3 (5.3%) carries essentially none
(+0.036). So era is a large axis of this geometry and it is NOT taste: it reflects when a user was
active on MovieLens, which nobody can volunteer at ask time.

TARGETED removal, not PC surgery: we know every item's release year, so the era direction is
constructible. `d_era = Wd^T y` (y = standardised year over items) is the state direction whose
decoded item scores are maximally year-aligned; project it out of every state and leave the rest of
the geometry untouched. Dropping whole PCs would discard the real taste PC0 carries alongside era.

Writes de-era'd states; the clustering rerun consumes them. Read-only w.r.t. the recommender: no
NDCG, no targets, no TEST.
"""
import os, sys, csv, re, json
_H = os.path.dirname(os.path.abspath(__file__)); _R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)
import numpy as np
from scipy import stats

M1 = os.path.join(_R, "experiments", "battery", "m1")


def main():
    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    files = sorted((f for f in os.listdir(os.path.join(M1, "train")) if f.startswith("slice_")),
                   key=lambda x: int(x.split("_")[1]))
    Z = np.concatenate([np.load(os.path.join(M1, "train", f))["z"] for f in files], 0)
    print(f"[in] Z {Z.shape}")

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1deera")
    Wd = np.asarray(E["ctx"].Wd)                                   # (n_items, d)
    with open(os.path.join(_R, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    year = {}
    with open(os.path.join(_R, "data", "movielens", "movies.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"\((\d{4})\)\s*$", r["title"].strip())
            if m:
                year[int(r["movieId"])] = int(m.group(1))
    yr = np.array([year.get(s, np.nan) for s in sids], float)
    ok = ~np.isnan(yr)
    y = np.zeros(len(sids))
    y[ok] = (yr[ok] - yr[ok].mean()) / yr[ok].std()                # standardised, unknown years -> 0
    print(f"[years] {int(ok.sum())} items carry a year; mean {yr[ok].mean():.1f}")

    d_era = Wd.T @ y
    d_era /= np.linalg.norm(d_era)
    sc = d_era @ Wd.T
    rho, _p = stats.spearmanr(sc[ok], yr[ok])
    print(f"[era-axis] rho(decoded score, release year) = {rho:+.3f}   <- the direction being removed")

    proj = Z @ d_era
    Zd = Z - np.outer(proj, d_era)
    print(f"[remove] projection onto the era axis: mean {proj.mean():+.3f} sd {proj.std():.3f}; "
          f"state norm {np.linalg.norm(Z,axis=1).mean():.3f} -> {np.linalg.norm(Zd,axis=1).mean():.3f}")

    # how much era survives in the leading directions afterwards
    rng = np.random.default_rng(98765)
    sub = Zd[rng.permutation(len(Zd))[:20000]]
    _u, s, vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    after = []
    for i in range(4):
        r, _ = stats.spearmanr((vt[i] @ Wd.T)[ok], yr[ok])
        after.append({"pc": i, "var": float(var[i]), "rho_year": float(r)})
        print(f"[after pc{i}] var {var[i]:.3f}  rho(year) = {r:+.3f}")

    off = 0
    for f in files:
        d = np.load(os.path.join(M1, "train", f))
        m = d["z"].shape[0]
        np.savez_compressed(os.path.join(M1, "train", f.replace("slice_", "deera_")),
                            z=Zd[off:off + m].astype(np.float32), n_liked=d["n_liked"],
                            end_indptr=d["end_indptr"], end_indices=d["end_indices"],
                            lo=d["lo"], hi=d["hi"], C=d["C"])
        off += m
    assert off == len(Zd)
    json.dump({"analysis": "m1_deera", "era_axis_rho": float(rho), "after": after},
              open(os.path.join(M1, "m1_deera.json"), "w"), indent=1)
    print(f"[done] wrote {len(files)} deera_* slices")


if __name__ == "__main__":
    main()
