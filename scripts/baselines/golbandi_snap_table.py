r"""golbandi_snap_table.py -- tabulate SNAP-5's per-depth JSON into the three tables the snap owes:
(1) adaptive vs each static comparator by depth on RMSE (the snap), (2) the same on NDCG@10 with the
like-rate leaf (the inversion question), (3) leaf occupancy by depth (the depth-vs-budget mechanism).

Seeds are averaged and the per-seed spread is printed, because a shape claim that holds on one seed
and flips on another is not a shape claim.
"""
import argparse
import json
import os
import sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def paired_ci(res, seeds, key_a, key_b, depth, field, boot=5000, seed=98765):
    """Paired-over-USERS bootstrap of (adaptive - static), pooled across seeds.

    RMSE is pooled correctly (sqrt of the resampled total SE over the resampled total count), not
    averaged over per-user RMSEs. Pairing is on the user, which is what makes the CI honest: both
    arms see the same cold user with the same fold-in side, so the between-user variance -- by far
    the largest term -- cancels. `paired-test-kills-the-tie-claim` is the reason this exists."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in seeds:
        if key_a not in res[s] or key_b not in res[s]:
            continue
        pa, pb = res[s][key_a]["per_user"], res[s][key_b]["per_user"]
        if field == "rmse":
            parts.append((np.asarray(pa["se"])[:, depth], np.asarray(pa["n"])[:, depth],
                          np.asarray(pb["se"])[:, depth], np.asarray(pb["n"])[:, depth]))
        else:
            parts.append((np.asarray(pa["ndcg"])[:, depth], None,
                          np.asarray(pb["ndcg"])[:, depth], None))
    if not parts:
        return None
    if field == "rmse":
        sa = np.concatenate([p[0] for p in parts]); na = np.concatenate([p[1] for p in parts])
        sb = np.concatenate([p[2] for p in parts])
        point = np.sqrt(sa.sum() / na.sum()) - np.sqrt(sb.sum() / na.sum())
        idx = rng.integers(0, len(sa), size=(boot, len(sa)))
        d = (np.sqrt(sa[idx].sum(1) / na[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / na[idx].sum(1)))
    else:
        ga = np.concatenate([p[0] for p in parts]); gb = np.concatenate([p[2] for p in parts])
        m = ~(np.isnan(ga) | np.isnan(gb))
        ga, gb = ga[m], gb[m]
        point = float(ga.mean() - gb.mean())
        idx = rng.integers(0, len(ga), size=(boot, len(ga)))
        d = ga[idx].mean(1) - gb[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(_ROOT, "experiments", "results", "snap5",
                                                   "golbandi_snap_ml100k_d8.json"))
    ap.add_argument("--lam1", type=float, default=None, help="restrict to one shrinkage value")
    a = ap.parse_args()
    blob = json.load(open(a.path))
    res = blob["results"]
    seeds = sorted(res.keys(), key=int)
    keys = sorted({k for s in seeds for k in res[s]})
    depth = blob["meta"]["depth"]

    def cells(key, field):
        """-> (mean_by_depth, per_seed_by_depth)"""
        rows = []
        for s in seeds:
            if key not in res[s]:
                continue
            pd = res[s][key]["per_depth"]
            rows.append([pd[str(d)][field] for d in range(depth + 1)])
        arr = np.asarray([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
        return np.nanmean(arr, axis=0), arr

    for kind, field, better in (("mean", "rmse", "lower"), ("like_rate", "ndcg10", "higher")):
        lams = sorted({float(k.split("lam1=")[1].split("|")[0]) for k in keys})
        for lam in lams:
            if a.lam1 is not None and lam != a.lam1:
                continue
            sel = [k for k in keys if k.endswith("|" + kind) and f"lam1={lam:g}|" in k]
            if not sel:
                continue
            adapt = next((k for k in sel if k.startswith("adaptive")), None)
            stats = [k for k in sel if k.startswith("static_")]
            am, aa = cells(adapt, field)
            print(f"\n=== {field.upper()} ({kind} leaf), lam1={lam:g}, {len(seeds)} seeds, "
                  f"{better} is better ===")
            hdr = f"{'depth':>5} | {'ADAPTIVE':>9} |" + "".join(
                f" {k.split('|')[0].replace('static_',''):>10}" for k in stats) + \
                " | " + " ".join(f"{'d(' + k.split('|')[0].replace('static_','') + ')':>12}"
                                 for k in stats)
            print(hdr)
            for d in range(depth + 1):
                line = f"{d:>5} | {am[d]:>9.4f} |"
                deltas = []
                for k in stats:
                    sm, _ = cells(k, field)
                    line += f" {sm[d]:>10.4f}"
                    dd = (sm[d] - am[d]) if better == "lower" else (am[d] - sm[d])
                    deltas.append(dd)
                line += " | " + " ".join(f"{x:>+12.4f}" for x in deltas)
                print(line)
            wins = []
            for d in range(1, depth + 1):
                ok = True
                for k in stats:
                    sm, sa = cells(k, field)
                    per_seed_ok = all((sa[i, d] > aa[i, d]) if better == "lower"
                                      else (aa[i, d] > sa[i, d]) for i in range(len(aa)))
                    ok = ok and per_seed_ok
                wins.append((d, ok))
            print("  adaptive beats EVERY static on EVERY seed at depth: "
                  + ", ".join(str(d) for d, ok in wins if ok) + "   |  fails at: "
                  + (", ".join(str(d) for d, ok in wins if not ok) or "none"))
            print(f"  paired bootstrap, adaptive - static, pooled over seeds "
                  f"({'negative' if better == 'lower' else 'positive'} = adaptive wins):")
            for k in stats:
                row = []
                for d in range(1, depth + 1):
                    ci = paired_ci(res, seeds, adapt, k, d, field)
                    if ci is None:
                        continue
                    star = "*" if (ci[1] > 0) == (ci[2] > 0) else " "
                    row.append(f"d{d}:{ci[0]:+.4f}[{ci[1]:+.4f},{ci[2]:+.4f}]{star}")
                print(f"    vs {k.split('|')[0].replace('static_',''):>10}: " + "  ".join(row))

    # ---------------------------------------------------------------- shrinkage sweep
    lams = sorted({float(k.split("lam1=")[1].split("|")[0]) for k in keys})
    if len(lams) > 2:
        for kind, field, better in (("mean", "rmse", "lower"), ("like_rate", "ndcg10", "higher")):
            print(f"\n=== SHRINKAGE SWEEP: adaptive {field.upper()} by depth, and whether the "
                  f"adaptive-minus-BEST-static paired CI excludes 0 at every depth ===")
            print(f"{'lam1':>6} | " + " ".join(f"{'d' + str(d):>8}" for d in range(depth + 1))
                  + " | best-static@dmax | separates from ALL statics at every depth 1..dmax?")
            for lam in lams:
                sel = [k for k in keys if k.endswith("|" + kind) and f"lam1={lam:g}|" in k]
                adapt = next((k for k in sel if k.startswith("adaptive")), None)
                stats = [k for k in sel if k.startswith("static_")]
                if adapt is None or not stats:
                    continue
                am, _ = cells(adapt, field)
                bs = [cells(k, field)[0][depth] for k in stats]
                best_static = min(bs) if better == "lower" else max(bs)
                allsep = True
                for k in stats:
                    for d in range(1, depth + 1):
                        ci = paired_ci(res, seeds, adapt, k, d, field, boot=2000)
                        if ci is None or not ((ci[1] > 0) == (ci[2] > 0)):
                            allsep = False
                        elif (ci[0] > 0) == (better == "lower"):
                            allsep = False              # separated but in the WRONG direction
                print(f"{lam:>6.0f} | " + " ".join(f"{am[d]:>8.4f}" for d in range(depth + 1))
                      + f" | {best_static:>16.4f} | {'YES' if allsep else 'no'}")

    print("\n=== LEAF OCCUPANCY (adaptive tree, train users routed to each node), seed-averaged ===")
    ak = next(k for k in keys if k.startswith("adaptive"))
    occ = [res[s][ak]["occupancy"] for s in seeds if ak in res[s]]
    print(f"{'depth':>5} {'#nodes':>7} {'users/node mean':>16} {'median':>8} "
          f"{'#nodes<25u':>11} {'#nodes<10u':>11} {'still splitting':>16}")
    for d in range(depth + 1):
        vals = [o[str(d)] for o in occ if str(d) in o]
        if not vals:
            continue
        g = lambda f: float(np.mean([v[f] for v in vals]))
        print(f"{d:>5} {g('n_nodes'):>7.0f} {g('users_mean'):>16.1f} {g('users_median'):>8.1f} "
              f"{g('n_lt_25'):>11.0f} {g('n_lt_10'):>11.0f} {g('internal'):>16.0f}")

    print("\n=== COLD-USER-SIDE occupancy: train users behind the node a COLD user lands in ===")
    print(f"{'depth':>5} {'adaptive':>10} {'static_pop':>11} {'leafed-early frac (adapt)':>26}")
    sp = next((k for k in keys if k.startswith("static_popularity") and k.endswith("|mean")), None)
    am, _ = cells(next(k for k in keys if k.startswith("adaptive") and k.endswith("|mean")),
                  "node_users_mean")
    sm, _ = cells(sp, "node_users_mean")
    lf, _ = cells(next(k for k in keys if k.startswith("adaptive") and k.endswith("|mean")),
                  "leafed_early_frac")
    for d in range(depth + 1):
        print(f"{d:>5} {am[d]:>10.1f} {sm[d]:>11.1f} {lf[d]:>26.3f}")


if __name__ == "__main__":
    main()
