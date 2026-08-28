r"""M6z -- WHERE DO REAL REDDIT REQUESTS LAND IN THE TRAIN-FITTED TASTE CLUSTERS?

`m1_reddit_clusters.json` already showed that real requests are NOT spread evenly over taste
clusters -- four of m1's twenty held 64.5% of requests but only 21% of users. That table was built
against m1's VALIDATION-side clustering, so its cluster ids do not correspond to the train-fitted
partition the teachers will use. This re-derives it against `m6y`'s centroids.

WHY IT MATTERS. It replaces two weak choices with one motivated by data:

  * WHICH clusters get a teacher. "Largest non-era-like" is arbitrary and picked c16, whose top films
    are an Australian crime flick, a stand-up tour doc and a 1981 Tarzan -- and whose m1 request
    share was 0.004. Fitting teachers where real requesters actually land covers most of the real
    population for a fraction of the compute: 20 teachers is ~60h, the top few are ~12-18h.
  * WHAT the Reddit corpus contributes. Until now it shaped the openers (film 0.6303 / genre 0.3979 /
    tone 0.2135 / popularity p93.3). It can also say which taste neighbourhoods real requests come
    from -- a stronger claim than "requesters name these six words", because it uses the requests
    themselves rather than a vocabulary we chose.

THE REQUESTS ARE FOLDED, NOT RE-EMBEDDED. `z_requests.npz` holds 3,078 real requests already passed
through the SAME frozen encoder from the SAME cold start as the user states (m1 Stage A). They are
projected onto the SAME era axis `m6y` removed, then assigned to the nearest `m6y` centroid. Any
other route would compare two geometries.

⚠ NEVER votes as graded labels; NEVER a Reddit API call; NO usernames anywhere in the output.

**THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import json
import time

import numpy as np

B = os.path.join(_R, "experiments", "battery")
M1 = os.path.join(B, "m1")
M6 = os.path.join(B, "m6")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6z -- request share per train-fitted cluster")
    ap.add_argument("--labels", default=os.path.join(M6, "m6y_clusters_train_k20.npz"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"

    cl = np.load(a.labels)
    C = cl["centroids"].astype(np.float64)         # already L2-normalised by m6y
    d_era = cl["d_era"].astype(np.float64)
    tr_lab = cl["train_labels"]
    k = C.shape[0]

    zr = np.load(os.path.join(M1, "z_requests.npz"))
    Zq = zr["z"].astype(np.float64)
    n_items = zr["n_items"]
    log(f"[requests] {len(Zq)} real requests, already folded through the frozen encoder (m1 Stage A)")
    log(f"[requests] items named per request: mean {n_items.mean():.2f} "
        f"median {np.median(n_items):.0f}; {int((n_items == 0).sum())} name none")

    # same axis removal, same normalisation, same centroids -> one geometry
    Zqd = Zq - np.outer(Zq @ d_era, d_era)
    Zqn = Zqd / (np.linalg.norm(Zqd, axis=1, keepdims=True) + 1e-12)
    lab_q = (Zqn @ C.T).argmax(1)

    tr_share = np.bincount(tr_lab, minlength=k) / len(tr_lab)
    rq_share = np.bincount(lab_q, minlength=k) / len(lab_q)
    meta = {c["cluster"]: c for c in
            json.load(open(a.labels.replace(".npz", ".json"), encoding="utf-8"))["clusters"]}

    order = np.argsort(-rq_share)
    log("=" * 92)
    log(f"{'cl':>3} {'n_train':>8} {'train%':>8} {'request%':>9} {'lift':>7} {'era?':>6}")
    cum = 0.0
    rows = []
    for c in order:
        lift = rq_share[c] / tr_share[c] if tr_share[c] > 0 else float("nan")
        m = meta.get(int(c), {})
        cum += rq_share[c]
        rows.append({"cluster": int(c), "n_train": int(m.get("n_train", 0)),
                     "train_share": float(tr_share[c]), "request_share": float(rq_share[c]),
                     "lift": float(lift), "era_like": bool(m.get("era_like", False)),
                     "cum_request_share": float(cum)})
        log(f"{int(c):>3} {int(m.get('n_train', 0)):>8} {tr_share[c]:>8.3f} {rq_share[c]:>9.3f} "
            f"{lift:>7.2f} {'YES' if m.get('era_like') else 'no':>6}")
    log("=" * 92)
    for n in (3, 4, 5, 6, 8):
        log(f"[coverage] top {n} clusters by request share cover "
            f"{rq_share[order[:n]].sum():.1%} of real requests and "
            f"{tr_share[order[:n]].sum():.1%} of users")
    cand = [r for r in rows if not r["era_like"]][:6]
    log(f"[gate] highest-request-share NON-era-like clusters: "
        + ", ".join(f"c{r['cluster']}({r['request_share']:.3f})" for r in cand[:4]))
    json.dump({"analysis": "m6z_request_share", "test_touched": False, "k": int(k),
               "n_requests": int(len(Zq)), "rows": rows},
              open(os.path.join(M6, "m6z_request_share.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
