r"""test_interview_curriculum.py -- pre-training verification of the interview curriculum.

RISK 7 in the design sheet: if the simulator's answered-per-asked distribution does not match evaluation,
the retrain re-creates the very train/test mismatch it exists to remove. So this is checked BEFORE any
training, not after.

Measured evaluation targets (arm-N protocol, 10,000 TEST users, k=8):
    HELF       1.6 / 8 answered      <- the HELD-OUT strategy
    entropy0   3.1 / 8
    popularity 3.1 / 8

Also asserts: no target ever appears in the asked set (leak check); the unseen channel is populated and
disjoint from the answered set; the k=0 bucket really is empty; regime proportions match the design.

  python src/instrument/test_interview_curriculum.py
"""
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from arm_n import load_arm_n
import interview_strategies as ST
from interview_curriculum import (StrategyFamily, make_interview_example, make_empty_example, draw,
                                  NAMED, P_FULL, P_INTERVIEW)

EVAL_ANSWERED_K8 = {"helf": 1.6, "entropy0": 3.1, "popularity": 3.1, "pure_entropy": 0.0}
TOL = 0.6           # absolute answers-per-8; the simulator must land in the same regime, not identically


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def profiles_from_graded(G, n_users, rng):
    """Train-user profiles straight from the cached graded train matrix: items / levels / vals / liked /
    disliked. Same shape as build_train_profiles produces, without re-reading the 25M-row csv."""
    G = G.tocsr()
    idx = rng.choice(G.shape[0], size=n_users, replace=False)
    out = []
    for u in idx:
        s, e = G.indptr[u], G.indptr[u + 1]
        sid = G.indices[s:e].astype(np.int64)
        lv = np.clip(np.rint(G.data[s:e] * 2).astype(np.int64) - 1, 0, 9)
        if len(sid) < 4:
            continue
        vals = ((lv.astype(np.float64) + 1.0) / 2.0 - 2.75) / 2.25
        out.append({"items": sid, "levels": lv, "vals": vals.astype(np.float32),
                    "liked": sid[lv >= 7], "disliked": sid[lv < 7]})
    return out


def main():
    D = load_arm_n(log=log)
    ni = D["n_items"]
    cnt = np.asarray(D["train"].sum(axis=0)).ravel().astype(np.float64)
    H, H0 = ST.entropies(D["g_train"], ni, float(D["g_train"].shape[0]))
    Hn = H / max(H.max(), 1e-9)
    Fn = np.log1p(cnt) / max(np.log1p(cnt).max(), 1e-9)
    helf = 2.0 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)

    rng = np.random.default_rng(0)
    users = profiles_from_graded(D["g_train"], 3000, rng)
    log(f"[test] {len(users)} train profiles; median size {int(np.median([len(u['items']) for u in users]))}")
    fam = StrategyFamily(cnt, H, H0, helf)
    ok = True

    # ---- 1. ANSWERED-RATE per NAMED strategy must match evaluation (risk 7)
    named = NAMED
    for name, w in named.items():
        rates = []
        for u in users:
            order = fam.order(np.random.default_rng(1), 8, weights=w)
            rated = set(u["items"].tolist())
            rates.append(sum(1 for i in order if int(i) in rated))
        got, want = float(np.mean(rates)), EVAL_ANSWERED_K8[name]
        hit = abs(got - want) <= TOL
        ok &= hit
        log(f"[test] answered@8 {name:11s}: simulator {got:.2f} vs eval {want:.2f} "
            f"-> {'PASS' if hit else 'FAIL'}")

    # ---- 2. The family must COVER the named corners (no strategy is withheld -- author ruling)
    near = {k: 0 for k in NAMED}
    for _ in range(4000):
        w = np.asarray(fam.sample_weights(rng)[:4])
        for k, wa in NAMED.items():
            if w[np.argmax(wa[:4])] > 0.55:
                near[k] += 1
    cov = all(v > 0 for v in near.values())
    log(f"[test] family covers each named corner (>0.55 mass): "
        f"{ {k: v for k, v in near.items()} } -> {'PASS' if cov else 'FAIL'}")
    ok &= cov

    # ---- 3. LEAK CHECK: a target must never be asked about, and unseen must be disjoint from answered
    leaks = overlap = n_unseen = n_ans = 0
    for u in users[:800]:
        ex = make_interview_example(u, rng, fam)
        if ex is None:
            continue
        inp_s, inp_l, inp_v, unseen, tg, negs = ex
        leaks += len(np.intersect1d(inp_s, tg))
        overlap += len(np.intersect1d(inp_s, unseen))
        n_unseen += len(unseen); n_ans += len(inp_s)
    log(f"[test] target-in-input leaks={leaks} answered/unseen overlap={overlap} "
        f"-> {'PASS' if leaks == 0 and overlap == 0 else 'FAIL'}")
    ok &= (leaks == 0 and overlap == 0)
    log(f"[test] channel volume over 800 examples: answered={n_ans} unseen={n_unseen} "
        f"(ratio {n_unseen / max(n_ans,1):.2f} unseen per answer)")
    ok &= n_unseen > 0

    # ---- 4. k=0 bucket really is empty, and still has targets to score
    e0 = [make_empty_example(u, rng, 0) for u in users[:200]]
    e0 = [x for x in e0 if x is not None]
    empt = all(len(x[0]) == 0 and len(x[4]) > 0 for x in e0)
    log(f"[test] k=0 bucket: {len(e0)} examples, all empty-input with targets "
        f"-> {'PASS' if empt else 'FAIL'}")
    ok &= empt

    # ---- 5. Regime proportions
    def fake_full(u, r):
        its = u["items"]
        keep = r.random(len(its)) >= 0.5
        if not keep.any():
            keep[0] = True
        tgt = np.setdiff1d(u["liked"], its[keep])
        return (its[keep], u["levels"][keep], u["vals"][keep], tgt, np.empty(0, np.int64)) \
            if len(tgt) else None
    counts = {"full": 0, "interview": 0, "empty": 0}
    for _ in range(4000):
        u = users[rng.integers(len(users))]
        r0 = rng.random()
        counts["full" if r0 < P_FULL else "interview" if r0 < P_FULL + P_INTERVIEW else "empty"] += 1
    tot = sum(counts.values())
    log(f"[test] regime mix: full={counts['full']/tot:.3f} (want {P_FULL}) "
        f"interview={counts['interview']/tot:.3f} (want {P_INTERVIEW}) "
        f"empty={counts['empty']/tot:.3f} (want {1-P_FULL-P_INTERVIEW:.2f})")

    # ---- 6. draw() end-to-end returns the 6-tuple shape the trainer will consume
    shapes = set()
    for _ in range(300):
        d = draw(users[rng.integers(len(users))], rng, fam, fake_full)
        if d is not None:
            shapes.add(len(d))
    log(f"[test] draw() tuple arity {shapes} -> {'PASS' if shapes == {6} else 'FAIL'}")
    ok &= shapes == {6}

    log(f"[test] {'ALL CHECKS PASS' if ok else '*** FAILURES -- do not train ***'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
