r"""igcn_ml100k_snap.py -- SNAP step S for IGCN (Rashid, Karypis & Riedl, SIGKDD Expl. 10(2) 2008).

WHAT IS AND IS NOT BEING SNAPPED, SAID FIRST BECAUSE IT DECIDES HOW THE RESULT MAY BE READ.

The paper's evaluation is (a) an OFFLINE simulation and (b) a 20-day ONLINE study with 381 valid
subjects, both on the live MovieLens corpus of the day: ~11,000 users, ~9,000 items, ~3M ratings.
That corpus is not a released dataset and cannot be obtained, and the numbers our full-text pass
recorded are the ONLINE ones -- which are irreproducible by construction, since they came from human
subjects. **A NUMERIC SNAP TO THIS PAPER IS THEREFORE IMPOSSIBLE AND IS NOT ATTEMPTED.** What is
snapped is the published ORDERING, which is the part that transfers, exactly as the standing snap
protocol prescribes for a paper that reports only curves or a study.

    THE PUBLISHED TARGET [V] -- online MAE with their user-kNN recommender:
        IGCN 0.67  (best)  <  Popularity 0.73  ==  Entropy0 0.73  <  HELF 0.84  (worst)
    verbatim verdict: "Overall, the Popularity measure performed the worst, and the Igcn measure
    performed the best."  (The two statements are about different metrics -- popularity is worst on
    Expected Utility and mid on MAE -- which is itself the paper's point.)
    Source: external_literature/findings/rating_elicitation_family_deep.md sec 2.3, from the repo's
    own full-text pass. The publisher PDF is CLOSED (Semantic Scholar openAccessPdf status CLOSED on
    DOI 10.1145/1540276.1540302); every mirror tried returned 404/403.

    THE GATE, stated before running: on MAE with a user-kNN consumer, at matched question budget,
        (G1) IGCN beats HELF, and
        (G2) IGCN is the best of the four arms
    PASS = both; PARTIAL = G1 only, diagnosed; FAIL = neither.

WHAT THIS SCRIPT DOES. ML-100k (the only MovieLens with ratings that is present in this repo besides
the ML-25M ruler itself), 200 held-out cold users x 3 seeds, each cold user's ratings split 70/30
into a fold-in side (the ONLY source of answers) and a DISJOINT eval side. Question budgets are the
paper's own 15/30/45/60/75. An asked-but-unrated item BURNS the question, as in the paper and as in
our harness. The consumer is user-kNN with mean-centred cosine, the paper's own recommender family.
Metrics: MAE on the eval side, and #seen (how many of the asked items the user could rate) -- the
paper's own second metric.

PARAMETER RESCALING, REPORTED AND NOT HIDDEN. The paper's (c, l) = (300 clusters, 150 neighbours) is
sized for ~11,000 users. On 743 training users 300 clusters is 2.5 users per cluster and 150
neighbours is 20% of the corpus, both degenerate. The PRIMARY row therefore uses the paper's RATIOS
(1 cluster per 37 users, l = 1.4% of the cohort) -> c=20, l=10, exactly as SNAP-5 rescaled Golbandi's
xi from Netflix to ML-100k. The literal (300, 150) is also run and reported, because "the constants
do not transfer" is a finding, not an excuse.

PERSISTENCE (HARD RULE 11): every (arm, budget, seed) cell is appended to the output JSON the moment
it is computed.

  python scripts/baselines/igcn_ml100k_snap.py
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import interview_strategies as ST
import igcn_selector as IG

ML100K = os.path.join(_ROOT, "data", "movielens", "ml-100k", "u.data")
OUT = os.path.join(_ROOT, "experiments", "results", "snapIGCN")
BUDGETS = (15, 30, 45, 60, 75)
KNN_K = 30            # neighbours of the CONSUMER (not of IGCN); the paper does not pin it -- ours


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ml100k(path=ML100K):
    raw = np.loadtxt(path, dtype=np.int64)
    u, i, r = raw[:, 0] - 1, raw[:, 1] - 1, raw[:, 2].astype(np.float64)
    return sparse.csr_matrix((r, (u, i)), shape=(int(u.max()) + 1, int(i.max()) + 1))


def split_cold(R, n_cold, seed):
    """Hold out `n_cold` users; split each held-out user's ratings 70/30 into a fold-in side (which
    alone may answer questions) and a DISJOINT eval side. No answer can ever come from an item the
    MAE is computed on -- the leak the archived SNAP-5 precedent had."""
    rng = np.random.default_rng(seed)
    n = R.shape[0]
    perm = rng.permutation(n)
    cold, train = perm[:n_cold], perm[n_cold:]
    Rc = R[cold].tocsr()
    fi_r, fi_c, fi_v, ev_r, ev_c, ev_v = [], [], [], [], [], []
    for j in range(Rc.shape[0]):
        s, e = Rc.indptr[j], Rc.indptr[j + 1]
        idx = np.arange(s, e)
        rng.shuffle(idx)
        cut = max(1, int(round(0.7 * len(idx))))
        for k, tgt in ((idx[:cut], (fi_r, fi_c, fi_v)), (idx[cut:], (ev_r, ev_c, ev_v))):
            tgt[0].extend([j] * len(k))
            tgt[1].extend(Rc.indices[k].tolist())
            tgt[2].extend(Rc.data[k].tolist())
    sh = (Rc.shape[0], R.shape[1])
    fold = sparse.csr_matrix((fi_v, (fi_r, fi_c)), shape=sh)
    ev = sparse.csr_matrix((ev_v, (ev_r, ev_c)), shape=sh)
    return R[train].tocsr(), fold, ev


# --------------------------------------------------------------------------- the consumer
class UserKNN:
    """Cosine user-kNN over BASELINE RESIDUALS -- the paper's own recommender family (user-kNN), in
    the textbook baseline-predictor form  rhat = mu + b_u + b_i + kNN(residual).

    WHY THE BASELINE AND NOT A PLAIN USER MEAN, and it is load-bearing rather than a detail. The
    first version of this consumer centred each cold user on the MEAN OF THEIR OWN ANSWERS. Every
    elicitation arm asks about popular, well-liked items, so that mean is biased upward by an amount
    that DEPENDS ON THE ARM -- and the existential control caught it: the `none` arm (ask nothing,
    predict from item means) beat every elicitation arm at every budget, i.e. the arms were being
    ranked on how little they corrupted the centring. Estimating b_u against the ITEM baselines
    instead removes the item-selection bias by construction:

        b_i = SUM_{u in R(i)} (r_ui - mu) / (|R(i)| + beta_i)
        b_u = SUM_{i in answers} (r_ui - mu - b_i) / (n_answers + beta_u)

    so an arm that asks only items with b_i = +0.6 contributes nothing to b_u through that channel.
    beta_i = 25, beta_u = 10 are the standard Koren shrinkages, applied IDENTICALLY to every arm and
    to the control, and never varied per arm."""

    BETA_I, BETA_U = 25.0, 10.0

    def __init__(self, R_train, k=KNN_K):
        self.R = R_train.tocsr()
        self.k = int(k)
        self.glob = float(self.R.data.mean())
        icnt = np.asarray((self.R > 0).sum(0)).ravel().astype(np.float64)
        isum = np.asarray(self.R.sum(0)).ravel() - self.glob * icnt
        self.b_i = isum / (icnt + self.BETA_I)
        cnt = np.diff(self.R.indptr)
        base_rated = self.glob + self.b_i[self.R.indices]
        dev = self.R.data - base_rated
        su = np.add.reduceat(dev, self.R.indptr[:-1]) * (cnt > 0)
        self.b_u = su / (cnt + self.BETA_U)
        C = self.R.copy().astype(np.float64)
        C.data = dev - np.repeat(self.b_u, cnt)               # residual off the full baseline
        nrm = np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel())
        self.CnT = sparse.diags(1.0 / np.maximum(nrm, 1e-12)).dot(C).tocsr().T.tocsr()
        self.Ccsc = C.tocsc()
        self.Rcsc = self.R.tocsc()

    def predict(self, answers, targets):
        """answers: list of (item, rating). targets: array of item ids. -> predicted ratings."""
        targets = np.asarray(targets, np.int64)
        if not answers:
            return np.clip(self.glob + self.b_i[targets], 1.0, 5.0)
        it = np.asarray([a[0] for a in answers], np.int64)
        rv = np.asarray([a[1] for a in answers], np.float64)
        b_u = float((rv - self.glob - self.b_i[it]).sum() / (len(it) + self.BETA_U))
        base = self.glob + b_u + self.b_i[targets]
        res = rv - self.glob - b_u - self.b_i[it]
        v = sparse.csr_matrix((res, (np.zeros(len(it), np.int64), it)), shape=(1, self.R.shape[1]))
        nv = float(np.sqrt(v.multiply(v).sum()))
        if nv <= 1e-12:
            return np.clip(base, 1.0, 5.0)
        sim = np.asarray((v @ self.CnT).todense()).ravel() / nv
        top = np.argsort(-np.abs(sim))[:self.k]
        top = top[np.abs(sim[top]) > 1e-9]
        keep = np.zeros(self.R.shape[0], bool)
        keep[top] = True
        out = base.copy()
        for j, t in enumerate(targets):
            s, e = self.Ccsc.indptr[t], self.Ccsc.indptr[t + 1]
            raters, resid = self.Ccsc.indices[s:e], self.Ccsc.data[s:e]
            m = keep[raters]
            if not m.any():
                continue
            w = sim[raters[m]]
            out[j] = base[j] + float((w * resid[m]).sum() / max(np.abs(w).sum(), 1e-12))
        return np.clip(out, 1.0, 5.0)


# --------------------------------------------------------------------------- the interview
def run_arm(order_fn, fold, budget, n_items):
    """Walk a per-user question order to `budget` asks; an unrated ask BURNS the turn."""
    asked, answered = [], []
    F = fold.tocsr()
    for u in range(F.shape[0]):
        s, e = F.indptr[u], F.indptr[u + 1]
        have = dict(zip(F.indices[s:e].tolist(), F.data[s:e].tolist()))
        a, t = [], []
        for i in order_fn(u, a, t, budget):
            if len(a) >= budget:
                break
            a.append(int(i))
            if int(i) in have:
                t.append((int(i), float(have[int(i)])))
        asked.append(a)
        answered.append(t)
    return asked, answered


def static_order_fn(order):
    return lambda u, a, t, b: order


def igcn_order_fn(sel, fold, n_items):
    """IGCN is adaptive, so its 'order' is generated one question at a time against the answers so
    far -- the same two-phase Algorithm 3.1 the ported selector runs, driven here per user."""
    F = fold.tocsr()

    def gen(u, a, t, budget):
        s, e = F.indptr[u], F.indptr[u + 1]
        have = dict(zip(F.indices[s:e].tolist(), F.data[s:e].tolist()))
        ask, ans = [], []
        for _ in range(budget):
            q = int(sel.select(len(ask), np.array([0]), [ask], [ans])[0])
            ask.append(q)
            if q in have:
                # levels, not stars: the selector's neighbour search speaks the harness's band units
                ans.append((q, int(np.clip(round(have[q] * 2) - 1, 0, IG.NLEV - 1))))
            yield q
    return gen


def mae_of(knn, answered, ev, per_user=False):
    """Pooled MAE over all eval ratings, and (optionally) the PER-USER mean absolute error.

    The per-user vector is the only thing a PAIRED bootstrap can be run from -- the unpaired
    1.96*sqrt(2)*SE form is banned repo-wide (ARM_N_PHASE1 sec 4b), and every gap in this table is
    small enough that the distinction decides whether there is a claim at all."""
    E = ev.tocsr()
    err, n = 0.0, 0
    pu = np.full(E.shape[0], np.nan)
    for u in range(E.shape[0]):
        s, e = E.indptr[u], E.indptr[u + 1]
        if e == s:
            continue
        tg, tr = E.indices[s:e], E.data[s:e]
        pr = knn.predict([(i, r) for i, r in answered[u]], tg)
        a = np.abs(pr - tr)
        err += float(a.sum())
        n += len(tg)
        pu[u] = float(a.mean())
    return (err / max(n, 1), pu) if per_user else err / max(n, 1)


def paired_ci(x, y, n_boot=10000, seed=98765):
    """Paired bootstrap over USERS of mean(x - y). Returns (delta, lo, hi)."""
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cold", type=int, default=200)
    ap.add_argument("--seeds", default="98765,1,2")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--label", default="ml100k")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    budgets = [int(x) for x in a.budgets.split(",")]

    R = load_ml100k()
    log(f"[snapIGCN] ML-100k: {R.shape[0]} users x {R.shape[1]} items, {R.nnz} ratings")
    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"igcn_snap_{a.label}.json")
    blob = {"meta": {
        "target": "Rashid, Karypis & Riedl, SIGKDD Explorations 10(2) 2008 -- IGCN",
        "published_ordering_MAE_userkNN": {"igcn": 0.67, "popularity": 0.73, "entropy0": 0.73,
                                           "helf": 0.84},
        "SNAP_KIND": "ORDERING snap on ML-100k -- NOT a numeric snap; the paper's corpus is the "
                     "live MovieLens site of 2008 and its headline numbers come from a 381-subject "
                     "online study, both irreproducible",
        "n_cold": a.n_cold, "seeds": seeds, "budgets": budgets, "knn_k": KNN_K,
        "gate": "G1 IGCN beats HELF; G2 IGCN best of four"}, "cells": []}

    def flush():
        tmp = outp + ".tmp"
        json.dump(blob, open(tmp, "w"), indent=2)
        os.replace(tmp, outp)

    variants = {"rescaled": dict(n_clusters=20, n_nbr=10),
                "literal": dict(n_clusters=IG.N_CLUSTERS, n_nbr=IG.N_NEIGHBOURS)}
    pu_vec = {}

    for seed in seeds:
        train, fold, ev = split_cold(R, a.n_cold, seed)
        n_items = R.shape[1]
        log(f"[snapIGCN] seed {seed}: train {train.shape[0]}u/{train.nnz}r, "
            f"cold {fold.shape[0]}u fold {fold.nnz}r eval {ev.nnz}r")
        glob, bank = ST.build_orders(train, (train > 0).astype(np.float64), n_items, log=log)
        knn = UserKNN(train)
        rng = np.random.default_rng(seed)

        arms = {nm: static_order_fn(glob[nm]) for nm in ("entropy0", "helf", "popularity")}
        arms["random"] = static_order_fn(rng.permutation(n_items))
        # EXISTENTIAL CONTROL (HARD RULE 5). `none` asks nothing, so the consumer answers from item
        # means alone. If an elicitation arm does not beat it, the CONSUMER is broken and no ordering
        # between the arms means anything -- the arms would be ranked on how little they interfere.
        arms["none"] = static_order_fn(np.empty(0, np.int64))
        sels = {}
        for vn, kw in variants.items():
            sel = IG.IGCNSelector(train, n_items, switch_i=IG.SWITCH_AFTER, recompute_every=1,
                                  seed=seed, log=log, **kw)
            sel.fit()
            sel.reset({"n_items": n_items, "glob": glob, "D": None})
            arms[f"igcn_{vn}"] = igcn_order_fn(sel, fold, n_items)
            sels[f"igcn_{vn}"] = sel

        # CONSUMER SANITY (HARD RULE 5, second control). The `none` floor alone cannot tell "the
        # interview is too short to help this consumer" from "this consumer ignores information".
        # Feeding it the cold user's WHOLE fold-in profile separates the two, once per seed.
        F0 = fold.tocsr()
        fullp = [[(int(i), float(v)) for i, v in
                  zip(F0.indices[F0.indptr[u]:F0.indptr[u + 1]],
                      F0.data[F0.indptr[u]:F0.indptr[u + 1]])] for u in range(F0.shape[0])]
        mae_full = mae_of(knn, fullp, ev)
        blob["cells"].append({"seed": seed, "arm": "_diag_full_profile", "budget": -1,
                              "mae": round(mae_full, 5),
                              "seen": round(float(np.mean([len(x) for x in fullp])), 3)})
        log(f"[snapIGCN] seed={seed} CONSUMER SANITY: full fold-in profile "
            f"({np.mean([len(x) for x in fullp]):.1f} ratings) MAE={mae_full:.4f}")

        for b in budgets:
            for nm, fn in arms.items():
                t0 = time.time()
                if nm.startswith("igcn"):
                    asked, answered = [], []
                    F = fold.tocsr()
                    for u in range(F.shape[0]):
                        gseq = list(fn(u, None, None, b))
                        asked.append(gseq)
                        s, e = F.indptr[u], F.indptr[u + 1]
                        have = dict(zip(F.indices[s:e].tolist(), F.data[s:e].tolist()))
                        answered.append([(i, have[i]) for i in gseq if i in have])
                else:
                    asked, answered = run_arm(fn, fold, b, n_items)
                mae, pu = mae_of(knn, answered, ev, per_user=True)
                pu_vec[(seed, nm, b)] = pu
                seen = float(np.mean([len(t) for t in answered]))
                distinct = len({tuple(x) for x in asked})
                cell = {"seed": seed, "arm": nm, "budget": b, "mae": round(mae, 5),
                        "seen": round(seen, 3), "distinct_ask_prefixes": distinct,
                        "minutes": round((time.time() - t0) / 60, 2)}
                if nm.startswith("igcn"):
                    # per-CELL adaptivity diagnostics, then zero the counters so the next budget's
                    # cell reports its own interview and not the cumulative total
                    s = sels[nm]
                    cell.update({k: v for k, v in s.stats().items()
                                 if k != "igcn_personalised_per_turn"})
                    s._n_pers = s._n_glob = s._n_degen = 0
                    s._pers_per_turn = []
                blob["cells"].append(cell)
                flush()
                log(f"[snapIGCN] seed={seed} {nm:16s} k={b:3d}  MAE={mae:.4f}  seen={seen:5.2f}  "
                    f"distinct-prefixes={distinct:4d}  ({cell['minutes']:.2f}m)")

    # ---------------- verdict
    log("=" * 96)
    names = sorted({c["arm"] for c in blob["cells"]})
    tab = {}
    for nm in names:
        tab[nm] = {b: float(np.mean([c["mae"] for c in blob["cells"]
                                     if c["arm"] == nm and c["budget"] == b])) for b in budgets}
    hdr = "arm".ljust(16) + "".join(f"  k={b:<7d}" for b in budgets)
    log(hdr)
    for nm in names:
        log(nm.ljust(16) + "".join(f"  {tab[nm][b]:<9.4f}" for b in budgets))
    # ---------------- paired bootstrap over users, IGCN vs each comparator, per budget
    log("-" * 96)
    log("paired over users (10,000 resamples, seed 98765); NEGATIVE delta = IGCN has LOWER MAE")
    cis = {}
    for b in budgets:
        for ign in ("igcn_rescaled", "igcn_literal"):
            for cmp_ in ("entropy0", "helf", "popularity", "random", "none"):
                x = np.concatenate([pu_vec[(s, ign, b)] for s in seeds])
                y = np.concatenate([pu_vec[(s, cmp_, b)] for s in seeds])
                d, lo, hi = paired_ci(x, y)
                sep = "SEP" if (lo > 0 or hi < 0) else "ns "
                cis[f"{ign}-{cmp_}|k{b}"] = {"delta": round(d, 5), "lo": round(lo, 5),
                                             "hi": round(hi, 5), "separated": sep == "SEP"}
                log(f"  k={b:3d}  {ign:14s} - {cmp_:11s} = {d:+.4f} "
                    f"[{lo:+.4f},{hi:+.4f}] {sep}")
    blob["meta"]["paired_ci"] = cis

    ig_nm = "igcn_rescaled"
    g1 = all(tab[ig_nm][b] < tab["helf"][b] for b in budgets)
    g2 = all(tab[ig_nm][b] <= min(tab[o][b] for o in ("helf", "entropy0", "popularity"))
             for b in budgets)
    verdict = "PASS" if (g1 and g2) else ("PARTIAL" if g1 else "FAIL")
    blob["meta"]["verdict"] = {"G1_beats_helf": bool(g1), "G2_best_of_four": bool(g2),
                               "verdict": verdict, "mean_mae": tab}
    flush()
    log(f"[snapIGCN] G1 (IGCN < HELF at every budget): {g1}")
    log(f"[snapIGCN] G2 (IGCN best of four at every budget): {g2}")
    log(f"[snapIGCN] VERDICT = {verdict}  ->  {outp}")


if __name__ == "__main__":
    main()
