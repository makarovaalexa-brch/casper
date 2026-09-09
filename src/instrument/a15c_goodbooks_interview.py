"""A15c -- RQ1 on goodbooks-10k: do published static selectors lose to a metric-optimised order?

Design sheet: docs/design/DS_A15_GOODBOOKS_PROBE.md

This is the paper's headline audit, ported to a second domain. One contract for every arm:

  * CONSUMER: the goodbooks RecVAE from A15b, frozen. It ingests binary likes only, which is exactly
    the "RecVAE (likes only, conservative)" consumer the paper already uses in its robustness sweep.
  * QUESTION: a book. ANSWER: the user's own rating, if they rated it in their fold-in profile.
    A rated book folds as a like iff rating > 3.5; a rated-but-disliked book folds nothing but still
    BURNS the turn; an unrated book burns the turn. Unanswerable questions are charged, never free.
  * COLD START: the user's profile at interview time is ONLY their answers. Their real fold-in
    history is used solely as the answer oracle, never folded.
  * CANDIDATE POOL: the full 10,000-book catalogue minus the books asked so far. No sampling.
  * METRIC: full and tail NDCG@10 against the user's held-out targets, tail = short-head removed.
  * ORDERS ARE FITTED ON VALIDATION AND SCORED ONCE ON TEST. No test user's targets enter any search.

ARMS
  greedy_ndcg   our comparator: lazy-greedy (CELF) prefix-optimal order maximising cohort-mean
                held-out NDCG@10 on the validation cohort -- the only rule that optimises the metric
  popularity    Rashid et al., most-rated first
  entropy0      Rashid et al., rating-distribution entropy
  helf          Rashid et al., harmonic mean of entropy and log popularity
  random        control, seeded
  oracle        clairvoyant ceiling: per user, the answerable books they rated highest (NOT a
                baseline; it reads the user's own ratings at selection time)

Run:  python src/instrument/a15c_goodbooks_interview.py --budgets 1,2,4,8,16
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "instrument2"))
from recvae import RecVAE, make_predict_fn   # noqa: E402

PROC = os.path.join(_ROOT, "data", "goodbooks-10k", "proc")
OUT = os.path.join(_ROOT, "experiments", "goodbooks")
CKPT = os.path.join(_ROOT, ".cache", "goodbooks")
LIKE_GT = 3.5
HEAD_MASS = 0.33
K = 10
POOL_N = 500
SEED = 98765
N_BOOT = 10000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_bootstrap(d, seed=SEED, n_boot=N_BOOT):
    d = np.asarray(d, np.float64); d = d[np.isfinite(d)]
    if not len(d):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


class Cohort:
    """Answer oracle + targets for one held-out cohort."""

    def __init__(self, name, ni):
        dtr = pd.read_csv(os.path.join(PROC, f"{name}_tr.csv"))
        dte = pd.read_csv(os.path.join(PROC, f"{name}_te.csv"))
        us = np.array(sorted(set(dtr.user_id) | set(dte.user_id)))
        self.users = us
        self.n = len(us)
        u2r = {u: i for i, u in enumerate(us)}
        self.ni = ni
        # answer oracle: rating of every fold-in book, per user (targets NOT included)
        self.rating = sparse.csr_matrix(
            (dtr.rating.values.astype(np.float32),
             ([u2r[u] for u in dtr.user_id], dtr.sid.values)), shape=(self.n, ni))
        self.rated = self.rating.copy(); self.rated.data = np.ones_like(self.rated.data)
        dte_l = dte[dte.rating > LIKE_GT]
        self.target = sparse.csr_matrix(
            (np.ones(len(dte_l), np.float32),
             ([u2r[u] for u in dte_l.user_id], dte_l.sid.values)), shape=(self.n, ni))
        # leak guard: an answered book must never be a target
        inter = self.rated.multiply(self.target).nnz
        assert inter == 0, f"{name}: {inter} fold-in books are also targets"
        self.tgt_rows = self.target.tolil().rows


def ndcg(P, asked_cols, cohort, head, k=K):
    """full and tail NDCG@k. `asked_cols` are the books asked of EVERY user, so they are dropped
    from the candidate pool by column rather than through an (n x ni) mask."""
    p = np.array(P, dtype=np.float32, copy=True)
    if len(asked_cols):
        p[:, np.asarray(asked_cols, np.int64)] = -np.inf
    out = np.zeros((2, p.shape[0]), np.float32)
    tp = 1.0 / np.log2(np.arange(2, k + 2))
    for axis, mask in ((0, None), (1, head)):
        if mask is None:
            q = p
        else:
            q = p.copy()
            q[:, mask] = -np.inf
        idx = np.argpartition(-q, k, 1)[:, :k]
        idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(q, idx, 1), 1), 1)
        for i in range(p.shape[0]):
            t = cohort.tgt_rows[i]
            if mask is not None:
                t = [x for x in t if not mask[x]]
            if not t:
                out[axis, i] = np.nan
                continue
            ts = set(t)
            dcg = sum(tp[j] for j, x in enumerate(idx[i]) if x in ts)
            out[axis, i] = dcg / tp[:min(len(ts), k)].sum()
    return out[0], out[1]


class Env:
    def __init__(self, cohort, predict, head, batch=1000):
        self.c = cohort; self.predict = predict; self.head = head; self.batch = batch
        self.memo = {}
        self.memo_cap = 4096          # HARD RULE 11c: bound every per-state cache
        self.n_evals = 0

    def fold_and_score(self, seq):
        """Ask `seq` (book indices, in order) of every user; return per-user full/tail NDCG@10."""
        key = tuple(seq)
        if key in self.memo:
            return self.memo[key]
        c = self.c
        n, ni = c.n, c.ni
        if seq:
            s = np.asarray(seq, np.int64)
            R = c.rating[:, s].toarray()             # 0 where the user did not rate it
            rows, cols = np.nonzero(R > LIKE_GT)     # answered AND liked -> folds as a like
            X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, s[cols])),
                                  shape=(n, ni))
        else:
            X = sparse.csr_matrix((n, ni), dtype=np.float32)
        outs = []
        for st in range(0, n, self.batch):
            outs.append(self.predict(X[st:st + self.batch]))
        P = np.vstack(outs)
        f, t = ndcg(P, seq, c, self.head)
        self.n_evals += 1
        if len(self.memo) >= self.memo_cap:
            self.memo.pop(next(iter(self.memo)))
        self.memo[key] = (f, t)
        return f, t


def greedy_order(env, pool, budget, seed_prefix=()):
    """Lazy-greedy (CELF) prefix-optimal order maximising cohort-mean full NDCG@10."""
    seq = list(seed_prefix)
    base = np.nanmean(env.fold_and_score(seq)[0]) if seq else np.nanmean(env.fold_and_score([])[0])
    heap = [[np.inf, int(c), -1] for c in pool]       # [upper bound, candidate, last updated at len]
    chosen = []
    while len(seq) < budget:
        heap.sort(key=lambda z: -z[0])
        while True:
            top = heap[0]
            if top[2] == len(seq):
                break
            g = np.nanmean(env.fold_and_score(seq + [top[1]])[0]) - base
            top[0] = g; top[2] = len(seq)
            heap.sort(key=lambda z: -z[0])
        best = heap.pop(0)
        seq.append(best[1]); base += best[0]; chosen.append({"book": best[1], "gain": float(best[0])})
        log(f"    pick {len(seq):2d}: book {best[1]:5d}  gain {best[0]:+.5f}  "
            f"cohort full {base:.5f}  ({env.n_evals} evals)")
    return seq, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--hidden", type=int, default=600)
    ap.add_argument("--latent", type=int, default=200)
    ap.add_argument("--label", default="a15c_rq1")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    B = max(budgets)
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    t00 = time.time()

    meta = json.load(open(os.path.join(PROC, "meta.json")))
    ni = meta["n_items"]
    tr = pd.read_csv(os.path.join(PROC, "train.csv"))
    trl = tr[tr.rating > LIKE_GT]
    cnt = np.zeros(ni, np.float64)
    vc = trl.sid.value_counts()
    cnt[vc.index.values] = vc.values
    order = np.argsort(-cnt); cum = np.cumsum(cnt[order]) / cnt.sum()
    head = np.zeros(ni, bool); head[order[:np.searchsorted(cum, HEAD_MASS) + 1]] = True
    log(f"[data] {ni} books; head {head.sum()}; train likes {len(trl)}")

    # rating-distribution statistics on TRAIN, for the heuristics
    ent = np.zeros(ni); nrat = np.zeros(ni)
    g = tr.groupby("sid").rating
    for sid, s in g:
        v = s.value_counts(normalize=True).values
        ent[sid] = -(v * np.log2(v)).sum()
        nrat[sid] = len(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        helf = 2 * (np.log2(nrat + 1) * ent) / np.maximum(np.log2(nrat + 1) + ent, 1e-9)
    log("[heuristics] entropy and HELF computed on the train cohort")

    dev = torch.device("cpu")
    model = RecVAE(a.hidden, a.latent, ni).to(dev)
    ck = os.path.join(CKPT, f"recvae_gb_h{a.hidden}_d{a.latent}.pt")
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    predict = make_predict_fn(model, dev)
    log(f"[consumer] frozen RecVAE {os.path.basename(ck)}")

    va, te = Cohort("validation", ni), Cohort("test", ni)
    log(f"[cohorts] validation {va.n}, test {te.n}; leak guards passed")

    pool = np.argsort(-cnt)[:POOL_N]
    arms = {
        "popularity": list(np.argsort(-cnt)[:B]),
        "entropy0": list(np.argsort(-ent)[:B]),
        "helf": list(np.argsort(-helf)[:B]),
        "random": list(np.random.default_rng(SEED).choice(ni, B, replace=False)),
    }

    log("[greedy] building the metric-optimised order on VALIDATION")
    genv = Env(va, predict, head)
    seq, picks = greedy_order(genv, pool, B)
    arms["greedy_ndcg"] = seq
    log(f"[greedy] done in {genv.n_evals} evals, {(time.time()-t00)/60:.1f}m")

    # score every arm on TEST
    tenv = Env(te, predict, head)
    res = {"label": a.label, "dataset": "goodbooks-10k", "consumer": "RecVAE (likes only)",
           "split": "fitted on validation, scored on test", "n_test_users": te.n,
           "budgets": budgets, "n_items": ni, "head_items": int(head.sum()),
           "greedy_picks": picks, "orders": {k: [int(x) for x in v] for k, v in arms.items()},
           "curves": {}, "per_user": {}}
    m0f, m0t = tenv.fold_and_score([])
    res["answer_free"] = {"full": float(np.nanmean(m0f)), "tail": float(np.nanmean(m0t))}
    log(f"[m0] answer-free reference full {np.nanmean(m0f):.5f} tail {np.nanmean(m0t):.5f}")

    peruser = {}
    for arm, seqa in arms.items():
        res["curves"][arm] = {}
        for kb in budgets:
            f, t = tenv.fold_and_score(seqa[:kb])
            ans = np.asarray(te.rated[:, np.asarray(seqa[:kb], np.int64)].sum(1)).ravel()
            res["curves"][arm][str(kb)] = {"full": float(np.nanmean(f)), "tail": float(np.nanmean(t)),
                                           "answers": float(ans.mean())}
            peruser[f"{arm}|{kb}|full"] = f; peruser[f"{arm}|{kb}|tail"] = t
        c = res["curves"][arm][str(B)]
        log(f"[test] {arm:12s} K={B}: full {c['full']:.5f} tail {c['tail']:.5f} "
            f"answers {c['answers']:.2f}")

    # contrasts against the metric-optimised order
    res["contrasts"] = {}
    for arm in arms:
        if arm == "greedy_ndcg":
            continue
        for kb in budgets:
            d = peruser[f"greedy_ndcg|{kb}|full"] - peruser[f"{arm}|{kb}|full"]
            dt = peruser[f"greedy_ndcg|{kb}|tail"] - peruser[f"{arm}|{kb}|tail"]
            res["contrasts"][f"greedy_minus_{arm}@K{kb}"] = {
                "full": paired_bootstrap(d), "tail": paired_bootstrap(dt)}

    res["seconds"] = time.time() - t00
    json.dump(res, open(os.path.join(OUT, f"{a.label}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, f"{a.label}_per_user.npz"), **peruser)
    log(f"\n[done] {os.path.join(OUT, a.label)}.json  ({res['seconds']/60:.1f}m, "
        f"{tenv.n_evals} test evals)")


if __name__ == "__main__":
    main()
