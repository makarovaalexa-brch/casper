"""A15b -- build the goodbooks-10k Liang split and train the RecVAE that will consume the answers.

Design sheet: docs/design/DS_A15_GOODBOOKS_PROBE.md

Two jobs, both resumable, both banked to disk:
  1. Persist a Liang strong-generalisation split of goodbooks-10k under data/goodbooks-10k/proc/,
     in the same shape as data/ml-25m/proc/ so the rest of the pipeline can read it.
  2. Train RecVAE on it. On ML-25M RecVAE is the frozen decoder the instrument is built on AND the
     published parity bar; here it is both the parity bar and, for the first replication, the
     CONSUMER itself, which removes the whole tower-certification step from the critical path.

PROTOCOL, mirroring the ML-25M canon exactly:
  * min_uc = 5; like <=> rating > 3.5 for the binary training matrix (Liang)
  * held-out users: per-user 80/20 fold-in/target, seed 98765
  * 43,424 train / 5,000 validation / 5,000 test users; full 10,000-book catalogue, NO truncation
  * metric: full AND tail NDCG@10, tail = Cremonesi short-head removed (head = items covering the
    top 33% of train interaction mass)
  * candidate pool excludes the user's own fold-in items; no popularity floor; no negative sampling

HARD RULE 11: the split is written once and reused; training banks a checkpoint and a log line every
epoch and resumes from the best checkpoint, so a killed run loses at most one epoch.

Run:  python src/instrument/a15b_goodbooks_recvae.py --epochs 30
      python src/instrument/a15b_goodbooks_recvae.py --eval_only
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
from recvae import RecVAE, run_updates, make_predict_fn   # noqa: E402

DATA = os.path.join(_ROOT, "data", "goodbooks-10k")
PROC = os.path.join(DATA, "proc")
OUT = os.path.join(_ROOT, "experiments", "goodbooks")
CKPT = os.path.join(_ROOT, ".cache", "goodbooks")
SEED = 98765
N_VAL = N_TEST = 5000
MIN_UC = 5
LIKE_GT = 3.5
TEST_PROP = 0.2
HEAD_MASS = 0.33
K = 10


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_split():
    """Write the Liang split once. Returns the metadata dict."""
    meta_p = os.path.join(PROC, "meta.json")
    if os.path.exists(meta_p):
        return json.load(open(meta_p))
    os.makedirs(PROC, exist_ok=True)
    r = pd.read_csv(os.path.join(DATA, "ratings.csv")).drop_duplicates(["user_id", "book_id"])
    cnt = r.groupby("user_id").size()
    r = r[r.user_id.isin(cnt[cnt >= MIN_UC].index)]
    users = np.array(sorted(r.user_id.unique()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(users))
    te_u = users[perm[:N_TEST]]
    va_u = users[perm[N_TEST:N_TEST + N_VAL]]
    tr_u = users[perm[N_TEST + N_VAL:]]
    books = np.array(sorted(r.book_id.unique()))
    np.savetxt(os.path.join(PROC, "unique_sid.txt"), books, fmt="%d")
    b2i = {b: i for i, b in enumerate(books)}

    def emit(df, name):
        df = df.copy()
        df["sid"] = df.book_id.map(b2i)
        df[["user_id", "sid", "rating"]].to_csv(os.path.join(PROC, name), index=False)

    emit(r[r.user_id.isin(set(tr_u.tolist()))], "train.csv")
    for cohort, nm in ((va_u, "validation"), (te_u, "test")):
        sub = r[r.user_id.isin(set(cohort.tolist()))]
        tr_rows, te_rows = [], []
        for u, g in sub.groupby("user_id"):
            idx = rng.permutation(len(g))
            g = g.iloc[idx]
            ntg = max(1, int(round(TEST_PROP * len(g))))
            te_rows.append(g.iloc[:ntg]); tr_rows.append(g.iloc[ntg:])
        emit(pd.concat(tr_rows), f"{nm}_tr.csv")
        emit(pd.concat(te_rows), f"{nm}_te.csv")

    meta = {"dataset": "goodbooks-10k", "n_items": int(len(books)),
            "n_train_users": int(len(tr_u)), "n_val_users": int(len(va_u)),
            "n_test_users": int(len(te_u)), "n_events_total": int(len(r)),
            "seed": SEED, "min_uc": MIN_UC, "rating_gt": LIKE_GT, "test_prop": TEST_PROP,
            "provenance": "Liang strong-generalisation recipe applied to goodbooks-10k; "
                          "full catalogue, no item truncation"}
    json.dump(meta, open(meta_p, "w"), indent=2)
    log(f"[split] written to {PROC}: {meta}")
    return meta


def load(name, ni):
    d = pd.read_csv(os.path.join(PROC, name))
    return d


def csr_from(d, ni, like_only=True):
    if like_only:
        d = d[d.rating > LIKE_GT]
    u = pd.factorize(d.user_id, sort=True)
    rows = u[0]
    return sparse.csr_matrix((np.ones(len(d), np.float32), (rows, d.sid.values)),
                            shape=(len(u[1]), ni)), np.asarray(u[1])


def ndcg_at_k(pred, tr_csr, te_csr, k, mask=None):
    """Strong-generalisation NDCG@k. Fold-in items excluded from candidates."""
    p = pred.copy()
    p[tr_csr.nonzero()] = -np.inf
    if mask is not None:
        p[:, mask] = -np.inf
    idx = np.argpartition(-p, k, axis=1)[:, :k]
    part = np.take_along_axis(p, idx, 1)
    idx = np.take_along_axis(idx, np.argsort(-part, axis=1), 1)
    tp = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.zeros(p.shape[0], np.float32)
    te = te_csr.tolil().rows
    for i in range(p.shape[0]):
        t = set(te[i])
        if mask is not None:
            t = {x for x in t if not mask[x]}
        if not t:
            out[i] = np.nan
            continue
        hits = np.array([1.0 if x in t else 0.0 for x in idx[i]])
        dcg = (hits * tp).sum()
        idcg = tp[:min(len(t), k)].sum()
        out[i] = dcg / idcg if idcg > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=600)
    ap.add_argument("--latent", type=int, default=200)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    torch.set_num_threads(4)

    meta = build_split()
    ni = meta["n_items"]
    tr = load("train.csv", ni)
    Xtr, _ = csr_from(tr, ni)
    log(f"[data] train matrix {Xtr.shape}, {Xtr.nnz} likes")

    cnt = np.asarray(Xtr.sum(0)).ravel()
    order = np.argsort(-cnt)
    cum = np.cumsum(cnt[order]) / max(cnt.sum(), 1)
    head = np.zeros(ni, bool); head[order[:np.searchsorted(cum, HEAD_MASS) + 1]] = True
    log(f"[tail] head = {head.sum()} items carrying the top {HEAD_MASS:.0%} of train mass")

    ev = {}
    for nm in ("validation", "test"):
        dtr, dte = load(f"{nm}_tr.csv", ni), load(f"{nm}_te.csv", ni)
        us = np.array(sorted(set(dtr.user_id) | set(dte.user_id)))
        u2r = {u: i for i, u in enumerate(us)}
        def mk(d, like):
            dd = d[d.rating > LIKE_GT] if like else d
            return sparse.csr_matrix((np.ones(len(dd), np.float32),
                                      ([u2r[u] for u in dd.user_id], dd.sid.values)),
                                     shape=(len(us), ni))
        ev[nm] = (mk(dtr, True), mk(dte, True))
        log(f"[eval] {nm}: {len(us)} users, fold-in nnz {ev[nm][0].nnz}, target nnz {ev[nm][1].nnz}")

    dev = torch.device("cpu")
    model = RecVAE(a.hidden, a.latent, ni).to(dev)
    ck = os.path.join(CKPT, f"recvae_gb_h{a.hidden}_d{a.latent}.pt")
    logp = os.path.join(OUT, "a15b_recvae_log.json")
    hist = json.load(open(logp)) if os.path.exists(logp) else []
    best = max([h["val_full"] for h in hist], default=-1.0)
    start = max([h["epoch"] for h in hist], default=0)
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location=dev))
        log(f"[resume] loaded {ck} at epoch {start}, best val full {best:.4f}")

    predict = make_predict_fn(model, dev)

    def evaluate(split):
        trm, tem = ev[split]
        outs = []
        for st in range(0, trm.shape[0], 500):
            outs.append(predict(trm[st:st + 500]))
        P = np.vstack(outs)
        f = np.nanmean(ndcg_at_k(P, trm, tem, K))
        t = np.nanmean(ndcg_at_k(P, trm, tem, K, mask=head))
        return float(f), float(t)

    if a.eval_only:
        for s in ("validation", "test"):
            f, t = evaluate(s)
            log(f"[{s}] full NDCG@{K} {f:.4f} tail {t:.4f}")
        return

    dec = [p for n, p in model.named_parameters() if n.startswith("decoder")]
    enc = [p for n, p in model.named_parameters() if not n.startswith("decoder")]
    opts = [torch.optim.Adam(enc, lr=5e-4), torch.optim.Adam(dec, lr=5e-4)]
    idx = np.arange(Xtr.shape[0])
    for ep in range(start + 1, a.epochs + 1):
        t0 = time.time()
        run_updates(model, opts, Xtr, idx, a.batch, dev, 1, 0.5, 0.005, None)
        vf, vt = evaluate("validation")
        hist.append({"epoch": ep, "val_full": vf, "val_tail": vt,
                     "minutes": round((time.time() - t0) / 60, 2)})
        json.dump(hist, open(logp, "w"), indent=1)
        star = ""
        if vf > best:
            best = vf; torch.save(model.state_dict(), ck); star = "  <- best, saved"
        log(f"[ep {ep:2d}] val full {vf:.4f} tail {vt:.4f}  ({hist[-1]['minutes']:.1f}m){star}")

    model.load_state_dict(torch.load(ck, map_location=dev))
    tf, tt = evaluate("test")
    res = {"dataset": "goodbooks-10k", "model": "RecVAE", "hidden": a.hidden, "latent": a.latent,
           "best_val_full": best, "test_full": tf, "test_tail": tt, "n_items": ni,
           "epochs_run": len(hist), "checkpoint": ck, "split_meta": meta}
    json.dump(res, open(os.path.join(OUT, "a15b_recvae_result.json"), "w"), indent=2)
    log(f"[TEST] full NDCG@{K} {tf:.4f} tail {tt:.4f}   (best val full {best:.4f})")


if __name__ == "__main__":
    main()
