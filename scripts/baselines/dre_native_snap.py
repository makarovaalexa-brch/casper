"""SNAP-3 — DRE native snap (Kweon, Kang, Hwang & Yu, "Deep Rating Elicitation for New Users
in Collaborative Filtering", WWW 2020, DOI 10.1145/3366423.3380042).

THEIR setup, THEIR data, THEIR metric, THEIR recommender. Nothing from our ruler enters here.

PUBLISHED TARGETS  [V] — arXiv:2402.16327 LaTeX source (`5ex.tex`), Table 2, k=50:
    MovieLens 1M   P@10 0.5396  P@20 0.4734  NDCG@10 0.5688  NDCG@20 0.5197
    MovieLens 20M  P@10 0.5712  P@20 0.4875  NDCG@10 0.6077  NDCG@20 0.5373
  best baseline (RMVA) ML-1M NDCG@10 0.5387 ; ML-20M NDCG@10 0.5444
  Paper: "we report the average value of five iterations for all methods."

SECOND TARGET  [V] — the committed output of the authors' own notebook
(`WonbinKweon/DRE_WWW2020/DRE_WWW2020.ipynb`, cell 14), ML-1M, k=10, d=300, retrain 150 ep:
    N@10 0.4979   N@20 0.4524   N@50 0.4133   N@100 0.4183

DATA
  ML-1M  : the authors' shipped tensor `u_implicit_5.pt` (6028 x 3533, sum 575242) = their Table 1 row.
  ML-20M : rebuilt here from raw `data/ml-20m/ratings.csv` under their rules. Their stated user
           filter ("at least twenty ratings") is off by one; the rule that reproduces Table 1
           EXACTLY (99,220 users / 20,660 items / 9,478,902 ratings / 99.54% sparsity) is
           `rating > 3.5` then users with STRICTLY MORE THAN 20 positives. Verified 2026-08-05.

FAITHFULNESS — every deviation from the released notebook, and why:
  1. `.cuda()` -> CPU.  Device only; no math changes.
  2. ML-20M held as scipy CSR and densified per batch (a dense 99220 x 20660 float32 tensor is
     8.2 GB and the box has ~6 GB free). Identical arithmetic, different storage.
  3. Batch size for ML-20M: the paper does not state one; the notebook's 128 is its ML-1M config.
     Stated explicitly in the output cell. ML-1M uses the notebook's 128 verbatim.
  Everything else -- init, optimiser, lr, annealing schedule, decoder shape, MSE loss, the 72/8/20
  slice, the retrain-decoder-with-frozen-encoder step, and the metric (seed columns deleted from
  BOTH prediction and ground truth) -- is the notebook verbatim.

PERSISTENCE (CLAUDE.md HARD RULE 11): every (dataset, k, seed) cell is written to the partial JSON
the moment it finishes; a re-run resumes from it and refuses to resume across a changed contract.

Usage:
  python scripts/baselines/dre_native_snap.py --dataset ml1m --k 10 --seeds 1
  python scripts/baselines/dre_native_snap.py --dataset ml1m --k 50 --seeds 5
  python scripts/baselines/dre_native_snap.py --dataset ml20m --k 50 --seeds 1 --epochs 400
"""
import argparse, json, os, time, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, ".cache", "snap3_dre")
OUTDIR = os.path.join(ROOT, "experiments", "baselines")
os.makedirs(OUTDIR, exist_ok=True)

TARGETS = {
    ("ml1m", 50): {"source": "Kweon WWW'20 Table 2 (arXiv:2402.16327 5ex.tex), k=50, avg of 5 runs",
                   "p@10": 0.5396, "p@20": 0.4734, "ndcg@10": 0.5688, "ndcg@20": 0.5197,
                   "best_baseline_ndcg@10": 0.5387, "best_baseline_name": "RMVA"},
    ("ml1m", 10): {"source": "authors' committed notebook output, cell 14 (k=10, d=300, retrain 150ep), single run",
                   "ndcg@10": 0.4979, "ndcg@20": 0.4524, "ndcg@50": 0.4133, "ndcg@100": 0.4183},
    ("ml20m", 50): {"source": "Kweon WWW'20 Table 2 (arXiv:2402.16327 5ex.tex), k=50, avg of 5 runs",
                    "p@10": 0.5712, "p@20": 0.4875, "ndcg@10": 0.6077, "ndcg@20": 0.5373,
                    "best_baseline_ndcg@10": 0.5444, "best_baseline_name": "RMVA"},
}


# --------------------------------------------------------------------------- data
def load_ml1m():
    """The authors' shipped preprocessed tensor, unmodified."""
    t = torch.load(os.path.join(CACHE, "data", "u_implicit_5.pt"), map_location="cpu")
    assert tuple(t.shape) == (6028, 3533) and float(t.sum()) == 575242.0, "shipped tensor changed"
    return sp.csr_matrix(t.numpy())


def load_ml20m():
    """Rebuilt from raw under DRE's rules; asserted against their Table 1."""
    cache = os.path.join(CACHE, "ml20m_dre.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    else:
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "ml-20m", "ratings.csv"),
                         usecols=["userId", "movieId", "rating"],
                         dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
        pos = df[df.rating > 3.5][["userId", "movieId"]]
        del df
        uc = pos.groupby("userId").size()
        pos = pos[pos.userId.isin(set(uc.index[uc > 20]))]        # STRICTLY > 20 -- see docstring
        uids = {u: i for i, u in enumerate(np.sort(pos.userId.unique()))}
        sids = {s: i for i, s in enumerate(np.sort(pos.movieId.unique()))}
        r = pos.userId.map(uids).values
        c = pos.movieId.map(sids).values
        X = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(uids), len(sids)))
        np.savez(cache, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape))
    assert X.shape == (99220, 20660) and int(X.nnz) == 9478902, \
        "ML-20M rebuild does not match DRE Table 1: got %s nnz=%d" % (X.shape, X.nnz)
    return X


def split(X):
    """Notebook cell 2 verbatim: deterministic 72/8/20 slice of the (already shuffled) matrix."""
    n = X.shape[0]
    return X[:int(n * .72)], X[int(n * .72):int(n * .8)], X[int(n * .8):]


# --------------------------------------------------------------------------- model (notebook verbatim)
class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.sm = nn.Softmax(dim=0)
        self.logpi = nn.Parameter(torch.ones(input_size, latent_dim) / input_size)  # uniform init

    def forward(self, x, T):
        g = torch.distributions.Gumbel(0, 1).sample((self.input_size, self.latent_dim))
        y = self.sm((self.logpi + g) / T)
        return torch.matmul(x, y)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim[-1], latent_dim[-2]), nn.Sigmoid(),
            nn.Linear(latent_dim[-2], output_size), nn.Sigmoid())

    def forward(self, z):
        return self.model(z)


# --------------------------------------------------------------------------- metrics (notebook verbatim)
def eval_masked(x_recon, x, seed, ks=(10, 20, 50, 100)):
    """Notebook `NDCG()` extended with P@N. Seed columns deleted from BOTH sides."""
    mask = torch.ones(x.size()[1])
    mask[seed] = 0
    x = x[:, mask == True]
    x_recon = x_recon[:, mask == True]
    out = {}
    kmax = max(ks)
    top = torch.topk(x_recon, kmax, dim=1)[1]
    npos = x.sum(1)
    for k in ks:
        mom = torch.FloatTensor(np.log2(np.arange(2, k + 2)))
        ideal = torch.ones(k) / mom
        hits = torch.gather(x, 1, top[:, :k])
        dcg = (hits / mom).sum(1)
        idcg = torch.stack([ideal[:min(int(npos[u]), k)].sum() for u in range(x.size(0))])
        out["ndcg@%d" % k] = float((dcg / idcg).sum())
        out["p@%d" % k] = float(hits.sum(1).div(k).sum())
    return out


# --------------------------------------------------------------------------- run one cell
def dense(Xs, lo, hi):
    return torch.from_numpy(np.asarray(Xs[lo:hi].todense(), dtype=np.float32))


def run_cell(Xtr, Xva, Xte, k, seed, args, log):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Xtr.shape[1]
    ntr = Xtr.shape[0]
    E = Encoder(m, k)
    De = Decoder([args.hidden, k], m)
    crit = nn.MSELoss()
    oE, oD = optim.Adam(E.parameters(), lr=args.lr), optim.Adam(De.parameters(), lr=args.lr)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        T = args.T0 * ((args.TE / args.T0) ** (epoch / args.epochs))
        E.train(); De.train()
        tot = 0.0
        for lo in range(0, ntr, args.batch):                      # shuffle=False, as in the notebook
            x = dense(Xtr, lo, min(lo + args.batch, ntr))
            oE.zero_grad(); oD.zero_grad()
            loss = crit(De(E(x, T)), x)
            loss.backward(); oE.step(); oD.step()
            tot += float(loss)
        if epoch % args.log_every == 0 or epoch == args.epochs:
            log("    ep %5d/%d T=%.5f train_mse=%.6f (%.0fs)"
                % (epoch, args.epochs, T, tot / max(1, (ntr + args.batch - 1) // args.batch),
                   time.time() - t0))
    itemseed = torch.max(E.logpi, dim=0)[1].tolist()
    n_dup = len(itemseed) - len(set(itemseed))

    # ---- retrain the decoder with the encoder frozen (paper Sec 4.3 / notebook `recommend`)
    De = Decoder([args.hidden, k], m)
    crit2 = nn.MSELoss(reduction="none")
    oD = optim.Adam(De.parameters(), lr=args.retrain_lr)
    idx = np.arange(ntr)
    for epoch in range(args.retrain_epochs):
        rng = np.random.RandomState(seed * 1000 + epoch); rng.shuffle(idx)
        De.train()
        for lo in range(0, ntr, args.retrain_batch):
            b = np.sort(idx[lo:lo + args.retrain_batch])
            y = torch.from_numpy(np.asarray(Xtr[b].todense(), dtype=np.float32))
            x = y[:, itemseed]
            oD.zero_grad()
            l2 = sum(W.norm(2) for W in De.parameters())
            (torch.sum(crit2(De(x), y)) + l2 * args.lamb).backward()
            oD.step()

    De.eval()
    res = {}
    with torch.no_grad():
        for name, Xs in (("val", Xva), ("test", Xte)):
            acc, n = {}, Xs.shape[0]
            for lo in range(0, n, 3000):
                y = dense(Xs, lo, min(lo + 3000, n))
                r = eval_masked(De(y[:, itemseed]), y, itemseed)
                for kk, v in r.items():
                    acc[kk] = acc.get(kk, 0.0) + v
            res[name] = {kk: v / n for kk, v in acc.items()}
    res.update(seconds=round(time.time() - t0, 1), seed=seed, k=k,
               n_duplicate_seeds=n_dup, n_distinct_seeds=len(set(itemseed)),
               itemseed=sorted(set(itemseed)))
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ml1m", "ml20m"], default="ml1m")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--T0", type=float, default=10.0)
    ap.add_argument("--TE", type=float, default=0.1)   # notebook uses 0.01 at k=10; paper Table 3 best is 0.1
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--retrain_epochs", type=int, default=150)
    ap.add_argument("--retrain_lr", type=float, default=0.001)
    ap.add_argument("--retrain_batch", type=int, default=256)
    ap.add_argument("--lamb", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    stem = "DRE_NATIVE_SNAP_%s_k%d%s" % (args.dataset, args.k, args.tag)
    jpath, lpath = os.path.join(OUTDIR, stem + ".json"), os.path.join(OUTDIR, stem + ".log")
    lf = open(lpath, "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); lf.write(s + "\n"); lf.flush()

    contract = hashlib.sha1(json.dumps(
        {kk: v for kk, v in sorted(vars(args).items()) if kk not in ("seeds", "log_every", "tag")}
    ).encode()).hexdigest()[:12]

    state = {"contract": contract, "config": vars(args), "cells": {}}
    if os.path.exists(jpath):
        old = json.load(open(jpath))
        if old.get("contract") == contract:
            state = old; log("resuming from %s (%d cells)" % (jpath, len(state["cells"])))
        else:
            log("contract changed (%s -> %s) -- refusing to resume, starting fresh"
                % (old.get("contract"), contract))

    log("=" * 78)
    log("SNAP-3 DRE native | %s k=%d | contract %s" % (args.dataset, args.k, contract))
    X = load_ml1m() if args.dataset == "ml1m" else load_ml20m()
    log("data %s nnz=%d  (DRE Table 1 verified)" % (X.shape, X.nnz))
    Xtr, Xva, Xte = split(X)
    log("split train=%d val=%d test=%d" % (Xtr.shape[0], Xva.shape[0], Xte.shape[0]))
    tgt = TARGETS.get((args.dataset, args.k))
    log("published target: %s" % json.dumps(tgt))
    state["target"] = tgt
    state["data_stats"] = {"n_users": X.shape[0], "n_items": X.shape[1], "nnz": int(X.nnz)}

    for seed in range(1, args.seeds + 1):
        if str(seed) in state["cells"]:
            log("seed %d already done -- skip" % seed); continue
        log("--- seed %d ---" % seed)
        cell = run_cell(Xtr, Xva, Xte, args.k, seed, args, log)
        state["cells"][str(seed)] = cell
        log("  seed %d test: ndcg@10=%.4f ndcg@20=%.4f p@10=%.4f p@20=%.4f  dup_seeds=%d  (%.0fs)"
            % (seed, cell["test"]["ndcg@10"], cell["test"]["ndcg@20"],
               cell["test"]["p@10"], cell["test"]["p@20"], cell["n_duplicate_seeds"],
               cell["seconds"]))
        # HARD RULE 11: write the cell the moment it completes
        json.dump(state, open(jpath, "w"), indent=1)

    cells = [state["cells"][s] for s in sorted(state["cells"])]
    summ = {}
    for metric in ("ndcg@10", "ndcg@20", "ndcg@50", "ndcg@100", "p@10", "p@20"):
        v = np.array([c["test"][metric] for c in cells])
        summ[metric] = {"mean": round(float(v.mean()), 4),
                        "sd": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
                        "n": len(v)}
    summ["mean_duplicate_seeds"] = float(np.mean([c["n_duplicate_seeds"] for c in cells]))
    state["summary"] = summ
    if tgt:
        state["verdict"] = {m: "target %.4f got %.4f (delta %+.4f)" % (tgt[m], summ[m]["mean"],
                                                                      summ[m]["mean"] - tgt[m])
                            for m in tgt if m.startswith(("ndcg", "p@"))}
    json.dump(state, open(jpath, "w"), indent=1)
    log("SUMMARY %s" % json.dumps(summ))
    log("VERDICT %s" % json.dumps(state.get("verdict")))
    lf.close()


if __name__ == "__main__":
    main()
