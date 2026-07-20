"""run_ml25m.py -- run the SAME baseline code on CASPER's canonical ML-25M strong-generalization harness.

The harness is REUSED verbatim from scripts/signed_latent.py (the eval that produced the canonical
pbC set-encoder full/tail NDCG@10 = 0.4946/0.3372 and reproduces RecVAE-d512 0.4998/0.3443):
  - split:  SL.load_arena_base() + SL.build_splits(base, seed)  (per-user half-split, arena convention)
  - eval cohort:  SL.cohort(base, SPL, "test") = te(500) MINUS the 300 quarantined study users
  - tail:  SL.ndcg10(..., tail=True) excludes head (top-33% popular) items and head targets
  - seed-avg over SL.SEEDS = [1,2,3,7,11]
This is the SAME split/eval/tail-definition as the interview recommender; NOT reinvented.

Train matrix X = (train users) x (ni items), binary r>=4 likes (base tr_u/tr_i) -- the same signal the
in-house EASE/RecVAE rulers in signed_latent use. Fold-in for a held-out user = their profile-half
likes (binary). PRIMARY metric: full + tail NDCG@10. SECONDARY: NDCG@100 (full, profile-masked).

Results -> experiments/baselines/ml25m/<name>.json. Resumable (skips existing JSON). Neural baselines
train on X with the arena VAL NDCG@10 as the early-stop signal (adapted from NDCG@100; documented).

Usage:  python scripts/baselines/run_ml25m.py [--only ease,pop] [--max_minutes 600]
NOTE: FULL run (no data caps). Do not launch until the review gate clears.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "instrument2"))

import signed_latent as SL   # canonical ML-25M harness (load_arena_base, build_splits, cohort, ndcg10)
import pop, itemknn, ease, ials, multvae, recvae, edlae

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m")
LOGP = os.path.join(OUTDIR, "run.log")
CKPT = os.path.join(_ROOT, ".cache", "baselines")
_W100 = 1.0 / np.log2(np.arange(2, 102))


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def build_train_csr(base):
    """X = (unique train users) x ni, binary likes. Same signal as SL.build_gram."""
    ni = base["ni"]
    uniqU, inv = np.unique(base["tr_u"], return_inverse=True)
    X = sparse.csr_matrix((np.ones(len(inv), np.float32), (inv, base["tr_i"])),
                          shape=(len(uniqU), ni), dtype=np.float32)
    return X


def ndcg_at_100_full(score, tlike, profset):
    """Full-catalog NDCG@100, profile items masked (Liang/arena convention). No tail restriction."""
    s = score.copy()
    s[list(profset)] = -1e30
    rel = set(tlike)
    if not rel:
        return None
    o = np.argpartition(-s, 100)[:100]
    o = o[np.argsort(-s[o])]
    dcg = sum(_W100[p] for p, t in enumerate(o) if int(t) in rel)
    idcg = _W100[:min(100, len(rel))].sum() + 1e-12
    return dcg / idcg


def _collect(base, SPL, users):
    """Build fold-in CSR (n x ni binary profile-half likes) + per-user (profset, tlike)."""
    ni = base["ni"]
    rows, cols, metas = [], [], []
    r = 0
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= SL.LO]
        if not tlike:
            continue
        liked_prof = [j for j in prof_r if prof_r[j] >= SL.LO]
        for j in liked_prof:
            rows.append(r); cols.append(j)
        metas.append((profset, tlike))
        r += 1
    X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(r, ni), dtype=np.float32)
    return X, metas


def evaluate_ml25m(predict, base, seeds, headmask, batch=500):
    """Seed-avg full/tail NDCG@10 (SL.ndcg10) + full NDCG@100."""
    accF, accT, acc100 = [], [], []
    for seed in seeds:
        SPL = SL.build_splits(base, seed)
        users = SL.cohort(base, SPL, "test")
        X, metas = _collect(base, SPL, users)
        ff, tt, hh = [], [], []
        for st in range(0, X.shape[0], batch):
            en = min(st + batch, X.shape[0])
            sc = predict(X[st:en]).astype(np.float64)
            for i in range(en - st):
                profset, tlike = metas[st + i]
                nf = SL.ndcg10(sc[i], tlike, profset, headmask, False)
                nt = SL.ndcg10(sc[i], tlike, profset, headmask, True)
                n100 = ndcg_at_100_full(sc[i], tlike, profset)
                if nf is not None: ff.append(nf)
                if nt is not None: tt.append(nt)
                if n100 is not None: hh.append(n100)
        accF.append(np.mean(ff)); accT.append(np.mean(tt)); acc100.append(np.mean(hh))
    return {"full_ndcg@10": float(np.mean(accF)), "tail_ndcg@10": float(np.mean(accT)),
            "full_ndcg@100": float(np.mean(acc100)),
            "full_ndcg@10_seedsd": float(np.std(accF)), "tail_ndcg@10_seedsd": float(np.std(accT))}


def val_ndcg10(base, headmask):
    """Neural early-stop signal: arena VAL cohort full NDCG@10 (adapted from Liang NDCG@100)."""
    SPL = SL.build_splits(base, SL.SEEDS[0])
    users = SL.cohort(base, SPL, "val")
    X, metas = _collect(base, SPL, users)

    def ev(predict):
        ff = []
        for st in range(0, X.shape[0], 500):
            en = min(st + 500, X.shape[0])
            sc = predict(X[st:en]).astype(np.float64)
            for i in range(en - st):
                profset, tlike = metas[st + i]
                nf = SL.ndcg10(sc[i], tlike, profset, headmask, False)
                if nf is not None: ff.append(nf)
        v = float(np.mean(ff))
        return v, {"val_full_ndcg@10": v}
    return ev


def build(name, X, ni, base, headmask, max_minutes, log):
    ck = os.path.join(CKPT, f"{name}_ml25m.pt"); os.makedirs(CKPT, exist_ok=True)
    import argparse as ap
    if name == "pop":
        return pop.fit(X, ni, log=log), {}
    if name == "itemknn":
        pr = itemknn.fit(X, ni, args=ap.Namespace(topk=itemknn.DEFAULTS["topk"]), log=log)
        return pr, {"topk": pr.topk}
    if name == "ease":
        pr = ease.fit(X, ni, args=ap.Namespace(lam=ease.DEFAULTS["lam"]), log=log)
        return pr, {"lambda": pr.lam}
    if name == "ials":
        pr = ials.fit(X, ni, args=ap.Namespace(factors=200, reg=0.01, alpha=10.0, iterations=15), log=log)
        return pr, pr.hp
    if name == "edlae":
        from ease import build_gram
        from edlae import edlae_B
        G = build_gram(X); B = edlae_B(G, 0.5); del G

        def pr(Xc, _B=B):
            return np.asarray(Xc @ _B, np.float32)
        return pr, {"p": 0.5}
    if name in ("dae", "multvae"):
        a = multvae._defaults("dae" if name == "dae" else "vae"); a.max_minutes = max_minutes
        pr = multvae.fit(X, ni, evaluator=val_ndcg10(base, headmask), args=a, ckpt=ck, log=log)
        return pr, {"best_val": pr.best_val, "best_epoch": pr.best_epoch, "mode": a.mode}
    if name == "recvae":
        a = recvae._defaults(); a.max_minutes = max_minutes
        pr = recvae.fit(X, ni, evaluator=val_ndcg10(base, headmask), args=a, ckpt=ck, log=log)
        return pr, {"best_val": pr.best_val, "best_epoch": pr.best_epoch}
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    order = ["pop", "itemknn", "ease", "ials", "dae", "multvae", "recvae", "edlae"]
    if args.only:
        want = set(args.only.split(",")); order = [n for n in order if n in want]

    logln(f"=== ml25m baselines start; order={order} ===")
    base = SL.load_arena_base(); ni = base["ni"]; headmask = base["headmask"]
    SL._POP = base["cnt"].astype(np.float64)
    X = build_train_csr(base)
    logln(f"ML-25M harness: ni={ni} train_users={X.shape[0]} nnz={X.nnz} "
          f"seeds={SL.SEEDS} (test = te minus 300 study)")

    for name in order:
        outp = os.path.join(OUTDIR, f"{name}.json")
        if os.path.exists(outp):
            logln(f"[skip] {name}: exists"); continue
        logln(f"--- {name} ---")
        t0 = time.time()
        predict, hp = build(name, X, ni, base, headmask, args.max_minutes, logln)
        res = evaluate_ml25m(predict, base, SL.SEEDS, headmask)
        res["seconds"] = time.time() - t0; res["hp"] = hp
        json.dump(res, open(outp, "w"), indent=2)
        logln(f"[done] {name}: full@10={res['full_ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
              f"full@100={res['full_ndcg@100']:.4f} {res['seconds']/60:.1f}m")
    logln("=== ml25m baselines complete ===")


if __name__ == "__main__":
    main()
