"""run_snap_ml20m.py -- sequential, resumable runner that snaps every baseline to its PUBLISHED
ML-20M (Liang strong-generalization split) number.

Order: pop, itemknn, ease, ials, dae, multvae, recvae, edlae.
Resumable: skips any baseline whose result JSON already exists in experiments/baselines/snap_ml20m/.
Each baseline logs its metrics + wall-time to <name>.json and appends to a rolling run.log. Prints a
PASS/FAIL vs the published NDCG@100 at +/-0.005 tolerance (verdict is advisory; the reviewer decides).

Neural baselines (dae, multvae, recvae) train with val-NDCG@100 early stopping and checkpoint to
.cache/baselines/<name>.pt (mid-training resume). Closed-form baselines fit once.

Usage:  python scripts/baselines/run_snap_ml20m.py [--only ease,recvae] [--max_minutes 600]
NOTE: this is the FULL run (all held-out users, full train). Do not launch until the review gate clears.
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import pop, itemknn, ease, ials, multvae, recvae, edlae

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "snap_ml20m")
LOGP = os.path.join(OUTDIR, "run.log")
CKPT = os.path.join(_ROOT, ".cache", "baselines")

# Published NDCG@100 anchors on the ML-20M Liang split. approx=True -> no single canonical constant
# (verdict left to reviewer, tolerance advisory). Sources: Mult-VAE (Liang WWW'18), RecVAE (WSDM'20),
# EASE (WWW'19), EDLAE (NeurIPS'20), landscape findings c4.
TARGETS = {
    "pop":     {"ndcg@100": None,  "note": "non-personalized floor (no published anchor)"},
    "itemknn": {"ndcg@100": 0.36,  "approx": True, "note": "~0.36 class (Cremonesi-era item-kNN)"},
    "ease":    {"ndcg@100": 0.420, "recall@20": 0.391, "recall@50": 0.521},
    "ials":    {"ndcg@100": 0.386, "approx": True, "note": "WMF/iALS reported ~0.386 (tuned)"},
    "dae":     {"ndcg@100": 0.419, "recall@20": 0.387, "recall@50": 0.524},
    "multvae": {"ndcg@100": 0.426, "recall@20": 0.395, "recall@50": 0.537},
    "recvae":  {"ndcg@100": 0.442, "recall@20": 0.414, "recall@50": 0.553},
    "edlae":   {"ndcg@100": 0.427, "approx": True, "note": "EDLAE full-rank ~RecVAE class (approx)"},
}
TOL = 0.005


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def build(name, train, n_items, D, max_minutes, log):
    """Return (predict_fn, hp_dict)."""
    va_tr, va_te = D["val"]

    def neural_eval(predict):
        m = M.evaluate(predict, va_tr, va_te, batch_size=500)
        return m["ndcg@100"], m

    ck = os.path.join(CKPT, f"{name}_ml20m.pt")
    os.makedirs(CKPT, exist_ok=True)
    if name == "pop":
        return pop.fit(train, n_items, log=log), {}
    if name == "itemknn":
        import argparse as ap
        pr = itemknn.fit(train, n_items, args=ap.Namespace(topk=itemknn.DEFAULTS["topk"]), log=log)
        return pr, {"topk": pr.topk}
    if name == "ease":
        import argparse as ap
        pr = ease.fit(train, n_items, args=ap.Namespace(lam=ease.DEFAULTS["lam"]), log=log)
        return pr, {"lambda": pr.lam}
    if name == "ials":
        # small val sweep (no canonical constant)
        best_v, best = -1.0, None
        import argparse as ap
        for reg in (0.001, 0.01, 0.1):
            for alpha in (1.0, 10.0, 50.0):
                pr = ials.fit(train, n_items, args=ap.Namespace(
                    factors=200, reg=reg, alpha=alpha, iterations=15), log=log)
                v = M.evaluate(pr, va_tr, va_te)["ndcg@100"]
                log(f"  [ials] VAL reg={reg} alpha={alpha} ndcg@100={v:.4f}")
                if v > best_v:
                    best_v, best = v, (reg, alpha)
        pr = ials.fit(train, n_items, args=ap.Namespace(
            factors=200, reg=best[0], alpha=best[1], iterations=15), log=log)
        return pr, pr.hp
    if name == "edlae":
        best_v, best_p = -1.0, None
        from ease import build_gram
        from edlae import edlae_B
        import numpy as np
        import gc
        B = None
        for pp in (0.3, 0.4, 0.5, 0.6, 0.7):
            B = None; gc.collect()               # free prev B BEFORE the next 3.2GB Gram (OOM guard)
            G = build_gram(train)               # edlae_B destroys G -> rebuild per p (~12 s)
            B = edlae_B(G, pp); del G; gc.collect()
            v = M.evaluate(lambda Xc, _B=B: np.asarray(Xc @ _B, np.float32), va_tr, va_te)["ndcg@100"]
            log(f"  [edlae] VAL p={pp} ndcg@100={v:.4f}")
            if v > best_v:
                best_v, best_p = v, pp
        B = None; gc.collect()
        G = build_gram(train)
        B = edlae_B(G, best_p); del G; gc.collect()

        def pr(Xc, _B=B):
            import numpy as np
            return np.asarray(Xc @ _B, np.float32)
        return pr, {"p": best_p}
    if name in ("dae", "multvae"):
        a = multvae._defaults("dae" if name == "dae" else "vae")
        a.max_minutes = max_minutes
        pr = multvae.fit(train, n_items, evaluator=neural_eval, args=a, ckpt=ck, log=log)
        return pr, {"best_val": pr.best_val, "best_epoch": pr.best_epoch, "mode": a.mode}
    if name == "recvae":
        a = recvae._defaults(); a.max_minutes = max_minutes
        pr = recvae.fit(train, n_items, evaluator=neural_eval, args=a, ckpt=ck, log=log)
        return pr, {"best_val": pr.best_val, "best_epoch": pr.best_epoch}
    raise ValueError(name)


def verdict(name, ndcg100):
    tgt = TARGETS[name]
    t = tgt.get("ndcg@100")
    if t is None:
        return f"(floor, target n/a) got {ndcg100:.4f}"
    d = ndcg100 - t
    if tgt.get("approx"):
        return f"~target {t:.3f} got {ndcg100:.4f} (delta {d:+.4f}) [APPROX - reviewer decides]"
    ok = abs(d) <= TOL
    return f"target {t:.3f} got {ndcg100:.4f} (delta {d:+.4f}) [{'PASS' if ok else 'FAIL'} @+/-{TOL}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of baseline names")
    ap.add_argument("--max_minutes", type=float, default=1e9, help="per-neural-baseline wall budget")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    order = ["pop", "itemknn", "ease", "ials", "dae", "multvae", "recvae", "edlae"]
    if args.only:
        want = set(args.only.split(","))
        order = [n for n in order if n in want]

    logln(f"=== snap_ml20m run start; order={order} ===")
    meta = M.load_meta(); n_items = meta["n_items"]
    logln(f"Liang split: n_items={n_items} meta={meta}")
    train = M.load_train(n_items)
    D = {"val": M.load_val(n_items)}
    te_tr, te_te = M.load_test(n_items)

    for name in order:
        outp = os.path.join(OUTDIR, f"{name}.json")
        if os.path.exists(outp):
            logln(f"[skip] {name}: {outp} exists")
            continue
        logln(f"--- {name} ---")
        t0 = time.time()
        predict, hp = build(name, train, n_items, D, args.max_minutes, logln)
        res = M.evaluate(predict, te_tr, te_te, batch_size=500)
        res["seconds"] = time.time() - t0
        res["hp"] = hp
        res["target"] = TARGETS[name]
        res["verdict"] = verdict(name, res["ndcg@100"])
        json.dump(res, open(outp, "w"), indent=2)
        logln(f"[done] {name}: ndcg@100={res['ndcg@100']:.4f} r@20={res['recall@20']:.4f} "
              f"r@50={res['recall@50']:.4f} {res['seconds']/60:.1f}m | {res['verdict']}")
    logln("=== snap_ml20m run complete ===")


if __name__ == "__main__":
    main()
