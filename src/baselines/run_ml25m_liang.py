"""run_ml25m_liang.py -- CANONICAL G0 full-profile runner (author ruling 2026-07-21).

Replaces run_ml25m.py (arena 500/500 split, now elicitation-line only) as the G0 / full-profile runner.
Evaluates every baseline on the Liang-recipe strong-generalization split applied to ML-25M
(data/ml-25m/proc/, built by scripts/baselines/liang_split.py --data ml-25m): 10k val + 10k test
held-out users, train = remainder, per-user 80/20 fold-in/target, seed 98765.

METRICS via src/baselines/metrics.py (the certified vae_cf port), same CSV loaders as the ML-20M snap:
  PRIMARY:   full NDCG@10  and  tail NDCG@10
  SECONDARY: NDCG@100, Recall@20, Recall@50
  TAIL DEFINITION (documented, computed from the TRAIN matrix -- identical to signed_latent.ndcg10):
    HEAD = the smallest set of items whose TRAIN interaction counts cover 33% of total train mass.
    Concretely: cnt = per-item train interaction count; sort items by DESCENDING cnt; the head is the
    top items up to and INCLUDING the first index where cumulative-mass/total >= 0.33
    (np.searchsorted(cumfrac, 0.33) + 1 items). Tail NDCG@10 masks head-item scores to -inf and drops
    head items from the held-out targets (see metrics.evaluate head_mask). This mask is passed straight
    into metrics.evaluate; NDCG_binary_at_k_batch is reused, not duplicated.

BASELINES (same interfaces / same modules as run_ml25m.py): pop, itemknn, ease, ials, dae, multvae,
  recvae, edlae.
  - iALS: uses the val-selected ML-20M-sweep hyperparams reg=0.1, alpha=50 (NO new sweep).
  - EASE:  lambda=500 verbatim (Steck 2019).
  - EDLAE: sweeps p in {0.3,0.4,0.5,0.6,0.7} on the NEW val (OOM guards: free B, rebuild Gram per p).
  - Neural (dae/multvae/recvae): early-stop on val FULL NDCG@10 (the primary metric); checkpoints to
    .cache/baselines/<name>_ml25m_liang.pt (NEVER overwrites the arena _ml25m.pt checkpoints).

Resumable: skips any baseline whose JSON already exists in experiments/baselines/ml25m_liang/.
Per-baseline <name>.json + a rolling run.log.

Usage:  python src/baselines/run_ml25m_liang.py [--only ease,pop] [--max_minutes 600]
NOTE: FULL run (no data caps). Do NOT launch ease/ials/edlae/neural until the review gate clears
(HARD RULE 10: training code committed first).
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import pop, itemknn, ease, ials, multvae, recvae, edlae

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")
LOGP = os.path.join(OUTDIR, "run.log")
CKPT = os.path.join(_ROOT, ".cache", "baselines")

# iALS: val-selected from the ML-20M sweep (no new sweep on ML-25M -- author directive 2026-07-21).
IALS_HP = {"factors": 200, "reg": 0.1, "alpha": 50.0, "iterations": 15}


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def compute_head_mask(train, n_items):
    """HEAD = smallest item set covering 33% of TRAIN interaction mass (signed_latent convention)."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order_pop = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order_pop]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order_pop[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head, cnt


def build(name, train, n_items, D, head_mask, max_minutes, log):
    """Return (predict_fn, hp_dict). Same module interfaces as run_ml25m.py."""
    va_tr, va_te = D["val"]

    def neural_eval(predict):
        # early-stop on the PRIMARY metric: val full NDCG@10 (head_mask also gives us tail for logging)
        m = M.evaluate(predict, va_tr, va_te, batch_size=500, head_mask=head_mask)
        return m["ndcg@10"], m

    ck = os.path.join(CKPT, f"{name}_ml25m_liang.pt")
    os.makedirs(CKPT, exist_ok=True)
    import argparse as ap
    if name == "pop":
        return pop.fit(train, n_items, log=log), {}
    if name == "itemknn":
        pr = itemknn.fit(train, n_items, args=ap.Namespace(topk=itemknn.DEFAULTS["topk"]), log=log)
        return pr, {"topk": pr.topk}
    if name == "ease":
        pr = ease.fit(train, n_items, args=ap.Namespace(lam=ease.DEFAULTS["lam"]), log=log)
        return pr, {"lambda": pr.lam}
    if name == "ials":
        pr = ials.fit(train, n_items, args=ap.Namespace(**IALS_HP), log=log)
        return pr, pr.hp
    if name == "edlae":
        best_v, best_p = -1.0, None
        from ease import build_gram
        from edlae import edlae_B
        import gc
        B = None
        for pp in (0.3, 0.4, 0.5, 0.6, 0.7):
            B = None; gc.collect()               # free prev B BEFORE the next ~2.7GB Gram (OOM guard)
            G = build_gram(train)                # edlae_B destroys G -> rebuild per p
            B = edlae_B(G, pp); del G; gc.collect()
            v = M.evaluate(lambda Xc, _B=B: np.asarray(Xc @ _B, np.float32), va_tr, va_te)["ndcg@10"]
            log(f"  [edlae] VAL p={pp} full_ndcg@10={v:.4f}")
            if v > best_v:
                best_v, best_p = v, pp
        B = None; gc.collect()
        G = build_gram(train)
        B = edlae_B(G, best_p); del G; gc.collect()

        def pr(Xc, _B=B):
            return np.asarray(Xc @ _B, np.float32)
        return pr, {"p": best_p}
    if name in ("dae", "multvae"):
        a = multvae._defaults("dae" if name == "dae" else "vae"); a.max_minutes = max_minutes
        pr = multvae.fit(train, n_items, evaluator=neural_eval, args=a, ckpt=ck, log=log)
        return pr, {"best_val": pr.best_val, "best_epoch": pr.best_epoch, "mode": a.mode}
    if name == "recvae":
        a = recvae._defaults(); a.max_minutes = max_minutes
        pr = recvae.fit(train, n_items, evaluator=neural_eval, args=a, ckpt=ck, log=log)
        return pr, {"best_val": pr.best_val, "best_epoch": pr.best_epoch}
    raise ValueError(name)


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

    logln(f"=== ml25m_liang run start; order={order} ===")
    if not os.path.exists(os.path.join(PROC, "meta.json")):
        raise SystemExit(f"[run] {PROC} not built. Run: python scripts/baselines/liang_split.py --data ml-25m")
    meta = M.load_meta(PROC); n_items = meta["n_items"]
    logln(f"Liang ML-25M split: n_items={n_items} meta={meta}")
    train = M.load_train(n_items, PROC)
    D = {"val": M.load_val(n_items, PROC)}
    te_tr, te_te = M.load_test(n_items, PROC)
    head_mask, cnt = compute_head_mask(train, n_items)
    logln(f"head/tail: {int(head_mask.sum())} head items (top-33% train mass) / "
          f"{int((~head_mask).sum())} tail items; train_users={train.shape[0]} nnz={train.nnz}")

    for name in order:
        outp = os.path.join(OUTDIR, f"{name}.json")
        if os.path.exists(outp):
            logln(f"[skip] {name}: {outp} exists")
            continue
        logln(f"--- {name} ---")
        t0 = time.time()
        predict, hp = build(name, train, n_items, D, head_mask, args.max_minutes, logln)
        res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=head_mask)
        res["seconds"] = time.time() - t0
        res["hp"] = hp
        res["n_items"] = n_items
        json.dump(res, open(outp, "w"), indent=2)
        logln(f"[done] {name}: full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
              f"ndcg@100={res['ndcg@100']:.4f} r@20={res['recall@20']:.4f} r@50={res['recall@50']:.4f} "
              f"{res['seconds']/60:.1f}m")
    logln("=== ml25m_liang run complete ===")


if __name__ == "__main__":
    main()
