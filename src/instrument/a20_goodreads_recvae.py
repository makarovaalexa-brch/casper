"""A20 -- RecVAE on the Goodreads arena, the frozen decoder the instrument is built on.

Design sheet: docs/design/DS_A17_GOODREADS_ARENA.md

This is `run_ml25m_liang.py`'s recvae arm pointed at the Goodreads proc. NOTHING about the recipe is
changed: it imports the SAME canonical module (`src/baselines/recvae.py`) with its SAME `_defaults()`
-- epochs 50, batch 500, lr 5e-4, gamma 0.005, hidden 600, latent 200, n_enc 3, n_dec 1, dropout 0.5,
patience 10 -- and the SAME evaluator (early-stop on validation full NDCG@10), the SAME head-mask
convention (smallest item set covering 33% of train interaction mass), and the SAME checkpointing.

⚠ Note for anyone reading the earlier goodbooks-10k detour: that run used
`scripts/instrument2/recvae.py`, a DIFFERENT module whose `run_updates` does not alternate the
encoder and decoder and never calls `update_prior()`. The canonical module does both. The goodbooks
number is therefore not comparable to this one and is not cited.

PARITY BAR to clear, from the ML-25M certified run (`experiments/baselines/ml25m_liang/recvae.json`):
    best val full NDCG@10 0.3512 at epoch 24; TEST full 0.3540 / tail 0.2497 on 18,359 items.
That is the number the ML-25M instrument was later certified against. The Goodreads instrument will
be certified against whatever RecVAE reaches HERE -- a dataset-local bar, since no one has published
this protocol on Goodreads.

HARD RULE 11: `recvae.fit` checkpoints every epoch and resumes, so a killed run loses one epoch.

Run:  python src/instrument/a20_goodreads_recvae.py
"""
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import metrics as M          # noqa: E402
import recvae                # noqa: E402

PROC = os.path.join(_ROOT, "data", "goodreads_ucsd", "proc")
OUT = os.path.join(_ROOT, "experiments", "goodreads")
CKPT = os.path.join(_ROOT, ".cache", "goodreads")
LABEL = "recvae_goodreads"
ML25M_BAR = {"best_val_ndcg@10": 0.3512, "test_ndcg@10": 0.3540, "test_tail_ndcg@10": 0.2497,
             "best_epoch": 24, "n_items": 18359}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    logp = os.path.join(OUT, f"{LABEL}.log")

    def log(*a):
        line = " ".join(str(x) for x in a)
        line = f"[{time.strftime('%H:%M:%S')}] {line}"
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.time()
    meta = M.load_meta(PROC)
    n_items = meta["n_items"]
    train = M.load_train(n_items, PROC)
    va_tr, va_te = M.load_val(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    log(f"[data] {meta['dataset']}: {n_items} items, train {train.shape} nnz={train.nnz}, "
        f"val {va_tr.shape[0]} users, test {te_tr.shape[0]} users")
    assert train.shape[0] == meta["n_train_users"], \
        f"train matrix has {train.shape[0]} rows but meta says {meta['n_train_users']} users -- " \
        f"uids are probably not contiguous from 0"

    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order_pop = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order_pop]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order_pop[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    log(f"[head/tail] {int(head.sum())} head items (top-33% train mass) / "
        f"{int((~head).sum())} tail items")

    def neural_eval(predict):
        m = M.evaluate(predict, va_tr, va_te, batch_size=500, head_mask=head)
        return m["ndcg@10"], m

    a = recvae._defaults()
    log(f"[recipe] {vars(a)}")
    ck = os.path.join(CKPT, f"{LABEL}.pt")
    pr = recvae.fit(train, n_items, evaluator=neural_eval, args=a, ckpt=ck, log=log)

    m = M.evaluate(pr, te_tr, te_te, batch_size=500, head_mask=head)
    res = {"label": LABEL, "dataset": meta["dataset"], "n_items": n_items,
           "ndcg@10": m["ndcg@10"], "tail_ndcg@10": m.get("tail_ndcg@10"),
           "ndcg@100": m.get("ndcg@100"), "recall@20": m.get("recall@20"),
           "recall@50": m.get("recall@50"),
           "ml25m_bar": ML25M_BAR, "checkpoint": ck, "split_meta": meta,
           "seconds": time.time() - t0}
    json.dump(res, open(os.path.join(OUT, f"{LABEL}.json"), "w"), indent=2)
    log(f"[TEST] full NDCG@10 {m['ndcg@10']:.4f} tail {m.get('tail_ndcg@10', float('nan')):.4f}   "
        f"(ML-25M's RecVAE: {ML25M_BAR['test_ndcg@10']:.4f}/{ML25M_BAR['test_tail_ndcg@10']:.4f})")
    log(f"[done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
