"""distill_edlae.py -- T3 trainer: a RecVAE STUDENT distilled from an EDLAE TEACHER.

Goal (the T3 rung): keep RecVAE's inductive form (a cheap, foldable amortized encoder) but pull its
ranking toward EDLAE's strong item-item structure via knowledge distillation, so the student inherits
EDLAE-grade full-profile accuracy while remaining a set-encoder we can fold interview answers into.

ARCHITECTURE: RecVAE, IMPORTED from recvae.py (recvae.RecVAE + its Encoder/CompositePrior/decoder and
log_norm_pdf) -- NOT copied. We reuse the model's own encoder/decoder/prior modules and only replace the
LOSS.  (One documented training-loop deviation: recvae alternates 3 encoder : 1 decoder sub-steps per
epoch; the distillation loss couples encoder+decoder+KD, so we optimize ALL params jointly with a single
Adam -- standard for a KD objective. update_prior() is still called each epoch to refresh the composite
prior's q_old, exactly as recvae does.)

LOSS (per batch, x = binary train row):
    recon_term = alpha * CE(student, teacher)  +  (1 - alpha) * multinomial_NLL(student, x)
    loss       = recon_term  +  kl_weight * KL(q(z|x) || composite_prior)          # RecVAE ELBO reg.
  where
    student log-probs = log_softmax(decoder(z)),  z ~ q(z|x)   (RecVAE reparam)
    multinomial_NLL   = -mean_u sum_i x_ui * log_softmax(student)_ui               # Liang WWW2018 term
    CE(student,teacher) = -mean_u sum_i teacher_prob_ui * log_softmax(student)_ui   # KD cross-entropy
    kl_weight = gamma * |x_u|   (gamma=0.005, recvae's per-user beta')

TEACHER, on the fly per batch (documented choices):
    teacher logits  = x @ B            (B = EDLAE item-item matrix, zero diagonal; Steck 2020)
    teacher_prob    = TOP-K TRUNCATED SOFTMAX with TEMPERATURE T on those logits:
        for each user, keep the top-k teacher logits, divide by T, softmax over the kept support, zero
        elsewhere.  top-k (default 1000) concentrates distillation on the teacher's confident items and
        bounds the CE support; T (default 2.0) softens the target (classic Hinton KD).  We do NOT mask
        held-in items -- the reconstruction/KD target is over the FULL catalogue (Mult-VAE convention).
        Temperature is applied to the TEACHER only; the student uses log_softmax at T=1 (documented).

  B is produced by:  python src/baselines/edlae.py --p <best> --export_B <path>.npy   (flag added there).

CHECKPOINT/RESUME + early stop on VAL FULL NDCG@10 (the primary metric), mirroring recvae.fit.
Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import recvae as R           # RecVAE architecture (imported, not copied)

torch.manual_seed(98765)
np.random.seed(98765)


def _teacher_probs(x_dense_np, B, topk, temp):
    """teacher_prob = top-k truncated softmax(x@B / T). x_dense_np: (b x m) float32. Returns torch (b x m)."""
    logits = x_dense_np @ B                                   # (b x m) EDLAE scores
    b, m = logits.shape
    kk = min(topk, m)
    idx = np.argpartition(-logits, kk - 1, axis=1)[:, :kk]    # top-k columns per row
    row = np.arange(b)[:, None]
    top = logits[row, idx] / float(temp)                     # (b x kk) temperature-softened
    top = top - top.max(axis=1, keepdims=True)
    ex = np.exp(top); ex /= ex.sum(axis=1, keepdims=True)    # softmax over kept support
    probs = np.zeros((b, m), dtype=np.float32)
    probs[row, idx] = ex.astype(np.float32)
    return torch.from_numpy(probs)


def _forward_losses(model, x, B, alpha, topk, temp, gamma):
    """Return (loss, kd, data_nll, kld) reusing RecVAE's encoder/decoder/prior modules."""
    mu, logvar = model.encoder(x, dropout_rate=model_dropout(model))
    z = model.reparameterize(mu, logvar)
    x_pred = model.decoder(z)                                 # student logits
    log_sm = F.log_softmax(x_pred, dim=-1)
    data_nll = -(log_sm * x).sum(dim=-1).mean()               # multinomial NLL (data term)
    with torch.no_grad():
        t_prob = _teacher_probs(x.detach().cpu().numpy(), B, topk, temp)
    kd = -(t_prob * log_sm).sum(dim=-1).mean()                # KD cross-entropy
    kl_weight = gamma * x.sum(dim=-1)
    kld = (R.log_norm_pdf(z, mu, logvar) - model.prior(x, z)).sum(dim=-1).mul(kl_weight).mean()
    recon = alpha * kd + (1.0 - alpha) * data_nll
    return recon + kld, kd, data_nll, kld


_DROPOUT = 0.5
def model_dropout(model):
    return _DROPOUT if model.training else 0.0


def make_predict_fn(model):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_csr.toarray(), dtype=torch.float32)
            return model(x, calculate_loss=False).cpu().numpy().astype(np.float32)
    return predict


def _defaults():
    return argparse.Namespace(
        epochs=50, batch=500, lr=5e-4, gamma=0.005, hidden=600, latent=200,
        alpha=0.5, topk=1000, temp=2.0, dropout=0.5, patience=10, max_minutes=1e9,
        resume=True, teacher_npy=None, mmap_teacher=False)


def _load_teacher(path, mmap):
    B = np.load(path, mmap_mode="r" if mmap else None)
    return np.asarray(B, dtype=np.float32)


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or _defaults()
    global _DROPOUT; _DROPOUT = a.dropout
    if not getattr(a, "teacher_npy", None) or not os.path.exists(a.teacher_npy):
        raise SystemExit(f"[distill] teacher B npy not found: {getattr(a,'teacher_npy',None)}. "
                         f"Produce it: python src/baselines/edlae.py --p <best> --export_B <path>.npy")
    B = _load_teacher(a.teacher_npy, getattr(a, "mmap_teacher", False))
    assert B.shape == (n_items, n_items), f"teacher B {B.shape} != ({n_items},{n_items})"
    log(f"[distill] teacher B {B.shape} loaded (mmap={a.mmap_teacher}); alpha={a.alpha} "
        f"topk={a.topk} T={a.temp}")
    model = R.RecVAE(a.hidden, a.latent, n_items)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)      # JOINT optimizer (documented deviation)
    N = train.shape[0]
    state = {"epoch": 0, "best": -1.0, "best_state": None, "best_epoch": 0, "history": []}
    if ckpt and getattr(a, "resume", True) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        log(f"[distill] RESUMED ep{state['epoch']} best={state['best']:.4f}")
    predict = make_predict_fn(model)
    idxlist = np.arange(N); t0 = time.time()
    for epoch in range(state["epoch"], a.epochs):
        model.train(); np.random.shuffle(idxlist); ep = {"kd": 0.0, "nll": 0.0, "kl": 0.0}; nb = 0
        for st in range(0, N, a.batch):
            X = train[idxlist[st:min(st + a.batch, N)]]
            x = torch.tensor(X.toarray(), dtype=torch.float32)
            opt.zero_grad()
            loss, kd, nll, kld = _forward_losses(model, x, B, a.alpha, a.topk, a.temp, a.gamma)
            loss.backward(); opt.step()
            ep["kd"] += float(kd); ep["nll"] += float(nll); ep["kl"] += float(kld); nb += 1
        model.update_prior()                                 # refresh composite prior q_old (as recvae)
        state["epoch"] = epoch + 1
        sc, _ = evaluator(predict) if evaluator is not None else (-1.0, {})
        el = (time.time() - t0) / 60.0
        state["history"].append({"epoch": epoch + 1, "kd": ep["kd"]/max(nb,1),
                                 "nll": ep["nll"]/max(nb,1), "val": sc, "min": round(el, 1)})
        log(f"[distill] ep{epoch+1:3d} kd {ep['kd']/max(nb,1):.4f} nll {ep['nll']/max(nb,1):.4f} "
            f"kl {ep['kl']/max(nb,1):.4f} val {sc:.4f} {el:.1f}m")
        if sc > state["best"]:
            state["best"] = sc; state["best_epoch"] = epoch + 1
            state["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                        "args": vars(a)}, ckpt)
        if evaluator is not None and (epoch + 1) - state["best_epoch"] >= a.patience:
            log(f"[distill] early stop @ep{epoch+1} best {state['best']:.4f} @ep{state['best_epoch']}")
            break
        if el >= a.max_minutes:
            log(f"[distill] wall budget {a.max_minutes}m hit @ep{epoch+1}; re-run to resume")
            break
    if state.get("best_state") is not None:
        model.load_state_dict(state["best_state"])
    predict.best_val = state["best"]; predict.best_epoch = state["best_epoch"]
    return predict


# --------------------------------------------------------------------------- CLI / smoke
def _head_mask(train, n_items):
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head


def _smoke():
    print("[distill][SMOKE] synthetic tiny data + synthetic teacher B (code-path only, NOT the split)")
    from scipy import sparse
    rng = np.random.RandomState(0)
    n_users, n_items = 120, 130   # >100 items: metrics.evaluate uses NDCG@100 / Recall@50
    X = (sparse.random(n_users, n_items, density=0.2, random_state=rng,
                       data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    tr, te_tr = X[:90], X[90:]
    te_te = (sparse.random(30, n_items, density=0.12, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    # synthetic teacher B (zero-diagonal item-item), saved to a temp npy to exercise the load path
    Bt = rng.randn(n_items, n_items).astype(np.float32) * 0.1
    np.fill_diagonal(Bt, 0.0)
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_teacherB.npy")
    np.save(tmp, Bt)
    hm = _head_mask(tr, n_items)
    a = _defaults(); a.epochs = 3; a.batch = 30; a.hidden = 32; a.latent = 8
    a.topk = 10; a.temp = 2.0; a.alpha = 0.5; a.patience = 99; a.teacher_npy = tmp
    try:
        pr = fit(tr, n_items, evaluator=lambda p: (M.evaluate(p, te_tr, te_te, batch_size=15,
                                                              head_mask=hm)["ndcg@10"], {}),
                 args=a, ckpt=None)
        res = M.evaluate(pr, te_tr, te_te, batch_size=15, head_mask=hm)
        print(f"[distill][SMOKE] OK  full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
              f"best_val={pr.best_val:.4f}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--teacher_npy", default=None, help="EDLAE B .npy from edlae.py --export_B")
    ap.add_argument("--mmap_teacher", action="store_true", help="mmap B (saves ~1.35GB RAM, slower)")
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--tag", default="distill_edlae_ml25m_liang")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args0 = ap.parse_args()
    if args0.smoke:
        _smoke(); return
    _HERE = os.path.dirname(os.path.abspath(__file__))
    PROC = os.path.abspath(os.path.join(_HERE, "..", "..", "data", "ml-25m", "proc"))
    a = _defaults()
    a.epochs = args0.epochs; a.alpha = args0.alpha; a.topk = args0.topk; a.temp = args0.temp
    a.teacher_npy = args0.teacher_npy; a.mmap_teacher = args0.mmap_teacher; a.max_minutes = args0.max_minutes
    meta = M.load_meta(PROC); n_items = meta["n_items"]
    train = M.load_train(n_items, PROC)
    va_tr, va_te = M.load_val(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    hm = _head_mask(train, n_items)
    ckpt = os.path.abspath(os.path.join(_HERE, "..", "..", ".cache", "baselines", f"{args0.tag}.pt"))
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    def evaluator(predict):
        m = M.evaluate(predict, va_tr, va_te, batch_size=a.batch, head_mask=hm)
        return m["ndcg@10"], m
    t0 = time.time()
    predict = fit(train, n_items, evaluator=evaluator, args=a, ckpt=ckpt)
    res = M.evaluate(predict, te_tr, te_te, batch_size=a.batch, head_mask=hm)
    res["seconds"] = time.time() - t0; res["best_val"] = predict.best_val
    res["alpha"] = a.alpha; res["topk"] = a.topk; res["temp"] = a.temp
    print(f"[distill] test full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f}")
    if args0.out:
        json.dump(res, open(args0.out, "w"), indent=2)


if __name__ == "__main__":
    main()
