"""multvae.py -- Mult-VAE^PR and Mult-DAE (Liang et al., "Variational Autoencoders for Collaborative
Filtering", WWW 2018). Faithful port of the official dawenl/vae_cf reference model.

Architecture (both modes): I -> 600 -> 200 (latent) -> 600 -> I, tanh activations, multinomial
log-likelihood, L2-normalized + 0.5-dropout input. Adam.
  - mode="vae" (Mult-VAE^PR): variational, reparameterized latent, KL term with beta annealed
    LINEARLY from 0 to anneal_cap=0.2 over total_anneal_steps=200000 updates. lr=1e-3.
  - mode="dae" (Mult-DAE): deterministic (no KL, no sampling), same 3-layer shape, weight_decay=0.01
    (the paper's regularized DAE). lr=1e-3.

Verbatim hyperparameters (vae_cf VAE_ML20M_WWW2018.ipynb):
  p_dims=[200,600,n_items]; anneal_cap=0.2; total_anneal_steps=200000; batch=500; lr=1e-3;
  ~200 epochs, early stop on validation NDCG@100.
Published ML-20M: Mult-VAE^PR NDCG@100=0.426 (R@20 0.395, R@50 0.537); Mult-DAE NDCG@100=0.419.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
  evaluator(predict_fn) -> (early_stop_score, metrics_dict); used per-epoch. Checkpoint-resumable.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

torch.manual_seed(98765)
np.random.seed(98765)


class MultVAE(nn.Module):
    def __init__(self, p_dims, dropout=0.5, mode="vae"):
        super().__init__()
        self.mode = mode
        self.p_dims = p_dims                 # [latent, hidden, n_items]
        self.q_dims = p_dims[::-1]           # [n_items, hidden, latent]
        last = self.q_dims[-1] * (2 if mode == "vae" else 1)   # VAE encoder outputs (mu, logvar)
        q_dims_ = self.q_dims[:-1] + [last]
        self.q_layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(q_dims_[:-1], q_dims_[1:])])
        self.p_layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in list(self.q_layers) + list(self.p_layers):
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, std=0.001)

    def encode(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        mu = logvar = None
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                if self.mode == "vae":
                    mu = h[:, :self.p_dims[0]]; logvar = h[:, self.p_dims[0]:]
                else:
                    mu = h; logvar = None
        return mu, logvar

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.training and self.mode == "vae":
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_fn(recon, x, mu, logvar, anneal, mode):
    log_softmax = F.log_softmax(recon, dim=1)
    neg_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))
    if mode == "dae":
        return neg_ll
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + anneal * kld


def make_predict_fn(model):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_csr.toarray(), dtype=torch.float32)
            recon, _, _ = model(x)
            return recon.cpu().numpy().astype(np.float32)
    return predict


def _defaults(mode):
    return argparse.Namespace(
        epochs=200, batch=500, lr=1e-3, anneal_cap=0.2, total_anneal_steps=200000,
        hidden=600, latent=200, dropout=0.5, patience=15, max_minutes=1e9,
        weight_decay=(0.01 if mode == "dae" else 0.0), mode=mode, resume=True)


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or _defaults("vae")
    mode = getattr(a, "mode", "vae")
    model = MultVAE([a.latent, a.hidden, n_items], dropout=a.dropout, mode=mode)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=getattr(a, "weight_decay", 0.0))
    N = train.shape[0]
    state = {"epoch": 0, "update": 0, "best": -1.0, "best_state": None, "best_epoch": 0, "history": []}
    if ckpt and getattr(a, "resume", True) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        log(f"[{mode}] RESUMED ep{state['epoch']} best={state['best']:.4f}")
    predict = make_predict_fn(model)
    idxlist = np.arange(N)
    t0 = time.time()
    for epoch in range(state["epoch"], a.epochs):
        model.train(); np.random.shuffle(idxlist); ep_loss = 0.0; nb = 0
        for st in range(0, N, a.batch):
            X = train[idxlist[st:min(st + a.batch, N)]]
            x = torch.tensor(X.toarray(), dtype=torch.float32)
            anneal = (min(a.anneal_cap, 1.0 * state["update"] / a.total_anneal_steps)
                      if a.total_anneal_steps > 0 else a.anneal_cap)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar, anneal, mode)
            loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1; state["update"] += 1
        state["epoch"] = epoch + 1
        if evaluator is not None:
            sc, metx = evaluator(predict)
        else:
            sc, metx = -1.0, {}
        el = (time.time() - t0) / 60.0
        state["history"].append({"epoch": epoch + 1, "loss": ep_loss / max(nb, 1),
                                 "val": sc, "min": round(el, 1)})
        log(f"[{mode}] ep{epoch+1:3d} loss {ep_loss/max(nb,1):.4f} anneal {anneal:.3f} val {sc:.4f} {el:.1f}m")
        if sc > state["best"]:
            state["best"] = sc; state["best_epoch"] = epoch + 1
            state["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                        "args": vars(a)}, ckpt)
        if evaluator is not None and (epoch + 1) - state["best_epoch"] >= a.patience:
            log(f"[{mode}] early stop @ep{epoch+1} best {state['best']:.4f} @ep{state['best_epoch']}")
            break
        if el >= a.max_minutes:
            log(f"[{mode}] wall budget {a.max_minutes}m hit @ep{epoch+1}; re-run to resume")
            break
    if state.get("best_state") is not None:
        model.load_state_dict(state["best_state"])
    predict.best_val = state["best"]; predict.best_epoch = state["best_epoch"]
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vae", "dae"], default="vae")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args0 = ap.parse_args()
    a = _defaults(args0.mode); a.epochs = args0.epochs; a.max_minutes = args0.max_minutes
    meta = M.load_meta(); n_items = meta["n_items"]
    train = M.load_train(n_items)
    va_tr, va_te = M.load_val(n_items)
    te_tr, te_te = M.load_test(n_items)
    tag = args0.tag or f"mult{args0.mode}_ml20m"
    ckpt = os.path.join(".cache", "baselines", f"{tag}.pt"); os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    def evaluator(predict):
        m = M.evaluate(predict, va_tr, va_te, batch_size=a.batch)
        return m["ndcg@100"], m
    t0 = time.time()
    predict = fit(train, n_items, evaluator=evaluator, args=a, ckpt=ckpt)
    res = M.evaluate(predict, te_tr, te_te, batch_size=a.batch)
    res["mode"] = args0.mode; res["seconds"] = time.time() - t0; res["best_val"] = predict.best_val
    print(f"[mult{args0.mode}] test {res}")
    if args0.out:
        json.dump(res, open(args0.out, "w"), indent=2)


if __name__ == "__main__":
    main()
