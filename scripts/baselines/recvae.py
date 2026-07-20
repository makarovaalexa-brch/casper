"""recvae.py -- RecVAE (Shenbin et al., WSDM 2020). Faithful port of the official repo
(github.com/ilya-shenbin/RecVAE: model.py + run.py).

Load-bearing recipe (verified against primary source):
  - Encoder: dense-connected feedforward, swish activations, LayerNorm(eps=0.1), 5 hidden blocks of
    width 600, latent d=200. L2-normalize input, then dropout.
  - Decoder: single linear layer (latent -> n_items), multinomial (softmax) likelihood.
  - Composite prior: mixture of (a) N(0,I) (b) previous-epoch posterior q_old(z|x) (c) wide
    N(0, exp(10) I); mixture weights [3/20, 3/4, 1/10].
  - Per-user beta' = gamma * |X_u|, gamma = 0.005 (ML-20M).
  - Alternating: 3 encoder steps : 1 decoder step per epoch; update_prior() (encoder -> encoder_old)
    between them. Denoising dropout (0.5) ONLY on encoder steps; decoder steps use dropout 0.
  - Adam lr=5e-4, batch=500, ~50 epochs, early stop on val NDCG@100.
Published ML-20M: NDCG@100=0.442 (R@20 0.414, R@50 0.553) -- top of the Liang split.

Interface: fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print) -> predict_fn
"""
import os
import sys
import json
import time
import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

torch.manual_seed(98765)
np.random.seed(98765)


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim); self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim); self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim); self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim); self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim); self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
        x = F.dropout(x, p=dropout_rate, training=self.training)
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=(3/20, 3/4, 1/10)):
        super().__init__()
        self.mixture_weights = mixture_weights
        self.mu_prior = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self.logvar_prior = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior = nn.Parameter(torch.full((1, latent_dim), 10.0), requires_grad=False)
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        stnd = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post = log_norm_pdf(z, post_mu, post_logvar)
        unif = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        gaussians = [g.add(np.log(w)) for g, w in zip([stnd, post, unif], self.mixture_weights)]
        return torch.logsumexp(torch.stack(gaussians, dim=-1), dim=-1)


class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super().__init__()
        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        if calculate_loss:
            if gamma:
                kl_weight = gamma * user_ratings.sum(dim=-1)
            else:
                kl_weight = beta
            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            return (mll, kld), -(mll - kld)
        return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


def _run_updates(model, opts, train, idxlist, batch, n_sub, dropout_rate, gamma):
    model.train(); N = train.shape[0]
    for _ in range(n_sub):
        np.random.shuffle(idxlist)
        for st in range(0, N, batch):
            X = train[idxlist[st:min(st + batch, N)]]
            x = torch.tensor(X.toarray(), dtype=torch.float32)
            for opt in opts:
                opt.zero_grad()
            _, neg_elbo = model(x, gamma=gamma, dropout_rate=dropout_rate)
            neg_elbo.backward()
            for opt in opts:
                opt.step()


def make_predict_fn(model):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_csr.toarray(), dtype=torch.float32)
            return model(x, calculate_loss=False).cpu().numpy().astype(np.float32)
    return predict


def _defaults():
    return argparse.Namespace(epochs=50, batch=500, lr=5e-4, gamma=0.005, hidden=600, latent=200,
                              n_enc=3, n_dec=1, dropout=0.5, patience=10, max_minutes=1e9, resume=True)


def fit(train, n_items, evaluator=None, args=None, ckpt=None, log=print):
    a = args or _defaults()
    model = RecVAE(a.hidden, a.latent, n_items)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=a.lr)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=a.lr)
    N = train.shape[0]
    state = {"epoch": 0, "best": -1.0, "best_state": None, "best_epoch": 0, "history": []}
    if ckpt and getattr(a, "resume", True) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"]); opt_enc.load_state_dict(blob["opt_enc"])
        opt_dec.load_state_dict(blob["opt_dec"]); state = blob["state"]
        log(f"[recvae] RESUMED ep{state['epoch']} best={state['best']:.4f}")
    predict = make_predict_fn(model)
    idxlist = np.arange(N); t0 = time.time()
    for epoch in range(state["epoch"], a.epochs):
        _run_updates(model, [opt_enc], train, idxlist, a.batch, a.n_enc, a.dropout, a.gamma)
        model.update_prior()
        _run_updates(model, [opt_dec], train, idxlist, a.batch, a.n_dec, 0.0, a.gamma)
        state["epoch"] = epoch + 1
        if evaluator is not None:
            sc, metx = evaluator(predict)
        else:
            sc, metx = -1.0, {}
        el = (time.time() - t0) / 60.0
        state["history"].append({"epoch": epoch + 1, "val": sc, "min": round(el, 1)})
        log(f"[recvae] ep{epoch+1:3d} val {sc:.4f} {el:.1f}m")
        if sc > state["best"]:
            state["best"] = sc; state["best_epoch"] = epoch + 1
            state["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        if ckpt:
            torch.save({"model": model.state_dict(), "opt_enc": opt_enc.state_dict(),
                        "opt_dec": opt_dec.state_dict(), "state": state, "args": vars(a)}, ckpt)
        if evaluator is not None and (epoch + 1) - state["best_epoch"] >= a.patience:
            log(f"[recvae] early stop @ep{epoch+1} best {state['best']:.4f} @ep{state['best_epoch']}")
            break
        if el >= a.max_minutes:
            log(f"[recvae] wall budget {a.max_minutes}m hit @ep{epoch+1}; re-run to resume")
            break
    if state.get("best_state") is not None:
        model.load_state_dict(state["best_state"])
    predict.best_val = state["best"]; predict.best_epoch = state["best_epoch"]
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--tag", default="recvae_ml20m")
    ap.add_argument("--out", default=None)
    args0 = ap.parse_args()
    a = _defaults(); a.epochs = args0.epochs; a.max_minutes = args0.max_minutes
    meta = M.load_meta(); n_items = meta["n_items"]
    train = M.load_train(n_items)
    va_tr, va_te = M.load_val(n_items)
    te_tr, te_te = M.load_test(n_items)
    ckpt = os.path.join(".cache", "baselines", f"{args0.tag}.pt"); os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    def evaluator(predict):
        m = M.evaluate(predict, va_tr, va_te, batch_size=a.batch)
        return m["ndcg@100"], m
    t0 = time.time()
    predict = fit(train, n_items, evaluator=evaluator, args=a, ckpt=ckpt)
    res = M.evaluate(predict, te_tr, te_te, batch_size=a.batch)
    res["seconds"] = time.time() - t0; res["best_val"] = predict.best_val
    print(f"[recvae] test {res}")
    if args0.out:
        json.dump(res, open(args0.out, "w"), indent=2)


if __name__ == "__main__":
    main()
