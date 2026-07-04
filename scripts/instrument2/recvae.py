"""
recvae.py -- RecVAE (Shenbin et al., WSDM 2020) faithful port of the official repo
(github.com/ilya-shenbin/RecVAE: model.py + run.py).

Load-bearing recipe (verified against primary source):
  - Encoder: dense-connected feedforward, swish activations, LayerNorm(eps=0.1),
    5 hidden blocks of width 600, latent d=200. L2-normalize input, then dropout.
  - Decoder: a single linear layer (latent -> n_items), softmax likelihood (multinomial).
  - Composite prior: mixture of  (a) N(0,I)  (b) previous-epoch posterior q_old(z|x)
    (c) a wide N(0, exp(10) I) "uniform" component; mixture weights [3/20, 3/4, 1/10].
  - Per-user beta' = gamma * |X_u|  with gamma = 0.005 (ML-20M).
  - Alternating updates: 3 encoder steps : 1 decoder step per epoch; update_prior()
    (copy encoder -> encoder_old) between them.
  - Denoising: dropout (Bernoulli mu=0.5) applied ONLY during encoder steps;
    decoder steps use dropout_rate=0.
  - Adam lr=5e-4, batch_size=500, n_epochs~50, early stop on val NDCG@100.
"""
import os, sys, json, time, argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_liang as E

torch.manual_seed(98765)
np.random.seed(98765)

CKPT_DIR = os.path.join(".cache", "instrument2")
os.makedirs(CKPT_DIR, exist_ok=True)


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
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
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        density_per_gaussian = torch.stack(gaussians, dim=-1)
        return torch.logsumexp(density_per_gaussian, dim=-1)


class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super().__init__()
        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            else:
                kl_weight = beta
            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)
            return (mll, kld), negative_elbo
        return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


def run_updates(model, opts, train, idxlist, batch, device, n_epochs, dropout_rate, gamma, beta):
    model.train()
    N = train.shape[0]
    for _ in range(n_epochs):
        np.random.shuffle(idxlist)
        for st in range(0, N, batch):
            en = min(st + batch, N)
            X = train[idxlist[st:en]]
            x = torch.tensor(X.toarray(), dtype=torch.float32, device=device)
            for opt in opts:
                opt.zero_grad()
            _, neg_elbo = model(x, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
            neg_elbo.backward()
            for opt in opts:
                opt.step()


def make_predict_fn(model, device):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_csr.toarray(), dtype=torch.float32, device=device)
            x_pred = model(x, calculate_loss=False)
            return x_pred.cpu().numpy().astype(np.float32)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--hidden", type=int, default=600)
    ap.add_argument("--latent", type=int, default=200)
    ap.add_argument("--n_enc", type=int, default=3)
    ap.add_argument("--n_dec", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="recvae_ml20m")
    args = ap.parse_args()

    device = torch.device("cpu")
    meta = E.load_meta()
    n_items = meta["n_items"]
    print(f"[recvae] n_items={n_items} device={device} gamma={args.gamma} "
          f"enc:dec={args.n_enc}:{args.n_dec}", flush=True)

    train = E.load_train(n_items)
    vad_tr, vad_te = E.load_val(n_items)
    N = train.shape[0]

    model = RecVAE(args.hidden, args.latent, n_items).to(device)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    ckpt_path = os.path.join(CKPT_DIR, f"{args.tag}.pt")
    log_path = os.path.join(CKPT_DIR, f"{args.tag}_log.json")
    state = {"epoch": 0, "best_ndcg": -1.0, "history": []}
    if args.resume and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(blob["model"])
        opt_enc.load_state_dict(blob["opt_enc"])
        opt_dec.load_state_dict(blob["opt_dec"])
        state = blob["state"]
        print(f"[recvae] RESUMED at epoch {state['epoch']} best_ndcg={state['best_ndcg']:.4f}", flush=True)

    predict = make_predict_fn(model, device)
    idxlist = np.arange(N)
    t_start = time.time()

    for epoch in range(state["epoch"], args.epochs):
        # 3 encoder steps (denoising dropout on), update prior, 1 decoder step (no dropout)
        run_updates(model, [opt_enc], train, idxlist, args.batch, device,
                    n_epochs=args.n_enc, dropout_rate=args.dropout, gamma=args.gamma, beta=None)
        model.update_prior()
        run_updates(model, [opt_dec], train, idxlist, args.batch, device,
                    n_epochs=args.n_dec, dropout_rate=0.0, gamma=args.gamma, beta=None)

        metrics = E.evaluate(predict, vad_tr, vad_te, batch_size=args.batch)
        state["epoch"] = epoch + 1
        elapsed = (time.time() - t_start) / 60.0
        rec = {"epoch": epoch + 1, "val_ndcg@100": metrics["ndcg@100"],
               "val_recall@20": metrics["recall@20"], "val_recall@50": metrics["recall@50"],
               "min": round(elapsed, 1)}
        state["history"].append(rec)
        print(f"[recvae] ep {epoch+1:3d} | val NDCG@100 {metrics['ndcg@100']:.4f} "
              f"R@20 {metrics['recall@20']:.4f} R@50 {metrics['recall@50']:.4f} | {elapsed:.1f}m", flush=True)

        if metrics["ndcg@100"] > state["best_ndcg"]:
            state["best_ndcg"] = metrics["ndcg@100"]
            state["best_epoch"] = epoch + 1
            torch.save({"model": model.state_dict(), "state": state, "args": vars(args)},
                       os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
        torch.save({"model": model.state_dict(), "opt_enc": opt_enc.state_dict(),
                    "opt_dec": opt_dec.state_dict(), "state": state, "args": vars(args)}, ckpt_path)
        with open(log_path, "w") as f:
            json.dump(state["history"], f, indent=2)

        since_best = (epoch + 1) - state.get("best_epoch", epoch + 1)
        if since_best >= args.patience:
            print(f"[recvae] early stop: no val improvement for {args.patience} epochs "
                  f"(best {state['best_ndcg']:.4f} @ep{state['best_epoch']})", flush=True)
            break
        if elapsed >= args.max_minutes:
            print(f"[recvae] wall budget {args.max_minutes}m hit at epoch {epoch+1}; "
                  f"re-run with --resume to continue", flush=True)
            return
    print(f"[recvae] BEST val NDCG@100 {state['best_ndcg']:.4f} @ep{state.get('best_epoch')}", flush=True)


if __name__ == "__main__":
    main()
