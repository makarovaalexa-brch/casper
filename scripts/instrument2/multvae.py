"""
multvae.py -- Mult-VAE^PR (Liang et al., WWW 2018) faithful reimplementation.

Architecture: I -> 600 -> 200 (latent) -> 600 -> I, tanh activations.
Multinomial log-likelihood. KL annealing beta: 0 -> 0.2 (anneal_cap) linearly over
total_anneal_steps. Dropout 0.5 at input. L2-normalize input. Adam lr=1e-3.
Early stopping on validation NDCG@100.

Verbatim hyperparameters (vae_cf VAE_ML20M_WWW2018.ipynb):
  p_dims = [200, 600, n_items]; anneal_cap=0.2; total_anneal_steps=200000;
  batch_size=500; lr=1e-3; weight_decay via lam=0.0 (PR = partially regularized,
  no explicit weight decay in the WWW2018 headline); ~200 epochs.
"""
import os, sys, json, time, argparse
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


class MultVAE(nn.Module):
    def __init__(self, p_dims, dropout=0.5):
        super().__init__()
        self.p_dims = p_dims                 # [latent, hidden, n_items]
        self.q_dims = p_dims[::-1]           # [n_items, hidden, latent]
        # encoder: last layer outputs 2*latent (mu, logvar)
        q_dims_ = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(q_dims_[:-1], q_dims_[1:])])
        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in list(self.q_layers) + list(self.p_layers):
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, std=0.001)

    def encode(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.p_dims[0]]
                logvar = h[:, self.p_dims[0]:]
        return mu, logvar

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_fn(recon, x, mu, logvar, anneal):
    # multinomial log-likelihood
    log_softmax = F.log_softmax(recon, dim=1)
    neg_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + anneal * kld


def make_predict_fn(model, device):
    def predict(X_csr):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_csr.toarray(), dtype=torch.float32, device=device)
            recon, _, _ = model(x)
            return recon.cpu().numpy().astype(np.float32)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anneal_cap", type=float, default=0.2)
    ap.add_argument("--total_anneal_steps", type=int, default=200000)
    ap.add_argument("--hidden", type=int, default=600)
    ap.add_argument("--latent", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--max_minutes", type=float, default=1e9,
                    help="soft wall-clock budget; stop after epoch if exceeded (RESUME via --resume)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="multvae_ml20m")
    args = ap.parse_args()

    device = torch.device("cpu")
    meta = E.load_meta()
    n_items = meta["n_items"]
    print(f"[multvae] n_items={n_items} device={device}", flush=True)

    train = E.load_train(n_items)
    vad_tr, vad_te = E.load_val(n_items)
    N = train.shape[0]

    model = MultVAE([args.latent, args.hidden, n_items], dropout=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(CKPT_DIR, f"{args.tag}.pt")
    log_path = os.path.join(CKPT_DIR, f"{args.tag}_log.json")
    state = {"epoch": 0, "update": 0, "best_ndcg": -1.0, "history": []}
    if args.resume and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        state = blob["state"]
        print(f"[multvae] RESUMED at epoch {state['epoch']} best_ndcg={state['best_ndcg']:.4f} "
              f"update={state['update']}", flush=True)

    predict = make_predict_fn(model, device)
    idxlist = np.arange(N)
    t_start = time.time()
    start_epoch = state["epoch"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        np.random.shuffle(idxlist)
        ep_loss = 0.0; nb = 0
        for st in range(0, N, args.batch):
            en = min(st + args.batch, N)
            X = train[idxlist[st:en]]
            x = torch.tensor(X.toarray(), dtype=torch.float32, device=device)
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1.0 * state["update"] / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar, anneal)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
            state["update"] += 1
        # validation
        metrics = E.evaluate(predict, vad_tr, vad_te, batch_size=args.batch)
        state["epoch"] = epoch + 1
        elapsed = (time.time() - t_start) / 60.0
        rec = {"epoch": epoch + 1, "loss": ep_loss / nb, "anneal": anneal,
               "val_ndcg@100": metrics["ndcg@100"], "val_recall@20": metrics["recall@20"],
               "val_recall@50": metrics["recall@50"], "min": round(elapsed, 1)}
        state["history"].append(rec)
        print(f"[multvae] ep {epoch+1:3d} loss {ep_loss/nb:.4f} anneal {anneal:.3f} "
              f"| val NDCG@100 {metrics['ndcg@100']:.4f} R@20 {metrics['recall@20']:.4f} "
              f"R@50 {metrics['recall@50']:.4f} | {elapsed:.1f}m", flush=True)

        improved = metrics["ndcg@100"] > state["best_ndcg"]
        if improved:
            state["best_ndcg"] = metrics["ndcg@100"]
            state["best_epoch"] = epoch + 1
            torch.save({"model": model.state_dict(), "state": state,
                        "args": vars(args)}, os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
        # always save resumable
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "state": state, "args": vars(args)}, ckpt_path)
        with open(log_path, "w") as f:
            json.dump(state["history"], f, indent=2)

        # early stopping
        since_best = (epoch + 1) - state.get("best_epoch", epoch + 1)
        if since_best >= args.patience:
            print(f"[multvae] early stop: no val improvement for {args.patience} epochs "
                  f"(best {state['best_ndcg']:.4f} @ep{state['best_epoch']})", flush=True)
            break
        if elapsed >= args.max_minutes:
            print(f"[multvae] wall budget {args.max_minutes}m hit at epoch {epoch+1}; "
                  f"re-run with --resume to continue", flush=True)
            return
    print(f"[multvae] BEST val NDCG@100 {state['best_ndcg']:.4f} @ep{state.get('best_epoch')}", flush=True)


if __name__ == "__main__":
    main()
