"""
smoke15.py -- INSTRUMENT 2.0 Phase-1.5 elicitation-compatibility smoke test.
EVAL-ONLY. Loads the replicated RecVAE best checkpoint and runs three checks on
ML-20M test users. No training, no commits.

Check 1: extreme-cold fold sweep (k-curve) + pure-popularity comparison at k=0.
Check 2: latent-direction sanity (ranking smoothness along random q; crude cos-answer elicitation).
Check 3: input-sparsity calibration of z at k=2 vs k=80%.
"""
import os, sys, json, time
import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_liang as E
from recvae import RecVAE

RNG = np.random.RandomState(1234)
torch.manual_seed(1234)
DEVICE = torch.device("cpu")
torch.set_num_threads(3)

CKPT = os.path.join(".cache", "instrument2", "recvae_ml20m_best.pt")
OUT_JSON = os.path.join("experiments", "instrument2", "phase15_smoke_results.json")


def load_model(n_items):
    blob = torch.load(CKPT, map_location=DEVICE)
    args = blob["args"]
    model = RecVAE(args["hidden"], args["latent"], n_items).to(DEVICE)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, args


def encode_mu(model, X_csr):
    """Return posterior mean mu (== eval-time z) for a batch of csr inputs."""
    with torch.no_grad():
        x = torch.tensor(X_csr.toarray(), dtype=torch.float32, device=DEVICE)
        mu, logvar = model.encoder(x, dropout_rate=0.0)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def decode(model, z_np):
    with torch.no_grad():
        z = torch.tensor(z_np, dtype=torch.float32, device=DEVICE)
        s = model.decoder(z)
    return s.cpu().numpy().astype(np.float32)


def predict_scores(model, X_csr):
    with torch.no_grad():
        x = torch.tensor(X_csr.toarray(), dtype=torch.float32, device=DEVICE)
        s = model(x, calculate_loss=False)
    return s.cpu().numpy().astype(np.float32)


def ndcg_recall(X_pred, mask_csr, held_csr, ks=(100,)):
    """X_pred dense; mask items in mask_csr (-inf); NDCG@100 on held_csr."""
    Xp = X_pred.copy()
    Xp[mask_csr.nonzero()] = -np.inf
    n100 = E.NDCG_binary_at_k_batch(Xp, held_csr, k=100)
    return n100


def build_folded_input(test_tr, k, n_items, rng):
    """For each user, keep k randomly-sampled fold-in items (all if k>=nnz).
    k='full' keeps all. Returns (input_csr, folded_mask_csr) both n_users x n_items.
    folded_mask == the items actually placed in the input (to mask from scoring)."""
    n = test_tr.shape[0]
    rows, cols = [], []
    tr = test_tr.tocsr()
    for u in range(n):
        items = tr.indices[tr.indptr[u]:tr.indptr[u + 1]]
        if len(items) == 0:
            continue
        if k == "full":
            sel = items
        else:
            kk = min(k, len(items))
            if kk == 0:
                sel = np.array([], dtype=int)
            else:
                sel = rng.choice(items, size=kk, replace=False)
        for c in sel:
            rows.append(u); cols.append(c)
    data = np.ones(len(rows), dtype=np.float32)
    inp = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_items), dtype=np.float32)
    return inp, inp  # input == folded mask


# ----------------------------------------------------------------------------
def check1_fold_sweep(model, test_tr, test_te, n_items, train_pop, batch=500):
    print("\n=== CHECK 1: extreme-cold fold sweep ===", flush=True)
    ks = [0, 1, 2, 4, 8, 16, 32, "full"]
    n = test_tr.shape[0]
    curve = {}
    for k in ks:
        rng = np.random.RandomState(777)  # fixed sampling per k for comparability
        inp, mask = build_folded_input(test_tr, k, n_items, rng)
        n100 = []
        for st in range(0, n, batch):
            en = min(st + batch, n)
            Xp = predict_scores(model, inp[st:en])
            m = mask[st:en]
            Xp[m.nonzero()] = -np.inf
            n100.append(E.NDCG_binary_at_k_batch(Xp, test_te[st:en], k=100))
        n100 = np.concatenate(n100)
        curve[str(k)] = (float(np.mean(n100)), float(np.std(n100) / np.sqrt(len(n100))))
        print(f"  k={str(k):>4}  NDCG@100 = {curve[str(k)][0]:.4f} +/- {curve[str(k)][1]:.4f}", flush=True)

    # pure popularity at k=0 (no fold-in, nothing masked)
    pop_scores_row = train_pop.astype(np.float32)
    n100 = []
    for st in range(0, n, batch):
        en = min(st + batch, n)
        Xp = np.tile(pop_scores_row, (en - st, 1))
        n100.append(E.NDCG_binary_at_k_batch(Xp, test_te[st:en], k=100))
    pop_ndcg = float(np.mean(np.concatenate(n100)))
    print(f"  pure-popularity NDCG@100 = {pop_ndcg:.4f}", flush=True)

    # NOTE: true empty encoder input -> encoder L2-normalizes x/||x|| = 0/0 = NaN.
    # The principled "no information" latent is the composite-prior mean z=0.
    z_empty_enc, _ = encode_mu(model, sparse.csr_matrix((1, n_items), dtype=np.float32))
    empty_enc_nan = bool(np.isnan(z_empty_enc).any())
    z0 = np.zeros((1, n_items and model.decoder.in_features), dtype=np.float32)  # prior mean
    prior_scores = decode(model, z0)[0]  # decoder(0) = decoder bias
    from scipy.stats import spearmanr
    top = np.argsort(-train_pop)[:2000]
    rho = spearmanr(prior_scores[top], train_pop[top]).correlation
    # prior-mean floor NDCG (z=0)
    n100 = []
    z0b = np.zeros((batch, model.decoder.in_features), dtype=np.float32)
    prior_row = prior_scores
    for st in range(0, n, batch):
        en = min(st + batch, n)
        Xp = np.tile(prior_row, (en - st, 1))
        n100.append(E.NDCG_binary_at_k_batch(Xp, test_te[st:en], k=100))
    prior_ndcg = float(np.mean(np.concatenate(n100)))
    print(f"  empty-encoder-input z is NaN (0/0 L2-norm): {empty_enc_nan}", flush=True)
    print(f"  prior-mean (z=0) decode NDCG@100 = {prior_ndcg:.4f}", flush=True)
    print(f"  spearman(prior-mean decode scores, popularity) over top-2000 = {rho:.3f}", flush=True)
    curve["0(prior-mean z=0)"] = (prior_ndcg, 0.0)
    return curve, pop_ndcg, float(rho), empty_enc_nan, prior_ndcg


# ----------------------------------------------------------------------------
def check2_latent(model, test_tr, test_te, n_items, batch=500):
    print("\n=== CHECK 2: latent-direction sanity ===", flush=True)
    from scipy.stats import spearmanr
    d = model.decoder.in_features  # latent dim
    # (a) ranking smoothness along random unit directions from the prior mean z0=0
    #     (encoder(empty)=NaN due to L2-norm; prior mean is the principled base)
    z0 = np.zeros(d, dtype=np.float32)
    alphas = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    smooth = []
    for _ in range(8):
        q = RNG.randn(d); q /= np.linalg.norm(q)
        Z = np.stack([z0 + a * q for a in alphas])
        S = decode(model, Z)  # (len(alphas) x n_items)
        # consecutive-alpha rank correlation (top-500 union)
        base = S[0]
        rhos = [spearmanr(S[0], S[i]).correlation for i in range(1, len(alphas))]
        smooth.append(rhos)
    smooth = np.array(smooth)  # (8 dirs x len(alphas)-1)
    mean_rho = smooth.mean(axis=0)
    print("  ranking rank-corr vs alpha (mean over 8 dirs), alpha=", list(alphas[1:]), flush=True)
    print("   ", [f"{r:.3f}" for r in mean_rho], flush=True)
    # adjacent-step smoothness: monotone decline (no jumps) => smooth
    adj = [spearmanr(decode(model, np.stack([z0 + alphas[i]*q, z0 + alphas[i+1]*q]))[0],
                     decode(model, np.stack([z0 + alphas[i]*q, z0 + alphas[i+1]*q]))[1]).correlation
           for i in range(len(alphas)-1)]

    # (b) crude cos-answer elicitation on 20 users. Probe q -> answer a=cos(z*,q);
    #     reconstruct z_hat = sum_i a_i q_i (an MC estimate of z*/d direction).
    #     Sweep n_probes to separate "no signal" from "budget underpowered in 200-d".
    n_users = 20
    rng = np.random.RandomState(42)
    uids = rng.choice(test_tr.shape[0], size=n_users, replace=False)
    full = (test_tr[uids] + test_te[uids])
    full.data[:] = 1.0
    zstar, _ = encode_mu(model, full)  # (20 x d) = full-profile latent (privileged target)
    zstar_u = zstar / (np.linalg.norm(zstar, axis=1, keepdims=True) + 1e-9)
    z_scale = float(np.linalg.norm(zstar, axis=1).mean())
    te_u = test_te[uids]
    S_floor = np.tile(decode(model, z0[None, :])[0], (n_users, 1))
    n_fl = float(np.mean(E.NDCG_binary_at_k_batch(S_floor, te_u, k=100)))
    # ceiling: decode(z*) directly (full-profile latent)
    n_ceil = float(np.mean(E.NDCG_binary_at_k_batch(decode(model, zstar), te_u, k=100)))

    sweep = {}
    for n_probes in [8, 32, 128, 512]:
        Q = rng.randn(n_probes, d); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        A = zstar_u @ Q.T                 # (20 x n_probes) geometric cos answers
        z_hat = A @ Q                     # (20 x d) reconstructed direction
        cos_recover = float(np.mean(np.sum(
            (z_hat / (np.linalg.norm(z_hat, axis=1, keepdims=True) + 1e-9)) * zstar_u, axis=1)))
        z_el = z_hat / (np.linalg.norm(z_hat, axis=1, keepdims=True) + 1e-9) * z_scale
        n_el = float(np.mean(E.NDCG_binary_at_k_batch(decode(model, z_el), te_u, k=100)))
        sweep[str(n_probes)] = {"ndcg": n_el, "cos_recover_zstar": cos_recover}
        print(f"  cos-elicit n_probes={n_probes:>4}: NDCG@100={n_el:.4f}  "
              f"cos(z_hat,z*)={cos_recover:.3f}", flush=True)
    print(f"  floor (z=0) NDCG@100={n_fl:.4f}  ceiling decode(z*) NDCG@100={n_ceil:.4f} "
          f"(20 users)", flush=True)
    return {
        "alphas": alphas[1:].tolist(),
        "rank_corr_vs_alpha": mean_rho.tolist(),
        "adjacent_step_rho": [float(x) for x in adj],
        "probe_sweep": sweep,
        "floor_ndcg": n_fl,
        "ceiling_ndcg": n_ceil,
    }


# ----------------------------------------------------------------------------
def check3_sparsity(model, test_tr, n_items, batch=500):
    print("\n=== CHECK 3: input-sparsity calibration of z ===", flush=True)
    n = test_tr.shape[0]
    out = {}
    for k in [2, "full"]:
        rng = np.random.RandomState(999)
        inp, _ = build_folded_input(test_tr, k, n_items, rng)
        mus = []
        for st in range(0, n, batch):
            en = min(st + batch, n)
            mu, _ = encode_mu(model, inp[st:en])
            mus.append(mu)
        mu = np.concatenate(mus, axis=0)  # (n x d)
        norms = np.linalg.norm(mu, axis=1)
        perdim_var = mu.var(axis=0)  # variance across users, per dim
        out[str(k)] = {
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "mean_perdim_var": float(perdim_var.mean()),
            "median_perdim_var": float(np.median(perdim_var)),
        }
        print(f"  k={str(k):>4}: mean||z||={norms.mean():.3f} (sd {norms.std():.3f}) "
              f"mean per-dim var={perdim_var.mean():.4f}", flush=True)
    # prior mean is 0; report distance of k=2 mean-z from origin
    return out


def main():
    t0 = time.time()
    meta = E.load_meta()
    n_items = meta["n_items"]
    model, margs = load_model(n_items)
    print(f"[smoke15] loaded RecVAE (hidden={margs['hidden']} latent={margs['latent']}) "
          f"n_items={n_items}", flush=True)

    test_tr, test_te = E.load_test(n_items)
    print(f"[smoke15] test users={test_tr.shape[0]}", flush=True)

    # training popularity (over train users)
    train = E.load_train(n_items)
    train_pop = np.asarray(train.sum(axis=0)).ravel()  # item counts

    c1, pop_ndcg, rho_pop, empty_nan, prior_ndcg = check1_fold_sweep(
        model, test_tr, test_te, n_items, train_pop)
    c2 = check2_latent(model, test_tr, test_te, n_items)
    c3 = check3_sparsity(model, test_tr, n_items)

    res = {"check1_kcurve": c1, "pop_ndcg": pop_ndcg,
           "prior_mean_vs_pop_spearman": rho_pop,
           "empty_encoder_nan": empty_nan, "prior_mean_ndcg": prior_ndcg,
           "check2_latent": c2, "check3_sparsity": c3,
           "elapsed_min": round((time.time() - t0) / 60, 2)}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[smoke15] wrote {OUT_JSON}  ({res['elapsed_min']} min)", flush=True)


if __name__ == "__main__":
    main()
