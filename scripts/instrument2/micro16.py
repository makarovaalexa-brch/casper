"""
micro16.py -- INSTRUMENT 2.0 Phase-1.6 elicitation micro-battery (RecVAE / ML-20M).
EVAL-ONLY (only tiny per-user z-optimization in Exp B). Foreground, 3 threads, no commits.

A. Concept pseudo-item folding  (genome-tag concept bags)
   (i)  FIDELITY   -- fold ONLY concept bag vs z=0 floor, per M in {10,50,200}
   (ii) ADDITIVITY -- k=2 items + 2 concept answers vs k=2 items alone
   (iii)DISLIKE     -- naive negative concept bag (input clamp@0) + z-space contrast
B. Inference derivation sweep -- amortized vs z-optimization vs hybrid, k in {1,2,4,8}
C. Posterior-variance sanity -- mean sigma vs k; Spearman(sigma_u, NDCG_u) at k=8

Conventions (smoke15): seed empty state with z=0, never hand the encoder an empty vector.
"""
import os, sys, json, time
import numpy as np
import torch
from scipy import sparse
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_liang as E
from recvae import RecVAE

DEVICE = torch.device("cpu")
torch.set_num_threads(3)
torch.manual_seed(1234)

CKPT = os.path.join(".cache", "instrument2", "recvae_ml20m_best.pt")
CONCEPTS = os.path.join(".cache", "instrument2", "concepts.npz")
OUT_JSON = os.path.join("experiments", "instrument2", "phase16_micro_results.json")


# --------------------------------------------------------------------------- model io
def load_model(n_items):
    blob = torch.load(CKPT, map_location=DEVICE)
    a = blob["args"]
    m = RecVAE(a["hidden"], a["latent"], n_items).to(DEVICE)
    m.load_state_dict(blob["model"]); m.eval()
    return m, a


def encode_mu_logvar(model, X_csr):
    with torch.no_grad():
        x = torch.tensor(X_csr.toarray(), dtype=torch.float32, device=DEVICE)
        mu, logvar = model.encoder(x, dropout_rate=0.0)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def encode_dense(model, X_dense):
    with torch.no_grad():
        x = torch.tensor(X_dense, dtype=torch.float32, device=DEVICE)
        mu, logvar = model.encoder(x, dropout_rate=0.0)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def decode(model, z_np):
    with torch.no_grad():
        z = torch.tensor(z_np, dtype=torch.float32, device=DEVICE)
        s = model.decoder(z)
    return s.cpu().numpy().astype(np.float32)


def ndcg_from_scores(S, tr_csr, te_csr):
    """Mask known (tr) items, NDCG@100 vs te. S dense (n x items)."""
    Sp = S.copy()
    Sp[tr_csr.nonzero()] = -np.inf
    return E.NDCG_binary_at_k_batch(Sp, te_csr, k=100)


# --------------------------------------------------------------------------- helpers
def sample_k_items(tr_csr, k, rng):
    """Per-user keep k random fold-in items -> dense input (n x items)."""
    n, ni = tr_csr.shape
    tr = tr_csr.tocsr()
    X = np.zeros((n, ni), dtype=np.float32)
    for u in range(n):
        items = tr.indices[tr.indptr[u]:tr.indptr[u + 1]]
        if len(items) == 0:
            continue
        kk = min(k, len(items))
        sel = items if kk == len(items) else rng.choice(items, size=kk, replace=False)
        X[u, sel] = 1.0
    return X


def build_bag(item_tag, tcol, M):
    """Concept bag for tag column tcol: top-M items by relevance, L1-normalized."""
    rel = item_tag[:, tcol]
    top = np.argpartition(-rel, M)[:M]
    w = rel[top].copy()
    w = w / (w.sum() + 1e-9)
    bag = np.zeros(item_tag.shape[0], dtype=np.float32)
    bag[top] = w
    return bag


# =========================================================================== EXP A
def exp_a(model, test_tr, test_te, n_items, concepts, n_users=2000):
    print("\n=== EXP A: concept pseudo-item folding ===", flush=True)
    item_tag = concepts["item_tag"]            # (n_items x 200)
    tag_names = concepts["tag_names"]
    rng = np.random.RandomState(20260704)
    # subsample users that have >=3 fold-in items (need real affinity + held-out signal)
    nnz = np.asarray((test_tr > 0).sum(1)).ravel()
    elig = np.where(nnz >= 3)[0]
    uids = rng.choice(elig, size=min(n_users, len(elig)), replace=False)
    tr, te = test_tr[uids], test_te[uids]

    # per-user affinity over the 200 tags = sum of relevance over fold-in items
    tr_dense = tr.toarray().astype(np.float32)          # (nu x items)
    affinity = tr_dense @ item_tag                       # (nu x 200)
    top_tag = affinity.argmax(1)
    # affinity concentration (max / sum) -> 'strong affinity' subset
    conc = affinity.max(1) / (affinity.sum(1) + 1e-9)
    strong = conc >= np.median(conc)
    nu = len(uids)

    # ---- z=0 floor (identical mask) ----
    d = model.decoder.in_features
    floor_scores = np.tile(decode(model, np.zeros((1, d), np.float32))[0], (nu, 1))
    ndcg_floor = ndcg_from_scores(floor_scores, tr, te)

    # ---- (i) FIDELITY: fold ONLY the concept bag, per M ----
    fidelity = {}
    bag_cache = {}
    for M in (10, 50, 200):
        X = np.zeros((nu, n_items), dtype=np.float32)
        for i in range(nu):
            tc = int(top_tag[i])
            key = (tc, M)
            if key not in bag_cache:
                bag_cache[key] = build_bag(item_tag, tc, M)
            X[i] = bag_cache[key]          # scale irrelevant (encoder L2-normalizes)
        mu, _ = encode_dense(model, X)
        S = decode(model, mu)
        ndcg = ndcg_from_scores(S, tr, te)
        fidelity[str(M)] = {
            "ndcg_all": float(ndcg.mean()),
            "ndcg_strong": float(ndcg[strong].mean()),
            "delta_vs_floor_all": float(ndcg.mean() - ndcg_floor.mean()),
            "delta_vs_floor_strong": float(ndcg[strong].mean() - ndcg_floor[strong].mean()),
        }
        print(f"  [fid] M={M:>3}: concept-only NDCG {ndcg.mean():.4f} "
              f"(strong {ndcg[strong].mean():.4f})  floor {ndcg_floor.mean():.4f} "
              f"(strong {ndcg_floor[strong].mean():.4f})  d+{ndcg.mean()-ndcg_floor.mean():+.4f}",
              flush=True)

    # ---- (ii) ADDITIVITY: k=2 items vs k=2 items + top-2 concept answers ----
    M_add = 50
    rng2 = np.random.RandomState(7)
    Xitems = sample_k_items(tr, 2, rng2)                 # (nu x items)
    mu_i, _ = encode_dense(model, Xitems)
    ndcg_items = ndcg_from_scores(decode(model, mu_i), tr, te)
    # add top-2 affinity tags' bags (answer strength s: each bag sums to 1, items sum to ~2)
    top2 = np.argsort(-affinity, axis=1)[:, :2]
    for s in (1.0,):
        Xboth = Xitems.copy()
        for i in range(nu):
            for tc in top2[i]:
                key = (int(tc), M_add)
                if key not in bag_cache:
                    bag_cache[key] = build_bag(item_tag, int(tc), M_add)
                Xboth[i] += s * bag_cache[key]
        mu_b, _ = encode_dense(model, Xboth)
        ndcg_both = ndcg_from_scores(decode(model, mu_b), tr, te)
    additivity = {
        "M": M_add, "answer_strength": 1.0,
        "ndcg_items_only": float(ndcg_items.mean()),
        "ndcg_items_plus_concept": float(ndcg_both.mean()),
        "delta": float(ndcg_both.mean() - ndcg_items.mean()),
        "ndcg_items_only_strong": float(ndcg_items[strong].mean()),
        "ndcg_items_plus_concept_strong": float(ndcg_both[strong].mean()),
        "delta_strong": float(ndcg_both[strong].mean() - ndcg_items[strong].mean()),
    }
    print(f"  [add] k=2 items {ndcg_items.mean():.4f} -> +2 concepts "
          f"{ndcg_both.mean():.4f}  d{ndcg_both.mean()-ndcg_items.mean():+.4f} "
          f"(strong d{ndcg_both[strong].mean()-ndcg_items[strong].mean():+.4f})", flush=True)

    # ---- (iii) DISLIKE preview: negative concept bag, input clamp@0 + z-space contrast ----
    # pick a few distinctive tags; for each, over all users, base = their k=2 items;
    # measure mean predicted score of the tag's top-M member items under:
    #  (a) items only, (b) items + positive bag, (c) items - bag clamp@0 [naive input dislike]
    #  (d) z-space: decode(z_items - s*||z_items|| * unit(z_bag))  [design-space contrast]
    M_dis = 50
    want = ["horror", "violence", "romance", "animation", "comedy", "dark", "scary", "gory"]
    name_list = [str(x) for x in tag_names]
    dis_tags = [name_list.index(w) for w in want if w in name_list][:5]
    dislike = {"tags": [name_list[t] for t in dis_tags], "per_tag": {}}
    Xb = sample_k_items(tr, 2, np.random.RandomState(11))
    mu_base, _ = encode_dense(model, Xb)
    z_base = mu_base
    for tc in dis_tags:
        bag = build_bag(item_tag, tc, M_dis)
        members = np.where(bag > 0)[0]
        # (a) items only
        Sa = decode(model, mu_base)
        # (b) items + positive bag
        mu_pos, _ = encode_dense(model, np.clip(Xb + 1.0 * bag[None, :], 0, None))
        Sb = decode(model, mu_pos)
        # (c) items - bag, clamp at 0 (naive input-space dislike)
        mu_neg, _ = encode_dense(model, np.clip(Xb - 1.0 * bag[None, :], 0, None))
        Sc = decode(model, mu_neg)
        # (d) z-space subtraction contrast
        mu_bag, _ = encode_dense(model, np.tile(bag[None, :], (z_base.shape[0], 1)))
        ubag = mu_bag / (np.linalg.norm(mu_bag, axis=1, keepdims=True) + 1e-9)
        zsub = z_base - 0.5 * np.linalg.norm(z_base, axis=1, keepdims=True) * ubag
        Sd = decode(model, zsub)
        # mean predicted score over member items (higher=promoted)
        ma = float(Sa[:, members].mean()); mb = float(Sb[:, members].mean())
        mc = float(Sc[:, members].mean()); md = float(Sd[:, members].mean())
        dislike["per_tag"][name_list[tc]] = {
            "mean_score_items_only": ma, "mean_score_plus_bag": mb,
            "mean_score_minus_bag_clamp": mc, "mean_score_zspace_sub": md,
            "demotion_inputclamp": mc - ma, "promotion_posbag": mb - ma,
            "demotion_zspace": md - ma,
        }
        print(f"  [dis] {name_list[tc]:>10}: items {ma:+.3f} | +bag {mb-ma:+.3f} | "
              f"-bag(clamp) {mc-ma:+.3f} | z-sub {md-ma:+.3f}", flush=True)

    return {"n_users": nu, "ndcg_floor": float(ndcg_floor.mean()),
            "ndcg_floor_strong": float(ndcg_floor[strong].mean()),
            "fidelity": fidelity, "additivity": additivity, "dislike": dislike}


# =========================================================================== EXP B
def exp_b(model, test_tr, test_te, n_items, n_users=500):
    print("\n=== EXP B: inference derivation sweep ===", flush=True)
    rng = np.random.RandomState(505)
    nnz = np.asarray((test_tr > 0).sum(1)).ravel()
    elig = np.where(nnz >= 8)[0]
    uids = rng.choice(elig, size=min(n_users, len(elig)), replace=False)
    tr, te = test_tr[uids], test_te[uids]
    d = model.decoder.in_features
    gamma = 0.005
    W = model.decoder.weight.detach()            # (n_items x d)
    b = model.decoder.bias.detach()              # (n_items,)

    def opt_infer(Xnp, steps, z_init):
        """Maximize multinomial log-lik of observed items over z (MAP, N(0,I) prior)."""
        x = torch.tensor(Xnp, dtype=torch.float32, device=DEVICE)
        kcount = x.sum(1, keepdim=True)          # (nu x 1)
        z = torch.tensor(z_init, dtype=torch.float32, device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            logits = z @ W.T + b                 # (nu x n_items)
            logp = torch.log_softmax(logits, dim=1)
            mll = (logp * x).sum(1)              # per-user log-lik
            prior = 0.5 * (z * z).sum(1)         # -log N(0,I)
            loss = -(mll - gamma * kcount.squeeze(1) * prior).mean()
            loss.backward()
            opt.step()
        return z.detach().cpu().numpy()

    out = {}
    for k in (1, 2, 4, 8):
        rk = np.random.RandomState(1000 + k)
        Xk = sample_k_items(tr, k, rk)
        # (1) amortized
        t0 = time.time()
        mu_a, _ = encode_dense(model, Xk)
        t_amort = time.time() - t0
        nd_a = ndcg_from_scores(decode(model, mu_a), tr, te).mean()
        # (2) optimization from z=0
        t0 = time.time()
        z_opt = opt_infer(Xk, steps=100, z_init=np.zeros((len(uids), d), np.float32))
        t_opt = time.time() - t0
        nd_o = ndcg_from_scores(decode(model, z_opt), tr, te).mean()
        # (3) hybrid: encoder init + 20 refine steps
        t0 = time.time()
        z_h = opt_infer(Xk, steps=20, z_init=mu_a)
        t_hyb = time.time() - t0 + t_amort
        nd_h = ndcg_from_scores(decode(model, z_h), tr, te).mean()
        out[str(k)] = {
            "amortized": {"ndcg": float(nd_a), "sec": round(t_amort, 2)},
            "optim100": {"ndcg": float(nd_o), "sec": round(t_opt, 2)},
            "hybrid20": {"ndcg": float(nd_h), "sec": round(t_hyb, 2)},
        }
        print(f"  k={k}: amort {nd_a:.4f} ({t_amort:.2f}s) | opt100 {nd_o:.4f} "
              f"({t_opt:.1f}s) | hyb20 {nd_h:.4f} ({t_hyb:.1f}s)", flush=True)
    return {"n_users": len(uids), "sweep": out}


# =========================================================================== EXP C
def exp_c(model, test_tr, test_te, n_items, n_users=2000):
    print("\n=== EXP C: posterior-variance sanity ===", flush=True)
    rng = np.random.RandomState(303)
    nnz = np.asarray((test_tr > 0).sum(1)).ravel()
    elig = np.where(nnz >= 8)[0]
    uids = rng.choice(elig, size=min(n_users, len(elig)), replace=False)
    tr, te = test_tr[uids], test_te[uids]
    curve = {}
    sigma_at_k = {}
    for k in (1, 2, 4, 8, "full"):
        rk = np.random.RandomState(2000 + (0 if k == "full" else k))
        Xk = tr.toarray().astype(np.float32) if k == "full" else sample_k_items(tr, k, rk)
        mu, logvar = encode_dense(model, Xk)
        sigma = np.exp(0.5 * logvar)             # (nu x d)
        mean_sigma_u = sigma.mean(1)             # per-user mean sigma
        curve[str(k)] = float(mean_sigma_u.mean())
        sigma_at_k[str(k)] = mean_sigma_u
        print(f"  k={str(k):>4}: mean sigma = {mean_sigma_u.mean():.4f}", flush=True)
    # k=0: encoder input empty -> NaN by design; prior std reference = 1.0
    print("  k=   0: encoder-undefined (empty input NaN); composite-prior std ref = 1.0",
          flush=True)

    # calibration at k=8: Spearman(mean sigma_u, per-user NDCG). Expect NEGATIVE.
    rk = np.random.RandomState(2008)
    X8 = sample_k_items(tr, 8, rk)
    mu8, logvar8 = encode_dense(model, X8)
    sig8 = np.exp(0.5 * logvar8).mean(1)
    nd8 = ndcg_from_scores(decode(model, mu8), tr, te)
    rho = spearmanr(sig8, nd8).correlation
    print(f"  calibration k=8: Spearman(mean sigma_u, NDCG_u) = {rho:.3f} "
          f"(want NEGATIVE)", flush=True)
    monotone = all(curve[str(a)] >= curve[str(b)] - 1e-9
                   for a, b in zip([1, 2, 4, 8], [2, 4, 8, "full"]))
    return {"n_users": len(uids), "mean_sigma_vs_k": curve,
            "monotone_decreasing": bool(monotone),
            "k0_note": "empty encoder input -> NaN; prior std ref 1.0",
            "calib_k8_spearman_sigma_ndcg": float(rho)}


def main():
    t0 = time.time()
    meta = E.load_meta(); n_items = meta["n_items"]
    model, margs = load_model(n_items)
    print(f"[micro16] RecVAE hidden={margs['hidden']} latent={margs['latent']} "
          f"n_items={n_items}", flush=True)
    test_tr, test_te = E.load_test(n_items)
    print(f"[micro16] test users={test_tr.shape[0]}", flush=True)
    concepts = np.load(CONCEPTS, allow_pickle=True)

    A = exp_a(model, test_tr, test_te, n_items, concepts)
    B = exp_b(model, test_tr, test_te, n_items)
    C = exp_c(model, test_tr, test_te, n_items)

    res = {"exp_a_concepts": A, "exp_b_inference": B, "exp_c_variance": C,
           "elapsed_min": round((time.time() - t0) / 60, 2)}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\n[micro16] wrote {OUT_JSON}  ({res['elapsed_min']} min)", flush=True)


if __name__ == "__main__":
    main()
