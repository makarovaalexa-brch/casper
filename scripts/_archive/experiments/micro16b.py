"""
micro16b.py -- Phase-1.6 follow-up probes for Exp A interpretation. EVAL-ONLY.

F1. FIDELITY-v2: does the concept-only FAIL come from generic tag selection?
    Pick per-user tag by LIFT (affinity / global tag mass) instead of raw affinity,
    and also report the raw-affinity chosen tag names (generic-tag diagnosis).
F2. DISLIKE specificity: z-space subtraction -- demotion of tag members vs ALL items
    (rank-based: mean percentile-rank shift of member items), + milder scale sweep.
F3. Input-clamp no-op confirmation: fraction of input mass actually changed.
"""
import os, sys, json, time
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_liang as E
from recvae import RecVAE
import torch

DEVICE = torch.device("cpu")
torch.set_num_threads(3)
CKPT = os.path.join(".cache", "instrument2", "recvae_ml20m_best.pt")
CONCEPTS = os.path.join(".cache", "instrument2", "concepts.npz")
OUT_JSON = os.path.join("experiments", "instrument2", "phase16_micro_followup.json")


def load_model(n_items):
    blob = torch.load(CKPT, map_location=DEVICE)
    a = blob["args"]
    m = RecVAE(a["hidden"], a["latent"], n_items).to(DEVICE)
    m.load_state_dict(blob["model"]); m.eval()
    return m


def encode_dense(model, X):
    with torch.no_grad():
        mu, lv = model.encoder(torch.tensor(X, dtype=torch.float32), dropout_rate=0.0)
    return mu.numpy(), lv.numpy()


def decode(model, z):
    with torch.no_grad():
        return model.decoder(torch.tensor(z, dtype=torch.float32)).numpy().astype(np.float32)


def ndcg(S, tr, te):
    Sp = S.copy(); Sp[tr.nonzero()] = -np.inf
    return E.NDCG_binary_at_k_batch(Sp, te, k=100)


def build_bag(item_tag, tc, M):
    rel = item_tag[:, tc]
    top = np.argpartition(-rel, M)[:M]
    w = rel[top] / (rel[top].sum() + 1e-9)
    bag = np.zeros(item_tag.shape[0], dtype=np.float32); bag[top] = w
    return bag


def sample_k_items(tr_csr, k, rng):
    n, ni = tr_csr.shape
    tr = tr_csr.tocsr()
    X = np.zeros((n, ni), dtype=np.float32)
    for u in range(n):
        it = tr.indices[tr.indptr[u]:tr.indptr[u+1]]
        if len(it) == 0: continue
        kk = min(k, len(it))
        sel = it if kk == len(it) else rng.choice(it, size=kk, replace=False)
        X[u, sel] = 1.0
    return X


def main():
    t0 = time.time()
    meta = E.load_meta(); n_items = meta["n_items"]
    model = load_model(n_items)
    test_tr, test_te = E.load_test(n_items)
    C = np.load(CONCEPTS, allow_pickle=True)
    item_tag = C["item_tag"]; names = [str(x) for x in C["tag_names"]]

    rng = np.random.RandomState(20260704)
    nnz = np.asarray((test_tr > 0).sum(1)).ravel()
    elig = np.where(nnz >= 3)[0]
    uids = rng.choice(elig, size=2000, replace=False)  # same seed/subsample as micro16 exp A
    tr, te = test_tr[uids], test_te[uids]
    nu = len(uids)
    tr_dense = tr.toarray().astype(np.float32)
    affinity = tr_dense @ item_tag                                  # (nu x 200)
    global_mass = item_tag.sum(0)                                   # (200,)
    lift = affinity / (global_mass[None, :] + 1e-9)                 # normalized affinity
    top_raw = affinity.argmax(1); top_lift = lift.argmax(1)

    d = model.decoder.in_features
    floor_S = np.tile(decode(model, np.zeros((1, d), np.float32))[0], (nu, 1))
    nd_floor = ndcg(floor_S, tr, te)

    from collections import Counter
    print("[F1] raw-affinity top tags:",
          Counter([names[t] for t in top_raw]).most_common(8), flush=True)
    print("[F1] lift top tags:       ",
          Counter([names[t] for t in top_lift]).most_common(8), flush=True)

    res = {"floor_ndcg": float(nd_floor.mean()),
           "raw_top_tags": Counter([names[t] for t in top_raw]).most_common(10),
           "lift_top_tags": Counter([names[t] for t in top_lift]).most_common(10),
           "fidelity_v2": {}}
    bag_cache = {}
    for M in (10, 50):
        for sel_name, top in (("raw", top_raw), ("lift", top_lift)):
            X = np.zeros((nu, n_items), dtype=np.float32)
            for i in range(nu):
                key = (int(top[i]), M)
                if key not in bag_cache:
                    bag_cache[key] = build_bag(item_tag, int(top[i]), M)
                X[i] = bag_cache[key]
            mu, _ = encode_dense(model, X)
            nd = ndcg(decode(model, mu), tr, te)
            res["fidelity_v2"][f"M{M}_{sel_name}"] = float(nd.mean())
            print(f"[F1] M={M} sel={sel_name}: concept-only NDCG {nd.mean():.4f} "
                  f"(floor {nd_floor.mean():.4f})", flush=True)

    # ---- F2: dislike z-sub specificity, on 500 users, tag=horror ----
    sub = np.arange(500)
    Xb = sample_k_items(tr[sub], 2, np.random.RandomState(11))
    mu_b, _ = encode_dense(model, Xb)
    tc = names.index("horror")
    bag = build_bag(item_tag, tc, 50)
    members = np.where(bag > 0)[0]
    mu_bag, _ = encode_dense(model, np.tile(bag[None, :], (len(sub), 1)))
    ubag = mu_bag / (np.linalg.norm(mu_bag, axis=1, keepdims=True) + 1e-9)
    S0 = decode(model, mu_b)
    r0 = S0.argsort(1).argsort(1) / n_items          # percentile ranks
    res["dislike_spec"] = {}
    for s in (0.25, 0.5):
        z = mu_b - s * np.linalg.norm(mu_b, axis=1, keepdims=True) * ubag
        S1 = decode(model, z)
        r1 = S1.argsort(1).argsort(1) / n_items
        dm = float((r1[:, members] - r0[:, members]).mean())        # member rank shift
        da = float((r1 - r0).mean())                                # all-items (==0 by constr.)
        # non-member top-1000-of-base control: do popular non-members move?
        base_top = np.argsort(-S0.mean(0))[:1000]
        ctl = np.setdiff1d(base_top, members)
        dc = float((r1[:, ctl] - r0[:, ctl]).mean())
        # NDCG cost of the subtraction (should stay near items-only)
        nd0 = ndcg(S0, tr[sub], te[sub]).mean(); nd1 = ndcg(S1, tr[sub], te[sub]).mean()
        res["dislike_spec"][f"s{s}"] = {"member_rankshift": dm, "control_top_rankshift": dc,
                                        "all_rankshift": da,
                                        "ndcg_before": float(nd0), "ndcg_after": float(nd1)}
        print(f"[F2] s={s}: horror-member percentile shift {dm:+.4f} | "
              f"top-nonmember ctrl {dc:+.4f} | NDCG {nd0:.4f}->{nd1:.4f}", flush=True)

    # ---- F3: input-clamp no-op confirmation ----
    Xneg = np.clip(Xb - bag[None, :], 0, None)
    changed = float(np.abs(Xneg - Xb).sum() / (np.abs(Xb).sum() + 1e-9))
    res["inputclamp_relative_mass_changed"] = changed
    print(f"[F3] input clamp: relative input mass changed = {changed:.5f} (no-op if ~0)", flush=True)

    res["elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[micro16b] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
