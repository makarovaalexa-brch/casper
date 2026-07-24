"""eval_cold_rating_baselines.py -- RATING-AWARE closed-form baselines on the ALL-BANDS graded
reconstruction (VAL cohort): the fair-fight table for the 'graded channel' claim.

CONTEXT (author decomposition insight + parity correction, 2026-07-24):
  * MASKING PARITY: every arm here masks ONLY the liked fold-in items (validation_tr), exactly like the
    canonical convention -- disliked/neutral fold-in items remain RANKABLE candidates. The earlier
    +0.065 all-bands tower claim was a masking artifact (extra ~74 rated-but-unliked candidates were
    masked); under parity the tower's all-bands numbers are full 0.3478 / tail 0.2482 vs likes-only
    0.3435 / 0.2463 (graded premium +0.0043 full).
  * DECOMPOSITION: G6's placebo showed most of the all-bands gain is REVEAL-SET SIZE, not values. Arm
    (g) is the same-information BINARY control: fold x=1 for EVERY rated item. The graded premium claim
    = tower_all_bands - max(binary controls) -- reported explicitly per model where applicable:
    likes-only / all-rated-binary / all-rated-membership-placebo / true-graded.
  * Cremonesi-2010 context line (in the output notes): explicit-rating models LOSE to binarized
    implicit models at top-N ranking -- the published reason the frontier binarizes. The
    'published-faithful' tier is therefore expected to be weak at top-N; it is here for fairness.

ARMS -- TIER 'published-faithful':
  (e) biasedMF (koren2009matrix; Koren, Bell, Volinsky 2009): explicit ALS with user/item biases on
      CENTERED ratings v=(stars-3)/2 (all-bands TRAIN ratings, 140,768 users). Paper-family hyperparams
      (documented): factors=50 (paper reports 50-200), reg=0.05, 15 ALS sweeps; global mean absorbed by
      centering. Fold-in: solve (b_u, p_u) from the user's fold-in ratings against frozen item params;
      score = b_u + b_i + p_u.q_i. Scored top-N on the same targets/masking as every other arm.
  (f) OrdRec (Koren & Sill 2011): CITED-NOT-RUN (no trivially available implementation; noted).

ARMS -- TIER 'our graded extension (best-effort)' (OUR upgrades of the binarized frontier -- generous
best-effort constructions, NOT published claims):
  (a) EASE-signedfold : standard binary-likes EASE B (lam=500); fold x = CENTERED values v for ALL
      rated fold-in items (likes + dislikes) -- the dislike-probe precedent.
  (b) EASE-ratingsfit : EASE closed form fit ON the centered-value train matrix (all bands); fold with
      centered values. Same lam=500 (documented; no sweep).
  (c) kNN-ratingweight: item-item cosine S from binary LIKES (as the frontier does); fold with centered
      values (signed weighting of neighborhoods).
  (d) iALS-explicit   : implicit lib with SIGNED matrix values alpha*v -- the lib's documented negative-
      preference construction: an entry with value c>0 means p=1, confidence 1+c; c<0 means p=0,
      confidence 1+|c| (verified in implicit.cpu.als docstring). Train on alpha*v over all bands,
      factors=200, reg=0.1, alpha=50 (the certified IALS_HP family). Fold-in: closed-form Hu-2008 user
      solve with c=1+alpha|v|, p=(v>0), against frozen item factors.

ARMS -- (g) SAME-INFORMATION BINARY CONTROLS (the decomposition):
  EASE-allratedbin  : binary-likes-fit B folded with x=1 for EVERY rated fold-in item.
  RecVAE-allratedbin: the frozen native RecVAE encoder folded with all-rated binary input.
  (plus their likes-only folds for the decomposition rows.)

CONDITIONS: full (all-bands fold-in) + k8 + k2 (fixed random k-subsets of the ALL-BANDS token set,
truncate_graded seeds COLD_SEED+1 / COLD_SEED -- note: sampled from all bands, so a k-subset may
contain dislikes; the same subsets feed every arm). full/tail NDCG@10.
Output: experiments/baselines/ml25m_liang/rating_baselines_val.json + printed table with the tower's
parity-corrected numbers alongside. Thread-capped; scoring + closed-form/ALS fits only (no training).
"""
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))
import metrics as M
import ease as ease_mod
import itemknn
from train_tower_t2 import (reproduce_partition, truncate_graded, COLD_SEED, NLEV,
                            load_recvae_teacher, log)
from run_battery_phaseA import ndcg10_from_scores

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
OUTP = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang", "rating_baselines_val.json")
IALS_HP = {"factors": 200, "reg": 0.1, "alpha": 50.0, "iterations": 15}

TOWER_REF = {  # parity-corrected tower numbers (masking-parity check 2026-07-24, ep4 snapshot)
    "likes_only": {"full": 0.3435, "tail": 0.2463},
    "all_bands_true_graded": {"full": 0.3478, "tail": 0.2482},
    "all_bands_TAINTED_mask": {"full": 0.4190, "tail": 0.2999},
    "coldk2_canonical": 0.1965, "coldk8_canonical": 0.2584}


def lvl_to_v(lvl):
    """centered value v = (stars-3)/2, stars = (lvl+1)/2  =>  v = (lvl-5)/4. Range [-1.125, +1]."""
    return ((np.asarray(lvl, np.float64) - 5.0) / 4.0).astype(np.float32)


def build_data():
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train = M.load_train(ni, PROC)
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt); cum = np.cumsum(cnt[order]) / cnt.sum()
    hm = np.zeros(ni, bool); hm[order[:np.searchsorted(cum, 0.33) + 1]] = True
    va_tr, va_te = M.load_val(ni, PROC)
    n = va_tr.shape[0]
    unique_uid, tr_set, vd_set, te_set, ntr, raw, show2id, usid = reproduce_partition()
    import pandas as pd
    # ---- VAL all-bands carrier (levels+1), te-half excluded ----
    nall = len(unique_uid); start_vd = nall - 20000
    val_ids = unique_uid[start_vd:start_vd + 10000]
    u2r = {int(u): i for i, u in enumerate(val_ids)}
    vdf = raw[raw["userId"].isin(set(val_ids.tolist()))].copy()
    vdf["sid"] = vdf["movieId"].map(show2id); vdf = vdf[vdf["sid"].notna()]
    vdf["sid"] = vdf["sid"].astype(np.int64)
    vdf["lvl"] = np.clip(np.rint(vdf["rating"].values * 2).astype(np.int64) - 1, 0, NLEV - 1)
    te_sets = [set(va_te[i].indices.tolist()) for i in range(n)]
    rr, cc, dd = [], [], []
    for uidv, g in vdf.groupby("userId"):
        r = u2r[int(uidv)]
        keep = ~g["sid"].isin(te_sets[r]).values
        rr.extend([r] * int(keep.sum())); cc.extend(g["sid"].values[keep].tolist())
        dd.extend((g["lvl"].values[keep] + 1.0).tolist())
    Lv = sparse.csr_matrix((np.asarray(dd, np.float32), (rr, cc)), shape=(n, ni))
    log(f"[data] VAL all-bands carrier: nnz={Lv.nnz}")
    # k-subsets sampled ONCE from the all-bands carrier (same items feed every arm)
    Lk2 = truncate_graded(Lv, 2, COLD_SEED); Lk8 = truncate_graded(Lv, 8, COLD_SEED + 1)
    # ---- TRAIN all-bands centered matrix (for ratings-fit EASE / explicit iALS / biasedMF) ----
    tdf = raw[raw["userId"].isin(tr_set)].copy()
    tdf["sid"] = tdf["movieId"].map(show2id); tdf = tdf[tdf["sid"].notna()]
    tdf["sid"] = tdf["sid"].astype(np.int64)
    uid_pos = {int(u): i for i, u in enumerate(unique_uid[:ntr])}
    tdf["urow"] = tdf["userId"].map(uid_pos)
    v = ((tdf["rating"].values - 3.0) / 2.0).astype(np.float32)
    v[v == 0.0] = 1e-6                                      # keep exact-3.0 entries in the sparse matrix
    Xtr_v = sparse.csr_matrix((v, (tdf["urow"].values, tdf["sid"].values)), shape=(ntr, ni))
    log(f"[data] TRAIN all-bands centered matrix: nnz={Xtr_v.nnz}")
    return dict(ni=ni, n=n, train=train, cnt=cnt, head_mask=hm, va_tr=va_tr, va_te=va_te,
                Lv=Lv, Lk2=Lk2, Lk8=Lk8, Xtr_v=Xtr_v)


def carriers_to_folds(L):
    """carrier (levels+1) -> (signed csr, all-rated-binary csr). Explicit ~0 signed kept via 1e-6."""
    sv = L.copy(); sv.data = lvl_to_v(sv.data - 1.0)
    sv.data[sv.data == 0.0] = 1e-6
    bi = L.copy(); bi.data = np.ones_like(bi.data)
    return sv.tocsr(), bi.tocsr()


def eval_scores(predict, X_fold, D, batch=500):
    """Mean full/tail NDCG@10, PARITY mask (likes-only va_tr)."""
    n = D["n"]; fulls, tails = [], []
    for st in range(0, n, batch):
        en = min(st + batch, n)
        S = np.asarray(predict(X_fold[st:en]), np.float32)
        f, t = ndcg10_from_scores(S, D["va_tr"][st:en], D["va_te"][st:en], D["head_mask"])
        fulls.append(f); tails.append(t)
    f = np.concatenate(fulls); t = np.concatenate(tails)
    return {"full": float(np.nanmean(f)), "tail": float(np.nanmean(t))}


# ------------------------------------------------------------------ published-faithful: biased MF
def fit_biased_mf(Xv, ni, f=50, reg=0.05, iters=15):
    """Koren 2009 (koren2009matrix) explicit MF with biases, ALS on centered ratings.
    v_ui ~ b_u + b_i + p_u.q_i (global mean absorbed by centering). Returns (bi, Q, solver)."""
    rng = np.random.default_rng(0)
    ntr = Xv.shape[0]
    Q = (rng.standard_normal((ni, f)) * 0.01).astype(np.float64)
    bi = np.zeros(ni); bu = np.zeros(ntr)
    P = np.zeros((ntr, f))
    Xu = Xv.tocsr(); Xi = Xv.tocsc()
    for it in range(iters):
        t0 = time.time()
        for u in range(ntr):                                 # user step (bias + factors jointly)
            s, e = Xu.indptr[u], Xu.indptr[u + 1]
            if s == e:
                continue
            cols = Xu.indices[s:e]; r = Xu.data[s:e].astype(np.float64)
            A = np.empty((len(cols), f + 1)); A[:, 0] = 1.0; A[:, 1:] = Q[cols]
            y = r - bi[cols]
            G = A.T @ A + reg * len(cols) * np.eye(f + 1)
            sol = np.linalg.solve(G, A.T @ y)
            bu[u] = sol[0]; P[u] = sol[1:]
        for i in range(ni):                                  # item step
            s, e = Xi.indptr[i], Xi.indptr[i + 1]
            if s == e:
                continue
            rows_u = Xi.indices[s:e]; r = Xi.data[s:e].astype(np.float64)
            A = np.empty((len(rows_u), f + 1)); A[:, 0] = 1.0; A[:, 1:] = P[rows_u]
            y = r - bu[rows_u]
            G = A.T @ A + reg * len(rows_u) * np.eye(f + 1)
            sol = np.linalg.solve(G, A.T @ y)
            bi[i] = sol[0]; Q[i] = sol[1:]
        log(f"  [biasedMF] sweep {it + 1}/{iters} ({(time.time() - t0)/60:.1f}m)")
    def predict(X_csr):
        Xc = X_csr.tocsr()
        out = np.zeros((Xc.shape[0], ni), np.float32)
        for j in range(Xc.shape[0]):
            cols = Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]]
            r = Xc.data[Xc.indptr[j]:Xc.indptr[j + 1]].astype(np.float64)
            if len(cols) == 0:
                out[j] = bi.astype(np.float32); continue
            A = np.empty((len(cols), f + 1)); A[:, 0] = 1.0; A[:, 1:] = Q[cols]
            y = r - bi[cols]
            G = A.T @ A + reg * len(cols) * np.eye(f + 1)
            sol = np.linalg.solve(G, A.T @ y)
            out[j] = (sol[0] + bi + Q @ sol[1:]).astype(np.float32)
        return out
    return predict


# ------------------------------------------------------------------ our extension: explicit iALS
def fit_ials_explicit(Xv, ni, hp=IALS_HP):
    """implicit lib on SIGNED alpha*v (documented negative-preference construction)."""
    from implicit.als import AlternatingLeastSquares
    Cui = Xv.tocsr().astype(np.float32) * hp["alpha"]
    model = AlternatingLeastSquares(factors=hp["factors"], regularization=hp["reg"],
                                    iterations=hp["iterations"], use_gpu=False)
    t0 = time.time()
    model.fit(Cui, show_progress=False)
    log(f"  [ials-explicit] trained f={hp['factors']} ({(time.time()-t0)/60:.1f}m)")
    Y = np.asarray(model.item_factors, np.float64)
    f = Y.shape[1]; YtY = Y.T @ Y; base = YtY + hp["reg"] * np.eye(f)
    alpha = hp["alpha"]
    def predict(X_csr):
        Xc = X_csr.tocsr()
        out = np.zeros((Xc.shape[0], ni), np.float32)
        for j in range(Xc.shape[0]):
            s, e = Xc.indptr[j], Xc.indptr[j + 1]
            cols = Xc.indices[s:e]; v = Xc.data[s:e].astype(np.float64)
            if len(cols) == 0:
                continue
            c = 1.0 + alpha * np.abs(v); p = (v > 0).astype(np.float64)
            Ys = Y[cols]
            A = base + Ys.T @ ((c - 1.0)[:, None] * Ys)      # Hu-2008 user solve, signed prefs
            b = Ys.T @ (c * p)
            out[j] = (Y @ np.linalg.solve(A, b)).astype(np.float32)
        return out
    return predict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    D = build_data()
    ni = D["ni"]
    folds = {}
    for cname, L in (("full", D["Lv"]), ("k8", D["Lk8"]), ("k2", D["Lk2"])):
        folds[cname] = carriers_to_folds(L)                  # (signed, all-rated-binary)
    likes_full = D["va_tr"].astype(np.float32)

    res = {"_notes": {
        "masking": "PARITY: only liked fold-in items masked, every arm (corrected convention)",
        "cremonesi2010": "explicit-rating models lose to binarized implicit at top-N ranking "
                         "(Cremonesi et al., RecSys 2010) -- the published reason the frontier "
                         "binarizes; the published-faithful tier is expected weak at top-N",
        "decomposition": "graded premium = tower_all_bands - max(same-information binary controls)",
        "ordrec": "OrdRec (Koren & Sill 2011) cited-not-run: no trivially available implementation",
        "tower_ref_parity": TOWER_REF},
        "published_faithful": {}, "graded_extension_best_effort": {}, "binary_controls": {}}
    if os.path.exists(OUTP) and not args.force:
        old = json.load(open(OUTP))
        for k in ("published_faithful", "graded_extension_best_effort", "binary_controls"):
            res[k].update(old.get(k, {}))

    def run_arm(tier, name, predict, use="signed", conds=("full", "k8", "k2")):
        if name in res[tier]:
            log(f"[skip] {name}"); return
        t0 = time.time(); row = {}
        for cname in conds:
            sv, bi = folds[cname]
            X = {"signed": sv, "binary": bi, "likes": likes_full}[use]
            row[cname] = eval_scores(predict, X, D)
        row["seconds"] = round(time.time() - t0, 1)
        res[tier][name] = row
        json.dump(res, open(OUTP, "w"), indent=2)
        log(f"[done] {name}: " + " ".join(f"{c}={row[c]['full']:.4f}/{row[c]['tail']:.4f}"
                                          for c in conds))

    os.makedirs(os.path.dirname(OUTP), exist_ok=True)
    ns = argparse.Namespace
    # ---- binary EASE (shared by (a) and (g)) ----
    log("--- EASE binary fit (shared) ---")
    ease_pred = ease_mod.fit(D["train"], ni, args=ns(lam=ease_mod.DEFAULTS["lam"]), log=log)
    B = ease_pred.B
    run_arm("graded_extension_best_effort", "EASE_signedfold",
            lambda X: np.asarray(X @ B, np.float32), use="signed")
    run_arm("binary_controls", "EASE_allratedbin",
            lambda X: np.asarray(X @ B, np.float32), use="binary")
    run_arm("binary_controls", "EASE_likesonly",
            lambda X: np.asarray(X @ B, np.float32), use="likes", conds=("full",))
    # ---- (c) rating-weighted kNN ----
    log("--- kNN ---")
    knn_pred = itemknn.fit(D["train"], ni, args=ns(topk=itemknn.DEFAULTS["topk"]), log=log)
    S = knn_pred.S if hasattr(knn_pred, "S") else None
    if S is None:                                            # fall back to module predict on signed X
        run_arm("graded_extension_best_effort", "kNN_ratingweight", knn_pred, use="signed")
    else:
        run_arm("graded_extension_best_effort", "kNN_ratingweight",
                lambda X: np.asarray(X @ S, np.float32), use="signed")
    # ---- (b) ratings-fit EASE ----
    if "EASE_ratingsfit" not in res["graded_extension_best_effort"]:
        log("--- EASE ratings-fit (centered-value Gram) ---")
        G = np.asarray((D["Xtr_v"].T @ D["Xtr_v"]).todense(), np.float64)
        from ease import inv_spd_lowmem
        G[np.diag_indices(ni)] += 500.0
        C = inv_spd_lowmem(G)
        d = 1.0 / np.diag(C).copy(); C *= -d[None, :]; np.fill_diagonal(C, 0.0)
        B2 = C.astype(np.float32); del G, C
        run_arm("graded_extension_best_effort", "EASE_ratingsfit",
                lambda X: np.asarray(X @ B2, np.float32), use="signed")
        del B2
    # ---- (d) explicit iALS ----
    if "iALS_explicit" not in res["graded_extension_best_effort"]:
        log("--- iALS explicit ---")
        run_arm("graded_extension_best_effort", "iALS_explicit",
                fit_ials_explicit(D["Xtr_v"], ni), use="signed")
    # ---- (g) RecVAE native binary controls ----
    if "RecVAE_allratedbin" not in res["binary_controls"]:
        import torch
        t = load_recvae_teacher(ni)
        def rv_pred(X):
            x = torch.tensor(np.asarray(X.todense(), np.float32))
            x = (x > 0).float()                              # binary membership
            keep = x.sum(1) > 0
            out = np.zeros((x.shape[0], ni), np.float32)
            if bool(keep.any()):
                with torch.no_grad():
                    mu, _ = t.encoder(x[keep], dropout_rate=0.0)
                    out[keep.numpy()] = (t.decoder(mu)).numpy()
            return out
        run_arm("binary_controls", "RecVAE_allratedbin", rv_pred, use="binary")
        run_arm("binary_controls", "RecVAE_likesonly", rv_pred, use="likes", conds=("full",))
    # ---- (e) biased MF (published-faithful) ----
    if "biasedMF_koren2009" not in res["published_faithful"]:
        log("--- biasedMF (Koren 2009 family: f=50 reg=0.05 ALS x15 on centered all-bands) ---")
        run_arm("published_faithful", "biasedMF_koren2009",
                fit_biased_mf(D["Xtr_v"], ni), use="signed")
    res["published_faithful"].setdefault("OrdRec", {"status": "cited-not-run (Koren & Sill 2011)"})
    json.dump(res, open(OUTP, "w"), indent=2)

    # ---- table ----
    def cell(row, c):
        return f"{row[c]['full']:.4f}/{row[c]['tail']:.4f}" if c in row else "-"
    print("\n=== RATING-AWARE BASELINES, VAL, PARITY MASK (full/tail NDCG@10) ===")
    print(f"{'arm':<28} {'full(all-bands)':>16} {'k=8':>14} {'k=2':>14}")
    for tier in ("published_faithful", "graded_extension_best_effort", "binary_controls"):
        print(f"[{tier}]")
        for name, row in res[tier].items():
            if "status" in row:
                print(f"  {name:<26} {row['status']}"); continue
            print(f"  {name:<26} {cell(row,'full'):>16} {cell(row,'k8'):>14} {cell(row,'k2'):>14}")
    tr = TOWER_REF
    print(f"[tower reference, parity mask]")
    print(f"  tower all-bands graded     {tr['all_bands_true_graded']['full']:.4f}/"
          f"{tr['all_bands_true_graded']['tail']:.4f}   (likes-only "
          f"{tr['likes_only']['full']:.4f}/{tr['likes_only']['tail']:.4f}; "
          f"coldk8 {tr['coldk8_canonical']}, coldk2 {tr['coldk2_canonical']})")
    print(f"JSON -> {OUTP}")


if __name__ == "__main__":
    main()
