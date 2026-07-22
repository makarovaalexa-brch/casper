"""probe_dislike_separability.py -- DECISIVE pre-training diagnostic (retarget of RUNG1 Check-2).

QUESTION: can the FROZEN canonical RecVAE decoder geometry separate a user's LIKED from DISLIKED
neighbourhoods? Gate for the T2' frozen-geometry tower (DESIGN_SHEET_STEP2_BELIEF.md sec:T2'-v2).

Faithful adaptation of scripts/_archive/experiments/rung1_checks.py Check-2 / Check-2b, retargeted to
the CANONICAL stack:
  - teacher ckpt   = .cache/baselines/recvae_ml25m_liang.pt  (SNAPSHOT-COPIED to scratch first; the file
                     updates per epoch while training runs -- we probe the snapshot, record its epoch).
  - canonical split= data/ml-25m/proc/  (Liang recipe on ML-25M, catalog-restricted; loaders in
                     src/baselines/metrics.py).
  - decoder        = the ckpt's Linear(200 -> n_items) + bias  (structure per src/baselines/recvae.py).
  - graded stars   = re-derived from data/movielens/ratings.csv for the VAL users' items ONLY
                     (dislikes are binarized away by the >3.5 split, so they can only come from raw).

OPERATOR (exactly as directed): z = encoder(binarized fold-in LIKES);
  correctly-signed dislike correction  z'(eta) = z - eta * mean_{i in foldin-dislikes} Wd[i] / ||.||
  (Wd[i] = decoder.weight row i; the mean disliked decoder direction, unit-normalized, subtracted).
  Sweep eta over Check-2's grid {2,4,8}, extended down {0.5,1} to characterize the small-eta regime.

MEASUREMENTS on the DISLIKE-HEAVY cohort (val users with >=3 fold-in disliked catalog items):
  (a) SEPARABILITY -- percentile shift (signed - blind, 1=top) of held-out LIKED vs held-out DISLIKED
      items in the full decode (known items masked). selectivity differential = liked_shift - disliked_shift
      (positive == disliked fall MORE than liked). Pre-registered cutoff: >= 0.10.
  (b) full & tail NDCG@10 delta (signed - blind), targets = held LIKES. Pre-registered cutoff: >= -0.005.

TEACHER COLD-CURVE (part 3, all val users, independent of the cohort): RecVAE encoder fed binarized
  k-item random subsets of each user's fold-in, k in {1,2,4,8,16,full} (seed fixed), decode, full+tail
  NDCG@10 per k. Decides the lambda_z gating question. Pop floor = 0.1345 full (experiments/baselines/
  ml25m_liang/pop.json). Report the k at which full clears the floor.

GRADED FOLD-IN/HELD PROTOCOL (stated, non-lossy):
  - LIKES (star>3.5): fold-in = validation_tr sids, held = validation_te sids  (the CANONICAL partition).
  - DISLIKES (star<=2.5, in vocab): split 80/20 per user with a fixed RNG -> 80% fold-in dislikes (feed
    the correction term) / 20% held dislikes (measured in (a)). MID (2.5<star<=3.5): unused (binarized out).
  All dislikes are used (just partitioned); no user/item/token is dropped beyond the stated cohort.

uid->userId RECONSTRUCTION: the split CSVs carry only uid=profile2id (not userId), and dislikes are
  absent from them -> we replay liang_split.py's DETERMINISTIC prefix (catalog>=20-total-ratings filter,
  star>3.5, filter_triplets(min_uc=5), seed-98765 permutation) to rebuild profile2id, then map the val
  uid block [n_train : n_train+n_val] back to userId. VERIFIED per-user: reconstructed liked-catalog-vocab
  items must equal validation_tr U validation_te for that uid (asserted on a sample; abort on mismatch).

NO TRAINING. Full item catalog. Read-only on .cache except the scratch snapshot copy.

Usage:
  python scripts/_verify/probe_dislike_separability.py \
      --ckpt <scratch snapshot .pt> [--etas 0.5,1,2,4,8] [--min_dislikes 3]
Results JSON -> experiments/baselines/probe_separability.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
import metrics as M          # canonical loaders + vae_cf metrics
import recvae as RV          # RecVAE architecture (Encoder/decoder)

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
RAW = os.path.join(_ROOT, "data", "movielens", "ratings.csv")
OUT_JSON = os.path.join(_ROOT, "experiments", "baselines", "probe_separability.json")
POP_FLOOR_FULL = 0.13453018766377675   # experiments/baselines/ml25m_liang/pop.json ndcg@10

CATALOG_MIN_RATINGS = 20               # liang_split.py: project catalog = movies w/ >=20 total ratings
CATALOG_N_EXPECTED = 18430
SEED = 98765                           # liang_split.py permutation + per-user split seed
MIN_UC = 5
N_HELDOUT = 10000
RATING_GT = 3.5
DISLIKE_LE = 2.5                       # star <= 2.5 == dislike (Check-2 used rating<=2 on a coarser scale;
                                       #  task directive: rating<=2.5)
DISLIKE_HELD_PROP = 0.2               # 80/20 fold-in/held split of a user's dislikes (fixed RNG)


# ------------------------------------------------------------------ model
def load_model(ckpt_path, n_items):
    blob = torch.load(ckpt_path, map_location="cpu")
    a = blob.get("args", {})
    hidden = a.get("hidden", 600); latent = a.get("latent", 200)
    model = RV.RecVAE(hidden, latent, n_items)
    # prefer the tracked best_state if present & != current; else the saved 'model' (== best when
    # best_epoch == current epoch, which is the case for a still-climbing run).
    st = blob.get("state", {})
    sd = st.get("best_state") if st.get("best_state") is not None else blob["model"]
    model.load_state_dict(sd)
    model.eval()
    epoch = st.get("epoch"); best = st.get("best"); best_epoch = st.get("best_epoch")
    return model, dict(epoch=epoch, best_val_ndcg10=best, best_epoch=best_epoch,
                       hidden=hidden, latent=latent)


@torch.no_grad()
def encode_mu(model, X_csr, batch=500):
    """z = encoder mu (deterministic) for a binary csr fold-in matrix."""
    n = X_csr.shape[0]; Z = np.zeros((n, model.encoder.fc_mu.out_features), np.float64)
    for st in range(0, n, batch):
        en = min(st + batch, n)
        x = torch.tensor(X_csr[st:en].toarray(), dtype=torch.float32)
        mu, _ = model.encoder(x, dropout_rate=0.0)
        Z[st:en] = mu.numpy().astype(np.float64)
    return Z


# ------------------------------------------------------------------ split loaders (grouped by uid)
def load_grouped(csv_path):
    import pandas as pd
    tp = pd.read_csv(csv_path)
    g = {}
    for uid, sid in zip(tp["uid"].to_numpy(), tp["sid"].to_numpy()):
        g.setdefault(int(uid), []).append(int(sid))
    return g


# ------------------------------------------------------------------ uid->userId reconstruction
def reconstruct_val_userids(log):
    """Replay liang_split.py's DETERMINISTIC prefix to rebuild profile2id, return:
       val_uid2userid: dict uid(profile2id) -> userId  for the val block,
       show2id: dict movieId -> sid (from unique_sid.txt),
       catalog: set of catalog movieIds.
    Also returns the raw dataframe restricted to (catalog movies) for downstream dislike lookup."""
    import pandas as pd
    log("[recon] loading raw ratings.csv (25M rows, usecols+dtypes to bound RAM) ...")
    raw = pd.read_csv(RAW, usecols=["userId", "movieId", "rating"],
                      dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
    icnt = raw.groupby("movieId").size()
    catalog = set(icnt[icnt >= CATALOG_MIN_RATINGS].index.tolist())
    if len(catalog) != CATALOG_N_EXPECTED:
        raise SystemExit(f"[recon] CATALOG MISMATCH derived {len(catalog)} != {CATALOG_N_EXPECTED}; abort")
    raw = raw[raw["movieId"].isin(catalog)]
    liked = raw[raw["rating"] > RATING_GT]
    # filter_triplets(min_uc=5, min_sc=0) on the liked interactions
    uc = liked[["userId"]].groupby("userId", as_index=False).size()
    keep_u = uc[uc["size"] >= MIN_UC]["userId"]
    liked = liked[liked["userId"].isin(keep_u)]
    user_activity = liked[["userId"]].groupby("userId", as_index=False).size()
    unique_uid = user_activity["userId"].values
    np.random.seed(SEED)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    n_users = unique_uid.size
    n_train = n_users - N_HELDOUT * 2
    vd_users = unique_uid[n_train:(n_users - N_HELDOUT)]
    # profile2id assigns uid = position in unique_uid; val uids are the contiguous block [n_train, ...]
    val_uid2userid = {n_train + i: int(vd_users[i]) for i in range(len(vd_users))}
    log(f"[recon] n_users={n_users} n_train={n_train} val block uids [{n_train},{n_train+len(vd_users)})")
    # show2id from unique_sid.txt (line i = movieId of sid i)
    show2id = {}
    with open(os.path.join(PROC, "unique_sid.txt")) as f:
        for i, line in enumerate(f):
            show2id[int(line.strip())] = i
    return val_uid2userid, show2id, catalog, raw


# ------------------------------------------------------------------ percentile helper (Check-2 _region_percentile)
def region_pctile(scores_row, members, known_idx):
    s = scores_row.copy()
    if known_idx:
        s[list(known_idx)] = -1e18
    order = np.argsort(-s)
    rankfrac = np.empty(len(s)); rankfrac[order] = 1.0 - np.arange(len(s)) / len(s)  # 1=top
    m = [j for j in members if 0 <= j < len(s)]
    return float(np.mean(rankfrac[m])) if m else float("nan")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def ndcg10_rows(scores, held_csr, k=10, head_mask=None, tail=False):
    """Per-user NDCG@10 using the certified vae_cf routine (no metric duplication). scores already has
    fold-in masked. tail=True: head scores -inf, head targets dropped, all-head users -> nan (skipped)."""
    S = scores
    he = held_csr
    if tail:
        S = S.copy(); S[:, head_mask] = -np.inf
        tail_row = (~head_mask).astype("float32")[np.newaxis, :]
        he = he.multiply(tail_row).tocsr(); he.eliminate_zeros()
    keep = np.asarray(he.getnnz(axis=1)).ravel() > 0
    out = np.full(S.shape[0], np.nan)
    if keep.any():
        out[keep] = M.NDCG_binary_at_k_batch(S[keep], he[keep], k=k)
    return out


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SNAPSHOT copy of recvae_ml25m_liang.pt")
    ap.add_argument("--etas", default="0.5,1,2,4,8")
    ap.add_argument("--min_dislikes", type=int, default=3, help="min fold-in dislikes to enter cohort")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    etas = [float(x) for x in args.etas.split(",")]
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    from scipy import sparse
    meta = M.load_meta(PROC); n_items = meta["n_items"]
    log(f"canonical split n_items={n_items} n_val={meta['n_val_users']}")

    # --- model + decoder ---
    model, ckpt_info = load_model(args.ckpt, n_items)
    log(f"ckpt epoch={ckpt_info['epoch']} best_val_ndcg10={ckpt_info['best_val_ndcg10']:.4f} "
        f"best_epoch={ckpt_info['best_epoch']}")
    Wd = model.decoder.weight.detach().numpy().astype(np.float64)   # (n_items, 200)
    bd = model.decoder.bias.detach().numpy().astype(np.float64)     # (n_items,)

    def decode(Z):
        return Z @ Wd.T + bd[None, :]

    # --- head/tail mask from TRAIN matrix (identical to run_ml25m_liang.compute_head_mask) ---
    train = M.load_train(n_items, PROC)
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order_pop = np.argsort(-cnt); cumfrac = np.cumsum(cnt[order_pop]) / cnt.sum()
    head_mask = np.zeros(n_items, dtype=bool)
    head_mask[order_pop[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    log(f"head items={int(head_mask.sum())} tail items={int((~head_mask).sum())}")
    del train

    # --- canonical val fold-in(likes)/held(likes) grouped by uid ---
    va_foldin = load_grouped(os.path.join(PROC, "validation_tr.csv"))
    va_held = load_grouped(os.path.join(PROC, "validation_te.csv"))
    val_uids = sorted(va_foldin.keys())
    log(f"val users with fold-in: {len(val_uids)}")

    # ================================================================
    # PART 3 first (cheap, cohort-independent): TEACHER COLD-CURVE
    # ================================================================
    log("=== PART 3: teacher cold-curve ===")
    ks = [1, 2, 4, 8, 16, "full"]
    # build held-likes csr aligned to val_uids order
    rows_h, cols_h = [], []
    for r, u in enumerate(val_uids):
        for s in va_held.get(u, []):
            rows_h.append(r); cols_h.append(s)
    held_csr = sparse.csr_matrix((np.ones(len(rows_h), np.float32), (rows_h, cols_h)),
                                 shape=(len(val_uids), n_items))
    cold_rows = []
    rng = np.random.default_rng(SEED)
    for k in ks:
        rows_f, cols_f = [], []
        for r, u in enumerate(val_uids):
            items = va_foldin[u]
            if k == "full" or len(items) <= k:
                sub = items
            else:
                sub = list(rng.choice(items, size=k, replace=False))
            for s in sub:
                rows_f.append(r); cols_f.append(s)
        Xk = sparse.csr_matrix((np.ones(len(rows_f), np.float32), (rows_f, cols_f)),
                               shape=(len(val_uids), n_items))
        Z = encode_mu(model, Xk)
        # eval in batches (mask fold-in, then full + tail NDCG@10)
        full_acc, tail_acc = [], []
        B = 500
        for st in range(0, len(val_uids), B):
            en = min(st + B, len(val_uids))
            S = decode(Z[st:en]).astype(np.float32)
            Xsub = Xk[st:en]
            S[Xsub.nonzero()] = -np.inf
            full_acc.append(ndcg10_rows(S, held_csr[st:en]))
            tail_acc.append(ndcg10_rows(S, held_csr[st:en], head_mask=head_mask, tail=True))
        full = np.concatenate(full_acc); tail = np.concatenate(tail_acc)
        row = dict(k=k, full_ndcg10=float(np.nanmean(full)), tail_ndcg10=float(np.nanmean(tail)),
                   clears_pop_floor=bool(np.nanmean(full) > POP_FLOOR_FULL))
        cold_rows.append(row)
        log(f"  k={str(k):>4s} full={row['full_ndcg10']:.4f} tail={row['tail_ndcg10']:.4f} "
            f"clears_pop({POP_FLOOR_FULL:.4f})={row['clears_pop_floor']}")
    first_clear = next((r["k"] for r in cold_rows if r["clears_pop_floor"]), None)

    # ================================================================
    # PART 1+2: reconstruct dislikes, build dislike-heavy cohort
    # ================================================================
    log("=== PART 1/2: reconstruct graded dislikes ===")
    val_uid2userid, show2id, catalog, raw = reconstruct_val_userids(log)

    # sanity: reconstructed liked-catalog-vocab items == validation_tr U validation_te for a sample
    sample = val_uids[:200]
    sample_userids = {val_uid2userid[u] for u in sample if u in val_uid2userid}
    liked_raw = raw[(raw["rating"] > RATING_GT) & (raw["userId"].isin(sample_userids))]
    liked_by_user = {}
    for uid, mid in zip(liked_raw["userId"].to_numpy(), liked_raw["movieId"].to_numpy()):
        if mid in show2id:
            liked_by_user.setdefault(int(uid), set()).add(show2id[mid])
    n_ok = 0
    for u in sample:
        userid = val_uid2userid.get(u)
        recon = liked_by_user.get(userid, set())
        canon = set(va_foldin.get(u, [])) | set(va_held.get(u, []))
        if recon == canon:
            n_ok += 1
    log(f"[recon] uid->userId verify: {n_ok}/{len(sample)} sample users match canonical liked set")
    if n_ok < len(sample):
        raise SystemExit(f"[recon] MAPPING MISMATCH ({n_ok}/{len(sample)}); reconstruction unreliable -- abort")

    # disliked catalog+vocab items per val userId (star <= 2.5) -- restrict to VAL userIds (RAM)
    val_userid_set = set(val_uid2userid.values())
    dis_raw = raw[(raw["rating"] <= DISLIKE_LE) & (raw["userId"].isin(val_userid_set))]
    dislikes_by_user = {}
    for uid, mid in zip(dis_raw["userId"].to_numpy(), dis_raw["movieId"].to_numpy()):
        if mid in show2id:
            dislikes_by_user.setdefault(int(uid), []).append(show2id[mid])
    del raw, liked_raw, dis_raw

    # build cohort: 80/20 split each user's dislikes (fixed RNG, sorted for determinism)
    rng2 = np.random.default_rng(SEED + 1)
    cohort = []   # (uid, foldin_likes, held_likes, foldin_dislikes, held_dislikes)
    for u in val_uids:
        userid = val_uid2userid.get(u)
        dis = sorted(set(dislikes_by_user.get(userid, [])))
        # exclude any dislike that also appears as a canonical like (shouldn't happen; guard)
        likeset = set(va_foldin.get(u, [])) | set(va_held.get(u, []))
        dis = [d for d in dis if d not in likeset]
        if len(dis) < args.min_dislikes:
            continue
        dis_arr = np.array(dis)
        n_held = int(DISLIKE_HELD_PROP * len(dis_arr))
        held_mask = np.zeros(len(dis_arr), bool)
        if n_held > 0:
            held_mask[rng2.choice(len(dis_arr), size=n_held, replace=False)] = True
        foldin_dis = list(dis_arr[~held_mask]); held_dis = list(dis_arr[held_mask])
        if len(foldin_dis) < args.min_dislikes:
            continue
        cohort.append((u, va_foldin.get(u, []), va_held.get(u, []), foldin_dis, held_dis))
    log(f"dislike-heavy cohort: {len(cohort)} users (>= {args.min_dislikes} fold-in dislikes)")
    n_with_held_dis = sum(1 for c in cohort if c[4])
    log(f"  cohort users with >=1 held dislike: {n_with_held_dis}")

    # encode blind z for cohort (input = binarized fold-in LIKES only)
    rows_f, cols_f = [], []
    for r, (u, fl, hl, fd, hd) in enumerate(cohort):
        for s in fl:
            rows_f.append(r); cols_f.append(s)
    Xc = sparse.csr_matrix((np.ones(len(rows_f), np.float32), (rows_f, cols_f)),
                           shape=(len(cohort), n_items))
    Zc = encode_mu(model, Xc)   # blind z

    # held-likes csr for cohort NDCG
    rows_hl, cols_hl = [], []
    for r, (u, fl, hl, fd, hd) in enumerate(cohort):
        for s in hl:
            rows_hl.append(r); cols_hl.append(s)
    cohort_held_lik = sparse.csr_matrix((np.ones(len(rows_hl), np.float32), (rows_hl, cols_hl)),
                                        shape=(len(cohort), n_items))

    # per-user correction direction d_u = normalized mean Wd[foldin_dislikes]
    dirs = np.zeros((len(cohort), Wd.shape[1]), np.float64)
    for r, (u, fl, hl, fd, hd) in enumerate(cohort):
        m = Wd[fd].mean(axis=0)
        nrm = np.linalg.norm(m)
        dirs[r] = m / nrm if nrm > 0 else m

    # ---- sweep eta: (a) separability percentile shift, (b) NDCG delta ----
    log("=== PART 2 sweep ===")
    # precompute blind scores + blind percentiles once
    B = 500
    eta_results = []
    # blind full/tail NDCG (targets = held likes)
    blind_full, blind_tail = [], []
    blind_pdis, blind_plik = {}, {}   # per-user blind percentile (held dislikes / held likes)
    for st in range(0, len(cohort), B):
        en = min(st + B, len(cohort))
        Sb = decode(Zc[st:en]).astype(np.float64)
        # mask known = fold-in likes + fold-in dislikes
        for i, r in enumerate(range(st, en)):
            u, fl, hl, fd, hd = cohort[r]
            known = fl + fd
            Sb[i, known] = -1e18
        blind_full.append(ndcg10_rows(Sb.astype(np.float32), cohort_held_lik[st:en]))
        blind_tail.append(ndcg10_rows(Sb.astype(np.float32), cohort_held_lik[st:en],
                                      head_mask=head_mask, tail=True))
        for i, r in enumerate(range(st, en)):
            u, fl, hl, fd, hd = cohort[r]
            known = set(fl) | set(fd)
            if hd:
                blind_pdis[r] = region_pctile(Sb[i], hd, known)
            if hl:
                blind_plik[r] = region_pctile(Sb[i], hl, known)
    blind_full = np.concatenate(blind_full); blind_tail = np.concatenate(blind_tail)
    blind_full_m = float(np.nanmean(blind_full)); blind_tail_m = float(np.nanmean(blind_tail))
    log(f"  blind full={blind_full_m:.4f} tail={blind_tail_m:.4f}")

    for eta in etas:
        Zs = Zc - eta * dirs
        sig_full, sig_tail = [], []
        dis_shift, lik_shift = [], []
        for st in range(0, len(cohort), B):
            en = min(st + B, len(cohort))
            Ss = decode(Zs[st:en]).astype(np.float64)
            for i, r in enumerate(range(st, en)):
                u, fl, hl, fd, hd = cohort[r]
                known = fl + fd
                Ss[i, known] = -1e18
            sig_full.append(ndcg10_rows(Ss.astype(np.float32), cohort_held_lik[st:en]))
            sig_tail.append(ndcg10_rows(Ss.astype(np.float32), cohort_held_lik[st:en],
                                        head_mask=head_mask, tail=True))
            for i, r in enumerate(range(st, en)):
                u, fl, hl, fd, hd = cohort[r]
                known = set(fl) | set(fd)
                if hd and r in blind_pdis:
                    p = region_pctile(Ss[i], hd, known)
                    dis_shift.append(p - blind_pdis[r])
                if hl and r in blind_plik:
                    p = region_pctile(Ss[i], hl, known)
                    lik_shift.append(p - blind_plik[r])
        sig_full = np.concatenate(sig_full); sig_tail = np.concatenate(sig_tail)
        d_full = float(np.nanmean(sig_full) - blind_full_m)
        d_tail = float(np.nanmean(sig_tail) - blind_tail_m)
        dis_m = float(np.mean(dis_shift)); lik_m = float(np.mean(lik_shift))
        sel_diff = lik_m - dis_m   # positive == disliked fall MORE than liked
        ci_dis = boot_ci(dis_shift, seed=9); ci_lik = boot_ci(lik_shift, seed=10)
        row = dict(eta=eta,
                   signed_full_ndcg10=float(np.nanmean(sig_full)),
                   signed_tail_ndcg10=float(np.nanmean(sig_tail)),
                   full_ndcg10_delta=d_full, tail_ndcg10_delta=d_tail,
                   held_disliked_pctile_shift=dis_m, ci_disliked=list(ci_dis),
                   held_liked_pctile_shift=lik_m, ci_liked=list(ci_lik),
                   selectivity_differential=sel_diff,
                   n_held_dislike_users=len(dis_shift), n_held_like_users=len(lik_shift))
        eta_results.append(row)
        log(f"  eta={eta:>4.1f} dFULL={d_full:+.4f} dTAIL={d_tail:+.4f} | "
            f"dis_shift={dis_m:+.4f} lik_shift={lik_m:+.4f} SELDIFF={sel_diff:+.4f}")

    # ---- verdicts ----
    # separability: best (max) selectivity differential across eta; and the NDCG delta at that eta
    best_sep = max(eta_results, key=lambda r: r["selectivity_differential"])
    # NDCG guard: least-harmful full delta on the cohort
    best_ndcg = max(eta_results, key=lambda r: r["full_ndcg10_delta"])
    sep_pass = bool(best_sep["selectivity_differential"] >= 0.10 and best_sep["full_ndcg10_delta"] >= -0.005)
    # also report the combined-cutoff view: an eta that satisfies BOTH cutoffs simultaneously
    both = [r for r in eta_results if r["selectivity_differential"] >= 0.10
            and r["full_ndcg10_delta"] >= -0.005]

    verdicts = dict(
        separability=dict(
            passed=sep_pass,
            any_eta_satisfies_both_cutoffs=bool(both),
            best_selectivity_differential=best_sep["selectivity_differential"],
            best_selectivity_eta=best_sep["eta"],
            full_ndcg_delta_at_best_sel=best_sep["full_ndcg10_delta"],
            best_full_ndcg_delta=best_ndcg["full_ndcg10_delta"],
            best_full_ndcg_delta_eta=best_ndcg["eta"],
            cutoffs="selectivity_differential>=0.10 AND cohort full-NDCG delta>=-0.005",
        ),
        teacher_cold_curve=dict(
            first_k_clearing_pop_floor=first_clear,
            pop_floor_full=POP_FLOOR_FULL,
            full_ndcg10_at_full_profile=cold_rows[-1]["full_ndcg10"],
        ),
    )

    results = dict(
        ckpt=os.path.abspath(args.ckpt),
        ckpt_info=ckpt_info,
        operator="z' = z - eta * mean_{i in foldin-dislikes} Wd[i] / ||mean Wd||  (Wd = decoder.weight row)",
        n_items=n_items,
        cohort_size=len(cohort), cohort_with_held_dislike=n_with_held_dis,
        min_dislikes=args.min_dislikes, dislike_threshold_star=DISLIKE_LE,
        etas=etas,
        blind_full_ndcg10=blind_full_m, blind_tail_ndcg10=blind_tail_m,
        eta_sweep=eta_results,
        cold_curve=cold_rows,
        verdicts=verdicts,
        minutes=round((time.time() - t0) / 60, 1),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=float)
    log(f"wrote {args.out}  [{results['minutes']}m]")

    print("\n================ VERDICTS ================")
    print(f"SEPARABILITY: passed={sep_pass}  best selectivity differential="
          f"{best_sep['selectivity_differential']:+.4f} (eta={best_sep['eta']}), "
          f"cohort full-NDCG delta there={best_sep['full_ndcg10_delta']:+.4f}")
    print(f"  (cutoffs: sel-diff>=0.10 AND full-NDCG delta>=-0.005; any eta meets both: {bool(both)})")
    print(f"TEACHER COLD-CURVE: first k clearing pop floor {POP_FLOOR_FULL:.4f} = {first_clear}; "
          f"full-profile full-NDCG={cold_rows[-1]['full_ndcg10']:.4f}")
    return results


if __name__ == "__main__":
    main()
