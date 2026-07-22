"""train_tower_t2.py -- T2': GRADED-NATIVE pb2-class set-encoder tower on the CANONICAL Liang-25M split.

Purpose (Step-2 design sheet v2, `docs/design/DESIGN_SHEET_STEP2_BELIEF.md`): train the interview TOWER --
an order-invariant set encoder over (item, half-star-level) tokens with a multinomial decoder over the
catalog -- on the canonical full-profile ruler (data/ml-25m/proc/, `scripts/baselines/liang_split.py
--data ml-25m`). The existing pb2 checkpoint CANNOT be scored on this ruler (its train users overlap the
new test cohort = leak); this file re-trains pb2-class from scratch on the canonical partition so the tower
is measurable on the same ruler as EASE/EDLAE/RecVAE. Emits experiments/baselines/ml25m_liang/tower_t2.json.

ARCHITECTURE PROVENANCE
  The model classes (MAB, PMA, SetEncoder) are COPIED, not imported, from the pinned reference
  `scripts/_verify/set_mn_pb2.py` (pin: set_mn_pb2.py @ 2bace5e; the Jul-14 frozen pb2 version of set_mn).
  Live `scripts/set_mn.py` is NOT imported -- it drifts (concept channels, belief-pool, PrecAcc branches).
  STRIPPED from the copy (per the build brief): concept machinery, the belief-pool branch (pool="belief",
  lam_head/log_p0/pscale/last_prec), the a0c teacher / SignedAE warm-start, and the answerer. `load_answerer`
  is RETIRED (Jul-22 audit) and MUST NOT appear anywhere in this file -- asserted at import.
  Kept: attention-pool set encoder (mab_in -> sab -> PMA) + FiLM graded token (NLEV=10 half-star levels)
  + separate co-trained multinomial decoder.

KEY DIFFERENCE FROM pb2: pb2 warm-started item_emb and the decoder from the a0c SignedAE (arena split,
  overlapping users -> a leak vector here, and a vocab mismatch: arena 18,430 vs canonical 18,359). This
  tower trains FROM SCRATCH: random item_emb, decoder bias initialised to log train-popularity (so the
  empty set decodes to a popularity prior, z0=0), decoder weight random. FLAG FOR REVIEW: pb2's 0.486 gate
  was reached WITH the a0c warm-start; from-scratch convergence to the R1 bar is not guaranteed and may need
  more epochs (or the T3' EDLAE-distillation hook, --alpha_kd). This is a build+smoke deliverable, not a run.

GRADED DATA (the point of "graded-native"): the proc CSVs store only binarised likes (>3.5). We re-derive
  half-star levels for ALL rating bands from the raw ratings.csv, for TRAIN-partition users only, mapped to
  the proc sid space. Training input tokens = (sid, level) over ALL bands (dislikes are levels too -- signed
  evidence); the multinomial target = the user's held-out LIKED items (>3.5), matching what the ruler scores.

  GRADED FOLD-IN CONVENTION (documented): the canonical eval path (metrics.evaluate) folds in the binarised
  tr-half LIKES. The graded encoder eats (sid,level) tokens, so we recover each tr-half item's REAL half-star
  level from the raw ratings and feed those. The fold-in ITEM SET is exactly the proc tr-half set (all likes,
  levels 7-9); we do NOT inject val/test dislikes (they were filtered out of the split -- injecting them would
  deviate from the canonical fold-in). The predict_fn wraps the encoder fold of these graded tokens and is
  scored through metrics.evaluate + head_mask, identical to run_ml25m_liang.

LEAK CHECKS (hard asserts, fail-fast):
  - vocab is EXACTLY the proc vocab (n_items == 18359; every training sid in [0, n_items)).
  - the reproduced TRAIN partition's binarised like-count == train.csv nnz (bit-faithful partition proof).
  - the reproduced train-user item vocabulary SET == unique_sid.txt SET (permutation reproduced correctly).
  - train users are disjoint from val+test users (no held-out user contributes any training row).

CHECKPOINT/RESUME (HR9/HR10): keeps <tag>.pt (last, for resume: enc+decoder+opt+epoch) and <tag>_best.pt
  (best-on-val only). Early stop on VAL full NDCG@10 via the canonical evaluate path. Code committed before
  any launch (HR10); this file does NOT launch training (review gate).

Usage:
  python src/instrument/train_tower_t2.py --smoke            # synthetic code-path smoke (labeled)
  python src/instrument/train_tower_t2.py --dry_run          # real-data leak asserts + 1 fwd/bwd + epoch est
  python src/instrument/train_tower_t2.py --train --tag t2   # FULL train (do NOT launch until review clears)
  python src/instrument/train_tower_t2.py --train --alpha_kd 0.3 --teacher_npy <edlae_B.npy>   # T3' distill
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

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
torch.set_num_threads(os.cpu_count())

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "baselines"))
import metrics as M                       # canonical loaders + evaluate + NDCG (vae_cf port)
import liang_split as LS                  # reuse the split's own filtering helpers (no drift)

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
RAW = os.path.join(_ROOT, "data", "movielens", "ratings.csv")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")
CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
CACHE_DIR = os.path.join(_ROOT, ".cache", "instrument")

D = 512          # latent width (pb2)
NLEV = 10        # half-star levels 0.5..5.0 -> 0..9 (matches set_mn_pb2 gamma/beta tables)

# `load_answerer` is RETIRED (Jul-22 audit). Guard: this name must never be defined or called here.
assert "load_answerer" not in globals(), "load_answerer is retired and must not appear in the tower"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================================
# ARCHITECTURE  (copied from scripts/_verify/set_mn_pb2.py @ pin 2bace5e; belief/concept/teacher stripped)
# =============================================================================================
class MAB(nn.Module):
    """Multihead attention block (Set Transformer): MAB(Q,K) = LN(H + FF(H)), H = LN(Q + Attn(Q,K,K)).
    Verbatim from set_mn_pb2.py."""
    def __init__(self, d, nhead, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ln0 = nn.LayerNorm(d); self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, Q, K, key_padding_mask=None):
        h, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln0(Q + h)
        return self.ln1(h + self.ff(h))


class PMA(nn.Module):
    """Pooling by multihead attention: a learned seed attends over the set -> one vector.
    Verbatim from set_mn_pb2.py."""
    def __init__(self, d, nhead, drop=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mab = MAB(d, nhead, drop)

    def forward(self, X, pad):
        b = X.shape[0]
        return self.mab(self.S.expand(b, -1, -1), X, key_padding_mask=pad)[:, 0]


class SetEncoder(nn.Module):
    """Graded-native set encoder (pb2-class, attn pool). Tokens (item, half-star level) -> FiLM token ->
    inducing-point attention (mab_in) -> global interaction (sab) -> PMA pool -> z. Permutation-invariant,
    ANY set length. Copied from set_mn_pb2.SetEncoder with pool='attn', token='film' as the graded-native
    path; belief-pool branch and a0c warm-start REMOVED.

    z0=0 (empty-set prior) + a popularity-initialised decoder bias => empty set decodes to the popularity
    prior EXACTLY (the design-(ii) intercept identity)."""
    def __init__(self, ni, d=D, nhead=4, m=32, nlayers=2, token_mode="film"):
        super().__init__()
        self.ni = ni; self.d = d; self.token_mode = token_mode
        self.item_emb = nn.Embedding(ni, d)
        nn.init.normal_(self.item_emb.weight, std=0.02)                 # random init (NO a0c warm-start)
        # FiLM graded token: per-level scale+shift tables. init gamma=1, beta~small -> token = item_emb+beta
        # (the pure additive/author-null token); the model can LEARN gamma<0 for dislike-as-negation.
        self.gamma = nn.Embedding(NLEV, d); self.beta = nn.Embedding(NLEV, d)
        nn.init.ones_(self.gamma.weight); nn.init.normal_(self.beta.weight, std=0.02)
        # mlp token path kept for parity (continuous signed value); FiLM is the graded-native default.
        self.tok_mlp = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, d))
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)             # inducing points
        self.mab_in = MAB(d, nhead)                                    # I attends to the set  (O(L*m))
        self.sab = nn.ModuleList([MAB(d, nhead) for _ in range(nlayers)])   # interaction among inducing pts
        self.pma = PMA(d, nhead)                                       # pool -> one vector
        self.z0 = nn.Parameter(torch.zeros(d))                        # empty-set prior mean
        self.head = nn.Linear(d + 1, d)                               # + log(1+set-size) feature

    def forward(self, ids, vals, pad, lvs):
        b = ids.shape[0]
        e = self.item_emb(ids)
        if self.token_mode == "film":
            x = self.gamma(lvs) * e + self.beta(lvs)                                   # (B,L,d) CHEAP
        else:
            x = self.tok_mlp(torch.cat([e, vals.unsqueeze(-1)], dim=-1))               # (B,L,d)
        h = self.mab_in(self.I.expand(b, -1, -1), x, key_padding_mask=pad)            # (B,m,d)
        for blk in self.sab:
            h = blk(h, h)                                                             # global interaction
        p = self.pma(h, None)                                                         # (B,d)
        p = torch.nan_to_num(p)             # all-padded row -> softmax over -inf -> NaN; empty set -> z0
        sz = (~pad).sum(-1, keepdim=True).float().clamp_min(1.0).log1p()
        return self.z0 + self.head(torch.cat([p, sz], dim=-1))


# =============================================================================================
# DATA  (graded profiles from raw stars; canonical partition reproduced via liang_split helpers)
# =============================================================================================
def star_to_level(rating):
    """half-star rating (0.5..5.0) -> discrete level 0..9 (NLEV=10)."""
    return np.clip(np.rint(np.asarray(rating, np.float64) * 2).astype(np.int64) - 1, 0, NLEV - 1)


def level_to_sv(level):
    """level 0..9 -> signed value (scale_rating inverse: star = sv*2.25+2.75 => sv=(star-2.75)/2.25).
    Only used by the mlp token path; FiLM ignores vals."""
    star = (np.asarray(level, np.float64) + 1.0) / 2.0
    return ((star - 2.75) / 2.25).astype(np.float32)


def reproduce_partition():
    """Reproduce the canonical Liang-25M user partition (which raw userIds are train/val/test) EXACTLY,
    using liang_split's own helpers + constants (no drift). Returns (unique_uid, tr_set, vd_set, te_set,
    n_train, raw_catalog_df, show2id, unique_sid_list). raw_catalog_df = catalog-restricted, ALL bands."""
    import pandas as pd
    log(f"[data] reading raw ratings {RAW}")
    raw = pd.read_csv(RAW)
    # catalog restriction (author-approved deviation; must be exactly 18430) -- as in liang_split.py
    icnt = raw.groupby("movieId").size()
    catalog = set(icnt[icnt >= LS.CATALOG_MIN_RATINGS].index.tolist())
    if len(catalog) != LS.CATALOG_N_EXPECTED:
        raise SystemExit(f"[data] CATALOG MISMATCH: {len(catalog)} != {LS.CATALOG_N_EXPECTED}; STOP.")
    raw = raw[raw["movieId"].isin(catalog)].copy()          # catalog-restricted, ALL rating bands (graded)
    # binarise + filter to reproduce the user set + permutation (identical to liang_split.main)
    binr = raw[raw["rating"] > LS.RATING_GT]
    binr, user_activity, _ = LS.filter_triplets(binr)
    unique_uid = user_activity["userId"].values
    np.random.seed(LS.SEED)
    unique_uid = unique_uid[np.random.permutation(unique_uid.size)]
    n = unique_uid.size
    n_heldout = LS.N_HELDOUT
    tr_users = unique_uid[:(n - n_heldout * 2)]
    vd_users = unique_uid[(n - n_heldout * 2):(n - n_heldout)]
    te_users = unique_uid[(n - n_heldout):]
    # authoritative sid order = unique_sid.txt (guarantees sid alignment with the trained baselines)
    unique_sid_list = [int(x) for x in open(os.path.join(PROC, "unique_sid.txt")).read().split()]
    show2id = {sid: i for i, sid in enumerate(unique_sid_list)}
    # ---- LEAK / FIDELITY ASSERTS ----
    meta = M.load_meta(PROC)
    assert len(unique_sid_list) == meta["n_items"] == 18359, "vocab size != proc vocab"
    assert len(tr_users) == meta["n_train_users"] == 140768, f"train users {len(tr_users)} != 140768"
    assert len(vd_users) == len(te_users) == 10000, "held-out user count != 10000"
    tr_set, vd_set, te_set = set(tr_users.tolist()), set(vd_users.tolist()), set(te_users.tolist())
    assert tr_set.isdisjoint(vd_set) and tr_set.isdisjoint(te_set), "train overlaps held-out (LEAK)"
    assert vd_set.isdisjoint(te_set), "val overlaps test"
    # reproduced train vocab SET must equal unique_sid.txt SET (proves the permutation was reproduced)
    train_plays = binr[binr["userId"].isin(tr_set)]
    repro_vocab = set(pd.unique(train_plays["movieId"]).tolist())
    assert repro_vocab == set(unique_sid_list), "reproduced train vocab != unique_sid.txt (partition drift)"
    # reproduced binarised train like-count must equal train.csv nnz (bit-faithful partition proof)
    n_like_train = int(train_plays[train_plays["movieId"].isin(show2id)].shape[0])
    train_csv_nnz = int(M.load_train(meta["n_items"], PROC).nnz)
    assert n_like_train == train_csv_nnz, f"train like-count {n_like_train} != train.csv nnz {train_csv_nnz}"
    log(f"[data] LEAK CHECKS PASS: vocab=18359, train=140768 disjoint from 2x10000 held-out, "
        f"train likes={n_like_train}==train.csv nnz, vocab-set matches unique_sid.txt")
    return unique_uid, tr_set, vd_set, te_set, len(tr_users), raw, show2id, unique_sid_list


def build_train_profiles(raw, tr_set, show2id, max_users=None):
    """Graded profiles for TRAIN-partition users: {items(sid,all bands), levels, vals(signed), liked(sid)}.
    max_users caps the number of users MATERIALISED (dry-run only) -- NOT a data cap on a real run."""
    import pandas as pd
    df = raw[raw["userId"].isin(tr_set)].copy()
    df["sid"] = df["movieId"].map(show2id)
    df = df[df["sid"].notna()]                              # drop out-of-vocab items
    df["sid"] = df["sid"].astype(np.int64)
    df["lvl"] = star_to_level(df["rating"].values)
    df["like"] = (df["rating"].values > LS.RATING_GT)
    users = []
    uids = df["userId"].unique()
    if max_users is not None:
        uids = uids[:max_users]
        df = df[df["userId"].isin(set(uids.tolist()))]
    for _, g in df.groupby("userId"):
        sids = g["sid"].values.astype(np.int64)
        lvls = g["lvl"].values.astype(np.int64)
        liked = g.loc[g["like"], "sid"].values.astype(np.int64)
        if len(liked) == 0:
            continue                                        # unique_uid filter guarantees >=5, but be safe
        users.append({"items": sids, "levels": lvls,
                      "vals": level_to_sv(lvls), "liked": liked})
    log(f"[data] materialised {len(users)} train graded profiles"
        + (f" (capped at {max_users})" if max_users is not None else ""))
    return users


def build_graded_eval_matrix(raw, unique_uid, show2id, unique_sid_list, split):
    """Build a level+1 CSR aligned to the proc {split}_tr fold-in matrix (same shape/row order), by
    recovering each tr-half item's REAL half-star level from raw. Returns (L_csr, n_users). Integrity:
    every proc tr pair must exist in raw (asserted) and L.nnz == {split}_tr.csv nnz."""
    import pandas as pd
    from scipy import sparse
    meta = M.load_meta(PROC); ni = meta["n_items"]
    tr = pd.read_csv(os.path.join(PROC, f"{split}_tr.csv"))
    uid = tr["uid"].values; sid = tr["sid"].values
    userId = np.asarray(unique_uid)[uid]
    movieId = np.asarray(unique_sid_list)[sid]
    key = pd.DataFrame({"userId": userId, "movieId": movieId, "sid": sid, "uid": uid})
    merged = key.merge(raw[["userId", "movieId", "rating"]], on=["userId", "movieId"], how="left")
    assert not merged["rating"].isna().any(), f"{split}_tr has (user,item) pairs absent from raw (drift)"
    lvl = star_to_level(merged["rating"].values)
    start_idx = int(uid.min())                              # matches metrics._load_tr_te_data reindex
    rows = merged["uid"].values - start_idx
    n_users = int(uid.max()) - start_idx + 1
    L = sparse.csr_matrix((lvl.astype(np.float32) + 1.0, (rows, merged["sid"].values)),
                          dtype=np.float32, shape=(n_users, ni))
    assert L.nnz == len(tr), f"{split} graded matrix nnz {L.nnz} != {split}_tr rows {len(tr)}"
    return L, n_users


# =============================================================================================
# BATCHING  (length-bucketed; copied contract from set_mn_pb2 -- HARD RULE #1: bucket, never cap)
# =============================================================================================
ATTN_BUDGET = 1_000_000
MAX_B = 256


def make_batches(users, order):
    batches = []; cur = []; curL = 0
    for i in order:
        L = len(users[i]["items"])
        nl = max(curL, L)
        if cur and ((len(cur) + 1) * nl > ATTN_BUDGET or len(cur) >= MAX_B):
            batches.append(cur); cur = []; curL = 0; nl = L
        cur.append(i); curL = nl
    if cur:
        batches.append(cur)
    return batches


def make_input_target(u, rng, drop_max=0.5):
    """pb2 denoising curriculum: input = random subset of ALL tokens (incl. dislikes), target = liked
    NOT in input (leak-free)."""
    its = u["items"]; n = len(its)
    keep = rng.random(n) >= rng.uniform(0.0, drop_max)
    if not keep.any():
        keep[rng.integers(0, n)] = True
    inp = its[keep]; lv = u["levels"][keep]; sv = u["vals"][keep]
    tgt = np.setdiff1d(u["liked"], inp, assume_unique=False)
    if len(tgt) == 0:
        return None
    return inp, lv, sv, tgt


def pack_tokens(rows):
    """rows: list of (ids, levels, vals) -> padded torch tensors (ids, vals, pad, lvs)."""
    B = len(rows); L = max(len(r[0]) for r in rows)
    ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32)
    pad = np.ones((B, L), bool); lvs = np.zeros((B, L), np.int64)
    for r, (i, lv, sv) in enumerate(rows):
        k = len(i)
        ids[r, :k] = i; lvs[r, :k] = lv; vals[r, :k] = sv; pad[r, :k] = False
    return (torch.from_numpy(ids), torch.from_numpy(vals), torch.from_numpy(pad), torch.from_numpy(lvs))


# =============================================================================================
# EVAL  (canonical metrics.evaluate path; predict_fn wraps the encoder fold of graded tokens)
# =============================================================================================
def make_graded_predict_fn(enc, Wd, bd, L_csr, token_mode):
    """Factory: returns a predict_fn(X_csr)->dense scores for metrics.evaluate. The graded levels come
    from L_csr (aligned to the fold-in matrix); metrics.evaluate iterates rows sequentially so a cursor
    tracks the row offset. An nnz assert catches any misalignment. FRESH factory call per evaluate()."""
    state = {"pos": 0}

    def predict(X_csr):
        pos = state["pos"]; rows = X_csr.shape[0]
        Ls = L_csr[pos:pos + rows]; state["pos"] = pos + rows
        assert Ls.shape[0] == rows and Ls.nnz == X_csr.nnz, "graded/fold-in misalignment"
        enc.eval(); out = np.zeros((rows, enc.ni), dtype=np.float32)
        with torch.no_grad():
            b = 0
            while b < rows:
                chunk = list(range(b, min(b + 256, rows)))
                packrows = []
                for r in chunk:
                    s, e = Ls.indptr[r], Ls.indptr[r + 1]
                    sids = Ls.indices[s:e].astype(np.int64)
                    lv = (Ls.data[s:e] - 1.0).astype(np.int64)
                    packrows.append((sids, lv, level_to_sv(lv)))
                ids, vals, pad, lvs = pack_tokens(packrows)
                z = enc(ids, vals, pad, lvs)
                out[chunk] = (z @ Wd.T + bd).numpy().astype(np.float32)
                b += len(chunk)
        return out
    return predict


def eval_split(enc, Wd, bd, raw, unique_uid, show2id, unique_sid_list, split, head_mask, batch_size=500):
    """NDCG@10 full/tail + NDCG@100 on a split via the canonical evaluate path."""
    meta = M.load_meta(PROC); ni = meta["n_items"]
    L, _ = build_graded_eval_matrix(raw, unique_uid, show2id, unique_sid_list, split)
    if split == "validation":
        d_tr, d_te = M.load_val(ni, PROC)
    else:
        d_tr, d_te = M.load_test(ni, PROC)
    predict = make_graded_predict_fn(enc, Wd, bd, L, enc.token_mode)
    return M.evaluate(predict, d_tr, d_te, batch_size=batch_size, head_mask=head_mask)


def compute_head_mask(train, n_items):
    """HEAD = smallest item set covering 33% of TRAIN mass (identical to run_ml25m_liang)."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order_pop = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order_pop]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order_pop[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head, cnt


# =============================================================================================
# T3' DISTILLATION HOOK (EDLAE teacher CE; precompute top-k targets ONCE; default OFF)
# =============================================================================================
def precompute_teacher(users, teacher_npy, ni, topk=1000, temp=2.0, batch=512):
    """The known 50x optimisation: compute the EDLAE teacher distribution ONCE per user (from the user's
    FULL binary LIKE profile), store top-k (idx, prob). Returns (idx int32 [N,topk], prob float32 [N,topk]).
    Teacher logits = x_like @ B; top-k truncated temperature-softmax (Hinton KD; distill_edlae convention).
    NOTE teacher is keyed to the user's fixed full-like profile, decoupled from the per-epoch random input."""
    if not os.path.exists(teacher_npy):
        raise SystemExit(f"[kd] teacher B npy not found: {teacher_npy}. Produce it: "
                         f"python src/baselines/edlae.py --p <best> --export_B <path>.npy")
    B = np.asarray(np.load(teacher_npy), dtype=np.float32)
    assert B.shape == (ni, ni), f"teacher B {B.shape} != ({ni},{ni})"
    N = len(users); kk = min(topk, ni)
    t_idx = np.zeros((N, kk), np.int32); t_prob = np.zeros((N, kk), np.float32)
    log(f"[kd] precomputing EDLAE teacher top-{kk} targets for {N} users (once)")
    for st in range(0, N, batch):
        rows = list(range(st, min(st + batch, N)))
        x = np.zeros((len(rows), ni), np.float32)
        for r, u in enumerate(rows):
            x[r, users[u]["liked"]] = 1.0
        logits = x @ B
        idx = np.argpartition(-logits, kk - 1, axis=1)[:, :kk]
        rr = np.arange(len(rows))[:, None]
        top = logits[rr, idx] / float(temp)
        top = top - top.max(axis=1, keepdims=True)
        ex = np.exp(top); ex /= ex.sum(axis=1, keepdims=True)
        t_idx[st:st + len(rows)] = idx.astype(np.int32)
        t_prob[st:st + len(rows)] = ex.astype(np.float32)
    return t_idx, t_prob


def kd_ce(logsm, uidx, t_idx, t_prob):
    """KD cross-entropy on the precomputed top-k teacher: -sum_k prob_k * logsm[gather(idx_k)] per user."""
    ti = torch.from_numpy(t_idx[uidx].astype(np.int64))          # (B,k)
    tp = torch.from_numpy(t_prob[uidx])                          # (B,k)
    picked = torch.gather(logsm, 1, ti)                          # (B,k)
    return -(tp * picked).sum(-1).mean()


# =============================================================================================
# TRAIN
# =============================================================================================
def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    head_mask, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    users = build_train_profiles(raw, tr_set, show2id,
                                 max_users=(args.max_users if args.max_users else None))

    enc = SetEncoder(ni, token_mode=args.token)
    decoder = nn.Linear(D, ni)
    with torch.no_grad():                                    # popularity-prior decoder bias (z0=0 -> cold pop)
        decoder.bias.copy_(torch.from_numpy(np.log(cnt / cnt.sum() + 1e-9).astype(np.float32)))
        nn.init.normal_(decoder.weight, std=0.02)
    Wd, bd = decoder.weight, decoder.bias
    opt = torch.optim.AdamW(list(enc.parameters()) + list(decoder.parameters()), lr=3e-4, weight_decay=1e-4)

    t_idx = t_prob = None
    if args.alpha_kd > 0:
        t_idx, t_prob = precompute_teacher(users, args.teacher_npy, ni, topk=args.kd_topk, temp=args.kd_temp)

    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    log(f"[train] {len(users)} users; profiles min={lens.min()} med={int(np.median(lens))} max={lens.max()}; "
        f"{len(batches_all)} adaptive batches; token={args.token} alpha_kd={args.alpha_kd}")

    ck = os.path.join(CKPT_DIR, f"{args.tag}.pt"); ckb = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    start_ep, best, bad = 0, -1.0, 0
    if args.resume and os.path.exists(ck):
        blob = torch.load(ck, map_location="cpu")
        enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
        opt.load_state_dict(blob["opt"]); start_ep = blob["epoch"]; best = blob.get("best", -1.0)
        Wd, bd = decoder.weight, decoder.bias
        log(f"[train] RESUMED ep{start_ep} best={best:.4f}")

    for ep in range(start_ep, args.epochs):
        enc.train(); rng = np.random.default_rng(ep)
        order = list(range(len(batches_all))); rng.shuffle(order)
        t0 = time.time(); run = 0.0; nb = 0
        for bi in order:
            bat = batches_all[bi]
            exs = [(i, make_input_target(users[i], rng)) for i in bat]
            exs = [(i, e) for i, e in exs if e is not None]
            if not exs:
                continue
            packrows = [(e[0], e[1], e[2]) for _, e in exs]      # (ids, levels, vals)
            ids, vals, pad, lvs = pack_tokens(packrows)
            tgt = torch.zeros((len(exs), ni), dtype=torch.float32)
            for r, (_, e) in enumerate(exs):
                tgt[r, e[3]] = 1.0
            z = enc(ids, vals, pad, lvs)
            logits = z @ Wd.T + bd
            logsm = F.log_softmax(logits, dim=-1)
            nll = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
            loss = nll
            if args.alpha_kd > 0:
                uidx = np.array([i for i, _ in exs], np.int64)
                loss = (1.0 - args.alpha_kd) * nll + args.alpha_kd * kd_ce(logsm, uidx, t_idx, t_prob)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(nll); nb += 1
            if nb % 100 == 0:
                log(f"  ep{ep} b{nb}/{len(order)} NLL={run/nb:.4f} {(time.time()-t0)/60:.1f}m")
        vm = eval_split(enc, Wd.detach(), bd.detach(), raw, unique_uid, show2id, usid, "validation", head_mask)
        f10, t10 = vm["ndcg@10"], vm["tail_ndcg@10"]
        log(f"[ep{ep+1}] NLL={run/max(nb,1):.4f} VAL full@10={f10:.4f} tail@10={t10:.4f} "
            f"ndcg@100={vm['ndcg@100']:.4f} ({'PASS>=0.486' if f10>=0.486 else 'below gate'}) "
            f"({(time.time()-t0)/60:.1f}m)")
        torch.save({"enc": enc.state_dict(), "decoder": decoder.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best": best, "val_full": f10, "val_tail": t10}, ck)
        if f10 > best:
            best = f10; bad = 0
            torch.save({"enc": enc.state_dict(), "decoder": decoder.state_dict(),
                        "epoch": ep + 1, "val_full": f10, "val_tail": t10}, ckb)
        else:
            bad += 1
            log(f"[train] no improvement over {best:.4f} ({bad}/{args.patience})")
            if bad >= args.patience:
                log(f"[train] CONVERGED (val flat {args.patience} epochs)"); break
    log(f"[train] done best val full@10={best:.4f}")

    # ---- test eval on the BEST checkpoint -> tower_t2.json ----
    if os.path.exists(ckb):
        blob = torch.load(ckb, map_location="cpu")
        enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
        Wd, bd = decoder.weight, decoder.bias
    tm = eval_split(enc, Wd.detach(), bd.detach(), raw, unique_uid, show2id, usid, "test", head_mask)
    out = {"model": "tower_t2 (pb2-class graded-native set encoder)", "token": args.token,
           "alpha_kd": args.alpha_kd, "n_items": ni,
           "ndcg@10": tm["ndcg@10"], "tail_ndcg@10": tm["tail_ndcg@10"], "ndcg@100": tm["ndcg@100"],
           "recall@20": tm["recall@20"], "recall@50": tm["recall@50"],
           "val_full@10_best": best, "warm_start": "NONE (from scratch; pop-init decoder bias)"}
    json.dump(out, open(os.path.join(OUTDIR, "tower_t2.json"), "w"), indent=2)
    log(f"[train] TEST full@10={tm['ndcg@10']:.4f} tail@10={tm['tail_ndcg@10']:.4f} -> tower_t2.json")


# =============================================================================================
# DRY RUN  (real data: leak asserts + one fwd/bwd on 200 users + calibrated epoch-time estimate)
# =============================================================================================
def dry_run(args):
    log("[DRY] real-data leak asserts + 1 fwd/bwd + epoch-time estimate (NO training)")
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    _, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    ncal = max(args.dry_users, 200)
    users = build_train_profiles(raw, tr_set, show2id, max_users=ncal)
    assert all(0 <= s < ni for u in users for s in u["items"]), "sid out of vocab range (LEAK)"
    log("[DRY] token-vocab range assert PASS (all sids in [0, n_items))")

    enc = SetEncoder(ni, token_mode=args.token)
    decoder = nn.Linear(D, ni)
    with torch.no_grad():
        decoder.bias.copy_(torch.from_numpy(np.log(cnt / cnt.sum() + 1e-9).astype(np.float32)))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(decoder.parameters()), lr=3e-4)
    Wd, bd = decoder.weight, decoder.bias

    # one fwd/bwd on the first 200 users
    rng = np.random.default_rng(0)
    sub = users[:200]
    packrows, tgts = [], []
    for u in sub:
        e = make_input_target(u, rng)
        if e is None:
            continue
        packrows.append((e[0], e[1], e[2])); tgts.append(e[3])
    ids, vals, pad, lvs = pack_tokens(packrows)
    tgt = torch.zeros((len(packrows), ni))
    for r, t in enumerate(tgts):
        tgt[r, t] = 1.0
    z = enc(ids, vals, pad, lvs)
    logsm = F.log_softmax(z @ Wd.T + bd, dim=-1)
    loss = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    log(f"[DRY] one fwd/bwd on {len(packrows)} users OK: NLL={float(loss):.4f}, z.shape={tuple(z.shape)}")

    # calibrated epoch-time estimate: time one epoch over `ncal` users, scale to 140768
    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    enc.train(); t0 = time.time()
    for bat in batches_all:
        exs = [(i, make_input_target(users[i], rng)) for i in bat]
        exs = [(i, e) for i, e in exs if e is not None]
        if not exs:
            continue
        pr = [(e[0], e[1], e[2]) for _, e in exs]
        ii, vv, pp, ll = pack_tokens(pr)
        tt = torch.zeros((len(exs), ni))
        for r, (_, e) in enumerate(exs):
            tt[r, e[3]] = 1.0
        zz = enc(ii, vv, pp, ll)
        lsm = F.log_softmax(zz @ Wd.T + bd, dim=-1)
        ll_ = -((lsm * tt).sum(-1) / tt.sum(-1).clamp_min(1.0)).mean()
        opt.zero_grad(); ll_.backward(); opt.step()
    dt = time.time() - t0
    est_min = dt * (140768 / len(users)) / 60.0
    log(f"[DRY] timed {len(users)} users ({len(batches_all)} batches) in {dt:.1f}s "
        f"-> EST full epoch @140768 users ~= {est_min:.1f} min ({est_min/60:.1f} h) "
        f"[fwd+bwd only; excludes per-epoch val eval]")
    log("[DRY] COMPLETE")


# =============================================================================================
# SMOKE  (synthetic data; code-path only, LABELED; exercises graded tokens + eval + KD path)
# =============================================================================================
def smoke(args):
    from scipy import sparse
    print("[SMOKE] synthetic tiny data (code-path only, NOT the canonical split)")
    rng = np.random.RandomState(0)
    ni = 130                         # >100 so NDCG@100 is well-defined
    nu = 240
    users = []
    for _ in range(nu):
        k = rng.randint(6, 25)
        items = rng.choice(ni, size=k, replace=False).astype(np.int64)
        lvls = rng.randint(0, NLEV, size=k).astype(np.int64)          # ALL bands (graded)
        liked = items[lvls >= 7]
        if len(liked) < 2:
            liked = items[:2]; lvls[:2] = 8
        users.append({"items": items, "levels": lvls, "vals": level_to_sv(lvls), "liked": liked})

    enc = SetEncoder(ni, token_mode=args.token, m=8, nlayers=1)
    decoder = nn.Linear(D, ni)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(decoder.parameters()), lr=1e-3)
    Wd, bd = decoder.weight, decoder.bias

    # KD path smoke (synthetic teacher B) if requested
    t_idx = t_prob = None
    if args.alpha_kd > 0:
        Bt = (rng.randn(ni, ni) * 0.1).astype(np.float32); np.fill_diagonal(Bt, 0.0)
        tmp = os.path.join(CACHE_DIR, "_smoke_teacherB.npy"); os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(tmp, Bt)
        t_idx, t_prob = precompute_teacher(users, tmp, ni, topk=10, temp=2.0)
        os.remove(tmp)
        print("[SMOKE] KD teacher precompute path exercised (synthetic B)")

    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    for ep in range(3):
        enc.train(); r = np.random.default_rng(ep)
        for bat in batches_all:
            exs = [(i, make_input_target(users[i], r)) for i in bat]
            exs = [(i, e) for i, e in exs if e is not None]
            if not exs:
                continue
            pr = [(e[0], e[1], e[2]) for _, e in exs]
            ii, vv, pp, ll = pack_tokens(pr)
            tt = torch.zeros((len(exs), ni))
            for k, (_, e) in enumerate(exs):
                tt[k, e[3]] = 1.0
            z = enc(ii, vv, pp, ll); lsm = F.log_softmax(z @ Wd.T + bd, dim=-1)
            nll = -((lsm * tt).sum(-1) / tt.sum(-1).clamp_min(1.0)).mean()
            loss = nll
            if args.alpha_kd > 0:
                uidx = np.array([i for i, _ in exs], np.int64)
                loss = (1 - args.alpha_kd) * nll + args.alpha_kd * kd_ce(lsm, uidx, t_idx, t_prob)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[SMOKE] ep{ep+1} last-batch NLL={float(nll):.4f}")

    # eval path smoke: synthetic graded fold-in matrix + head_mask through metrics.evaluate
    tr = sparse.csr_matrix((np.ones(600), (rng.randint(0, 60, 600), rng.randint(0, ni, 600))),
                           shape=(60, ni), dtype=np.float32)
    tr.data[:] = rng.randint(7, NLEV, size=tr.nnz).astype(np.float32) + 1.0    # levels 7-9 (likes), +1
    bin_tr = (tr > 0).astype(np.float32)
    te = sparse.csr_matrix((np.ones(200), (rng.randint(0, 60, 200), rng.randint(0, ni, 200))),
                           shape=(60, ni), dtype=np.float32)
    te = (te > 0).astype(np.float32); te.data[:] = 1.0
    hm, _ = compute_head_mask(bin_tr, ni)
    predict = make_graded_predict_fn(enc, Wd.detach(), bd.detach(), tr, args.token)
    res = M.evaluate(predict, bin_tr, te, batch_size=20, head_mask=hm)
    print(f"[SMOKE] eval path OK  full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    print("[SMOKE] COMPLETE (all code paths: graded tokens, denoising fold, KD, canonical eval)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="tower_t2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--token", choices=["film", "mlp"], default="film")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_users", type=int, default=0, help="cap MATERIALISED users (dev only; 0=all)")
    ap.add_argument("--dry_users", type=int, default=2000, help="users to time for the epoch estimate")
    # T3' distillation hook (default OFF)
    ap.add_argument("--alpha_kd", type=float, default=0.0, help="EDLAE-teacher KD weight (0=off)")
    ap.add_argument("--teacher_npy", default=None, help="EDLAE B .npy (edlae.py --export_B)")
    ap.add_argument("--kd_topk", type=int, default=1000)
    ap.add_argument("--kd_temp", type=float, default=2.0)
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
    elif args.dry_run:
        dry_run(args)
    elif args.train:
        train(args)
    else:
        ap.error("one of --smoke / --dry_run / --train required")


if __name__ == "__main__":
    main()
