"""train_tower_t2.py -- T2': GRADED-NATIVE pb2-class set-encoder tower on the CANONICAL Liang-25M split.

Purpose (Step-2 design sheet v2, `docs/design/DESIGN_SHEET_STEP2_BELIEF.md`): train the interview TOWER --
an order-invariant set encoder over (item, half-star-level) tokens -- on the canonical full-profile ruler
(data/ml-25m/proc/, `scripts/baselines/liang_split.py --data ml-25m`). The existing pb2 checkpoint CANNOT
be scored on this ruler (its train users overlap the new test cohort = leak); this file re-trains pb2-class
on the canonical partition. Emits experiments/baselines/ml25m_liang/tower_t2.json.

TWO TEACHER MODES (author-approved architecture revision, 2026-07-22):
  --teacher recvae  (DEFAULT): distill T1 RecVAE into the set encoder with FROZEN geometry.
      * Loads `.cache/baselines/recvae_ml25m_liang.pt` (src/baselines/recvae.py state-dict; prefers
        state.best_state = the best-on-val weights, HR9, else the last `model`). d_latent=200.
      * FROZEN + reused: the RecVAE decoder Linear(200 -> n_items) INCLUDING its bias. The set encoder
        outputs 200-d (internal width D=512, projected out by the head).
      * FROZEN input item embeddings = the decoder rows Wd (n_items x 200), lifted to the internal width
        by a TRAINABLE in_proj Linear(200 -> 512). Fallback arm: --unfreeze_emb trains the embeddings.
      * Trainable params ONLY: attention blocks (mab_in / sab / PMA + inducing points), FiLM gamma/beta
        tables, in_proj, out head (zero-init) + z0. The FiLM tables REMAIN trainable by design: with the
        item embeddings frozen, gamma/beta are where graded value/sign lives.
      * SIGN INIT ON GAMMA [repair 2; --sign_prior default ON, --no_sign_prior ablation]: insurance
        against the RUNG1 gamma(hated)==gamma(loved) failure -- the sign is UNREACHABLE from a pure null
        init (memory `token-fusion-signed-values`). Init gamma(level) = v(level) in [-1,+1] (v = centered
        valence, NEUTRAL at the 3-star point, clipped), so a hated token STARTS as -e_i = PER-ITEM
        negation, a loved token as +e_i; beta ZERO-init (learned small). The earlier beta-along-u-bar
        variant is REMOVED: it seeded valence on the POPULARITY axis (G3a false-pass, G5 structurally
        unpassable).
      * TOKEN IDENTITY [repair 3]: input item identity = NORMALIZED decoder rows Wd[i]/||Wd[i]||, with
        ||Wd[i]|| appended as a SCALAR token feature (in_proj eats d_emb+1). The OUTPUT decoder stays the
        raw frozen Wd + bias -- normalization is input-side only (identity direction decoupled from the
        popularity-correlated row norm).
      * FREEZE INTEGRITY [repair 6]: frozen tensors are EXCLUDED from the optimizer param list entirely
        (AdamW weight-decay mutates even at zero grad) AND a per-epoch trunk-drift assert checks the
        frozen decoder W/b and item identities are bit-identical to the checkpoint.
      * LOSS (v3, post-adversarial-review 2026-07-22) = per-example
            lam_i * ||z_set(graded S'_i) - z_T_i||^2  +  (1 - lam_i) * rank_i
        rank_i = NLL(decoder(z_set), held likes_i) + w_neg * mean-logprob(held DISLIKES_i)   [repair 4:
        held disliked items (rating<=2.5) enter as explicit DOWN-WEIGHTED negatives -- without this,
        dislikes are input-only and get ~zero target-side gradient. --w_neg, default 0.1.]
        z_T = FROZEN RecVAE encoder MEAN on the BINARIZED S' (same denoising subset, presence-binarized).
        lam_i = lambda_z SCHEDULE [repair 1], NOT flat: lam_i = anneal(epoch) * ramp(k_i) with
        ramp(k)=0 for k<=8 (the teacher latent is covariate-shift garbage at tiny k: RecVAE L2-normalises
        its input, putting 1-8-item subsets far outside training scale), linear ramp to 1 at k>=30; and
        anneal(ep) = lambda_z * max(0, 1 - ep/anneal_epochs) -- the latent KD is a full-profile
        WARM-START only; the NLL owns the endgame. Teacher forward SKIPPED for ramp==0 rows.
        Subset sizes sampled BROADLY: P_INTERVIEW=50% of examples draw k ~ U{1..8}; rest pb2 dropout.
      * EXACT EMPTY-SET INTERCEPT [repair 5]: evidence gate z = z0 + g(n_tok) * fold with
        g(n) = 1 - exp(-softplus(a)*n), so g(0)=0 EXACTLY and FOREVER (the RUNG1 intercept fix -- not an
        init-only property). z0 is FROZEN at 0 in teacher mode => enc(empty) decodes EXACTLY to
        softmax(RecVAE's trained bias) at every point in training. NOTE the intercept is RecVAE's LEARNED
        MARGINAL, not log-count popularity (no log-pop bias init in teacher mode; the bias is the ckpt's).
      * G0 REPORTING SPLIT [repair 7], emitted in tower_t2.json:
          G0-strength = CI-tie test of the tower's full-fold test NDCG@10 vs the frozen RecVAE's own
          full-profile NDCG@10 on the same ruler (|diff| <= 1.96*sqrt(se_a^2+se_b^2));
          G0-identity  = mechanical bit-identity checks (empty-set decode == frozen bias; the graded
          predict harness is a pure pass-through of the encoder fold).
  --teacher none: the previous from-scratch arm (trainable decoder, pop-log bias init, no latent teacher).

ARCHITECTURE PROVENANCE
  The model classes (MAB, PMA, SetEncoder) are COPIED, not imported, from the pinned reference
  `scripts/_verify/set_mn_pb2.py` (pin: set_mn_pb2.py @ 2bace5e; the Jul-14 frozen pb2 version of set_mn).
  Live `scripts/set_mn.py` is NOT imported -- it drifts (concept machinery, belief-pool, PrecAcc branches).
  STRIPPED from the copy: concept machinery, the belief-pool branch (pool="belief", lam_head/log_p0/
  pscale/last_prec), the a0c teacher / SignedAE warm-start, and the answerer. `load_answerer` is RETIRED
  (Jul-22 audit) and MUST NOT appear anywhere in this file -- asserted at import.
  Deviations from the pin (all revision-mandated): d_out decoupled from the internal width (head
  Linear(d+1 -> d_out) ZERO-INIT for the intercept identity), optional frozen item_emb + in_proj lift.
  The RecVAE teacher is IMPORTED from src/baselines/recvae.py (the snap-certified port), not copied.

GRADED DATA (the point of "graded-native"): the proc CSVs store only binarised likes (>3.5). We re-derive
  half-star levels for ALL rating bands from the raw ratings.csv, for TRAIN-partition users only, mapped to
  the proc sid space. Training input tokens = (sid, level) over ALL bands (dislikes are levels too -- signed
  evidence); the multinomial target = the user's held-out LIKED items (>3.5), matching what the ruler scores.
  G3 CANARY (graded-vs-binarised ablation arm): --ablate_binarized collapses every INPUT token's level to a
  constant like-level (8 == 4.5 stars) so the encoder sees presence-only evidence; targets/eval unchanged.

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

CHECKPOINT/RESUME (HR9/HR10): keeps <tag>.pt (last, for resume) and <tag>_best.pt (best-on-val only).
  Early stop on VAL full NDCG@10 via the canonical evaluate path. Code committed before any launch (HR10);
  this file does NOT launch training (review gate).

Usage:
  python src/instrument/train_tower_t2.py --smoke            # synthetic code-path smoke (fake teacher)
  python src/instrument/train_tower_t2.py --dry_run          # real-data leak asserts + 1 fwd/bwd + epoch est
  python src/instrument/train_tower_t2.py --train --tag t2   # FULL train (do NOT launch until review clears)
  python src/instrument/train_tower_t2.py --train --teacher none            # from-scratch fallback arm
  python src/instrument/train_tower_t2.py --train --alpha_kd 0.3 --teacher_npy <edlae_B.npy>  # EDLAE CE hook
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
import recvae as R                        # T1 teacher architecture (IMPORTED, snap-certified port)

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
RAW = os.path.join(_ROOT, "data", "movielens", "ratings.csv")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")
CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
RECVAE_CKPT = os.path.join(_ROOT, ".cache", "baselines", "recvae_ml25m_liang.pt")

D = 512          # internal attention width (pb2)
NLEV = 10        # half-star levels 0.5..5.0 -> 0..9 (matches set_mn_pb2 gamma/beta tables)
LIKE_LEVEL = 8   # constant level (4.5 stars) used by the --ablate_binarized G3-canary arm

# `load_answerer` is RETIRED (Jul-22 audit). Guard: this name must never be defined or called here.
assert "load_answerer" not in globals(), "load_answerer is retired and must not appear in the tower"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================================
# ARCHITECTURE  (copied from scripts/_verify/set_mn_pb2.py @ pin 2bace5e; belief/concept/teacher stripped;
#                d_out decoupling + frozen-emb lift + zero-init head = the 2026-07-22 revision)
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
    """Graded-native set encoder (pb2-class, attn pool). Tokens (item, half-star level) -> [in_proj] ->
    FiLM token -> inducing-point attention (mab_in) -> global interaction (sab) -> PMA pool -> zero-init
    head -> z (d_out). Permutation-invariant, ANY set length.

    Copied from set_mn_pb2.SetEncoder (pool='attn', token='film'); belief-pool branch REMOVED. Revisions
    (2026-07-22, v3 post-adversarial-review): d_out decoupled from the internal width d; item_emb may be
    FROZEN to NORMALIZED teacher decoder rows with ||Wd[i]|| as an appended scalar feature (repair 3);
    EVIDENCE GATE z = z0 + g(n_tok)*fold, g(n)=1-exp(-softplus(a)*n) => g(0)=0 EXACTLY, so the empty set
    decodes to decoder(z0) forever, not just at init (repair 5; z0 frozen at 0 in teacher mode)."""
    def __init__(self, ni, d=D, d_out=D, d_emb=None, nhead=4, m=32, nlayers=2, token_mode="film",
                 norm_feat=False):
        super().__init__()
        d_emb = d_emb or d
        self.ni = ni; self.d = d; self.d_out = d_out; self.token_mode = token_mode
        self.norm_feat = norm_feat
        self.item_emb = nn.Embedding(ni, d_emb)
        nn.init.normal_(self.item_emb.weight, std=0.02)     # overwritten+frozen in teacher mode
        self.register_buffer("emb_norm", torch.ones(ni))    # ||Wd[i]|| scalar feature (repair 3)
        d_in = d_emb + (1 if norm_feat else 0)
        self.in_proj = nn.Linear(d_in, d) if d_in != d else nn.Identity()
        # FiLM graded token: per-level scale+shift tables. init gamma=1, beta~small -> token = e + beta
        # (the pure additive/author-null token); the model can LEARN gamma<0 for dislike-as-negation.
        self.gamma = nn.Embedding(NLEV, d); self.beta = nn.Embedding(NLEV, d)
        nn.init.ones_(self.gamma.weight); nn.init.normal_(self.beta.weight, std=0.02)
        # mlp token path kept for parity (continuous signed value); FiLM is the graded-native default.
        self.tok_mlp = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, d))
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)             # inducing points
        self.mab_in = MAB(d, nhead)                                    # I attends to the set  (O(L*m))
        self.sab = nn.ModuleList([MAB(d, nhead) for _ in range(nlayers)])   # interaction among inducing pts
        self.pma = PMA(d, nhead)                                       # pool -> one vector
        self.z0 = nn.Parameter(torch.zeros(d_out))                     # empty-set prior mean (0)
        self.head = nn.Linear(d + 1, d_out)                            # + log(1+set-size) feature
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # zero-init: stable start
        self.gate_a = nn.Parameter(torch.tensor(0.5413))               # softplus(0.5413) ~= 1.0

    def forward(self, ids, vals, pad, lvs):
        b = ids.shape[0]
        e_in = self.item_emb(ids)
        if self.norm_feat:                                             # repair 3: [Wd_i/||Wd_i||, ||Wd_i||]
            e_in = torch.cat([e_in, self.emb_norm[ids].unsqueeze(-1)], dim=-1)
        e = self.in_proj(e_in)
        if self.token_mode == "film":
            x = self.gamma(lvs) * e + self.beta(lvs)                                   # (B,L,d) CHEAP
        else:
            x = self.tok_mlp(torch.cat([e, vals.unsqueeze(-1)], dim=-1))               # (B,L,d)
        h = self.mab_in(self.I.expand(b, -1, -1), x, key_padding_mask=pad)            # (B,m,d)
        for blk in self.sab:
            h = blk(h, h)                                                             # global interaction
        p = self.pma(h, None)                                                         # (B,d)
        p = torch.nan_to_num(p)             # all-padded row -> softmax over -inf -> NaN; empty set -> z0
        n_tok = (~pad).sum(-1, keepdim=True).float()
        sz = n_tok.clamp_min(1.0).log1p()
        fold = self.head(torch.cat([p, sz], dim=-1))
        g = 1.0 - torch.exp(-F.softplus(self.gate_a) * n_tok)          # repair 5: g(0)=0 EXACTLY, always
        return self.z0 + g * fold


# =============================================================================================
# T1 RECVAE TEACHER  (frozen; encoder mean supervises the latent, decoder+bias reused for ranking)
# =============================================================================================
def load_recvae_teacher(ni, path=RECVAE_CKPT, hidden=600, latent=200):
    """Load the T1 RecVAE (src/baselines/recvae.py checkpoint {'model','state',...}); prefer the
    best-on-val weights (state.best_state, HR9), else the last 'model'. Returns frozen eval model."""
    if not os.path.exists(path):
        raise SystemExit(f"[teacher] RecVAE checkpoint not found: {path} (still training?). "
                         f"Use --teacher none for the from-scratch arm.")
    blob = torch.load(path, map_location="cpu")
    st = blob.get("state", {})
    sd = st.get("best_state") or blob["model"]
    src = "state.best_state" if st.get("best_state") else "model (last)"
    t = R.RecVAE(hidden, latent, ni)
    t.load_state_dict(sd)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    log(f"[teacher] RecVAE loaded from {src}: ep{st.get('epoch','?')} best_val={st.get('best', float('nan')):.4f} "
        f"(hidden={hidden} latent={latent}) -- FROZEN")
    return t


def teacher_latent(teacher, sids_list, ni):
    """z_T = frozen RecVAE encoder MEAN on the BINARIZED subset (presence of ALL subset tokens ->
    1.0; the encoder L2-normalises internally; subsets are never empty by construction).
    sids_list: list of int arrays. Returns (B, latent)."""
    x = torch.zeros((len(sids_list), ni), dtype=torch.float32)
    for r, s in enumerate(sids_list):
        x[r, s] = 1.0
    with torch.no_grad():
        mu, _ = teacher.encoder(x, dropout_rate=0.0)
    return mu


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
    df["dislike"] = (df["rating"].values <= 2.5)            # repair 4: explicit dislike band (<=2.5 stars)
    users = []
    uids = df["userId"].unique()
    if max_users is not None:
        uids = uids[:max_users]
        df = df[df["userId"].isin(set(uids.tolist()))]
    for _, g in df.groupby("userId"):
        sids = g["sid"].values.astype(np.int64)
        lvls = g["lvl"].values.astype(np.int64)
        liked = g.loc[g["like"], "sid"].values.astype(np.int64)
        disliked = g.loc[g["dislike"], "sid"].values.astype(np.int64)
        if len(liked) == 0:
            continue                                        # unique_uid filter guarantees >=5, but be safe
        users.append({"items": sids, "levels": lvls,
                      "vals": level_to_sv(lvls), "liked": liked, "disliked": disliked})
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


P_INTERVIEW = 0.5      # fraction of training examples drawn in the interview regime (tiny subsets)
INTERVIEW_KMAX = 8


def make_input_target(u, rng, drop_max=0.5):
    """Denoising curriculum, revision 2026-07-22: with prob P_INTERVIEW sample an INTERVIEW-REGIME subset
    (k ~ U{1..8} tokens -- supervises the small-set fold the interview lives in); otherwise the pb2
    random-dropout subset. Input includes dislikes (graded); target = liked NOT in input (leak-free)."""
    its = u["items"]; n = len(its)
    if rng.random() < P_INTERVIEW:
        k = int(rng.integers(1, min(INTERVIEW_KMAX, n) + 1))
        keep = np.zeros(n, bool); keep[rng.choice(n, size=k, replace=False)] = True
    else:
        keep = rng.random(n) >= rng.uniform(0.0, drop_max)
        if not keep.any():
            keep[rng.integers(0, n)] = True
    inp = its[keep]; lv = u["levels"][keep]; sv = u["vals"][keep]
    tgt = np.setdiff1d(u["liked"], inp, assume_unique=False)
    if len(tgt) == 0:
        return None
    negs = np.setdiff1d(u.get("disliked", np.empty(0, np.int64)), inp, assume_unique=False)  # repair 4
    return inp, lv, sv, tgt, negs


def pack_tokens(rows, binarize=False):
    """rows: list of (ids, levels, vals) -> padded torch tensors (ids, vals, pad, lvs).
    binarize=True is the G3-canary arm: every input level collapsed to LIKE_LEVEL (presence-only)."""
    B = len(rows); L = max(len(r[0]) for r in rows)
    ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32)
    pad = np.ones((B, L), bool); lvs = np.zeros((B, L), np.int64)
    for r, (i, lv, sv) in enumerate(rows):
        k = len(i)
        ids[r, :k] = i; pad[r, :k] = False
        if binarize:
            lvs[r, :k] = LIKE_LEVEL; vals[r, :k] = level_to_sv(np.full(k, LIKE_LEVEL))
        else:
            lvs[r, :k] = lv; vals[r, :k] = sv
    return (torch.from_numpy(ids), torch.from_numpy(vals), torch.from_numpy(pad), torch.from_numpy(lvs))


# =============================================================================================
# EVAL  (canonical metrics.evaluate path; predict_fn wraps the encoder fold of graded tokens)
# =============================================================================================
def make_graded_predict_fn(enc, Wd, bd, L_csr, binarize=False):
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
                ids, vals, pad, lvs = pack_tokens(packrows, binarize=binarize)
                z = enc(ids, vals, pad, lvs)
                out[chunk] = (z @ Wd.T + bd).numpy().astype(np.float32)
                b += len(chunk)
        return out
    return predict


def eval_split(enc, Wd, bd, raw, unique_uid, show2id, unique_sid_list, split, head_mask,
               batch_size=500, binarize=False):
    """NDCG@10 full/tail + NDCG@100 on a split via the canonical evaluate path."""
    meta = M.load_meta(PROC); ni = meta["n_items"]
    L, _ = build_graded_eval_matrix(raw, unique_uid, show2id, unique_sid_list, split)
    if split == "validation":
        d_tr, d_te = M.load_val(ni, PROC)
    else:
        d_tr, d_te = M.load_test(ni, PROC)
    predict = make_graded_predict_fn(enc, Wd, bd, L, binarize=binarize)
    return M.evaluate(predict, d_tr, d_te, batch_size=batch_size, head_mask=head_mask)


def compute_head_mask(train, n_items):
    """HEAD = smallest item set covering 33% of TRAIN mass (identical to run_ml25m_liang)."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order_pop = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order_pop]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order_pop[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head, cnt


def report_empty_set(enc, Wd, bd, cnt, label):
    """Verify + document what enc(empty) decodes to under the (frozen) decoder: |z_empty| and the
    Spearman of the empty-set scores against the decoder bias and against train popularity."""
    from scipy.stats import spearmanr
    enc.eval()
    with torch.no_grad():
        ids = torch.zeros((1, 1), dtype=torch.long); vals = torch.zeros((1, 1))
        pad = torch.ones((1, 1), dtype=torch.bool); lvs = torch.zeros((1, 1), dtype=torch.long)
        z_empty = enc(ids, vals, pad, lvs)[0]
        sc = (z_empty @ Wd.T + bd).numpy()
    znorm = float(z_empty.norm())
    rho_bias = float(spearmanr(sc, bd.numpy()).statistic)
    rho_pop = float(spearmanr(sc, cnt).statistic) if cnt is not None else float("nan")
    log(f"[{label}] EMPTY-SET: |z|={znorm:.6f}; decode-vs-frozen-bias Spearman={rho_bias:.4f}; "
        f"vs train-popularity Spearman={rho_pop:.4f} "
        f"({'EXACT intercept identity (z=0 -> decoder bias)' if znorm < 1e-6 else 'z0/head trained away from 0 (allowed; measured here)'})")
    return znorm, rho_bias


def count_params(enc, decoder):
    tr_p = sum(p.numel() for p in enc.parameters() if p.requires_grad) \
         + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    fr_p = sum(p.numel() for p in enc.parameters() if not p.requires_grad) \
         + sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    return tr_p, fr_p


# =============================================================================================
# T3' EDLAE DISTILLATION HOOK (score-CE; precompute top-k targets ONCE; default OFF)
# =============================================================================================
def precompute_teacher(users, teacher_npy, ni, topk=1000, temp=2.0, batch=512):
    """The known 50x optimisation: compute the EDLAE teacher distribution ONCE per user (from the user's
    FULL binary LIKE profile), store top-k (idx, prob). Returns (idx int32 [N,topk], prob float32 [N,topk]).
    Teacher logits = x_like @ B; top-k truncated temperature-softmax (Hinton KD; distill_edlae convention)."""
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
    """KD cross-entropy on the precomputed top-k teacher, PER EXAMPLE: -sum_k prob_k * logsm[idx_k]."""
    ti = torch.from_numpy(t_idx[uidx].astype(np.int64))          # (B,k)
    tp = torch.from_numpy(t_prob[uidx])                          # (B,k)
    picked = torch.gather(logsm, 1, ti)                          # (B,k)
    return -(tp * picked).sum(-1)                                # (B,)


# ---- repair 1: lambda_z schedule (subset-size ramp x epoch anneal) ----
RAMP_LO, RAMP_HI = 8, 30      # ramp(k)=0 for k<=8 (teacher covariate shift at tiny k), 1 for k>=30


def lam_ramp(k):
    """ramp(k): 0 for k<=RAMP_LO, linear to 1 at k>=RAMP_HI."""
    return float(np.clip((k - RAMP_LO) / (RAMP_HI - RAMP_LO), 0.0, 1.0))


def lam_anneal(ep, lambda_z, anneal_epochs):
    """Global anneal high->0: latent KD is a full-profile WARM-START; NLL owns the endgame."""
    return lambda_z * max(0.0, 1.0 - ep / max(anneal_epochs, 1))


# =============================================================================================
# MODEL BUILD  (teacher-mode wiring shared by train / dry_run / smoke)
# =============================================================================================
def level_valence(levels=None):
    """Centered valence v(level) in [-1,+1], NEUTRAL at the 3-star point (level 5), linear in stars,
    clipped: v = clip((star - 3.0)/2.0, -1, 1). Levels 0..9 <-> stars 0.5..5.0."""
    lv = np.arange(NLEV) if levels is None else np.asarray(levels)
    star = (lv + 1.0) / 2.0
    return np.clip((star - 3.0) / 2.0, -1.0, 1.0)


def apply_sign_prior(enc):
    """SIGN INIT ON GAMMA (repair 2, 2026-07-22 adversarial review; insurance vs the RUNG1
    sign-unreachable-from-null-init failure, memory `token-fusion-signed-values`):
    gamma(level) = v(level) in [-1,+1] broadcast across dims -- a hated token STARTS as -e_i (PER-ITEM
    negation, not a global direction), a loved token as +e_i; beta ZERO-init (learned small).
    The earlier beta-along-u-bar variant is REMOVED: it seeded valence on the popularity axis
    (G3a false-pass; G5 structurally unpassable)."""
    with torch.no_grad():
        v = torch.from_numpy(level_valence().astype(np.float32))          # (NLEV,)
        enc.gamma.weight.copy_(v.unsqueeze(1).expand(-1, enc.d).contiguous())
        nn.init.zeros_(enc.beta.weight)
    log(f"[model] SIGN INIT applied: gamma(level)=v(level)={np.round(level_valence(), 2).tolist()} "
        f"(per-item negation for dislikes; neutral at 3 stars); beta=0")



def build_model(args, ni, cnt, teacher_override=None):
    """Returns (enc, decoder, teacher, trainable_params). teacher_override lets smoke inject a fake."""
    if args.teacher == "recvae":
        teacher = teacher_override if teacher_override is not None else \
            load_recvae_teacher(ni, hidden=args.t_hidden, latent=args.t_latent)
        d_out = args.t_latent
        enc = SetEncoder(ni, d=D, d_out=d_out, d_emb=d_out, token_mode=args.token, norm_feat=True)
        decoder = nn.Linear(d_out, ni)
        with torch.no_grad():                                # FROZEN RecVAE decoder + bias, reused
            decoder.weight.copy_(teacher.decoder.weight)     # NOTE: bias = RecVAE's LEARNED marginal
            decoder.bias.copy_(teacher.decoder.bias)         # (NOT log-count popularity; repair 5)
            # repair 3: input token identity = NORMALIZED decoder rows; ||Wd[i]|| as scalar feature
            norms = teacher.decoder.weight.norm(dim=1).clamp_min(1e-8)
            enc.item_emb.weight.copy_(teacher.decoder.weight / norms.unsqueeze(1))
            enc.emb_norm.copy_(norms)
        decoder.weight.requires_grad_(False); decoder.bias.requires_grad_(False)
        enc.z0.requires_grad_(False)                         # repair 5: z0 FROZEN at 0 -> exact intercept
        if not args.unfreeze_emb:
            enc.item_emb.weight.requires_grad_(False)
    else:                                                    # --teacher none: from-scratch fallback arm
        teacher = None
        enc = SetEncoder(ni, d=D, d_out=D, token_mode=args.token)
        decoder = nn.Linear(D, ni)
        with torch.no_grad():                                # pop-prior bias (z0=0 -> cold pop)
            decoder.bias.copy_(torch.from_numpy(np.log(cnt / cnt.sum() + 1e-9).astype(np.float32)))
            nn.init.normal_(decoder.weight, std=0.02)
    if getattr(args, "sign_prior", True):
        apply_sign_prior(enc)                                # repair 2: signed gamma init (both modes)
    else:
        log("[model] sign_prior OFF (ablation arm): gamma=1, beta random (additive-null init)")
    # repair 6: frozen tensors EXCLUDED from the optimizer entirely (AdamW decay mutates at zero grad)
    params = [p for p in list(enc.parameters()) + list(decoder.parameters()) if p.requires_grad]
    tr_p, fr_p = count_params(enc, decoder)
    log(f"[model] teacher={args.teacher} d_int={D} d_out={enc.d_out} "
        f"TRAINABLE={tr_p:,} FROZEN={fr_p:,} (unfreeze_emb={args.unfreeze_emb})")
    return enc, decoder, teacher, params


def batch_loss(enc, decoder, teacher, exs, ni, args, lam_glob=None, t_idx=None, t_prob=None):
    """Shared loss for one packed batch (v3 per-example schedule, repairs 1+4).
    exs: list of (user_idx, (inp, lv, sv, tgt, negs)). lam_glob = epoch-annealed global lambda_z.
    loss = mean_i [ lam_i * zMSE_i + (1-lam_i) * rank_i ],   lam_i = lam_glob * ramp(k_i)
    rank_i = like-NLL_i + w_neg * mean-logprob(held dislikes_i)  [+ alpha_kd EDLAE CE mix].
    Returns (loss, nll_float, zmse_float)."""
    B = len(exs)
    packrows = [(e[0], e[1], e[2]) for _, e in exs]
    ids, vals, pad, lvs = pack_tokens(packrows, binarize=args.ablate_binarized)
    tgt = torch.zeros((B, ni), dtype=torch.float32)
    neg = torch.zeros((B, ni), dtype=torch.float32)
    for r, (_, e) in enumerate(exs):
        tgt[r, e[3]] = 1.0
        if len(e) > 4 and len(e[4]) > 0:
            neg[r, e[4]] = 1.0
    z = enc(ids, vals, pad, lvs)
    logits = z @ decoder.weight.T + decoder.bias
    logsm = F.log_softmax(logits, dim=-1)
    nll_vec = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0))               # (B,)
    # repair 4: held dislikes as down-weighted explicit negatives (push their log-prob DOWN).
    # mean logprob of negs is negative; ADDING it penalises high probability on disliked items.
    neg_vec = (logsm * neg).sum(-1) / neg.sum(-1).clamp_min(1.0)                  # (B,) 0 where no negs
    rank_vec = nll_vec + args.w_neg * neg_vec
    if args.alpha_kd > 0:
        uidx = np.array([i for i, _ in exs], np.int64)
        rank_vec = (1.0 - args.alpha_kd) * rank_vec + args.alpha_kd * kd_ce(logsm, uidx, t_idx, t_prob)
    zmse_mean = float("nan")
    if teacher is not None and lam_glob is not None and lam_glob > 0:
        ramp = torch.tensor([lam_ramp(len(e[0])) for _, e in exs], dtype=torch.float32)   # (B,)
        lam = lam_glob * ramp
        sel = (ramp > 0).nonzero(as_tuple=True)[0]
        mse_vec = torch.zeros(B)
        if sel.numel() > 0:                                  # teacher forward ONLY for ramp>0 rows
            z_T = teacher_latent(teacher, [exs[int(r)][1][0] for r in sel], ni)
            mse_sel = ((z[sel] - z_T) ** 2).mean(-1)
            mse_vec = mse_vec.index_copy(0, sel, mse_sel)
            zmse_mean = float(mse_sel.mean())
        loss = (lam * mse_vec + (1.0 - lam) * rank_vec).mean()
        return loss, float(nll_vec.mean()), zmse_mean
    return rank_vec.mean(), float(nll_vec.mean()), zmse_mean


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

    enc, decoder, teacher, params = build_model(args, ni, cnt)
    report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), cnt, "init")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    t_idx = t_prob = None
    if args.alpha_kd > 0:
        t_idx, t_prob = precompute_teacher(users, args.teacher_npy, ni, topk=args.kd_topk, temp=args.kd_temp)

    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    log(f"[train] {len(users)} users; profiles min={lens.min()} med={int(np.median(lens))} max={lens.max()}; "
        f"{len(batches_all)} adaptive batches; token={args.token} teacher={args.teacher} "
        f"lambda_z={args.lambda_z} alpha_kd={args.alpha_kd} ablate_binarized={args.ablate_binarized}")

    ck = os.path.join(CKPT_DIR, f"{args.tag}.pt"); ckb = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    start_ep, best, bad = 0, -1.0, 0
    if args.resume and os.path.exists(ck):
        blob = torch.load(ck, map_location="cpu")
        enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
        opt.load_state_dict(blob["opt"]); start_ep = blob["epoch"]; best = blob.get("best", -1.0)
        log(f"[train] RESUMED ep{start_ep} best={best:.4f}")

    W0 = decoder.weight.detach().clone(); b0 = decoder.bias.detach().clone()   # repair 6 drift ref
    E0 = enc.item_emb.weight.detach().clone()
    for ep in range(start_ep, args.epochs):
        enc.train(); rng = np.random.default_rng(ep)
        lam_glob = lam_anneal(ep, args.lambda_z, args.anneal_epochs)           # repair 1 anneal
        order = list(range(len(batches_all))); rng.shuffle(order)
        t0 = time.time(); run_n = 0.0; run_z = 0.0; nb = 0
        for bi in order:
            bat = batches_all[bi]
            exs = [(i, make_input_target(users[i], rng)) for i in bat]
            exs = [(i, e) for i, e in exs if e is not None]
            if not exs:
                continue
            loss, nll_v, mse_v = batch_loss(enc, decoder, teacher, exs, ni, args, lam_glob, t_idx, t_prob)
            opt.zero_grad(); loss.backward(); opt.step()
            run_n += nll_v; run_z += (0.0 if np.isnan(mse_v) else mse_v); nb += 1
            if nb % 100 == 0:
                log(f"  ep{ep} b{nb}/{len(order)} NLL={run_n/nb:.4f} zMSE={run_z/nb:.4f} "
                    f"{(time.time()-t0)/60:.1f}m")
        if args.teacher == "recvae":                                           # repair 6: trunk drift
            assert torch.equal(decoder.weight, W0) and torch.equal(decoder.bias, b0), \
                "FROZEN decoder drifted (||dWd|| != 0)"
            assert args.unfreeze_emb or torch.equal(enc.item_emb.weight, E0), \
                "FROZEN item identities drifted"
        vm = eval_split(enc, decoder.weight.detach(), decoder.bias.detach(), raw, unique_uid, show2id,
                        usid, "validation", head_mask, binarize=args.ablate_binarized)
        f10, t10 = vm["ndcg@10"], vm["tail_ndcg@10"]
        log(f"[ep{ep+1}] NLL={run_n/max(nb,1):.4f} zMSE={run_z/max(nb,1):.4f} lam_glob={lam_glob:.3f} "
            f"VAL full@10={f10:.4f} tail@10={t10:.4f} ndcg@100={vm['ndcg@100']:.4f} "
            f"({'PASS>=0.486' if f10>=0.486 else 'below gate'}) ({(time.time()-t0)/60:.1f}m)")
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
    report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), cnt, "final")
    tm = eval_split(enc, decoder.weight.detach(), decoder.bias.detach(), raw, unique_uid, show2id,
                    usid, "test", head_mask, binarize=args.ablate_binarized)
    tr_p, fr_p = count_params(enc, decoder)
    out = {"model": "tower_t2 (pb2-class graded-native set encoder, v3)", "token": args.token,
           "teacher": args.teacher, "lambda_z": args.lambda_z, "anneal_epochs": args.anneal_epochs,
           "w_neg": args.w_neg, "alpha_kd": args.alpha_kd, "sign_prior": args.sign_prior,
           "unfreeze_emb": args.unfreeze_emb, "ablate_binarized": args.ablate_binarized,
           "trainable_params": tr_p, "frozen_params": fr_p, "n_items": ni,
           "ndcg@10": tm["ndcg@10"], "ndcg@10_se": tm["ndcg@10_se"],
           "tail_ndcg@10": tm["tail_ndcg@10"], "ndcg@100": tm["ndcg@100"],
           "recall@20": tm["recall@20"], "recall@50": tm["recall@50"], "val_full@10_best": best,
           "geometry": ("FROZEN RecVAE decoder+bias+identities (recvae_ml25m_liang.pt best_state)"
                        if args.teacher == "recvae" else "from scratch; pop-init decoder bias")}
    if args.teacher == "recvae":                             # repair 7: G0 reporting split
        # G0-strength: CI-tie of the tower's full-fold test NDCG@10 vs the frozen RecVAE's own
        te_tr, te_te = M.load_test(ni, PROC)
        rec_res = M.evaluate(R.make_predict_fn(teacher), te_tr, te_te, batch_size=500,
                             head_mask=head_mask)
        diff = tm["ndcg@10"] - rec_res["ndcg@10"]
        ci = 1.96 * float(np.sqrt(tm["ndcg@10_se"] ** 2 + rec_res["ndcg@10_se"] ** 2))
        out["G0_strength"] = {"tower_full@10": tm["ndcg@10"], "recvae_full@10": rec_res["ndcg@10"],
                              "recvae_tail@10": rec_res["tail_ndcg@10"], "diff": diff,
                              "ci95_halfwidth": ci, "tie": bool(abs(diff) <= ci)}
        # G0-identity: mechanical bit-identity of the intercept + harness pass-through
        with torch.no_grad():
            z_e = enc(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
                      torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long))
            sc_e = (z_e @ decoder.weight.T + decoder.bias)[0]
        out["G0_identity"] = {
            "empty_set_equals_frozen_bias": bool(torch.equal(sc_e, decoder.bias.detach())),
            "z_empty_norm": float(z_e.norm())}
        log(f"[G0] strength: tower {tm['ndcg@10']:.4f} vs recvae {rec_res['ndcg@10']:.4f} "
            f"diff={diff:+.4f} ci={ci:.4f} tie={out['G0_strength']['tie']}; "
            f"identity: empty==bias {out['G0_identity']['empty_set_equals_frozen_bias']}")
    json.dump(out, open(os.path.join(OUTDIR, "tower_t2.json"), "w"), indent=2)
    log(f"[train] TEST full@10={tm['ndcg@10']:.4f} tail@10={tm['tail_ndcg@10']:.4f} -> tower_t2.json")


# =============================================================================================
# DRY RUN  (real data: leak asserts + one fwd/bwd + calibrated epoch-time estimate; NO training)
# =============================================================================================
def dry_run(args):
    log(f"[DRY] real-data leak asserts + 1 fwd/bwd + epoch-time estimate (teacher={args.teacher})")
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    _, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    ncal = max(args.dry_users, 200)
    users = build_train_profiles(raw, tr_set, show2id, max_users=ncal)
    assert all(0 <= s < ni for u in users for s in u["items"]), "sid out of vocab range (LEAK)"
    log("[DRY] token-vocab range assert PASS (all sids in [0, n_items))")

    enc, decoder, teacher, params = build_model(args, ni, cnt)
    report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), cnt, "DRY-init")
    opt = torch.optim.AdamW(params, lr=args.lr)

    # one fwd/bwd on the first 200 users
    rng = np.random.default_rng(0)
    exs = [(i, make_input_target(users[i], rng)) for i in range(min(200, len(users)))]
    exs = [(i, e) for i, e in exs if e is not None]
    lam0 = lam_anneal(0, args.lambda_z, args.anneal_epochs)
    loss, nll_v, mse_v = batch_loss(enc, decoder, teacher, exs, ni, args, lam0)
    opt.zero_grad(); loss.backward(); opt.step()
    log(f"[DRY] one fwd/bwd on {len(exs)} users OK: NLL={nll_v:.4f} zMSE={mse_v:.4f} lam_glob={lam0:.3f}")
    if args.teacher == "recvae":
        assert not decoder.weight.requires_grad and not decoder.bias.requires_grad, "decoder not frozen"
        assert not enc.z0.requires_grad, "z0 not frozen (intercept identity needs z0 fixed at 0)"
        assert args.unfreeze_emb or not enc.item_emb.weight.requires_grad, "item_emb not frozen"
        opt_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
        assert id(decoder.weight) not in opt_ids and id(decoder.bias) not in opt_ids \
            and id(enc.z0) not in opt_ids, "frozen tensor leaked into the optimizer (repair 6)"
        log("[DRY] frozen-geometry asserts PASS (decoder/z0/item_emb frozen + excluded from optimizer)")

    # calibrated epoch-time estimate: time one epoch over `ncal` users, scale to 140768
    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    enc.train(); t0 = time.time()
    for bat in batches_all:
        exs = [(i, make_input_target(users[i], rng)) for i in bat]
        exs = [(i, e) for i, e in exs if e is not None]
        if not exs:
            continue
        loss, _, _ = batch_loss(enc, decoder, teacher, exs, ni, args, lam0)
        opt.zero_grad(); loss.backward(); opt.step()
    dt = time.time() - t0
    est_min = dt * (140768 / len(users)) / 60.0
    log(f"[DRY] timed {len(users)} users ({len(batches_all)} batches) in {dt:.1f}s "
        f"-> EST full epoch @140768 users ~= {est_min:.1f} min ({est_min/60:.1f} h) "
        f"[fwd+bwd only; excludes per-epoch val eval]")
    log("[DRY] COMPLETE")


# =============================================================================================
# SMOKE  (synthetic data + FAKE teacher weights; code-path only, LABELED)
# =============================================================================================
def smoke(args):
    from scipy import sparse
    print("[SMOKE] synthetic tiny data + FAKE RecVAE teacher (code-path only, NOT the canonical split)")
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

    teacher_override = None
    if args.teacher == "recvae":
        # fake teacher: random-weight RecVAE saved+loaded through the REAL load path
        args.t_hidden, args.t_latent = 24, 16
        fake = R.RecVAE(args.t_hidden, args.t_latent, ni)
        tmp = os.path.join(CKPT_DIR, "_smoke_recvae.pt"); os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save({"model": fake.state_dict(), "state": {"epoch": 1, "best": 0.0, "best_state": None}}, tmp)
        teacher_override = load_recvae_teacher(ni, path=tmp, hidden=args.t_hidden, latent=args.t_latent)
        os.remove(tmp)
    cnt = np.ones(ni)
    enc, decoder, teacher, params = build_model(args, ni, cnt, teacher_override=teacher_override)
    znorm, rho = report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), None, "SMOKE-init")
    assert znorm < 1e-6, "zero-init intercept identity broken (enc(empty) != 0 at init)"
    if args.teacher == "recvae":
        with torch.no_grad():
            sc0 = (enc(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
                       torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long))
                   @ decoder.weight.T + decoder.bias)[0]
        assert torch.equal(sc0, decoder.bias), "enc(empty) does not decode BIT-EXACTLY to frozen bias"
        print("[SMOKE] intercept identity PASS: enc(empty) decodes BIT-EXACTLY to the frozen decoder "
              "bias (gate g(0)=0 + frozen z0=0 -- holds at every training step, not just init)")
        if args.sign_prior:
            g = enc.gamma.weight.detach()
            v = level_valence()
            assert torch.allclose(g[0], torch.full_like(g[0], -1.0)), "gamma(hated) != -1"
            assert torch.allclose(g[NLEV - 1], torch.full_like(g[0], 1.0)), "gamma(loved) != +1"
            assert float(g[5].abs().max()) < 1e-6, "gamma(3-star) should be exactly 0 (neutral)"
            assert float(enc.beta.weight.detach().abs().max()) < 1e-12, "beta not zero-init"
            print(f"[SMOKE] sign init PASS: gamma(level)=v={np.round(v,2).tolist()} "
                  f"(hated token = -e_i, per-item negation), beta=0")

    t_idx = t_prob = None
    if args.alpha_kd > 0:
        Bt = (rng.randn(ni, ni) * 0.1).astype(np.float32); np.fill_diagonal(Bt, 0.0)
        tmp = os.path.join(CKPT_DIR, "_smoke_teacherB.npy"); os.makedirs(CKPT_DIR, exist_ok=True)
        np.save(tmp, Bt)
        t_idx, t_prob = precompute_teacher(users, tmp, ni, topk=10, temp=2.0)
        os.remove(tmp)
        print("[SMOKE] EDLAE KD teacher precompute path exercised (synthetic B)")

    opt = torch.optim.AdamW(params, lr=1e-3)
    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    for ep in range(3):
        enc.train(); r = np.random.default_rng(ep)
        last = (float("nan"), float("nan"))
        for bat in batches_all:
            exs = [(i, make_input_target(users[i], r)) for i in bat]
            exs = [(i, e) for i, e in exs if e is not None]
            if not exs:
                continue
            loss, nll_v, mse_v = batch_loss(enc, decoder, teacher, exs, ni, args, t_idx, t_prob)
            opt.zero_grad(); loss.backward(); opt.step()
            last = (nll_v, mse_v)
        print(f"[SMOKE] ep{ep+1} last-batch NLL={last[0]:.4f} zMSE={last[1]:.4f}")
    if args.teacher == "recvae":
        assert torch.equal(decoder.weight, teacher_override.decoder.weight) and \
               torch.equal(decoder.bias, teacher_override.decoder.bias), \
               "frozen decoder CHANGED during training"
        assert torch.equal(enc.item_emb.weight, teacher_override.decoder.weight) or args.unfreeze_emb, \
               "frozen item_emb CHANGED during training"
        print("[SMOKE] frozen-geometry immutability PASS (decoder + item_emb bit-identical after 3 epochs)")

    # eval path smoke: synthetic graded fold-in matrix + head_mask through metrics.evaluate
    tr = sparse.csr_matrix((np.ones(600), (rng.randint(0, 60, 600), rng.randint(0, ni, 600))),
                           shape=(60, ni), dtype=np.float32)
    tr.data[:] = rng.randint(7, NLEV, size=tr.nnz).astype(np.float32) + 1.0    # levels 7-9 (likes), +1
    bin_tr = (tr > 0).astype(np.float32)
    te = sparse.csr_matrix((np.ones(200), (rng.randint(0, 60, 200), rng.randint(0, ni, 200))),
                           shape=(60, ni), dtype=np.float32)
    te = (te > 0).astype(np.float32); te.data[:] = 1.0
    hm, _ = compute_head_mask(bin_tr, ni)
    predict = make_graded_predict_fn(enc, decoder.weight.detach(), decoder.bias.detach(), tr)
    res = M.evaluate(predict, bin_tr, te, batch_size=20, head_mask=hm)
    print(f"[SMOKE] eval path OK  full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")
    print("[SMOKE] COMPLETE (paths: graded tokens, interview-regime subsets, frozen-geometry latent "
          "distill, EDLAE KD hook, canonical eval)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="tower_t2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--token", choices=["film", "mlp"], default="film")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_users", type=int, default=0, help="cap MATERIALISED users (dev only; 0=all)")
    ap.add_argument("--dry_users", type=int, default=2000, help="users to time for the epoch estimate")
    # teacher-mode revision (2026-07-22)
    ap.add_argument("--teacher", choices=["recvae", "none"], default="recvae",
                    help="recvae = frozen T1 geometry + latent distill (DEFAULT); none = from-scratch arm")
    ap.add_argument("--lambda_z", type=float, default=0.5, help="latent-regression weight (recvae mode)")
    ap.add_argument("--unfreeze_emb", action="store_true", help="fallback arm: train the item embeddings")
    ap.add_argument("--no_sign_prior", dest="sign_prior", action="store_false", default=True,
                    help="ablation: skip the signed FiLM-beta init (addendum 2026-07-22)")
    ap.add_argument("--t_hidden", type=int, default=600)
    ap.add_argument("--t_latent", type=int, default=200)
    ap.add_argument("--ablate_binarized", action="store_true",
                    help="G3 canary: collapse INPUT levels to a constant like-level (presence-only)")
    # T3' EDLAE score-CE hook (default OFF)
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
