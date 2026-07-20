"""i25_fold_recovered.py -- FOLD (RECOVERED): fixed prior + ATTENTION-POOL delta + CONTENT-confidence
shrinkage gate, over the FROZEN RecVAE-d512 decoder. Per DESIGN_SHEET_FOLD_RECOVERED.md (LOCKED
2026-07-10). RECOVERS the June attention-pool + Bayesian-shrinkage no-harm design, ports it to the
RecVAE-d512 world, keeps fold-v3's two-channel contribution. NOT a reinvention.

z = prior + w * delta, where
  PRIOR    = the FIXED population/cold RecVAE encoding enc_items([]) (== 0 vector), INDEPENDENT of the
             revealed items. A true shared intercept: cold == prior. (Undoes the Q1-drop bug that came
             from jumping the prior to a single-item RecVAE encoding.)
  DELTA    = ATTENTION-POOL over answer tokens (softmax weights SUM TO 1 -> count-normalized), then a
             small residual MLP. NOT sum-pool. NOT set-transformer/MHSA (rejected 3x).
  w (GATE) = CONTENT confidence, NOT count: per-token fidelity (data>ease>vague) x multi-token
             coherence (attention-pool agreement). w == 0 with zero tokens (empty -> prior exactly).
             w NEVER receives ntok / log(ntok). Restores Bayesian shrinkage: weak evidence -> stay near
             prior (no Q1 drop); consistent -> confident; noise -> down-weighted (no full-profile harm).

FEATURES (sheet 1): NO hand-crafted surprise ratio. Raw per-token log(n_E), log(V), log(p_E) + 512-d
  emb + type/kind/level/value/fidelity; the net learns the combination. Two-channel implicit/explicit
  tokens KEPT; concepts/entities/items are learned channels (type one-hot).
ANSWERS (sheet 2): DATA-SIDE ONLY (real ratings U EASE, via the arena's gated v2.1 value backbone).
  cos(z*, q) / recommender-geometry answers appear NOWHERE (G-firewall asserts this).
TRAINING (sheet 3): masked-reveal reconstruction of held-out LIKED items through the FROZEN RecVAE
  decoder (multinomial LL, IPS / inverse-popularity weighted). Curriculum = DeOODGen mixture of
  realistic blind selection strategies (realistic tails ~0.18) at arbitrary lengths 1..24 + ~30% clean
  full profiles; ONE strategy (blind_eig) HELD OUT for G-generalize. Frozen decoder, deterministic,
  best-on-disjoint-val.
SPLIT (sheet 4): population trU users (300 study ids EXCLUDED by make_cohorts). TRAIN ~20000 / disjoint
  VAL ~2000 / disjoint TEST ~2000. Per-user known/held split pinned (seed-123 population_split); held
  targets never in inputs; every gate on TEST users unseen in train/val.

NO LLM calls. $0. Deterministic. Reduced threads.

Run:
  python scripts/i25_fold_recovered.py sanity                                  # ~100 users, 3 epochs
  python scripts/i25_fold_recovered.py full --n_train 20000 --n_val 2000 --n_test 2000 --epochs 12
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
import sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import i25_lib as L
import i25_fold_v4 as V4
from i25_fold_v3_sampler import (TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY,
                                  KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG,
                                  FID_DATA, FID_EASE, FID_LLM)

D_LAT = L.D_LAT
BEST = ".cache/i25_fold_recovered_best.pt"
CKPT = ".cache/i25_fold_recovered.pt"
LOGP = ".cache/i25_fold_recovered_log.json"
RESULTS = ".cache/arena/fold_recovered_results.json"
BUILD_MD = "experiments/ARENA_BUILD.md"
CLEAN_FRAC = 0.30
MICRO_TOK = 20000                          # length-bucket micro-batch budget: cap on sum(#tokens) per
                                           # forward sub-batch (bounds memory without dropping tokens)
K = 10                                   # primary NDCG endpoint

# realistic curriculum: tails (off_niche+adversarial) ~0.18, on_profile 0.22 (NOT 0.7), NOT fat.
STRAT_W = {"random": 0.18, "popularity": 0.12, "entropy": 0.14, "on_profile": 0.22,
           "off_niche": 0.10, "adversarial": 0.08, "mixed": 0.16}
SEEN = list(STRAT_W.keys())
HELDOUT = "blind_eig"                     # never trained; probed in G-generalize
VBIN = AC.VBIN_CENTERED                    # centered fold value per 4-level value index
ANS_CACHE_DIR = ".cache/arena/ans_lazy"    # PERSISTENT per-user answer cache (content-keyed)
N_SHARD = 64


# =============================================================== LAZY answerer arena
class LazyArena(AC.Arena):
    """Arena with LAZY, on-demand answer computation + a PERSISTENT per-user cache.

    Two fixes vs the eager path (which regenerated all 2428 answers x 24k users every run):
      (1) SPARSE EASE fold-in: pred = ease_mu + rc @ ease_B, but rc is sparse (only the user's rated
          items are nonzero). We sum ONLY those ~10-30 rows of ease_B instead of the full dense
          (9352 x 9352) matmul -> ~6x faster per user, MATHEMATICALLY IDENTICAL (zeros add nothing).
      (2) CONTENT-keyed persistent cache: (answerer_sha, uid, known_hash) -> table, sharded to disk
          and reused ACROSS epochs AND runs. Independent of any run-config hash (the per-run-config
          keying is exactly what forced the full regen). Tables are computed only for users a reveal
          or gate actually touches (selection uses precomputed population orders, not answerer calls).
    Verified lazy == eager (see verify_lazy_eager)."""

    def __init__(self, verbose=True):
        super().__init__(verbose=verbose)
        self.answerer_sha = self.models_sha
        self._shard_dir = os.path.join(ANS_CACHE_DIR, self.answerer_sha)
        os.makedirs(self._shard_dir, exist_ok=True)
        self._shards = {}                      # sid -> {(uid,known_hash): table}
        self._shard_dirty = set()

    def _shard_path(self, sid):
        return os.path.join(self._shard_dir, f"shard{sid}.pkl")

    def _load_shard(self, sid):
        if sid not in self._shards:
            import pickle
            p = self._shard_path(sid); d = {}
            if os.path.exists(p):
                try:
                    d = pickle.load(open(p, "rb"))
                except Exception:
                    d = {}
            self._shards[sid] = d
        return self._shards[sid]

    def flush_cache(self):
        import pickle
        for sid in list(self._shard_dirty):
            try:
                tmp = self._shard_path(sid) + ".tmp"
                pickle.dump(self._shards[sid], open(tmp, "wb"), protocol=4)
                os.replace(tmp, self._shard_path(sid))
            except Exception as e:
                print(f"[lazy] shard {sid} flush failed: {e}", flush=True)
        self._shard_dirty.clear()

    def user_table(self, uid, known):
        if uid in self.user_tables:
            return self.user_tables[uid]
        kh = int(AC.Arena._known_hash(known))
        sid = int(uid) % N_SHARD
        shard = self._load_shard(sid)
        key = (int(uid), kh)
        t = shard.get(key)
        if t is None:
            t = self._gen_user(uid, known)
            shard[key] = t; self._shard_dirty.add(sid)
        self.user_tables[uid] = t
        return t

    def _gen_user(self, uid, known):
        """IDENTICAL to arena_core.Arena._gen_user except the EASE fold-in is computed SPARSELY.
        Knowledge (which does not use EASE) is bit-identical; only value bins depend on EASE and
        float ordering never flips an ordinal bin (verify_lazy_eager checks this)."""
        import dans_stages as DS
        import dans_build as DB
        u = self.uni
        f = u.user_features(known)
        rng = np.random.default_rng(AC.SEED * 7_777 + int(uid))
        Pk = DS.know_probs(u, f, self.models, rng=rng)
        know = np.concatenate([DS._sample_cat(Pk["concept"], rng),
                               DS._sample_cat(Pk["entity"], rng),
                               DS._sample_cat(Pk["item"], rng)]).astype(np.int8)
        cmean = float(f["cmean"])
        # ---- SPARSE EASE fold-in (only the user's rated columns contribute) ----
        nz_idx, nz_val = [], []
        for j, r in known.items():
            c = self.ease_index.get(int(j))
            if c is not None:
                nz_idx.append(c); nz_val.append(r - self.ease_mu[c])
        if nz_idx:
            delta = np.asarray(nz_val, np.float64) @ self.ease_B[nz_idx, :]   # (k,)@(k,9352)
        else:
            delta = 0.0
        pred = np.clip(self.ease_mu + delta, 0.5, 5.0)
        t_full = np.full(u.ni, np.nan)
        t_full[self.ease_uni] = pred
        known_ind = np.zeros(u.ni, np.float64)
        for j, r in known.items():
            if 0 <= int(j) < u.ni:
                t_full[int(j)] = r; known_ind[int(j)] = 1.0
        tm = np.isfinite(t_full).astype(np.float64)
        tv = np.where(tm > 0, t_full, 0.0)
        wt = u.pr * tm
        mv = self.models["value"]
        num_c = np.asarray(u.tagM.dot(wt * tv)).ravel()
        den_c = np.asarray(u.tagM.dot(wt)).ravel()
        ctaste = np.where(den_c > 1e-9, num_c / np.maximum(den_c, 1e-9), cmean)
        nr_c = np.asarray(u.tagM.dot(known_ind)).ravel()
        Xc = np.column_stack([ctaste - 3.5, (nr_c > 0).astype(float), np.log1p(nr_c),
                              np.full(self.ntag, cmean)])
        Pv_c = DB.ord_prob(mv["concept"]["theta"], DB.zscale(Xc, mv["concept"]["mu"],
                           mv["concept"]["sd"]), 4)
        num_e = np.asarray(u.entM.dot(wt * tv)).ravel()
        den_e = np.asarray(u.entM.dot(wt)).ravel()
        etaste = np.where(den_e > 1e-9, num_e / np.maximum(den_e, 1e-9), cmean)
        nr_e = np.asarray(u.entM.dot(known_ind)).ravel()
        Xe = np.column_stack([etaste - 3.5, (nr_e > 0).astype(float), np.log1p(nr_e),
                              np.full(self.nent, cmean)])
        Pv_e = DB.ord_prob(mv["entity"]["theta"], DB.zscale(Xe, mv["entity"]["mu"],
                           mv["entity"]["sd"]), 4)
        t_bank = t_full[u.bank]
        t_bank = np.where(np.isfinite(t_bank), t_bank, cmean)
        Xi = np.column_stack([t_bank - 3.5, f["item_val"][:, 1], f["item_val"][:, 0],
                              np.full(self.nbank, cmean)])
        Pv_i = DB.ord_prob(mv["item"]["theta"], DB.zscale(Xi, mv["item"]["mu"], mv["item"]["sd"]), 4)
        val = np.full(self.nQ, -1, np.int8)
        for off, P in ((0, Pv_c), (self.off_ent, Pv_e), (self.off_item, Pv_i)):
            seg = know[off:off + P.shape[0]]
            nz = seg > 0
            if nz.any():
                dv = DS._sample_cat(P[nz], rng)
                block = np.full(P.shape[0], -1, np.int8); block[nz] = dv
                val[off:off + P.shape[0]] = block
        mu_known = float(np.mean(list(known.values())))
        crval = np.full(self.nbank, np.nan, np.float32)
        rated = f["rated_flag"] > 0.5
        if rated.any():
            know[self.off_item:][rated] = 2
            for r in np.where(rated)[0]:
                star = known[int(u.bank[r])]
                val[self.off_item + r] = 3 if star >= 4.5 else 2 if star >= 3.5 else \
                    1 if star >= 2.5 else 0
                crval[r] = np.float32(star - mu_known)
        V = max(len(known), 1)
        n_E = np.concatenate([np.asarray(u.tagM.dot(known_ind)).ravel(),
                              np.asarray(u.entM.dot(known_ind)).ravel(),
                              known_ind[u.bank]])
        p_E = self.q_popmass / self.totalpop
        surp = np.log((n_E + 0.5) / (V * p_E + 0.5)).astype(np.float32)
        return dict(know=know, val=val, crval=crval, surp=surp)


def verify_lazy_eager(n_users=200, seed=0):
    """CORRECTNESS: for n_users sampled users, compute the FULL answer table via the LAZY sparse-EASE
    path AND the ORIGINAL dense-EASE eager path; assert every (user, question) answer is IDENTICAL
    (know/val/surp exact; crval exact incl NaN pattern). A wrong cache would leak/alter training data,
    so this runs BEFORE trusting the fold."""
    print(f"[verify] lazy(sparse-EASE) vs eager(dense-EASE) on {n_users} users ...", flush=True)
    ar = LazyArena(verbose=False)
    coh = AC.make_cohorts(ar, n_train=max(n_users + 50, 300), n_devval=10, n_devtest=10)
    users = coh["train"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(users), size=min(n_users, len(users)), replace=False)
    n_cells = 0; mism_know = 0; mism_val = 0; mism_crval = 0
    for c, i in enumerate(idx):
        u = users[int(i)]; known = u["known"]
        t_lazy = ar._gen_user(u["u"], known)                       # sparse (override)
        t_eager = AC.Arena._gen_user(ar, u["u"], known)            # dense (base class)
        n_cells += ar.nQ
        mism_know += int((t_lazy["know"] != t_eager["know"]).sum())
        mism_val += int((t_lazy["val"] != t_eager["val"]).sum())
        cl, ce = t_lazy["crval"], t_eager["crval"]
        nan_ok = np.array_equal(np.isnan(cl), np.isnan(ce))
        fin = ~np.isnan(cl)
        mism_crval += (0 if nan_ok else 1) + int((cl[fin] != ce[fin]).sum())
    ok = bool(mism_know == 0 and mism_val == 0 and mism_crval == 0)
    print(f"[verify] users={len(idx)}  cells={n_cells}  mismatch know={mism_know} val={mism_val} "
          f"crval={mism_crval}  -> {'IDENTICAL (PASS)' if ok else 'MISMATCH (FAIL)'}", flush=True)
    return dict(n_users=len(idx), n_cells=n_cells, mism_know=mism_know, mism_val=mism_val,
                mism_crval=mism_crval, identical=ok)


# =============================================================== the recovered fold
class FoldRecovered(nn.Module):
    """z = prior + w * delta.
    Per-token input x = [type_oh(4), kind_oh(2), lvl_oh(3)*impl, log_nE, log_V, log_pE,
                         fid_oh(3)*expl, value, emb(d), value*emb(d)].
    ATTENTION pool: alpha = softmax(att(phi(x))) over valid tokens (sum to 1). pool = sum alpha*phi.
    CONFIDENCE gate w = sigmoid(wnet([agg_conf, agreement])), agg_conf = sum alpha*conf (content only,
    NO count), agreement = ||pool|| / sum(alpha*||phi||) in [0,1] (coherence). w, pool = 0 for empty.
    delta = rho(pool). Zero-init rho head so z starts at the prior."""

    PHI_IN = TYPE_ATTR + 2  # placeholder, set below
    def __init__(self, d=D_LAT):
        super().__init__()
        ntype = 4
        phi_in = ntype + 2 + 3 + 3 + 3 + 1 + d + d      # 4+2+3 oh | 3 raw logs | 3 fid | value | emb | v*emb
        conf_in = ntype + 2 + 3 + 3 + 1 + 1 + 1         # type,kind,lvl,fid oh | value | log_nE | log_pE
        self.phi = nn.Sequential(nn.Linear(phi_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.att = nn.Linear(d, 1)                       # attention score per token
        self.conf = nn.Sequential(nn.Linear(conf_in, d // 4), nn.ReLU(), nn.Linear(d // 4, 1))
        self.wnet = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.rho = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)   # residual: start prior
        self.d = d; self.ntype = ntype

    def forward(self, tt, tk, tl, lnE, lV, lpE, tf, tv, te, mask, prior_z, impl_ablate=False):
        B, Kk, d = te.shape
        is_impl = (tk == KIND_IMPL).to(te.dtype)
        is_expl = 1.0 - is_impl
        type_oh = F.one_hot(tt.clamp(min=0), self.ntype).to(te.dtype)
        kind_oh = F.one_hot(tk.clamp(min=0), 2).to(te.dtype)
        lvl_oh = F.one_hot(tl.clamp(min=0), 3).to(te.dtype) * is_impl.unsqueeze(-1)
        fid_oh = F.one_hot(tf.clamp(min=0), 3).to(te.dtype) * is_expl.unsqueeze(-1)
        value = tv.unsqueeze(-1)
        logs = torch.stack([lnE, lV, lpE], dim=-1)       # (B,K,3)
        ve = value * te
        x = torch.cat([type_oh, kind_oh, lvl_oh, logs, fid_oh, value, te, ve], dim=-1)
        eff_mask = mask * (is_expl if impl_ablate else torch.ones_like(mask))   # (B,K)
        h = self.phi(x)                                  # (B,K,d)
        # --- attention pool (softmax weights sum to 1 = count-normalized) ---
        att_logit = self.att(h).squeeze(-1)              # (B,K)
        att_logit = att_logit.masked_fill(eff_mask == 0, -1e9)
        alpha = torch.softmax(att_logit, dim=1)          # (B,K)
        has = (eff_mask.sum(1, keepdim=True) > 0).to(te.dtype)   # (B,1)
        pool = (alpha.unsqueeze(-1) * h).sum(1) * has     # (B,d); exact 0 for empty
        # --- CONTENT-confidence gate (no count) ---
        xc = torch.cat([type_oh, kind_oh, lvl_oh, fid_oh, value,
                        lnE.unsqueeze(-1), lpE.unsqueeze(-1)], dim=-1)
        conf = torch.sigmoid(self.conf(xc)).squeeze(-1)  # (B,K) per-token reliability
        agg_conf = (alpha * conf).sum(1)                 # (B,) attention-weighted mean reliability
        hn = h.norm(dim=-1)                              # (B,K)
        agree = pool.norm(dim=-1) / ((alpha * hn).sum(1) + 1e-6)   # (B,) coherence in [0,1]
        w = torch.sigmoid(self.wnet(torch.stack([agg_conf, agree], dim=-1)).squeeze(-1))  # (B,)
        w = w * has.squeeze(-1)                          # empty -> w == 0 -> z == prior exactly
        delta = self.rho(pool)
        return prior_z + w.unsqueeze(-1) * delta


# =============================================================== packing / folding (9-field tokens)
def pack_batch(FR, tok_lists, prior_vec):
    B = len(tok_lists)
    Kk = max((len(t) for t in tok_lists), default=1); Kk = max(Kk, 1)
    d = FR.W.shape[1]
    tt = torch.zeros((B, Kk), dtype=torch.long); tk = torch.zeros((B, Kk), dtype=torch.long)
    tl = torch.zeros((B, Kk), dtype=torch.long); tf = torch.zeros((B, Kk), dtype=torch.long)
    lnE = torch.zeros((B, Kk), dtype=torch.float32); lV = torch.zeros((B, Kk), dtype=torch.float32)
    lpE = torch.zeros((B, Kk), dtype=torch.float32); tv = torch.zeros((B, Kk), dtype=torch.float32)
    te = torch.zeros((B, Kk, d), dtype=torch.float32); mask = torch.zeros((B, Kk), dtype=torch.float32)
    for b, toks in enumerate(tok_lists):
        for k, tok in enumerate(toks):
            typ, kind, lvl, l_n, l_v, l_p, fid, val, emb = tok
            tt[b, k] = typ; tk[b, k] = kind; tl[b, k] = lvl; tf[b, k] = fid
            lnE[b, k] = l_n; lV[b, k] = l_v; lpE[b, k] = l_p; tv[b, k] = val
            te[b, k] = torch.as_tensor(emb); mask[b, k] = 1.0
    prior = prior_vec.unsqueeze(0).expand(B, -1).contiguous()
    return tt, tk, tl, lnE, lV, lpE, tf, tv, te, mask, prior


def micro_groups(tok_lists, budget=MICRO_TOK):
    """Length-bucket indices into groups so the padded tensor (n_users x max_len) stays <= budget
    tokens. NO tokens dropped -- prolific users (thousands of tokens) simply land in small groups
    (down to a group of one). Bounds memory without any cap/subsample."""
    idx = sorted(range(len(tok_lists)), key=lambda i: len(tok_lists[i]))
    groups, cur, cmax = [], [], 0
    for i in idx:
        li = max(len(tok_lists[i]), 1); nmax = max(cmax, li)
        if cur and (len(cur) + 1) * nmax > budget:
            groups.append(cur); cur = [i]; cmax = li
        else:
            cur.append(i); cmax = nmax
    if cur:
        groups.append(cur)
    return groups


def fold_batch(FR, model, tok_lists, prior_vec, impl_ablate=False):
    if not tok_lists:
        return np.zeros((0, FR.W.shape[1]))
    out = np.zeros((len(tok_lists), FR.W.shape[1]))
    for g in micro_groups(tok_lists):                       # length-bucketed (memory-safe, no caps)
        args = pack_batch(FR, [tok_lists[i] for i in g], prior_vec)
        with torch.no_grad():
            z = model(*args, impl_ablate=impl_ablate)
        zz = z.numpy().astype(np.float64)
        for k, i in enumerate(g):
            out[i] = zz[k]
    return out


def fold_np(FR, model, toks, prior_vec, impl_ablate=False):
    return fold_batch(FR, model, [toks], prior_vec, impl_ablate=impl_ablate)[0].astype(np.float64)


# =============================================================== recovered curriculum generator
class RecoveredGen:
    """Reuses DeOODGen SELECTION (arena 2,428-q universe, realistic strategies through the real gated
    v2.1 answerer). Emits RECOVERED tokens carrying raw log(n_E)/log(V)/log(p_E) (no surprise ratio)."""

    def __init__(self, ar, train_sub, verbose=True):
        self.ar = ar
        self.deo = V4.DeOODGen(ar, train_sub, verbose=verbose)
        self.totalpop = float(ar.totalpop)
        self.cnt = ar.D["cnt"].astype(np.float64)
        self._ctx_cache = {}

    def _ctx(self, r):
        ctx = self.ar.user_ctx(r)                  # LAZY: computes/loads this user's answer table
        ctx["nE"] = self.deo._nE(r["known"])       # per-question n_E vector (concept|entity|item)
        ctx["V"] = max(len(r["known"]), 1)
        return ctx

    def ctx(self, r):
        """Lazy per-user context (answer table + n_E), cached across epochs/gates."""
        uid = r["u"]
        c = self._ctx_cache.get(uid)
        if c is None:
            c = self._ctx(r); self._ctx_cache[uid] = c
        return c

    def tokens_for(self, uid, qidx, ctx):
        ar = self.ar; t = ctx["table"]; k = int(t["know"][qidx]); ch = int(ar.q_channel[qidx])
        emb = ar.Qemb[qidx]
        n_E = float(ctx["nE"][qidx]); V = float(ctx["V"]); p_E = float(ar.q_popmass[qidx] / self.totalpop)
        l_n = float(np.log(n_E + 0.5)); l_v = float(np.log(V + 0.5)); l_p = float(np.log(p_E + 1e-9))
        if k == 0:                                 # refusal: low-reliability implicit-negative token
            return [(ch, KIND_IMPL, LVL_NEG, l_n, l_v, l_p, FID_DATA, 0.0, emb)]
        lvl = LVL_KW if k == 2 else LVL_ROUGH
        toks = [(ch, KIND_IMPL, lvl, l_n, l_v, l_p, FID_DATA, 0.0, emb)]
        if qidx >= ar.off_item and np.isfinite(t["crval"][qidx - ar.off_item]):
            toks.append((ch, KIND_EXPL, LVL_ROUGH, l_n, l_v, l_p, FID_DATA,
                         float(t["crval"][qidx - ar.off_item]), emb))
        else:
            v = int(t["val"][qidx])
            if v >= 0:
                fid = FID_EASE if k == 2 else FID_LLM
                toks.append((ch, KIND_EXPL, LVL_ROUGH, l_n, l_v, l_p, fid, float(VBIN[v]), emb))
        return toks

    def interview_tokens(self, r, qs, ctx):
        toks = []
        for qi in qs:
            toks += self.tokens_for(r["u"], int(qi), ctx)
        return toks

    def clean_qs(self, r, ctx):
        """CLEAN full-profile question set = EVERY question the user is answerable on = ALL arena qi
        with n_E>0 (their rated bank items + ALL implied concepts/attributes/entities). NO CAPS, NO
        subsampling: we paid to label every question, so all answerable tokens are used (capping both
        discards paid-for information AND biases which channel survives). Featurized by the SAME
        tokens_for as interviews. Memory stays bounded via length-bucketed micro-batching, not caps."""
        nE = ctx["nE"]
        return [int(q) for q in np.where(nE > 0)[0]]

    def clean_reveal(self, r, ctx):
        return self.interview_tokens(r, self.clean_qs(r, ctx), ctx)

    def select(self, r, strat, rng, B):
        return self.deo.select(r, strat, rng, B=B)

    def build_reveal(self, r, ctx, rng, clean_frac=CLEAN_FRAC):
        if rng.random() < clean_frac:
            return self.clean_reveal(r, ctx)
        strat = rng.choice(SEEN, p=[STRAT_W[s] for s in SEEN])
        B = int(rng.integers(1, 25))
        return self.interview_tokens(r, self.select(r, strat, rng, B), ctx)


# =============================================================== training
def _loss_sum(FR, model, tl, tgt, prof, prior_vec, ipw_t):
    """SUM (not mean) of per-user IPS-weighted multinomial neg-LL for a micro-batch."""
    args = pack_batch(FR, tl, prior_vec)
    z = model(*args)
    Smat = z @ FR.W.T + FR.bdec
    B, ni = Smat.shape
    negmask = torch.zeros((B, ni), dtype=torch.bool)
    tgw = torch.zeros((B, ni), dtype=torch.float32)
    for b in range(B):
        negmask[b, list(prof[b])] = True
        idx = list(tgt[b])
        tgw[b, idx] = ipw_t[idx]                          # IPS / inverse-pop target weights
    Smat = Smat.masked_fill(negmask, -1e9)
    logp = torch.log_softmax(Smat, dim=1)
    denom = tgw.sum(1).clamp(min=1e-6)
    return -((logp * tgw).sum(1) / denom).sum()


def batch_loss(FR, model, gen, users, prior_vec, ipw_t, rng, opt):
    """Build every user's FULL reveal (no caps), then LENGTH-BUCKET into micro-batches and accumulate
    gradients so a few prolific (thousands-of-tokens) users can't blow up a padded tensor. Loss is the
    mean per-user neg-LL over the whole logical batch. Returns the float loss (grads left on model;
    caller clips + steps). Returns None (grads zeroed) on a non-finite micro-loss."""
    tl, tgt, prof = [], [], []
    for u in users:
        toks = gen.build_reveal(u, gen.ctx(u), rng)
        if not toks:
            continue
        tl.append(toks); tgt.append(u["held"]); prof.append(set(u["known"].keys()))
    if not tl:
        return None
    N = len(tl)
    opt.zero_grad()
    total = 0.0
    for g in micro_groups(tl):
        sub_tl = [tl[i] for i in g]; sub_tgt = [tgt[i] for i in g]; sub_prof = [prof[i] for i in g]
        lsum = _loss_sum(FR, model, sub_tl, sub_tgt, sub_prof, prior_vec, ipw_t)
        if not torch.isfinite(lsum):
            opt.zero_grad(); return None                  # never poison weights with NaN/inf
        (lsum / N).backward()                             # accumulate; mean over the full batch
        total += float(lsum.item())
    return total / N


@torch.no_grad()
def _fold_ndcg(ar, model, tok_lists, held, prof, prior_vec, kk=K):
    if not tok_lists:
        return 0.0
    Z = fold_batch(ar.FR, model, tok_lists, prior_vec)
    vv = AC.ndcg_at_k_batch(ar.FR, Z, held, prof, kk)
    vv = [v for v in vv if v is not None]
    return float(np.mean(vv)) if vv else 0.0


@torch.no_grad()
def val_ndcg(ar, model, gen, val_users, prior_vec, seed):
    """Deterministic val: fixed 30% clean + rotated SEEN strategies at varied length."""
    model.eval()
    rng = np.random.default_rng(seed)
    tl, held, prof = [], [], []
    for i, u in enumerate(val_users):
        if rng.random() < CLEAN_FRAC:
            toks = gen.clean_reveal(u, gen.ctx(u))
        else:
            strat = SEEN[i % len(SEEN)]; B = int(rng.integers(1, 25))
            toks = gen.interview_tokens(u, gen.select(u, strat, rng, B), gen.ctx(u))
        if toks:
            tl.append(toks); held.append(u["held"]); prof.append(set(u["known"].keys()))
    return _fold_ndcg(ar, model, tl, held, prof, prior_vec)


def train(ar, gen, tr_users, val_users, prior_vec, args, tag="full"):
    t0 = time.time()
    ipw = 1.0 / np.sqrt(ar.D["cnt"].astype(np.float64) + 1.0)
    ipw_t = torch.as_tensor(ipw, dtype=torch.float32)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = FoldRecovered()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    step_rng = np.random.default_rng(args.seed + 909)
    state = dict(epoch=0, best_val=-1.0, best_epoch=-1, history=[], clean_frac=CLEAN_FRAC,
                 strategies_seen=SEEN, strategies_heldout=[HELDOUT], tag=tag)
    print(f"[rec:{tag}] === training FoldRecovered ({int(CLEAN_FRAC*100)}% clean, lengths 1..24), "
          f"epochs={args.epochs}, {len(tr_users)} train / {len(val_users)} val ===", flush=True)
    for ep in range(args.epochs):
        model.train()
        order = np.arange(len(tr_users)); step_rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), args.batch):
            us = [tr_users[i] for i in order[b0:b0 + args.batch]]
            loss = batch_loss(ar.FR, model, gen, us, prior_vec, ipw_t, step_rng, opt)
            if loss is None or not np.isfinite(loss):     # batch_loss already accumulated grads
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        vN = val_ndcg(ar, model, gen, val_users, prior_vec, args.seed + 7)
        if isinstance(ar, LazyArena):
            ar.flush_cache()                          # persist newly-computed tables each epoch
        state["epoch"] = ep + 1
        rec = dict(epoch=ep + 1, train_loss=round(tot / max(nb, 1), 4), val_ndcg=round(vN, 4),
                   min=round((time.time() - t0) / 60, 1))
        state["history"].append(rec)
        print(f"[rec:{tag}] ep {ep+1:2d} | loss {rec['train_loss']:.4f} | val NDCG@{K} {vN:.4f} | "
              f"{rec['min']}m", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state), CKPT)
        if vN > state["best_val"]:
            state["best_val"] = vN; state["best_epoch"] = ep + 1
            torch.save(dict(model=model.state_dict(), state=state), BEST)
        json.dump(state["history"], open(LOGP, "w"), indent=1)
    print(f"[rec:{tag}] BEST val {state['best_val']:.4f} @ep{state['best_epoch']} -> {BEST} "
          f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    return state


def load_best(FR):
    blob = torch.load(BEST, map_location="cpu")
    m = FoldRecovered(); m.load_state_dict(blob["model"]); m.eval()
    return m, blob.get("state", {})


# =============================================================== gate helpers
def _boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _region_score(FR, z, members):
    S = FR.decode_np(z[None, :])[0]
    return float(np.mean(S[members]))


def _top_q(ar, nE, lo, hi):
    seg = nE[lo:hi]
    j = int(np.argmax(seg))
    return (lo + j) if seg[j] > 0 else None


def _tok_logs(ar, gen, n_E, V, qidx):
    p_E = float(ar.q_popmass[qidx] / gen.totalpop)
    return float(np.log(n_E + 0.5)), float(np.log(V + 0.5)), float(np.log(p_E + 1e-9))


# =============================================================== the full gate suite (on TEST)
def _print_thresholds():
    print("\n=== PRE-REGISTERED GATE THRESHOLDS (printed before results; all on TEST users) ===", flush=True)
    print("  G-intercept : fold([]) == prior (enc_items([])) to 1e-5 AND cold NDCG == native cold NDCG.", flush=True)
    print("  G-falsify   : duplicate answers x2/x3 -> NDCG delta ~0 (|d|<0.002); uninformative padding FLAT.", flush=True)
    print("  G-generalize: HELD-OUT blind_eig NDCG >= mean(SEEN strategies) - 0.02.", flush=True)
    print("  G-order     : shuffle-invariance, |NDCG(shuffled)-NDCG(orig)| < 1e-4.", flush=True)
    print("  G-noharm    : adding ONE real answer to a reveal never hurts; mean delta >= -0.003 (CI).", flush=True)
    print("  G-noQ1drop  : turn-1 NDCG >= cold; per-turn max decline > -0.003.", flush=True)
    print("  G-clean     : clean full-profile fold >= native RecVAE - 0.02.", flush=True)
    print("  G-caplength : elicitation (B=8) NDCG >= cold prior baseline.", flush=True)
    print("  G-canaries  : 1 answer per channel x {implicit,explicit} from cold has lift > 0.", flush=True)
    print("  G-GoT       : watched-all-X-rated-badly pulled toward X vs never-heard. CI excl 0.", flush=True)
    print("  G-prolific  : selective-all-X pulled toward X > prolific-all-X-plus-everything. CI excl 0.", flush=True)
    print("  G-implicit  : zeroing implicit tokens DROPS NDCG. CI excl 0.", flush=True)
    print("  G-firewall  : cos(z*,q)/recommender-geometry answers ABSENT from train+eval.", flush=True)


def gate_intercept(ar, model, gen, test_users, prior_vec):
    z_empty = fold_np(ar.FR, model, [], prior_vec)
    prior_np = prior_vec.numpy().astype(np.float64)
    max_dev = float(np.max(np.abs(z_empty - prior_np)))
    # cold NDCG (fold empty) vs native cold NDCG (enc_items([]))
    held = [u["held"] for u in test_users]; prof = [set(u["known"].keys()) for u in test_users]
    cold_fold = _fold_ndcg(ar, model, [[]] * len(test_users), held, prof, prior_vec)
    Zn = ar.FR.enc_items([[]]).numpy().astype(np.float64)
    vv = AC.ndcg_at_k_batch(ar.FR, np.repeat(Zn, len(test_users), 0), held, prof, K)
    native_cold = float(np.mean([v for v in vv if v is not None]))
    passed = bool(max_dev < 1e-5 and abs(cold_fold - native_cold) < 1e-6)
    print(f"\n[G-intercept] max|fold([])-prior|={max_dev:.2e}  cold_fold={cold_fold:.4f}  "
          f"native_cold={native_cold:.4f}  -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(max_dev=max_dev, cold_fold=cold_fold, native_cold=native_cold, **{"pass": passed})


def gate_falsify(ar, model, gen, test_users, prior_vec):
    rng = np.random.default_rng(31)
    d_dup, d_pad = [], []
    for u in test_users:
        ctx = gen.ctx(u); held = u["held"]; prof = set(u["known"].keys())
        toks = gen.interview_tokens(u, gen.select(u, "on_profile", rng, 6), ctx)
        if not toks:
            continue
        base = AC.ndcg_at_k(ar.FR, fold_np(ar.FR, model, toks, prior_vec), held, prof, K)
        dup = AC.ndcg_at_k(ar.FR, fold_np(ar.FR, model, toks * 3, prior_vec), held, prof, K)
        # uninformative padding: add vague low-fidelity concept tokens with no consumption (n_E=0)
        pad = list(toks)
        for _ in range(8):
            qi = int(rng.integers(0, ar.off_ent))       # a concept q
            l_n, l_v, l_p = _tok_logs(ar, gen, 0.0, ctx["V"], qi)
            pad.append((TYPE_CONCEPT, KIND_IMPL, LVL_ROUGH, l_n, l_v, l_p, FID_DATA, 0.0, ar.Qemb[qi]))
        padn = AC.ndcg_at_k(ar.FR, fold_np(ar.FR, model, pad, prior_vec), held, prof, K)
        if None not in (base, dup, padn):
            d_dup.append(dup - base); d_pad.append(padn - base)
    md, mp = float(np.mean(d_dup)), float(np.mean(d_pad))
    passed = bool(abs(md) < 0.002 and abs(mp) < 0.01)
    print(f"[G-falsify]   duplicate x3 delta {md:+.5f} (|d|<0.002); padding delta {mp:+.5f} (|d|<0.01) "
          f"n={len(d_dup)} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(dup_delta=md, pad_delta=mp, n=len(d_dup), **{"pass": passed})


def gate_generalize(ar, model, gen, test_users, prior_vec, cold0):
    out = {}
    for strat in SEEN + [HELDOUT]:
        rng = np.random.default_rng(abs(hash(strat)) % (2 ** 31))
        tl, held, prof = [], [], []
        for u in test_users:
            toks = gen.interview_tokens(u, gen.select(u, strat, rng, 8), gen.ctx(u))
            if toks:
                tl.append(toks); held.append(u["held"]); prof.append(set(u["known"].keys()))
        nd = _fold_ndcg(ar, model, tl, held, prof, prior_vec)
        out[strat] = dict(ndcg=nd, lift=nd - cold0)
    mean_seen = float(np.mean([out[s]["ndcg"] for s in SEEN]))
    eig = out[HELDOUT]["ndcg"]
    passed = bool(eig >= mean_seen - 0.02)
    print(f"[G-generalize] mean(SEEN)={mean_seen:.4f}  held-out {HELDOUT}={eig:.4f}  "
          f"-> {'PASS' if passed else 'FAIL'}", flush=True)
    for s in SEEN + [HELDOUT]:
        tagh = "  (HELD OUT)" if s == HELDOUT else ""
        print(f"      {s:12s} NDCG {out[s]['ndcg']:.4f}  lift {out[s]['lift']:+.4f}{tagh}", flush=True)
    return dict(per_strategy=out, mean_seen=mean_seen, heldout=eig, tol=0.02, **{"pass": passed})


def gate_order(ar, model, gen, test_users, prior_vec):
    rng = np.random.default_rng(42); devs = []
    for u in test_users:
        toks = gen.interview_tokens(u, gen.select(u, "mixed", rng, 10), gen.ctx(u))
        if len(toks) < 2:
            continue
        z1 = fold_np(ar.FR, model, toks, prior_vec)
        sh = list(toks); rng.shuffle(sh)
        z2 = fold_np(ar.FR, model, sh, prior_vec)
        devs.append(float(np.max(np.abs(z1 - z2))))
    md = float(np.max(devs)) if devs else 0.0
    passed = bool(md < 1e-4)
    print(f"[G-order]     max|z(orig)-z(shuffled)| = {md:.2e}  n={len(devs)} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(max_dev=md, n=len(devs), **{"pass": passed})


def gate_noharm(ar, model, gen, test_users, prior_vec):
    """Adding ONE more real (answered) question to a partial reveal must not hurt (Bayesian shrinkage)."""
    rng = np.random.default_rng(5); deltas = []
    for u in test_users:
        ctx = gen.ctx(u); held = u["held"]; prof = set(u["known"].keys())
        qs = gen.select(u, "on_profile", rng, 8)
        # find an answered question to add as the (k+1)-th real answer
        answered = [int(q) for q in qs if int(ctx["table"]["know"][int(q)]) >= 1]
        if len(answered) < 2:
            continue
        base_qs = answered[:-1]; add_q = answered[-1]
        z_base = fold_np(ar.FR, model, gen.interview_tokens(u, base_qs, ctx), prior_vec)
        z_more = fold_np(ar.FR, model, gen.interview_tokens(u, base_qs + [add_q], ctx), prior_vec)
        a = AC.ndcg_at_k(ar.FR, z_base, held, prof, K); b = AC.ndcg_at_k(ar.FR, z_more, held, prof, K)
        if None not in (a, b):
            deltas.append(b - a)
    ci = _boot_ci(deltas); m = float(np.mean(deltas))
    passed = bool(ci[0] > -0.003)
    print(f"[G-noharm]    add-one-real NDCG delta {m:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] n={len(deltas)} "
          f"-> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=m, ci=ci, n=len(deltas), **{"pass": passed})


def gate_perturn(ar, model, gen, test_users, prior_vec, maxB=12):
    """G-noQ1drop + per-turn monotone: on-profile prefix curve; turn1>=cold, max decline>-0.003."""
    rng = np.random.default_rng(77)
    per_user = []
    for u in test_users:
        ctx = gen.ctx(u); held = u["held"]; prof = set(u["known"].keys())
        qs = gen.select(u, "on_profile", rng, maxB)
        curve = []
        cold = AC.ndcg_at_k(ar.FR, fold_np(ar.FR, model, [], prior_vec), held, prof, K)
        curve.append(cold)
        for t in range(1, maxB + 1):
            toks = gen.interview_tokens(u, qs[:t], ctx)
            curve.append(AC.ndcg_at_k(ar.FR, fold_np(ar.FR, model, toks, prior_vec), held, prof, K))
        if all(c is not None for c in curve):
            per_user.append(curve)
    arr = np.array(per_user)                         # (n, maxB+1)
    curve = arr.mean(0)
    q1_drop = float(curve[1] - curve[0])
    steps = np.diff(curve)
    max_decline = float(steps.min())
    passed = bool(q1_drop >= -0.003 and max_decline > -0.003)
    print(f"[G-noQ1drop]  cold {curve[0]:.4f} -> turn1 {curve[1]:.4f} (drop {q1_drop:+.4f}); "
          f"max per-step decline {max_decline:+.4f} -> {'PASS' if passed else 'FAIL'}", flush=True)
    print("      curve: " + " ".join(f"{c:.3f}" for c in curve), flush=True)
    return dict(curve=[float(c) for c in curve], q1_drop=q1_drop, max_step_decline=max_decline,
                n=len(per_user), **{"pass": passed})


def gate_clean(ar, model, gen, test_users, prior_vec):
    rng = np.random.default_rng(41)
    tl, held, prof, natez = [], [], [], []
    for u in test_users:
        tl.append(gen.clean_reveal(u, gen.ctx(u))); held.append(u["held"])
        prof.append(set(u["known"].keys())); natez.append(list(u["known"].keys()))
    fold = _fold_ndcg(ar, model, tl, held, prof, prior_vec)
    vals = []
    for s in range(0, len(natez), 1500):
        e = min(s + 1500, len(natez))
        Zn = ar.FR.enc_items(natez[s:e]).numpy().astype(np.float64)
        vv = AC.ndcg_at_k_batch(ar.FR, Zn, held[s:e], prof[s:e], K)
        vals += [v for v in vv if v is not None]
    nat = float(np.mean(vals)) if vals else 0.0
    passed = bool(fold >= nat - 0.02)
    print(f"[G-clean]     fold(full) {fold:.4f}  native RecVAE {nat:.4f}  gap {fold-nat:+.4f}  "
          f"(>= -0.02) -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(fold_ndcg=fold, native_ndcg=nat, gap=fold - nat, n=len(tl), **{"pass": passed})


def gate_caplength(ar, model, gen, test_users, prior_vec, cold0):
    rng = np.random.default_rng(88)
    tl, held, prof = [], [], []
    for u in test_users:
        toks = gen.interview_tokens(u, gen.select(u, "mixed", rng, 8), gen.ctx(u))
        if toks:
            tl.append(toks); held.append(u["held"]); prof.append(set(u["known"].keys()))
    elic = _fold_ndcg(ar, model, tl, held, prof, prior_vec)
    passed = bool(elic >= cold0 - 0.002)
    print(f"[G-caplength] elicitation@8 {elic:.4f} vs cold prior {cold0:.4f}  -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(elic_ndcg=elic, cold=cold0, **{"pass": passed})


def gate_canaries(ar, model, gen, test_users, prior_vec):
    spans = {"item": (ar.off_item, ar.nQ), "concept": (0, ar.off_ent),
             "entity": (ar.off_ent, ar.off_item)}
    out = {}
    for name, (lo, hi) in spans.items():
        li_e, li_i = [], []
        for u in test_users:
            ctx = gen.ctx(u); held = u["held"]; prof = set(u["known"].keys())
            qi = _top_q(ar, ctx["nE"], lo, hi)
            if qi is None or int(ctx["table"]["know"][qi]) < 1:
                continue
            toks = gen.tokens_for(u["u"], qi, ctx)
            z0 = fold_np(ar.FR, model, [], prior_vec)
            n0 = AC.ndcg_at_k(ar.FR, z0, held, prof, K)
            impl = [t for t in toks if t[1] == KIND_IMPL]
            expl = [t for t in toks if t[1] == KIND_EXPL]
            ni = AC.ndcg_at_k(ar.FR, fold_np(ar.FR, model, impl, prior_vec), held, prof, K) if impl else None
            ne = AC.ndcg_at_k(ar.FR, fold_np(ar.FR, model, expl, prior_vec), held, prof, K) if expl else None
            if n0 is not None and ni is not None:
                li_i.append(ni - n0)
            if n0 is not None and ne is not None:
                li_e.append(ne - n0)
        mi = float(np.mean(li_i)) if li_i else float("nan")
        me = float(np.mean(li_e)) if li_e else float("nan")
        ci_i = _boot_ci(li_i); ci_e = _boot_ci(li_e)
        out[name] = dict(implicit_lift=mi, implicit_ci=ci_i, explicit_lift=me, explicit_ci=ci_e,
                         n_impl=len(li_i), n_expl=len(li_e))
        print(f"[G-canaries]  {name:8s} implicit lift {mi:+.4f} CI[{ci_i[0]:+.4f},{ci_i[1]:+.4f}] | "
              f"explicit lift {me:+.4f} CI[{ci_e[0]:+.4f},{ci_e[1]:+.4f}]", flush=True)
    passed = all((out[c]["implicit_lift"] > 0) and
                 (np.isnan(out[c]["explicit_lift"]) or out[c]["explicit_lift"] > 0) for c in out)
    out["pass"] = bool(passed)
    print(f"      canaries all-positive -> {'PASS' if passed else 'FAIL'}", flush=True)
    return out


def gate_got(ar, model, gen, test_users, prior_vec):
    diffs = []
    for u in test_users:
        ctx = gen.ctx(u)
        qi = _top_q(ar, ctx["nE"], ar.off_ent, ar.off_item)
        if qi is None:
            continue
        members = ar.region_members(qi)
        n_E = float(ctx["nE"][qi])
        if n_E < 3 or len(members) < 3:
            continue
        emb = ar.Qemb[qi]
        l_n, l_v, l_p = _tok_logs(ar, gen, n_E, ctx["V"], qi)
        bad = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, l_n, l_v, l_p, FID_DATA, 0.0, emb),
               (TYPE_ENTITY, KIND_EXPL, LVL_ROUGH, l_n, l_v, l_p, FID_EASE, VBIN[0], emb)]
        z_bad = fold_np(ar.FR, model, bad, prior_vec)
        z_never = fold_np(ar.FR, model, [], prior_vec)
        diffs.append(_region_score(ar.FR, z_bad, members) - _region_score(ar.FR, z_never, members))
    ci = _boot_ci(diffs); m = float(np.mean(diffs))
    passed = bool(ci[0] > 0)
    print(f"[G-GoT]       pull(bad-rater)-pull(never) {m:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] n={len(diffs)}"
          f" -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=m, ci=ci, n=len(diffs), **{"pass": passed})


def gate_prolific(ar, model, gen, test_users, prior_vec):
    diffs = []
    for u in test_users:
        ctx = gen.ctx(u)
        qi = _top_q(ar, ctx["nE"], ar.off_ent, ar.off_item)
        if qi is None:
            continue
        members = ar.region_members(qi); n_E = float(ctx["nE"][qi])
        if n_E < 3 or len(members) < 3:
            continue
        emb = ar.Qemb[qi]
        ln_sel, lv_sel, lp = _tok_logs(ar, gen, n_E, n_E, qi)           # selective: V == n_E
        ln_pro, lv_pro, _ = _tok_logs(ar, gen, n_E, n_E + 4000.0, qi)   # prolific: huge V, same n_E
        t_sel = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, ln_sel, lv_sel, lp, FID_DATA, 0.0, emb)]
        t_pro = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, ln_pro, lv_pro, lp, FID_DATA, 0.0, emb)]
        z_sel = fold_np(ar.FR, model, t_sel, prior_vec)
        z_pro = fold_np(ar.FR, model, t_pro, prior_vec)
        diffs.append(_region_score(ar.FR, z_sel, members) - _region_score(ar.FR, z_pro, members))
    ci = _boot_ci(diffs); m = float(np.mean(diffs))
    passed = bool(ci[0] > 0)
    print(f"[G-prolific]  pull(selective)-pull(prolific) {m:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"n={len(diffs)} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=m, ci=ci, n=len(diffs), **{"pass": passed})


def gate_implicit(ar, model, gen, test_users, prior_vec):
    rng = np.random.default_rng(9); drops = []
    for u in test_users:
        ctx = gen.ctx(u); held = u["held"]; prof = set(u["known"].keys())
        toks = gen.interview_tokens(u, gen.select(u, "mixed", rng, int(rng.integers(6, 20))), ctx)
        if not toks:
            continue
        zf = fold_np(ar.FR, model, toks, prior_vec, impl_ablate=False)
        za = fold_np(ar.FR, model, toks, prior_vec, impl_ablate=True)
        a = AC.ndcg_at_k(ar.FR, zf, held, prof, K); b = AC.ndcg_at_k(ar.FR, za, held, prof, K)
        if None not in (a, b):
            drops.append(a - b)
    ci = _boot_ci(drops); m = float(np.mean(drops))
    passed = bool(ci[0] > 0)
    print(f"[G-implicit]  NDCG(full)-NDCG(implicit-zeroed) {m:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"n={len(drops)} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=m, ci=ci, n=len(drops), **{"pass": passed})


def gate_firewall(ar, gen, test_users):
    """DATA-FLOW firewall (not a string scan): verify answers depend ONLY on real ratings U EASE and
    are INVARIANT to the recommender geometry z*.
    (A) BELIEF-INVARIANCE: regenerate a user's answer table with a FRESH RANDOM belief model swapped
        into the arena; if any answer changes, a cos(z*,q) path leaked. Must be byte-identical.
    (B) VALUE PROVENANCE: every EXPLICIT token value must equal its data-side source exactly -- the
        real centered rating (rated bank item) or VBIN[EASE-ordinal] from the answer table -- so no
        value can originate from recommender geometry."""
    # (A) belief-invariance
    r = test_users[0]; uid = r["u"]
    ar.user_tables.pop(uid, None)
    t_base = ar._gen_user(uid, r["known"])
    saved = ar.model
    try:
        ar.model = FoldRecovered()                       # fresh random belief -> different z*
        t_pert = ar._gen_user(uid, r["known"])
    finally:
        ar.model = saved
        ar.user_tables.pop(uid, None)
    invariant = bool(np.array_equal(t_base["know"], t_pert["know"]) and
                     np.array_equal(t_base["val"], t_pert["val"]) and
                     np.array_equal(np.nan_to_num(t_base["crval"], nan=-9.0),
                                    np.nan_to_num(t_pert["crval"], nan=-9.0)))
    # (B) value provenance
    rng = np.random.default_rng(0); bad = 0; checked = 0
    for u in test_users[:200]:
        ctx = gen.ctx(u); t = ctx["table"]
        qs = np.where(t["know"] >= 1)[0]
        if len(qs) == 0:
            continue
        for qi in rng.choice(qs, size=min(5, len(qs)), replace=False):
            qi = int(qi)
            for tok in gen.tokens_for(u["u"], qi, ctx):
                if tok[1] != KIND_EXPL:
                    continue
                val = float(tok[7]); checked += 1
                if qi >= ar.off_item and np.isfinite(t["crval"][qi - ar.off_item]):
                    exp = float(t["crval"][qi - ar.off_item])            # real rating (data)
                else:
                    exp = float(VBIN[int(t["val"][qi])])                 # EASE-ordinal (data)
                if abs(val - exp) > 1e-6:
                    bad += 1
    passed = bool(invariant and bad == 0)
    print(f"[G-firewall]  belief-invariance (answers independent of z*): {invariant}; "
          f"value-provenance real-U-EASE: {bad}/{checked} off-source -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(belief_invariant=invariant, provenance_bad=bad, provenance_checked=checked,
                note="answers = real ratings U EASE ordinal only; regenerating with a random belief "
                "model leaves every answer byte-identical (no cos(z*,q) dependence)", **{"pass": passed})


# =============================================================== orchestration
def build_world(n_train, n_val, n_test, verbose=True):
    print("[rec] building LAZY arena (gated v2.1 world + EASE + 2,428-q universe) ...", flush=True)
    ar = LazyArena(verbose=verbose)
    V4._sanitize_qemb(ar)
    coh = AC.make_cohorts(ar, n_train=n_train, n_devval=n_val, n_devtest=n_test)
    tr, val, test = coh["train"], coh["devval"], coh["devtest"]
    # firewall assert: no study ids leaked into any cohort
    import dans_build as DB
    excl = DB.study_ids()
    for split in (tr, val, test):
        assert all(u["u"] not in excl for u in split), "STUDY ID LEAK into fold cohort"
    print(f"[rec] cohorts: train {len(tr)} / val {len(val)} / TEST {len(test)} (study ids excluded); "
          f"answers computed LAZILY on-touch (sparse EASE) + persistent content-keyed cache "
          f"({ar._shard_dir})", flush=True)
    train_sub = tr[:min(800, len(tr))]
    gen = RecoveredGen(ar, train_sub, verbose=verbose)
    prior_vec = ar.FR.enc_items([[]])[0].detach()            # FIXED cold prior (== 0)
    return ar, gen, tr, val, test, prior_vec


def run_gates(ar, gen, model, test_users, prior_vec):
    _print_thresholds()
    held = [u["held"] for u in test_users]; prof = [set(u["known"].keys()) for u in test_users]
    cold0 = _fold_ndcg(ar, model, [[]] * len(test_users), held, prof, prior_vec)
    R = {}
    R["G_intercept"] = gate_intercept(ar, model, gen, test_users, prior_vec)
    R["G_falsify"] = gate_falsify(ar, model, gen, test_users, prior_vec)
    R["G_generalize"] = gate_generalize(ar, model, gen, test_users, prior_vec, cold0)
    R["G_order"] = gate_order(ar, model, gen, test_users, prior_vec)
    R["G_noharm"] = gate_noharm(ar, model, gen, test_users, prior_vec)
    R["G_noQ1drop"] = gate_perturn(ar, model, gen, test_users, prior_vec)
    R["G_clean"] = gate_clean(ar, model, gen, test_users, prior_vec)
    R["G_caplength"] = gate_caplength(ar, model, gen, test_users, prior_vec, cold0)
    R["G_canaries"] = gate_canaries(ar, model, gen, test_users, prior_vec)
    R["G_GoT"] = gate_got(ar, model, gen, test_users, prior_vec)
    R["G_prolific"] = gate_prolific(ar, model, gen, test_users, prior_vec)
    R["G_implicit"] = gate_implicit(ar, model, gen, test_users, prior_vec)
    R["G_firewall"] = gate_firewall(ar, gen, test_users)
    R["cold0"] = cold0
    names = [k for k in R if k.startswith("G_")]
    allpass = all(R[k].get("pass") for k in names)
    R["all_pass"] = bool(allpass)
    print("\n=== GATE VERDICTS (TEST) ===", flush=True)
    for k in names:
        print(f"  {k:14s} {'PASS' if R[k].get('pass') else 'FAIL'}", flush=True)
    print(f"  ALL PASS: {allpass}", flush=True)
    if isinstance(ar, LazyArena):
        ar.flush_cache()
    return R


def _append_md(state, sanity, fold_val, R):
    def _v(b):
        return "PASS" if b else "FAIL"
    o = ["\n## FOLD RECOVERED: attention-pool + shrinkage + full gate suite\n\n"]
    o.append("Per DESIGN_SHEET_FOLD_RECOVERED.md (LOCKED 2026-07-10). Recovers the June attention-pool "
             "+ Bayesian-shrinkage no-harm design over the FROZEN RecVAE-d512 decoder. "
             "z = FIXED cold prior + w*delta; delta = ATTENTION-POOL (softmax weights sum to 1) -> "
             "residual MLP; w = CONTENT confidence (fidelity x coherence, NEVER count). Raw per-token "
             "log(n_E)/log(V)/log(p_E) (NO hand-crafted surprise). Answers DATA-SIDE (real U EASE). "
             "Curriculum = DeOODGen realistic strategy mixture (tails ~0.18) at lengths 1..24 + 30% "
             "clean; blind_eig HELD OUT. IPS-weighted multinomial reconstruction of held-out likes. "
             "Split: TRAIN/VAL/TEST disjoint population trU users, 300 study ids excluded; all gates "
             "on TEST.\n\n")
    if sanity:
        o.append(f"### Sanity (tiny run): intercept {_v(sanity['G_intercept']['pass'])}, "
                 f"no-Q1-drop {_v(sanity['G_noQ1drop']['pass'])} "
                 f"(cold {sanity['G_noQ1drop']['curve'][0]:.3f} -> turn1 "
                 f"{sanity['G_noQ1drop']['curve'][1]:.3f}), loss finite-decreasing.\n\n")
    o.append(f"### Fold: best val NDCG@{K} {state['best_val']:.4f} @ep{state['best_epoch']} "
             f"(`{BEST}`); cold prior baseline {R['cold0']:.4f}.\n\n")
    o.append("### Gate table (TEST users)\n\n| gate | value | verdict |\n|---|---|:--:|\n")
    gi = R["G_intercept"]; o.append(f"| G-intercept | max_dev {gi['max_dev']:.1e}, cold {gi['cold_fold']:.4f}"
                                    f"=={gi['native_cold']:.4f} | {_v(gi['pass'])} |\n")
    gf = R["G_falsify"]; o.append(f"| G-falsify-count | dup {gf['dup_delta']:+.5f}, pad {gf['pad_delta']:+.4f}"
                                  f" | {_v(gf['pass'])} |\n")
    gg = R["G_generalize"]; o.append(f"| G-generalize | held-out {gg['heldout']:.4f} vs mean-seen "
                                     f"{gg['mean_seen']:.4f} | {_v(gg['pass'])} |\n")
    go = R["G_order"]; o.append(f"| G-order | max_dev {go['max_dev']:.1e} | {_v(go['pass'])} |\n")
    gh = R["G_noharm"]; o.append(f"| G-noharm | delta {gh['mean']:+.4f} CI[{gh['ci'][0]:+.4f},"
                                 f"{gh['ci'][1]:+.4f}] | {_v(gh['pass'])} |\n")
    gq = R["G_noQ1drop"]; o.append(f"| G-noQ1drop | Q1 {gq['q1_drop']:+.4f}, max decline "
                                   f"{gq['max_step_decline']:+.4f} | {_v(gq['pass'])} |\n")
    gc = R["G_clean"]; o.append(f"| G-clean | fold {gc['fold_ndcg']:.4f} vs native {gc['native_ndcg']:.4f}"
                                f" ({gc['gap']:+.4f}) | {_v(gc['pass'])} |\n")
    gl = R["G_caplength"]; o.append(f"| G-caplength | elic@8 {gl['elic_ndcg']:.4f} vs cold {gl['cold']:.4f}"
                                    f" | {_v(gl['pass'])} |\n")
    gca = R["G_canaries"]
    for ch in ("item", "concept", "entity"):
        d = gca[ch]
        o.append(f"| G-canary {ch} | impl {d['implicit_lift']:+.4f}, expl {d['explicit_lift']:+.4f} | "
                 f"{_v((d['implicit_lift']>0) and (np.isnan(d['explicit_lift']) or d['explicit_lift']>0))} |\n")
    gt = R["G_GoT"]; o.append(f"| G-GoT | {gt['mean']:+.4f} CI[{gt['ci'][0]:+.4f},{gt['ci'][1]:+.4f}] | "
                              f"{_v(gt['pass'])} |\n")
    gp = R["G_prolific"]; o.append(f"| G-prolific | {gp['mean']:+.4f} CI[{gp['ci'][0]:+.4f},"
                                   f"{gp['ci'][1]:+.4f}] | {_v(gp['pass'])} |\n")
    gim = R["G_implicit"]; o.append(f"| G-implicit-ablation | {gim['mean']:+.4f} CI[{gim['ci'][0]:+.4f},"
                                    f"{gim['ci'][1]:+.4f}] | {_v(gim['pass'])} |\n")
    gfw = R["G_firewall"]; o.append(f"| G-firewall | {gfw['note'][:40]}... | {_v(gfw['pass'])} |\n")
    o.append(f"\n**ALL GATES PASS (TEST): {R['all_pass']}**\n")
    with open(BUILD_MD, "a", encoding="utf-8") as fh:
        fh.write("".join(o))
    assert os.path.exists(BUILD_MD)


def cmd_sanity(args):
    t0 = time.time()
    ar, gen, tr, val, test, prior_vec = build_world(args.n_train, args.n_val, args.n_test)
    state = train(ar, gen, tr, val, prior_vec, args, tag="sanity")
    model, _ = load_best(ar.FR)
    # sanity checks: (a) finite decreasing loss, (b) G-intercept, (c) no Q1 drop
    losses = [h["train_loss"] for h in state["history"]]
    loss_ok = bool(np.all(np.isfinite(losses)) and losses[-1] < losses[0])
    gi = gate_intercept(ar, model, gen, test, prior_vec)
    gq = gate_perturn(ar, model, gen, test, prior_vec, maxB=6)
    # (d) NO-CAP clean-reveal check on the MOST PROLIFIC user: unified builder emits ALL answerable
    #     tokens (item+concept+entity, real surprise), count == full answerable set, NOT capped.
    uc = max(test, key=lambda u: len(u["known"]))       # most prolific test user
    cctx = gen.ctx(uc)
    n_answerable = int((cctx["nE"] > 0).sum())           # full answerable question set (no cap)
    n_qs = len(gen.clean_qs(uc, cctx))
    creveal = gen.clean_reveal(uc, cctx)
    ctypes = sorted(set(int(t[0]) for t in creveal))
    per_ch = {c: int(sum(1 for t in creveal if int(t[0]) == c)) for c in (0, 1, 2, 3)}
    has_multichannel = any(int(t[0]) in (TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY) for t in creveal)
    max_ln = max((float(t[3]) for t in creveal), default=0.0)
    real_surprise = bool(max_ln > float(np.log(1.5)) + 1e-6)
    no_cap = bool(n_qs == n_answerable)                  # every answerable question used, none dropped
    enriched = bool(has_multichannel and real_surprise and no_cap and len(creveal) > 0)
    ar.flush_cache()
    R = dict(G_intercept=gi, G_noQ1drop=gq, loss_finite_decreasing=loss_ok, losses=losses,
             clean_enriched=enriched, clean_token_types=ctypes, clean_max_logn=max_ln,
             clean_n_tokens=len(creveal), prolific_uid=int(uc["u"]),
             prolific_n_ratings=len(uc["known"]), clean_n_answerable_q=n_answerable,
             clean_n_selected_q=n_qs, clean_no_cap=no_cap, clean_per_channel_tokens=per_ch)
    sane = bool(loss_ok and gi["pass"] and gq["pass"] and enriched)
    print(f"\n=== SANITY VERDICT ===\n  loss finite-decreasing: {loss_ok} ({losses})", flush=True)
    print(f"  G-intercept: {gi['pass']}   no-Q1-drop: {gq['pass']} "
          f"(cold {gq['curve'][0]:.4f} -> turn1 {gq['curve'][1]:.4f})", flush=True)
    print(f"  NO-CAP clean reveal on PROLIFIC user u={uc['u']} ({len(uc['known'])} ratings): "
          f"selected {n_qs}/{n_answerable} answerable q (no_cap={no_cap}); {len(creveal)} tokens "
          f"by channel item={per_ch[0]} concept={per_ch[1]} attr={per_ch[2]} entity={per_ch[3]}; "
          f"types {ctypes}; max log(n_E+.5) {max_ln:.3f} vs log(1.5)=0.405 -> real surprise "
          f"{real_surprise}", flush=True)
    print(f"  SANITY {'PASS' if sane else 'FAIL'}  [{(time.time()-t0)/60:.1f}m]", flush=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    blob = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    blob["sanity"] = dict(fold_val=state["best_val"], best_epoch=state["best_epoch"],
                          sanity_pass=sane, **{k: R[k] for k in R})
    json.dump(blob, open(RESULTS, "w"), indent=1, default=float)
    print(f"[rec] wrote sanity -> {RESULTS}", flush=True)
    return sane


def cmd_full(args):
    t0 = time.time()
    ar, gen, tr, val, test, prior_vec = build_world(args.n_train, args.n_val, args.n_test)
    state = train(ar, gen, tr, val, prior_vec, args, tag="full")
    model, _ = load_best(ar.FR)
    # re-derive the deterministic fold val for the record
    fold_val = state["best_val"]
    R = run_gates(ar, gen, model, test, prior_vec)
    # sanity snapshot on TEST (for the deliverable)
    sanity = dict(G_intercept=R["G_intercept"], G_noQ1drop=R["G_noQ1drop"])
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    blob = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    blob["fold"] = dict(best_val=state["best_val"], best_epoch=state["best_epoch"],
                        history=state["history"], n_train=len(tr), n_val=len(val), n_test=len(test),
                        strategies_seen=SEEN, heldout=HELDOUT, clean_frac=CLEAN_FRAC, ckpt=BEST)
    blob["gates"] = R
    json.dump(blob, open(RESULTS, "w"), indent=1, default=float)
    _append_md(state, blob.get("sanity"), state, R)
    print(f"\n[rec] wrote fold val + full gate suite -> {RESULTS}  and appended -> {BUILD_MD}  "
          f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sanity", "full", "verifyans"])
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_verify", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "verifyans":
        verify_lazy_eager(n_users=a.n_verify, seed=a.seed)
    elif a.cmd == "sanity":
        a.n_train = 100; a.n_val = 40; a.n_test = 60; a.epochs = 3
        cmd_sanity(a)
    else:
        cmd_full(a)
