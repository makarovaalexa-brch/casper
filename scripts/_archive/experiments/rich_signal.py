"""rich_signal.py -- RICH TWO-AXIS SIGNAL recommender + capacity-equalized channel ablation.

Contract: casper/DESIGN_RICH_SIGNAL.md (author-signed; Fable pass-1 sec.8 + pass-2 sec.9 BINDING) +
casper/RICH_DATASET_REVIEW.md (exact encodings). Reuses arena_core (distilled answerer v2.1 world,
_gen_user tables, Universe member-bags, EASE backbone) and signed_latent (SignedAE infra, safe_save).

WHAT: a deep residual recommender whose input is FIVE item-space channels (ni=18430) built from a
REALISTIC distilled-answerer interview over the fixed 2,428-question universe:
  value_vec  = centered graded value (hated -1 / meh -1/3 / liked +1/3 / loved +1; rated crval/2.5)
  know_vec   = knowledge scaled (no_clue 0 / rough 0.5 / know_well 1.0)
  ref_vec    = asked-and-refused (know=0) marker
  source_flag= direct-item vs concept/entity-derived coverage weight
  mask       = weighted coverage ("asked") -- the C0 crude baseline
  (+ ease_vec, the EASE-t-on-asked control channel, used only by C1e in the value slot)
Concepts/entities spread to member items with L2-unit pop weights; DIRECT item answers OVERRIDE
concept-derived (order-invariant). Held-liked targets are ZEROED from every input channel (leak-free).

MODEL: one 5xni deep-residual encoder (SignedAE-class, 5 dense blocks) -> z(512); Linear decoder ->
item logits. Loss = multinomial log-lik of held-liked (no dislike margin). CAPACITY-EQUALIZED
ablation: SAME arch for all configs; a config is realized by only feeding its active channels (zeroed
channels carry no gradient -> identical capacity, no OOD). Configs:
  C0=mask ; C1=value+source+mask ; C2=know+ref+mask ; C3=all ; C1e=ease+source+mask.

Firewall: population trU users only (300 study ids excluded); the 173 real-LLM users are QUARANTINED,
touched ONLY by the final transfer eval. NO DATA CAPS (full catalog, full universe, full held sets).
"""
import os, sys, json, time, math, argparse, hashlib
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())

import arena_core as AC
import dans_build as DB
import dans_stages as DS
import llm_answerability_gate as G
from signed_latent import safe_save

OUTDIR = "C:/dev/phd/casper/.cache/rich_signal"
os.makedirs(OUTDIR, exist_ok=True)

LO = 4.0                                  # liked threshold
TMAX = 24
K_TURNS = [1, 2, 4, 8, 16]
EARLY_KS = [1, 2, 4, 8]                   # PRIMARY early-AUC support (design 9-B2)
_W10 = 1.0 / np.log2(np.arange(2, 12))    # NDCG@10 gains
CENTERED = AC.VBIN_CENTERED               # [-1,-1/3,+1/3,+1]

# ---- 5 input channels (fixed slot order) ----
CH = ["value", "know", "ref", "source", "mask"]
CH_IX = {c: i for i, c in enumerate(CH)}
# config -> active channel slots (capacity-equalized; unused channels zeroed in data).
# C1e feeds EASE-t into the 'value' slot (source-flag present per sec.9-E2).
CONFIGS = {
    "C0":  ["mask"],
    "C1":  ["value", "source", "mask"],
    "C2":  ["know", "ref", "mask"],
    "C3":  ["value", "know", "ref", "source", "mask"],
    "C1e": ["value", "source", "mask"],     # 'value' slot fed by ease_vec (see build_channels use_ease)
}
PRIORITY = ["C3", "C0", "C1", "C1e", "C2"]  # design: if time-limited


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---- peak resident memory (Windows working set; no external deps) ----
def peak_rss_gb():
    """Return (peak_working_set_GB, current_working_set_GB). Peak is monotone over process life."""
    try:
        import ctypes, ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32"); psapi = ctypes.WinDLL("psapi")
        f = getattr(psapi, "GetProcessMemoryInfo", None) or getattr(psapi, "K32GetProcessMemoryInfo")
        h = k32.GetCurrentProcess()
        if f(wt.HANDLE(h), ctypes.byref(c), wt.DWORD(c.cb)):
            return c.PeakWorkingSetSize / 1e9, c.WorkingSetSize / 1e9
    except Exception:
        pass
    return float("nan"), float("nan")


_PEAK_GB = 0.0


def log_mem(where=""):
    global _PEAK_GB
    pk, cur = peak_rss_gb()
    if pk == pk:
        _PEAK_GB = max(_PEAK_GB, pk)
    log(f"[mem] {where} resident={cur:.2f}GB peak={pk:.2f}GB")
    return pk, cur


# ======================================================================== faithful FAST answerer gen
# _gen_user's cost is the per-user dense EASE fold-in rc@ease_B (9352^2). We batch that matmul across
# users (float64, matching numpy's upcast accumulation) and feed the precomputed pred into a verbatim
# copy of _gen_user. Validated bit-for-faithful against ar._gen_user (see cmd_prefill assertion).
def ease_pred_batch(ar, known_list):
    """Batched EASE prediction for a list of known-dicts -> (n, len(ease_mu)) float64, clipped."""
    m = len(ar.ease_mu)
    RC = np.zeros((len(known_list), m), np.float64)
    for a, known in enumerate(known_list):
        for j, r in known.items():
            c = ar.ease_index.get(int(j))
            if c is not None:
                RC[a, c] = r - ar.ease_mu[c]
    pred = np.clip(ar.ease_mu[None, :] + RC @ ar.ease_B.astype(np.float64), 0.5, 5.0)
    return pred


def gen_user_fast(ar, uid, known, pred):
    """Verbatim copy of arena_core.Arena._gen_user, EXCEPT the EASE `pred` vector is passed in
    (precomputed by ease_pred_batch) rather than recomputed per-user. Returns the same dict."""
    u = ar.uni
    f = u.user_features(known)
    rng = np.random.default_rng(AC.SEED * 7_777 + int(uid))
    Pk = DS.know_probs(u, f, ar.models, rng=rng)
    know = np.concatenate([DS._sample_cat(Pk["concept"], rng),
                           DS._sample_cat(Pk["entity"], rng),
                           DS._sample_cat(Pk["item"], rng)]).astype(np.int8)
    cmean = float(f["cmean"])
    t_full = np.full(u.ni, np.nan)
    t_full[ar.ease_uni] = pred
    known_ind = np.zeros(u.ni, np.float64)
    for j, r in known.items():
        if 0 <= int(j) < u.ni:
            t_full[int(j)] = r
            known_ind[int(j)] = 1.0
    tm = np.isfinite(t_full).astype(np.float64)
    tv = np.where(tm > 0, t_full, 0.0)
    wt = u.pr * tm
    mv = ar.models["value"]
    num_c = np.asarray(u.tagM.dot(wt * tv)).ravel()
    den_c = np.asarray(u.tagM.dot(wt)).ravel()
    ctaste = np.where(den_c > 1e-9, num_c / np.maximum(den_c, 1e-9), cmean)
    nr_c = np.asarray(u.tagM.dot(known_ind)).ravel()
    Xc = np.column_stack([ctaste - 3.5, (nr_c > 0).astype(float), np.log1p(nr_c),
                          np.full(ar.ntag, cmean)])
    Pv_c = DB.ord_prob(mv["concept"]["theta"], DB.zscale(Xc, mv["concept"]["mu"],
                       mv["concept"]["sd"]), 4)
    num_e = np.asarray(u.entM.dot(wt * tv)).ravel()
    den_e = np.asarray(u.entM.dot(wt)).ravel()
    etaste = np.where(den_e > 1e-9, num_e / np.maximum(den_e, 1e-9), cmean)
    nr_e = np.asarray(u.entM.dot(known_ind)).ravel()
    Xe = np.column_stack([etaste - 3.5, (nr_e > 0).astype(float), np.log1p(nr_e),
                          np.full(ar.nent, cmean)])
    Pv_e = DB.ord_prob(mv["entity"]["theta"], DB.zscale(Xe, mv["entity"]["mu"],
                       mv["entity"]["sd"]), 4)
    t_bank = t_full[u.bank]
    t_bank = np.where(np.isfinite(t_bank), t_bank, cmean)
    Xi = np.column_stack([t_bank - 3.5, f["item_val"][:, 1], f["item_val"][:, 0],
                          np.full(ar.nbank, cmean)])
    Pv_i = DB.ord_prob(mv["item"]["theta"], DB.zscale(Xi, mv["item"]["mu"], mv["item"]["sd"]), 4)
    val = np.full(ar.nQ, -1, np.int8)
    for off, P in ((0, Pv_c), (ar.off_ent, Pv_e), (ar.off_item, Pv_i)):
        seg = know[off:off + P.shape[0]]
        nz = seg > 0
        if nz.any():
            dv = DS._sample_cat(P[nz], rng)
            block = np.full(P.shape[0], -1, np.int8); block[nz] = dv
            val[off:off + P.shape[0]] = block
    mu_known = float(np.mean(list(known.values())))
    crval = np.full(ar.nbank, np.nan, np.float32)
    rated = f["rated_flag"] > 0.5
    if rated.any():
        know[ar.off_item:][rated] = 2
        for r in np.where(rated)[0]:
            star = known[int(u.bank[r])]
            val[ar.off_item + r] = 3 if star >= 4.5 else 2 if star >= 3.5 else \
                1 if star >= 2.5 else 0
            crval[r] = np.float32(star - mu_known)
    return dict(know=know, val=val, crval=crval)


# ======================================================================== cohorts (study ids excluded)
def build_cohorts(ar, n_val, n_test, max_train, seed=AC.SEED):
    """All-passing population trU users (study ids excluded), split into val / test / train. Same
    per-user half-split convention as arena make_cohorts (uid-seeded)."""
    excl = DB.study_ids()
    d = np.load(DB.META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float32)
    order = np.argsort(uu, kind="stable"); uu, ii, rr = uu[order], ii[order], rr[order]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
    trU = d["trU"].astype(np.int64)
    cand = np.array([u for u in trU if u not in excl], np.int64)
    rng = np.random.default_rng(seed); rng.shuffle(cand)
    want = n_val + n_test + max_train
    out = []
    for u in cand:
        s, e = bnd[u], bnd[u + 1]
        its = ii[s:e]; rat = rr[s:e]
        if len(its) < 8:
            continue
        ru = np.random.default_rng(seed * 1_000_003 + int(u))
        perm = ru.permutation(len(its)); half = len(its) // 2
        known = {int(its[k]): float(rat[k]) for k in perm[:half]}
        held = set(int(its[k]) for k in perm[half:] if rat[k] >= LO)
        if len(known) < 4 or not held:
            continue
        out.append(dict(u=int(u), known=known, held=held))
        if len(out) >= want:
            break
    log(f"cohorts: {len(out)} passing users (requested up to {want})")
    return dict(val=out[:n_val], test=out[n_val:n_val + n_test], train=out[n_val + n_test:])


# ======================================================================== member-weight matrix W_ce
def build_Wce(ar):
    """Sparse (n_ce, ni) member-weight matrix for concept+entity questions: row q = L2-unit
    pop-weights over member items (genome membership x popularity). Cached."""
    p = f"{OUTDIR}/Wce.npz"
    if os.path.exists(p):
        b = np.load(p); return sp.csr_matrix((b["data"], b["indices"], b["indptr"]),
                                              shape=tuple(b["shape"]))
    u = ar.uni; cnt = u.D["cnt"].astype(np.float64)
    rows, cols, data = [], [], []
    for q in range(u.ntag):
        mem = u.tagM[q].indices
        if len(mem) == 0:
            continue
        w = cnt[mem] + 1.0; w = w / (np.linalg.norm(w) + 1e-12)
        rows += [q] * len(mem); cols += mem.tolist(); data += w.tolist()
    for e in range(u.nent):
        mem = u.entM[e].indices
        if len(mem) == 0:
            continue
        w = cnt[mem] + 1.0; w = w / (np.linalg.norm(w) + 1e-12)
        q = u.ntag + e
        rows += [q] * len(mem); cols += mem.tolist(); data += w.tolist()
    W = sp.csr_matrix((data, (rows, cols)), shape=(u.ntag + u.nent, u.ni)).astype(np.float32)
    np.savez_compressed(p, data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape)
    log(f"built Wce {W.shape} nnz={W.nnz} -> {p}")
    return W


# ======================================================================== channel builder (batched)
def _crval_rescale(x):
    return np.clip(x / 2.5, -1.0, 1.0)


def build_channels(ar, Wce, tables, asked_list, held_list, ni, n_ce, off_item,
                   use_ease=False, pred_list=None, cmean_list=None):
    """Build the 6 item-space channels for a batch. tables[b] = dict(know,val,crval); asked_list[b]
    = int array of asked question indices; held_list[b] = set of held item ids (zeroed from inputs).
    Returns dict of (B,ni) float32 arrays: value, know, ref, source, mask, ease.
    DIRECT item answers override concept/entity-derived; concept/entity spread via Wce."""
    B = len(tables)
    # ---- concept/entity sparse (B, n_ce) accumulators ----
    rV, cV, dV = [], [], []       # value
    rK, cK, dK = [], [], []       # know scaled
    rR, cR, dR = [], [], []       # refusal
    rM, cM, dM = [], [], []       # asked
    rE, cE, dE = [], [], []       # ease value (concept/entity: region t)
    # ---- direct item dense (B, ni) accumulators (via lists then scatter) ----
    it_rows, it_cols, it_v, it_k, it_r, it_m, it_e = [], [], [], [], [], [], []
    for b in range(B):
        t = tables[b]; asked = asked_list[b]
        know = t["know"]; val = t["val"]; crval = t["crval"]
        pr = pred_full = None
        if use_ease:
            pr = pred_list[b]; cm = cmean_list[b]
        for q in asked:
            q = int(q); k = int(know[q])
            if q >= off_item:                                   # DIRECT item
                it = int(ar.uni.bank[q - off_item]); b_ = b
                if k == 0:                                       # refusal
                    it_rows.append(b_); it_cols.append(it); it_v.append(0.0); it_k.append(0.0)
                    it_r.append(1.0); it_m.append(1.0); it_e.append(0.0)
                    continue
                cr = crval[q - off_item]
                v = _crval_rescale(cr) if np.isfinite(cr) else (
                    CENTERED[int(val[q])] if int(val[q]) >= 0 else 0.0)
                ks = 1.0 if k == 2 else 0.5
                ev = 0.0
                if use_ease:
                    pv = pr[ar.ease_index[it]] if it in ar.ease_index else cm
                    ev = float(np.clip((pv - cm) / 2.5, -1.0, 1.0))
                it_rows.append(b_); it_cols.append(it); it_v.append(v); it_k.append(ks)
                it_r.append(0.0); it_m.append(1.0); it_e.append(ev)
            else:                                                # concept / entity
                if k == 0:                                       # refusal
                    rR.append(b); cR.append(q); dR.append(1.0)
                    rM.append(b); cM.append(q); dM.append(1.0)
                    continue
                vv = int(val[q])
                if vv < 0:
                    rM.append(b); cM.append(q); dM.append(1.0); continue
                rV.append(b); cV.append(q); dV.append(float(CENTERED[vv]))
                ks = 1.0 if k == 2 else 0.5
                rK.append(b); cK.append(q); dK.append(ks)
                rM.append(b); cM.append(q); dM.append(1.0)
                if use_ease:
                    # region EASE value = pop-weighted member t centered (recompute from Wce row & pred)
                    rE.append(b); cE.append(q); dE.append(0.0)   # placeholder; filled below via matmul
    def spm(r, c, dta):
        return sp.csr_matrix((dta, (r, c)), shape=(B, n_ce)).astype(np.float32)
    Vce = spm(rV, cV, dV) @ Wce
    Kce = spm(rK, cK, dK) @ Wce
    Rce = spm(rR, cR, dR) @ Wce
    Mce = spm(rM, cM, dM) @ Wce
    Vce = np.asarray(Vce.todense()); Kce = np.asarray(Kce.todense())
    Rce = np.asarray(Rce.todense()); Mce = np.asarray(Mce.todense())
    # direct item scatter
    value = np.zeros((B, ni), np.float32); know_c = np.zeros((B, ni), np.float32)
    ref = np.zeros((B, ni), np.float32); cover_it = np.zeros((B, ni), np.float32)
    ease = np.zeros((B, ni), np.float32)
    if it_rows:
        ir = np.asarray(it_rows); ic = np.asarray(it_cols)
        np.add.at(value, (ir, ic), np.asarray(it_v, np.float32))
        np.add.at(know_c, (ir, ic), np.asarray(it_k, np.float32))
        np.add.at(ref, (ir, ic), np.asarray(it_r, np.float32))
        np.add.at(cover_it, (ir, ic), np.asarray(it_m, np.float32))
        np.add.at(ease, (ir, ic), np.asarray(it_e, np.float32))
    direct = cover_it > 0
    # override: direct where present, else concept/entity-derived
    value = np.where(direct, value, Vce)
    know_c = np.where(direct, know_c, Kce)
    ref = np.where(direct, ref, Rce)
    source = cover_it.copy()                                     # direct-derived weight
    mask = cover_it + np.where(direct, 0.0, Mce)                 # total weighted coverage
    # ease concept/entity spread (region t) for covered ce items
    if use_ease:
        # ease_vec on ce positions = rescale(pred-cmean) on covered members, weighted by coverage
        # compute per-user pred spread lazily: pred_ce[b,i] = pred[i]; covered = Mce>0
        covered_ce = (Mce > 0) & (~direct)
        for b in range(B):
            cm = cmean_list[b]; pr = pred_list[b]
            cols = np.where(covered_ce[b])[0]
            if len(cols) == 0:
                continue
            pv = np.array([pr[ar.ease_index[int(i)]] if int(i) in ar.ease_index else cm
                           for i in cols])
            ease[b, cols] = np.clip((pv - cm) / 2.5, -1.0, 1.0)
    # ---- zero held-liked targets from ALL channels (leak-free, design sec.1/4/8-A) ----
    for b in range(B):
        h = list(held_list[b])
        if h:
            for arr in (value, know_c, ref, source, mask, ease):
                arr[b, h] = 0.0
    return dict(value=value, know=know_c, ref=ref, source=source, mask=mask, ease=ease)


def channels_to_input(chan, active, use_ease_in_value=False):
    """Stack active channels into (B,5,ni) with the fixed slot order; zero inactive slots."""
    B, ni = chan["value"].shape
    X = np.zeros((B, 5, ni), np.float32)
    for c in active:
        src = c
        if c == "value" and use_ease_in_value:
            src = "ease"
        X[:, CH_IX[c], :] = chan[src]
    return X


# ======================================================================== model
def swish(x):
    return x.mul(torch.sigmoid(x))


class RichEncoder(nn.Module):
    """5-channel deep residual encoder (SignedAE-class). fc1 is a per-channel weight bank (5,ni,hid)
    so a config feeds only its active channel blocks (inactive -> no gradient, capacity-equalized)."""
    def __init__(self, ni, hidden=600, latent=512):
        super().__init__()
        self.ni = ni
        self.W1 = nn.Parameter(torch.empty(5, ni, hidden)); nn.init.kaiming_uniform_(self.W1, a=5**0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.ln1 = nn.LayerNorm(hidden, eps=0.1)
        self.fc2 = nn.Linear(hidden, hidden); self.ln2 = nn.LayerNorm(hidden, eps=0.1)
        self.fc3 = nn.Linear(hidden, hidden); self.ln3 = nn.LayerNorm(hidden, eps=0.1)
        self.fc4 = nn.Linear(hidden, hidden); self.ln4 = nn.LayerNorm(hidden, eps=0.1)
        self.fc5 = nn.Linear(hidden, hidden); self.ln5 = nn.LayerNorm(hidden, eps=0.1)
        self.fc_z = nn.Linear(hidden, latent)

    def forward(self, X, active_ix, dropout=0.0):
        # X: (B,5,ni). Per-channel L2 norm (not over the concat) then sum active projections.
        h = self.b1.clone().expand(X.shape[0], -1).contiguous()
        for ci in active_ix:
            xc = X[:, ci, :]
            n = xc.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
            xc = xc / n
            if dropout > 0:
                xc = F.dropout(xc, p=dropout, training=self.training)
            h = h + xc @ self.W1[ci]
        h1 = self.ln1(swish(h))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_z(h5)


class RichAE(nn.Module):
    def __init__(self, ni, hidden=600, latent=512):
        super().__init__()
        self.encoder = RichEncoder(ni, hidden, latent)
        self.decoder = nn.Linear(latent, ni)
        self.ni = ni

    def forward(self, X, active_ix, dropout=0.0):
        z = self.encoder(X, active_ix, dropout)
        return self.decoder(z), z


# ======================================================================== NDCG (full catalog)
def ndcg10(score, tlike, profset, headmask, tail):
    s = score.copy(); s[list(profset)] = -1e30
    if tail:
        s[headmask] = -1e30
        rel = set(t for t in tlike if not headmask[t])
    else:
        rel = set(tlike)
    if not rel:
        return None
    o = np.argsort(-s)[:10]
    dcg = sum(_W10[p] for p, t in enumerate(o) if int(t) in rel)
    idcg = _W10[:min(10, len(rel))].sum() + 1e-12
    return dcg / idcg


# ======================================================================== interview samplers
def q_static_score(ar):
    """User-agnostic informativeness proxy = log popmass (answerable + broad). For pop/entropy strat."""
    return np.log(ar.q_popmass + 1.0)


def user_overlap_score(ar, Wce, known_liked, ni, n_ce, off_item):
    """Per-question overlap with the user's known-LIKED items (on-profile signal). Returns (nQ,)."""
    kl = np.zeros(ni, np.float32)
    if len(known_liked):
        kl[np.asarray(list(known_liked), np.int64)] = 1.0
    ce = np.asarray(Wce @ kl).ravel()                     # concept+entity overlap
    it = kl[ar.uni.bank]                                  # direct item overlap
    return np.concatenate([ce, it])


def eval_interview(ar, Wce, rec, ni, n_ce, off_item, tmax=TMAX):
    """FIXED per-user realistic static interview: rank questions by on-profile overlap (+ popmass
    tiebreak), take top tmax; deterministic -> identical sequence across all configs (paired)."""
    known = rec["known"]
    kl = set(j for j, r in known.items() if r >= LO)
    ov = user_overlap_score(ar, Wce, kl, ni, n_ce, off_item)
    score = ov + 1e-3 * q_static_score(ar)
    order = np.argsort(-score, kind="stable")[:tmax]
    return order.astype(np.int64)


def train_interview(u_known_liked, u_known_dis, ar, Wce, ni, n_ce, off_item, rng, stat_score):
    """One curriculum interview: strategy mix x log-uniform length (+ dense full-profile regime)."""
    r = rng.random()
    if r < 0.15:                                           # full-profile-via-answerer (dense end)
        T = int(rng.integers(80, ar.nQ))
        strat = "pop"
    else:
        T = int(round(math.exp(rng.uniform(0.0, math.log(TMAX)))))
        T = max(1, min(T, TMAX))
        strat = ["random", "pop", "entropy", "on_profile", "off_profile", "adversarial", "mixed"][
            int(rng.integers(0, 7))]
    nQ = ar.nQ
    if strat == "random":
        keys = rng.random(nQ)
    elif strat in ("pop", "entropy"):
        keys = stat_score + rng.gumbel(size=nQ)
    elif strat == "on_profile":
        keys = user_overlap_score(ar, Wce, u_known_liked, ni, n_ce, off_item) + rng.gumbel(size=nQ) * 0.3
    elif strat == "off_profile":
        keys = user_overlap_score(ar, Wce, u_known_dis, ni, n_ce, off_item) + rng.gumbel(size=nQ) * 0.3
    elif strat == "adversarial":
        keys = -stat_score + rng.gumbel(size=nQ)
    else:
        keys = rng.gumbel(size=nQ) + 0.5 * stat_score
    return np.argpartition(-keys, T - 1)[:T].astype(np.int64) if T < nQ else np.arange(nQ)


# ======================================================================== streaming table store
class TableStore:
    """Lazy per-user answer-table accessor backed by memmap'd raw .npy fields (know,val,crval).
    Maps uid -> row position and returns ONLY the requested rows on demand, so peak resident memory
    is the arena + model + one batch, never the full N x 2428 tables. Drop-in for the old
    {uid: {know,val,crval}} dict: supports store[uid], `uid in store`, len(store), .items(), .keys().
    Rows are read-only views into the memmap (never mutated by build_channels); callers that need to
    edit (refusal_sweep) .copy() first, as before."""
    def __init__(self, uids, kp, vp, cp):
        self.uids = np.asarray(uids, np.int64)
        self.pos = {int(u): i for i, u in enumerate(self.uids)}
        self._know = np.load(kp, mmap_mode="r")
        self._val = np.load(vp, mmap_mode="r")
        self._crval = np.load(cp, mmap_mode="r")

    def __contains__(self, uid):
        return int(uid) in self.pos

    def __len__(self):
        return len(self.uids)

    def row(self, uid):
        p = self.pos[int(uid)]
        return dict(know=self._know[p], val=self._val[p], crval=self._crval[p])

    def __getitem__(self, uid):
        return self.row(uid)

    def keys(self):
        return [int(u) for u in self.uids]

    def items(self):
        for u in self.uids:
            yield int(u), self.row(int(u))


# ======================================================================== prefill -> memmap store
def prefill(ar, recs, name, chunk=4000):
    """Generate per-user answer tables (know,val,crval) with batched EASE and STREAM them into
    memmap'd raw .npy fields (mm_{name}_{know,val,crval}.npy). Sharded npz stay the resumable compute
    layer; the .npy fields are written chunk-by-chunk so nothing larger than one chunk is ever
    resident. Returns a lazy TableStore (NOT a full dict) -- this is the OOM fix.

    Reuse: existing 40k shards from the earlier run are valid (build_cohorts order is deterministic in
    max_train), so only the new users are generated."""
    from numpy.lib.format import open_memmap
    n = len(recs); uids = np.array([r["u"] for r in recs], np.int64)
    kp = f"{OUTDIR}/mm_{name}_know.npy"; vp = f"{OUTDIR}/mm_{name}_val.npy"
    cp = f"{OUTDIR}/mm_{name}_crval.npy"; up = f"{OUTDIR}/mm_{name}_uids.npy"
    if all(os.path.exists(p) for p in (kp, vp, cp, up)):
        old = np.load(up)
        if len(old) == n and np.array_equal(old, uids):
            log(f"tables_{name}: memmap store present ({n} users)")
            return TableStore(uids, kp, vp, cp)
        log(f"tables_{name}: memmap mismatch ({len(old)} vs {n}) -> rebuild")
    shard_dir = f"{OUTDIR}/shards_{name}"; os.makedirs(shard_dir, exist_ok=True)
    t0 = time.time()
    KM = open_memmap(kp + ".tmp", mode="w+", dtype=np.int8, shape=(n, ar.nQ))
    VM = open_memmap(vp + ".tmp", mode="w+", dtype=np.int8, shape=(n, ar.nQ))
    CM = open_memmap(cp + ".tmp", mode="w+", dtype=np.float32, shape=(n, ar.nbank))
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        sh = f"{shard_dir}/{st}.npz"
        if os.path.exists(sh):
            b = np.load(sh); kk, vv, cc = b["know"], b["val"], b["crval"]
        else:
            idx = list(range(st, en))
            pred = ease_pred_batch(ar, [recs[i]["known"] for i in idx])
            kk = np.zeros((len(idx), ar.nQ), np.int8); vv = np.zeros((len(idx), ar.nQ), np.int8)
            cc = np.zeros((len(idx), ar.nbank), np.float32)
            for a, i in enumerate(idx):
                t = gen_user_fast(ar, recs[i]["u"], recs[i]["known"], pred[a])
                kk[a] = t["know"]; vv[a] = t["val"]; cc[a] = t["crval"]
            safe_save_npz(sh, know=kk, val=vv, crval=cc)
            log(f"  prefill {name}: {en}/{n} [{(time.time()-t0)/60:.1f}m]")
        KM[st:en] = kk; VM[st:en] = vv; CM[st:en] = cc
        del kk, vv, cc
    KM.flush(); VM.flush(); CM.flush(); del KM, VM, CM
    os.replace(kp + ".tmp", kp); os.replace(vp + ".tmp", vp); os.replace(cp + ".tmp", cp)
    np.save(up, uids)
    log(f"tables_{name}: memmap store built {n} users [{(time.time()-t0)/60:.1f}m]")
    return TableStore(uids, kp, vp, cp)


def safe_save_npz(path, **kw):
    import shutil
    try:
        if shutil.disk_usage(os.path.dirname(path) or ".").free / 1e6 < 500:
            log(f"  [warn] low disk; SKIP {os.path.basename(path)}"); return
    except OSError:
        pass
    tmpbase = path + ".tmp"
    try:
        np.savez_compressed(tmpbase, **kw)     # writes tmpbase + '.npz'
        os.replace(tmpbase + ".npz", path)
    except (OSError, RuntimeError) as e:
        log(f"  [warn] npz save failed {os.path.basename(path)}: {e}")


# ======================================================================== headmask + shared setup
def get_headmask(ar):
    cnt = ar.uni.D["cnt"].astype(np.float64)
    order = np.argsort(-cnt); cum = np.cumsum(cnt[order]) / cnt.sum()
    hm = np.zeros(ar.uni.ni, bool); hm[order[:np.searchsorted(cum, 0.33) + 1]] = True
    return hm


def active_ix_for(cfg):
    return [CH_IX[c] for c in CONFIGS[cfg]]


# ======================================================================== channel build for a cohort @k
def build_for_asked(ar, Wce, tables, recs, asked_seqs, ni, n_ce, off_item, use_ease):
    """Build channels for a list of users given each user's asked-question array (already truncated)."""
    tabs = [tables[r["u"]] for r in recs]
    held = [r["held"] for r in recs]
    pred_list = cmean_list = None
    if use_ease:
        knowns = [r["known"] for r in recs]
        pred_list = list(ease_pred_batch(ar, knowns))
        cmean_list = [float(np.mean(list(r["known"].values()))) for r in recs]
    return build_channels(ar, Wce, tabs, asked_seqs, held, ni, n_ce, off_item,
                          use_ease=use_ease, pred_list=pred_list, cmean_list=cmean_list)


def score_cohort(model, ar, Wce, tables, recs, asked_seqs, cfg, ni, n_ce, off_item, headmask,
                 bs=256):
    """Return (full_ndcg_list, tail_ndcg_list) aligned to recs for a given config + asked sequences."""
    active = active_ix_for(cfg); use_ease = (cfg == "C1e")
    model.eval()
    fN, tN = [], []
    for st in range(0, len(recs), bs):
        rb = recs[st:st + bs]; ab = asked_seqs[st:st + bs]
        chan = build_for_asked(ar, Wce, tables, rb, ab, ni, n_ce, off_item, use_ease)
        X = channels_to_input(chan, CONFIGS[cfg], use_ease_in_value=use_ease)
        with torch.no_grad():
            sc = model(torch.from_numpy(X), active)[0].numpy().astype(np.float64)
        for b, r in enumerate(rb):
            tlike = list(r["held"]); profset = set(r["known"].keys())
            fN.append(ndcg10(sc[b], tlike, profset, headmask, False))
            tN.append(ndcg10(sc[b], tlike, profset, headmask, True))
    return fN, tN


# ======================================================================== TRAIN one config
def train_one(ar, Wce, headmask, tables_tr, tables_va, tr, kl_list, kd_list, va, va_full_seqs,
              stat, ni, n_ce, off_item, cfg, epochs, resume):
    active = active_ix_for(cfg); use_ease = (cfg == "C1e")
    if os.path.exists(f"{OUTDIR}/{cfg}_best.pt") and not resume and \
            not os.path.exists(f"{OUTDIR}/{cfg}.pt"):
        log(f"SKIP {cfg}: best checkpoint already exists (no latest to resume)")
        return
    log(f"TRAIN config={cfg} active={CONFIGS[cfg]} use_ease={use_ease} "
        f"train={len(tr)} val={len(va)}")
    model = RichAE(ni)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    ckpt = f"{OUTDIR}/{cfg}.pt"; best = f"{OUTDIR}/{cfg}_best.pt"
    start_ep, best_val, hist = 0, -1.0, []
    if resume and os.path.exists(ckpt):
        try:
            blob = torch.load(ckpt, map_location="cpu"); model.load_state_dict(blob["model"])
            opt.load_state_dict(blob["opt"]); start_ep = blob["epoch"]; hist = blob["hist"]
            best_val = max([h["val"] for h in hist], default=-1.0)
            log(f"RESUMED ep{start_ep} best_val={best_val:.4f}")
        except Exception as e:
            log(f"resume failed ({e}); fresh")
    B = 256; nU = len(tr); idx = np.arange(nU)
    for ep in range(start_ep, epochs):
        model.train(); r2 = np.random.default_rng(ep * 7919 + 17)
        np.random.default_rng(ep).shuffle(idx)
        t0 = time.time(); run = 0.0; nb = 0
        for st in range(0, nU, B):
            bat = idx[st:st + B]
            asked = [train_interview(kl_list[i], kd_list[i], ar, Wce, ni, n_ce, off_item, r2, stat)
                     for i in bat]
            rb = [tr[i] for i in bat]
            chan = build_for_asked(ar, Wce, tables_tr, rb, asked, ni, n_ce, off_item, use_ease)
            X = torch.from_numpy(channels_to_input(chan, CONFIGS[cfg], use_ease_in_value=use_ease))
            tgt = torch.zeros((len(bat), ni))
            for b, i in enumerate(bat):
                h = list(tr[i]["held"])
                if h:
                    tgt[b, h] = 1.0
            logits = model(X, active, dropout=0.5)[0]
            logsm = F.log_softmax(logits, dim=-1)
            denom = tgt.sum(-1).clamp_min(1.0)
            loss = -((logsm * tgt).sum(-1) / denom).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss); nb += 1
            if nb % 100 == 0:
                log(f"  {cfg} ep{ep} b{nb}/{(nU+B-1)//B} loss={run/nb:.4f} {(time.time()-t0)/60:.1f}m")
        vf, _ = score_cohort(model, ar, Wce, tables_va, va, va_full_seqs, cfg, ni, n_ce, off_item,
                             headmask)
        vf = float(np.mean([x for x in vf if x is not None]))
        hist.append({"epoch": ep + 1, "loss": run / max(nb, 1), "val": vf})
        log(f"[{cfg} ep{ep+1}] loss={run/max(nb,1):.4f} VAL(full-answerer)={vf:.4f} "
            f"({(time.time()-t0)/60:.1f}m)")
        safe_save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1,
                   "hist": hist, "cfg": cfg}, ckpt)
        if vf > best_val:
            best_val = vf
            safe_save({"model": model.state_dict(), "val": vf, "epoch": ep + 1, "cfg": cfg}, best)
        try:
            json.dump(hist, open(f"{OUTDIR}/{cfg}_history.json", "w"))
        except OSError:
            pass
    # delete latest (keep only best) to save disk
    try:
        if os.path.exists(ckpt):
            os.remove(ckpt)
    except OSError:
        pass
    log(f"TRAIN done tag={cfg} best_val={best_val:.4f}")


def build_context(args):
    """Load arena once, build cohorts, prefill tables, precompute per-user helpers. Shared by
    cmd_train / cmd_full / cmd_eval."""
    ar = AC.Arena(verbose=True)
    log_mem("arena loaded")
    ni = ar.uni.ni; n_ce = ar.ntag + ar.nent; off_item = ar.off_item
    Wce = build_Wce(ar); headmask = get_headmask(ar)
    coh = build_cohorts(ar, args.n_val, args.n_test, args.max_train)
    stat = q_static_score(ar)
    # prefill ONCE into a lazy streaming store and reuse across all configs (never resident; OOM fix)
    tr = coh["train"]; va = coh["val"]
    tables_tr = prefill(ar, tr, "train"); tables_va = prefill(ar, va, "val")
    log_mem("prefill(train+val) done (streaming store)")
    kl_list = [set(j for j, r in x["known"].items() if r >= LO) for x in tr]
    kd_list = [set(j for j, r in x["known"].items() if r < LO) for x in tr]
    va_full_seqs = [np.arange(ar.nQ) for _ in va]
    ctx = dict(ar=ar, ni=ni, n_ce=n_ce, off_item=off_item, Wce=Wce, headmask=headmask, coh=coh,
               stat=stat, tables_tr=tables_tr, tables_va=tables_va, kl_list=kl_list, kd_list=kd_list,
               va_full_seqs=va_full_seqs)
    return ctx


def _train_ctx(ctx, cfg, epochs, resume):
    ar = ctx["ar"]; Wce = ctx["Wce"]; coh = ctx["coh"]
    tr = coh["train"]; va = coh["val"]
    train_one(ar, Wce, ctx["headmask"], ctx["tables_tr"], ctx["tables_va"], tr, ctx["kl_list"],
              ctx["kd_list"], va, ctx["va_full_seqs"], ctx["stat"], ctx["ni"], ctx["n_ce"],
              ctx["off_item"], cfg, epochs, resume)


def cmd_train(args):
    ctx = build_context(args)
    _train_ctx(ctx, args.tag, args.epochs, args.resume)


def cmd_full(args):
    """ONE-process pipeline: prefill -> train all configs (priority order) -> full eval."""
    ctx = build_context(args)
    for cfg in PRIORITY:
        if os.path.exists(f"{OUTDIR}/{cfg}_best.pt") and not os.path.exists(f"{OUTDIR}/{cfg}.pt"):
            log(f"== config {cfg}: best exists, skipping ==")
            continue
        log(f"== training config {cfg} ({PRIORITY.index(cfg)+1}/{len(PRIORITY)}) ==")
        _train_ctx(ctx, cfg, args.epochs, args.resume)
    log("== all configs trained; running eval ==")
    _eval_ctx(ctx, args)


# ======================================================================== paired eval + early-AUC
def load_model(cfg, ni):
    blob = torch.load(f"{OUTDIR}/{cfg}_best.pt", map_location="cpu")
    m = RichAE(ni); m.load_state_dict(blob["model"]); m.eval()
    return m, blob.get("val")


def per_user_ndcg_by_turn(model, ar, Wce, tables, recs, fixed_seqs, cfg, ni, n_ce, off_item,
                          headmask, turns):
    """For each turn k in `turns` (int or 'full'), score all users -> dict k -> (fullN, tailN)."""
    out = {}
    for k in turns:
        if k == "full":
            seqs = [np.arange(ar.nQ) for _ in recs]
        else:
            seqs = [s[:k] for s in fixed_seqs]
        fN, tN = score_cohort(model, ar, Wce, tables, recs, seqs, cfg, ni, n_ce, off_item, headmask)
        out[k] = (fN, tN)
    return out


def early_auc(ndcg_by_turn, ks=EARLY_KS):
    """Per-user trapezoid area under NDCG@10-vs-turn over ks -> per-user array (None-safe)."""
    n = len(ndcg_by_turn[ks[0]][0])
    auc = np.full(n, np.nan)
    for i in range(n):
        ys = [ndcg_by_turn[k][0][i] for k in ks]
        if any(y is None for y in ys):
            continue
        auc[i] = np.trapz(ys, ks)
    return auc


def paired_boot(delta, n_boot=5000, seed=0):
    d = np.asarray([x for x in delta if np.isfinite(x)], float)
    if len(d) < 2:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=int(len(d)),
                    p_gt0=float("nan"))
    rng = np.random.default_rng(seed); bs = np.empty(n_boot)
    for b in range(n_boot):
        bs[b] = d[rng.integers(0, len(d), len(d))].mean()
    return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n=int(len(d)),
                p_gt0=float((bs > 0).mean()), se=float(bs.std()))


# ======================================================================== C3 full battery
def battery_c3(model, ar, Wce, tables, recs, ni, n_ce, off_item, headmask):
    """IG1 genre purity, IG2 genre polarity, IG4 graded-value monotonicity, hygiene."""
    res = {}
    Gmat = ar.uni.Gmat; GEN = G.GENRES; cnt = ar.uni.D["cnt"].astype(np.float64)
    active = active_ix_for("C3")

    def score_from_direct_items(item_vals):
        chan = {c: np.zeros((1, ni), np.float32) for c in
                ["value", "know", "ref", "source", "mask", "ease"]}
        for it, (v, kk, refuse) in item_vals.items():
            if refuse:
                chan["ref"][0, it] = 1.0; chan["mask"][0, it] = 1.0
            else:
                chan["value"][0, it] = v; chan["know"][0, it] = kk
                chan["source"][0, it] = 1.0; chan["mask"][0, it] = 1.0
        X = channels_to_input(chan, CONFIGS["C3"])
        with torch.no_grad():
            return model(torch.from_numpy(X), active)[0].numpy()[0].astype(np.float64)

    def pct_rank(scores, items):
        order = np.argsort(scores); ranks = np.empty(len(scores))
        ranks[order] = np.arange(len(scores)) / (len(scores) - 1)
        return float(np.mean(ranks[np.asarray(items, np.int64)]))

    probe_genres = ["Sci-Fi", "Horror", "Romance", "Documentary", "Children", "War"]
    NP = 12; flips = {}
    for g in probe_genres:
        gi = GEN.index(g); members = np.where(Gmat[:, gi] > 0)[0]
        if len(members) < NP * 2:
            continue
        pm = members[np.argsort(-cnt[members])]
        probe = pm[:NP]; held = pm[NP:NP + 200]
        s_like = score_from_direct_items({int(i): (1.0, 1.0, False) for i in probe})
        s_dis = score_from_direct_items({int(i): (-1.0, 1.0, False) for i in probe})
        pl = pct_rank(s_like, held); pd = pct_rank(s_dis, held)
        flips[g] = dict(pct_like=pl, pct_dislike=pd, flip_gap=pl - pd)
    res["IG1_IG2_genre"] = dict(
        flips=flips,
        mean_flip_gap=float(np.mean([f["flip_gap"] for f in flips.values()])) if flips else float("nan"),
        mean_like_purity=float(np.mean([f["pct_like"] for f in flips.values()])) if flips else float("nan"))

    gi = GEN.index("Sci-Fi"); members = np.where(Gmat[:, gi] > 0)[0]
    pm = members[np.argsort(-cnt[members])]; probe = pm[:NP]; held = pm[NP:NP + 200]
    sweep = {}
    for lvl, v in [("hated", -1.0), ("meh", -1 / 3), ("liked", 1 / 3), ("loved", 1.0)]:
        s = score_from_direct_items({int(i): (v, 1.0, False) for i in probe})
        sweep[lvl] = pct_rank(s, held)
    vals = [sweep[l] for l in ["hated", "meh", "liked", "loved"]]
    res["IG4_graded_sweep"] = dict(sweep=sweep,
                                   monotone=bool(all(vals[i] <= vals[i + 1] + 1e-6 for i in range(3))))

    hy = {}
    cold = score_from_direct_items({})
    hy["cold_pop_corr"] = float(np.corrcoef(cold, np.log(cnt + 1))[0, 1])
    r0 = recs[0]; seq = eval_interview(ar, Wce, r0, ni, n_ce, off_item)
    c1 = build_for_asked(ar, Wce, tables, [r0], [seq], ni, n_ce, off_item, False)
    c2 = build_for_asked(ar, Wce, tables, [r0], [seq[::-1]], ni, n_ce, off_item, False)
    hy["order_invariant"] = bool(np.allclose(c1["value"], c2["value"], atol=1e-5))
    leak = 0.0
    for r in recs[:200]:
        ch = build_for_asked(ar, Wce, tables, [r], [np.arange(ar.nQ)], ni, n_ce, off_item, False)
        h = list(r["held"])
        if h:
            leak += float(np.abs(ch["value"][0, h]).sum() + np.abs(ch["mask"][0, h]).sum())
    hy["held_input_leak_absmass"] = leak
    res["hygiene"] = hy
    return res


# ======================================================================== 173 real-LLM transfer
def build_173_channels(ar, Wce, users173, ni, n_ce, off_item):
    KIDX = {"no_clue": 0, "rough_idea": 1, "know_well": 2}
    VIDX = {"hated": 0, "meh": 1, "liked": 2, "loved": 3}
    recs = []; tables = {}
    for x in users173:
        know = np.zeros(ar.nQ, np.int8); val = np.full(ar.nQ, -1, np.int8)
        crval = np.full(ar.nbank, np.nan, np.float32); asked = []
        mu_known = float(np.mean(list(x["known"].values()))) if x["known"] else 3.5
        for (ch, rid, k, v, stars) in x["cells"]:
            if ch == "concept":
                q = rid
            elif ch in ("attribute", "entity"):
                q = ar.ntag + rid
            elif ch == "item":
                q = ar.off_item + rid
            else:
                continue
            kk = KIDX.get(k)
            if kk is None:
                continue
            know[q] = kk; asked.append(q)
            if kk >= 1 and v in VIDX:
                val[q] = VIDX[v]
            if ch == "item" and stars is not None and kk >= 1:
                crval[rid] = np.float32(float(stars) - mu_known)
        for (rid, stars) in x["data"]:
            if rid is None:
                continue
            q = ar.off_item + rid; know[q] = 2
            val[q] = 3 if stars >= 4.5 else 2 if stars >= 3.5 else 1 if stars >= 2.5 else 0
            crval[rid] = np.float32(float(stars) - mu_known); asked.append(q)
        tables[x["u"]] = dict(know=know, val=val, crval=crval)
        recs.append(dict(u=x["u"], known=x["known"], held=x["held"],
                         asked=np.array(sorted(set(asked)), np.int64)))
    return recs, tables


def transfer_173(models, ar, Wce, ni, n_ce, off_item, headmask):
    uni = ar.uni; split = G.build_split(uni.D); raw = DB.load_173(uni)
    users173 = []
    for x in raw:
        u = x["u"]
        if u not in split:
            continue
        kn, ho = split[u]; rat = dict(uni.D["rat_by_u"][u])
        held = set(int(j) for j in ho if rat.get(j, 0) >= LO)
        if not held or len(x["known"]) < 4:
            continue
        x["held"] = held; users173.append(x)
    recs, tables = build_173_channels(ar, Wce, users173, ni, n_ce, off_item)
    log(f"173 transfer: {len(recs)} usable users")
    stat = q_static_score(ar); fixed = []
    for r in recs:
        a = r["asked"]; fixed.append(a[np.argsort(-stat[a], kind="stable")])
    turns = EARLY_KS; res = {}; aucs = {}
    for cfg, m in models.items():
        nbt = per_user_ndcg_by_turn(m, ar, Wce, tables, recs, fixed, cfg, ni, n_ce, off_item,
                                    headmask, turns)
        aucs[cfg] = early_auc(nbt)
        res[cfg] = {str(k): float(np.nanmean([x for x in nbt[k][0] if x is not None])) for k in turns}
    res["C3_minus_C0_earlyAUC"] = paired_boot(aucs["C3"] - aucs["C0"], seed=173)
    res["n"] = len(recs)
    return res


# ======================================================================== EVAL (full analysis)
def cmd_eval(args):
    ctx = build_context(args)
    _eval_ctx(ctx, args)


def _eval_ctx(ctx, args):
    ar = ctx["ar"]; ni = ctx["ni"]; n_ce = ctx["n_ce"]; off_item = ctx["off_item"]
    Wce = ctx["Wce"]; headmask = ctx["headmask"]; coh = ctx["coh"]
    test = coh["test"]
    tables_te = prefill(ar, test, "test")
    cfgs = [c for c in ["C0", "C1", "C2", "C3", "C1e"] if os.path.exists(f"{OUTDIR}/{c}_best.pt")]
    log(f"eval configs present: {cfgs}")
    models = {c: load_model(c, ni)[0] for c in cfgs}

    # fixed PAIRED eval interviews (identical across configs)
    fixed = [eval_interview(ar, Wce, r, ni, n_ce, off_item) for r in test]
    turns_curve = [0, 1, 2, 4, 8, 16, "full"]
    A = {"n_test": len(test), "configs": cfgs, "turns": [str(t) for t in turns_curve]}

    # per-config k-curve + per-user early-AUC
    nbt_by_cfg = {}; auc_by_cfg = {}; ceil_by_cfg = {}
    for c in cfgs:
        nbt = per_user_ndcg_by_turn(models[c], ar, Wce, tables_te, test, fixed, c, ni, n_ce,
                                    off_item, headmask, turns_curve)
        nbt_by_cfg[c] = nbt
        auc_by_cfg[c] = early_auc(nbt)
        ceil = np.array([x if x is not None else np.nan for x in nbt["full"][0]])
        ceil_by_cfg[c] = ceil
        A[c] = {"kcurve_full": {str(k): float(np.nanmean([x for x in nbt[k][0] if x is not None]))
                                for k in turns_curve},
                "kcurve_tail": {str(k): float(np.nanmean([x for x in nbt[k][1] if x is not None]))
                                for k in turns_curve},
                "early_auc_mean": float(np.nanmean(auc_by_cfg[c]))}
        log(f"  {c}: early-AUC={np.nanmean(auc_by_cfg[c]):.4f} "
            f"full-answerer NDCG={A[c]['kcurve_full']['full']:.4f}")

    # PRIMARY: C3 - C0 early-AUC (paired, one-sided)
    if "C3" in cfgs and "C0" in cfgs:
        d = auc_by_cfg["C3"] - auc_by_cfg["C0"]
        A["PRIMARY_C3_minus_C0_earlyAUC"] = paired_boot(d, seed=1)
        # ceiling-normalized: each per-user curve / its own full-answerer NDCG, then early-AUC
        def norm_auc(c):
            nbt = nbt_by_cfg[c]; ceil = ceil_by_cfg[c]; n = len(test)
            out = np.full(n, np.nan)
            for i in range(n):
                if not (ceil[i] and np.isfinite(ceil[i]) and ceil[i] > 1e-6):
                    continue
                ys = [nbt[k][0][i] for k in EARLY_KS]
                if any(y is None for y in ys):
                    continue
                out[i] = np.trapz([y / ceil[i] for y in ys], EARLY_KS)
            return out
        A["PRIMARY_ceilnorm_C3_minus_C0"] = paired_boot(norm_auc("C3") - norm_auc("C0"), seed=2)
    if "C3" in cfgs and "C1" in cfgs:
        A["C3_minus_C1_earlyAUC"] = paired_boot(auc_by_cfg["C3"] - auc_by_cfg["C1"], seed=3)
    if "C3" in cfgs and "C1e" in cfgs:
        A["C3_minus_C1e_earlyAUC"] = paired_boot(auc_by_cfg["C3"] - auc_by_cfg["C1e"], seed=4)
    if "C1" in cfgs and "C0" in cfgs:
        A["C1_minus_C0_earlyAUC"] = paired_boot(auc_by_cfg["C1"] - auc_by_cfg["C0"], seed=5)

    # REFUSAL-rate robustness sweep on C3 vs C0 (force extra refusals)
    A["refusal_sweep"] = {}
    for rate in [0.0, 0.15, 0.30, 0.50]:
        rng = np.random.default_rng(int(rate * 100) + 7)
        tab_r = {}
        for u, t in tables_te.items():
            know = t["know"].copy(); val = t["val"].copy()
            ans = np.where(know >= 1)[0]
            if len(ans):
                flip = ans[rng.random(len(ans)) < rate]
                know[flip] = 0; val[flip] = -1
            tab_r[u] = dict(know=know, val=val, crval=t["crval"])
        row = {}
        for c in [x for x in ["C0", "C3"] if x in cfgs]:
            nbt = per_user_ndcg_by_turn(models[c], ar, Wce, tab_r, test, fixed, c, ni, n_ce,
                                        off_item, headmask, EARLY_KS)
            row[c] = float(np.nanmean(early_auc(nbt)))
            row[c + "_auc_arr"] = early_auc(nbt)
        if "C0" in row and "C3" in row:
            row["C3_minus_C0"] = paired_boot(row["C3_auc_arr"] - row["C0_auc_arr"], seed=99)
        for kk in ("C0_auc_arr", "C3_auc_arr"):
            row.pop(kk, None)
        A["refusal_sweep"][str(rate)] = row
        log(f"  refusal {rate}: " + " ".join(f"{c}={row.get(c, float('nan')):.4f}"
                                              for c in ("C0", "C3")))

    # C3 full battery
    if "C3" in cfgs:
        A["battery_C3"] = battery_c3(models["C3"], ar, Wce, tables_te, test, ni, n_ce, off_item,
                                     headmask)
        # strength anchor
        full = A["C3"]["kcurve_full"]["full"]; tail = A["C3"]["kcurve_tail"]["full"]
        A["battery_C3"]["strength"] = {"answerer_full_profile_full": full,
                                       "answerer_full_profile_tail": tail,
                                       "anchor_EASE_full": 0.510, "anchor_RecVAE_full": 0.526,
                                       "note": "answerer-full-profile != raw-rating full profile; "
                                       "distinct anchor (design 8-F). Cohort = arena trU population "
                                       "(not ruler.json te split)."}

    # 173 real-LLM transfer (arbiter)
    A["transfer_173"] = transfer_173(models, ar, Wce, ni, n_ce, off_item, headmask)

    pk, cur = log_mem("eval done")
    A["mem"] = {"peak_resident_gb": float(_PEAK_GB), "final_resident_gb": float(cur),
                "max_train": int(getattr(args, "max_train", -1)),
                "note": "peak working set over the whole process; streaming TableStore keeps the "
                        "N x 2428 answer tables on disk (memmap), only per-batch rows resident."}
    json.dump(A, open(f"{OUTDIR}/analysis.json", "w"), indent=1, default=float)
    log(f"wrote {OUTDIR}/analysis.json")
    write_report(A)
    return A


def write_report(A):
    L = ["# RICH TWO-AXIS SIGNAL -- analysis", "", f"Test users (population trU): {A['n_test']}. "
         f"Configs: {A['configs']}.", "",
         "Primary statistic = paired per-user EARLY-AUC (trapezoid of NDCG@10 over turns k in "
         "{1,2,4,8}). Pre-registered confirmatory: C3-C0 early-AUC > 0, one-sided paired.", ""]
    L.append("## Per-config early-AUC + full-answerer NDCG@10")
    L.append("| config | early-AUC | k=1 | k=2 | k=4 | k=8 | k=16 | full-answerer |")
    L.append("|--|--|--|--|--|--|--|--|")
    for c in A["configs"]:
        kc = A[c]["kcurve_full"]
        L.append(f"| {c} | {A[c]['early_auc_mean']:.4f} | {kc['1']:.4f} | {kc['2']:.4f} | "
                 f"{kc['4']:.4f} | {kc['8']:.4f} | {kc['16']:.4f} | {kc['full']:.4f} |")
    L.append("")

    def fmt(d):
        return (f"mean **{d['mean']:+.4f}** [95% CI {d['lo']:+.4f}, {d['hi']:+.4f}], "
                f"P(>0)={d['p_gt0']:.3f}, n={d['n']}")
    L.append("## PRIMARY contrast")
    if "PRIMARY_C3_minus_C0_earlyAUC" in A:
        d = A["PRIMARY_C3_minus_C0_earlyAUC"]
        verdict = "PASS (CI excludes 0, positive)" if d["lo"] > 0 else (
            "FAIL (CI includes 0)" if d["hi"] > 0 else "FAIL (negative)")
        L.append(f"- **C3 - C0 early-AUC**: {fmt(d)} -> pre-registered verdict: **{verdict}**")
        if "PRIMARY_ceilnorm_C3_minus_C0" in A:
            L.append(f"- ceiling-normalized C3 - C0: {fmt(A['PRIMARY_ceilnorm_C3_minus_C0'])} "
                     "(speed, not asymptote)")
    L.append("")
    L.append("## Channel-contribution contrasts (descriptive)")
    for key, lab in [("C1_minus_C0_earlyAUC", "C1 - C0 (value adds over mask)"),
                     ("C3_minus_C1_earlyAUC", "C3 - C1 (knowledge adds over value)"),
                     ("C3_minus_C1e_earlyAUC", "C3 - C1e (rich beats EASE-plumbing control)")]:
        if key in A:
            L.append(f"- {lab}: {fmt(A[key])}")
    L.append("")
    L.append("## Refusal-rate robustness (early-AUC)")
    L.append("| forced refusal | C0 | C3 | C3-C0 (CI) |")
    L.append("|--|--|--|--|")
    for rate, row in A["refusal_sweep"].items():
        d = row.get("C3_minus_C0", {})
        cis = f"{d.get('mean', float('nan')):+.4f} [{d.get('lo', float('nan')):+.4f}, {d.get('hi', float('nan')):+.4f}]" if d else "-"
        L.append(f"| {rate} | {row.get('C0', float('nan')):.4f} | {row.get('C3', float('nan')):.4f} | {cis} |")
    L.append("")
    if "battery_C3" in A:
        b = A["battery_C3"]
        L.append("## C3 full battery")
        s = b["strength"]
        L.append(f"- **Strength** (answerer-full-profile): full {s['answerer_full_profile_full']:.4f} / "
                 f"tail {s['answerer_full_profile_tail']:.4f}  vs EASE {s['anchor_EASE_full']} / "
                 f"RecVAE {s['anchor_RecVAE_full']} (distinct anchor; see note).")
        g2 = b["IG1_IG2_genre"]
        L.append(f"- **IG1/IG2 genre**: mean like-purity {g2['mean_like_purity']:.3f}, mean "
                 f"like-vs-dislike flip gap {g2['mean_flip_gap']:+.3f}")
        sw = b["IG4_graded_sweep"]
        L.append(f"- **IG4 graded-value sweep** (Sci-Fi pct-rank): "
                 + " ".join(f"{k}={sw['sweep'][k]:.3f}" for k in ['hated', 'meh', 'liked', 'loved'])
                 + f" -> monotone={sw['monotone']}")
        hy = b["hygiene"]
        L.append(f"- **Hygiene**: cold~pop corr {hy['cold_pop_corr']:.3f}; order-invariant "
                 f"{hy['order_invariant']}; held-input-leak abs-mass {hy['held_input_leak_absmass']:.2e} "
                 f"(0 = leak-free).")
    L.append("")
    t = A["transfer_173"]
    L.append("## 173 real-LLM transfer (arbiter; low power)")
    d = t["C3_minus_C0_earlyAUC"]
    L.append(f"- n={t['n']} real-LLM users. **C3 - C0 early-AUC**: {fmt(d)}")
    L.append(f"- per-config early NDCG@10 by turn: " +
             "; ".join(f"{c}: " + "/".join(f"{t[c][str(k)]:.3f}" for k in EARLY_KS)
                       for c in A["configs"] if c in t))
    L.append("")
    if "mem" in A:
        m = A["mem"]
        L.append(f"## Memory (streaming fix)")
        L.append(f"- Peak resident (working set) over the run: **{m['peak_resident_gb']:.2f} GB** "
                 f"(max_train={m['max_train']}). Answer tables are memmap'd on disk; only per-batch "
                 f"rows are resident. The old full-dict prefill OOM'd at 40k on this 16 GB box.")
        L.append("")
    L.append("_Caveats: concept/unrated values are EASE-derived (C1/C3 upper bound; C1e is the "
             "circularity arbiter). ease_v21 B saw pop val/test held-halves (diffuse at 161k). "
             "Answerer v2.1 dial-refit on the 173 (transfer distribution tuned; not label leak)._")
    open(f"C:/dev/phd/casper/RICH_SIGNAL_ANALYSIS.md", "w", encoding="utf-8").write("\n".join(L))
    log("wrote RICH_SIGNAL_ANALYSIS.md")


# ======================================================================== validate faithfulness
def cmd_validate(args):
    ar = AC.Arena(verbose=True)
    coh = build_cohorts(ar, 0, 0, 12)
    recs = coh["train"][:12]
    pred = ease_pred_batch(ar, [r["known"] for r in recs])
    ok = True
    for a, r in enumerate(recs):
        t_slow = ar._gen_user(r["u"], r["known"])
        t_fast = gen_user_fast(ar, r["u"], r["known"], pred[a])
        for k in ("know", "val"):
            if not np.array_equal(t_slow[k], t_fast[k]):
                ok = False; log(f"MISMATCH user {r['u']} {k}")
        cs = np.nan_to_num(t_slow["crval"]); cf = np.nan_to_num(t_fast["crval"])
        if not np.allclose(cs, cf, atol=1e-4):
            ok = False; log(f"MISMATCH user {r['u']} crval")
    log(f"FAITHFULNESS {'PASS' if ok else 'FAIL'} (fast gen == arena _gen_user on 12 users)")

    # ---- streaming round-trip faithfulness: memmap store rows == direct gen, bit-identical ----
    import shutil
    for junk in ("mm_vstream_know.npy", "mm_vstream_val.npy", "mm_vstream_crval.npy",
                 "mm_vstream_uids.npy"):
        try:
            os.remove(f"{OUTDIR}/{junk}")
        except OSError:
            pass
    shutil.rmtree(f"{OUTDIR}/shards_vstream", ignore_errors=True)
    store = prefill(ar, recs, "vstream", chunk=5)      # tiny chunk -> exercises multi-shard streaming
    ok2 = True
    for a, r in enumerate(recs):
        direct = gen_user_fast(ar, r["u"], r["known"], pred[a])
        row = store[r["u"]]
        for k in ("know", "val"):
            if not np.array_equal(np.asarray(direct[k]), np.asarray(row[k])):
                ok2 = False; log(f"STREAM MISMATCH user {r['u']} {k}")
        if not np.array_equal(np.nan_to_num(direct["crval"]),
                              np.nan_to_num(np.asarray(row["crval"]))):
            ok2 = False; log(f"STREAM MISMATCH user {r['u']} crval")
    log(f"STREAM ROUNDTRIP {'PASS' if ok2 else 'FAIL'} (memmap store rows == direct gen, "
        f"bit-identical on {len(recs)} users)")
    for junk in ("mm_vstream_know.npy", "mm_vstream_val.npy", "mm_vstream_crval.npy",
                 "mm_vstream_uids.npy"):
        try:
            os.remove(f"{OUTDIR}/{junk}")
        except OSError:
            pass
    shutil.rmtree(f"{OUTDIR}/shards_vstream", ignore_errors=True)
    return ok and ok2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "prefill", "train", "eval", "full"])
    ap.add_argument("--tag", default="C3")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n_val", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--max_train", type=int, default=150000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "prefill":
        ar = AC.Arena(verbose=True)
        coh = build_cohorts(ar, args.n_val, args.n_test, args.max_train)
        prefill(ar, coh["val"], "val"); prefill(ar, coh["test"], "test")
        prefill(ar, coh["train"], "train")
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "full":
        cmd_full(args)
