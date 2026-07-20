"""signed_latent.py -- SIGNED LATENT RECOMMENDER for cold-start conversational movie rec (CASPER, ML-25M).

A deep autoencoder-style recommender with a continuous latent z (d=512) and a SIGNED ranking objective:
  ENCODER: signed per-item value vector (rated -> [-1,+1] from 0.5-5 stars; unrated=0) (+ optional
           consumed-mask channel) -> L2-norm -> denoising dropout -> 5 dense blocks (width 600) -> z.
  DECODER: Linear(z -> item logits).
  LOSS = L_pos (multinomial over LIKED items; the RecVAE/EASE strength source)
       + alpha * L_neg (margin: score(unrated) > score(disliked); gentle dislike expression).
  DENOISING + STRATEGY-VARIED interview CURRICULUM (item-level): full profiles + log-uniform sampled
  interviews, strategy mix {random/pop/entropy/on-profile/off-profile/adversarial/mixed}, refusals.

NOT RecVAE (positive-only) and NOT EASE (no latent): this expresses DISLIKE and carries a latent.

RULER (make-or-break): full-catalog held-liked NDCG@10 on the byte-validated ml25m_arena half-split.
  ANCHORS on this SAME ruler: MOSTPOP 0.2522, RecVAE d512 0.4998 (docs), EASE reproduced here.
  ONE ruler for full-profile AND short-k (shared held set -> full-profile >= short-k by construction).

Quarantine: 300 LLM study users (subset of te) NEVER in train/val/test. NO DATA CAPS.
Durable ckpts + peak -> .cache/signed_latent/.
"""
import os, sys, json, time, argparse, math
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count())
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "instrument2"))
sys.path.insert(0, _HERE)

META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
STUDY = "C:/dev/phd/casper/.cache/instrument2/answerability_grid_ml25m.json"
RECVAE_CKPT = "C:/dev/phd/casper/.cache/instrument2/ml25m_recvae_d512_best.pt"
OUTDIR = "C:/dev/phd/casper/.cache/signed_latent"
os.makedirs(OUTDIR, exist_ok=True)

LO, HI = 4.0, 2.0                      # liked / disliked thresholds (project canon)
SEEDS = [1, 2, 3, 7, 11]               # arena seed-avg protocol
_W = 1.0 / np.log2(np.arange(2, 12))   # NDCG@10 gains


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def safe_save(obj, path, min_free_mb=500):
    """Disk-frugal, crash-safe checkpoint save: skip if <min_free_mb free; delete-before-write to a
    temp then atomic rename; NEVER raise (log + continue). Prevents disk-full from killing training."""
    import shutil
    try:
        free_mb = shutil.disk_usage(os.path.dirname(path) or ".").free / 1e6
    except OSError:
        free_mb = 0.0
    if free_mb < min_free_mb:
        log(f"  [warn] only {free_mb:.0f} MB free (<{min_free_mb}); SKIP save {os.path.basename(path)}")
        return False
    tmp = path + ".tmp"
    try:
        if os.path.exists(path):
            os.remove(path)                       # delete-before-write (keep only ONE file)
        torch.save(obj, tmp)
        os.replace(tmp, path)                     # atomic
        return True
    except (OSError, RuntimeError) as e:
        log(f"  [warn] save failed for {os.path.basename(path)}: {e}; CONTINUING")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


# ======================================================================== value scaling
def scale_rating(r):
    """0.5-5 star -> [-1,+1].  0.5->-1, 2.75->0, 5->+1.  Vectorized."""
    return (np.asarray(r, np.float64) - 2.75) / 2.25


# ======================================================================== arena / data
def load_arena_base():
    """Load full ML-25M meta + build train-like CSR (trU x ni binary r>=4) + eval rating dicts
    for va+te users. Reuses the byte-stable meta.npz split (ml25m_arena convention)."""
    d = np.load(META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float32)
    ni = int(d["ni"]); nu = int(d["nu"])
    trU = d["trU"].astype(np.int64); va = d["va"].astype(np.int64); te = d["te"].astype(np.int64)
    cnt = d["cnt"].astype(np.float64)
    popb = np.log(cnt + 1.0).astype(np.float32)
    order_pop = np.argsort(-cnt); cum = np.cumsum(cnt[order_pop]) / cnt.sum()
    headmask = np.zeros(ni, bool); headmask[order_pop[:np.searchsorted(cum, 0.33) + 1]] = True
    # quarantine
    study = set(int(u) for u in json.load(open(STUDY))["users"].keys())
    assert study <= set(te.tolist()), "study cohort must be subset of te"
    # train-like rows (r>=4, train users) -> signed values too (for signed AE input)
    trmask = np.zeros(nu, bool); trmask[trU] = True
    like = rr >= LO
    sel_tr_like = like & trmask[uu]
    tr_u = uu[sel_tr_like]; tr_i = ii[sel_tr_like]
    # FULL train ratings (all bands) for signed train matrix
    sel_tr_all = trmask[uu]
    tra_u = uu[sel_tr_all]; tra_i = ii[sel_tr_all]; tra_r = rr[sel_tr_all]
    # eval cohorts: full rating dicts for va + te users
    evalset = np.zeros(nu, bool); evalset[va] = True; evalset[te] = True
    sel_ev = evalset[uu]
    ev_u = uu[sel_ev]; ev_i = ii[sel_ev]; ev_r = rr[sel_ev]
    rat_by_u = {}
    for k in range(len(ev_u)):
        rat_by_u.setdefault(int(ev_u[k]), []).append((int(ev_i[k]), float(ev_r[k])))
    return dict(ni=ni, nu=nu, trU=trU, va=va, te=te, study=study, cnt=cnt, popb=popb,
                headmask=headmask, tr_u=tr_u, tr_i=tr_i,
                tra_u=tra_u, tra_i=tra_i, tra_r=tra_r, rat_by_u=rat_by_u)


def build_splits(base, seed):
    """Per-user half-split (arena convention): shuffle unique rated items, first half = profile
    (fold-in + exclusion), second half held-out; tlike = held r>=4. Requires >=6 rated items.
    Returns SPL {u: (profset, held_list, prof_ratings{}, held_ratings{})}."""
    rat_by_u = base["rat_by_u"]
    rs = np.random.default_rng(seed)
    SPL = {}
    for u, v in rat_by_u.items():
        dd = dict(v)                       # unique item -> rating (last wins; items unique anyway)
        its = list(dd.keys())
        if len(its) < 6:
            continue
        il = its[:]; rs.shuffle(il)
        half = len(il) // 2
        prof = il[:half]; held = il[half:]
        profset = set(prof)
        prof_r = {j: dd[j] for j in prof}
        held_r = {j: dd[j] for j in held}
        SPL[u] = (profset, held, prof_r, held_r)
    return SPL


def cohort(base, SPL, which):
    """which='val' -> va users in SPL; 'test' -> te MINUS study users in SPL."""
    if which == "val":
        pool = base["va"].tolist()
        return [u for u in pool if u in SPL]
    pool = [u for u in base["te"].tolist() if u not in base["study"]]
    return [u for u in pool if u in SPL]


def ndcg10(score, tlike, profset, headmask, tail):
    """Full-catalog NDCG@10 (arena metric, identical to ml25m_arena.ndcg_at10)."""
    s = score.copy()
    s[list(profset)] = -1e30
    if tail:
        s[headmask] = -1e30
        rel = set(t for t in tlike if not headmask[t])
    else:
        rel = set(tlike)
    if not rel:
        return None
    o = np.argsort(-s)[:10]
    dcg = sum(_W[p] for p, t in enumerate(o) if int(t) in rel)
    idcg = _W[:min(10, len(rel))].sum() + 1e-12
    return dcg / idcg


# ======================================================================== EASE ruler
def build_gram(base, chunk=8000):
    """G = X^T X, X = trU x ni binary likes. Chunked dense torch float32 accumulation (no cap)."""
    ni = base["ni"]
    # CSR of train likes
    data = np.ones(len(base["tr_u"]), np.float32)
    # dense-remap train users to 0..K-1
    uniqU, inv = np.unique(base["tr_u"], return_inverse=True)
    X = sp.csr_matrix((data, (inv, base["tr_i"])), shape=(len(uniqU), ni), dtype=np.float32)
    log(f"EASE gram: X = {X.shape[0]} train users x {ni} items, nnz={X.nnz}")
    G = torch.zeros((ni, ni), dtype=torch.float32)
    n = X.shape[0]
    for b in range(0, n, chunk):
        Xc = torch.from_numpy(X[b:b + chunk].toarray())
        G.addmm_(Xc.T, Xc)
        if (b // chunk) % 3 == 0:
            log(f"    gram {min(b + chunk, n)}/{n}")
    return G


def ease_B(G, lam):
    ni = G.shape[0]
    A = G.clone(); A.diagonal().add_(lam)
    P = torch.linalg.inv(A); del A
    dP = torch.diag(P).clone()
    P /= -dP[None, :]
    P.fill_diagonal_(0.0)
    return P     # torch float32 (ni x ni), B[j,i]


def eval_ease(B, base, SPL, users, revealed="full", k=None, alpha=0.0, seedoff=0):
    """Full-catalog held-liked NDCG@10. revealed='full' -> all profile-half likes; 'k' -> k random
    profile-half likes; alpha>0 -> add profile-half dislikes as -alpha entries (GATE 3)."""
    headmask = base["headmask"]
    Bn = B.numpy()
    full_n, full_t = [], []
    rng = np.random.default_rng(1234 + seedoff)
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        liked_prof = [j for j in prof_r if prof_r[j] >= LO]
        dis_prof = [j for j in prof_r if prof_r[j] <= HI]
        if revealed == "full":
            idx = np.array(liked_prof, np.int64); w = np.ones(len(idx), np.float32)
        else:
            if len(liked_prof) < k:
                continue
            sel = rng.choice(len(liked_prof), size=k, replace=False)
            idx = np.array([liked_prof[i] for i in sel], np.int64); w = np.ones(len(idx), np.float32)
        if alpha > 0 and dis_prof:
            idx = np.concatenate([idx, np.array(dis_prof, np.int64)])
            w = np.concatenate([w, -alpha * np.ones(len(dis_prof), np.float32)])
        if len(idx) == 0:
            continue
        score = (w @ Bn[idx, :]).astype(np.float64)
        nf = ndcg10(score, tlike, profset, headmask, False)
        nt = ndcg10(score, tlike, profset, headmask, True)
        if nf is not None: full_n.append(nf)
        if nt is not None: full_t.append(nt)
    return (float(np.mean(full_n)) if full_n else float("nan"),
            float(np.mean(full_t)) if full_t else float("nan"), len(full_n))


# ======================================================================== RecVAE ruler reproduction
def eval_recvae(base, SPL, users, revealed="full", k=None):
    """Reproduce RecVAE d512 full-profile NDCG@10 on THIS harness (ruler cross-check)."""
    from recvae import RecVAE
    blob = torch.load(RECVAE_CKPT, map_location="cpu")
    a = blob["args"]; model = RecVAE(a["hidden"], a["latent"], base["ni"])
    model.load_state_dict(blob["model"]); model.eval()
    ni = base["ni"]; headmask = base["headmask"]
    full_n, full_t = [], []
    rng = np.random.default_rng(999)
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        liked_prof = [j for j in prof_r if prof_r[j] >= LO]
        if revealed == "full":
            idx = liked_prof
        else:
            if len(liked_prof) < k:
                continue
            sel = rng.choice(len(liked_prof), size=k, replace=False)
            idx = [liked_prof[i] for i in sel]
        x = torch.zeros((1, ni), dtype=torch.float32)
        if idx:
            x[0, idx] = 1.0
        with torch.no_grad():
            score = model(x, calculate_loss=False).numpy()[0].astype(np.float64)
        nf = ndcg10(score, tlike, profset, headmask, False)
        nt = ndcg10(score, tlike, profset, headmask, True)
        if nf is not None: full_n.append(nf)
        if nt is not None: full_t.append(nt)
    return (float(np.mean(full_n)) if full_n else float("nan"),
            float(np.mean(full_t)) if full_t else float("nan"), len(full_n))


def cmd_ruler(args):
    """Validate the eval: reproduce RecVAE 0.4998 + EASE anchor + MOSTPOP on the arena half-split."""
    base = load_arena_base()
    ni = base["ni"]
    # MOSTPOP + RecVAE + EASE, seed-avg over SEEDS
    out = {"dataset": "ML-25M", "ni": ni, "eval": "full-catalog held-liked NDCG@10, arena half-split",
           "cohort_test": "te(500) minus 300 study users", "seeds": SEEDS}

    # ---- MOSTPOP ----
    mp_f, mp_t = [], []
    for seed in SEEDS:
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        popb = base["popb"]; headmask = base["headmask"]
        ff, tt = [], []
        for u in users:
            profset, held, prof_r, held_r = SPL[u]
            tlike = [j for j in held if held_r[j] >= LO]
            if not tlike: continue
            nf = ndcg10(popb.astype(np.float64).copy(), tlike, profset, headmask, False)
            nt = ndcg10(popb.astype(np.float64).copy(), tlike, profset, headmask, True)
            if nf is not None: ff.append(nf)
            if nt is not None: tt.append(nt)
        mp_f.append(np.mean(ff)); mp_t.append(np.mean(tt))
    out["MOSTPOP"] = {"full": float(np.mean(mp_f)), "tail": float(np.mean(mp_t))}
    log(f"MOSTPOP full={np.mean(mp_f):.4f} tail={np.mean(mp_t):.4f}  (doc 0.2522/0.0502)")

    # ---- RecVAE reproduction ----
    rv_f, rv_t = [], []
    for seed in SEEDS:
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        f, t, n = eval_recvae(base, SPL, users, "full")
        rv_f.append(f); rv_t.append(t)
        log(f"  RecVAE seed{seed} full={f:.4f} tail={t:.4f} n={n}")
    out["RecVAE_d512"] = {"full": float(np.mean(rv_f)), "tail": float(np.mean(rv_t))}
    log(f"RecVAE full={np.mean(rv_f):.4f} tail={np.mean(rv_t):.4f}  (doc 0.4998/0.3443)")

    # ---- EASE: build gram once, sweep lambda on VAL, eval TEST ----
    G = build_gram(base)
    lam_grid = [100.0, 250.0, 500.0, 1000.0, 2000.0]
    val_by_lam = {}
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    for lam in lam_grid:
        B = ease_B(G, lam)
        vf, vt, vn = eval_ease(B, base, SPLv, vusers, "full")
        val_by_lam[lam] = vf
        log(f"  EASE lam={lam} VAL full={vf:.4f} (n={vn})")
        del B
    best_lam = max(val_by_lam, key=val_by_lam.get)
    log(f"EASE best lam={best_lam} (val {val_by_lam[best_lam]:.4f})")
    B = ease_B(G, best_lam)
    ef, et = [], []
    for seed in SEEDS:
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        f, t, n = eval_ease(B, base, SPL, users, "full")
        ef.append(f); et.append(t)
        log(f"  EASE seed{seed} TEST full={f:.4f} tail={t:.4f} n={n}")
    out["EASE"] = {"full": float(np.mean(ef)), "tail": float(np.mean(et)), "lambda": best_lam,
                   "val_by_lambda": val_by_lam}
    log(f"EASE TEST full={np.mean(ef):.4f} tail={np.mean(et):.4f}")

    # ruler verdict
    ease_full = out["EASE"]["full"]; rv_full = out["RecVAE_d512"]["full"]
    ruler_ok = (abs(ease_full - rv_full) < 0.03) and (ease_full > 0.45)
    out["ruler_trustworthy"] = bool(ruler_ok)
    out["ruler_note"] = (f"EASE {ease_full:.4f} vs RecVAE {rv_full:.4f} (doc 0.4998). ML-25M EASE-class "
                         f"~0.50, NOT the ML-1M 0.55 the brief cited. Ruler OK if EASE~RecVAE and >0.45.")
    json.dump(out, open(os.path.join(OUTDIR, "ruler.json"), "w"), indent=2)
    # save gram-B best for reuse in gates
    np.save(os.path.join(OUTDIR, "ease_B_best.npy"), B.numpy())
    json.dump({"lambda": best_lam}, open(os.path.join(OUTDIR, "ease_meta.json"), "w"))
    log(f"RULER trustworthy={ruler_ok}. wrote ruler.json")
    return out


# ======================================================================== signed AE model
def swish(x):
    return x.mul(torch.sigmoid(x))


class SignedEncoder(nn.Module):
    """RecVAE-class dense-connected encoder (5 blocks width 600, swish, LayerNorm), signed input +
    consumed-mask channel, L2-norm + denoising dropout -> latent z (deterministic AE head)."""
    def __init__(self, ni, hidden=600, latent=512, use_mask=True):
        super().__init__()
        self.use_mask = use_mask
        din = 2 * ni if use_mask else ni
        self.fc1 = nn.Linear(din, hidden); self.ln1 = nn.LayerNorm(hidden, eps=0.1)
        self.fc2 = nn.Linear(hidden, hidden); self.ln2 = nn.LayerNorm(hidden, eps=0.1)
        self.fc3 = nn.Linear(hidden, hidden); self.ln3 = nn.LayerNorm(hidden, eps=0.1)
        self.fc4 = nn.Linear(hidden, hidden); self.ln4 = nn.LayerNorm(hidden, eps=0.1)
        self.fc5 = nn.Linear(hidden, hidden); self.ln5 = nn.LayerNorm(hidden, eps=0.1)
        self.fc_z = nn.Linear(hidden, latent)

    def forward(self, xv, dropout=0.0):
        # xv: (B, ni) signed values. mask channel = (xv != 0).
        if self.use_mask:
            m = (xv != 0).to(xv.dtype)
            x = torch.cat([xv, m], dim=-1)
        else:
            x = xv
        norm = x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        x = x / norm
        x = F.dropout(x, p=dropout, training=self.training)
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_z(h5)


class SignedAE(nn.Module):
    def __init__(self, ni, hidden=600, latent=512, use_mask=True):
        super().__init__()
        self.encoder = SignedEncoder(ni, hidden, latent, use_mask)
        self.decoder = nn.Linear(latent, ni)
        self.ni = ni

    def forward(self, xv, dropout=0.0):
        z = self.encoder(xv, dropout)
        return self.decoder(z), z

    def encode(self, xv):
        return self.encoder(xv, 0.0)


# ======================================================================== train-user data (signed)
def build_train_users(base):
    """Per train user: item ids + signed values + liked/disliked lists. Full data, no cap."""
    tra_u = base["tra_u"]; tra_i = base["tra_i"]; tra_r = base["tra_r"]
    order = np.argsort(tra_u, kind="stable")
    tra_u = tra_u[order]; tra_i = tra_i[order]; tra_r = tra_r[order]
    users = []
    N = len(tra_u); b = 0
    while b < N:
        e = b
        while e < N and tra_u[e] == tra_u[b]:
            e += 1
        its = tra_i[b:e].astype(np.int64); rs = tra_r[b:e].astype(np.float64)
        srt = np.argsort(its)                        # sort by item for searchsorted value lookup
        its = its[srt]; rs = rs[srt]
        liked = its[rs >= LO]
        if len(liked) >= 2:                          # need >=2 likes to hold one out
            users.append(dict(items=its, sv=scale_rating(rs).astype(np.float32),
                              liked=liked, disliked=its[rs <= HI]))
        b = e
    return users


# ---- item informativeness for entropy/EIG-style strategies (popularity-derived, cheap) ----
def item_info(base):
    cnt = base["cnt"]
    p = cnt / (cnt.sum() + 1e-9)
    # binary entropy of "popular vs not" proxy -> peaks at mid popularity (informative middle)
    pr = np.clip(cnt / (cnt.max() + 1e-9), 1e-6, 1 - 1e-6)
    ent = -(pr * np.log(pr) + (1 - pr) * np.log(1 - pr))
    return ent


STRATEGIES = ["random", "pop", "entropy", "on_profile", "off_profile", "adversarial", "mixed"]
STRAT_W = np.array([0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10])


def _gumbel_topk(pool, k, logw, rng):
    """Fast weighted sampling WITHOUT replacement via Gumbel-top-k (O(n))."""
    if k >= len(pool):
        return pool
    keys = logw + rng.gumbel(size=len(pool))
    idx = np.argpartition(-keys, k)[:k]
    return pool[idx]


def select_items(u, k, strat, rng, info):
    """Select k of the user's rated items by strategy (item-level interview curriculum). Fast."""
    its = u["items"]; n = len(its)
    if k >= n:
        return its
    if strat == "random":
        return _gumbel_topk(its, k, np.zeros(n), rng)
    if strat == "pop":
        return _gumbel_topk(its, k, np.log(_POP[its] + 1.0), rng)
    if strat == "entropy":
        return _gumbel_topk(its, k, np.log(info[its] + 1e-6), rng)
    if strat == "on_profile":
        L = u["liked"]
        pool = L if len(L) >= k else its
        return _gumbel_topk(pool, k, np.zeros(len(pool)), rng)
    if strat == "off_profile":
        D = u["disliked"]
        pool = D if len(D) >= k else its
        return _gumbel_topk(pool, k, np.zeros(len(pool)), rng)
    if strat == "adversarial":
        return _gumbel_topk(its, k, -np.log(info[its] + 1e-3), rng)
    return _gumbel_topk(its, k, np.zeros(n), rng)   # mixed default


_POP = None


_CUM_STRAT = np.cumsum(STRAT_W)


def make_input_target(u, rng, info, w_clean=0.4, p_ans=0.85):
    """Build ONE curriculum training example: signed input spec + held-liked target (numpy arrays).
    Leak-free: target = liked NOT in input. Fast (searchsorted value lookup, array setdiff)."""
    its = u["items"]; n = len(its)
    if rng.random() < w_clean:
        drop = rng.uniform(0.0, 0.5)
        in_items = its[rng.random(n) >= drop]
    else:
        k = int(round(math.exp(rng.uniform(0.0, math.log(max(2, n))))))
        k = max(1, min(k, n))
        strat = STRATEGIES[np.searchsorted(_CUM_STRAT, rng.random())]
        sel = np.asarray(select_items(u, k, strat, rng, info))
        in_items = sel[rng.random(len(sel)) < p_ans] if len(sel) else sel
    in_items = np.unique(in_items.astype(np.int64))
    in_val = u["sv"][np.searchsorted(its, in_items)] if len(in_items) else np.zeros(0, np.float32)
    target = np.setdiff1d(u["liked"], in_items, assume_unique=False)
    if len(target) == 0:
        target = u["liked"]
    return in_items, in_val, target


def cmd_train(args):
    global _POP
    base = load_arena_base(); ni = base["ni"]
    _POP = base["cnt"].astype(np.float64)
    info = item_info(base)
    users = build_train_users(base)
    log(f"train users={len(users)} (>=2 likes), ni={ni}, alpha={args.alpha}")
    model = SignedAE(ni, use_mask=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    tag = args.tag
    ckpt = os.path.join(OUTDIR, f"{tag}.pt"); best_ckpt = os.path.join(OUTDIR, f"{tag}_best.pt")
    peak_txt = os.path.join(OUTDIR, f"{tag}_peak.txt")
    start_ep = 0; best_val = -1.0; history = []
    if args.resume and os.path.exists(ckpt):
        try:
            blob = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(blob["model"])
            if "opt" in blob:
                opt.load_state_dict(blob["opt"])
            start_ep = blob.get("epoch", 0); history = blob.get("history", [])
            best_val = max([h.get("val_full", -1.0) for h in history], default=-1.0)
            log(f"RESUMED ep{start_ep} best_val={best_val:.4f}")
        except (RuntimeError, OSError, KeyError, EOFError) as e:
            log(f"  [warn] resume failed ({e}); starting FRESH")
            start_ep = 0; best_val = -1.0; history = []

    B = 256; m_margin = 0.5; n_neg = 4
    rng = np.random.default_rng(20260711 + int(args.alpha * 1000))
    # val cohort for early stop (arena val users)
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    idxlist = np.arange(len(users))

    for ep in range(start_ep, args.epochs):
        model.train(); rng2 = np.random.default_rng(ep * 7919 + int(args.alpha * 1000))
        np.random.default_rng(ep).shuffle(idxlist)
        t0 = time.time(); running = 0.0; nb = 0
        for st in range(0, len(users), B):
            bat = idxlist[st:st + B]
            xv = torch.zeros((len(bat), ni), dtype=torch.float32)
            tgt = torch.zeros((len(bat), ni), dtype=torch.float32)      # multinomial positives (liked held)
            neg_idx = []; pos_present = []
            for r, ui in enumerate(bat):
                u = users[ui]
                in_idx, in_val, target = make_input_target(u, rng2, info)
                if len(in_idx):
                    xv[r, in_idx] = torch.from_numpy(in_val)
                if len(target):
                    tgt[r, target] = 1.0
                    pos_present.append(r)
                # dislikes for margin
                D = u["disliked"]
                neg_idx.append(D)
            logits, z = model(xv, dropout=0.0)
            logsm = F.log_softmax(logits, dim=-1)
            denom = tgt.sum(-1).clamp_min(1.0)
            L_pos = -((logsm * tgt).sum(-1) / denom).mean()
            # dislike margin: score(unrated) > score(disliked)
            L_neg = torch.zeros(())
            if args.alpha > 0:
                terms = []
                for r, D in enumerate(neg_idx):
                    if len(D) == 0:
                        continue
                    dsel = D[rng2.integers(0, len(D), size=min(n_neg, len(D)))]
                    usel = rng2.integers(0, ni, size=n_neg)
                    sd = logits[r, dsel]
                    su = logits[r, usel]
                    terms.append(F.relu(m_margin + sd.mean() - su.mean()))
                if terms:
                    L_neg = torch.stack(terms).mean()
            loss = L_pos + args.alpha * L_neg
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss); nb += 1
            if nb % 100 == 0:
                log(f"  ep{ep} batch {nb}/{(len(users)+B-1)//B} loss={running/nb:.4f} {(time.time()-t0)/60:.1f}m")
        # ---- val (full-profile NDCG@10 on arena val cohort) ----
        vf = eval_model(model, base, SPLv, vusers, "full")[0]
        history.append({"epoch": ep + 1, "loss": running / max(nb, 1), "val_full": vf})
        log(f"[ep{ep+1}] loss={running/max(nb,1):.4f} VAL full={vf:.4f} ({(time.time()-t0)/60:.1f}m)")
        # crash-safe, disk-frugal (model-only; delete-before-write; disk check; never crash on save)
        safe_save({"model": model.state_dict(), "epoch": ep + 1, "history": history,
                   "alpha": args.alpha}, ckpt)
        if vf > best_val:
            best_val = vf
            safe_save({"model": model.state_dict(), "epoch": ep + 1, "val_full": vf,
                       "alpha": args.alpha}, best_ckpt)
            try:
                open(peak_txt, "w").write(f"best val_full={vf:.4f} @ep{ep+1} alpha={args.alpha}\n")
            except OSError as e:
                log(f"  [warn] peak_txt write failed: {e}")
        # always persist the loss/val history as a tiny json (survives even if ckpt skipped)
        try:
            json.dump(history, open(os.path.join(OUTDIR, f"{tag}_history.json"), "w"))
        except OSError:
            pass
    log(f"TRAIN done tag={tag} best_val={best_val:.4f}")


def eval_model(model, base, SPL, users, revealed="full", k=None, alpha=0.0, add_dislikes=False, seedoff=0):
    """Full-catalog held-liked NDCG@10 for the signed AE. revealed='full'|'k'. add_dislikes -> feed
    profile-half dislikes as signed-negative input (GATE 3)."""
    model.eval(); ni = base["ni"]; headmask = base["headmask"]
    ff, tt = [], []
    rng = np.random.default_rng(4242 + seedoff)
    xs = []; metas = []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        liked_prof = [j for j in prof_r if prof_r[j] >= LO]
        dis_prof = [j for j in prof_r if prof_r[j] <= HI]
        if revealed == "full":
            idx = list(liked_prof)
        else:
            if len(liked_prof) < k:
                continue
            sel = rng.choice(len(liked_prof), size=k, replace=False)
            idx = [liked_prof[i] for i in sel]
        xv = np.zeros(ni, np.float32)
        for j in idx:
            xv[j] = scale_rating(prof_r[j])
        if add_dislikes:
            for j in dis_prof:
                xv[j] = scale_rating(prof_r[j])   # already negative
        xs.append(xv); metas.append((profset, tlike))
    if not xs:
        return float("nan"), float("nan"), 0
    with torch.no_grad():
        Xt = torch.from_numpy(np.stack(xs))
        scores = model(Xt, dropout=0.0)[0].numpy().astype(np.float64)
    for r, (profset, tlike) in enumerate(metas):
        nf = ndcg10(scores[r], tlike, profset, headmask, False)
        nt = ndcg10(scores[r], tlike, profset, headmask, True)
        if nf is not None: ff.append(nf)
        if nt is not None: tt.append(nt)
    return (float(np.mean(ff)) if ff else float("nan"),
            float(np.mean(tt)) if tt else float("nan"), len(ff))


# ======================================================================== genre membership (GATE 2)
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]


def build_gmat(base):
    d = np.load(META); keepI = d["keepI"].astype(np.int64); ni = base["ni"]
    gix = {g: i for i, g in enumerate(GENRES)}
    Gmat = np.zeros((ni, len(GENRES)), np.float32)
    mv = {}
    with open("data/movielens/movies.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            i0 = line.find(","); i1 = line.rfind(",")
            try:
                mid = int(line[:i0])
            except ValueError:
                continue
            genres = line[i1 + 1:].strip().split("|")
            mv[mid] = genres
    for i in range(ni):
        for g in mv.get(int(keepI[i]), []):
            if g in gix:
                Gmat[i, gix[g]] = 1.0
    return Gmat, gix


def score_from_input(model, base, idx, val):
    ni = base["ni"]; xv = np.zeros(ni, np.float32)
    xv[np.asarray(idx, np.int64)] = np.asarray(val, np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(xv[None, :]), 0.0)[0].numpy()[0].astype(np.float64)


def percentile_rank(scores, items):
    """Mean fraction-of-catalog scored BELOW each item (1.0 = top)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores)) / (len(scores) - 1)
    return float(np.mean(ranks[np.asarray(items, np.int64)]))


def gate2_ig2(model, base, Gmat, gix):
    """IG2 genre flip: like-g vs dislike-g -> g items high vs low; specificity; taste separation."""
    cnt = base["cnt"]; res = {}
    probe_genres = ["Sci-Fi", "Horror", "Romance", "Documentary", "Children", "War"]
    NP = 12                                          # probe items per genre (popular members)
    flips = {}
    for g in probe_genres:
        gi = gix[g]; members = np.where(Gmat[:, gi] > 0)[0]
        if len(members) < NP * 2:
            continue
        pop_members = members[np.argsort(-cnt[members])]
        probe = pop_members[:NP]                      # revealed as answers
        held_members = pop_members[NP:NP + 200]       # measured (not in input)
        # untouched control genre = the genre least co-occurring; use a fixed distinct one
        ctrl = "Documentary" if g != "Documentary" else "War"
        ctrl_members = np.where(Gmat[:, gix[ctrl]] > 0)[0]
        ctrl_probe = ctrl_members[np.argsort(-cnt[ctrl_members])][:200]
        s_like = score_from_input(model, base, probe, np.ones(NP))
        s_dis = score_from_input(model, base, probe, -np.ones(NP))
        pr_like = percentile_rank(s_like, held_members)
        pr_dis = percentile_rank(s_dis, held_members)
        ctrl_like = percentile_rank(s_like, ctrl_probe)
        ctrl_dis = percentile_rank(s_dis, ctrl_probe)
        flips[g] = dict(genre_pct_like=pr_like, genre_pct_dislike=pr_dis,
                        flip_gap=pr_like - pr_dis,
                        ctrl_genre=ctrl, ctrl_pct_like=ctrl_like, ctrl_pct_dislike=ctrl_dis,
                        ctrl_displacement=abs(ctrl_like - ctrl_dis))
    res["genre_flips"] = flips
    res["mean_flip_gap"] = float(np.mean([f["flip_gap"] for f in flips.values()])) if flips else float("nan")
    res["mean_ctrl_displacement"] = float(np.mean([f["ctrl_displacement"] for f in flips.values()])) if flips else float("nan")
    # taste separation in latent: encode liked-only vs disliked-only for real users
    SPL = build_splits(base, SEEDS[0]); users = cohort(base, SPL, "test")
    ni = base["ni"]; seps = []
    for u in users[:120]:
        profset, held, prof_r, held_r = SPL[u]
        L = [j for j in prof_r if prof_r[j] >= LO]; D = [j for j in prof_r if prof_r[j] <= HI]
        if len(L) < 2 or len(D) < 2:
            continue
        xl = np.zeros(ni, np.float32); xl[L] = 1.0
        xd = np.zeros(ni, np.float32); xd[D] = 1.0
        with torch.no_grad():
            zl = model.encode(torch.from_numpy(xl[None, :]))[0]
            zd = model.encode(torch.from_numpy(xd[None, :]))[0]
        cos = float(F.cosine_similarity(zl, zd, dim=0))
        seps.append(cos)
    res["taste_sep_cos_like_vs_dislike"] = float(np.mean(seps)) if seps else float("nan")
    res["taste_sep_n"] = len(seps)
    return res


def gate3_dislike_lift(model, base):
    """Short-k fold-in: does adding profile-half oracle dislikes raise held-liked NDCG@10?"""
    tbl = {}
    for k in (2, 4, 8):
        f0s, f1s = [], []
        for seed in SEEDS:
            SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
            f0 = eval_model(model, base, SPL, users, "k", k=k, add_dislikes=False)[0]
            f1 = eval_model(model, base, SPL, users, "k", k=k, add_dislikes=True)[0]
            f0s.append(f0); f1s.append(f1)
        tbl[k] = dict(likes_only=float(np.mean(f0s)), plus_dislikes=float(np.mean(f1s)),
                      lift=float(np.mean(f1s) - np.mean(f0s)))
    return tbl


def cmd_gates(args):
    base = load_arena_base(); ni = base["ni"]
    global _POP; _POP = base["cnt"].astype(np.float64)
    tag = args.tag
    best = os.path.join(OUTDIR, f"{tag}_best.pt")
    blob = torch.load(best, map_location="cpu")
    model = SignedAE(ni, use_mask=True); model.load_state_dict(blob["model"]); model.eval()
    alpha_trained = blob.get("alpha", args.alpha)
    log(f"loaded {best} (trained alpha={alpha_trained}, val_full={blob.get('val_full')})")

    # ---- GATE 1: strength (full-profile + k-curve, seed-avg, full catalog) ----
    strength = {"tag": tag, "alpha": alpha_trained}
    full_f, full_t = [], []
    kcurve = {k: [] for k in (2, 4, 8)}
    for seed in SEEDS:
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        f, t, n = eval_model(model, base, SPL, users, "full")
        full_f.append(f); full_t.append(t)
        for k in (2, 4, 8):
            kf = eval_model(model, base, SPL, users, "k", k=k)[0]
            kcurve[k].append(kf)
    strength["full_profile"] = {"full": float(np.mean(full_f)), "tail": float(np.mean(full_t))}
    strength["kcurve"] = {k: float(np.mean(v)) for k, v in kcurve.items()}
    log(f"GATE1 full-profile full={np.mean(full_f):.4f} tail={np.mean(full_t):.4f}")
    log(f"GATE1 kcurve " + " ".join(f"k{k}={strength['kcurve'][k]:.4f}" for k in (2,4,8)))

    # sanity: full-profile must be >= every short-k (one ruler)
    strength["monotone_ok"] = bool(all(strength["full_profile"]["full"] >= strength["kcurve"][k] - 1e-9
                                        for k in (2, 4, 8)))
    json.dump(strength, open(os.path.join(OUTDIR, f"strength_{tag}.json"), "w"), indent=2)

    # ---- GATE 2 + GATE 3 ----
    Gmat, gix = build_gmat(base)
    g2 = gate2_ig2(model, base, Gmat, gix)
    log(f"GATE2 mean_flip_gap={g2['mean_flip_gap']:.3f} ctrl_disp={g2['mean_ctrl_displacement']:.3f} "
        f"taste_sep_cos={g2['taste_sep_cos_like_vs_dislike']:.3f}")
    g3 = gate3_dislike_lift(model, base)
    for k in (2, 4, 8):
        log(f"GATE3 k={k} likes_only={g3[k]['likes_only']:.4f} +dislikes={g3[k]['plus_dislikes']:.4f} "
            f"lift={g3[k]['lift']:+.4f}")
    gates = {"tag": tag, "alpha": alpha_trained, "gate1_strength": strength,
             "gate2_dislike_expresses": g2, "gate3_dislike_lift": g3}
    json.dump(gates, open(os.path.join(OUTDIR, f"gates_{tag}.json"), "w"), indent=2)
    log(f"wrote gates_{tag}.json + strength_{tag}.json")
    return gates


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ruler", "train", "gates"])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--tag", default="signed")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.cmd == "ruler":
        cmd_ruler(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "gates":
        cmd_gates(args)
