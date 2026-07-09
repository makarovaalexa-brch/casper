"""dans_build.py -- BUILD THE DISTILLED ANSWERER (D-ANS) v1.

Contract: casper/DESIGN_SHEET_DISTILLED_ANSWERER.md (SIGNED 2026-07-09). NO LLM API CALLS anywhere.
Error ledger E1-E7 (casper/STATE_2026-07-08.md) honored: deterministic seeds, ASCII prints,
LOUO firewall (never an eval user's own cells in their fit), pre-registered thresholds printed
before results, paired bootstrap CIs, incremental checkpointing + write verification.

A hand-designed generative answerer: converts ANY ML-25M user's KNOWN-HALF ratings into interview
answers (knowledge {no_clue<rough_idea<know_well} + value {hated<meh<liked<loved}) statistically
faithful to the 173-user LLM grid.  GENERATION = SAMPLING ONLY (categorical draws; no argmax; no
injected noise -- the fitted distributions carry the uncertainty).

Stages (each writes incrementally to casper/experiments/DANS_BUILD.md + JSON sidecars):
  pre     G-pre reputation-confidence prevalence (informational, no correction)
  fit     features (known-half only) + per-channel ordered-logistic FIT (LOUO) for knowledge+value
  g1      agreement gate (LOUO accuracy/kappa vs LLM; value MAE)
  g2      FUEL REPRODUCTION (decisive): sample synthetic answers for the 173, recompute Stage-A stats
  g3      error profile on masked rated cells (value rule disabled)
  scale   20k synthetic population + G4 sanity
  full    extend to all ~162k population users (author condition; est-gated)

Run:  python scripts/dans_build.py --stage pre|fit|g1|g2|g3|scale|full
"""
import os, sys, json, time, argparse, collections, re, hashlib
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import warnings
warnings.filterwarnings("ignore")
import llm_answerability_gate as G

CACHE = ".cache/instrument2"
DANS = ".cache/dans"
EXP = "experiments"
WORKING = f"{CACHE}/answerer_v1_grid173_WORKING.json"
KMAP_EMB = f"{CACHE}/kmap_emb.npz"
KMAP_INT = f"{CACHE}/kmap_intercepts.npz"
TAG_MEMB = f"{CACHE}/tag_membership.json"
TAG_Q = f"{CACHE}/tag_questions.json"
ATTR_BAT = f"{CACHE}/attr_battery_500.json"
ITEM_LISTS = f"{CACHE}/item_lists.json"
META = "data/movielens/.cache/ml25m/meta.npz"
MD = f"{EXP}/DANS_BUILD.md"

KLAB = ["no_clue", "rough_idea", "know_well"]
VLAB = ["hated", "meh", "liked", "loved"]
KIDX = {k: i for i, k in enumerate(KLAB)}
VIDX = {v: i for i, v in enumerate(VLAB)}
VSTAR = np.array([1.5, 3.0, 4.0, 4.75])          # representative stars per 4-level value (bin midpoints)
SEED = 123
BOOT = 4000
os.makedirs(DANS, exist_ok=True)
os.makedirs(EXP, exist_ok=True)


def md(txt, mode="a"):
    open(MD, mode, encoding="utf-8").write(txt)


def sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:12]


# ============================================================ STATIC UNIVERSE (shared by fit + generate)
class Universe:
    """All population-side structure needed to featurize ANY user's known half. Built once."""
    def __init__(self, verbose=True):
        t0 = time.time()
        self.D = G.load_data()
        D = self.D
        self.ni = int(D["ni"])
        # per-item fame
        self.pr = D["pr"].astype(np.float64)
        self.logcnt = np.log(D["cnt"].astype(np.float64) + 1.0)
        self.Gmat = D["Gmat"].astype(np.float64)                       # (ni,20) binary genre
        gn = np.linalg.norm(self.Gmat, axis=1)
        self.Gnorm = np.where(gn > 0, gn, 1.0)
        # decade per item (from title year)
        pat = re.compile(r"\((\d{4})\)")
        yr = np.full(self.ni, -1, np.int64)
        for i, t in enumerate(D["title"]):
            m = pat.findall(t or "")
            if m:
                yr[i] = int(m[-1])
        self.year = yr
        self.decade = np.where(yr > 0, (yr // 10) * 10, -1)
        self.dec_levels = sorted(set(int(d) for d in self.decade if d > 0))
        self.dec_row = {d: r for r, d in enumerate(self.dec_levels)}
        # kmap knowledge embeddings (row map dense_id -> emb row, else -1)
        ke = np.load(KMAP_EMB)
        self.km_ids = ke["item_ids"].astype(np.int64)
        E = ke["E"].astype(np.float64)
        self.Enorm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        self.km_row = -np.ones(self.ni, np.int64)
        self.km_row[self.km_ids] = np.arange(len(self.km_ids))
        # ---- item bank (top800) ----
        il = json.load(open(ITEM_LISTS))
        self.bank = np.array([int(x) for x in il["lists"]["top800"]["ids"]], np.int64)
        self.nbank = len(self.bank)
        self.bank_row = {int(j): r for r, j in enumerate(self.bank)}
        self.bank_km = self.km_row[self.bank]                          # emb row per bank item (-1 if none)
        bm = self.bank_km >= 0
        self.bank_emb = np.zeros((self.nbank, self.Enorm.shape[1]))
        self.bank_emb[bm] = self.Enorm[self.bank_km[bm]]
        self.bank_has_emb = bm
        self.bank_genre = self.Gmat[self.bank]
        self.bank_gn = self.Gnorm[self.bank]
        self.bank_pr = self.pr[self.bank]
        self.bank_logcnt = self.logcnt[self.bank]
        self.bank_decade = self.decade[self.bank]
        # ---- concepts (1128 tags) ----
        tq = json.load(open(TAG_Q))["tags"]
        self.tags = tq
        self.ntag = len(tq)
        self.tag_row = {int(t["tagId"]): r for r, t in enumerate(tq)}
        self.tag_logmemb = np.array([np.log(t.get("membership_size", 0) + 1.0) for t in tq])
        self.tag_logpw = np.array([np.log(t.get("pop_weighted_membership", 0.0) + 1.0) for t in tq])
        memb = json.load(open(TAG_MEMB))["membership"]
        rows, cols = [], []
        for t in tq:
            r = self.tag_row[int(t["tagId"])]
            for j in memb.get(str(t["tagId"]), []):
                if 0 <= j < self.ni:
                    rows.append(r); cols.append(j)
        self.tagM = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                      shape=(self.ntag, self.ni))
        self.tag_size = np.asarray(self.tagM.sum(1)).ravel()
        # concept genre vec = mean Gmat over members
        cg = self.tagM.dot(self.Gmat)
        self.tag_genre = cg / (self.tag_size[:, None] + 1e-9)
        self.tag_gn = np.linalg.norm(self.tag_genre, axis=1)
        # ---- entities (500) ----
        bat = json.load(open(ATTR_BAT))["entities"]
        self.ents = bat
        self.nent = len(bat)
        self.ent_row = {e["entity_id"]: r for r, e in enumerate(bat)}
        rows, cols = [], []
        self.ent_nmov = np.zeros(self.nent)
        self.ent_logpop = np.zeros(self.nent)
        for r, e in enumerate(bat):
            self.ent_nmov[r] = max(int(e.get("n_movies", 0)), 1)
            self.ent_logpop[r] = np.log(float(e.get("popularity", 0.0)) + 1.0)
            for j in e.get("member_dense_ids", []):
                if 0 <= j < self.ni:
                    rows.append(r); cols.append(j)
        self.entM = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                      shape=(self.nent, self.ni))
        eg = self.entM.dot(self.Gmat)
        esz = np.asarray(self.entM.sum(1)).ravel()
        self.ent_genre = eg / (esz[:, None] + 1e-9)
        self.ent_gn = np.linalg.norm(self.ent_genre, axis=1)
        if verbose:
            print(f"[universe] ni={self.ni} bank={self.nbank} tags={self.ntag} ents={self.nent} "
                  f"kmap_items={len(self.km_ids)} [{time.time()-t0:.1f}s]", flush=True)

    # -------- per-user feature builder (known half only) --------
    def user_features(self, known):
        """known: dict {dense_id: rating}. Returns feature matrices for ALL bank items / concepts /
        entities plus per-channel value features. Identical transform for fit and generation."""
        ni = self.ni
        kids = np.fromiter(known.keys(), np.int64, len(known))
        krat = np.fromiter(known.values(), np.float64, len(known))
        cmean = float(krat.mean()) if len(krat) else 3.5
        # ---- ITERATION-1 user-level consumption statistics (observable from the known half for real
        #      AND synthetic users alike; carry cross-user knowledge propensity) ----
        lognk = float(np.log(len(kids) + 1.0))
        csd = float(krat.std()) if len(krat) > 1 else 0.0
        kg = self.Gmat[kids].sum(0) if len(kids) else np.zeros(20)
        pgd = kg / max(kg.sum(), 1e-9); pgd = pgd[pgd > 0]
        gent = float(-(pgd * np.log(pgd)).sum()) if len(pgd) else 0.0
        ndec = float(len(set(int(self.decade[j]) for j in kids if self.decade[j] > 0)))
        uvec = np.array([lognk, gent, ndec, csd])
        # taste genre vector = sum genre over liked known (>=4), else all known
        likemask = krat >= 4.0
        base = kids[likemask] if likemask.any() else kids
        taste = self.Gmat[base].sum(0) if len(base) else np.zeros(20)
        tn = np.linalg.norm(taste)
        taste_u = taste / tn if tn > 0 else np.zeros(20)
        # decade distribution over known
        ddist = np.zeros(len(self.dec_levels) + 1)                 # last bucket = unknown decade
        for j in kids:
            d = int(self.decade[j])
            ddist[self.dec_row.get(d, len(self.dec_levels))] += 1
        if ddist.sum() > 0:
            ddist = ddist / ddist.sum()
        # indicator / rating dense vectors (for sparse member counts)
        ind = np.zeros(ni); ind[kids] = 1.0
        ratv = np.zeros(ni); ratv[kids] = krat

        # ===== ITEM features (nbank) =====
        rated_flag = ind[self.bank]
        # co-knowledge proximity to rated set in kmap space
        kk = kids[self.km_row[kids] >= 0]
        if len(kk):
            Ku = self.Enorm[self.km_row[kk]]                       # (m,16)
            sim = self.bank_emb @ Ku.T                             # (nbank,m)
            coprox = sim.mean(1); comax = sim.max(1)
            coprox[~self.bank_has_emb] = 0.0; comax[~self.bank_has_emb] = 0.0
        else:
            coprox = np.zeros(self.nbank); comax = np.zeros(self.nbank)
        genre_align = (self.bank_genre @ taste_u) / self.bank_gn
        dec_align = np.array([ddist[self.dec_row.get(int(d), len(self.dec_levels))]
                              for d in self.bank_decade])
        Uc = np.tile(uvec, (self.nbank, 1))
        item_know = np.column_stack([coprox, comax, genre_align, self.bank_pr,
                                     self.bank_logcnt, dec_align, Uc, rated_flag])
        # item value features
        item_val = np.column_stack([genre_align, self.bank_pr, self.bank_logcnt,
                                    np.full(self.nbank, cmean)])

        # ===== CONCEPT features (ntag) =====
        rmc = self.tagM.dot(ind)                                    # rated-member count
        rms = self.tagM.dot(ratv)                                   # rated-member rating sum
        mmean = np.where(rmc > 0, rms / np.maximum(rmc, 1), 0.0)    # mean rating over rated members
        c_align = np.where(self.tag_gn > 0, self.tag_genre @ taste_u / np.maximum(self.tag_gn, 1e-9), 0.0)
        concept_raw_count = rmc                                     # saturating term (gamma applied later)
        concept_know_lin = np.column_stack([self.tag_logmemb, self.tag_logpw, c_align,
                                            np.tile(uvec, (self.ntag, 1))])
        concept_val = np.column_stack([mmean - 3.5, (rmc > 0).astype(float), c_align, self.tag_logpw,
                                       np.full(self.ntag, cmean)])

        # ===== ENTITY features (nent) =====
        rec = self.entM.dot(ind)                                    # rated filmography count
        res = self.entM.dot(ratv)
        emean = np.where(rec > 0, res / np.maximum(rec, 1), 0.0)
        e_align = np.where(self.ent_gn > 0, self.ent_genre @ taste_u / np.maximum(self.ent_gn, 1e-9), 0.0)
        efrac = rec / self.ent_nmov
        entity_raw_count = rec
        entity_know_lin = np.column_stack([efrac, self.ent_logpop, e_align,
                                           np.tile(uvec, (self.nent, 1))])
        entity_val = np.column_stack([emean - 3.5, (rec > 0).astype(float), e_align, self.ent_logpop,
                                      np.full(self.nent, cmean)])
        return dict(cmean=cmean,
                    item_know=item_know, item_val=item_val, rated_flag=rated_flag,
                    concept_raw=concept_raw_count, concept_lin=concept_know_lin, concept_val=concept_val,
                    entity_raw=entity_raw_count, entity_lin=entity_know_lin, entity_val=entity_val)


# ============================================================ split machinery
def study_ids():
    ids = set()
    for gp in (f"{CACHE}/answerability_grid_ml25m.json", f"{CACHE}/answerability_mainstudy_grid.json",
               WORKING):
        if os.path.exists(gp):
            try:
                ids |= {int(u) for u in json.load(open(gp))["users"].keys()}
            except Exception:
                pass
    return ids


def load_173(uni):
    """Load the 173-user WORKING grid into per-user records with known halves + labeled cells."""
    split = G.build_split(uni.D)
    grid = json.load(open(WORKING))["users"]
    users = []
    for us, rec in grid.items():
        u = int(us)
        if u not in split:
            continue
        kn, ho = split[u]
        rat = dict(uni.D["rat_by_u"][u])
        known = {int(j): float(rat[j]) for j in kn if j in rat}
        if len(known) < 4:
            continue
        cells = []
        for i, c in rec["Q"].items():
            ans = c.get("ans")
            if not ans or "knowledge" not in ans:
                continue
            ch = c["channel"]
            if ch == "concept":
                rid = uni.tag_row.get(int(c["tagId"]))
            elif ch == "attribute":
                rid = uni.ent_row.get(c["entity_id"])
            elif ch == "item":
                rid = uni.bank_row.get(int(c["j"]))
            else:
                rid = None
            if rid is None:
                continue
            cells.append((ch, rid, ans.get("knowledge"), ans.get("value"), ans.get("stars")))
        data = []
        for jstr, c in rec.get("data", {}).items():
            data.append((uni.bank_row.get(int(c["j"])), float(c["stars"])))
        users.append(dict(u=u, known=known, cells=cells, data=data))
    print(f"[load_173] {len(users)} users, "
          f"{sum(len(x['cells']) for x in users)} labeled LLM cells, "
          f"{sum(len(x['data']) for x in users)} rated data cells", flush=True)
    return users


# ============================================================ ORDINAL (proportional-odds) ENGINE
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _cuts(a):
    """monotone cutpoints from unconstrained a: c1=a0, c_k=c_{k-1}+softplus(a_k)."""
    c = np.empty(len(a))
    c[0] = a[0]
    for k in range(1, len(a)):
        c[k] = c[k - 1] + np.log1p(np.exp(-abs(a[k]))) + max(a[k], 0.0)
    return c


def _cuts_jac(a):
    """d c_j / d a_k."""
    n = len(a)
    sp = _sig(a)                                    # softplus'(a_k)=sigmoid(a_k) for k>=1
    J = np.zeros((n, n))
    for j in range(n):
        J[j, 0] = 1.0
        for k in range(1, j + 1):
            J[j, k] = sp[k]
    return J


def ord_nll_grad(theta, X, y, ncat, l2=1e-4):
    """Negative log-likelihood + analytic grad for proportional-odds ordinal logistic.
    theta = [a(ncat-1), beta(p)]. eta=X beta. c=_cuts(a). P(y=k)=sig(c_k-eta)-sig(c_{k-1}-eta)."""
    na = ncat - 1
    a = theta[:na]; beta = theta[na:]
    c = _cuts(a)
    eta = X @ beta
    cL = np.concatenate([[-1e9], c])                # c_0..c_{K-1}; index by category
    cU = np.concatenate([c, [1e9]])
    A = _sig(cU[y] - eta)                           # sigma(c_y - eta)
    B = _sig(cL[y] - eta)                           # sigma(c_{y-1} - eta)
    P = np.maximum(A - B, 1e-12)
    nll = -np.sum(np.log(P)) + l2 * np.sum(beta * beta)
    # grad wrt eta
    dA = A * (1 - A); dB = B * (1 - B)
    dEta = (dA - dB) / P                            # dNLL/deta_i
    gbeta = X.T @ dEta + 2 * l2 * beta
    # grad wrt cutpoints c_k: contributions where y==k (upper) or y==k+1 (lower)
    gc = np.zeros(len(c))
    for k in range(len(c)):
        m_up = (y == k)                             # A uses c_k
        m_lo = (y == k + 1)                         # B uses c_k
        gc[k] = -np.sum((1.0 / P[m_up]) * dA[m_up]) + np.sum((1.0 / P[m_lo]) * dB[m_lo])
    ga = _cuts_jac(a).T @ gc
    return nll, np.concatenate([ga, gbeta])


def ord_fit(X, y, ncat, warm=None, l2=1e-4, maxiter=300):
    na = ncat - 1
    if warm is None:
        base = np.log(np.arange(1, ncat) / ncat / (1 - np.arange(1, ncat) / ncat))
        theta0 = np.concatenate([base, np.zeros(X.shape[1])])
    else:
        theta0 = warm.copy()
    res = minimize(lambda th: ord_nll_grad(th, X, y, ncat, l2), theta0,
                   jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x


def ord_prob(theta, X, ncat, shift=0.0):
    """shift = per-user random intercept added to eta (ITERATION-1 model amendment)."""
    na = ncat - 1
    c = _cuts(theta[:na]); eta = X @ theta[na:] + shift
    cL = np.concatenate([[-1e9], c]); cU = np.concatenate([c, [1e9]])
    P = np.zeros((len(X), ncat))
    for k in range(ncat):
        P[:, k] = _sig(cU[k] - eta) - _sig(cL[k] - eta)
    return np.clip(P, 1e-9, 1.0)


def eb_user_sigma(theta, Xs, y, grp, ncat, cap=4.0):
    """ITERATION-1: empirical-Bayes per-user random intercept on top of the fixed-effects ordinal fit.
    For each user solve the scalar shift b_u minimizing the ordinal NLL of their cells (Newton, numeric
    Hessian); return (sigma_u, b_by_user) where sigma_u^2 = var(b_u) - mean sampling variance (1/info),
    floored at 0. The DRAW (not the estimate) is used in generation -- population-level fidelity, E5-clean."""
    na = ncat - 1
    c = _cuts(theta[:na]); eta0 = Xs @ theta[na:]
    cL = np.concatenate([[-1e9], c]); cU = np.concatenate([c, [1e9]])

    def grad_b(idx, b):
        e = eta0[idx] + b
        A = _sig(cU[y[idx]] - e); Bv = _sig(cL[y[idx]] - e)
        P = np.maximum(A - Bv, 1e-12)
        dA = A * (1 - A); dB = Bv * (1 - Bv)
        return float(np.sum((dA - dB) / P))

    bs, infos = {}, {}
    uq = np.unique(grp)
    for u in uq:
        idx = np.where(grp == u)[0]
        b = 0.0
        for _ in range(30):
            g0 = grad_b(idx, b)
            h = (grad_b(idx, b + 1e-4) - grad_b(idx, b - 1e-4)) / 2e-4
            if h <= 1e-9:
                break
            step = g0 / h
            b -= step
            if abs(step) < 1e-7:
                break
        b = float(np.clip(b, -cap, cap))
        h = (grad_b(idx, b + 1e-4) - grad_b(idx, b - 1e-4)) / 2e-4
        bs[int(u)] = b; infos[int(u)] = max(float(h), 1e-6)
    bv = np.array(list(bs.values()))
    samp = np.mean([1.0 / infos[u] for u in bs])
    var = max(float(np.var(bv, ddof=1)) - samp, 0.0)
    return float(np.sqrt(var)), bs


# ============================================================ standardization
def zscale(X, mu, sd):
    return (X - mu) / sd


def fit_scale(X):
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd > 1e-9, sd, 1.0)
    return mu, sd


# ============================================================ STAGE pre
def stage_pre():
    uni = Universe()
    users = load_173(uni)
    print("\n==== G-pre: reputation-confidence prevalence (INFORMATIONAL; no correction) ====", flush=True)
    print("Definition: an ITEM cell is 'reputation-confidence' if knowledge==know_well on an UNRATED item"
          "\n  (not in the user's known half) with NO NEARBY ENGAGEMENT, operationalized as BOTH:"
          "\n    co-knowledge proximity (max cosine to rated set in kmap space) < 0.30, AND"
          "\n    genre affinity (cos taste,item genre) < per-user median."
          "\n  Reported per within-bank popularity tercile. NO correction applied (author decision).",
          flush=True)
    # per-user featurize, examine item cells
    strat = {s: dict(kw=0, kw_rep=0, tot=0) for s in ("low", "mid", "high")}
    allcnt = uni.bank_logcnt
    q = np.quantile(allcnt, [1 / 3, 2 / 3])
    band = np.where(allcnt <= q[0], "low", np.where(allcnt <= q[1], "mid", "high"))
    for rec in users:
        f = uni.user_features(rec["known"])
        ga = f["item_know"][:, 2]                       # genre_align per bank row
        med = np.median(ga)
        comax = f["item_know"][:, 1]
        rated = f["rated_flag"] > 0.5
        for ch, rid, k, v, st in rec["cells"]:
            if ch != "item":
                continue
            b = band[rid]; strat[b]["tot"] += 1
            if k == "know_well" and not rated[rid]:
                strat[b]["kw"] += 1
                if comax[rid] < 0.30 and ga[rid] < med:
                    strat[b]["kw_rep"] += 1
    tot_kw = sum(s["kw"] for s in strat.values())
    tot_rep = sum(s["kw_rep"] for s in strat.values())
    tot_cells = sum(s["tot"] for s in strat.values())
    print(f"\n  overall: item cells={tot_cells}  know_well(unrated)={tot_kw}  "
          f"reputation-confidence={tot_rep}  "
          f"(rep share of know_well={tot_rep/max(tot_kw,1):.4f}; of all item cells={tot_rep/max(tot_cells,1):.4f})",
          flush=True)
    rows = []
    for s in ("low", "mid", "high"):
        d = strat[s]
        share_kw = d["kw_rep"] / max(d["kw"], 1)
        share_all = d["kw_rep"] / max(d["tot"], 1)
        rows.append((s, d["tot"], d["kw"], d["kw_rep"], share_kw, share_all))
        print(f"  [{s:4s}] cells={d['tot']:6d} kw_unrated={d['kw']:5d} rep={d['kw_rep']:5d} "
              f"rep/kw={share_kw:.4f} rep/all={share_all:.4f}", flush=True)
    out = dict(overall=dict(item_cells=tot_cells, know_well_unrated=tot_kw, reputation_conf=tot_rep,
                            rep_share_of_kw=tot_rep / max(tot_kw, 1),
                            rep_share_of_all=tot_rep / max(tot_cells, 1)),
               per_stratum={r[0]: dict(cells=r[1], kw_unrated=r[2], rep=r[3], rep_of_kw=r[4],
                                       rep_of_all=r[5]) for r in rows},
               operationalization="know_well on unrated item with comax<0.30 AND genre_align<per-user-median")
    json.dump(out, open(f"{DANS}/g_pre.json", "w"), indent=1)
    md("# D-ANS BUILD LOG\n\nContract: DESIGN_SHEET_DISTILLED_ANSWERER.md (signed 2026-07-09). "
       "NO LLM calls. Seeds deterministic (123). Script `scripts/dans_build.py`.\n\n"
       "## G-pre -- reputation-confidence prevalence (informational; NO correction, author decision)\n\n"
       "Operationalization: an item cell is reputation-confidence if it is `know_well` on an UNRATED "
       "item with NO nearby engagement -- co-knowledge proximity (max cosine to the rated set in the "
       "population kmap knowledge space) < 0.30 AND genre affinity below the user's median.\n\n"
       f"Overall: {tot_cells} item cells; {tot_kw} know_well-on-unrated; {tot_rep} reputation-confidence "
       f"(**{tot_rep/max(tot_kw,1):.3f}** of know_well-unrated; {tot_rep/max(tot_cells,1):.3f} of all item cells).\n\n"
       "| within-bank pop tercile | item cells | know_well(unrated) | reputation-conf | rep/know_well | rep/all |\n"
       "|---|--:|--:|--:|--:|--:|\n" +
       "".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.3f} |\n" for r in rows) +
       "\n", mode="w")
    print(f"\n[pre] wrote {DANS}/g_pre.json + {MD}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pre", "fit", "g1", "g2", "g3", "scale", "full"])
    a = ap.parse_args()
    if a.stage == "pre":
        stage_pre()
    else:
        import dans_stages as S
        getattr(S, f"stage_{a.stage}")()
