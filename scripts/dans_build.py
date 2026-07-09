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
        # ---- ITERATION-2 BUFFNESS static structure (population-side, no new joins) ----
        # item popularity percentile pr in [0,1] (1=most popular); top-N cutoffs as pr thresholds.
        self.TOPN = (1000, 5000)
        self.topn_thr = {n: 1.0 - float(n) / self.ni for n in self.TOPN}
        # low-fame genome tags = bottom tercile of pop-weighted membership (relevance proxy already
        # thresholded at genome-membership); niche-tag engagement measures contact with these.
        self.lowfame_tag = self.tag_logpw <= np.quantile(self.tag_logpw, 1.0 / 3.0)
        # foreign markers from genome tag names (existing metadata; no new join).
        FOREIGN_KW = ("foreign", "subtitle", "subtitled", "anime", "bollywood", "criterion",
                      "french", "german", "germany", "east germany", "italian", "japanese",
                      "korean", "spanish", "world cinema", "kung fu", "martial arts")
        fmask = np.array([any(k in (str(t.get("tag", "")).lower()) for k in FOREIGN_KW)
                          for t in tq], bool)
        self.n_foreign_tags = int(fmask.sum())
        self.item_foreign = (np.asarray(self.tagM[fmask].sum(0)).ravel() > 0).astype(np.float64)
        # ---- ADDENDUM (territory matching): era distance + canon/cult + franchise structure ----
        # per-concept / per-entity member mean YEAR -> decade (for per-question era distance).
        yr_pos = np.where(self.year > 0, self.year.astype(np.float64), 0.0)
        yr_val = (self.year > 0).astype(np.float64)
        tsum = np.asarray(self.tagM.dot(yr_pos)).ravel(); tcnt = np.asarray(self.tagM.dot(yr_val)).ravel()
        self.tag_year = np.where(tcnt > 0, tsum / np.maximum(tcnt, 1), -1.0)
        self.tag_decade = np.where(self.tag_year > 0, (self.tag_year // 10) * 10, -1.0)
        esum = np.asarray(self.entM.dot(yr_pos)).ravel(); ecnt = np.asarray(self.entM.dot(yr_val)).ravel()
        self.ent_year = np.where(ecnt > 0, esum / np.maximum(ecnt, 1), -1.0)
        self.ent_decade = np.where(self.ent_year > 0, (self.ent_year // 10) * 10, -1.0)
        # bank era mass (for the era-pocket check): rating-count-weighted mean decade of the bank.
        bd = self.bank_decade[self.bank_decade > 0]
        self.bank_era_mean = float(bd.mean()) if len(bd) else 1990.0
        # canon/cult item markers: pre-1970; non-English title heuristic (non-ASCII glyph or 'a.k.a.').
        self.item_pre1970 = ((self.year > 0) & (self.year < 1970)).astype(np.float64)
        self.item_nonascii = np.array([1.0 if (t and (any(ord(c) > 127 for c in t)
                                       or "a.k.a" in t.lower())) else 0.0 for t in D["title"]])
        # franchise membership: member of any type=='franchise' entity in the battery.
        frows = [r for r, e in enumerate(bat) if e.get("type") == "franchise"]
        self.n_franchise = len(frows)
        fr_items = np.asarray(self.entM[frows].sum(0)).ravel() if frows else np.zeros(self.ni)
        self.item_franchise = (fr_items > 0).astype(np.float64)
        # ---- DISTINCTIVENESS reference (author addition): population user taste centroids in kmap
        #      space; deterministic sample; population-side structure like popularity (uses reference
        #      users' full rated sets -- these are structure, not the featurized user's own data). ----
        self._build_ref_centroids(n_ref=4000, seed=SEED)
        if verbose:
            print(f"[universe] ni={self.ni} bank={self.nbank} tags={self.ntag} ents={self.nent} "
                  f"kmap_items={len(self.km_ids)} [{time.time()-t0:.1f}s]", flush=True)

    def _build_ref_centroids(self, n_ref=4000, seed=SEED):
        """Sample n_ref population users (>=20 ratings, >=10 with kmap embeddings) and cache their
        rating-weighted taste centroids (unit 16-d) + the population centroid. Cached to disk."""
        cpath = f"{DANS}/ref_centroids_{n_ref}_{seed}.npz"
        if os.path.exists(cpath):
            d = np.load(cpath)
            self.refC = d["refC"]; self.popC = d["popC"]
            return
        d = np.load(META)
        uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
        order = np.argsort(uu, kind="stable")
        uu = uu[order]; ii = ii[order]; rr = rr[order]
        bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
        rng = np.random.default_rng(seed)
        cand = rng.permutation(int(uu[-1]) + 1)
        excl = study_ids()                     # eval firewall: study users never enter the reference set
        refc = []
        for u in cand:
            if int(u) in excl:
                continue
            s, e = bnd[u], bnd[u + 1]
            if e - s < 20:
                continue
            its = ii[s:e]; rat = rr[s:e]
            rows = self.km_row[its]; m = rows >= 0
            if m.sum() < 10:
                continue
            w = rat[m]
            c = (self.Enorm[rows[m]] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
            n = np.linalg.norm(c)
            if n > 0:
                refc.append(c / n)
            if len(refc) >= n_ref:
                break
        self.refC = np.asarray(refc)
        pc = self.refC.mean(0)
        self.popC = pc / (np.linalg.norm(pc) + 1e-12)
        np.savez_compressed(cpath, refC=self.refC, popC=self.popC)
        print(f"[universe] built {len(self.refC)} reference taste centroids -> {cpath}", flush=True)

    # -------- per-user feature builder (known half only) --------
    def user_features(self, known):
        """known: dict {dense_id: rating}. Returns feature matrices for ALL bank items / concepts /
        entities plus per-channel value features. Identical transform for fit and generation."""
        ni = self.ni
        kids = np.fromiter(known.keys(), np.int64, len(known))
        krat = np.fromiter(known.values(), np.float64, len(known))
        cmean = float(krat.mean()) if len(krat) else 3.5
        nk = max(len(kids), 1)
        # indicator / rating dense vectors (needed early for niche-tag engagement)
        ind = np.zeros(ni); ratv = np.zeros(ni)
        if len(kids):
            ind[kids] = 1.0; ratv[kids] = krat
        rmc_tag = self.tagM.dot(ind)                                    # per-tag rated-member count
        # ---- ITERATION-1 user-level consumption statistics (observable from the known half for real
        #      AND synthetic users alike; carry cross-user knowledge propensity) ----
        lognk = float(np.log(len(kids) + 1.0))
        csd = float(krat.std()) if len(krat) > 1 else 0.0
        kg = self.Gmat[kids].sum(0) if len(kids) else np.zeros(20)
        pgd = kg / max(kg.sum(), 1e-9); pgd = pgd[pgd > 0]
        gent = float(-(pgd * np.log(pgd)).sum()) if len(pgd) else 0.0
        ndec = float(len(set(int(self.decade[j]) for j in kids if self.decade[j] > 0)))
        # taste genre vector = sum genre over liked known (>=4), else all known
        likemask = krat >= 4.0
        gbase = kids[likemask] if likemask.any() else kids
        taste = self.Gmat[gbase].sum(0) if len(gbase) else np.zeros(20)
        tn = np.linalg.norm(taste)
        taste_u = taste / tn if tn > 0 else np.zeros(20)
        # decade distribution over known (density-at-decade); user decades with mass (min-distance)
        ddist = np.zeros(len(self.dec_levels) + 1)                 # last bucket = unknown decade
        for j in kids:
            d = int(self.decade[j])
            ddist[self.dec_row.get(d, len(self.dec_levels))] += 1
        if ddist.sum() > 0:
            ddist = ddist / ddist.sum()
        user_decs = np.array(sorted(set(int(self.decade[j]) for j in kids if self.decade[j] > 0)),
                             np.float64)

        def era_feats(dec_years):
            """Per-question era (density-at-decade + min-decade-distance) for an array of member decades
            (years; -1 unknown). Unknown-decade entries get the user's mean valid distance (neutral)."""
            dens = np.array([ddist[self.dec_row.get(int(d), len(self.dec_levels))] if d > 0 else 0.0
                             for d in dec_years])
            dist = np.zeros(len(dec_years))
            valid = dec_years > 0
            if len(user_decs) and valid.any():
                dv = np.abs(dec_years[valid][:, None] - user_decs[None, :]).min(1) / 10.0
                dist[valid] = dv
                dist[~valid] = float(dv.mean())
            return dens, dist

        # ===== ITEM features (nbank) =====
        rated_flag = ind[self.bank]
        kk = kids[self.km_row[kids] >= 0]                          # co-knowledge proximity (kmap space)
        if len(kk):
            Ku = self.Enorm[self.km_row[kk]]
            sim = self.bank_emb @ Ku.T
            coprox = sim.mean(1); comax = sim.max(1)
            coprox[~self.bank_has_emb] = 0.0; comax[~self.bank_has_emb] = 0.0
        else:
            coprox = np.zeros(self.nbank); comax = np.zeros(self.nbank)
        genre_align = (self.bank_genre @ taste_u) / self.bank_gn
        dec_align, era_dist_i = era_feats(self.bank_decade)        # dec_align = density variant (existing)

        # ---- ITERATION-2 BUFFNESS + ADDENDUM territory features (all user-level; known-half only) ----
        pr_k = self.pr[kids] if len(kids) else np.array([0.5])
        b_mean_pr = float(pr_k.mean()); b_med_pr = float(np.median(pr_k))
        out1 = pr_k < self.topn_thr[self.TOPN[0]]                       # outside top-1000
        out5 = pr_k < self.topn_thr[self.TOPN[1]]                       # outside top-5000
        b_share_out1k = float(out1.mean()); b_logcnt_out1k = float(np.log(1.0 + out1.sum()))
        b_share_out5k = float(out5.mean()); b_logcnt_out5k = float(np.log(1.0 + out5.sum()))
        niche_incid = float(rmc_tag[self.lowfame_tag].sum()); total_incid = float(rmc_tag.sum())
        b_niche_share = niche_incid / max(total_incid, 1e-9)
        b_niche_logcnt = float(np.log(1.0 + niche_incid))
        nfor = float(self.item_foreign[kids].sum()) if len(kids) else 0.0
        b_foreign_share = nfor / nk; b_foreign_logcnt = float(np.log(1.0 + nfor))
        yv = self.year[kids] if len(kids) else np.array([-1])
        vy = yv > 0
        b_share_old = float((yv[vy] < 1980).mean()) if vy.any() else 0.0
        b_int_size_out5k = lognk * b_share_out5k                        # log rating-count x obscurity
        b_int_size_niche = lognk * b_niche_share
        # ADDENDUM: universe coverage = share of the bank inside the user's territory (co-knowledge OR
        # era+genre proximity) = 'how much info is obtainable from this user'.
        gmed = float(np.median(genre_align))
        territory = (comax >= 0.30) | ((genre_align >= gmed) & (era_dist_i <= 1.0))
        b_coverage = float(territory.mean())
        # ADDENDUM canon/cult markers.
        npre = float(self.item_pre1970[kids].sum()) if len(kids) else 0.0
        b_pre1970_share = npre / nk; b_pre1970_logcnt = float(np.log(1.0 + npre))
        b_foreign_title_share = float(self.item_nonascii[kids].mean()) if len(kids) else 0.0
        b_franchise_share = float(self.item_franchise[kids].mean()) if len(kids) else 0.0
        # AUTHOR ADDITION: taste-cloud dispersion = trace of the covariance of the user's rated items'
        # co-knowledge embeddings (subculture BREADTH; genre entropy under-measures it).
        if len(kk) >= 2:
            Kv = self.Enorm[self.km_row[kk]]
            b_taste_disp = float(np.trace(np.cov(Kv, rowvar=False)))
        else:
            Kv = None
            b_taste_disp = 0.0
        # AUTHOR ADDITION (distinctiveness): taste typicality + neighborhood density vs the population
        # reference centroids (kmap space; reference = 4000 non-study users, fixed).
        if len(kk):
            w = np.array([known[int(j)] for j in kk])
            cen = (self.Enorm[self.km_row[kk]] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
            cn2 = np.linalg.norm(cen)
            cen = cen / cn2 if cn2 > 0 else cen
            b_typicality = 1.0 - float(cen @ self.popC)          # high = unusual taste centroid
            sims = self.refC @ cen
            b_nbr_density = float(np.sort(sims)[-20:].mean())    # high = dense neighborhood (common)
        else:
            b_typicality = 0.0; b_nbr_density = 0.0
        # AUTHOR ADDITION (pockets): top-cluster concentration + effective pocket count via small-k
        # k-means on the rated-item embeddings (k in 2..4 by silhouette; k=1 if weak structure).
        b_pocket_conc, b_pocket_eff = 1.0, 1.0
        if Kv is not None and len(Kv) >= 8:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            best_k, best_s, best_lab = 1, -1.0, None
            for k in (2, 3, 4):
                km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(Kv)
                try:
                    s = float(silhouette_score(Kv, km.labels_))
                except Exception:
                    s = -1.0
                if s > best_s:
                    best_k, best_s, best_lab = k, s, km.labels_
            if best_s >= 0.15 and best_lab is not None:          # weak structure -> single pocket
                shares = np.bincount(best_lab, minlength=best_k) / len(best_lab)
                b_pocket_conc = float(shares.max())
                b_pocket_eff = float(1.0 / np.sum(shares ** 2))
        # RATING STYLE (census family h): generosity/decisiveness as the LLM could read it.
        b_rate_mean = cmean - 3.5
        b_share_max = float((krat >= 5.0).mean()) if len(krat) else 0.0
        b_share_extreme = float(((krat <= 1.0) | (krat >= 5.0)).mean()) if len(krat) else 0.0
        # census markers + era spread + interaction
        b_doc_share = float(self.Gmat[kids, 6].mean()) if len(kids) else 0.0      # Documentary
        b_anim_share = float(self.Gmat[kids, 2].mean()) if len(kids) else 0.0     # Animation
        b_era_spread = float(yv[vy].std()) if vy.sum() > 1 else 0.0
        b_int_size_disp = lognk * b_taste_disp
        uvec = np.array([lognk, gent, ndec, csd,
                         b_mean_pr, b_med_pr,
                         b_share_out1k, b_logcnt_out1k, b_share_out5k, b_logcnt_out5k,
                         b_niche_share, b_niche_logcnt,
                         b_foreign_share, b_foreign_logcnt,
                         b_share_old,
                         b_int_size_out5k, b_int_size_niche,
                         b_coverage, b_pre1970_share, b_pre1970_logcnt,
                         b_foreign_title_share, b_franchise_share, b_taste_disp,
                         b_typicality, b_nbr_density, b_pocket_conc, b_pocket_eff,
                         b_rate_mean, b_share_max, b_share_extreme,
                         b_doc_share, b_anim_share, b_era_spread, b_int_size_disp])
        Uc = np.tile(uvec, (self.nbank, 1))
        item_know = np.column_stack([coprox, comax, genre_align, self.bank_pr,
                                     self.bank_logcnt, dec_align, Uc, era_dist_i, rated_flag])
        item_val = np.column_stack([genre_align, self.bank_pr, self.bank_logcnt,
                                    np.full(self.nbank, cmean)])

        # ===== CONCEPT features (ntag) =====
        rmc = rmc_tag
        rms = self.tagM.dot(ratv)
        mmean = np.where(rmc > 0, rms / np.maximum(rmc, 1), 0.0)
        c_align = np.where(self.tag_gn > 0, self.tag_genre @ taste_u / np.maximum(self.tag_gn, 1e-9), 0.0)
        c_dens, c_dist = era_feats(self.tag_decade)
        concept_raw_count = rmc
        concept_know_lin = np.column_stack([self.tag_logmemb, self.tag_logpw, c_align,
                                            np.tile(uvec, (self.ntag, 1)), c_dens, c_dist])
        concept_val = np.column_stack([mmean - 3.5, (rmc > 0).astype(float), c_align, self.tag_logpw,
                                       np.full(self.ntag, cmean)])

        # ===== ENTITY features (nent) =====
        rec = self.entM.dot(ind)
        res = self.entM.dot(ratv)
        emean = np.where(rec > 0, res / np.maximum(rec, 1), 0.0)
        e_align = np.where(self.ent_gn > 0, self.ent_genre @ taste_u / np.maximum(self.ent_gn, 1e-9), 0.0)
        efrac = rec / self.ent_nmov
        e_dens, e_dist = era_feats(self.ent_decade)
        entity_raw_count = rec
        entity_know_lin = np.column_stack([efrac, self.ent_logpop, e_align,
                                           np.tile(uvec, (self.nent, 1)), e_dens, e_dist])
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
    theta = [a(ncat-1), beta(p)]. eta=X beta. c=_cuts(a). P(y=k)=sig(c_k-eta)-sig(c_{k-1}-eta).
    l2: scalar OR per-coefficient vector (ITERATION-2 ridge on the user-feature block)."""
    na = ncat - 1
    a = theta[:na]; beta = theta[na:]
    c = _cuts(a)
    eta = X @ beta
    cL = np.concatenate([[-1e9], c])                # c_0..c_{K-1}; index by category
    cU = np.concatenate([c, [1e9]])
    A = _sig(cU[y] - eta)                           # sigma(c_y - eta)
    B = _sig(cL[y] - eta)                           # sigma(c_{y-1} - eta)
    P = np.maximum(A - B, 1e-12)
    nll = -np.sum(np.log(P)) + np.sum(l2 * beta * beta)
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


def eb_logit_sigma(offset, yb, grp, cap=6.0):
    """ITERATION-2 (calibration fix a): empirical-Bayes per-user random intercept for a SINGLE binary
    margin.  logit P(yb=1) = offset + b_u ; offset is the fixed linear predictor at this cut (eta - c_cut).
    Returns sampling-corrected sigma = sqrt(max(var(b_u) - mean(1/info), 0)).  Newton per user."""
    bs, infos = {}, {}
    for u in np.unique(grp):
        idx = np.where(grp == u)[0]
        off = offset[idx]; y = yb[idx]
        b = 0.0
        for _ in range(40):
            p = _sig(off + b)
            g = float(np.sum(y - p)); h = float(np.sum(p * (1 - p)))
            if h <= 1e-9:
                break
            step = g / h; b += step
            if abs(step) < 1e-7:
                break
        b = float(np.clip(b, -cap, cap))
        p = _sig(off + b); info = float(np.sum(p * (1 - p)))
        bs[int(u)] = b; infos[int(u)] = max(info, 1e-6)
    bv = np.array(list(bs.values()))
    samp = np.mean([1.0 / infos[u] for u in bs])
    var = max(float(np.var(bv, ddof=1)) - samp, 0.0)
    return float(np.sqrt(var))


def eb_percut_sigma(theta, Xs, y, grp, ncat):
    """ITERATION-2 (calibration fix a): per-cut-margin user random effects for an ordinal channel.
    For each threshold cut k (Y>=k+1) fit an independent between-user variance on the binary margin,
    with the ordinal fixed effects as offset (eta - c_k).  Returns [sigma_cut1, ..., sigma_cut{ncat-1}].
    Lets the knows-of-it margin (k>=1) carry MORE user variance than the know-well margin (k>=2), which a
    single shared intercept cannot -- the named fix for the attribute ICC(k>=1) miss."""
    na = ncat - 1
    c = _cuts(theta[:na]); eta = Xs @ theta[na:]
    sig = []
    for k in range(na):
        yb = (y > k).astype(np.float64)           # 1 if Y >= k+1
        sig.append(eb_logit_sigma(eta - c[k], yb, grp))
    return sig


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
