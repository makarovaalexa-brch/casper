"""E0 GO/NO-GO diagnostics for the coarse->granular adaptive agent.

Implements FABLE_AGENT_DESIGN_2026-07-07.md Q6 + RUN ORDER E0:
  P1  O-ans decomposition (the answerability-discovery prize): a selector GIVEN the true per-user
      answerability table but with a REALIZABLE belief (z starts 0, updated only by answers actually
      received; question VALUE scored from the current belief, never from z*). vs O-full(=A,0.568),
      static(=B,0.281). Anytime + endpoint NDCG, paired bootstrap CI on (O-ans - B).
  P2  Decision loss of the fitted pmodel surrogate: same selector but the true table is replaced by
      p_hat (using the user's TRUE known-profile genre_match). NDCG forfeited vs O-ans; pmodel
      within-user AUC vs pooled AUC.
  P3  Posterior-sharpening curve: online g_hat updated by B's simulated turns (answers AND refusals);
      AUC(pmodel-with-g_hat vs judged grid) at t=0..8.
  P4  Fuel-value scatter: per judged question, x=answerability variance across users, y=cold-belief
      value proxy. CSV + quadrant report.

REUSES the gate machinery EXACTLY (scripts/llm_answerability_gate.py): data loading, fixed answerer
split (seed 123), RecVAE-d512 fold-in, belief operator z'=z+eta*a*q (eta=16, a=cos(z*,q)), NDCG@10.
Reproduces G2 A=0.568 / B=0.281 BEFORE anything else. Makes NO LLM calls. Never modifies canonical
scripts or caches; all outputs to experiments/E0_*.

Run:  python scripts/e0_gonogo.py
"""
import os, sys, json, time, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")

import llm_answerability_gate as G          # gate machinery (canonical, unmodified)
import answerability_main_study as MS       # feature helpers (item_year, is_franchise, concept_pop_pct)

ETA = G.ETA                                 # 16.0
T = G.T_TURNS                               # 8
SEED = 0
SIGMA2 = 1.0                                # nuisance measurement variance for the belief-covariance value
GATE_GRID = G.GRID_CACHE                    # .cache/instrument2/answerability_grid_ml25m.json
MAIN_GRID = MS.GRID_CACHE                   # .cache/instrument2/answerability_mainstudy_grid.json
PMODEL_JSON = MS.PMODEL_JSON
BLIND_SCHEDULE = ["C:77", "C:11", "C:136", "C:10", "C:67", "C:70", "I:1279", "C:5"]  # gate's static B

OUT_JSON = "experiments/E0_gonogo_results.json"
OUT_MD = "experiments/E0_GONOGO_RESULTS.md"
OUT_SCATTER = "experiments/E0_fuel_value_scatter.csv"


# ============================================================ pmodel surrogate
class PModel:
    def __init__(self, path):
        m = json.load(open(path))
        self.feats = m["features"]           # [pop_pct, log_rcount, decade, genre_match, franchise, is_concept]
        self.mean = np.array(m["mean"]); self.std = np.array(m["std"])
        self.coef = np.array(m["coef"]); self.b = float(m["intercept"])

    def p(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        z = (X - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.b)))


def concept_pct_vec(D):
    return MS.concept_pop_pct(D)


def cand_features(D, kind, meta, dgv, concept_pct):
    """Mirror answerability_main_study.build_rows feature computation for ONE (user,question)."""
    nv = np.linalg.norm(dgv)
    if kind == "concept":
        cv = G.concept_genre_vec(D, meta["ctag"])
        gm = float(dgv @ cv / nv) if nv > 0 else 0.0
        pop = float(concept_pct[meta["ctag"]])
        lrc = float(np.log(D["concepts"]["coverage"][meta["ctag"]] + 1.0))
        dec, fr, isc = 0.0, 0.0, 1.0
    else:
        j = meta["j"]
        gv = D["Gmat"][j].astype(np.float64); n2 = np.linalg.norm(gv)
        gm = float(dgv @ gv / (nv * n2)) if (nv > 0 and n2 > 0) else 0.0
        pop = float(D["pr"][j])
        lrc = float(np.log(D["cnt"][j] + 1.0))
        yr = MS.item_year(D["title"][j]); dec = ((yr - 1900) / 100.0) if yr else 0.5
        fr = float(MS.is_franchise(D["title"][j])); isc = 0.0
    return [pop, lrc, dec, gm, fr, isc]


# ============================================================ fold-in machinery (copied from g2_exploitability)
def build_foldin(D):
    import torch
    from recvae import RecVAE
    import ml25m_arena as A
    blob = torch.load(G.RECVAE_CKPT, map_location="cpu")
    a = blob["args"]; model = RecVAE(a["hidden"], a["latent"], int(D["ni"]))
    model.load_state_dict(blob["model"]); model.eval()
    d = model.decoder.in_features
    W = model.decoder.weight.detach().numpy().astype(np.float64)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    headmask = A.load_arena(seed=G.ANSWERER_SPLIT_SEED)["headmask"]
    # item-decoder-row covariance for divisiveness value: div(q)=Var_i(W_i.q)=q^T C q
    Wm = W - W.mean(0, keepdims=True)
    C = (Wm.T @ Wm) / W.shape[0]

    item_tag = D["concepts"]["item_tag"]; Mbag = 50
    cdir_cache = {}

    def concept_dir(ctag):
        if ctag not in cdir_cache:
            rel = item_tag[:, ctag].astype(np.float64)
            top = np.argpartition(-rel, Mbag)[:Mbag]
            b = np.zeros(int(D["ni"]), np.float32); b[top] = rel[top]; b /= b.sum() + 1e-9
            with torch.no_grad():
                mu, _ = model.encoder(torch.tensor(b[None, :]), dropout_rate=0.0)
            z = mu.numpy()[0].astype(np.float64)
            cdir_cache[ctag] = z / (np.linalg.norm(z) + 1e-9)
        return cdir_cache[ctag]

    def enc_bag(items):
        x = np.zeros((1, int(D["ni"])), np.float32)
        if items:
            x[0, list(items)] = 1.0
        with torch.no_grad():
            mu, _ = model.encoder(torch.tensor(x), dropout_rate=0.0)
        return mu.numpy()[0].astype(np.float64)

    def decode(Z):
        return Z @ W.T + bdec

    return dict(d=d, W=W, bdec=bdec, Wn=Wn, C=C, headmask=headmask,
                concept_dir=concept_dir, enc_bag=enc_bag, decode=decode)


def per_user_from_gate(D):
    """Assemble per_user dict from the cached gate grid (== the 300 study users, seed-123 split)."""
    grid = json.load(open(GATE_GRID))["users"]
    split = G.build_split(D)
    per_user = {}
    for us, g in grid.items():
        u = int(us)
        Q = [(k, m) for k, m in g["Q"]]
        ans = {int(k): v for k, v in g["ans"].items()}
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        _, dgv = G.dominant_genre(D, sorted(kn), rat)
        per_user[u] = dict(Q=Q, ans=ans, dg_vec=dgv)
    return per_user, split


def assemble_users(D, per_user, split, FI, pm, concept_pct):
    """Per eligible user: z*, targets, candidate directions + true-answerable + p_hat + latent div value."""
    Wn, C = FI["Wn"], FI["C"]
    U = []
    for u, rec in per_user.items():
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        prof = [j for j in kn if rat.get(j, 0) >= 4]
        tlike = set(j for j in ho if rat.get(j, 0) >= 4)
        if not prof or not tlike:
            continue
        zstar = FI["enc_bag"](prof); nz = np.linalg.norm(zstar) + 1e-9
        dgv = rec["dg_vec"]
        dirs, isyes, keys, phat, divv = [], [], [], [], []
        for i, (k, m) in enumerate(rec["Q"]):
            if k == "concept":
                q = FI["concept_dir"](m["ctag"]); key = f"C:{m['ctag']}"
            elif k == "item":
                q = Wn[m["j"]]; key = f"I:{m['j']}"
            else:
                continue
            dirs.append(q); keys.append(key)
            isyes.append(G.is_yes(rec["ans"].get(i, {})))
            feats = cand_features(D, k, m, dgv, concept_pct)
            phat.append(float(pm.p(feats)[0]))
            divv.append(float(q @ C @ q))
        if not dirs:
            continue
        U.append(dict(u=u, zstar=zstar, nz=nz, prof=set(kn), tlike=tlike,
                      dir=np.array(dirs), ans=np.array(isyes, bool), key=keys,
                      phat=np.array(phat), div=np.array(divv)))
    return U


# ============================================================ selectors (anytime NDCG curves)
def ndcg(FI, z, rec):
    S = FI["decode"](z[None, :])[0]
    return G._ndcg_top10(S, rec["tlike"], rec["prof"], FI["headmask"], False)


def sherman_update(Sig, q, sigma2=SIGMA2):
    Sq = Sig @ q
    return Sig - np.outer(Sq, Sq) / (sigma2 + float(q @ Sq))


def sel_Oans(FI, rec):
    """O-ans: true table (only answerable picked), realizable belief, value = div(q)*(q^T Sigma q)."""
    d = FI["d"]; dirs = rec["dir"]; ok = set(np.where(rec["ans"])[0].tolist())
    z = np.zeros(d); Sig = np.eye(d); used = set(); curve = []
    for _ in range(T):
        avail = [c for c in ok if c not in used]
        if avail:
            val = [rec["div"][c] * float(dirs[c] @ Sig @ dirs[c]) for c in avail]
            c = avail[int(np.argmax(val))]
            a = float((dirs[c] @ rec["zstar"]) / rec["nz"])
            z = z + ETA * a * dirs[c]; Sig = sherman_update(Sig, dirs[c]); used.add(c)
        curve.append(ndcg(FI, z, rec))
    return curve


def sel_Ofull(FI, rec):
    """O-full (=selector A): true table + true-taste value (picks best true-target NDCG each turn)."""
    d = FI["d"]; dirs = rec["dir"]; ok = np.where(rec["ans"])[0]
    z = np.zeros(d); used = set(); curve = []
    for _ in range(T):
        avail = [c for c in ok if c not in used]
        if avail:
            av = (dirs[avail] @ rec["zstar"]) / rec["nz"]
            Zn = z[None, :] + ETA * av[:, None] * dirs[avail]
            S = FI["decode"](Zn)
            best, bi = -1.0, None
            for r, c in enumerate(avail):
                n = G._ndcg_top10(S[r], rec["tlike"], rec["prof"], FI["headmask"], False)
                if n is not None and n > best:
                    best, bi = n, c
            if bi is not None:
                a = float((dirs[bi] @ rec["zstar"]) / rec["nz"])
                z = z + ETA * a * dirs[bi]; used.add(bi)
        curve.append(ndcg(FI, z, rec))
    return curve


def sel_static(FI, rec, schedule):
    """Static B: fixed schedule, per-user answer gating (update only if this user can answer)."""
    d = FI["d"]; z = np.zeros(d); curve = []
    kmap = {k: c for c, k in enumerate(rec["key"])}
    for t in range(T):
        if t < len(schedule):
            k = schedule[t]; c = kmap.get(k)
            if c is not None and rec["ans"][c]:
                a = float((rec["dir"][c] @ rec["zstar"]) / rec["nz"])
                z = z + ETA * a * rec["dir"][c]
        curve.append(ndcg(FI, z, rec))
    return curve


def sel_surrogate(FI, rec):
    """P2: p_hat replaces the true table. Rank by p_hat*div*(q^T Sigma q) over ALL candidates;
    OUTCOME uses the true table (a wrong-p_hat pick that is truly unanswerable => refusal, turn wasted)."""
    d = FI["d"]; dirs = rec["dir"]; z = np.zeros(d); Sig = np.eye(d); used = set(); curve = []
    n = len(dirs)
    for _ in range(T):
        avail = [c for c in range(n) if c not in used]
        if avail:
            val = [rec["phat"][c] * rec["div"][c] * float(dirs[c] @ Sig @ dirs[c]) for c in avail]
            c = avail[int(np.argmax(val))]; used.add(c)
            if rec["ans"][c]:                                  # truly answerable -> answer + updates
                a = float((dirs[c] @ rec["zstar"]) / rec["nz"])
                z = z + ETA * a * dirs[c]; Sig = sherman_update(Sig, dirs[c])
            # else: refusal -> turn consumed, no update
        curve.append(ndcg(FI, z, rec))
    return curve


def anytime(curve):
    v = [x for x in curve if x is not None]
    return float(np.mean(v)) if v else None


# ============================================================ P1/P2 driver
def run_p1_p2(D, U, FI):
    print(f"[P1/P2] running selectors over {len(U)} eligible users ...", flush=True)
    rows = {"Oans": [], "Ofull": [], "static": [], "surr": []}
    endpoint = {"Oans": [], "Ofull": [], "static": [], "surr": []}
    for i, rec in enumerate(U):
        for name, fn in (("Oans", sel_Oans), ("Ofull", sel_Ofull),
                         ("surr", sel_surrogate)):
            cur = fn(FI, rec)
            rows[name].append(cur); endpoint[name].append(cur[-1])
        cur = sel_static(FI, rec, BLIND_SCHEDULE)
        rows["static"].append(cur); endpoint["static"].append(cur[-1])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(U)}", flush=True)

    # per-user anytime + endpoint arrays, keeping only users with all curves defined at t=8
    def col(name, t):
        return np.array([c[t] if c[t] is not None else np.nan for c in rows[name]])
    users = [rec["u"] for rec in U]
    keep = np.ones(len(U), bool)
    for name in rows:
        for t in range(T):
            keep &= ~np.isnan(col(name, t))
    idx = np.where(keep)[0]
    n = len(idx)

    def anytime_arr(name):
        M = np.array([[rows[name][i][t] for t in range(T)] for i in idx])
        return M.mean(1), M            # per-user anytime, full matrix
    at = {name: anytime_arr(name) for name in rows}
    end = {name: np.array([rows[name][i][-1] for i in idx]) for name in rows}

    def boot_ci(diff, nb=5000):
        rng = np.random.default_rng(SEED)
        b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
        return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())

    # anytime curves (mean over users at each t)
    per_turn = {name: [float(np.nanmean(col(name, t)[idx])) for t in range(T)] for name in rows}

    d_anytime, ci_anytime, p_anytime = boot_ci(at["Oans"][0] - at["static"][0])
    d_end, ci_end, p_end = boot_ci(end["Oans"] - end["static"])
    # P2 forfeit
    forf_any, forf_any_ci, _ = boot_ci(at["Oans"][0] - at["surr"][0])
    forf_end, forf_end_ci, _ = boot_ci(end["Oans"] - end["surr"])

    P1 = dict(
        n_users=n,
        anytime=dict(Oans=float(at["Oans"][0].mean()), Ofull=float(at["Ofull"][0].mean()),
                     static=float(at["static"][0].mean()), surrogate=float(at["surr"][0].mean())),
        endpoint=dict(Oans=float(end["Oans"].mean()), Ofull=float(end["Ofull"].mean()),
                      static=float(end["static"].mean()), surrogate=float(end["surr"].mean())),
        per_turn_ndcg=per_turn,
        prize_anytime=dict(delta=d_anytime, ci95=ci_anytime, p_gt0=p_anytime,
                           threshold=0.015, GO=bool(ci_anytime[0] > 0 and d_anytime >= 0.015)),
        prize_endpoint=dict(delta=d_end, ci95=ci_end, p_gt0=p_end))
    P2 = dict(
        forfeit_anytime=dict(delta_Oans_minus_surr=forf_any, ci95=forf_any_ci),
        forfeit_endpoint=dict(delta_Oans_minus_surr=forf_end, ci95=forf_end_ci),
        surrogate_anytime=float(at["surr"][0].mean()),
        surrogate_endpoint=float(end["surr"].mean()))
    return P1, P2


# ============================================================ P2 AUC (within-user vs pooled)
def run_p2_auc(D, pm, concept_pct):
    from sklearn.metrics import roc_auc_score
    grid = json.load(open(MAIN_GRID))["users"]
    split = G.build_split(D)
    pooled_y, pooled_p = [], []
    within = []
    for us, rec in grid.items():
        u = int(us)
        ans = {int(k): v for k, v in rec["ans"].items()}
        dgv = np.asarray(rec.get("dg_vec")) if rec.get("dg_vec") is not None else None
        if dgv is None:
            kn, _ = split[u]; rat = dict(D["rat_by_u"][u]); _, dgv = G.dominant_genre(D, sorted(kn), rat)
        ys, ps = [], []
        for i, (k, m) in enumerate(rec["Q"]):
            if i not in ans or k not in ("concept", "item"):
                continue
            y = 1 if G.is_yes(ans[i]) else 0
            feats = cand_features(D, k, m, dgv, concept_pct)
            p = float(pm.p(feats)[0])
            ys.append(y); ps.append(p); pooled_y.append(y); pooled_p.append(p)
        ys = np.array(ys); ps = np.array(ps)
        if len(ys) > 5 and ys.min() != ys.max():
            within.append(roc_auc_score(ys, ps))
    pooled = float(roc_auc_score(pooled_y, pooled_p))
    return dict(pooled_auc=pooled, within_user_auc_mean=float(np.mean(within)),
                within_user_auc_median=float(np.median(within)), n_users_within=len(within),
                n_pooled_obs=len(pooled_y))


# ============================================================ P3 posterior-sharpening
def run_p3(D, pm, FI, concept_pct):
    from sklearn.metrics import roc_auc_score
    grid = json.load(open(MAIN_GRID))["users"]
    split = G.build_split(D)
    NG = len(G.GENRES)

    # genre vector of a schedule key
    def key_genre(key):
        typ, val = key.split(":"); val = int(val)
        if typ == "C":
            return G.concept_genre_vec(D, val)          # unit genre-space vec
        gv = D["Gmat"][val].astype(np.float64); nn = np.linalg.norm(gv)
        return gv / nn if nn > 0 else gv

    sched_gvec = {k: key_genre(k) for k in BLIND_SCHEDULE}

    # population prior g_hat0 = mean of per-user known-half dominant-genre distributions
    prior = np.zeros(NG); nu = 0
    dgv_by_u = {}
    for us in grid:
        u = int(us); kn, _ = split[u]; rat = dict(D["rat_by_u"][u])
        _, dgv = G.dominant_genre(D, sorted(kn), rat)
        dgv_by_u[u] = dgv; prior += dgv; nu += 1
    prior /= max(nu, 1)

    # per-user schedule answerability lookup (from the gate grid, matching the selectors) + main grid labels
    gate = json.load(open(GATE_GRID))["users"]
    sched_answer = {}   # u -> {key: bool answerable}
    for us, g in gate.items():
        u = int(us); ans = {int(k): v for k, v in g["ans"].items()}
        m = {}
        for i, (k, meta) in enumerate(g["Q"]):
            key = f"C:{meta['ctag']}" if k == "concept" else (f"I:{meta['j']}" if k == "item" else None)
            if key in BLIND_SCHEDULE and i in ans:
                m[key] = G.is_yes(ans[i])
        sched_answer[u] = m

    # pmodel population answerability for schedule keys (for refusal down-weight): use pop-prior genre_match
    def phat_pop(key):
        gv = sched_gvec[key]; nv = np.linalg.norm(prior)
        gm = float(prior @ gv / nv) if nv > 0 else 0.0
        typ, val = key.split(":"); val = int(val)
        if typ == "C":
            feats = [float(concept_pct[val]), float(np.log(D["concepts"]["coverage"][val] + 1.0)),
                     0.0, gm, 0.0, 1.0]
        else:
            yr = MS.item_year(D["title"][val]); dec = ((yr - 1900) / 100.0) if yr else 0.5
            feats = [float(D["pr"][val]), float(np.log(D["cnt"][val] + 1.0)), dec, gm,
                     float(MS.is_franchise(D["title"][val])), 0.0]
        return float(pm.p(feats)[0])
    phat_pop_cache = {k: phat_pop(k) for k in BLIND_SCHEDULE}

    # precompute per-user judged-question fixed features + genre vecs + labels (main grid)
    users_eval = []
    for us, rec in grid.items():
        u = int(us); ans = {int(k): v for k, v in rec["ans"].items()}
        qg, qfix, qy = [], [], []
        for i, (k, m) in enumerate(rec["Q"]):
            if i not in ans or k not in ("concept", "item"):
                continue
            if k == "concept":
                gvec = G.concept_genre_vec(D, m["ctag"])
                fixed = [float(concept_pct[m["ctag"]]), float(np.log(D["concepts"]["coverage"][m["ctag"]] + 1.0)),
                         0.0, None, 0.0, 1.0]
            else:
                j = m["j"]; gv = D["Gmat"][j].astype(np.float64); nn = np.linalg.norm(gv)
                gvec = gv / nn if nn > 0 else gv
                yr = MS.item_year(D["title"][j]); dec = ((yr - 1900) / 100.0) if yr else 0.5
                fixed = [float(D["pr"][j]), float(np.log(D["cnt"][j] + 1.0)), dec, None,
                         float(MS.is_franchise(D["title"][j])), 0.0]
            qg.append(gvec); qfix.append(fixed); qy.append(1 if G.is_yes(ans[i]) else 0)
        if len(qy) > 5 and min(qy) != max(qy):
            users_eval.append(dict(u=u, gvec=np.array(qg), fix=qfix, y=np.array(qy)))

    def auc_for(user, ghat):
        nv = np.linalg.norm(ghat)
        X = []
        for r in range(len(user["y"])):
            gm = float(ghat @ user["gvec"][r] / nv) if nv > 0 else 0.0
            f = list(user["fix"][r]); f[3] = gm; X.append(f)
        p = pm.p(np.array(X))
        return roc_auc_score(user["y"], p)

    curve = []
    # t=0: population prior for everyone
    aucs0 = [auc_for(us, prior.copy()) for us in users_eval]
    curve.append(float(np.mean(aucs0)))

    # per-user running g_hat over B's schedule
    ghat = {us["u"]: prior.copy() for us in users_eval}
    for t in range(T):
        key = BLIND_SCHEDULE[t] if t < len(BLIND_SCHEDULE) else None
        if key is not None:
            gv = sched_gvec[key]; pp = phat_pop_cache[key]
            for us in users_eval:
                u = us["u"]; answ = sched_answer.get(u, {}).get(key, None)
                if answ is True:
                    ghat[u] = ghat[u] + gv                       # answered -> deposit genre vector
                elif answ is False:
                    ghat[u] = ghat[u] - pp * gv                  # refusal -> down-weight prop. to predicted answerability
                # unknown (key not judged for this user) -> no update
        aucs = [auc_for(us, ghat[us["u"]]) for us in users_eval]
        curve.append(float(np.mean(aucs)))
    return dict(auc_curve_t0_to_t8=curve, n_users=len(users_eval),
                t0_population_auc=curve[0], t4_auc=curve[4], t8_auc=curve[8],
                sharpened_by_t4=bool(curve[4] - curve[0] >= 0.01))


# ============================================================ P4 fuel-value scatter
def run_p4(D, FI, concept_pct):
    grid = json.load(open(MAIN_GRID))["users"]
    C = FI["C"]
    # answerability across users per question key
    lab = collections.defaultdict(list)
    meta_of = {}
    for us, rec in grid.items():
        ans = {int(k): v for k, v in rec["ans"].items()}
        for i, (k, m) in enumerate(rec["Q"]):
            if i not in ans or k not in ("concept", "item"):
                continue
            key = f"C:{m['ctag']}" if k == "concept" else f"I:{m['j']}"
            lab[key].append(1 if G.is_yes(ans[i]) else 0)
            meta_of[key] = (k, m)
    rows = []
    for key, ys in lab.items():
        if len(ys) < 20:            # need enough users to estimate variance
            continue
        ys = np.array(ys); var = float(ys.mean() * (1 - ys.mean()))   # Bernoulli variance across users
        k, m = meta_of[key]
        if k == "concept":
            q = FI["concept_dir"](m["ctag"])
        else:
            q = FI["Wn"][m["j"]]
        val = float(q @ C @ q)      # cold-belief value proxy = divisiveness (q^T C q, Sigma0=I)
        rows.append(dict(key=key, kind=k, n_users=len(ys), ans_rate=float(ys.mean()),
                         ans_variance=var, cold_value=val))
    # quadrant analysis: median splits
    if rows:
        vv = np.array([r["ans_variance"] for r in rows]); yy = np.array([r["cold_value"] for r in rows])
        mv, my = float(np.median(vv)), float(np.median(yy))
        for r in rows:
            r["hi_var"] = r["ans_variance"] >= mv; r["hi_val"] = r["cold_value"] >= my
        upper_right = [r for r in rows if r["hi_var"] and r["hi_val"]]
    else:
        mv = my = 0.0; upper_right = []
    # write CSV
    import csv
    with open(OUT_SCATTER, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "kind", "n_users", "ans_rate", "ans_variance", "cold_value", "hi_var", "hi_val"])
        for r in sorted(rows, key=lambda x: -x["ans_variance"]):
            w.writerow([r["key"], r["kind"], r["n_users"], f"{r['ans_rate']:.4f}",
                        f"{r['ans_variance']:.5f}", f"{r['cold_value']:.6f}",
                        int(r.get("hi_var", False)), int(r.get("hi_val", False))])
    return dict(n_questions=len(rows), median_variance=mv, median_value=my,
                upper_right_quadrant_count=len(upper_right),
                upper_right_fraction=float(len(upper_right) / max(len(rows), 1)),
                quadrant_populated=bool(len(upper_right) >= 5),
                csv=OUT_SCATTER,
                examples_hi_var_hi_val=[r["key"] for r in
                                        sorted(upper_right, key=lambda x: -(x["ans_variance"] * x["cold_value"]))[:12]])


# ============================================================ main
def main():
    t0 = time.time()
    print("[E0] loading data + fold-in ...", flush=True)
    D = G.load_data()
    FI = build_foldin(D)
    pm = PModel(PMODEL_JSON)
    concept_pct = concept_pct_vec(D)
    per_user, split = per_user_from_gate(D)
    print(f"[E0] gate grid users={len(per_user)}", flush=True)

    # ---- reproduce G2 (A=0.568, B=0.281) with the CANONICAL code before anything ----
    print("[E0] reproducing G2 via canonical g2_exploitability ...", flush=True)
    g2 = G.g2_exploitability(D, split, per_user, n_boot=2000)
    repro = dict(A_mean_ndcg=g2["A_mean_ndcg"], B_mean_ndcg=g2["B_mean_ndcg"],
                 dndcg_mean=g2["dndcg_mean"], ci95=g2["ci95"], n_users=g2["n_users"],
                 blind_schedule=g2["blind_schedule"],
                 matches_published=bool(abs(g2["A_mean_ndcg"] - 0.5681) < 0.01
                                        and abs(g2["B_mean_ndcg"] - 0.2812) < 0.01))
    print(f"  G2 repro: A={g2['A_mean_ndcg']:.4f} B={g2['B_mean_ndcg']:.4f} "
          f"dNDCG={g2['dndcg_mean']:.4f} (published A=0.5681 B=0.2812) "
          f"match={repro['matches_published']}", flush=True)
    if not repro["matches_published"]:
        print("  !! G2 reproduction FAILED — stopping per instructions.", flush=True)
        json.dump(dict(repro_g2=repro, STOP="G2 reproduction failed"), open(OUT_JSON, "w"), indent=1)
        return
    if g2["blind_schedule"] != BLIND_SCHEDULE:
        print(f"  note: reproduced blind schedule {g2['blind_schedule']} differs from hard-coded "
              f"{BLIND_SCHEDULE}; using the reproduced one for static B.", flush=True)
        globals()["BLIND_SCHEDULE"] = g2["blind_schedule"]

    # ---- assemble users for the realizable selectors ----
    U = assemble_users(D, per_user, split, FI, pm, concept_pct)
    print(f"[E0] assembled {len(U)} eligible users for P1/P2 selectors", flush=True)

    P1, P2sel = run_p1_p2(D, U, FI)
    print(f"  P1 O-ans anytime={P1['anytime']['Oans']:.4f} static={P1['anytime']['static']:.4f} "
          f"prize={P1['prize_anytime']['delta']:.4f} CI={P1['prize_anytime']['ci95']} "
          f"GO={P1['prize_anytime']['GO']}", flush=True)

    print("[E0] P2 within/pooled AUC ...", flush=True)
    P2auc = run_p2_auc(D, pm, concept_pct)
    P2 = dict(**P2sel, **P2auc)
    print(f"  P2 forfeit(anytime)={P2['forfeit_anytime']['delta_Oans_minus_surr']:.4f} "
          f"pooled_auc={P2auc['pooled_auc']:.4f} within_auc={P2auc['within_user_auc_mean']:.4f}", flush=True)

    print("[E0] P3 posterior-sharpening curve ...", flush=True)
    P3 = run_p3(D, pm, FI, concept_pct)
    print(f"  P3 AUC t0={P3['t0_population_auc']:.4f} t4={P3['t4_auc']:.4f} t8={P3['t8_auc']:.4f} "
          f"sharpened_by_t4={P3['sharpened_by_t4']}", flush=True)

    print("[E0] P4 fuel-value scatter ...", flush=True)
    P4 = run_p4(D, FI, concept_pct)
    print(f"  P4 questions={P4['n_questions']} upper-right={P4['upper_right_quadrant_count']} "
          f"populated={P4['quadrant_populated']}", flush=True)

    out = dict(
        config=dict(dataset="ML-25M", n_study_users=300, answerer_split_seed=123,
                    instrument="RecVAE-d512", eta=ETA, T=T, sigma2_value=SIGMA2,
                    value_model="value(q|z)=divisiveness(q)*(q^T Sigma q); divisiveness=Var_i(W_i.q)=q^T Cov(W) q; "
                                "Sigma=linear-Gaussian belief covariance (Sigma0=I, Sigma^-1+=qq^T/sigma2 per answered q); "
                                "NO z* peek. See assumptions.",
                    static_schedule=BLIND_SCHEDULE),
        repro_g2=repro, P1_Oans=P1, P2_surrogate=P2, P3_posterior=P3, P4_fuel_value=P4,
        wall_min=round((time.time() - t0) / 60, 2))
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"[saved] {OUT_JSON}  (wall {out['wall_min']} min)", flush=True)
    write_md(out)
    print(f"[saved] {OUT_MD}", flush=True)


def write_md(o):
    P1, P2, P3, P4, R = o["P1_Oans"], o["P2_surrogate"], o["P3_posterior"], o["P4_fuel_value"], o["repro_g2"]
    L = []
    L.append("# E0 GO/NO-GO diagnostics — results (ML-25M, RecVAE-d512, 300 study users)\n")
    L.append(f"Date 2026-07-07. Script `scripts/e0_gonogo.py`. NO LLM calls (all cached grids). "
             f"Wall {o['wall_min']} min. Reuses the canonical gate fold-in / belief / NDCG machinery "
             f"(`scripts/llm_answerability_gate.py`), the fitted pmodel "
             f"(`.cache/instrument2/answerability_pmodel.json`), and the cached judged grids.\n")

    L.append("## Reproduction check (done FIRST, canonical code)\n")
    L.append(f"Ran the unmodified `g2_exploitability` on the cached gate grid:\n")
    L.append(f"- **Selector A (O-full) = {R['A_mean_ndcg']:.4f}** (published 0.5681)")
    L.append(f"- **Selector B (static) = {R['B_mean_ndcg']:.4f}** (published 0.2812)")
    L.append(f"- dNDCG = {R['dndcg_mean']:.4f}, CI95 {[round(x,4) for x in R['ci95']]}, n={R['n_users']}")
    L.append(f"- **Matches published: {R['matches_published']}** -> proceeded.\n")

    L.append("## P1 — O-ans decomposition (the answerability-discovery prize)\n")
    L.append("O-ans = given the true per-user answerability table (asks only answerable questions) but "
             "with a REALIZABLE belief (z starts 0, updated only by answers received; value scored from "
             "the current belief, never from z*). Endpoint NDCG@10 and anytime NDCG@10 "
             f"(mean over t=1..8), n={P1['n_users']} users.\n")
    L.append("| selector | anytime NDCG | endpoint NDCG |")
    L.append("|---|---|---|")
    for nm, lab in (("Ofull", "O-full (A, table+true taste)"), ("Oans", "O-ans (table, realizable belief)"),
                    ("static", "static B (population schedule)"), ("surrogate", "surrogate (P2)")):
        L.append(f"| {lab} | {P1['anytime'][nm]:.4f} | {P1['endpoint'][nm]:.4f} |")
    pa = P1["prize_anytime"]; pe = P1["prize_endpoint"]
    L.append(f"\n**Answerability-discovery prize (O-ans − static):**")
    L.append(f"- anytime Δ = **{pa['delta']:.4f}**, 95% CI [{pa['ci95'][0]:.4f}, {pa['ci95'][1]:.4f}], "
             f"P(Δ>0)={pa['p_gt0']:.3f}")
    L.append(f"- endpoint Δ = {pe['delta']:.4f}, 95% CI [{pe['ci95'][0]:.4f}, {pe['ci95'][1]:.4f}]")
    L.append(f"- decision threshold: anytime prize ≥ 0.015 -> **{'GO' if pa['GO'] else 'NO-GO'}** "
             f"(measured {pa['delta']:.4f}).\n")
    L.append("Per-turn NDCG@10 (mean over users):\n")
    L.append("| t | " + " | ".join(str(t) for t in range(1, T + 1)) + " |")
    L.append("|" + "---|" * (T + 1))
    for nm, lab in (("Ofull", "O-full"), ("Oans", "O-ans"), ("static", "static B"), ("surr", "surrogate")):
        key = nm if nm in P1["per_turn_ndcg"] else nm
        L.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in P1["per_turn_ndcg"][key]) + " |")

    L.append("\n## P2 — Decision loss of the fitted surrogate\n")
    L.append("Same selector as O-ans but the true table is replaced by p̂ from the pmodel (true "
             "known-profile genre_match; isolates surrogate quality, not online estimation). A pick that "
             "is truly unanswerable costs a turn (refusal).\n")
    L.append(f"- NDCG forfeited vs O-ans: **anytime {P2['forfeit_anytime']['delta_Oans_minus_surr']:.4f}** "
             f"(CI {[round(x,4) for x in P2['forfeit_anytime']['ci95']]}), endpoint "
             f"{P2['forfeit_endpoint']['delta_Oans_minus_surr']:.4f} "
             f"(CI {[round(x,4) for x in P2['forfeit_endpoint']['ci95']]}).")
    L.append(f"- surrogate anytime NDCG {P2['surrogate_anytime']:.4f}, endpoint {P2['surrogate_endpoint']:.4f}.")
    L.append(f"- pmodel **pooled AUC = {P2['pooled_auc']:.4f}** vs **within-user AUC = "
             f"{P2['within_user_auc_mean']:.4f}** (median {P2['within_user_auc_median']:.4f}, "
             f"n={P2['n_users_within']} users, {P2['n_pooled_obs']} pooled obs). The within-user AUC is "
             f"the decision-relevant number (the agent ranks questions inside one user); the gap to the "
             f"pooled .92 is the between-user separation the agent never uses.\n")

    L.append("## P3 — Posterior-sharpening curve\n")
    L.append("Online ĝ (genre distribution) starts at the population prior; updated by B's simulated turns "
             "(answered concept/item -> +genre vector; refusal -> −p̂·genre vector). At each t we score "
             "AUC(pmodel-with-ĝ-genre_match vs the user's true judged grid), averaged over users.\n")
    L.append("| t | " + " | ".join(str(t) for t in range(0, T + 1)) + " |")
    L.append("|" + "---|" * (T + 2))
    L.append("| AUC | " + " | ".join(f"{v:.4f}" for v in P3["auc_curve_t0_to_t8"]) + " |")
    L.append(f"\n- t0 (population prior) AUC = {P3['t0_population_auc']:.4f}; t4 = {P3['t4_auc']:.4f}; "
             f"t8 = {P3['t8_auc']:.4f} (n={P3['n_users']} users).")
    L.append(f"- **Sharpens by t≤4 (Δ≥0.01): {P3['sharpened_by_t4']}.** If AUC@t4 ≈ AUC@t0 this is "
             f"failure-mode-1 evidence (posterior can't sharpen inside the budget).\n")

    L.append("## P4 — Fuel-value scatter\n")
    L.append("Per judged question (≥20 users): x = answerability variance across users (Bernoulli "
             "p(1−p)), y = cold-belief value proxy = divisiveness q^T Cov(W) q (the same per-question "
             "value the selectors use at z=0). CSV: `experiments/E0_fuel_value_scatter.csv`.\n")
    L.append(f"- {P4['n_questions']} questions; median variance {P4['median_variance']:.4f}, median value "
             f"{P4['median_value']:.4f}.")
    L.append(f"- **High-variance × high-value quadrant: {P4['upper_right_quadrant_count']} questions "
             f"({100*P4['upper_right_fraction']:.0f}%). Populated: {P4['quadrant_populated']}.**")
    if P4["examples_hi_var_hi_val"]:
        L.append(f"- top upper-right keys: {', '.join(P4['examples_hi_var_hi_val'])}.\n")

    L.append("## Assumptions / approximations (explicit)\n")
    L.append("1. **Value model.** The design doc's V(q) references \"the same divisiveness/entropy/expected-"
             "info machinery selector B uses.\" The canonical selector B is in fact a population greedy-"
             "forward-NDCG *schedule* whose value is not a per-turn belief-scorable quantity, so it cannot "
             "be reused verbatim for a realizable per-turn scorer. I substituted a belief-only expected-info "
             "value: value(q|z)=divisiveness(q)·(qᵀΣq), where divisiveness(q)=Var_i(W_i·q)=qᵀCov(W)q gives "
             "turn-1 discrimination (belief-independent catalog reordering power) and (qᵀΣq) with the "
             "linear-Gaussian posterior covariance Σ (Σ0=I, Σ⁻¹+=qqᵀ/σ² per answered q, σ²=1) supplies the "
             "novelty/diversity term. It uses ONLY the belief state — no z* peek. This is a defensible "
             "realization of \"expected-info machinery\" but is MY choice; a different realizable value "
             "model could shift O-ans (the prize should be read as: what a realizable-value agent that "
             "knows the answerability table earns over the static schedule).")
    L.append("2. **σ²=1** in the covariance update is a nuisance scale (answers are cos∈[−1,1], not stars); "
             "it only tunes how fast Σ shrinks along asked directions. Not tuned.")
    L.append("3. **Static B endpoint** is recomputed here as a per-turn curve (frozen schedule + per-user "
             "answer gating); its t8 endpoint matches the canonical B (0.281) by construction, and the "
             "anytime value is the mean of that curve.")
    L.append("4. **O-full anytime** is reported for context; O-full's per-turn pick maximizes true-target "
             "NDCG (privileged), so its curve is an upper reference, not a realizable one.")
    L.append("5. **P2 refusal cost** is folded in as opportunity cost (a refused pick yields zero belief "
             "gain and consumes a turn); I did not add a separately-tuned c_refusal scalar (selection by "
             "expected gain p̂·value already penalizes low-p̂ picks in a unit-consistent way).")
    L.append("6. **P3 ĝ** lives in the 20-dim genre space (the pmodel's genre_match feature space), updated "
             "additively; the population prior is the mean of per-user known-half dominant-genre "
             "distributions. Refusal down-weight uses the population-prior p̂ of the refused question.")
    L.append("7. **Candidate pool** for P1/P2 = the gate grid's per-user bank (39 concepts + 30 designed "
             "items + 6 taste-adjacent), identical to the G2 selectors, so O-full/static match 0.568/0.281. "
             "P2-AUC, P3, P4 use the richer main-study grid (200 concepts + ~1.7k items).\n")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
