"""ANSWER CHANNEL DESIGN DRAFT (Fable, 2026-07-17). DO NOT RUN until the author signs a design sheet.

Non-circular, model-free concept answer for the Kalman belief pool (bpool5 lineage).

THE ANSWER: t_raw(u,c) = SEL + VAL, computed ONLY from (rated item ids, ratings, item popularity,
genome membership). No encoder, no decoder, no u*. Population-level calibration (fit on TRAIN users)
maps t_raw -> (obs value y, obs noise sig2) per concept; that map is shared by all users, so it can
inject ZERO per-user information beyond what raw behavior carries => non-circular by construction.

  SEL(u,c) = log2( (n_uc + 0.5) / (E_uc + 0.5) )          # watch-lift vs popularity expectation
             n_uc = #rated members of c;  E_uc = |hist_u| * sum_{i in c} cnt_i / sum_i cnt_i
  VAL(u,c) = (n_uc / (n_uc + LAM)) * mean_{i in rated members}(r_ui - item_mean_i)   # shrunk residual
  REFUSAL:  E_uc < TAU_EXPO  ->  user had no realistic exposure -> "can't answer" -> NO observation.
            (n_uc == 0 with E_uc >= TAU_EXPO is NOT refusal: it is a strong negative SEL answer.)

HARD RULE #1: use ALL train/val users with >=8 profile ratings. No 4k subsample (bpool5's samp=4000
was a shortcut; this experiment is cheap sparse ops -- run it on everyone).
LEAK GUARD: t_raw must be computed from the SAME profile items that feed the oracle belief u*
(the arena profile split), NEVER from eval-target items.
"""
import numpy as np
import scipy.sparse as sp

LAM = 3.0          # valence shrinkage pseudo-count
TAU_EXPO = 1.5     # refusal threshold on expected member exposure
K_LEV = 11         # quantizer levels ("half-star" scale -5..+5)
S2_FLOOR_SHRINK = 20.0  # cell-variance shrinkage pseudo-count

# ---------------------------------------------------------------- raw answers
def raw_answers(user_items, user_resid, Mbin, cnt):
    """user_items: list of int arrays (profile item ids per user)
       user_resid: list of float arrays (r_ui - item_mean_i), same shapes
       Mbin: csr (NC x ni) binary genome membership; cnt: (ni,) item rating counts.
       Returns SEL (U,NC), VAL (U,NC), NMEM (U,NC), EXPO (U,NC), REFUSED (U,NC) bool."""
    NC, ni = Mbin.shape
    MT = Mbin.T.tocsr()                                   # ni x NC
    p_item = cnt.astype(np.float64) / cnt.sum()
    pexp = np.asarray(Mbin @ p_item).ravel()              # (NC,) popularity mass per concept
    U = len(user_items)
    SEL = np.zeros((U, NC)); VAL = np.zeros((U, NC))
    NMEM = np.zeros((U, NC)); EXPO = np.zeros((U, NC))
    for u, (its, res) in enumerate(zip(user_items, user_resid)):
        sub = MT[its]                                     # n_u x NC sparse
        n_c = np.asarray(sub.sum(0)).ravel()              # members rated per concept
        rsum = np.asarray(sub.T @ res).ravel()            # residual-rating sum over members
        e_c = len(its) * pexp
        SEL[u] = np.log2((n_c + 0.5) / (e_c + 0.5))
        mean_res = np.divide(rsum, n_c, out=np.zeros(NC), where=n_c > 0)
        VAL[u] = (n_c / (n_c + LAM)) * mean_res
        NMEM[u] = n_c; EXPO[u] = e_c
    REFUSED = EXPO < TAU_EXPO
    return SEL, VAL, NMEM, EXPO, REFUSED

# ------------------------------------------- R2 decomposition (RUN THIS FIRST)
def r2_decomposition(AFF, SEL, VAL, WDPROJ, REFUSED, usable):
    """AFF: (U,NC) oracle affinities <u*_u, d_c> (TARGET ONLY -- never an input to answers).
       WDPROJ: optional (U,NC) hand-built Wd projection = UPPER-BOUND DIAGNOSTIC, not a candidate.
       Per usable concept, OLS R2 of AFF on each feature set, on non-refused cells. Pooled = n-weighted."""
    out = {}
    for name, feats in [("SEL", (SEL,)), ("VAL", (VAL,)), ("SEL+VAL", (SEL, VAL)),
                        ("WDPROJ*", (WDPROJ,)) if WDPROJ is not None else ("WDPROJ*", None)]:
        if feats is None: continue
        r2s, ws = [], []
        for c in np.where(usable)[0]:
            m = ~REFUSED[:, c]
            if m.sum() < 50: continue
            y = AFF[m, c]; X = np.column_stack([f[m, c] for f in feats] + [np.ones(m.sum())])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2s.append(1 - np.var(y - X @ beta) / np.var(y)); ws.append(m.sum())
        out[name] = (float(np.average(r2s, weights=ws)), np.array(r2s))
    return out  # compare against LLM-ordinal pooled R2 ~= 0.01 (known)

# -------------------------------- calibration + quantizer (population-level, TRAIN only)
def fit_calibration(AFF_tr, SEL_tr, VAL_tr, REFUSED_tr, usable, K=K_LEV):
    """Per concept: (1) 2-feature OLS t_hat = b0 + b1*SEL + b2*VAL (weights are POPULATION constants);
       (2) K-quantile edges of t_hat on train; (3) codebook y[c,k] = mean TRAIN oracle affinity in bin,
       s2[c,k] = shrunk var of TRAIN oracle affinity in bin  ->  the Kalman (obs value, obs noise).
       All maps are shared across users => observation model, not a leak."""
    NC = AFF_tr.shape[1]
    B = np.zeros((NC, 3)); EDGES = np.zeros((NC, K - 1))
    Y = np.zeros((NC, K)); S2 = np.ones((NC, K))
    for c in np.where(usable)[0]:
        m = ~REFUSED_tr[:, c]
        if m.sum() < 200: usable[c] = False; continue
        y = AFF_tr[m, c]; X = np.column_stack([SEL_tr[m, c], VAL_tr[m, c], np.ones(m.sum())])
        B[c], *_ = np.linalg.lstsq(X, y, rcond=None)
        t = X @ B[c]
        EDGES[c] = np.quantile(t, np.linspace(0, 1, K + 1)[1:-1])
        lev = np.searchsorted(EDGES[c], t)
        gvar = np.var(y)
        for k in range(K):
            sel = lev == k; n = sel.sum()
            Y[c, k] = y[sel].mean() if n else 0.0
            v = np.var(y[sel]) if n > 1 else gvar
            S2[c, k] = (n * v + S2_FLOOR_SHRINK * gvar) / (n + S2_FLOOR_SHRINK)
    return B, EDGES, Y, S2

def answer(u_sel, u_val, u_refused, c, B, EDGES, Y, S2):
    """A val-user's answer to concept c -> (level, y_obs, sig2) or None (refusal -> skip, turn consumed)."""
    if u_refused[c]: return None
    t = B[c, 0] * u_sel[c] + B[c, 1] * u_val[c] + B[c, 2]
    k = int(np.searchsorted(EDGES[c], t))
    return k, Y[c, k], S2[c, k]

# ------------------------------------------------------- Kalman update (heteroscedastic)
def kalman_update(mu, Sig, d, y, s2):
    Sd = Sig @ d; g = Sd / (d @ Sd + s2)
    return mu + g * (y - d @ mu), Sig - np.outer(g, Sd)

# ------------------------------------------------ NON-CIRCULARITY CHECK (Opus: run before trusting)
def rotation_invariance_check(gen_answers_fn, model_tensors, rng):
    """Randomly rotate EVERY frozen model tensor visible to the answer generator (Wd, Dc, encoder
       weights) and regenerate all val answers. They must be BIT-IDENTICAL (np.array_equal) to the
       unrotated run. Any change => the model leaked into the answer => circular. Calibration maps
       (B, EDGES, Y, S2) are exempt ONLY if fit on TRAIN and frozen before touching val."""
    a0 = gen_answers_fn(model_tensors)
    Q = np.linalg.qr(rng.standard_normal((512, 512)))[0]
    rot = {k: (v @ Q if getattr(v, 'ndim', 0) == 2 and v.shape[-1] == 512 else v)
           for k, v in model_tensors.items()}
    a1 = gen_answers_fn(rot)
    assert all(np.array_equal(x, y) for x, y in zip(a0, a1)), "CIRCULARITY: answers depend on the model"
