r"""signed_answers.py -- THE shared four-band signed concept-answer construction (design:
docs/design/DESIGN_SIGNED_CONCEPTS.md; author GO 2026-07-25). Implemented ONCE; every consumer
(concept_fold --signed, train_tower_t2 --signed_concepts, tradeoff_ledger signed rungs, eval
harnesses) imports from here. Replaces the [0.25,1] clip path (born in sel_top_concepts, 7f45fd2).

CONSTRUCTION (exact bpool_r2 components; 'watched' = rated at any band; targets excluded upstream):
    n_c  = user's watched members of c          e_c = |watches| * pexp_c
    SEL  = log2((n_c + 0.5) / (e_c + 0.5))
    NPMI = SEL / (-log2((n_c + 0.5) / (nu + 1)))              # normalized to [-1, 1]
    VAL  = (n_c / (n_c + 3)) * mean(star - TRAIN item mean)
    v_raw = NPMI + w_val * VAL          # w_val pre-registered: p90|w_val*VAL| = 0.15 on train
    v     = clip(v_raw, -1, 1)
FOUR BANDS (thresholds pre-registered from the TRAIN v_raw distribution, cached at
.cache/instrument/signed_answer_prereg.json -- computed once, never from eval outcomes):
    like     v > t_like (train p60 of positive v_raw)   -> fold graded positive v
    meh      -t_neg <= v <= t_like                       -> fold neutral (v = 0)
    dislike  v < -t_neg (t_neg = |train p25 of v_raw|)   -> fold graded negative v
    refuse   EXPO = e_c < TAU (1.5, bpool_r2's rule)     -> NO fold (burns the turn)
PER-USER NEGATIVE-CHANNEL VOLUME CAP (KT-A3 mitigation; leak R2 0.025->0.260 without it):
    over each user's folded negative set: v_neg <- v_neg * min(1, C_NEG / sum|v_neg|), C_NEG = 2.0.
C-FULL LEVEL MAP: like -> levels 6..9 by v; meh -> 5; dislike -> 4..0 by |v| (the tower's EXISTING
graded dislike levels -- sign handling at parity with the item channel).
"""
import os
import json
import numpy as np
from scipy import sparse

TAU_REF = 1.5
LAM_VAL = 3.0
C_NEG = 2.0
W_VAL_TARGET_P90 = 0.15
PREREG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                           ".cache", "instrument", "signed_answer_prereg.json")
ITEM_MEAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                              ".cache", "instrument", "train_item_mean.npy")
BAND_LIKE, BAND_MEH, BAND_DISLIKE, BAND_REFUSE = 0, 1, 2, 3
BAND_NAMES = {0: "like", 1: "meh", 2: "dislike", 3: "refuse"}


def _components(Xbin, Xres, Mm, pexp):
    """Dense (n,C) SEL/NPMI/VAL/EXPO/n_c from sparse user-item binary + residual matrices."""
    nc = np.asarray((Xbin @ Mm).todense(), np.float32)
    rs = np.asarray((Xres @ Mm).todense(), np.float32)
    nu = np.asarray(Xbin.sum(axis=1)).ravel().astype(np.float32)
    E = nu[:, None] * pexp[None, :].astype(np.float32)
    SEL = np.log2((nc + 0.5) / (E + 0.5))
    denom = -np.log2((nc + 0.5) / (nu[:, None] + 1.0))
    NPMI = SEL / np.maximum(denom, 1e-6)
    mr = np.divide(rs, nc, out=np.zeros_like(rs), where=nc > 0)
    VAL = (nc / (nc + LAM_VAL)) * mr
    return SEL, NPMI, VAL, E, nc


def user_matrices(items_list, stars_list, item_mean, ni):
    """[(sids,), (stars,)] per user -> sparse binary + residual matrices."""
    rows = []; cols = []; bdat = []; rdat = []
    for r, (its, st) in enumerate(zip(items_list, stars_list)):
        rows.extend([r] * len(its)); cols.extend(np.asarray(its).tolist())
        bdat.extend([1.0] * len(its))
        rdat.extend((np.asarray(st, np.float32) - item_mean[np.asarray(its, np.int64)]).tolist())
    n = len(items_list)
    Xb = sparse.csr_matrix((np.asarray(bdat, np.float32), (rows, cols)), shape=(n, ni))
    Xr = sparse.csr_matrix((np.asarray(rdat, np.float32), (rows, cols)), shape=(n, ni))
    return Xb, Xr


def compute_prereg(raw_df, tr_set, show2id, ni, Mm, pexp, chunk=20000, force=False):
    """ONE-TIME pre-registration from ALL train users (graded, train-only item means). Caches the
    thresholds JSON + item_mean npy; loads from cache thereafter."""
    if os.path.exists(PREREG_PATH) and os.path.exists(ITEM_MEAN_PATH) and not force:
        return json.load(open(PREREG_PATH)), np.load(ITEM_MEAN_PATH)
    import pandas as pd
    df = raw_df[raw_df["userId"].isin(tr_set)]
    sid = df["movieId"].map(show2id); ok = sid.notna()
    sid = sid[ok].astype(np.int64).values
    stars = df.loc[ok, "rating"].values.astype(np.float32)
    uid, _ = pd.factorize(df.loc[ok, "userId"].values)
    n_u = int(uid.max()) + 1
    imm = np.zeros(ni); ctt = np.zeros(ni)
    np.add.at(imm, sid, stars); np.add.at(ctt, sid, 1.0)
    item_mean = np.where(ctt > 0, imm / np.maximum(ctt, 1), stars.mean()).astype(np.float32)
    resid = (stars - item_mean[sid]).astype(np.float32)
    Xb = sparse.csr_matrix((np.ones(len(sid), np.float32), (uid, sid)), shape=(n_u, ni))
    Xr = sparse.csr_matrix((resid, (uid, sid)), shape=(n_u, ni))
    np_samples = []; val_samples = []
    npairs = 0
    for st in range(0, n_u, chunk):
        SEL, NPMI, VAL, E, nc = _components(Xb[st:st + chunk], Xr[st:st + chunk], Mm, pexp)
        m = (nc >= 2) & (E >= TAU_REF)
        np_samples.append(NPMI[m].astype(np.float32))
        val_samples.append(VAL[m].astype(np.float32))
        npairs += int(m.sum())
    NP = np.concatenate(np_samples); VA = np.concatenate(val_samples)
    w_val = W_VAL_TARGET_P90 / max(float(np.percentile(np.abs(VA), 90)), 1e-6)
    vraw = NP + w_val * VA
    pos = vraw[vraw > 0]
    prereg = {"source": f"ALL {n_u} train users, {npairs:,} answerable cells",
              "w_val": float(w_val),
              "t_like_p60pos": float(np.percentile(pos, 60)) if len(pos) else 0.2,
              "t_neg_absp25": float(abs(np.percentile(vraw, 25))),
              "TAU_refuse": TAU_REF, "LAM_val": LAM_VAL, "C_NEG": C_NEG,
              "vraw_quantiles": {str(q): float(np.percentile(vraw, q))
                                 for q in (5, 25, 50, 75, 95)},
              "formula": "v = clip(NPMI + w_val*VAL, -1, 1); bands like>t_like / meh / "
                         "dislike<-t_neg / refuse E<TAU; C_NEG per-user negative cap"}
    os.makedirs(os.path.dirname(PREREG_PATH), exist_ok=True)
    json.dump(prereg, open(PREREG_PATH, "w"), indent=2)
    np.save(ITEM_MEAN_PATH, item_mean)
    return prereg, item_mean


def load_prereg():
    assert os.path.exists(PREREG_PATH), \
        f"signed-answer prereg missing ({PREREG_PATH}) -- run compute_prereg (training does this once)"
    return json.load(open(PREREG_PATH)), np.load(ITEM_MEAN_PATH)


def signed_values(items_list, stars_list, item_mean, Mm, pexp, ni, prereg, apply_neg_cap=True):
    """The shared valuation: per (user, concept) -> (V clipped [-1,1] w/ meh zeroed + C_NEG cap,
    FOLD mask, BAND int8, answerable mask (n>=2))."""
    Xb, Xr = user_matrices(items_list, stars_list, item_mean, ni)
    SEL, NPMI, VAL, E, nc = _components(Xb, Xr, Mm, pexp)
    vraw = NPMI + prereg["w_val"] * VAL
    V = np.clip(vraw, -1.0, 1.0).astype(np.float32)
    t_like = prereg["t_like_p60pos"]; t_neg = prereg["t_neg_absp25"]
    B = np.full(V.shape, BAND_MEH, np.int8)
    B[V > t_like] = BAND_LIKE
    B[V < -t_neg] = BAND_DISLIKE
    B[E < TAU_REF] = BAND_REFUSE
    F = B != BAND_REFUSE
    V = np.where(B == BAND_MEH, 0.0, V).astype(np.float32)
    if apply_neg_cap:                                        # KT-A3: per-user negative-mass cap
        negmask = (B == BAND_DISLIKE)
        negsum = np.abs(np.where(negmask, V, 0.0)).sum(axis=1)
        scale = np.minimum(1.0, prereg.get("C_NEG", C_NEG) / np.maximum(negsum, 1e-9))
        V = np.where(negmask, V * scale[:, None], V).astype(np.float32)
    answerable = nc >= 2
    return V, F, B, answerable


def signed_for_pool(pool_sids, pool_stars, item_mean, Mm, pexp, prereg):
    """Single-user fast path (training examples): signed v per concept from the INPUT POOL only.
    Returns (V (C,), B (C,), answerable (C,)). NO neg-cap here -- cap post-selection over the
    REVEALED negatives via cap_negatives (the folded set is what the cap governs)."""
    pool_sids = np.asarray(pool_sids, np.int64)
    sub = Mm[pool_sids]                                          # (k, C) sparse
    nc = np.asarray(sub.sum(axis=0)).ravel().astype(np.float32)
    res = (np.asarray(pool_stars, np.float32) - item_mean[pool_sids])
    rs = np.asarray(sub.T @ res).ravel().astype(np.float32)
    nu = float(len(pool_sids))
    E = (nu * pexp).astype(np.float32)
    SEL = np.log2((nc + 0.5) / (E + 0.5))
    NPMI = SEL / np.maximum(-np.log2((nc + 0.5) / (nu + 1.0)), 1e-6)
    mr = np.divide(rs, nc, out=np.zeros_like(rs), where=nc > 0)
    VAL = (nc / (nc + LAM_VAL)) * mr
    V = np.clip(NPMI + prereg["w_val"] * VAL, -1.0, 1.0).astype(np.float32)
    B = np.full(V.shape, BAND_MEH, np.int8)
    B[V > prereg["t_like_p60pos"]] = BAND_LIKE
    B[V < -prereg["t_neg_absp25"]] = BAND_DISLIKE
    B[E < TAU_REF] = BAND_REFUSE
    V = np.where(B == BAND_MEH, 0.0, V).astype(np.float32)
    return V, B, nc >= 2


def cap_negatives(vals, c_neg=C_NEG):
    """Per-user negative-channel volume cap over a REVEALED value list (KT-A3 mitigation)."""
    s = sum(abs(v) for v in vals if v < 0)
    if s <= c_neg:
        return list(vals)
    f = c_neg / max(s, 1e-9)
    return [v * f if v < 0 else v for v in vals]


def value_to_level(v):
    """C-full tower level map: like (v>0) -> 6..9 by v; meh (v==0) -> 5; dislike (v<0) -> 4..0 by |v|
    (the tower's EXISTING graded dislike levels)."""
    v = float(v)
    if v > 0:
        return int(6 + round(3 * min(v, 1.0)))
    if v < 0:
        return int(4 - round(4 * min(abs(v), 1.0)))
    return 5


def smoke():
    rng = np.random.RandomState(0)
    ni, C, n = 200, 12, 30
    members = {c: np.sort(rng.choice(ni, size=rng.randint(20, 40), replace=False)) for c in range(C)}
    rows = np.concatenate([members[c] for c in range(C)])
    cols = np.concatenate([np.full(len(members[c]), c) for c in range(C)])
    Mm = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(ni, C))
    cnt = rng.rand(ni) * 100
    pexp = (cnt @ np.asarray(Mm.todense())) / cnt.sum()
    item_mean = np.full(ni, 3.5, np.float32)
    items = [rng.choice(ni, size=rng.randint(10, 60), replace=False).astype(np.int64)
             for _ in range(n)]
    stars = [np.clip(rng.normal(3.5, 1.0, len(it)), 0.5, 5.0).astype(np.float32) for it in items]
    prereg = {"w_val": 0.3, "t_like_p60pos": 0.15, "t_neg_absp25": 0.1, "C_NEG": C_NEG}
    V, F, B, ans = signed_values(items, stars, item_mean, Mm, pexp, ni, prereg)
    assert V.shape == (n, C) and (np.abs(V) <= 1.0 + 1e-6).all()
    assert (V[B == BAND_MEH] == 0).all(), "meh band must fold neutral"
    assert (V[B == BAND_DISLIKE] < 0).all() and (V[B == BAND_LIKE] > 0).all()
    assert (~F[B == BAND_REFUSE]).all(), "refuse band must not fold"
    negsum = np.abs(np.where(B == BAND_DISLIKE, V, 0)).sum(1)
    assert (negsum <= C_NEG + 1e-5).all(), "C_NEG per-user negative cap violated"
    assert value_to_level(1.0) == 9 and value_to_level(0.3) == 7 and value_to_level(0.0) == 5
    assert value_to_level(-0.25) == 3 and value_to_level(-1.0) == 0
    lv = [value_to_level(v) for v in np.linspace(-1, 1, 21)]
    assert all(lv[i] <= lv[i + 1] for i in range(20)), "value_to_level must be monotone"
    # single-user fast path agrees with the batch path (no cap on either side)
    Vb, Fb, Bb, _ = signed_values(items[:1], stars[:1], item_mean, Mm, pexp, ni, prereg,
                                  apply_neg_cap=False)
    Vs, Bs, ans_s = signed_for_pool(items[0], stars[0], item_mean, Mm, pexp, prereg)
    assert np.allclose(Vb[0], Vs, atol=1e-5) and (Bb[0] == Bs).all(), "pool path != batch path"
    cv = cap_negatives([0.9, -0.8, -0.9, -0.7], c_neg=C_NEG)
    assert abs(sum(-v for v in cv if v < 0) - C_NEG) < 1e-9 and cv[0] == 0.9
    print(f"[SMOKE signed_answers] PASS: bands {np.bincount(B.ravel(), minlength=4).tolist()} "
          f"(like/meh/dislike/refuse); neg-cap holds; pool==batch; level map monotone 0..9")


if __name__ == "__main__":
    smoke()
