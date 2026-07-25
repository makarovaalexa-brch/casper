r"""selplus.py -- ARM (S) SEL+ : the assembled, non-circular, model-free concept-answer imputer (the
middle-ground candidate the answer-model contrast is built toward; deep-research recipe, coordinator
2026-07-25). Labeled "assembled, non-circular candidate" -- built from cited prior art (NPMI watch-lift,
BM25-CF, Tagommenders/tag-genome content affinity, empirical-Bayes / James-Stein shrinkage). Uses NO
recommender parameters (no tower, no decoder, no u*) -> non-circular by construction.

SEL+(u, g) = shrink( sign * BM25_lift(u, members(g)) + w_val * resid_rating(u, g),
                     prior = center(genome_proj(u, g)),  weight = support(u, g) )
  BM25_lift    : IR/BM25-weighted exposure-corrected watch-lift of g's member items in u's FOLD-IN --
                 IDF de-emphasizes popular member items, user-length norm de-emphasizes heavy users
                 (replaces raw NPMI counts). NPMI-normalized to [-1, 1] like the SEL estimator.
  resid_rating : the existing shrunk residual-rating VAL term (signed_answers._components VAL).
  genome_proj  : u's tag-genome content affinity to g = mean over u's rated items of their genome
                 relevance to tag g (data/movielens/genome-scores.csv) -- MODEL-FREE; fills the
                 low-support attribute tail SEL is blind on. Per-concept z-scored -> the content prior.
  shrink       : v = w * v_behavioral + (1 - w) * prior,  w = support / (support + K0)  (James-Stein /
                 EB toward the content prior; low support leans on content, high support on behavior).
FOUR BANDS + C_NEG cap identical to signed_answers (shared thresholds from the train prereg). Answerable
set is EXTENDED modestly into the low-support tail: nc>=2 OR (nc>=1 AND |prior_z|>=1) -- the content
prior lets a real profile answer a well-defined tag with <2 rated members (the tail-filling claim).

For ITEM questions SEL+ keeps the real rating (it is a concept-channel imputer only).
"""
import os
import sys
import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
GENOME = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
K1_BM25 = 1.2
B_BM25 = 0.75
K0_SHRINK = 5.0                       # James-Stein shrinkage constant (support half-weight point)
NTR_USERS = 140768                    # canonical train user count (IDF document count)


def _idf(cnt, N=NTR_USERS):
    """BM25 IDF over item popularity (df = train rating count)."""
    df = np.asarray(cnt, np.float64)
    return np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)


def load_genome_relevance(ctx, tags, smoke=False, members=None):
    """Dense (ni, C) genome relevance restricted to the catalog items and the concept `tags`
    (columns aligned to `tags`). Model-free content signal. Smoke: synthesize from membership."""
    C = len(tags)
    G = np.zeros((ctx.ni, C), np.float32)
    if smoke:
        rng = np.random.RandomState(7)
        for j, t in enumerate(tags):
            G[members[t], j] = 0.6 + 0.4 * rng.rand(len(members[t]))
            G[:, j] += 0.05 * rng.rand(ctx.ni)
        return G
    import pandas as pd
    usid = [int(x) for x in open(os.path.join(PROC, "unique_sid.txt")).read().split()]
    m2s = {m: i for i, m in enumerate(usid)}
    tagcol = {int(t): j for j, t in enumerate(tags)}
    tagset = set(tagcol)
    for chunk in pd.read_csv(GENOME, chunksize=2_000_000):
        chunk = chunk[chunk["tagId"].isin(tagset)]
        chunk = chunk[chunk["movieId"].isin(m2s)]
        if len(chunk) == 0:
            continue
        si = chunk["movieId"].map(m2s).astype(np.int64).values
        ci = chunk["tagId"].map(tagcol).astype(np.int64).values
        G[si, ci] = chunk["relevance"].values.astype(np.float32)
    return G


def build_selplus_arrays(ctx, sh, Mm, pexp, smoke=False):
    """(V, F, answerable) at (n, C) for the SEL+ imputer over the FOLD-IN history (ctx.allb)."""
    from signed_answers import (load_prereg, user_matrices, _components, TAU_REF, C_NEG,
                                BAND_LIKE, BAND_MEH, BAND_DISLIKE, BAND_REFUSE)
    prereg, item_mean = load_prereg()
    tags = sh["tags"]; members = sh["members"]
    items_l = [np.asarray(s, np.int64) for s, l in ctx.allb]
    stars_l = [((np.asarray(l, np.float64) + 1) / 2).astype(np.float32) for s, l in ctx.allb]
    Xb, Xr = user_matrices(items_l, stars_l, item_mean, ctx.ni)
    # --- behavioral SEL/VAL/support (the honest terms) ---
    SEL, NPMI, VAL, E, nc = _components(Xb, Xr, Mm, pexp)
    # --- BM25-weighted exposure-corrected watch-lift (replaces raw NPMI counts) ---
    idf = _idf(ctx.cnt)                                            # (ni,)
    dl = np.asarray(Xb.sum(axis=1)).ravel().astype(np.float32)    # fold-in length
    avgdl = max(float(dl[dl > 0].mean()) if (dl > 0).any() else 1.0, 1e-6)
    Lu = (1.0 - B_BM25 + B_BM25 * dl / avgdl)                     # user-length norm
    tf = (K1_BM25 + 1.0) / (K1_BM25 * np.maximum(Lu, 1e-6) + 1.0)  # tf saturation (tf=1 per item)
    Wrow = sparse.diags((tf).astype(np.float32))                 # per-user weight
    Xw = Wrow @ Xb                                               # user-weighted
    Xw = Xw.multiply(idf[None, :]).tocsr()                       # + per-item IDF
    n_bm = np.asarray((Xw @ Mm).todense(), np.float32)           # BM25 observed member mass (n, C)
    nu_bm = np.asarray(Xw.sum(axis=1)).ravel().astype(np.float32)  # user BM25 total mass
    cnt_idf = (ctx.cnt.astype(np.float32) * idf)
    pexp_bm = (cnt_idf @ np.asarray(Mm.todense())) / max(cnt_idf.sum(), 1e-9)
    E_bm = nu_bm[:, None] * pexp_bm[None, :].astype(np.float32)
    SEL_bm = np.log2((n_bm + 0.5) / (E_bm + 0.5))
    denom = -np.log2((n_bm + 0.5) / (nu_bm[:, None] + 1.0))
    NPMI_bm = SEL_bm / np.maximum(denom, 1e-6)
    v_behav = np.clip(NPMI_bm + prereg["w_val"] * VAL, -1.0, 1.0).astype(np.float32)
    # --- content prior (model-free genome projection; per-concept z-score) ---
    Grel = load_genome_relevance(ctx, tags, smoke=smoke, members=members)   # (ni, C)
    dlc = np.maximum(dl, 1.0)
    gproj = (np.asarray((Xb @ Grel)) / dlc[:, None]).astype(np.float32)      # mean rated-item relevance
    mu = gproj.mean(0, keepdims=True); sd = gproj.std(0, keepdims=True) + 1e-6
    prior_z = ((gproj - mu) / sd).astype(np.float32)                         # centered content affinity
    prior = np.clip(0.5 * prior_z, -1.0, 1.0).astype(np.float32)            # to the v-scale (half a sd ~ unit)
    # --- James-Stein / EB shrinkage toward the content prior, support-weighted ---
    w = (nc / (nc + K0_SHRINK)).astype(np.float32)
    V = np.clip(w * v_behav + (1.0 - w) * prior, -1.0, 1.0).astype(np.float32)
    # --- answerable set: behavioral support OR low-support-with-strong-content-prior (tail fill) ---
    answerable = (nc >= 2) | ((nc >= 1) & (np.abs(prior_z) >= 1.0))
    # --- four bands (shared thresholds); refuse where exposure below TAU AND no content prior ---
    t_like = prereg["t_like_p60pos"]; t_neg = prereg["t_neg_absp25"]
    Bnd = np.full(V.shape, BAND_MEH, np.int8)
    Bnd[V > t_like] = BAND_LIKE
    Bnd[V < -t_neg] = BAND_DISLIKE
    refuse = (E < TAU_REF) & (np.abs(prior_z) < 1.0)              # unmeasurable AND no content signal
    Bnd[refuse] = BAND_REFUSE
    F = (Bnd != BAND_REFUSE) & answerable
    V = np.where(Bnd == BAND_MEH, 0.0, V).astype(np.float32)
    # per-user negative-channel volume cap (KT-A3 parity with signed_answers)
    negmask = (Bnd == BAND_DISLIKE) & F
    negsum = np.abs(np.where(negmask, V, 0.0)).sum(axis=1)
    scale = np.minimum(1.0, C_NEG / np.maximum(negsum, 1e-9))
    V = np.where(negmask, V * scale[:, None], V).astype(np.float32)
    return V, F, answerable


class SELPlus:
    """(S) SEL+ answer model. Concepts = the assembled imputer; items = the real fold-in rating."""
    geometric = False

    def __init__(self, V, F, answerable, lvl_lookup):
        self.V = V; self.F = F; self.ans = answerable; self.lk = lvl_lookup

    def concept_value(self, r, c):
        if self.ans[r, c] and self.F[r, c]:
            return float(self.V[r, c]), True
        return 0.0, False

    def item_value(self, r, i):
        d = self.lk[r]
        return (float(d[i]), True) if i in d else (0.0, False)
