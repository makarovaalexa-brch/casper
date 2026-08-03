r"""concept_curriculum_v2.py -- the rebuilt concept-channel curriculum, transposed from the item tower.

WHY A REBUILD. The item tower gained its interview capability only when a MISSING REGIME was restored:
it had been trained on random-dropout profiles and on sparse refusal-dominated interviews, but never on
DENSE small sets of real answered ratings. Adding that bucket moved val sel 0.1370 -> 0.1408 and closed a
0.0104/0.0142/0.0133 deficit on dense k=2/4/8 fold-ins. Two further levers mattered: sampling examples
FRESH per draw rather than reusing a fixed set, and up-weighting RARE items so they actually appear
(uniform sampling left the median tail item in ~25 dense sets per run and 10.7% of the catalogue in fewer
than five).

WHAT THE OLD CONCEPT CURRICULUM ACTUALLY DID (read from the code, not assumed -- two of my four suspected
defects were wrong and are recorded here so nobody re-fixes a non-problem):
  * kc was ALREADY random, m ~ U{1..16}.                                        NOT a defect.
  * selection was ALREADY half-polarised / half-uniform: top = order[:ceil(m/2)] by |V| descending, then
    `rest` drawn uniformly from the remainder.                                  NOT a defect.
  * item context ALREADY existed: p_conc_only=0.35, so 65% of examples carried k ~ U{1..8} items with the
    revealed concepts' members dropped.                                         NOT a defect.
  * D1 THE EXAMPLES WERE MATERIALISED ONCE. build_cache() drew one example per user under a fixed seed,
    precomputed its latents, saved to disk, and every epoch looped over that same frozen set. One draw
    per user for the whole run -- a degenerate case of exactly what limited the item tower, where even
    nine draws per user proved thin.                                            THE MAIN DEFECT.
  * D3 REFUSALS WERE EXCLUDED. `cand = flatnonzero(ans & (B != BAND_REFUSE))` filtered refusal-band
    concepts out of the candidate pool, so the channel never once saw "I don't know this concept" --
    although an interview will ask them, and the item tower's own gain came from learning exactly that
    signal.                                                                     REAL.
  * D4 no rare-concept weighting; the uniform half samples concepts at their natural rate.
                                                                                REAL, milder than D1/D3.

WHAT THIS MODULE CHANGES, AND NOTHING ELSE:
  1. FRESH DRAWS. draw() is called per example from a live RNG; there is no cache and no fixed example
     set. Affordable because the distillation term is gone (below), so there is no teacher fold to
     precompute -- only the base fold, which the student needs anyway.
  2. REFUSALS ARE FED. Refusal-band concepts enter as tokens with value 0.0 and are reported separately,
     instead of being filtered out. A refusal is evidence about the user, not an absence of evidence.
  3. RARE CONCEPTS UP-WEIGHTED. The uniform half of the pick is drawn with probability proportional to
     member_count^-RARE_POW (default 0.5, the same exponent the item tower used -- no evidence to
     deviate).
  4. AN EXPLICIT REGIME MIXTURE, transposed from the item tower's 25/30/35/5/5, so the sparse and dense
     ends are both supervised rather than only whatever the old single generator happened to produce.

WHAT IT DELIBERATELY DOES NOT CHANGE: the signed four-band SEL answer, the member-drop rule, the
answerability rule, and the target hold-out. Those were verified (specificity: like 0.626 / dislike 0.416
/ unrelated 0.498; SIGN +0.2096, IDENTITY +0.1278) and are the acceptance gate for the refit.

THE DISTILLATION TERM IS GONE. lam>0 never beat lam=0 at equal budget -- the teacher was NULL. Training
is pure held-out-like NLL. This is what makes (1) affordable.
"""
import numpy as np

from signed_answers import signed_for_pool, cap_negatives, BAND_REFUSE

# ---- regime mixture, transposed from the item tower's 25/30/35/5/5 -----------------------------------
P_BROAD, P_DENSE, P_INTERVIEW, P_KC0, P_KC1 = 0.25, 0.30, 0.35, 0.05, 0.05

RARE_POW = 0.5          # uniform-half pick ~ clip(member_count, RARE_CLIP)^-RARE_POW; 0 = natural rate
RARE_CLIP = (30.0, 1000.0)  # clip counts BEFORE exponentiating. Concept member counts span 3+ orders,
                        # so an unclipped ^-0.5 gave broad concepts 0.8% of picks -- and broad concepts
                        # are exactly what an interview asks FIRST. Clipping caps the weight ratio near
                        # 6:1 instead of 13:1. Target niche share ~0.95-0.97, not 0.99.
DENSE_KCMAX = 8         # dense bucket: kc ~ U{1..8}
BROAD_FRAC = (0.5, 1.0) # broad bucket: reveal this fraction of answerable concepts
BROAD_CAP = 64          # ... but never more than this many. BROAD plays the item tower's dropout-profile
                        # role (teach aggregation beyond interview lengths); uncantained it would produce
                        # token sets far longer than anything an interview generates.
INTERVIEW_KCS = (1, 2, 4, 8, 16, 32)
ITEM_KMAX = 8           # item context per example, k ~ U{1..ITEM_KMAX}
P_CONC_ONLY = 0.35      # unchanged from the old curriculum: 35% concepts-only, 65% with item context


def set_mixture(p_broad, p_dense, p_interview, p_kc0, p_kc1):
    global P_BROAD, P_DENSE, P_INTERVIEW, P_KC0, P_KC1
    tot = p_broad + p_dense + p_interview + p_kc0 + p_kc1
    assert abs(tot - 1.0) < 1e-9, f"mixture must sum to 1, got {tot}"
    P_BROAD, P_DENSE, P_INTERVIEW, P_KC0, P_KC1 = p_broad, p_dense, p_interview, p_kc0, p_kc1


def _pick_concepts(cand, V, kc, rng, member_count, mode):
    """Choose kc concepts from `cand`.

    mode='mixed'  -- half most-polarised by |V|, half rare-weighted uniform. This preserves the old
                     curriculum's half/half shape (which was NOT a defect) and only changes the uniform
                     half from natural-rate to rare-weighted.
    mode='random' -- entirely rare-weighted uniform; used by the dense bucket so mild concepts appear.
    """
    kc = min(kc, len(cand))
    if kc <= 0:
        return np.empty(0, np.int64)
    if mode == "mixed":
        n_top = int(np.ceil(kc / 2))
        top = cand[np.argsort(-np.abs(V[cand]))][:n_top]
        rest_pool = np.setdiff1d(cand, top)
        n_rest = kc - len(top)
    else:
        top = np.empty(0, np.int64)
        rest_pool = cand
        n_rest = kc
    rest = np.empty(0, np.int64)
    if n_rest > 0 and len(rest_pool) > 0:
        n_rest = min(n_rest, len(rest_pool))
        if RARE_POW > 0.0 and member_count is not None:
            w = np.clip(np.asarray(member_count, np.float64)[rest_pool],
                        RARE_CLIP[0], RARE_CLIP[1]) ** (-RARE_POW)
            ssum = w.sum()
            rest = (rng.choice(rest_pool, size=n_rest, replace=False, p=w / ssum) if ssum > 0
                    else rng.choice(rest_pool, size=n_rest, replace=False))
        else:
            rest = rng.choice(rest_pool, size=n_rest, replace=False)
    return np.concatenate([top, rest]).astype(np.int64)


def draw(u, rng, Mm, grate, member_sets, item_mean, prereg, member_count=None):
    """One FRESH curriculum example. Returns
        (in_s, in_l, cids, cvals, crefs, tgt, regime, n_refusal)
    or None. No teacher basis is returned: the distillation term is deleted.

    in_s/in_l  item context (may be empty), members of revealed concepts dropped
    cids/cvals concept ids and their signed four-band values; refusals carry 0.0
    tgt        held-out liked items, chosen BEFORE anything is revealed so nothing can leak
    """
    items = np.asarray(u["items"], np.int64)
    lv = np.asarray(u["levels"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(items) < 4 or len(liked) < 2:
        return None

    # targets first -- an interview must never be able to ask about a target
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(items, tg)
    pool_s, pool_l = items[keep], lv[keep]
    if len(pool_s) < 2:
        return None

    stars = (pool_l.astype(np.float32) + 1.0) / 2.0
    V, B, ans = signed_for_pool(pool_s, stars, item_mean, Mm, grate, prereg)
    answerable = np.flatnonzero(ans & (B != BAND_REFUSE))     # can give a signed value
    refusable = np.flatnonzero(ans & (B == BAND_REFUSE))      # asked -> "I don't know this one"

    r = rng.random()
    n_ref = 0
    if r < P_BROAD:
        regime = "broad"
        if len(answerable) == 0:
            return None
        frac = rng.uniform(*BROAD_FRAC)
        kc = max(1, min(BROAD_CAP, int(round(frac * len(answerable)))))
        cids = _pick_concepts(answerable, V, kc, rng, member_count, "mixed")
    elif r < P_BROAD + P_DENSE:
        regime = "dense"
        if len(answerable) == 0:
            return None
        kc = int(rng.integers(1, DENSE_KCMAX + 1))
        cids = _pick_concepts(answerable, V, kc, rng, member_count, "random")
    elif r < P_BROAD + P_DENSE + P_INTERVIEW:
        # THE REGIME THE OLD CURRICULUM NEVER HAD: refusals are ASKED and FED, not filtered away.
        regime = "interview"
        kc = int(rng.choice(INTERVIEW_KCS))
        askable = np.concatenate([answerable, refusable]) if len(refusable) else answerable
        if len(askable) == 0:
            return None
        cids = _pick_concepts(askable, V, kc, rng, member_count, "mixed")
        n_ref = int(np.isin(cids, refusable).sum())
    elif r < P_BROAD + P_DENSE + P_INTERVIEW + P_KC0:
        regime = "kc0"                                        # no concepts at all: the channel must be
        cids = np.empty(0, np.int64)                          # a no-op-safe additive input
    else:
        regime = "kc1"
        if len(answerable) == 0:
            return None
        cids = _pick_concepts(answerable, V, 1, rng, member_count, "random")

    # refusal-band concepts contribute a token with value 0.0 -- asked, and the user could not answer
    cvals = cap_negatives([float(V[c]) if B[c] != BAND_REFUSE else 0.0 for c in cids])
    crefs = [1.0 if B[c] == BAND_REFUSE else 0.0 for c in cids]   # marks refusal vs a genuine meh

    # member-drop: the revealed concepts' own rated members leave the item input, so the concept token
    # must carry signal the items do not already supply
    if len(cids):
        support = [np.intersect1d(pool_s, member_sets[int(c)]) for c in cids]
        drop = np.unique(np.concatenate(support)) if support else np.empty(0, np.int64)
    else:
        drop = np.empty(0, np.int64)
    mk = ~np.isin(pool_s, drop)
    rem_s, rem_l = pool_s[mk], pool_l[mk]

    if rng.random() < P_CONC_ONLY or len(rem_s) == 0:
        in_s = np.empty(0, np.int64); in_l = np.empty(0, np.int64)
    else:
        k = int(rng.integers(1, min(ITEM_KMAX, len(rem_s)) + 1))
        pick = rng.choice(len(rem_s), size=k, replace=False)
        in_s, in_l = rem_s[pick], rem_l[pick]

    tgt = np.setdiff1d(tg, in_s)
    if len(tgt) == 0:
        return None
    return (in_s, in_l, [int(c) for c in cids], [float(v) for v in cvals],
            [float(x) for x in crefs], tgt, regime, n_ref)
