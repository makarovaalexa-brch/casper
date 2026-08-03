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

# ---- regime mixture v3 (author + Fable 2026-08-03) --------------------------------------------------
# BROAD is GONE. It revealed 50-100% of a user's answerable concepts, but the median user can answer 573
# of 1031 -- there is no "warm concept user", so that bucket trained a state that never occurs.
P_INTERVIEW, P_DENSE, P_KC0, P_KC1 = 0.55, 0.30, 0.05, 0.10
# cv5: the COLD eval hands the model zero item history, but a uniform k~U{0..8} draw makes only ~11% of
# training examples zero-item -- the model mostly learns to fold concepts ON TOP OF items, then is scored
# folding them onto the bare prior. P_ITEM_ZERO forces the zero-item case on top of the uniform draw.
P_ITEM_ZERO = 0.0
KEEP_MEH = False        # keep the graded MEH value instead of zeroing it (see signed_for_pool)

# WARNING (2026-08-03): the kc0 regime is a PURE NO-OP for the concept bolt-on. With zero concept answers
# the fold loop never executes, U = z - z_items = 0, the output is exactly the tower's prior, and NO
# parameter of the net is touched -- the gradient is identically zero, with or without item context. A
# zero-concept bucket earns its place in the ITEM tower, which must learn a prior; here it only burns
# steps. Keep P_KC0 at 0 unless something downstream starts depending on it.

# Ask policies for the INTERVIEW regime. MEASURED answerability over 3000 users x 1031 concepts:
#   ask 50 broadest -> 97.3% answered ( 2.7% refusal)
#   ask uniformly   -> 57.4% answered (42.6% refusal)
#   ask 50 nichest  -> 16.3% answered (84%   refusal)
# Training samples these UNIFORMLY so the model sees the whole 3-84% refusal range; deployment realism
# (broad-first, coarse-to-fine) belongs in the VAL METRIC, not in the training distribution. Skewing
# training broad-first would starve exactly the refusal signal this regime exists to supply.
ASK_POLICIES = ("broad", "uniform", "niche")
P_DENSE_RARE = 0.5      # half the DENSE bucket rare-weighted, half uniform-over-answerable.
                        # "dense + rare + answerable" silently selects HEAVY RATERS, because niche
                        # concepts are answerable mainly by users with large profiles -- the bucket would
                        # confound rarity with user activity. The niche-refusal signal is already carried
                        # by INTERVIEW.

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


def set_mixture(p_interview, p_dense, p_kc0, p_kc1):
    global P_INTERVIEW, P_DENSE, P_KC0, P_KC1
    tot = p_interview + p_dense + p_kc0 + p_kc1
    assert abs(tot - 1.0) < 1e-9, f"mixture must sum to 1, got {tot}"
    P_INTERVIEW, P_DENSE, P_KC0, P_KC1 = p_interview, p_dense, p_kc0, p_kc1


def sample_kc_loguniform(rng, budgets=INTERVIEW_KCS):
    """kc sampled so mass sits at 1-8; 32 is a TAIL REGULARISER, not a target. kc=32 is ~6% of the
    median user's 573 answerable concepts, so it is not the warm-user state that killed the v2 BROAD
    bucket -- but a uniform draw over {1..32} would put a sixth of training at a length no interview
    reaches."""
    # The grid is GEOMETRIC (1,2,4,8,16,32), so a UNIFORM draw over it is already log-uniform -- and it
    # is exactly what the item tower sampled. A 1/b harmonic weighting put 50% of interviews at kc=1,
    # which with the separate kc=1 bucket would have made ~38% of all training a single question.
    return int(rng.choice(budgets))


def ask_pool(policy, n_concepts, member_count, rng, kc):
    """Draw kc concept ids from the WHOLE bank under an ask policy. Answerability is then decided by the
    USER'S HISTORY, exactly as an item ask is decided by whether they rated it -- not by pre-filtering
    the candidate set, which is what capped refusals at 2.9%."""
    mcv = np.clip(np.asarray(member_count, np.float64), RARE_CLIP[0], RARE_CLIP[1])
    if policy == "broad":
        w = mcv
    elif policy == "niche":
        w = mcv ** (-1.0)
    else:
        w = np.ones_like(mcv)
    w = w / w.sum()
    kc = min(kc, n_concepts)
    return rng.choice(n_concepts, size=kc, replace=False, p=w)


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


def draw(u, rng, Mm, grate, member_sets, item_mean, prereg, member_count=None, n_concepts=None):
    """One FRESH curriculum example (v3). Returns
        (in_s, in_l, cids, cvals, crefs, tgt, regime, n_refusal)
    or None.

    THE KEY CHANGE vs v2: the INTERVIEW regime draws its asks from the WHOLE concept bank and lets the
    USER'S HISTORY decide answerability, exactly as an item ask is decided by whether they rated it.
    v2 pre-filtered candidates to concepts the user already had >=2 rated members for, which excluded
    every concept they had never encountered and capped refusals at 2.9%. Measured truth: asking
    uniformly gives 42.6% refusals, the 50 nichest give 84%, the 50 broadest 2.7%.
    """
    items = np.asarray(u["items"], np.int64)
    lv = np.asarray(u["levels"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(items) < 4 or len(liked) < 2:
        return None

    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(items, tg)
    pool_s, pool_l = items[keep], lv[keep]
    if len(pool_s) < 2:
        return None

    stars = (pool_l.astype(np.float32) + 1.0) / 2.0
    V, B, ans = signed_for_pool(pool_s, stars, item_mean, Mm, grate, prereg, keep_meh=KEEP_MEH)
    nC = n_concepts if n_concepts is not None else len(V)
    answerable = np.flatnonzero(ans & (B != BAND_REFUSE))

    r = rng.random()
    n_ref = 0
    if r < P_INTERVIEW:
        regime = "interview"
        kc = sample_kc_loguniform(rng)
        policy = str(rng.choice(ASK_POLICIES))
        asked = ask_pool(policy, nC, member_count, rng, kc)
        # answerability decided by the user's own history, not by pre-filtering the bank
        ok = np.array([bool(ans[c]) and B[c] != BAND_REFUSE for c in asked])
        cids = asked
        n_ref = int((~ok).sum())
        regime = f"interview_{policy}"
    elif r < P_INTERVIEW + P_DENSE:
        regime = "dense"
        if len(answerable) == 0:
            return None
        kc = int(rng.integers(1, DENSE_KCMAX + 1))
        mode = "random" if rng.random() < P_DENSE_RARE else "uniform_answerable"
        if mode == "random":
            cids = _pick_concepts(answerable, V, kc, rng, member_count, "random")
        else:
            kk = min(kc, len(answerable))
            cids = rng.choice(answerable, size=kk, replace=False)
        ok = np.ones(len(cids), bool)
    elif r < P_INTERVIEW + P_DENSE + P_KC0:
        regime = "kc0"
        cids = np.empty(0, np.int64); ok = np.empty(0, bool)
    else:
        regime = "kc1"
        if len(answerable) == 0:
            return None
        cids = _pick_concepts(answerable, V, 1, rng, member_count, "random")
        ok = np.ones(len(cids), bool)

    cvals = cap_negatives([float(V[c]) if ok[j] else 0.0 for j, c in enumerate(cids)])
    crefs = [0.0 if ok[j] else 1.0 for j in range(len(cids))]

    # member-drop applies ONLY to concepts the user could actually answer: a refused concept reveals
    # nothing about their ratings, so dropping its members would leak the answer.
    ansd = [int(c) for j, c in enumerate(cids) if ok[j]]
    if ansd:
        support = [np.intersect1d(pool_s, member_sets[int(c)]) for c in ansd]
        drop = np.unique(np.concatenate(support)) if support else np.empty(0, np.int64)
    else:
        drop = np.empty(0, np.int64)
    mk = ~np.isin(pool_s, drop)
    rem_s, rem_l = pool_s[mk], pool_l[mk]

    if len(rem_s) == 0:
        in_s = np.empty(0, np.int64); in_l = np.empty(0, np.int64)
    else:
        k = 0 if rng.random() < P_ITEM_ZERO else int(rng.integers(0, min(ITEM_KMAX, len(rem_s)) + 1))
        if k == 0:
            in_s = np.empty(0, np.int64); in_l = np.empty(0, np.int64)
        else:
            pick = rng.choice(len(rem_s), size=k, replace=False)
            in_s, in_l = rem_s[pick], rem_l[pick]

    tgt = np.setdiff1d(tg, in_s)
    if len(tgt) == 0:
        return None
    return (in_s, in_l, [int(c) for c in cids], [float(v) for v in cvals],
            [float(x) for x in crefs], tgt, regime, n_ref)
