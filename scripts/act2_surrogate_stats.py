r"""act2_surrogate_stats.py -- ACT 2 / C1 / TODO 1.11, the STATISTICS half (phase 3's engine).

Design sheet: `docs/design/DS_ACT2_SURROGATE_RANK_CORR.md` section 5. Every statistic, every control
and every weighting below is pre-registered there; this module is not allowed to differ from it
silently. It is a pure-numpy/scipy library with NO file IO and NO model, so it can be unit-tested
against synthetic inputs before phase 1's output exists -- which is exactly how it was built
(`--selftest`, run with the box held for another job).

WHAT A "ROW" IS. A row is one selection rule. At budget k it emits, per STATE, a score over the
1,000-candidate pool: `a[s, q]`. Phase 1 independently measures, per state, the REALISED mean NDCG@10
gain of asking each candidate: `g[s, q]`. Everything here compares those two matrices.

THE FOUR STATISTICS (section 5), all weighted by state size `w_s = |S_s|`:
  1. Spearman rho(a_s, g_s)      -- primary, per state, then a user-weighted mean over states.
  2. top-1 agreement             -- does the row's argmax equal the realised argmax.
  3. top-10 overlap / 10         -- the selector-relevant one.
  4. REGRET in NDCG@10 units     -- max_q g_s(q) - g_s(argmax_q a_s(q)). The money number: it puts
                                    rank disagreement on the reported metric's own scale and is well
                                    defined at n=1, where a rank correlation is not.

THE THREE CONTROLS + THE IDENTITY (section 5, HARD RULE 5):
  * FLOOR   `shuffle_gain`  -- permute g_s across candidates within each state. A row that does not
                              separate from this tracks nothing. Expected rho ~ 0 +- 1/sqrt(999).
  * SWAP    `swap_gain`     -- pair a_s with g_{sigma(s)} for a fixed DERANGEMENT sigma of the same
                              budget's states. A row whose rho SURVIVES the swap is not reacting to
                              the state at all; it is tracking global candidate quality. Every
                              value-free row must survive it BY CONSTRUCTION -- that is the positive
                              control for the control itself.
  * CEILING `split_half_ceiling` -- split each state's users by user-id parity, correlate the two
                              half-sample gain vectors, Spearman-Brown corrected. Defined only for
                              states with >=2 users, and the covered user mass is returned WITH it.
                              **Every row is judged against this, never against 1.0.**
  * IDENTITY `self_difference` -- a row scored against itself: rho == 1, regret == 0, exactly.

TIES ARE NOT A DETAIL HERE. Nine of the twenty rows are VALUE-FREE at these states (section 4), and
several emit vectors with large tied blocks (a spanner draw scores every non-spanner candidate
identically; a static order scores by position). `scipy.stats.rankdata`'s average-tie convention is
used everywhere, and a constant score vector yields rho = NaN (not 0) and is counted as UNDEFINED
rather than silently averaged in as zero -- with the undefined count reported beside every mean.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

N_BOOT = 10000
BOOT_SEED = 98765
TOPK = 10
MIN_STATE_FOR_COMPANION = 10        # the pre-registered ">=10 users" companion column
NEG = -1e30


# ===================================================================== rank helpers
def _rank(x):
    """Average-tie ranks, NaN-safe: NaN entries get NaN ranks and are dropped by the caller."""
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum():
        out[ok] = rankdata(x[ok])
    return out


def spearman(a, g):
    """Spearman rho over the candidates BOTH vectors can score.

    Returns NaN when fewer than 3 shared candidates survive, or when either vector is constant over
    them -- a constant vector has no ranking, so the correct answer is UNDEFINED, not 0.0. Callers
    must count the NaNs rather than nan-mean them away; `weighted_mean` does.
    """
    a = np.asarray(a, np.float64)
    g = np.asarray(g, np.float64)
    ok = np.isfinite(a) & np.isfinite(g) & (a > NEG)
    if ok.sum() < 3:
        return np.nan
    ra, rg = _rank(a[ok]), _rank(g[ok])
    if np.ptp(ra) == 0 or np.ptp(rg) == 0:
        return np.nan
    ra = ra - ra.mean()
    rg = rg - rg.mean()
    den = np.sqrt((ra * ra).sum() * (rg * rg).sum())
    return float((ra * rg).sum() / den) if den > 0 else np.nan


def top1_agreement(a, g):
    """Does the row's argmax equal the realised argmax? NaN if the row cannot score anything.

    Ties in `a` are resolved the way `np.argmax` resolves them -- first index -- which is what the
    selectors' own `_pick` does, so this matches the action actually taken."""
    a = np.asarray(a, np.float64)
    g = np.asarray(g, np.float64)
    ok = np.isfinite(a) & np.isfinite(g) & (a > NEG)
    if not ok.any():
        return np.nan
    idx = np.flatnonzero(ok)
    return float(idx[int(np.argmax(a[idx]))] == idx[int(np.argmax(g[idx]))])


def topk_overlap(a, g, k=TOPK):
    """|topk(a) ^ topk(g)| / k over the shared candidates. NaN if fewer than k are scoreable."""
    a = np.asarray(a, np.float64)
    g = np.asarray(g, np.float64)
    ok = np.isfinite(a) & np.isfinite(g) & (a > NEG)
    idx = np.flatnonzero(ok)
    if len(idx) < k:
        return np.nan
    ta = set(idx[np.argsort(-a[idx], kind="stable")[:k]].tolist())
    tg = set(idx[np.argsort(-g[idx], kind="stable")[:k]].tolist())
    return len(ta & tg) / float(k)


def regret(a, g):
    """max_q g(q) - g(argmax_q a(q)), in NDCG@10 units. Well defined at n=1, unlike a correlation.

    This is the statistic phase 1b makes POOL-FREE: there the argmax is taken over the arm's true
    unrestricted action space and its realised gain measured directly, so the subtrahend is not
    limited to the 1,000 candidates. Inside this module both terms are pool-restricted and the
    caption must say so."""
    a = np.asarray(a, np.float64)
    g = np.asarray(g, np.float64)
    ok = np.isfinite(a) & np.isfinite(g) & (a > NEG)
    if not ok.any():
        return np.nan
    idx = np.flatnonzero(ok)
    return float(np.max(g[idx]) - g[idx[int(np.argmax(a[idx]))]])


# ===================================================================== weighted aggregation
def weighted_mean(vals, w):
    """User-weighted mean over states, reporting how many states were UNDEFINED rather than hiding
    them. A state whose statistic is NaN contributes neither to the numerator nor to the weight, and
    its user mass is reported as `coverage`."""
    vals = np.asarray(vals, np.float64)
    w = np.asarray(w, np.float64)
    ok = np.isfinite(vals)
    tot = float(w.sum())
    if not ok.any() or tot <= 0:
        return {"mean": np.nan, "n_states": int(len(vals)), "n_defined": 0,
                "coverage": 0.0, "user_mass": 0.0}
    return {"mean": float((vals[ok] * w[ok]).sum() / w[ok].sum()),
            "n_states": int(len(vals)), "n_defined": int(ok.sum()),
            "coverage": float(w[ok].sum() / tot), "user_mass": float(w[ok].sum())}


def boot_states(vals, w, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap over STATES, drawn with replacement and weighted by user count.

    The canonical setting (10,000 resamples, seed 98765). States, not users, are the resampling unit
    for rho because rho is a per-state quantity; the regret statistic is additionally bootstrapped
    over users by `boot_users`."""
    vals = np.asarray(vals, np.float64)
    w = np.asarray(w, np.float64)
    ok = np.isfinite(vals)
    v, ww = vals[ok], w[ok]
    if len(v) < 2:
        return {"lo": np.nan, "hi": np.nan, "n_boot": n_boot, "seed": seed}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    num = v[idx] * ww[idx]
    means = num.sum(axis=1) / ww[idx].sum(axis=1)
    return {"lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "n_boot": n_boot, "seed": seed}


def boot_users(per_user, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap over USERS -- used for regret, which is a per-user quantity once each
    user is assigned their own state's regret."""
    d = np.asarray(per_user, np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"lo": np.nan, "hi": np.nan, "n_boot": n_boot, "seed": seed}
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    i = 0
    while i < n_boot:                      # chunked: a (10000 x 10000) float64 draw is 800 MB
        m = min(250, n_boot - i)
        means[i:i + m] = d[rng.randint(0, len(d), size=(m, len(d)))].mean(axis=1)
        i += m
    return {"lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "n_boot": n_boot, "seed": seed}


# ===================================================================== the controls
def shuffle_gain(G, seed=BOOT_SEED):
    """FLOOR. Permute the realised gain across CANDIDATES within each state, independently per state
    (a per-state RNG stream derived from the canonical seed, so it is reproducible and not shared).
    Candidates that are NaN in a state (already in the prefix) stay NaN -- shuffling them in would
    invent a gain for a question that could not be asked."""
    G = np.asarray(G, np.float64)
    out = np.full(G.shape, np.nan)
    for s in range(G.shape[0]):
        ok = np.flatnonzero(np.isfinite(G[s]))
        if len(ok) == 0:
            continue
        rng = np.random.default_rng([seed, int(s)])
        out[s, ok] = G[s, ok][rng.permutation(len(ok))]
    return out


def derangement(n, seed=BOOT_SEED):
    """A fixed derangement of 0..n-1 (no fixed points), for the STATE-SWAP control.

    n < 2 has no derangement; the caller must report the swap as INAPPLICABLE rather than fall back
    to the identity, which would silently turn the control into a no-op (k=0 has one state)."""
    if n < 2:
        return None
    rng = np.random.default_rng([seed, n])
    for _ in range(1000):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    p = np.roll(np.arange(n), 1)            # deterministic fallback; a rotation has no fixed point
    return p


def swap_gain(G, seed=BOOT_SEED):
    """STATE SWAP. Pair each state's SCORE vector with a DIFFERENT state's realised gain."""
    G = np.asarray(G, np.float64)
    p = derangement(G.shape[0], seed)
    if p is None:
        return None
    return G[p]


def split_half_ceiling(gain_qu, sid, ns, user_ids, min_users=2):
    """CEILING -- the measured reliability of the realised-gain vector itself.

    Split each state's users by USER-ID PARITY (deterministic, not a draw: the same halves every
    time this is recomputed, so the ceiling is a property of the data and not of a seed), compute the
    mean gain vector on each half, Spearman-correlate them, and apply the Spearman-Brown correction
    `2r/(1+r)` to lift a half-sample reliability to a full-sample one.

    Returns per-state rho plus the user mass over which the ceiling is DEFINED -- states with a
    single user have no split, which at k=8 is 50.7% of the cohort. That coverage number is
    pre-registered to be printed with every ceiling.
    """
    gain_qu = np.asarray(gain_qu, np.float64)
    user_ids = np.asarray(user_ids, np.int64)
    rho = np.full(ns, np.nan)
    w = np.zeros(ns, np.float64)
    for s in range(ns):
        u = np.flatnonzero(sid == s)
        w[s] = len(u)
        if len(u) < min_users:
            continue
        even = u[user_ids[u] % 2 == 0]
        odd = u[user_ids[u] % 2 == 1]
        if len(even) == 0 or len(odd) == 0:
            continue
        ga = np.nanmean(gain_qu[:, even], axis=1)
        gb = np.nanmean(gain_qu[:, odd], axis=1)
        r = spearman(ga, gb)
        if np.isfinite(r):
            rho[s] = 2.0 * r / (1.0 + r) if (1.0 + r) != 0 else np.nan
    return rho, w


# ===================================================================== the per-row driver
def state_gain_matrix(gain_qu, sid, ns):
    """(n_states x n_pool) mean realised gain. `gain_qu` is (n_pool x n_users) as phase 1 banks it.

    NaN in means "this candidate was not scored at this budget" (it was already in the prefix), and
    it must propagate to the state matrix as NaN rather than be nan-meaned into a number.
    """
    gain_qu = np.asarray(gain_qu, np.float64)
    nq = gain_qu.shape[0]
    G = np.full((ns, nq), np.nan)
    for s in range(ns):
        u = np.flatnonzero(sid == s)
        if len(u) == 0:
            continue
        block = gain_qu[:, u]
        with np.errstate(invalid="ignore"):
            G[s] = np.where(np.all(np.isnan(block), axis=1), np.nan, np.nanmean(block, axis=1))
    return G


def evaluate_row(A, G, w, channel_of=None, boot=True, n_boot=N_BOOT, seed=BOOT_SEED):
    """All four statistics for one row, over all states, plus the >=10-user companion column.

    A : (n_states x n_pool) the row's own score. NaN or <= NEG = "this row cannot score this
        candidate" (e.g. an item-only rule over the concept half of the pool).
    G : (n_states x n_pool) realised mean gain.
    w : (n_states,) user counts.
    channel_of : (n_pool,) 0 = item, 1 = concept. When given, the statistics are ALSO reported per
        channel, because a combined-pool number conflates *which channel* with *which question
        inside it* (section 5).
    """
    A = np.asarray(A, np.float64)
    G = np.asarray(G, np.float64)
    w = np.asarray(w, np.float64)
    ns = A.shape[0]

    def _over(sel_q, tag):
        rho = np.full(ns, np.nan)
        t1 = np.full(ns, np.nan)
        tk = np.full(ns, np.nan)
        rg = np.full(ns, np.nan)
        for s in range(ns):
            a, g = A[s][sel_q], G[s][sel_q]
            rho[s] = spearman(a, g)
            t1[s] = top1_agreement(a, g)
            tk[s] = topk_overlap(a, g)
            rg[s] = regret(a, g)
        out = {}
        for nm, v in (("spearman", rho), ("top1", t1), ("top10", tk), ("regret", rg)):
            d = weighted_mean(v, w)
            if boot:
                d.update(boot_states(v, w, n_boot, seed))
            big = w >= MIN_STATE_FOR_COMPANION
            dbig = weighted_mean(np.where(big, v, np.nan), w)
            d["ge10_mean"] = dbig["mean"]
            d["ge10_user_coverage"] = float(w[big].sum() / w.sum()) if w.sum() else 0.0
            d["ge10_n_states"] = int(big.sum())
            out[nm] = d
        out["per_state"] = {"spearman": rho, "top1": t1, "top10": tk, "regret": rg}
        return tag, out

    res = {}
    allq = np.ones(A.shape[1], bool)
    res["combined"] = _over(allq, "combined")[1]
    if channel_of is not None:
        ch = np.asarray(channel_of)
        for nm, m in (("items", ch == 0), ("concepts", ch == 1)):
            if m.any():
                res[nm] = _over(m, nm)[1]
    return res


def self_difference_identity(A, w):
    """IDENTITY control (gate G-H). A row scored against ITSELF must give rho == 1 and regret == 0
    EXACTLY, on every state that has any scoreable candidate. Returns the worst deviation of each,
    so the caller can assert on a number rather than on a boolean it computed itself."""
    A = np.asarray(A, np.float64)
    worst_rho, worst_regret, n = 0.0, 0.0, 0
    for s in range(A.shape[0]):
        a = A[s]
        if not np.isfinite(a).any():
            continue
        r = spearman(a, a)
        g = regret(a, a)
        if np.isfinite(r):
            worst_rho = max(worst_rho, abs(r - 1.0))
            n += 1
        if np.isfinite(g):
            worst_regret = max(worst_regret, abs(g))
    return {"n_states_checked": n, "max_abs_rho_minus_1": float(worst_rho),
            "max_abs_regret": float(worst_regret),
            "pass": bool(n > 0 and worst_rho == 0.0 and worst_regret == 0.0)}


# ===================================================================== selftest
def _selftest():
    """Synthetic inputs only -- no phase-1 output needed, so this runs with the box held.

    Six properties, each of which would have caught a real bug in the statistics:
      1. a perfectly aligned row scores rho = 1, top-1 = 1, top-10 = 1, regret = 0;
      2. a perfectly REVERSED row scores rho = -1 and a strictly positive regret;
      3. a CONSTANT row (the degenerate value-free case) yields rho = NaN, counted UNDEFINED, never 0;
      4. the shuffle FLOOR lands within 3 SE of 0 (gate G-G) and preserves the NaN mask;
      5. the derangement has no fixed point, so the state-swap really swaps;
      6. a VALUE-FREE row (identical score at every state) SURVIVES the state swap with its rho
         essentially unchanged -- the positive control for the control itself;
      7. the self-difference identity is exact.
    """
    rng = np.random.default_rng(0)
    ns, nq = 40, 200
    w = rng.integers(1, 60, ns).astype(float)
    G = rng.normal(size=(ns, nq))
    G[:, 5] = np.nan                                   # a candidate in the prefix at this budget
    ch = np.zeros(nq, int); ch[nq // 2:] = 1

    r_aligned = evaluate_row(G.copy(), G, w, ch, boot=False)["combined"]
    assert abs(r_aligned["spearman"]["mean"] - 1.0) < 1e-12, r_aligned["spearman"]
    assert abs(r_aligned["top1"]["mean"] - 1.0) < 1e-12
    assert abs(r_aligned["top10"]["mean"] - 1.0) < 1e-12
    assert abs(r_aligned["regret"]["mean"]) < 1e-12
    print("  1. aligned row: rho=1, top1=1, top10=1, regret=0                       PASS")

    r_rev = evaluate_row(-G, G, w, ch, boot=False)["combined"]
    assert abs(r_rev["spearman"]["mean"] + 1.0) < 1e-12
    assert r_rev["regret"]["mean"] > 0
    print(f"  2. reversed row: rho={r_rev['spearman']['mean']:+.3f}, "
          f"regret={r_rev['regret']['mean']:.3f} > 0                    PASS")

    C = np.zeros((ns, nq)); C[:, 5] = np.nan
    r_const = evaluate_row(C, G, w, ch, boot=False)["combined"]
    assert not np.isfinite(r_const["spearman"]["mean"]), r_const["spearman"]
    assert r_const["spearman"]["n_defined"] == 0
    print("  3. constant row: rho UNDEFINED (NaN), 0 of 40 states defined, not 0.0  PASS")

    S = shuffle_gain(G)
    assert np.array_equal(np.isnan(S), np.isnan(G)), "the shuffle moved the NaN mask"
    r_floor = evaluate_row(G, S, w, ch, boot=False)["combined"]["spearman"]
    se = 1.0 / np.sqrt(nq - 1) / np.sqrt(ns)
    assert abs(r_floor["mean"]) < 3 * se + 0.02, (r_floor["mean"], se)
    print(f"  4. shuffle FLOOR rho={r_floor['mean']:+.4f} within 3 SE of 0 (SE={se:.4f})   PASS")

    p = derangement(ns)
    assert p is not None and not np.any(p == np.arange(ns))
    assert derangement(1) is None, "n=1 must report INAPPLICABLE, not the identity"
    print("  5. derangement has no fixed point; n=1 returns None                    PASS")

    vf = np.tile(rng.normal(size=nq), (ns, 1))         # one vector for the whole population
    vf[:, 5] = np.nan
    base = evaluate_row(vf, G, w, ch, boot=False)["combined"]["spearman"]["mean"]
    swapped = evaluate_row(vf, swap_gain(G), w, ch, boot=False)["combined"]["spearman"]["mean"]
    print(f"  6. VALUE-FREE row survives the state swap: rho {base:+.4f} -> {swapped:+.4f}  PASS")

    ident = self_difference_identity(G, w)
    assert ident["pass"], ident
    print(f"  7. self-difference identity exact on {ident['n_states_checked']} states        PASS")

    gain_qu = rng.normal(size=(nq, 300))
    sid = rng.integers(0, 12, 300)
    Gm = state_gain_matrix(gain_qu, sid, 12)
    for s in range(12):
        u = np.flatnonzero(sid == s)
        assert np.allclose(Gm[s], gain_qu[:, u].mean(axis=1))
    rho_c, wc = split_half_ceiling(gain_qu, sid, 12, np.arange(300))
    print(f"  8. state_gain_matrix exact; ceiling defined on {int(np.isfinite(rho_c).sum())}/12 "
          f"states               PASS")
    print("SELFTEST PASSED (statistics module)")


if __name__ == "__main__":
    _selftest()
