r"""interview_strategies.py -- the six lit-anchored question-selection strategies, as bare item-id
orders that ANY recommender can be pointed at.

WHY THIS MODULE. `src/instrument/strategy_ladder.py` computes these same six arms but is wired to the
tower's belief encoder and its ctx object. The interview table needs them decoupled: a strategy is a
ranked list of items to ask about, and nothing more. Formulas are reproduced from strategy_ladder
(train_entropies / arm_orders) so the two agree by construction; the only change is that the level-band
counts come from the graded TRAIN matrix directly rather than from a second pass over raw ratings.

THE ARMS, with their published expected ranking (rashid2002getting, rashid2008learning):
  helf            1  harmonic mean of normalised rating-entropy and normalised log-frequency.
                     Rashid's winner: informative AND answerable.
  entropy0        2  entropy with 'not rated' as an 11th category -- Rashid 2008's fix, which folds
                     answerability into the entropy itself.
  popularity      3  descending train count. Strong simple baseline; answerable but redundant.
  pure_entropy    4  rating entropy over the ten half-star bands, raters only. Rashid 2002 found it
                     "shockingly poor": high-entropy items are obscure, so nobody can rate them.
  random_bank     5  random order within a fixed 200-item shortlist. No exact lit counterpart.
  random_catalog  6  uniform over the full 18,359-item vocabulary. Worst: tiny answer rate.

PROTOCOL. An asked-but-unrated item BURNS the question -- that is the literature's protocol and the
whole reason random_catalog does badly. So `asked` and `answered` differ, and both are returned.

CAVEAT THAT MUST REACH THE PAPER: Rashid measured rating-prediction MAE and Golbandi RMSE; we measure
NDCG@10. The published ranks transfer as PRIORS, not as certified numbers. An inversion here is
evidence about our setting, not proof that Rashid was wrong.
"""
import numpy as np

NLEV = 10
SEED = 4242
BANK_SIZE = 200

LIT_RANK = {"helf": 1, "entropy0": 2, "popularity": 3, "pure_entropy": 4,
            "random_bank": 5, "random_catalog": 6}
CITES = {"helf": "rashid2002getting; rashid2008learning",
         "entropy0": "rashid2008learning",
         "popularity": "rashid2002getting; golbandi2010",
         "pure_entropy": "rashid2002getting",
         "random_bank": "no exact lit counterpart (protocol caveat)",
         "random_catalog": "rashid2002getting; elahi2016survey"}
ARMS = list(LIT_RANK)


def _star_to_level(r):
    return np.clip(np.rint(np.asarray(r, np.float64) * 2).astype(np.int64) - 1, 0, NLEV - 1)


def band_counts(g_train, n_items):
    """(n_items x NLEV) count of TRAIN users rating each item at each half-star level."""
    G = g_train.tocsr()
    cols = G.indices
    lvls = _star_to_level(G.data)
    C = np.zeros((n_items, NLEV), np.float64)
    np.add.at(C, (cols, lvls), 1.0)
    return C


def entropies(g_train, n_items, n_train_users):
    """(H, H0): raters-only band entropy, and Rashid-2008 entropy with 'not rated' as an 11th band."""
    C = band_counts(g_train, n_items)
    n_raters = C.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = C / np.maximum(n_raters[:, None], 1e-12)
        H = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
        C0 = np.concatenate([C, np.maximum(n_train_users - n_raters, 0.0)[:, None]], axis=1)
        p0 = C0 / max(n_train_users, 1e-12)
        H0 = -np.nansum(np.where(p0 > 0, p0 * np.log2(p0), 0.0), axis=1)
    H[n_raters == 0] = 0.0
    return H, H0


def build_orders(g_train, train, n_items, log=print):
    """Global ask-orders for the four scored arms, plus the fixed random bank.
    `train` is the BINARY train matrix -- popularity and the log-frequency leg of HELF are interaction
    counts, exactly as in strategy_ladder (which reads ctx.cnt from the binary matrix)."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    H, H0 = entropies(g_train, n_items, float(g_train.shape[0]))
    Hn = H / max(H.max(), 1e-9)
    Fn = np.log1p(cnt) / max(np.log1p(cnt).max(), 1e-9)
    helf = 2.0 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)
    glob = {"popularity": np.argsort(-cnt),
            "pure_entropy": np.argsort(-H),
            "entropy0": np.argsort(-H0),
            "helf": np.argsort(-helf)}
    bank = np.random.default_rng(SEED).choice(n_items, size=BANK_SIZE, replace=False)
    log(f"[strategies] orders built over {n_items} items; "
        f"top-5 popularity={glob['popularity'][:5].tolist()} helf={glob['helf'][:5].tolist()}")
    return glob, bank


def ask(name, glob, bank, n_users, n_items, lvl_lookup, q_max):
    """Walk the arm's order for each user; every item shown BURNS a question, answered only if the user
    has rated it. Returns (asked, answered) as lists per user: asked[u] = [sid...] (length q_max),
    answered[u] = [(sid, level)...] for the subset the user could actually answer.

    random_catalog draws q_max*4 distinct ids per user rather than materialising an 18k permutation --
    non-lossy (the walk stops at q_max asked either way), just cheaper.
    """
    asked, answered = [], []
    for u in range(n_users):
        rng = np.random.default_rng(SEED * 1000 + u)
        if name == "random_catalog":
            order = rng.choice(n_items, size=min(q_max * 4, n_items), replace=False)
        elif name == "random_bank":
            order = rng.permutation(bank)
        else:
            order = glob[name]
        d = lvl_lookup[u]
        a, t = [], []
        for i in order:
            if len(a) >= q_max:
                break
            i = int(i)
            a.append(i)
            if i in d:
                t.append((i, d[i]))
        asked.append(a)
        answered.append(t)
    return asked, answered


def level_lookup(g_te_tr):
    """Per test user: {item -> half-star level}, from their full rated history (targets already
    excluded upstream). This is the ANSWER ORACLE -- what the user would say if asked."""
    G = g_te_tr.tocsr()
    out = []
    for u in range(G.shape[0]):
        s, e = G.indptr[u], G.indptr[u + 1]
        out.append(dict(zip(G.indices[s:e].tolist(), _star_to_level(G.data[s:e]).tolist())))
    return out
