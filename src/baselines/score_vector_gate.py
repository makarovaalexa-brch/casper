r"""score_vector_gate.py -- GATE G-F, on synthetic inputs: does each arm's `score_vector` really
emit the vector its own `select()` argmaxes?

Design sheet `docs/design/DS_ACT2_SURROGATE_RANK_CORR.md` section 7, gate G-F. The C1 measurement
compares a selector's acquisition score against the realised NDCG@10 gain; if the emitted vector is
not *the* vector the selector argmaxes, the correlation measures the wrong object and every row in
the table is wrong in a way no downstream check would catch.

WHY SYNTHETIC, AND WHY THAT IS ENOUGH HERE. G-F is a statement about MASKING AND ARGMAX AGREEMENT --
"whatever this arm scores, its own action is the argmax of what it emits". That is a property of the
plumbing, not of the numbers, so a stub posterior with random factors exercises it exactly as well as
the certified stack and costs no CPU. It is run again against the REAL selectors and the REAL banked
states inside `--surrogates`, where the same assertion fires per row per state; this file is the
version that can run while the box belongs to another job.

The stub belief implements only what `_PosteriorSelector` touches: `Y`, `posterior(cols, vals)`,
`precision(cols)`, `inv_s2`. It is the conjugate Gaussian update written out, so the arms exercise
their real code paths (Cholesky, Sherman-Morrison, the width cache) rather than a mock.

  python src/baselines/score_vector_gate.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import bandit_selectors as B                                             # noqa: E402

N_ITEMS = 240
F = 12
N_USERS = 60
SEED = 98765


class StubBelief:
    """The conjugate Gaussian posterior of `belief_mf`, written out over random factors."""

    def __init__(self, n_items=N_ITEMS, f=F, seed=SEED):
        rng = np.random.default_rng(seed)
        self.Y = rng.normal(size=(n_items, f)) / np.sqrt(f)
        self.mu0 = rng.normal(size=f) * 0.1
        self.S0inv = np.eye(f) * 2.0
        self.inv_s2 = 4.0

    def precision(self, cols):
        A = self.S0inv.copy()
        if len(cols):
            Ys = self.Y[np.asarray(cols, np.int64)]
            A = A + self.inv_s2 * (Ys.T @ Ys)
        return A

    def posterior(self, cols, vals):
        A = self.precision(cols)
        b = self.S0inv @ self.mu0
        if len(cols):
            Ys = self.Y[np.asarray(cols, np.int64)]
            b = b + self.inv_s2 * (Ys.T @ np.asarray(vals, np.float64))
        return np.linalg.solve(A, b), A


def synthetic_state(rng, n_asked, n_items=N_ITEMS):
    """(asked, answered) for one user. Some asks are BURNED (asked, never answered) on purpose --
    that is the case where the asked-mask and the posterior key disagree, and it is where a wrong
    mask would hide."""
    asked = rng.choice(n_items, size=n_asked, replace=False).astype(np.int64).tolist()
    answered = []
    for i in asked:
        if rng.random() < 0.6:                       # ~40% of asks burn, as on the real cohort
            answered.append((int(i), int(rng.integers(1, 11))))
    return asked, answered


def check_arm(sel, name, pool=None, turns=(0, 1, 4, 8), seed=SEED):
    """G-F for one arm: over several turns and a population of mixed states, every user's
    `argmax(score_vector)` must equal `select`'s action, and the emitted vector must be masked --
    the user's own asked items must be unscoreable."""
    rng = np.random.default_rng(seed)
    ctx = {"n_users": N_USERS, "n_items": N_ITEMS, "q_max": 16, "D": None, "glob": {}, "bank": None,
           "log": lambda *a, **k: None, "cand_pool": pool, "conc_state": None}
    sel.reset(ctx)
    users = np.arange(N_USERS, dtype=np.int64)
    n_checked = 0
    for turn in turns:
        asked, answered = {}, {}
        for u in range(N_USERS):
            a, w = synthetic_state(rng, turn)
            asked[u], answered[u] = a, w
        act = sel.select(turn, users, asked, answered)
        vec = sel.score_vector(turn, users, asked, answered)
        assert vec is not None, f"{name}: score_vector returned None"
        assert vec.shape == (N_USERS, N_ITEMS), f"{name}: shape {vec.shape}"
        for j, u in enumerate(users.tolist()):
            got = int(np.argmax(vec[j]))
            assert got == int(act[j]), (f"{name}: turn {turn} user {u}: score_vector argmax {got} "
                                        f"!= select action {int(act[j])} -- G-F FAILED")
            if asked[u]:
                m = vec[j][np.asarray(asked[u], np.int64)]
                assert np.all(~np.isfinite(m) | (m <= -1e17)), \
                    f"{name}: an already-asked item is still scoreable -- the mask did not apply"
            if pool is not None:
                out_of_pool = np.setdiff1d(np.arange(N_ITEMS), np.asarray(pool, np.int64))
                m2 = vec[j][out_of_pool]
                assert np.all(~np.isfinite(m2) | (m2 <= -1e17)), \
                    f"{name}: an out-of-pool candidate is still scoreable"
            n_checked += 1
    return n_checked


def main():
    bel = StubBelief()
    pool = np.sort(np.random.default_rng(1).choice(N_ITEMS, size=120, replace=False))
    rows = [
        ("glm_ucb", lambda p: B.GLMUCBSelector(bel, alpha=1.0, pool=p)),
        ("bald", lambda p: B.BALDSelector(bel, pool=p)),
        ("aopt", lambda p: B.AOptimalSelector(bel, pool=p)),
        ("thompson", lambda p: B.ThompsonSelector(bel, seed=SEED, pool=p)),
    ]
    print("=" * 92)
    print("GATE G-F (synthetic): argmax(score_vector) == select(), and the masks really mask")
    print("=" * 92)
    total = 0
    for name, mk in rows:
        n1 = check_arm(mk(None), name + " [no pool]")
        n2 = check_arm(mk(pool), name + " [pooled]", pool=pool)
        total += n1 + n2
        print(f"  {name:10s} {n1 + n2:5d} (user, turn) cells   PASS")

    # Thompson is the one randomised arm: its vector is a DRAW, so the gate above also proves the
    # single-user form reproduces the block form bit-identically. Check that explicitly, because it
    # is the property that makes the row reportable at all.
    t = B.ThompsonSelector(bel, seed=SEED, pool=None)
    ctx = {"n_users": N_USERS, "n_items": N_ITEMS, "q_max": 16, "D": None, "glob": {}, "bank": None,
           "log": lambda *a, **k: None, "cand_pool": None, "conc_state": None}
    t.reset(ctx)
    rng = np.random.default_rng(7)
    asked, answered = {}, {}
    for u in range(N_USERS):
        a, w = synthetic_state(rng, 5)
        asked[u], answered[u] = a, w
    users = np.arange(N_USERS, dtype=np.int64)
    v1 = t.score_vector(3, users, asked, answered)
    v2 = t.score_vector(3, users, asked, answered)
    assert np.array_equal(v1, v2), "thompson: the (seed, turn, user) draw is not reproducible"
    v3 = t.score_vector(4, users, asked, answered)
    assert not np.array_equal(v1, v3), "thompson: the draw does not depend on the turn"
    print(f"  thompson   draw reproducible across calls at fixed turn, and turn-dependent   PASS")

    # ---------------------------------------------------------------- the refactor is behaviour-free
    # `_pick`'s loop body was MOVED into `_mask_vec` and GLM-UCB's inline index was MOVED into
    # `_ucb_vec`. A pure extraction must not move a selection, and "must not" is worth an assert:
    # replay the PRE-REFACTOR masking verbatim and require the same action, item for item.
    def legacy_pick(base_scores, asked_u, sel):
        s = np.asarray(base_scores).copy()
        if sel.pool is not None:
            keep = np.full(s.shape, B.NEG, dtype=s.dtype)
            keep[sel.pool] = s[sel.pool]
            s = keep
        s[np.asarray(asked_u, dtype=np.int64)] = B.NEG
        return int(np.argmax(s))

    n_reg = 0
    for name, mk in rows:
        for p in (None, pool):
            sel = mk(p)
            sel.reset(ctx | {"cand_pool": p})
            rng = np.random.default_rng(11)
            for turn in (0, 2, 6):
                asked, answered = {}, {}
                for u in range(N_USERS):
                    a, w = synthetic_state(rng, turn)
                    asked[u], answered[u] = a, w
                act = sel.select(turn, users, asked, answered)
                for j, u in enumerate(users.tolist()):
                    key = tuple(answered[u])
                    want = legacy_pick(sel._acq_vec(key, turn, u), asked[u], sel)
                    assert want == int(act[j]), (f"{name}: the extraction MOVED a selection at turn "
                                                 f"{turn} user {u}: legacy {want} != {int(act[j])}")
                    n_reg += 1
    print(f"  refactor   {n_reg} actions identical to the pre-extraction masking          PASS")
    print(f"TOTAL {total} (user, turn) cells checked -- G-F PASSES on the posterior family")


if __name__ == "__main__":
    main()
