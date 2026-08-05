r"""mask_contract.py -- POLICY B candidate pool + the two invariants that keep it honest.

RULING (2026-08-05, `external_literature/findings/candidate_masking_protocol.md` section 3):

    POLICY B -- the candidate pool masks EVERY rated fold-in-side item (likes AND dislikes),
    fixed per user BEFORE the interview starts, identical for every arm, every channel, every
    recommender and every budget. An item is never removed from the candidate list because it was
    ASKED about. Held-out targets are never in the fold-in, so they stay rankable.

`src/baselines/arm_n.py::build_pool` is the reference implementation (`binarize(g_te_tr) |
binarize(te_tr)`); this module is the same object for the instrument-side cohorts, plus the two
assertions the ruling asks for (section 3, item 3):

  (a) within a budget, the mask used is BIT-IDENTICAL across arms;
  (b) the target matrix is NEVER modified by any arm.

Both are enforced in the STRONGER form that is actually true under policy B: for a given cohort
slice, the mask and the target set must be bit-identical across every arm AND every budget, because
neither is allowed to depend on what was asked. That catches (a) as a special case and catches (b)
directly -- `strategy_ladder.py` used to do `te[j, i] = 0` for every asked item, which changes the
target fingerprint the moment a second arm is scored.

Policy A (mask the liked fold-in only, Liang verbatim) is RETAINED for the full-profile
certification table and for the greedy run's certification-bridge gate. It is not this module's
business; nothing here touches it.

USAGE
  pool = policy_b_pool(ctx.allb_mask, ctx.va_tr, targets=ctx.va_te)     # csr cohorts
  assert_fixed_mask("greedy_env", rows, mask_csr, te_csr)               # per scored batch
  assert_fixed_user_mask("cold_check", uid, pool_ids, target_ids)       # per scored user
"""
import hashlib

import numpy as np
from scipy import sparse

__all__ = ["policy_b_pool", "assert_fixed_mask", "assert_fixed_user_mask", "reset_contract",
           "contract_state"]

_REG = {}


# ============================================================================= the pool
def policy_b_pool(rated, foldin=None, targets=None, log=None):
    """The policy-B candidate exclusion for a cohort: every rated fold-in-side item, likes u dislikes.

    `rated`   -- the all-bands reconstruction (every catalogue rating of the cohort user EXCLUDING
                 their held-out targets). On the test cohort this is exactly arm N's `g_te_tr`; on
                 the instrument's cohorts it is `ctx.allb_mask`.
    `foldin`  -- the canonical binary (>3.5) fold-in, unioned in DEFENSIVELY rather than assumed to
                 be a subset: the binary fold-in comes from the split's own csv while the all-bands
                 matrix is a catalogue-ratings reconstruction, and nothing has ever asserted the two
                 agree (same reasoning as arm_n.build_pool).
    `targets` -- if given, the held-out target matrix; the pool/target leak check is run here so it
                 cannot be skipped by a caller.
    """
    P = rated.tocsr().copy()
    P.data[:] = 1.0
    if foldin is not None:
        B = foldin.tocsr().copy()
        B.data[:] = 1.0
        P = (P + B).tocsr()
        P.data[:] = 1.0
    P.eliminate_zeros()
    if targets is not None:
        leak = P.multiply(targets.tocsr()).nnz
        assert leak == 0, f"POOL/TARGET LEAK: {leak} held-out targets are inside the exclusion mask"
    if log is not None:
        n = max(P.shape[0], 1)
        extra = "" if foldin is None else f" (binary fold-in {foldin.nnz / n:.1f}/user)"
        log(f"[policy-B] pool nnz={P.nnz} = {P.nnz / n:.1f} items/user{extra}; target leak 0")
    return P


# ============================================================================= the invariants
def _fp_csr(X):
    X = X.tocsr()
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(X.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(X.indices, dtype=np.int64).tobytes())
    h.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    return h.hexdigest()


def _fp_ids(a):
    a = np.unique(np.asarray(a, dtype=np.int64))
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def _check(tag, key, mask_fp, tgt_fp, what):
    prev = _REG.setdefault(tag, {}).get(key)
    if prev is None:
        _REG[tag][key] = (mask_fp, tgt_fp)
        return
    assert prev[0] == mask_fp, (
        f"MASK CONTRACT (a) VIOLATED [{tag}/{what}={key}]: the candidate mask is not bit-identical "
        f"across arms/budgets. Under policy B the pool is a property of the PROTOCOL, not of the "
        f"arm -- an arm-dependent pool makes two arms score on different haystacks and voids the "
        f"paired bootstrap. See external_literature/findings/candidate_masking_protocol.md section 3.")
    assert prev[1] == tgt_fp, (
        f"TARGET CONTRACT (b) VIOLATED [{tag}/{what}={key}]: the held-out target set changed. No arm "
        f"may modify the target matrix -- two arms scored against different ground truths are not on "
        f"a common ruler (this is exactly the `te[j, i] = 0` deletion removed from strategy_ladder).")


def assert_fixed_mask(tag, rows, mask_csr, te_csr):
    """(a)+(b) for a scored batch. `rows` identifies the cohort slice; mask/te are its csr slices."""
    _check(tag, _fp_ids(rows), _fp_csr(mask_csr), _fp_csr(te_csr), "rows")


def assert_fixed_user_mask(tag, uid, pool_ids, target_ids):
    """(a)+(b) for one scored user, for harnesses that carry per-user id lists instead of matrices."""
    _check(tag, int(uid), _fp_ids(pool_ids), _fp_ids(target_ids), "user")


def reset_contract(tag=None):
    """Drop the recorded fingerprints. ONLY legitimate when the cohort itself is rebuilt."""
    if tag is None:
        _REG.clear()
    else:
        _REG.pop(tag, None)


def contract_state():
    return {t: len(v) for t, v in _REG.items()}
