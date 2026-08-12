r"""pool_contract_guard_test.py -- proof that a policy-A partial can no longer be resumed silently.

WHAT THIS EXISTS FOR. `docs/results/POLICY_A_PROVENANCE_AUDIT.md` (2026-08-12) found that the greedy
resume sidecar's signature carried no field describing the candidate pool:

    sig = {tower, conc, objective, item_bank, conc_bank, n_items, n_concepts, L,
           n_build_users, build_head, build_tail}

Every one of those fields is INVARIANT under a change of candidate-mask policy. So a policy-B re-run
of `strategy_channel_suite.py` writing to `experiments/battery/greedy_static_i26c.partial.json` would
have matched that signature exactly and RESUMED THE POLICY-A PICKS -- silently, with no log line, and
the resulting artifact would have carried a policy-B header over a policy-A order.

The fix is `mask_contract.pool_contract`: an identity DERIVED from the mask's own sparsity structure
(nnz, items/user, blake2b of indptr+indices), not a hand-typed label -- the audit's other finding was
that the CELF heap signature keys on a DISPLAY label (`contract_tag`), which is itself invariant under
a policy change. Plus `mask_contract.resume_ok`, which treats a MISSING field as a refusal with a
named reason rather than a pass.

WHAT IS REAL HERE, AND WHAT IS NOT. Everything below runs on artifacts on disk:
  * the two pools are the REAL ones -- policy A = binarize(te_tr) (the canonical binary fold-in),
    policy B = binarize(g_te_tr) | binarize(te_tr) (`arm_n.build_pool`), on the real 10,000-user test
    cohort, read from `data/ml-25m/proc/test_tr.csv` and `.cache/baselines/graded_ml25m.npz`;
  * the refused signature is the REAL `sig` block of the REAL policy-A partial on disk;
  * `pool_contract` and `resume_ok` are the shipped functions, not re-implementations.
No recommender is loaded and no question is scored: this is a contract test, not an experiment.

    python src/instrument/pool_contract_guard_test.py
"""
import os
import sys
import json
import tempfile

import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from mask_contract import pool_contract, resume_ok            # noqa: E402

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
GRADED = os.path.join(_ROOT, ".cache", "baselines", "graded_ml25m.npz")
PARTIAL_A = os.path.join(_ROOT, "experiments", "battery", "greedy_static_i26c.partial.json")
LINES = []


def log(m):
    print(m, flush=True)
    LINES.append(str(m))


def _csr(z, name):
    return sparse.csr_matrix((z[name + ".data"], z[name + ".indices"], z[name + ".indptr"]),
                             shape=tuple(z[name + ".shape"]))


def real_pools():
    """The two REAL candidate pools of the 10k test cohort, straight off disk."""
    import pandas as pd
    n_items = len(open(os.path.join(PROC, "unique_sid.txt")).read().split())
    tp = pd.read_csv(os.path.join(PROC, "test_tr.csv"))
    u = tp["uid"].values - tp["uid"].min()
    te_tr = sparse.csr_matrix((np.ones_like(u, dtype="float32"), (u, tp["sid"].values)),
                              shape=(10000, n_items))
    z = np.load(GRADED)
    g = _csr(z, "g_te_tr")
    a = te_tr.copy(); a.data[:] = 1.0; a.eliminate_zeros()          # POLICY A: liked fold-in only
    b = g.copy(); b.data[:] = 1.0
    b = (b + a).tocsr(); b.data[:] = 1.0; b.eliminate_zeros()        # POLICY B: arm_n.build_pool
    return a, b


def main():
    ok, fail = [], []

    # ---------------------------------------------------------------- 1. the fingerprint separates
    log("=" * 104)
    log("[1] the DERIVED pool contract, on the two REAL pools of the 10,000-user test cohort")
    A, B = real_pools()
    ca = pool_contract(A, policy="A")
    cb = pool_contract(B, policy="B (arm_n.build_pool)")
    log(f"    policy A : nnz={ca['nnz']:>9,}  per_user={ca['per_user']:>7}  fp={ca['fp']}")
    log(f"    policy B : nnz={cb['nnz']:>9,}  per_user={cb['per_user']:>7}  fp={cb['fp']}")
    (ok if ca["fp"] != cb["fp"] else fail).append(
        "1a the two policies' pools have DIFFERENT fingerprints")
    # the audit's published detection signature for policy B: nnz 1,411,761 = 141.2 items/user
    (ok if (cb["nnz"] == 1411761 and abs(cb["per_user"] - 141.2) < 0.05) else fail).append(
        f"1b policy-B pool reproduces the audit's signature (nnz 1,411,761 / 141.2 per user): "
        f"got {cb['nnz']} / {cb['per_user']}")
    (ok if abs(ca["per_user"] - 62.6) < 0.1 else fail).append(
        f"1c policy-A pool reproduces the audit's ~62.6 items/user: got {ca['per_user']}")
    (ok if pool_contract(B, policy="B")["fp"] == cb["fp"] else fail).append(
        "1d the fingerprint is stable across calls (a resume can actually succeed)")
    (ok if pool_contract(B, policy="mislabelled-as-A")["fp"] == cb["fp"] else fail).append(
        "1e the LABEL cannot move the fingerprint -- the mask is the evidence, the label is not")

    # ---------------------------------------------------------------- 2. the refusal, on the real file
    log("=" * 104)
    log(f"[2] REFUSAL: the real policy-A sidecar {os.path.relpath(PARTIAL_A, _ROOT)}")
    part_a = json.load(open(PARTIAL_A))
    sig_a = part_a["sig"]
    log(f"    its banked sig fields: {sorted(sig_a)}")
    log(f"    its banked items-only head: {part_a['arms']['items-only']['sequence'][:4]} "
        f"(the picks that would have been resumed)")
    sig_b = dict(sig_a, pool_contract=cb)          # same run, today's code, policy-B pool
    log("    -> resume_ok(banked policy-A sig, this run's policy-B sig):")
    verdict = resume_ok(sig_a, sig_b, lambda m: log("       " + m), os.path.basename(PARTIAL_A))
    (ok if verdict is False else fail).append(
        "2a the real policy-A partial is REFUSED by the new guard")
    # and the pre-fix comparison, for the record
    old_style = {k: v for k, v in sig_b.items() if k != "pool_contract"}
    (ok if (sig_a == old_style) else fail).append(
        "2b under the OLD signature (no pool field) that same sidecar compared EQUAL -- i.e. the "
        "pre-fix code would have resumed the policy-A picks")
    log(f"    [pre-fix control] old-style sig == banked sig ? {sig_a == old_style}  "
        f"<- this is the defect, reproduced")

    # ---------------------------------------------------------------- 3. policy-A-stamped sidecar
    log("=" * 104)
    log("[3] REFUSAL: a sidecar that DOES state its pool, but states policy A")
    sig_a_stamped = dict(sig_a, pool_contract=ca)
    verdict = resume_ok(sig_a_stamped, sig_b, lambda m: log("       " + m), "policyA_stamped.partial")
    (ok if verdict is False else fail).append("3a a policy-A-stamped sidecar is REFUSED")

    # ---------------------------------------------------------------- 4. a genuine resume succeeds
    log("=" * 104)
    log("[4] ACCEPT: a same-policy sidecar still resumes, round-tripped through JSON")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "greedy_same_policy.partial.json")
        json.dump({"sig": sig_b, "arms": part_a["arms"]}, open(p, "w"), indent=2)
        part = json.load(open(p))                                  # exactly what greedy_main does
        verdict = resume_ok(part.get("sig"), sig_b, lambda m: log("       " + m), os.path.basename(p))
        (ok if verdict is True else fail).append(
            "4a a same-policy sidecar is ACCEPTED after a JSON round trip (floats survive)")
        resumed = sorted(part["arms"]) if verdict else []
        log(f"    accepted={verdict}; arms that would be reused instead of re-searched: {resumed}")
        (ok if resumed == ["combined", "concepts-only", "items-only"] else fail).append(
            "4b the accepted sidecar's banked arms are readable and reused")

    # ---------------------------------------------------------------- 5. every call site is wired
    log("=" * 104)
    log("[5] every instrument resume path signs the pool and gates on resume_ok")
    for f in ("strategy_channel_suite.py", "group_greedy_genre.py", "group_greedy_d1.py",
              "ndcg_tree.py"):
        src = open(os.path.join(_HERE, f)).read()
        has_sig = '"pool_contract": pool_contract(' in src
        has_gate = "resume_ok(part.get(\"sig\")" in src
        log(f"    {f:<28} pool_contract in sig={has_sig}   gated on resume_ok={has_gate}")
        (ok if (has_sig and has_gate) else fail).append(f"5 {f} signs the pool and gates on it")

    log("=" * 104)
    for s in ok:
        log(f"  PASS  {s}")
    for s in fail:
        log(f"  FAIL  {s}")
    log(f"[guard-test] {len(ok)} passed, {len(fail)} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
