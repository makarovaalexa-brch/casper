r"""M6s -- the STATIC COMBINED order, fitted on TRAIN users.

The static comparator must be fitted on the same population as the trees it is compared against.
`greedy_static_policyB_combined.json` is validation-fitted; this builds its train twin.

HOW IT IS CHEAP. A global greedy order over the whole cohort and the tree's ROOT node search are the
same computation: both are CELF over the full 1,000-candidate pool at the empty prefix on every user
in the cohort. `coldtree_v2_train10000` already paid that seed pass (1,002 evals at 7.37 s/eval,
123 minutes), so this resumes its banked root heap and extends to L=15 instead of re-seeding. That is
exactly what the validation order did with the validation tree's root heap
(`ndcg_tree_extend16.py`, TASK 1).

The heap is COPIED, never shared: `greedy_certified` writes back to `heap_path` as it picks, and the
tree's own sidecar must not be mutated by a different search.

The resume is contract-signed. `celf_heap_sig(E, rows, pool, objective, prefix)` must reproduce the
signature the tree's root banked, which pins the tower, the concept checkpoint, the split, the exact
row set (sha1), the pool (sha1) and the objective. If any of those differ the sidecar is REFUSED and
the search re-seeds cold rather than silently mixing populations -- so a mismatched cohort cannot
produce a quietly wrong order, it just costs 1.9 h.

⚠ Fitted on TRAIN, so its own NDCG is measured on a memorised ruler and is NEVER reported. This
object exists to be APPLIED to validation.

**THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import json
import shutil
import time

import numpy as np
import torch

B = os.path.join(_R, "experiments", "battery")
SEED = 98765


def main():
    ap = argparse.ArgumentParser(description="M6s -- static combined order on TRAIN users")
    ap.add_argument("--slice_lo", type=int, default=0)
    ap.add_argument("--slice_hi", type=int, default=34000)
    ap.add_argument("--n_users", type=int, default=10000)
    ap.add_argument("--total_L", type=int, default=15)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--tree_heaps", default=os.path.join(B, "coldtree_v2_train10000_heaps"))
    ap.add_argument("--label", default="greedy_static_policyB_combined_train10000")
    ap.add_argument("--no_seed_reuse", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (log, build_greedy_env, set_active_rows, greedy_certified,
                                        genv_mean, contract_selftest, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert
    from run_guard import require_mains

    require_mains(f"{a.label}", log=log)
    log("#" * 100)
    log(f"# M6s -- STATIC COMBINED ORDER on TRAIN users {a.slice_lo}..{a.slice_hi}, "
        f"L={a.total_L}")
    log("# Fitted on a memorised ruler: its own NDCG is NEVER reported. It is APPLIED to validation.")
    log("#" * 100)

    E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, a.label,
                         cohort_slice=(a.slice_lo, a.slice_hi))
    rows = E["rows"]
    if a.n_users and len(rows) > a.n_users:
        rows = list(rows)[:a.n_users]
    assert len(rows) == a.n_users, f"cohort is {len(rows)}, expected {a.n_users}"
    log(f"[cohort] {len(rows):,} TRAIN users (the same cohort coldtree_v2_train10000 was fitted on)")
    contract_selftest(E, rows)
    leak_assert(E, rows, "train")
    set_active_rows(E, rows)

    pool, _ip, _cp = build_pool(E, a.n_items, a.n_concepts)
    log(f"[pool] {len(pool)} candidates ({a.n_items} items + {a.n_concepts} concepts)")

    # ---- resume the tree's root heap, on a COPY --------------------------------------------------
    heap = os.path.join(B, f"{a.label}_heaps", "search.json")
    os.makedirs(os.path.dirname(heap), exist_ok=True)
    src = os.path.join(a.tree_heaps, "node_root.json")
    if a.no_seed_reuse:
        log("[seed] --no_seed_reuse: paying the full 1,000-candidate seed pass (~1.9 h)")
    elif os.path.exists(src) and not os.path.exists(heap):
        shutil.copyfile(src, heap)           # COPY: greedy_certified writes back as it picks
        log(f"[seed] copied the tree's root heap from {src}")
        log("[seed] if its contract does not match this cohort it will be REFUSED and re-seeded "
            "cold -- that is the guard, not a failure")
    elif os.path.exists(heap):
        log(f"[seed] resuming this run's own sidecar {heap}")
    else:
        log(f"[seed] no root heap at {src} -- seeding cold")

    t0 = time.time()
    base = genv_mean(E, [], rows)[0]
    log(f"[base] empty-interview full NDCG@10 on this cohort = {base:.8f} "
        f"(a TRAIN figure: memorised, never quoted)")
    seq, gains, nev = greedy_certified(E, rows, pool, a.total_L, "full", a.label, heap_path=heap)
    assert len(seq) == a.total_L, f"got {len(seq)} of {a.total_L} picks"

    out = {"analysis": "static_global_greedy_combined_POLICY_B_TRAIN",
           "label": a.label, "split": "train", "test_touched": False,
           "cohort_slice": [a.slice_lo, a.slice_hi], "n_cohort": len(rows),
           "tower": os.path.basename(CERT_TOWER), "concept_ckpt": os.path.basename(CERT_CONC),
           "objective": "full", "mask_policy": "B", "L": a.total_L,
           "n_items_pool": a.n_items, "n_concepts_pool": a.n_concepts, "n_pool": len(pool),
           "seed_pass_reused_from": None if a.no_seed_reuse else src,
           "n_evals": int(nev), "minutes": round((time.time() - t0) / 60, 1),
           "base_full_train_MEMORISED_DO_NOT_QUOTE": float(base),
           "sequence": [[int(c), int(i)] for c, i in seq],
           "gains_train_MEMORISED_DO_NOT_QUOTE": [float(g) for g in gains],
           "ruler_note": ("Fitted on TRAIN users, whom the tower was trained on. Every NDCG in this "
                          "file is on a memorised ruler and must NEVER be quoted or compared with a "
                          "validation or test number. What this file provides is the SEQUENCE, "
                          "which is re-scored on validation wherever it is used.")}
    p = os.path.join(B, f"{a.label}.json")
    json.dump(out, open(p, "w"), indent=1)
    log(f"[done] {a.total_L} picks, {nev} evals, {out['minutes']:.1f} min -> {os.path.basename(p)}")
    for k, (c, i) in enumerate(seq, 1):
        log(f"   {k:>2}. {'concept' if c else 'item':<8} {i}   gain {gains[k-1]:+.5f}")


if __name__ == "__main__":
    main()
