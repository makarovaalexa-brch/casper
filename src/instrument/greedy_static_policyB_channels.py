r"""A3 / A4 -- REBUILD THE SINGLE-CHANNEL STATIC GREEDY ORDERS UNDER POLICY B.

`greedy_static_i26c.json`'s `items-only` and `concepts-only` arms were SELECTED under the retired
policy-A candidate mask (its recorded intercept 0.12987336 is the policy-A m=0; policy B gives
0.15686538 on the same 10,000 validation users). Their SEQUENCES are re-scored under policy B
wherever they are used, but their ORDER was chosen on the wrong ruler -- and A1 measured what that
is worth on the combined pool: the rebuilt order shared 4/16 questions and 0/16 POSITIONS with the
policy-A one. The contamination is not an offset, it moves the argmax at every position.

This script rebuilds ONE CHANNEL's order, cold in every sense that matters:

  * the SAME 10,000 canonical validation users (every one of them -- no sampling, HARD RULE 1),
  * the SAME declared pool: `--channel items` = the 500 popularity-ranked items, `--channel
    concepts` = the 500 answerable concepts, taken by the SAME `build_pool` call the combined build
    uses so the ids and their order are identical to the combined pool's two blocks,
  * the SAME objective (cohort-mean held-out full NDCG@10) and the SAME certified stack,
  * under POLICY B, stamped with the DERIVED `pool_contract` fingerprint of the mask itself.

An items-only order is NOT the combined order with the concepts deleted, and it is not derivable
from it: greedy conditions each pick on the folded answers to everything already picked, so
restricting the pool changes every subsequent pick. It is a fresh search.

THE SEED PASS IS TRANSPLANTED, NOT RE-PAID -- AND IT IS VERIFIED, NOT TRUSTED.
  The tree's ROOT node holds a completed CELF seed pass: 1,000 candidates, EMPTY prefix, all 10,000
  validation users, policy B, `full` objective -- 999 heap entries all still at update-level 0 plus
  the one it picked. A seed gain is `mean_ndcg([q]) - base`, which depends on the candidate, the
  cohort and the environment, and NOT on which other candidates are in the pool. So every one of
  this search's 500 seed values is already on disk, exactly.

  Rather than hand-patch a memo (which would leave the heap sidecar's contract unverified), this
  script WRITES ITS OWN sidecar: a heap over ITS OWN sub-pool, signed with `celf_heap_sig` computed
  from THIS environment by the current code (v2 -- keyed on tower/concept ckpt/derived mask
  fingerprint, not on a display label). `greedy_certified` then resumes it through the normal,
  unmodified guard. Nothing is weakened; the sidecar is simply already correct. Three checks make
  that claim testable instead of asserted:

    1. The banked blob's own signature fields are matched: split, objective, empty prefix, the row
       set (sha1), and the COMBINED pool's sha1 -- so the transplant can only come from a heap over
       a superset pool on this exact cohort.
    2. `base` is recomputed in this process and must equal the banked `base` EXACTLY (`==`, not a
       tolerance). This is the check that would have caught a policy-A heap instantly: 0.1299 vs
       0.1569.
    3. `--verify_n` candidates (default 8, drawn with a fixed seed) are RE-EVALUATED from scratch
       and their reconstructed values must match EXACTLY. Then, after the search, pick #1's gain is
       re-derived from a fresh evaluation and must match EXACTLY.

  `--no_seed_reuse` runs the cold seed pass instead, for anyone who wants the ~500 evaluations paid
  in full.

TEST IS NOT BUILT, NOT LOADED AND NOT TOUCHED.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import hashlib
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, greedy_certified,
                                    genv_eval, genv_mean, genv_answered, contract_selftest,
                                    celf_heap_sig, CERT_TOWER, CERT_CONC)
from ndcg_tree import build_pool, leak_assert, read_global, RULER_NOTE

# Per-dataset results dir, single source of truth (run_battery_phaseA keys it off
# CASPER_DATASET). Was hardcoded here, so a Goodreads run would have written on top of the
# certified ML-25M tree/greedy artifacts -- HARD RULE 11e.
from run_battery_phaseA import OUTDIR                  # noqa: E402
# The tree root's search heap IS the 1,000-candidate seed pass these channel builds transplant.
# It was hardcoded to the default tree label; a tree grown with --label ndcgtree16 (the K=16 tree
# the paper reports) writes ndcgtree16_heaps instead, so take whichever exists.
_TH = [os.path.join(OUTDIR, d) for d in ("ndcgtree16_heaps", "ndcgtree_heaps")]
TREE_HEAPS = next((d for d in _TH if os.path.exists(os.path.join(d, "node_root.json"))), _TH[-1])
ROOT_HEAP = os.path.join(TREE_HEAPS, "node_root.json")      # VALIDATION only -- see transplant_seed
GLOBAL_JSON = os.path.join(OUTDIR, "greedy_static_i26c.json")

TOTAL_L = 16
BUDGETS = (1, 2, 4, 8, 16)
# `combined` builds the WHOLE 1,000-candidate pool, of which items-only and concepts-only are the
# two blocks. It is the order both headline comparisons are measured against, and on ml-25m it was
# built before this script existed, so the channel was never added here. chan=None means "no
# channel filter"; it also has nothing to transplant a seed from, since it IS the source build.
CHANNELS = {"items": ("items-only", 0), "concepts": ("concepts-only", 1),
            "combined": ("combined", None)}

SUPERSEDES = (
    "This order SUPERSEDES the '{arm}' arm of greedy_static_i26c.json. That arm was SELECTED under "
    "the retired policy-A candidate mask, so any comparison against it carried a selection-protocol "
    "offset. Same pool ({n} candidates, the same ids in the same order as the combined build's "
    "{arm} block), same objective (cohort-mean held-out full NDCG@10), same 10,000 canonical "
    "validation users, same certified stack -- under POLICY B.")


def _sha(o):
    return hashlib.sha1(json.dumps(o, separators=(",", ":")).encode()).hexdigest()


def transplant_seed(E, rows, sub_pool, combined_pool, objective, heap_path, verify_n, tag):
    """Write a CELF sidecar for `sub_pool` from the tree root's completed 1,000-candidate seed pass.

    Returns (base, n_seed_values) or raises. The sidecar is signed by `celf_heap_sig` computed here,
    so `greedy_certified` resumes it through its own unmodified guard.
    """
    assert objective == "full", "the transplanted seed pass is a FULL-objective pass; --objective " \
                                "tail must pay its own seed (use --no_seed_reuse)"
    assert os.path.exists(ROOT_HEAP), f"root heap missing: {ROOT_HEAP}"
    b = json.load(open(ROOT_HEAP))
    s = b["sig"]
    # ---- check 1: the banked signature's own fields -------------------------------------------
    want = {"split": "validation", "objective": "full", "prefix": [],
            "n_pool": len(combined_pool), "pool_sha1": _sha([[int(a), int(c)] for a, c in combined_pool]),
            "n_rows": len(rows), "rows_sha1": _sha(sorted(int(r) for r in rows))}
    bad = {k: (s.get(k), v) for k, v in want.items() if s.get(k) != v}
    assert not bad, (f"the root heap is NOT a seed pass over this cohort/pool: {bad}. Refusing to "
                     f"transplant -- run with --no_seed_reuse.")
    assert b["nev"] == len(combined_pool) and len(b["seq"]) == 1, \
        f"root heap is not one-pick-after-seed (nev={b['nev']}, {len(b['seq'])} picks)"
    upd = {u for _a, _q, u in b["heap"]}
    assert upd == {0}, f"root heap holds re-evaluated entries (update levels {sorted(upd)}); only " \
                       f"update-level-0 entries are seed values"
    # ---- check 2: the base value, recomputed here, EXACTLY ------------------------------------
    base_now = genv_mean(E, [], rows)[0]
    assert base_now == float(b["base"]), (
        f"BASE MISMATCH: this environment gives m=0 = {base_now!r}, the banked heap says "
        f"{b['base']!r}. These are different rulers (policy A m=0 is 0.12987336, policy B is "
        f"0.15686538) -- REFUSING the transplant.")
    log(f"[seed/{tag}] base check PASS: m=0 = {base_now!r} == banked base, exactly")
    # ---- assemble the sub-pool's seed values ---------------------------------------------------
    seed = {(int(q[0]), int(q[1])): -float(a) for a, q, _u in b["heap"]}
    picked = (int(b["seq"][0][0]), int(b["seq"][0][1]))
    assert picked not in seed
    seed[picked] = float(b["gains"][0])
    assert len(seed) == len(combined_pool), f"{len(seed)} seed values for {len(combined_pool)} candidates"
    missing = [q for q in sub_pool if tuple(q) not in seed]
    assert not missing, f"{len(missing)} sub-pool candidates have no banked seed value: {missing[:5]}"
    # ---- check 3: re-evaluate a sample from scratch, EXACT equality -----------------------------
    rng = np.random.default_rng(98765)
    for k in rng.choice(len(sub_pool), size=min(verify_n, len(sub_pool)), replace=False):
        q = tuple(sub_pool[int(k)])
        f, _t = genv_eval(E, [q], rows)
        v = float(np.nanmean(f))
        assert v == base_now + seed[q], (
            f"SEED TRANSPLANT VERIFICATION FAILED at {q}: fresh {v!r} != base + banked gain "
            f"{base_now + seed[q]!r}")
    log(f"[seed/{tag}] {min(verify_n, len(sub_pool))} candidates re-evaluated from scratch; every "
        f"reconstructed value EXACT")
    sig = celf_heap_sig(E, rows, sub_pool, objective, [])
    blob = {"sig": sig, "tag": tag, "base": float(base_now), "cur": float(base_now),
            "nev": len(sub_pool), "seq": [], "gains": [],
            "heap": [[-seed[tuple(q)], [int(q[0]), int(q[1])], 0] for q in sub_pool],
            "PROVENANCE": (
                "Seed values TRANSPLANTED from experiments/battery/ndcgtree_heaps/node_root.json -- a "
                "completed 1,000-candidate CELF seed pass over the identical cohort, empty prefix, "
                "policy B, full objective. A seed gain does not depend on the rest of the pool, so "
                "these are the exact values a cold seed pass would produce; base equality and a "
                "re-evaluated sample were asserted exactly before this file was written.")}
    os.makedirs(os.path.dirname(heap_path), exist_ok=True)
    with open(heap_path + ".tmp", "w") as fh:
        json.dump(blob, fh)
    os.replace(heap_path + ".tmp", heap_path)
    log(f"[seed/{tag}] wrote a verified {len(sub_pool)}-entry seed sidecar -> "
        f"{os.path.relpath(heap_path, _ROOT)} ({len(sub_pool)} evaluations NOT re-paid)")
    return float(base_now)


def curve(E, seq, rows, budgets):
    out = {}
    for K in budgets:
        f, t = genv_eval(E, seq[:K], rows)
        ia, ca = genv_answered(E, seq[:K], rows)
        out[str(K)] = {"full@10": float(np.nanmean(f)), "tail@10": float(np.nanmean(t)),
                       "items_answered": round(ia, 3), "concepts_answered": round(ca, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=sorted(CHANNELS))
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--total_L", type=int, default=TOTAL_L)
    ap.add_argument("--label", default=None)
    ap.add_argument("--verify_n", type=int, default=8)
    ap.add_argument("--split", default="validation", choices=["validation", "train"],
                    help="fitting population. 'train' keeps this comparator on the same footing as "
                         "the train-fitted trees it is compared against; validation then holds no "
                         "arm's fitting data. NDCG is never REPORTED on train.")
    ap.add_argument("--slice_lo", type=int, default=0)
    ap.add_argument("--slice_hi", type=int, default=34000)
    ap.add_argument("--n_users", type=int, default=0, help="cap the fitting cohort (0 = all)")
    ap.add_argument("--no_seed_reuse", action="store_true")
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv)")
    args = ap.parse_args()

    arm, chan = CHANNELS[args.channel]
    label = args.label or f"greedy_static_policyB_{args.channel}"
    t00 = time.time()
    OUT = os.path.join(OUTDIR, f"{label}.json")
    PARTIAL = os.path.join(OUTDIR, f"{label}_partial.json")
    HEAP = os.path.join(OUTDIR, f"{label}_heaps", "search.json")
    total_L = args.total_L

    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("global greedy json", GLOBAL_JSON)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"

    log("#" * 110)
    log(f"# A3/A4 -- REBUILDING THE '{arm}' STATIC ORDER UNDER POLICY B, {total_L} picks, "
        f"10,000 canonical validation users, objective={args.objective}.")
    log("# TEST IS NOT BUILT, NOT LOADED, NOT TOUCHED.")
    log("#" * 110)

    assert args.split != "test", "the TEST look is reserved -- this file never builds it"
    if args.split == "validation":
        E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"{label}(val)")
    else:
        E = build_greedy_env(args.snapshot, "train", args.concept_ckpt, f"{label}(train)",
                             cohort_slice=(args.slice_lo, args.slice_hi))
        log(f"[cohort] FITTING ON TRAIN USERS {args.slice_lo}..{args.slice_hi} -- a fitting "
            f"objective only; this order is REPORTED on validation.")
    rows = E["rows"]
    if args.n_users and len(rows) > args.n_users:
        rows = list(rows)[:args.n_users]
        log(f"[cohort] capped to the first {len(rows):,} users")
    contract_selftest(E, rows)
    leak_assert(E, rows, args.split)
    want_n = 10000 if not args.n_users else args.n_users
    assert len(rows) == want_n, f"cohort is {len(rows)} users, expected {want_n}"

    combined_pool, item_pool, conc_pool = build_pool(E, args.n_items, args.n_concepts)
    if chan is None:
        sub_pool = combined_pool
        assert len(sub_pool) == args.n_items + args.n_concepts
    else:
        sub_pool = item_pool if chan == 0 else conc_pool
        assert len(sub_pool) == (args.n_items if chan == 0 else args.n_concepts)
        assert all(int(q[0]) == chan for q in sub_pool)
    log(f"[pool] {len(sub_pool)} {args.channel}-only candidates (the {arm} block of the combined "
        f"{len(combined_pool)}-candidate pool, same ids, same order). NO truncation: this IS the "
        f"full declared bank.")
    log(f"[pool] contract (DERIVED): {json.dumps(E['pool_contract'])}")

    comp_old, _gblob = read_global(GLOBAL_JSON, total_L)
    old_seq = comp_old[arm]["sequence"][:total_L]

    set_active_rows(E, rows); E["memo"] = {}; E["n_evals"] = 0; E["t_evals"] = 0.0
    if args.no_seed_reuse:
        if os.path.exists(HEAP):
            os.remove(HEAP)
        log(f"[seed/{args.channel}] --no_seed_reuse: paying the full {len(sub_pool)}-candidate cold "
            f"seed pass")
    elif os.path.exists(HEAP):
        log(f"[seed/{args.channel}] a sidecar already exists at {os.path.relpath(HEAP, _ROOT)} -- "
            f"greedy_certified's own guard decides whether it may be resumed")
    else:
        if chan is None:
            log(f"[seed/combined] this IS the source build -- paying the full "
                f"{len(sub_pool)}-candidate cold seed pass, nothing to transplant from")
        else:
            transplant_seed(E, rows, sub_pool, combined_pool, args.objective, HEAP,
                            args.verify_n, args.channel)

    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    t0 = time.time()

    def on_pick(seq, gains, nev):
        el = (time.time() - t0) / 60
        part.update({"label": label, "channel": args.channel, "arm": arm,
                     "sequence": [[int(a), int(b)] for a, b in seq],
                     "gains": [float(g) for g in gains], "n_evals": int(nev),
                     "minutes": round(el, 2),
                     "minutes_per_pick": round(el / max(len(seq), 1), 2),
                     "done": False})
        with open(PARTIAL + ".tmp", "w") as fh:
            json.dump(part, fh, indent=2, default=float)
        os.replace(PARTIAL + ".tmp", PARTIAL)
        log(f"[cost] {len(seq)} picks in {el:.1f}m ({el / max(len(seq), 1):.1f} m/pick, "
            f"{E['t_evals'] / max(E['n_evals'], 1):.2f}s/eval measured this process)")

    seq, gains, nev = greedy_certified(E, rows, sub_pool, total_L, args.objective, args.channel,
                                       on_pick=on_pick, heap_path=HEAP)
    assert len(seq) == total_L, f"got {len(seq)} picks, wanted {total_L}"
    minutes = round((time.time() - t0) / 60, 2)

    # ---- post-search gate: pick #1's gain re-derived from a fresh evaluation, EXACTLY -----------
    base_now = genv_mean(E, [], rows)[0]
    f1, _t1 = genv_eval(E, [tuple(seq[0])], rows)
    g1 = float(np.nanmean(f1)) - base_now
    assert g1 == gains[0], (f"PICK-1 GATE FAILED: fresh gain {g1!r} != recorded {gains[0]!r} -- the "
                            f"resumed seed values are not what this environment produces")
    log(f"[gate] pick #1 gain re-derived from a fresh evaluation = {g1!r} == recorded, exactly")

    c_new = curve(E, seq, rows, BUDGETS)
    c_old = curve(E, old_seq, rows, BUDGETS)
    same_pos = sum(1 for a, b in zip(seq, old_seq) if tuple(a) == tuple(b))
    same_set = len({tuple(q) for q in seq} & {tuple(q) for q in old_seq})

    blob = {"analysis": f"static_global_greedy_{args.channel}_only_POLICY_B",
            "label": label, "citable": True, "arm": arm, "channel": args.channel,
            "supersedes": SUPERSEDES.format(arm=arm, n=len(sub_pool)),
            "ruler_note": RULER_NOTE,
            "split": args.split, "n_cohort": len(rows), "test_touched": False,
            "tower": os.path.basename(args.snapshot),
            "concept_ckpt": os.path.basename(args.concept_ckpt),
            "objective": args.objective, "item_bank": "pop", "conc_bank": "answerable",
            "n_items_pool": len(item_pool), "n_concepts_pool": len(conc_pool),
            "n_pool": len(sub_pool),
            "mask_policy": "B", "pool_policy": "B", "pool_contract": E["pool_contract"],
            "n_cohort": len(rows), "L": total_L,
            "sequence": [[int(a), int(b)] for a, b in seq], "gains": [float(g) for g in gains],
            "intercept_policyB_full": base_now,
            "n_evals": int(nev),
            "seed_pass": ("transplanted+verified from ndcgtree_heaps/node_root.json"
                          if not args.no_seed_reuse else "paid cold"),
            "curve_policyB": c_new,
            "policyA_order_for_reference": [[int(a), int(b)] for a, b in old_seq],
            "policyA_order_rescored_under_policyB": c_old,
            "positions_identical_to_policyA_order": same_pos,
            "questions_shared_with_policyA_order": same_set,
            "minutes": minutes,
            "minutes_per_pick": round(minutes / total_L, 2),
            "sec_per_eval": round(E["t_evals"] / max(E["n_evals"], 1), 3),
            "seconds": round(time.time() - t00, 1)}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    part.update(dict(blob, done=True))
    with open(PARTIAL + ".tmp", "w") as fh:
        json.dump(part, fh, indent=2, default=float)
    os.replace(PARTIAL + ".tmp", PARTIAL)

    log("=" * 110)
    log(f"[{args.channel}] policy-B order = {[[int(a), int(b)] for a, b in seq]}")
    log(f"[{args.channel}] vs the policy-A order: {same_pos}/{total_L} positions identical, "
        f"{same_set}/{total_L} questions shared")
    for K in BUDGETS:
        n, o = c_new[str(K)], c_old[str(K)]
        log(f"[{args.channel}] K={K:<3} policy-B order {n['full@10']:.5f}/{n['tail@10']:.5f}   "
            f"policy-A order rescored {o['full@10']:.5f}/{o['tail@10']:.5f}   "
            f"delta {n['full@10'] - o['full@10']:+.5f}/{n['tail@10'] - o['tail@10']:+.5f}")
    log(f"[done] wrote {os.path.basename(OUT)} in {(time.time() - t00) / 60:.1f}m "
        f"({minutes:.1f}m search, {nev} cumulative evals, {blob['sec_per_eval']}s/eval)")
    return blob


if __name__ == "__main__":
    main()
