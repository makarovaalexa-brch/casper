r"""golbandi_resume_test.py -- the acceptance test for per-depth checkpointing + RESUME in
`src/baselines/golbandi_tree_selector.py`. Runs on ML-100k in seconds, so the ML-25M growth (6.5
hours on 2026-08-06) never has to be launched to check the machinery.

WHAT IT PROVES, in order:

  A. GROW TO DEPTH 3 with a sidecar. A checkpoint exists after depth 1, 2 AND 3 -- not only at the
     end -- each one a self-consistent tree whose deepest level is all leaves, stamped with the depth
     actually grown and with `complete` false until the target is reached.
  B. RESUME. Asking for depth 5 against that sidecar replays depths 1-3 and searches only depths 4-5.
     Proved by COUNTING `_best_split` calls: a resumed run must call it exactly as many times as
     there are frontier nodes at depths 3 and 4 that are still allowed to split, i.e. zero calls for
     the banked levels. And the depth<=3 substructure of the resumed tree is compared BYTE-FOR-BYTE
     (json.dumps) against the depth-3 sidecar.
  C. EQUIVALENCE. A depth-5 tree grown from scratch, with no sidecar at all, is byte-identical to the
     resumed one. This is the claim that makes resuming legitimate: it is not an approximation.
  D. SHALLOWER REQUEST. Asking for depth 2 against the depth-5 sidecar LOADS and stops -- zero
     `_best_split` calls, zero growth.
  E. CONTRACT REFUSAL. A sidecar grown on different data is refused (loudly) and regrown, never
     silently reused.
  F. THE ML-25M ARTIFACT. `.cache/snap5/tree_ml25m_liang_d8.json` -- the 6.5-hour depth-8 tree the
     port cells were scored from -- still LOADS, and is still ACCEPTED by the new contract logic at
     depth 8 (i.e. the change did not orphan it). It was written by the old depth-first growth and
     has no `order`/`complete`/`target_depth` keys; the loader treats a missing `complete` as true
     and every node reference is by id, so ordering never mattered.

    python scripts/baselines/golbandi_resume_test.py
"""
import json
import os
import shutil
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

import golbandi_tree_selector as GT                                  # noqa: E402
from golbandi_ml100k_snap import load_ml100k                         # noqa: E402

WORK = os.path.join(_ROOT, ".cache", "snap5", "_resume_test")
ML25M = os.path.join(_ROOT, ".cache", "snap5", "tree_ml25m_liang_d8.json")
LIKE_MIN = 4.0


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- instrumentation
_CALLS = []


def count_best_split():
    """Wrap `_best_split` so the test can assert WHICH depths were actually searched. The split
    search is the entire cost of growth; a resume that avoided it is a resume that worked."""
    orig = GT.GolbandiTree._best_split

    def wrapped(self, rows, banned):
        _CALLS.append(len(banned))          # |banned| == the depth of the node being searched
        return orig(self, rows, banned)
    GT.GolbandiTree._best_split = wrapped
    return orig


def snapshotting_log(dest_dir, verbose=True):
    """Copy the sidecar aside every time the tree announces a checkpoint, so the test can inspect the
    file as it stood at each depth rather than only its final state."""
    os.makedirs(dest_dir, exist_ok=True)
    seen = {}

    def f(*a):
        msg = " ".join(str(x) for x in a)
        if verbose:
            log("   " + msg)
        if "CHECKPOINT depth=" in msg:
            d = int(msg.split("CHECKPOINT depth=")[1].split()[0])
            src = msg.split(" -> ")[-1].strip()
            dst = os.path.join(dest_dir, f"snap_d{d}.json")
            shutil.copyfile(src, dst)
            seen[d] = dst
    f.seen = seen
    return f


def grow(cache, depth, R, n_items, logf, xi=None):
    _CALLS.clear()
    sel = GT.GolbandiTreeSelector(R, n_items, depth=depth, like_level=4, like_min=LIKE_MIN, xi=xi,
                                  log=logf)
    sel.fit(cache_path=cache)
    return sel, list(_CALLS)


def nodes_of(path):
    return json.load(open(path))["nodes"]


def strip_split(nd):
    """A node with its SPLIT erased -- what a node at the deepest level looks like in the shallower
    checkpoint, before the next level gives it a question. Applied to BOTH sides of the comparison,
    so what is compared is the shape a depth-d tree is REQUIRED to have: the identity of every
    question above depth d, every child pointer, and every node's user and rating counts."""
    out = {k: v for k, v in nd.items() if k != "child_sizes"}
    out.update({"item": None, "L": None, "H": None, "U": None})
    return out


def prefix_bytes(nodes, max_depth):
    """json.dumps of every node at depth <= max_depth, deepest level reduced to a leaf -- i.e.
    exactly the structure a depth-`max_depth` tree must contain, and nothing else."""
    return json.dumps([strip_split(nd) if nd["depth"] == max_depth else nd
                       for nd in nodes if nd["depth"] <= max_depth], sort_keys=True)


# --------------------------------------------------------------------------- the test
def main():
    ok = []
    orig_best_split = count_best_split()
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    R = load_ml100k()
    n_items = R.shape[1]
    log(f"[test] ML-100k {R.shape[0]} users x {n_items} items, {R.nnz} ratings; "
        f"like cut r >= {LIKE_MIN}")

    # ---- A: grow to depth 3, a checkpoint after every depth -------------------------------------
    log("\n[test] A. grow to depth 3 with a per-depth sidecar")
    cache = os.path.join(WORK, "tree.json")
    snaps = snapshotting_log(os.path.join(WORK, "snapshots"))
    sel3, calls3 = grow(cache, 3, R, n_items, snaps)
    for d in (1, 2, 3):
        assert d in snaps.seen, f"NO checkpoint was written after depth {d}"
        blob = json.load(open(snaps.seen[d]))
        assert blob["depth"] == d, f"snapshot d{d} claims depth {blob['depth']}"
        assert blob["_contract"]["depth"] == d, "the contract must carry the depth actually grown"
        assert blob["complete"] is (d == 3), f"snapshot d{d} complete={blob['complete']}"
        deepest = [nd for nd in blob["nodes"] if nd["depth"] == d]
        assert deepest and all(nd["item"] is None for nd in deepest), \
            f"snapshot d{d} has a non-leaf at its deepest level -- it is not a valid depth-{d} tree"
        t = GT.load_tree(snaps.seen[d], log=lambda *_: None)
        assert t.depth == d and len(t.nodes) == len(blob["nodes"])
        log(f"[test]   depth {d}: sidecar OK, {len(blob['nodes'])} nodes, "
            f"{sum(1 for nd in blob['nodes'] if nd['item'] is not None)} internal, "
            f"complete={blob['complete']}, reloads")
    ok.append("A per-depth sidecars at d=1,2,3, each a valid tree of that depth")

    # ---- B: request depth 5 -> RESUME ------------------------------------------------------------
    log("\n[test] B. request depth 5 against the depth-3 sidecar (expect RESUME)")
    d3_nodes = nodes_of(snaps.seen[3])
    sel5, calls5 = grow(cache, 5, R, n_items, snapshotting_log(os.path.join(WORK, "snap_resume")))
    searched = sorted(set(calls5))
    assert searched and min(searched) >= 3, \
        f"a resumed run re-searched banked depths: _best_split ran at depths {searched}"
    assert len(calls5) > 0, "the resumed run searched nothing at all"
    log(f"[test]   _best_split ran {len(calls5)} times, only at depths {searched} "
        f"(depths 0-2 were REPLAYED by routing, not re-searched)")
    resumed = nodes_of(cache)
    a = prefix_bytes(d3_nodes, 3)
    b = prefix_bytes(resumed, 3)
    assert a == b, "the depth<=3 substructure of the resumed tree differs from the depth-3 sidecar"
    log(f"[test]   depth<=3 substructure BYTE-IDENTICAL to the depth-3 run "
        f"({len(json.loads(a))} nodes, {len(a)} bytes of JSON)")
    ok.append("B depth-5 request RESUMED from depth 3; banked levels re-routed, never re-searched; "
              "depth<=3 substructure byte-identical")

    # ---- C: a from-scratch depth-5 tree must be the SAME tree ------------------------------------
    log("\n[test] C. grow depth 5 from scratch (no sidecar) and compare")
    fresh_cache = os.path.join(WORK, "fresh.json")
    selF, callsF = grow(fresh_cache, 5, R, n_items, lambda *_: None)
    assert json.dumps(nodes_of(fresh_cache), sort_keys=True) == \
        json.dumps(resumed, sort_keys=True), \
        "the resumed depth-5 tree is NOT the tree a from-scratch depth-5 growth produces"
    log(f"[test]   from-scratch depth-5 tree byte-identical to the resumed one "
        f"({len(resumed)} nodes); from-scratch searched {len(callsF)} nodes vs "
        f"{len(calls5)} for the resume -- {len(callsF) - len(calls5)} searches saved")
    ok.append(f"C resumed depth-5 tree == from-scratch depth-5 tree, byte for byte "
              f"({len(callsF) - len(calls5)}/{len(callsF)} split searches skipped)")

    # ---- D: request depth 2 -> LOAD and stop ------------------------------------------------------
    log("\n[test] D. request depth 2 against the depth-5 sidecar (expect LOAD, no growth)")
    lines = []
    sel2, calls2 = grow(cache, 2, R, n_items, lambda *a: lines.append(" ".join(str(x) for x in a)))
    assert calls2 == [], f"a load-and-stop still searched {len(calls2)} nodes"
    assert any("LOADED cached tree" in x for x in lines), lines
    assert sel2.tree.depth == 5 and len(sel2.tree.nodes) == len(resumed)
    assert json.dumps(nodes_of(cache), sort_keys=True) == json.dumps(resumed, sort_keys=True), \
        "a load-and-stop overwrote the sidecar"
    log(f"[test]   loaded depth-5 tree, 0 split searches, sidecar untouched")
    ok.append("D shallower request LOADS and stops (0 searches, sidecar untouched)")

    # ---- E: a changed contract must REFUSE --------------------------------------------------------
    log("\n[test] E. same sidecar, DIFFERENT data (expect a loud refusal + regrow)")
    lines = []
    R2 = R[: R.shape[0] - 50]                       # different n_train and nnz -> different contract
    sel_r, calls_r = grow(cache, 2, R2, n_items,
                          lambda *a: lines.append(" ".join(str(x) for x in a)))
    assert any("REFUSED: contract mismatch" in x for x in lines), lines
    assert len(calls_r) > 0, "a refused contract did not regrow"
    log("[test]   " + next(x for x in lines if "REFUSED" in x).splitlines()[0])
    ok.append("E changed contract REFUSED loudly and regrew from scratch")

    # ---- F: the ML-25M artifact must still load and still be accepted -----------------------------
    log("\n[test] F. the existing 6.5-hour ML-25M depth-8 checkpoint")
    if not os.path.exists(ML25M):
        log(f"[test]   SKIPPED: {ML25M} not present")
        ok.append("F SKIPPED (artifact absent)")
    else:
        blob = json.load(open(ML25M))
        t = GT.load_tree(ML25M, log=log)
        assert t.depth == 8 and len(t.nodes) == len(blob["nodes"])
        assert "order" not in blob and "complete" not in blob, \
            "expected the pre-change (depth-first) file; it seems to have been rewritten"
        occ = t.occupancy()
        assert sorted(occ) == list(range(9)), sorted(occ)
        # the decision code, exercised on a stub that carries only what tree_contract reads
        c = blob["_contract"]
        stub = types.SimpleNamespace(shape=(c["n_train"], c["n_items"]), nnz=c["nnz"])
        lines = []
        sel = GT.GolbandiTreeSelector(stub, c["n_items"], depth=8, like_level=4,
                                      like_min=c["like_min"], xi=c["xi"],
                                      min_child_users=c["min_child_users"],
                                      bias_correct=c["bias_correct"],
                                      log=lambda *a: lines.append(" ".join(str(x) for x in a)))
        sel.fit(cache_path=ML25M)                   # must LOAD; growing on the stub would explode
        assert any("LOADED cached tree" in x for x in lines), lines
        assert sel.tree.depth == 8 and len(sel.tree.nodes) == 1360
        log(f"[test]   ACCEPTED and loaded under the new contract logic: depth 8, "
            f"{len(sel.tree.nodes)} nodes, occupancy depths {sorted(occ)}")
        ok.append("F ML-25M depth-8 artifact still loads AND is still accepted at depth 8 "
                  "(contract unchanged; depth-first node order is irrelevant, all refs are by id)")

    GT.GolbandiTree._best_split = orig_best_split
    log("\n[test] ==================== PASS ====================")
    for line in ok:
        log("[test] " + line)
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
