"""Seed coldtree_v2 from the banked v1 tree instead of rebuilding it.

v2's rule is STRICTLY more permissive than v1's: any node v1 split, v2 splits too, with the SAME
splitter (the splitter search does not depend on the stopping rule). So the depth-5 v1 tree is a
valid PREFIX of the v2 tree, and every one of its 15 leaves can be re-decided from its already
recorded `child_sizes` WITHOUT a single new evaluation.
"""
import io, json, os, shutil

B = r"C:\dev\phd\casper\experiments\battery"
SRC = os.path.join(B, "ndcgtree_partial.json")
DST = os.path.join(B, "coldtree_v2_partial.json")
SRC_HEAPS = os.path.join(B, "ndcgtree_heaps")
DST_HEAPS = os.path.join(B, "coldtree_v2_heaps")

d = json.load(io.open(SRC, encoding="utf-8"))
sig, nodes = d["sig"], d["nodes"]
total_L, min_leaf = int(sig["total_L"]), int(sig["min_leaf"])
print(f"v1: {len(nodes)} nodes, total_L={total_L}, min_leaf={min_leaf}")

sig = dict(sig)
sig["stopping"] = ("min_leaf v2: split iff >=2 non-empty children and >=1 child >= min_leaf; "
                   "children below the floor inherit the parent tail (no depth cap)")
sig["cohort_concept"] = -1
sig["seed_prefix"] = []

flipped, kept, already = [], [], 0
for nid, rec in nodes.items():
    depth = int(rec["depth"])
    remaining = total_L - depth
    rec["remaining_budget"] = remaining
    sizes = rec.get("child_sizes") or {}
    nonempty = [k for k, v in sizes.items() if v > 0]
    big = [k for k, v in sizes.items() if v >= min_leaf]
    rec["children_above_floor"] = big
    rec["children_inheriting"] = [k for k in nonempty if k not in big]
    rec["min_nonempty_child"] = min([sizes[k] for k in nonempty], default=0)

    if rec.get("split"):
        already += 1
        rec.setdefault("tail", rec.get("steps"))          # v1 split nodes stored only the splitter
        continue

    will_split = remaining >= 2 and len(nonempty) >= 2 and len(big) >= 1
    if will_split:
        # v1's leaf `steps` IS the full greedy tail from this node -- exactly what an inheriting
        # child needs, and it was fitted on this node's own (above-floor) cohort.
        rec["tail"] = list(rec["steps"])
        rec["tail_gains"] = list(rec.get("gains", []))
        rec["split"] = True
        rec["stop_reason"] = None
        flipped.append((nid, depth, sizes))
    else:
        rec["tail"] = list(rec["steps"])
        rec["stop_reason"] = (f"degenerate partition: {len(nonempty)} non-empty child(ren)"
                              if len(nonempty) < 2 else
                              f"no child reaches min_leaf {min_leaf} (largest {max(sizes.values())})")
        kept.append((nid, depth, sizes))

json.dump({"sig": sig, "nodes": nodes}, io.open(DST, "w", encoding="utf-8"), indent=2)

if os.path.isdir(SRC_HEAPS):
    if os.path.isdir(DST_HEAPS):
        shutil.rmtree(DST_HEAPS)
    shutil.copytree(SRC_HEAPS, DST_HEAPS)
    print(f"copied {len(os.listdir(DST_HEAPS))} CELF heaps -> coldtree_v2_heaps")

print(f"\nalready split under v1 : {already}")
print(f"leaves NOW SPLITTING   : {len(flipped)}")
for nid, dep, sz in flipped:
    print(f"   d{dep} {nid:<48} {sz}")
print(f"leaves STAYING leaves  : {len(kept)}")
for nid, dep, sz in kept:
    print(f"   d{dep} {nid:<48} {sz}")
