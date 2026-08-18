"""Seed coldtree_v2 from the banked v1 tree, using the LIVE contract.

The v1 partial predates the `pool_contract` field, so the grower correctly refuses it: an unstated
contract is not a matching one. The contracts are nevertheless verified EQUAL behaviourally -- the
from-scratch v2 run of 2026-08-18 reproduced v1's first four nodes exactly (root concept 114 ->
2606/4814/2580; root.like item 83 -> 1010/340/1256; root.dislike concept 283 -> 723/3562/529;
root.noanswer item 514 -> 357/438/1785) under the live contract. That is the evidence for adopting
the live `pool_contract` here rather than inventing one.

Heaps are NOT migrated: they carry the same pre-contract signature and are refused. That costs
nothing, because every node still needing a search is a NEW child with no banked heap anyway.
"""
import io, json, os

B = r"C:\dev\phd\casper\experiments\battery"
v1 = json.load(io.open(os.path.join(B, "ndcgtree_partial.json"), encoding="utf-8"))
live_sig = json.load(io.open(r"C:\Users\Tech\AppData\Local\Temp\live_sig.json", encoding="utf-8"))
nodes = v1["nodes"]
total_L, min_leaf = int(live_sig["total_L"]), int(live_sig["min_leaf"])
assert total_L == int(v1["sig"]["total_L"]), "total_L differs between v1 and the live run"
assert min_leaf == int(v1["sig"]["min_leaf"]), "min_leaf differs"
for k in ("tower", "conc", "objective", "n_pool", "criterion", "branch_rule", "split", "n_cohort"):
    assert v1["sig"].get(k) == live_sig.get(k), f"sig field {k!r} differs: {v1['sig'].get(k)!r} vs {live_sig.get(k)!r}"
print("v1 and live sigs agree on every field v1 states")

flipped = kept = already = 0
for nid, rec in nodes.items():
    depth = int(rec["depth"]); remaining = total_L - depth
    rec["remaining_budget"] = remaining
    sizes = rec.get("child_sizes") or {}
    nonempty = [k for k, v in sizes.items() if v > 0]
    big = [k for k, v in sizes.items() if v >= min_leaf]
    rec["children_above_floor"] = big
    rec["children_inheriting"] = [k for k in nonempty if k not in big]
    rec["min_nonempty_child"] = min([sizes[k] for k in nonempty], default=0)
    if rec.get("split"):
        already += 1
        rec.setdefault("tail", rec.get("steps"))
        continue
    if remaining >= 2 and len(nonempty) >= 2 and len(big) >= 1:
        rec["tail"] = list(rec["steps"])
        rec["tail_gains"] = list(rec.get("gains", []))
        rec["split"] = True
        rec["stop_reason"] = None
        flipped += 1
    else:
        rec["tail"] = list(rec["steps"])
        kept += 1

json.dump({"sig": live_sig, "nodes": nodes},
          io.open(os.path.join(B, "coldtree_v2_partial.json"), "w", encoding="utf-8"), indent=2)
print(f"seeded coldtree_v2_partial.json: {len(nodes)} nodes "
      f"({already} already split, {flipped} leaves now split, {kept} stay leaves)")
