"""Print each forest teacher's first levels as NAMED questions, to eyeball semantic coherence.

Rebuilds the concept vocabulary the same way `run_battery_phaseA._g5_members` does -- genome tags
with >= 30 in-catalogue members, keyed by ascending tagId -- so concept id i is the i-th such tag.
Items map through unique_sid.txt row index -> movieId -> title. No env is built; this is read-only
and safe to run beside a growing teacher.
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
B = os.path.join(_R, "experiments", "battery")
PROC = os.path.join(_R, "data", "ml-25m", "proc")
DATA = os.path.join(_R, "data", "movielens")
MIN_MEMBERS = 30


def build_names():
    usid = [int(x) for x in open(os.path.join(PROC, "unique_sid.txt")).read().split()]
    m2s = {m: i for i, m in enumerate(usid)}
    g = pd.read_csv(os.path.join(DATA, "genome-scores.csv"),
                    usecols=["movieId", "tagId", "relevance"],
                    dtype={"movieId": np.int32, "tagId": np.int16, "relevance": np.float32})
    g = g[(g["relevance"] >= 0.5) & (g["movieId"].isin(m2s))]
    cnt = g.groupby("tagId").size()
    tags = sorted(int(t) for t, n in cnt.items() if n >= MIN_MEMBERS)
    tagname = {}
    with open(os.path.join(DATA, "genome-tags.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tagname[int(r["tagId"])] = r["tag"].strip()
    cname = {i: tagname.get(t, f"tag{t}") for i, t in enumerate(tags)}
    title = {}
    with open(os.path.join(DATA, "movies.csv"), newline="", encoding="utf-8") as fh:
        mt = {int(r["movieId"]): r["title"] for r in csv.DictReader(fh)}
    iname = {i: mt.get(m, str(m)) for i, m in enumerate(usid)}
    print(f"[vocab] {len(tags)} concepts with >= {MIN_MEMBERS} members "
          f"(expected 1031); {len(iname)} items")
    return cname, iname


def qname(q, cname, iname):
    ch, i = int(q[0]), int(q[1])
    return (f"concept '{cname.get(i, '?')}'" if ch == 1 else f"item '{iname.get(i, '?')}'")


def walk(nodes, cname, iname, max_depth, anchor=None):
    out = []

    def rec(nid, prefix):
        rec_ = nodes.get(nid)
        if rec_ is None:
            return
        d = int(rec_.get("depth", 0))
        if d > max_depth:
            return
        sp = rec_.get("splitter")
        n = rec_.get("n_users")
        if sp is None:
            out.append(f"{'  ' * d}{prefix}[leaf, n={n}] inherits parent order")
            return
        sizes = rec_.get("child_sizes", {})
        ar = rec_.get("answer_rate")
        out.append(f"{'  ' * d}{prefix}ask {qname(sp, cname, iname)}  "
                   f"(n={n}, answered {ar}, like/dis/no = "
                   f"{sizes.get('like')}/{sizes.get('dislike')}/{sizes.get('noanswer')})")
        if not rec_.get("split"):
            return
        for br, sym in (("like", "LIKE  -> "), ("dislike", "DISLIKE-> "), ("noanswer", "NOANSW-> ")):
            if sizes.get(br):
                rec(f"{nid}.{br}", sym)

    rec("root", "")
    return out


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cname, iname = build_names()
    anchors = {"arthouse": 68, "dialogue": 275, "marvel": 581,
               "gore": 417, "teenagers": 922, "inspiring": 503}
    for name, cid in anchors.items():
        f = os.path.join(B, f"m3v2_{name}.json")
        if not os.path.exists(f):
            print(f"\n=== {name}: not finished yet ===")
            continue
        d = json.load(open(f, encoding="utf-8"))
        nodes = {r["nid"]: r for r in d.get("nodes", [])} if isinstance(d.get("nodes"), list) \
            else d.get("nodes", {})
        if not nodes:
            part = os.path.join(B, f"m3v2_{name}_partial.json")
            nodes = json.load(open(part, encoding="utf-8")).get("nodes", {}) \
                if os.path.exists(part) else {}
        print(f"\n=== TEACHER '{name}'  (anchor: concept '{cname.get(cid, '?')}')  "
              f"depth {d.get('depth_reached')}, {d.get('n_nodes')} nodes, "
              f"{d.get('n_leaves')} leaves ===")
        print(f"  the user has VOLUNTEERED '{cname.get(cid, '?')}' -- folded free, then:")
        for line in walk(nodes, cname, iname, depth):
            print("  " + line)


if __name__ == "__main__":
    main()
