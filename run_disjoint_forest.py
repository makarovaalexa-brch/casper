#!/usr/bin/env python
"""Refit the six teachers on DISJOINT groups with exclusive opener material, then the forest."""
import json, os, subprocess, sys, time
R = r"C:\dev\phd\casper"; PY = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(R, "experiments", "battery"); M6 = os.path.join(LOG, "m6")
os.chdir(R); env = dict(os.environ)
DJ = os.path.join(M6, "m7b_disjoint.npz")
G = ["arthouse", "dialogue", "marvel", "gore", "teenagers", "inspiring"]

def run(tag, args, log, need=None):
    print(f"[dj] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    txt = open(os.path.join(LOG, log), encoding="utf-8", errors="replace").read()
    ok = rc == 0 and (need is None or need in txt)
    print(f"[dj] {tag} rc={rc} ok={ok} {time.strftime('%H:%M:%S')}", flush=True)
    return ok

for g in G:
    if os.path.exists(os.path.join(M6, f"m6k_dj{g}_train2000.json")):
        print(f"[dj] {g} already fitted -- skipping", flush=True); continue
    if not run(f"fit-{g}", ["src/instrument/m6k_cluster_tree.py", "--i_have_author_goahead",
                            "--threads", "6", "--cluster", g, "--disjoint", DJ,
                            "--split", "train", "--slice_hi", "34000", "--n_users", "2000",
                            "--depth", "6", "--min_leaf", "200"],
               f"m6k_dj{g}.out", need="[done]"):
        print(f"[dj] {g} FAILED -- continuing", flush=True)

bad = {}
for g in G:
    p = os.path.join(M6, f"m6k_dj{g}_train2000.json")
    if not os.path.exists(p): bad[g] = "missing"; continue
    d = json.load(open(p, encoding="utf-8")); n = d["nodes"]
    n = n if isinstance(n, dict) else {r["nid"]: r for r in n}
    k = sum(1 for x in n.values()
            if set(map(tuple, x.get("path", [])))
            & set(map(tuple, (x.get("tail") or x.get("steps") or []))))
    if k: bad[g] = f"{k} duplicate-question nodes"
print(f"[dj] audit: {bad or 'ALL SIX CLEAN'}", flush=True)
if bad: sys.exit(f"[dj] {bad} -- ABORT before measuring")
print(f"[dj] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
