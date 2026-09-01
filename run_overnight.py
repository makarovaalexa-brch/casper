#!/usr/bin/env python
"""Overnight: refit inspiring, then the whole forest pipeline. All teachers TRAIN-fitted,
every number on VALIDATION. Each stage aborts the chain rather than feeding a broken bank forward.
"""
import json, os, subprocess, sys, time
R = r"C:\dev\phd\casper"; PY = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(R, "experiments", "battery"); M6 = os.path.join(LOG, "m6")
os.chdir(R)
env = dict(os.environ)
COLD = "coldtree_v2_train10000.json"
GROUPS = ["arthouse", "dialogue", "gore", "inspiring", "marvel", "teenagers"]

def run(tag, args, log, need=None):
    print(f"[night] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    txt = open(os.path.join(LOG, log), encoding="utf-8", errors="replace").read()
    ok = rc == 0 and (need is None or need in txt)
    print(f"[night] {tag} rc={rc} ok={ok} {time.strftime('%H:%M:%S')}", flush=True)
    return ok

# ---- 1. the last teacher ----------------------------------------------------------------------
if not os.path.exists(os.path.join(M6, "m6k_inspiring_train2000.json")):
    if not run("fit-inspiring", ["src/instrument/m6k_cluster_tree.py", "--i_have_author_goahead",
                                 "--threads", "6", "--cluster", "inspiring", "--split", "train",
                                 "--slice_hi", "34000", "--n_users", "2000",
                                 "--depth", "6", "--min_leaf", "200"],
               "m6k_inspiring_tr2k_v3.out", need="[done]"):
        sys.exit("[night] inspiring refit failed -- ABORT")

# ---- 2. verify EVERY tree is free of the duplicate-question defect ------------------------------
bad = {}
for g in GROUPS:
    d = json.load(open(os.path.join(M6, f"m6k_{g}_train2000.json"), encoding="utf-8"))
    n = d["nodes"]; n = n if isinstance(n, dict) else {r["nid"]: r for r in n}
    k = sum(1 for x in n.values()
            if set(map(tuple, x.get("path", [])))
            & set(map(tuple, (x.get("tail") or x.get("steps") or []))))
    if k: bad[g] = k
print(f"[night] duplicate-node audit: {bad or 'ALL SIX CLEAN'}", flush=True)
if bad:
    sys.exit(f"[night] {bad} still carry duplicate questions -- ABORT before measuring")

# ---- 3. the measurement, then the forest --------------------------------------------------------
run("order-check-6", ["src/instrument/m6t_order_check.py", "--i_have_author_goahead",
                      "--threads", "6"], "m6t_order6.out", need="[done]")

if not run("gen", ["src/instrument/m6c_distill_gen.py", "--i_have_author_goahead", "--threads", "6",
                   "--half", "A", "--tree_suffix", "_train2000", "--out_dir", "traj_forest",
                   "--cold_tree", COLD], "m6c_gen_forest.out", need="[done] all slices banked"):
    sys.exit("[night] generation failed -- ABORT")

if not run("distil", ["src/instrument/m4b_policy.py", "--i_have_author_goahead", "--threads", "6",
                      "--epochs", "6", "--anchor_encoding", "none", "--no_cold_all",
                      "--traj_dir", os.path.join(M6, "traj_forest"),
                      "--trees", "cold," + ",".join(GROUPS), "--tag", "m6forest"],
           "m6forest_train.out"):
    sys.exit("[night] distillation failed -- ABORT")
ck = os.path.join(LOG, "m4", "m6forest_policy.pt")
assert os.path.exists(ck), ck

run("generalise", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
                   "--mode", "generalise", "--half", "B", "--tree_suffix", "_train2000",
                   "--cold_tree", COLD, "--ckpt", ck], "m6d_gen_forest.out")
run("mix", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
            "--mode", "mix", "--half", "B", "--tree_suffix", "_train2000",
            "--cold_tree", COLD, "--ckpt", ck], "m6d_mix_forest.out")
print(f"[night] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
