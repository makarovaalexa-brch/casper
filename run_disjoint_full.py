#!/usr/bin/env python
"""Wait for the six disjoint teachers, then measure and distil. All arms use the disjoint partition."""
import json, os, subprocess, sys, time
R = r"C:\dev\phd\casper"; PY = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(R, "experiments", "battery"); M6 = os.path.join(LOG, "m6")
os.chdir(R); env = dict(os.environ)
DJ = os.path.join(M6, "m7b_disjoint.npz")
COLD = "coldtree_v2_train10000.json"
G = ["arthouse", "dialogue", "marvel", "gore", "teenagers", "inspiring"]

def run(tag, args, log, need=None):
    print(f"[full] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    txt = open(os.path.join(LOG, log), encoding="utf-8", errors="replace").read()
    ok = rc == 0 and (need is None or need in txt)
    print(f"[full] {tag} rc={rc} ok={ok} {time.strftime('%H:%M:%S')}", flush=True)
    return ok

# ---- 1. wait for the FITTING DRIVER to finish -------------------------------------------------
# NOT for the artifacts to exist: m6k writes its artifact incrementally at every leaf (HARD RULE
# 11a), so a file appears as soon as a tree's FIRST leaf is fitted. Gating on file existence would
# start measuring a partial tree and report it as the result. The driver's own completion line is
# the only signal that all six are actually finished.
CHAIN = os.path.join(LOG, "chain_disjoint.out")
for _ in range(1440):                                    # up to 12 h
    t = open(CHAIN, encoding="utf-8", errors="replace").read() if os.path.exists(CHAIN) else ""
    if "[dj] ALL DONE" in t or "[dj] audit:" in t:
        break
    if "-- ABORT" in t:
        sys.exit("[full] fitting driver aborted -- ABORT")
    time.sleep(30)
else:
    sys.exit("[full] teachers never completed -- ABORT")
missing = [g for g in G if not os.path.exists(os.path.join(M6, f"m6k_dj{g}_train2000.json"))]
if missing:
    sys.exit(f"[full] driver finished but {missing} have no artifact -- ABORT")

# ---- 2. audit BEFORE measuring ----------------------------------------------------------------
bad = {}
for g in G:
    d = json.load(open(os.path.join(M6, f"m6k_dj{g}_train2000.json"), encoding="utf-8"))
    n = d["nodes"]; n = n if isinstance(n, dict) else {r["nid"]: r for r in n}
    k = sum(1 for x in n.values()
            if set(map(tuple, x.get("path", [])))
            & set(map(tuple, (x.get("tail") or x.get("steps") or []))))
    if k: bad[g] = k
print(f"[full] duplicate-question audit: {bad or 'ALL SIX CLEAN'}", flush=True)
if bad: sys.exit(f"[full] {bad} -- ABORT before measuring")

# ---- 3. does teacher VALUE survive disjointing? THE gate --------------------------------------
run("order-check", ["src/instrument/m6t_order_check.py", "--i_have_author_goahead",
                    "--threads", "6", "--disjoint", DJ], "m6t_disjoint.out", need="[done]")

# ---- 4. distil the disjoint forest -------------------------------------------------------------
if not run("gen", ["src/instrument/m6c_distill_gen.py", "--i_have_author_goahead", "--threads", "6",
                   "--half", "A", "--tree_suffix", "_train2000", "--out_dir", "traj_dj",
                   "--cold_tree", COLD, "--disjoint", DJ],
           "m6c_gen_dj.out", need="[done] all slices banked"):
    sys.exit("[full] generation failed -- ABORT")
if not run("distil", ["src/instrument/m4b_policy.py", "--i_have_author_goahead", "--threads", "6",
                      "--epochs", "6", "--anchor_encoding", "none", "--no_cold_all",
                      "--traj_dir", os.path.join(M6, "traj_dj"),
                      "--trees", "cold," + ",".join(G), "--tag", "m7dj"], "m7dj_train.out"):
    sys.exit("[full] distillation failed -- ABORT")
ck = os.path.join(LOG, "m4", "m7dj_policy.pt")
assert os.path.exists(ck), ck

# ---- 5. routing, unseen films, and the deployment mix ------------------------------------------
run("confusion", ["src/instrument/m6v_route_confusion.py", "--i_have_author_goahead",
                  "--threads", "6", "--ckpt", ck, "--tree_suffix", "_train2000"], "m6v_dj.out")
run("generalise", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
                   "--mode", "generalise", "--half", "B", "--tree_suffix", "_train2000",
                   "--cold_tree", COLD, "--disjoint", DJ, "--ckpt", ck], "m6d_gen_dj.out")
run("mix", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
            "--mode", "mix", "--half", "B", "--tree_suffix", "_train2000",
            "--cold_tree", COLD, "--disjoint", DJ, "--ckpt", ck], "m6d_mix_dj.out")
print(f"[full] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
