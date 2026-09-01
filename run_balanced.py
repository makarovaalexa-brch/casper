#!/usr/bin/env python
"""Retrain the forest policy with balanced cohort banks, then re-measure routing and value."""
import os, subprocess, sys, time
R = r"C:\dev\phd\casper"; PY = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(R, "experiments", "battery"); M6 = os.path.join(LOG, "m6")
os.chdir(R); env = dict(os.environ)
COLD = "coldtree_v2_train10000.json"
G = "arthouse,dialogue,gore,inspiring,marvel,teenagers"

def run(tag, args, log):
    print(f"[bal] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(f"[bal] {tag} rc={rc} {time.strftime('%H:%M:%S')}", flush=True)
    return rc == 0

if not run("distil-balanced", ["src/instrument/m4b_policy.py", "--i_have_author_goahead",
                               "--threads", "6", "--epochs", "6", "--anchor_encoding", "none",
                               "--no_cold_all", "--balance_banks",
                               "--traj_dir", os.path.join(M6, "traj_forest"),
                               "--trees", "cold," + G, "--tag", "m6bal"], "m6bal_train.out"):
    sys.exit("[bal] distillation failed -- ABORT")
ck = os.path.join(LOG, "m4", "m6bal_policy.pt")
assert os.path.exists(ck), ck
run("confusion", ["src/instrument/m6v_route_confusion.py", "--i_have_author_goahead",
                  "--threads", "6", "--ckpt", ck], "m6v_conf_bal.out")
run("generalise", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
                   "--mode", "generalise", "--half", "B", "--tree_suffix", "_train2000",
                   "--cold_tree", COLD, "--ckpt", ck], "m6d_gen_bal.out")
run("mix", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
            "--mode", "mix", "--half", "B", "--tree_suffix", "_train2000",
            "--cold_tree", COLD, "--ckpt", ck], "m6d_mix_bal.out")
print(f"[bal] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
