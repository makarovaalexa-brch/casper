#!/usr/bin/env python
"""Six-teacher forest: order check, then distillation, generalisation and the mix.

All teachers TRAIN-fitted (cold 10,000; six cohort trees 2,000 each); every number on VALIDATION.
"""
import os, subprocess, sys, time
R = r"C:\dev\phd\casper"; PY = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(R, "experiments", "battery"); M6 = os.path.join(LOG, "m6")
os.chdir(R)
env = dict(os.environ)
COLD = "coldtree_v2_train10000.json"

def run(tag, args, log, need=None):
    print(f"[fin] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    txt = open(os.path.join(LOG, log), encoding="utf-8", errors="replace").read()
    ok = rc == 0 and (need is None or need in txt)
    print(f"[fin] {tag} rc={rc} ok={ok} {time.strftime('%H:%M:%S')}", flush=True)
    return ok

run("order-check-6", ["src/instrument/m6t_order_check.py", "--i_have_author_goahead",
                      "--threads", "6"], "m6t_order6.out")

if not run("gen", ["src/instrument/m6c_distill_gen.py", "--i_have_author_goahead", "--threads", "6",
                   "--half", "A", "--tree_suffix", "_train2000", "--out_dir", "traj_forest",
                   "--cold_tree", COLD], "m6c_gen_forest.out", need="[done] all slices banked"):
    sys.exit("[fin] generation failed -- ABORT")

if not run("distil", ["src/instrument/m4b_policy.py", "--i_have_author_goahead", "--threads", "6",
                      "--epochs", "6", "--anchor_encoding", "none", "--no_cold_all",
                      "--traj_dir", os.path.join(M6, "traj_forest"),
                      "--trees", "cold,arthouse,dialogue,marvel,gore,teenagers,inspiring",
                      "--tag", "m6forest"], "m6forest_train.out"):
    sys.exit("[fin] distillation failed -- ABORT")
ck = os.path.join(LOG, "m4", "m6forest_policy.pt")
assert os.path.exists(ck), ck

run("generalise", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
                   "--mode", "generalise", "--half", "B", "--tree_suffix", "_train2000",
                   "--cold_tree", COLD, "--ckpt", ck], "m6d_gen_forest.out")
run("mix", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
            "--mode", "mix", "--half", "B", "--tree_suffix", "_train2000",
            "--cold_tree", COLD, "--ckpt", ck], "m6d_mix_forest.out")
print(f"[fin] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
