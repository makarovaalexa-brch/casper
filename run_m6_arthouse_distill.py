#!/usr/bin/env python
"""Continue the arthouse pipeline from the distillation stage (fit, m6n and m6c are done)."""
import os, subprocess, sys, time
R = r"C:\dev\phd\casper"
PY = r"C:\Program Files\Python312\python.exe"
M6 = os.path.join(R, "experiments", "battery", "m6")
LOG = os.path.join(R, "experiments", "battery")
os.chdir(R)
env = dict(os.environ, OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6", MKL_NUM_THREADS="6")

def run(tag, args, log):
    print(f"[pipe] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(f"[pipe] {tag} rc={rc} {time.strftime('%H:%M:%S')}", flush=True)
    return rc == 0

if not run("distil", ["src/instrument/m4b_policy.py", "--i_have_author_goahead", "--threads", "6",
                      "--epochs", "6", "--anchor_encoding", "none", "--no_cold_all",
                      "--expect_cover", "0.66",
                      "--traj_dir", os.path.join(M6, "traj2k"), "--trees", "cold,arthouse",
                      "--tag", "m6a2k"], "m6a2k_train.out"):
    sys.exit("[pipe] distillation failed -- ABORT")
ck = os.path.join(R, "experiments", "battery", "m4", "m6a2k_policy.pt")
assert os.path.exists(ck), ck
run("generalise", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
                   "--mode", "generalise", "--half", "B", "--groups", "arthouse",
                   "--tree_suffix", "_train2000", "--ckpt", ck], "m6d_arthouse2k.out")
print(f"[pipe] DONE {time.strftime('%H:%M:%S')}", flush=True)
