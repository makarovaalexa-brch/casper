#!/usr/bin/env python
"""Arthouse-first pipeline (author, 2026-08-21): *"i want the first group tested on hold-out as it
finishes, and whatever experiments we can do with distill, if it works, a few days is worth it"*.

Waits for the running `arthouse` fit, then in order:
  1. m6n   in-sample AND held-out evaluation of the tree (no refitting)
  2. m6c   trajectories on TRAIN users for {cold, arthouse}, openers from film-pool HALF A
  3. train state-only distillation over those two banks, 6 epochs (the depth-1 forest was still
           climbing at 3, so this trains longer)
  4. m6d   generalisation on HALF B -- films the policy never saw in arthouse openers
  5. restart the remaining five group fits

Two banks make the generalisation test a clean BINARY: given an unseen film from an arthouse user,
does the policy emit arthouse's order or the cold one? That is a sharper probe than the six-group
version, which confounded routing with a six-way choice.

VALIDATION + TRAIN only. The test set is never built, loaded or named.
"""
import glob, json, os, subprocess, sys, time

R = r"C:\dev\phd\casper"
PY = r"C:\Program Files\Python312\python.exe"
M6 = os.path.join(R, "experiments", "battery", "m6")
LOG = os.path.join(R, "experiments", "battery")
os.chdir(R)
env = dict(os.environ, OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6", MKL_NUM_THREADS="6")
ART = os.path.join(M6, "m6k_arthouse_train2000.json")


def run(tag, args, log, need=None):
    print(f"[pipe] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    txt = open(os.path.join(LOG, log), encoding="utf-8", errors="replace").read()
    ok = rc == 0 and (need is None or need in txt)
    print(f"[pipe] {tag} rc={rc} ok={ok} {time.strftime('%H:%M:%S')}", flush=True)
    return ok


# ---- 1. wait for the fit --------------------------------------------------------------------
for _ in range(2880):                      # up to 24h
    f = os.path.join(LOG, "m6k_arthouse_tr2k_v2.out")
    if os.path.exists(f) and "[done]" in open(f, encoding="utf-8", errors="replace").read():
        break
    time.sleep(30)
else:
    sys.exit("[pipe] arthouse never finished -- ABORT")
d = json.load(open(ART, encoding="utf-8"))
print(f"[pipe] arthouse fit complete: {len(d['nodes'])} nodes, "
      f"{d.get('minutes', 0) / 60:.1f} h, split={d.get('split')}", flush=True)

# ---- 2. evaluate the tree, in-sample and held-out --------------------------------------------
run("m6n-eval", ["src/instrument/m6n_train2k_eval.py", "--i_have_author_goahead", "--threads", "6",
                 "--artifact", ART, "--cluster", "arthouse"], "m6n_arthouse_v2.out")

# ---- 3. trajectories for {cold, arthouse}, half-A openers ------------------------------------
if not run("m6c-gen", ["src/instrument/m6c_distill_gen.py", "--i_have_author_goahead",
                       "--threads", "6", "--half", "A", "--groups", "arthouse",
                       "--tree_suffix", "_train2000", "--out_dir", "traj2k"],
           "m6c_gen2k.out", need="[done] all slices banked"):
    sys.exit("[pipe] generation failed -- ABORT")

# ---- 4. distil, state-only, longer -------------------------------------------------------------
if not run("distil", ["src/instrument/m4b_policy.py", "--i_have_author_goahead", "--threads", "6",
                      "--epochs", "6", "--anchor_encoding", "none", "--no_cold_all",
                      "--traj_dir", os.path.join(M6, "traj2k"), "--trees", "cold,arthouse",
                      "--tag", "m6a2k"], "m6a2k_train.out"):
    sys.exit("[pipe] distillation failed -- ABORT")
ck = os.path.join(R, "experiments", "battery", "m4", "m6a2k_policy.pt")
if not os.path.exists(ck):
    sys.exit(f"[pipe] no checkpoint at {ck} -- ABORT")

# ---- 5. generalisation on UNSEEN films ---------------------------------------------------------
run("generalise", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
                   "--mode", "generalise", "--half", "B", "--groups", "arthouse",
                   "--tree_suffix", "_train2000", "--ckpt", ck], "m6d_arthouse2k.out")

# ---- 6. resume the remaining group fits ---------------------------------------------------------
print(f"[pipe] arthouse pipeline done {time.strftime('%H:%M:%S')}; resuming group fits", flush=True)
for g in ("marvel", "gore", "dialogue", "teenagers", "inspiring"):
    if glob.glob(os.path.join(M6, f"m6k_{g}_train2000.json")):
        print(f"[pipe] {g} already fitted -- skipping", flush=True)
        continue
    run(f"fit-{g}", ["src/instrument/m6k_cluster_tree.py", "--i_have_author_goahead",
                     "--threads", "6", "--cluster", g, "--split", "train", "--slice_hi", "34000",
                     "--n_users", "2000", "--depth", "6", "--min_leaf", "200"],
        f"m6k_{g}_tr2k.out", need="[done]")
print(f"[pipe] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
