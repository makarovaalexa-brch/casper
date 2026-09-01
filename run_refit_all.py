#!/usr/bin/env python
"""Refit chain: gate on the m6k smoke test, then the static order and five opener trees.

Design (author, 2026-08-25): every tree AND the static order fitted on TRAIN, evaluated on
VALIDATION. Sizes: cold 10,000 (done), static 10,000, opener trees N=2,000 per cohort.

One job at a time. Each step's log is separate. A failed step aborts the chain rather than
poisoning the next one.
"""
import json, os, subprocess, sys, time

R = r"C:\dev\phd\casper"; PY = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(R, "experiments", "battery"); M6 = os.path.join(LOG, "m6")
os.chdir(R)
env = dict(os.environ)
SMOKE = os.path.join(LOG, "m6k_SMOKE_dialogue.out")


def run(tag, args, log, need=None):
    print(f"[chain] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(LOG, log), "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    txt = open(os.path.join(LOG, log), encoding="utf-8", errors="replace").read()
    ok = rc == 0 and (need is None or need in txt)
    print(f"[chain] {tag} rc={rc} ok={ok} {time.strftime('%H:%M:%S')}", flush=True)
    return ok


# ---- 0. gate on the smoke test -----------------------------------------------------------------
print(f"[chain] waiting on the m6k smoke test {time.strftime('%H:%M:%S')}", flush=True)
for _ in range(120):                                    # up to 60 min
    t = open(SMOKE, encoding="utf-8", errors="replace").read() if os.path.exists(SMOKE) else ""
    if "[done]" in t:
        print("[chain] smoke PASSED -- heap reuse and sweep banking work inside m6k", flush=True)
        break
    if "Traceback" in t or "AssertionError" in t:
        sys.exit("[chain] smoke FAILED -- ABORT, do not run the queue on a broken fitter")
    time.sleep(30)
else:
    sys.exit("[chain] smoke never finished -- ABORT")

# ---- 1. static combined order on TRAIN ----------------------------------------------------------
if not run("static-train", ["src/instrument/m6s_static_train.py", "--i_have_author_goahead",
                            "--threads", "6", "--n_users", "10000"],
           "m6s_static_train.out", need="[done]"):
    sys.exit("[chain] static order failed -- ABORT")

# ---- 2. five opener trees, N=2,000 --------------------------------------------------------------
for g in ("dialogue", "marvel", "gore", "teenagers", "inspiring"):
    art = os.path.join(M6, f"m6k_{g}_train2000.json")
    if os.path.exists(art):
        print(f"[chain] {g} already fitted -- skipping", flush=True)
        continue
    if not run(f"fit-{g}", ["src/instrument/m6k_cluster_tree.py", "--i_have_author_goahead",
                            "--threads", "6", "--cluster", g, "--split", "train",
                            "--slice_hi", "34000", "--n_users", "2000",
                            "--depth", "6", "--min_leaf", "200"],
               f"m6k_{g}_tr2k.out", need="[done]"):
        print(f"[chain] {g} FAILED -- continuing with the rest", flush=True)
print(f"[chain] ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
