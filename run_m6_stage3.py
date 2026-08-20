#!/usr/bin/env python
"""M6 stage 3-5, with the group list derived in PYTHON rather than in a shell variable.

The shell chain mangled `GROUPS` (CRLF line endings) into "197121" and would have invoked the
trainer with a bank that does not exist. Generation was unaffected -- it never read that variable --
so the banked trajectories are reused as-is. VALIDATION ONLY throughout; the test set stays sealed.
"""
import glob, os, subprocess, sys, time

R = r"C:\dev\phd\casper"
PY = r"C:\Program Files\Python312\python.exe"
os.chdir(R)
env = dict(os.environ, OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6", MKL_NUM_THREADS="6")

groups = sorted(os.path.basename(p)[len("m6k_"):-len(".json")]
                for p in glob.glob("experiments/battery/m6/m6k_*.json"))
assert groups, "no opener-fitted trees"
trees = "cold," + ",".join(groups)
print(f"[stage3] groups: {groups}", flush=True)

def run(tag, args, log, need=None):
    print(f"[stage3] {tag} START {time.strftime('%H:%M:%S')}", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.call([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=env)
    ok = rc == 0 and (need is None or need in open(log, encoding="utf-8", errors="replace").read())
    print(f"[stage3] {tag} rc={rc} ok={ok} {time.strftime('%H:%M:%S')}", flush=True)
    return ok

# wait for generation (started by the shell chain) to finish
for _ in range(180):
    g = "experiments/battery/m6c_gen.out"
    if os.path.exists(g) and "[done] all slices banked" in open(g, encoding="utf-8", errors="replace").read():
        break
    time.sleep(30)
else:
    sys.exit("[stage3] generation never completed -- ABORT")
print("[stage3] generation complete", flush=True)

if not run("distil", ["src/instrument/m4b_policy.py", "--i_have_author_goahead", "--threads", "6",
                      "--epochs", "3", "--anchor_encoding", "none", "--no_cold_all",
                      "--traj_dir", "experiments/battery/m6/traj", "--trees", trees,
                      "--tag", "m6"], "experiments/battery/m6_train.out"):
    sys.exit("[stage3] distillation failed -- ABORT")

for src in ("experiments/battery/m4/m6_policy.pt", "experiments/battery/m6/m6_policy.pt"):
    if os.path.exists(src):
        dst = "experiments/battery/m6/m6_policy.pt"
        if src != dst:
            import shutil; shutil.copy(src, dst)
        break
else:
    sys.exit("[stage3] no checkpoint written -- ABORT")

run("generalise", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
                   "--mode", "generalise", "--half", "B"], "experiments/battery/m6d_generalise.out")
run("mix", ["src/instrument/m6d_eval.py", "--i_have_author_goahead", "--threads", "6",
            "--mode", "mix", "--half", "B"], "experiments/battery/m6e_mix.out")
print(f"[stage3] M6 PIPELINE DONE {time.strftime('%H:%M:%S')}", flush=True)
