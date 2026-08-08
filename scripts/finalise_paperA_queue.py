r"""finalise_paperA_queue.py -- run the last two Paper A jobs in sequence, after user-kNN frees the box.

AUTHOR SAID (2026-08-08, verbatim): "ok, do all of the above and try to finalise"

"All of the above" resolves to, in order:
  1. wait for the live user-kNN job (run_userknn_proper.py) to finish -- HARD RULE 11(d), one heavy job
     at a time, and a third process alongside adaptive_harness would risk both;
  2. re-select the shared MF setting on the PRIMARY metric (run_reselect_on_ndcg10.py);
  3. the within-model weight-vs-estimand test (run_weight_vs_estimand.py), which the author ruled should
     stay in the chapter and therefore needs support rather than deletion.

Not in scope, by author decision earlier the same day: the NATIVE column is FROZEN -- failed cells get a
dash and one sentence, not another run -- and the RBMF native cell is dropped under that freeze.

A failing step REPORTS and the queue continues (standing author ruling: gates report, they never
terminate). Each step writes its own snapshot-stamped output, so a crash in one cannot damage another.

  python scripts/finalise_paperA_queue.py
"""
import os
import sys
import time
import json
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
LOG = os.path.join(_ROOT, "experiments", "baselines", "arm_n", "finalise_queue.log")
UKNN_OUT = os.path.join(_ROOT, "experiments", "baselines", "arm_n", "userknn_proper_armn.json")

STEPS = [
    ("re-select the shared MF setting on NDCG@10 (both arms)",
     [sys.executable, os.path.join(_ROOT, "src", "baselines", "run_reselect_on_ndcg10.py")]),
    ("within-model weight-vs-estimand test",
     [sys.executable, os.path.join(_ROOT, "src", "baselines", "run_weight_vs_estimand.py")]),
]


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def uknn_running():
    """True while run_userknn_proper.py is still alive. Matched on the command line, because
    Get-Process alone cannot distinguish our jobs from any other python.exe (memory: pgrep is blind
    to python.exe on this box, so command-line matching is the only reliable test)."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
             "Where-Object { $_.CommandLine -like '*run_userknn_proper*' } | Measure-Object).Count"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return int(out or "0") > 0
    except Exception as e:
        logln(f"process check failed ({e.__class__.__name__}); assuming the box is free")
        return False


def main():
    waited = 0
    while uknn_running():
        if waited % 600 == 0:
            logln(f"user-kNN still running; holding the queue ({waited//60} min waited)")
        time.sleep(60)
        waited += 60
    if os.path.exists(UKNN_OUT):
        d = json.load(open(UKNN_OUT))
        g = d.get("graded", {})
        logln(f"user-kNN finished: graded full={g.get('ndcg@10')} tail={g.get('tail_ndcg@10')} "
              f"citable={d.get('citable')}")
    else:
        logln("user-kNN produced no output file -- continuing, but that cell must be re-run")

    for i, (name, cmd) in enumerate(STEPS, 1):
        logln(f"--- step {i}/{len(STEPS)}: {name}")
        t0 = time.time()
        env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        rc = subprocess.call(cmd, cwd=_ROOT, env=env)
        logln(f"--- step {i} exit={rc} in {(time.time()-t0)/60:.1f} min")
        if rc != 0:
            logln(f"*** step {i} FAILED -- continuing; this one must be re-run ***")
    logln("queue done")


if __name__ == "__main__":
    main()
