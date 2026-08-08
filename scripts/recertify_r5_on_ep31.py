r"""recertify_r5_on_ep31.py -- re-measure R5's whole evidence base on the CERTIFIED tower, in order.

WHY. Chapter A's graded-input claim -- that the instrument consumes signed/graded answers natively and
that this pays on the SHORT interview -- rests on exactly two artefacts, and on 2026-08-08 both were found
to have been measured on `t2final_best.pt`, the PRE-certification tower, five days after `t2i26c_ep31.pt`
was certified and frozen:

    experiments/battery/premium_vs_k_t2final.json   -> the graded-premium curve behind fig:gradedcurve
    experiments/baselines/arm_n/sign_probe.json     -> the "+0.023 at k=2" quoted in prose

Three chained defaults produced that (run_battery_phaseA, arm_n_tower, plot_graded_curve), all repointed
in 58152d9. This driver re-runs both on ep31 so the claim is measured on the tower the chapter certifies.

WHY A DRIVER RATHER THAN TWO LAUNCHES. HARD RULE 11(d): never two heavy jobs on the box at once. These
run strictly in sequence, and each writes its own snapshot-stamped output, so a crash in the second
cannot damage the first.

WHAT TO COMPARE AGAINST (the pre-certification values, for the delta -- NOT for citation):
    premium_vs_k  k=2 +0.0147 [+0.0127,+0.0167] | k=4 +0.0083 | k=8 -0.0006 | k=16 -0.0007 | full -0.0002
    sign_probe    k=1 +0.0169 | k=2 +0.0229 | k=4 +0.0151 | k=8 -0.0009 | k=16 -0.0018 | k=32 +0.0005

NOTE, and it needs saying in the chapter: these two are DIFFERENT ESTIMANDS, not a number and its
replication. premium_vs_k is the VAL cohort under the parity mask (only liked fold-in masked);
sign_probe is the TEST cohort under the protocol pool, with the question set drawn from the full rated
history. Both are legitimate; the chapter currently quotes +0.023 in prose and +0.0147 in the figure as
though they were one quantity.

  python scripts/recertify_r5_on_ep31.py
"""
import os
import sys
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
SNAP = os.path.join(_ROOT, ".cache", "instrument", "t2i26c_ep31.pt")
LOG = os.path.join(_ROOT, "experiments", "battery", "recertify_r5.log")

STEPS = [
    ("premium_vs_k (graded premium vs interview length, VAL, parity mask)",
     [sys.executable, os.path.join(_ROOT, "src", "instrument", "premium_vs_k.py"), "--snapshot", SNAP]),
    ("sign_probe (same questions, different answer expressiveness, TEST, protocol pool)",
     [sys.executable, os.path.join(_ROOT, "src", "baselines", "arm_n_sign_probe.py"),
      "--snapshot", SNAP]),
]


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    if not os.path.exists(SNAP):
        raise SystemExit(f"certified snapshot missing: {SNAP}")
    logln(f"re-certifying R5 on {os.path.basename(SNAP)}; {len(STEPS)} steps, strictly sequential")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for i, (name, cmd) in enumerate(STEPS, 1):
        logln(f"--- step {i}/{len(STEPS)}: {name}")
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=_ROOT, env=env)
        logln(f"--- step {i} exit={rc} in {(time.time() - t0) / 60:.1f} min")
        if rc != 0:
            # Report and continue: a failure in one estimand must not silently cancel the other, and the
            # author's standing ruling is that gates REPORT, they do not terminate.
            logln(f"*** step {i} FAILED -- continuing to the next step; this one must be re-run ***")
    logln("done")


if __name__ == "__main__":
    main()
