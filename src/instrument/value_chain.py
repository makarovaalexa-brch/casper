r"""value_chain.py -- run the value-dataset workstream BEHIND whatever owns the box right now.

Phase 3 (`policy_labels.py --mode sweep --states deep`) holds the CPU. HARD RULE 11(d): never two
heavy jobs at once. So this waits for a SUSTAINED idle box and then runs, in order:

  1. sample     value_dataset.py --mode sample   ~2.2 h   the unbiased random-prefix scatter
  2. scrape     value_dataset.py --mode scrape   seconds  re-ingest: banked + phase 3 + the samples
  3. meanvec    value_dataset.py --mode meanvec  seconds  the parameter-free bar, by source
  4. train      value_dataset.py --mode train    hours    the refit, with the zeroed-state control

★ IDLE IS DETECTED BY PROCESS NAME AND FOOTPRINT, NEVER BY COMMAND-LINE TEXT. A command line is a
string a job can be launched without; a resident set of gigabytes is what actually contends for the
box. `python.exe` serving `http.server` sits at a few tens of MB and is correctly ignored; a scoring
job sits above HEAVY_RSS. Idle must hold for IDLE_POLLS CONSECUTIVE polls -- a single quiet sample
catches the gap between two states of a sweep and would launch straight into a live job.

Each stage is skipped if its artifact is already on disk, so a restart resumes.
"""
import os
import sys
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
BAT = os.path.join(_ROOT, "experiments", "battery")
PY = sys.executable
LOG = os.path.join(BAT, "value_chain.log")

# MEASURED on this box 2026-08-14, which is why it is not a guessed constant: the four idle
# `python -m http.server` processes sit at 2-10 MB, and the live phase-3 sweep sits at 919 MB. A
# threshold of 1 GB -- the obvious guess -- called a live 919 MB sweep IDLE on the first test.
HEAVY_RSS = 200_000_000
POLL = 120
IDLE_POLLS = 5                 # 10 minutes of sustained quiet before anything is launched

STAGES = [
    {"name": "sample", "artifact": os.path.join(BAT, "value_sample_partial", "prefix_018.json"),
     "cmd": [PY, "-u", os.path.join(_HERE, "value_dataset.py"), "--mode", "sample",
             "--n_prefix", "19", "--threads", "6"], "log": "value_sample"},
    {"name": "scrape", "artifact": None,
     "cmd": [PY, "-u", os.path.join(_HERE, "value_dataset.py"), "--mode", "scrape"],
     "log": "value_scrape"},
    {"name": "meanvec", "artifact": None,
     "cmd": [PY, "-u", os.path.join(_HERE, "value_dataset.py"), "--mode", "meanvec"],
     "log": "value_meanvec"},
    {"name": "train", "artifact": os.path.join(BAT, "value_head_union.json"),
     "cmd": [PY, "-u", os.path.join(_HERE, "value_dataset.py"), "--mode", "train",
             "--threads", "6"], "log": "value_train"},
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def heavy_pythons():
    """Every python.exe on the box with a scoring-job footprint, EXCLUDING this process."""
    me = os.getpid()
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "ForEach-Object { \"$($_.ProcessId) $($_.WorkingSetSize)\" }"],
        capture_output=True, text=True, timeout=120).stdout
    hits = []
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) != 2:
            continue
        pid, rss = int(parts[0]), int(parts[1])
        if pid != me and rss >= HEAVY_RSS:
            hits.append((pid, rss))
    return hits


def wait_for_idle():
    quiet = 0
    while True:
        h = heavy_pythons()
        if h:
            quiet = 0
            log(f"box busy: {len(h)} heavy python.exe "
                + ", ".join(f"pid {p} {r / 2**30:.1f} GB" for p, r in h))
        else:
            quiet += 1
            log(f"box quiet {quiet}/{IDLE_POLLS}")
            if quiet >= IDLE_POLLS:
                log("SUSTAINED IDLE -- starting")
                return
        time.sleep(POLL)


def main():
    log("=" * 96)
    log("value chain armed; waiting for a sustained idle box before any stage")
    wait_for_idle()
    for st in STAGES:
        if st["artifact"] and os.path.exists(st["artifact"]):
            log(f"--- stage {st['name']}: artifact present, SKIP")
            continue
        log(f"--- stage {st['name']}: {' '.join(st['cmd'])}")
        o = open(os.path.join(BAT, st["log"] + ".out"), "a")
        e = open(os.path.join(BAT, st["log"] + ".err"), "a")
        rc = subprocess.call(st["cmd"], cwd=_ROOT, stdout=o, stderr=e)
        o.close(); e.close()
        log(f"--- stage {st['name']} exit={rc}")
        if rc != 0:
            log("=== ABORT: a stage failed; the chain stops rather than building on it")
            return
    log("CHAIN COMPLETE")


if __name__ == "__main__":
    main()
