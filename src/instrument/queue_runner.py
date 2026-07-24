r"""queue_runner.py -- OVERNIGHT self-driving queue (author directive 2026-07-24; Fable low-token).

Sequential decision-independent stages with a resumable state file
(experiments/battery/queue_state.json) and a heartbeat thread (60 s). A Task-Scheduler watchdog
(queue_watchdog.ps1, every 15 min) relaunches this runner if it dies before stage 'done'; the state
file makes the relaunch resume at the interrupted stage (stages are idempotent re-runs).

  S1  wait for the C-full trainer (cmdline match 'tag cfull'); if it died DIRTY (no '[train] done' in
      the log) relaunch it with --resume (detached) and keep waiting.
  S2  tradeoff_ledger.py --full_threads with both trained ckpts; commit the JSON.
  S3  clite_gates.py (trained-module concept gates); commit the JSON.
  S4  FILLER: run_snap_ml20m.py --only dae,multvae (~16 h, decision-independent roadmap item).
  done  HARD STOP -- NO certification retrain (the author selects the winning config first).

Usage: python -u src/instrument/queue_runner.py     (launch via Start-Process, session-independent)
"""
import os
import sys
import json
import time
import threading
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
BATT = os.path.join(_ROOT, "experiments", "battery")
STATE = os.path.join(BATT, "queue_state.json")
RUNLOG = os.path.join(BATT, "queue_runner.log")
PY = sys.executable
STAGES = ["S1", "S2", "S3", "S4", "done"]

_cur = {"stage": "S1"}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUNLOG, "a") as f:
        f.write(line + "\n")


def write_state():
    os.makedirs(BATT, exist_ok=True)
    json.dump({"stage": _cur["stage"], "heartbeat": time.time(),
               "pid": os.getpid()}, open(STATE, "w"))


def heartbeat_loop():
    while True:
        try:
            write_state()
        except Exception:
            pass
        time.sleep(60)


def read_stage():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE)).get("stage", "S1")
        except Exception:
            return "S1"
    return "S1"


def ps(cmd):
    """Run a PowerShell one-liner, return stdout text."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip()


def find_pid(pattern):
    out = ps(f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
             f"Select-Object -ExpandProperty ProcessId")
    pids = [int(x) for x in out.split() if x.strip().isdigit() and int(x) != os.getpid()]
    return pids[0] if pids else None


def run_stage_cmd(args_list, out_path):
    """Run a child python stage, appending stdout+stderr to out_path. Returns returncode."""
    log(f"exec: {' '.join(args_list)} -> {out_path}")
    with open(out_path, "a") as f:
        r = subprocess.run([PY, "-u"] + args_list, cwd=_ROOT, stdout=f, stderr=subprocess.STDOUT)
    log(f"exit code {r.returncode}")
    return r.returncode


def git_commit(paths, msg):
    try:
        subprocess.run(["git", "add"] + paths, cwd=_ROOT, capture_output=True)
        r = subprocess.run(["git", "commit", "-m",
                            msg + "\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>"],
                           cwd=_ROOT, capture_output=True, text=True)
        log(f"git commit: {'OK' if r.returncode == 0 else r.stdout.strip() or r.stderr.strip()}")
    except Exception as e:
        log(f"git commit failed (non-fatal): {e}")


def s1_wait_cfull():
    trainer_log = os.path.join(BATT, "cfull_train.log")
    while True:
        pid = find_pid("tag cfull")
        if pid:
            time.sleep(60)
            continue
        txt = open(trainer_log).read() if os.path.exists(trainer_log) else ""
        if "[train] done" in txt:
            log("S1: C-full trainer finished CLEAN")
            return
        # dead dirty -> relaunch with --resume (detached; append logs)
        log("S1: C-full trainer DEAD DIRTY -> relaunching with --resume")
        ps("Start-Process -FilePath '" + PY + "' -ArgumentList "
           "'-u','src/instrument/train_tower_t2.py','--train','--tag','cfull',"
           "'--concept_tokens','--select_cold','--p_interview','0.5','--resume' "
           f"-WorkingDirectory '{_ROOT}' "
           f"-RedirectStandardOutput '{os.path.join(BATT, 'cfull_train_resume.log')}' "
           f"-RedirectStandardError '{os.path.join(BATT, 'cfull_train_resume.err')}' "
           "-WindowStyle Hidden")
        time.sleep(180)


def s2_ledger():
    rc = run_stage_cmd(["src/instrument/tradeoff_ledger.py", "--full_threads",
                        "--clite_ckpt", ".cache/instrument/cfold_best.pt",
                        "--cfull_ckpt", ".cache/instrument/cfull_best.pt"],
                       os.path.join(BATT, "ledger_run.log"))
    if rc == 0:
        git_commit(["experiments/battery/tradeoff_ledger.json"],
                   "Tradeoff ledger COMPLETE (all four rungs: per-answer + redundancy + split-G5 + "
                   "deployment/crossovers) -- the author decision table [queue S2]")
    else:
        log("S2 FAILED (ledger rc != 0) -- continuing to S3; ledger left for manual rerun")


def s3_gates():
    rc = run_stage_cmd(["src/instrument/clite_gates.py", "--full_threads"],
                       os.path.join(BATT, "clite_gates_run.log"))
    if rc == 0:
        git_commit(["experiments/battery/clite_gates.json"],
                   "C-lite trained-module concept gates (G3a-c flip, G6-c existential + duplicates, "
                   "G8-c order) [queue S3]")
    else:
        log("S3 FAILED (gates rc != 0) -- continuing to S4")


def s4_snap():
    rc = run_stage_cmd([os.path.join("scripts", "baselines", "run_snap_ml20m.py"),
                        "--only", "dae,multvae"],
                       os.path.join(BATT, "snap_ml20m_run.log"))
    log(f"S4 snap finished rc={rc}")


def main():
    start = read_stage()
    if start == "done":
        log("queue already done; exiting")
        return
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    log(f"queue runner START (pid {os.getpid()}) at stage {start}")
    idx = STAGES.index(start) if start in STAGES else 0
    for stage in STAGES[idx:]:
        if stage == "done":
            break
        _cur["stage"] = stage; write_state()
        log(f"=== STAGE {stage} ===")
        {"S1": s1_wait_cfull, "S2": s2_ledger, "S3": s3_gates, "S4": s4_snap}[stage]()
    _cur["stage"] = "done"; write_state()
    log("=== QUEUE DONE (HARD STOP -- no certification retrain; author rules first) ===")


if __name__ == "__main__":
    main()
