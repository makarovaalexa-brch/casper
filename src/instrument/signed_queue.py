r"""signed_queue.py -- SIGNED-CONCEPTS overnight queue (DESIGN_SIGNED_CONCEPTS; author GO 2026-07-25).

Stages (state file experiments/battery/signed_queue_state.json; heartbeat 60 s; watchdog =
signed_watchdog.ps1 via schtasks every 15 min; stages idempotent on relaunch):
  U0  demote the running DAE/MultVAE snap to IDLE priority (stated judgment: the author-GO retrains
      outrank the decision-independent filler; snap TIMINGS are not results, only final metrics)
  U1  train signed C-lite  (concept_fold --train --signed --tag cfold_signed; ~2 h; auto-resume on
      dirty death via --resume_from_best)
  U2  train signed C-full  (train_tower_t2 --train --tag cfull_signed --concept_tokens
      --signed_concepts --select_cold --p_interview 0.5; ~3-5 h; auto --resume on dirty death)
  U3  acceptance batch: signed_sel_gate on the RETRAINED C-lite (--ckpt cfold_signed_best
      --out_tag signed_retrain: deployment curve + counterfeit + leak R2) + the FULL ledger with
      rungs armA,fixab,clite,cfull,sclite,scfull; commit the JSONs.
  done HARD STOP -- NO certification retrain (author picks the final winner first).
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
STATE = os.path.join(BATT, "signed_queue_state.json")
RUNLOG = os.path.join(BATT, "signed_queue.log")
PY = sys.executable
STAGES = ["U0", "U1", "U2", "U3", "done"]
_cur = {"stage": "U0"}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUNLOG, "a") as f:
        f.write(line + "\n")


def write_state():
    os.makedirs(BATT, exist_ok=True)
    json.dump({"stage": _cur["stage"], "heartbeat": time.time(), "pid": os.getpid()},
              open(STATE, "w"))


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
            return json.load(open(STATE)).get("stage", "U0")
        except Exception:
            return "U0"
    return "U0"


def ps(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True, timeout=180)
    return r.stdout.strip()


def find_pid(pattern):
    out = ps(f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
             f"Select-Object -ExpandProperty ProcessId")
    pids = [int(x) for x in out.split() if x.strip().isdigit() and int(x) != os.getpid()]
    return pids[0] if pids else None


def start_detached(arglist, out_log, err_log):
    args_ps = ",".join(f"'{a}'" for a in ["-u"] + arglist)
    ps(f"Start-Process -FilePath '{PY}' -ArgumentList {args_ps} -WorkingDirectory '{_ROOT}' "
       f"-RedirectStandardOutput '{out_log}' -RedirectStandardError '{err_log}' -WindowStyle Hidden")


def run_train(tag_pattern, launch_args, resume_args, done_marker, log_path):
    """Launch a detached trainer, babysit it, auto-resume on dirty death, return when done clean."""
    if find_pid(tag_pattern) is None:
        log(f"launch: {' '.join(launch_args)}")
        start_detached(launch_args, log_path, log_path.replace(".log", ".err"))
        time.sleep(60)
    while True:
        pid = find_pid(tag_pattern)
        if pid:
            time.sleep(90)
            continue
        txt = open(log_path).read() if os.path.exists(log_path) else ""
        if done_marker in txt:
            log(f"{tag_pattern}: trainer finished CLEAN")
            return True
        log(f"{tag_pattern}: DEAD DIRTY -> relaunch w/ resume")
        start_detached(resume_args, log_path.replace(".log", "_resume.log"),
                       log_path.replace(".log", "_resume.err"))
        log_path = log_path.replace(".log", "_resume.log")
        time.sleep(120)


def run_stage_cmd(args_list, out_path):
    log(f"exec: {' '.join(args_list)} -> {out_path}")
    with open(out_path, "a") as f:
        r = subprocess.run([PY, "-u"] + args_list, cwd=_ROOT, stdout=f, stderr=subprocess.STDOUT)
    log(f"exit {r.returncode}")
    return r.returncode


def git_commit(paths, msg):
    try:
        subprocess.run(["git", "add"] + paths, cwd=_ROOT, capture_output=True)
        r = subprocess.run(["git", "commit", "-m",
                            msg + "\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>"],
                           cwd=_ROOT, capture_output=True, text=True)
        log(f"git: {'OK' if r.returncode == 0 else (r.stdout or r.stderr).strip()[:120]}")
    except Exception as e:
        log(f"git failed (non-fatal): {e}")


def u0_demote_snap():
    pid = find_pid("run_snap_ml20m")
    if pid:
        ps(f"(Get-Process -Id {pid}).PriorityClass = 'Idle'")
        log(f"U0: snap pid {pid} demoted to IDLE (author-GO retrains take precedence; snap timings "
            "are not results)")
    else:
        log("U0: no snap process found (already finished)")


def u1_clite():
    run_train("tag cfold_signed",
              ["src/instrument/concept_fold.py", "--train", "--signed", "--tag", "cfold_signed"],
              ["src/instrument/concept_fold.py", "--train", "--signed", "--tag", "cfold_signed",
               "--resume_from_best"],
              "[cfold] done", os.path.join(BATT, "cfold_signed_train.log"))


def u2_cfull():
    run_train("tag cfull_signed",
              ["src/instrument/train_tower_t2.py", "--train", "--tag", "cfull_signed",
               "--concept_tokens", "--signed_concepts", "--select_cold", "--p_interview", "0.5"],
              ["src/instrument/train_tower_t2.py", "--train", "--tag", "cfull_signed",
               "--concept_tokens", "--signed_concepts", "--select_cold", "--p_interview", "0.5",
               "--resume"],
              "[train] done", os.path.join(BATT, "cfull_signed_train.log"))


def u3_acceptance():
    rc1 = run_stage_cmd(["src/instrument/signed_sel_gate.py", "--full_threads",
                         "--ckpt", ".cache/instrument/cfold_signed_best.pt",
                         "--out_tag", "signed_retrain"],
                        os.path.join(BATT, "signed_retrain_gate.log"))
    rc2 = run_stage_cmd(["src/instrument/tradeoff_ledger.py", "--full_threads",
                         "--rungs", "armA,fixab,clite,cfull,sclite,scfull"],
                        os.path.join(BATT, "ledger_signed_run.log"))
    paths = []
    if rc1 == 0:
        paths.append("experiments/battery/signed_sel_gate_signed_retrain.json")
    if rc2 == 0:
        paths.append("experiments/battery/tradeoff_ledger.json")
    if paths:
        git_commit(paths, "Signed-retrain acceptance batch: retrained deployment gate "
                          "(counterfeit + leak R2) + 6-rung tradeoff ledger "
                          "(armA/fixab/clite/cfull/sclite/scfull) [signed queue U3]")


def main():
    start = read_stage()
    if start == "done":
        log("signed queue already done"); return
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    log(f"signed queue START (pid {os.getpid()}) at {start}")
    idx = STAGES.index(start) if start in STAGES else 0
    for stage in STAGES[idx:]:
        if stage == "done":
            break
        _cur["stage"] = stage; write_state()
        log(f"=== STAGE {stage} ===")
        {"U0": u0_demote_snap, "U1": u1_clite, "U2": u2_cfull, "U3": u3_acceptance}[stage]()
    _cur["stage"] = "done"; write_state()
    log("=== SIGNED QUEUE DONE (HARD STOP -- author picks the final winner) ===")


if __name__ == "__main__":
    main()
