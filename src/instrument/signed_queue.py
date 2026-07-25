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
    """PowerShell query; NEVER raises (2026-07-25 incident: TimeoutExpired from an inherited pipe
    killed both runner instances ~3 min after each launch)."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=120)
        return r.stdout.strip()
    except Exception:
        return ""



def find_pid(pattern):
    out = ps(f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
             f"Select-Object -ExpandProperty ProcessId")
    pids = [int(x) for x in out.split() if x.strip().isdigit() and int(x) != os.getpid()]
    return pids[0] if pids else None


def start_detached(arglist, out_log, err_log):
    """Direct DETACHED Popen (2026-07-25 fix): no powershell middleman, no pipes (the Start-Process
    route left an inherited pipe that hung the runner's subprocess.run and ALSO overwrote the child
    log on relaunch). Append-mode file handles; child survives runner death."""
    DETACHED = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    fo = open(out_log, "a"); fe = open(err_log, "a")
    p = subprocess.Popen([PY, "-u"] + arglist, cwd=_ROOT, stdout=fo, stderr=fe,
                         stdin=subprocess.DEVNULL, creationflags=DETACHED)
    fo.close(); fe.close()
    log(f"detached pid {p.pid}: {' '.join(arglist)}")
    return p.pid


def _log_fresh(path, secs=300):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < secs


def _done(log_paths, marker):
    for lp in log_paths:
        if os.path.exists(lp) and marker in open(lp, errors="replace").read():
            return True
    return False


def run_train(tag_pattern, launch_args, resume_args, done_marker, log_path):
    """Idempotent babysitter (2026-07-25 rules): (a) ARTIFACT FIRST -- done marker in any log =>
    stage complete, never relaunch; (b) LIVENESS = pid match OR log freshness (<5 min) -- a flaky
    CIM query alone can NEVER trigger a relaunch onto a live trainer; (c) resume-launch only when
    provably dead AND not done."""
    logs = [log_path, log_path.replace(".log", "_resume.log")]
    launched_once = False
    while True:
        if _done(logs, done_marker):
            log(f"{tag_pattern}: done marker present -> stage COMPLETE")
            return True
        alive = (find_pid(tag_pattern) is not None) or any(_log_fresh(lp) for lp in logs)
        if alive:
            time.sleep(90)
            continue
        if not launched_once and not any(os.path.exists(lp) for lp in logs):
            log(f"launch: {' '.join(launch_args)}")
            start_detached(launch_args, log_path, log_path.replace(".log", ".err"))
        else:
            log(f"{tag_pattern}: provably dead (no pid, logs stale, no done marker) -> resume")
            start_detached(resume_args, logs[1], logs[1].replace(".log", ".err"))
        launched_once = True
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
    # single-instance lock: a FRESH heartbeat from another pid means a live runner -- exit
    if os.path.exists(STATE):
        try:
            st = json.load(open(STATE))
            if st.get("pid") != os.getpid() and (time.time() - st.get("heartbeat", 0)) < 180:
                other = find_pid("signed_queue")
                if other and other != os.getpid():
                    log(f"another live runner (pid {other}, fresh heartbeat) -- exiting")
                    return
        except Exception:
            pass
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
