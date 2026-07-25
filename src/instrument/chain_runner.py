r"""chain_runner.py -- SESSION-INDEPENDENT overnight chain for the answer-decomposition battery (author
directive 2026-07-25: run everything, no babysitting). Idempotent artifact-checked stages so a restart
RESUMES rather than restarts; a heartbeat + pid in the state file let a schtasks watchdog detect death.

Stages (each skipped if its artifact already exists):
  1 main        answer_contrast.py --arms G,B,O,K,U,S,C   -> answer_contrast_newrec.json
  2 granularity concept_granularity.py                    -> concept_granularity.json
  3 imputers    answer_contrast.py --arms B,O,E,P ...      -> answer_contrast_imputers.json  (OPTIONAL:
                 a failure here does not block consolidation)
  4 consolidate consolidate.py                            -> answer_decomposition_FINAL.json + RESULT md

Stage 1 special-cases an ALREADY-LIVE manual run: if its artifact is missing but a producer log
(answer_contrast_diag.out or the stage's own log) is fresh, it WAITS instead of relaunching (never
relaunch over a live job). Done marker: chain_DONE.marker.

Usage (normally launched detached by the watchdog / operator):
  python src/instrument/chain_runner.py
"""
import os
import sys
import json
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
BAT = os.path.join(_ROOT, "experiments", "battery")
STATE = os.path.join(BAT, "chain_state.json")
DONE = os.path.join(BAT, "chain_DONE.marker")
PY = sys.executable
LIVE_AGE = 2400                                                 # a log fresher than 40 min => live producer
POLL = 30
SINGLETON_AGE = 90                                              # another instance w/ heartbeat < 90s = alive

STAGES = [
    {"name": "main", "artifact": "answer_contrast_newrec.json", "need_key": "K",
     "cmd": [PY, "-u", os.path.join(_HERE, "answer_contrast.py"), "--arms", "G,B,O,K,U,S,C"],
     "log": "chain_main", "watch": ["answer_contrast_diag.out", "chain_main.out"], "optional": False},
    {"name": "granularity", "artifact": "concept_granularity.json",
     "cmd": [PY, "-u", os.path.join(_HERE, "concept_granularity.py")],
     "log": "chain_gran", "watch": ["chain_gran.out"], "optional": False},
    {"name": "imputers", "artifact": "answer_contrast_imputers.json",
     "cmd": [PY, "-u", os.path.join(_HERE, "answer_contrast.py"), "--arms", "B,O,E,P",
             "--out", "answer_contrast_imputers.json"],
     "log": "chain_imp", "watch": ["chain_imp.out"], "optional": True},
    {"name": "consolidate", "artifact": "answer_decomposition_FINAL.json",
     "cmd": [PY, "-u", os.path.join(_HERE, "consolidate.py")],
     "log": "chain_cons", "watch": ["chain_cons.out"], "optional": False},
]


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(BAT, "chain_runner.log"), "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"stages": {}, "pid": os.getpid(), "started": time.time()}


def save_state(st):
    st["pid"] = os.getpid()
    st["heartbeat"] = time.time()
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)


def artifact_ok(stage):
    p = os.path.join(BAT, stage["artifact"])
    if not os.path.exists(p):
        return False
    if stage.get("need_key"):
        try:
            return stage["need_key"] in json.load(open(p)).get("arms", [])
        except Exception:
            return False
    return True


def log_age(stage):
    ages = []
    for w in stage.get("watch", []):
        p = os.path.join(BAT, w)
        if os.path.exists(p):
            ages.append(time.time() - os.path.getmtime(p))
    return min(ages) if ages else 1e18


def run_stage(stage, st):
    name = stage["name"]
    if artifact_ok(stage):
        _log(f"stage {name}: artifact present -> SKIP")
        st["stages"][name] = "done"; save_state(st)
        return True
    # never relaunch over a live producer (e.g. the manual main run): wait for its artifact
    if log_age(stage) < LIVE_AGE:
        _log(f"stage {name}: a producer log is fresh -> WAIT for artifact (not relaunching)")
        while log_age(stage) < LIVE_AGE:
            if artifact_ok(stage):
                _log(f"stage {name}: artifact appeared -> done")
                st["stages"][name] = "done"; save_state(st)
                return True
            st["stages"][name] = "waiting_live"; save_state(st)
            time.sleep(POLL)
        if artifact_ok(stage):
            st["stages"][name] = "done"; save_state(st); return True
        _log(f"stage {name}: producer log went stale without an artifact -> will launch")
    # launch the stage
    out = open(os.path.join(BAT, stage["log"] + ".out"), "a")
    err = open(os.path.join(BAT, stage["log"] + ".err"), "a")
    _log(f"stage {name}: LAUNCH {' '.join(os.path.basename(c) for c in stage['cmd'])}")
    st["stages"][name] = "running"; save_state(st)
    proc = subprocess.Popen(stage["cmd"], cwd=_ROOT, stdout=out, stderr=err)
    while proc.poll() is None:
        save_state(st)                                          # heartbeat during the long stage
        time.sleep(POLL)
    rc = proc.returncode
    out.close(); err.close()
    if artifact_ok(stage):
        _log(f"stage {name}: DONE (rc={rc})")
        st["stages"][name] = "done"; save_state(st)
        return True
    _log(f"stage {name}: FAILED (rc={rc}, no artifact)"
         + ("  [optional -> continue]" if stage.get("optional") else ""))
    st["stages"][name] = f"failed_rc{rc}"; save_state(st)
    return bool(stage.get("optional"))


def main():
    os.makedirs(BAT, exist_ok=True)
    if os.path.exists(DONE):
        _log("chain already DONE (marker present) -> exit")
        return
    st = load_state()
    if (st.get("pid") and st["pid"] != os.getpid()
            and (time.time() - st.get("heartbeat", 0)) < SINGLETON_AGE):
        _log(f"another chain instance is alive (pid {st['pid']}, "
             f"hb {int(time.time()-st.get('heartbeat',0))}s ago) -> exit")
        return
    save_state(st)
    _log(f"chain start (pid {os.getpid()})")
    for stage in STAGES:
        ok = run_stage(stage, st)
        if not ok and not stage.get("optional"):
            _log(f"chain HALT at required stage {stage['name']} (watchdog will retry)")
            return
    # consolidate is the last required stage; if its artifact exists we are done
    if artifact_ok(STAGES[-1]):
        open(DONE, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        st["stages"]["_ALL"] = "done"; save_state(st)
        _log("chain COMPLETE -> DONE marker written")
    else:
        _log("chain finished loop but FINAL artifact missing (watchdog will retry)")


if __name__ == "__main__":
    main()
