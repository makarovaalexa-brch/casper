#!/usr/bin/env bash
# Auto-resume driver for arena_b2_budget.py: the long greedy build keeps getting killed (~30min
# watchdog). build_b2 checkpoints after every turn and resumes from the partial cache, so we just
# relaunch until the script prints its DONE marker. Curve+anchors eval (post-greedy) is short enough
# to finish within one launch window once the 25-pick sequence is cached.
set -u
cd /c/dev/phd/casper
LOG=.cache/arena/b2budget_direct.log
PY="/c/Program Files/Python312/python.exe"
for i in $(seq 1 15); do
  if grep -q "b2budget\] DONE" "$LOG" 2>/dev/null; then
    echo "[autoresume] DONE detected before attempt $i" >> "$LOG"; break
  fi
  echo "[autoresume] === attempt $i $(date) ===" >> "$LOG"
  "$PY" -u scripts/arena_b2_budget.py >> "$LOG" 2>&1
  rc=$?
  echo "[autoresume] attempt $i exited rc=$rc $(date)" >> "$LOG"
  if grep -q "b2budget\] DONE" "$LOG" 2>/dev/null; then break; fi
  sleep 5
done
echo "B2_AUTORESUME_FINISHED" >> "$LOG"
