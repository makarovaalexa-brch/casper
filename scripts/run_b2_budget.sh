#!/usr/bin/env bash
# Driver: wait for the learnability gate's completed T24 b2@10 greedy (24 picks), then run the
# budget-finding per-turn rerun (seeds T25 from T24, extends 1 turn, builds curve + anchors).
set -u
cd /c/dev/phd/casper
LOG=.cache/arena/b2budget.log
: > "$LOG"
T24=.cache/arena/b2v31_T24_K10_n300.json

echo "[driver] $(date) waiting for completed T24 b2@10 greedy ..." >> "$LOG"
while true; do
  if [ -f "$T24" ]; then
    n=$(python -c "import json;print(len(json.load(open(r'$T24'))['seq']))" 2>/dev/null || echo 0)
    echo "[driver] $(date) T24 picks = $n" >> "$LOG"
    if [ "$n" -ge 24 ]; then
      echo "[driver] $(date) T24 complete -> launching budget rerun" >> "$LOG"
      break
    fi
  fi
  sleep 120
done

"/c/Program Files/Python312/python.exe" -u scripts/arena_b2_budget.py >> "$LOG" 2>&1
echo "[driver] $(date) DONE rc=$?" >> "$LOG"
echo "B2_BUDGET_ALL_DONE" >> "$LOG"
