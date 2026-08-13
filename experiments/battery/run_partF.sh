#!/bin/sh
# Part F run chain. One heavy job at a time (HARD RULE 11d): each stage waits for the previous.
# Every stage refuses to start if the stage it depends on did not actually finish.
set -u
PY="/c/Program Files/Python312/python.exe"
cd /c/dev/phd/casper || exit 1
B=experiments/battery

# --- wait for the trajectory harvest ---------------------------------------------------------
while ! grep -q "gen\] all 15 chunks banked" $B/policy_traj/gen.out; do
  if ! grep -qE "chunk .*DONE in" $B/policy_traj/gen.out; then :; fi
  sleep 20
done
echo "=== HARVEST DONE ==="
grep "all 15 chunks banked" $B/policy_traj/gen.out

# --- 1. the outcome head, main arm ------------------------------------------------------------
"$PY" src/instrument/outcome_head.py --arm main --epochs 3 --batch 1024 --d_k 64 \
  --threads 8 --full_threads --i_have_author_goahead > $B/outcome_main.log 2>&1
echo "=== OUTCOME MAIN exit=$? ==="
tail -6 $B/outcome_main.log

# --- 2. the lift head + the combined P x V policy ----------------------------------------------
if [ -f $B/outcome_head_main_both.pt ]; then
  "$PY" src/instrument/lift_head.py --threads 8 --full_threads \
    --i_have_author_goahead > $B/lift_head.log 2>&1
  echo "=== LIFT HEAD exit=$? ==="
  tail -8 $B/lift_head.log
else
  echo "=== LIFT HEAD SKIPPED: no outcome-head checkpoint ==="
fi

# --- 3. the shuffle control (HARD RULE 5), one epoch is enough to show the collapse ------------
"$PY" src/instrument/outcome_head.py --arm shuffle --epochs 1 --batch 1024 --d_k 64 \
  --threads 8 --full_threads --i_have_author_goahead > $B/outcome_shuffle.log 2>&1
echo "=== OUTCOME SHUFFLE exit=$? ==="
tail -6 $B/outcome_shuffle.log
echo "=== CHAIN COMPLETE ==="
