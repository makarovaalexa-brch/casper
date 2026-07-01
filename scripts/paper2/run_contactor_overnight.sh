#!/bin/bash
# PAPER C overnight: continuous unified-action actor.
#  Phase 1 (snap)  = Wolpertinger replicate of CASPER-R (emit query -> snap to nearest pool entity -> fold real answer)
#  Phase 2 (cont)  = fold the continuous off-pool point directly, geometric answer -> optimise cos(belief,u*)
# Objective throughout = reconstruction 1-cos(u,u*). Train deterministic (seed0); eval seed-avg {1,2,3,7,11} on te[300:].
# Anchors: CASPER-R (discrete winner) = 0.360 full / 0.152 tail ; fullprof ceiling ~0.407/0.216.
set -u
PY="python C:/dev/phd/casper/scripts/paper2/continuous_actor.py"
CK="C:/dev/phd/casper/data/movielens/.cache"
LOG="C:/dev/phd/casper/experiments/paper2/contactor_overnight_2026-06-26.log"
EP=30
echo "=== CONTINUOUS ACTOR overnight | start $(date) ===" > "$LOG"

# ---------- Phase 1: TRAIN snap (replicate) ----------
echo "### TRAIN snap EP=$EP  $(date)" >> "$LOG"
CONTMODE=snap NOBC=1 OBJ=ustar TAG=cont_snap EP=$EP SKIPVAL=1 SEED=1 $PY 2>&1 | grep -E "ep[0-9]+ return|saved .*last" >> "$LOG"

# ---------- Phase 2: TRAIN cont (optimise cos, off-pool) ----------
echo "### TRAIN cont EP=$EP  $(date)" >> "$LOG"
CONTMODE=cont NOBC=1 OBJ=ustar TAG=cont_off EP=$EP SKIPVAL=1 SEED=1 $PY 2>&1 | grep -E "ep[0-9]+ return|saved .*last" >> "$LOG"

# ---------- Stage A: epoch curve (single seed=1) to locate the peak ----------
echo "### STAGE A epoch curve (SEED=1)  $(date)" >> "$LOG"
for E in 3 6 9 12 15 18 21 24 27 30; do
  CONTMODE=snap NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_snap_ep${E}.pt MODES=contactor   TESTRANGE=300:99999 SEED=1 $PY 2>&1 | grep -E "contactor:"   | sed "s/^/[snap ep$E s1] /" >> "$LOG"
  CONTMODE=cont NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_off_ep${E}.pt  MODES=contactor_c TESTRANGE=300:99999 SEED=1 $PY 2>&1 | grep -E "contactor_c:" | sed "s/^/[cont ep$E s1] /" >> "$LOG"
done

# ---------- Stage B: seed-avg {1,2,3,7,11} at 3 epochs ----------
echo "### STAGE B seed-avg {1,2,3,7,11}  $(date)" >> "$LOG"
for E in 12 21 30; do
  for s in 1 2 3 7 11; do
    CONTMODE=snap NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_snap_ep${E}.pt MODES=contactor   TESTRANGE=300:99999 SEED=$s $PY 2>&1 | grep -E "contactor:"   | sed "s/^/[snap ep$E s$s] /" >> "$LOG"
    CONTMODE=cont NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_off_ep${E}.pt  MODES=contactor_c TESTRANGE=300:99999 SEED=$s $PY 2>&1 | grep -E "contactor_c:" | sed "s/^/[cont ep$E s$s] /" >> "$LOG"
  done
done

# ---------- Reference anchors: conc_pop + fullprof, seed-avg ----------
echo "### REFERENCE conc_pop + fullprof (seed-avg)  $(date)" >> "$LOG"
for s in 1 2 3 7 11; do
  CONTMODE=snap NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_snap_ep3.pt MODES=conc_pop,fullprof TESTRANGE=300:99999 SEED=$s $PY 2>&1 | grep -E "conc_pop:|fullprof :" | sed "s/^/[ref s$s] /" >> "$LOG"
done
echo "=== DONE $(date) ===" >> "$LOG"
