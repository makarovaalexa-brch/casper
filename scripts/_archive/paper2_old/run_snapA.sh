#!/bin/bash
# PAPER C path (A): answerability-masked continuous SNAP actor (Wolpertinger over ANSWERABLE pool).
# Fix vs overnight: ANSMASK=1 restricts the snap to answerable entities (train+eval matched) -> no un-askable cheating.
# Train deterministic (seed0); eval seed-avg {1,2,3,7,11} on te[300:]. Anchors: CASPER-R 0.360/0.152 ; fullprof ~0.407/0.216.
set -u
PY="python C:/dev/phd/casper/scripts/paper2/continuous_actor.py"
CK="C:/dev/phd/casper/data/movielens/.cache"
LOG="C:/dev/phd/casper/experiments/paper2/snapA_2026-06-26.log"
EP=30
echo "=== ANSMASK SNAP actor (path A) | start $(date) ===" > "$LOG"
echo "### TRAIN snapA EP=$EP  $(date)" >> "$LOG"
CONTMODE=snap ANSMASK=1 NOBC=1 OBJ=ustar TAG=cont_snapA EP=$EP SKIPVAL=1 SEED=1 $PY 2>&1 | grep -E "ep[0-9]+ return|saved .*last" >> "$LOG"

echo "### STAGE A curve (SEED=1)  $(date)" >> "$LOG"
for E in 2 4 6 9 12 16 20 24 30; do
  CONTMODE=snap ANSMASK=1 NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_snapA_ep${E}.pt MODES=contactor TESTRANGE=300:99999 SEED=1 $PY 2>&1 | grep -E "contactor:" | sed "s/^/[snapA ep$E s1] /" >> "$LOG"
done

echo "### STAGE B seed-avg {1,2,3,7,11}  $(date)" >> "$LOG"
for E in 6 12 20 30; do
  for s in 1 2 3 7 11; do
    CONTMODE=snap ANSMASK=1 NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_snapA_ep${E}.pt MODES=contactor TESTRANGE=300:99999 SEED=$s $PY 2>&1 | grep -E "contactor:" | sed "s/^/[snapA ep$E s$s] /" >> "$LOG"
  done
done

echo "### REFERENCE conc_pop + fullprof (seed-avg)  $(date)" >> "$LOG"
for s in 1 2 3 7 11; do
  CONTMODE=snap NOBC=1 NOTRAIN=1 LOADCK=$CK/policy_cont_snapA_ep2.pt MODES=conc_pop,fullprof TESTRANGE=300:99999 SEED=$s $PY 2>&1 | grep -E "conc_pop:|fullprof :" | sed "s/^/[ref s$s] /" >> "$LOG"
done
echo "=== DONE $(date) ===" >> "$LOG"
