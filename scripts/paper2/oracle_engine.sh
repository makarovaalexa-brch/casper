#!/bin/bash
# ORACLE-DISTILL bigger-win attempt (owner preference): oracle BC floor + the WINNING ustar refinement.
# poracle used tail-RL refine (-> 0.140). Untried = oracle floor + OBJ=ustar (what made entropy-distill hit 0.152).
# Evals raw floors first (reference), then trains oracle+ustar variants, seed-evals each. Aggregate: evalcks_agg.py evalcks_oracle.csv
cd "C:/dev/phd/casper" || exit 1
LOG=experiments/paper2/oracle_engine.log
CSV=data/movielens/.cache/evalcks_oracle.csv
SEEDS=123,1,2,3,7,11
ts() { date '+%H:%M:%S'; }
seval() { env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,2,4,8 EVALSEEDS="$SEEDS" EVALCSV="$CSV" EVALCKS="$1" python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1; }
rm -f "$CSV"; echo "===== $(ts) ORACLE ENGINE START =====" >> "$LOG"
# (1) reference: raw oracle BC floors + the prior poracle winner
echo "===== $(ts) eval raw oracle floors =====" >> "$LOG"
seval poracle_bc,poracle_full_bc,poracle_reg_best,entdistill_ep4
# (2) oracle floor + ustar refine (the untried winning combo)
for SPEC in "roracle_ustar|poracle_bc" "roracle_full_ustar|poracle_full_bc"; do
  TAG="${SPEC%%|*}"; FLOOR="${SPEC#*|}"
  echo "===== $(ts) TRAIN $TAG (BCFROM=$FLOOR OBJ=ustar) =====" >> "$LOG"
  env OBJ=ustar SKIPVAL=1 FEATS=ext,ans TAG="$TAG" EP=12 BCFROM="$FLOOR" python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
  echo "===== $(ts) EVAL $TAG =====" >> "$LOG"
  seval "$(seq -s, -f "${TAG}_ep%g" 3 12)"
done
echo "===== $(ts) ORACLE ENGINE COMPLETE =====" >> "$LOG"
python scripts/paper2/evalcks_agg.py "$CSV" >> "$LOG" 2>&1
echo "===== $(ts) AGG DONE =====" >> "$LOG"
