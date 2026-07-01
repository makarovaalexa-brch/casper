#!/bin/bash
# RECONSTRUCTION OBJECTIVE x PRETRAINING matrix, evaluated on VAL (te[:300], seed-avg).
# OBJECTIVE: prof (fold of KNOWN profile) | held (fold of HELD-OUT likes = NDCG target) | bce (ranking score predicts held-out likes, inv-pop weighted)
# PRETRAINING: scratch (NOBC=pure SL from random) vs entropy-floor (BCFROM=entdistill_bc)
# Answers (1) which reconstruction target wins on val, (2) does BC pretraining help when the objective is SL.
# Pre-existing cells reused: entdistill (prof/entropy), rheld (held/entropy, finishing), roracle_ustar (prof/ORACLE-floor).
cd "C:/dev/phd/casper" || exit 1
LOG=experiments/paper2/recon_matrix.log
CSV=data/movielens/.cache/evalcks_reconmatrix.csv
SEEDS=123,1,2,3,7,11
ts(){ date '+%H:%M:%S'; }
seval(){ env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,4,8 EVALSEEDS="$SEEDS" TESTRANGE=0:300 EVALCSV="$CSV" EVALCKS="$1" python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1; }   # VAL=te[:300]; entropy+conc_pop auto-included each call
te(){ local TAG=$1 EP=$2; shift 2; echo "===== $(ts) TRAIN $TAG : $* =====" >> "$LOG"; env SKIPVAL=1 FEATS=ext,ans "$@" TAG="$TAG" EP="$EP" python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1; echo "===== $(ts) EVAL $TAG (val) =====" >> "$LOG"; seval "$(seq -s, -f "${TAG}_ep%g" 3 "$EP")"; }
rm -f "$CSV"; echo "===== $(ts) RECON MATRIX START =====" >> "$LOG"
te bce_ent  10 OBJ=bce               BCFROM=entdistill_bc    # ranking objective, entropy floor
te prof_scr 12 OBJ=ustar             NOBC=1                  # known-profile, SCRATCH (SL, no pretraining)
te held_scr 12 OBJ=ustar USTARHELD=1 NOBC=1                  # held-out target, SCRATCH
te bce_scr  12 OBJ=bce               NOBC=1                  # ranking objective, SCRATCH
echo "===== $(ts) EVAL pre-trained cells on val =====" >> "$LOG"
seval entdistill_ep3,entdistill_ep4,entdistill_ep5,entdistill_ep6                  # prof / entropy-floor
seval rheld_ep3,rheld_ep4,rheld_ep5,rheld_ep6,rheld_ep7,rheld_ep8                  # held / entropy-floor
seval roracle_ustar_ep3,roracle_ustar_ep5,roracle_ustar_ep7,roracle_ustar_ep9     # prof / ORACLE-floor (teacher axis)
echo "===== $(ts) RECON MATRIX COMPLETE =====" >> "$LOG"
python scripts/paper2/evalcks_agg.py "$CSV" >> "$LOG" 2>&1
