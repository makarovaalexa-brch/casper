#!/bin/bash
# Overnight gap-chasing retrains. Each = RL-only from the fair-entropy BC floor (BCFROM=entdistill_bc),
# ONE variable changed, then seed-eval ep3-10 vs entropy. Logs to overnight.log + evalcks_overnight.csv.
# Launch when the box is free (sweep done). Aggregate in the morning: python scripts/paper2/evalcks_agg.py <csv>
cd "C:/dev/phd/casper" || exit 1
LOG=experiments/paper2/overnight.log
CSV=data/movielens/.cache/evalcks_overnight.csv
SEEDS=123,1,2,3,7,11
ts() { date '+%H:%M:%S'; }
eveps() { seq -s, -f "${1}_ep%g" 3 "${2:-10}"; }

train_eval() {            # $1=TAG $2=EP $3..=one-variable overrides
  local TAG=$1 EP=$2; shift 2
  echo "===== $(ts) TRAIN $TAG EP=$EP : $* =====" >> "$LOG"
  env OBJ=reinforce REW=dbf REWTAIL=1 SKIPVAL=1 FEATS=ext,ans "$@" TAG="$TAG" EP="$EP" BCFROM=entdistill_bc \
      python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
  echo "===== $(ts) EVAL  $TAG (seeds) =====" >> "$LOG"
  env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,2,4,8 EVALSEEDS="$SEEDS" EVALCSV="$CSV" \
      EVALCKS="$(eveps "$TAG" "$EP")" python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
  echo "===== $(ts) DONE  $TAG =====" >> "$LOG"
}

echo "===== $(ts) OVERNIGHT CAMPAIGN v2 (OBJ=reinforce FIX) START =====" >> "$LOG"
train_eval rbase    8                             # entdistill REPLICA (OBJ=reinforce defaults) -- SANITY GATE: must match ~0.152 tail@q8 before trusting ablations
train_eval rpen0    8 PEN=0                        # answerability CONFOUNDER ablation (does the win survive w/o the penalty?)
train_eval rpenhi   8 PEN=0.08                     # push answerability harder (bigger early tail)
train_eval rfulleff 8 REW=ndcg REWTAIL=0           # maximize low-q EFFICIENCY gap (full objective)
echo "===== $(ts) CAMPAIGN COMPLETE =====" >> "$LOG"
python scripts/paper2/evalcks_agg.py "$CSV" >> "$LOG" 2>&1
echo "===== $(ts) AGG DONE =====" >> "$LOG"
