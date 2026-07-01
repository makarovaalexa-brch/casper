#!/bin/bash
# REWARD sweep incl. PEN=0 (owner: "did you experiment with NOT penalising unanswerable?").
# REINFORCE from the entropy BC floor, vary {reward, PEN}. Isolates the unanswerable-penalty effect
# (rbase = REW=dbf PEN=0.02 -> 0.126; this adds the clean PEN=0 + alt rewards). Seed-eval each.
# Run AFTER the oracle engine. Aggregate: evalcks_agg.py evalcks_reward.csv
cd "C:/dev/phd/casper" || exit 1
LOG=experiments/paper2/reward_sweep.log
CSV=data/movielens/.cache/evalcks_reward.csv
SEEDS=123,1,2,3,7,11
ts() { date '+%H:%M:%S'; }
seval() { env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,2,4,8 EVALSEEDS="$SEEDS" EVALCSV="$CSV" EVALCKS="$1" python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1; }
te() {  # TAG EP overrides...
  local TAG=$1 EP=$2; shift 2
  echo "===== $(ts) TRAIN $TAG : $* =====" >> "$LOG"
  env OBJ=reinforce SKIPVAL=1 FEATS=ext,ans "$@" TAG="$TAG" EP="$EP" BCFROM=entdistill_bc python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
  echo "===== $(ts) EVAL $TAG =====" >> "$LOG"; seval "$(seq -s, -f "${TAG}_ep%g" 3 "$EP")"
}
rm -f "$CSV"; echo "===== $(ts) REWARD SWEEP START =====" >> "$LOG"
te rew_dbf_p0     8 REW=dbf  REWTAIL=1 PEN=0       # dbf, NO penalty (vs rbase dbf PEN=0.02=0.126) -> isolates PEN
te rew_ndcg_p0    8 REW=ndcg REWTAIL=1 PEN=0       # direct tail-NDCG, NO penalty
te rew_ndcg_p02   8 REW=ndcg REWTAIL=1 PEN=0.02    # direct tail-NDCG, WITH penalty (the other half of the isolation)
te rew_cov_p0     8 REW=cov            PEN=0       # coverage reward, NO penalty
echo "===== $(ts) REWARD SWEEP COMPLETE =====" >> "$LOG"
python scripts/paper2/evalcks_agg.py "$CSV" >> "$LOG" 2>&1
