#!/bin/bash
# HARMONIZED v2: 5 held-out seeds {1,2,3,7,11} (drop dev-seed 123), MRR (full/tail) instead of RMSE,
# full-profile ceiling, lit heuristics (random/helf/logpop_ent/pop_ent) + popularity/entropy 3x2 grid,
# recipe ablations (BC floor; +reconstruction=ours; +RL-on-NDCG; +SL-ranking-BCE; reconstruction-from-scratch).
# Oracle/dbf/poracle DROPPED. q-curve points for saturation figure.
cd "C:/dev/phd/casper" || exit 1
LOG=experiments/paper2/harm2.log; CSV=data/movielens/.cache/evalcks_harm2.csv
echo "===== TRAIN ablations $(date '+%H:%M:%S') =====" > "$LOG"
# entropy-pretrained + RL on (tail) NDCG reward
env SKIPVAL=1 FEATS=ext,ans BCFROM=entdistill_bc OBJ=reinforce REW=ndcg REWTAIL=1 TAG=entrl EP=8 python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
# entropy-pretrained + SL ranking objective (BCE)
env SKIPVAL=1 FEATS=ext,ans BCFROM=entdistill_bc OBJ=bce TAG=bce_ent EP=10 python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
echo "===== HARMONIZED EVAL v2 (5 held-out seeds; MRR; lit heuristics; ceiling) $(date '+%H:%M:%S') =====" >> "$LOG"
rm -f "$CSV"
env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,1,2,4,8 EVALSEEDS=1,2,3,7,11 TESTRANGE=300:99999 \
    EVALBASE=fullprof,entropy,entropy_item,entropy_uni,conc_pop,pop_item,mix4,random,helf,logpop_ent,pop_ent \
    EVALCKS=entdistill_ep4,entdistill_bc,prof_scr_ep9,entrl_ep3,entrl_ep5,entrl_ep7,bce_ent_ep4,bce_ent_ep7,bce_ent_ep10 \
    EVALCSV="$CSV" python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
echo "===== AGG $(date '+%H:%M:%S') =====" >> "$LOG"
PYTHONIOENCODING=utf-8 python scripts/paper2/evalcks_agg_full.py "$CSV" >> "$LOG" 2>&1
echo "HARM2_DONE" >> "$LOG"
