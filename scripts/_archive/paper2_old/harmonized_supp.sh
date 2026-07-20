#!/bin/bash
# SUPPLEMENTARY baselines/ablations APPENDED to evalcks_harmonized.csv (does NOT rm the CSV).
# Adds: interleaved item+concept askers (mix2=25% items, mix4=50%); the PEAK answerability variant
# (dbf_uni_best/ep7, NOT the drifted latest); the oracle-distill+RL ablation (poracle_reg_best);
# the entropy-clone BC floor (entdistill_bc, should ~= entropy); entdistill epoch-robustness (ep3,ep5);
# and the privileged unified tail-oracle CEILING (2 seeds, slow lookahead).
cd "C:/dev/phd/casper" || exit 1
CSV=data/movielens/.cache/evalcks_harmonized.csv
LOG=experiments/paper2/harmonized_eval.log
echo "===== SUPP (extra baselines + peak ablations) $(date '+%H:%M:%S') =====" >> "$LOG"
env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,2,4,8 \
    EVALSEEDS=123,1,2,3,7,11 TESTRANGE=300:99999 \
    EVALBASE=mix2,mix4 \
    EVALCKS=dbf_uni_best,dbf_uni_ep7,poracle_reg_best,entdistill_bc,entdistill_ep3,entdistill_ep5,prof_scr_ep3,prof_scr_ep6,prof_scr_ep9 \
    EVALCSV="$CSV" \
    python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
echo "===== SUPP unified tail-oracle CEILING (2 seeds) $(date '+%H:%M:%S') =====" >> "$LOG"
env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,2,4,8 REWTAIL=1 \
    EVALSEEDS=123,1 TESTRANGE=300:99999 \
    EVALBASE=conc_oracle \
    EVALCSV="$CSV" \
    python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
echo "===== AGGREGATE (combined harmonized table) $(date '+%H:%M:%S') =====" >> "$LOG"
python scripts/paper2/evalcks_agg_full.py "$CSV" >> "$LOG" 2>&1
echo "SUPP_DONE" >> "$LOG"
