#!/bin/bash
# HARMONIZED seed-averaged eval on te[300:] (PAPER test), ALL metrics: NDCG full/tail, Rec full/tail, RMSE, cos-to-u*, ans.
# ANSWERABILITY COUNTERFACTUAL (criterion held = divisiveness, pool varies):
#   entropy(concepts) vs entropy_item(items) vs entropy_uni(both) -> if items crater, answerability is the cause.
# Plus popular baselines (conc_pop, pop_item) and our policies (entdistill_ep4 winner, dbf_uni answerability variant).
cd "C:/dev/phd/casper" || exit 1
CSV=data/movielens/.cache/evalcks_harmonized.csv
LOG=experiments/paper2/harmonized_eval.log
rm -f "$CSV"
echo "===== HARMONIZED EVAL START =====" > "$LOG"
env NOBC=1 NOTRAIN=1 FEATS=ext,ans QPTS=0,2,4,8 \
    EVALSEEDS=123,1,2,3,7,11 TESTRANGE=300:99999 \
    EVALBASE=entropy,entropy_item,entropy_uni,conc_pop,pop_item \
    EVALCKS=entdistill_ep4,dbf_uni \
    EVALCSV="$CSV" \
    python scripts/paper2/continuous_policy2.py >> "$LOG" 2>&1
echo "===== AGGREGATE =====" >> "$LOG"
python scripts/paper2/evalcks_agg_full.py "$CSV" >> "$LOG" 2>&1
echo "HARMONIZED_DONE" >> "$LOG"
