#!/bin/bash
# Rerun the key Paper-B/C policy experiments on V2-A (GRU+attn on V1 recipe), binary answers (Paper-A/B convention).
# Baseline to beat on V2-A: entropy item 0.345/0.135, conc 0.340/0.107 (from the gate, seed 1).
cd C:/dev/phd/casper
REC=data/movielens/.cache/enc_concept_gru.pt
LOG=experiments/paper2/v2a_grid_$(date +%H%M).log
echo "=== V2-A policy grid | rec=enc_concept_gru | baseline entropy 0.345/0.135 ===" | tee $LOG
run(){ echo "--- $1 ---" | tee -a $LOG; shift; env LOADREC=$REC "$@" NOTRAIN=0 SEED=1 python scripts/paper2/continuous_actor.py 2>&1 | grep -E "contactor|ORADISTILL ep40|ep14 |best-val|init TEST" | tail -3 | tee -a $LOG; }

# 1) distilled CONCEPT oracle (CASPER-R-style selection)
run "concept-oracle-distill"  ORADISTILL=1 MODES=contactor COSSNAP=1 ANSMASK=1 DEMON=800 DISTEP=40 TAG=v2a_concorc NOBC=1
# 2) distilled RECON oracle (reconstruct u*)
run "recon-oracle-distill"    ORADISTILL=1 RECON=1 GRAW=1 ITEMSANS=1 MODES=contactor COSSNAP=1 ANSMASK=1 DEMON=800 DISTEP=40 TAG=v2a_reconorc NOBC=1
# 3) actor, reconstruct KNOWN profile, from SCRATCH
run "ustar-known-scratch"     OBJ=ustar NOBC=1 EP=14 TAG=v2a_ustar_scr MODES=contactor COSSNAP=1
# 4) actor, reconstruct KNOWN profile, BC-floor fine-tuned
run "ustar-known-bc"          OBJ=ustar EP=14 TAG=v2a_ustar_bc MODES=contactor COSSNAP=1
# 5) actor, reconstruct HELD-OUT profile, from scratch
run "ustar-held-scratch"      OBJ=ustar USTARHELD=1 NOBC=1 EP=14 TAG=v2a_held_scr MODES=contactor COSSNAP=1
# 6) actor, direct NDCG (reinforce), from scratch
run "ndcg-reinforce-scratch"  OBJ=reinforce REW=ndcg NOBC=1 EP=14 TAG=v2a_ndcg_scr MODES=contactor COSSNAP=1
echo "=== DONE ===" | tee -a $LOG; echo "LOG: $LOG"
