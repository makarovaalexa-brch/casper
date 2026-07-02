cd /c/dev/phd/casper
for ts in 1 2; do
  echo "###### PAIRTRAIN d1warm TRSEED=$ts ######"
  PAIRTRAIN=1 TRSEED=$ts INIT=data/movielens/.cache/policy_phase3_d1divw_last.pt DIVW=1.0 DTAU=2.0 NUMAT=2500 CONTMODE=cont GRADED=1 OBJ=ustar NOBC=1 FEATS=ext,ans EP=10 SELVAL=tail TAG=pairtrain_d1warm_ts$ts QPTS=0 MODES=pop_item python -u scripts/paper2/continuous_actor.py 2>&1 | grep -E "INIT|ep[0-9]+ return|pair-realization|BEST|saved|Traceback|Error"
  echo "--- eval ts$ts (best-val ckpt, canonical 5 seeds) ---"
  NOBC=1 EP=0 PAIRSNAP=1 EVALSEEDS=1,2,3,7,11 PAIRCHECK=0 CONTMODE=cont FEATS=ext,ans ANSF=1 ACTORCK=data/movielens/.cache/policy_pairtrain_d1warm_ts${ts}_best.pt python -u scripts/paper2/continuous_actor.py 2>&1 | grep -E "PAIRSNAP|seed|PAIR-SNAP|realization|Traceback"
done
echo "--- eval from-scratch ts0 best (diagnostic row) ---"
NOBC=1 EP=0 PAIRSNAP=1 EVALSEEDS=1,2,3,7,11 PAIRCHECK=0 CONTMODE=cont FEATS=ext,ans ANSF=1 ACTORCK=data/movielens/.cache/policy_pairtrain_ts0_best.pt python -u scripts/paper2/continuous_actor.py 2>&1 | grep -E "PAIRSNAP|seed|PAIR-SNAP|realization|Traceback"
echo ALLDONE_PAIRTRAIN_TS12
