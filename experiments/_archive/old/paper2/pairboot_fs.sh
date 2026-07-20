cd /c/dev/phd/casper
echo "###### PAIRBOOT paired per-user bootstrap (A=pairtrain ts0 best vs B=uent+GRAW) ######"
NOBC=1 EP=0 PAIRBOOT=1 EVALSEEDS=1,2,3,7,11 CONTMODE=cont FEATS=ext,ans ANSF=1 ACTORCK=data/movielens/.cache/policy_pairtrain_d1warm_best.pt python -u scripts/paper2/continuous_actor.py 2>&1 | grep -E "PAIRBOOT|seed|FULL|TAIL|Traceback|Error"
echo "###### PAIRFS field-rerank (div only, M=32) ######"
NOBC=1 EP=0 PAIRSNAP=1 PAIRFS=1 PAIRM=32 DIVW=1.0 DTAU=2.0 NUMAT=2500 EVALSEEDS=1,2,3,7,11 PAIRCHECK=0 CONTMODE=cont FEATS=ext,ans ANSF=1 ACTORCK=data/movielens/.cache/policy_pairtrain_d1warm_best.pt python -u scripts/paper2/continuous_actor.py 2>&1 | grep -E "PAIRFS|seed|PAIR-SNAP|realization|Traceback|Error"
echo "###### PAIRFS field-rerank (div+pop+rat, M=32) ######"
NOBC=1 EP=0 PAIRSNAP=1 PAIRFS=1 PAIRM=32 FSP=1.0 FSR=1.0 DIVW=1.0 DTAU=2.0 NUMAT=2500 EVALSEEDS=1,2,3,7,11 PAIRCHECK=0 CONTMODE=cont FEATS=ext,ans ANSF=1 ACTORCK=data/movielens/.cache/policy_pairtrain_d1warm_best.pt python -u scripts/paper2/continuous_actor.py 2>&1 | grep -E "PAIRFS|seed|PAIR-SNAP|realization|Traceback|Error"
echo ALLDONE_BOOT_FS
