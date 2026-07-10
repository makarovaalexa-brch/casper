"""Fast end-to-end validation of the GATED arena (every policy + world fuel machinery)."""
import os, sys, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings; warnings.filterwarnings("ignore")
import arena_core as AC
import arena_policies as AP
from arena_policies import (StaticSeq, B0Cold, B4Myopic, TrueTableRouter, Clairvoyant, AskGradient,
                            CATRouter, GolbandiTree, ScorerA, B2Anchored, run_policy)

ar = AC.Arena()
coh = AC.make_cohorts(ar, n_train=40, n_devval=12, n_devtest=15)
tr, dv, dt = coh["train"], coh["devval"], coh["devtest"]
cfg = dict(coh["cfg"]); cfg["val_run"] = 1
ar.prefill_answers(tr, "vtr", cfg, verbose=False)
ar.prefill_answers(dt, "vdt", cfg, verbose=False)
ar.set_pop_prior(AP.train_pop_prior(ar, tr))
print("[val] gated cohorts + tables + prior ok", flush=True)

b1 = AP.build_b1(ar, tr, verbose=False)
b2_seq, _ = AP.build_b2(ar, tr, Tmax=6, prescreen_top=60, verbose=True, tag="_val")
tail = AP.b3_tail(ar, tr, b2_seq, tag="_val")
gol_tree = AP.build_golbandi(ar, tr, max_depth=2, min_users=6, cand_cap=60, verbose=True, tag="_val")
gbm = AP.build_scorerA(ar, tr, n_samples=150, M2=6, verbose=True, tag="_val")
print("[val] constructions ok", flush=True)

Tmax = 6; KS = (50, 10)
def check(name, pol, skip=False):
    t0 = time.time()
    o = run_policy(ar, dt, pol, Tmax, Ks=KS, skip=skip, tag=name)
    e = np.nanmean(o["curves"][50][:, Tmax])
    print(f"[val] {name:8s} ok  endpoint@50={e:.4f}  [{time.time()-t0:.1f}s]", flush=True)

check("b0", B0Cold())
check("b1", StaticSeq("b1", b1))
check("b2", StaticSeq("b2", b2_seq))
check("b3", StaticSeq("b3", b2_seq, tail=tail), skip=True)
check("b4", B4Myopic(M=60))
check("A", B2Anchored("A", b2_seq, ScorerA(gbm, M=60, Tmax=Tmax), 2))
check("B", B2Anchored("B", b2_seq, AskGradient(M=60), 0))
check("C", B2Anchored("C", b2_seq, CATRouter(M=60), 2))
check("D", GolbandiTree(gol_tree, b2_seq, tail))
check("ttab", TrueTableRouter(M=60))
check("clair", Clairvoyant(M=100))
# fuel check machinery guard (needs >=400 users; here just exercise the assert)
try:
    AC.world_fuel_check(ar, tr)
except AssertionError as e:
    print(f"[val] fuel-check guard ok (needs >=400): {e}", flush=True)
print("[val] ALL POLICIES RAN CLEAN", flush=True)
