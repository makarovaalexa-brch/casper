"""Fast end-to-end validation of every arena policy + report writer (no long b2 greedy)."""
import os, sys, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings; warnings.filterwarnings("ignore")
import arena_core as AC
import arena_policies as AP
from arena_policies import (StaticSeq, B0Cold, B4Myopic, TrueTableRouter, Clairvoyant, AskGradient,
                            CATRouter, GolbandiTree, ScorerA, B2Anchored, run_policy)

ar = AC.Arena(n_item_universe=800)
coh = AC.make_cohorts(ar, n_train=40, n_devval=12, n_devtest=15)
tr, dv, dt = coh["train"], coh["devval"], coh["devtest"]
ar.prefill_answers(tr, "vtr40", verbose=False)
ar.prefill_answers(dt, "vdt15", verbose=False)
print("[val] cohorts + prefill ok", flush=True)

# fake b2 seq = b1 concept-entropy order truncated (fast) so eval paths run without the long greedy
b1 = AP.build_b1(ar, tr, verbose=False)
b2_seq = b1[:24]
gol_tree, gol_tail = AP.build_golbandi(ar, tr, b2_seq, max_depth=3, min_users=8, cand_cap=120, verbose=True)
gbm = AP.build_scorerA(ar, tr, n_samples=200, M2=8, verbose=True)
print("[val] constructions ok", flush=True)

Tmax = 6; KS = (50, 10)
def check(name, pol, skip=False):
    t0 = time.time()
    o = run_policy(ar, dt, pol, Tmax, Ks=KS, skip=skip, tag=name)
    e = np.nanmean(o["curves"][50][:, Tmax])
    print(f"[val] {name:16s} ok  endpoint@50={e:.4f}  [{time.time()-t0:.1f}s]", flush=True)
    return o

check("b0", B0Cold())
check("b1", StaticSeq("b1", b1))
check("b2", StaticSeq("b2", b2_seq))
check("b3", StaticSeq("b3", b2_seq), skip=True)
check("b4", B4Myopic(M=60))
check("A", B2Anchored("A", b2_seq, ScorerA(gbm, M=80, Tmax=Tmax), 2))
check("B", B2Anchored("B", b2_seq, AskGradient(M=80), 0))
check("C", B2Anchored("C", b2_seq, CATRouter(M=80), 2))
check("D", GolbandiTree(gol_tree, gol_tail))
check("ttab", TrueTableRouter(M=60))
check("clair", Clairvoyant())
print("[val] ALL POLICIES RAN CLEAN", flush=True)
# cleanup validation caches
for f in ["answers_vtr40.npz", "answers_vdt15.npz"]:
    p = f".cache/arena/{f}"
    if os.path.exists(p): os.remove(p)
