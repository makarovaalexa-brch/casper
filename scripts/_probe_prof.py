import os, sys, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arena_core as A

ar = A.Arena(verbose=False)
coh = A.make_cohorts(ar, n_train=60, n_devval=5, n_devtest=5)
recs = coh["train"][:40]
# warm
ar._gen_user(recs[0]["u"], recs[0]["known"])
pr = cProfile.Profile()
pr.enable()
for r in recs[1:36]:
    ar._gen_user(r["u"], r["known"])
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(20)
print(s.getvalue())
print("PROF done", flush=True)
