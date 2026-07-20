import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import arena_core as A

t0 = time.time()
ar = A.Arena(verbose=True)
print(f"[probe] arena init {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
coh = A.make_cohorts(ar, n_train=200, n_devval=10, n_devtest=20)
print(f"[probe] make_cohorts(230) {time.time()-t0:.1f}s; train={len(coh['train'])}", flush=True)

# time per-user answer-table gen on 30 users
recs = coh["train"][:30]
t0 = time.time()
for r in recs:
    ar.user_table(r["u"], r["known"])
dt = time.time() - t0
print(f"[probe] gen 30 user tables {dt:.1f}s = {dt/30*1000:.0f} ms/user", flush=True)
print(f"[probe] EXTRAPOLATE 155k users = {dt/30*155000/3600:.1f} hours", flush=True)

# inspect one table + member-bag sizes
r = recs[0]; t = ar.user_table(r["u"], r["known"])
print(f"[probe] table keys {list(t.keys())} know shape {t['know'].shape} val {t['val'].shape} crval {t['crval'].shape}", flush=True)
print(f"[probe] answered rate {(t['know']>=1).mean():.3f} refusal rate {(t['know']==0).mean():.3f}", flush=True)
# member bag sizes
sz_tag = np.asarray(ar.uni.tagM.sum(1)).ravel()
sz_ent = np.asarray(ar.uni.entM.sum(1)).ravel()
print(f"[probe] concept members: min{sz_tag.min():.0f} med{np.median(sz_tag):.0f} max{sz_tag.max():.0f}", flush=True)
print(f"[probe] entity members: min{sz_ent.min():.0f} med{np.median(sz_ent):.0f} max{sz_ent.max():.0f}", flush=True)
print(f"[probe] ni={ar.uni.ni} nbank={ar.nbank} ntag={ar.ntag} nent={ar.nent} nQ={ar.nQ}", flush=True)
print("PROBE done", flush=True)
