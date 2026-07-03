"""
ML-25M PHASE-1B resumable biased-SVD trainer. Loads prep.npz, runs EPOCHS_THIS_RUN epochs of the
identical Koren09 mini-batch SGD (D=64 LAMF=0.05 LR=0.01, total EP=15), checkpointing state to
.cache/ml25m/svd_state.npz so it can be run in foreground chunks. Writes Q_svd.npy/bi_svd.npy each run.
"""
import os, time, numpy as np
out='C:/dev/phd/casper/data/movielens/.cache/ml25m'
D=64; LAMF=0.05; LR=0.01; EP_TOTAL=15
EPR=int(os.environ.get('EPOCHS_THIS_RUN',3))
t0=time.time()
P=np.load(f'{out}/prep.npz')
ruD=P['ruD']; ri=P['ri']; rr=P['rr']; mu=float(P['mu']); ntr=int(P['ntr']); ni=int(P['ni'])
sp=f'{out}/svd_state.npz'
if os.path.exists(sp):
    S=np.load(sp); Pm=S['P']; Qm=S['Q']; bu=S['bu']; bi=S['bi']; done=int(S['done'])
    rng=np.random.default_rng(1000+done)
    print(f"resume: {done}/{EP_TOTAL} epochs done",flush=True)
else:
    rng=np.random.default_rng(0)
    bu=np.zeros(ntr,np.float32); bi=np.zeros(ni,np.float32)
    Pm=(0.1*rng.standard_normal((ntr,D))).astype(np.float32); Qm=(0.1*rng.standard_normal((ni,D))).astype(np.float32)
    done=0; print("fresh init",flush=True)
idx=np.arange(len(rr))
target=min(done+EPR,EP_TOTAL)
for ep in range(done,target):
    rng.shuffle(idx)
    for b0 in range(0,len(idx),16384):
        bidx=idx[b0:b0+16384]; us=ruD[bidx]; it=ri[bidx]
        pred=mu+bu[us]+bi[it]+np.sum(Pm[us]*Qm[it],1); e=(rr[bidx]-pred).astype(np.float32)
        np.add.at(bu,us,LR*(e-LAMF*bu[us])); np.add.at(bi,it,LR*(e-LAMF*bi[it]))
        gP=LR*(e[:,None]*Qm[it]-LAMF*Pm[us]); gQ=LR*(e[:,None]*Pm[us]-LAMF*Qm[it])
        np.add.at(Pm,us,gP); np.add.at(Qm,it,gQ)
    trrmse=np.sqrt(np.mean((rr-(mu+bu[ruD]+bi[ri]+np.sum(Pm[ruD]*Qm[ri],1)))**2))
    print(f"  ep{ep+1}/{EP_TOTAL} train RMSE={trrmse:.4f} ({time.time()-t0:.0f}s)",flush=True)
np.savez(sp,P=Pm,Q=Qm,bu=bu,bi=bi,done=target)
np.save(f'{out}/Q_svd.npy',Qm); np.save(f'{out}/bi_svd.npy',bi)
print(f"SAVED state ({target}/{EP_TOTAL}) + Q_svd.npy bi_svd.npy ({time.time()-t0:.0f}s)",flush=True)
print("DONE" if target>=EP_TOTAL else f"MORE ({EP_TOTAL-target} epochs left)",flush=True)
