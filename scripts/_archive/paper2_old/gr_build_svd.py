"""
Goodreads MTC: biased-SVD item factors (linear reference). Reads base.npz, trains biased SVD
(Koren09) mini-batch SGD on TRAIN-user explicit ratings. D=64 LAMF=0.05 LR=0.01 EP=15 —
identical recipe to ML-1M/ML-25M. Saves Q_svd.npy, bi_svd.npy.
"""
import os, time, numpy as np
GR='C:/dev/phd/casper/.cache/goodreads'
D=int(os.environ.get('D',64)); LAMF=float(os.environ.get('LAMF',0.05)); LR=float(os.environ.get('LR',0.01))
EP=int(os.environ.get('EP',15)); rng=np.random.default_rng(0); t0=time.time()
B=np.load(f'{GR}/base.npz')
uu=B['uu']; ii=B['ii']; R=B['rr']; ni=int(B['ni']); nu=int(B['nu']); trU=B['trU']; mu=float(B['mu'])
trU_mask=np.zeros(nu,bool); trU_mask[trU]=True; tr_row=trU_mask[uu]
ru=uu[tr_row]; ri=ii[tr_row]; rr=R[tr_row].astype(np.float32)
truniq=np.unique(ru); trmap=np.zeros(nu,np.int32); trmap[truniq]=np.arange(len(truniq)); ruD=trmap[ru]; ntr=len(truniq)
print(f"train ratings={len(rr)} ntr={ntr} ni={ni} mu={mu:.3f} ({time.time()-t0:.0f}s)",flush=True)
bu=np.zeros(ntr,np.float32); bi=np.zeros(ni,np.float32)
P=(0.1*rng.standard_normal((ntr,D))).astype(np.float32); Q=(0.1*rng.standard_normal((ni,D))).astype(np.float32)
idx=np.arange(len(rr))
for ep in range(EP):
    rng.shuffle(idx)
    for b0 in range(0,len(idx),16384):
        bidx=idx[b0:b0+16384]; us=ruD[bidx]; it=ri[bidx]
        pred=mu+bu[us]+bi[it]+np.sum(P[us]*Q[it],1); e=(rr[bidx]-pred).astype(np.float32)
        np.add.at(bu,us,LR*(e-LAMF*bu[us])); np.add.at(bi,it,LR*(e-LAMF*bi[it]))
        gP=LR*(e[:,None]*Q[it]-LAMF*P[us]); gQ=LR*(e[:,None]*P[us]-LAMF*Q[it]); np.add.at(P,us,gP); np.add.at(Q,it,gQ)
    if (ep+1)%5==0 or ep==0:
        tr=np.sqrt(np.mean((rr-(mu+bu[ruD]+bi[ri]+np.sum(P[ruD]*Q[ri],1)))**2))
        print(f"  ep{ep+1} train RMSE={tr:.4f} ({time.time()-t0:.0f}s)",flush=True)
np.save(f'{GR}/Q_svd.npy',Q); np.save(f'{GR}/bi_svd.npy',bi)
print(f"SAVED Q_svd.npy bi_svd.npy ({time.time()-t0:.0f}s)",flush=True)
print("DONE",flush=True)
