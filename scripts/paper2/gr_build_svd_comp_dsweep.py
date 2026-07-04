"""
Goodreads COMPOSITE biased-SVD at arbitrary dimension D (dimension sweep).
Same recipe as gr_build_svd_comp.py (LAMF=0.05 LR=0.01 EP=15, rng(0)) apart from D.
Saves Q_svd_comp_d{D}.npy + bi_svd_comp_d{D}.npy and a peak/RMSE log.
Epoch-level checkpoint/resume: _dsweep_ckpt_d{D}.npz holds P,Q,bu,bi,ep so a
foreground chunk can stop and the next call resumes. Set MAXEP_CHUNK to cap
epochs per invocation.
"""
import os, time, numpy as np
GR='C:/dev/phd/casper/.cache/goodreads'
D=int(os.environ['D']); LAMF=0.05; LR=0.01; EP=15
MAXEP=int(os.environ.get('MAXEP_CHUNK',15))   # epochs to run THIS invocation
rng_seed=0
t0=time.time()
B=np.load(f'{GR}/base_comp.npz')
uu=B['uu']; ii=B['ii']; R=B['rr']; ni=int(B['ni']); nu=int(B['nu']); trU=B['trU']; mu=float(B['mu'])
trU_mask=np.zeros(nu,bool); trU_mask[trU]=True; tr_row=trU_mask[uu]
ru=uu[tr_row]; ri=ii[tr_row]; rr=R[tr_row].astype(np.float32)
truniq=np.unique(ru); trmap=np.zeros(nu,np.int32); trmap[truniq]=np.arange(len(truniq)); ruD=trmap[ru]; ntr=len(truniq)
print(f"[D={D}] train ratings={len(rr)} ntr={ntr} ni={ni} mu={mu:.3f} ({time.time()-t0:.0f}s)",flush=True)

ckpt=f'{GR}/_dsweep_ckpt_d{D}.npz'
if os.path.exists(ckpt):
    Z=np.load(ckpt)
    P=Z['P']; Q=Z['Q']; bu=Z['bu']; bi=Z['bi']; start_ep=int(Z['ep'])
    rng=np.random.default_rng(int(Z['rng_state_ep'])+1000)  # fresh but deterministic per resume
    print(f"[resume] loaded ckpt at ep={start_ep} ({time.time()-t0:.0f}s)",flush=True)
else:
    rng=np.random.default_rng(rng_seed)
    bu=np.zeros(ntr,np.float32); bi=np.zeros(ni,np.float32)
    P=(0.1*rng.standard_normal((ntr,D))).astype(np.float32); Q=(0.1*rng.standard_normal((ni,D))).astype(np.float32)
    start_ep=0
idx=np.arange(len(rr))
def train_rmse():
    se=0.0
    for c0 in range(0,len(rr),2_000_000):
        s=slice(c0,c0+2_000_000)
        pr=mu+bu[ruD[s]]+bi[ri[s]]+np.sum(P[ruD[s]]*Q[ri[s]],1)
        se+=float(np.sum((rr[s]-pr)**2))
    return np.sqrt(se/len(rr))
ran=0
for ep in range(start_ep,EP):
    if ran>=MAXEP:
        print(f"[chunk] reached MAXEP_CHUNK={MAXEP}; stopping at ep={ep} (resume next call)",flush=True)
        break
    rng.shuffle(idx)
    for b0 in range(0,len(idx),16384):
        bidx=idx[b0:b0+16384]; us=ruD[bidx]; it=ri[bidx]
        pred=mu+bu[us]+bi[it]+np.sum(P[us]*Q[it],1); e=(rr[bidx]-pred).astype(np.float32)
        np.add.at(bu,us,LR*(e-LAMF*bu[us])); np.add.at(bi,it,LR*(e-LAMF*bi[it]))
        gP=LR*(e[:,None]*Q[it]-LAMF*P[us]); gQ=LR*(e[:,None]*P[us]-LAMF*Q[it]); np.add.at(P,us,gP); np.add.at(Q,it,gQ)
    ran+=1
    tr=train_rmse()
    print(f"  ep{ep+1} train RMSE={tr:.4f} ({time.time()-t0:.0f}s)",flush=True)
    np.savez(ckpt,P=P,Q=Q,bu=bu,bi=bi,ep=ep+1,rng_state_ep=ep+1)

done = (start_ep+ran)>=EP or (os.path.exists(ckpt) and False)
final_ep=start_ep+ran
if final_ep>=EP:
    tr=train_rmse()
    np.save(f'{GR}/Q_svd_comp_d{D}.npy',Q); np.save(f'{GR}/bi_svd_comp_d{D}.npy',bi)
    with open(f'{GR}/Q_svd_comp_d{D}_peak.txt','w') as f:
        f.write(f"D={D} LAMF={LAMF} LR={LR} EP={EP} final_train_RMSE={tr:.4f} ntr={ntr} ni={ni}\n")
    if os.path.exists(ckpt): os.remove(ckpt)
    print(f"[D={D}] SAVED Q_svd_comp_d{D}.npy final train RMSE={tr:.4f} ({time.time()-t0:.0f}s)",flush=True)
    print("DONE",flush=True)
else:
    print(f"[D={D}] PARTIAL through ep={final_ep}/{EP}; rerun to continue ({time.time()-t0:.0f}s)",flush=True)
    print("PARTIAL",flush=True)
