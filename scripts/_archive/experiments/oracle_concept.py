"""CEILING test: in a FEW-question interview, can concepts help IF selected adaptively (oracle)?
For each user, oracle-pick the single concept whose whitened-alignment most lifts their HELD likes (leaky =
upper bound on 1-concept adaptivity), fold it on popb; greedy-add for 2,3. Compare to popularity intercept and
STATIC top-WC selection. If oracle-1 >> intercept, low-kc weakness is a SELECTION problem (buildable policy).
If oracle-1 also hurts, concepts are fundamentally too coarse for few-shot.
"""
import os,sys; sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from set_mn import RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m):
    import time; print('['+time.strftime('%H:%M:%S')+'] '+m, flush=True)
S.set_grading('ordinal'); NC=S.NC; OUT='.cache/set_mn'; LO=4.0
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']; Wd=pa['decoder']['weight'].numpy().astype(np.float32)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
d=np.load('data/movielens/.cache/ml25m/meta.npz'); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cd=np.zeros((NC,512),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cd[c]=v/n if n>0 else 0
ALIGN=(Wd@Cd.T).astype(np.float64)   # (ni x NC)
usable=(csz>=20)
popb=np.log(cnt.astype(np.float64)+1.0)
KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy'); uidsV=np.load(RSD+'/mm_val_uids.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3); GMAP={3:1.,2:.5,1:0.,0:-1.}
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,hi,hr=its[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    cc=np.where(ANSV[r]&usable)[0]; g=np.array([GMAP[int(CVALV[r,c])] for c in cc]); nz=g!=0
    if not nz.any(): continue
    US.append((hl.astype(np.int64), cc[nz], g[nz]))
log('val users %d'%len(US))
def tm(ps): t=np.array([p for p in ps if p is not None]); return t.mean()
BETA=float(os.environ.get('BETA','8'))
pt=tm([ndcg10(popb.copy(),list(hl),set(),head,True) for hl,cc,g in US]); pf=tm([ndcg10(popb.copy(),list(hl),set(),head,False) for hl,cc,g in US])
log('popularity: FULL %.4f TAIL %.4f (beta=%.0f)'%(pf,pt,BETA))
# oracle greedy: at each step pick concept maximizing held-like alignment lift; fold cumulatively
for STEP,label in [(1,'oracle-1'),(2,'oracle-2'),(3,'oracle-3')]:
    ft=[]; ff=[]
    for hl,cc,g in US:
        chosen=[]; cur=np.zeros(ni)
        pool=list(range(len(cc)))
        for s in range(STEP):
            best=None; bestv=-1e9
            for j in pool:
                lift=ALIGN[hl,cc[j]].mean()*g[j]   # signed lift on held-likes
                if lift>bestv: bestv=lift; best=j
            chosen.append(best); pool.remove(best)
            cur=cur+BETA*g[chosen[-1]]*ALIGN[:,cc[chosen[-1]]]
        sc=popb+cur
        ft.append(ndcg10(sc.copy(),list(hl),set(),head,True)); ff.append(ndcg10(sc.copy(),list(hl),set(),head,False))
    log('  %s: FULL %.4f (%+.4f)  TAIL %.4f (%+.4f)'%(label,tm(ff),tm(ff)-pf,tm(ft),tm(ft)-pt))
# static top-3 by member size (engagement proxy, non-oracle) for contrast
ft=[]
for hl,cc,g in US:
    order=np.argsort(-csz[cc])[:3]; cur=np.zeros(ni)
    for j in order: cur=cur+BETA*g[j]*ALIGN[:,cc[j]]
    ft.append(ndcg10((popb+cur).copy(),list(hl),set(),head,True))
log('  static-top3(size): TAIL %.4f (%+.4f)'%(tm(ft),tm(ft)-pt))
log('done')
