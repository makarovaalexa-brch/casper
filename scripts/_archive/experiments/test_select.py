"""REALIZABLE concept selection (no held-like leak). Simple geometric fold score = popb + beta*sum g_c*align_c.
Selectors (use only the elicited answer + concept stats):
  eng       : biggest membership (broad genres) -- the weak baseline
  loved_niche: among LOVED/LIKED concepts, smallest membership (specific taste the user endorsed)
  infogain  : highest global ANSWER-ENTROPY (polarizing concepts, from TRAIN population -- no leak)
  spec_ans  : |answer strength| * (1/log members)  (strong answer on a specific concept)
  oracle    : align with HELD likes (CEILING, leaky)
Report tail@10 at k=1,2,3.
"""
import os,sys
sys.path.insert(0,'scripts')
import numpy as np
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import set_mn as S, arena_core as AC
import torch
def log(m): print(m,flush=True)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; RS=".cache/rich_signal"; LO=4.0
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load('data/movielens/.cache/ml25m/meta.npz'); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy(); Wd=pa['decoder']['weight'].numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cw=np.zeros((NC,D),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cw[c]=v/n if n>0 else 0
ALIGN=(Wd@Cw.T.astype(np.float64))   # (ni x NC)
popb=np.log(cnt.astype(np.float64)+1.0)
# global answer-entropy per concept from TRAIN (no leak)
KT=np.load(RS+'/mm_train_know.npy'); VT=np.load(RS+'/mm_train_val.npy')
ansT=(KT[:,:NC]>=1)&(VT[:,:NC]>=0); valT=np.clip(VT[:,:NC],0,3)
ENT=np.zeros(NC)
for c in range(NC):
    m=ansT[:,c]
    if m.sum()>=50:
        h=np.bincount(valT[m,c],minlength=4).astype(float); p=h/h.sum(); p=p[p>0]; ENT[c]=-(p*np.log(p)).sum()
# val cohort
KV=np.load(RS+'/mm_val_know.npy'); VV=np.load(RS+'/mm_val_val.npy'); uidsV=np.load(RS+'/mm_val_uids.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3); GMAP={3:1.,2:.5,1:0.,0:-1.}; usable=(csz>=20)
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
log(f"val users {len(US)}  (concept answer-entropy from {ansT.shape[0]} train users)")
BETA=float(os.environ.get('BETA','8'))
def tm(ps): t=np.array([x for x in ps if x is not None]); return t.mean()
pt=tm([ndcg10(popb.copy(),list(hl),set(),head,True) for hl,cc,g in US]); log(f"popb tail {pt:.4f} (beta={BETA:.0f})")
def sel(cc,g,hl,how,k):
    if how=='eng': idx=np.argsort(-csz[cc])
    elif how=='loved_niche': lov=np.where(g>=0.5)[0];  idx=lov[np.argsort(csz[cc[lov]])] if len(lov) else np.argsort(csz[cc])
    elif how=='infogain': idx=np.argsort(-ENT[cc])
    elif how=='spec_ans': idx=np.argsort(-(np.abs(g)/np.log(csz[cc].clip(2))))
    elif how=='oracle': idx=np.argsort(-(ALIGN[hl][:,cc].mean(0)*g))
    return idx[:k]
for how in ['eng','infogain','loved_niche','spec_ans','oracle']:
    row=[]
    for k in [1,2,3]:
        ps=[]
        for hl,cc,g in US:
            s=sel(cc,g,hl,how,k); dd=BETA*(g[s][:,None]*ALIGN[:,cc[s]].T).sum(0)
            ps.append(ndcg10((popb+dd).copy(),list(hl),set(),head,True))
        row.append(tm(ps)-pt)
    log(f"  {how:>11}: k1 {row[0]:+.4f}  k2 {row[1]:+.4f}  k3 {row[2]:+.4f}")
log("done")
