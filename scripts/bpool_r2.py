"""R2 DECOMPOSITION (Fable step 2, decisive): how much of the oracle affinity <u*,d_c> does each MODEL-FREE
item-derived answer explain? SEL (watch-lift, implicit) / VAL (shrunk residual rating, explicit) / SEL+VAL /
WDPROJ* (hand-built Wd projection = UPPER-BOUND DIAGNOSTIC, uses model geometry -> not a real candidate) / LLM
ordinal (~0.01, the flat channel). If SEL+VAL pooled R2 >~0.1 -> the item-derived answer is viable; implicit
should beat explicit (author's bet). Probe sample (noted); design-sheet run uses all users.
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn, scipy.sparse as sp
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; D=512; OUT=".cache/set_mn"; LAM=3.0; TAU=1.5
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
dd=np.load('data/movielens/.cache/ml25m/meta.npz'); uu=dd['uu'].astype(np.int64); ii=dd['ii'].astype(np.int64); rr=dd['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
imm=np.zeros(ni); ctt=np.zeros(ni); np.add.at(imm,ii,rr); np.add.at(ctt,ii,1.0); item_mean=np.where(ctt>0,imm/np.maximum(ctt,1),rr.mean())
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64); csz=np.array([len(Mbin[c].indices) for c in range(NC)]); usable=(csz>=20)
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
enc=SetEncoder(ni, token_mode='film', pool='attn', nlev=NLEV, nknow=3); enc.load_state_dict(pa['student'],strict=False); enc.eval()
def sv2lv(x): return S.sv_to_level((x-2.75)/2.25).astype(np.int64)
def beliefs(seqs):
    out=[]
    for i in range(0,len(seqs),256):
        ch=seqs[i:i+256]; B=len(ch); L=max(1,max(len(s[0]) for s in ch))
        ids=np.zeros((B,L),np.int64); lv=np.zeros((B,L),np.int64); kn=np.zeros((B,L),np.int64); pad=np.ones((B,L),bool)
        for j,(it,le) in enumerate(ch):
            if len(it): ids[j,:len(it)]=it; lv[j,:len(it)]=le; kn[j,:len(it)]=2; pad[j,:len(it)]=False
        with torch.no_grad(): out.append(enc(torch.from_numpy(ids),torch.zeros(B,L),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn)).numpy().astype(np.float64))
    return np.concatenate(out,0)
uids=np.load(RSD+'/mm_train_uids.npy'); KN=np.load(RSD+'/mm_train_know.npy'); VL=np.load(RSD+'/mm_train_val.npy')
ansT=(KN[:,:NC]>=1)&(VL[:,:NC]>=0); cvalT=np.clip(VL[:,:NC],0,3)
rs=np.random.default_rng(1); samp=rs.choice(len(uids),15000,replace=False)
seqs=[]; items=[]; resid=[]; rows=[]
for r in samp:
    uid=uids[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    seqs.append((its.astype(np.int64),sv2lv(rat))); items.append(its.astype(np.int64)); resid.append(rat-item_mean[its]); rows.append(r)
log(f"folding {len(seqs)} beliefs ...")
UT=beliefs(seqs); AFF=UT@Dc.T                      # (U,NC) oracle target
log("beliefs done; raw answers ...")
MT=Mbin.T.tocsr(); p_item=cnt.astype(np.float64)/cnt.sum(); pexp=np.asarray(Mbin@p_item).ravel()
U=len(items); SEL=np.zeros((U,NC)); VAL=np.zeros((U,NC)); EXPO=np.zeros((U,NC)); WDPROJ=np.zeros((U,NC))
for u in range(U):
    its=items[u]; res=resid[u]; sub=MT[its]; n_c=np.asarray(sub.sum(0)).ravel(); rsum=np.asarray(sub.T@res).ravel()
    e_c=len(its)*pexp; SEL[u]=np.log2((n_c+0.5)/(e_c+0.5)); mr=np.divide(rsum,n_c,out=np.zeros(NC),where=n_c>0)
    VAL[u]=(n_c/(n_c+LAM))*mr; EXPO[u]=e_c
    raw=(res[:,None]*Wcw[its]).sum(0); WDPROJ[u]=raw@Dc.T          # hand-built proj (diagnostic, model geometry)
REF=EXPO<TAU
LLM=np.full((U,NC),np.nan)
for u,r in enumerate(rows):
    ac=np.where(ansT[r])[0]; LLM[u,ac]=cvalT[r,ac]
def pooled_r2(feats,mask_extra=None):
    r2s=[];ws=[]
    for c in np.where(usable)[0]:
        m=~REF[:,c]
        if mask_extra is not None: m=m&mask_extra[:,c]
        if m.sum()<50: continue
        y=AFF[m,c]
        if y.var()<1e-9: continue
        X=np.column_stack([f[m,c] for f in feats]+[np.ones(m.sum())])
        beta,*_=np.linalg.lstsq(X,y,rcond=None); r2s.append(1-np.var(y-X@beta)/np.var(y)); ws.append(m.sum())
    return float(np.average(r2s,weights=ws)), len(r2s)
log("R2 decomposition:")
for name,feats in [("SEL",(SEL,)),("VAL",(VAL,)),("SEL+VAL",(SEL,VAL)),("WDPROJ*diag",(WDPROJ,))]:
    r2,ncc=pooled_r2(feats); log(f"  {name:>12}: pooled R2 {r2:+.3f}  ({ncc} concepts)")
# LLM: only cells where LLM not nan
llm_mask=~np.isnan(LLM); LLMf=np.nan_to_num(LLM)
r2l,ncl=pooled_r2((LLMf,),mask_extra=llm_mask); log(f"  {'LLM ordinal':>12}: pooled R2 {r2l:+.3f}  ({ncl} concepts)  [the flat channel]")
np.savez(OUT+'/r2_answers.npz', SEL=SEL, VAL=VAL, EXPO=EXPO, AFF=AFF, rows=np.array(rows))
log("done")
