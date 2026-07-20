"""BELIEF-POOL concept elicitation (Kalman / Bayesian-linear update on the FROZEN decoder factor space).
Belief u~N(mu,Sig). score(i)=popb_i+<mu,Wd_i>. Prior mu=0,Sig=tau2*I -> cold=popb.
Each concept answer = linear-Gaussian obs of affinity <u,d_c>=t_c (d_c=decoder-side whitened member centroid).
Sequential update: K=Sig d/(dᵀSig d+sig2); mu+=K(t_c-<d,mu>); Sig-=K (Sig d)ᵀ.  NO training, NO attention.
INVARIANT TEST: fold answers ONE AT A TIME (fixed realizable order = |answer| desc), record NDCG per step;
report the per-step curve + per-step delta -> must be NON-DECREASING, never negative.
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import set_mn as S, arena_core as AC
def log(m): print(m,flush=True)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; RS=".cache/rich_signal"; LO=4.0
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load('data/movielens/.cache/ml25m/meta.npz'); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wd=pa['decoder']['weight'].numpy().astype(np.float64)  # (ni x D)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
# decoder-side whitened concept directions
m=Wd.mean(0); Wc=Wd-m; U,Sg,Vt=np.linalg.svd(Wc[np.random.default_rng(0).choice(ni,4000,replace=False)],full_matrices=False)
v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0)
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
    cc=cc[nz]; g=g[nz]; oo=np.argsort(-np.abs(g))                # fold confident answers first (realizable)
    US.append((hl.astype(np.int64), cc[oo], g[oo]))
log(f"val users {len(US)}")
MAXQ=12
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
pt=np.mean([ndt(popb,hl) for hl,cc,g in US]); log(f"popb intercept tail {pt:.4f}")
for SCALE,SIG2,TAU2 in [(2,1,1),(4,1,1),(4,2,1),(8,2,1),(8,4,1)]:
    step_nd=[[] for _ in range(MAXQ+1)]
    for hl,cc,g in US:
        mu=np.zeros(D); Sig=TAU2*np.eye(D)
        step_nd[0].append(ndt(popb+Wd@mu,hl))
        K=min(MAXQ,len(cc))
        for j in range(K):
            dvec=Dc[cc[j]]; t=SCALE*g[j]; Sd=Sig@dvec
            k=Sd/(dvec@Sd+SIG2); mu=mu+k*(t-dvec@mu); Sig=Sig-np.outer(k,Sd)
            step_nd[j+1].append(ndt(popb+Wd@mu,hl))
    means=[np.mean(s) if s else np.nan for s in step_nd]
    deltas=[means[i]-means[i-1] for i in range(1,len(means)) if not np.isnan(means[i])]
    mono='MONOTONE' if all(dl>=-1e-4 for dl in deltas) else 'NON-MONO'
    log(f"[scale{SCALE} sig{SIG2} tau{TAU2}] "+" ".join(f"{i}:{means[i]:.4f}" for i in range(min(9,MAXQ+1)))+f"  vs popb@q{min(8,MAXQ)}: {means[min(8,MAXQ)]-pt:+.4f}  {mono}")
log("done")
