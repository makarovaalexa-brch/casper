"""BLEND-ANSWER HONESTY pre-check (gate for DKQS). DKQS trains on a blend answer y=sum_i w_i y_{c_i} for a blend
direction d=normalize(sum_i w_i d_{c_i}). Honest ONLY if that blend answer tracks the TRUE projection <u*,d>.
Test over random 2- and 3-sparse blends: per user, true=<us,d>, blend=sum w_i y_{c_i} (component calibrated
answers). Report corr(blend,true) vs the mean SINGLE-COMPONENT fidelity corr(y_c,<us,d_c>) (the base ~0.62 that
concepts already achieve). GO if blend fidelity is NOT degraded by blending (>= component fidelity).
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from signed_latent import load_arena_base
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"
base=load_arena_base(); ni=base['ni']; cnt=base['cnt']
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64)
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
z=np.load(CACHE,allow_pickle=True); Bc=z['Bc']; EDG=z['EDG']; Yc=z['Yc']; UIDX=z['uidx']; US=list(z['US'])
NCA=len(UIDX)
# per-user: belief us, and component answer y over UIDX columns
USv=np.array([u[7] for u in US])                      # nusers x D beliefs
YU=np.zeros((len(US),NCA))
for r,u in enumerate(US):
    selU,valU=u[1],u[2]; tt=Bc[UIDX,0]*selU+Bc[UIDX,1]*valU+Bc[UIDX,2]
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)]); YU[r]=Yc[UIDX,lev]
log(f"{len(US)} users, {NCA} candidate concepts, catalog {ni} items")
DcU=Dc[UIDX]                                          # NCA x D
AFF=USv@DcU.T                                         # nusers x NCA : true <us,d_c> for each single concept
def corr(a,b):
    a=a-a.mean(); b=b-b.mean(); da=np.sqrt((a*a).sum()); db=np.sqrt((b*b).sum())
    return float((a*b).sum()/(da*db)) if da>0 and db>0 else 0.0
# single-component fidelity baseline
comp_fid=np.array([corr(YU[:,j],AFF[:,j]) for j in range(NCA)])
log(f"single-concept fidelity corr(y_c, <us,d_c>): median {np.median(comp_fid):.3f}  IQR [{np.quantile(comp_fid,.25):.3f},{np.quantile(comp_fid,.75):.3f}]  (R2 median {np.median(comp_fid**2):.3f})")
rng2=np.random.default_rng(11)
for KSP in [2,3]:
    bc=[]; base=[]
    for _ in range(400):
        cols=rng2.choice(NCA,KSP,replace=False); w=rng2.random(KSP); w=w/w.sum()
        d=(w[:,None]*DcU[cols]).sum(0); nd=np.linalg.norm(d)
        if nd<1e-6: continue
        d=d/nd
        true=USv@d                                    # <us, d>  per user
        blend=(w[None,:]*YU[:,cols]).sum(1)           # sum w_i y_{c_i}
        bc.append(corr(blend,true))
        base.append(comp_fid[cols].mean())            # mean single-component fidelity of the parts
    bc=np.array(bc); base=np.array(base)
    log(f"[{KSP}-sparse] blend corr(blend,true): median {np.median(bc):.3f} IQR [{np.quantile(bc,.25):.3f},{np.quantile(bc,.75):.3f}]  | mean-component-fidelity {np.median(base):.3f}  | blend>=component in {100*np.mean(bc>=base):.0f}% of blends")
log("VERDICT: honest if blend median corr >= ~0.6 (single-concept level) and blend>=component majority.")
log("done")
