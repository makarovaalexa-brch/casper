"""Faithful Paper-B fold: score(i) = popb[i] + beta * (Wd[i] . u_c), on a PRESERVED log-pop floor.
u_c = sum_c g_c * concept_dir[c]  (graded geometric alignment, NOT flat membership -> within-genre
discrimination + preserved popularity floor -> cannot sink below intercept at beta->0).
Compare concept_dir = WHITENED centroid vs RAW centroid. Sweep beta small->large. Cold + graded.
Reports concept-only NDCG vs the log-pop intercept (Paper-B's own bar: intercept=0.2551 there).
No retraining, no data reduction.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV
META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; OUT="C:/dev/phd/casper/.cache/set_mn"; LO=4.0; D=512
BASE="paord_best"

base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(os.path.join(OUT,BASE+'.pt'),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
Wd=pa['decoder']['weight'] if 'decoder' in pa else None
# decoder rows for scoring geometry
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach().numpy().astype(np.float64)  # (ni x D)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])

# concept directions
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
def dirs(kind):
    Cd=np.zeros((NC,D),np.float64)
    for c in range(NC):
        idx=Mbin[c].indices
        if len(idx)<20: continue
        v=(Xw[idx] if kind=="whit" else IE[idx]).mean(0); n=np.linalg.norm(v)
        if n>0: Cd[c]=v/n
    return Cd

popb=np.log(cnt.astype(np.float64)+1.0)

# val cohort with graded answered concepts
uidsV=np.load(RSD+'/mm_val_uids.npy'); KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3)
GMAP={3:1.0,2:0.5,1:0.0,0:-1.0}
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(hl)==0: continue
    ans=np.where(ANSV[r])[0]; g=np.array([GMAP[int(CVALV[r,c])] for c in ans],np.float64); nz=g!=0
    if not nz.any(): continue
    US.append((hl, ans[nz], g[nz]))
log(f"{len(US)} val users")

def ndf(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,False), ndcg10(sc.copy(),list(hl),set(),head,True)
def agg(ps): f=np.array([p[0] for p in ps if p[0] is not None]); t=np.array([p[1] for p in ps if p[1] is not None]); return f.mean(),t.mean()

bf,bt=agg([ndf(popb,u[0]) for u in US]); log(f"=== log-pop intercept FULL {bf:.4f} TAIL {bt:.4f} ===")
for kind in ["whit","raw"]:
    Cd=dirs(kind)
    ALIGN=Wd@Cd.T                     # (ni x NC): per-item alignment to each concept
    astd=ALIGN.std()
    log(f"--- concept_dir={kind}  align std {astd:.3f} ---")
    for beta in [0.1,0.25,0.5,1.0,2.0,4.0]:
        ps=[]
        for hl,ans,g in US:
            u=(ALIGN[:,ans]*g).sum(1)          # sum_c g_c * align_c(i)
            ps.append(ndf(popb+beta*u, hl))
        f,t=agg(ps); log(f"  beta={beta:<4} FULL {f:.4f} ({f-bf:+.4f})  TAIL {t:.4f} ({t-bt:+.4f})")
log("done")
