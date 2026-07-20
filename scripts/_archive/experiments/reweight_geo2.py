"""Refined Paper-B fold (Fable's converting changes): score = popb + beta * align_residual, WHITENED dir,
with (a) IDF concept weighting w_c = 1/log(1+|members_c|) [broad genres do the head damage],
(b) per-user normalization by #answered [uniform beta across users], (c) fine beta sweep,
(d) beta->inf endpoint approximated by large beta = intercept<->filter interpolation.
Report FULL + TAIL vs log-pop intercept. Whit only (raw is dead). Optimized: ALIGN as (NC x ni) row-major.
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
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])

Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cd=np.zeros((NC,D),np.float64)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20:
        v=Xw[idx].mean(0); n=np.linalg.norm(v); Cd[c]=v/n if n>0 else 0
ALIGN_T=(Wd@Cd.T).T.copy()                      # (NC x ni) ROW-major -> fast row slicing
IDF=1.0/np.log(1.0+csz.clip(1))                 # down-weight broad genres
popb=np.log(cnt.astype(np.float64)+1.0)

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

# precompute per-user residual direction (unit-ish), then only beta scales it
UDLT=[]
for hl,ans,g in US:
    w=g*IDF[ans]                                # graded * IDF
    u=(ALIGN_T[ans]*w[:,None]).sum(0)/max(len(ans),1)   # per-user normalized
    UDLT.append((hl,u))
bf,bt=agg([ndf(popb,u[0]) for u in US]); log(f"=== log-pop intercept FULL {bf:.4f} TAIL {bt:.4f} ===")
for beta in [8.0,20.0,50.0,120.0,300.0,1000.0]:
    ps=[ndf(popb+beta*u, hl) for hl,u in UDLT]
    f,t=agg(ps); log(f"  beta={beta:<5} FULL {f:.4f} ({f-bf:+.4f})  TAIL {t:.4f} ({t-bt:+.4f})")
log("done")
