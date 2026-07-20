"""HYPOTHESIS H1 (score-space re-weight, Paper-B duality / DESIGN_CONCEPT_OPERATOR V3):
the concept fold DRAGS NDCG down because we inject a free vector into the BELIEF (blurs off-manifold).
The correct operator touches items through THEIR OWN attribute evidence:
    score(i) = sc0(i) + sum_c g_c * beta * align(c,i),   align = binary membership Mbin[c,i]
Additive residual on the base -> beta->0 == intercept exactly -> CANNOT drag NDCG below intercept.
Test cold (empty profile) + graded concepts (loved +1 / liked +0.5 / hated -1), sweep beta.
Base = model cold sc0 AND pure log-popularity, both reported. No retraining. No data reduction.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn, scipy.sparse as sp
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
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr()   # (C x ni) binary membership

# base rankings: model cold sc0 (wmat_ep1 empty fold = item-path popularity prior), and pure log-popularity
ck=torch.load(os.path.join(OUT,"wmat_ep1.pt"),map_location='cpu')
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
enc.load_state_dict(ck['student']); enc.eval()
dec=nn.Linear(D,ni); dec.load_state_dict(ck['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
with torch.no_grad():
    z0=enc(torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long))
    sc0_model=(z0@Wd.T+bd).detach().numpy().astype(np.float64)[0]
sc0_pop=np.log(cnt.astype(np.float64)+1.0)
log(f"sc0_model std {None if sc0_model is None else round(float(sc0_model.std()),3)}  logpop std {sc0_pop.std():.3f}")

# val cohort
uidsV=np.load(RSD+'/mm_val_uids.npy'); KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3)
GMAP={3:1.0,2:0.5,1:0.0,0:-1.0}   # loved+1 / liked+0.5 / meh 0 / hated -1
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,hi,hr=its[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    ans=np.where(ANSV[r])[0]
    if len(ans)==0: continue
    g=np.array([GMAP[int(CVALV[r,c])] for c in ans],np.float64); nz=g!=0
    if not nz.any(): continue
    US.append((int(uid), hl, ans[nz], g[nz]))
log(f"{len(US)} val users with graded answered concepts")

def ndf(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,False), ndcg10(sc.copy(),list(hl),set(),head,True)
def agg(ps): f=np.array([p[0] for p in ps if p[0] is not None]); t=np.array([p[1] for p in ps if p[1] is not None]); return f.mean(),t.mean()

def delta(u, beta):
    uid,hl,ans,g=u
    # sum_c g_c * Mbin[c]  -> dense ni
    sub=Mbin[ans]                       # (k x ni)
    d=(sub.T @ g)                       # ni
    return np.asarray(d).ravel()*beta

for base_name, sc0 in [("model", sc0_model), ("logpop", sc0_pop)]:
    if sc0 is None: continue
    bf,bt=agg([ndf(sc0,u[1]) for u in US])
    log(f"=== BASE={base_name}  intercept FULL {bf:.4f} TAIL {bt:.4f} ===")
    for beta in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        ps=[ndf(sc0+delta(u,beta), u[1]) for u in US]
        f,t=agg(ps)
        log(f"  beta={beta:<4} FULL {f:.4f} ({f-bf:+.4f})  TAIL {t:.4f} ({t-bt:+.4f})")
log("done")
