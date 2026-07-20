"""2-minute test of the 'whitening shot us in the foot' hypothesis. Keep the TRAINED W(v), but swap the
frozen concept input rows: whitened-unit (current) vs RAW centroid (on-manifold, has popularity) vs
raw-unit-normed. Re-run cold NDCG (niche + oracle heldcov). If raw >> whitened, whitening-at-input is the
off-manifold culprit and a raw-input retrain is justified. NOTE: feeding raw through a W(v) trained on
whitened is OUT-OF-DISTRIBUTION -> directional signal, not final proof.
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
CKPT="wmat_ep1"; BASE="paord_best"

base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(os.path.join(OUT,BASE+'.pt'),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr() if hasattr(cb.Mbin,'tocsr') else cb.Mbin
csz=np.array([len(Mbin[c].indices) for c in range(NC)]); memset=[set(Mbin[c].indices.tolist()) for c in range(NC)]

# three concept-row variants
def build(kind):
    C=np.zeros((NC,512),np.float32)
    for c in range(NC):
        idx=Mbin[c].indices
        if len(idx)<20: continue
        if kind=="whit": v=Xw[idx].mean(0); v=v/max(np.linalg.norm(v),1e-9)
        elif kind=="raw": v=IE[idx].mean(0)                                   # on-manifold, natural magnitude
        elif kind=="rawunit": v=IE[idx].mean(0); v=v/max(np.linalg.norm(v),1e-9)
        C[c]=v
    return torch.from_numpy(C)

enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
ck=torch.load(os.path.join(OUT,CKPT+'.pt'),map_location='cpu'); enc.load_state_dict(ck['student']); enc.eval()
dec=nn.Linear(D,ni); dec.load_state_dict(ck['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
def decode(z): return (z@Wd.T+bd).detach().numpy().astype(np.float64)
def enc_seqs(seqs):
    B=len(seqs); Lm=max(len(s[0]) for s in seqs)
    ids=np.zeros((B,Lm),np.int64); lv=np.zeros((B,Lm),np.int64); kn=np.zeros((B,Lm),np.int64); pad=np.ones((B,Lm),bool)
    for i,(a,l,k) in enumerate(seqs): ids[i,:len(a)]=a; lv[i,:len(a)]=l; kn[i,:len(a)]=k; pad[i,:len(a)]=False
    with torch.no_grad(): return enc(torch.from_numpy(ids),torch.zeros(B,Lm),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))

uidsV=np.load(RSD+'/mm_val_uids.npy'); KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3)
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,hi,hr=its[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    loved=np.where(ANSV[r] & (CVALV[r]==3))[0]; loved=loved[csz[loved]>=20]
    if len(loved)==0: continue
    US.append((int(uid), ki, hl, loved))
log(f"{len(US)} val users")

z0=enc(torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long))
sc0=decode(z0)[0]
def nd(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,False), ndcg10(sc.copy(),list(hl),set(),head,True)
def agg(ps): f=np.array([p[0] for p in ps if p[0] is not None]); t=np.array([p[1] for p in ps if p[1] is not None]); return f.mean(),t.mean()
bf,bt=agg([nd(sc0,u[2]) for u in US]); log(f"intercept  FULL {bf:.4f}  TAIL {bt:.4f}")
def pickniche(u): return u[3][np.argmin(csz[u[3]])]
def pickoracle(u): hs=set(u[2].tolist()); return u[3][int(np.argmax([len(memset[c]&hs) for c in u[3]]))]
for kind in ["whit","raw","rawunit"]:
    with torch.no_grad(): enc.item_emb.weight[ni:ni+NC].copy_(build(kind))
    for sel,pf in [("niche",pickniche),("oracle",pickoracle)]:
        ps=[nd(decode(enc_seqs([(np.array([ni+pf(u)]),np.array([3]),np.array([2]))]))[0], u[2]) for u in US]
        f,t=agg(ps); log(f"  {kind:<7} {sel:<7} FULL {f:.4f} ({f-bf:+.4f})  TAIL {t:.4f} ({t-bt:+.4f})")
log("done")
