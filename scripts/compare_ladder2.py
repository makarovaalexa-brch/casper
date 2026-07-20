"""Corrected ladder. Concept selection = highest head weight w_c (informativeness the policy would use),
among the user's answered USABLE concepts. Verify all-concepts == known 0.0723. Askable-item arm reused.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV
OUT="C:/dev/phd/casper/.cache/set_mn"; META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; LO=4.0; D=512
pa=torch.load(os.path.join(OUT,"paord_best.pt"),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach().numpy().astype(np.float32); bd=dec.bias.detach()
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)]); NTAG=cb.ntag
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
enc.load_state_dict(torch.load(os.path.join(OUT,"wmat_ep1.pt"),map_location='cpu')['student']); enc.eval()
def enc_seqs(seqs):
    B=len(seqs); Lm=max(len(s[0]) for s in seqs)
    ids=np.zeros((B,Lm),np.int64); lv=np.zeros((B,Lm),np.int64); kn=np.zeros((B,Lm),np.int64); pad=np.ones((B,Lm),bool)
    for i,(a,l,k) in enumerate(seqs): ids[i,:len(a)]=a; lv[i,:len(a)]=l; kn[i,:len(a)]=k; pad[i,:len(a)]=False
    with torch.no_grad(): return enc(torch.from_numpy(ids),torch.zeros(B,Lm),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))
def decm(z): return (z@torch.from_numpy(Wd).T+bd).detach().numpy().astype(np.float64)
def sv2lv(rat): return S.sv_to_level((rat-2.75)/2.25).astype(np.int64)
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cd=np.zeros((NC,D),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cd[c]=v/n if n>0 else 0
ALIGN_T=torch.from_numpy((Wd@Cd.T).T.copy()); IDF=torch.from_numpy((1.0/np.log(1.0+csz.clip(1))).astype(np.float32))
usable_c=torch.from_numpy((csz>=20)).float()
feat=np.stack([np.log(csz.clip(1)),(csz/ni),(np.arange(NC)>=NTAG).astype(np.float64)],1).astype(np.float32)
feat=(feat-feat.mean(0))/(feat.std(0)+1e-6); FEAT=torch.from_numpy(feat)
class Head(nn.Module):
    def __init__(s):
        super().__init__(); s.mlp=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,1))
        s.rp=nn.Parameter(torch.tensor(0.54)); s.rn=nn.Parameter(torch.tensor(0.54)); s.gate=nn.Parameter(torch.tensor(20.0))
    def w(s): return IDF*torch.exp(s.mlp(FEAT).squeeze(-1).clamp(-3,3))*usable_c
    def delta(s,cids,g):
        phi=torch.where(g>0,g*F.softplus(s.rp),g*F.softplus(s.rn))
        return s.gate*((phi*s.w()[cids])@ALIGN_T[cids])/max(len(cids),1)
hd=Head(); hd.load_state_dict(torch.load(f"{OUT}/v4head_best.pt",map_location='cpu')['head']); hd.eval()
WC=hd.w().detach().numpy()   # per-concept learned weight (informativeness)
popb=np.log(cnt.astype(np.float64)+1.0)
uidsV=np.load(RSD+'/mm_val_uids.npy'); KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3); GMAP={3:1.0,2:0.5,1:0.0,0:-1.0}
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr,hi,hr=its[p[:h]],rat[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    cc=np.where(ANSV[r])[0]; g=np.array([GMAP[int(CVALV[r,c])] for c in cc],np.float64); nz=g!=0
    if not nz.any(): continue
    cc=cc[nz]; g=g[nz]; ordc=np.argsort(-WC[cc])            # rank by head weight (informative, usable-first)
    ask=np.where(head[ki])[0]; ask=ask[np.argsort(-cnt[ki[ask]])]
    US.append((hl.astype(np.int64), cc[ordc].astype(np.int64), g[ordc].astype(np.float32), ki[ask].astype(np.int64), kr[ask]))
log(f"{len(US)} val users")
def nd(sc,hl,prof): return ndcg10(sc.copy(),list(hl),prof,head,False), ndcg10(sc.copy(),list(hl),prof,head,True)
def agg(ps): f=np.array([p[0] for p in ps if p[0] is not None]); t=np.array([p[1] for p in ps if p[1] is not None]); return f.mean(),t.mean()
pf,pt=agg([nd(popb,u[0],set()) for u in US]); log(f"popularity              FULL {pf:.4f}  TAIL {pt:.4f}")
with torch.no_grad():
    for K in [1,2,3,99]:
        ps=[]
        for hl,cc,g,aki,akr in US:
            k=min(K,len(cc)); dd=hd.delta(torch.from_numpy(cc[:k]),torch.from_numpy(g[:k])).numpy().astype(np.float64)
            ps.append(nd(popb+dd,hl,set()))
        f,t=agg(ps); lab="all" if K==99 else str(K); nk=sum(1 for u in US if len(u[1])>=K)
        log(f"+{lab:>3} concept(s) [askable]  FULL {f:.4f} ({f-pf:+.4f})  TAIL {t:.4f} ({t-pt:+.4f})   (n>=K {nk})")
for K in [1,2,3,5]:
    ps=[]
    for hl,cc,g,aki,akr in US:
        if len(aki)==0: ps.append((None,None)); continue
        it=aki[:K]; lv=sv2lv(akr[:K]); sc=decm(enc_seqs([(it,lv,np.full(len(it),2,np.int64))]))[0]
        ps.append(nd(sc,hl,set(it.tolist())))
    f,t=agg(ps); nk=sum(1 for u in US if len(u[3])>=K)
    log(f"+{K:>3} ASKABLE item(s)     FULL {f:.4f} ({f-pf:+.4f})  TAIL {t:.4f} ({t-pt:+.4f})   (n>=K {nk})")
log("done")
