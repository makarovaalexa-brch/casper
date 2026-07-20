"""Concept FOLD-IN encoder on the FROZEN paord tower (Paper-B operator, the one that COMPOUNDS).
Fold answered concept tokens (whitened centroid + graded value) through a small nonlinear ATTENTION encoder
to a POINT u in the decoder factor space; score = popb + gate * <Wd_i, u>. gate zero-init -> u=0 -> intercept
(never-drop safety). Frozen: item tower, decoder Wd/bd, whitened centroids. Trains ~120k params fast (no tower
forward). Concepts-only masked supervision, CE over held-likes. Reports the concept-only k-curve (does it
COMPOUND like Paper B, not saturate like the linear V4 head?).
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
META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; OUT="C:/dev/phd/casper/.cache/set_mn"; LO=4.0; D=512
BASE="paord_best"; EPOCHS=int(os.environ.get("EPOCHS","6")); TAG=os.environ.get("TAG","foldin")

base=load_arena_base(); ni=base['ni']; head=base['headmask']; headt=torch.from_numpy(head); cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(os.path.join(OUT,BASE+'.pt'),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach()   # (ni x D) frozen
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
# whitened centroids (frozen)
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cw=np.zeros((NC,D),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cw[c]=v/n if n>0 else 0
Cw=torch.from_numpy(Cw); usable=(csz>=20)
popb=torch.from_numpy(np.log(cnt.astype(np.float32)+1.0)).float()

# ---- fold-in encoder (nonlinear attention pool -> point); concept embeddings NOW LEARNED ----
class FoldIn(nn.Module):
    def __init__(s, cinit, d=D, dh=128):
        super().__init__()
        s.cemb=nn.Parameter(cinit.clone())              # LEARNED concept embeddings (init = whitened centroids)
        s.register_buffer('cinit', cinit.clone())       # anchor: keep near whitened taste-geometry
        s.proj=nn.Linear(d,dh); s.val=nn.Linear(1,dh); s.tok=nn.Linear(2*dh,dh)
        s.attn=nn.Linear(dh,1); s.vhead=nn.Linear(dh,d)
        s.gate=nn.Parameter(torch.tensor(0.0))          # ZERO-INIT -> u=0 -> score=popb (intercept)
    def belief(s, cids, g, mask):                       # cids (B,L) g (B,L) mask (B,L) bool
        emb=s.cemb[cids]                                 # (B,L,d) LEARNED
        h=F.relu(s.tok(torch.cat([s.proj(emb), s.val(g.unsqueeze(-1))],-1)))
        a=s.attn(h).squeeze(-1).masked_fill(~mask,-1e9); a=torch.softmax(a,-1)
        u=(a.unsqueeze(-1)*s.vhead(h)).sum(1)
        return s.gate*u
    def anchor(s): return ((s.cemb-s.cinit)**2).sum(-1).mean()   # L2-to-whitened-init (guard vs popularity drift)
net=FoldIn(Cw)
opt=torch.optim.Adam(net.parameters(), lr=0.01)
GUARD=0.003; ANCHOR=float(os.environ.get("ANCHOR","0.3"))   # weight on stay-near-whitened penalty

# ---- data ----
GMAP={3:1.0,2:0.5,1:0.0,0:-1.0}
def build(uf,vf,kf):
    uids=np.load(RSD+f'/{uf}'); KN=np.load(RSD+f'/{kf}'); VL=np.load(RSD+f'/{vf}')
    ans=(KN[:,:NC]>=1)&(VL[:,:NC]>=0); cval=np.clip(VL[:,:NC],0,3); out=[]
    for r,uid in enumerate(uids):
        a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
        if len(its)<8: continue
        ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
        hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
        if len(hl)==0: continue
        cc=np.where(ans[r]&usable)[0]; g=np.array([GMAP[int(cval[r,c])] for c in cc],np.float32); nz=g!=0
        if not nz.any(): continue
        cc=cc[nz]; g=g[nz]; order=np.argsort(-csz[cc])          # engagement order (big concepts first)
        out.append((hl.astype(np.int64), cc[order].astype(np.int64), g[order]))
    return out
TR=build('mm_train_uids.npy','mm_train_val.npy','mm_train_know.npy')
VA=build('mm_val_uids.npy','mm_val_val.npy','mm_val_know.npy')
log(f"train {len(TR)} val {len(VA)} users; foldin params {sum(p.numel() for p in net.parameters())}")

def score_batch(seqs):        # list of (cc,g) -> (B,ni) scores
    B=len(seqs); L=max(len(s[0]) for s in seqs)
    cids=torch.zeros(B,L,dtype=torch.long); g=torch.zeros(B,L); mask=torch.zeros(B,L,dtype=torch.bool)
    for i,(cc,gg) in enumerate(seqs):
        k=len(cc); cids[i,:k]=torch.from_numpy(cc); g[i,:k]=torch.from_numpy(gg); mask[i,:k]=True
    u=net.belief(cids,g,mask)                    # (B,D)
    return popb.unsqueeze(0)+u@Wd.T              # (B,ni)

def evaluate(kc=None):
    net.eval(); bf=[];bt=[];ff=[];ft=[]
    with torch.no_grad():
        pb=popb.numpy().astype(np.float64)
        for i in range(0,len(VA),256):
            chunk=VA[i:i+256]
            seqs=[(cc[:kc] if kc else cc, g[:kc] if kc else g) for _,cc,g in chunk]
            sc=score_batch(seqs).numpy().astype(np.float64)
            for j,(hl,_,_) in enumerate(chunk):
                nb=ndcg10(pb.copy(),list(hl),set(),head,False); nbt=ndcg10(pb.copy(),list(hl),set(),head,True)
                nf=ndcg10(sc[j].copy(),list(hl),set(),head,False); nft=ndcg10(sc[j].copy(),list(hl),set(),head,True)
                if nb is not None and nf is not None: bf.append(nb); ff.append(nf)
                if nbt is not None and nft is not None: bt.append(nbt); ft.append(nft)
    net.train(); return np.mean(bf),np.mean(bt),np.mean(ff),np.mean(ft)

bf,bt,ff,ft=evaluate()
log(f"[step0] intercept FULL {bf:.4f} TAIL {bt:.4f} | +foldin FULL {ff:.4f} TAIL {ft:.4f} (gate0 -> must match)")
INTC_F,INTC_T=bf,bt; order=np.arange(len(TR)); best=-1
for ep in range(1,EPOCHS+1):
    net.train(); rng=np.random.default_rng(ep); rng.shuffle(order); run=0.0; nb=0; t0=time.time()
    for i in range(0,len(order),256):
        idx=order[i:i+256]; chunk=[TR[j] for j in idx]
        seqs=[(cc,g) for _,cc,g in chunk]
        sc=score_batch(seqs)
        logp=F.log_softmax(sc,-1)
        ce=torch.stack([-logp[k][torch.from_numpy(chunk[k][0])].mean() for k in range(len(chunk))]).mean()
        loss=ce+ANCHOR*net.anchor()
        opt.zero_grad(); loss.backward(); opt.step(); run+=float(ce); nb+=1
    bf,bt,ff,ft=evaluate()
    ok=ff>=INTC_F-GUARD; star='*' if (ok and ft>best) else ''
    log(f"[ep{ep}] loss {run/nb:.4f} gate {float(net.gate):.3f} | FULL {ff:.4f} ({ff-INTC_F:+.4f}) TAIL {ft:.4f} ({ft-INTC_T:+.4f}) {'OK' if ok else 'VIOL'} {star} {(time.time()-t0)/60:.1f}m")
    if ok and ft>best: best=ft; torch.save({'net':net.state_dict(),'ep':ep,'tail':ft,'full':ff}, f"{OUT}/{TAG}_best.pt")
# k-curve on best
net.load_state_dict(torch.load(f"{OUT}/{TAG}_best.pt",map_location='cpu')['net']); net.eval()
log("[k-curve] concept-only, does it COMPOUND?")
for kc in [1,2,4,8,16,32,None]:
    bf,bt,ff,ft=evaluate(kc=kc)
    log(f"  kc={str(kc):>4}: TAIL {ft:.4f} ({ft-bt:+.4f})  FULL {ff:.4f} ({ff-bf:+.4f})")
log(f"done best_tail {best:.4f}")
