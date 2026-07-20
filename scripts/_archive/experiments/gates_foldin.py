"""GATES battery for the fold-in concept channel (runs on a SNAPSHOT of the current best ckpt, in parallel
with training). Checks:
  G1 KNOCKOUT: gate->0 must == popb intercept exactly (no-harm structural).
  G2 CONCEPT-ONLY K-CURVE: tail+full at kc=1,2,4,8,16,32,all -> does it COMPOUND?
  G3 POPULARITY-DRIFT: did learned concept embeddings re-acquire the popularity (PC1) component the whitening
     removed? (mean |<cemb,pc1>| learned vs init; and cemb drift from whitened init).
  G4 WARM-COMPOSE: fold K real items through paord (belief z) + concept belief u -> score = z@Wd+bd + u@Wd.
     does the concept channel ADD on top of items? (K=0,3,5,10, +/- concepts). NOTE: foldin trained on popb
     base, so composing with paord's z is mildly OOD -> directional.
  G5 ITEM-TOWER GUARD: paord item-only full-profile NDCG (must be ~0.4859; frozen -> sanity only).
"""
import os,sys,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; D=512; OUT="C:/dev/phd/casper/.cache/set_mn"; LO=4.0
META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; SNAP=os.environ.get("SNAP","foldin_snap")
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]
Xw=Xc-(Xc@V1.T)@V1; pc1=torch.from_numpy(V1[0].astype(np.float32))
Cw=np.zeros((NC,D),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cw[c]=v/n if n>0 else 0
Cw=torch.from_numpy(Cw); usable=(csz>=20); popb=torch.from_numpy(np.log(cnt.astype(np.float32)+1.0)).float()
class FoldIn(nn.Module):
    def __init__(s,cinit,d=D,dh=128):
        super().__init__(); s.cemb=nn.Parameter(cinit.clone()); s.register_buffer('cinit',cinit.clone())
        s.proj=nn.Linear(d,dh); s.val=nn.Linear(1,dh); s.tok=nn.Linear(2*dh,dh); s.attn=nn.Linear(dh,1); s.vhead=nn.Linear(dh,d); s.gate=nn.Parameter(torch.tensor(0.0))
    def belief(s,cids,g,mask):
        emb=s.cemb[cids]; h=F.relu(s.tok(torch.cat([s.proj(emb),s.val(g.unsqueeze(-1))],-1)))
        a=s.attn(h).squeeze(-1).masked_fill(~mask,-1e9); a=torch.softmax(a,-1)
        return s.gate*(a.unsqueeze(-1)*s.vhead(h)).sum(1)
net=FoldIn(Cw); ck=torch.load(OUT+f'/{SNAP}.pt',map_location='cpu'); net.load_state_dict(ck['net']); net.eval()
log(f"loaded {SNAP} ep{ck['ep']} tail {ck['tail']:.4f} gate {float(net.gate):.3f}")
# item recommender for warm-compose
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
enc.load_state_dict(torch.load(OUT+'/wmat_ep1.pt',map_location='cpu')['student']); enc.eval()
def enc_items(seqs):
    B=len(seqs); Lm=max(1,max(len(s[0]) for s in seqs))
    ids=np.zeros((B,Lm),np.int64); lv=np.zeros((B,Lm),np.int64); kn=np.zeros((B,Lm),np.int64); pad=np.ones((B,Lm),bool)
    for i,(a,l) in enumerate(seqs):
        if len(a): ids[i,:len(a)]=a; lv[i,:len(a)]=l; kn[i,:len(a)]=2; pad[i,:len(a)]=False
    with torch.no_grad(): return enc(torch.from_numpy(ids),torch.zeros(B,Lm),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))
def sv2lv(rat): return S.sv_to_level((rat-2.75)/2.25).astype(np.int64)
# val cohort with known items + concepts + held(tail)
uidsV=np.load(RSD+'/mm_val_uids.npy'); KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3); GMAP={3:1.,2:.5,1:0.,0:-1.}
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr,hi,hr=its[p[:h]],rat[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    cc=np.where(ANSV[r]&usable)[0]; g=np.array([GMAP[int(CVALV[r,c])] for c in cc],np.float32); nz=g!=0
    if not nz.any(): continue
    cc=cc[nz]; g=g[nz]; oc=np.argsort(-csz[cc]); oi=np.argsort(-kr)
    US.append((hl.astype(np.int64), ki[oi].astype(np.int64), kr[oi], cc[oc].astype(np.int64), g[oc]))
log(f"val users {len(US)}")
def ndf(sc,hl,prof): return ndcg10(sc.copy(),list(hl),prof,head,False), ndcg10(sc.copy(),list(hl),prof,head,True)
def agg(ps): f=np.array([p[0] for p in ps if p[0] is not None]); t=np.array([p[1] for p in ps if p[1] is not None]); return f.mean(),t.mean()
def cbelief(cc,g):
    with torch.no_grad():
        cids=torch.from_numpy(cc).long().unsqueeze(0); gg=torch.from_numpy(g).unsqueeze(0); m=torch.ones(1,len(cc),dtype=torch.bool)
        return (net.belief(cids,gg,m)@Wd.T)[0].numpy().astype(np.float64)
pb=popb.numpy().astype(np.float64)
# G1 knockout
with torch.no_grad():
    saved=float(net.gate); net.gate.fill_(0.0)
    d0=cbelief(US[0][3],US[0][4]); net.gate.fill_(saved)
log(f"G1 KNOCKOUT: gate0 concept-delta max|.| = {np.abs(d0).max():.2e}  (must be ~0)")
# G3 popularity drift
with torch.no_grad():
    ali_init=(net.cinit @ pc1).abs().mean().item(); ali_lrn=(net.cemb @ pc1).abs().mean().item()
    drift=(net.cemb-net.cinit).norm(dim=-1)[torch.from_numpy(usable)].mean().item()
log(f"G3 POP-DRIFT: |<cemb,pc1>| init {ali_init:.4f} -> learned {ali_lrn:.4f}  (rise=popularity creeping back); cemb drift {drift:.3f}")
# G2 concept-only k-curve
log("G2 CONCEPT-ONLY K-CURVE (tail | full vs popb intercept):")
pf,pt=agg([ndf(pb,u[0],set()) for u in US])
for kc in [1,2,4,8,16,32,None]:
    ps=[ndf(pb+cbelief(u[3][:kc] if kc else u[3], u[4][:kc] if kc else u[4]), u[0], set()) for u in US]
    f,t=agg(ps); log(f"   kc={str(kc):>4}: TAIL {t:.4f} ({t-pt:+.4f})  FULL {f:.4f} ({f-pf:+.4f})")
# G4 warm-compose: items via paord (z) + concept u
log("G4 WARM-COMPOSE (paord item base z@Wd+bd  +/- concept fold; OOD-caveat: foldin trained on popb):")
for K in [0,3,5,10]:
    io=[]; ic=[]
    for hl,ki,kr,cc,g in US:
        it=ki[:K]; lv=sv2lv(kr[:K]) if K else np.array([],np.int64)
        z=enc_items([(it,lv)]); base=(z@Wd.T+bd)[0].numpy().astype(np.float64); prof=set(it.tolist())
        io.append(ndf(base,hl,prof)); ic.append(ndf(base+cbelief(cc,g),hl,prof))
    fo,to=agg(io); fc,tc=agg(ic)
    log(f"   K={K:>2}: items FULL {fo:.4f} TAIL {to:.4f} | +concept FULL {fc:.4f} ({fc-fo:+.4f}) TAIL {tc:.4f} ({tc-to:+.4f})")
log("done")
