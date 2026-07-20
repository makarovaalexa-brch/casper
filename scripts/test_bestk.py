"""Does the TRAINED fold-in model give high NDCG from a FEW WELL-CHOSEN concepts (not engagement-order broad
genres)? Select each user's top-k concepts by alignment of the concept direction with their held-liked items
(oracle-ish), fold through the trained model, tail NDCG. Compare foldin2c (variable-kc) vs A1 (all-concepts).
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import set_mn as S
from set_mn import RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(m,flush=True)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; LO=4.0
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load('data/movielens/.cache/ml25m/meta.npz'); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
Wd=pa['decoder']['weight']; bd=torch.zeros(ni)
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cw=np.zeros((NC,D),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cw[c]=v/n if n>0 else 0
Cw=torch.from_numpy(Cw); usable=(csz>=20); popb=np.log(cnt.astype(np.float64)+1.0)
ALIGN=(Wd.numpy()@Cw.numpy().T).astype(np.float64)   # (ni x NC) item<->concept alignment (for selection)
class FoldIn(nn.Module):   # matches train_foldin2 (val=2: graded, refused)
    def __init__(s,cinit,d=D,dh=128):
        super().__init__(); s.cemb=nn.Parameter(cinit.clone()); s.register_buffer('cinit',cinit.clone())
        s.proj=nn.Linear(d,dh); s.val=nn.Linear(2,dh); s.tok=nn.Linear(2*dh,dh); s.attn=nn.Linear(dh,1); s.vhead=nn.Linear(dh,d); s.gate=nn.Parameter(torch.tensor(0.0))
    def belief(s,cids,g,ref,mask):
        emb=s.cemb[cids]; h=F.relu(s.tok(torch.cat([s.proj(emb),s.val(torch.stack([g,ref],-1))],-1)))
        a=s.attn(h).squeeze(-1).masked_fill(~mask,-1e9); a=torch.softmax(a,-1)
        return s.gate*(a.unsqueeze(-1)*s.vhead(h)).sum(1)
KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy'); uidsV=np.load(RSD+'/mm_val_uids.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3); GMAP={3:1.,2:.5,1:0.,0:-1.}
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,hi,hr=its[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    cc=np.where(ANSV[r]&usable)[0]; g=np.array([GMAP[int(CVALV[r,c])] for c in cc],np.float32); nz=g!=0
    if not nz.any(): continue
    US.append((hl.astype(np.int64), cc[nz], g[nz]))
log(f"val users {len(US)}")
def tm(ps): t=np.array([x for x in ps if x is not None]); return t.mean()
for TAG in ['foldin2c_ep6','foldin_A1']:
    try: ck=torch.load(OUT+f'/{TAG}_best.pt',map_location='cpu') if 'A1' in TAG else torch.load(OUT+f'/foldin2c_ep6.pt',map_location='cpu')
    except Exception as e:
        # fallback: latest foldin2c epoch
        import glob; fs=sorted(glob.glob(OUT+'/foldin2c_ep*.pt'));
        if not fs: log(f'{TAG}: no ckpt'); continue
        ck=torch.load(fs[-1],map_location='cpu'); TAG=os.path.basename(fs[-1])
    net=FoldIn(Cw)
    try: net.load_state_dict(ck['net']); 
    except Exception as e: log(f'{TAG} load fail {str(e)[:80]}'); continue
    net.eval(); log(f"--- {TAG} gate {float(net.gate):.3f} ---")
    pt=tm([ndcg10(popb.copy(),list(hl),set(),head,True) for hl,cc,g in US]); log(f"  popb tail {pt:.4f}")
    def foldk(cc,g):
        with torch.no_grad():
            cids=torch.from_numpy(cc).long().unsqueeze(0); gg=torch.from_numpy(g).unsqueeze(0); ref=torch.zeros(1,len(cc)); m=torch.ones(1,len(cc),dtype=torch.bool)
            return (net.belief(cids,gg,ref,m)@Wd.T)[0].numpy().astype(np.float64)
    for K,sel in [(1,'best'),(2,'best'),(3,'best'),(1,'eng'),(3,'eng')]:
        ps=[]
        for hl,cc,g in US:
            if sel=='best': sc=ALIGN[hl][:,cc].mean(0)*g; order=np.argsort(-sc)[:K]   # concept whose dir aligns w/ held-likes
            else: order=np.argsort(-csz[cc])[:K]
            ps.append(ndcg10((popb+foldk(cc[order],g[order])).copy(),list(hl),set(),head,True))
        log(f"  K={K} {sel:>4}: tail {tm(ps):.4f} ({tm(ps)-pt:+.4f})")
log("done")
