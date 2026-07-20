"""Fold-in v2 — RECIPE FIX (author): interview simulation. Per user/step sample kc ASKED concepts (from the
usable pool) + K known items; asked concepts split answered(graded token)/REFUSED(refuse token) by the
answerability table; base = FROZEN paord item belief z from the K items; concept fold u ADDS on top:
    score(i) = z@Wd + bd + gate * <u, Wd_i>
Learns concept embeddings + fold-in encoder; item tower + decoder FROZEN. Fixes: few-shot (variable kc),
warm-compose (item-composed base), refusals (folded as signal). All 150k users, no reduction.
Eval: concept-only k-curve (few-shot now in-dist) + warm-compose (K items +/- concepts).
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
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; D=512; OUT="C:/dev/phd/casper/.cache/set_mn"; LO=4.0
META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; BASE="paord_best"
EPOCHS=int(os.environ.get("EPOCHS","6")); TAG=os.environ.get("TAG","foldin2"); ANCHOR=float(os.environ.get("ANCHOR","0.3"))
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(OUT+f'/{BASE}.pt',map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
popb=torch.from_numpy(np.log(cnt.astype(np.float32)+1.0)).float()   # A1 base (concepts-only training on this)
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cw=np.zeros((NC,D),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cw[c]=v/n if n>0 else 0
Cw=torch.from_numpy(Cw); usable=np.where(csz>=20)[0]
# frozen item recommender (paord set-encoder) for the item belief base
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
enc.load_state_dict(torch.load(OUT+'/wmat_ep1.pt',map_location='cpu')['student']); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
def item_base(seqs):     # list of (items,levels) -> (B,ni) frozen paord score
    B=len(seqs); Lm=max(1,max(len(s[0]) for s in seqs))
    ids=np.zeros((B,Lm),np.int64); lv=np.zeros((B,Lm),np.int64); kn=np.zeros((B,Lm),np.int64); pad=np.ones((B,Lm),bool)
    for i,(a,l) in enumerate(seqs):
        if len(a): ids[i,:len(a)]=a; lv[i,:len(a)]=l; kn[i,:len(a)]=2; pad[i,:len(a)]=False
    with torch.no_grad():
        z=enc(torch.from_numpy(ids),torch.zeros(B,Lm),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))
        return z@Wd.T+bd
def sv2lv(rat): return S.sv_to_level((rat-2.75)/2.25).astype(np.int64)

class FoldIn(nn.Module):
    def __init__(s,cinit,d=D,dh=128):
        super().__init__(); s.cemb=nn.Parameter(cinit.clone()); s.register_buffer('cinit',cinit.clone())
        s.proj=nn.Linear(d,dh); s.val=nn.Linear(2,dh); s.tok=nn.Linear(2*dh,dh)   # val=[graded, is_refused]
        s.attn=nn.Linear(dh,1); s.vhead=nn.Linear(dh,d); s.gate=nn.Parameter(torch.tensor(0.0))
    def belief(s,cids,g,ref,mask):
        emb=s.cemb[cids]; vin=torch.stack([g,ref],-1)                     # (B,L,2)
        h=F.relu(s.tok(torch.cat([s.proj(emb),s.val(vin)],-1)))
        a=s.attn(h).squeeze(-1).masked_fill(~mask,-1e9); a=torch.softmax(a,-1)
        return s.gate*(a.unsqueeze(-1)*s.vhead(h)).sum(1)
    def anchor(s): return ((s.cemb-s.cinit)**2).sum(-1).mean()
net=FoldIn(Cw); opt=torch.optim.Adam(net.parameters(), lr=0.01); GUARD=0.003

GMAP={3:1.0,2:0.5,1:0.0,0:-1.0}
def build(uf,vf,kf):
    uids=np.load(RSD+f'/{uf}'); KN=np.load(RSD+f'/{kf}'); VL=np.load(RSD+f'/{vf}')
    ansmask=(KN[:,:NC]>=1)&(VL[:,:NC]>=0); cval=np.clip(VL[:,:NC],0,3); out=[]
    for r,uid in enumerate(uids):
        a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
        if len(its)<8: continue
        ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
        ki,kr,hi,hr=its[p[:h]],rat[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
        if len(ki)<4 or len(hl)==0: continue
        ans=ansmask[r][usable]                                   # (nusable,) answerable?
        g=np.where(ans, np.array([GMAP[int(cval[r,c])] for c in usable]), 0.0).astype(np.float32)
        out.append((hl.astype(np.int64), ki.astype(np.int64), kr.astype(np.float32), usable, ans, g))
    return out
TR=build('mm_train_uids.npy','mm_train_val.npy','mm_train_know.npy')
VA=build('mm_val_uids.npy','mm_val_val.npy','mm_val_know.npy')
log(f"train {len(TR)} val {len(VA)} users; params {sum(p.numel() for p in net.parameters())}")

def cbelief_batch(rows, kc_mode, rng, kmax_item):
    """rows: list of user tuples. Sample kc asked-concepts + K items per user. Return (item_base_seqs, cbelief)."""
    B=len(rows); iseqs=[]; C=[]; G=[]; Rf=[]; L=[]
    for hl,ki,kr,us,ans,g in rows:
        # CONCEPTS-ONLY training (A1 recipe): no items -> concept channel carries the load (gate won't starve).
        K=0; oi=rng.permutation(len(ki))[:K]
        iseqs.append((ki[oi], sv2lv(kr[oi])))
        # sample kc asked concepts (log-uniform up to a realistic interview length), split answered/refused
        kc=min(int(np.exp(rng.uniform(0,np.log(64)))), len(us))     # interview asks 1..~64 concepts, not 1300
        sel=rng.permutation(len(us))[:kc]
        C.append(us[sel]); G.append(g[sel]); Rf.append((~ans[sel]).astype(np.float32)); L.append(kc)
    Lm=max(1,max(L))
    cids=torch.zeros(B,Lm,dtype=torch.long); gg=torch.zeros(B,Lm); ref=torch.zeros(B,Lm); mask=torch.zeros(B,Lm,dtype=torch.bool)
    for i in range(B):
        k=L[i]; cids[i,:k]=torch.from_numpy(C[i]); gg[i,:k]=torch.from_numpy(G[i]); ref[i,:k]=torch.from_numpy(Rf[i]); mask[i,:k]=True
    return iseqs, net.belief(cids,gg,ref,mask)

# ---- eval helpers ----
def cbel_fixed(us, ans, g, kc):     # fold top-kc ANSWERED concepts (signal), engagement order
    cc=us[ans]; gv=g[ans]
    if kc: order=np.argsort(-csz[cc])[:kc]; cc=cc[order]; gv=gv[order]
    if len(cc)==0: return np.zeros(ni)
    with torch.no_grad():
        cids=torch.from_numpy(cc).long().unsqueeze(0); gg=torch.from_numpy(gv).unsqueeze(0)
        ref=torch.zeros(1,len(cc)); m=torch.ones(1,len(cc),dtype=torch.bool)
        return (net.belief(cids,gg,ref,m)@Wd.T)[0].numpy().astype(np.float64)
def ndf(sc,hl,prof): return ndcg10(sc.copy(),list(hl),prof,head,False), ndcg10(sc.copy(),list(hl),prof,head,True)
def agg(ps): f=np.array([p[0] for p in ps if p[0] is not None]); t=np.array([p[1] for p in ps if p[1] is not None]); return f.mean(),t.mean()
def kcurve():
    net.eval()
    with torch.no_grad():
        base0=popb.numpy().astype(np.float64)
        pf,pt=agg([ndf(base0,VA[i][0],set()) for i in range(len(VA))])
        log(f"  [k-curve] popb base FULL {pf:.4f} TAIL {pt:.4f}")
        for kc in [1,2,4,8,16,None]:
            ps=[ndf(base0+cbel_fixed(VA[i][3],VA[i][4],VA[i][5],kc), VA[i][0], set()) for i in range(len(VA))]
            f,t=agg(ps); log(f"    kc={str(kc):>4}: TAIL {t:.4f} ({t-pt:+.4f})  FULL {f:.4f} ({f-pf:+.4f})")
    net.train()
def warm():
    net.eval()
    with torch.no_grad():
        for K in [0,3,5,10]:
            io=[];ic=[]
            for hl,ki,kr,us,ans,g in VA:
                oi=np.argsort(-kr)[:K]; b=item_base([(ki[oi],sv2lv(kr[oi]))]).numpy().astype(np.float64)[0]; prof=set(ki[oi].tolist())
                io.append(ndf(b,hl,prof)); ic.append(ndf(b+cbel_fixed(us,ans,g,None),hl,prof))
            fo,to=agg(io);fc,tc=agg(ic); log(f"    K={K:>2}: items T {to:.4f} | +con T {tc:.4f} ({tc-to:+.4f}) F {fc:.4f} ({fc-fo:+.4f})")
    net.train()

# step0 sanity
net.eval()
with torch.no_grad():
    base0=popb.numpy().astype(np.float64)
    b,t=agg([ndf(base0,VA[i][0],set()) for i in range(len(VA))])
log(f"[step0] popb base FULL {b:.4f} TAIL {t:.4f}; gate {float(net.gate):.3f} (gate0 -> concept adds 0)")
INTC_F=b
order=np.arange(len(TR)); best=-1
for ep in range(1,EPOCHS+1):
    net.train(); rng=np.random.default_rng(ep); rng.shuffle(order); run=0.0; nb=0; t0=time.time()
    for i in range(0,len(order),256):
        rows=[TR[j] for j in order[i:i+256]]
        iseqs,u=cbelief_batch(rows, 'log', rng, kmax_item=8)
        base=popb.unsqueeze(0)                   # concepts-only on the popb floor (A1 base)
        sc=base + u@Wd.T
        logp=F.log_softmax(sc,-1)
        ce=torch.stack([-logp[k][torch.from_numpy(rows[k][0])].mean() for k in range(len(rows))]).mean()
        loss=ce+ANCHOR*net.anchor(); opt.zero_grad(); loss.backward(); opt.step(); run+=float(ce); nb+=1
    log(f"[ep{ep}] loss {run/nb:.4f} gate {float(net.gate):.3f} ({(time.time()-t0)/60:.1f}m)")
    kcurve(); log("  [warm-compose]"); warm()
    torch.save({'net':net.state_dict(),'ep':ep}, f"{OUT}/{TAG}_ep{ep}.pt")
log("done")
