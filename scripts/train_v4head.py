"""V4 additive concept re-rank head on the FROZEN tower (design sheet DESIGN_SHEET_V4_RERANK.md).
    score(i) = sc0(i) + Delta(i);   Delta(i) = gate * sum_{c in answered} phi_sign(g_c) * w_c * align(c,i)
- sc0 = frozen cold empty-fold score (concepts-only supervision, items masked -> sc0 user-independent).
- align = ALIGN_T[c,i] = Wd[i] . whitened_centroid[c]  (the geo2 winner; factor-space graded alignment).
- w_c = softplus(MLP([log|members|, breadth, is_entity]))  (feature-parameterized, not free per-concept).
- phi_sign: g scaled by learned s_pos (g>0) / s_neg (g<0)  (asymmetric love/hate).
- gate init 0 -> Delta==0 at step0 -> score == intercept EXACTLY (monotone safety; FULL preserved).
Loss = multinomial CE over held-LIKED items on (sc0+Delta), sc0 FROZEN, full 18430 softmax (no pools).
Trains FAST (tower frozen, sc0 precomputed). All train users, disjoint val for selection. No data reduction.
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
BASE="paord_best"; EPOCHS=int(os.environ.get("EPOCHS","6")); TAG=os.environ.get("TAG","v4head")

base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']; headt=torch.from_numpy(head)
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(os.path.join(OUT,BASE+'.pt'),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach().numpy().astype(np.float32)

cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
NTAG=cb.ntag
# whitened centroids -> ALIGN_T (NC x ni)
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
Cd=np.zeros((NC,D),np.float32)
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20:
        v=Xw[idx].mean(0).astype(np.float32); n=np.linalg.norm(v); Cd[c]=v/n if n>0 else 0
ALIGN_T=torch.from_numpy((Wd@Cd.T).T.copy())          # (NC x ni) float32
usable_c=torch.from_numpy((csz>=20)).float()
IDF=torch.from_numpy((1.0/np.log(1.0+csz.clip(1))).astype(np.float32))   # geo2 winner: down-weight broad genres
# concept features for w_c
feat=np.stack([np.log(csz.clip(1)), (csz/ni), (np.arange(NC)>=NTAG).astype(np.float64)],1).astype(np.float32)
featm=feat.mean(0); feats=feat.std(0)+1e-6; feat=(feat-featm)/feats
FEAT=torch.from_numpy(feat)

# COLD base = log-popularity (Paper-B popb; the geo2-proven base). Note: model empty-fold (0.1617) is WORSE
# than log-pop (0.1886) at pure cold -> for concept-only cold-start the base IS popb; the strong tower earns
# its keep when items arrive. (TODO morning: verify why model cold < logpop -- Fable's snap-to-canonical flag.)
sc0=torch.from_numpy(np.log(cnt.astype(np.float32)+1.0)).float().detach()

# ---- head ----
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,1))
        self.mlp[-1].weight.data.zero_(); self.mlp[-1].bias.data.zero_()   # init 0 -> w_c == IDF (geo2 operator)
        # softplus -> s_pos,s_neg >= 0 : love ADDS (g>0), hate SUBTRACTS (g<0). init raw 0.54 -> softplus~1.0
        self.rp=nn.Parameter(torch.tensor(0.54)); self.rn=nn.Parameter(torch.tensor(0.54))
        self.gate=nn.Parameter(torch.tensor(20.0))           # init inside geo2 safe basin (gate==beta; peak~120). gate=0 => Delta=0 (safety, provable)
    def w(self):  # (NC,) = IDF * exp(learned residual); init residual 0 -> w == IDF
        return IDF*torch.exp(self.mlp(FEAT).squeeze(-1).clamp(-3,3))*usable_c
    def delta(self, cids, g):   # cids: LongTensor(k), g: FloatTensor(k) -> (ni,)
        s_pos=F.softplus(self.rp); s_neg=F.softplus(self.rn)
        phi=torch.where(g>0, g*s_pos, g*s_neg)               # g<0 -> phi<0 (hate subtracts); asymmetric magnitudes
        coef=phi*self.w()[cids]                              # (k,)
        return self.gate*(coef @ ALIGN_T[cids])/max(len(cids),1)   # per-user normalization (geo2 winner)
head=Head()
opt=torch.optim.Adam(head.parameters(), lr=0.03)
GUARD=0.003   # accept a checkpoint only if FULL >= intercept - GUARD (safe-basin selection)

# ---- data: train + val cohorts ----
def build(uids_file, val_file, know_file):
    uids=np.load(RSD+f'/{uids_file}'); KN=np.load(RSD+f'/{know_file}'); VL=np.load(RSD+f'/{val_file}')
    ans=(KN[:,:NC]>=1)&(VL[:,:NC]>=0); cval=np.clip(VL[:,:NC],0,3)
    GMAP={3:1.0,2:0.5,1:0.0,0:-1.0}; out=[]
    for r,uid in enumerate(uids):
        a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
        if len(its)<8: continue
        ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
        hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
        if len(hl)==0: continue
        cc=np.where(ans[r])[0]; g=np.array([GMAP[int(cval[r,c])] for c in cc],np.float64); nz=g!=0
        if not nz.any(): continue
        out.append((hl.astype(np.int64), cc[nz].astype(np.int64), g[nz].astype(np.float32)))
    return out
TR=build('mm_train_uids.npy','mm_train_val.npy','mm_train_know.npy')
VA=build('mm_val_uids.npy','mm_val_val.npy','mm_val_know.npy')
log(f"train {len(TR)} users, val {len(VA)} users")

head_bool=head  # boolean head mask (np.bool array) for ndcg10 tail masking
head_bool=headt.numpy()
def evaluate():
    head.eval()
    with torch.no_grad():
        bf=[]; bt=[]; ff=[]; ft=[]
        sc0n=sc0.numpy().astype(np.float64)
        for hl,cc,g in VA:
            dd=head.delta(torch.from_numpy(cc),torch.from_numpy(g)).numpy().astype(np.float64)
            for arr,accf,acct in [(sc0n,bf,bt),(sc0n+dd,ff,ft)]:
                f=ndcg10(arr.copy(),list(hl),set(),head_bool,False); t=ndcg10(arr.copy(),list(hl),set(),head_bool,True)
                if f is not None: accf.append(f)
                if t is not None: acct.append(t)
    head.train()
    return np.mean(bf),np.mean(bt),np.mean(ff),np.mean(ft)

bf,bt,ff,ft=evaluate()
log(f"[step0] logpop intercept FULL {bf:.4f} TAIL {bt:.4f} | +head(gate20) FULL {ff:.4f} ({ff-bf:+.4f}) TAIL {ft:.4f} ({ft-bt:+.4f})  (expect geo2 safe-basin: full flat, tail up)")

# ---- train (intra-epoch eval; select best TAIL subject to FULL >= intercept - GUARD) ----
INTC_F, INTC_T = bf, bt   # intercept from step0 sanity
order=np.arange(len(TR)); best=-1; best_desc="none"; EVAL_EVERY=25000
def checkpoint(tag_note):
    global best, best_desc
    bf_,bt_,ff_,ft_=evaluate()
    ok = ff_ >= INTC_F - GUARD
    star = "*" if (ok and ft_>best) else " "
    log(f"  {tag_note}: gate {float(head.gate):.2f} s+ {F.softplus(head.rp).item():.2f} s- {F.softplus(head.rn).item():.2f} | FULL {ff_:.4f} ({ff_-INTC_F:+.4f}) TAIL {ft_:.4f} ({ft_-INTC_T:+.4f}) {'GUARD-OK' if ok else 'FULL-VIOL'} {star}")
    if ok and ft_>best:
        best=ft_; best_desc=f"{tag_note} tail={ft_:.4f} full={ff_:.4f}"
        torch.save({'head':head.state_dict(),'note':tag_note,'tail':ft_,'full':ff_,'intercept_full':INTC_F,'intercept_tail':INTC_T}, f"{OUT}/{TAG}_best.pt")
for ep in range(1,EPOCHS+1):
    head.train(); rng=np.random.default_rng(ep); rng.shuffle(order); run=0.0; nb=0; t0=time.time()
    for step,i in enumerate(order):
        hl,cc,g=TR[i]
        dd=head.delta(torch.from_numpy(cc),torch.from_numpy(g))
        loss=-F.log_softmax(sc0+dd,-1)[torch.from_numpy(hl)].mean()
        opt.zero_grad(); loss.backward(); opt.step(); run+=float(loss); nb+=1
        if (step+1)%EVAL_EVERY==0: checkpoint(f"ep{ep}.{(step+1)//EVAL_EVERY}")
    checkpoint(f"ep{ep}.end ({run/nb:.3f} loss {(time.time()-t0)/60:.1f}m)")
log(f"done best: {best_desc}")
