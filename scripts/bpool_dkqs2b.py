"""DKQS diagnostic: retrain concept-only non-myopic DKQS (tail obj), SAVE checkpoint, report per-turn curve on
EVAL *and* TRAIN, and the ROUTES (are different users routed to different concepts = truly adaptive?).
Loads cached static/vg (fast). Answers: Q2 adaptive-routes, Q6 train-vs-eval (overfit test), Q7 per-turn curve.
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from signed_latent import load_arena_base
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"; T=10
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0)
z=np.load(CACHE,allow_pickle=True); Sig0=z['Sig0']; Bc=z['Bc']; EDG=z['EDG']; Yc=z['Yc']; S2=z['S2']; UIDX=z['uidx']; US=list(z['US'])
NCA=len(UIDX); DcU=Dc[UIDX]; LOG2=1.0/np.log2(np.arange(2,12))
def prep(u):
    hlt,selU,valU,refU,aki,akr,hlall,us=u; tt=Bc[UIDX,0]*selU+Bc[UIDX,1]*valU+Bc[UIDX,2]
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)])
    hlt=np.asarray(hlt,np.int64); hla=np.asarray(hlall,np.int64)
    return dict(hlt=hlt,hla=hla,y=Yc[UIDX,lev],s2=np.maximum(S2[UIDX,lev],1e-3),
                it=LOG2[:min(len(hlt),10)].sum() if len(hlt) else 1.0, iff=LOG2[:min(len(hla),10)].sum() if len(hla) else 1.0)
P=[prep(u) for u in US]
rs=np.random.default_rng(7); perm=rs.permutation(len(P)); half=len(P)//2
TRi=perm[:half]; EVi=perm[half:]; log(f"train {len(TRi)} eval {len(EVi)} bank {NCA}")
_R=np.full(ni,-1,np.int64)
def fast_nd(sc,held,idcg,tail):
    s=sc.copy()
    if tail: s[head]=-1e30
    top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]; _R[order]=np.arange(10)
    r=_R[held]; m=r>=0; v=LOG2[r[m]].sum()/idcg if m.any() else 0.0; _R[order]=-1; return v
d=np.load(OUT+"/dkqs2_static.npz"); statT=list(d['st']); statF=list(d['sf']); vgT=list(d['vt']); SEQ=list(d['seq'])
log(f"static tail q10 {statT[10]:.4f} | seq(first5 concepts)={[int(UIDX[c]) for c in SEQ[:5]]}")
Wd_t=torch.tensor(Wd); popb_t=torch.tensor(popb); DcU_t=torch.tensor(DcU); Sig0_t=torch.tensor(Sig0)
dSig0=torch.tensor(np.diag(Sig0).copy()); head_t=torch.tensor(head); YU=torch.tensor(np.array([p['y'] for p in P])); S2U=torch.tensor(np.array([p['s2'] for p in P]))
TE=torch.eye(T,dtype=torch.float64)
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(2*D+T,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,NCA))
    def forward(s,st): return s.net(st)
def soft_ndcg(mu_b,idxs):
    sc=(mu_b@Wd_t.T+popb_t).masked_fill(head_t.unsqueeze(0),-1e30); logp=torch.log_softmax(sc,1); loss=0.0; nB=0
    for bi,gi in enumerate(idxs):
        h=P[gi]['hlt']
        if len(h)==0: continue
        w=torch.tensor(LOG2[:min(len(h),10)]); loss=loss-(w*logp[bi,torch.tensor(h[:len(w)])]).sum()/w.sum(); nB+=1
    return loss/max(nB,1)
def unroll(act,idxs,train=False,collect=False):
    B=len(idxs); mu=torch.zeros(B,D,dtype=torch.float64); Vst=torch.zeros(B,T,D,dtype=torch.float64)
    asked=torch.zeros(B,NCA,dtype=torch.bool); Yb=YU[idxs]; S2b=S2U[idxs]; loss=0.0
    lam=[(t+1)**2 for t in range(T)]; Z=sum(lam); mus=[]; routes=[]
    for t in range(T):
        diagS=dSig0.unsqueeze(0)-(Vst*Vst).sum(1)
        st=torch.cat([mu,diagS,TE[t].unsqueeze(0).expand(B,-1)],1)
        logits=act(st).masked_fill(asked,-1e30); w=torch.softmax(logits,1)
        d_=w@DcU_t; d_=d_/d_.norm(dim=1,keepdim=True).clamp_min(1e-8)
        Sd=d_@Sig0_t-torch.einsum('bt,btd->bd',torch.einsum('btd,bd->bt',Vst,d_),Vst)
        y=(w*Yb).sum(1); s2=(w*w*S2b).sum(1); denom=(d_*Sd).sum(1)+s2
        mu=mu+(Sd/denom.unsqueeze(1))*(y-(d_*mu).sum(1)).unsqueeze(1)
        Vst=Vst.clone(); Vst[:,t,:]=Sd/denom.clamp_min(1e-9).sqrt().unsqueeze(1)
        am=w.argmax(1); asked=asked.clone(); asked[torch.arange(B),am]=True
        if train: loss=loss+lam[t]/Z*soft_ndcg(mu,idxs)
        if collect: mus.append(mu.detach()); routes.append(am.numpy().copy())
    return loss if train else (mus,routes)
def curve(act,idx):
    ft=[[[] for _ in range(T+1)] for _ in range(2)]
    with torch.no_grad():
        for b in range(0,len(idx),256):
            ib=idx[b:b+256]; mus,_=unroll(act,ib,collect=True)
            for bi,gi in enumerate(ib):
                p=P[gi]; ft[0][0].append(fast_nd(popb,p['hla'],p['iff'],False)); ft[1][0].append(fast_nd(popb,p['hlt'],p['it'],True))
                for t in range(T):
                    m2=mus[t][bi].numpy(); ft[0][t+1].append(fast_nd(popb+Wd@m2,p['hla'],p['iff'],False)); ft[1][t+1].append(fast_nd(popb+Wd@m2,p['hlt'],p['it'],True))
    return [np.mean(x) for x in ft[0]],[np.mean(x) for x in ft[1]]
act=Actor().double(); opt=torch.optim.Adam(act.parameters(),lr=2e-3); best=(-1,None,-1)
for ep in range(16):
    act.train(); ordr=np.random.default_rng(ep).permutation(len(TRi))
    for b in range(0,len(ordr),256):
        ib=TRi[ordr[b:b+256]]; loss=unroll(act,ib,train=True); opt.zero_grad(); loss.backward(); opt.step()
    act.eval(); ef,et=curve(act,EVi)
    if et[T]>best[0]: best=(et[T],{k:v.clone() for k,v in act.state_dict().items()},ep)
    log(f"  ep{ep} EVAL tail@10 {et[T]:.4f} full@10 {ef[T]:.4f}")
act.load_state_dict(best[1]); torch.save(best[1],OUT+"/dkqs2b_ckpt.pt"); log(f"best ep{best[2]} tail@10 {best[0]:.4f} (ckpt saved)")
ef,et=curve(act,EVi); tf,tt=curve(act,TRi)
log("PER-QUESTION (best ckpt):")
log("  EVAL TAIL "+" ".join(f"{et[q]:.4f}" for q in range(T+1)))
log("  TRAIN TAIL "+" ".join(f"{tt[q]:.4f}" for q in range(T+1)))
log("  static TAIL "+" ".join(f"{statT[q]:.4f}" for q in range(T+1)))
log("  var-gr TAIL "+" ".join(f"{vgT[q]:.4f}" for q in range(T+1)))
log("  EVAL FULL "+" ".join(f"{ef[q]:.4f}" for q in range(T+1)))
log(f"OVERFIT CHECK: train tail@10 {tt[T]:.4f} vs eval {et[T]:.4f} vs static {statT[T]:.4f}")
# routes
with torch.no_grad(): _,routes=unroll(act,EVi[:800],collect=True)
routes=np.array(routes)  # T x B (concept bank idx)
log("ROUTES (adaptivity):")
for t in range(min(T,6)):
    vals,cnts=np.unique(routes[t],return_counts=True); o=np.argsort(-cnts)
    top=[(int(UIDX[vals[i]]),int(cnts[i])) for i in o[:3]]; ndist=len(vals)
    log(f"  q{t+1}: {ndist} distinct concepts across 800 users; top3 (concept,#users)={top}; modal share {cnts.max()/routes.shape[1]:.2f}")
seqs=set(tuple(routes[:,b]) for b in range(routes.shape[1]))
log(f"  DISTINCT FULL SEQUENCES across 800 users: {len(seqs)} (1 => degenerated to static; 800 => fully personalised)")
log("done")
