"""FAST PROOF: BC the actor directly from the cached GREEDY-STATIC sequence (no teacher grow). Tests whether the
actor CAN represent+learn static (0.1244) -- if yes, the from-scratch failure was the OBJECTIVE (soft-CE != tail-
NDCG), not capacity/representation. Also probes: does the actor, cloned from a *fixed* sequence, stay static
(expected), confirming the machinery. Runs in minutes.
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
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64)
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
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)]); hlt=np.asarray(hlt,np.int64); hla=np.asarray(hlall,np.int64)
    return dict(hlt=hlt,hla=hla,y=Yc[UIDX,lev],s2=np.maximum(S2[UIDX,lev],1e-3),it=LOG2[:min(len(hlt),10)].sum() if len(hlt) else 1.0,iff=LOG2[:min(len(hla),10)].sum() if len(hla) else 1.0)
P=[prep(u) for u in US]; rs=np.random.default_rng(7); perm=rs.permutation(len(P)); half=len(P)//2
TRi=perm[:half]; EVi=perm[half:]
_R=np.full(ni,-1,np.int64)
def fast_nd(sc,held,idcg,tail):
    s=sc.copy()
    if tail: s[head]=-1e30
    top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]; _R[order]=np.arange(10)
    r=_R[held]; m=r>=0; v=LOG2[r[m]].sum()/idcg if m.any() else 0.0; _R[order]=-1; return v
def kal(mu,Sig,col,y,s2): d=DcU[col]; Sd=Sig@d; k=Sd/(d@Sd+s2); return mu+k*(y-d@mu),Sig-np.outer(k,Sd)
d=np.load(OUT+"/dkqs2_static.npz"); SEQ=[int(c) for c in d['seq']]; statT=list(d['st']); log(f"static seq (bank idx) {SEQ}; static tail@10 {statT[T]:.4f}")
# BC data from the FIXED greedy-static sequence
states=[]; labels=[]
for gi in TRi:
    p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy()
    for t,col in enumerate(SEQ):
        states.append(np.concatenate([mu,np.diag(Sig),np.eye(T)[t]])); labels.append(col)
        mu,Sig=kal(mu,Sig,col,p['y'][col],p['s2'][col])
Xs=torch.tensor(np.array(states)); Ys=torch.tensor(np.array(labels,dtype=np.int64)); log(f"BC pairs {len(labels)}")
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(2*D+T,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,NCA))
    def forward(s,st): return s.net(st)
def actor_curve(act,idx):
    ft=[[[] for _ in range(T+1)] for _ in range(2)]; routes=[[] for _ in range(T)]
    act.eval()
    with torch.no_grad():
        for gi in idx:
            p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy(); asked=set()
            ft[0][0].append(fast_nd(popb,p['hla'],p['iff'],False)); ft[1][0].append(fast_nd(popb,p['hlt'],p['it'],True))
            for t in range(T):
                st=torch.tensor(np.concatenate([mu,np.diag(Sig),np.eye(T)[t]])); lg=act(st).numpy()
                for c in asked: lg[c]=-1e30
                col=int(np.argmax(lg)); asked.add(col); routes[t].append(int(UIDX[col]))
                mu,Sig=kal(mu,Sig,col,p['y'][col],p['s2'][col])
                ft[0][t+1].append(fast_nd(popb+Wd@mu,p['hla'],p['iff'],False)); ft[1][t+1].append(fast_nd(popb+Wd@mu,p['hlt'],p['it'],True))
    return [np.mean(x) for x in ft[0]],[np.mean(x) for x in ft[1]],routes
act=Actor().double(); opt=torch.optim.Adam(act.parameters(),lr=1e-3); lossf=nn.CrossEntropyLoss(); best=(-1,None)
log("BC training from greedy-static ...")
for ep in range(150):
    act.train(); pm=np.random.default_rng(ep).permutation(len(Ys))
    for b in range(0,len(pm),512):
        ix=pm[b:b+512]; opt.zero_grad(); l=lossf(act(Xs[ix]),Ys[ix]); l.backward(); opt.step()
    if ep%20==0 or ep==149:
        ef,et,_=actor_curve(act,EVi)
        if et[T]>best[0]: best=(et[T],{k:v.clone() for k,v in act.state_dict().items()})
        log(f"  bc ep{ep} eval tail@10 {et[T]:.4f} (train CE {float(l):.3f})")
act.load_state_dict(best[1]); ef,et,routes=actor_curve(act,EVi)
log("PER-Q TAIL:")
log("  BC-ACTOR  "+" ".join(f"{et[q]:.4f}" for q in range(T+1)))
log("  static    "+" ".join(f"{statT[q]:.4f}" for q in range(T+1)))
log(f"BC-actor eval tail@10 {et[T]:.4f} vs static {statT[T]:.4f}  (match => actor CAN learn static; failure was the OBJECTIVE)")
routes=np.array(routes); seqmatch=np.mean([routes[t]==int(UIDX[SEQ[t]]) for t in range(T)])
log(f"route agreement with greedy-static seq: {seqmatch:.2f} (should be ~1.0 if BC reproduced it)")
log("done")
