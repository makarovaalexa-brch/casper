"""SNAPPED eval of the saved DKQS checkpoint (the control skipped earlier): at each turn fold the ARGMAX single
concept (CLEAN fold), not the diffuse softmax blend. Separates two failures: (i) blend-dilution (blend<clean) vs
(ii) the learned SEQUENCE being genuinely worse than greedy-static. Also logs snapped routes (adaptivity).
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from signed_latent import load_arena_base
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count()); S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"; T=10
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
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)]); hlt=np.asarray(hlt,np.int64)
    return dict(hlt=hlt,y=Yc[UIDX,lev],s2=np.maximum(S2[UIDX,lev],1e-3),it=LOG2[:min(len(hlt),10)].sum() if len(hlt) else 1.0)
P=[prep(u) for u in US]; rs=np.random.default_rng(7); perm=rs.permutation(len(P)); half=len(P)//2; EVi=perm[half:]
_R=np.full(ni,-1,np.int64)
def fast_t(sc,held,idcg):
    s=sc.copy(); s[head]=-1e30; top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]; _R[order]=np.arange(10)
    r=_R[held]; m=r>=0; v=LOG2[r[m]].sum()/idcg if m.any() else 0.0; _R[order]=-1; return v
Sig0_t=torch.tensor(Sig0); dSig0=torch.tensor(np.diag(Sig0).copy()); DcU_t=torch.tensor(DcU); TE=torch.eye(T,dtype=torch.float64)
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(2*D+T,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,NCA))
    def forward(s,st): return s.net(st)
act=Actor().double(); act.load_state_dict(torch.load(OUT+"/dkqs2b_ckpt.pt")); act.eval()
d=np.load(OUT+"/dkqs2_static.npz"); statT=list(d['st'])
# SNAPPED unroll (numpy, exact clean folds), actor chooses argmax; also BLEND for contrast
def run(mode):
    curve=[[] for _ in range(T+1)]; routes=[[] for _ in range(T)]
    for gi in EVi:
        p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy(); Vst=[]; asked=set(); curve[0].append(fast_t(popb,p['hlt'],p['it']))
        for t in range(T):
            diagS=np.diag(Sig0)-(np.array(Vst)**2).sum(0) if Vst else np.diag(Sig0)
            st=torch.tensor(np.concatenate([mu,diagS,np.eye(T)[t]]))
            with torch.no_grad(): logits=act(st).numpy()
            for c in asked: logits[c]=-1e30
            if mode=='snap':
                col=int(np.argmax(logits)); dvec=DcU[col]; y=p['y'][col]; s2=p['s2'][col]; asked.add(col); routes[t].append(int(UIDX[col]))
            else:
                w=np.exp(logits-logits.max()); w/=w.sum(); dvec=w@DcU; nrm=np.linalg.norm(dvec); dvec=dvec/max(nrm,1e-8)
                y=float((w*p['y']).sum()); s2=float((w*w*p['s2']).sum()); asked.add(int(np.argmax(logits)))
            Sd=Sig@dvec; k=Sd/(dvec@Sd+s2); mu=mu+k*(y-dvec@mu); Sig=Sig-np.outer(k,Sd); Vst.append(Sd/np.sqrt(dvec@Sd+s2))
            curve[t+1].append(fast_t(popb+Wd@mu,p['hlt'],p['it']))
    return [np.mean(x) for x in curve],routes
snapC,routes=run('snap'); blendC,_=run('blend')
log("PER-Q TAIL:")
log("  SNAPPED  "+" ".join(f"{snapC[q]:.4f}" for q in range(T+1)))
log("  BLEND    "+" ".join(f"{blendC[q]:.4f}" for q in range(T+1)))
log("  static   "+" ".join(f"{statT[q]:.4f}" for q in range(T+1)))
log(f"SNAP tail@10 {snapC[T]:.4f} | BLEND {blendC[T]:.4f} | static {statT[T]:.4f}")
for t in range(min(T,6)):
    v,c=np.unique(routes[t],return_counts=True); log(f"  q{t+1} snapped: {len(v)} distinct, modal share {c.max()/len(EVi):.2f}, top {int(v[np.argmax(c)])}")
log("done")
