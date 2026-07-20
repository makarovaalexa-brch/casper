"""FABLE FIX: BC-distill a VALIDATED-BRANCH tree teacher into the DKQS actor (escape the collapse basin;
guarantee >= greedy-static by construction; inherit certified adaptive branches).
Teacher = greedy-static SPINE (nbin=1) as default; at each node accept an answer-SPLIT only if per-cluster
best-next beats the node's shared best-next on a HELD-OUT CROSS-FIT of that node's users (E3-style validation).
BC: roll train users down the teacher (real answers -> real belief), label = node concept at each visited state;
CE on actor logits at [mu,diagSig,turn]. Eval on eval half: per-turn tail/full + ROUTES + controls, vs static.
2500 cache probe (HARD RULE #1/#2). LLM-free.
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from signed_latent import load_arena_base
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"; T=10; NBIN=3; MINLEAF=40
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
TRi=perm[:half]; EVi=perm[half:]; log(f"train {len(TRi)} eval {len(EVi)} bank {NCA}")
_R=np.full(ni,-1,np.int64)
def fast_nd(sc,held,idcg,tail):
    s=sc.copy()
    if tail: s[head]=-1e30
    top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]; _R[order]=np.arange(10)
    r=_R[held]; m=r>=0; v=LOG2[r[m]].sum()/idcg if m.any() else 0.0; _R[order]=-1; return v
def kal(mu,Sig,col,y,s2): d=DcU[col]; Sd=Sig@d; k=Sd/(d@Sd+s2); return mu+k*(y-d@mu),Sig-np.outer(k,Sd)
def score_cand(users,mu,Sig,col):     # mean post-answer tail-NDCG folding col with each user's REAL answer from shared belief
    by={}
    for p in users: by.setdefault(p['y'][col],[]).append(p)
    tot=0.0
    for yv,grp in by.items():
        mu2,_=kal(mu,Sig,col,yv,grp[0]['s2'][col]); s=popb+Wd@mu2; s[head]=-1e30
        top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]; _R[order]=np.arange(10)
        for p in grp:
            r=_R[p['hlt']]; m=r>=0; tot+=(LOG2[r[m]].sum()/p['it']) if m.any() else 0.0
        _R[order]=-1
    return tot/len(users)
def best_col(users,mu,Sig,avail):
    b=(-1,avail[0])
    for c in avail:
        s=score_cand(users,mu,Sig,c)
        if s>b[0]: b=(s,c)
    return b[1]
def meanfold(users,mu,Sig,col):
    ym=float(np.mean([p['y'][col] for p in users])); sm=float(np.mean([p['s2'][col] for p in users])); return kal(mu,Sig,col,ym,sm)
def validate_split(users,mu,Sig,col,av2):
    # cross-fit: does per-answer-cluster best-next beat one shared best-next? (E3 test)
    rg=np.random.default_rng(len(users)*7+col); idx=rg.permutation(len(users)); A=[users[i] for i in idx[:len(idx)//2]]; B=[users[i] for i in idx[len(idx)//2:]]
    if len(A)<MINLEAF or len(B)<MINLEAF: return False,None
    muc,Sigc=meanfold(users,mu,Sig,col)
    ncs=best_col(A,muc,Sigc,av2); static_B=score_cand(B,muc,Sigc,ncs)
    levs=np.array([p['y'][col] for p in users]); edges=np.quantile(levs,np.linspace(0,1,NBIN+1)[1:-1])
    tot=0.0;nn_=0
    for b in range(NBIN):
        lo=-1e9 if b==0 else edges[b-1]; hi=edges[b] if b<NBIN-1 else 1e9
        Ab=[p for p in A if lo<p['y'][col]<=hi]; Bb=[p for p in B if lo<p['y'][col]<=hi]
        if len(Ab)<10 or len(Bb)<5: Ab,Bb=A,B
        ncb=best_col(Ab,muc,Sigc,av2); tot+=score_cand(Bb,muc,Sigc,ncb)*len(Bb); nn_+=len(Bb)
    adapt_B=tot/max(nn_,1); return (adapt_B>static_B+1e-4),edges
def grow(users,mu,Sig,depth,avail):
    node=dict(col=None,children=None,edges=None,mu=mu.copy(),Sig=Sig.copy(),n=len(users))
    if not avail or depth>=T: return node
    col=best_col(users,mu,Sig,avail); node['col']=col; av2=[c for c in avail if c!=col]
    if depth>=T-1 or len(users)<2*MINLEAF or not av2:
        mu2,Sig2=meanfold(users,mu,Sig,col); node['children']={0:grow(users,mu2,Sig2,depth+1,av2)}; node['edges']=[]; return node
    acc,edges=validate_split(users,mu,Sig,col,av2)
    if acc:
        node['edges']=list(edges); node['children']={}
        for b in range(NBIN):
            lo=-1e9 if b==0 else edges[b-1]; hi=edges[b] if b<NBIN-1 else 1e9
            sub=[p for p in users if lo<p['y'][col]<=hi]
            if len(sub)<MINLEAF: sub=users
            mu2,Sig2=meanfold(sub,mu,Sig,col); node['children'][b]=grow(sub,mu2,Sig2,depth+1,av2)
    else:
        mu2,Sig2=meanfold(users,mu,Sig,col); node['edges']=[]; node['children']={0:grow(users,mu2,Sig2,depth+1,av2)}
    return node
log("growing validated-branch teacher ..."); t0=time.time(); TREE=grow([P[i] for i in TRi],np.zeros(D),Sig0.copy(),0,list(range(NCA))); log(f"grown ({time.time()-t0:.0f}s)")
def nsplits(nd):
    if nd['col'] is None or nd['children'] is None: return 0
    s=1 if len(nd['children'])>1 else 0
    return s+sum(nsplits(c) for c in nd['children'].values())
log(f"teacher validated SPLITS: {nsplits(TREE)}")
# roll a user down the teacher: yield (mu,diagSig,turn) states + label concept; fold REAL answers; track Sig via V
def rollout(nd,p,collect_bc,states,labels,mus):
    mu=np.zeros(D); Sig=Sig0.copy(); node=nd
    for t in range(T):
        if node is None or node['col'] is None:
            if mus is not None: mus.append(mu.copy())
            continue
        col=node['col']
        if collect_bc:
            diagS=np.diag(Sig); states.append(np.concatenate([mu,diagS,np.eye(T)[t]])); labels.append(col)
        mu,Sig=kal(mu,Sig,col,p['y'][col],p['s2'][col])
        if mus is not None: mus.append(mu.copy())
        if node['children'] is None: node=None
        elif node['edges']: b=int(np.searchsorted(node['edges'],p['y'][col])); node=node['children'].get(b,list(node['children'].values())[0])
        else: node=node['children'][0]
def teacher_curve(idx):
    ft=[[[] for _ in range(T+1)] for _ in range(2)]; routes=[]
    for gi in idx:
        p=P[gi]; mus=[]; rollout(TREE,p,False,None,None,mus)
        ft[0][0].append(fast_nd(popb,p['hla'],p['iff'],False)); ft[1][0].append(fast_nd(popb,p['hlt'],p['it'],True))
        for t in range(T):
            m=mus[t] if t<len(mus) else mus[-1]; ft[0][t+1].append(fast_nd(popb+Wd@m,p['hla'],p['iff'],False)); ft[1][t+1].append(fast_nd(popb+Wd@m,p['hlt'],p['it'],True))
    return [np.mean(x) for x in ft[0]],[np.mean(x) for x in ft[1]]
d=np.load(OUT+"/dkqs2_static.npz"); statT=list(d['st'])
tf,tt=teacher_curve(EVi); log(f"TEACHER eval tail@10 {tt[T]:.4f} (static {statT[T]:.4f})")
# ---- BC data ----
log("building BC data ..."); states=[]; labels=[]
for gi in TRi: rollout(TREE,P[gi],True,states,labels,None)
Xs=torch.tensor(np.array(states)); Ys=torch.tensor(np.array(labels,dtype=np.int64))
log(f"BC pairs {len(labels)}")
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(2*D+T,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,NCA))
    def forward(s,st): return s.net(st)
Sig0_t=torch.tensor(Sig0); dSig0=torch.tensor(np.diag(Sig0).copy()); DcU_t=torch.tensor(DcU); TE=torch.eye(T,dtype=torch.float64)
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
log("BC training ...")
for ep in range(120):
    act.train(); pm=np.random.default_rng(ep).permutation(len(Ys))
    for b in range(0,len(pm),512):
        ix=pm[b:b+512]; opt.zero_grad(); l=lossf(act(Xs[ix]),Ys[ix]); l.backward(); opt.step()
    if ep%15==0 or ep==119:
        ef,et,_=actor_curve(act,EVi)
        if et[T]>best[0]: best=(et[T],{k:v.clone() for k,v in act.state_dict().items()})
        log(f"  bc ep{ep} eval tail@10 {et[T]:.4f}")
act.load_state_dict(best[1]); torch.save(best[1],OUT+"/dkqs3_ckpt.pt")
ef,et,routes=actor_curve(act,EVi); tf_,tt_,_=actor_curve(act,TRi)
log("PER-Q TAIL:")
log("  ACTOR EVAL "+" ".join(f"{et[q]:.4f}" for q in range(T+1)))
log("  ACTOR TRAIN "+" ".join(f"{tt_[q]:.4f}" for q in range(T+1)))
log("  TEACHER    "+" ".join(f"{tt[q]:.4f}" for q in range(T+1)))
log("  static     "+" ".join(f"{statT[q]:.4f}" for q in range(T+1)))
log(f"BARS: actor eval tail@10 {et[T]:.4f} | teacher {tt[T]:.4f} | static {statT[T]:.4f}")
routes=np.array(routes)
for t in range(min(T,6)):
    v,c=np.unique(routes[t],return_counts=True); log(f"  q{t+1}: {len(v)} distinct, modal share {c.max()/routes.shape[1]:.2f}")
seqs=set(tuple(routes[:,b]) for b in range(routes.shape[1])); log(f"DISTINCT SEQUENCES: {len(seqs)} across {routes.shape[1]} users (teacher splits {nsplits(TREE)})")
log("done")
