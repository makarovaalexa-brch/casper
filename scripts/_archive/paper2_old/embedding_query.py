"""
PAPER B — E2' proof-of-concept: DIRECT CONTINUOUS-EMBEDDING QUERY (no vocab snap). The action is a point a in R^D; the
user model (embeddings) answers it directly: answer(a)=sim-weighted mean profile residual; answerable iff alignment
mass>tau. Fold (a, answer) straight into the frozen encoder. Question: does the continuous action space have headroom
BEYOND discrete asking? Compare (NDCG@10 full+tail):
  pop ; oracle-ITEM @{1,2,4} (best discrete single items, peek) ; DIRECT-EMB ceiling @{1,2,4} (oracle over a rich
  CANDIDATE-DIRECTION set incl. COMBINATION directions u_full / pairwise-means, peek) ; full-profile fold (recommender
  ceiling). If direct-emb @1 ~= full-profile and >> oracle-item @1, one ideal combination-direction question captures
  near-full info that NO single discrete item can => large continuous headroom (verbalization is the remaining cost).
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; rng=np.random.default_rng(0); torch.manual_seed(0)
NEVAL=int(os.environ.get('NEVAL',200)); TAU=float(os.environ.get('TAU',2.0))
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
Qn=Q/np.clip(np.linalg.norm(Q,axis=1,keepdims=True),1e-8,None)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
def enc_u_tokens(toks):     # toks: list of (factor_vec D, value)
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
_W=1./np.log2(np.arange(2,12))
def ndcg(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in tlike if not headmask[t]]
    else: rel=list(tlike)
    if not rel: return None
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]; rs=set(rel)
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rs)/(_W[:min(10,len(rel))].sum()+1e-12)
def ridge(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
def answer(a,prof,rd):                                   # direct embedding-query answer model (leakage-safe: profile only)
    Qp=Q[prof]; sims=np.maximum((Qp@a)/ (np.linalg.norm(a)+1e-9) /np.clip(np.linalg.norm(Qp,axis=1),1e-9,None),0.0)
    mass=sims.sum()
    if mass<TAU: return None,mass                        # "don't know"
    res=np.array([rd[j]-mu-bi[j] for j in prof]); return float((sims*res).sum()/(mass+1e-9)), mass
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:NEVAL]
def cand_dirs(prof,rd):                                   # rich candidate DIRECTIONS incl. combinations (not single items)
    dirs=[Q[j] for j in prof]                             # single-item directions (= item asking)
    u_full=ridge([Q[j] for j in prof],[rd[j]-mu-bi[j] for j in prof]); dirs.append(u_full)   # COMBINATION direction
    for _ in range(20):                                   # random combinations of 2-4 profile items
        k=rng.integers(2,min(5,len(prof)+1)); idx=rng.choice(len(prof),size=k,replace=False); dirs.append(Q[[prof[i] for i in idx]].mean(0))
    return dirs
def run(mode,K,tail):
    acc=0.;m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        if mode=='full':
            v=ndcg(enc_u_tokens([(Q[j],rd[j]-mu-bi[j]) for j in prof]),tlike,set(prof),tail)
        elif mode=='item':                                # oracle over single items (peek)
            asked=[];toks=[]
            for _ in range(K):
                cands=[j for j in prof if j not in asked]; best=None
                for c in cands:
                    v=ndcg(enc_u_tokens(toks+[(Q[c],rd[c]-mu-bi[c])]),tlike,set(prof)|set(asked)|{c},tail)
                    if v is not None and (best is None or v>best[0]): best=(v,c)
                if best is None: break
                asked.append(best[1]); toks.append((Q[best[1]],rd[best[1]]-mu-bi[best[1]]))
            v=ndcg(enc_u_tokens(toks),tlike,set(prof),tail)
        else:                                             # direct-embedding oracle over candidate DIRECTIONS (peek)
            dirs=cand_dirs(prof,rd); toks=[]; used=set()
            for _ in range(K):
                best=None
                for di,a in enumerate(dirs):
                    if di in used: continue
                    ans,mass=answer(a,prof,rd)
                    if ans is None: continue
                    v=ndcg(enc_u_tokens(toks+[(a,ans)]),tlike,set(prof),tail)
                    if v is not None and (best is None or v>best[0]): best=(v,di,a,ans)
                if best is None: break
                used.add(best[1]); toks.append((best[2],best[3]))
            v=ndcg(enc_u_tokens(toks),tlike,set(prof),tail) if toks else ndcg(np.zeros(D),tlike,set(prof),tail)
        if v is not None: acc+=v;m+=1
    return acc/m
print(f"E2' direct-embedding-query ceiling (N={len(TE)}, TAU={TAU})",flush=True)
for tail in [False,True]:
    print(f"=== {'TAIL' if tail else 'FULL'} NDCG@10 ===",flush=True)
    print(f"  pop(q0)           : {run('item',0,tail):.3f}",flush=True)
    for K in [1,2,4]:
        print(f"  oracle-ITEM   @{K}  : {run('item',K,tail):.3f}    direct-EMB @{K} : {run('dir',K,tail):.3f}",flush=True)
    print(f"  full-profile fold : {run('full',0,tail):.3f}",flush=True)
