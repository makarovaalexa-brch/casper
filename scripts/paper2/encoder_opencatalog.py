"""
OPEN-CATALOGUE deployability check: the policy asks about ANY catalogue item (not just the user's rated set).
If the picked item is rated by the user -> fold the true answer; else "don't know" -> no fold, turn still consumed.
Budget = q ASKED turns. This is the hardest, most realistic setting (most questions get "don't know").
Policies (all belief-only, deployable):
  random_cat : ask a random catalogue item
  pop_cat    : ask the most-popular unasked item (high answer-rate, low info)
  eig_cat    : pick item maximizing EXPECTED belief shift  E_ans[||u_next - u_now||]  (max info, answerability-blind)
  eig_ans    : answerability-aware = eig_cat * P(rated|popularity)  (balances info vs being answerable)
Report full+tail NDCG@10 and the answer-rate. Question: does an adaptive catalogue policy still beat random/pop?
Candidate vocabulary = top-1200 popular items (the realistically askable set; niche items are ~never answerable).
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; K=int(os.environ.get('K',12)); rng=np.random.default_rng(0); torch.manual_seed(0)
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
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Qt=torch.tensor(Q)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1])
headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
POOL=list(order_pop[:1200])                                  # askable vocabulary
prated=(cnt/cnt.max()).astype(np.float32)                    # P(rated|popularity) proxy, in [0,1]
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4])
NEG=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); opt=torch.optim.Adam(list(enc.parameters())+[Qp],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def make_batch(users,kk):
    toks=np.zeros((len(users),kk,D+1),np.float32); msk=np.zeros((len(users),kk),np.float32)
    tgt=np.zeros((len(users),ni),np.float32); wt=np.ones((len(users),ni),np.float32); seen=np.zeros((len(users),ni),bool)
    for b,x in enumerate(users):
        rv=[(j,resid_by_u[x][j]) for j,_ in rat_by_u[x]]; rng.shuffle(rv); rev=rv[:kk]
        for q,(j,res) in enumerate(rev): toks[b,q,:D]=Q[j]; toks[b,q,D]=res; msk[b,q]=1; seen[b,j]=True
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.0; wt[b,j]=ipsw[j]; posw+=ipsw[j]
        nneg=ni-int(seen[b].sum())-int(tgt[b].sum())
        if nneg>0: wt[b][(tgt[b]==0)&(~seen[b])]=posw/nneg
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
print("train encoder (joint)...",flush=True)
for ep in range(int(os.environ.get('EP',30))):
    rng.shuffle(trbig)
    for b0 in range(0,len(trbig),256):
        us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk); sc=u@Qp.t()
        loss=(wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')).masked_fill(seen,0.).mean()
        opt.zero_grad(); loss.backward(); opt.step()
enc.eval(); Ql=Qp.detach().numpy()
_W=1./np.log2(np.arange(2,12))
def ndcg_at(u,tlike,excl,tailonly):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tailonly: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12) if rel else None
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(j,res) in enumerate(rev): arr[b,q,:D]=Q[j]; arr[b,q,D]=res; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def enc_u(rev): return enc_u_batch([rev])[0] if rev else np.zeros(D)
def sig(z): return 1/(1+np.exp(-z))
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def pick(sel,rev,asked,rd):
    cands=[j for j in POOL if j not in asked]
    if sel=='random_cat': return cands[int(rng.integers(len(cands)))]
    if sel=='pop_cat': return max(cands,key=lambda j:cnt[j])
    u_now=enc_u(rev); cl=np.array(cands)
    ul=enc_u_batch([rev+[(c,POS)] for c in cands]); ud=enc_u_batch([rev+[(c,NEG)] for c in cands])
    p=sig(popb[cl]+Ql[cl]@u_now)
    shift=p*np.linalg.norm(ul-u_now,axis=1)+(1-p)*np.linalg.norm(ud-u_now,axis=1)   # expected belief shift
    if sel=='eig_ans': shift=shift*prated[cl]                                          # answerability-aware
    return cands[int(shift.argmax())]
def run(sel,tailonly,NEVAL=200):
    acc={q:0. for q in [1,2,4,8]}; ans={q:0. for q in [1,2,4,8]}; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); profset=set(prof)
        if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
        asked=[]; rev=[]; nans=0
        for turn in range(1,9):
            c=pick(sel,rev,asked,rd); asked.append(c)
            if c in profset:                          # user can answer only about held-IN profile items (no test leak)
                rev.append((c,rd[c]-mu-bi[c])); nans+=1
            if turn in (1,2,4,8):
                v=ndcg_at(enc_u(rev),tlike,profset,tailonly)
                if v is not None: acc[turn]+=v; ans[turn]+=nans
        m+=1
    return {q:acc[q]/m for q in acc}, {q:ans[q]/m for q in ans}, m
for tailonly in [False,True]:
    print(f"\n=== {'TAIL' if tailonly else 'FULL'} NDCG@10 — OPEN-CATALOGUE (q = asked turns; #ans = answered) ===",flush=True)
    for sel in ['random_cat','pop_cat','eig_cat','eig_ans']:
        r,a,m=run(sel,tailonly); print(f"  {sel:<11}: "+" ".join(f"q{q}={r[q]:.3f}(ans{a[q]:.1f})" for q in [1,2,4,8])+f" | n={m}",flush=True)
