"""
PAPER B — CONTINUOUS policy via PATHWISE GRADIENTS (the real continuous method; not concepts, not EIG-imitation, not
REINFORCE). The action is a RAW embedding a in R^D. The user model rates it directly: answer(a)=sim-weighted residual
over the user's ask-set (DIFFERENTIABLE). Fold (a, answer) via the frozen encoder (DIFFERENTIABLE). Reconstruction loss
on held-out target likes is backprop'd straight through the multi-turn rollout to the actor. So the actor DISCOVERS
embeddings that best reconstruct the user -- free to leave any vocab. Eval: tail NDCG@10 / Recall@50 vs EIG/random/oracle.
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=int(os.environ.get('T',6)); EP=int(os.environ.get('EP',12)); rng=np.random.default_rng(0); torch.manual_seed(0)
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
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=8]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=-1.0
Qt=torch.tensor(Q); Qlt=torch.tensor(Ql); popbt=torch.tensor(popb); ipswt=torch.tensor(ipsw)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+2,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,D))
    def forward(s,u,t): return s.f(torch.cat([u,t],1))
actor=Actor(); opt=torch.optim.Adam(actor.parameters(),3e-4)
# ---- per-user ask-set (factors+resid) and reconstruction target ----
def prep(users):
    KA=12; Fa=torch.zeros(len(users),KA,D); Ya=torch.zeros(len(users),KA); Ma=torch.zeros(len(users),KA)
    tgt=torch.zeros(len(users),ni); wt=torch.zeros(len(users),ni)
    for b,x in enumerate(users):
        lk=likes_by_u[x][:]; rng.shuffle(lk); allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit)
        ask=allit[:KA]                                       # ask-set (any rated items)
        for q,j in enumerate(ask): Fa[b,q]=Qt[j]; Ya[b,q]=resid[x][j]; Ma[b,q]=1
        tg=[j for j in lk if j not in set(ask)]              # target = held-out likes
        if not tg: tg=lk[:1]
        posw=0.
        for j in tg: tgt[b,j]=1.; wt[b,j]=float(ipsw[j]); posw+=float(ipsw[j])
        seen=set(ask); nneg=ni-len(seen)-len(tg)
        m=torch.ones(ni); m[list(seen)]=0
        wtb=wt[b]; wtb[(tgt[b]==0)&(m>0)]=float(posw)/max(nneg,1); wt[b]=wtb
    return Fa,Ya,Ma,tgt,wt
def rollout(actor, Fa,Ya,Ma, Tn, explore=0.0):
    B=Fa.shape[0]; toks=torch.zeros(B,Tn,D+1); tmask=torch.zeros(B,Tn); u=torch.zeros(B,D)
    Fan=Fa/ (Fa.norm(dim=2,keepdim=True)+1e-9)
    for t in range(Tn):
        tt=torch.full((B,1),t/8.); tt2=torch.full((B,1),float(t)); a=actor(u,torch.cat([tt,tt2],1))
        if explore>0: a=a+explore*torch.randn_like(a)
        an=a/(a.norm(dim=1,keepdim=True)+1e-9)
        sims=torch.relu((Fan*an.unsqueeze(1)).sum(2))*Ma                      # (B,KA) differentiable alignment
        ans=(sims*Ya).sum(1)/(sims.sum(1)+1e-6)                               # (B,) sim-weighted residual answer
        toks=toks.clone(); toks[:,t,:D]=a; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1
        u=enc(toks,tmask)
    return u
def recon_loss(u,tgt,wt):
    sc=u@Qlt.t()+popbt                                                       # (B,ni)
    bce=nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')
    return (wt*bce).sum()/wt.sum()
trbig=[x for x in trU if len(likes_by_u[x])>=6 and len(rat_by_u[x])>=14]
print(f"train differentiable continuous actor ({len(trbig)} users, T={T})...",flush=True)
for ep in range(EP):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),128):
        us=trbig[b0:b0+128]; Fa,Ya,Ma,tgt,wt=prep(us)
        u=rollout(actor,Fa,Ya,Ma,T,explore=0.1); loss=recon_loss(u,tgt,wt)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item();nb+=1
    print(f"  ep{ep+1} recon-loss={tot/nb:.4f}",flush=True)
actor.eval()
# ---- eval vs EIG/random/oracle on held-out, tail+full ----
def sig(z): return 1/(1+np.exp(-z))
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-sc); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel); return nd,rc
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=8: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:200]
def actor_pick_dir(u, ask_fac, ask_res, t):
    with torch.no_grad():
        a=actor(torch.tensor(u[None],dtype=torch.float32),torch.tensor([[t/8.,float(t)]])).numpy()[0]
    an=a/(np.linalg.norm(a)+1e-9); Fn=ask_fac/(np.linalg.norm(ask_fac,axis=1,keepdims=True)+1e-9); sims=np.maximum(Fn@an,0)
    ans=float((sims*ask_res).sum()/(sims.sum()+1e-6)); return a,ans
def run(mode,tail):
    M={q:0. for q in [0,1,2,4,8]};Rc={q:0. for q in [0,1,2,4,8]};m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        ask_fac=np.array([Q[j] for j in prof]); ask_res=np.array([rd[j]-mu-bi[j] for j in prof])
        toks=[]; asked=[]
        for q in [0,1,2,4,8]:
            while len(toks)<q:
                if mode=='actor':
                    a,ans=actor_pick_dir(enc_u_np(toks),ask_fac,ask_res,len(toks)); toks.append((a,ans))
                else:
                    cset=[j for j in prof if j not in asked]
                    if not cset: break
                    if mode=='random': pick=cset[int(rng.integers(len(cset)))]
                    elif mode=='item':                                        # EIG: coverage over rest-of-profile
                        ul=enc_u_batch([toks+[(Q[j],rd[j]-mu-bi[j])] for j in cset]); val=sig(popb[np.array(prof)]+ul@Ql[prof].T).sum(1); pick=cset[int(val.argmax())]
                    else:                                                     # oracle: peek at held-out
                        ul=enc_u_batch([toks+[(Q[j],rd[j]-mu-bi[j])] for j in cset]); best=None
                        for li,j in enumerate(cset):
                            mt=metr(ul[li],tlike,set(prof),tail); a=mt[0] if mt else -1
                            if best is None or a>best[0]: best=(a,j)
                        pick=best[1]
                    asked.append(pick); toks.append((Q[pick],rd[pick]-mu-bi[pick]))
            mt=metr(enc_u_np(toks),tlike,set(prof),tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},m
for tail in [False,True]:
    print(f"\n=== {'TAIL' if tail else 'FULL'}: NDCG@10 / Recall@50 (continuous actor vs EIG) ===",flush=True)
    for mode in ['random','item','actor','oracle']:
        M,Rc,m=run(mode,tail); print(f"  {mode:<7}: NDCG "+" ".join(f"{M[q]:.3f}" for q in [0,1,2,4,8])+" | Rec "+" ".join(f"{Rc[q]:.3f}" for q in [0,1,2,4,8]),flush=True)
