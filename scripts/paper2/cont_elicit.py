"""
PC — CONTINUOUS elicitation (the CASPER thesis). The policy ACTOR emits a continuous embedding e in R^D that is NOT
snapped to any existing item/concept — it can be a novel MIDDLE-GROUND point (e.g. between "dark" and "comedy").
  - Answer: geometric, sign(u*.e - thr) — works for ANY point => fine-grained AND answerable (the gap items can't fill).
  - Fold e into the belief via the frozen concept-aware encoder; reward = held-out NDCG@10 gain (same ruler as the ladder).
  - Train: Gaussian policy gradient (REINFORCE) with exploration; WARM-START the actor from the frequency-concept order
    (a successful discrete policy) so it doesn't cold-collapse like the old DDPG actor.
  - DECODE: each emitted e -> nearest genome concepts (cosine) => the natural-language translation of the middle-ground.
Eval on the locked 150 split: NDCG@10/Rec@50 full+tail vs conc_pop, + a decode report (how often e is a blend vs snapped).
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; rng=np.random.default_rng(0); torch.manual_seed(0)
NEP=int(os.environ.get('NEP',25)); SIGMA=float(os.environ.get('SIGMA',0.3)); WARM=int(os.environ.get('WARM',6))
U,I,Rr=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U);I=np.array(I);Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(Rr[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); nK=len(keep); trU=keep[:int(0.8*nK)]; te=keep[int(0.9*nK):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_concept.npy'); Ec=np.load(f'{base}/.cache/Ec_concept.npy'); ctags=list(np.load(f'{base}/.cache/ctags_concept.npy'))
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
tagitems={}; tagset=set(int(t) for t in ctags)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1])
        if m in iids and tg in tagset and float(a[2])>0.5: tagitems.setdefault(tg,[]).append(iids[m])
citems=[set(tagitems.get(int(t),[])) for t in ctags]; cfreq=np.array([len(s) for s in citems]); NC=len(ctags)
tagname={}
try:
    with open(f'{base}/genome-tags.csv') as f:
        next(f)
        for line in f:
            a=line.strip().split(','); tagname[int(a[0])]=a[1]
except Exception: pass
cnames=[tagname.get(int(t),f'tag{int(t)}') for t in ctags]
print(f"  {NC} concepts",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
Ect=torch.tensor(Ec); Qlt=torch.tensor(Ql); popbt=torch.tensor(popb); cnorm=float(np.linalg.norm(Ec,axis=1).mean())   # typical concept-embedding norm
freq_order=list(np.argsort(-cfreq))                                                                                   # conc_pop order (warm-start target)
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
ML=64
def prep(users):
    B=len(users); USTAR=torch.zeros(B,D); THR=torch.zeros(B); LIKED=torch.zeros(B,ML,dtype=torch.long); LMASK=torch.zeros(B,ML); PROFM=torch.zeros(B,ni)
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=allit[:max(len(allit)//2,4)]; held=[j for j in likes_by_u[x] if j not in set(prof)] or likes_by_u[x][:1]
        us=enc_u_np([(Q[j],resid[x][j]) for j in prof]); USTAR[b]=torch.tensor(us); THR[b]=float((us@Ec.T).mean())   # true-taste vector + neutral threshold
        for j in prof: PROFM[b,j]=1.
        for hi,j in enumerate(held[:ML]): LIKED[b,hi]=j; LMASK[b,hi]=1.
    return USTAR,THR,LIKED,LMASK,PROFM
class Actor(nn.Module):                                                                # state (belief u, turn) -> continuous embedding (the query point)
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,D))
    def forward(s,u,tt):
        B=u.shape[0]; tn=tt.view(B,1) if torch.is_tensor(tt) else torch.full((B,1),float(tt)); return s.f(torch.cat([u,tn],1))
actor=Actor(); opt=torch.optim.Adam(actor.parameters(),1e-3); POSt=torch.tensor(float(POS)); NEGt=torch.tensor(float(NEG))
_disc=1.0/torch.log2(torch.arange(2,12).float()); _cd=torch.cumsum(_disc,0)
def scale(e): return e/(e.norm(dim=1,keepdim=True)+1e-9)*cnorm                          # put the emitted point on the concept-embedding scale
def fold_step(toks,tmask,t,e,ans):
    toks=toks.clone(); toks[:,t,:D]=e; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1; return toks,tmask,enc(toks,tmask)
def ndcg(u,LIKED,LMASK,PROFM,idcg):
    sc=(u@Qlt.t()+popbt).masked_fill(PROFM>0,-1e9); ti=sc.topk(10,1).indices
    hit=((ti.unsqueeze(2)==LIKED.unsqueeze(1))&(LMASK.unsqueeze(1)>0)).any(2).float(); return (hit*_disc).sum(1)/idcg.clamp(min=1e-6)
trbig=[x for x in trU if len(rat_by_u[x])>=14 and len(likes_by_u[x])>=6]
# ---- WARM-START: actor(u,t) -> embedding of the t-th frequency concept (mimic conc_pop) so continuous search starts competent ----
if WARM>0:
    print(f"warm-start actor -> freq-concept order ({WARM} ep)...",flush=True)
    wtarg=torch.tensor(np.stack([Ec[freq_order[t]] for t in range(T)]))                # (T,D) conc_pop targets per turn
    for ep in range(WARM):
        rng.shuffle(trbig)
        for b0 in range(0,len(trbig),96):
            us=trbig[b0:b0+96]; B=len(us); USTAR,THR,_,_,_=prep(us); u=torch.zeros(B,D); loss=0.
            toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T)
            for t in range(T):
                e=actor(u,t/8.); loss=loss+((e-wtarg[t].unsqueeze(0))**2).sum(1).mean()
                with torch.no_grad():
                    es=scale(e); ans=torch.where((USTAR*es).sum(1)>THR,POSt,NEGt)
                    toks,tmask,u=fold_step(toks,tmask,t,es,ans)
            opt.zero_grad(); (loss/T).backward(); opt.step()
    print("  warm-start done",flush=True)
for g in opt.param_groups: g['lr']=float(os.environ.get('FTLR',3e-4))
print(f"train continuous actor (REINFORCE, SIGMA={SIGMA}, NDCG reward)...",flush=True)
for ep in range(NEP):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),96):
        us=trbig[b0:b0+96]; B=len(us); USTAR,THR,LIKED,LMASK,PROFM=prep(us)
        idcg=_cd[(LMASK.sum(1).clamp(1,10).long()-1)]; toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D)
        sc0=popbt.unsqueeze(0).expand(B,ni).masked_fill(PROFM>0,-1e9); ti0=sc0.topk(10,1).indices
        prev=(((ti0.unsqueeze(2)==LIKED.unsqueeze(1))&(LMASK.unsqueeze(1)>0)).any(2).float()*_disc).sum(1)/idcg.clamp(min=1e-6)
        logps=[];rews=[]
        for t in range(T):
            mean_e=actor(u,t/8.); e_raw=(mean_e+SIGMA*torch.randn(B,D)).detach()         # sampled action (fixed); grad flows through mean_e in the log-density
            logps.append(-((e_raw-mean_e)**2).sum(1)/(2*SIGMA**2))                       # Gaussian score function (REINFORCE)
            with torch.no_grad():
                es=scale(e_raw); ans=torch.where((USTAR*es).sum(1)>THR,POSt,NEGt)
                toks,tmask,u=fold_step(toks,tmask,t,es,ans); cov=ndcg(u,LIKED,LMASK,PROFM,idcg); rews.append(cov-prev); prev=cov
        Gs=[None]*T; acc=torch.zeros(B)
        for t in reversed(range(T)): acc=rews[t]+acc; Gs[t]=acc.clone()
        loss=0.
        for t in range(T): adv=Gs[t]-Gs[t].mean(); loss=loss-(logps[t]*adv).mean()
        opt.zero_grad(); (loss/T).backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.0); opt.step(); tot+=Gs[0].mean().item(); nb+=1
    print(f"  ep{ep+1} return(over q0)={tot/nb:.4f}",flush=True)
torch.save(actor.state_dict(),f'{base}/.cache/actor_cont.pt'); print("saved actor_cont.pt",flush=True)
actor.eval()
# ---- eval on locked ruler + decode-to-words ----
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    s=(popb+Ql@u).copy(); s[list(excl)]=-1e9
    if tail: s[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-s); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel); return nd,rc
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
TE=[x for x in te if x in SPL][:150]
Ecn=Ec/(np.linalg.norm(Ec,axis=1,keepdims=True)+1e-9)
def decode(e):                                                                          # nearest genome concepts to a continuous point
    en=e/(np.linalg.norm(e)+1e-9); cs=Ecn@en; o=np.argsort(-cs)[:3]; return [(cnames[int(k)],round(float(cs[int(k)]),2)) for k in o]
for tail in [False,True]:
    M={q:0. for q in [0,2,4,8]};Rc={q:0. for q in [0,2,4,8]};m=0; blends=0; tot_e=0; examples=[]
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        us=enc_u_np([(Q[j],resid[x][j]) for j in profset]); thr=float((us@Ec.T).mean()); toks=[]; u=np.zeros(D)
        for q in [0,2,4,8]:
            while len(toks)<q:
                with torch.no_grad(): e=actor(torch.tensor(u[None],dtype=torch.float32),len(toks)/8.).numpy()[0]
                es=e/(np.linalg.norm(e)+1e-9)*cnorm; ans=POS if (us@es)>thr else NEG; toks.append((es,ans))
                if not tail:
                    nn3=decode(es); tot_e+=1; blends+= (1 if nn3[0][1]<0.9 else 0)            # cos<0.9 to nearest => a genuine middle-ground
                    if len(examples)<6: examples.append(nn3)
                u=enc_u_np(toks)
            mt=metr(enc_u_np(toks),tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1
    print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | CONTINUOUS actor | NDCG@10 / Rec@50 (vs conc_pop {0.315 if not tail else 0.116}) ===",flush=True)
    print(f"  cont  : NDCG "+" ".join(f"{M[q]/m:.3f}" for q in [0,2,4,8])+" | Rec "+" ".join(f"{Rc[q]/m:.3f}" for q in [0,2,4,8]),flush=True)
    if not tail:
        print(f"  DECODE: {blends}/{tot_e} emitted points are genuine middle-grounds (cos<0.9 to nearest concept)",flush=True)
        for ex in examples: print(f"    e ~ {ex}",flush=True)
