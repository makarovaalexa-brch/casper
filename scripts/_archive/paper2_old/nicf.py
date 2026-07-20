"""
NICF baseline (Zou, Chen, Chen, Yang, Ye, Du, Zha; "Neural Interactive Collaborative Filtering", SIGIR 2020).
The canonical DEEP-RL cold-start interviewer. Faithful traits vs our policy-gradient method:
  - VALUE-BASED Q-learning (DQN: target network, eps-greedy, bootstrapped TD)  [we use REINFORCE]
  - a LEARNED self-attentive state encoder over the (entity, answer) interaction history  [we reuse the frozen recommender belief]
  - cumulative-relevance reward (here: per-turn held-out coverage gain, same reward as O12 for apples-to-apples)
Action space = the SAME unified item+concept pool as our policy; recommendations scored on the IDENTICAL ruler
(frozen concept-aware encoder fold-in, full-catalogue NDCG@10/Rec@50, full+tail, q-curve). Checkpointed (LOAD=1 to re-eval).
"""
import os, time, copy, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; rng=np.random.default_rng(0); torch.manual_seed(0)
NEP=int(os.environ.get('NEP',25)); GAMMA=0.95
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
t0=time.time(); tagitems={}; tagset=set(int(t) for t in ctags)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1])
        if m in iids and tg in tagset and float(a[2])>0.5: tagitems.setdefault(tg,[]).append(iids[m])
citems=[set(tagitems.get(int(t),[])) for t in ctags]; cfreq=np.array([len(s) for s in citems]); NC=len(ctags)
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems.get(int(t),[]): item2c[j].append(ki)
pe_cnt=np.zeros(NC)
for x in trU[:2500]:
    cn={}
    for j,_ in rat_by_u[x]:
        for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
    for ki,c2 in cn.items():
        if c2>=2: pe_cnt[ki]+=1
pe_ans=(pe_cnt/min(len(trU),2500)).astype(np.float32)+1e-6; p_seen=(cnt/max(len(trU),1)).astype(np.float32)
print(f"  {NC} concepts ({time.time()-t0:.0f}s)",flush=True)
PITEMS=list(order_pop[:600]); NI=len(PITEMS); NP=NI+NC
POOL=np.concatenate([Q[np.array(PITEMS)], Ec],0).astype(np.float32); PTYPE=np.array([0]*NI+[1]*NC)
PRIOR=np.concatenate([np.log(p_seen[np.array(PITEMS)]+1e-4), np.log(pe_ans+1e-4)]).astype(np.float32); PRIOR=(PRIOR-PRIOR.mean())/(PRIOR.std()+1e-6)
POOLt=torch.tensor(POOL); PRIORt=torch.tensor(PRIOR); Qlt=torch.tensor(Ql); popbt=torch.tensor(popb)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def cans_np(x,profset):
    ac=[c for c in range(NC) if len(citems[c]&profset)>=2]; ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset])
    pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())) if pr else 0.; return {c:(POS if pr[c]>thr else NEG) for c in ac}
# population info-gain prior per pool entity (reuse cache from continuous_policy2)
IGC=f'{base}/.cache/pool_ig2.npy'; POOL_IG=np.load(IGC) if os.path.exists(IGC) else np.zeros(NP)
POOL_IG=((POOL_IG-POOL_IG.mean())/(POOL_IG.std()+1e-6)).astype(np.float32); POOL_IGt=torch.tensor(POOL_IG)
ML=64
def nicf_prep(users):
    B=len(users); ANSVAL=torch.zeros(B,NP); ANSMASK=torch.zeros(B,NP); LIKED=torch.zeros(B,ML,dtype=torch.long); LMASK=torch.zeros(B,ML)
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=set(allit[:max(len(allit)//2,4)]); _hl=[j for j in likes_by_u[x] if j not in prof]; held=(([j for j in _hl if not headmask[j]] or _hl) if os.environ.get('TAILREW','1')!='0' else _hl) or likes_by_u[x][:1]   # TAILREW(default 1): head-masked held likes -> reward is NOT popularity-saturated (popb already covers HEAD likes -> ~0 signal -> DQN can't learn); tail gives a learnable gradient
        for k,j in enumerate(PITEMS):
            if j in prof: ANSVAL[b,k]=resid[x][j]; ANSMASK[b,k]=1.
        for cc,v in cans_np(x,prof).items(): ANSVAL[b,NI+cc]=v; ANSMASK[b,NI+cc]=1.
        for hi,j in enumerate(held[:ML]): LIKED[b,hi]=j; LMASK[b,hi]=1.
    return ANSVAL,ANSMASK,LIKED,LMASK
def coverage(u,LIKED,LMASK):
    QlL=Qlt[LIKED]; popL=popbt[LIKED]; s=torch.sigmoid((u.unsqueeze(1)*QlL).sum(2)+popL); return (s*LMASK).sum(1)/LMASK.sum(1).clamp(min=1)
class NICF(nn.Module):                                                            # self-attentive history encoder + per-candidate Q-head
    def __init__(s):
        super().__init__(); s.tok=nn.Linear(D+1,D); s.attn=nn.MultiheadAttention(D,4,batch_first=True); s.ln=nn.LayerNorm(D)
        s.start=nn.Parameter(torch.zeros(D)); s.qh=nn.Sequential(nn.Linear(D+4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def encode(s,toks,tmask):
        B=toks.shape[0]; h=s.tok(toks); empty=(tmask.sum(1)==0); kpm=(tmask==0).clone(); kpm[empty,0]=False  # avoid all-masked rows -> NaN
        a,_=s.attn(h,h,h,key_padding_mask=kpm); h=s.ln(h+a)
        state=(h*tmask.unsqueeze(2)).sum(1)/tmask.sum(1,keepdim=True).clamp(min=1)
        return torch.where(empty.unsqueeze(1), s.start.unsqueeze(0).expand(B,D), state)
    def qval(s,state,tt):
        B=state.shape[0]; align=(state@POOLt.t()).unsqueeze(2); emb=POOLt.unsqueeze(0).expand(B,NP,D)
        pri=PRIORt.view(1,NP,1).expand(B,NP,1); ig=POOL_IGt.view(1,NP,1).expand(B,NP,1); tn=torch.full((B,NP,1),float(tt))
        return s.qh(torch.cat([emb,align,pri,ig,tn],2)).squeeze(2)
net=NICF(); tgt=copy.deepcopy(net); opt=torch.optim.Adam(net.parameters(),1e-3); huber=nn.SmoothL1Loss()
_CK=f'{base}/.cache/nicf.pt'
if os.environ.get('LOAD') and os.path.exists(_CK):
    net.load_state_dict(torch.load(_CK)); print(f"LOADED {_CK} -- skip training",flush=True); NEP=0
trbig=[x for x in trU if len(rat_by_u[x])>=8][:1500]
print(f"train NICF DQN (pool={NP}, {NEP} epochs)...",flush=True)
for ep in range(NEP):
    rng.shuffle(trbig); eps=max(0.05, 0.5-0.45*ep/max(NEP-1,1)); tot=0.;nb=0
    for b0 in range(0,len(trbig),64):
        users=trbig[b0:b0+64]; B=len(users); ANSVAL,ANSMASK,LIKED,LMASK=nicf_prep(users)
        toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); asked=torch.zeros(B,NP)
        prev=coverage(torch.zeros(B,D),LIKED,LMASK); states=[];acts=[];rews=[];asks=[]
        for t in range(T):
            st=toks.clone(); stm=tmask.clone(); asks.append(asked.clone()); states.append((st,stm))
            with torch.no_grad():
                a=net.qval(net.encode(st,stm),t/8.).masked_fill(asked>0,-1e9).argmax(1)
            for b in range(B):
                if rng.random()<eps:
                    av=(asked[b]==0).nonzero().squeeze(1); a[b]=int(av[rng.integers(len(av))])
                k=int(a[b]); asked[b,k]=1.
                if ANSMASK[b,k]>0:
                    pos=int(tmask[b].sum().item()); toks[b,pos,:D]=POOLt[k]; toks[b,pos,D]=ANSVAL[b,k]; tmask[b,pos]=1.
            with torch.no_grad():
                ub=enc(toks,tmask); ub[tmask.sum(1)==0]=0.
            cov=coverage(ub,LIKED,LMASK); rews.append(cov-prev); prev=cov; acts.append(a.clone())
        loss=0.
        for t in range(T):
            st,stm=states[t]; qa=net.qval(net.encode(st,stm),t/8.).gather(1,acts[t].unsqueeze(1)).squeeze(1)
            if t<T-1:
                with torch.no_grad():
                    st2,stm2=states[t+1]; q2=tgt.qval(tgt.encode(st2,stm2),(t+1)/8.).masked_fill(asks[t+1]>0,-1e9).max(1).values
                target=rews[t]+GAMMA*q2
            else: target=rews[t]
            loss=loss+huber(qa,target.detach())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); tot+=float(rews[-1].mean()+rews[0].mean()); nb+=1
        if nb%8==0: tgt.load_state_dict(net.state_dict())
    print(f"  ep{ep+1} eps={eps:.2f} cov~{tot/nb:.4f}",flush=True)
if NEP>0: torch.save(net.state_dict(),_CK); print(f"saved {_CK}",flush=True)
net.eval()
# ---- eval on the locked ruler ----
BETAC=f'{base}/.cache/rmse_beta.npy'                                              # RMSE rating head (cached BETA, shared across scripts)
if os.path.exists(BETAC) and not os.environ.get('REGEN'):
    BETA=float(np.load(BETAC))
else:
    num=den=0.
    for x in [x for x in trU if len(rat_by_u[x])>=10][:600]:
        allit=[j for j,_ in rat_by_u[x]]; kn=allit[:len(allit)//2]; hl=allit[len(allit)//2:]
        ux=enc_u_np([(Q[j],resid[x][j]) for j in kn])
        for j in hl: pr=float(ux@Ql[j]); num+=resid[x][j]*pr; den+=pr*pr
    BETA=num/den if den>1e-9 else 1.0; np.save(BETAC,np.array(BETA))
def rmse_of(u,held):
    if not held: return None
    return (sum((r-min(max(mu+bi[j]+BETA*float(u@Ql[j]),1.),5.))**2 for j,r in held.items())/len(held))**0.5
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    s=(popb+Ql@u).copy(); s[list(excl)]=-1e9
    if tail: s[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-s); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel); return nd,rc
_rs=np.random.default_rng(int(os.environ.get('SEED',123))); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
_tsel=[x for x in te if x in SPL]; _tr=os.environ.get('TESTRANGE','300:99999')        # HARMONIZED to continuous_policy2: te[300:] = representative disjoint test
TE=(_tsel[:150]+_tsel[400:]) if _tr=='clean' else (_tsel if _tr=='all' else _tsel[int(_tr.split(':')[0]):int(_tr.split(':')[1])])
def run(tail):
    M={q:0. for q in [0,2,4,8]};Rc={q:0. for q in [0,2,4,8]};RM={q:0. for q in [0,2,4,8]};m=0;na=0.;ni_=0;nc_=0;FP=[]
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); held={j:rd[j] for j in test}
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        cans=cans_np(x,profset); toks=[]; asked=set(); nq=0; nans=0; first=None
        for q in [0,2,4,8]:
            while nq<q:
                ar=np.zeros((1,max(len(toks),1),D+1),np.float32); mk=np.zeros((1,max(len(toks),1)),np.float32)
                for qi,(f,v) in enumerate(toks): ar[0,qi,:D]=f; ar[0,qi,D]=v; mk[0,qi]=1
                with torch.no_grad():
                    state=net.encode(torch.tensor(ar),torch.tensor(mk)); sc=net.qval(state,nq/8.).numpy()[0]
                for k in np.argsort(-sc):
                    if k not in asked: break
                if first is None: first=int(k)
                asked.add(int(k)); nq+=1
                if PTYPE[k]==0:
                    j=PITEMS[k]
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; ni_+=1
                else:
                    cc=k-NI
                    if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
            uu=enc_u_np(toks); mt=metr(uu,tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
            rv=rmse_of(uu,held); RM[q]+=(rv if rv is not None else 0.)
        m+=1; na+=nans
        if first is not None: FP.append(first)
    extra=f" | picks {ni_}i/{nc_}c | distinct-1st {len(set(FP))}/{max(len(FP),1)}"
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},{q:RM[q]/m for q in RM},na/m,extra
for tail in [False,True]:
    Mp,Rcp,RMp,na,extra=run(tail); print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | NICF (DQN) | NDCG@10 / Rec@50 / RMSE ===",flush=True)
    print(f"  nicf     : NDCG "+" ".join(f"{Mp[q]:.3f}" for q in [0,2,4,8])+" | Rec "+" ".join(f"{Rcp[q]:.3f}" for q in [0,2,4,8])+" | RMSE "+" ".join(f"{RMp[q]:.3f}" for q in [0,2,4,8])+f" | ans/{T}={na:.1f}{extra}",flush=True)
    print(f"  (compare: conc_pop FULL 0.315/TAIL 0.116 ; pop_item 0.307/0.085 ; q0 0.293/0.088)",flush=True)
