"""
PAPER B policy v2 (user's fix): give the policy the POPULARITY signal it was missing (population prior, NOT user-specific
-> no leak), PLUS the belief from revealed answers. A learned SCORER scores each entity in the unified item+concept pool
from per-candidate features [embedding, belief-alignment u.E, popularity-prior, belief-strength, turn]; soft-pick (train)
/ argmax (eval) over the pool; fold the answer; trained end-to-end on RECONSTRUCTION. Should match conc_pop (use
popularity when belief weak) AND personalize (use belief as answers accumulate) => beat conc_pop. Frozen unified model.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; TAU=float(os.environ.get('TAU',0.3)); EP=int(os.environ.get('EP',14)); rng=np.random.default_rng(0); torch.manual_seed(0)
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
PRIOR=np.concatenate([np.log(p_seen[np.array(PITEMS)]+1e-4), np.log(pe_ans+1e-4)]).astype(np.float32); PRIOR=(PRIOR-PRIOR.mean())/(PRIOR.std()+1e-6)  # POPULARITY/answerability prior (population-level)
POOLt=torch.tensor(POOL); PRIORt=torch.tensor(PRIOR); Qlt=torch.tensor(Ql); popbt=torch.tensor(popb)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
def sig(z): return 1/(1+np.exp(-z))
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_batch_np(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def cans_np(x,profset):
    ac=[c for c in range(NC) if len(citems[c]&profset)>=2]; ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset])
    pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())) if pr else 0.; return {c:(POS if pr[c]>thr else NEG) for c in ac}
# POPULATION-AVERAGE info-gain per pool entity (offline; population-level, no per-user leak): avg single-fold coverage-gain of an answerer's own profile
IGC=f'{base}/.cache/pool_ig2.npy'
if not os.path.exists(IGC) or os.environ.get('REGEN'):
    print("computing population info-gain per entity...",flush=True); igv=np.zeros(NP); cig=np.zeros(NP); samp=[x for x in trU if len(rat_by_u[x])>=10][:600]
    for x in samp:
        prof=[j for j,_ in rat_by_u[x]]; profset=set(prof); profa=np.array(prof); base_cov=sig(popb[profa]).sum(); ents=[];idxs=[]
        for k,j in enumerate(PITEMS):
            if j in profset: ents.append((Q[j],resid[x][j])); idxs.append(k)
        for c,v in cans_np(x,profset).items(): ents.append((Ec[c],v)); idxs.append(NI+c)
        if not ents: continue
        ul=enc_batch_np([[e] for e in ents]); cov=sig(popb[profa]+ul@Ql[profa].T).sum(1)-base_cov
        for q2,k in enumerate(idxs): igv[k]+=cov[q2]; cig[k]+=1
    POOL_IG=(igv/np.maximum(cig,1)).astype(np.float32); np.save(IGC,POOL_IG)
else: POOL_IG=np.load(IGC)
POOL_IG=((POOL_IG-POOL_IG.mean())/(POOL_IG.std()+1e-6)).astype(np.float32); POOL_IGt=torch.tensor(POOL_IG)
class Scorer(nn.Module):                                                      # per-candidate score from [emb, align(u.E), popularity prior, POPULATION info-gain, belief-strength, turn]
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+5,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(s,u,tt):                                                      # u (B,D), tt scalar turn frac -> scores (B,NP)
        B=u.shape[0]; align=(u@POOLt.t()).unsqueeze(2)                        # (B,NP,1) belief alignment
        bnorm=u.norm(dim=1,keepdim=True).unsqueeze(1).expand(B,NP,1)          # belief strength
        emb=POOLt.unsqueeze(0).expand(B,NP,D); pri=PRIORt.view(1,NP,1).expand(B,NP,1); ig=POOL_IGt.view(1,NP,1).expand(B,NP,1); tn=torch.full((B,NP,1),tt)
        feat=torch.cat([emb,align,pri,ig,bnorm,tn],2); return s.f(feat).squeeze(2)
scorer=Scorer(); opt=torch.optim.Adam(scorer.parameters(),1e-3)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
def prep(users):
    B=len(users); ANS=torch.zeros(B,NP); tgt=torch.zeros(B,ni); wt=torch.zeros(B,ni)
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=set(allit[:max(len(allit)//2,4)]); held=[j for j in likes_by_u[x] if j not in prof] or likes_by_u[x][:1]
        for k,j in enumerate(PITEMS):
            if j in prof: ANS[b,k]=resid[x][j]
        cans=cans_np(x,prof)
        for c,v in cans.items(): ANS[b,NI+c]=v
        posw=0.
        for j in held: tgt[b,j]=1.; wt[b,j]=float(ipsw[j]); posw+=float(ipsw[j])
        seen=prof; nneg=ni-len(seen)-len(held); m=torch.ones(ni); m[list(seen)]=0; wb=wt[b]; wb[(tgt[b]==0)&(m>0)]=float(posw)/max(nneg,1); wt[b]=wb
    return ANS,tgt,wt
def rollout(users,ANS,explore=0.0):
    B=len(users); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D); asked=torch.zeros(B,NP)
    for t in range(T):
        sc=scorer(u,t/8.)                                                    # (B,NP) popularity+belief-aware scores
        if explore>0: sc=sc+explore*torch.randn_like(sc)
        sc=sc.masked_fill(asked>0,-1e9)                                      # no re-asking (matches eval)
        w_soft=torch.softmax(sc/TAU,1); idx=w_soft.argmax(1)
        w_hard=torch.zeros_like(w_soft).scatter_(1,idx.unsqueeze(1),1.0)
        w=w_hard+(w_soft-w_soft.detach())                                    # STRAIGHT-THROUGH: fold the single PICKED entity (hard, == eval), soft gradient
        asked=asked+w_hard
        a_used=w@POOLt; ans=(w*ANS).sum(1)
        toks=toks.clone(); toks[:,t,:D]=a_used; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask)
    return u
def recon(u,tgt,wt): s=u@Qlt.t()+popbt; bce=nn.functional.binary_cross_entropy_with_logits(s,tgt,reduction='none'); return (wt*bce).sum()/wt.sum()
trbig=[x for x in trU if len(rat_by_u[x])>=14 and len(likes_by_u[x])>=6]
print(f"train popularity-aware scorer policy (pool={NP}, TAU={TAU})...",flush=True)
for ep in range(EP):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),96):
        us=trbig[b0:b0+96]; ANS,tgt,wt=prep(us); u=rollout(us,ANS,explore=0.1); loss=recon(u,tgt,wt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(scorer.parameters(),5.0); opt.step(); tot+=loss.item();nb+=1
    print(f"  ep{ep+1} recon={tot/nb:.4f}",flush=True)
scorer.eval()
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
def run(mode,tail):
    M={q:0. for q in [0,2,4,8]};Rc={q:0. for q in [0,2,4,8]};m=0;na=0.;ni_=0;nc_=0
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        cans=cans_np(x,profset) if mode in('policy','conc_pop') else {}; toks=[]; asked=set(); nq=0; nans=0
        for q in [0,2,4,8]:
            while nq<q:
                if mode=='policy':
                    with torch.no_grad(): sc=scorer(torch.tensor(enc_u_np(toks)[None],dtype=torch.float32),len(toks)/8.).numpy()[0]
                    for k in np.argsort(-sc):
                        if k not in asked: break
                    asked.add(int(k)); nq+=1
                    if PTYPE[k]==0:
                        j=PITEMS[k]
                        if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; ni_+=1
                    else:
                        cc=k-NI
                        if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
                elif mode=='conc_pop':
                    ci=[c for c in range(NC) if c not in asked]; cc=max(ci,key=lambda c:cfreq[c]); asked.add(cc); nq+=1
                    if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1
                elif mode=='pop_item':
                    cs=[j for j in PITEMS if j not in asked]; j=cs[0]; asked.add(j); nq+=1
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
            mt=metr(enc_u_np(toks),tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1; na+=nans
    extra=f" | picks {ni_}i/{nc_}c" if mode=='policy' else ""
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},na/m,extra
for tail in [False,True]:
    print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | popularity-aware scorer | NDCG@10 / Rec@50 / ans ===",flush=True)
    for mode in ['pop_item','conc_pop','policy']:
        Mp,Rcp,na,extra=run(mode,tail); print(f"  {mode:<9}: NDCG "+" ".join(f"{Mp[q]:.3f}" for q in [0,2,4,8])+" | Rec "+" ".join(f"{Rcp[q]:.3f}" for q in [0,2,4,8])+f" | ans/{T}={na:.1f}{extra}",flush=True)
    print(f"  GATE: policy must beat conc_pop @q8 (full+tail)",flush=True)
