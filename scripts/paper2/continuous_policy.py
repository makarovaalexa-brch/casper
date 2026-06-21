"""
PAPER B HEADLINE (Claim 4): UNIFIED CONTINUOUS item+concept elicitation policy. One actor emits a query vector a in the
shared 64-d space; soft-snaps (differentiable, training) / hard-snaps (eval) to the nearest entity in a UNIFIED pool of
items AND concepts (all levers available). Trained end-to-end on RECONSTRUCTION (metric-aligned, not the popb-coverage
proxy that made greedy lose). Realistic cold-start answerability: item answered iff seen, concept iff >=2 items;
unanswerable pick folds a near-zero token (no signal) so the actor LEARNS to prefer answerable questions. Frozen unified
model (enc_concept). GATE: beat conc_pop AND pop_item on full+tail.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; TAU=float(os.environ.get('TAU',0.3)); EP=int(os.environ.get('EP',12)); rng=np.random.default_rng(0); torch.manual_seed(0)
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
citems=[set(tagitems.get(int(t),[])) for t in ctags]; cfreq=np.array([len(s) for s in citems])
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems.get(int(t),[]): item2c[j].append(ki)
pe_cnt=np.zeros(NC if False else len(ctags))                                 # population concept answerability prior P(user can answer c)
for x in trU[:2500]:
    cn={}
    for j,_ in rat_by_u[x]:
        for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
    for ki,c2 in cn.items():
        if c2>=2: pe_cnt[ki]+=1
pe_ans=(pe_cnt/min(len(trU),2500)).astype(np.float32)+1e-6
p_seen=(cnt/max(len(trU),1)).astype(np.float32)                              # item answerability prior P(user has seen item)
print(f"  {len(ctags)} concepts ({time.time()-t0:.0f}s)",flush=True)
# ---- UNIFIED POOL: top item-pool + all concepts ----
PITEMS=list(order_pop[:600]); NI=len(PITEMS); NC=len(ctags); NP=NI+NC
POOL=np.concatenate([Q[np.array(PITEMS)], Ec],0).astype(np.float32)          # (NP, D) unified entity embeddings
PTYPE=np.array([0]*NI+[1]*NC)                                                # 0=item, 1=concept
POOLt=torch.tensor(POOL); POOLn=POOLt/ (POOLt.norm(dim=1,keepdim=True)+1e-9)
Qlt=torch.tensor(Ql); popbt=torch.tensor(popb)
PRIORW=float(os.environ.get('PRIORW',2.0))                                   # steer toward answerable-LIKELY entities (population prior; not per-user peeking)
PP=np.concatenate([np.log(p_seen[np.array(PITEMS)]+1e-4), np.log(pe_ans+1e-4)]).astype(np.float32)
PP=(PP-PP.mean())/(PP.std()+1e-6); POOL_PRIOR=torch.tensor(PP); POOL_PRIOR_np=PP
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
# ---- per-user pool answers/answerability (training: profile known) ----
def pool_answers(x, profset):                                                # returns ans(NP), answerable-mask(NP) using known profile
    ans=np.zeros(NP,np.float32); msk=np.zeros(NP,np.float32)
    rd=dict(rat_by_u[x])
    for k,j in enumerate(PITEMS):
        if j in profset: ans[k]=resid[x][j]; msk[k]=1
    ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset])                     # true taste vector (geometric concept answer)
    pr=[]; cidx=[]
    for k in range(NC):
        inter=citems[k]&profset
        if len(inter)>=2: pr.append(float(ustar@Ec[k])); cidx.append(k)
    if pr:
        thr=np.mean(pr)
        for q,k in enumerate(cidx): ans[NI+k]=POS if pr[q]>thr else NEG; msk[NI+k]=1
    return ans, msk
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+2,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,D))
    def forward(s,u,tt): return s.f(torch.cat([u,tt],1))
actor=Actor(); opt=torch.optim.Adam(actor.parameters(),1e-3)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
# ---- Phase BC: distill conc_gprof (personalized teacher: greedy concept covering THIS user's profile) into the belief-only actor ----
GCACHE=f'{base}/.cache/gprof_traj.npz'
if not os.path.exists(GCACHE) or os.environ.get('REGEN'):
    print("gen conc_gprof teacher trajectories...",flush=True); S=[];TG=[];TT=[]
    samp=[x for x in trU if len(rat_by_u[x])>=14][:1200]
    for n_,x in enumerate(samp):
        prof=[j for j,_ in rat_by_u[x]]; profset=set(prof); profa=np.array(prof)
        ustar=enc_u_np([(Q[j],resid[x][j]) for j in prof]); ac=[c for c in range(NC) if len(citems[c]&profset)>=2]
        if len(ac)<3: continue
        pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())); cans={c:(POS if pr[c]>thr else NEG) for c in ac}
        acf=sorted(ac,key=lambda c:-cfreq[c])[:100]; toks=[]; asked=set()
        for t in range(T):
            ci=[c for c in acf if c not in asked]
            if not ci: break
            ul=enc_batch_np([toks+[(Ec[c],cans[c])] for c in ci]); cov=sig(popb[profa]+ul@Ql[profa].T).sum(1); cstar=ci[int(cov.argmax())]
            S.append(enc_u_np(toks).astype(np.float32)); TG.append(Ec[cstar].astype(np.float32)); TT.append(t); asked.add(cstar); toks.append((Ec[cstar],cans[cstar]))
        if (n_+1)%300==0: print(f"  gen {n_+1} ({len(S)} pairs)",flush=True)
    S=np.array(S);TG=np.array(TG);TT=np.array(TT); np.savez(GCACHE,S=S,TG=TG,TT=TT)
else:
    d=np.load(GCACHE); S,TG,TT=d['S'],d['TG'],d['TT']
print(f"  BC distil conc_gprof -> actor ({len(S)} pairs)...",flush=True)
St=torch.tensor(S); TGt=torch.tensor(TG); TTt2=torch.stack([torch.tensor(TT/8.,dtype=torch.float32),torch.tensor(TT.astype(np.float32))],1)
for ep in range(40):
    idx=torch.randperm(len(St))
    for b0 in range(0,len(St),512):
        bb=idx[b0:b0+512]; pred=actor(St[bb],TTt2[bb]); loss=((pred-TGt[bb])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
for g in opt.param_groups: g['lr']=3e-4                                       # lower LR for the recon finetune
def prep(users):
    B=len(users); ANS=torch.zeros(B,NP); MSK=torch.zeros(B,NP); tgt=torch.zeros(B,ni); wt=torch.zeros(B,ni)
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=set(allit[:max(len(allit)//2,4)]); held=[j for j in likes_by_u[x] if j not in prof] or likes_by_u[x][:1]
        ans,msk=pool_answers(x,prof); ANS[b]=torch.tensor(ans); MSK[b]=torch.tensor(msk)
        posw=0.
        for j in held: tgt[b,j]=1.; wt[b,j]=float(ipsw[j]); posw+=float(ipsw[j])
        seen=prof; nneg=ni-len(seen)-len(held); m=torch.ones(ni); m[list(seen)]=0
        wb=wt[b]; wb[(tgt[b]==0)&(m>0)]=float(posw)/max(nneg,1); wt[b]=wb
    return ANS,MSK,tgt,wt
def rollout(users,ANS,MSK,explore=0.0):                                       # differentiable soft-snap rollout
    B=len(users); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D)
    for t in range(T):
        tt=torch.cat([torch.full((B,1),t/8.),torch.full((B,1),float(t))],1); a=actor(u,tt)
        if explore>0: a=a+explore*torch.randn_like(a)
        an=a/(a.norm(dim=1,keepdim=True)+1e-9); sim=an@POOLn.t()+PRIORW*POOL_PRIOR           # cos to pool + population answerability prior (consistent train/eval; NO per-user mask)
        w=torch.softmax(sim/TAU,1)                                                           # soft-snap weights; unanswerable picks fold a 0-value token (ANS=0) -> actor learns to avoid
        a_used=w@POOLt; ans=(w*ANS).sum(1)                                                   # continuous proto-action + its (soft) answer
        toks=toks.clone(); toks[:,t,:D]=a_used; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask)
    return u
def recon(u,tgt,wt): sc=u@Qlt.t()+popbt; bce=nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none'); return (wt*bce).sum()/wt.sum()
trbig=[x for x in trU if len(rat_by_u[x])>=14 and len(likes_by_u[x])>=6]
print(f"train unified continuous policy (pool={NP}: {NI} items+{NC} concepts, TAU={TAU})...",flush=True)
for ep in range(EP):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),96):
        us=trbig[b0:b0+96]; ANS,MSK,tgt,wt=prep(us); u=rollout(us,ANS,MSK,explore=0.05); loss=recon(u,tgt,wt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.0); opt.step(); tot+=loss.item();nb+=1
    print(f"  ep{ep+1} recon={tot/nb:.4f}",flush=True)
actor.eval()
# ---- EVAL: realistic cold-start, hard-snap, real answers ----
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
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
TE=[x for x in te if x in SPL][:150]; R=order_pop[:500]
def emit(u,t):
    with torch.no_grad(): return actor(torch.tensor(u[None],dtype=torch.float32),torch.tensor([[t/8.,float(t)]])).numpy()[0]
def cans_of(x,profset):
    ac=[c for c in range(NC) if len(citems[c]&profset)>=2]; ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset])
    pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())) if pr else 0.; return {c:(POS if pr[c]>thr else NEG) for c in ac}
def infogain_concepts(toks,cidx,pe):
    u=enc_u_np(toks); ca=np.array(cidx); pc=sig(Ec[ca]@u); ul=enc_batch_np([toks+[(Ec[c],POS)] for c in cidx]); ud=enc_batch_np([toks+[(Ec[c],NEG)] for c in cidx])
    return pe[ca]*(pc*sig(ul@Ql[R].T).sum(1)+(1-pc)*sig(ud@Ql[R].T).sum(1))
def run(mode,tail):
    M={q:0. for q in [0,2,4,8]};Rc={q:0. for q in [0,2,4,8]};m=0;na=0.;nitem=0;ncc=0
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        cans=cans_of(x,profset) if mode in('policy','conc_pop','conc_eig','conc_gprof') else {}
        toks=[]; asked=set(); nq=0; nans=0
        for q in [0,2,4,8]:
            while nq<q:
                if mode=='policy':
                    a=emit(enc_u_np(toks),len(toks)); an=a/(np.linalg.norm(a)+1e-9); sims=POOLn.numpy()@an+PRIORW*POOL_PRIOR_np
                    for k in np.argsort(-sims):
                        if k not in asked: break
                    asked.add(int(k)); nq+=1
                    if PTYPE[k]==0:
                        j=PITEMS[k]
                        if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; nitem+=1
                    else:
                        c=k-NI
                        if c in cans: toks.append((Ec[c],cans[c])); nans+=1; ncc+=1
                elif mode=='conc_pop':
                    ci=[c for c in range(NC) if c not in asked]; c=max(ci,key=lambda c:cfreq[c]); asked.add(c); nq+=1
                    if c in cans: toks.append((Ec[c],cans[c])); nans+=1
                elif mode=='conc_eig':
                    ci=[c for c in range(NC) if c not in asked]; c=ci[int(infogain_concepts(toks,ci,pe_ans).argmax())]; asked.add(c); nq+=1
                    if c in cans: toks.append((Ec[c],cans[c])); nans+=1
                elif mode=='conc_gprof':                                                # personalized teacher (uses profile): the realizable-warm reference to match
                    ci=[c for c in range(NC) if c in cans and c not in asked]
                    if not ci: break
                    ci=sorted(ci,key=lambda c:-cfreq[c])[:150]; profa=np.array(list(profset))
                    ul=enc_batch_np([toks+[(Ec[c],cans[c])] for c in ci]); cov=sig(popb[profa]+ul@Ql[profa].T).sum(1)
                    c=ci[int(cov.argmax())]; asked.add(c); nq+=1; toks.append((Ec[c],cans[c])); nans+=1
                elif mode=='pop_item':
                    cs=[j for j in PITEMS if j not in asked]; j=cs[0]; asked.add(j); nq+=1
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
            mt=metr(enc_u_np(toks),tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1; na+=nans
    extra=f" | picks:{nitem}i/{ncc}c" if mode=='policy' else ""
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},na/m,extra
for tail in [False,True]:
    print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | unified continuous policy | NDCG@10 / Rec@50 / ans ===",flush=True)
    for mode in ['pop_item','conc_pop','conc_gprof','policy']:
        Mp,Rcp,na,extra=run(mode,tail); print(f"  {mode:<9}: NDCG "+" ".join(f"{Mp[q]:.3f}" for q in [0,2,4,8])+" | Rec "+" ".join(f"{Rcp[q]:.3f}" for q in [0,2,4,8])+f" | ans/{T}={na:.1f}{extra}",flush=True)
    print(f"  GATE: policy must beat conc_pop AND pop_item @q8",flush=True)
