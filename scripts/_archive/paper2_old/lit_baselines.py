"""
LIT BASELINES (be true to lit; don't reinvent the wheel). Cold-start CONCEPT elicitation on the locked ruler
(concept-aware enc, ANSWER=geom, realistic cold-start, full+tail NDCG@10/Rec@50, q-curve). Implements the canonical
non-personalized active-learning heuristics + Golbandi's population optimizer:
  random ; pop (most-frequent concept) ; ENTROPY (Rashid 2002) ; POP*ENTROPY ; log(pop)*ENTROPY (Rashid 2002) ;
  HELF = harmonic mean of normalized entropy & normalized log-frequency (Rashid/Karypis/Riedl 2008) ;
  GREEDYEXTEND = static seed set greedily minimizing POPULATION reconstruction error (Golbandi/Koren/Lempel 2010).
vs conc_pop and q0. All concept answers = geometric (the user simulator).
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; rng=np.random.default_rng(0)
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
print(f"  {NC} concepts ({time.time()-t0:.0f}s)",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
def sig(z): return 1/(1+np.exp(-z))
def enc_u(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_batch(revs):
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def cans_of(x,profset):
    ac=[c for c in range(NC) if len(citems[c]&profset)>=2]; ustar=enc_u([(Q[j],resid[x][j]) for j in profset])
    pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())) if pr else 0.; return {c:(POS if pr[c]>thr else NEG) for c in ac}
# ---- population stats over train sample: like-rate, entropy, HELF (geometric answers) ----
print("population stats (entropy/HELF)...",flush=True); lc=np.zeros(NC); ac_=np.zeros(NC)
for x in [x for x in trU if len(rat_by_u[x])>=10][:2000]:
    profset=set(j for j,_ in rat_by_u[x]); cans=cans_of(x,profset)
    for c,v in cans.items(): ac_[c]+=1; lc[c]+= (1 if v>0 else 0)
lr=np.where(ac_>0, lc/np.maximum(ac_,1), 0.5); ent=-(lr*np.log(lr+1e-9)+(1-lr)*np.log(1-lr+1e-9))   # answer entropy
ent=np.where(ac_>0, ent, 0.)                                                                          # undefined (nobody can answer) -> rank last, not top (fair to naive entropy)
pe=ac_/max(1,len([x for x in trU if len(rat_by_u[x])>=10][:2000]))                                  # answerability (pop)
nent=ent/(ent.max()+1e-9); nlp=np.log(ac_+1)/(np.log(ac_+1).max()+1e-9); helf=2*nent*nlp/(nent+nlp+1e-9)  # HELF
popent=pe*ent; logpopent=np.log(ac_+1)*ent
# ---- GreedyExtend (Golbandi 2010): static seed greedily minimizing population recon error ----
GXC=f'{base}/.cache/greedyext.npy'
if not os.path.exists(GXC) or os.environ.get('REGEN'):
    print("GreedyExtend (population-greedy seed set)...",flush=True); samp=[x for x in trU if len(rat_by_u[x])>=12][:400]
    pdata=[]
    for x in samp:
        profset=set(j for j,_ in rat_by_u[x]); pdata.append((np.array(list(profset)), cans_of(x,profset)))
    cand=[c for c in range(NC) if ac_[c]>=20]; chosen=[]; toks_u=[[] for _ in pdata]
    for step in range(8):
        best=-1;bc=None
        for c in sorted(cand,key=lambda c:-ac_[c])[:120]:
            if c in chosen: continue
            tot=0.
            tk=[ [ (Ec[c], cans[c]) ] if c in cans else [] for (_,cans) in pdata]
            full=[toks_u[i]+tk[i] for i in range(len(pdata))]
            ul=enc_batch([f if f else [(np.zeros(D),0.)] for f in full])
            for i,(profa,_) in enumerate(pdata): tot+=sig(popb[profa]+ul[i]@Ql[profa].T).sum()
            if tot>best: best=tot;bc=c
        chosen.append(bc)
        for i,(_,cans) in enumerate(pdata):
            if bc in cans: toks_u[i]=toks_u[i]+[(Ec[bc],cans[bc])]
        print(f"  gx step {step+1}: concept {bc}",flush=True)
    np.save(GXC,np.array(chosen))
gxorder=list(np.load(GXC))
# ---- shared artifacts for ADAPTIVE baselines (tree + IGCN): u* embeddings, answer matrix, user clusters ----
def ustar_of(x,profset): return enc_u([(Q[j],resid[x][j]) for j in profset])
ART=f'{base}/.cache/adaptive_artifacts.npz'; K=20
if os.path.exists(ART) and not os.environ.get('REGEN'):
    d=np.load(ART,allow_pickle=True); US=d['US']; A=d['A']; cand_t=[int(c) for c in d['cand_t']]; CL=d['CL'].astype(int); print("adaptive artifacts: loaded cached",flush=True)
else:
    print("adaptive artifacts (u*, answer matrix, user clusters)...",flush=True)
    samp_t=[x for x in trU if len(rat_by_u[x])>=10][:1200]; nS=len(samp_t)
    US=np.array([ustar_of(x,set(j for j,_ in rat_by_u[x])) for x in samp_t])          # u* per train user
    CANS=[cans_of(x,set(j for j,_ in rat_by_u[x])) for x in samp_t]
    cand_t=[int(c) for c in sorted([c for c in range(NC) if ac_[c]>=30],key=lambda c:-ac_[c])[:200]]
    A=np.zeros((nS,len(cand_t)),np.int8)                                              # answer matrix +1 like / -1 dislike / 0 unanswerable
    for i,cans in enumerate(CANS):
        for cj,c in enumerate(cand_t):
            if c in cans: A[i,cj]=1 if cans[c]>0 else -1
    def kmeans(X,Kk,iters=25,seed=0):
        rs=np.random.default_rng(seed); C=X[rs.choice(len(X),Kk,replace=False)].copy()
        for _ in range(iters):
            dd=((X[:,None,:]-C[None,:,:])**2).sum(2); lab=dd.argmin(1)
            for k in range(Kk):
                if (lab==k).any(): C[k]=X[lab==k].mean(0)
        return lab.astype(int)
    CL=kmeans(US,K); np.savez(ART,US=US,A=A,cand_t=np.array(cand_t),CL=CL); print(f"  {len(US)} users x {len(cand_t)} concepts, {K} clusters",flush=True)
nS=len(US)
# ---- Golbandi ternary tree (uses shared US/A/cand_t) ----
TREEC=f'{base}/.cache/golbandi_tree.npy'
if os.path.exists(TREEC) and not os.environ.get('REGEN'):
    TREE=np.load(TREEC,allow_pickle=True).item(); print("Golbandi ternary tree: loaded cached",flush=True)
else:
    print("Golbandi ternary tree (building)...",flush=True)
    def varsum(idx):
        if len(idx)==0: return 0.
        e=US[idx]; return float(((e-e.mean(0))**2).sum())
    nid=[0]
    def build(idx,depth):
        if depth>=8 or len(idx)<60: return None
        best=(varsum(idx)-1e-9,None,None)
        for cj,c in enumerate(cand_t):
            col=A[idx,cj]; L=idx[col>0]; Dn=idx[col<0]; Uk=idx[col==0]
            if len(L)<20 or len(Dn)<20: continue
            v=varsum(L)+varsum(Dn)+varsum(Uk)
            if v<best[0]: best=(v,c,(L,Dn,Uk))
        if best[1] is None: return None
        nid[0]+=1; _,c,(L,Dn,Uk)=best
        return {'c':int(c),'L':build(L,depth+1),'D':build(Dn,depth+1),'U':build(Uk,depth+1)}
    TREE=build(np.arange(nS),0); np.save(TREEC,np.array(TREE,dtype=object)); print(f"  tree built ({nid[0]} nodes), saved",flush=True)
# ---- IGCN (Rashid/Karypis/Riedl 2008): max info-gain about user-CLUSTER over still-consistent neighbours; adaptive/personalized ----
def entropy_of(lab):
    if len(lab)==0: return 0.
    cnt=np.bincount(lab,minlength=K).astype(float); p=cnt[cnt>0]/len(lab); return float(-(p*np.log(p)).sum())
# RMSE rating head: r_hat=mu+bi[j]+BETA*(u.Ql[j]); BETA fit once on training item-folds (cached, SHARED across scripts)
BETAC=f'{base}/.cache/rmse_beta.npy'
if os.path.exists(BETAC) and not os.environ.get('REGEN'):
    BETA=float(np.load(BETAC))
else:
    num=den=0.
    for x in [x for x in trU if len(rat_by_u[x])>=10][:600]:
        allit=[j for j,_ in rat_by_u[x]]; kn=allit[:len(allit)//2]; hl=allit[len(allit)//2:]
        ux=enc_u([(Q[j],resid[x][j]) for j in kn])
        for j in hl: pr=float(ux@Ql[j]); num+=resid[x][j]*pr; den+=pr*pr
    BETA=num/den if den>1e-9 else 1.0; np.save(BETAC,np.array(BETA)); print(f"  RMSE BETA={BETA:.3f}",flush=True)
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
SPL={}; TE=[]
def build_split(seed):                                                                # seed=123 is the canonical locked ruler; other seeds for seed-averaging
    global SPL,TE; _rs=np.random.default_rng(seed); SPL={}
    for x in te:
        its=list(dict(rat_by_u[x]))
        if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
    _tsel=[x for x in te if x in SPL]; _tr=os.environ.get('TESTRANGE','300:99999')   # HARMONIZED to continuous_policy2: te[300:] = representative disjoint test
    TE=(_tsel[:150]+_tsel[400:]) if _tr=='clean' else (_tsel if _tr=='all' else _tsel[int(_tr.split(':')[0]):int(_tr.split(':')[1])])
ORD={'pop':list(np.argsort(-cfreq)),'entropy':list(np.argsort(-ent)),'popent':list(np.argsort(-popent)),'logpopent':list(np.argsort(-logpopent)),'helf':list(np.argsort(-helf)),'greedyext':gxorder+list(np.argsort(-cfreq))}
def run(mode,tail):
    M={q:0. for q in [0,2,4,8]};Rc={q:0. for q in [0,2,4,8]};RM={q:0. for q in [0,2,4,8]};m=0;na=0.
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); held={j:rd[j] for j in test}
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        cans=cans_of(x,profset); toks=[]; asked=set(); nq=0; nans=0; node=TREE; Sst=np.arange(nS)
        seq=ORD.get(mode)
        for q in [0,2,4,8]:
            while nq<q:
                cj_sel=None
                if mode=='random':
                    cc=[c for c in range(NC) if c not in asked and c in cans]
                    if not cc: break
                    c=cc[int(rng.integers(len(cc)))]
                elif mode=='tree' and node is not None:
                    c=node['c']
                elif mode=='igcn':
                    clS=CL[Sst]; HS=entropy_of(clS); nSc=max(len(Sst),1); AS=A[Sst]; best=(-1.,None)
                    for cj in range(len(cand_t)):
                        if cand_t[cj] in asked: continue
                        col=AS[:,cj]; ig=HS
                        for a in (-1,0,1):
                            mm=col==a; nm=int(mm.sum())
                            if nm: ig-=(nm/nSc)*entropy_of(clS[mm])
                        if ig>best[0]: best=(ig,cj)
                    cj_sel=best[1]; c=cand_t[cj_sel] if cj_sel is not None else next((cc for cc in ORD['pop'] if cc not in asked),None)
                    if c is None: break
                else:
                    c=next((cc for cc in (seq or ORD['pop']) if cc not in asked), None)
                    if c is None: break
                asked.add(c); nq+=1; ans=cans.get(c)
                if ans is not None: toks.append((Ec[c],ans)); nans+=1
                if mode=='tree' and node is not None:
                    node=(node['L'] if ans>0 else node['D']) if ans is not None else node['U']
                if mode=='igcn' and cj_sel is not None:                            # filter neighbours by target's answer (incl. unanswerable=0)
                    av=1 if (ans is not None and ans>0) else (-1 if ans is not None else 0)
                    nxt=Sst[A[Sst,cj_sel]==av]
                    if len(nxt)>=15: Sst=nxt
            uu=enc_u(toks); mt=metr(uu,tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
            rv=rmse_of(uu,held)
            if rv is not None: RM[q]+=rv
        m+=1; na+=nans
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},{q:RM[q]/m for q in RM},na/m
SEEDS=[int(s) for s in os.environ.get('SEEDS','123').split(',')]
MODES=os.environ.get('MODES','random,pop,entropy,popent,logpopent,helf,greedyext,tree,igcn').split(',')
if len(SEEDS)==1:
    build_split(SEEDS[0])
    for tail in [False,True]:
        print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | LIT BASELINES | NDCG@10 / Rec@50 / RMSE / ans ===",flush=True)
        for mode in MODES:
            Mp,Rcp,RMp,na=run(mode,tail); tag=' <- conc_pop' if mode=='pop' else ''; print(f"  {mode:<10}: NDCG "+" ".join(f"{Mp[q]:.3f}" for q in [0,2,4,8])+" | Rec "+" ".join(f"{Rcp[q]:.3f}" for q in [0,2,4,8])+" | RMSE "+" ".join(f"{RMp[q]:.3f}" for q in [0,2,4,8])+f" | ans/{T}={na:.1f}{tag}",flush=True)
else:
    agg={}
    for si,sd in enumerate(SEEDS):
        build_split(sd)
        for tail in [False,True]:
            for mode in MODES:
                Mp,Rcp,RMp,na=run(mode,tail); agg.setdefault((mode,tail),[]).append((Mp[8],Rcp[8],RMp[8]))
        print(f"  ...seed {sd} done ({si+1}/{len(SEEDS)})",flush=True)
    for tail in [False,True]:
        print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | SEED-AVG q8 (n={len(SEEDS)}) | NDCG@10 / Rec@50 / RMSE (mean±std) ===",flush=True)
        for mode in MODES:
            a=np.array(agg[(mode,tail)]); mn=a.mean(0); st=a.std(0); tag=' <- conc_pop' if mode=='pop' else ''
            print(f"  {mode:<10}: NDCG {mn[0]:.3f}±{st[0]:.3f} | Rec {mn[1]:.3f}±{st[1]:.3f} | RMSE {mn[2]:.3f}±{st[2]:.3f}{tag}",flush=True)
