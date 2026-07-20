"""
Concept ANSWER model, leakage-free (answers from PROFILE-ONLY = rated minus held-out test) + 3-valued
(like/dislike/DON'T-KNOW=abstain). Test whether a principled answer makes concept elicitation grow monotonically.
Models:
  C2  relweighted-debiased (Sen'09)         -- graded, no abstain (baseline)
  C4  C2 + CONFIDENCE GATE (abstain if <SUP movies about concept)  [differentiate don't-know from dislike]
  C5  Bayesian shrinkage (Sen movie-bayes: denom + lambda0)        [soft confidence]
  C6  per-user ridge regression beta_u over tag-relevance (Nguyen&Riedl UMAP'13) + gate  [learn user's concepts]
  G   genre graded-lift (profile-only) -- control (should still grow)
Selection: affinity order (ask most-confident |answer| first). Leakage-free.
"""
import os, numpy as np, torch
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=1.0
W=float(os.environ.get('W',2.0)); T=int(os.environ.get('T',10)); NU=int(os.environ.get('NU',120))
NC=int(os.environ.get('NC',150)); SUP=int(os.environ.get('SUP',5)); rng=np.random.default_rng(0)
I=[];R=[];Uu=[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); Uu.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I);R=np.array(R,np.float32);Uu=np.array(Uu)
uids={x:k for k,x in enumerate(np.unique(Uu))}; iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); nu=len(uids)
u=np.array([uids[x] for x in Uu]); ic=np.array([iids[x] for x in I])
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ic[k],R[k]))
keep=[x for x in range(nu) if sum(1 for it,r in rated[x] if r>=4)>=8]; rng.shuffle(keep)
trU=keep[:int(0.8*len(keep))]; te=keep[int(0.9*len(keep)):]
cnt=np.zeros(ni)
for x in trU:
    for it,r in rated[x]:
        if r>=4: cnt[it]+=1
popb=np.log(cnt+1.0).astype(np.float32); zpop=(popb-popb.mean())/(popb.std()+1e-9)
Q=torch.load(f'{base}/.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32); qn=np.linalg.norm(Q,axis=1).mean()
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
Ag=np.zeros((na,D),np.float32)
for g in range(na):
    s=np.where(item_g[:,g])[0]; Ag[g]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn if len(s) else 0
Pg=item_g[cnt>0].mean(0)
THR=0.5; tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f: a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
tcount={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in iids and float(a[2])>THR: tcount[int(a[1])]=tcount.get(int(a[1]),0)+1
glow=set(g.lower() for g in GEN)
ctags=[t for t,_ in sorted(tcount.items(),key=lambda kv:-kv[1]) if tagname[t] not in glow][:NC]; cidx={t:k for k,t in enumerate(ctags)}
rel=np.zeros((ni,NC),np.float32)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); t=int(a[1])
        if m in iids and t in cidx: rel[iids[m],cidx[t]]=float(a[2])
Ac=np.zeros((NC,D),np.float32)
for c in range(NC):
    s=np.where(rel[:,c]>THR)[0]; Ac[c]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn if len(s) else 0
print(f"loaded {NC} concepts; cold-test {min(NU,len(te))} users W={W} T={T} SUP={SUP}",flush=True)
def foldin(X,y,w=None):                          # precision-weighted ridge fold-in: u=(XᵀCX+λI)⁻¹XᵀCy
    if not len(X): return np.zeros(D)
    if w is None: w=np.ones(len(y))
    return np.linalg.solve((X.T*w)@X+LAM*np.eye(D), X.T@(w*y))
def score(u_,m):
    sc=(W*zpop).copy()
    if m>0: z=Q@u_; z=(z-z.mean())/(z.std()+1e-9); sc=sc+(m/(m+5.))*z
    return sc
def ndcg(sc,rel_,excl):
    if not len(rel_): return 0.
    s=sc.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(int(t) for t in rel_)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel_))))
    return sum(1./np.log2(p+2) for p,t in enumerate(top) if int(t) in rs)/idcg if idcg else 0.
def answers(model, p_items, p_r):
    umean=p_r.mean() if len(p_r) else 0.0; out={}
    if model=='G':
        for g in range(na):
            ing=item_g[p_items,g]; pu=ing.mean() if len(p_items) else 0.
            out[g]=float(np.tanh(np.log((pu+1e-3)/(Pg[g]+1e-3)+1e-6)));
        return out
    relP=rel[p_items] if len(p_items) else np.zeros((0,NC))
    beta=None
    if model=='C6' and len(p_items)>0:
        X=relP; yv=p_r-umean; beta=np.linalg.solve(X.T@X+1.0*np.eye(NC), X.T@yv)
    for c in range(NC):
        w=relP[:,c] if len(p_items) else np.zeros(0); sup=int((w>THR).sum())
        if model in ('C4','C6') and sup<SUP: out[c]=None; continue       # 'don't know' -> abstain
        if w.sum()<1e-6 and model!='C6': out[c]=None; continue
        if model=='C2' or model=='C4': out[c]=float(np.clip((w*(p_r-umean)).sum()/(w.sum()+1e-6)/1.5,-1,1))
        elif model=='C5': out[c]=float(np.clip((w*(p_r-umean)).sum()/(w.sum()+5.0)/1.5,-1,1))
        elif model=='C6': out[c]=float(np.clip(beta[c]*2.0,-1,1))
    return out
def zc(v): return (v-v.mean())/(v.std()+1e-9)
def run(model, mode='foldin'):
    A = Ag if model=='G' else Ac; NDt=np.zeros(T+1); m_=0
    for x in te[:NU]:
        recs=rated[x]; allit=np.array([it for it,r in recs]); allr=np.array([r for it,r in recs],np.float32)
        likes=[it for it,r in recs if r>=4]
        if len(likes)<4: continue
        ll=likes[:]; rng.shuffle(ll); test=set(ll[len(ll)//2:])
        pmask=np.array([int(i) not in test for i in allit]); p_items=allit[pmask]; p_r=allr[pmask]   # PROFILE-ONLY (no leak)
        profile=set(int(i) for i in p_items); rel_=list(test)
        ans=answers(model,p_items,p_r)
        order=sorted([e for e in ans if ans[e] is not None],key=lambda e:-abs(ans[e]))   # affinity selection
        rows=[];y=[]; gpref=np.zeros(D); curve=[ndcg(W*zpop,rel_,profile)]
        for e in order:
            if len(curve)>T: break
            rows.append(A[e]); y.append(ans[e])
            if mode=='foldin': pers=Q@foldin(np.array(rows),np.array(y))   # ridge (Gram inverse)
            else: gpref=gpref+ans[e]*A[e]; pers=Q@gpref                    # EAR additive (plain sum)
            wt = WACAP if mode=='additive' else (len(rows)/(len(rows)+5.)) # additive: BOUNDED constant weight
            sc=W*zpop+wt*zc(pers)
            curve.append(ndcg(sc,rel_,profile))
        while len(curve)<=T: curve.append(curve[-1])
        NDt+=np.array(curve[:T+1]); m_+=1
    return NDt/m_
def mono(c): return all(c[t]>=c[t-1]-0.003 for t in range(1,len(c)))
print(f"\n{'model (mechanism)':<26} | NDCG@10 q0..qT | mono?",flush=True)
for mdl in ['G','C6']:
    c=run(mdl,'foldin'); print(f"{mdl+' foldin(grow)':<26} | "+" ".join(f"{v:.3f}" for v in c)+f" | {'YES' if mono(c) else 'no'}",flush=True)
    for wa in [0.2,0.4,0.8]:
        WACAP=wa; c=run(mdl,'additive'); print(f"{mdl+' additive w='+str(wa):<26} | "+" ".join(f"{v:.3f}" for v in c)+f" | {'YES' if mono(c) else 'no'}",flush=True)
