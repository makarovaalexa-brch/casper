"""
Test user-ANSWER models for genre/concept elicitation. Goal: a profile-based answer such that POPULAR-attribute
elicitation grows MONOTONICALLY. One change at a time. Logs to ATTR_ANSWER_LOG.md analysis.
Genre: M1 count(old) | M2 diff-in-means(graded) | M3 lift(binary) | M4 lift(graded).
Concept: C1 count(old) | C2 relweighted-debiased(Sen'09, graded) | C3 relweighted-sign(binary).
"""
import os, numpy as np, torch
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=1.0
W=float(os.environ.get('W',2.0)); T=int(os.environ.get('T',10)); NU=int(os.environ.get('NU',120)); NC=int(os.environ.get('NC',150))
rng=np.random.default_rng(0)
I=[];R=[];Uu=[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); Uu.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I);R=np.array(R,np.float32);Uu=np.array(Uu)
uids={x:k for k,x in enumerate(np.unique(Uu))}; iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); nu=len(uids)
u=np.array([uids[x] for x in Uu]); ic=np.array([iids[x] for x in I])
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ic[k],R[k]))     # (item, RAW rating)
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
Pg_global=item_g[ (cnt>0) ].mean(0) if (cnt>0).any() else item_g.mean(0)   # base-rate P(g)
gpop=list(np.argsort(-(cnt@item_g)))                                       # popular-genre order
# ---- genome: rel matrix [ni,NC] for top-NC concepts ----
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
cpop=list(np.argsort(-(rel>THR).sum(0)))                                   # popular-concept order
print(f"loaded {NC} concepts; ML-1M cold-test {min(NU,len(te))} users W={W} T={T}",flush=True)
def foldin(rows,y): return np.linalg.solve(rows.T@rows+LAM*np.eye(D), rows.T@y) if len(rows) else np.zeros(D)
def score(u_,m):
    sc=(W*zpop).copy()
    if m>0: z=Q@u_; z=(z-z.mean())/(z.std()+1e-9); sc=sc+(m/(m+5.))*z
    return sc
def ndcg(sc,rel_,excl):
    if not len(rel_): return 0.
    s=sc.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(int(t) for t in rel_)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel_))))
    return sum(1./np.log2(p+2) for p,t in enumerate(top) if int(t) in rs)/idcg if idcg else 0.
# ---- answer models: return (y or None) given user rating arrays ----
def genre_ans(model, ritems, rr, gmask_user, umean):
    # ritems: idx of user's rated items; rr: their raw ratings; for each genre return y
    out={}
    for g in range(na):
        ing=item_g[ritems,g]
        if model=='M1': out[g]= 1.0 if (ing & (rr>=4)).sum()>=2 else -1.0
        elif model=='M2':
            if ing.sum()==0: out[g]=None
            else: a=rr[ing].mean()-umean; out[g]=float(np.clip(a/1.5,-1,1))
        elif model in ('M3','M4'):
            pu=ing.mean(); lift=(pu+1e-3)/(Pg_global[g]+1e-3)
            if model=='M3': out[g]= 1.0 if lift>1.1 else (-1.0 if lift<0.9 else None)
            else: out[g]=float(np.tanh(np.log(lift+1e-6)))
    return out
def concept_ans(model, ritems, rr, umean):
    out={}
    rc=rel[ritems]                              # [m, NC]
    for c in range(NC):
        w=rc[:,c]
        if model=='C1': out[c]= 1.0 if ((w>THR)&(rr>=4)).sum()>=2 else -1.0
        else:
            if w.sum()<1e-6: out[c]=None
            else:
                a=(w*(rr-umean)).sum()/(w.sum());
                out[c]= float(np.clip(a/1.5,-1,1)) if model=='C2' else (1.0 if a>0 else -1.0)
    return out
def run(modality, model, sel):
    NDt=np.zeros(T+1); m_=0; baseorder = gpop if modality=='genre' else cpop
    for x in te[:NU]:
        recs=rated[x]; ritems=np.array([it for it,r in recs]); rr=np.array([r for it,r in recs],np.float32)
        likes=[it for it,r in recs if r>=4]
        if len(likes)<4: continue
        ll=likes[:]; rng.shuffle(ll); test=set(ll[len(ll)//2:]); profile=set(int(i) for i in ritems)-test
        umean=rr.mean()
        ans = genre_ans(model,ritems,rr,None,umean) if modality=='genre' else concept_ans(model,ritems,rr,umean)
        A = Ag if modality=='genre' else Ac
        if sel=='aff':                                   # per-user: ask most-opinionated (|affinity|) first
            order=sorted([e for e in ans if ans[e] is not None],key=lambda e:-abs(ans[e]))
        else: order=baseorder
        rows=[];y=[]; rel_=list(test); curve=[ndcg(score(np.zeros(D),0),rel_,profile)]
        for e in order:
            if len(curve)>T: break
            yy=ans.get(e)
            if yy is None: continue
            rows.append(A[e]); y.append(yy)
            curve.append(ndcg(score(foldin(np.array(rows),np.array(y)),len(rows)),rel_,profile))
        while len(curve)<=T: curve.append(curve[-1])      # pad with last value
        NDt+=np.array(curve[:T+1]); m_+=1
    return NDt/m_
def mono(c): return all(c[t]>=c[t-1]-0.003 for t in range(1,len(c)))
print(f"\n{'model (sel)':<30} | NDCG@10 q0..qT | mono?",flush=True)
for modality,models in [('genre',['M1','M3','M4']),('concept',['C1','C2'])]:
    for mdl in models:
        for sel in ['pop','aff']:
            c=run(modality,mdl,sel)
            print(f"{modality+':'+mdl+' ('+sel+')':<30} | "+" ".join(f"{v:.3f}" for v in c)+f" | {'YES' if mono(c) else 'no'}",flush=True)
