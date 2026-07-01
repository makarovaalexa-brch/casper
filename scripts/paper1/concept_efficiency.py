"""
Does asking GENERAL CONCEPTS / ATTRIBUTES improve NDCG faster than asking MOVIES, at least for some users?
Per-user ORACLE (privileged) greedy interview on the CASPER-U instrument: at each turn pick the question whose
TRUE answer most increases held-out NDCG. Three arms: ITEMS-only, GENRES-only, MIXED. This measures the *potential*
(ceiling) of concept/attribute elicitation per user ("backwards from user embeddings": the oracle finds the
most-informative question for each user's true taste). Reports per-turn NDCG curves + the fraction of users for whom
genres beat items, with illustrative examples.
"""
import os, numpy as np, torch
base='C:/dev/phd/casper/data/movielens/ml-1m'; D=64; LAM=1.0; W=float(os.environ.get('W',4.0)); T=int(os.environ.get('T',5))
NU=int(os.environ.get('NU',150)); rng=np.random.default_rng(0)
I=[];R=[];Uu=[]
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); Uu.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I);R=np.array(R,np.float32);Uu=np.array(Uu)
uids={x:k for k,x in enumerate(np.unique(Uu))}; iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); nu=len(uids)
u=np.array([uids[x] for x in Uu]); ii=np.array([iids[x] for x in I])
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ii[k],(R[k]-3.)/2.,1.0 if R[k]>=4 else 0.0))
keep=[x for x in range(nu) if sum(t[2] for t in rated[x])>=8]; rng.shuffle(keep); te=keep[int(0.9*len(keep)):]
cnt=np.zeros(ni)
for x in keep[:int(0.8*len(keep))]:
    for it,_,lk in rated[x]:
        if lk>0: cnt[it]+=1
popb=np.log(cnt+1.0).astype(np.float32); zpop=(popb-popb.mean())/(popb.std()+1e-9)
Q=torch.load(f'{base}/../.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32)
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{base}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); mid=int(p[0])
        if mid in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[mid],gid[g]]=True
A=np.zeros((na,D),np.float32)
for g in range(na):
    s=np.where(item_g[:,g])[0]
    if len(s): c=Q[s].mean(0); A[g]=c/(np.linalg.norm(c)+1e-9)*np.linalg.norm(Q,axis=1).mean()
def foldin(rows,y): return np.linalg.solve(rows.T@rows+LAM*np.eye(D), rows.T@y) if len(rows) else np.zeros(D)
def ndcg(u_,rel,excl,m):
    sc=(W*zpop).copy()
    if m>0:
        z=Q@u_; z=(z-z.mean())/(z.std()+1e-9); sc=sc+(m/(m+5.))*z
    sc=sc.copy(); sc[list(excl)]=-1e9; top=np.argsort(-sc)[:10]; rs=set(rel)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs)/idcg if idcg else 0.
def oracle(user, mode):                       # greedy: pick question whose TRUE answer most improves NDCG
    recs=rated[user]; rd={it:(cr,lk) for it,cr,lk in recs}
    likes=[it for it,cr,lk in recs if lk>0]; rng.shuffle(likes)
    test=set(likes[:len(likes)//2])                       # held-out target
    item_cands=[it for it,cr,lk in recs if it not in test]  # askable items (rated, not target)
    liked_g=set(g for g in range(na) if sum(item_g[it,g] for it in likes)>=2)
    rows=[]; yv=[]; asked_i=set(); asked_g=set(); excl=set(); curve=[ndcg(np.zeros(D),list(test),excl,0)]
    for t in range(T):
        best=None;bg=None;bkind=None
        if mode in ('item','mixed'):
            for it in item_cands:
                if it in asked_i: continue
                u_=foldin(np.array(rows+[Q[it]]),np.array(yv+[rd[it][0]]))
                g=ndcg(u_,list(test),excl|{it},len(rows)+1)
                if best is None or g>best: best=g;bg=it;bkind='item'
        if mode in ('genre','mixed'):
            for gg in range(na):
                if gg in asked_g: continue
                yg=1.0 if gg in liked_g else -1.0
                u_=foldin(np.array(rows+[A[gg]]),np.array(yv+[yg]))
                g=ndcg(u_,list(test),excl,len(rows)+1)
                if best is None or g>best: best=g;bg=gg;bkind='genre'
        if bkind=='item': rows.append(Q[bg]); yv.append(rd[bg][0]); asked_i.add(bg); excl.add(bg)
        elif bkind=='genre': rows.append(A[bg]); yv.append(1.0 if bg in liked_g else -1.0); asked_g.add(bg)
        else: break
        curve.append(best)
    return np.array(curve+[curve[-1]]*(T+1-len(curve)))
users=te[:NU]
cur={m:[] for m in ('item','genre','mixed')}
for n,x in enumerate(users):
    for m in cur: cur[m].append(oracle(x,m))
    if (n+1)%50==0: print(f"  {n+1}/{len(users)}",flush=True)
CI={m:np.array(cur[m]) for m in cur}
print("\n=== ORACLE elicitation efficiency: NDCG@10 vs #questions (avg over %d users) ==="%len(users),flush=True)
print(f"{'#q':>3} {'ITEMS':>8} {'GENRES':>8} {'MIXED':>8}",flush=True)
for t in range(T+1): print(f"{t:>3} {CI['item'][:,t].mean():>8.4f} {CI['genre'][:,t].mean():>8.4f} {CI['mixed'][:,t].mean():>8.4f}",flush=True)
# per-user: where do genres beat items?
for t in [1,3,T]:
    win=(CI['genre'][:,t]>CI['item'][:,t]+1e-4).mean()
    gap=(CI['genre'][:,t]-CI['item'][:,t])
    print(f"\n@ {t} questions: genres>items for {100*win:.0f}% of users; "
          f"mean gap among those {gap[gap>0].mean():.3f}; mixed>items {100*(CI['mixed'][:,t]>CI['item'][:,t]+1e-4).mean():.0f}%",flush=True)
# illustrative: users where 3 genres beat 3 items most
d=CI['genre'][:,3]-CI['item'][:,3]; topu=np.argsort(-d)[:5]
print("\nillustrative users (3 genres vs 3 items NDCG@10):",flush=True)
for j in topu: print(f"  user {users[j]}: genres {CI['genre'][j,3]:.3f} vs items {CI['item'][j,3]:.3f}",flush=True)
