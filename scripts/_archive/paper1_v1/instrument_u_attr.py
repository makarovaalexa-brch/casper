"""
S3 — CASPER-U unified instrument: items + ATTRIBUTES (genres) in ONE space.
Loads the S2 instrument's WRMF item factors Q (instrument_u_ml1m.pt). Attribute (genre) factors A live in the
SAME space as centroids of their items. A reveal = an item OR an attribute; the differentiable ridge fold-in
stacks the revealed factor rows ([Q[items]; A[attrs]]) and solves one user vector u. Score = pop + Q.u.
GATES (DoD): (a) attribute reveals move predictions correctly (genre 'like' -> that genre's items rise);
(b) mixed item+attribute reveals >= item-only at equal #questions; (c) item-only path unchanged (regression).
Ruler: ML-1M, std NDCG@10/Recall@10, full-cat, same fixed split as S2.
"""
import os, time, numpy as np, torch, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens/ml-1m'; D=64; LIKE=4.0; LAM=float(os.environ.get('LAM',1.0)); K=50
rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U); I=np.array(I); R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((i[k],(R[k]-3.)/2., 1.0 if R[k]>=LIKE else 0.0))
keep=[x for x in range(nu) if sum(t[2] for t in rated[x])>=5]; rng.shuffle(keep); n=len(keep)
tr=keep[:int(0.8*n)]; va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in tr:
    for it,_,lk in rated[x]:
        if lk>0: cnt[it]+=1
popb=np.log(cnt+1.0).astype(np.float32)
ck=torch.load(f'{base}/../.cache/checkpoints/instrument_u_ml1m.pt'); Q=ck['Q'].numpy().astype(np.float32)
# ---- genres (attributes) from movies.dat, mapped to item indices ----
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{base}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); mid=int(p[0])
        if mid in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[mid],gid[g]]=True
A=np.zeros((na,D),np.float32)                         # genre factor = normalized centroid of its items in Q-space
for g in range(na):
    items=np.where(item_g[:,g])[0]
    if len(items): c=Q[items].mean(0); A[g]=c/ (np.linalg.norm(c)+1e-9) * np.linalg.norm(Q,axis=1).mean()
print(f"unified instrument | ML-1M {ni} items + {na} genres in one d={D} space | test {len(te)}",flush=True)

def foldin(rows, y):                                  # rows [m,D] (item and/or attribute factors), y [m]
    if len(rows)==0: return np.zeros(D)
    A_=rows.T@rows+LAM*np.eye(D); return np.linalg.solve(A_, rows.T@y)
def ndcg_recall(score, rel, excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/len(rel)
def zsc(v): return (v-v.mean())/(v.std()+1e-9)
zpop=zsc(popb)
def rmva(k):
    cand=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[cand].T,pivoting=True); return [int(cand[p]) for p in piv[:k]]
RMVA=rmva(K)
def liked_genres(x):                                  # user's confirmed genre prefs (>=2 liked films of that genre)
    gc=np.zeros(na)
    for it,_,lk in rated[x]:
        if lk>0: gc+=item_g[it]
    return [g for g in range(na) if gc[g]>=2]
# ---------- GATE (a): a single genre 'like' raises that genre's items ----------
print("\n=== GATE (a): attribute reveal moves predictions (mean rank-percentile of genre items) ===",flush=True)
def genre_lift(g):
    u=foldin(A[g:g+1], np.array([1.0])); sc=Q@u; pct=(sc.argsort().argsort())/ni   # percentile of each item's score
    gi=np.where(item_g[:,g])[0]; return pct[gi].mean()-pct.mean()
lifts=[genre_lift(g) for g in range(na)]
print("  mean score-percentile lift of a genre's items when that genre is 'liked':",flush=True)
for g in [gid['Sci-Fi'],gid['Horror'],gid['Romance'],gid['Documentary'],gid['Animation']]:
    print(f"    {GEN[g]:<12} lift={lifts[g]:+.3f}",flush=True)
gate_a = np.mean(lifts)>0.05
print(f"  GATE(a) attribute-directional: mean lift {np.mean(lifts):+.3f} -> {'PASS' if gate_a else 'FAIL'}",flush=True)
# ---------- GATE (b): mixed item+attribute >= item-only at equal #questions ----------
def eval_arm(users, n_item, n_attr, w):
    nd=rc=0.; m=0
    iseeds=RMVA[:n_item]; ss=set(iseeds)
    for x in users:
        rd={it:lk for it,cr,lk in rated[x]}; rel=[it for it,cr,lk in rated[x] if lk>0 and it not in ss]
        if not rel: continue
        rows=[]; y=[]
        for it in iseeds: rows.append(Q[it]); y.append(rd.get(it,0.0))      # item reveals
        lg=liked_genres(x)
        for g in range(n_attr):                                            # attribute reveals (ask top-n_attr genres)
            rows.append(A[g]); y.append(1.0 if g in lg else 0.0)
        ninf=int(sum(1 for v in y if v>0)); uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
        conf=ninf/(ninf+5.0); sc=conf*zsc(Q@uu)+w*zpop
        a,b=ndcg_recall(sc,rel,ss); nd+=a; rc+=b; m+=1
    return nd/m, rc/m
print("\n=== GATE (b): equal #questions — item-only vs mixed vs attr-heavy (val-tuned w) ===",flush=True)
WG=[0,1,2,3,4,6,8,12]
def best(users_eval, ni_, na_):
    w=max(WG,key=lambda w: eval_arm(va,ni_,na_,w)[0]); return eval_arm(te,ni_,na_,w)+(w,)
for (ni_,na_,tag) in [(10,0,'10 items (item-only)'),(5,5,'5 items + 5 genres'),(0,10,'10 genres (attr-only)'),(0,18,'all 18 genres')]:
    nd,rc,w=best(te,ni_,na_); print(f"  {tag:<24} NDCG@10={nd:.4f} Recall@10={rc:.4f} (w*={w})",flush=True)
print("\n(GATE (b) PASS if mixed/attr >= item-only at equal #q; GATE (c) item-only path = S2, unchanged by construction)",flush=True)
