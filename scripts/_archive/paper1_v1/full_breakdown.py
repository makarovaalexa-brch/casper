"""
Full per-step breakdown on the CALIBRATED biased-SVD instrument: every selector x {RMSE, Recall@10, NDCG@10},
printed q0..qT so monotonicity is visible. Includes GENRES (attribute) selector. Reuses Q_svd.npy/bi_svd.npy.
Protocol: per user hold out HALF of rated items as TEST; rest = answerable PROFILE.
 item selectors: ask profile items in order, fold taste-residual (r-mu-bi).
 genres selector: ask genres in affinity order; fold genre-centroid pseudo-item with target = user's mean
   taste-residual over that genre's profile items (biased-SVD extension of S3).
Ranking metrics: full-catalogue, exclude profile+asked, rel = held-out LIKES. RMSE over all held-out ratings.
"""
import os, numpy as np, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=int(os.environ.get('T',15)); LAMFI=float(os.environ.get('LAMFI',5.0))
rng=np.random.default_rng(0)
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
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
trU=set(keep[:int(0.8*n)]); te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=0.;c=0
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVArank={int(ca[p]):r for r,p in enumerate(piv)}
H5=np.zeros((ni,5))
for k in range(len(uu)):
    if uu[k] in trU: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.zeros((na,D))
for g in range(na):
    s=np.where(item_g[:,g])[0]; gcent[g]=Q[s].mean(0) if len(s) else 0
def foldin(F,targ):
    if not len(F): return np.zeros(D)
    F=np.array(F); A=F.T@F+LAMFI*np.eye(D); return np.linalg.solve(A,F.T@np.array(targ,np.float32))
def ndcg(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/len(rel)
SPL={}; _rs=np.random.default_rng(123)            # FIXED held-out split per user, SHARED across all selectors
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6:
        il=items[:]; _rs.shuffle(il); cut=len(il)//2; SPL[x]=(il[:cut],il[cut:])
_rord=np.random.default_rng(7)                    # separate rng for the 'random' selector's question order
def run(kind,maxu=None):
    RMSE=np.zeros(T+1); R10=np.zeros(T+1); ND=np.zeros(T+1); m=0
    for x in (te[:maxu] if maxu else te):
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test,prof=SPL[x]
        tlike=[j for j in test if rd[j]>=4]
        if not tlike or not prof: continue
        tj=np.array(test); ty=np.array([rd[j] for j in test],np.float32); umean_res=None
        if kind=='genres':
            # genre affinity from profile: mean taste-residual over genre items
            ga={}
            for g in range(na):
                gi=[j for j in prof if item_g[j,g]]
                if len(gi)>=2: ga[g]=np.mean([rd[j]-mu-bi[j] for j in gi])
            order=sorted(ga,key=lambda g:-abs(ga[g]))
            F=[]; tg=[]
            seq=[('g',g) for g in order]
        else:
            if kind=='rmva': order=sorted(prof,key=lambda j:RMVArank.get(j,10**9))
            elif kind=='helf': order=sorted(prof,key=lambda j:-helf[j])
            elif kind=='pop': order=sorted(prof,key=lambda j:-cnt[j])
            elif kind=='entropy': order=sorted(prof,key=lambda j:-ent[j])
            else: order=prof[:]; _rord.shuffle(order)
            seq=[('i',j) for j in order]
        F=[]; tg=[]
        for t in range(T+1):
            if t>0 and t-1<len(seq):
                typ,e=seq[t-1]
                if typ=='i': F.append(Q[e]); tg.append(rd[e]-mu-bi[e])
                else: F.append(gcent[e]); tg.append(ga[e])
            u=foldin(F,tg)
            pred=mu+bi[tj]+Q[tj]@u; RMSE[t]+=np.sqrt(np.mean((pred-ty)**2))
            sc=popb+Q@u; a,b=ndcg(sc,set(tlike),set(prof)); ND[t]+=a; R10[t]+=b
        m+=1
    return RMSE/m,R10/m,ND/m,m
def run_oracle(maxu):                              # privileged greedy: pick next profile item that maximizes held-out NDCG
    ND=np.zeros(T+1); m=0
    for x in te[:maxu]:
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test,prof=SPL[x]; tlike=set(j for j in test if rd[j]>=4)
        if not tlike or not prof: continue
        nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,set(prof))[0]
        F=[];tg=[];asked=set()
        for t in range(1,T+1):
            best=None
            for c in prof:
                if c in asked: continue
                u=foldin(F+[Q[c]],tg+[rd[c]-mu-bi[c]]); a=ndcg(popb+Q@u,tlike,set(prof))[0]
                if best is None or a>best[0]: best=(a,c)
            if best is None: nd[t:]=nd[t-1]; break          # profile exhausted -> carry forward
            c=best[1]; asked.add(c); F.append(Q[c]); tg.append(rd[c]-mu-bi[c]); nd[t]=best[0]
        ND+=nd; m+=1
    return ND/m,m
print(f"CALIBRATED biased-SVD; {len(te)} test users; T={T}; held-out=half of rated. Per-step q0..q{T}.\n",flush=True)
for kind in ['random','pop','entropy','helf','rmva','genres']:
    rmse,r10,nd,m=run(kind)
    print(f"=== {kind}  (n={m}) ===",flush=True)
    print("  RMSE  : "+" ".join(f"{v:.3f}" for v in rmse),flush=True)
    print("  Rec@10: "+" ".join(f"{v:.3f}" for v in r10),flush=True)
    print("  NDCG10: "+" ".join(f"{v:.3f}" for v in nd),flush=True)
NU_OR=int(os.environ.get('NU_OR',150))
ond,om=run_oracle(NU_OR)
_,_,rnd,_=run('random',NU_OR); _,_,hnd,_=run('helf',NU_OR)   # 3rd return value = NDCG curve
print(f"\n=== ORACLE headroom (greedy on held-out NDCG; {om} users; random/helf on same {NU_OR}) ===",flush=True)
print("  ORACLE NDCG10: "+" ".join(f"{v:.3f}" for v in ond),flush=True)
print("  helf   NDCG10: "+" ".join(f"{v:.3f}" for v in hnd),flush=True)
print("  random NDCG10: "+" ".join(f"{v:.3f}" for v in rnd),flush=True)
print(f"  -> oracle q15={ond[T]:.3f} vs helf {hnd[T]:.3f} vs random {rnd[T]:.3f}; ORACLE headroom over random={ond[T]-rnd[T]:+.3f} (realizable HELF captures {(hnd[T]-rnd[T])/(ond[T]-rnd[T]+1e-9)*100:.0f}%)",flush=True)
