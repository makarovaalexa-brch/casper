"""
Realizable-ceiling probe on the preprocessed binary-implicit profile matrices (lastfm / yelp-multicity /
amazon-balanced / amazon-crossdomain): 1.0 = liked, else not. Implicit biased-MF (negative-sampled SGD) instrument;
then PROFILE-mode realizable-ceiling test (HELF vs REALIZABLE-ORACLE[profile-internal val] vs TRUE-ORACLE[test peek]).
Question: do these weak(er)-prior / cross-domain testbeds have REALIZABLE selection headroom (realizable-oracle >> HELF)?

Usage: NPZ=lastfm/lastfm_profiles | yelp/yelp_multicity_profiles | amazon/amazon_crossdomain_profiles | amazon/amazon_balanced_profiles
"""
import os, numpy as np
base='C:/dev/phd/casper/data'; D=int(os.environ.get('D',32)); LAMF=0.05; LR=float(os.environ.get('LR',0.01)); EP=int(os.environ.get('EP',30))
LAM=2.0; T=8; NEG=2; rng=np.random.default_rng(0); NPZ=os.environ.get('NPZ','lastfm/lastfm_profiles')
z=np.load(f'{base}/{NPZ}.npz',allow_pickle=True)
M=np.vstack([z['train'],z['test']]); L=(M==1.0)              # binary like matrix (NaN/else -> not like)
nu,ni=L.shape; likes_by_u={x:list(np.where(L[x])[0]) for x in range(nu)}
keep=[x for x in range(nu) if len(likes_by_u[x])>=5]; rng.shuffle(keep); n=len(keep)
trU=set(keep[:int(0.8*n)]); te=keep[int(0.9*n):]
print(f"NPZ={NPZ}: {nu} users x {ni} items, like-density {L.mean():.3f}; usable(>=5 likes)={n}, test={len(te)}",flush=True)
cnt=np.zeros(ni)
for x in trU: cnt[np.array(likes_by_u[x],int)]+=1
popb=np.log(cnt+1.0).astype(np.float32)
# implicit biased-MF via negative-sampled SGD on train users (target 1=like, 0=sampled neg)
pos=[(x,j) for x in trU for j in likes_by_u[x]]; pos=np.array(pos)
mu=float(L[sorted(trU)].mean())
trlist=sorted(trU); truser={x:k for k,x in enumerate(trlist)}; ntr=len(trlist)
bu=np.zeros(ntr,np.float32); bi=np.zeros(ni,np.float32)
P=(0.1*rng.standard_normal((ntr,D))).astype(np.float32); Q=(0.1*rng.standard_normal((ni,D))).astype(np.float32)
likeset={x:set(likes_by_u[x]) for x in trU}
for ep in range(EP):
    rng.shuffle(pos)
    # build training batch: positives + NEG random items per positive (label 0 if not liked)
    us=np.repeat(pos[:,0],1+NEG);
    it=np.empty(len(us),int); lab=np.empty(len(us),np.float32)
    it[0::1+NEG]=pos[:,1]; lab[0::1+NEG]=1.0
    rndi=rng.integers(0,ni,(len(pos),NEG))
    for q in range(NEG):
        it[1+q::1+NEG]=rndi[:,q]
        lab[1+q::1+NEG]=[1.0 if rndi[k,q] in likeset[pos[k,0]] else 0.0 for k in range(len(pos))]
    ud=np.array([truser[x] for x in us]); idx=np.arange(len(ud)); rng.shuffle(idx)
    for b0 in range(0,len(idx),16384):
        bx=idx[b0:b0+16384]; uS=ud[bx]; iS=it[bx]
        pred=mu+bu[uS]+bi[iS]+np.sum(P[uS]*Q[iS],1); e=(lab[bx]-pred).astype(np.float32)
        np.add.at(bu,uS,LR*(e-LAMF*bu[uS])); np.add.at(bi,iS,LR*(e-LAMF*bi[iS]))
        gP=LR*(e[:,None]*Q[iS]-LAMF*P[uS]); gQ=LR*(e[:,None]*P[uS]-LAMF*Q[iS]); np.add.at(P,uS,gP); np.add.at(Q,iS,gQ)
print(f"  trained: Q finite={np.isfinite(Q).all()}, |Q| mean={np.abs(Q).mean():.3f}, |bi| mean={np.abs(bi).mean():.3f}, mu={mu:.3f}",flush=True)
# HELF over items (entropy of like/not * log freq) from train
p1=cnt/max(len(trU),1); p1=np.clip(p1,1e-6,1-1e-6); H=-(p1*np.log(p1)+(1-p1)*np.log(1-p1))
lf=np.log(cnt+1)/np.log(cnt.max()+1); helf=2*lf*(H/np.log(2))/(lf+H/np.log(2)+1e-9)
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(score,rel,excl):
    if not rel: return 0.
    s=score.copy(); s[list(excl)]=-1e9; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
_rs=np.random.default_rng(123); SPL={}
for x in te:
    lk=likes_by_u[x]
    if len(lk)>=6: ll=lk[:]; _rs.shuffle(ll); SPL[x]=(set(ll[:len(ll)//2]),set(ll[len(ll)//2:]))  # test-likes, profile-likes
def run(kind,NEVAL=400):
    ND=np.zeros(T+1); m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        tlike,prof=SPL[x]; prof=list(prof)
        if len(tlike)<1 or len(prof)<4: continue
        po=prof[:]; rng.shuffle(po); cut=len(po)//2; pa=po[:cut]; pv=set(po[cut:])
        if not pa or (kind=='realizable' and not pv): continue
        excl=set(prof); F=[];y=[];asked=set(); nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,excl)
        for t in range(1,T+1):
            cands=[j for j in pa if j not in asked]
            if not cands: nd[t:]=nd[t-1]; break
            if kind=='helf': e=max(cands,key=lambda j:helf[j])
            else:
                tgt=tlike if kind=='true' else pv; best=None
                for j in cands:
                    a=ndcg(popb+Q@foldin(F+[Q[j]],y+[1.0-mu-bi[j]]),tgt,excl|asked|{j})
                    if best is None or a>best[0]: best=(a,j)
                e=best[1]
            asked.add(e); F.append(Q[e]); y.append(1.0-mu-bi[e]); nd[t]=ndcg(popb+Q@foldin(F,y),tlike,excl|asked)
        ND+=nd; m+=1
    return ND/m,m
print(f"\n=== {NPZ} PROFILE-mode realizable-ceiling (NDCG@10 on held-out likes) ===",flush=True)
res={}
for kind in ['helf','realizable','true']:
    nd,m=run(kind); res[kind]=nd; tag={'realizable':'REALIZABLE-ORACLE','true':'TRUE-ORACLE'}.get(kind,'HELF')
    print(f"{tag:<18}(n={m}): "+" ".join(f"{v:.3f}" for v in nd)+f" | delta {nd[T]-nd[0]:+.3f}",flush=True)
rg=res['realizable'][T]-res['helf'][T]; gap=res['true'][T]-res['helf'][T]
print(f"\nVERDICT [{NPZ}]: realizable-over-HELF={rg:+.3f}; privileged-gap={gap:+.3f}; "
      f"{'*** SELECTION-FRIENDLY (build policy) ***' if rg>0.02 else 'same negative'}",flush=True)
