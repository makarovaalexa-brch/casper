"""
DECISIVE diagnostic: does elicitation reduce MODEL ERROR (RMSE) and grow RECALL@K with #questions, even though
top-10 NDCG is flat? The elicitation literature (Golbandi'11, Rashid'02/08) measures RMSE/MAE, not top-10 NDCG.
Reuses the already-trained calibrated biased-SVD factors (Q_svd.npy, bi_svd.npy) -- NO retraining.

Per test user: hold out HALF of their rated items (any rating) as TEST; the rest is the answerable PROFILE.
Interview asks profile items in selector order (rmva/helf/popular/random); each answer folds the taste residual
(r - mu - bi) via ridge. After q questions, predict every held-out item:  r_hat_j = mu + bi_j + q_j . u.
Report, vs q:  RMSE (held-out ratings)  +  Recall@10/@50 (held-out LIKES, full-catalogue ranking, exclude profile).
"""
import os, numpy as np
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
# mu over train ratings
sm=0.;c=0
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
import scipy.linalg as sla
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVArank={int(ca[p]):r for r,p in enumerate(piv)}
H5=np.zeros((ni,5))
for k in range(len(uu)):
    if uu[k] in trU: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
def order_key(kind,prof):
    if kind=='rmva': return sorted(prof,key=lambda j:RMVArank.get(j,10**9))
    if kind=='helf': return sorted(prof,key=lambda j:-helf[j])
    if kind=='pop':  return sorted(prof,key=lambda j:-cnt[j])
    pr=prof[:]; rng.shuffle(pr); return pr
def foldin(seeds,targ):
    Qs=Q[seeds]; A=Qs.T@Qs+LAMFI*np.eye(D); return np.linalg.solve(A,Qs.T@targ)
_rs=np.random.default_rng(123)
def run(kind):
    RMSE=np.zeros(T+1); R10=np.zeros(T+1); R50=np.zeros(T+1); m=0
    for x in te:
        rd=dict(rat_by_u[x]); items=list(rd)
        if len(items)<6: continue
        il=items[:]; _rs.shuffle(il); cut=len(il)//2
        test=il[:cut]; prof=il[cut:]                      # held-out (any rating) vs answerable profile
        tlike=[j for j in test if rd[j]>=4]
        if not tlike or not prof: continue
        tj=np.array(test); ty=np.array([rd[j] for j in test],np.float32)
        order=order_key(kind,prof); seeds=[]; targ=[]
        for t in range(T+1):
            if t>0 and t-1<len(order):
                e=order[t-1]; seeds.append(e); targ.append(rd[e]-mu-bi[e])
            u=foldin(np.array(seeds),np.array(targ,np.float32)) if seeds else np.zeros(D)
            pred=mu+bi[tj]+Q[tj]@u
            RMSE[t]+=np.sqrt(np.mean((pred-ty)**2))
            # ranking over full catalogue, exclude profile+seeds
            sc=popb+Q@u; sc[prof]=-1e9
            top=np.argsort(-sc)[:50]; rs=set(tlike)
            R10[t]+=len(set(top[:10].tolist())&rs)/len(tlike); R50[t]+=len(set(top.tolist())&rs)/len(tlike)
        m+=1
    return RMSE/m,R10/m,R50/m
print(f"calibrated biased-SVD factors; {len(te)} test users; T={T}; held-out=half of rated items\n",flush=True)
for kind in ['rmva','helf','pop','random']:
    rmse,r10,r50=run(kind)
    print(f"--- {kind} ---",flush=True)
    print(f"  RMSE     q0={rmse[0]:.4f}  q5={rmse[5]:.4f}  q10={rmse[10]:.4f}  q15={rmse[T]:.4f}  (delta {rmse[T]-rmse[0]:+.4f})",flush=True)
    print(f"  Recall10 q0={r10[0]:.4f}  q5={r10[5]:.4f}  q10={r10[10]:.4f}  q15={r10[T]:.4f}  (delta {r10[T]-r10[0]:+.4f})",flush=True)
    print(f"  Recall50 q0={r50[0]:.4f}  q5={r50[5]:.4f}  q10={r50[10]:.4f}  q15={r50[T]:.4f}  (delta {r50[T]-r50[0]:+.4f})",flush=True)
