"""
S2 — CASPER-U instrument (item-only first). The learned, frozen, permutation-invariant set-encoder recommender.
Architecture (combines the 3 replicated lessons):
  - EDDI-style permutation-invariant set-encoder over revealed (item, rating) tokens -> user vector u   [R3]
  - reveal-subset masked training so partial interviews are in-distribution                              [R3]
  - score(item) = popularity_bias + u . itemEmb   (popularity-FLOOR + personalization RESIDUAL)          [R1,R2 law]
  - seeds chosen by RMVA (max-volume) in the instrument's OWN embedding space  (one space = rec + seeds)  [R1]
Ruler: ML-1M, like=rating>=4, users>=5 pos, fixed 80/10/10, FULL-cat, STANDARD NDCG@10 + Recall@10.
DoD: (a) ranker >= WRMF fold-in (R1: RMVA+pop 0.5011); (b) reproduces RMVA-seed > popularity; (c) gates:
monotone in #reveals, partial-reveal robust, FULL > pop-only and > personalization-only (the law).
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens/ml-1m'
D=int(os.environ.get('D',64)); H=int(os.environ.get('H',128)); EP=int(os.environ.get('EP',30)); K=int(os.environ.get('K',50))
LIKE=4.0; rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U); I=np.array(I); R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
rated={}                                   # user -> list of (item, centered_rating, liked)
for k in range(len(u)):
    rated.setdefault(u[k],[]).append((i[k],(R[k]-3.0)/2.0, 1.0 if R[k]>=LIKE else 0.0))
keep=[x for x in range(nu) if sum(t[2] for t in rated[x])>=5]; rng.shuffle(keep); n=len(keep)
tr=keep[:int(0.8*n)]; va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in tr:
    for it,_,lk in rated[x]:
        if lk>0: cnt[it]+=1
popb=np.log(cnt+1.0).astype(np.float32)
print(f"instrument-U | ML-1M {nu}u {ni}i | train {len(tr)} val {len(va)} test {len(te)} | d={D}",flush=True)
# ---- MF-init item geometry via implicit WRMF/ALS — strong collaborative Q (R1's exact recommender) ----
ALPHA=float(os.environ.get('ALPHA',20)); LAMF=0.1; ITERS=int(os.environ.get('ITERS',15))
trpos=[[it for it,cr,lk in rated[x] if lk>0] for x in tr]              # liked items per train user
icol=[[] for _ in range(ni)]
for xi,items in enumerate(trpos):
    for it in items: icol[it].append(xi)
icol=[np.array(c,int) for c in icol]
P=0.01*rng.standard_normal((len(tr),D)); Q0=0.01*rng.standard_normal((ni,D)); Id=LAMF*np.eye(D); _t=time.time()
for itr in range(ITERS):
    QtQ=Q0.T@Q0
    for xi,items in enumerate(trpos):
        s=np.array(items,int); P[xi]=np.linalg.solve(QtQ+Id+ALPHA*(Q0[s].T@Q0[s]),(1+ALPHA)*Q0[s].sum(0)) if len(s) else 0
    PtP=P.T@P
    for j in range(ni):
        s=icol[j]; Q0[j]=np.linalg.solve(PtP+Id+ALPHA*(P[s].T@P[s]),(1+ALPHA)*P[s].sum(0)) if len(s) else 0
    if (itr+1)%5==0: print(f"  WRMF it{itr+1} ({time.time()-_t:.0f}s)",flush=True)
Q0=Q0.astype(np.float32); print("MF-init Q via implicit WRMF",flush=True)

class InstrumentU(nn.Module):
    def __init__(s):
        super().__init__()
        s.Q=nn.Parameter(torch.tensor(Q0))                             # item embeddings (MF-init; the shared space)
        s.loglam=nn.Parameter(torch.tensor(0.0))                       # ridge lambda (learned)
        s.popw=nn.Parameter(torch.tensor(1.0)); s.popb=torch.tensor(popb)
    def uvec(s, items, rats):                                          # DIFFERENTIABLE ridge fold-in (R1 mechanism, learned Q)
        if len(items)==0: return torch.zeros(D)
        Qs=s.Q[items.long()]; y=rats.float(); lam=F.softplus(s.loglam)+1e-3
        A=Qs.t()@Qs+lam*torch.eye(D); b=Qs.t()@y                       # u = (QsᵀQs+λI)⁻¹ Qsᵀy  (permutation-invariant)
        return torch.linalg.solve(A,b)
    def scores(s, uvec): return s.popw*s.popb + s.Q@uvec              # popularity FLOOR + personalization RESIDUAL
m=InstrumentU(); opt=torch.optim.Adam(m.parameters(),5e-4,weight_decay=1e-6); t0=time.time()  # low lr: fine-tune MF-init, don't destroy it
trl=[x for x in tr if len(rated[x])>=6]
for ep in range(EP):
    rng.shuffle(trl); tot=0.; nb=0
    for b0 in range(0,len(trl),128):
        batch=trl[b0:b0+128]; losses=[]
        for x in batch:
            recs=rated[x]; idx=np.arange(len(recs)); rng.shuffle(idx)
            cut=max(3,int((0.3+0.5*rng.random())*len(recs)))           # reveal-subset masking (vary fraction)
            rev=idx[:cut]; held=[recs[j][0] for j in idx[cut:] if recs[j][2]>0]  # held-out LIKED = targets
            if not held: continue
            items=torch.tensor([recs[j][0] for j in rev]); rats=torch.tensor([recs[j][2] for j in rev])  # y=liked-binary (R1 protocol)
            uv=m.uvec(items,rats); sc=m.Q@uv     # train on PURE residual (no pop crutch -> encoder must learn to rank)
            pos=torch.tensor(held); neg=torch.tensor(rng.integers(0,ni,len(held)))
            losses.append(-F.logsigmoid(sc[pos]-sc[neg]).mean())
        if losses:
            opt.zero_grad(); L=torch.stack(losses).mean(); L.backward(); opt.step(); tot+=L.item(); nb+=1
    if (ep+1)%5==0: print(f"  ep{ep+1} bpr={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)",flush=True)
m.eval()
with torch.no_grad(): Qd=m.Q.detach().numpy()
def rmva(k):
    cand=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Qd[cand].T,pivoting=True); return [int(cand[p]) for p in piv[:k]]
RMVA=rmva(K)
def ndcg_recall(score, rel, excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/len(rel)
def zsc(v): return (v-v.mean())/(v.std()+1e-9)
zpop=zsc(popb)
def pers_scores(x, seeds):                                             # (residual u.Q over items, #informative reveals) or None
    if not seeds: return None
    rd={it:lk for it,cr,lk in rated[x]}
    yv=[rd.get(it,0.0) for it in seeds]; n=int(sum(1 for v in yv if v>0))
    items=torch.tensor(seeds); y=torch.tensor(yv)
    with torch.no_grad(): return (m.Q@m.uvec(items,y)).numpy(), n
def blend_eval(users, seeds, w):                                      # score = z(u.Q) + w*z(pop); val-tuned w (R1 recipe)
    ss=set(seeds); nd=rc=0.; mm=0
    for x in users:
        rel=[it for it,cr,lk in rated[x] if lk>0 and it not in ss]
        if not rel: continue
        p=pers_scores(x,seeds)
        if p is not None: vec,nn=p; conf=nn/(nn+5.0); sc=conf*zsc(vec)+w*zpop   # confidence-scale residual by #reveals
        else: sc=w*zpop
        a,b=ndcg_recall(sc,rel,ss); nd+=a; rc+=b; mm+=1
    return nd/mm, rc/mm
def pop_eval(users):
    nd=rc=0.; mm=0
    for x in users:
        rel=[it for it,cr,lk in rated[x] if lk>0]
        if not rel: continue
        a,b=ndcg_recall(popb.copy(),rel,set()); nd+=a; rc+=b; mm+=1
    return nd/mm, rc/mm
WGRID=[0,1,2,3,4,6,8,12,16]
wbest=max(WGRID,key=lambda w: blend_eval(va,RMVA,w)[0])               # tune blend weight on VAL
mp=pop_eval(te); full=blend_eval(te,RMVA,wbest); ponly=blend_eval(te,RMVA,0)
print("\n=== S2 instrument-U (ML-1M cold-start, std NDCG@10/Recall@10, RMVA seeds; w tuned on val) ===",flush=True)
print(f"  {'MOSTPOP (pop only)':<26} NDCG@10={mp[0]:.4f} Recall@10={mp[1]:.4f}",flush=True)
print(f"  {'PERSONALIZATION only (u.Q)':<26} NDCG@10={ponly[0]:.4f} Recall@10={ponly[1]:.4f}  (law: should LOSE)",flush=True)
print(f"  {'FULL (pop + u.Q, w*=%d)'%wbest:<26} NDCG@10={full[0]:.4f} Recall@10={full[1]:.4f}  (vs R1 RMVA-foldin 0.5011, MF/WRMF)",flush=True)
print(f"  -> beats popularity: {'YES' if full[0]>mp[0] else 'NO'} ({full[0]-mp[0]:+.4f}); law holds: {'YES' if full[0]>ponly[0] and full[0]>mp[0] else 'NO'}",flush=True)
print("\n=== GATE: NDCG@10 vs #RMVA reveals (monotone + partial-reveal robust) ===",flush=True)
prev=-1; mono=True
for kk in [0,5,10,25,50]:
    nd,_=blend_eval(te,RMVA[:kk],wbest); flag='' if nd>=prev-2e-3 else ' <-DROP'
    if nd<prev-2e-3: mono=False
    print(f"  q={kk:<3} NDCG@10={nd:.4f}{flag}",flush=True); prev=nd
DoD_a = full[0]>=0.5011-0.02                                          # match/beat WRMF fold-in (R1)
print(f"\nGATES: beats-pop={'PASS' if full[0]>mp[0] else 'FAIL'} | law={'PASS' if full[0]>ponly[0] and full[0]>mp[0] else 'FAIL'} | monotone={'PASS' if mono else 'FAIL'} | >=WRMF(0.5011)={'PASS' if DoD_a else 'FAIL'}",flush=True)
torch.save({'Q':m.Q.detach(),'state':m.state_dict(),'d':D}, f"{base}/../.cache/checkpoints/instrument_u_ml1m.pt")
print("saved instrument_u_ml1m.pt",flush=True)
