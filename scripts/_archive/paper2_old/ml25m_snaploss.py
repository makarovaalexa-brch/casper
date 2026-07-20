"""
ML-25M PHASE-2 STEP 3 -- SNAP-LOSS: post-hoc realization of the trained continuous actor's queries.
Variants (same actor, same graded geometric answers, same frozen enc_v1_ml25m, same ruler):
  cont     : fold the raw off-manifold unit query (the actor's native regime)  [reference]
  concsnap : snap each query to the nearest CONCEPT direction (cosine over unit Ac), fold the CONCEPT vector
  pairsnap : realize each query as the best ITEM-PAIR difference d=(e_i-e_j)/|.| (top/bottom-K over pool items)
Off-manifold gain = cont - concsnap (and cont - pairsnap). ML-1M anchor: snap-loss -0.037/-0.040 on headline.
Ruler: NDCG@50 (primary) full+tail, @10 alongside optional; te cohort, seed-avg EVALSEEDS.
Env: ACTORCK(policy_ml25m_d1_best.pt) EVALSEEDS(1,2,3,7,11) PAIRK(64) NPAIRITEMS(600)
"""
import os, time, numpy as np, torch, torch.nn as nn
t0=time.time(); out='C:/dev/phd/casper/data/movielens/.cache/ml25m'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0; T=8; HID=128
M=np.load(f'{out}/meta.npz'); uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; mu=float(M['mu'])
ni=int(M['ni']); trU=M['trU']; te=M['te']
Q=np.load(f'{out}/Q_svd.npy').astype(np.float32); bi=np.load(f'{out}/bi_svd.npy').astype(np.float32)
popb=np.log(cnt+1.0).astype(np.float32)
Ac=np.load(f'{out}/Ac_concept.npy').astype(np.float32); nc=Ac.shape[0]
Acn=(Ac/(np.linalg.norm(Ac,axis=1,keepdims=True)+1e-9)).astype(np.float32)
_CN=float(np.linalg.norm(Ac,axis=1).mean())
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/max(cnt.sum(),1)
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
PAIRK=int(os.environ.get('PAIRK',64)); NPI=int(os.environ.get('NPAIRITEMS',600))
PIT=order_pop[:NPI]; PE=Q[PIT]  # pair items = top-600 popular (the askable pool)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{out}/enc_v1_ml25m.pt')); enc.eval()
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,HID),nn.ReLU(),nn.Linear(HID,HID),nn.ReLU(),nn.Linear(HID,D))
    def forward(s,u,tt):
        B=u.shape[0]; tn=torch.full((B,1),float(tt)); return s.f(torch.cat([u,tn],1))
actor=Actor(); ck=os.environ.get('ACTORCK',f'{out}/policy_ml25m_d1_best.pt')
sd=torch.load(ck); actor.load_state_dict(sd['actor'] if 'actor' in sd else sd); actor.eval()
print(f"loaded actor {os.path.basename(ck)}",flush=True)
def enc_u_np(rows):
    if not rows: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(rows),D+1),np.float32); mk=np.ones((1,len(rows)),np.float32)
    for q,(f,r) in enumerate(rows): tk[0,q,:D]=f; tk[0,q,D]=r
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def pair_realize(qn):  # best (e_i - e_j)/|.| by cos to qn, i from top-K of s, j from bottom-K
    s=PE@qn; ti=np.argsort(-s)[:PAIRK]; bj=np.argsort(s)[:PAIRK]
    best=-2.; bd=None
    for i in ti:
        d=PE[i]-PE[bj]; dn=np.linalg.norm(d,axis=1)+1e-9
        c=(s[i]-s[bj])/dn; c[dn<1e-5]=-2.
        k=int(np.argmax(c))
        if c[k]>best: best=float(c[k]); bd=d[k]/dn[k]
    return bd.astype(np.float32), best
# cohort
selm=np.isin(uu,te); euu=uu[selm]; eii=ii[selm]; eR=rr[selm]
rat_by_u={}
for k in range(len(euu)): rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
POSNEG=np.load(f'{out}/posneg.npy') if os.path.exists(f'{out}/posneg.npy') else None
POS,NEG=(float(POSNEG[0]),float(POSNEG[1])) if POSNEG is not None else (0.734,-0.664)  # train-sample residual means (matches trainer)
_Wk={K:1./np.log2(np.arange(2,K+2)) for K in (10,50)}
def ndcg(u,rel,excl,K,tail):
    sc=(popb+Q@u).astype(np.float64).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in rel if not headmask[t]]
    if not rel: return None
    W=_Wk[K]; top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]; rs=set(rel)
    dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
seeds=[int(s) for s in os.environ.get('EVALSEEDS','1,2,3,7,11').split(',')]
VAR=['cont','concsnap','pairsnap']
res={v:{'F':[], 'T':[]} for v in VAR}; simacc={v:[] for v in VAR}
for sdv in seeds:
    r=np.random.default_rng(sdv); acc={v:[0.,0.] for v in VAR}; m=0; mt=0; sims={v:[] for v in VAR}
    for x in te.tolist():
        if x not in rat_by_u: continue
        its=[j for j,_ in rat_by_u[x]]
        if len(its)<6: continue
        il=its[:]; r.shuffle(il); prof=set(il[:len(il)//2]); vtest=il[len(il)//2:]
        rd=dict(rat_by_u[x]); tlike=set(j for j in vtest if rd[j]>=LIKE)
        if not tlike: continue
        rel=list(tlike); rel_t=[t for t in rel if not headmask[t]]; ht=1 if rel_t else 0; m+=1; mt+=ht
        us=enc_u_np([(Q[j],rd[j]-mu-bi[j]) for j in prof]); usn=us/(np.linalg.norm(us)+1e-9)
        TK={v:[] for v in VAR}
        for v in VAR:
            u=np.zeros(D,np.float32)
            for t in range(T):
                with torch.no_grad(): qv=actor(torch.tensor(u[None],dtype=torch.float32),t/8.).numpy()[0]
                qn=(qv/(np.linalg.norm(qv)+1e-9)).astype(np.float32)
                if v=='concsnap':
                    k=int(np.argmax(Acn@qn)); sims[v].append(float(Acn[k]@qn)); dirn=Acn[k]; fe=Ac[k].astype(np.float32)
                elif v=='pairsnap':
                    d,c=pair_realize(qn); sims[v].append(c); dirn=d; fe=(d*_CN).astype(np.float32)
                else:
                    dirn=qn; fe=(qn*_CN).astype(np.float32); sims[v].append(1.0)
                cf=float(usn@dirn); a=float(NEG+(POS-NEG)*(cf+1)/2)
                TK[v].append((fe,a)); u=enc_u_np(TK[v])
            uf=enc_u_np(TK[v]); excl=set(prof)
            vf=ndcg(uf,rel,excl,50,False); acc[v][0]+=vf if vf else 0
            if ht:
                vt=ndcg(uf,rel,excl,50,True); acc[v][1]+=vt if vt else 0
    for v in VAR:
        res[v]['F'].append(acc[v][0]/max(m,1)); res[v]['T'].append(acc[v][1]/max(mt,1)); simacc[v].append(np.mean(sims[v]))
    print(f"seed{sdv}: "+" | ".join(f"{v} F={acc[v][0]/max(m,1):.4f} T={acc[v][1]/max(mt,1):.4f}" for v in VAR)+f"  ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== SNAP-LOSS (te, seed-avg {seeds}, n={m}, n_tail={mt}) @q8 NDCG@50 ===",flush=True)
for v in VAR:
    print(f"  {v:<9}: FULL {np.mean(res[v]['F']):.4f}+/-{np.std(res[v]['F']):.4f}  TAIL {np.mean(res[v]['T']):.4f}+/-{np.std(res[v]['T']):.4f}  cos(q,realized)={np.mean(simacc[v]):.3f}",flush=True)
cf_=np.mean(res['cont']['F']); ct=np.mean(res['cont']['T'])
print(f"\n>>> SNAP-LOSS concsnap: full={np.mean(res['concsnap']['F'])-cf_:+.4f} tail={np.mean(res['concsnap']['T'])-ct:+.4f}",flush=True)
print(f">>> SNAP-LOSS pairsnap: full={np.mean(res['pairsnap']['F'])-cf_:+.4f} tail={np.mean(res['pairsnap']['T'])-ct:+.4f}",flush=True)
print(f"DONE ({time.time()-t0:.0f}s)",flush=True)
