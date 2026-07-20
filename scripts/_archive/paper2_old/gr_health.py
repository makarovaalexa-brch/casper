"""
Goodreads MTC PHASE-1 health gate on the V1 neural attention fold-in encoder (mirrors ml25m_health_enc).
Ruler (pre-stated): primary NDCG@142 (catalogue-fraction-matched), NDCG@10 alongside; tail = Cremonesi
head-33%. V1 decoder: sc = popb + Q.u, popb=log(cnt+1) (beta=1); ridge lam=5. MOSTPOP q0 = popb.
Held-out disjoint-targets. Compares MOSTPOP(q0), ridge, ENCODER at reveals {0,8,20} with selectors
{random,entropy}, and ENCODER/ridge FULL-PROFILE fold. NDCG@{10,142} full + tail.
GATE (K=142): encoder full-profile - MOSTPOP substantially positive AND monotone no-harm.
"""
import os, time, numpy as np, torch, torch.nn as nn
t0=time.time(); GR='C:/dev/phd/casper/.cache/goodreads'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0; LAM=5.0
NEVAL=int(os.environ.get('NEVAL',500)); REVEALS=[0,8,20]; KP=int(os.environ.get('KP',142)); Ks=[10,KP]
B=np.load(f'{GR}/base.npz')
uu=B['uu']; ii=B['ii']; rr=B['rr']; cnt=B['cnt']; mu=float(B['mu']); ni=int(B['ni']); va=B['va']; te=B['te']
Q=np.load(f'{GR}/Q_svd.npy').astype(np.float32); bi=np.load(f'{GR}/bi_svd.npy').astype(np.float32)
popb=np.log(cnt+1.0).astype(np.float32)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/max(cnt.sum(),1)
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
# entropy selector (per-item, 5-bin over all rows)
H5=np.zeros((ni,5)); b5=np.clip(rr.astype(int),1,5)-1; np.add.at(H5,(ii,b5.astype(int)),1.0)
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); entv=-(pm*np.log(pm+1e-12)).sum(1)
ORD={'entropy':list(np.argsort(-entv)),'pop':list(np.argsort(-cnt))}
evalset=np.array(sorted(set(va.tolist())|set(te.tolist()))); selm=np.isin(uu,evalset)
euu=uu[selm]; eii=ii[selm]; eR=rr[selm]; rat_by_u={}
for k in range(len(euu)): rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
likes_by_u={x:[j for j,r in v if r>=LIKE] for x,v in rat_by_u.items()}
print(f"cohort dicts built ({time.time()-t0:.0f}s); ni={ni} KP={KP}",flush=True)
_rs=np.random.default_rng(123); SPL={}
for x in (va.tolist()+te.tolist()):
    lk=list(likes_by_u.get(x,[]))
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{GR}/enc_v1_gr.pt')); enc.eval()
def enc_u(rev):
    if not rev: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(rev),D+1),np.float32); mk=np.ones((1,len(rev)),np.float32)
    for q,(j,res) in enumerate(rev): tk[0,q,:D]=Q[j]; tk[0,q,D]=res
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def ridge_u(rev):
    if not rev: return np.zeros(D,np.float32)
    F=np.array([Q[j] for j,_ in rev],np.float32); y=np.array([r for _,r in rev],np.float32)
    return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@y)
def ndcg_at(u,rel,excl,K,tail):
    sc=(popb+Q@u).astype(np.float64).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in rel if not headmask[t]]
    if not rel: return None
    W=1./np.log2(np.arange(2,K+2)); top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]
    rs=set(rel); dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
def eval_curve(kind):
    accE={(q,K,tl):0. for q in REVEALS for K in Ks for tl in (0,1)}; m=0; mt=0
    for x in te[:NEVAL].tolist():
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profile=set(rd)-test; rel=list(test)
        if len(rel)<2: continue
        rel_t=[t for t in rel if not headmask[t]]
        order=list(np.random.default_rng(x).permutation(ni)) if kind=='random' else ORD[kind]
        m+=1; ht=1 if rel_t else 0; mt+=ht; asked=set(); seeds=[]
        for t in range(max(REVEALS)+1):
            if t>0:
                e=next((it for it in order if it not in asked and it not in test),None)
                if e is not None:
                    asked.add(e)
                    if e in rd: seeds.append((e,rd[e]-mu-bi[e]))
            if t in REVEALS:
                uE=enc_u(seeds); excl=profile|asked
                for K in Ks:
                    v=ndcg_at(uE,rel,excl,K,False)
                    if v is not None: accE[(t,K,0)]+=v
                    if ht:
                        vt=ndcg_at(uE,rel,excl,K,True)
                        if vt is not None: accE[(t,K,1)]+=vt
    for k in accE: accE[k]/=(mt if k[2]==1 else m) or 1
    return accE,m,mt
def eval_fullprof():
    accE={(K,tl):0. for K in Ks for tl in (0,1)}; accR=dict(accE); accP=dict(accE); m=0; mt=0
    for x in te[:NEVAL].tolist():
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profile=[j for j in rd if j not in test]; rel=list(test)
        if len(rel)<2 or len(profile)<2: continue
        rel_t=[t for t in rel if not headmask[t]]; ht=1 if rel_t else 0; m+=1; mt+=ht
        seeds=[(j,rd[j]-mu-bi[j]) for j in profile]; uE=enc_u(seeds); uR=ridge_u(seeds); excl=set(profile)
        for K in Ks:
            accE[(K,0)]+=ndcg_at(uE,rel,excl,K,False) or 0; accR[(K,0)]+=ndcg_at(uR,rel,excl,K,False) or 0
            accP[(K,0)]+=ndcg_at(np.zeros(D),rel,excl,K,False) or 0
            if ht:
                accE[(K,1)]+=ndcg_at(uE,rel,excl,K,True) or 0; accR[(K,1)]+=ndcg_at(uR,rel,excl,K,True) or 0
                accP[(K,1)]+=ndcg_at(np.zeros(D),rel,excl,K,True) or 0
    for d in (accE,accR,accP):
        for k in d: d[k]/=(mt if k[1]==1 else m) or 1
    return accE,accR,accP,m,mt
print(f"\n=== GOODREADS PHASE-1 V1 NEURAL ENCODER health gate; ni={ni}; n_test={NEVAL} ===",flush=True)
print("[held-out disjoint targets; sc=popb+Q.u; tail=Cremonesi head-33%]",flush=True)
accEfp,accRfp,accP,mfp,mtfp=eval_fullprof()
def row(tag,d): return (f"{tag:<14}| @10 full={d[(10,0)]:.4f} tail={d[(10,1)]:.4f} | @{KP} full={d[(KP,0)]:.4f} tail={d[(KP,1)]:.4f}")
print(f"\n-- FULL-PROFILE fold (n={mfp}, n_tail={mtfp}) --",flush=True)
print(row("MOSTPOP(q0)",accP),flush=True); print(row("ridge",accRfp),flush=True); print(row("ENCODER",accEfp),flush=True)
hr=accEfp[(KP,0)]-accP[(KP,0)]
print(f"\n>>> HEADROOM encoder full-profile - MOSTPOP: @{KP} full={hr:+.4f} tail={accEfp[(KP,1)]-accP[(KP,1)]:+.4f} @10 full={accEfp[(10,0)]-accP[(10,0)]:+.4f}",flush=True)
print(f">>> HEADROOM ridge   full-profile - MOSTPOP: @{KP} full={accRfp[(KP,0)]-accP[(KP,0)]:+.4f}",flush=True)
for kind in ['random','entropy','pop']:
    accE,m,mt=eval_curve(kind)
    print(f"\n-- selector={kind} (n={m}, n_tail={mt}); ENCODER fold at reveals {REVEALS} --",flush=True)
    for q in REVEALS:
        print(f"  q{q:<2} ENC | @10 full={accE[(q,10,0)]:.4f} tail={accE[(q,10,1)]:.4f} | @{KP} full={accE[(q,KP,0)]:.4f} tail={accE[(q,KP,1)]:.4f}",flush=True)
    mono10=accE[(REVEALS[-1],10,0)]>=accE[(0,10,0)]-3e-3; monoK=accE[(REVEALS[-1],KP,0)]>=accE[(0,KP,0)]-3e-3
    print(f"  ENC monotone no-harm @10={'OK' if mono10 else 'HURTS'} @{KP}={'OK' if monoK else 'HURTS'} | elicit gain @{KP} full={accE[(REVEALS[-1],KP,0)]-accE[(0,KP,0)]:+.4f} tail={accE[(REVEALS[-1],KP,1)]-accE[(0,KP,1)]:+.4f}",flush=True)
print(f"\n=== GATE (primary K={KP}): encoder full-profile - MOSTPOP = {hr:+.4f} ===",flush=True)
print("GATE VERDICT: PASS" if hr>0.02 else "GATE VERDICT: WEAK/FAIL", flush=True)
print(f"DONE ({time.time()-t0:.0f}s)",flush=True)
