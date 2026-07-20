"""
ML-25M PHASE-1C health gate on the V1 NEURAL attention fold-in encoder.
Ruler (pre-stated): primary NDCG@50 (catalogue-depth-matched to ML-1M's K=10), NDCG@10 alongside;
tail = Cremonesi head-33% (cumulative-popularity head masked from candidates + rel). V1 decoder
convention (encoder_recon): sc = popb + Q.u, popb=log(cnt+1) (beta=1); ridge lam=5. MOSTPOP q0 = popb.

Held-out disjoint-targets protocol (the correct no-harm / headroom ruler). Compares:
MOSTPOP (q0), ridge fold, ENCODER fold at reveals {0,8,20} with selectors {random,rmva,entropy},
and ENCODER / ridge FULL-PROFILE fold. NDCG@{10,50} full + tail.
GATE (K=50): encoder full-profile - MOSTPOP substantially positive AND monotone no-harm.
If FAIL: depth curve full-profile - MOSTPOP at K={10,20,50,100}.
"""
import os, time, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
t0=time.time(); out='C:/dev/phd/casper/data/movielens/.cache/ml25m'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0; LAM=5.0
NEVAL=int(os.environ.get('NEVAL',500)); REVEALS=[0,8,20]
M=np.load(f'{out}/meta.npz')
uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; mu=float(M['mu']); ni=int(M['ni'])
va=M['va']; te=M['te']
Q=np.load(f'{out}/Q_svd.npy').astype(np.float32); bi=np.load(f'{out}/bi_svd.npy').astype(np.float32)
popb=np.log(cnt+1.0).astype(np.float32)
# head-33% cumulative popularity (Cremonesi)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/max(cnt.sum(),1)
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
# per-item entropy (10-bin) from train-like histogram in meta? meta has no H5 -> compute from te+va-excluded train via cnt only.
# entropy selector needs H5; recompute quickly over ALL rows (fine for a global item stat).
H5=np.zeros((ni,10)); b5=np.clip(np.round(rr*2).astype(int),1,10)-1; np.add.at(H5,(ii,b5),1.0)
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); entv=-(pm*np.log(pm+1e-12)).sum(1)
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVA=[int(ca[p]) for p in piv]
ORD={'rmva':RMVA,'entropy':list(np.argsort(-entv))}
# eval cohort dicts (va+te)
evalset=np.array(sorted(set(va.tolist())|set(te.tolist()))); selm=np.isin(uu,evalset)
euu=uu[selm]; eii=ii[selm]; eR=rr[selm]
rat_by_u={}
for k in range(len(euu)):
    rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
likes_by_u={x:[j for j,r in v if r>=LIKE] for x,v in rat_by_u.items()}
print(f"cohort dicts built ({time.time()-t0:.0f}s)",flush=True)
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
enc=Enc(); enc.load_state_dict(torch.load(f'{out}/enc_v1_ml25m.pt')); enc.eval()
def enc_u(rev):  # rev list of (item,residual)
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
Ks=[10,50]
def eval_curve(kind):  # held-out disjoint targets, encoder + ridge at REVEALS
    accE={(q,K,tl):0. for q in REVEALS for K in Ks for tl in (0,1)}
    accR={(q,K,tl):0. for q in REVEALS for K in Ks for tl in (0,1)}
    m=0; mt=0
    for x in te[:NEVAL].tolist():
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profile=set(rd)-test; rel=list(test)
        if len(rel)<2: continue
        rel_t=[t for t in rel if not headmask[t]]
        order=list(np.random.default_rng(x).permutation(ni)) if kind=='random' else ORD[kind]
        m+=1; ht = 1 if rel_t else 0; mt+=ht
        asked=set(); seeds=[]
        maxq=max(REVEALS)
        for t in range(maxq+1):
            if t>0:
                e=next((it for it in order if it not in asked and it not in test),None)
                if e is not None:
                    asked.add(e)
                    if e in rd: seeds.append((e,rd[e]-mu-bi[e]))
            if t in REVEALS:
                uE=enc_u(seeds); uR=ridge_u(seeds); excl=profile|asked
                for K in Ks:
                    vE=ndcg_at(uE,rel,excl,K,False); vR=ndcg_at(uR,rel,excl,K,False)
                    if vE is not None: accE[(t,K,0)]+=vE
                    if vR is not None: accR[(t,K,0)]+=vR
                    if ht:
                        vEt=ndcg_at(uE,rel,excl,K,True); vRt=ndcg_at(uR,rel,excl,K,True)
                        if vEt is not None: accE[(t,K,1)]+=vEt
                        if vRt is not None: accR[(t,K,1)]+=vRt
    for d in (accE,accR):
        for k in d: d[k]/= (mt if k[2]==1 else m) or 1
    return accE,accR,m,mt
def eval_fullprof():
    accE={(K,tl):0. for K in Ks for tl in (0,1)}; accR=dict(accE); accP=dict(accE)
    m=0; mt=0
    for x in te[:NEVAL].tolist():
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profile=[j for j in rd if j not in test]; rel=list(test)
        if len(rel)<2 or len(profile)<2: continue
        rel_t=[t for t in rel if not headmask[t]]; ht=1 if rel_t else 0; m+=1; mt+=ht
        seeds=[(j,rd[j]-mu-bi[j]) for j in profile]
        uE=enc_u(seeds); uR=ridge_u(seeds); excl=set(profile)
        for K in Ks:
            accE[(K,0)]+=ndcg_at(uE,rel,excl,K,False) or 0; accR[(K,0)]+=ndcg_at(uR,rel,excl,K,False) or 0
            accP[(K,0)]+=ndcg_at(np.zeros(D),rel,excl,K,False) or 0   # MOSTPOP (u=0 => popb only)
            if ht:
                accE[(K,1)]+=ndcg_at(uE,rel,excl,K,True) or 0; accR[(K,1)]+=ndcg_at(uR,rel,excl,K,True) or 0
                accP[(K,1)]+=ndcg_at(np.zeros(D),rel,excl,K,True) or 0
    for d in (accE,accR,accP):
        for k in d: d[k]/=(mt if k[1]==1 else m) or 1
    return accE,accR,accP,m,mt
print(f"\n=== ML-25M PHASE-1C V1 NEURAL ENCODER health gate; ni={ni}; n_test={NEVAL} ===",flush=True)
print("[held-out disjoint targets; sc=popb+Q.u (V1 conv); tail=Cremonesi head-33%]",flush=True)
accEfp,accRfp,accP,mfp,mtfp=eval_fullprof()
def row(tag,d):
    return (f"{tag:<16}| "
            f"@10 full={d[(10,0)]:.4f} tail={d[(10,1)]:.4f} | @50 full={d[(50,0)]:.4f} tail={d[(50,1)]:.4f}")
print(f"\n-- FULL-PROFILE fold (n={mfp}, n_tail={mtfp}) --",flush=True)
print(row("MOSTPOP(q0)",accP),flush=True)
print(row("ridge",accRfp),flush=True)
print(row("ENCODER",accEfp),flush=True)
print(f"\n>>> HEADROOM encoder full-profile - MOSTPOP:  @50 full={accEfp[(50,0)]-accP[(50,0)]:+.4f}  @50 tail={accEfp[(50,1)]-accP[(50,1)]:+.4f}  @10 full={accEfp[(10,0)]-accP[(10,0)]:+.4f}",flush=True)
print(f">>> HEADROOM ridge    full-profile - MOSTPOP:  @50 full={accRfp[(50,0)]-accP[(50,0)]:+.4f}",flush=True)
for kind in ['random','rmva','entropy']:
    accE,accR,m,mt=eval_curve(kind)
    print(f"\n-- selector={kind} (n={m}, n_tail={mt}); ENCODER fold at reveals {REVEALS} --",flush=True)
    for q in REVEALS:
        print(f"  q{q:<2} ENC | @10 full={accE[(q,10,0)]:.4f} tail={accE[(q,10,1)]:.4f} | @50 full={accE[(q,50,0)]:.4f} tail={accE[(q,50,1)]:.4f}",flush=True)
    mono10=accE[(REVEALS[-1],10,0)]>=accE[(0,10,0)]-3e-3; mono50=accE[(REVEALS[-1],50,0)]>=accE[(0,50,0)]-3e-3
    print(f"  ENC monotone no-harm @10={'OK' if mono10 else 'HURTS'} @50={'OK' if mono50 else 'HURTS'} | ENC elicit gain @50 full={accE[(REVEALS[-1],50,0)]-accE[(0,50,0)]:+.4f} tail={accE[(REVEALS[-1],50,1)]-accE[(0,50,1)]:+.4f}",flush=True)
# GATE
hr=accEfp[(50,0)]-accP[(50,0)]
print(f"\n=== GATE (primary K=50): encoder full-profile - MOSTPOP = {hr:+.4f} (ML-1M ref +0.10) ===",flush=True)
if hr>0.05:
    print("GATE VERDICT: PASS-band (substantial). See report.",flush=True)
else:
    print("GATE VERDICT: FAIL-band (headroom small). Running depth curve...",flush=True)
    # depth curve full-profile - MOSTPOP at K=10/20/50/100
    KD=[10,20,50,100]
    accE2={K:0. for K in KD}; accP2={K:0. for K in KD}; m=0
    for x in te[:NEVAL].tolist():
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profile=[j for j in rd if j not in test]; rel=list(test)
        if len(rel)<2 or len(profile)<2: continue
        seeds=[(j,rd[j]-mu-bi[j]) for j in profile]; uE=enc_u(seeds); excl=set(profile); m+=1
        for K in KD:
            accE2[K]+=ndcg_at(uE,rel,excl,K,False) or 0; accP2[K]+=ndcg_at(np.zeros(D),rel,excl,K,False) or 0
    print("  K     encoder   MOSTPOP   headroom",flush=True)
    for K in KD:
        e=accE2[K]/m; p=accP2[K]/m; print(f"  {K:<5} {e:.4f}   {p:.4f}   {e-p:+.4f}",flush=True)
print(f"\nDONE ({time.time()-t0:.0f}s)",flush=True)
