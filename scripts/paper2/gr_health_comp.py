"""
Goodreads COMPOSITE PHASE-1E health gate (eval-only). Reads composite artifacts. Ruler pre-stated:
primary NDCG@KP (KP=round(0.0027*ni) passed via env), NDCG@10 alongside; tail=Cremonesi head-33%.
Decoder sc=popb+Q.u, popb=log(cnt+1); ridge lam=5. Held-out disjoint targets (rng123).
Reports, at primary K:
  (A) FULL cohort full-profile fold MOSTPOP/ridge/ENCODER (+ headroom, + headroom by profile-size);
  (B) CONCEPT-channel monotone no-harm elicitation (random/entropy/pop) with answered-token counts;
  (C) COLD cohort (<=10 ratings) full-profile fold + headroom + concept answered-tokens.
GATE (primary K): encoder full-profile - MOSTPOP >= +0.025 AND monotone no-harm.
"""
import os, time, numpy as np, torch, torch.nn as nn
t0=time.time(); GR='C:/dev/phd/casper/.cache/goodreads'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0; LAM=5.0
NEVAL=int(os.environ.get('NEVAL',500)); REVEALS=[0,8,20]; KP=int(os.environ.get('KP',142)); Ks=[10,KP]
COLD_TH=int(os.environ.get('COLD_TH',10))
B=np.load(f'{GR}/base_comp.npz')
uu=B['uu']; ii=B['ii']; rr=B['rr']; cnt=B['cnt']; mu=float(B['mu']); ni=int(B['ni']); nu=int(B['nu'])
va=B['va']; te=B['te']; H5=B['H5']
Q=np.load(f'{GR}/Q_svd_comp.npy').astype(np.float32); bi=np.load(f'{GR}/bi_svd_comp.npy').astype(np.float32)
popb=np.log(cnt+1.0).astype(np.float32)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/max(cnt.sum(),1)
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
C=np.load(f'{GR}/concepts_comp.npz',allow_pickle=True); Ac=C['Ac'].astype(np.float32)
cf_flat=C['citems_flat']; cf_off=C['citems_off']; ctags=C['ctags']; nc=Ac.shape[0]
citems=[cf_flat[cf_off[k]:cf_off[k+1]] for k in range(nc)]
ccount=np.array([len(c) for c in citems],np.float64)
cH=np.zeros((nc,5))
for c in range(nc): cH[c]=H5[citems[c]].sum(0)
cpm=cH/np.clip(cH.sum(1,keepdims=True),1,None); cdiv=-(cpm*np.log(cpm+1e-12)).sum(1)
CONC_ORD={'entropy':list(np.argsort(-cdiv)),'pop':list(np.argsort(-ccount))}
item2c=[[] for _ in range(ni)]
for c in range(nc):
    for j in citems[c]: item2c[int(j)].append(c)
upc=np.bincount(uu,minlength=nu)
for th in (10,15):
    print(f"[share] users <= {th} ratings: ALL {(upc<=th).mean()*100:.1f}% | test {(upc[te]<=th).mean()*100:.1f}% ({int((upc[te]<=th).sum())}/500)",flush=True)
COLD=set(int(x) for x in te if upc[x]<=COLD_TH)
print(f"[cold] THRESH<={COLD_TH}: test qualifying={len(COLD)}",flush=True)
evalset=np.array(sorted(set(va.tolist())|set(te.tolist()))); selm=np.isin(uu,evalset)
euu=uu[selm]; eii=ii[selm]; eR=rr[selm]; rat_by_u={}
for k in range(len(euu)): rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
_rs=np.random.default_rng(123); SPL={}
for x in (va.tolist()+te.tolist()):
    lk=[j for j,r in rat_by_u.get(x,[]) if r>=LIKE]
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
print(f"cohort dicts built ({time.time()-t0:.0f}s); ni={ni} nc={nc} KP={KP}",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
ENCF=os.environ.get('ENCF',f'{GR}/enc_v1_grcomp_GATE.pt')
enc=Enc(); enc.load_state_dict(torch.load(ENCF)); enc.eval()
def enc_u(toks):
    if not toks: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(toks),D+1),np.float32); mk=np.ones((1,len(toks)),np.float32)
    for q,(f,res) in enumerate(toks): tk[0,q,:D]=f; tk[0,q,D]=res
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def ridge_u(rev):
    if not rev: return np.zeros(D,np.float32)
    F=np.array([f for f,_ in rev],np.float32); y=np.array([r for _,r in rev],np.float32)
    return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@y)
def ndcg_at(u,rel,excl,K,tail):
    sc=(popb+Q@u).astype(np.float64).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in rel if not headmask[t]]
    if not rel: return None
    W=1./np.log2(np.arange(2,K+2)); top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]
    rs=set(rel); dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
def conc_answers(profset, rd):
    acc={}
    for j in profset:
        for c in item2c[j]: acc.setdefault(c,[]).append(rd[j]-mu-bi[j])
    return {c:float(np.mean(v)) for c,v in acc.items() if len(v)>=2}
def eval_fullprof(cohort=None):
    accE={(K,tl):0. for K in Ks for tl in (0,1)}; accR=dict(accE); accP=dict(accE); m=0; mt=0; np_prof=0
    bucket={b:[0.,0.,0] for b in ['2-5','6-15','16-40','41+']}  # sumE,sumP,n
    for x in te[:NEVAL].tolist():
        if (cohort is not None and x not in cohort) or x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profile=[j for j in rd if j not in test]; rel=list(test)
        if len(rel)<2 or len(profile)<2: continue
        rel_t=[t for t in rel if not headmask[t]]; ht=1 if rel_t else 0; m+=1; mt+=ht; np_prof+=len(profile)
        seeds=[(Q[j],rd[j]-mu-bi[j]) for j in profile]; uE=enc_u(seeds); uR=ridge_u(seeds); excl=set(profile)
        eK=ndcg_at(uE,rel,excl,KP,False) or 0; pK=ndcg_at(np.zeros(D),rel,excl,KP,False) or 0
        for K in Ks:
            accE[(K,0)]+=ndcg_at(uE,rel,excl,K,False) or 0; accR[(K,0)]+=ndcg_at(uR,rel,excl,K,False) or 0
            accP[(K,0)]+=ndcg_at(np.zeros(D),rel,excl,K,False) or 0
            if ht:
                accE[(K,1)]+=ndcg_at(uE,rel,excl,K,True) or 0; accR[(K,1)]+=ndcg_at(uR,rel,excl,K,True) or 0
                accP[(K,1)]+=ndcg_at(np.zeros(D),rel,excl,K,True) or 0
        nsz=len(profile); b=('2-5' if nsz<=5 else '6-15' if nsz<=15 else '16-40' if nsz<=40 else '41+')
        bucket[b][0]+=eK; bucket[b][1]+=pK; bucket[b][2]+=1
    for d in (accE,accR,accP):
        for k in d: d[k]/=(mt if k[1]==1 else m) or 1
    return accE,accR,accP,m,mt,np_prof/max(m,1),bucket
def eval_concept(selector, cohort=None):
    acc={(q,K,tl):0. for q in REVEALS for K in Ks for tl in (0,1)}; m=0; mt=0; ans_at={q:0. for q in REVEALS}
    for x in te[:NEVAL].tolist():
        if (cohort is not None and x not in cohort) or x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profset=set(rd)-test; rel=list(test)
        if len(rel)<2 or len(profset)<2: continue
        rel_t=[t for t in rel if not headmask[t]]; ht=1 if rel_t else 0; m+=1; mt+=ht
        cans=conc_answers(profset,rd)
        order=list(np.random.default_rng(x).permutation(nc)) if selector=='random' else CONC_ORD[selector]
        toks=[]; asked=set(); ei=0
        for t in range(max(REVEALS)+1):
            if t>0:
                e=None
                while ei<len(order):
                    cand=order[ei]; ei+=1
                    if cand in asked: continue
                    e=cand; break
                if e is not None:
                    asked.add(e)
                    if e in cans: toks.append((Ac[e],cans[e]))
            if t in REVEALS:
                ans_at[t]+=len(toks); u=enc_u(toks); excl=set(profset)
                for K in Ks:
                    v=ndcg_at(u,rel,excl,K,False)
                    if v is not None: acc[(t,K,0)]+=v
                    if ht:
                        vt=ndcg_at(u,rel,excl,K,True)
                        if vt is not None: acc[(t,K,1)]+=vt
    for k in acc: acc[k]/=(mt if k[2]==1 else m) or 1
    for q in ans_at: ans_at[q]/=max(m,1)
    return acc,ans_at,m,mt
def row(tag,d): return f"{tag:<14}| @10 full={d[(10,0)]:.4f} tail={d[(10,1)]:.4f} | @{KP} full={d[(KP,0)]:.4f} tail={d[(KP,1)]:.4f}"

print(f"\n########## (A) FULL COHORT full-profile fold; ni={ni} KP={KP} ##########",flush=True)
accE,accR,accP,m,mt,avgp,bucket=eval_fullprof(None)
print(f"-- n={m} n_tail={mt} avg_profile={avgp:.1f} --",flush=True)
print(row("MOSTPOP(q0)",accP),flush=True); print(row("ridge",accR),flush=True); print(row("ENCODER",accE),flush=True)
hr_full=accE[(KP,0)]-accP[(KP,0)]
print(f">>> HEADROOM enc-MOSTPOP @{KP} full={hr_full:+.4f} tail={accE[(KP,1)]-accP[(KP,1)]:+.4f} @10 full={accE[(10,0)]-accP[(10,0)]:+.4f}",flush=True)
print(f">>> HEADROOM ridge-MOSTPOP @{KP} full={accR[(KP,0)]-accP[(KP,0)]:+.4f}",flush=True)
print("  headroom by profile size (@%d full): "%KP + " ".join(f"{b}:{(v[0]-v[1])/v[2]:+.4f}(n={v[2]})" for b,v in bucket.items() if v[2]>0),flush=True)

print(f"\n########## (B) CONCEPT-channel elicitation (FULL cohort); reveals {REVEALS} ##########",flush=True)
mono_all=True
for sel in ['random','entropy','pop']:
    acc,ans_at,mc,mtc=eval_concept(sel,None)
    print(f"-- selector={sel} (n={mc}) --",flush=True)
    for q in REVEALS:
        print(f"  q{q:<2} | @10 full={acc[(q,10,0)]:.4f} | @{KP} full={acc[(q,KP,0)]:.4f} tail={acc[(q,KP,1)]:.4f} | ans_tok={ans_at[q]:.2f}",flush=True)
    monoK=acc[(REVEALS[-1],KP,0)]>=acc[(0,KP,0)]-3e-3; mono10=acc[(REVEALS[-1],10,0)]>=acc[(0,10,0)]-3e-3
    print(f"  monotone no-harm @10={'OK' if mono10 else 'HURTS'} @{KP}={'OK' if monoK else 'HURTS'} | gain @{KP} q8={acc[(8,KP,0)]-acc[(0,KP,0)]:+.4f} q20={acc[(20,KP,0)]-acc[(0,KP,0)]:+.4f}",flush=True)
    if sel in ('random','entropy') and not (monoK and mono10): mono_all=False

print(f"\n########## (C) COLD cohort (<= {COLD_TH} ratings) ##########",flush=True)
accEc,accRc,accPc,mc,mtc,avgpc,_=eval_fullprof(COLD)
print(f"-- n={mc} n_tail={mtc} avg_profile={avgpc:.1f} --",flush=True)
print(row("MOSTPOP(q0)",accPc),flush=True); print(row("ridge",accRc),flush=True); print(row("ENCODER",accEc),flush=True)
hr_cold=accEc[(KP,0)]-accPc[(KP,0)]
print(f">>> COLD HEADROOM enc-MOSTPOP @{KP} full={hr_cold:+.4f} @10 full={accEc[(10,0)]-accPc[(10,0)]:+.4f}",flush=True)
acc,ans_at,_,_=eval_concept('entropy',COLD)
print(f"  COLD concept(entropy) ans_tok q8={ans_at[8]:.2f} q20={ans_at[20]:.2f} | gain @{KP} q8={acc[(8,KP,0)]-acc[(0,KP,0)]:+.4f} q20={acc[(20,KP,0)]-acc[(0,KP,0)]:+.4f}",flush=True)

print(f"\n========== GATE (primary K={KP}) ==========",flush=True)
print(f"  FULL-cohort encoder headroom = {hr_full:+.4f} (target >= +0.025); monotone no-harm (rand+ent) = {'OK' if mono_all else 'HURTS'}",flush=True)
print(f"  COLD-cohort encoder headroom = {hr_cold:+.4f}",flush=True)
verdict='PASS' if (hr_full>=0.025 and mono_all) else ('WEAK/FAIL' if hr_full>0 else 'FAIL')
print(f"GATE VERDICT: {verdict}",flush=True)
print(f"DONE ({time.time()-t0:.0f}s)",flush=True)
