"""
ML-25M PHASE-2 STEP 1 -- B battery (answerability + selection), frozen V1 enc_v1_ml25m instrument.
Strict WASTED-TURN regime: asking consumes a turn; only ANSWERABLE asks fold a token.
  item  answerable  <=> user rated it (in profile)          -> token (Q[j],  r-mu-bi[j])
  concept answerable <=> user has >=2 profile items tagged c -> token (Ac[c], mean rating-residual)
Held-out disjoint targets (same protocol as ml25m_health_enc): profile = rated - held-half-likes.
Selectors (static global order): entropy(divisiveness) / popularity / random.  Item pool = top-600 popular.
Ruler: NDCG@50 primary + @10 alongside, full + Cremonesi head-33% tail. q-curve {0,2,4,6,8}.
Anchors (ML-1M Paper B): answerable concepts beat items (tail +~50%); entropy >> random.
Env: NEVAL(500), COHORT(te|va).
"""
import os, time, numpy as np, torch, torch.nn as nn
t0=time.time(); out='C:/dev/phd/casper/data/movielens/.cache/ml25m'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0
NEVAL=int(os.environ.get('NEVAL',500)); COHORT=os.environ.get('COHORT','te')
REVEALS=[0,2,4,6,8]; Ks=[10,50]
M=np.load(f'{out}/meta.npz')
uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; mu=float(M['mu']); ni=int(M['ni'])
va=M['va']; te=M['te']; EVU=(te if COHORT=='te' else va)[:NEVAL]
Q=np.load(f'{out}/Q_svd.npy').astype(np.float32); bi=np.load(f'{out}/bi_svd.npy').astype(np.float32)
Ac=np.load(f'{out}/Ac_concept.npy').astype(np.float32); nc=Ac.shape[0]
popb=np.log(cnt+1.0).astype(np.float32)
mem=np.load(f'{out}/membership.npz'); cf_flat=mem['citems_flat']; cf_off=mem['citems_off']
cdiv=mem['cdiv']; ccount=mem['ccount']; idiv=mem['idiv']; PITEMS=mem['PITEMS'].astype(int)
citems=[set(cf_flat[cf_off[k]:cf_off[k+1]].tolist()) for k in range(nc)]
# item -> concepts (only for pool bookkeeping not needed); build user concept membership on the fly
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/max(cnt.sum(),1)
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
# selector orders
CONC_ORD={'entropy':list(np.argsort(-cdiv)),'pop':list(np.argsort(-ccount))}
ITEM_ORD={'entropy':[int(PITEMS[k]) for k in np.argsort(-idiv[PITEMS])],'pop':list(PITEMS)}
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
_ENCF=os.environ.get('ENCF','enc_v1_ml25m.pt')
enc=Enc(); enc.load_state_dict(torch.load(f'{out}/{_ENCF}')); enc.eval(); print(f"[ENC] {_ENCF}",flush=True)
def enc_u(toks):  # toks list of (D-vec, residual)
    if not toks: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(toks),D+1),np.float32); mk=np.ones((1,len(toks)),np.float32)
    for q,(f,res) in enumerate(toks): tk[0,q,:D]=f; tk[0,q,D]=res
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def ndcg_at(u,rel,excl,K,tail):
    sc=(popb+Q@u).astype(np.float64).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in rel if not headmask[t]]
    if not rel: return None
    W=1./np.log2(np.arange(2,K+2)); top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]
    rs=set(rel); dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
# cohort ratings
selm=np.isin(uu,EVU); euu=uu[selm]; eii=ii[selm]; eR=rr[selm]
rat_by_u={}
for k in range(len(euu)): rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
_rs=np.random.default_rng(123); SPL={}
for x in EVU.tolist():
    lk=[j for j,r in rat_by_u.get(x,[]) if r>=LIKE]
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
print(f"cohort dicts built: {len(SPL)} users w/ held targets ({time.time()-t0:.0f}s)",flush=True)
def user_conc_answers(profset, rd):  # concept -> mean rating-residual over profile items tagged, if >=2
    cn={}
    for j in profset:
        pass
    # count per concept
    ans={}; cnt_c={}; sum_c={}
    for c in range(nc):
        its=citems[c]&profset
        if len(its)>=2:
            r=np.mean([rd[j]-mu-bi[j] for j in its]); ans[c]=float(r)
    return ans
# faster: invert once per user via item->concepts membership
item2c=[[] for _ in range(ni)]
for c in range(nc):
    for j in cf_flat[cf_off[c]:cf_off[c+1]]: item2c[int(j)].append(c)
def conc_answers(profset, rd):
    acc={};
    for j in profset:
        for c in item2c[j]: acc.setdefault(c,[]).append(rd[j]-mu-bi[j])
    return {c:float(np.mean(v)) for c,v in acc.items() if len(v)>=2}
def run(channel, selector):
    acc={(q,K,tl):0. for q in REVEALS for K in Ks for tl in (0,1)}
    m=0; mt=0; ans_at={q:0. for q in REVEALS}  # avg #answered tokens by turn q
    for x in EVU.tolist():
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]; profset=set(rd)-test; rel=list(test)
        if len(rel)<2 or len(profset)<2: continue
        rel_t=[t for t in rel if not headmask[t]]; ht=1 if rel_t else 0; m+=1; mt+=ht
        if channel=='concept':
            cans=conc_answers(profset,rd)
            order=list(np.random.default_rng(x).permutation(nc)) if selector=='random' else CONC_ORD[selector]
        else:
            order=[int(j) for j in np.random.default_rng(x).permutation(PITEMS)] if selector=='random' else ITEM_ORD[selector]
        toks=[]; asked=set(); ei=0
        for t in range(max(REVEALS)+1):
            if t>0:
                # consume one turn: pick next entity in order (no re-ask), fold IF answerable
                e=None
                while ei<len(order):
                    cand=order[ei]; ei+=1
                    if cand in asked: continue
                    e=cand; break
                if e is not None:
                    asked.add(e)
                    if channel=='concept':
                        if e in cans: toks.append((Ac[e],cans[e]))
                    else:
                        if e in profset: toks.append((Q[e],rd[e]-mu-bi[e]))
            if t in REVEALS:
                ans_at[t]+=len(toks)
                u=enc_u(toks); excl=profset|asked
                for K in Ks:
                    v=ndcg_at(u,rel,excl,K,False)
                    if v is not None: acc[(t,K,0)]+=v
                    if ht:
                        vt=ndcg_at(u,rel,excl,K,True)
                        if vt is not None: acc[(t,K,1)]+=vt
    for k in acc: acc[k]/=(mt if k[2]==1 else m) or 1
    for q in ans_at: ans_at[q]/=max(m,1)
    return acc,ans_at,m,mt
print(f"\n=== ML-25M PHASE-2 STEP 1 B-BATTERY (cohort={COHORT}, n<={NEVAL}) ===",flush=True)
print("[frozen enc_v1_ml25m; wasted-turn; graded rating-residual answers; tail=Cremonesi head-33%]",flush=True)
def show(tag,acc,ans_at,m,mt):
    print(f"\n-- {tag} (n={m},n_tail={mt}) --",flush=True)
    for q in REVEALS:
        print(f"  q{q} | @10 full={acc[(q,10,0)]:.4f} tail={acc[(q,10,1)]:.4f} | @50 full={acc[(q,50,0)]:.4f} tail={acc[(q,50,1)]:.4f} | ans_tok={ans_at[q]:.2f}",flush=True)
RES={}
for ch in ['concept','item']:
    for sel in ['entropy','pop','random']:
        acc,ans_at,m,mt=run(ch,sel); RES[(ch,sel)]=(acc,ans_at,m,mt)
        show(f"{ch}-asking / {sel}",acc,ans_at,m,mt)
# headline compares @q8
def g(ch,sel,q,K,tl): return RES[(ch,sel)][0][(q,K,tl)]
print("\n=== HEADLINE @q8 (entropy selector) ===",flush=True)
for K in Ks:
    cf_=g('concept','entropy',8,K,0); if_=g('item','entropy',8,K,0); ct=g('concept','entropy',8,K,1); it=g('item','entropy',8,K,1)
    print(f"  @{K}: CONCEPT full={cf_:.4f} tail={ct:.4f} | ITEM full={if_:.4f} tail={it:.4f} | concept-item full={cf_-if_:+.4f} tail={ct-it:+.4f} ({(ct/it-1)*100:+.0f}% tail)" if it>1e-6 else f"  @{K}: CONCEPT full={cf_:.4f} tail={ct:.4f} | ITEM full={if_:.4f} tail={it:.4f} | diff full={cf_-if_:+.4f} tail={ct-it:+.4f}",flush=True)
print("\n=== SELECTION-MATTERS @q8 (concept channel) ===",flush=True)
for K in Ks:
    print(f"  @{K} full: entropy={g('concept','entropy',8,K,0):.4f} pop={g('concept','pop',8,K,0):.4f} random={g('concept','random',8,K,0):.4f}",flush=True)
    print(f"  @{K} tail: entropy={g('concept','entropy',8,K,1):.4f} pop={g('concept','pop',8,K,1):.4f} random={g('concept','random',8,K,1):.4f}",flush=True)
# concept channel saturation: marginal @50 full per turn (entropy)
acc=RES[('concept','entropy')][0]; ans=RES[('concept','entropy')][1]
print("\n=== CONCEPT-CHANNEL SATURATION (entropy, @50 full) ===",flush=True)
prev=acc[(0,50,0)]
for q in REVEALS[1:]:
    print(f"  q{q}: @50full={acc[(q,50,0)]:.4f} (marginal {acc[(q,50,0)]-prev:+.4f}) ans_tok={ans[q]:.2f}",flush=True); prev=acc[(q,50,0)]
print(f"\nDONE ({time.time()-t0:.0f}s)",flush=True)
