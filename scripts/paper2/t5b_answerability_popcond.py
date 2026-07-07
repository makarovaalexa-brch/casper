"""
T5b (Paper B review response): item-vs-concept elicitation under TWO answerability regimes.

REGIME 'orig'  (the existing, reviewer-considered-too-generous model):
    item question answerable  iff the user actually rated it (j in known-half profile)   [P=1 if seen]
    concept question answerable iff the user has >=2 rated items in that concept.

REGIME 'popcond' (NEW, popularity-conditioned seen-probability):
    A user can only ANSWER a question about something they can RECALL. Recall is popularity-gated:
        P(recall item j) = sigmoid( ALPHA * ( log(cnt_j+1) - log(cnt_med+1) ) )
    i.e. a logistic in log-popularity, anchored at the MEDIAN popularity of rated items (P=0.5 there),
    rising toward 1 for head items and falling toward 0 for obscure tail items. ALPHA sets steepness.
    We draw, once per user (fixed seed), a RECALLABLE subset of the known-half profile.
        item question answerable  iff j in the recallable subset.
        concept question answerable iff the user can recall >=2 of its member items.
    This treats items AND concepts consistently: a concept made of obscure items is now hard to answer,
    so the concept channel no longer gets 'free' answerability from tail items the user can't recall.

Everything else is IDENTICAL to the canonical headline setup (scripts/paper2/answerability_concept.py):
    ML-1M, learned concept channel (enc_concept/Ec/Ql_concept), GEOMETRIC answer, locked 150-user split
    (seed 123), popb+Ql ruler, HEAD=top-33% by like-count, FULL + TAIL NDCG@10, q in {0,2,4,8}.
Adds bootstrap CIs over users and dumps JSON. u* (true taste) is ALWAYS the full known-half fold
(ground-truth target); only which questions are ANSWERABLE changes between regimes.
"""
import os, time, json, numpy as np, torch, torch.nn as nn
ALPHA=float(os.environ.get('ALPHA','1.0')); NBOOT=int(os.environ.get('NBOOT','2000'))
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; rng=np.random.default_rng(0)
U,I,Rr=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U);I=np.array(I);Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(Rr[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); nK=len(keep); trU=keep[:int(0.8*nK)]; te=keep[int(0.9*nK):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
Ql=np.load(f'{base}/.cache/Ql_concept.npy'); Ec=np.load(f'{base}/.cache/Ec_concept.npy'); ctags=list(np.load(f'{base}/.cache/ctags_concept.npy'))
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
p_seen=(cnt/max(len(trU),1)).astype(np.float32); R=order_pop[:500]
t0=time.time(); tagitems={}; tagset=set(int(t) for t in ctags)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1])
        if m in iids and tg in tagset and float(a[2])>0.5: tagitems.setdefault(tg,[]).append(iids[m])
citems=[set(tagitems.get(int(t),[])) for t in ctags]; cfreq=np.array([len(s) for s in citems])
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems.get(int(t),[]): item2c[j].append(ki)
# concept answerability PRIOR (fraction of train users who could answer c) - reused for EIG weighting, ORIG definition
pe_cnt=np.zeros(len(ctags))
for x in trU[:2500]:
    cn={}
    for j,_ in rat_by_u[x]:
        for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
    for ki,ct in cn.items():
        if ct>=2: pe_cnt[ki]+=1
pe_ans=(pe_cnt/min(len(trU),2500)).astype(np.float32)+1e-6
# ---- POPULARITY-CONDITIONED seen/recall probability (logistic in log-pop, anchored at median rated-item pop) ----
rated_pop=cnt[cnt>0]; cnt_med=float(np.median(rated_pop))
logpop=np.log(cnt+1.0); logmed=np.log(cnt_med+1.0)
p_recall=(1.0/(1.0+np.exp(-ALPHA*(logpop-logmed)))).astype(np.float32)   # P(user can recall/answer about item j)
print(f"  {len(ctags)} concepts | ALPHA={ALPHA} cnt_med={cnt_med:.0f} | p_recall: head~{p_recall[order_pop[0]]:.2f} med~{np.median(p_recall[cnt>0]):.2f} tail~{p_recall[order_pop[-1]]:.3f} ({time.time()-t0:.0f}s)",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
def sig(z): return 1/(1+np.exp(-z))
def enc_u(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_batch(revs):
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def infogain_items(toks,cands):
    u=enc_u(toks); p=sig(popb[cands]+Ql[cands]@u); ul=enc_batch([toks+[(Q[j],POS)] for j in cands]); ud=enc_batch([toks+[(Q[j],NEG)] for j in cands])
    return p*sig(popb[R]+ul@Ql[R].T).sum(1)+(1-p)*sig(popb[R]+ud@Ql[R].T).sum(1)
def infogain_concepts(toks,cidx):
    u=enc_u(toks); ca=np.array(cidx); pc=sig(Ec[ca]@u)
    ul=enc_batch([toks+[(Ec[c],POS)] for c in cidx]); ud=enc_batch([toks+[(Ec[c],NEG)] for c in cidx])
    cov=pc*sig(ul@Ql[R].T).sum(1)+(1-pc)*sig(ud@Ql[R].T).sum(1)
    return pe_ans[ca]*cov
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-sc); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel); return nd,rc
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
TE=[x for x in te if x in SPL][:150]; ITEMC=list(order_pop[:1500])
# per-user RECALLABLE subset for popcond regime (drawn once, fixed per-user seed for reproducibility)
recall_set={}
for x in TE:
    profset,_=SPL[x]; pr=sorted(profset)
    ru=np.random.default_rng(1000+int(x)); draws=ru.random(len(pr))
    recall_set[x]=set(j for j,d in zip(pr,draws) if d < p_recall[j])
def answerable_set(x,regime):
    profset,_=SPL[x]
    return profset if regime=='orig' else (recall_set[x]&profset)
def user_answers(x,ansset):                                  # geometric concept answers, over the answerable member-set
    ac=[c for c in range(len(ctags)) if len(citems[c]&ansset)>=2]
    ustar=enc_u([(Q[j],resid[x][j]) for j in SPL[x][0]])      # true taste = full known-half fold (ground truth target)
    pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())) if pr else 0.
    return {c:(POS if pr[c]>thr else NEG) for c in ac}
MODES=['pop_item','eig_item','conc_pop','conc_eig']
def run(mode,tail,regime):
    per={q:[] for q in [0,2,4,8]}; perR={q:[] for q in [0,2,4,8]}; ans_tot=0.; m=0
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        ansset=answerable_set(x,regime); cans=user_answers(x,ansset)
        toks=[]; asked=set(); nq=0; nans=0
        for q in [0,2,4,8]:
            while nq<q:
                if mode=='pop_item':
                    cs=[j for j in ITEMC if j not in asked]
                    if not cs: break
                    j=cs[0]; asked.add(j); nq+=1
                    if j in ansset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode=='eig_item':
                    cs=[j for j in ITEMC if j not in asked][:500]
                    if not cs: break
                    csa=np.array(cs); j=cs[int((p_seen[csa]*infogain_items(toks,csa)).argmax())]; asked.add(j); nq+=1
                    if j in ansset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode in ('conc_pop','conc_eig'):
                    ci=[c for c in range(len(ctags)) if c not in asked]
                    if not ci: break
                    if mode=='conc_pop': c=max(ci,key=lambda c:cfreq[c])
                    else: c=ci[int(infogain_concepts(toks,ci).argmax())]
                    asked.add(c); nq+=1; v=cans.get(c)
                    if v is not None: toks.append((Ec[c],v)); nans+=1
            mt=metr(enc_u(toks),tlike,profset,tail)
            per[q].append(mt[0] if mt else 0.); perR[q].append(mt[1] if mt else 0.)
        m+=1; ans_tot+=nans
    return per,perR,ans_tot/max(m,1)
def boot_ci(vals,nb=NBOOT):
    v=np.array(vals); n=len(v); bs=np.random.default_rng(7)
    means=v[bs.integers(0,n,size=(nb,n))].mean(1); return float(v.mean()), float(np.percentile(means,2.5)), float(np.percentile(means,97.5))
out={'ALPHA':ALPHA,'cnt_med':cnt_med,'n_users':len(TE),'regimes':{}}
for regime in ['orig','popcond']:
    out['regimes'][regime]={}
    for tail in [False,True]:
        key='tail' if tail else 'full'; out['regimes'][regime][key]={}
        print(f"\n=== regime={regime} | {'TAIL' if tail else 'FULL'} | NDCG@10 (mean [95% CI]) / ans-obtained ===",flush=True)
        for mode in MODES:
            per,perR,na=run(mode,tail,regime)
            row={'ans':na,'ndcg':{},'rec':{}}
            cells=[]
            for q in [0,2,4,8]:
                mn,lo,hi=boot_ci(per[q]); rmn,rlo,rhi=boot_ci(perR[q])
                row['ndcg'][str(q)]={'mean':mn,'lo':lo,'hi':hi}; row['rec'][str(q)]={'mean':rmn,'lo':rlo,'hi':rhi}
                cells.append(f"q{q}={mn:.3f}[{lo:.3f},{hi:.3f}]")
            out['regimes'][regime][key][mode]=row
            print(f"  {mode:<9}: "+" ".join(cells)+f" | ans/{T}={na:.1f}",flush=True)
        # paired diff conc_pop - pop_item at q8 (bootstrap over the SAME users)
        pcp,_,_=run('conc_pop',tail,regime); ppi,_,_=run('pop_item',tail,regime)
        d=np.array(pcp[8])-np.array(ppi[8]); bs=np.random.default_rng(11); dm=d[bs.integers(0,len(d),size=(NBOOT,len(d)))].mean(1)
        dmean,dlo,dhi=float(d.mean()),float(np.percentile(dm,2.5)),float(np.percentile(dm,97.5))
        out['regimes'][regime][key]['delta_concpop_minus_popitem_q8']={'mean':dmean,'lo':dlo,'hi':dhi}
        print(f"  DELTA conc_pop-pop_item @q8: {dmean:+.3f} [{dlo:+.3f},{dhi:+.3f}] {'(sig)' if dlo>0 else '(n.s.)' }",flush=True)
os.makedirs(f'{base}/.cache/paper2',exist_ok=True)
TAG=os.environ.get('TAG','')
fn=f'{base}/.cache/paper2/t5b_answerability{TAG}.json'
with open(fn,'w') as f: json.dump(out,f,indent=2)
print(f"\nsaved {fn}",flush=True)
