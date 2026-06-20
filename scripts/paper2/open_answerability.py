"""
PAPER B — E1 (answerability centerpiece): OPEN-setting interview, item vs genre vs concept asking, with "don't know"
for unanswerable. Same deployable selector across types (argmax expected belief-shift over a shortlist); budget = q
QUESTIONS ASKED (don't-know turns consume budget). Report NDCG@10 (full+tail) per #asked + avg #answered.
Gate A1: concept-asking > item-asking at fixed #asked (items ~unanswerable -> wasted turns). Frozen encoder.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; THR=0.5; MINIT=30; SHORT=int(os.environ.get('SHORT',60)); NEVAL=int(os.environ.get('NEVAL',120)); rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids); keepids=set(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
POS=np.mean([resid_by_u[x][j] for x in te[:500] for j,r in rat_by_u[x] if r>=4 and x in resid_by_u]); NEG=-1.0
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) for g in range(na)]).astype(np.float32); gitems=[set(np.where(item_g[:,g])[0].tolist()) for g in range(na)]
tagitems={}
print("streaming genome...",flush=True); t0=time.time()
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>THR: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=MINIT]
Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32); citems=[set(tagitems[t]) for t in ctags]; ccov=np.array([len(s) for s in citems])
print(f"  {len(ctags)} concepts ({time.time()-t0:.0f}s)",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def enc_u(rev): return enc_u_batch([rev])[0] if rev else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(u,tlike,excl,tail):                              # NOW Recall@50 (the un-saturated metric)
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    top=np.argpartition(-sc,50)[:50]; return len(set(top.tolist())&rel)/len(rel)
# type pools: factor matrix + answerability/answer closures
itempool=list(order_pop[:400])
def cov_select(prev_toks, ut, cand_factors, ref):        # expected COVERAGE over current top-recs (value-oriented EIG)
    ul=enc_u_batch([prev_toks+[(f,POS)] for f in cand_factors]); ud=enc_u_batch([prev_toks+[(f,NEG)] for f in cand_factors])
    cl=(1/(1+np.exp(-(popb[ref]+ul@Ql[ref].T)))).sum(1); cd=(1/(1+np.exp(-(popb[ref]+ud@Ql[ref].T)))).sum(1)
    return 0.5*cl+0.5*cd
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:NEVAL]
def run(typ,tail):
    acc={q:0. for q in [1,2,4,8]}; ans={q:0. for q in [1,2,4,8]}; m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); profset=set(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        asked=set(); toks=[]; nans=0
        for turn in range(1,9):
            ut=enc_u(toks); ref=np.argpartition(-(popb+Ql@ut),150)[:150]   # current top-recs as coverage reference
            if typ=='item':
                pool=[j for j in itempool if j not in asked]; fac=Q[pool]
            elif typ=='genre':
                pool=[g for g in range(na) if g not in asked]; fac=gcent[pool]
            else:
                cl=[k for k in np.argsort(-ccov) if k not in asked][:SHORT]; pool=cl; fac=Ac[pool]
            sh=cov_select(toks,ut,fac,ref); pick=pool[int(sh.argmax())]; asked.add(pick)
            # answer or don't-know
            if typ=='item':
                if pick in profset: toks.append((Q[pick],rd[pick]-mu-bi[pick])); nans+=1
            elif typ=='genre':
                inb=[j for j in prof if j in gitems[pick]]
                if len(inb)>=2: toks.append((gcent[pick],float(np.mean([rd[j]-mu-bi[j] for j in inb])))); nans+=1
            else:
                inb=[j for j in prof if j in citems[pick]]
                if len(inb)>=2: toks.append((Ac[pick],float(np.mean([rd[j]-mu-bi[j] for j in inb])))); nans+=1
            if turn in (1,2,4,8):
                v=ndcg(enc_u(toks),tlike,profset,tail)
                if v is not None: acc[turn]+=v; ans[turn]+=nans
        m+=1
    return {q:acc[q]/m for q in acc},{q:ans[q]/m for q in ans},m
for tail in [False,True]:
    print(f"\n=== {'TAIL' if tail else 'FULL'} Recall@50 — OPEN setting, per #ASKED (ans=avg answered) ===",flush=True)
    for typ in ['item','genre','concept']:
        r,a,m=run(typ,tail); print(f"  {typ:<8}: "+" ".join(f"q{q}={r[q]:.3f}(a{a[q]:.1f})" for q in [1,2,4,8])+f" | n={m}",flush=True)
