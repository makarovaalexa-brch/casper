"""
PAPER B — decisive channel comparison with the VALIDATED EIG selector (expected coverage over rest-of-profile, the one
that gave item-EIG +0.127 tail), on NDCG@10 + Recall@50 (full+tail) -- NOT AUC. Profile-restricted (answerable).
Channels: item-EIG, genre-EIG, concept-EIG (proper), DIRECT-EMBEDDING continuous (concepts + interpolations, fold the
raw chosen direction). Bounds: q0, random, oracle, full-profile. Question: do concept-EIG / continuous capture the
headroom (q0~0.29 -> oracle~0.51 full) comparably to item-EIG, and does continuity (interpolated directions) add?
Frozen canonical encoder enc_unified.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; THR=0.5; MINIT=30; NEVAL=int(os.environ.get('NEVAL',200)); rng=np.random.default_rng(0)
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
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
POS=np.mean([resid[x][j] for x in te[:400] for j,r in rat_by_u[x] if r>=4]); NEG=-1.0
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) for g in range(na)]).astype(np.float32); gitems=[set(np.where(item_g[:,g])[0].tolist()) for g in range(na)]
print("streaming genome...",flush=True); t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>THR: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=MINIT]; Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32); citems=[set(tagitems[t]) for t in ctags]
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems[t]: item2c[j].append(ki)
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
def sig(z): return 1/(1+np.exp(-z))
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-sc); top10=o[:10]; top50=o[:50]
    nd=sum(_W[p] for p,t in enumerate(top10) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    rc=len(set(top50.tolist())&rel)/len(rel); return nd,rc
def eig_cov(prev, fac, vals, rest):                       # VALIDATED EIG: belief-weighted expected coverage over rest-of-profile
    p=sig((fac@_ut(prev)));                                # belief along each candidate direction
    ul=enc_u_batch([prev+[(f,POS)] for f in fac]); ud=enc_u_batch([prev+[(f,NEG)] for f in fac])
    cl=sig(popb[rest]+ul@Ql[rest].T).sum(1); cd=sig(popb[rest]+ud@Ql[rest].T).sum(1)
    return p*cl+(1-p)*cd
def _ut(prev): return enc_u(prev)
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:NEVAL]
def uconcepts(x,prof):
    cn={}
    for j in prof:
        for ki in item2c[j]: cn[ki]=cn.get(ki,[])+[j]
    return [(Ac[ki], float(np.mean([resid[x][j] for j in js]))) for ki,js in cn.items() if len(js)>=2]
def run(mode,tail):
    M={q:0. for q in [0,1,2,4,8]};Rc={q:0. for q in [0,1,2,4,8]};m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        if mode in('item','random','oracle','full'): cand=[(Q[j],rd[j]-mu-bi[j]) for j in prof]
        elif mode=='genre': cand=[(gcent[g],float(np.mean([rd[j]-mu-bi[j] for j in prof if j in gitems[g]]))) for g in range(na) if len([j for j in prof if j in gitems[g]])>=2]
        elif mode=='concept': cand=uconcepts(x,prof)
        else:  # direct-embedding continuous: concepts + interpolations (pairwise means)
            cc=uconcepts(x,prof); cand=list(cc)
            if len(cc)>=2:
                for _ in range(min(40,len(cc)*2)):
                    i,j=rng.choice(len(cc),2,replace=False); f=(cc[i][0]+cc[j][0])/2; v=(cc[i][1]+cc[j][1])/2; cand.append((f,v))
        if not cand: cand=[(Q[j],rd[j]-mu-bi[j]) for j in prof]
        rest=np.array(prof); toks=[]; used=set()
        for q in [0,1,2,4,8]:
            while len(toks)<q and len(used)<len(cand):
                ci=[i for i in range(len(cand)) if i not in used]
                if mode=='random': pi=ci[int(rng.integers(len(ci)))]
                elif mode=='oracle':
                    cu=enc_u_batch([toks+[cand[i]] for i in ci]); best=None
                    for t_,i in enumerate(ci):
                        mt=metr(cu[t_],tlike,set(prof),tail); a=mt[0] if mt else -1
                        if best is None or a>best[0]: best=(a,i)
                    pi=best[1]
                else:
                    fac=np.array([cand[i][0] for i in ci]); val=eig_cov(toks,fac,None,rest); pi=ci[int(val.argmax())]
                used.add(pi); toks.append(cand[pi])
            u=enc_u([(f,v) for f,v in cand]) if mode=='full' else enc_u(toks)
            if mode=='full' and 0<q<8: u=(np.zeros(D) if q==0 else None);
            if u is None: continue
            mt=metr(u,tlike,set(prof),tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},m
for tail in [False,True]:
    print(f"\n=== {'TAIL' if tail else 'FULL'}: NDCG@10 / Recall@50, profile-restricted, VALIDATED EIG (N={len(TE)}) ===",flush=True)
    for mode in ['random','item','genre','concept','direct','oracle']:
        M,Rc,m=run(mode,tail); print(f"  {mode:<8}: NDCG "+" ".join(f"{M[q]:.3f}" for q in [0,1,2,4,8])+" | Rec@50 "+" ".join(f"{Rc[q]:.3f}" for q in [0,1,2,4,8]),flush=True)
