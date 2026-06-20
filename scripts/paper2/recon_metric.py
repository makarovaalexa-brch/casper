"""
PAPER B — T1 (foundational metric reframe): measure PROFILE RECONSTRUCTION (not NDCG) as elicitation proceeds.
Primary metric = recon-AUC: rank ALL unrevealed items by folded score; AUC of held-out LIKES vs non-likes (dense,
sensitive, NOT top-10 saturated). Secondary: Recall@50, NDCG@10. Profile-restricted answerable asking (isolate metric
from answerability). Item vs genre vs concept (EIG = expected coverage over rest-of-profile). FLOOR/CEILING bounds:
q0(prior), random, EIG, full-profile, oracle-subset(peek). Question: does asking (esp. concepts) clearly improve
profile reconstruction, where NDCG@10 hid it? Frozen encoder.
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
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
POS=np.mean([resid_by_u[x][j] for x in te[:400] for j,r in rat_by_u[x] if r>=4]); NEG=-1.0
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) for g in range(na)]).astype(np.float32); gitems=[set(np.where(item_g[:,g])[0].tolist()) for g in range(na)]
tagitems={}; print("streaming genome...",flush=True); t0=time.time()
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>THR: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=MINIT]
Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32); citems=[set(tagitems[t]) for t in ctags]
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
allidx=np.arange(ni); _W=1./np.log2(np.arange(2,12))
def metrics(u,tlike,excl,tail=False):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: cand=np.array([j for j in allidx if (j not in excl) and not headmask[j]])
    else: cand=np.array([j for j in allidx if j not in excl])
    s=sc[cand]; pos=np.array([1 if j in tlike else 0 for j in cand]); npos=pos.sum(); nneg=len(pos)-npos
    if npos==0 or nneg==0: return None
    order=np.argsort(s); ranks=np.empty(len(s)); ranks[order]=np.arange(1,len(s)+1)
    auc=(ranks[pos==1].sum()-npos*(npos+1)/2)/(npos*nneg)
    top=cand[np.argsort(-s)[:50]]; rec50=len(set(top.tolist())&tlike)/npos
    return auc,rec50
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:NEVAL]
def pick_eig(prev, cands_fac, restitems):                # choose candidate maximizing expected coverage over rest-of-profile items
    ul=enc_u_batch([prev+[(f,POS)] for f in cands_fac]); ud=enc_u_batch([prev+[(f,NEG)] for f in cands_fac])
    cl=sig(popb[restitems]+ul@Ql[restitems].T).sum(1); cd=sig(popb[restitems]+ud@Ql[restitems].T).sum(1)
    return int((0.5*cl+0.5*cd).argmax())
def run(mode,tail):
    A={q:0. for q in [0,1,2,4,8]};Rc={q:0. for q in [0,1,2,4,8]};m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        if mode in ('item','random','full','oracle'):
            cand=[(Q[j], rd[j]-mu-bi[j]) for j in prof]
        elif mode=='genre':
            cand=[(gcent[g], float(np.mean([rd[j]-mu-bi[j] for j in prof if j in gitems[g]]))) for g in range(na) if len([j for j in prof if j in gitems[g]])>=2]
        else:
            cand=[(Ac[k], float(np.mean([rd[j]-mu-bi[j] for j in prof if j in citems[k]]))) for k in range(len(ctags)) if len([j for j in prof if j in citems[k]])>=2]
        toks=[]; used=set()
        for q in [0,1,2,4,8]:
            while len(toks)<q and len(used)<len(cand):
                ci=[i for i in range(len(cand)) if i not in used]; rest=np.array(prof)
                if mode=='random': pi=ci[int(rng.integers(len(ci)))]
                elif mode=='oracle':
                    cu=enc_u_batch([toks+[cand[i]] for i in ci]); best=None
                    for t_,i in enumerate(ci):
                        mt=metrics(cu[t_],tlike,set(prof),tail); a=mt[0] if mt else -1
                        if best is None or a>best[0]: best=(a,i)
                    pi=best[1]
                else:
                    facs=[cand[i][0] for i in ci]; sel=pick_eig(toks,facs,rest); pi=ci[sel]
                used.add(pi); toks.append(cand[pi])
            u=enc_u([(f,v) for f,v in cand]) if mode=='full' else enc_u(toks)
            if mode=='full' and q<8:
                u=np.zeros(D) if q==0 else None
                if u is None: continue
            mt=metrics(u,tlike,set(prof),tail)
            if mt: A[q]+=mt[0];Rc[q]+=mt[1]
        m+=1
    return {q:A[q]/m for q in A},{q:Rc[q]/m for q in Rc},m
for tail in [False,True]:
    print(f"\n=== T1 PROFILE RECONSTRUCTION ({'TAIL' if tail else 'FULL'}) as elicitation proceeds (N={len(TE)}) ===",flush=True)
    print(f"{'method':<9} | recon-AUC q0->q8                 | Recall@50 q0->q8",flush=True)
    for mode in ['random','item','genre','concept','oracle']:
        A,Rc,m=run(mode,tail)
        print(f"{mode:<9} | "+" ".join(f"{A[q]:.3f}" for q in [0,1,2,4,8])+" | "+" ".join(f"{Rc[q]:.3f}" for q in [0,1,2,4,8]),flush=True)
