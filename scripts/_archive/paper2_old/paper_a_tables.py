"""
PAPER A — consolidated key tables, ONE unified encoder, ONE protocol, fixed ruler (ML-1M, like>=4, users>=5 likes,
full-catalogue + Cremonesi head-33% tail, NDCG@10 + Recall@10). Reveal order HELF where a fixed order is needed.
Tables:
  A1 fold-in / recommender quality : MOSTPOP, ItemKNN(NEW), MF-ridge(closed form on Q_svd), encoder(learned) at
                                     q={0,2,4,8} + FULL profile (= pure recommender perf). NDCG+Recall, full+tail.
  A2 selection panel               : random, popularity, entropy, HELF, RMVA, Golbandi-tree, EIG(deployable), oracle.
  A3 unified instrument            : item reveals vs genre(attribute) reveals through the same encoder.
  A4 elicitation ceiling           : q0 / full-profile / EIG@8 / oracle-best-subset.
ridge-vs-encoder kept as the control that isolates the gain to the fold-in. EIG = belief-only deployable policy.
"""
import os, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; K=int(os.environ.get('K',12)); rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Qt=torch.tensor(Q)
Qn=Q/np.clip(np.linalg.norm(Q,axis=1,keepdims=True),1e-8,None)            # for ItemKNN cosine
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1])
headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) if item_g[:,g].any() else np.zeros(D) for g in range(na)]).astype(np.float32)
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
def genre_tokens(x,avail):
    out=[]
    for g in range(na):
        gi=[j for j in avail if item_g[j,g]]
        if len(gi)>=2: out.append((gcent[g],float(np.mean([resid_by_u[x][j] for j in gi]))))
    return out
POS=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4])
NEG=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); opt=torch.optim.Adam(list(enc.parameters())+[Qp],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def make_batch(users,kk):
    toks=np.zeros((len(users),kk,D+1),np.float32); msk=np.zeros((len(users),kk),np.float32)
    tgt=np.zeros((len(users),ni),np.float32); wt=np.ones((len(users),ni),np.float32); seen=np.zeros((len(users),ni),bool)
    for b,x in enumerate(users):
        its=[j for j,_ in rat_by_u[x]]; rng.shuffle(its); gtok=genre_tokens(x,its)
        pool=[('i',j) for j in its]+[('g',k) for k in range(len(gtok))]; rng.shuffle(pool); pick=pool[:kk]
        for q,(typ,v) in enumerate(pick):
            if typ=='i': toks[b,q,:D]=Q[v]; toks[b,q,D]=resid_by_u[x][v]; msk[b,q]=1; seen[b,v]=True
            else: toks[b,q,:D]=gtok[v][0]; toks[b,q,D]=gtok[v][1]; msk[b,q]=1
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.0; wt[b,j]=ipsw[j]; posw+=ipsw[j]
        nneg=ni-int(seen[b].sum())-int(tgt[b].sum())
        if nneg>0: wt[b][(tgt[b]==0)&(~seen[b])]=posw/nneg
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
print(f"train UNIFIED encoder (mixed item+genre, joint) on {len(trbig)} users...",flush=True)
for ep in range(int(os.environ.get('EP',30))):
    rng.shuffle(trbig)
    for b0 in range(0,len(trbig),256):
        us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk); sc=u@Qp.t()
        loss=(wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')).masked_fill(seen,0.).mean()
        opt.zero_grad(); loss.backward(); opt.step()
enc.eval(); Ql=Qp.detach().numpy()
_W=1./np.log2(np.arange(2,12))
def metr(score,tlike,excl,tailonly):
    sc=score.copy(); sc[list(excl)]=-1e9
    if tailonly: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None,None
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    ndcg=sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    rec=len([t for t in top if int(t) in rel])/len(rel)
    return ndcg,rec
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def enc_u(rev): return enc_u_batch([rev])[0] if rev else np.zeros(D)
def ridge(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
def sig(z): return 1/(1+np.exp(-z))
def rmva_order(prof):
    if len(prof)<2: return prof
    _,_,piv=sla.qr(Q[np.array(prof)].T,pivoting=True); return [prof[p] for p in piv]
# Golbandi global tree on top-pop candidates, impurity = within-node var of full-profile ridge latent
def ridge_full(x):
    F=np.array([Q[j] for j,_ in rat_by_u[x]]); y=np.array([resid_by_u[x][j] for j,_ in rat_by_u[x]],np.float32)
    return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@y) if len(F) else np.zeros(D)
CAND=list(np.argsort(-cnt)[:80]); GTR=list(rng.choice(trU,size=min(2500,len(trU)),replace=False))
Uc={x:ridge_full(x) for x in GTR}; like_of={x:set(likes_by_u[x]) for x in GTR}; rated_of={x:set(j for j,_ in rat_by_u[x]) for x in GTR}
def imp(us):
    if len(us)<2: return 0.
    M=np.stack([Uc[x] for x in us]); return ((M-M.mean(0))**2).sum()
def build(us,depth,used):
    if depth==0 or len(us)<10: return {'leaf':True}
    best=None
    for it in CAND:
        if it in used: continue
        L=[x for x in us if it in like_of[x]]; Dd=[x for x in us if it in rated_of[x] and it not in like_of[x]]; Uk=[x for x in us if it not in rated_of[x]]
        s=imp(L)+imp(Dd)+imp(Uk)
        if best is None or s<best[0]: best=(s,it,L,Dd,Uk)
    _,it,L,Dd,Uk=best
    return {'leaf':False,'item':it,'L':build(L,depth-1,used|{it}),'D':build(Dd,depth-1,used|{it}),'Uk':build(Uk,depth-1,used|{it})}
print("building Golbandi tree...",flush=True); TREE=build(GTR,8,set())
def gol_path(rd,maxq):
    node=TREE; path=[]
    while node and not node.get('leaf') and len(path)<maxq:
        it=node['item']; path.append(it); node=node['L'] if (it in rd and rd[it]>=4) else (node['D'] if it in rd else node['Uk'])
    return path
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL]
# ---------- A1 fold-in / recommender quality ----------
def foldin_score(method,rev_items,rd):       # rev_items = list of item idx (HELF order); returns score vector
    if method=='mostpop': return popb.copy()
    if method=='itemknn':
        s=popb.copy()
        for j in rev_items: s+= (1 if rd[j]>=4 else -1)*(Qn@Qn[j])
        return s
    if method=='ridge':
        u=ridge([Q[j] for j in rev_items],[rd[j]-mu-bi[j] for j in rev_items]); return popb+Q@u
    u=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in rev_items]); return popb+Ql@u   # encoder
def A1():
    print("\n===== TABLE A1: fold-in / recommender quality (NDCG@10 / Recall@10) =====",flush=True)
    for tailonly in [False,True]:
        print(f"--- {'TAIL' if tailonly else 'FULL'} ---",flush=True)
        for method in ['mostpop','itemknn','ridge','encoder']:
            nd={q:0. for q in [0,2,4,8,'full']}; rc={q:0. for q in [0,2,4,8,'full']}; m=0
            for x in TE[:300]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
                order=sorted(prof,key=lambda j:-helf[j])
                for q in [0,2,4,8,'full']:
                    ri=order if q=='full' else order[:q]
                    nd_,rc_=metr(foldin_score(method,ri,rd),tlike,set(prof),tailonly)
                    if nd_ is not None: nd[q]+=nd_; rc[q]+=rc_
                m+=1
            print(f"  {method:<8}: "+" ".join(f"q{q}={nd[q]/m:.3f}/{rc[q]/m:.3f}" for q in [0,2,4,8,'full']),flush=True)
# ---------- A2 selection panel ----------
def A2():
    print("\n===== TABLE A2: selection panel on the encoder (NDCG@10) =====",flush=True)
    for tailonly in [False,True]:
        print(f"--- {'TAIL' if tailonly else 'FULL'} ---",flush=True)
        for sel in ['random','pop','entropy','helf','rmva','golbandi','eig','oracle']:
            acc={q:0. for q in [0,1,2,4,8]}; m=0
            for x in TE[:250]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
                if sel=='random': o=prof[:]; rng.shuffle(o); seq=o
                elif sel=='pop': seq=sorted(prof,key=lambda j:-cnt[j])
                elif sel=='entropy': seq=sorted(prof,key=lambda j:-ent[j])
                elif sel=='helf': seq=sorted(prof,key=lambda j:-helf[j])
                elif sel=='rmva': seq=rmva_order(prof)
                elif sel=='golbandi': seq=[j for j in gol_path(rd,8) if j in rd]
                else: seq=None
                for q in [0,1,2,4,8]:
                    if sel in('eig','oracle'):
                        asked=[];rev=[]
                        while len(asked)<q:
                            cands=[j for j in prof if j not in asked]
                            if not cands: break
                            if sel=='eig':
                                un=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in asked]); p=sig(popb[cands]+Ql[cands]@un)
                                ul=enc_u_batch([[(Q[j],rd[j]-mu-bi[j]) for j in asked]+[(Q[c],POS)] for c in cands])
                                ud=enc_u_batch([[(Q[j],rd[j]-mu-bi[j]) for j in asked]+[(Q[c],NEG)] for c in cands])
                                val=p*sig(ul@Ql[cands].T).sum(1)+(1-p)*sig(ud@Ql[cands].T).sum(1); c=cands[int(val.argmax())]
                            else:
                                cu=enc_u_batch([[(Q[j],rd[j]-mu-bi[j]) for j in asked]+[(Q[c],rd[c]-mu-bi[c])] for c in cands]); best=None
                                for idx,c2 in enumerate(cands):
                                    nd_,_=metr(popb+Ql@cu[idx],tlike,set(prof)|set(asked)|{c2},tailonly)
                                    if nd_ is not None and (best is None or nd_>best[0]): best=(nd_,c2)
                                c=best[1] if best else cands[0]
                            asked.append(c); rev.append(c)
                        u=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in asked])
                    else:
                        u=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in seq[:q]])
                    nd_,_=metr(popb+Ql@u,tlike,set(prof),tailonly)
                    if nd_ is not None: acc[q]+=nd_
                m+=1
            print(f"  {sel:<9}: "+" ".join(f"q{q}={acc[q]/m:.3f}" for q in [0,1,2,4,8])+f" | +{(acc[8]-acc[0])/m:+.3f}",flush=True)
# ---------- A3 attributes ----------
def A3():
    print("\n===== TABLE A3: unified instrument items vs genres (NDCG@10) =====",flush=True)
    for tailonly in [False,True]:
        print(f"--- {'TAIL' if tailonly else 'FULL'} ---",flush=True)
        for mode in ['items','genres']:
            acc={q:0. for q in [0,1,2,4,8]}; m=0
            for x in TE[:300]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
                if mode=='genres': seq=sorted(genre_tokens(x,prof),key=lambda t:-abs(t[1]))
                else: seq=[(Q[j],rd[j]-mu-bi[j]) for j in sorted(prof,key=lambda j:-cnt[j])]
                for q in [0,1,2,4,8]:
                    nd_,_=metr(popb+Ql@enc_u(seq[:q]),tlike,set(prof),tailonly)
                    if nd_ is not None: acc[q]+=nd_
                m+=1
            print(f"  {mode:<7}: "+" ".join(f"q{q}={acc[q]/m:.3f}" for q in [0,1,2,4,8]),flush=True)
# ---------- A4 ceiling ----------
def A4():
    print("\n===== TABLE A4: elicitation ceiling (NDCG@10) =====",flush=True)
    for tailonly in [False,True]:
        out={}
        for mode in ['q0','eig8','full','oracle']:
            acc=0.;m=0
            for x in TE[:250]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
                if mode=='q0': u=np.zeros(D)
                elif mode=='full': u=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in prof])
                elif mode=='eig8':
                    asked=[]
                    while len(asked)<8 and len([j for j in prof if j not in asked]):
                        cands=[j for j in prof if j not in asked]; un=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in asked]); p=sig(popb[cands]+Ql[cands]@un)
                        ul=enc_u_batch([[(Q[j],rd[j]-mu-bi[j]) for j in asked]+[(Q[c],POS)] for c in cands]); ud=enc_u_batch([[(Q[j],rd[j]-mu-bi[j]) for j in asked]+[(Q[c],NEG)] for c in cands])
                        val=p*sig(ul@Ql[cands].T).sum(1)+(1-p)*sig(ud@Ql[cands].T).sum(1); asked.append(cands[int(val.argmax())])
                    u=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in asked])
                else:
                    asked=[];cur=metr(popb+Ql@np.zeros(D),tlike,set(prof),tailonly)[0] or 0.
                    while True:
                        cands=[j for j in prof if j not in asked]
                        if not cands: break
                        cu=enc_u_batch([[(Q[j],rd[j]-mu-bi[j]) for j in asked]+[(Q[c],rd[c]-mu-bi[c])] for c in cands]); best=None
                        for idx,c2 in enumerate(cands):
                            v=metr(popb+Ql@cu[idx],tlike,set(prof),tailonly)[0]
                            if v is not None and (best is None or v>best[0]): best=(v,c2)
                        if best is None or best[0]<=cur+1e-9: break
                        cur=best[0]; asked.append(best[1])
                    out.setdefault('oracle',None); acc+=cur; m+=1; continue
                v=metr(popb+Ql@u,tlike,set(prof),tailonly)[0]
                if v is not None: acc+=v; m+=1
            out[mode]=acc/m
        print(f"--- {'TAIL' if tailonly else 'FULL'} ---  "+" ".join(f"{k}={out[k]:.3f}" for k in ['q0','eig8','full','oracle']),flush=True)
A1(); A2(); A3(); A4()
print("\nDONE",flush=True)
