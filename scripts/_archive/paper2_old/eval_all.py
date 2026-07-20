"""
CANONICAL REGRESSION HARNESS. One command = all key tables + comparisons + sanity + a COLLAPSE CHECK.
Flags (env): POLARITY (fix A: disliked items as explicit NEGATIVE reconstruction targets), COFACTOR (fix B, later).
Tables: A1 fold-in (mostpop/itemknn/ridge/encoder, NDCG+Recall, full+tail, q-curve); A2 selection panel
(random/pop/entropy/helf/rmva/golbandi/eig/oracle); A3 items vs genres; A4 ceiling; SANITY (genre purity w/ & w/o popb,
polarity gap, item coherence). COLLAPSE CHECK gates: encoder>ridge, eig>random, monotone, polarity gap.
"""
import os, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
POLARITY=int(os.environ.get('POLARITY',0)); COFACTOR=int(os.environ.get('COFACTOR',0)); EXPO=int(os.environ.get('EXPO',0))
CONTRA=int(os.environ.get('CONTRA',0)); CL=float(os.environ.get('CL',0.5)); MARGIN=float(os.environ.get('MARGIN',0.5))
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; K=12; rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids); inv={k:x for x,k in iids.items()}
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
dis_by_u={x:[j for j,r in v if r<3] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Qt=torch.tensor(Q); Qn=Q/np.clip(np.linalg.norm(Q,axis=1,keepdims=True),1e-8,None)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
expo_prop=((cnt/max(cnt.max(),1))**0.5).astype(np.float32)                    # P(exposed) proxy for MNAR negative weighting
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1])
headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool); title={}
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            title[iids[m]]=pp[1]
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) if item_g[:,g].any() else np.zeros(D) for g in range(na)]).astype(np.float32)
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
def gtok(x,avail):
    out=[]
    for g in range(na):
        gi=[j for j in avail if item_g[j,g]]
        if len(gi)>=2: out.append((gcent[g],float(np.mean([resid_by_u[x][j] for j in gi]))))
    return out
POS=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); opt=torch.optim.Adam(list(enc.parameters())+[Qp],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def make_batch(us,kk):
    toks=np.zeros((len(us),kk,D+1),np.float32); msk=np.zeros((len(us),kk),np.float32); tgt=np.zeros((len(us),ni),np.float32); wt=np.ones((len(us),ni),np.float32); seen=np.zeros((len(us),ni),bool)
    for b,x in enumerate(us):
        its=[j for j,_ in rat_by_u[x]]; rng.shuffle(its); gt=gtok(x,its); pool=[('i',j) for j in its]+[('g',k) for k in range(len(gt))]; rng.shuffle(pool)
        for q,(typ,v) in enumerate(pool[:kk]):
            if typ=='i': toks[b,q,:D]=Q[v]; toks[b,q,D]=resid_by_u[x][v]; msk[b,q]=1; seen[b,v]=True
            else: toks[b,q,:D]=gt[v][0]; toks[b,q,D]=gt[v][1]; msk[b,q]=1
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.;wt[b,j]=ipsw[j];posw+=ipsw[j]
        negmask=(tgt[b]==0)&(~seen[b]); nneg=int(negmask.sum())
        if EXPO and nneg>0:                                                   # MNAR: negatives weighted by exposure propensity (popular-unrated = strong neg)
            pw=expo_prop[negmask]; wt[b][negmask]=posw*(pw/pw.sum())
        elif nneg>0: wt[b][negmask]=posw/nneg                                 # background balanced (same as baseline)
        if POLARITY:                                                          # GENTLE: each known dislike == one positive
            npos=max(len(likes_by_u[x]),1)
            for j in dis_by_u[x]:
                if not seen[b,j]: wt[b,j]=posw/npos
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
LOAD=os.environ.get('LOAD','')
if LOAD:                                                                       # test a SAVED model (e.g. enc_concept = the UNIFIED A+B model) on ALL Paper A gates, no training
    enc.load_state_dict(torch.load(f'{base}/.cache/enc_{LOAD}.pt')); enc.eval(); Ql=np.load(f'{base}/.cache/Ql_{LOAD}.npy')
    print(f"LOADED enc_{LOAD}.pt + Ql_{LOAD}.npy (NO training) -- running Paper A gates on the {LOAD} model",flush=True)
else:
    print(f"train encoder (POLARITY={POLARITY} COFACTOR={COFACTOR})...",flush=True)
    for ep in range(30):
        rng.shuffle(trbig)
        for b0 in range(0,len(trbig),256):
            us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); t,m,tg,w,se=make_batch(us,kk); u=enc(t,m); sc=u@Qp.t()
            loss=(w*nn.functional.binary_cross_entropy_with_logits(sc,tg,reduction='none')).masked_fill(se,0.).mean()
            if CONTRA:                                                            # value-channel polarity: like must score the item higher than dislike
                jt=torch.randint(0,ni,(256,)); o=torch.ones(256,1)
                tp=torch.zeros(256,1,D+1); tp[:,0,:D]=Qt[jt]; tp[:,0,D]=POS
                tm=torch.zeros(256,1,D+1); tm[:,0,:D]=Qt[jt]; tm[:,0,D]=NEG
                up=enc(tp,o); um=enc(tm,o); diff=(Qp[jt]*up).sum(1)-(Qp[jt]*um).sum(1)
                loss=loss+CL*torch.nn.functional.softplus(MARGIN-diff).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); Ql=Qp.detach().numpy()
_W=1./np.log2(np.arange(2,12))
def metr(score,tlike,excl,tailonly):
    sc=score.copy(); sc[list(excl)]=-1e9
    if tailonly: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None,None
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    return (sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12), len([t for t in top if int(t) in rel])/len(rel))
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
TREE=build(GTR,8,set())
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
RES={}
def A1():
    print("\n== A1 fold-in (NDCG/Recall) ==",flush=True)
    for tail in [False,True]:
        print(f" {'TAIL' if tail else 'FULL'}",flush=True)
        for meth in ['mostpop','itemknn','ridge','encoder']:
            nd={q:0. for q in [0,2,4,8,'full']};rc={q:0. for q in [0,2,4,8,'full']};m=0
            for x in TE[:300]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tl=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tl or (tail and not any(not headmask[t] for t in tl)): continue
                order=sorted(prof,key=lambda j:-helf[j])
                for q in [0,2,4,8,'full']:
                    ri=order if q=='full' else order[:q]
                    if meth=='mostpop': s=popb.copy()
                    elif meth=='itemknn':
                        s=popb.copy()
                        for j in ri: s+=(1 if rd[j]>=4 else -1)*(Qn@Qn[j])
                    elif meth=='ridge': s=popb+Q@ridge([Q[j] for j in ri],[rd[j]-mu-bi[j] for j in ri])
                    else: s=popb+Ql@enc_u([(Q[j],rd[j]-mu-bi[j]) for j in ri])
                    a,b=metr(s,tl,set(prof),tail)
                    if a is not None: nd[q]+=a;rc[q]+=b
                m+=1
            RES[(meth,'tail' if tail else 'full')]={q:nd[q]/m for q in nd}
            print(f"  {meth:<8}: "+" ".join(f"q{q}={nd[q]/m:.3f}/{rc[q]/m:.3f}" for q in [0,2,4,8,'full']),flush=True)
def A2():
    print("\n== A2 selection panel (NDCG@10) ==",flush=True)
    for tail in [False,True]:
        print(f" {'TAIL' if tail else 'FULL'}",flush=True)
        for sel in ['random','pop','entropy','helf','rmva','golbandi','eig','oracle']:
            acc={q:0. for q in [0,1,2,4,8]};m=0
            for x in TE[:200]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tl=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tl or (tail and not any(not headmask[t] for t in tl)): continue
                if sel=='random': o=prof[:];rng.shuffle(o);seq=o
                elif sel=='pop': seq=sorted(prof,key=lambda j:-cnt[j])
                elif sel=='entropy': seq=sorted(prof,key=lambda j:-ent[j])
                elif sel=='helf': seq=sorted(prof,key=lambda j:-helf[j])
                elif sel=='rmva': seq=rmva_order(prof)
                elif sel=='golbandi': seq=[j for j in gol_path(rd,8) if j in rd]
                else: seq=None
                for q in [0,1,2,4,8]:
                    if sel in ('eig','oracle'):
                        asked=[]
                        while len(asked)<q:
                            cset=[j for j in prof if j not in asked]
                            if not cset: break
                            pre=[(Q[j],rd[j]-mu-bi[j]) for j in asked]
                            if sel=='eig':
                                un=enc_u(pre); p=sig(popb[cset]+Ql[cset]@un)
                                ul=enc_u_batch([pre+[(Q[c],POS)] for c in cset]); ud=enc_u_batch([pre+[(Q[c],NEG)] for c in cset])
                                val=p*sig(ul@Ql[cset].T).sum(1)+(1-p)*sig(ud@Ql[cset].T).sum(1); c=cset[int(val.argmax())]
                            else:
                                cu=enc_u_batch([pre+[(Q[c],rd[c]-mu-bi[c])] for c in cset]);best=None
                                for idx,c2 in enumerate(cset):
                                    a,_=metr(popb+Ql@cu[idx],tl,set(prof)|set(asked)|{c2},tail)
                                    if a is not None and (best is None or a>best[0]): best=(a,c2)
                                c=best[1] if best else cset[0]
                            asked.append(c)
                        u=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in asked])
                    else: u=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in seq[:q]])
                    a,_=metr(popb+Ql@u,tl,set(prof),tail)
                    if a is not None: acc[q]+=a
                m+=1
            RES[(sel,'tail' if tail else 'full')]={q:acc[q]/m for q in acc}
            print(f"  {sel:<9}: "+" ".join(f"q{q}={acc[q]/m:.3f}" for q in [0,1,2,4,8])+f" | +{(acc[8]-acc[0])/m:+.3f}",flush=True)
def A3():
    print("\n== A3 items vs genres (NDCG@10) ==",flush=True)
    for tail in [False,True]:
        print(f" {'TAIL' if tail else 'FULL'}",flush=True)
        for mode in ['items','genres']:
            acc={q:0. for q in [0,1,2,4,8]};m=0
            for x in TE[:300]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tl=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tl or (tail and not any(not headmask[t] for t in tl)): continue
                seq=sorted(gtok(x,prof),key=lambda t:-abs(t[1])) if mode=='genres' else [(Q[j],rd[j]-mu-bi[j]) for j in sorted(prof,key=lambda j:-cnt[j])]
                for q in [0,1,2,4,8]:
                    a,_=metr(popb+Ql@enc_u(seq[:q]),tl,set(prof),tail)
                    if a is not None: acc[q]+=a
                m+=1
            print(f"  {mode:<7}: "+" ".join(f"q{q}={acc[q]/m:.3f}" for q in [0,1,2,4,8]),flush=True)
def SANITY():
    print("\n== SANITY ==",flush=True)
    def top(u,k=10,excl=(),pop=True):
        s=(popb+Ql@u) if pop else (Ql@u).copy(); s[list(excl)]=-1e9; t=np.argpartition(-s,k)[:k]; return t[np.argsort(-s[t])]
    for pop in [True,False]:
        pur=[]
        for g in range(na): pur.append(np.mean([item_g[j,g] for j in top(enc_u([(gcent[g],POS)]),10,pop=pop)]))
        print(f"  genre purity mean ({'with' if pop else 'no'} popb) = {np.mean(pur)*100:.0f}% (mostpop 16%)",flush=True)
    lp=[np.mean([item_g[j,g] for j in top(enc_u([(gcent[g],POS)]),10,pop=False)]) for g in range(na)]
    dp=[np.mean([item_g[j,g] for j in top(enc_u([(gcent[g],NEG)]),10,pop=False)]) for g in range(na)]
    print(f"  POLARITY gap (no popb): like {np.mean(lp)*100:.0f}% vs dislike {np.mean(dp)*100:.0f}% = {(np.mean(lp)-np.mean(dp))*100:+.0f}pp",flush=True)
    RES['polgap']=(np.mean(lp)-np.mean(dp))*100
    def find(s):
        for j,t in title.items():
            if s.lower() in t.lower(): return j
        return None
    for nm in ['Star Wars: Episode IV','Toy Story','Silence of the Lambs','Godfather, The (1972)']:
        j=find(nm)
        if j is not None: print(f"  like '{title[j]}' -> "+" | ".join(title[int(r)] for r in top(enc_u([(Q[j],POS)]),5,excl={j},pop=False)),flush=True)
A1();A2();A3();SANITY()
print("\n== COLLAPSE CHECK ==",flush=True)
g1f=RES[('encoder','full')][8]>RES[('ridge','full')][8]; g1t=RES[('encoder','tail')][8]>RES[('ridge','tail')][8]
g2f=RES[('eig','full')][8]>RES[('random','full')][8]; g2t=RES[('eig','tail')][8]>RES[('random','tail')][8]
mono=all(RES[('eig','full')][q]<=RES[('eig','full')][nq]+2e-3 for q,nq in [(0,1),(1,2),(2,4),(4,8)]) and \
     all(RES[('eig','tail')][q]<=RES[('eig','tail')][nq]+2e-3 for q,nq in [(0,1),(1,2),(2,4),(4,8)])
print(f"  G1 encoder>ridge @q8: FULL {'OK' if g1f else 'FAIL'} ({RES[('encoder','full')][8]:.3f} vs {RES[('ridge','full')][8]:.3f}) | TAIL {'OK' if g1t else 'FAIL'} ({RES[('encoder','tail')][8]:.3f} vs {RES[('ridge','tail')][8]:.3f})",flush=True)
print(f"  G2 eig>random @q8:    FULL {'OK' if g2f else 'FAIL'} ({RES[('eig','full')][8]:.3f} vs {RES[('random','full')][8]:.3f}) | TAIL {'OK' if g2t else 'FAIL'} ({RES[('eig','tail')][8]:.3f} vs {RES[('random','tail')][8]:.3f})",flush=True)
print(f"  G3 EIG-POLICY monotone (full+tail): {'OK' if mono else 'FAIL'}  [static-order fold-in may dip at q1 by design]",flush=True)
print(f"  G4 polarity gap (no popb): {RES['polgap']:+.0f}pp  (baseline was +9pp; target >=+20pp)",flush=True)
print(f"  VERDICT: {'ELICITATION INTACT' if (g1f and g1t and g2f and g2t and mono) else '*** COLLAPSE - RETHINK ***'}",flush=True)
print("DONE",flush=True)
