"""
TEST 3 (principled): TWO-HEAD encoder = EXPOSURE x PREFERENCE (ExpoMF/Liang2016 spirit; your instrument-v6 dual head).
 - exposure head e_i = expo_bias_i + q_i.u_e : target = rated(1)/unrated(0) over all unrevealed  (the watch/not signal)
 - preference head p_i = q_i.u_p : target = like(1)/dislike(0) among RATED unrevealed ONLY  (clean polarity, no popularity)
Recommendation/score downstream = popb + Ql.u_p  (taste head + floor). The exposure head ABSORBS popularity/selection so
the preference head learns clean like/dislike. Same eval tables + COLLAPSE-CHECK gates as eval_all.py.
"""
import os, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; K=12; rng=np.random.default_rng(0); torch.manual_seed(0)
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
rd_by_u={x:{j:r for j,r in rat_by_u[x]} for x in (trU+te)}; resid_by_u={x:{j:(rd_by_u[x][j]-mu-bi[j]) for j in rd_by_u[x]} for x in (trU+te)}
def gtok(x,avail):
    out=[]
    for g in range(na):
        gi=[j for j in avail if item_g[j,g]]
        if len(gi)>=2: out.append((gcent[g],float(np.mean([resid_by_u[x][j] for j in gi]))))
    return out
POS=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1)
        s.val_p=nn.Linear(128,D); s.val_e=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); pooled=(al.unsqueeze(-1)*h).sum(1)
        return s.val_p(pooled), s.val_e(pooled)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); ebias=torch.nn.Parameter(torch.tensor(popb.copy()))
opt=torch.optim.Adam(list(enc.parameters())+[Qp,ebias],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def make_batch(us,kk):
    toks=np.zeros((len(us),kk,D+1),np.float32); msk=np.zeros((len(us),kk),np.float32); seen=np.zeros((len(us),ni),bool)
    ptg=np.zeros((len(us),ni),np.float32); pw=np.zeros((len(us),ni),np.float32)
    etg=np.zeros((len(us),ni),np.float32); ew=np.zeros((len(us),ni),np.float32)
    for b,x in enumerate(us):
        its=[j for j,_ in rat_by_u[x]]; rng.shuffle(its); gt=gtok(x,its); pool=[('i',j) for j in its]+[('g',k) for k in range(len(gt))]; rng.shuffle(pool)
        for q,(typ,v) in enumerate(pool[:kk]):
            if typ=='i': toks[b,q,:D]=Q[v]; toks[b,q,D]=resid_by_u[x][v]; msk[b,q]=1; seen[b,v]=True
            else: toks[b,q,:D]=gt[v][0]; toks[b,q,D]=gt[v][1]; msk[b,q]=1
        rated=[j for j in its if not seen[b,j]]
        lk=[j for j in rated if rd_by_u[x][j]>=4]; dk=[j for j in rated if rd_by_u[x][j]<4]
        for j in lk: ptg[b,j]=1.; pw[b,j]=0.5/max(len(lk),1)               # PREFERENCE: balanced like vs dislike, RATED only
        for j in dk: ptg[b,j]=0.; pw[b,j]=0.5/max(len(dk),1)
        for j in rated: etg[b,j]=1.; ew[b,j]=0.5/max(len(rated),1)         # EXPOSURE: rated=1 (balanced vs unrated)
        bg=(~seen[b]).copy()
        for j in its: bg[j]=False
        nbg=int(bg.sum())
        if nbg>0: ew[b][bg]=0.5/nbg                                        # unrated=0
    return (torch.tensor(toks),torch.tensor(msk),torch.tensor(ptg),torch.tensor(pw),torch.tensor(etg),torch.tensor(ew))
print("train TWO-HEAD encoder (exposure x preference)...",flush=True)
bce=nn.functional.binary_cross_entropy_with_logits
for ep in range(30):
    rng.shuffle(trbig)
    for b0 in range(0,len(trbig),256):
        usb=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); t,m,ptg,pw,etg,ew=make_batch(usb,kk)
        u_p,u_e=enc(t,m); pl=u_p@Qp.t(); el=ebias+u_e@Qp.t()
        lp=(pw*bce(pl,ptg,reduction='none')).sum()/(pw.sum()+1e-6); le=(ew*bce(el,etg,reduction='none')).sum()/(ew.sum()+1e-6)
        loss=lp+le; opt.zero_grad(); loss.backward(); opt.step()
enc.eval(); Ql=Qp.detach().numpy()
_W=1./np.log2(np.arange(2,12))
def metr(score,tlike,excl,tail):
    sc=score.copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None,None
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    return (sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12), len([t for t in top if int(t) in rel])/len(rel))
def enc_u_batch(revs):                       # returns PREFERENCE head u_p (used for scoring)
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): up,_=enc(torch.tensor(arr),torch.tensor(m)); return up.numpy()
def enc_u(rev): return enc_u_batch([rev])[0] if rev else np.zeros(D)
def ridge(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
def sig(z): return 1/(1+np.exp(-z))
def rmva_order(prof):
    if len(prof)<2: return prof
    _,_,piv=sla.qr(Q[np.array(prof)].T,pivoting=True); return [prof[p] for p in piv]
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL]; RES={}
def A1():
    print("\n== A1 fold-in (NDCG/Recall) ==",flush=True)
    for tail in [False,True]:
        print(f" {'TAIL' if tail else 'FULL'}",flush=True)
        for meth in ['mostpop','ridge','encoder']:
            nd={q:0. for q in [0,2,4,8,'full']};rc={q:0. for q in [0,2,4,8,'full']};m=0
            for x in TE[:300]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tl=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tl or (tail and not any(not headmask[t] for t in tl)): continue
                order=sorted(prof,key=lambda j:-helf[j])
                for q in [0,2,4,8,'full']:
                    ri=order if q=='full' else order[:q]
                    if meth=='mostpop': s=popb.copy()
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
        for sel in ['random','helf','eig','oracle']:
            acc={q:0. for q in [0,1,2,4,8]};m=0
            for x in TE[:200]:
                test,prof=SPL[x]; rd=dict(rat_by_u[x]); tl=set(j for j in test if rd[j]>=4); prof=list(prof)
                if len(prof)<4 or not tl or (tail and not any(not headmask[t] for t in tl)): continue
                if sel=='random': o=prof[:];rng.shuffle(o);seq=o
                elif sel=='helf': seq=sorted(prof,key=lambda j:-helf[j])
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
        pur=[np.mean([item_g[j,g] for j in top(enc_u([(gcent[g],POS)]),10,pop=pop)]) for g in range(na)]
        print(f"  genre purity mean ({'with' if pop else 'no'} popb) = {np.mean(pur)*100:.0f}% (mostpop 16%)",flush=True)
    lp=[np.mean([item_g[j,g] for j in top(enc_u([(gcent[g],POS)]),10,pop=False)]) for g in range(na)]
    dp=[np.mean([item_g[j,g] for j in top(enc_u([(gcent[g],NEG)]),10,pop=False)]) for g in range(na)]
    RES['polgap']=(np.mean(lp)-np.mean(dp))*100
    print(f"  POLARITY gap (no popb): like {np.mean(lp)*100:.0f}% vs dislike {np.mean(dp)*100:.0f}% = {RES['polgap']:+.0f}pp",flush=True)
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
mono=all(RES[('encoder','full')][q]<=RES[('encoder','full')][nq]+1e-3 for q,nq in [(0,2),(2,4),(4,8)])
print(f"  G1 encoder>ridge @q8: FULL {'OK' if g1f else 'FAIL'} ({RES[('encoder','full')][8]:.3f} vs {RES[('ridge','full')][8]:.3f}) | TAIL {'OK' if g1t else 'FAIL'} ({RES[('encoder','tail')][8]:.3f} vs {RES[('ridge','tail')][8]:.3f})",flush=True)
print(f"  G2 eig>random @q8:    FULL {'OK' if g2f else 'FAIL'} ({RES[('eig','full')][8]:.3f} vs {RES[('random','full')][8]:.3f}) | TAIL {'OK' if g2t else 'FAIL'} ({RES[('eig','tail')][8]:.3f} vs {RES[('random','tail')][8]:.3f})",flush=True)
print(f"  G3 encoder monotone FULL: {'OK' if mono else 'FAIL'}",flush=True)
print(f"  G4 polarity gap (no popb): {RES['polgap']:+.0f}pp  (baseline +9pp; target >=+20pp)",flush=True)
print(f"  VERDICT: {'ELICITATION INTACT' if (g1f and g1t and g2f and g2t and mono) else '*** COLLAPSE - RETHINK ***'}",flush=True)
print("DONE",flush=True)
