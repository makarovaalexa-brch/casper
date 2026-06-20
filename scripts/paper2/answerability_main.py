"""
PAPER B CORE TEST (honest): REALISTIC COLD-START with many UNKNOWN items. Asker starts empty, asks T questions over the
FULL CATALOGUE. An ITEM question yields signal ONLY if the user has actually seen it (item in known-half profile);
otherwise "unknown" = wasted turn. A CONCEPT question yields signal if the user has >=2 items in that concept. Compare
item askers (random / POPULAR = strong realistic baseline / answerability-aware EIG) vs concept askers (popular / EIG),
on FULL-FIRST + tail NDCG@10/Recall@50 vs held-out likes. Report ANSWERS-OBTAINED (why one wins). Canonical keep>=5,
correct EIG, frozen encoder. HYPOTHESIS: answerable(concept) >> item under realistic unknown-item answerability.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; rng=np.random.default_rng(0)
U,I,Rr=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U);I=np.array(I);Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids); keepids=set(iids)
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
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=-1.0
p_seen=(cnt/max(len(trU),1)).astype(np.float32)                       # item answerability prior (P a user has seen item j)
R=order_pop[:500]                                                     # popular reference set for cold-start info-gain
t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>0.5: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=30]; Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32); citems=[set(tagitems[t]) for t in ctags]; cfreq=np.array([len(s) for s in citems])
print(f"  {len(ctags)} concepts ({time.time()-t0:.0f}s)",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
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
def infogain_items(toks,cands):                # expected coverage over popular reference R (cold-start; no profile needed)
    u=enc_u(toks); p=sig(popb[cands]+Ql[cands]@u); ul=enc_batch([toks+[(Q[j],POS)] for j in cands]); ud=enc_batch([toks+[(Q[j],NEG)] for j in cands])
    return p*sig(popb[R]+ul@Ql[R].T).sum(1)+(1-p)*sig(popb[R]+ud@Ql[R].T).sum(1)
def infogain_concepts(toks,cidx):
    u=enc_u(toks); pc=np.array([sig(np.mean(Ql[list(citems[c])]@u+popb[list(citems[c])])) for c in cidx]); ul=enc_batch([toks+[(Ac[c],POS)] for c in cidx]); ud=enc_batch([toks+[(Ac[c],NEG)] for c in cidx])
    return pc*sig(popb[R]+ul@Ql[R].T).sum(1)+(1-pc)*sig(popb[R]+ud@Ql[R].T).sum(1)
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
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])   # prof=known(answerable) half, test=held-out
TE=[x for x in te if x in SPL][:150]
ITEMC=list(order_pop[:1500])                   # realistic item candidate pool (popular-ish; asking obscure items is pointless)
orc_pick={'i':0,'c':0}                          # diagnostic: does the oracle ever pick concepts?
def run(mode,tail):
    M={q:0. for q in [0,2,4,8]};Rc={q:0. for q in [0,2,4,8]};m=0;ans_tot=0.
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        toks=[]; asked=set(); nans=0
        for q in [0,2,4,8]:
            while len(toks)<q if mode!='oracle' else (len([1 for _ in toks])<q):
                if mode=='rand_item':
                    j=ITEMC[int(rng.integers(len(ITEMC)))]
                    if j in asked: continue
                    asked.add(j)
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode=='pop_item':
                    cs=[j for j in ITEMC if j not in asked]
                    if not cs: break
                    j=cs[0]; asked.add(j)
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode=='eig_item':                              # answerability-AWARE: p_seen * infogain
                    cs=[j for j in ITEMC if j not in asked][:500]
                    if not cs: break
                    csa=np.array(cs); j=cs[int((p_seen[csa]*infogain_items(toks,csa)).argmax())]; asked.add(j)
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode=='conc_pop':
                    ci=[c for c in range(len(ctags)) if c not in asked]; ci=sorted(ci,key=lambda c:-cfreq[c])
                    c=ci[0]; asked.add(c); inter=citems[c]&profset
                    if len(inter)>=2: toks.append((Ac[c],float(np.mean([resid[x][j] for j in inter])))); nans+=1
                elif mode=='conc_eig':
                    ci=[c for c in range(len(ctags)) if c not in asked]
                    ig=infogain_concepts(toks,ci); c=ci[int(ig.argmax())]; asked.add(c); inter=citems[c]&profset
                    if len(inter)>=2: toks.append((Ac[c],float(np.mean([resid[x][j] for j in inter])))); nans+=1
                elif mode=='oracle':                               # best answerable question (items OR concepts) by true held-out coverage
                    bestv=-1;bestt=None
                    for j in [jj for jj in ITEMC if jj in profset and ('i',jj) not in asked][:150]:
                        mt=metr(enc_u(toks+[(Q[j],resid[x][j])]),tlike,profset,tail)
                        if mt and mt[0]>bestv: bestv=mt[0];bestt=('i',j)
                    for c in [cc for cc in range(len(ctags)) if len(citems[cc]&profset)>=2 and ('c',cc) not in asked][:150]:
                        inter=citems[c]&profset; mt=metr(enc_u(toks+[(Ac[c],float(np.mean([resid[x][j] for j in inter])))]),tlike,profset,tail)
                        if mt and mt[0]>bestv: bestv=mt[0];bestt=('c',c)
                    if bestt is None: break
                    asked.add(bestt); orc_pick[bestt[0]]+=1
                    if bestt[0]=='i': toks.append((Q[bestt[1]],resid[x][bestt[1]]))
                    else: inter=citems[bestt[1]]&profset; toks.append((Ac[bestt[1]],float(np.mean([resid[x][j] for j in inter]))))
                    nans+=1
                if len(asked)>200: break
            mt=metr(enc_u(toks),tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1; ans_tot+=nans
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},m,ans_tot/m
for tail in [False,True]:
    print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'}: realistic cold-start, T={T} | NDCG@10 / Recall@50 / answers-obtained ===",flush=True)
    for mode in ['rand_item','pop_item','eig_item','conc_pop','conc_eig','oracle']:
        M,Rc,m,na=run(mode,tail); print(f"  {mode:<10}: NDCG "+" ".join(f"{M[q]:.3f}" for q in [0,2,4,8])+" | Rec "+" ".join(f"{Rc[q]:.3f}" for q in [0,2,4,8])+f" | ans/{T}={na:.1f}",flush=True)
    print(f"  (item answerable iff user saw it; concept iff >=2 items. GATE: conc_eig > pop_item/eig_item on full+tail)",flush=True)
print(f"\nORACLE PICK COMPOSITION (what the best answerable question actually is): items={orc_pick['i']} concepts={orc_pick['c']} -> {100*orc_pick['c']/max(orc_pick['i']+orc_pick['c'],1):.0f}% concepts",flush=True)
