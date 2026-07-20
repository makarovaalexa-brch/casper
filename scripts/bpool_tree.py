"""ADAPTIVE NDCG-TREE on the Kalman belief pool (deployable Golbandi adaptive questionnaire) vs the STATIC
questionnaire and the VARIANCE-GREEDY champion -- HEAD-TO-HEAD, SAME eval users, same budget. Runs on the
REFUSAL cache (bpool_cache_ref.npz): every usable concept is answerable (genuine answer or refusal) => universal
coverage, no non-answerer path.

Method: select each question by its ACTUAL expected tail-NDCG on a disjoint TRAIN population (the real objective
E3 used, not the variance surrogate VoI failed with). Golbandi property: users in a node share ONE belief
(same ancestor answer-bins => same mu,Sig), so a candidate is scored with one matvec per answer-level.

Arms (same eval users, concept-only shared questions, disjoint halves):
  TREE     : adaptive, node asks its own best question, split children by answer bin (NBIN), depth MAXD.
  STATIC   : same growth NBIN=1 (no split) = fixed questionnaire.
  VARGREEDY: per-user variance-greedy argmax dᵀSig d (answer-independent selection) = current champion.
  LIN      : per-user MAP-point NDCG-greedy (no Sig) = Lin WWW'23-style heuristic; isolates posterior value.
Probe on 2500 cache; headline needs ALL users + sign-off (HARD RULE #1/#2).
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from set_mn import RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"
MAXD=3; NBIN=3; MINLEAF=40
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0)
assert os.path.exists(CACHE), "run bpool_cacheref.py first"
z=np.load(CACHE,allow_pickle=True); Sig0=z['Sig0']; Bc=z['Bc']; EDG=z['EDG']; Yc=z['Yc']; S2=z['S2']; UIDX=z['uidx']; US=list(z['US'])
NCA=len(UIDX); Dm=Dc[UIDX]  # NCA x D candidate directions
log(f"ref cache: {len(US)} users, {NCA} usable concepts (refusal => universal coverage)")
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
def ndf(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,False)
# --- fast SHARED-ranking tail-NDCG@10: sort a score vector ONCE, cheap per-user lookup (only top-10 matters) ---
LOG2=1.0/np.log2(np.arange(2,12))          # DCG weights ranks 0..9
_RANKS=np.full(ni,-1,dtype=np.int64)
def top10_tail(sc):
    s=sc.copy(); s[head]=-1e30; idx=np.argpartition(-s,10)[:10]; return idx[np.argsort(-s[idx])]  # 10 tail ids, rank order
def prep(u):
    hlt,selU,valU,refU,aki,akr,hlall,us=u; tt=Bc[UIDX,0]*selU+Bc[UIDX,1]*valU+Bc[UIDX,2]
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)])
    nt=min(len(hlt),10); idcg=LOG2[:nt].sum() if nt>0 else 1.0
    return dict(hlt=np.asarray(hlt,np.int64),hlall=hlall,lev=lev,y=Yc[UIDX,lev],s2=np.maximum(S2[UIDX,lev],1e-3),idcg=idcg)
P=[prep(u) for u in US]
def fast_ndt_group(sc,users_idx,users):   # shared sc; return sum of tail-NDCG over the given users
    order=top10_tail(sc); _RANKS[order]=np.arange(10); tot=0.0
    for i in users_idx:
        p=users[i]; r=_RANKS[p['hlt']]; m=r>=0
        if m.any(): tot+=LOG2[r[m]].sum()/p['idcg']
    _RANKS[order]=-1; return tot
# self-test: fast shared-ranking tail-NDCG must match ndcg10 exactly
_rng=np.random.default_rng(3)
for _ti in range(20):
    _p=P[_rng.integers(len(P))]; _sc=popb+_rng.standard_normal(ni)*0.3
    _f=fast_ndt_group(_sc,[0],[_p]); _r=ndt(_sc,_p['hlt'])
    assert abs(_f-_r)<1e-9, f"fast-ndcg mismatch {_f} vs {_r}"
log("fast-ndcg self-test PASSED")
rs=np.random.default_rng(7); perm=rs.permutation(len(P)); half=len(P)//2
TR=[P[i] for i in perm[:half]]; EV=[P[i] for i in perm[half:]]
log(f"train {len(TR)} eval {len(EV)}")
def kal(mu,Sig,col,k):
    d=Dm[col]; y=Yc[UIDX[col],k]; s2=max(S2[UIDX[col],k],1e-3); Sd=Sig@d; kk=Sd/(d@Sd+s2); return mu+kk*(y-d@mu),Sig-np.outer(kk,Sd)
# score candidate col for a node of users sharing (mu,Sig): mean post-answer tail-NDCG. Everyone answers.
def score_cand(users,mu,Sig,col):
    by={}
    for i,p in enumerate(users): by.setdefault(p['lev'][col],[]).append(i)
    tot=0.0
    for k,idxs in by.items():
        mu2,_=kal(mu,Sig,col,k); sc=popb+Wd@mu2; tot+=fast_ndt_group(sc,idxs,users)
    return tot/len(users)
def best_col(users,mu,Sig):
    b=(-1,0)
    for col in range(NCA):
        s=score_cand(users,mu,Sig,col)
        if s>b[0]: b=(s,col)
    return b[1]
def grow(users,mu,Sig,depth,nbin):
    node=dict(col=None,mu=mu,Sig=Sig,children={},edges=None,n=len(users))
    if depth>=MAXD or len(users)<MINLEAF*max(nbin,1): return node
    col=best_col(users,mu,Sig); node['col']=col
    levs=np.array([p['lev'][col] for p in users])
    if nbin<=1:
        kbar=int(round(levs.mean())); mu2,Sig2=kal(mu,Sig,col,kbar); node['edges']=[]; node['children'][0]=grow(users,mu2,Sig2,depth+1,1); return node
    edges=np.quantile(levs,np.linspace(0,1,nbin+1)[1:-1]); node['edges']=list(edges)
    for b in range(nbin):
        lo=-1 if b==0 else edges[b-1]; hi=edges[b] if b<nbin-1 else 999
        sub=[p for p in users if (p['lev'][col]> (-1 if b==0 else edges[b-1]-1e-9)) and (p['lev'][col]<= (edges[b] if b<nbin-1 else 999))]
        if len(sub)<MINLEAF: sub=users
        kbar=int(round(np.mean([p['lev'][col] for p in sub]))); mu2,Sig2=kal(mu,Sig,col,kbar)
        node['children'][b]=grow(sub,mu2,Sig2,depth+1,nbin)
    return node
def route(node,p,depth,mus):
    mus.append(node['mu'])
    if node['col'] is None or depth>=MAXD:
        while len(mus)<=MAXD: mus.append(mus[-1]);
        return
    col=node['col']; k=p['lev'][col]; mu2,_=kal(node['mu'],node['Sig'],col,k)  # user's REAL update
    if not node['edges']: child=node['children'][0]
    else: b=int(np.searchsorted(node['edges'],k)); child=node['children'].get(b,list(node['children'].values())[0])
    c2=dict(child); c2['mu']=mu2; route(c2,p,depth+1,mus)
def eval_tree(node):
    tct=[[] for _ in range(MAXD+1)]; tcf=[[] for _ in range(MAXD+1)]
    for p in EV:
        mus=[]; route(node,p,0,mus)
        for dp in range(MAXD+1): sc=popb+Wd@mus[dp]; tct[dp].append(ndt(sc,p['hlt'])); tcf[dp].append(ndf(sc,p['hlall']))
    return [np.mean(x) for x in tct],[np.mean(x) for x in tcf]
def per_user(mode):
    tct=[[] for _ in range(MAXD+1)]; tcf=[[] for _ in range(MAXD+1)]
    for p in EV:
        mu=np.zeros(D); Sig=Sig0.copy(); avail=list(range(NCA))
        tct[0].append(ndt(popb+Wd@mu,p['hlt'])); tcf[0].append(ndf(popb+Wd@mu,p['hlall']))
        for dp in range(MAXD):
            if mode=='var':
                SD=Sig@Dm[avail].T; g=np.einsum('dq,dq->q',SD,SD); col=avail[int(np.argmax(g))]
            else:
                best=(-1,avail[0])
                for col in avail:
                    k=p['lev'][col]; mu2,_=kal(mu,Sig,col,k); v=ndt(popb+Wd@mu2,p['hlt'])
                    if v>best[0]: best=(v,col)
                col=best[1]
            k=p['lev'][col]; mu,Sig=kal(mu,Sig,col,k); avail.remove(col)
            tct[dp+1].append(ndt(popb+Wd@mu,p['hlt'])); tcf[dp+1].append(ndf(popb+Wd@mu,p['hlall']))
    return [np.mean(x) for x in tct],[np.mean(x) for x in tcf]
log("growing adaptive tree ..."); t0=time.time(); TREE=grow(TR,np.zeros(D),Sig0.copy(),0,NBIN); log(f"tree grown ({time.time()-t0:.0f}s)")
def describe(node,depth=0,pre=""):
    if node['col'] is None: return
    c=UIDX[node['col']]; log(f"  {pre}d{depth} q=concept{c} members={csz[c]} n={node['n']}")
    for b,ch in node['children'].items(): describe(ch,depth+1,pre+f"[{b}]")
describe(TREE)
log("static questionnaire ..."); STAT=grow(TR,np.zeros(D),Sig0.copy(),0,1)
at,af=eval_tree(TREE); st,sf=eval_tree(STAT)
log("variance-greedy ..."); vt,vf=per_user('var')
pt=np.mean([ndt(popb,p['hlt']) for p in EV]); pf=np.mean([ndf(popb,p['hlall']) for p in EV])
log("="*70); log(f"popb-only      TAIL {pt:.4f} FULL {pf:.4f}")
for nm,ct,cf in [('ADAPTIVE-TREE',at,af),('STATIC-QUEST',st,sf),('VAR-GREEDY',vt,vf)]:
    log(f"{nm:>14}  TAIL "+" ".join(f"d{i}:{ct[i]:.4f}" for i in range(MAXD+1))+"  FULL "+" ".join(f"d{i}:{cf[i]:.4f}" for i in range(MAXD+1)))
log(f"ADAPTIVITY PRIZE (tree-static)   TAIL@{MAXD} {at[MAXD]-st[MAXD]:+.4f}")
log(f"TREE vs CHAMPION (tree-vargreedy) TAIL@{MAXD} {at[MAXD]-vt[MAXD]:+.4f}  <-- decisive")
log("done")
