"""
Replicate Golbandi et al. (WSDM 2011) adaptive ternary decision-tree interview on
ML-100k. At each node pick the item minimizing within-group SSE; split users into
like(>=4)/dislike(<4)/unknown; leaf = shrunk group-mean profile. Cold user traverses
by answering -> leaf profile predicts. Compare to popularity (global-mean profile)
and a NON-adaptive most-popular tree. Metrics: RMSE + NDCG@10 vs interview depth.
"""
import os, time, numpy as np
base='C:/dev/phd/casper/data/movielens/ml-100k'
MAXD=int(os.environ.get('MAXD',6)); POOL=int(os.environ.get('POOL',80)); MINN=20; LAM=float(os.environ.get('LAM',8))
N_COLD=200; rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{base}/u.data') as f:
    for line in f:
        a=line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U); I=np.array(I); R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
cold=set(np.random.default_rng(0).choice(nu,N_COLD,replace=False).tolist()); warm=[x for x in range(nu) if x not in cold]
Rm=np.full((nu,ni),np.nan,np.float32)
for k in range(len(u)): Rm[u[k],i[k]]=R[k]
cnt=np.array([(~np.isnan(Rm[warm,j])).sum() for j in range(ni)]); pool=list(np.argsort(-cnt)[:POOL])
gmean=np.nanmean(Rm[warm],0); gmean=np.where(np.isnan(gmean),np.nanmean(R),gmean)
Rf=np.nan_to_num(Rm,nan=0.0); M=(~np.isnan(Rm)).astype(np.float32); Rsq=Rf*Rf
def grp_sse(rows):
    if len(rows)==0: return 0.0
    s=Rf[rows].sum(0); sq=Rsq[rows].sum(0); c=M[rows].sum(0)
    return float(np.where(c>0, sq - s*s/np.maximum(c,1), 0.0).sum())
def profile(rows):  # shrunk group-mean per item
    s=Rf[rows].sum(0); c=M[rows].sum(0); return (s+LAM*gmean)/(c+LAM)
def build(rows, depth):
    node={'profile':profile(rows)}
    if depth>=MAXD or len(rows)<MINN: return node
    col_all=Rm[rows]; best=None; bj=None
    for j in pool:
        col=col_all[:,j]; L=rows[(col>=4)]; Dd=rows[(col<4)&(~np.isnan(col))]; Uu=rows[np.isnan(col)]
        if len(L)<5 or len(Dd)<5: continue
        e=grp_sse(L)+grp_sse(Dd)+grp_sse(Uu)
        if best is None or e<best: best=e; bj=j; split=(L,Dd,Uu)
    if bj is None: return node
    node['item']=bj; node['L']=build(split[0],depth+1); node['D']=build(split[1],depth+1); node['U']=build(split[2],depth+1)
    return node
def build_static(rows, depth, k):
    # NON-adaptive: at each level ask the SAME fixed popular item (pool[depth]) to everyone; same group-mean leaf
    node={'profile':profile(rows)}
    if depth>=k or len(rows)<MINN or depth>=len(pool): return node
    j=pool[depth]; col=Rm[rows][:,j]
    L=rows[(col>=4)]; Dd=rows[(col<4)&(~np.isnan(col))]; Uu=rows[np.isnan(col)]
    node['item']=j; node['L']=build_static(L,depth+1,k); node['D']=build_static(Dd,depth+1,k); node['U']=build_static(Uu,depth+1,k)
    return node
print("building Golbandi tree...",flush=True); t0=time.time()
root=build(np.array(warm),0); print(f"built ({time.time()-t0:.0f}s)",flush=True)
static_root=build_static(np.array(warm),0,MAXD); print("built static-popular tree",flush=True)
# also a non-adaptive most-popular tree: ask pool[0..d] in fixed order, predict via reached group means
def traverse(node, rd, depth):
    prof=node['profile']
    if depth==0 or 'item' not in node: return [prof]
    j=node['item']; v=rd.get(j,None)
    child=node['U'] if v is None else (node['L'] if v>=4 else node['D'])
    return [prof]+traverse(child,rd,depth-1)
# cold eval
cases=[]
for cu in cold:
    rd={int(k):float(Rm[cu,k]) for k in np.where(~np.isnan(Rm[cu]))[0]}
    items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))])
    rel=[j for j in test if rd[j]>=4];
    if not rel: continue
    cases.append((rd,test,rel,np.array([j for j in range(ni) if j not in rd])))
print(f"cold cases {len(cases)}",flush=True)
def ndcg(prof,rel,unr):
    return np.mean([(1./np.log2(2+int((prof[unr]>=prof[ri]).sum())) if 1+int((prof[unr]>=prof[ri]).sum())<=10 else 0.) for ri in rel])
def rmse(prof,test,rd):
    hi=np.array(list(test)); ht=np.array([rd[j] for j in test]); return np.sqrt(np.mean((ht-np.clip(prof[hi],1,5))**2))
# evaluate tree per depth
nd=np.zeros(MAXD+1); rm=np.zeros(MAXD+1); nds=np.zeros(MAXD+1); rms=np.zeros(MAXD+1)
for (rd,test,rel,unr) in cases:
    profs=traverse(root,rd,MAXD); sprofs=traverse(static_root,rd,MAXD)
    for d in range(MAXD+1):
        p=profs[min(d,len(profs)-1)]; nd[d]+=ndcg(p,rel,unr); rm[d]+=rmse(p,test,rd)
        ps=sprofs[min(d,len(sprofs)-1)]; nds[d]+=ndcg(ps,rel,unr); rms[d]+=rmse(ps,test,rd)
nd/=len(cases); rm/=len(cases); nds/=len(cases); rms/=len(cases)
# popularity static = depth-0 (global mean profile) ; also popularity-rank by count
poprank=cnt.astype(np.float32)
nd_pop=np.mean([ndcg(poprank,rel,unr) for (rd,test,rel,unr) in cases])
print("\n=== Golbandi ADAPTIVE tree vs STATIC-popular interview (ML-100k cold-test, equal #questions) ===")
print(f"{'depth':>5} | {'ADAPT RMSE':>10} {'STATIC RMSE':>11} | {'ADAPT NDCG':>10} {'STATIC NDCG':>11}")
for d in range(MAXD+1): print(f"{d:>5} | {rm[d]:>10.4f} {rms[d]:>11.4f} | {nd[d]:>10.4f} {nds[d]:>11.4f}")
print(f"\nKEY (Golbandi's metric = RMSE): adaptive {rm[-1]:.4f} vs static {rms[-1]:.4f} at depth {MAXD} "
      f"-> adaptive {'WINS' if rm[-1]<rms[-1] else 'LOSES'} by {rms[-1]-rm[-1]:+.4f}")
print(f"popularity-rank NDCG@10 = {nd_pop:.4f} (group-mean leaf is a poor RANKER -> see DRE for NDCG)")
