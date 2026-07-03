"""
ML-25M PHASE-1 step B: genome concept channel — build concept direction vectors (centroid Ac, same construction
as ML-1M's Ec init in freeze_concept_encoder.py), measure EFFECTIVE RANK (participation ratio) of concept vs item
direction subspaces (mirrors effrank_concept_vs_item.py), and ANSWERABILITY stats (concept answer-rate vs item
answer-rate). Reuses .cache/ml25m/{Q_svd.npy,bi_svd.npy,meta.npz} from ml25m_build_svd.py.
"""
import os, time, numpy as np
base='C:/dev/phd/casper/data/movielens'; out=f'{base}/.cache/ml25m'
Q=np.load(f'{out}/Q_svd.npy'); bi=np.load(f'{out}/bi_svd.npy')
M=np.load(f'{out}/meta.npz',allow_pickle=True)
uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; ni=int(M['ni']); nu=int(M['nu'])
trU=set(int(x) for x in M['trU']); keepI=M['keepI']; D=Q.shape[1]
# dense item id map: catalog movieId -> dense index (same order as build: sorted keepI)
iids={int(x):k for k,x in enumerate(np.sort(keepI))}; keepids=set(iids)
likes_by_u={}; rat_by_u={}
for k in range(len(uu)):
    rat_by_u.setdefault(int(uu[k]),[]).append((int(ii[k]),float(rr[k])))
    if rr[k]>=4.0: likes_by_u.setdefault(int(uu[k]),[]).append(int(ii[k]))
# ---- genome concepts: tag -> catalog items with relevance>0.5, keep tags with >=30 items ----
t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>0.5: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=30]; nc=len(ctags)
Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32)
np.save(f'{out}/Ac_concept.npy',Ac); np.save(f'{out}/ctags_concept.npy',np.array(ctags))
tagnames={}
with open(f'{base}/genome-tags.csv',encoding='utf-8') as f:
    next(f)
    for line in f:
        p=line.rstrip('\n').split(','); tagnames[int(p[0])]=p[1]
print(f"{nc} genome concepts (>=30 catalog items) built in {time.time()-t0:.0f}s",flush=True)
# ---- EFFECTIVE RANK: concepts vs top-600 popular items (both unit-normalized) ----
order_pop=np.argsort(-cnt); PITEMS=order_pop[:600]; Vitem=Q[PITEMS]
def analyze(name,Mv):
    Mn=Mv/(np.linalg.norm(Mv,axis=1,keepdims=True)+1e-9)
    C=Mn.T@Mn/Mn.shape[0]; w=np.clip(np.linalg.eigvalsh(C)[::-1],0,None); s=w.sum()
    pr=(s*s)/(np.sum(w*w)+1e-12); cum=np.cumsum(w)/s
    Mc=Mn-Mn.mean(0,keepdims=True); Cc=Mc.T@Mc/Mn.shape[0]; wc=np.clip(np.linalg.eigvalsh(Cc)[::-1],0,None); sc=wc.sum()
    prc=(sc*sc)/(np.sum(wc*wc)+1e-12); cumc=np.cumsum(wc)/sc
    print(f"\n=== {name} ({Mv.shape[0]} unit vectors in R^{Mv.shape[1]}) ===")
    print(f"  UNCENTERED participation ratio (eff. rank) = {pr:.2f} / {Mv.shape[1]}")
    print("   top-k var: "+"  ".join(f"k={k}:{cum[k-1]*100:4.1f}%" for k in (1,2,3,4,5,8,16,32)))
    print(f"  CENTERED  participation ratio (eff. rank) = {prc:.2f} / {Mv.shape[1]}")
    print("   top-k var: "+"  ".join(f"k={k}:{cumc[k-1]*100:4.1f}%" for k in (1,2,3,4,5,8,16,32)))
    return pr,prc
print("\nEFFECTIVE-RANK: concept direction vectors vs pool-item direction vectors")
prc_c=analyze("CONCEPTS (Ac centroid, %d tags)"%nc,Ac)
prc_i=analyze("POOL ITEMS (Q[top-600 popular])",Vitem)
# ---- ANSWERABILITY: concept answer-rate vs item answer-rate ----
# item answer-rate: fraction of TRAIN users who rated the item (an item Q is answerable if the user has an opinion)
rated_by_item=np.zeros(ni)
seen={x:set(j for j,_ in rat_by_u[x]) for x in trU}
for x in trU:
    for j in seen[x]: rated_by_item[j]+=1
item_ans=rated_by_item/max(len(trU),1)
# concept answer-rate: fraction of TRAIN users with >=2 rated items carrying that tag (matches ctok >=2 rule)
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems[t]: item2c[j].append(ki)
NC=nc; pe_cnt=np.zeros(NC); SAMP=list(trU)[:3000]
for x in SAMP:
    cn={}
    for j in seen[x]:
        for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
    for ki,c2 in cn.items():
        if c2>=2: pe_cnt[ki]+=1
conc_ans=pe_cnt/len(SAMP)
# popular-item answer rate over the 600-pool (the entity set the policy actually asks from)
print("\n=== ANSWERABILITY (first cross-dataset datapoint) ===")
print(f"  ITEM answer-rate (frac train-users who rated item):  mean {item_ans.mean():.3f}  median {np.median(item_ans):.3f}")
print(f"    top-600 popular pool items:                        mean {item_ans[PITEMS].mean():.3f}  min {item_ans[PITEMS].min():.3f}")
print(f"  CONCEPT answer-rate (frac users with >=2 tagged items): mean {conc_ans.mean():.3f}  median {np.median(conc_ans):.3f}  max {conc_ans.max():.3f}")
print(f"    concepts with answer-rate >=0.5: {(conc_ans>=0.5).sum()}/{NC};  >=0.8: {(conc_ans>=0.8).sum()}/{NC}")
# most answerable concepts (the natural opening questions)
topc=np.argsort(-conc_ans)[:12]
print("  most-answerable concepts:")
for ki in topc:
    print(f"    {conc_ans[ki]:.3f}  {tagnames.get(int(ctags[ki]),'?')}")
print(f"\nSUMMARY: concepts effrank(uncentered)={prc_c[0]:.2f}/{D}  items effrank={prc_i[0]:.2f}/{D}  "
      f"concept-ans mean={conc_ans.mean():.3f}  item-ans(pool) mean={item_ans[PITEMS].mean():.3f}")
print("DONE",flush=True)
