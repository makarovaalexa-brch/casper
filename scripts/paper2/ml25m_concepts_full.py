"""
ML-25M PHASE-1B step B (FULL DATASET): genome concept channel. Build concept centroid directions Ac,
measure EFFECTIVE RANK (participation ratio) concepts vs top-600 items, and ANSWERABILITY (concept vs item
answer-rate) on the realistic FULL cohort. Reads prep.npz (uu,ii,R,cnt,trU,keepI) + genome-scores/tags.csv.
Memory-lean: vectorized item answer-rate; concept answer-rate on a 3000 train-user sample.
"""
import os, time, numpy as np
base='C:/dev/phd/casper/data/movielens'; out=f'{base}/.cache/ml25m'
Q=np.load(f'{out}/Q_svd.npy'); D=Q.shape[1]
P=np.load(f'{out}/prep.npz')
uu=P['uu']; ii=P['ii']; R=P['R']; cnt=P['cnt']; ni=int(P['ni']); nu=int(P['nu']); trU=P['trU']; keepI=P['keepI']
iids={int(x):k for k,x in enumerate(np.sort(keepI))}; keepids=set(iids)
trU_mask=np.zeros(nu,bool); trU_mask[trU]=True; ntr=len(trU)
t0=time.time()
# ---- genome concepts ----
tagitems={}
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
print(f"{nc} genome concepts (>=30 catalog items) built ({time.time()-t0:.0f}s)",flush=True)
# ---- effective rank ----
order_pop=np.argsort(-cnt); PITEMS=order_pop[:600]; Vitem=Q[PITEMS]
def analyze(name,Mv):
    Mn=Mv/(np.linalg.norm(Mv,axis=1,keepdims=True)+1e-9)
    C=Mn.T@Mn/Mn.shape[0]; w=np.clip(np.linalg.eigvalsh(C)[::-1],0,None); s=w.sum()
    pr=(s*s)/(np.sum(w*w)+1e-12); cum=np.cumsum(w)/s
    Mc=Mn-Mn.mean(0,keepdims=True); Cc=Mc.T@Mc/Mn.shape[0]; wc=np.clip(np.linalg.eigvalsh(Cc)[::-1],0,None); sc=wc.sum()
    prc=(sc*sc)/(np.sum(wc*wc)+1e-12); cumc=np.cumsum(wc)/sc
    print(f"\n=== {name} ({Mv.shape[0]} unit vectors in R^{Mv.shape[1]}) ===")
    print(f"  UNCENTERED participation ratio = {pr:.2f} / {Mv.shape[1]}")
    print("   top-k var: "+"  ".join(f"k={k}:{cum[k-1]*100:4.1f}%" for k in (1,2,3,4,5,8,16,32)))
    print(f"  CENTERED  participation ratio = {prc:.2f} / {Mv.shape[1]}")
    return pr,prc
prc_c=analyze("CONCEPTS (Ac centroid, %d tags)"%nc,Ac)
prc_i=analyze("POOL ITEMS (Q[top-600 popular])",Vitem)
# ---- answerability ----
# item answer-rate: fraction of TRAIN users who rated the item
tr_row=trU_mask[uu]
rated_by_item=np.bincount(ii[tr_row],minlength=ni).astype(np.float64)
item_ans=rated_by_item/max(ntr,1)
# concept answer-rate on 3000 train-user sample
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems[t]: item2c[j].append(ki)
SAMP=set(int(x) for x in trU[:3000]); pe_cnt=np.zeros(nc)
# gather rated items per sampled user
samp_rows=np.isin(uu,np.array(sorted(SAMP)))
suu=uu[samp_rows]; sii=ii[samp_rows]
byu={}
for k in range(len(suu)): byu.setdefault(int(suu[k]),[]).append(int(sii[k]))
for x,items in byu.items():
    cn={}
    for j in items:
        for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
    for ki,c2 in cn.items():
        if c2>=2: pe_cnt[ki]+=1
conc_ans=pe_cnt/max(len(byu),1)
print("\n=== ANSWERABILITY (FULL realistic cohort) ===")
print(f"  ITEM answer-rate (frac train-users who rated item): mean {item_ans.mean():.3f} median {np.median(item_ans):.4f}",flush=True)
print(f"    top-600 popular pool items:                       mean {item_ans[PITEMS].mean():.3f} min {item_ans[PITEMS].min():.4f}",flush=True)
print(f"  CONCEPT answer-rate (frac users >=2 tagged items):  mean {conc_ans.mean():.3f} median {np.median(conc_ans):.3f} max {conc_ans.max():.3f}",flush=True)
print(f"    concepts >=0.5: {(conc_ans>=0.5).sum()}/{nc};  >=0.8: {(conc_ans>=0.8).sum()}/{nc}",flush=True)
topc=np.argsort(-conc_ans)[:12]
print("  most-answerable concepts:")
for ki in topc: print(f"    {conc_ans[ki]:.3f}  {tagnames.get(int(ctags[ki]),'?')}")
print(f"\nSUMMARY: concepts effrank={prc_c[0]:.2f}/{D} items effrank={prc_i[0]:.2f}/{D}  "
      f"concept-ans mean={conc_ans.mean():.3f} item-ans(all) mean={item_ans.mean():.3f} item-ans(pool) mean={item_ans[PITEMS].mean():.3f}",flush=True)
print("DONE",flush=True)
