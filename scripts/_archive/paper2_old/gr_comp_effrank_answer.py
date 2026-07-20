"""
Goodreads MULTI-GENRE COMPOSITE: effective rank (participation ratio) of shelf-concept directions vs
item directions, + answerability (concept vs item answer-rate) on the FULL realistic cohort.
Mirrors gr_effrank_answer.py. Reads base_comp.npz + concepts_comp.npz + Q_svd_comp.npy.
"""
import os, time, numpy as np
GR='C:/dev/phd/casper/.cache/goodreads'; t0=time.time()
B=np.load(f'{GR}/base_comp.npz'); uu=B['uu']; ii=B['ii']; cnt=B['cnt']; ni=int(B['ni']); nu=int(B['nu']); trU=B['trU']
Q=np.load(f'{GR}/Q_svd_comp.npy'); D=Q.shape[1]
C=np.load(f'{GR}/concepts_comp.npz',allow_pickle=True); Ac=C['Ac']; ctags=C['ctags']; flat=C['citems_flat']; off=C['citems_off']
nc=len(ctags); trU_mask=np.zeros(nu,bool); trU_mask[trU]=True; ntr=len(trU)
citems=[flat[off[k]:off[k+1]] for k in range(nc)]
order_pop=np.argsort(-cnt); PITEMS=order_pop[:600]; Vitem=Q[PITEMS]
def analyze(name,Mv):
    Mn=Mv/(np.linalg.norm(Mv,axis=1,keepdims=True)+1e-9)
    Cc=Mn.T@Mn/Mn.shape[0]; w=np.clip(np.linalg.eigvalsh(Cc)[::-1],0,None); s=w.sum()
    pr=(s*s)/(np.sum(w*w)+1e-12); cum=np.cumsum(w)/s
    Mc=Mn-Mn.mean(0,keepdims=True); Cc2=Mc.T@Mc/Mn.shape[0]; wc=np.clip(np.linalg.eigvalsh(Cc2)[::-1],0,None)
    prc=(wc.sum()**2)/(np.sum(wc*wc)+1e-12)
    print(f"\n=== {name} ({Mv.shape[0]} unit vectors in R^{Mv.shape[1]}) ===")
    print(f"  UNCENTERED participation ratio = {pr:.2f} / {Mv.shape[1]}")
    print("   top-k var: "+"  ".join(f"k={k}:{cum[k-1]*100:4.1f}%" for k in (1,2,3,4,5,8,16,32)))
    print(f"  CENTERED  participation ratio = {prc:.2f} / {Mv.shape[1]}")
    return pr,prc
prc_c=analyze(f"SHELF CONCEPTS (Ac centroid, {nc} shelves)",Ac)
prc_i=analyze("POOL ITEMS (Q[top-600 popular])",Vitem)
# answerability
tr_row=trU_mask[uu]
rated_by_item=np.bincount(ii[tr_row],minlength=ni).astype(np.float64); item_ans=rated_by_item/max(ntr,1)
item2c=[[] for _ in range(ni)]
for ki in range(nc):
    for j in citems[ki]: item2c[j].append(ki)
SAMP=set(int(x) for x in trU[:3000]); pe_cnt=np.zeros(nc)
samp_rows=np.isin(uu,np.array(sorted(SAMP))); suu=uu[samp_rows]; sii=ii[samp_rows]
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
print(f"  ITEM answer-rate (frac train-users who rated item): mean {item_ans.mean():.4f} median {np.median(item_ans):.5f}",flush=True)
print(f"    top-600 popular pool items: mean {item_ans[PITEMS].mean():.4f} min {item_ans[PITEMS].min():.5f}",flush=True)
print(f"  CONCEPT answer-rate (frac users >=2 tagged items): mean {conc_ans.mean():.4f} median {np.median(conc_ans):.4f} max {conc_ans.max():.4f}",flush=True)
print(f"    concepts >=0.5: {(conc_ans>=0.5).sum()}/{nc};  >=0.8: {(conc_ans>=0.8).sum()}/{nc}",flush=True)
topc=np.argsort(-conc_ans)[:12]
print("  most-answerable concepts:")
for ki in topc: print(f"    {conc_ans[ki]:.4f}  {ctags[ki]}")
print(f"\nSUMMARY: concepts effrank(uncentered)={prc_c[0]:.2f}/{D} items effrank={prc_i[0]:.2f}/{D}  "
      f"concept-ans mean={conc_ans.mean():.4f} item-ans(all) mean={item_ans.mean():.4f} item-ans(pool) mean={item_ans[PITEMS].mean():.4f}",flush=True)
print("DONE",flush=True)
