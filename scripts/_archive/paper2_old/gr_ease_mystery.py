"""
VARIANT B (implicit track) + explicit-EASE isolation control, on the single-genre MYSTERY slice
(phase-1's arena; base.npz / Q_svd / enc_v1_gr, K=142). Restricted top-N-by-count universe.

Reproduces gr_prep's user index (umap = first-appearance among rating>0 rows) so that is_read
implicit interactions align to the SAME te/va/trU as base.npz (verified against base.npz keepB+cnt+te).
Then a 2x2 comparison on ONE arena / one ruler / SAME held-out explicit targets (rng123), IDENTICAL
candidate universe and IDENTICAL exclusion set (everything the user has interacted with):
  - MOSTPOP (explicit like-popularity)
  - ENCODER (enc_v1_gr, explicit fold-in)          [our signal, our model]
  - EASE-explicit (rating>=4 binary)               [our signal, EASE model]  -> isolates MODEL
  - EASE-implicit (is_read=True binary)            [implicit signal, EASE]   -> isolates SIGNAL  (VARIANT B)
lambda sweep {1,10,100,500} on val for each EASE. Mystery interactions only (stated fallback).
No encoder training, no commits.
"""
import os, time, gzip, json, numpy as np, torch, torch.nn as nn, scipy.sparse as sp, array
t0=time.time(); GR='C:/dev/phd/casper/.cache/goodreads'
INTER=f'{GR}/goodreads_interactions_mystery_thriller_crime.json.gz'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0; LAM_R=5.0; MINIT=20
N=int(os.environ.get('NUNIV',20000)); KP=int(os.environ.get('KP',142)); Ks=[10,KP]
LAMS=[1.0,10.0,100.0,500.0]
Bnp=np.load(f'{GR}/base.npz')
uu=Bnp['uu']; ii=Bnp['ii']; rr=Bnp['rr']; cnt=Bnp['cnt']; mu=float(Bnp['mu']); ni=int(Bnp['ni']); nu=int(Bnp['nu'])
trU=Bnp['trU']; va=Bnp['va']; te=Bnp['te']; keepB=Bnp['keepB']
Q=np.load(f'{GR}/Q_svd.npy').astype(np.float32); bi=np.load(f'{GR}/bi_svd.npy').astype(np.float32)
popb=np.log(cnt+1.0).astype(np.float32)
univ=np.sort(np.argsort(-cnt)[:N]); umask=np.zeros(ni,bool); umask[univ]=True
loc=-np.ones(ni,np.int64); loc[univ]=np.arange(N)
print(f"[universe] N={N} of {ni}; like-count share={cnt[univ].sum()/cnt.sum()*100:.1f}%",flush=True)

# ================= PASS 1: reproduce umap + explicit, verify against base.npz =================
umap={}; bmap={}; au=array.array('i'); ab=array.array('i'); nrow=0
with gzip.open(INTER,'rt',encoding='utf-8') as f:
    for line in f:
        nrow+=1
        if nrow%4000000==0: print(f"  p1 {nrow/1e6:.0f}M rows ({time.time()-t0:.0f}s)",flush=True)
        try: d=json.loads(line)
        except Exception: continue
        r=d.get('rating',0)
        if not r or r<=0: continue
        uid=d['user_id']; bid=d['book_id']
        ui=umap.get(uid)
        if ui is None: ui=len(umap); umap[uid]=ui
        bi_=bmap.get(bid)
        if bi_ is None: bi_=len(bmap); bmap[bid]=bi_
        au.append(ui); ab.append(bi_)
Uraw=np.frombuffer(au,dtype=np.int32).copy(); Braw=np.frombuffer(ab,dtype=np.int32).copy(); del au,ab
bid_by_idx=np.zeros(len(bmap),np.int64)
for bid,k in bmap.items(): bid_by_idx[k]=np.int64(bid)
bcnt=np.bincount(Braw,minlength=len(bid_by_idx)); keepB_local=np.nonzero(bcnt>=MINIT)[0]
imap=-np.ones(len(bid_by_idx),np.int32); imap[keepB_local]=np.arange(len(keepB_local))
mask=imap[Braw]>=0; Uk=Uraw[mask]; iik=imap[Braw][mask]
uniqU=np.unique(Uk); umap2=np.zeros(int(uniqU.max())+1,np.int32); umap2[uniqU]=np.arange(len(uniqU))
keepB_rep=bid_by_idx[keepB_local]
# verify alignment
ok=(len(keepB_rep)==ni) and np.array_equal(keepB_rep,keepB)
print(f"[verify] reproduced ni={len(keepB_rep)} vs base ni={ni}; keepB match={ok} ({time.time()-t0:.0f}s)",flush=True)
assert ok,"keepB mismatch -> user index would not align; abort"
# uid -> dense user idx (final compacted)  = umap2[umap[uid]] (only if umap[uid] in uniqU)
uid2dense={}
inuniq=np.zeros(int(uniqU.max())+1,bool); inuniq[uniqU]=True
for uid,raw in umap.items():
    if raw<len(inuniq) and inuniq[raw]: uid2dense[uid]=int(umap2[raw])
# book_id string -> dense item idx
bid_str2dense={}
for lidx,braw in enumerate(keepB_local): bid_str2dense[str(int(bid_by_idx[braw]))]=lidx
del Uraw,Braw,Uk,iik,bmap,bid_by_idx

# ================= PASS 2: implicit is_read interactions -> (dense_user, dense_item) =================
iu=array.array('i'); it=array.array('i'); nread=0; nrow=0
with gzip.open(INTER,'rt',encoding='utf-8') as f:
    for line in f:
        nrow+=1
        if nrow%4000000==0: print(f"  p2 {nrow/1e6:.0f}M rows, kept_read {nread} ({time.time()-t0:.0f}s)",flush=True)
        try: d=json.loads(line)
        except Exception: continue
        if not d.get('is_read',False): continue
        du=uid2dense.get(d['user_id'])
        if du is None: continue
        di=bid_str2dense.get(d['book_id'])
        if di is None: continue           # book not in mystery >=20 catalog
        iu.append(du); it.append(di); nread+=1
IU=np.frombuffer(iu,dtype=np.int32).copy(); IT=np.frombuffer(it,dtype=np.int32).copy(); del iu,it
print(f"[implicit] {nread} is_read interactions over catalog ({time.time()-t0:.0f}s)",flush=True)
# read-set per eval user (universe-restricted)
readmask_u=np.isin(IU,np.array(sorted(set(va.tolist())|set(te.tolist())),np.int32))
read_by_u={}
for k in np.nonzero(readmask_u)[0]:
    j=int(IT[k])
    if umask[j]: read_by_u.setdefault(int(IU[k]),set()).add(j)

# ================= explicit rating dicts + SPL held-out targets =================
evalset=np.array(sorted(set(va.tolist())|set(te.tolist()))); selm=np.isin(uu,evalset)
euu=uu[selm]; eii=ii[selm]; eR=rr[selm]; rat_by_u={}
for k in range(len(euu)): rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
_rs=np.random.default_rng(123); SPL={}
for x in (va.tolist()+te.tolist()):
    lk=[j for j,r in rat_by_u.get(x,[]) if r>=LIKE]
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])

# ================= EASE matrices (train users x universe) =================
trset=np.zeros(nu,bool); trset[trU]=True
# explicit like X
rm=(rr>=LIKE)&umask[ii]&trset[uu]
Xe=sp.csr_matrix((np.ones(rm.sum(),np.float32),(uu[rm],loc[ii[rm]])),shape=(nu,N))
# implicit read X
trrows=trset[IU]&umask[IT]
Xi=sp.csr_matrix((np.ones(int(trrows.sum()),np.float32),(IU[trrows],loc[IT[trrows]])),shape=(nu,N))
print(f"[EASE X] explicit nnz={Xe.nnz} (users {np.asarray((Xe.sum(1)>0)).sum()}); "
      f"implicit nnz={Xi.nnz} (users {np.asarray((Xi.sum(1)>0)).sum()}) ({time.time()-t0:.0f}s)",flush=True)
def ease_B(X,lam):
    G=np.asarray((X.T@X).todense(),dtype=np.float64); G[np.diag_indices_from(G)]+=lam
    P=np.linalg.inv(G); dP=np.diag(P).copy(); Bm=-P/dP[None,:]; Bm[np.diag_indices_from(Bm)]=0.0
    del G,P; return Bm.astype(np.float32)

# ================= encoder =================
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{GR}/enc_v1_gr.pt')); enc.eval()
def enc_u(toks):
    if not toks: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(toks),D+1),np.float32); mk=np.ones((1,len(toks)),np.float32)
    for q,(f,res) in enumerate(toks): tk[0,q,:D]=f; tk[0,q,D]=res
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
W=1./np.log2(np.arange(2,KP+2))
def ndcg(sc,rel,K):
    if not rel: return None
    top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]; rs=set(rel)
    dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
def eval_all(users, Be, Bi):
    acc={m:{K:0. for K in Ks} for m in ['MOSTPOP','ENCODER','EASE-expl','EASE-impl']}; m=0
    neg=np.full(ni,-1e9,np.float64)
    for x in users:
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]
        prof_e=[j for j in rd if (j not in test) and umask[j]]         # explicit profile in universe
        prof_i=[j for j in read_by_u.get(x,()) if j not in test]       # implicit profile in universe
        rel=[t for t in test if umask[t]]
        if len(rel)<1 or len(prof_e)<1: continue                       # need explicit target+profile to exist
        m+=1
        excl=set(prof_e)|set(prof_i)                                   # identical exclusion for all models
        uE=enc_u([(Q[j],rd[j]-mu-bi[j]) for j in prof_e])
        scP=neg.copy(); scP[univ]=popb[univ]
        scE=neg.copy(); scE[univ]=popb[univ]+Q[univ]@uE
        ee=np.zeros(N,np.float64)
        for j in prof_e: ee+=Be[loc[j]]
        scSe=neg.copy(); scSe[univ]=ee
        ei=np.zeros(N,np.float64)
        for j in prof_i: ei+=Bi[loc[j]]
        scSi=neg.copy(); scSi[univ]=ei
        el=list(excl)
        for s in (scP,scE,scSe,scSi): s[el]=-1e9
        for K in Ks:
            acc['MOSTPOP'][K]+=ndcg(scP,rel,K) or 0
            acc['ENCODER'][K]+=ndcg(scE,rel,K) or 0
            acc['EASE-expl'][K]+=ndcg(scSe,rel,K) or 0
            acc['EASE-impl'][K]+=ndcg(scSi,rel,K) or 0
    for mm in acc:
        for K in Ks: acc[mm][K]/=max(m,1)
    return acc,m
# lambda sweep (val) for each EASE independently
print(f"\n=== MYSTERY arena; universe N={N}; K={KP} ===",flush=True)
def sweep(which):
    best=None
    for lam in LAMS:
        Bm=ease_B(Xe if which=='e' else Xi,lam)
        # cheap val: build score only for this EASE
        acc={K:0. for K in Ks}; m=0
        for x in va.tolist():
            if x not in SPL: continue
            rd=dict(rat_by_u[x]); test=SPL[x]
            prof=[j for j in (rd if which=='e' else read_by_u.get(x,())) if (j not in test) and umask[j]]
            prof_e=[j for j in rd if (j not in test) and umask[j]]
            rel=[t for t in test if umask[t]]
            if len(rel)<1 or len(prof_e)<1 or len(prof)<1: continue
            m+=1; ss=np.zeros(N,np.float64)
            for j in prof: ss+=Bm[loc[j]]
            sc=np.full(ni,-1e9,np.float64); sc[univ]=ss
            sc[list(set(prof_e)|set(read_by_u.get(x,())))]=-1e9
            v=ndcg(sc,rel,KP)
            if v is not None: acc[KP]+=v
        acc[KP]/=max(m,1)
        print(f"[val {which} lam={lam:>5.0f}] EASE @{KP}={acc[KP]:.4f} (n={m})",flush=True)
        if best is None or acc[KP]>best[1]: best=(lam,acc[KP],Bm)
        else: del Bm
    print(f">>> {which}: val-selected lam={best[0]}",flush=True); return best[0],best[2]
lam_e,Be=sweep('e'); lam_i,Bi=sweep('i')
accT,mte=eval_all(te.tolist(),Be,Bi)
print(f"\n-- TEST (n={mte}); universe N={N}; K={KP}; excl=union(explicit,implicit) --",flush=True)
print(f"   (val-selected lambda: explicit={lam_e}, implicit={lam_i})",flush=True)
for mm in ['MOSTPOP','ENCODER','EASE-expl','EASE-impl']:
    print(f"  {mm:<10}| @10={accT[mm][10]:.4f} | @{KP}={accT[mm][KP]:.4f}",flush=True)
mp=accT['MOSTPOP']
print(f"\n>>> HEADROOM over MOSTPOP @{KP}: ENCODER={accT['ENCODER'][KP]-mp[KP]:+.4f} | "
      f"EASE-expl={accT['EASE-expl'][KP]-mp[KP]:+.4f} | EASE-impl(VAR-B)={accT['EASE-impl'][KP]-mp[KP]:+.4f}",flush=True)
print(f">>> HEADROOM over MOSTPOP @10 : ENCODER={accT['ENCODER'][10]-mp[10]:+.4f} | "
      f"EASE-expl={accT['EASE-expl'][10]-mp[10]:+.4f} | EASE-impl(VAR-B)={accT['EASE-impl'][10]-mp[10]:+.4f}",flush=True)
# implicit profile density diagnostic
dens=[len(read_by_u.get(x,())) for x in te.tolist() if x in SPL]
print(f">>> implicit read-set size on test users: median={np.median(dens):.0f} mean={np.mean(dens):.1f} (universe-restricted)",flush=True)
print(f"DONE ({time.time()-t0:.0f}s)",flush=True)
