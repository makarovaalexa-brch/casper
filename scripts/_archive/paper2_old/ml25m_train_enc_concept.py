"""
ML-25M PHASE-2 STEP 1b: train a CONCEPT-AWARE V1 encoder (the "learned concept channel" / OOD fix).
Same recipe as ml25m_train_enc.py but each user's revealed reveal-set MIXES item tokens (Q[j],res) and
CONCEPT tokens (Ac[c], mean residual over the user's >=2 rated tagged items). Teaches the attention fold-in
to USE concept centroids. Reconstruction target unchanged (unrevealed like-set). Frozen Q/bi decoder.
Warm-start from enc_v1_ml25m (best), fine-tune. Val = item full-profile NDCG@50 (stays a good recommender)
+ a concept-elicitation val (8 divisive concept reveals) for selection on the concept ability.
Durable: enc_v1c_ml25m.pt (best concept-val), _state.pt (resume), _peak.txt.
Env: EP_CHUNK(2), EP_MAX(8), K(12), CFRAC(0.5 concept fraction of reveals), FRESH(0), SELMET(concept|item).
"""
import os, time, numpy as np, torch, torch.nn as nn
t0=time.time(); out='C:/dev/phd/casper/data/movielens/.cache/ml25m'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0
K=int(os.environ.get('K',12)); BATCH=int(os.environ.get('BATCH',512))
EP_CHUNK=int(os.environ.get('EP_CHUNK',2)); EP_MAX=int(os.environ.get('EP_MAX',8))
CFRAC=float(os.environ.get('CFRAC',0.5)); SELMET=os.environ.get('SELMET','concept')
rng=np.random.default_rng(0); torch.manual_seed(0)
M=np.load(f'{out}/meta.npz'); uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; mu=float(M['mu'])
ni=int(M['ni']); nu=int(M['nu']); trU=M['trU']; va=M['va']
Q=np.load(f'{out}/Q_svd.npy').astype(np.float32); bi=np.load(f'{out}/bi_svd.npy').astype(np.float32); Qt=torch.tensor(Q)
Ac=np.load(f'{out}/Ac_concept.npy').astype(np.float32); nc=Ac.shape[0]
mem=np.load(f'{out}/membership.npz'); cf_flat=mem['citems_flat']; cf_off=mem['citems_off']; cdiv=mem['cdiv']
item2c=[[] for _ in range(ni)]
for c in range(nc):
    for j in cf_flat[cf_off[c]:cf_off[c+1]]: item2c[int(j)].append(c)
item2c=[np.array(v,np.int32) for v in item2c]
CONC_ENT_ORD=list(np.argsort(-cdiv))
popb=np.log(cnt+1.0).astype(np.float32)
ipsw=np.clip((1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32),None,float(os.environ.get('IPSCLIP',8.0)))
CTOK=int(os.environ.get('CTOK',6))          # extra concept tokens appended on top of the full item-reveal set
PCONLY=float(os.environ.get('PCONLY',0.35)) # prob a user is trained CONCEPT-ONLY (drop items) so pure-concept folds are in-distribution
order=np.argsort(uu,kind='stable'); uu_s=uu[order]; ii_s=ii[order].astype(np.int32); rr_s=rr[order].astype(np.float32)
res_s=(rr_s-mu-bi[ii_s]).astype(np.float32)
off=np.zeros(nu+1,np.int64); np.add.at(off,uu_s+1,1); off=np.cumsum(off)
print(f"CSR built ({time.time()-t0:.0f}s); nu={nu} ni={ni} nc={nc}",flush=True)
lens=(off[1:]-off[:-1]); trbig=np.array([x for x in trU if lens[x]>=K+1],dtype=np.int64)
print(f"train users (>=K+1): {len(trbig)}",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc()
def user_concepts(its,rss):  # -> dict concept->mean residual over rated tagged (>=2)
    acc={}
    for j,r in zip(its,rss):
        for c in item2c[j]: acc.setdefault(int(c),[]).append(r)
    return {c:float(np.mean(v)) for c,v in acc.items() if len(v)>=2}
def make_batch(users,kk):
    W=kk+CTOK; B=len(users); toks=np.zeros((B,W,D+1),np.float32); msk=np.zeros((B,W),np.float32)
    tgt=np.zeros((B,ni),np.float32); wt=np.ones((B,ni),np.float32); seen=np.zeros((B,ni),bool)
    for b,x in enumerate(users):
        a,c=off[x],off[x+1]; idx=np.arange(a,c); rng.shuffle(idx); rev=idx[:kk]
        its=ii_s[rev]; rss=res_s[rev]
        conly=(rng.random()<PCONLY)                                   # CONCEPT-ONLY user: teach pure-concept folds (in-distribution)
        cans=user_concepts(its.tolist(),rss.tolist())
        cs=[cc for cc in CONC_ENT_ORD if cc in cans][:CTOK]
        L=0
        if not conly:                                                 # full item reveal set (stable, == item-only trainer) + appended concept tokens
            for q in range(len(its)): toks[b,L,:D]=Q[its[q]]; toks[b,L,D]=rss[q]; msk[b,L]=1; L+=1
            seen[b,its]=True
        for cc in cs:
            toks[b,L,:D]=Ac[cc]; toks[b,L,D]=cans[cc]; msk[b,L]=1; L+=1
        if conly and L==0:                                            # no concepts to reveal -> fall back to items
            for q in range(len(its)): toks[b,L,:D]=Q[its[q]]; toks[b,L,D]=rss[q]; msk[b,L]=1; L+=1
            seen[b,its]=True
        allit=ii_s[a:c]; alll=allit[rr_s[a:c]>=LIKE]; unl=alll[~seen[b,alll]]
        if len(unl): tgt[b,unl]=1.0; wt[b,unl]=ipsw[unl]
        posw=float(ipsw[unl].sum()) if len(unl) else 0.
        negmask=(~seen[b])&(tgt[b]==0); nneg=int(negmask.sum())
        if nneg>0 and posw>0: wt[b,negmask]=posw/nneg
    return (torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen))
# ---- val ----
_rs=np.random.default_rng(123); VSPL={}; vpool=list(va)
for x in vpool:
    a,c=off[x],off[x+1]; its=ii_s[a:c]; lk=its[rr_s[a:c]>=LIKE]
    if len(lk)>=4: ll=lk.copy(); _rs.shuffle(ll); VSPL[x]=set(int(j) for j in ll[len(ll)//2:])
_Wv=1./np.log2(np.arange(2,52))
def enc_u_np(rows):
    if not rows: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(rows),D+1),np.float32); mk=np.ones((1,len(rows)),np.float32)
    for q,(f,r) in enumerate(rows): tk[0,q,:D]=f; tk[0,q,D]=r
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def val_ndcg50(concept=False):
    enc.eval(); tot=0.; m=0
    for x in vpool:
        if x not in VSPL: continue
        a,c=off[x],off[x+1]; its=ii_s[a:c]; rss=res_s[a:c]; test=VSPL[x]
        pm=np.array([int(j) not in test for j in its]); prof_it=its[pm]; prof_rs=rss[pm]; rel=test
        if len(prof_it)<2 or not rel: continue
        if concept:  # 8 divisive answerable concept reveals (wasted-turn), fold concept tokens only
            cans=user_concepts(prof_it.tolist(),prof_rs.tolist())
            cs=[cc for cc in CONC_ENT_ORD if cc in cans][:8]
            rows=[(Ac[cc],cans[cc]) for cc in cs]
        else:
            rows=[(Q[j],r) for j,r in zip(prof_it,prof_rs)]
        u=enc_u_np(rows); sc=popb+Q@u; sc[prof_it]=-1e9
        top=np.argpartition(-sc,50)[:50]; top=top[np.argsort(-sc[top])]
        dcg=sum(_Wv[p] for p,t in enumerate(top) if int(t) in rel); idcg=_Wv[:min(50,len(rel))].sum()
        tot+=dcg/idcg if idcg>0 else 0.; m+=1
    enc.train(); return tot/max(m,1), m
# ---- resume / warm-start ----
st=f'{out}/enc_v1c_ml25m_state.pt'; bestf=f'{out}/enc_v1c_ml25m.pt'; peakf=f'{out}/enc_v1c_ml25m_peak.txt'
ep0=0; best=-1.0
opt=torch.optim.Adam(enc.parameters(),float(os.environ.get('LR','5e-4')),weight_decay=1e-5)
if os.path.exists(st):
    ck=torch.load(st); enc.load_state_dict(ck['enc']); opt.load_state_dict(ck['opt']); ep0=ck['epoch']; best=ck['best']
    print(f"[RESUME] ep{ep0} best-concept-val={best:.4f}",flush=True)
elif not os.environ.get('FRESH'):
    enc.load_state_dict(torch.load(f'{out}/enc_v1_ml25m.pt')); print("[WARM] from enc_v1_ml25m (item-only)",flush=True)
nrun=min(EP_CHUNK,EP_MAX-ep0)
if nrun<=0:
    print(f"[DONE] {ep0}/{EP_MAX} epochs; best-concept-val={best:.4f}",flush=True)
else:
    for e in range(nrun):
        ep=ep0+e; rng.shuffle(trbig); tl=0.; nb=0
        for b0 in range(0,len(trbig),BATCH):
            us=trbig[b0:b0+BATCH]; kk=int(rng.integers(2,K+1))
            toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk); sc=u@Qt.t()
            loss=wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')
            loss=loss.masked_fill(seen,0.).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        vC,mC=val_ndcg50(concept=True); vI,mI=val_ndcg50(concept=False); sel=vC if SELMET=='concept' else vI
        improved=sel>best
        if improved: best=sel; torch.save(enc.state_dict(),bestf)
        torch.save({'enc':enc.state_dict(),'opt':opt.state_dict(),'epoch':ep+1,'best':best},st)
        with open(peakf,'a') as f: f.write(f"ep{ep+1} loss={tl/nb:.4f} concept_val50={vC:.4f} item_val50={vI:.4f} best({SELMET})={best:.4f}{' *SAVED' if improved else ''}\n")
        print(f"ep{ep+1}/{EP_MAX} loss={tl/nb:.4f} concept_val50={vC:.4f} item_val50={vI:.4f} best={best:.4f}{' *SAVED' if improved else ''} ({time.time()-t0:.0f}s)",flush=True)
    print(f"[CHUNK DONE] now {ep0+nrun}/{EP_MAX}; best-concept-val={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
