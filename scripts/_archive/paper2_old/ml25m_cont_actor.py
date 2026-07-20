"""
ML-25M PHASE-2 STEP 2 -- the money experiment: D1-style CONTINUOUS actor (differentiable unroll, graded
geometric answers, divisiveness field) vs the strongest static GRADED DISCRETE selector (entropy over
concepts) vs a binary control. Faithful port of the D1 recipe in continuous_actor.py to ml25m artifacts.
Frozen instrument = enc_v1_ml25m (canonical). From-scratch actor (D1 used NOBC=1; warm-start N/A cross-dataset).

Recipe (D1): actor MLP [belief u(64), turn]->q(64); cont rollout T=8: qn=q/|q|, fe=qn*_CN (_CN=mean|Ac|),
graded geometric answer a=NEG+(POS-NEG)*(cos(u*,fe)+1)/2, fold (fe,a) into frozen enc -> belief; accumulate
divisiveness field. Objective OBJ=ustar: (1-cos(u,u*)) + SNDCG*softNDCG(held likes vs popular) - DIVW*div.
u* = profile fold (simulated taste). Ruler: NDCG@50 primary + @10, full + Cremonesi head-33% tail.
Val-select on the 500 va cohort (tail@50); TEST = 500 te cohort, seed-avg profile splits.

Env: EP(14) TRSEED(0) SNDCG(0.3) DIVW(1.0) DTAU(2.0) NTRAIN(4000) SELVAL(tail) MODE(train|eval)
     TAG(d1_ml25m) EVALSEEDS(1,2,3,7,11) LOADCK(path) DISC(0/1 run discrete baselines) GRADED(1)
"""
import os, sys, time, numpy as np, torch, torch.nn as nn
t0=time.time(); out='C:/dev/phd/casper/data/movielens/.cache/ml25m'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0; T=8
EP=int(os.environ.get('EP',14)); TRS=int(os.environ.get('TRSEED',0)); rng=np.random.default_rng(TRS); torch.manual_seed(TRS)
SNDCG=float(os.environ.get('SNDCG',0.3)); DIVW=float(os.environ.get('DIVW',1.0)); DTAU=float(os.environ.get('DTAU',2.0))
NTRAIN=int(os.environ.get('NTRAIN',4000)); SELVAL=os.environ.get('SELVAL','tail'); TAG=os.environ.get('TAG','d1_ml25m')
HID=128; MODE=os.environ.get('MODE','train'); GRADED=bool(int(os.environ.get('GRADED','1')))
M=np.load(f'{out}/meta.npz'); uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; mu=float(M['mu'])
ni=int(M['ni']); trU=M['trU']; va=M['va']; te=M['te']
Q=np.load(f'{out}/Q_svd.npy').astype(np.float32); bi=np.load(f'{out}/bi_svd.npy').astype(np.float32)
Ql=Q; Qlt=torch.tensor(Q); popb=np.log(cnt+1.0).astype(np.float32); popbt=torch.tensor(popb)
Ac=np.load(f'{out}/Ac_concept.npy').astype(np.float32); nc=Ac.shape[0]
mem=np.load(f'{out}/membership.npz'); cf_flat=mem['citems_flat']; cf_off=mem['citems_off']; cdiv=mem['cdiv']
CONC_ENT_ORD=list(np.argsort(-cdiv))
item2c=[[] for _ in range(ni)]
for c in range(nc):
    for j in cf_flat[cf_off[c]:cf_off[c+1]]: item2c[int(j)].append(c)
item2c=[np.array(v,np.int32) for v in item2c]
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/max(cnt.sum(),1)
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
_CN=float(np.linalg.norm(Ac,axis=1).mean())
# ---- ratings by user (train subset + full va/te) ----
need=set(int(x) for x in va.tolist()+te.tolist())
_tr_pool=[int(x) for x in trU]; rng.shuffle(_tr_pool)
rat_by_u={}
# build via single pass over cohort rows + sampled train rows
def build_rat(users):
    us=set(int(x) for x in users); m=np.isin(uu,np.array(sorted(us))); a_uu=uu[m]; a_ii=ii[m]; a_rr=rr[m]
    d={}
    for k in range(len(a_uu)): d.setdefault(int(a_uu[k]),[]).append((int(a_ii[k]),float(a_rr[k])))
    return d
rat_cohort=build_rat(list(need))
# train users: sample NTRAIN with >=14 rated & >=6 likes
cand=_tr_pool[:NTRAIN*3]
rat_tr=build_rat(cand)
trbig=[x for x in cand if x in rat_tr and len(rat_tr[x])>=14 and sum(1 for _,r in rat_tr[x] if r>=LIKE)>=6][:NTRAIN]
rat_by_u.update(rat_cohort); rat_by_u.update({x:rat_tr[x] for x in trbig})
print(f"cohort {len(rat_cohort)} + train {len(trbig)} users ({time.time()-t0:.0f}s)",flush=True)
def resid_of(x): rd=dict(rat_by_u[x]); return rd,{j:(r-mu-bi[j]) for j,r in rat_by_u[x]}
_res={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in rat_by_u}
# POS/NEG from train sample
_pos=[]; _neg=[]
for x in trbig[:1500]:
    for j,r in rat_by_u[x]:
        (_pos if r>=LIKE else _neg).append(_res[x][j])
POS=float(np.mean(_pos)); NEG=float(np.mean(_neg)); print(f"POS={POS:.3f} NEG={NEG:.3f} _CN={_CN:.3f}",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{out}/enc_v1_ml25m.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
def enc_u_np(rows):
    if not rows: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(rows),D+1),np.float32); mk=np.ones((1,len(rows)),np.float32)
    for q,(f,r) in enumerate(rows): tk[0,q,:D]=f; tk[0,q,D]=r
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,HID),nn.ReLU(),nn.Linear(HID,HID),nn.ReLU(),nn.Linear(HID,D))
    def forward(s,u,tt):
        B=u.shape[0]; tn=(tt.view(B,1) if torch.is_tensor(tt) else torch.full((B,1),float(tt))); return s.f(torch.cat([u,tn],1))
actor=Actor(); opt=torch.optim.Adam(actor.parameters(),1e-3)
def save_ck(p): torch.save({'actor':actor.state_dict()},p)
def load_ck(p):
    sd=torch.load(p); actor.load_state_dict(sd['actor'] if isinstance(sd,dict) and 'actor' in sd else sd)
# ---- divisiveness field ----
_uu=[x for x in trbig if len(rat_by_u[x])>=8][:2500]
UMATt=torch.tensor(np.stack([enc_u_np([(Q[j],_res[x][j]) for j,_ in rat_by_u[x]]) for x in _uu]).astype(np.float32))
UMATt=UMATt/(UMATt.norm(dim=1,keepdim=True)+1e-9)
def field_div(q):
    p=torch.sigmoid((q@UMATt.t())/DTAU).mean(-1).clamp(1e-4,1-1e-4); return -(p*torch.log2(p)+(1-p)*torch.log2(1-p))
print(f"div field over {UMATt.shape[0]} tastes ({time.time()-t0:.0f}s)",flush=True)
_POPNEG=torch.tensor(np.argsort(-popb)[:200].copy()); ML=64
def prep(users):
    B=len(users); USTAR=torch.zeros(B,D); tgt=torch.zeros(B,ni); wt=torch.zeros(B,ni)
    LIKED=torch.zeros(B,ML,dtype=torch.long); LMASK=torch.zeros(B,ML); PROFM=torch.zeros(B,ni)
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=allit[:max(len(allit)//2,4)]
        likes=[j for j,r in rat_by_u[x] if r>=LIKE]; held=[j for j in likes if j not in set(prof)] or likes[:1]
        for j in prof: PROFM[b,j]=1.
        USTAR[b]=torch.tensor(enc_u_np([(Q[j],_res[x][j]) for j in prof]))
        posw=0.
        for j in held: tgt[b,j]=1.; wt[b,j]=float(ipsw[j]); posw+=float(ipsw[j])
        for hi,j in enumerate(held[:ML]): LIKED[b,hi]=j; LMASK[b,hi]=1.
        nneg=ni-len(prof)-len(held); m=torch.ones(ni); m[prof]=0; wb=wt[b]; wb[(tgt[b]==0)&(m>0)]=float(posw)/max(nneg,1); wt[b]=wb
    return USTAR,tgt,wt,LIKED,LMASK,PROFM
_DIVH=[torch.zeros(1)]
def rollout(users,USTAR):
    B=len(users); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D); _DIVH[0]=torch.zeros(B)
    for t in range(T):
        q=actor(u,t/8.); qn=q/(q.norm(dim=1,keepdim=True)+1e-9); fe=qn*_CN
        _DIVH[0]=_DIVH[0]+field_div(qn)
        cf=(fe*USTAR).sum(1)/(fe.norm(dim=1)*USTAR.norm(dim=1)+1e-9)
        ans=(NEG+(POS-NEG)*(cf+1)/2) if GRADED else torch.sign(cf)
        toks=toks.clone(); toks[:,t,:D]=fe; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask)
    return u
def softndcg(u,LIKED,LMASK,tgt,PROFM,tau=1.0):
    B=u.shape[0]; sL=(u.unsqueeze(1)*Qlt[LIKED]).sum(2)+popbt[LIKED]; sP=u@Qlt[_POPNEG].t()+popbt[_POPNEG]
    S=torch.cat([sL,sP],1); rel=torch.cat([LMASK,tgt[:,_POPNEG]],1); valid=torch.cat([LMASK,1.-PROFM[:,_POPNEG]],1)
    d=(S.unsqueeze(1)-S.unsqueeze(2))/tau; rank=1.+(torch.sigmoid(d)*valid.unsqueeze(2)).sum(1)-0.5*valid
    dcg=((rel/torch.log2(1.+rank))*valid).sum(1); npos=rel.sum(1).clamp(min=1.)
    _it=1./torch.log2(torch.arange(2,2+rel.shape[1]).float()); _ic=torch.cumsum(_it,0)
    idcg=_ic[(npos.long()-1).clamp(0,rel.shape[1]-1)]; return -(dcg/(idcg+1e-9)).mean()
# ---- eval helpers ----
Ks=[10,50]; _Wk={K:1./np.log2(np.arange(2,K+2)) for K in Ks}
def ndcg(u,rel,excl,K,tail):
    sc=(popb+Q@u).astype(np.float64).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in rel if not headmask[t]]
    if not rel: return None
    W=_Wk[K]; top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]; rs=set(rel)
    dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
def user_conc(prof, rd):
    acc={}
    for j in prof:
        for c in item2c[j]: acc.setdefault(int(c),[]).append(rd[j]-mu-bi[j])
    return {c for c,v in acc.items() if len(v)>=2}
def eval_policy(cohort, seeds, qsnap=(8,), method='cont', graded=True):
    # method: cont (actor), disc (entropy concepts geometric answer)
    resF={q:[] for q in qsnap}; resT={q:[] for q in qsnap}
    for sdv in seeds:
        r=np.random.default_rng(sdv); accF={q:0. for q in qsnap}; accT={q:0. for q in qsnap}; m=0; mt=0
        for x in cohort:
            if x not in rat_by_u: continue
            its=[j for j,_ in rat_by_u[x]]
            if len(its)<6: continue
            il=its[:]; r.shuffle(il); prof=set(il[:len(il)//2]); vtest=il[len(il)//2:]
            rd=dict(rat_by_u[x]); tlike=set(j for j in vtest if rd[j]>=LIKE)
            if not tlike: continue
            rel=list(tlike); rel_t=[t for t in rel if not headmask[t]]; ht=1 if rel_t else 0; m+=1; mt+=ht
            us=enc_u_np([(Q[j],_res[x][j]) for j in prof]); usn=us/(np.linalg.norm(us)+1e-9)
            toks=[]; snaps={}
            if method=='cont':
                u=np.zeros(D,np.float32)
                for t in range(max(qsnap)+1):
                    if t in qsnap: snaps[t]=enc_u_np(toks)
                    if t==max(qsnap): break
                    with torch.no_grad(): qv=actor(torch.tensor(u[None],dtype=torch.float32),t/8.).numpy()[0]
                    qn=qv/(np.linalg.norm(qv)+1e-9); fe=(qn*_CN).astype(np.float32); cf=float(usn@qn)
                    a=float(NEG+(POS-NEG)*(cf+1)/2) if graded else (POS if cf>0 else NEG)
                    toks.append((fe,a)); u=enc_u_np(toks)
            else:  # disc: entropy concepts, answerable only, geometric graded/binary answer; fixed order -> fold prefixes
                cans=user_conc(prof,rd); cs=[c for c in CONC_ENT_ORD if c in cans][:max(qsnap)]
                seq=[]
                for c in cs:
                    acn=Ac[c]/(np.linalg.norm(Ac[c])+1e-9); cf=float(usn@acn)
                    a=float(NEG+(POS-NEG)*(cf+1)/2) if graded else (POS if cf>0 else NEG)
                    seq.append((Ac[c].astype(np.float32),a))
                for q in qsnap: snaps[q]=enc_u_np(seq[:q])   # wasted turns beyond answerable concepts => fewer tokens
            for q in qsnap:
                u=snaps[q]; excl=set(prof)
                v=ndcg(u,rel,excl,50,False); accF[q]+=v if v else 0
                if ht: vt=ndcg(u,rel,excl,50,True); accT[q]+=vt if vt else 0
        for q in qsnap: resF[q].append(accF[q]/max(m,1)); resT[q].append(accT[q]/max(mt,1))
    return resF,resT,m,mt
def val_metric():
    rF,rT,m,mt=eval_policy(list(va),[123],qsnap=(8,),method='cont',graded=GRADED)
    return rF[8][0], rT[8][0]
if os.environ.get('LOADCK'): load_ck(os.environ['LOADCK']); print(f"LOADED {os.environ['LOADCK']}",flush=True); MODE='eval'
elif MODE=='train':
    _CKB=f'{out}/policy_{TAG}_best.pt'; _ST=f'{out}/policy_{TAG}_state.pt'; _PK=f'{out}/policy_{TAG}_peak.txt'
    best=-1; bestep=0; ep0=0; EP_CHUNK=int(os.environ.get('EP_CHUNK',EP))
    if os.path.exists(_ST):
        ck=torch.load(_ST); actor.load_state_dict(ck['actor']); opt.load_state_dict(ck['opt'])
        ep0=ck['epoch']; best=ck['best']; bestep=ck['bestep']; print(f"[RESUME] ep{ep0} best={best:.4f}@ep{bestep}",flush=True)
    nrun=min(EP_CHUNK,EP-ep0)
    for ep in range(ep0,ep0+nrun):
        rng.shuffle(trbig); tot=0.; nb=0
        for b0 in range(0,len(trbig),96):
            us=trbig[b0:b0+96]; USTAR,tgt,wt,LIKED,LMASK,PROFM=prep(us)
            u=rollout(us,USTAR)
            loss=(1-torch.nn.functional.cosine_similarity(u,USTAR,dim=1)).mean()
            if SNDCG>0: loss=loss+SNDCG*softndcg(u,LIKED,LMASK,tgt,PROFM)
            if DIVW>0: loss=loss-DIVW*(_DIVH[0]/T).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.0); opt.step()
            tot+=float(loss); nb+=1
        vf,vt=val_metric(); sel=vt if SELVAL=='tail' else vf
        star=''
        if sel>best: best=sel; bestep=ep+1; save_ck(_CKB); star=' *BEST'
        save_ck(f'{out}/policy_{TAG}_last.pt'); save_ck(f'{out}/policy_{TAG}_ep{ep+1}.pt')
        torch.save({'actor':actor.state_dict(),'opt':opt.state_dict(),'epoch':ep+1,'best':best,'bestep':bestep},_ST)
        with open(_PK,'a') as f: f.write(f"ep{ep+1}/{EP} loss={tot/nb:.4f} VAL@50 full={vf:.4f} tail={vt:.4f} best={best:.4f}@ep{bestep}{star}\n")
        print(f"ep{ep+1}/{EP} loss={tot/nb:.4f} VAL@50 full={vf:.4f} tail={vt:.4f} best={best:.4f}@ep{bestep}{star} ({time.time()-t0:.0f}s)",flush=True)
    if ep0+nrun<EP and not os.environ.get('FINAL'):
        print(f"[CHUNK DONE] {ep0+nrun}/{EP}; rerun to continue",flush=True); sys.exit(0)
    load_ck(_CKB); print(f"loaded BEST @ep{bestep} for eval",flush=True)
# ---- TEST eval (seed-avg) ----
seeds=[int(s) for s in os.environ.get('EVALSEEDS','1,2,3,7,11').split(',')]
print(f"\n=== TEST eval (te cohort, seed-avg {seeds}) ===",flush=True)
rF,rT,m,mt=eval_policy(list(te),seeds,qsnap=(0,2,4,6,8),method='cont',graded=True)
print(f"[cont-graded actor] n={m} n_tail={mt}",flush=True)
for q in (0,2,4,6,8): print(f"  q{q}: @50 full={np.mean(rF[q]):.4f}+/-{np.std(rF[q]):.4f} tail={np.mean(rT[q]):.4f}+/-{np.std(rT[q]):.4f}",flush=True)
np.save(f'{out}/qcurve_{TAG}_cont.npy',np.array([[np.mean(rF[q]),np.mean(rT[q])] for q in (0,2,4,6,8)]))
if os.environ.get('DISC'):
    dF,dT,_,_=eval_policy(list(te),seeds,qsnap=(8,),method='disc',graded=True)
    print(f"[entropy-concept GRADED discrete] q8: @50 full={np.mean(dF[8]):.4f}+/-{np.std(dF[8]):.4f} tail={np.mean(dT[8]):.4f}+/-{np.std(dT[8]):.4f}",flush=True)
    bF,bT,_,_=eval_policy(list(te),seeds,qsnap=(8,),method='disc',graded=False)
    print(f"[entropy-concept BINARY discrete]  q8: @50 full={np.mean(bF[8]):.4f}+/-{np.std(bF[8]):.4f} tail={np.mean(bT[8]):.4f}+/-{np.std(bT[8]):.4f}",flush=True)
    cbF,cbT,_,_=eval_policy(list(te),seeds,qsnap=(8,),method='cont',graded=False)
    print(f"[cont BINARY actor]                q8: @50 full={np.mean(cbF[8]):.4f}+/-{np.std(cbF[8]):.4f} tail={np.mean(cbT[8]):.4f}+/-{np.std(cbT[8]):.4f}",flush=True)
    print(f"\n>>> cont-graded - entropy-concept-graded @q8: full={np.mean(rF[8])-np.mean(dF[8]):+.4f} tail={np.mean(rT[8])-np.mean(dT[8]):+.4f}",flush=True)
print(f"DONE ({time.time()-t0:.0f}s)",flush=True)
