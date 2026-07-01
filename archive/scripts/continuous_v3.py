"""
PAPER B — continuous actor, PARITY-INIT then PATHWISE-FINETUNE (the user's recipe: 'pretrain a continuous model equal
to EIG, then teach it to go further'). Phase P: BC the actor to emit the concept-EIG choice's embedding at each state
(=> actor ~ concept-EIG parity, the GOOD policy, avoids from-scratch degeneracy). Phase F: pathwise-finetune the actor
to maximize reconstruction, with a TRUST REGION (L2 to the frozen parity actor) to prevent degenerate drift; emit RAW
embedding (folded directly by the differentiable user model + frozen encoder). Gate: finetuned actor BEATS concept-EIG
parity on BOTH full+tail. Sweep trust-region lambda. Frozen encoder.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=int(os.environ.get('T',6)); NU_TR=int(os.environ.get('NU_TR',1200)); TR=float(os.environ.get('TR',1.0)); FTEP=int(os.environ.get('FTEP',10)); rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids); keepids=set(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=8]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}; ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=-1.0
t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>0.5: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=30]; Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32); citems=[set(tagitems[t]) for t in ctags]
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems[t]: item2c[j].append(ki)
print(f"  {len(ctags)} concepts ({time.time()-t0:.0f}s)",flush=True)
Qt=torch.tensor(Q); Qlt=torch.tensor(Ql); popbt=torch.tensor(popb); ipswt=torch.tensor(ipsw)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
def sig(z): return 1/(1+np.exp(-z))
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def uconcepts(x,prof):
    cn={}
    for j in prof:
        for ki in item2c[j]: cn[ki]=cn.get(ki,[])+[j]
    return [(ki, Ac[ki], float(np.mean([resid[x][j] for j in js]))) for ki,js in cn.items() if len(js)>=2]
# ---- Phase P data: concept-EIG rollouts (state u_t -> chosen concept centroid) ----
CACHE=f'{base}/.cache/parity_traj.npz'
if not os.path.exists(CACHE) or os.environ.get('REGEN'):
    print("gen concept-EIG parity trajectories...",flush=True); S=[];Targ=[];TT=[]
    sample=[x for x in trU if len(rat_by_u[x])>=12][:NU_TR]
    for n_,x in enumerate(sample):
        prof=[j for j,_ in rat_by_u[x]]; cc=uconcepts(x,prof)
        if len(cc)<2: continue
        rest=np.array(prof); toks=[]; used=set()
        for t in range(T):
            ci=[i for i in range(len(cc)) if i not in used]
            if not ci: break
            fac=np.array([cc[i][1] for i in ci]); ul=enc_u_batch([toks+[(cc[i][1],POS)] for i in ci]); ud=enc_u_batch([toks+[(cc[i][1],NEG)] for i in ci])
            val=0.5*sig(popb[rest]+ul@Ql[rest].T).sum(1)+0.5*sig(popb[rest]+ud@Ql[rest].T).sum(1); bi_=ci[int(val.argmax())]
            S.append(enc_u_np(toks).astype(np.float32)); Targ.append(cc[bi_][1].astype(np.float32)); TT.append(t)
            used.add(bi_); toks.append((cc[bi_][1],cc[bi_][2]))
        if (n_+1)%300==0: print(f"  {n_+1}/{len(sample)} ({len(S)})",flush=True)
    S=np.array(S);Targ=np.array(Targ);TT=np.array(TT); np.savez(CACHE,S=S,Targ=Targ,TT=TT); print(f"  {len(S)} parity pairs",flush=True)
else:
    d=np.load(CACHE); S,Targ,TT=d['S'],d['Targ'],d['TT']; print(f"  loaded {len(S)} parity pairs",flush=True)
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+2,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,D))
    def forward(s,u,t): return s.f(torch.cat([u,t],1))
actor=Actor(); opt=torch.optim.Adam(actor.parameters(),1e-3)
St=torch.tensor(S); Tgt=torch.tensor(Targ); TTt=torch.stack([torch.tensor(TT/8.,dtype=torch.float32),torch.tensor(TT,dtype=torch.float32)],1)
print("Phase P: BC actor -> concept-EIG choice...",flush=True)
for ep in range(40):
    idx=torch.randperm(len(St))
    for b0 in range(0,len(St),512):
        bb=idx[b0:b0+512]; pred=actor(St[bb],TTt[bb]); loss=((pred-Tgt[bb])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
import copy; parity=copy.deepcopy(actor)
for p in parity.parameters(): p.requires_grad_(False)
# ---- Phase F: pathwise finetune with trust region to parity ----
def prep(users):
    KA=12; Fa=torch.zeros(len(users),KA,D); Ya=torch.zeros(len(users),KA); Ma=torch.zeros(len(users),KA); tgt=torch.zeros(len(users),ni); wt=torch.zeros(len(users),ni)
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); ask=allit[:KA]
        for q,j in enumerate(ask): Fa[b,q]=Qt[j]; Ya[b,q]=resid[x][j]; Ma[b,q]=1
        lk=likes_by_u[x]; tg=[j for j in lk if j not in set(ask)] or lk[:1]; posw=0.
        for j in tg: tgt[b,j]=1.; wt[b,j]=float(ipsw[j]); posw+=float(ipsw[j])
        seen=set(ask); nneg=ni-len(seen)-len(tg); m=torch.ones(ni); m[list(seen)]=0; wtb=wt[b]; wtb[(tgt[b]==0)&(m>0)]=float(posw)/max(nneg,1); wt[b]=wtb
    return Fa,Ya,Ma,tgt,wt
def rollout(net,Fa,Ya,Ma,Tn,explore=0.05,trust=False,netp=None):
    B=Fa.shape[0]; toks=torch.zeros(B,Tn,D+1); tmask=torch.zeros(B,Tn); u=torch.zeros(B,D); Fan=Fa/(Fa.norm(dim=2,keepdim=True)+1e-9); treg=0.
    for t in range(Tn):
        tt=torch.cat([torch.full((B,1),t/8.),torch.full((B,1),float(t))],1); a=net(u,tt)
        if trust: treg=treg+((a-netp(u,tt))**2).mean()
        if explore>0: a=a+explore*torch.randn_like(a)
        an=a/(a.norm(dim=1,keepdim=True)+1e-9); sims=torch.relu((Fan*an.unsqueeze(1)).sum(2))*Ma; ans=(sims*Ya).sum(1)/(sims.sum(1)+1e-6)
        toks=toks.clone(); toks[:,t,:D]=a; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask)
    return u,treg/Tn
def recon(u,tgt,wt): sc=u@Qlt.t()+popbt; bce=nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none'); return (wt*bce).sum()/wt.sum()
optf=torch.optim.Adam(actor.parameters(),1e-4); trbig=[x for x in trU if len(likes_by_u[x])>=6 and len(rat_by_u[x])>=14]
print(f"Phase F: pathwise finetune (trust-region lambda={TR})...",flush=True)
for ep in range(FTEP):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),128):
        Fa,Ya,Ma,tgt,wt=prep(trbig[b0:b0+128]); u,treg=rollout(actor,Fa,Ya,Ma,T,trust=True,netp=parity); loss=recon(u,tgt,wt)+TR*treg
        optf.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.0); optf.step(); tot+=loss.item();nb+=1
    print(f"  ft{ep+1} loss={tot/nb:.4f}",flush=True)
actor.eval()
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
    if len(its)>=8: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:200]
def emit_np(net,u,t):
    with torch.no_grad(): return net(torch.tensor(u[None],dtype=torch.float32),torch.tensor([[t/8.,float(t)]])).numpy()[0]
def ans_of(a,fac,res): an=a/(np.linalg.norm(a)+1e-9); Fn=fac/(np.linalg.norm(fac,axis=1,keepdims=True)+1e-9); s=np.maximum(Fn@an,0); return float((s*res).sum()/(s.sum()+1e-6))
def run(mode,tail):
    M={q:0. for q in [0,1,2,4,8]};Rc={q:0. for q in [0,1,2,4,8]};m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        fac=np.array([Q[j] for j in prof]); res=np.array([rd[j]-mu-bi[j] for j in prof]); toks=[]; asked=[]; cc=uconcepts(x,prof)
        for q in [0,1,2,4,8]:
            while len(toks)<q:
                if mode in('actorP','actorF'):
                    net=parity if mode=='actorP' else actor; a=emit_np(net,enc_u_np(toks),len(toks)); toks.append((a,ans_of(a,fac,res)))
                elif mode=='conceptEIG':
                    ci=[i for i in range(len(cc)) if i not in asked]
                    if not ci: break
                    ul=enc_u_batch([toks+[(cc[i][1],POS)] for i in ci]); ud=enc_u_batch([toks+[(cc[i][1],NEG)] for i in ci]); val=0.5*sig(popb[np.array(prof)]+ul@Ql[prof].T).sum(1)+0.5*sig(popb[np.array(prof)]+ud@Ql[prof].T).sum(1); pk=ci[int(val.argmax())]; asked.append(pk); toks.append((cc[pk][1],cc[pk][2]))
                elif mode=='item':
                    cs=[j for j in prof if j not in asked]; ul=enc_u_batch([toks+[(Q[j],rd[j]-mu-bi[j])] for j in cs]); val=sig(popb[np.array(prof)]+ul@Ql[prof].T).sum(1); pk=cs[int(val.argmax())]; asked.append(pk); toks.append((Q[pk],rd[pk]-mu-bi[pk]))
                else: j=prof[int(rng.integers(len(prof)))]; toks.append((Q[j],rd[j]-mu-bi[j]))
            mt=metr(enc_u_np(toks),tlike,set(prof),tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},m
for tail in [False,True]:
    print(f"\n=== {'TAIL' if tail else 'FULL'}: NDCG@10 / Recall@50 (parity-init + pathwise finetune, TR={TR}) ===",flush=True)
    for mode in ['random','item','conceptEIG','actorP','actorF']:
        M,Rc,m=run(mode,tail); print(f"  {mode:<10}: NDCG "+" ".join(f"{M[q]:.3f}" for q in [0,1,2,4,8])+" | Rec "+" ".join(f"{Rc[q]:.3f}" for q in [0,1,2,4,8]),flush=True)
    print(f"  (actorP=parity~=concept-EIG; actorF=pathwise-finetuned; GATE: actorF>conceptEIG on full+tail)",flush=True)
