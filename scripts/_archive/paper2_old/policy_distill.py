"""
PAPER B — B1 (corrected): distill a teacher's SELECTION into a policy, with an ADEQUATE distiller.
Fixes the earlier broken version: (1) features now include SET CONTEXT (mean candidate embedding + current coverage),
because EIG(j) is a function of the candidate SET, not just (u_t, Q_j); (2) DENSE supervision — regress the teacher's
per-candidate target VALUES (standardized per step), not weak argmax cross-entropy.
SANITY GATE: TEACHER=eig must RECOVER EIG (policy NDCG ~= EIG). Only if EIG is recovered is a TEACHER=oracle failure
evidence that the oracle is privileged (vs a machinery artefact).
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; rng=np.random.default_rng(0); torch.manual_seed(0)
NU_TR=int(os.environ.get('NU_TR',1200)); TEACHER=os.environ.get('TEACHER','eig'); EP=int(os.environ.get('EP',40))
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); entv=-(pm*np.log(pm+1e-12)).sum(1); lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=entv/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(j,res) in enumerate(rev): arr[b,q,:D]=Q[j]; arr[b,q,D]=res; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def enc_u(rev): return enc_u_batch([rev])[0] if rev else np.zeros(D)
def sig(z): return 1/(1+np.exp(-z))
_W=1./np.log2(np.arange(2,12))
def ndcg(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in tlike if not headmask[t]]
    else: rel=list(tlike)
    if not rel: return None
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]; rs=set(rel)
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rs)/(_W[:min(10,len(rel))].sum()+1e-12)
def eig_vals(ut,cands,asked,rd):                       # EIG expected-coverage value per candidate (the teacher signal)
    p=sig(popb[cands]+Ql[cands]@ut)
    pre=[(j,rd[j]-mu-bi[j]) for j in asked]
    ul=enc_u_batch([pre+[(c,POS)] for c in cands]); ud=enc_u_batch([pre+[(c,NEG)] for c in cands])
    return p*sig(ul@Ql[cands].T).sum(1)+(1-p)*sig(ud@Ql[cands].T).sum(1)
def feats(ut,cands):                                   # (C, 3D+3) with SET CONTEXT
    C=len(cands); utt=np.tile(ut,(C,1)); Qc=Q[cands]; bel=(Ql[cands]*ut).sum(1,keepdims=True); pb=popb[cands][:,None]
    meanc=np.tile(Qc.mean(0),(C,1)); cov=sig(popb[cands]+Ql[cands]@ut); covtot=np.full((C,1),cov.sum()); nc=np.full((C,1),C/30.)
    return np.concatenate([utt,Qc,bel,pb,meanc,covtot,nc],1).astype(np.float32)
FD=3*D+4
# ---------- teacher trajectories with DENSE per-candidate target values ----------
_rs=np.random.default_rng(123)
def split(x):
    its=list(dict(rat_by_u[x]))
    if len(its)<6: return None
    il=its[:]; _rs.shuffle(il); return il[:len(il)//2], il[len(il)//2:]
print(f"generating {TEACHER} trajectories (dense values) on {NU_TR} users...",flush=True)
STEPS=[]
trsel=[x for x in trU if len(rat_by_u[x])>=12][:NU_TR]
for n_,x in enumerate(trsel):
    sp=split(x)
    if sp is None: continue
    test,prof=sp; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
    if len(prof)<4 or not tlike: continue
    asked=[]
    for t in range(8):
        cands=[j for j in prof if j not in asked]
        if not cands: break
        ut=enc_u([(j,rd[j]-mu-bi[j]) for j in asked])
        if TEACHER=='eig': vals=eig_vals(ut,np.array(cands),asked,rd)
        else:
            cu=enc_u_batch([[(j,rd[j]-mu-bi[j]) for j in asked]+[(c,rd[c]-mu-bi[c])] for c in cands])
            vals=np.array([ (ndcg(cu[li],tlike,set(prof)|set(asked)|{c},False) or 0.) for li,c in enumerate(cands)])
        STEPS.append((feats(ut,np.array(cands)), vals.astype(np.float32))); asked.append(cands[int(vals.argmax())])
    if (n_+1)%300==0: print(f"  {n_+1}/{len(trsel)} ({len(STEPS)} steps)",flush=True)
print(f"  {len(STEPS)} steps",flush=True)
# ---------- policy: regress standardized teacher values ----------
class Pol(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(FD,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(s,x): return s.f(x).squeeze(-1)
pol=Pol(); opt=torch.optim.Adam(pol.parameters(),1e-3,weight_decay=1e-5)
def zz(v): return (v-v.mean())/(v.std()+1e-6)
print("training policy (dense value regression)...",flush=True)
order=np.arange(len(STEPS))
for ep in range(EP):
    rng.shuffle(order); tot=0.
    for i in order:
        cf,vals=STEPS[i]
        pred=pol.f(torch.tensor(cf)).squeeze(-1); tgt=torch.tensor(zz(vals))
        loss=((pred-tgt)**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%10==0:
        # rank-agreement with teacher argmax on a sample
        agree=np.mean([int(pol.f(torch.tensor(STEPS[i][0])).squeeze(-1).argmax().item()==STEPS[i][1].argmax()) for i in order[:500]])
        print(f"  ep{ep+1} MSE={tot/len(STEPS):.3f} argmax-agree={agree:.2f}",flush=True)
pol.eval()
def pol_pick(ut,cands):
    with torch.no_grad(): sc=pol.f(torch.tensor(feats(ut,np.array(cands)))).squeeze(-1).numpy()
    return cands[int(sc.argmax())]
SPL={}
for x in te:
    sp=split(x)
    if sp is not None: SPL[x]=sp
TE=[x for x in te if x in SPL][:250]
def run(sel,tail):
    acc={q:0. for q in [0,1,2,4,8]}; m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        for q in [0,1,2,4,8]:
            asked=[]
            while len(asked)<q:
                cands=[j for j in prof if j not in asked]
                if not cands: break
                ut=enc_u([(j,rd[j]-mu-bi[j]) for j in asked])
                if sel=='random': c=cands[int(rng.integers(len(cands)))]
                elif sel=='helf': c=max(cands,key=lambda j:helf[j])
                elif sel=='policy': c=pol_pick(ut,cands)
                elif sel=='eig': c=cands[int(eig_vals(ut,np.array(cands),asked,rd).argmax())]
                else:
                    cu=enc_u_batch([[(j,rd[j]-mu-bi[j]) for j in asked]+[(c2,rd[c2]-mu-bi[c2])] for c2 in cands]); best=None
                    for li,c2 in enumerate(cands):
                        v=ndcg(cu[li],tlike,set(prof)|set(asked)|{c2},tail)
                        if v is not None and (best is None or v>best[0]): best=(v,c2)
                    c=best[1] if best else cands[0]
                asked.append(c)
            v=ndcg(enc_u([(j,rd[j]-mu-bi[j]) for j in asked]),tlike,set(prof),tail)
            if v is not None: acc[q]+=v
        m+=1
    return {q:acc[q]/m for q in acc}
for tail in [False,True]:
    print(f"\n=== {'TAIL' if tail else 'FULL'} NDCG@10 — distill {TEACHER} (corrected distiller) ===",flush=True)
    Rr={}
    for sel in ['random','helf','eig','policy','oracle']:
        Rr[sel]=run(sel,tail); print(f"  {sel:<7}: "+" ".join(f"q{q}={Rr[sel][q]:.3f}" for q in [0,1,2,4,8])+f" | +{Rr[sel][8]-Rr[sel][0]:+.3f}",flush=True)
    if TEACHER=='eig':
        print(f"  SANITY policy vs eig @q8: {Rr['policy'][8]:.3f} vs {Rr['eig'][8]:.3f} (recovered if ~equal)",flush=True)
    else:
        print(f"  policy vs eig @q8: {Rr['policy'][8]:.3f} vs {Rr['eig'][8]:.3f} (policy>eig => realizable headroom above greedy)",flush=True)
