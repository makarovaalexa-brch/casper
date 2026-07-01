"""
PAPER B — distill + fine-tune an elicitation policy on ANSWERABLE questions (efficient: cached teacher trajectories +
BATCHED training). Controlled comparison: SAME states (from EIG rollout), distill EIG-values vs ORACLE-values vs BLEND;
then optional REINFORCE fine-tune (reward = NDCG). Frozen canonical encoder (enc_unified.pt). Item action space.
GATES: distill-EIG must RECOVER EIG (sanity); does distill-ORACLE or fine-tune BEAT EIG? (matching EIG = justifiable.)
Env: TARGET=eig|oracle|blend ; FT=0|1 (RL finetune) ; REGEN=1 to rebuild trajectory cache.
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; rng=np.random.default_rng(0); torch.manual_seed(0)
NU_TR=int(os.environ.get('NU_TR',1500)); TARGET=os.environ.get('TARGET','eig'); FT=int(os.environ.get('FT',0)); REGEN=int(os.environ.get('REGEN',0)); MAXC=50
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
def eig_vals(ut,cands,asked,rd):
    p=sig(popb[cands]+Ql[cands]@ut); pre=[(j,rd[j]-mu-bi[j]) for j in asked]
    ul=enc_u_batch([pre+[(c,POS)] for c in cands]); ud=enc_u_batch([pre+[(c,NEG)] for c in cands])
    return p*sig(ul@Ql[cands].T).sum(1)+(1-p)*sig(ud@Ql[cands].T).sum(1)
FD=3*D+4
def feats(ut,cands):
    C=len(cands); utt=np.tile(ut,(C,1)); Qc=Q[cands]; bel=(Ql[cands]*ut).sum(1,keepdims=True); pb=popb[cands][:,None]
    meanc=np.tile(Qc.mean(0),(C,1)); cov=sig(popb[cands]+Ql[cands]@ut); covt=np.full((C,1),cov.sum()); nc=np.full((C,1),C/30.)
    return np.concatenate([utt,Qc,bel,pb,meanc,covt,nc],1).astype(np.float32)
_rs=np.random.default_rng(123)
def split(x):
    its=list(dict(rat_by_u[x]))
    if len(its)<6: return None
    il=its[:]; _rs.shuffle(il); return il[:len(il)//2], il[len(il)//2:]
CACHE=f'{base}/.cache/policy_traj.npz'
if REGEN or not os.path.exists(CACHE):
    print(f"generating teacher trajectories (EIG-rollout states; eig+oracle values) on {NU_TR} users...",flush=True)
    F=[];Ev=[];Ov=[];M=[]
    trsel=[x for x in trU if len(rat_by_u[x])>=12][:NU_TR]
    for n_,x in enumerate(trsel):
        sp=split(x)
        if sp is None: continue
        test,prof=sp; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike: continue
        asked=[]
        for t in range(8):
            cands=[j for j in prof if j not in asked][:MAXC]
            if len(cands)<2: break
            ut=enc_u([(j,rd[j]-mu-bi[j]) for j in asked])
            ev=eig_vals(ut,np.array(cands),asked,rd)
            cu=enc_u_batch([[(j,rd[j]-mu-bi[j]) for j in asked]+[(c,rd[c]-mu-bi[c])] for c in cands])
            ov=np.array([(ndcg(cu[li],tlike,set(prof)|set(asked)|{c},False) or 0.) for li,c in enumerate(cands)])
            ff=feats(ut,np.array(cands)); C=len(cands)
            fp=np.zeros((MAXC,FD),np.float32); fp[:C]=ff; evp=np.zeros(MAXC,np.float32); evp[:C]=ev; ovp=np.zeros(MAXC,np.float32); ovp[:C]=ov; mk=np.zeros(MAXC,np.float32); mk[:C]=1
            F.append(fp);Ev.append(evp);Ov.append(ovp);M.append(mk); asked.append(cands[int(ev.argmax())])
        if (n_+1)%400==0: print(f"  {n_+1}/{len(trsel)} ({len(F)} steps)",flush=True)
    F=np.array(F);Ev=np.array(Ev);Ov=np.array(Ov);M=np.array(M); np.savez(CACHE,F=F,Ev=Ev,Ov=Ov,M=M); print(f"  cached {len(F)} steps",flush=True)
else:
    d=np.load(CACHE); F,Ev,Ov,M=d['F'],d['Ev'],d['Ov'],d['M']; print(f"loaded {len(F)} cached steps",flush=True)
Ft=torch.tensor(F); Mt=torch.tensor(M)
def zrow(v,m):  # standardize per row over masked entries
    s=(v*m).sum(1,keepdims=True)/m.sum(1,keepdims=True); v2=(v-s)*m; sd=np.sqrt((v2**2).sum(1,keepdims=True)/m.sum(1,keepdims=True)+1e-6); return v2/sd
if TARGET=='eig': TG=zrow(Ev,M)
elif TARGET=='oracle': TG=zrow(Ov,M)
else: TG=0.5*zrow(Ev,M)+0.5*zrow(Ov,M)
TGt=torch.tensor(TG.astype(np.float32))
class Pol(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(FD,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(s,x): return s.f(x).squeeze(-1)
pol=Pol(); opt=torch.optim.Adam(pol.parameters(),1e-3,weight_decay=1e-5)
print(f"BATCHED distill (TARGET={TARGET}) on {len(F)} steps...",flush=True)
N=len(F); idx=np.arange(N)
for ep in range(60):
    rng.shuffle(idx); tot=0.
    for b0 in range(0,N,512):
        bi_=idx[b0:b0+512]; x=Ft[bi_]; m=Mt[bi_]; tg=TGt[bi_]
        pred=pol(x); loss=(((pred-tg)**2)*m).sum()/m.sum(); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%20==0:
        with torch.no_grad():
            pr=pol(Ft).masked_fill(Mt==0,-1e9); ag_e=(pr.argmax(1).numpy()==(Ev*M+(-1e9)*(1-M)).argmax(1)).mean(); ag_o=(pr.argmax(1).numpy()==(Ov*M+(-1e9)*(1-M)).argmax(1)).mean()
        print(f"  ep{ep+1} MSE={tot/(N/512):.3f} argmax-agree eig={ag_e:.2f} oracle={ag_o:.2f}",flush=True)
pol.eval()
def pol_pick(ut,cands):
    with torch.no_grad(): sc=pol(torch.tensor(feats(ut,np.array(cands)))).numpy()
    return cands[int(sc.argmax())]
# ---------- optional REINFORCE fine-tune (reward = tail NDCG) ----------
if FT:
    print("REINFORCE fine-tune (reward=tail NDCG)...",flush=True)
    optf=torch.optim.Adam(pol.parameters(),3e-4); ftU=[x for x in trU if len(rat_by_u[x])>=12]
    for it in range(int(os.environ.get('FTIT',300))):
        us=list(rng.choice(ftU,size=32,replace=False)); logps=[]; Rs=[]
        for x in us:
            sp=split(x)
            if sp is None: continue
            test,prof=sp; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
            if len(prof)<4 or not tlike or not any(not headmask[t] for t in tlike): continue
            asked=[]; lp=0.
            for t in range(4):
                cands=[j for j in prof if j not in asked][:MAXC]
                if len(cands)<2: break
                ut=enc_u([(j,rd[j]-mu-bi[j]) for j in asked])
                sc=pol(torch.tensor(feats(ut,np.array(cands)))); pr=torch.softmax(sc,0); a=int(torch.multinomial(pr,1)); lp=lp+torch.log(pr[a]+1e-9); asked.append(cands[a])
            r=ndcg(enc_u([(j,rd[j]-mu-bi[j]) for j in asked]),tlike,set(prof),True)
            if r is not None: logps.append(lp); Rs.append(r)
        if not Rs: continue
        Rt=torch.tensor(Rs,dtype=torch.float32); b=Rt.mean(); loss=-((Rt-b)*torch.stack(logps)).mean()
        optf.zero_grad(); loss.backward(); optf.step()
        if (it+1)%100==0: print(f"  ft{it+1} meanR={b.item():.3f}",flush=True)
    pol.eval()
# ---------- eval ----------
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
                elif sel=='policy': c=pol_pick(ut,cands[:MAXC])
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
print(f"\n##### TARGET={TARGET} FT={FT} #####",flush=True)
for tail in [False,True]:
    print(f"=== {'TAIL' if tail else 'FULL'} NDCG@10 ===",flush=True); Rr={}
    for sel in ['random','helf','eig','policy','oracle']:
        Rr[sel]=run(sel,tail); print(f"  {sel:<7}: "+" ".join(f"q{q}={Rr[sel][q]:.3f}" for q in [0,1,2,4,8])+f" | +{Rr[sel][8]-Rr[sel][0]:+.3f}",flush=True)
    e,p=Rr['eig'],Rr['policy']; print(f"  policy vs eig @q4/q8: {p[4]:.3f}/{p[8]:.3f} vs {e[4]:.3f}/{e[8]:.3f} -> {'BEATS' if p[8]>e[8]+.003 else ('MATCHES' if abs(p[8]-e[8])<=.005 else 'BELOW')} eig",flush=True)
