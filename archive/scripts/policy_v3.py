"""
PAPER B — BEST-SHOT SUPERVISED distillation: set-transformer (attention OVER candidates) instead of a pointwise MLP,
so the policy can represent the set-dependent EIG/oracle value function. Soft-listwise distillation (KL to softmax of
standardized teacher values = dense). Reuses cached trajectories (policy_traj.npz). Frozen encoder. Item action space.
Env: TARGET=eig|oracle|blend ; EP epochs ; H hidden ; LYR attention layers.
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; rng=np.random.default_rng(0); torch.manual_seed(0)
TARGET=os.environ.get('TARGET','oracle'); EP=int(os.environ.get('EP',150)); H=int(os.environ.get('H',256)); LYR=int(os.environ.get('LYR',3)); MAXC=50; TEMP=float(os.environ.get('STEMP',1.0))
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
IN=2*D+2
def feat2(ut,cands):                                   # per-candidate base features [u_t, Q_j, belief, popb] (set-attn adds context)
    C=len(cands); return np.concatenate([np.tile(ut,(C,1)),Q[cands],(Ql[cands]*ut).sum(1,keepdims=True),popb[cands][:,None]],1).astype(np.float32)
d=np.load(f'{base}/.cache/policy_traj.npz'); F,Ev,Ov,M=d['F'],d['Ev'],d['Ov'],d['M']; Xin=F[:,:,:IN].copy(); print(f"loaded {len(F)} steps",flush=True)
class SetPol(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Linear(IN,H)
        s.layers=nn.ModuleList([nn.TransformerEncoderLayer(H,8,H*2,batch_first=True,dropout=0.0) for _ in range(LYR)])
        s.out=nn.Sequential(nn.Linear(H,H),nn.ReLU(),nn.Linear(H,1))
    def forward(s,x,mask):     # x (B,C,IN), mask (B,C) 1=valid
        h=s.emb(x); kpm=(mask==0)
        for L in s.layers: h=L(h,src_key_padding_mask=kpm)
        return s.out(h).squeeze(-1)
pol=SetPol(); opt=torch.optim.Adam(pol.parameters(),5e-4,weight_decay=1e-5)
Xt=torch.tensor(Xin); Mt=torch.tensor(M)
def zrow(v,m): s=(v*m).sum(1,keepdims=True)/m.sum(1,keepdims=True); v2=(v-s)*m; sd=np.sqrt((v2**2).sum(1,keepdims=True)/m.sum(1,keepdims=True)+1e-6); return v2/sd
TV = zrow(Ev,M) if TARGET=='eig' else (zrow(Ov,M) if TARGET=='oracle' else 0.5*zrow(Ev,M)+0.5*zrow(Ov,M))
# soft listwise target = softmax(standardized values / TEMP) over valid
TVt=torch.tensor(TV.astype(np.float32))
print(f"BEST-SHOT distill: set-transformer H={H} L={LYR} TARGET={TARGET} EP={EP}...",flush=True)
N=len(F); idx=np.arange(N)
for ep in range(EP):
    rng.shuffle(idx); tot=0.
    for b0 in range(0,N,256):
        bb=idx[b0:b0+256]; x=Xt[bb]; m=Mt[bb]; tv=TVt[bb]
        logit=pol(x,m).masked_fill(m==0,-1e9); logp=torch.log_softmax(logit,1)
        soft=torch.softmax(tv.masked_fill(m==0,-1e9)/TEMP,1)
        loss=-(soft*logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%30==0:
        with torch.no_grad():
            pr=pol(Xt,Mt).masked_fill(Mt==0,-1e9); age=(pr.argmax(1).numpy()==(Ev*M+(-1e9)*(1-M)).argmax(1)).mean(); ago=(pr.argmax(1).numpy()==(Ov*M+(-1e9)*(1-M)).argmax(1)).mean()
        print(f"  ep{ep+1} loss={tot/(N/256):.3f} argmax-agree eig={age:.2f} oracle={ago:.2f}",flush=True)
pol.eval()
def pol_pick(ut,cands):
    cs=cands[:MAXC]; x=torch.tensor(feat2(ut,cs)).unsqueeze(0); m=torch.ones(1,len(cs))
    with torch.no_grad(): sc=pol(x,m).squeeze(0).numpy()
    return cs[int(sc.argmax())]
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
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
                if sel=='eig': c=cands[int(eig_vals(ut,np.array(cands),asked,rd).argmax())]
                elif sel=='policy': c=pol_pick(ut,cands)
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
print(f"\n##### SET-TRANSFORMER distill TARGET={TARGET} #####",flush=True)
for tail in [False,True]:
    print(f"=== {'TAIL' if tail else 'FULL'} NDCG@10 ===",flush=True); Rr={}
    for sel in ['eig','policy','oracle']:
        Rr[sel]=run(sel,tail); print(f"  {sel:<7}: "+" ".join(f"q{q}={Rr[sel][q]:.3f}" for q in [0,1,2,4,8])+f" | +{Rr[sel][8]-Rr[sel][0]:+.3f}",flush=True)
    e,p=Rr['eig'],Rr['policy']; print(f"  policy vs eig @q4/q8: {p[4]:.3f}/{p[8]:.3f} vs {e[4]:.3f}/{e[8]:.3f} -> {'BEATS' if p[8]>e[8]+.004 else ('MATCHES' if abs(p[8]-e[8])<=.006 else 'BELOW')}",flush=True)
