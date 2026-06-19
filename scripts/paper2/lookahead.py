"""
PAPER B — best shot to BEAT greedy EIG on discrete item selection: 2-STEP LOOKAHEAD PLANNING (NOCTA-style; the method
the literature says beats greedy where RL only matches). Realizable (belief-only, no test peek): for each candidate c,
value2(c) = E_ans[ max_{c2} EIG(c2 | state after c=ans) ], expectation over c's answer under the model's belief.
Frozen encoder. Compare random/helf/eig(1-step)/lookahead(2-step)/oracle on full+tail NDCG.
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; rng=np.random.default_rng(0); torch.manual_seed(0)
NEVAL=int(os.environ.get('NEVAL',200)); OUTER=int(os.environ.get('OUTER',12))
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
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
POS=np.mean([ (rat_by_u[x][k][1]-mu-bi[rat_by_u[x][k][0]]) for x in trU[:3000] for k in range(len(rat_by_u[x])) if rat_by_u[x][k][1]>=4]);
NEG=np.mean([ (rat_by_u[x][k][1]-mu-bi[rat_by_u[x][k][0]]) for x in trU[:3000] for k in range(len(rat_by_u[x])) if rat_by_u[x][k][1]<4])
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
def eig_vals(ut,cands,pre):                              # expected coverage over `cands` after folding each candidate
    cands=np.array(cands); p=sig(popb[cands]+Ql[cands]@ut)
    ul=enc_u_batch([pre+[(c,POS)] for c in cands]); ud=enc_u_batch([pre+[(c,NEG)] for c in cands])
    return p*sig(ul@Ql[cands].T).sum(1)+(1-p)*sig(ud@Ql[cands].T).sum(1)
def two_step(asked,prof,rd,ut):
    cands=[j for j in prof if j not in asked]; pre=[(j,rd[j]-mu-bi[j]) for j in asked]
    e1=eig_vals(ut,cands,pre)
    short=[cands[i] for i in np.argsort(-e1)[:OUTER]]
    bestc=None;bestv=None
    for c in short:
        p=sig(popb[c]+Ql[c]@ut); rest=[j for j in cands if j!=c]
        if not rest: v=0.
        else:
            uln=enc_u(pre+[(c,POS)]); udn=enc_u(pre+[(c,NEG)])
            vN_like=eig_vals(uln,rest,pre+[(c,POS)]).max(); vN_dis=eig_vals(udn,rest,pre+[(c,NEG)]).max()
            v=p*vN_like+(1-p)*vN_dis
        if bestv is None or v>bestv: bestv=v;bestc=c
    return bestc
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:NEVAL]
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
                ut=enc_u([(j,rd[j]-mu-bi[j]) for j in asked]); pre=[(j,rd[j]-mu-bi[j]) for j in asked]
                if sel=='random': c=cands[int(rng.integers(len(cands)))]
                elif sel=='helf': c=max(cands,key=lambda j:helf[j])
                elif sel=='eig': c=cands[int(eig_vals(ut,cands,pre).argmax())]
                elif sel=='lookahead': c=two_step(asked,prof,rd,ut)
                else:
                    cu=enc_u_batch([pre+[(c2,rd[c2]-mu-bi[c2])] for c2 in cands]); best=None
                    for li,c2 in enumerate(cands):
                        v=ndcg(cu[li],tlike,set(prof)|set(asked)|{c2},tail)
                        if v is not None and (best is None or v>best[0]): best=(v,c2)
                    c=best[1] if best else cands[0]
                asked.append(c)
            v=ndcg(enc_u([(j,rd[j]-mu-bi[j]) for j in asked]),tlike,set(prof),tail)
            if v is not None: acc[q]+=v
        m+=1
    return {q:acc[q]/m for q in acc}
print(f"2-step lookahead vs 1-step EIG (OUTER={OUTER}, N={len(TE)})",flush=True)
for tail in [False,True]:
    print(f"=== {'TAIL' if tail else 'FULL'} NDCG@10 ===",flush=True); Rr={}
    for sel in ['random','helf','eig','lookahead','oracle']:
        Rr[sel]=run(sel,tail); print(f"  {sel:<10}: "+" ".join(f"q{q}={Rr[sel][q]:.3f}" for q in [0,1,2,4,8])+f" | +{Rr[sel][8]-Rr[sel][0]:+.3f}",flush=True)
    e,l=Rr['eig'],Rr['lookahead']; print(f"  lookahead vs eig @q2/q4/q8: {l[2]:.3f}/{l[4]:.3f}/{l[8]:.3f} vs {e[2]:.3f}/{e[4]:.3f}/{e[8]:.3f} -> {'BEATS' if l[8]>e[8]+.004 else ('MATCHES' if abs(l[8]-e[8])<=.006 else 'BELOW')}",flush=True)
