"""E3: RECONSTRUCTION metric (held-out liked-vs-disliked AUC per turn) — the
IJCNN-style 'recommendation-loss reduction' lens. Does an INFORMATIVE strategy
beat popularity at predicting the user's whole held-out profile?"""
import numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens/ml-100k'; CK='C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
T=20; N_COLD=200; rng=np.random.default_rng(0)
U,I,Rr=[],[],[]
with open(f'{base}/u.data') as f:
    for line in f:
        a=line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U); I=np.array(I); Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
cnt=np.bincount(i,minlength=ni); pop_order=list(np.argsort(-cnt)); popn=cnt/cnt.max()
ratings_by={}
for k in range(len(u)): ratings_by.setdefault(u[k],{})[i[k]]=Rr[k]
cold=list(np.random.default_rng(0).choice(nu,N_COLD,replace=False))
_ck=torch.load(CK); mu_ck=_ck['mu']; DI=_ck['d']; TIE=_ck.get('tie',False)
class Inst(nn.Module):
    def __init__(s,ni,d):
        super().__init__(); s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d); s.cls=nn.Parameter(torch.randn(d)*0.1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.2,batch_first=True),2)
        s.rank_bias=nn.Parameter(torch.zeros(ni))
        if not TIE: s.rank_head=nn.Linear(d,ni)
        s.rate_head=nn.Linear(d,ni); s.rate_bias=nn.Parameter(torch.zeros(ni))
    def fwd(s,ci,cr,cm):
        tok=s.item(ci)+s.rate(cr); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),cm],1)
        h=s.tf(x,src_key_padding_mask=pad)[:,0]
        return s.rank_bias+(h@s.item.weight.t() if TIE else s.rank_head(h)), mu_ck+s.rate_bias+s.rate_head(h)
W=Inst(ni,DI); W.load_state_dict(_ck['state'],strict=False); W.eval()
def belief(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): rk,rt=W.fwd(ci,cr,cm)
    return rk[0].numpy(), rt[0].numpy()
def auc(sc, lab):
    pos=sc[lab==1]; neg=sc[lab==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    al=np.concatenate([pos,neg]); o=np.argsort(al,kind='mergesort'); r=np.empty(len(al)); r[o]=np.arange(1,len(al)+1)
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
# cases with held-out liked AND disliked
cases=[]
for cu in cold:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<12: continue
    ii=items[:]; rng.shuffle(ii); test=list(ii[:max(6,int(0.35*len(ii)))])
    lab=np.array([1 if rd[j]>=4 else 0 for j in test])
    if lab.sum()<1 or (1-lab).sum()<1: continue
    cases.append((rd,set(test),np.array(test),lab))
print(f"recon cases {len(cases)}",flush=True)
def run(name,pick):
    a=np.zeros(T+1)
    for (rd,testset,tarr,lab) in cases:
        rev=[]; asked=set(testset)
        for t in range(T+1):
            rk,rt=belief(rev); a[t]+=auc(rk[tarr],lab)
            if t==T: break
            q=pick(rev,asked,rk,rt)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    a/=len(cases); print(f"  {name:<18} AUC q0={a[0]:.3f} q5={a[5]:.3f} q10={a[10]:.3f} q20={a[20]:.3f} gain={a[20]-a[0]:+.3f}",flush=True)
def pick_pop(rev,asked,rk,rt): return next((e for e in pop_order if e not in asked),None)
def pick_unc(rev,asked,rk,rt):
    un=-np.abs(rt-3.5); un[list(asked)]=-1e9; return int(np.argmax(un))
def pick_popunc(rev,asked,rk,rt):
    sc=popn*(-np.abs(rt-3.5)+2.0); sc[list(asked)]=-1e9; return int(np.argmax(sc))
def pick_belief(rev,asked,rk,rt):
    s=rk.copy(); s[list(asked)]=-1e9; return int(np.argmax(s))
def pick_popbel(rev,asked,rk,rt):
    sc=popn*(rk-rk.min()); sc[list(asked)]=-1e9; return int(np.argmax(sc))
def pick_rand(rev,asked,rk,rt):
    c=[e for e in range(ni) if e not in asked]; return int(rng.choice(c))
print("=== E3: RECONSTRUCTION (held-out liked-vs-disliked AUC) ===",flush=True)
run('random',pick_rand); run('popularity',pick_pop); run('uncertainty',pick_unc)
run('pop x uncertainty',pick_popunc); run('belief-greedy',pick_belief); run('pop x belief',pick_popbel)
