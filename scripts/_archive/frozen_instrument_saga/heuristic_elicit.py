"""E1: realizable adaptive heuristics on the accepted instrument vs popularity.
Does ANY observable strategy beat popularity? If yes -> that's the teacher to distill."""
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
cnt=np.bincount(i,minlength=ni); pop_order=list(np.argsort(-cnt)); pop_n=cnt/cnt.max()
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
        rank=s.rank_bias+(h@s.item.weight.t() if TIE else s.rank_head(h)); rate=mu_ck+s.rate_bias+s.rate_head(h)
        return rank, rate
W=Inst(ni,DI); W.load_state_dict(_ck['state'],strict=False); W.eval()
def pad1(rev):
    if rev: return torch.tensor([[j for j,_ in rev]]),torch.tensor([[int(round(r)) for _,r in rev]]),torch.zeros(1,len(rev),dtype=torch.bool)
    return torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool)
def belief(rev):
    ci,cr,cm=pad1(rev)
    with torch.no_grad(): rk,rt=W.fwd(ci,cr,cm)
    return rk[0].numpy(), rt[0].numpy()
def ndcg_rec(rk,rel,unr):
    nd=rc=0.
    for ri in rel:
        rank=1+int((rk[unr]>=rk[ri]).sum())
        if rank<=10: nd+=1./np.log2(rank+1); rc+=1.
    return nd/len(rel), rc/len(rel)
cases=[]
for cu in cold:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); ncut=max(5,int(0.3*len(ii))); test=set(ii[:ncut]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    cases.append((rd,test,rel,np.array([j for j in range(ni) if j not in rd])))
print(f"cold cases {len(cases)}",flush=True)
def run(name,pick):
    nd=np.zeros(T+1); rc=np.zeros(T+1)
    for (rd,test,rel,unr) in cases:
        rev=[]; asked=set(test)
        for t in range(T+1):
            rk,rt=belief(rev); a,b=ndcg_rec(rk,rel,unr); nd[t]+=a; rc[t]+=b
            if t==T: break
            q=pick(rev,asked,rk,rt)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    nd/=len(cases); rc/=len(cases)
    print(f"  {name:<22} NDCG q5={nd[5]:.3f} q10={nd[10]:.3f} q20={nd[20]:.3f} AUC={nd.mean():.4f} | Rec q20={rc[20]:.3f}",flush=True)
# strategies
def pick_pop(rev,asked,rk,rt): return next((e for e in pop_order if e not in asked),None)
def pick_belief(rev,asked,rk,rt):
    s=rk.copy(); s[list(asked)]=-1e9; return int(np.argmax(s))                  # ask top predicted
def pick_uncert(rev,asked,rk,rt):
    un=-np.abs(rt-3.5); un[list(asked)]=-1e9; return int(np.argmax(un))         # most-uncertain rating
def pick_pop_uncert(rev,asked,rk,rt):
    sc=pop_n*(-np.abs(rt-3.5)+2.0); sc[list(asked)]=-1e9; return int(np.argmax(sc))  # answerable & uncertain
def pick_pop_belief(rev,asked,rk,rt):
    sc=pop_n*(rk-rk.min()); sc[list(asked)]=-1e9; return int(np.argmax(sc))     # answerable & promising
print("=== E1: realizable heuristics vs popularity ===",flush=True)
run('popularity',pick_pop)
run('belief-greedy',pick_belief)
run('uncertainty',pick_uncert)
run('pop x uncertainty',pick_pop_uncert)
run('pop x belief',pick_pop_belief)
