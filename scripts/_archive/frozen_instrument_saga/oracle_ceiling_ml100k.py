"""Headroom check on accepted ml-100k instrument: popularity vs clairvoyant ORACLE
(picks the answerable movie whose true reveal most improves held-out NDCG) vs a
realizable greedy-uncertainty. Tells us if any adaptive policy can beat popularity."""
import numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens/ml-100k'; CK='C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
T=15; N_COLD=200; rng=np.random.default_rng(0)
U,I,Rr=[],[],[]
with open(f'{base}/u.data') as f:
    for line in f:
        a=line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U); I=np.array(I); Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
cnt=np.bincount(i,minlength=ni); pop_order=list(np.argsort(-cnt))
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
    def enc(s,ci,cr,cm):
        tok=s.item(ci)+s.rate(cr); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),cm],1)
        h=s.tf(x,src_key_padding_mask=pad)[:,0]; return s.rank_bias+(h@s.item.weight.t() if TIE else s.rank_head(h))
W=Inst(ni,DI); W.load_state_dict(_ck['state'],strict=False); W.eval()
def pad_batch(revs):
    L=max(1,max(len(r) for r in revs)); B=len(revs); ci=np.zeros((B,L),np.int64); cr=np.zeros((B,L),np.int64); cm=np.ones((B,L),bool)
    for b,r in enumerate(revs):
        for j,(e,v) in enumerate(r): ci[b,j]=e; cr[b,j]=int(round(v)); cm[b,j]=False
    return torch.from_numpy(ci),torch.from_numpy(cr),torch.from_numpy(cm)
def sc_batch(revs):
    ci,cr,cm=pad_batch([r if r else [] for r in revs])
    with torch.no_grad(): return W.enc(ci,cr,cm).numpy()
def ndcg(pr,rel,unr): return np.mean([(1./np.log2(2+int((pr[unr]>=pr[ri]).sum())) if 1+int((pr[unr]>=pr[ri]).sum())<=10 else 0.) for ri in rel])
cases=[]
for cu in cold:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); ncut=max(5,int(0.3*len(ii))); test=set(ii[:ncut]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    cases.append((rd,test,rel,[x for x in ii if x not in test],np.array([j for j in range(ni) if j not in rd])))
def run(name, kind):
    nd=np.zeros(T+1)
    for (rd,test,rel,pool,unr) in cases:
        rev=[]; asked=set(test)
        for t in range(T+1):
            nd[t]+=ndcg(sc_batch([rev])[0],rel,unr)
            if t==T: break
            if kind=='rand': q=int(rng.choice([e for e in range(ni) if e not in asked]))
            elif kind=='pop': q=next((e for e in pop_order if e not in asked),None)
            elif kind=='oracle':
                cands=[c for c in pool if c not in asked]
                if not cands: q=None
                else:
                    sets=[rev+[(c,rd[c])] for c in cands]; s=sc_batch(sets)
                    q=cands[int(np.argmax([ndcg(s[k],rel,unr) for k in range(len(cands))]))]
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    nd/=len(cases); print(f"  {name:<10} NDCG q5={nd[5]:.3f} q10={nd[10]:.3f} q15={nd[-1]:.3f}",flush=True); return nd
print(f"=== headroom ({len(cases)} cold users) ===",flush=True)
run('random','rand'); run('popularity','pop'); run('ORACLE','oracle')
