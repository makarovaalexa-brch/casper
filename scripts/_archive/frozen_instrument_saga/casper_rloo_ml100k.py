"""
CASPER via RLOO on the accepted ml-100k instrument (avoids the imitation gap).
Actor = transformer(revealed)->next-question logits. Reward = held-out NDCG@10
(instrument, full-catalogue) after the dialogue. RLOO baseline over G rollouts.
Warm-train; eval cold-test (argmax) vs popularity + oracle ceiling.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
base='C:/dev/phd/casper/data/movielens/ml-100k'; CK='C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
T=12; G=int(os.environ.get('G',5)); EPOCHS=int(os.environ.get('EPOCHS',25)); NPE=int(os.environ.get('NPE',160))
ENT=float(os.environ.get('ENT',0.02)); LR=float(os.environ.get('LR',5e-4)); D=48; N_COLD=200
rng=np.random.default_rng(0); torch.manual_seed(0)
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
cold=set(np.random.default_rng(0).choice(nu,N_COLD,replace=False).tolist()); warm=[x for x in range(nu) if x not in cold]
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
def inst_sc(revs):
    ci,cr,cm=pad_batch([r if r else [] for r in revs])
    with torch.no_grad(): return W.enc(ci,cr,cm).numpy()
def ndcg(pr,rel,unr): return np.mean([(1./np.log2(2+int((pr[unr]>=pr[ri]).sum())) if 1+int((pr[unr]>=pr[ri]).sum())<=10 else 0.) for ri in rel])
class Actor(nn.Module):
    def __init__(s,ni,d):
        super().__init__(); s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d); s.cls=nn.Parameter(torch.randn(d)*0.1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.1,batch_first=True),2); s.head=nn.Linear(d,ni)
    def forward(s,ci,cr,cm):
        tok=s.item(ci)+s.rate(cr); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),cm],1)
        return s.head(s.tf(x,src_key_padding_mask=pad)[:,0])
actor=Actor(ni,D); opt=torch.optim.Adam(actor.parameters(),LR)
def actor_logits(rev):
    ci,cr,cm=pad_batch([rev if rev else []]); return actor(ci,cr,cm)[0]
# warm users with held-out rel
wcases=[]
for cu in warm:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<12: continue
    wcases.append(cu)
print(f"warm cases {len(wcases)}; oracle ceiling ~0.26, popularity ~0.12",flush=True)
t0=time.time()
for ep in range(EPOCHS):
    actor.train(); batch=list(rng.choice(wcases,min(NPE,len(wcases)),replace=False)); losses=[]; Rs=[]
    for cu in batch:
        rd=ratings_by[cu]; items=list(rd.keys()); ii=items[:]; rng.shuffle(ii)
        ncut=max(5,int(0.3*len(ii))); test=set(ii[:ncut]); rel=[j for j in test if rd[j]>=4]
        if not rel: continue
        unr=np.array([j for j in range(ni) if j not in rd])
        roll_lp=[]; roll_R=[]
        for g in range(G):
            rev=[]; asked=set(test); lps=[]
            for _ in range(T):
                lg=actor_logits(rev).clone()
                idxbad=list(asked)+[e for e,_ in rev]; lg[idxbad]=-1e9
                dist=torch.distributions.Categorical(logits=lg); a=dist.sample()
                lps.append(dist.log_prob(a)+ENT*dist.entropy()); ai=int(a.item()); asked.add(ai)
                if ai in rd: rev.append((ai,rd[ai]))
            R=ndcg(inst_sc([rev])[0],rel,unr); roll_lp.append(torch.stack(lps).sum()); roll_R.append(R)
        Rs.append(np.mean(roll_R)); Rarr=np.array(roll_R)
        base=(Rarr.sum()-Rarr)/(G-1)
        for g in range(G):
            adv=Rarr[g]-base[g]
            if abs(adv)>1e-9: losses.append(-adv*roll_lp[g])
    if losses:
        opt.zero_grad(); torch.stack(losses).mean().backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.); opt.step()
    if (ep+1)%5==0: print(f"  ep{ep+1}/{EPOCHS} meanR={np.mean(Rs):.4f} ({time.time()-t0:.0f}s)",flush=True)
actor.eval()
# eval cold-test
cases=[]
for cu in cold:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); ncut=max(5,int(0.3*len(ii))); test=set(ii[:ncut]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    cases.append((rd,test,rel,np.array([j for j in range(ni) if j not in rd])))
def run(name,pick):
    nd=np.zeros(T+1); rc=np.zeros(T+1)
    for (rd,test,rel,unr) in cases:
        rev=[]; asked=set(test)
        for t in range(T+1):
            sc=inst_sc([rev])[0]; nd[t]+=ndcg(sc,rel,unr)
            rc[t]+=np.mean([1. if 1+int((sc[unr]>=sc[ri]).sum())<=10 else 0. for ri in rel])
            if t==T: break
            q=pick(rev,asked,rd)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    nd/=len(cases); rc/=len(cases)
    print(f"  {name:<12} NDCG q5={nd[5]:.3f} q10={nd[10]:.3f} q12={nd[-1]:.3f} | Rec q12={rc[-1]:.3f}",flush=True)
def pick_actor(rev,asked,rd):
    with torch.no_grad(): lg=actor_logits(rev).numpy()
    lg[list(asked)]=-1e9
    for (e,_) in rev: lg[e]=-1e9
    return int(np.argmax(lg))
def pick_pop(rev,asked,rd): return next((e for e in pop_order if e not in asked),None)
print(f"\n=== COLD-TEST ({len(cases)} users) ===",flush=True)
run('popularity',pick_pop); run('CASPER-RLOO',pick_actor)
