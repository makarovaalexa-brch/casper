"""
E2: belief-conditioned CASPER. Actor sees per-item [belief rank-score, predicted
rating, log-popularity, revealed] -> logit. Warm-start by BC on the pop x belief
teacher (E1 winner), then RLOO finetune on held-out NDCG. Eval cold-test vs
popularity and pop x belief.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
base='C:/dev/phd/casper/data/movielens/ml-100k'; CK='C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
T=12; G=int(os.environ.get('G',6)); RL_EP=int(os.environ.get('RL_EP',20)); NPE=int(os.environ.get('NPE',160))
BC_EP=int(os.environ.get('BC_EP',8)); ENT=float(os.environ.get('ENT',0.02)); N_COLD=200
rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,Rr=[],[],[]
with open(f'{base}/u.data') as f:
    for line in f:
        a=line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U); I=np.array(I); Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
cnt=np.bincount(i,minlength=ni); pop_order=list(np.argsort(-cnt)); logpop=(np.log1p(cnt)/np.log1p(cnt.max())).astype(np.float32); popn=(cnt/cnt.max()).astype(np.float32)
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
    def fwd(s,ci,cr,cm):
        tok=s.item(ci)+s.rate(cr); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),cm],1)
        h=s.tf(x,src_key_padding_mask=pad)[:,0]
        return s.rank_bias+(h@s.item.weight.t() if TIE else s.rank_head(h)), mu_ck+s.rate_bias+s.rate_head(h)
W=Inst(ni,DI); W.load_state_dict(_ck['state'],strict=False); W.eval()
lp_t=torch.tensor(logpop); popn_t=torch.tensor(popn)
def belief(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): rk,rt=W.fwd(ci,cr,cm)
    return rk[0], rt[0]
def feats(rk, rt, revset):
    rkn=(rk-rk.mean())/(rk.std()+1e-6); rtn=(rt-3.)/2.
    rev=torch.zeros(ni);
    if revset: rev[list(revset)]=1.
    return torch.stack([rkn, rtn, lp_t, rev], 1)   # [ni,4]
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
    def forward(s,F_): return s.net(F_).squeeze(-1)   # [ni]
actor=Actor()
def ndcg(rk,rel,unr):
    rk=rk.detach().numpy(); return np.mean([(1./np.log2(2+int((rk[unr]>=rk[ri]).sum())) if 1+int((rk[unr]>=rk[ri]).sum())<=10 else 0.) for ri in rel])
# ---- BC warm-start on pop x belief ----
print("BC warm-start on pop x belief...",flush=True)
Xb=[]; Yb=[]
for cu in list(rng.choice(warm,300,replace=False)):
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,len(ii)//3)])
    rev=[]; asked=set(test)
    for _ in range(T):
        rk,rt=belief(rev); F_=feats(rk,rt,set(e for e,_ in rev))
        sc=(popn*(rk.numpy()-rk.numpy().min())); sc[list(asked)]=-1e9; a=int(np.argmax(sc))
        Xb.append(F_); Yb.append(a); asked.add(a)
        if a in rd: rev.append((a,rd[a]))
opt=torch.optim.Adam(actor.parameters(),1e-3)
for ep in range(BC_EP):
    idx=torch.randperm(len(Xb))
    for s in range(0,len(idx),64):
        b=idx[s:s+64]; loss=0
        opt.zero_grad()
        for k in b: loss=loss+F.cross_entropy(actor(Xb[k]).unsqueeze(0), torch.tensor([Yb[k]]))
        (loss/len(b)).backward(); opt.step()
print("BC done; RLOO finetune...",flush=True)
# ---- RLOO ----
opt=torch.optim.Adam(actor.parameters(),3e-4); t0=time.time()
wc=[cu for cu in warm if len(ratings_by[cu])>=12]
for ep in range(RL_EP):
    actor.train(); batch=list(rng.choice(wc,min(NPE,len(wc)),replace=False)); losses=[]; Rs=[]
    for cu in batch:
        rd=ratings_by[cu]; items=list(rd.keys()); ii=items[:]; rng.shuffle(ii)
        test=set(ii[:max(5,int(0.3*len(ii)))]); rel=[j for j in test if rd[j]>=4]
        if not rel: continue
        unr=np.array([j for j in range(ni) if j not in rd]); roll_lp=[]; roll_R=[]
        for g in range(G):
            rev=[]; asked=set(test); lps=[]
            for _ in range(T):
                rk,rt=belief(rev); lg=actor(feats(rk,rt,set(e for e,_ in rev)))
                lg=lg.clone(); lg[list(asked)]=-1e9
                d=torch.distributions.Categorical(logits=lg); a=d.sample(); lps.append(d.log_prob(a)+ENT*d.entropy())
                ai=int(a.item()); asked.add(ai)
                if ai in rd: rev.append((ai,rd[ai]))
            rk,_=belief(rev); roll_R.append(ndcg(rk,rel,unr)); roll_lp.append(torch.stack(lps).sum())
        Rs.append(np.mean(roll_R)); Ra=np.array(roll_R); bse=(Ra.sum()-Ra)/(G-1)
        for g in range(G):
            adv=Ra[g]-bse[g]
            if abs(adv)>1e-9: losses.append(-adv*roll_lp[g])
    if losses:
        opt.zero_grad(); torch.stack(losses).mean().backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.); opt.step()
    if (ep+1)%5==0: print(f"  rl ep{ep+1}/{RL_EP} meanR={np.mean(Rs):.4f} ({time.time()-t0:.0f}s)",flush=True)
actor.eval()
# ---- eval ----
cases=[]
for cu in cold:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    cases.append((rd,test,rel,np.array([j for j in range(ni) if j not in rd])))
def run(name,pick):
    nd=np.zeros(T+1)
    for (rd,test,rel,unr) in cases:
        rev=[]; asked=set(test)
        for t in range(T+1):
            rk,rt=belief(rev); nd[t]+=ndcg(rk,rel,unr)
            if t==T: break
            q=pick(rev,asked,rk,rt)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    nd/=len(cases); print(f"  {name:<16} NDCG q5={nd[5]:.3f} q8={nd[8]:.3f} q12={nd[-1]:.3f} AUC={nd.mean():.4f}",flush=True)
def pick_pop(rev,asked,rk,rt): return next((e for e in pop_order if e not in asked),None)
def pick_popbel(rev,asked,rk,rt):
    sc=popn*(rk.numpy()-rk.numpy().min()); sc[list(asked)]=-1e9; return int(np.argmax(sc))
def pick_actor(rev,asked,rk,rt):
    with torch.no_grad(): lg=actor(feats(rk,rt,set(e for e,_ in rev))).numpy()
    lg[list(asked)]=-1e9; return int(np.argmax(lg))
print(f"\n=== E2 COLD-TEST ({len(cases)} users) ===",flush=True)
run('popularity',pick_pop); run('pop x belief',pick_popbel); run('CASPER-belief',pick_actor)
