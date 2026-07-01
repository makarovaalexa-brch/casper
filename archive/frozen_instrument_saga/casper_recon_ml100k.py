"""
E4: CASPER on the RECONSTRUCTION metric. Belief-conditioned actor, reward =
held-out liked-vs-disliked AUC, warm-started by BC on the uncertainty teacher,
then RLOO. Eval cold-test recon-AUC vs popularity + uncertainty.
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
cnt=np.bincount(i,minlength=ni); pop_order=list(np.argsort(-cnt)); logpop=(np.log1p(cnt)/np.log1p(cnt.max())).astype(np.float32)
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
lp_t=torch.tensor(logpop)
def belief(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): rk,rt=W.fwd(ci,cr,cm)
    return rk[0], rt[0]
def feats(rk,rt,revset):
    rkn=(rk-rk.mean())/(rk.std()+1e-6); rtn=(rt-3.)/2.; unc=-torch.abs(rt-3.5)
    rev=torch.zeros(ni)
    if revset: rev[list(revset)]=1.
    return torch.stack([rkn,rtn,unc,lp_t,rev],1)   # [ni,5]
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
    def forward(s,F_): return s.net(F_).squeeze(-1)
actor=Actor()
def auc(sc,lab):
    pos=sc[lab==1]; neg=sc[lab==0]
    if len(pos)==0 or len(neg)==0: return 0.5
    al=np.concatenate([pos,neg]); o=np.argsort(al,kind='mergesort'); r=np.empty(len(al)); r[o]=np.arange(1,len(al)+1)
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
# ---- BC warm-start on uncertainty ----
print("BC warm-start on uncertainty...",flush=True)
Xb=[]; Yb=[]
for cu in list(rng.choice(warm,300,replace=False)):
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(6,int(0.35*len(ii)))]); rev=[]; asked=set(test)
    for _ in range(T):
        rk,rt=belief(rev); F_=feats(rk,rt,set(e for e,_ in rev))
        un=-np.abs(rt.numpy()-3.5); un[list(asked)]=-1e9; a=int(np.argmax(un))
        Xb.append(F_); Yb.append(a); asked.add(a)
        if a in rd: rev.append((a,rd[a]))
opt=torch.optim.Adam(actor.parameters(),1e-3)
for ep in range(BC_EP):
    idx=torch.randperm(len(Xb))
    for s in range(0,len(idx),64):
        b=idx[s:s+64]; opt.zero_grad(); loss=0
        for k in b: loss=loss+F.cross_entropy(actor(Xb[k]).unsqueeze(0),torch.tensor([Yb[k]]))
        (loss/len(b)).backward(); opt.step()
print("RLOO finetune (recon reward)...",flush=True)
opt=torch.optim.Adam(actor.parameters(),3e-4); t0=time.time()
wc=[cu for cu in warm if len(ratings_by[cu])>=12]
for ep in range(RL_EP):
    actor.train(); batch=list(rng.choice(wc,min(NPE,len(wc)),replace=False)); losses=[]; Rs=[]
    for cu in batch:
        rd=ratings_by[cu]; items=list(rd.keys()); ii=items[:]; rng.shuffle(ii)
        test=list(ii[:max(6,int(0.35*len(ii)))]); lab=np.array([1 if rd[j]>=4 else 0 for j in test]); tarr=np.array(test)
        if lab.sum()<1 or (1-lab).sum()<1: continue
        roll_lp=[]; roll_R=[]
        for g in range(G):
            rev=[]; asked=set(test); lps=[]
            for _ in range(T):
                rk,rt=belief(rev); lg=actor(feats(rk,rt,set(e for e,_ in rev))).clone(); lg[list(asked)]=-1e9
                d=torch.distributions.Categorical(logits=lg); a=d.sample(); lps.append(d.log_prob(a)+ENT*d.entropy())
                ai=int(a.item()); asked.add(ai)
                if ai in rd: rev.append((ai,rd[ai]))
            rk,_=belief(rev); roll_R.append(auc(rk.numpy()[tarr],lab)); roll_lp.append(torch.stack(lps).sum())
        Rs.append(np.mean(roll_R)); Ra=np.array(roll_R); bse=(Ra.sum()-Ra)/(G-1)
        for g in range(G):
            adv=Ra[g]-bse[g]
            if abs(adv)>1e-9: losses.append(-adv*roll_lp[g])
    if losses:
        opt.zero_grad(); torch.stack(losses).mean().backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.); opt.step()
    if (ep+1)%5==0: print(f"  rl ep{ep+1}/{RL_EP} meanR={np.mean(Rs):.4f} ({time.time()-t0:.0f}s)",flush=True)
actor.eval()
# ---- eval recon-AUC ----
cases=[]
for cu in cold:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<12: continue
    ii=items[:]; rng.shuffle(ii); test=list(ii[:max(6,int(0.35*len(ii)))]); lab=np.array([1 if rd[j]>=4 else 0 for j in test])
    if lab.sum()<1 or (1-lab).sum()<1: continue
    cases.append((rd,set(test),np.array(test),lab))
def run(name,pick):
    a=np.zeros(T+1)
    for (rd,testset,tarr,lab) in cases:
        rev=[]; asked=set(testset)
        for t in range(T+1):
            rk,rt=belief(rev); a[t]+=auc(rk.numpy()[tarr],lab)
            if t==T: break
            q=pick(rev,asked,rk,rt)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    a/=len(cases); print(f"  {name:<16} AUC q0={a[0]:.3f} q5={a[5]:.3f} q12={a[-1]:.3f} gain={a[-1]-a[0]:+.3f}",flush=True)
def pick_pop(rev,asked,rk,rt): return next((e for e in pop_order if e not in asked),None)
def pick_unc(rev,asked,rk,rt):
    un=-np.abs(rt.numpy()-3.5); un[list(asked)]=-1e9; return int(np.argmax(un))
def pick_actor(rev,asked,rk,rt):
    with torch.no_grad(): lg=actor(feats(rk,rt,set(e for e,_ in rev))).numpy()
    lg[list(asked)]=-1e9; return int(np.argmax(lg))
print(f"\n=== E4 COLD-TEST recon-AUC ({len(cases)} users) ===",flush=True)
run('popularity',pick_pop); run('uncertainty',pick_unc); run('CASPER-recon',pick_actor)
