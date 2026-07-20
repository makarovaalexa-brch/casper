"""
E7: CASPER = BC-pretrain on (myopic) oracle  ->  RLOO finetune (AUC-over-turns NDCG
reward, which repairs the oracle's myopia). Actor = transformer over revealed
(item,rating). Eval cold-test vs popularity.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
base='C:/dev/phd/casper/data/movielens/ml-100k'; CK='C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
T=12; N_TEACH=int(os.environ.get('N_TEACH',350)); BC_EP=int(os.environ.get('BC_EP',12))
G=int(os.environ.get('G',5)); RL_EP=int(os.environ.get('RL_EP',15)); NPE=int(os.environ.get('NPE',140)); ENT=0.02; D=48; N_COLD=200
rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,Rr=[],[],[]
for line in open(f'{base}/u.data'):
    a=line.split('\t'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U); I=np.array(I); Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
cnt=np.bincount(i,minlength=ni); pop_order=list(np.argsort(-cnt))
rb={}
for k in range(len(u)): rb.setdefault(u[k],{})[i[k]]=Rr[k]
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
# ---- oracle demos ----
print(f"oracle demos ({N_TEACH} warm)...",flush=True); t0=time.time(); X,A=[],[]
for n,cu in enumerate(rng.choice(warm,min(N_TEACH,len(warm)),replace=False)):
    rd=rb[cu]; items=list(rd.keys())
    if len(items)<12: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); pool=[x for x in ii if x not in test]
    rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    unr=np.array([j for j in range(ni) if j not in rd]); rev=[]; asked=set()
    for _ in range(T):
        cands=[c for c in pool if c not in asked]
        if not cands: break
        s=inst_sc([rev+[(c,rd[c])] for c in cands]); best=cands[int(np.argmax([ndcg(s[k],rel,unr) for k in range(len(cands))]))]
        X.append(rev.copy()); A.append(int(best)); asked.add(best); rev.append((best,rd[best]))
    if (n+1)%100==0: print(f"  {n+1} ({time.time()-t0:.0f}s, {len(X)} demos)",flush=True)
print(f"{len(X)} demos",flush=True)
class Actor(nn.Module):
    def __init__(s,ni,d):
        super().__init__(); s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d); s.cls=nn.Parameter(torch.randn(d)*0.1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.1,batch_first=True),2); s.head=nn.Linear(d,ni)
    def forward(s,ci,cr,cm):
        tok=s.item(ci)+s.rate(cr); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),cm],1)
        return s.head(s.tf(x,src_key_padding_mask=pad)[:,0])
actor=Actor(ni,D); At=torch.tensor(A,dtype=torch.long)
opt=torch.optim.Adam(actor.parameters(),1e-3,weight_decay=1e-5)
print("BC pretrain...",flush=True)
for ep in range(BC_EP):
    idx=torch.randperm(len(X))
    for s in range(0,len(idx),128):
        bi=idx[s:s+128].tolist(); ci,cr,cm=pad_batch([X[k] for k in bi]); opt.zero_grad(); lg=actor(ci,cr,cm)
        for r_,row in enumerate([X[k] for k in bi]):
            for (e,_) in row: lg[r_,e]=-1e9
        F.cross_entropy(lg,At[idx[s:s+128]]).backward(); opt.step()
print("RLOO finetune (AUC reward)...",flush=True)
opt=torch.optim.Adam(actor.parameters(),2e-4); t0=time.time(); wc=[cu for cu in warm if len(rb[cu])>=12]
def alogits(rev):
    ci,cr,cm=pad_batch([rev if rev else []]); return actor(ci,cr,cm)[0]
for ep in range(RL_EP):
    actor.train(); batch=list(rng.choice(wc,min(NPE,len(wc)),replace=False)); losses=[]; Rs=[]
    for cu in batch:
        rd=rb[cu]; items=list(rd.keys()); ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); rel=[j for j in test if rd[j]>=4]
        if not rel: continue
        unr=np.array([j for j in range(ni) if j not in rd]); roll_lp=[]; roll_R=[]
        for g in range(G):
            rev=[]; asked=set(test); lps=[]; ndc=[]
            for _ in range(T):
                lg=alogits(rev).clone(); lg[list(asked)]=-1e9
                for (e,_) in rev: lg[e]=-1e9
                d=torch.distributions.Categorical(logits=lg); a=d.sample(); lps.append(d.log_prob(a)+ENT*d.entropy())
                ai=int(a.item()); asked.add(ai)
                if ai in rd: rev.append((ai,rd[ai]))
                ndc.append(ndcg(inst_sc([rev])[0],rel,unr))
            roll_R.append(np.mean(ndc)); roll_lp.append(torch.stack(lps).sum())  # AUC reward
        Rs.append(np.mean(roll_R)); Ra=np.array(roll_R); bse=(Ra.sum()-Ra)/(G-1)
        for g in range(G):
            adv=Ra[g]-bse[g]
            if abs(adv)>1e-9: losses.append(-adv*roll_lp[g])
    if losses:
        opt.zero_grad(); torch.stack(losses).mean().backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.); opt.step()
    if (ep+1)%5==0: print(f"  rl ep{ep+1}/{RL_EP} meanR={np.mean(Rs):.4f} ({time.time()-t0:.0f}s)",flush=True)
actor.eval()
cases=[]
for cu in cold:
    rd=rb[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    cases.append((rd,test,rel,np.array([j for j in range(ni) if j not in rd])))
def run(name,pick):
    nd=np.zeros(T+1)
    for (rd,test,rel,unr) in cases:
        rev=[]; asked=set(test)
        for t in range(T+1):
            nd[t]+=ndcg(inst_sc([rev])[0],rel,unr)
            if t==T: break
            q=pick(rev,asked,rd)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    nd/=len(cases); print(f"  {name:<14} NDCG q5={nd[5]:.3f} q8={nd[8]:.3f} q12={nd[-1]:.3f} AUC={nd.mean():.4f}",flush=True)
def pick_pop(rev,asked,rd): return next((e for e in pop_order if e not in asked),None)
def pick_actor(rev,asked,rd):
    with torch.no_grad(): lg=alogits(rev).numpy()
    lg[list(asked)]=-1e9
    for (e,_) in rev: lg[e]=-1e9
    return int(np.argmax(lg))
print(f"\n=== E7 COLD-TEST ({len(cases)}) ===",flush=True)
run('popularity',pick_pop); run('CASPER-BC+RL',pick_actor)
