"""
E8: belief-conditioned actor + BC-on-ORACLE init + RLOO (AUC reward).
Actor sees per-item [belief, rate, uncertainty, logpop, revealed] (E1's signal) AND
is pretrained to imitate the oracle, then RL-finetuned. Reports POST-BC eval too.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DATASET=os.environ.get('DATASET','ml-100k'); base=f'C:/dev/phd/casper/data/movielens/{DATASET}'
CK=f"C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_{DATASET.replace('-','')}_s42.pt"
_fn,_sep=('u.data','\t') if DATASET=='ml-100k' else ('ratings.dat','::')
T=12; N_TEACH=int(os.environ.get('N_TEACH',350)); BC_EP=int(os.environ.get('BC_EP',12))
G=int(os.environ.get('G',5)); RL_EP=int(os.environ.get('RL_EP',25)); NPE=int(os.environ.get('NPE',140)); ENT=float(os.environ.get('ENT',0.005)); N_COLD=200
rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,Rr=[],[],[]
for line in open(f'{base}/{_fn}'):
    a=line.strip().split(_sep); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U); I=np.array(I); Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
cnt=np.bincount(i,minlength=ni); pop_order=list(np.argsort(-cnt)); logpop=(np.log1p(cnt)/np.log1p(cnt.max())).astype(np.float32)
nov=-np.log2(np.clip(cnt/nu,1e-4,None)).astype(np.float32)   # novelty (self-information): serendipity weight
rb={}
for k in range(len(u)): rb.setdefault(u[k],{})[i[k]]=Rr[k]
cold=set(np.random.default_rng(0).choice(nu,N_COLD,replace=False).tolist()); warm=[x for x in range(nu) if x not in cold]
rng.shuffle(warm); warm_val=warm[:80]; warm_tr=warm[80:]   # held-out warm-val for RL early-stop
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
    return rk[0],rt[0]
def scb(revs):
    L=max(1,max(len(r) for r in revs)); B=len(revs); ci=np.zeros((B,L),np.int64); cr=np.zeros((B,L),np.int64); cm=np.ones((B,L),bool)
    for b,r in enumerate(revs):
        for j,(e,v) in enumerate(r): ci[b,j]=e; cr[b,j]=int(round(v)); cm[b,j]=False
    with torch.no_grad(): return W.fwd(torch.from_numpy(ci),torch.from_numpy(cr),torch.from_numpy(cm))[0].numpy()
def feats(rk,rt,revset):
    rkn=(rk-rk.mean())/(rk.std()+1e-6); rtn=(rt-3.)/2.; unc=-torch.abs(rt-3.5)
    rev=torch.zeros(ni)
    if revset: rev[list(revset)]=1.
    return torch.stack([rkn,rtn,unc,lp_t,rev],1)
def ndcg(pr,rel,unr):   # plain NDCG@10 (primary accepted metric) — reward + eval
    return np.mean([(1./np.log2(2+int((pr[unr]>=pr[ri]).sum())) if 1+int((pr[unr]>=pr[ri]).sum())<=10 else 0.) for ri in rel])
POPW=float(os.environ.get('POPW',6.0))
class Actor(nn.Module):   # popularity BASE + zero-init residual -> starts EXACTLY at popularity
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
        nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias)   # residual==0 at init
    def forward(s,F_): return POPW*F_[:,3] + s.net(F_).squeeze(-1)         # F_[:,3]=logpop
actor=Actor()
LAM_AUX=float(os.environ.get('LAM_AUX',0.5))
# ---- collect ORACLE demos (features, action) for DQfD-style auxiliary imitation during RL ----
print(f"oracle demos for aux-imitation ({N_TEACH})...",flush=True); _t=time.time(); Xo=[]; Ao=[]
for n,cu in enumerate(rng.choice(warm_tr,min(N_TEACH,len(warm_tr)),replace=False)):
    rd=rb[cu]; items=list(rd.keys())
    if len(items)<12: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); pool=[x for x in ii if x not in test]
    rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    unr=np.array([j for j in range(ni) if j not in rd]); rev=[]; asked=set()
    for _ in range(T):
        cands=[c for c in pool if c not in asked]
        if not cands: break
        rk,rt=belief(rev); Xo.append(feats(rk,rt,set(e for e,_ in rev)))
        s=scb([rev+[(c,rd[c])] for c in cands]); best=cands[int(np.argmax([ndcg(s[k],rel,unr) for k in range(len(cands))]))]
        Ao.append(int(best)); asked.add(best); rev.append((best,rd[best]))
    if (n+1)%120==0: print(f"  {n+1} ({time.time()-_t:.0f}s, {len(Xo)} demos)",flush=True)
Ao=torch.tensor(Ao,dtype=torch.long); print(f"{len(Xo)} oracle demos",flush=True)
# No BC: residual is zero-init => policy == popularity at start (floor guaranteed by construction).
# eval helper
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
            rk,rt=belief(rev); nd[t]+=ndcg(rk.numpy(),rel,unr)
            if t==T: break
            q=pick(rev,asked,rk,rt)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q,rd[q]))
    nd/=len(cases); print(f"  {name:<16} sNDCG q5={nd[5]:.3f} q8={nd[8]:.3f} q12={nd[-1]:.3f} AUC={nd.mean():.4f}",flush=True)
def pick_pop(rev,asked,rk,rt): return next((e for e in pop_order if e not in asked),None)
def pick_actor(rev,asked,rk,rt):
    with torch.no_grad(): lg=actor(feats(rk,rt,set(e for e,_ in rev))).numpy()
    lg[list(asked)]=-1e9; return int(np.argmax(lg))
print("\n=== E13 POST-INIT (== popularity) ===",flush=True); run('popularity',pick_pop); run('CASPER-init',pick_actor)
# ---- RLOO finetune ----
val_cases=[]
for cu in warm_val:
    rd=rb[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    val_cases.append((rd,test,rel,np.array([j for j in range(ni) if j not in rd])))
def val_seren():
    actor.eval(); tot=0.
    for (rd,test,rel,unr) in val_cases:
        rev=[]; asked=set(test); ss=0.
        for t in range(T+1):
            rk,rt=belief(rev); ss+=ndcg(rk.numpy(),rel,unr)
            if t==T: break
            q=pick_actor(rev,asked,rk,rt); asked.add(q)
            if q in rd: rev.append((q,rd[q]))
        tot+=ss/(T+1)
    return tot/len(val_cases)
print("\nRLOO finetune (val-early-stop)...",flush=True); opt=torch.optim.Adam(actor.parameters(),2e-4,weight_decay=1e-4); t0=time.time(); wc=[cu for cu in warm_tr if len(rb[cu])>=12]
best=val_seren(); bstate={k:v.clone() for k,v in actor.state_dict().items()}; bad=0
print(f"  post-BC val_seren={best:.4f}",flush=True)
for ep in range(RL_EP):
    actor.train(); batch=list(rng.choice(wc,min(NPE,len(wc)),replace=False)); losses=[]; Rs=[]
    for cu in batch:
        rd=rb[cu]; items=list(rd.keys()); ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); rel=[j for j in test if rd[j]>=4]
        if not rel: continue
        unr=np.array([j for j in range(ni) if j not in rd]); roll_lp=[]; roll_R=[]
        for g in range(G):
            rev=[]; asked=set(test); lps=[]; ndc=[]
            for _ in range(T):
                rk,rt=belief(rev); lg=actor(feats(rk,rt,set(e for e,_ in rev))).clone(); lg[list(asked)]=-1e9
                d=torch.distributions.Categorical(logits=lg); a=d.sample(); lps.append(d.log_prob(a)+ENT*d.entropy())
                ai=int(a.item()); asked.add(ai)
                if ai in rd: rev.append((ai,rd[ai]))
                ndc.append(ndcg(belief(rev)[0].numpy(),rel,unr))
            roll_R.append(np.mean(ndc)); roll_lp.append(torch.stack(lps).sum())
        Rs.append(np.mean(roll_R)); Ra=np.array(roll_R); bse=(Ra.sum()-Ra)/(G-1)
        for g in range(G):
            adv=Ra[g]-bse[g]
            if abs(adv)>1e-9: losses.append(-adv*roll_lp[g])
    # DQfD-style auxiliary oracle imitation (pull residual toward adaptive picks; escape pop basin)
    di=torch.randint(0,len(Xo),(128,)); aux=torch.stack([F.cross_entropy(actor(Xo[k]).unsqueeze(0),Ao[k:k+1]) for k in di]).mean()
    if losses:
        opt.zero_grad(); (torch.stack(losses).mean()+LAM_AUX*aux).backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.); opt.step()
    else:
        opt.zero_grad(); (LAM_AUX*aux).backward(); opt.step()
    vs=val_seren()
    if vs>best+1e-4: best=vs; bstate={k:v.clone() for k,v in actor.state_dict().items()}; bad=0
    else: bad+=1
    print(f"  rl ep{ep+1}/{RL_EP} trainR={np.mean(Rs):.4f} valSeren={vs:.4f} best={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
    if bad>=5: print(f"  early stop ep{ep+1}",flush=True); break
actor.load_state_dict(bstate); actor.eval()
print("\n=== E13 FINAL NDCG (RPL + DQfD oracle-aux) ===",flush=True); run('popularity',pick_pop); run('CASPER-E13',pick_actor)
# ---- ADAPTIVITY CHECK: flip a real reveal -> does next question change? ----
flips=tot=0
for (rd,test,rel,unr) in cases:
    rev=[]; asked=set(test)
    for _ in range(T):
        rk,rt=belief(rev); a=pick_actor(rev,asked,rk,rt); asked.add(a); v=rd.get(a,None)
        if v is not None:
            rev.append((a,v))
            rev_cf=rev[:-1]+[(a,1.0 if v<4 else 5.0)]   # flip polarity of this answer
            rkc,rtc=belief(rev_cf)
            if pick_actor(rev_cf,asked,rkc,rtc)!=pick_actor(rev,asked,rk,rt): flips+=1
            tot+=1
print(f"\nADAPTIVITY: next question changes when a reveal is flipped in {flips}/{tot} = {100*flips/max(tot,1):.0f}% of decision points",flush=True)
