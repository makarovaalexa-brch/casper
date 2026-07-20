"""
SYNTHETIC HIERARCHICAL WORLD on the CURRENT ruler (NDCG@10 + the current model lineup).
Replaces the OUTDATED AUAC + paper-1 policies (PPO/DQN/REINFORCE/SCPR) with the SAME
metric and SAME models we report on ML-1M, so the two testbeds share one ruler.

WHY a separate harness (not continuous_policy2 directly): continuous_policy2's eval is a
WARM profile-split (half the items known) with PROFILE-based answerability -> the known half
already reveals the user's group, answerability pre-routes to own-group concepts, and the
adaptivity headroom (the whole point of this world) vanishes (== the ML-1M "adaptivity flat"
finding). The synthetic world's +headroom EXISTS ONLY in COLD-START with TASTE-based
answerability: you must ASK to discover the group. So: same NDCG@10 metric + same model
lineup, run in the synthetic world's native cold-start regime. (Documented in the paper.)

WORLD: user in 1 of 2 GROUPS (4 clusters each); per-own-cluster like/dislike polarity;
rates own-group cluster movies (p=0.7, 5% noise). 96 items (8x12).
- ITEMS unanswerable in cold-start (unseen) -> asking one wastes a turn (answerability lever).
- CONCEPTS = 8 cluster-concepts (answerable IFF cluster in user's group: taste-based)
             + 1 GROUP-INDICATOR (always answerable; the adaptive lever).
- Adaptive optimum: ask indicator -> learn group -> ask the 4 own-group clusters. A STATIC
  policy wastes ~half its budget on wrong-group (unanswerable) concepts.

MODELS (same as ML-1M): prior, pop_item, random_concept, conc_pop, FAIR entropy
(most-divisive concept), oracle_adaptive (privileged greedy NDCG), static_best (no-adaptivity
ceiling), and OUR learned policy (BC-distill the entropy heuristic -> REINFORCE on NDCG reward
-- identical recipe to the ML-1M entropy-distill winner).
METRIC: NDCG@10 full + Recall@10. (Tail reported but ~=full: uniform popularity by design.)
"""
import os, json, numpy as np, torch, torch.nn as nn
SEED=int(os.environ.get('SEED',0)); rng=np.random.default_rng(SEED); torch.manual_seed(SEED)
NCl=8; MPC=12; NI=NCl*MPC; D=int(os.environ.get('D',48)); T=int(os.environ.get('T',4))   # T=4: oracle fills 4/4 own-group clusters; a STATIC seq must split 2+2 across both groups -> adaptivity headroom (loose T lets static cover both -> headroom collapses)
GROUPS={0:list(range(4)),1:list(range(4,8))}; CL_OF=np.repeat(np.arange(NCl),MPC)
GRP_OF=np.array([0 if CL_OF[m] in GROUPS[0] else 1 for m in range(NI)])
RATE_P=0.7; NOISE_P=0.05; NU=int(os.environ.get('NU',6000)); NTE=int(os.environ.get('NTE',600))
IND=NCl; NC=NCl+1; TAU=0.3                                            # concepts: 8 clusters + indicator(id=8)

def make_user(r):
    g=int(r.integers(2)); pol={c:int(r.integers(2)) for c in GROUPS[g]}   # 1=likes cluster, 0=dislikes
    rd={}
    for m in range(NI):
        c=int(CL_OF[m])
        if c in pol and r.random()<RATE_P:
            lab=pol[c]
            if r.random()<NOISE_P: lab=1-lab
            rd[m]=5.0 if lab else 1.0
    return g,pol,rd
print(f"[synth] generating {NU}+{NTE} users (seed {SEED})...",flush=True)
US=[make_user(rng) for _ in range(NU+NTE)]; TRU=list(range(NU)); TEU=list(range(NU,NU+NTE))
# ---- residual SVD item factors (mirror Q_svd) ----
R=np.zeros((NU,NI),np.float32); M=np.zeros((NU,NI),np.float32)
for u in TRU:
    for m,r in US[u][2].items(): R[u,m]=r; M[u,m]=1
mu=float(R[M>0].mean()); bi=np.where(M.sum(0)>0,(np.where(M>0,R-mu,0)).sum(0)/np.maximum(M.sum(0),1),0).astype(np.float32)
Rc=np.where(M>0,R-mu-bi[None],0).astype(np.float32)
_,S_,Vt=np.linalg.svd(Rc,full_matrices=False); Q=(Vt[:D].T*np.sqrt(np.maximum(S_[:D],0))).astype(np.float32)
if Q.shape[1]<D: Q=np.pad(Q,((0,0),(0,D-Q.shape[1])))
cnt=np.array([sum(1 for u in TRU if US[u][2].get(m,0)>=4) for m in range(NI)],np.float32); popb=np.log(cnt+1).astype(np.float32)
order=np.argsort(-cnt); cum=np.cumsum(cnt[order])/max(cnt.sum(),1); HEAD=set(int(j) for j in order[:np.searchsorted(cum,0.33)+1]); headm=np.zeros(NI,bool); headm[list(HEAD)]=True
# ---- concepts: cluster centroids + indicator (group-0 minus group-1 direction) ----
Ec=np.zeros((NC,D),np.float32)
for c in range(NCl): Ec[c]=Q[CL_OF==c].mean(0)
Ec[IND]=Q[GRP_OF==0].mean(0)-Q[GRP_OF==1].mean(0)
def resid(u): return {m:(r-mu-bi[m]) for m,r in US[u][2].items()}
def ustar_ridge(u):
    rd=US[u][2]
    if not rd: return np.zeros(D,np.float32)
    F=np.array([Q[m] for m in rd],np.float32); y=np.array([resid(u)[m] for m in rd],np.float32)
    return np.linalg.solve(F.T@F+5*np.eye(D),F.T@y).astype(np.float32)
UST={u:ustar_ridge(u) for u in TRU+TEU}
_allr=[resid(u)[m] for u in TRU[:2000] for m in US[u][2]]; _rr=np.array(_allr)
POS=float(_rr[_rr>0].mean()); NEG=float(_rr[_rr<0].mean())                 # fold answer magnitudes (resid scale), matches continuous_policy2
def tlike_of(u): return set(m for m,r in US[u][2].items() if r>=4)          # liked items = NDCG targets (cold-start: all held out)
# ---- TASTE-based answerability + STRUCTURAL geometric answer (we have ground-truth polarity) ----
def cans(u):                                                                # answerable concepts {c: fold-answer-value}
    g,pol,_=US[u]; ca={c:(POS if pol[c] else NEG) for c in GROUPS[g]}        # own-group clusters: like/dislike polarity
    ca[IND]=POS if g==0 else NEG; return ca                                  # indicator: reveals group (always answerable)
# population priors for the heuristics
ans_cnt=np.zeros(NC); like_cnt=np.zeros(NC)
for u in TRU:
    for c,v in cans(u).items(): ans_cnt[c]+=1; like_cnt[c]+=(1 if v>0 else 0)
cfreq=ans_cnt/len(TRU)                                                       # answerability frequency (conc_pop)
clr=np.where(ans_cnt>0,like_cnt/np.maximum(ans_cnt,1),0.); Hb=lambda p:-(p*np.log(p+1e-9)+(1-p)*np.log(1-p+1e-9))
CENT=Hb(clr).astype(np.float32)                                             # concept divisiveness (entropy heuristic signal)
OPENER=int(np.argmax(cfreq))                                                # cold-start opener = MOST-ANSWERABLE concept (the indicator, cfreq=1.0): over an empty belief you can't act on taste, so bootstrap with the universal probe. The LEARNED policy's adaptivity is the post-opener BRANCH (== ML-1M's "adaptive after a fixed opening"); fixes the turn-0 shared-state credit cancellation.
print(f"[synth] {NI} items, {NC} concepts | POS={POS:.2f} NEG={NEG:.2f} | most-divisive={int(CENT.argmax())} (H={CENT.max():.3f}) | opener(most-answerable)={OPENER}{' =INDICATOR' if OPENER==IND else ''} cfreq={cfreq[OPENER]:.2f}",flush=True)

# ================= concept-aware fold-in encoder (same arch as continuous_policy2 enc_concept) =================
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
ENCC=f'C:/dev/phd/casper/data/movielens/.cache/synth_enc_{SEED}.pt'; enc=Enc()
def enc_u(toks):                                                            # list[(emb,answer)] -> user factor
    if not toks: return np.zeros(D,np.float32)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_batch(revs):
    if not revs: return np.zeros((0,D),np.float32)
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
if os.path.exists(ENCC) and not os.environ.get('REGEN'):
    enc.load_state_dict(torch.load(ENCC)); print(f"[synth] loaded cached encoder {os.path.basename(ENCC)}",flush=True)
else:                                                                       # train: reconstruct ustar from a random subset of CONCEPT (+some item) reveals -> learns to fold concepts incl. the indicator
    print("[synth] fitting concept-aware fold-in encoder...",flush=True); opt=torch.optim.Adam(enc.parameters(),1e-3); trE=[u for u in TRU if US[u][2]]
    for ep in range(int(os.environ.get('ENCEP',14))):
        rng.shuffle(trE); tot=0.;nb=0
        for b0 in range(0,len(trE),256):
            us=trE[b0:b0+256]; B=len(us); ml=6; toks=torch.zeros(B,ml,D+1); tm=torch.zeros(B,ml); tgt=torch.zeros(B,D)
            for b,u in enumerate(us):
                ca=list(cans(u).items()); rng.shuffle(ca)                   # concept reveals (the eval channel)
                its=[(m,resid(u)[m]) for m in US[u][2]]; rng.shuffle(its)
                pool=[(Ec[c],v) for c,v in ca]+([ (Q[m],rv) for m,rv in its[:2]] if rng.random()<0.4 else [])  # mostly concepts, sometimes a couple items
                rng.shuffle(pool); k=min(len(pool),max(1,int(rng.integers(1,ml+1))))
                for q,(f,v) in enumerate(pool[:k]): toks[b,q,:D]=torch.tensor(f); toks[b,q,D]=v; tm[b,q]=1
                tgt[b]=torch.tensor(UST[u])
            u_=enc(toks,tm); loss=(1-torch.nn.functional.cosine_similarity(u_,tgt,dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss); nb+=1
        if ep%3==0 or ep==13: print(f"   enc ep{ep+1} recon {tot/nb:.4f}",flush=True)
    torch.save(enc.state_dict(),ENCC)
for p in enc.parameters(): p.requires_grad_(False); enc.eval()

# ================= metric =================
_W=1./np.log2(np.arange(2,12))
def ndcg_recall(u_vec,tlike,tail=False):
    s=(popb+Q@u_vec).copy()
    if tail: s[headm]=-1e9; rel=set(t for t in tlike if not headm[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-s); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    rc=len(set(int(t) for t in o[:10])&rel)/len(rel); return nd,rc

# ================= heuristic / oracle policies (cold-start) =================
def fold_concept(toks,u,c,ca):                                              # ask concept c; append token if answerable (else wasted turn)
    if c in ca: toks.append((Ec[c],ca[c]))
    return toks
def run_heuristic(mode,users,static_seq=None):
    nf=rf=nt=0.; m=0
    for u in users:
        ca=cans(u); tl=tlike_of(u); toks=[]; asked=set()
        if not tl: continue                                                 # ~6% all-dislike users have no NDCG target -> skip (same users skipped across modes)
        for t in range(T):
            if mode=='prior': break
            if mode=='pop_item': asked.add(('i',int(order[t]))); continue   # ask popular ITEM -> unanswerable in cold-start -> wasted
            if mode=='random_concept':
                cand=[c for c in range(NC) if c not in asked]; c=int(rng.choice(cand))
            elif mode=='conc_pop':
                cand=[c for c in range(NC) if c not in asked]; c=max(cand,key=lambda c:cfreq[c])
            elif mode=='entropy':                                           # FAIR entropy: most-divisive concept (population), blind to the user's group
                cand=[c for c in range(NC) if c not in asked]; c=max(cand,key=lambda c:CENT[c])
            elif mode=='static':                                            # fixed sequence for everyone (no adaptivity)
                c=static_seq[t]
            elif mode=='oracle':                                            # privileged greedy: pick the answerable concept maximizing this user's NDCG
                cand=[c for c in ca if c not in asked]
                if not cand: c=-1
                else:
                    best=-1.;c=cand[0]
                    for cc in cand:
                        nd=ndcg_recall(enc_u(toks+[(Ec[cc],ca[cc])]),tl,False)[0]
                        if nd>best: best=nd; c=cc
            if c==-1 or c in asked: asked.add(('skip',t)); continue
            asked.add(c); fold_concept(toks,u,c,ca)
        uv=enc_u(toks); r=ndcg_recall(uv,tl,False)
        if r: nf+=r[0]; rf+=r[1]; m+=1
        rt=ndcg_recall(uv,tl,True)
        if rt: nt+=rt[0]
    return nf/max(m,1), rf/max(m,1), nt/max(m,1)
def best_static_seq():                                                      # greedy forward-select the best FIXED concept sequence (no-adaptivity ceiling) on a train subval
    sub=TRU[:1500]; seq=[]
    for _ in range(T):
        best=-1.; bc=0
        for c in range(NC):
            if c in seq: continue
            tot=0.;m=0
            for u in sub:
                ca=cans(u); toks=[]
                for cc in seq+[c]: fold_concept(toks,u,cc,ca)
                r=ndcg_recall(enc_u(toks),tlike_of(u),False)
                if r: tot+=r[0]; m+=1
            v=tot/max(m,1)
            if v>best: best=v; bc=c
        seq.append(bc)
    return seq

# ================= OUR learned policy: BC-distill entropy -> REINFORCE on NDCG (same recipe as ML-1M) =================
Ect=torch.tensor(Ec); CENTt=torch.tensor(((CENT-CENT.mean())/(CENT.std()+1e-6)).astype(np.float32)); CFt=torch.tensor(((cfreq-cfreq.mean())/(cfreq.std()+1e-6)).astype(np.float32))
class Scorer(nn.Module):                                                    # per-concept score from [emb, align(u.E), pop-prior, divisiveness-prior, turn]
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+4,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(s,u,tt):                                                    # u (B,D), tt scalar OR (B,) -> (B,NC)
        B=u.shape[0]; emb=Ect.unsqueeze(0).expand(B,NC,D); align=(u@Ect.t()).unsqueeze(2)
        pri=CFt.view(1,NC,1).expand(B,NC,1); ent=CENTt.view(1,NC,1).expand(B,NC,1)
        tn=(tt.view(B,1,1).expand(B,NC,1).float() if torch.is_tensor(tt) else torch.full((B,NC,1),float(tt)))
        return s.f(torch.cat([emb,align,pri,ent,tn],2)).squeeze(2)
def train_policy():
    sc=Scorer(); opt=torch.optim.Adam(sc.parameters(),1e-3)
    # --- BC floor: distill a heuristic teacher (BCTGT=entropy default | oracle = privileged greedy-NDCG branch) ---
    BCTGT=os.environ.get('BCTGT','entropy')
    print(f"[synth] BC floor: distilling {BCTGT} teacher...",flush=True)
    Sb=[];Tb=[];TT=[]
    for u in TRU[:2500]:
        ca=cans(u); tl=tlike_of(u); toks=[]; asked=set()
        if not tl: continue
        order_ent=sorted(range(NC),key=lambda c:-CENT[c])
        for t in range(T):
            if t==0: cc=OPENER                                              # fixed answerability-bootstrap opener (so turns>=1 distil from the post-opener, group-revealed belief)
            elif BCTGT=='oracle':                                            # ORACLE teacher: greedy answerable concept maximizing THIS user's NDCG (privileged; teaches the own-group branch -> hits the privileged gap at cold turn-0)
                cand=[c for c in ca if c not in asked]
                if not cand: break
                cc=max(cand,key=lambda c:ndcg_recall(enc_u(toks+[(Ec[c],ca[c])]),tl,False)[0])
            else:
                cc=next((c for c in order_ent if c not in asked),None)
                if cc is None: break
            Sb.append(enc_u(toks)); Tb.append(cc); TT.append(t/8.); asked.add(cc); fold_concept(toks,u,cc,ca)
    Sb=torch.tensor(np.array(Sb,np.float32)); Tb=torch.tensor(Tb).long(); TT=torch.tensor(np.array(TT,np.float32))
    for ep in range(int(os.environ.get('BCEP',5))):                          # SOFT floor (5 ep): a hard floor (loss~0) -> near-deterministic CONSTANT sequence (entropy is degenerate here: all concepts ~equally divisive) -> REINFORCE can't explore -> collapse
        idx=torch.randperm(len(Sb)); tl=0.;nb=0
        for b0 in range(0,len(Sb),256):
            bb=idx[b0:b0+256]; out=sc(Sb[bb],TT[bb]); loss=nn.functional.cross_entropy(out,Tb[bb]); opt.zero_grad(); loss.backward(); opt.step(); tl+=float(loss); nb+=1
        if ep%2==0: print(f"   BC ep{ep+1} loss {tl/nb:.3f}",flush=True)
    # --- REINFORCE on per-turn delta-NDCG reward (dense); ENTROPY BONUS for exploration (the ML-1M ENT_COEF) so RL can DISCOVER the branch beyond the entropy floor ---
    print("[synth] REINFORCE refine on NDCG reward (entropy-bonus exploration)...",flush=True)
    for g in opt.param_groups: g['lr']=float(os.environ.get('FTLR',5e-4))
    trp=[u for u in TRU if tlike_of(u)][:int(os.environ.get('RLUSERS',3000))]; RLTAU=float(os.environ.get('RLTAU',0.8)); EC=float(os.environ.get('ENT_COEF',0.04)); PEN=float(os.environ.get('PEN',0.06))   # PEN: penalty for an UNANSWERABLE (wasted) ask -> indicator (always answerable) becomes the turn-0 move; after it reveals the group, wrong-group asks are penalised => the policy learns to BRANCH (fixes turn-0 group-blindness credit cancellation)
    for ep in range(int(os.environ.get('RLEP',15))):
        rng.shuffle(trp); tot=0.;nb=0;hh=0.
        for b0 in range(0,len(trp),64):
            us=trp[b0:b0+64]; B=len(us); CAS=[cans(u) for u in us]; TLS=[tlike_of(u) for u in us]
            toks=[[] for _ in range(B)]; asked=[set() for _ in range(B)]; logps=[]; rews=[]; ents=[]; prev=np.zeros(B)
            for t in range(T):
                uv=np.array([enc_u(toks[b]) for b in range(B)],np.float32); out=sc(torch.tensor(uv),t/8.)
                mask=torch.zeros(B,NC)
                for b in range(B):
                    for c in asked[b]: mask[b,c]=-1e9
                p=torch.softmax((out+mask)/RLTAU,1)
                if t==0:                                                     # FIXED opener: no gradient on the cold turn (avoids shared-state credit cancellation); the learned adaptivity is turns>=1
                    a=torch.full((B,),OPENER,dtype=torch.long); logps.append(torch.zeros(B)); ents.append(torch.tensor(0.))
                else:
                    a=torch.multinomial(p,1).squeeze(1); logps.append(torch.log(p[torch.arange(B),a]+1e-9)); ents.append(-(p*torch.log(p+1e-9)).sum(1).mean())  # per-turn policy entropy -> bonus keeps the sampler exploratory
                rew=np.zeros(B,np.float32)
                for b in range(B):
                    c=int(a[b]); asked[b].add(c); un=0.
                    if c in CAS[b]: toks[b].append((Ec[c],CAS[b][c]))
                    else: un=1.                                              # wasted (unanswerable) ask
                    r=ndcg_recall(enc_u(toks[b]),TLS[b],False); nd=r[0] if r else 0.
                    rew[b]=(nd-prev[b])-PEN*un; prev[b]=nd                   # shaped reward; prev tracks clean NDCG (penalty only shapes the gradient)
                rews.append(torch.tensor(rew))
            loss=0.; acc=torch.zeros(B)
            for t in reversed(range(T)): acc=rews[t]+acc; adv=acc-acc.mean(); loss=loss-(logps[t]*adv).mean()
            loss=loss/T-EC*(sum(ents)/T); opt.zero_grad(); loss.backward(); opt.step(); tot+=float(prev.mean()); nb+=1; hh+=float(sum(ents)/T)
        print(f"   RL ep{ep+1} mean-final-NDCG {tot/nb:.4f}  H(pi)={hh/nb:.3f}",flush=True)
    return sc
def run_policy(sc,users):
    nf=rf=nt=0.;m=0
    for u in users:
        ca=cans(u); tl=tlike_of(u); toks=[]; asked=set()
        if not tl: continue
        for t in range(T):
            if t==0: c=OPENER                                               # fixed answerability-bootstrap opener; argmax branch from turn 1
            else:
                with torch.no_grad(): out=sc(torch.tensor(enc_u(toks)[None]),t/8.).numpy()[0]
                c=next((int(k) for k in np.argsort(-out) if int(k) not in asked),None)
            if c is None: break
            asked.add(c); fold_concept(toks,u,c,ca)
        uv=enc_u(toks); r=ndcg_recall(uv,tl,False)
        if r: nf+=r[0]; rf+=r[1]; m+=1
        rt=ndcg_recall(uv,tl,True)
        if rt: nt+=rt[0]
    return nf/max(m,1), rf/max(m,1), nt/max(m,1)

# ================= run all models on the SAME ruler =================
print("\n[synth] === cold-start elicitation, NDCG@10 over %d test users (T=%d) ==="%(len(TEU),T),flush=True)
RES={}
for mode in ['prior','pop_item','random_concept','conc_pop','entropy']:
    RES[mode]=run_heuristic(mode,TEU)
sseq=best_static_seq(); RES['static_best']=run_heuristic('static',TEU,static_seq=sseq); print(f"[synth] best-static concept seq = {sseq}",flush=True)
RES['oracle_adaptive']=run_heuristic('oracle',TEU)
_BCTGT=os.environ.get('BCTGT','entropy'); PKEY='policy_ours' if _BCTGT=='entropy' else f'policy_{_BCTGT}distill'
sc=train_policy(); RES[PKEY]=run_policy(sc,TEU)
print("\n%-18s %8s %8s %8s"%('model','NDCG@10','tailND','Rec@10'),flush=True)
for k in ['prior','pop_item','random_concept','conc_pop','entropy','static_best',PKEY,'oracle_adaptive']:
    nf,rf,nt=RES[k]; print("%-18s %8.4f %8.4f %8.4f"%(k,nf,nt,rf),flush=True)
hl=RES['oracle_adaptive'][0]-RES['static_best'][0]; pg=RES[PKEY][0]-RES['static_best'][0]
print(f"\n[synth] ADAPTIVITY HEADROOM (oracle_adaptive - static_best) = +{hl:.4f} NDCG@10",flush=True)
print(f"[synth] {PKEY} - static_best = +{pg:.4f}; captures {100*pg/max(hl,1e-6):.0f}% of the oracle headroom over the best STATIC policy",flush=True)
_sfx='' if _BCTGT=='entropy' else f'_{_BCTGT}'; out=f'C:/dev/phd/casper/experiments/paper2/synth_ndcg{_sfx}_seed{SEED}.json'; os.makedirs(os.path.dirname(out),exist_ok=True)
json.dump({'seed':SEED,'T':T,'NU':NU,'NTE':NTE,'results':{k:{'ndcg':v[0],'tail':v[2],'recall':v[1]} for k,v in RES.items()},
    'headroom':hl,'policy_gain_over_static':pg,'policy_key':PKEY,'static_seq':[int(x) for x in sseq]},open(out,'w'),indent=2)
print(f"[synth] saved -> {out}",flush=True)
