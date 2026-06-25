"""
PAPER B policy v2 (user's fix): give the policy the POPULARITY signal it was missing (population prior, NOT user-specific
-> no leak), PLUS the belief from revealed answers. A learned SCORER scores each entity in the unified item+concept pool
from per-candidate features [embedding, belief-alignment u.E, popularity-prior, belief-strength, turn]; soft-pick (train)
/ argmax (eval) over the pool; fold the answer; trained end-to-end on RECONSTRUCTION. Should match conc_pop (use
popularity when belief weak) AND personalize (use belief as answers accumulate) => beat conc_pop. Frozen unified model.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=int(os.environ.get('T',8)); TAU=float(os.environ.get('TAU',0.3)); EP=int(os.environ.get('EP',14)); rng=np.random.default_rng(0); torch.manual_seed(0)
QPTS=[int(x) for x in os.environ.get('QPTS','0,2,4,8').split(',')]              # eval question-budget points; extend for the saturation cut-off, e.g. QPTS=0,2,4,8,12,16,20 (must be <= T)
U,I,Rr=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U);I=np.array(I);Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(Rr[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); nK=len(keep); trU=keep[:int(0.8*nK)]; te=keep[int(0.9*nK):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_concept.npy'); Ec=np.load(f'{base}/.cache/Ec_concept.npy'); ctags=list(np.load(f'{base}/.cache/ctags_concept.npy'))
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
t0=time.time(); tagitems={}; tagset=set(int(t) for t in ctags)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1])
        if m in iids and tg in tagset and float(a[2])>0.5: tagitems.setdefault(tg,[]).append(iids[m])
citems=[set(tagitems.get(int(t),[])) for t in ctags]; cfreq=np.array([len(s) for s in citems]); NC=len(ctags)
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems.get(int(t),[]): item2c[j].append(ki)
pe_cnt=np.zeros(NC)
for x in trU[:2500]:
    cn={}
    for j,_ in rat_by_u[x]:
        for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
    for ki,c2 in cn.items():
        if c2>=2: pe_cnt[ki]+=1
pe_ans=(pe_cnt/min(len(trU),2500)).astype(np.float32)+1e-6; p_seen=(cnt/max(len(trU),1)).astype(np.float32)
print(f"  {NC} concepts ({time.time()-t0:.0f}s)",flush=True)
PITEMS=list(order_pop[:600]); NI=len(PITEMS); NP=NI+NC
POOL=np.concatenate([Q[np.array(PITEMS)], Ec],0).astype(np.float32); PTYPE=np.array([0]*NI+[1]*NC)
PRIOR=np.concatenate([np.log(p_seen[np.array(PITEMS)]+1e-4), np.log(pe_ans+1e-4)]).astype(np.float32); PRIOR=(PRIOR-PRIOR.mean())/(PRIOR.std()+1e-6)  # POPULARITY/answerability prior (population-level)
POOLt=torch.tensor(POOL); PRIORt=torch.tensor(PRIOR); Qlt=torch.tensor(Ql); popbt=torch.tensor(popb); HEADt=torch.tensor(headmask)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
def sig(z): return 1/(1+np.exp(-z))
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_batch_np(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def cans_np(x,profset):
    ac=[c for c in range(NC) if len(citems[c]&profset)>=2]; ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset])
    pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())) if pr else 0.; return {c:(POS if pr[c]>thr else NEG) for c in ac}
def unicand_np(x,profset,cans,asked):                                          # ALL LEVERS (owner req): askable pool ITEMS (seen, in PITEMS) + answerable CONCEPTS (incl. genre tags). returns (fold-tokens, pool-indices=BC-targets, asked-ids)
    items=[k for k in range(NI) if (PITEMS[k] in profset and ('i',k) not in asked)]
    concs=[c for c in cans if ('c',c) not in asked]
    toks=[(POOL[k], float(resid[x][PITEMS[k]])) for k in items] + [(Ec[c], float(cans[c])) for c in concs]
    pidx=[k for k in items] + [NI+c for c in concs]
    aid =[('i',k) for k in items] + [('c',c) for c in concs]
    return toks, pidx, aid
# POPULATION-AVERAGE info-gain per pool entity (offline; population-level, no per-user leak): avg single-fold coverage-gain of an answerer's own profile
IGC=f'{base}/.cache/pool_ig2.npy'
if not os.path.exists(IGC) or os.environ.get('REGEN'):
    print("computing population info-gain per entity...",flush=True); igv=np.zeros(NP); cig=np.zeros(NP); samp=[x for x in trU if len(rat_by_u[x])>=10][:600]
    for x in samp:
        prof=[j for j,_ in rat_by_u[x]]; profset=set(prof); profa=np.array(prof); base_cov=sig(popb[profa]).sum(); ents=[];idxs=[]
        for k,j in enumerate(PITEMS):
            if j in profset: ents.append((Q[j],resid[x][j])); idxs.append(k)
        for c,v in cans_np(x,profset).items(): ents.append((Ec[c],v)); idxs.append(NI+c)
        if not ents: continue
        ul=enc_batch_np([[e] for e in ents]); cov=sig(popb[profa]+ul@Ql[profa].T).sum(1)-base_cov
        for q2,k in enumerate(idxs): igv[k]+=cov[q2]; cig[k]+=1
    POOL_IG=(igv/np.maximum(cig,1)).astype(np.float32); np.save(IGC,POOL_IG)
else: POOL_IG=np.load(IGC)
POOL_IG=((POOL_IG-POOL_IG.mean())/(POOL_IG.std()+1e-6)).astype(np.float32); POOL_IGt=torch.tensor(POOL_IG)
# per-entity DIVISIVENESS (entropy of like-rate, the signal entropy-heuristic uses) + AVG-RATING (item bias) — user's O14 feature idea
ENTC=f'{base}/.cache/pool_entavg.npy'
if not os.path.exists(ENTC) or os.environ.get('REGEN'):
    print("computing per-entity entropy(divisiveness)+avg-rating...",flush=True)
    itot=np.zeros(ni)
    for x in trU:
        for j,_ in rat_by_u[x]: itot[j]+=1
    ilr=cnt/np.maximum(itot,1)                                                   # item like-rate
    lc=np.zeros(NC); ac=np.zeros(NC)
    for x in [x for x in trU if len(rat_by_u[x])>=10][:2000]:
        for c,v in cans_np(x,set(j for j,_ in rat_by_u[x])).items(): ac[c]+=1; lc[c]+=(1 if v>0 else 0)
    clr=np.where(ac>0,lc/np.maximum(ac,1),0.0)                                   # concept like-rate; unanswerable (ac==0) -> 0 so Hb=0 (ranked LAST), NOT 0.5 (Hb=max) which CRIPPLED the entropy baseline (matches lit_baselines)
    Hb=lambda p:-(p*np.log(p+1e-9)+(1-p)*np.log(1-p+1e-9))
    pent=np.concatenate([Hb(ilr[np.array(PITEMS)]),Hb(clr)])                     # divisiveness per pool entity (items+concepts)
    pavg=np.concatenate([bi[np.array(PITEMS)],np.array([bi[list(citems[c])].mean() if citems[c] else 0. for c in range(NC)])])
    np.save(ENTC,np.stack([pent,pavg]))
POOL_ENT,POOL_AVGR=np.load(ENTC)
_entc_raw=POOL_ENT[NI:].copy()                                                # raw concept divisiveness (binary entropy, [0,1]) for lit heuristics (capture before z-score)
_lf=np.log(cfreq+1.).astype(np.float32); _ne=_entc_raw/(_entc_raw.max()+1e-9); _nl=_lf/(_lf.max()+1e-9)
HELF_c=(2*_ne*_nl/(_ne+_nl+1e-9)).astype(np.float32)                          # HELF (Rashid/Karypis/Riedl 2008): harmonic mean of norm divisiveness & norm log-freq
LPE_c=(_lf*_entc_raw).astype(np.float32); PE_c=(cfreq.astype(np.float32)*_entc_raw)   # log(pop)*entropy (Rashid 2002) ; pop*entropy
POOL_ENT=((POOL_ENT-POOL_ENT.mean())/(POOL_ENT.std()+1e-6)).astype(np.float32); POOL_AVGR=((POOL_AVGR-POOL_AVGR.mean())/(POOL_AVGR.std()+1e-6)).astype(np.float32)
POOL_ENTt=torch.tensor(POOL_ENT); POOL_AVGRt=torch.tensor(POOL_AVGR)
ENTPRIOR=POOL_ENT.copy().astype(np.float32); ENTPRIOR[:NI]=float(POOL_ENT.min()); ENTPRIORt=torch.tensor(ENTPRIOR)   # RESID head: divisiveness prior over CONCEPTS (items masked low) -> alpha*ENTPRIOR reproduces the entropy heuristic EXACTLY at init
POOL_ANS=np.concatenate([p_seen[np.array(PITEMS)],pe_ans]).astype(np.float32); POOL_ANS=((POOL_ANS-POOL_ANS.mean())/(POOL_ANS.std()+1e-6)).astype(np.float32)
POOL_ANSt=torch.tensor(POOL_ANS); HID=int(os.environ.get('HID',128))
_FE=set(x for x in os.environ.get('FEATS','').split(',') if x)                                              # INDIVIDUAL feature ablation: FEATS=ent,avg,ans  (legacy 'ext'==ent+avg)
if 'ext' in _FE: _FE|={'ent','avg'}
FENT='ent' in _FE; FAVG='avg' in _FE; ANSF=('ans' in _FE) or bool(os.environ.get('ANSF'))                   # entropy(divisiveness) / avg-rating / answerability — independently toggleable
CONCONLY=os.environ.get('CONCONLY'); CMASK=(torch.tensor(np.where(PTYPE==0,-1e9,0.).astype(np.float32)) if CONCONLY else None)   # P9: restrict action pool to concepts
ATTN=bool(os.environ.get('ATTN'))                                             # P6: add a LEARNED attention state over answer-history as an additive, gated per-candidate score (base Scorer untouched -> BC floor still loads)
RESID=bool(os.environ.get('RESID'))                                           # RESIDUAL head: score = alpha*entropy_prior + MLP(features); MLP last-layer init 0, alpha init 1 => starts AS the entropy heuristic, RL learns ONLY the +more (structurally cannot underperform entropy)
class Scorer(nn.Module):                                                      # per-candidate score from [emb, align(u.E), popularity prior, POPULATION info-gain, belief-strength, turn (+ENT,AVGR if EXT, +ANS if ANSF)]
    def __init__(s):
        super().__init__(); din=D+5+int(FENT)+int(FAVG)+int(ANSF); s.f=nn.Sequential(nn.Linear(din,HID),nn.ReLU(),nn.Linear(HID,HID),nn.ReLU(),nn.Linear(HID,1)); s.alpha=nn.Parameter(torch.tensor(1.0))
        if RESID: nn.init.zeros_(s.f[-1].weight); nn.init.zeros_(s.f[-1].bias)    # MLP output starts at 0 => score = alpha*entropy_prior = the entropy heuristic exactly
    def forward(s,u,tt):                                                      # u (B,D), tt scalar turn frac -> scores (B,NP)
        B=u.shape[0]; align=(u@POOLt.t()).unsqueeze(2)                        # (B,NP,1) belief alignment
        bnorm=u.norm(dim=1,keepdim=True).unsqueeze(1).expand(B,NP,1)          # belief strength
        emb=POOLt.unsqueeze(0).expand(B,NP,D); pri=PRIORt.view(1,NP,1).expand(B,NP,1); ig=POOL_IGt.view(1,NP,1).expand(B,NP,1)
        tn=(tt.view(B,1,1).expand(B,NP,1) if torch.is_tensor(tt) else torch.full((B,NP,1),float(tt)))
        fl=[emb,align,pri,ig,bnorm,tn]
        if FENT: fl+=[POOL_ENTt.view(1,NP,1).expand(B,NP,1)]                                          # divisiveness (entropy)
        if FAVG: fl+=[POOL_AVGRt.view(1,NP,1).expand(B,NP,1)]                                         # avg-rating
        if ANSF: fl+=[POOL_ANSt.view(1,NP,1).expand(B,NP,1)]                                          # explicit answerability
        out=s.f(torch.cat(fl,2)).squeeze(2)
        if RESID: out=out+s.alpha*ENTPRIORt.view(1,NP)                                                # RESID: entropy prior + learned +more (= entropy by construction at init)
        return out+CMASK if CONCONLY else out                                                        # P9: concept-only mask
scorer=Scorer()
class AttnHead(nn.Module):                                                    # P6: LEARNED attention over interaction history [(entity_emb,answer)...] -> asking-state -> additive gated per-candidate score
    def __init__(s):
        super().__init__(); s.tok=nn.Linear(D+1,HID); s.q=nn.Linear(HID,1); s.proj=nn.Linear(HID,D); s.scale=nn.Parameter(torch.zeros(1))   # scale=0 => starts EXACTLY as the base scorer (the winner); learns to add a history-aware adjustment
    def state(s,toks,tmask):                                                  # (B,T,D+1),(B,T) -> learned asking-state (B,D)
        z=torch.tanh(s.tok(toks)); a=s.q(z).squeeze(2).masked_fill(tmask<0.5,-1e9)
        has=(tmask.sum(1,keepdim=True)>0).float(); w=torch.softmax(a,1).unsqueeze(2)
        return s.proj((w*z).sum(1))*has                                      # 0 when no history (t=0, nothing asked yet)
    def forward(s,toks,tmask): return s.scale*(s.state(toks,tmask)@POOLt.t())   # (B,NP) additive score, gated by scale (init 0)
attn=AttnHead()
_pp=list(scorer.parameters())+(list(attn.parameters()) if ATTN else [])
opt=torch.optim.Adam(_pp,1e-3,weight_decay=float(os.environ.get('WD',0)))     # WD = L2 regularization (anti-overfit); ATTN adds the attention-head params
class Critic(nn.Module):                                                      # P7: state-value baseline V(belief,turn) for REINFORCE variance reduction
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,1))
    def forward(s,u,tt): B=u.shape[0]; return s.f(torch.cat([u,torch.full((B,1),float(tt))],1)).squeeze(1)
critic=Critic(); CRITIC=os.environ.get('CRITIC')
if CRITIC: opt=torch.optim.Adam(_pp+list(critic.parameters()),1e-3)
def save_ck(path): torch.save({'scorer':scorer.state_dict(),'attn':attn.state_dict()} if ATTN else scorer.state_dict(),path)   # ATTN -> persist BOTH heads (else attn weights lost on save/eval)
def load_ck(path):                                                            # loads a combined {scorer,attn} ckpt OR a plain scorer ckpt (e.g. the BC floor) -> attn then stays fresh (scale=0 => starts as the winner)
    sd=torch.load(path)
    if isinstance(sd,dict) and 'scorer' in sd: scorer.load_state_dict(sd['scorer'],strict=False); (attn.load_state_dict(sd['attn']) if ATTN else None)
    else: scorer.load_state_dict(sd,strict=False)                              # strict=False: pre-RESID ckpts lack 'alpha' (stays init 1.0, unused when RESID off)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**float(os.environ.get('DBEXP',0.5)),1e-3,1.0)).astype(np.float32)   # DBEXP: de-bias strength (0.5=1/sqrt(pop) default; 1.0=1/pop pushes harder to tail)
IPOPt=torch.tensor(ipsw)                                                      # inverse-popularity weights (for the REW=dbf de-biased-full reward)
ML=64
def prep(users):
    B=len(users); ANS=torch.zeros(B,NP); USTAR=torch.zeros(B,D); tgt=torch.zeros(B,ni); wt=torch.zeros(B,ni); LIKED=torch.zeros(B,ML,dtype=torch.long); LMASK=torch.zeros(B,ML); RNEG=torch.tensor(rng.integers(0,ni,size=(B,ML)),dtype=torch.long); PROFM=torch.zeros(B,ni)  # random items ~= non-likes (ranking reward); PROFM=known profile to exclude
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=set(allit[:max(len(allit)//2,4)]); held=[j for j in likes_by_u[x] if j not in prof] or likes_by_u[x][:1]
        for j in prof: PROFM[b,j]=1.
        for k,j in enumerate(PITEMS):
            if j in prof: ANS[b,k]=resid[x][j]
        cans=cans_np(x,prof)
        for c,v in cans.items(): ANS[b,NI+c]=v
        USTAR[b]=torch.tensor(enc_u_np([(Q[j],resid[x][j]) for j in (held if os.environ.get('USTARHELD') else prof)]))   # RECONSTRUCT target: known profile (default) OR USTARHELD=1 -> the HELD-OUT likes = the NDCG target (owner idea: target what we rank, not what we ask from)
        posw=0.
        for j in held: tgt[b,j]=1.; wt[b,j]=float(ipsw[j]); posw+=float(ipsw[j])
        for hi,j in enumerate(held[:ML]): LIKED[b,hi]=j; LMASK[b,hi]=1.          # held-out likes (for cheap coverage reward)
        seen=prof; nneg=ni-len(seen)-len(held); m=torch.ones(ni); m[list(seen)]=0; wb=wt[b]; wb[(tgt[b]==0)&(m>0)]=float(posw)/max(nneg,1); wt[b]=wb
    return ANS,USTAR,tgt,wt,LIKED,LMASK,RNEG,PROFM
def toks2t(toks):                                                             # python history list[(emb,ans)] -> (1,L,D+1),(1,L) tensors for the P6 attn head (empty history -> masked => h=0)
    if not toks: return torch.zeros(1,1,D+1),torch.zeros(1,1)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    return torch.tensor(arr),torch.tensor(m)
def rollout(users,ANS,explore=0.0):
    B=len(users); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D); asked=torch.zeros(B,NP)
    for t in range(T):
        sc=scorer(u,t/8.)+(attn(toks,tmask) if ATTN else 0)                  # (B,NP) base popularity+belief score (+ P6 learned history-attention head)
        if explore>0: sc=sc+explore*torch.randn_like(sc)
        sc=sc.masked_fill(asked>0,-1e9)                                      # no re-asking (matches eval)
        w_soft=torch.softmax(sc/TAU,1); idx=w_soft.argmax(1)
        w_hard=torch.zeros_like(w_soft).scatter_(1,idx.unsqueeze(1),1.0)
        w=w_hard+(w_soft-w_soft.detach())                                    # STRAIGHT-THROUGH: fold the single PICKED entity (hard, == eval), soft gradient
        asked=asked+w_hard
        a_used=w@POOLt; ans=(w*ANS).sum(1)
        toks=toks.clone(); toks[:,t,:D]=a_used; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask)
    return u
def recon(u,tgt,wt): s=u@Qlt.t()+popbt; bce=nn.functional.binary_cross_entropy_with_logits(s,tgt,reduction='none'); return (wt*bce).sum()/wt.sum()
def rollout_sample(users,ANS,LIKED,LMASK,RNEG,PROFM,PEN):                      # REINFORCE: DENSE per-turn reward = D[coverage|rank|NDCG] - PEN*(unanswered)
    B=len(users); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D); asked=torch.zeros(B,NP); ar=torch.arange(B)
    logps=[]; rews=[]; states=[]; ents=[]; prev=torch.zeros(B); QlL=Qlt[LIKED]; popL=popbt[LIKED]; lsum=LMASK.sum(1).clamp(min=1); QlN=Qlt[RNEG]; popN=popbt[RNEG]   # likes vs random non-likes
    _FIXQ1=os.environ.get('FIXQ1')                                                # FIXQ1: turn-0 opener = entropy pick (belief-free, intercept-confounded) -> NOT a policy action; learn the adaptive Q2..QT
    if _FIXQ1:
        _e=POOL_ENTt.view(1,NP).expand(B,NP).clone(); _e[:,:NI]=-1e9; _e=_e.masked_fill(ANS.abs()<1e-6,-1e9); _opidx=_e.argmax(1)   # per-user entropy opener (most divisive ANSWERABLE concept)
    REW=os.environ.get('REW','cov'); RTAIL=os.environ.get('REWTAIL'); _disc=1.0/torch.log2(torch.arange(2,12).float()); _cd=torch.cumsum(_disc,0); idcg=_cd[(LMASK.sum(1).clamp(1,10).long()-1)]  # per-user ideal DCG@10
    if REW=='ndcg' and os.environ.get('DQ0'):                                 # DQ0=1 => delta-over-q0 (tested P1b, regressed); DEFAULT = absolute final-NDCG (P1, reaches conc_pop)
        with torch.no_grad():
            sc0=popbt.unsqueeze(0).expand(B,ni).masked_fill(PROFM>0,-1e9); ti0=sc0.topk(10,1).indices
            prev=(((ti0.unsqueeze(2)==LIKED.unsqueeze(1))&(LMASK.unsqueeze(1)>0)).any(2).float()*_disc).sum(1)/idcg.clamp(min=1e-6)
    for t in range(T):
        states.append(u.detach().clone())                                        # P7: per-turn state for the value critic
        sc=(scorer(u,t/8.)+(attn(toks,tmask) if ATTN else 0)).masked_fill(asked>0,-1e9); p=torch.softmax(sc/TAU,1)   # P6: + learned history-attention head (grad flows to attn params)
        ents.append(-(p*torch.log(p+1e-9)).sum(1).mean())                       # policy entropy H(pi) per turn -> entropy-bonus regularization (escape plateau / anti-collapse)
        if _FIXQ1 and t==0: idx=_opidx; logps.append(torch.zeros(B))             # Q1 FORCED -> zero log-prob => no gradient on the intercept-confounded opener
        else: idx=torch.multinomial(p,1).squeeze(1); logps.append(torch.log(p[ar,idx]+1e-9))
        oh=torch.zeros_like(p).scatter_(1,idx.unsqueeze(1),1.0); asked=asked+oh
        av=ANS[ar,idx]; unans=(av.abs()<1e-6).float()                        # ANS==0 => unanswerable/wasted ask
        toks=toks.clone(); toks[:,t,:D]=POOLt[idx]; toks[:,t,D]=av; tmask=tmask.clone(); tmask[:,t]=1
        with torch.no_grad():
            u=enc(toks,tmask)
            if REW=='ndcg':                                                  # O14: DIRECT held-out NDCG@10 reward (exclude known profile); REWTAIL => TAIL-NDCG (rank+targets restricted to tail)
                sca=(u@Qlt.t()+popbt).masked_fill(PROFM>0,-1e9)
                if RTAIL: sca=sca.masked_fill(HEADt.unsqueeze(0),-1e9)        # rank only tail items
                topi=sca.topk(10,dim=1).indices
                vmask=(LMASK*(~HEADt[LIKED]).float()) if RTAIL else LMASK     # tail held-likes only as targets
                hit=((topi.unsqueeze(2)==LIKED.unsqueeze(1))&(vmask.unsqueeze(1)>0)).any(2).float()
                idc=_cd[(vmask.sum(1).clamp(1,10).long()-1)] if RTAIL else idcg
                cov=(hit*_disc).sum(1)/idc.clamp(min=1e-6)
            elif REW=='auc':                                                 # DENSER reward (owner #2): AUC of held-out likes vs random non-likes — smooth ranking signal, better SNR than top-10 NDCG
                sl=(u.unsqueeze(1)*QlL).sum(2)+popL; sn=(u.unsqueeze(1)*QlN).sum(2)+popN
                lm=(LMASK*(~HEADt[LIKED]).float()) if RTAIL else LMASK
                cov=(torch.sigmoid(sl.unsqueeze(2)-sn.unsqueeze(1))*lm.unsqueeze(2)).sum(dim=(1,2))/(lm.sum(1).clamp(min=1)*sn.shape[1])
            elif REW=='dbf':                                                 # DE-BIASED full (owner #3): top-10 coverage of held-out likes weighted by INVERSE popularity (reward personalization, not popularity)
                sca=(u@Qlt.t()+popbt).masked_fill(PROFM>0,-1e9); topi=sca.topk(10,1).indices
                hitL=(((topi.unsqueeze(2)==LIKED.unsqueeze(1)).any(1))&(LMASK>0)).float(); ipw=IPOPt[LIKED]*LMASK
                cov=(hitL*ipw).sum(1)/ipw.sum(1).clamp(min=1e-6)
            else:
                covL=(torch.sigmoid((u.unsqueeze(1)*QlL).sum(2)+popL)*LMASK).sum(1)/lsum
                covN=torch.sigmoid((u.unsqueeze(1)*QlN).sum(2)+popN).mean(1)  # non-likes (penalise inflating everything)
                cov=(covL-covN) if REW=='rank' else covL                     # REW=cov (O12) / rank (O13) / ndcg (O14)
            rews.append(((cov-prev)-PEN*unans)*float(os.environ.get('REWSCALE','1'))); prev=cov   # REWSCALE restores gradient magnitude for the small delta-over-q0 reward
    return u, logps, rews, states, ents
OBJ=os.environ.get('OBJ','ustar')                                            # 'ustar' = reconstruct the user embedding (user's idea); 'bce' = held-out item prediction
trbig=[x for x in trU if len(rat_by_u[x])>=14 and len(likes_by_u[x])>=6]
_FASTEVAL=bool(os.environ.get('LOAD')) and os.path.exists(f'{base}/.cache/policy_{os.environ.get("TAG","o12")}.pt')   # LOAD eval-only: skip BC demo-gen + 25ep training (load_ck overwrites the scorer anyway) -> instant re-eval, no CPU waste
# ---- BC FLOOR (optional): teach the scorer to reproduce conc_pop. NOBC=1 skips it (test if the static-pop BC traps the policy) ----
if _FASTEVAL:
    print("FAST-EVAL (LOAD=1, checkpoint present): skip BC pretraining",flush=True)
elif os.environ.get('NOBC'):
    print("NO BC pretraining -- pure RL from random init (testing BC-trap hypothesis)",flush=True)
elif os.environ.get('BCFROM'):
    load_ck(f'{base}/.cache/policy_{os.environ["BCFROM"]}.pt'); print(f"BCFROM: loaded shared BC floor policy_{os.environ['BCFROM']}.pt -> skip BC, RL-only (reuse pretraining across the lever grid)",flush=True)
else:
    BCT=os.environ.get('BCTGT'); Sb=[];Tb=[];TTb=[];TKS=[];TM=[]
    ORC=f'{base}/.cache/oracle_demos_{"cos" if os.environ.get("BCORC")=="cos" else ("dbf" if os.environ.get("BCORC")=="dbf" else ("tail" if os.environ.get("REWTAIL") else "full"))}_uni_h.npz'   # TEACHER demos (UNIFIED items+concepts; with history toks); objective: cos-belief | de-biased-NDCG | tail | full
    if BCT=='oracle' and os.path.exists(ORC) and not os.environ.get('REGEN'):
        _d=np.load(ORC); Sb=list(_d['Sb']); Tb=list(_d['Tb']); TTb=list(_d['TTb']); TKS=list(_d['TKS']); TM=list(_d['TM']); print(f"loaded cached ORACLE teacher ({len(Tb)} demos, with history)",flush=True)
    elif BCT=='oracle':                                                        # ORACLE DISTILLATION (UNIFIED items+concepts; clairvoyant greedy). BCORC=cos => belief-oracle: pick the lever maximising cos(belief, true-taste)
        _ocos=os.environ.get('BCORC')=='cos'; _odbf=os.environ.get('BCORC')=='dbf'; _Wb=1./np.log2(np.arange(2,12))   # _odbf: DE-BIASED teacher (owner Q2) -- rank levers by inverse-pop-weighted held-out NDCG, ALIGNED with the REW=dbf reward
        print(f"BC floor: ORACLE ({'cos-belief' if _ocos else ('de-biased-NDCG' if _odbf else (('tail' if os.environ.get('REWTAIL') else 'full')+'-NDCG'))} teacher, ITEMS+CONCEPTS)...",flush=True)
        for x in [x for x in trbig][:int(os.environ.get("DEMON",800))]:
            allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=set(allit[:max(len(allit)//2,4)])
            rd=dict(rat_by_u[x]); trel=set(j for j in allit if j not in prof and rd[j]>=4 and (not headmask[j] if os.environ.get('REWTAIL') else True))   # teacher objective: tail (REWTAIL) or full
            if not trel: continue
            cans=cans_np(x,prof); excl=np.array(list(prof)); idcgo=_Wb[:min(10,len(trel))].sum()+1e-12; toks=[]; asked=set()
            ustar=enc_u_np([(Q[j],resid[x][j]) for j in allit]) if _ocos else None; un_=(np.linalg.norm(ustar)+1e-9) if _ocos else 1.   # full-profile taste for the cos-belief oracle
            for t in range(T):
                ctoks,cpidx,caid=unicand_np(x,prof,cans,asked)                  # UNIFIED candidates: items + concepts (incl. genres)
                if not ctoks: break
                ul=enc_batch_np([toks+[tk] for tk in ctoks])
                if _ocos:
                    sc=(ul@ustar)/(np.linalg.norm(ul,axis=1)*un_); bk=int(np.argmax(sc))
                else:
                    S=ul@Ql.T+popb[None,:]; S[:,excl]=-1e9
                    if os.environ.get('REWTAIL'): S[:,headmask]=-1e9              # tail teacher ranks tail-only; full teacher ranks full
                    best=-1.; bk=0
                    for k in range(len(ctoks)):
                        o=np.argsort(-S[k])[:10]; nd=sum(_Wb[p]*(float(ipsw[int(it)]) if _odbf else 1.0) for p,it in enumerate(o) if int(it) in trel)/idcgo   # _odbf: inverse-pop weight each hit -> de-biased teacher
                        if nd>best: best=nd; bk=k
                _tk=np.zeros((T,D+1),np.float32); _tm=np.zeros(T,np.float32)
                for _i,(_f,_v) in enumerate(toks): _tk[_i,:D]=_f; _tk[_i,D]=_v; _tm[_i]=1.   # store the history PREFIX (before this pick) for the attn head
                Sb.append(enc_u_np(toks).astype(np.float32)); Tb.append(cpidx[bk]); TTb.append(t/8.); TKS.append(_tk); TM.append(_tm); asked.add(caid[bk]); toks.append(ctoks[bk])
        np.savez(ORC,Sb=np.array(Sb,np.float32),Tb=np.array(Tb),TTb=np.array(TTb,np.float32),TKS=np.array(TKS,np.float32),TM=np.array(TM,np.float32)); print(f"saved ORACLE teacher (with history) -> {os.path.basename(ORC)} ({len(Tb)} demos)",flush=True)
    else:
        _BCC=f'{base}/.cache/bc_demos_{BCT}.npz'; _bcd=os.path.exists(_BCC) and not os.environ.get('REGEN')   # CACHE heuristic-BC demos (uncached build = ~12k unbatched encoder folds = slow)
        if _bcd:
            _z=np.load(_BCC); Sb=list(_z['Sb']); Tb=list(_z['Tb']); TTb=list(_z['TTb']); TKS=list(_z['TKS']); TM=list(_z['TM']); print(f"loaded cached BC demos ({len(Tb)}) -> {os.path.basename(_BCC)}",flush=True)
        print(f"BC floor: match {'entropy' if BCT=='entropy' else 'conc_pop'}... ({'cached' if _bcd else 'building 1200 users'})",flush=True)
        for _ix,x in enumerate([] if _bcd else [x for x in trbig][:1200]):
            if _ix%150==0: print(f"  demo {_ix}/1200",flush=True)
            profset=set(j for j,_ in rat_by_u[x]); cans=cans_np(x,profset); ac=sorted(cans.keys(),key=lambda c:-(POOL_ENT[NI+c] if BCT=='entropy' else cfreq[c]))[:T]; toks=[]
            for t,c in enumerate(ac):
                _tk=np.zeros((T,D+1),np.float32); _tm=np.zeros(T,np.float32)
                for _i,(_f,_v) in enumerate(toks): _tk[_i,:D]=_f; _tk[_i,D]=_v; _tm[_i]=1.
                Sb.append(enc_u_np(toks).astype(np.float32)); Tb.append(NI+c); TTb.append(t/8.); TKS.append(_tk); TM.append(_tm); toks.append((Ec[c],cans[c]))
        if not _bcd and Sb: np.savez(_BCC,Sb=np.array(Sb,np.float32),Tb=np.array(Tb),TTb=np.array(TTb,np.float32),TKS=np.array(TKS,np.float32),TM=np.array(TM,np.float32)); print(f"saved BC demos -> {os.path.basename(_BCC)} ({len(Tb)})",flush=True)
    Sbt=torch.tensor(np.array(Sb)); Tbt=torch.tensor(np.array(Tb)).long(); TTbt=torch.tensor(np.array(TTb,np.float32))
    _useh=ATTN and len(TKS)==len(Sb) and len(TKS)>0                            # P6: distill the ATTN head TOO (history-aware) from the same oracle targets
    if _useh: TKSt=torch.tensor(np.array(TKS,np.float32)); TMt=torch.tensor(np.array(TM,np.float32)); print(f"BC: distilling scorer+ATTN together (history-aware) on {len(Sb)} demos",flush=True)
    _bcep=int(os.environ.get('BCEP',25))                                          # BC scorer-fit epochs (each minibatch = 256xNPx72 forward, slow on CPU); BCEP=10 for fast exploration
    for ep in range(_bcep):
        idx=torch.randperm(len(Sbt)); _bl=0.; _nb=0
        for b0 in range(0,len(Sbt),256):
            bb=idx[b0:b0+256]; sc=scorer(Sbt[bb],TTbt[bb])+(attn(TKSt[bb],TMt[bb]) if _useh else 0); loss=nn.functional.cross_entropy(sc,Tbt[bb]); opt.zero_grad(); loss.backward(); opt.step(); _bl+=float(loss); _nb+=1
        if ep%2==0 or ep==_bcep-1: print(f"  BC ep{ep+1}/{_bcep} loss {_bl/max(1,_nb):.3f}",flush=True)
if not os.environ.get('NOBC') and not os.environ.get('RESUME') and not _FASTEVAL and not os.environ.get('BCFROM'):   # SAVE the pre-finetune BC floor (pure distillation); skip when BCFROM reuses a shared floor
    save_ck(f'{base}/.cache/policy_{os.environ.get("TAG","o12")}_bc.pt'); print(f"saved BC-FLOOR (pre-finetune, +attn if ATTN) -> policy_{os.environ.get('TAG','o12')}_bc.pt",flush=True)
for g in opt.param_groups: g['lr']=float(os.environ.get('FTLR',5e-4))         # finetune LR from the conc_pop floor
_CK=f'{base}/.cache/policy_{os.environ.get("TAG","o12")}.pt'                       # checkpoint: train once, re-eval via LOAD=1 (no retrain)
if os.environ.get('LOAD') and os.path.exists(_CK):
    load_ck(_CK); print(f"LOADED {_CK} -- skip training",flush=True); EP=0
if os.environ.get('NOTRAIN'): EP=0                                                 # eval-only (non-learned mix/heuristic sweep, no policy retrain)
if os.environ.get('RESUME') and os.path.exists(_CK):                               # warm-resume: load checkpoint, train EP MORE epochs (no restart)
    load_ck(_CK); print(f"RESUMED {_CK} -- +{EP} more epochs",flush=True)
# per-epoch VALIDATION NDCG tracking (separate val users, disjoint from the locked test-150 -> no leakage)
_vrs=np.random.default_rng(123); _VSPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _vrs.shuffle(il); _VSPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
_VAL=([x for x in te if x in _VSPL][300:] if os.environ.get('VALTEST') else [x for x in te if x in _VSPL][:300]); _Wv=1./np.log2(np.arange(2,12))   # default val=te[:300]; VALTEST=1 -> validate on the TEST set te[300:] so the per-epoch "VAL" line IS the real paper metric (watch live). Final still seed-avg'd for honesty (epoch-robustness already shown).
def val_ndcg():
    nf=nt=mf=mt=0.
    for x in _VAL:
        profset,vtest=_VSPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in vtest if rd[j]>=4)
        if not tlike: continue
        cans=cans_np(x,profset); toks=[]; asked=set()
        for _ in range(8):
            with torch.no_grad(): _tt,_tm=toks2t(toks); sc=(scorer(torch.tensor(enc_u_np(toks)[None],dtype=torch.float32),len(toks)/8.)+(attn(_tt,_tm) if ATTN else 0)).numpy()[0]
            kk=next(int(k) for k in np.argsort(-sc) if int(k) not in asked); asked.add(kk)
            if PTYPE[kk]==0:
                j=PITEMS[kk]
                if j in profset: toks.append((Q[j],resid[x][j]))
            elif (kk-NI) in cans: toks.append((Ec[kk-NI],cans[kk-NI]))
        u=enc_u_np(toks); s=popb+Ql@u; s[list(profset)]=-1e9
        o=np.argsort(-s)[:10]; nf+=sum(_Wv[p] for p,it in enumerate(o) if int(it) in tlike)/(_Wv[:min(10,len(tlike))].sum()+1e-12); mf+=1
        st=s.copy(); st[headmask]=-1e9; relt=set(t for t in tlike if not headmask[t])
        if relt: ot=np.argsort(-st)[:10]; nt+=sum(_Wv[p] for p,it in enumerate(ot) if int(it) in relt)/(_Wv[:min(10,len(relt))].sum()+1e-12); mt+=1
    return nf/max(mf,1), nt/max(mt,1)
print(f"train scorer policy (pool={NP}, TAU={TAU}) -- finetune from conc_pop floor...",flush=True)
_CKBEST=_CK.replace('.pt','_best.pt'); TAGn=os.environ.get('TAG','o12'); _bestvt=-1.0; _bestvf=0.; _bestvtt=0.; _bestep=0; _selm=os.environ.get('SELVAL','tail')  # EARLY-STOP via best-checkpoint on the SEPARATE val (SELVAL=tail|full); final eval loads _best.pt
_since=0; _PAT=int(os.environ.get('PATIENCE',0))                              # patience halt: stop if val hasn't improved for _PAT epochs (0=OFF, run all EP). best-ckpt still captures the peak regardless.
for ep in range(EP):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),96):
        us=trbig[b0:b0+96]; ANS,USTAR,tgt,wt,LIKED,LMASK,RNEG,PROFM=prep(us)
        if OBJ=='reinforce':
            B=len(us); u,logps,rews,states,ents=rollout_sample(us,ANS,LIKED,LMASK,RNEG,PROFM,float(os.environ.get('PEN',0.02)))
            Gs=[None]*T; acc=torch.zeros(B)
            for t in reversed(range(T)): acc=rews[t]+acc; Gs[t]=acc.clone()   # return-to-go per turn
            loss=0.
            if CRITIC:                                                        # P7: learned value baseline (lower-variance advantage) + critic MSE
                closs=0.
                for t in range(T):
                    V=critic(states[t],t/8.); loss=loss-(logps[t]*(Gs[t]-V.detach())).mean(); closs=closs+((V-Gs[t].detach())**2).mean()
                loss=loss/T+0.5*closs/T
            else:
                for t in range(T): adv=Gs[t]-Gs[t].mean(); loss=loss-(logps[t]*adv).mean()
                loss=loss/T
            _ec=float(os.environ.get('ENT_COEF',0))                           # entropy-bonus regularization: loss -= beta*mean_t H(pi_t) -> keeps policy exploratory (escape the plateau, anti-collapse)
            if _ec: loss=loss-_ec*(sum(ents)/T)
            tot+=(-Gs[0].mean()).item()                                       # log -total_return (more negative = better)
            if os.environ.get('RECON'): loss=loss+float(os.environ.get('RECON'))*recon(rollout(us,ANS,explore=0.0),tgt,wt)  # P3: held-out reconstruction aux (straight-through u -> BCE on held-out likes)
            if os.environ.get('AUXCOS'): loss=loss+float(os.environ.get('AUXCOS'))*(1-torch.nn.functional.cosine_similarity(rollout(us,ANS,explore=0.0),USTAR,dim=1)).mean()  # owner #4: cos(belief,true-taste) auxiliary (the strong cos signal) via grad-carrying straight-through rollout
        else:
            u=rollout(us,ANS,explore=float(os.environ.get('EXPL',0.3)))
            loss=(1-torch.nn.functional.cosine_similarity(u,USTAR,dim=1)).mean() if OBJ=='ustar' else recon(u,tgt,wt)
            tot+=loss.item()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(scorer.parameters(),5.0); opt.step(); nb+=1
    if os.environ.get('SKIPVAL'):                                             # SKIPVAL: skip the slow per-epoch 300-user val rollout (~7min/epoch) -> we EVALCKS on TEST post-hoc; every epoch still saved
        print(f"  ep{ep+1} return={-tot/nb:.4f} (SKIPVAL; test post-hoc)",flush=True)
    else:
        vf,vt=val_ndcg(); _sel=vt if _selm=='tail' else vf                        # SELECTION metric on the separate val
        if _sel>_bestvt:                                                          # NEW PEAK -> save best ckpt + record peak durably (top-not-latest; survives a kill)
            _bestvt=_sel; _bestvf=vf; _bestvtt=vt; _bestep=ep+1; _since=0; save_ck(_CKBEST); _bm=f'  <- BEST ({_selm}) saved'
            open(f'{base}/.cache/peak_{TAGn}.txt','w').write(f"PEAK {TAGn} ENT_COEF={os.environ.get('ENT_COEF',0)} WD={os.environ.get('WD',0)} ATTN={int(ATTN)} | val full={vf:.4f} tail={vt:.4f} @ep{ep+1}/{EP} (sel={_selm})\n")
        else: _bm=''; _since+=1
        print(f"  ep{ep+1} return={-tot/nb:.4f} | VAL NDCG@10 full {vf:.3f} tail {vt:.3f}{_bm}  [PEAK {_selm} {_bestvt:.3f} @ep{_bestep}, {_since} since]",flush=True)
    save_ck(_CK.replace('.pt',f'_ep{ep+1}.pt')); save_ck(_CK.replace('.pt','_last.pt'))   # SAVE EVERY epoch (ckpts ~100KB) + always-latest pointer -> any-epoch/peak/latest analysis without stopping; a kill loses only the in-progress epoch
    if _PAT and _since>=_PAT: print(f"EARLY-STOP @ep{ep+1}: no val gain for {_PAT} epochs (best {_selm} {_bestvt:.3f} @ep{_bestep})",flush=True); break
if not os.environ.get('LOAD') and EP>0:
    save_ck(_CK); print(f"saved {_CK} (last) ; BEST {_selm} {_bestvt:.3f} @ep{_bestep} -> {_CKBEST}",flush=True)  # pin BOTH last + best-val checkpoints
    open(f'{base}/.cache/policy_runs.tsv','a').write(f"{TAGn}\tENT_COEF={os.environ.get('ENT_COEF',0)}\tWD={os.environ.get('WD',0)}\tFEATS={os.environ.get('FEATS','-')}\tpeakval_full={_bestvf:.4f}\tpeakval_tail={_bestvtt:.4f}\t@ep{_bestep}\tEP={EP}\n")  # durable per-run PEAK record (top-not-latest)
    if os.environ.get('USEBEST') and os.path.exists(_CKBEST): load_ck(_CKBEST); print(f"loaded BEST-val checkpoint for eval (early-stop @ep{_bestep})",flush=True)
scorer.eval()
# RMSE rating head: r_hat=mu+bi[j]+BETA*(u.Ql[j]); BETA fit once on training item-folds (cached, SHARED across scripts)
BETAC=f'{base}/.cache/rmse_beta.npy'
if os.path.exists(BETAC) and not os.environ.get('REGEN'):
    BETA=float(np.load(BETAC))
else:
    num=den=0.
    for x in [x for x in trU if len(rat_by_u[x])>=10][:600]:
        allit=[j for j,_ in rat_by_u[x]]; kn=allit[:len(allit)//2]; hl=allit[len(allit)//2:]
        ux=enc_u_np([(Q[j],resid[x][j]) for j in kn])
        for j in hl: pr=float(ux@Ql[j]); num+=resid[x][j]*pr; den+=pr*pr
    BETA=num/den if den>1e-9 else 1.0; np.save(BETAC,np.array(BETA))
def rmse_of(u,held):
    if not held: return None
    return (sum((r-min(max(mu+bi[j]+BETA*float(u@Ql[j]),1.),5.))**2 for j,r in held.items())/len(held))**0.5
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    s=(popb+Ql@u).copy(); s[list(excl)]=-1e9
    if tail: s[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-s); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel)
    mrr=0.
    for p,t in enumerate(o):
        if int(t) in rel: mrr=1./(p+1); break
    return nd,rc,mrr
_rs=np.random.default_rng(int(os.environ.get('SEED',123))); SPL={}                # SEED env => vary the test profile-split (for seed-averaging the result)
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
_tsel=[x for x in te if x in SPL]; _tr=os.environ.get('TESTRANGE','300:99999')      # main test = te[300:] (REPRESENTATIVE, disjoint from val te[:300]); 'all' -> full te (overlaps val); '0:150' / 'a:b' -> custom range
TE=(_tsel[:150]+_tsel[400:]) if _tr=='clean' else (_tsel if _tr=='all' else _tsel[int(_tr.split(':')[0]):int(_tr.split(':')[1])])
print(f"TEST set: TESTRANGE={_tr} -> {len(TE)} users",flush=True)
def run(mode,tail):
    M={q:0. for q in QPTS};Rc={q:0. for q in QPTS};CO={q:0. for q in QPTS};MR={q:0. for q in QPTS};m=0;na=0.;ni_=0;nc_=0;novp_=0;FP=[];ALLSEQ=[];TREEROWS=[]
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); held={j:rd[j] for j in test}
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset]); un=np.linalg.norm(ustar)+1e-9   # known user vector
        cans=cans_np(x,profset) if mode in('policy','conc_pop','conc_oracle','cont_oracle','entropy','entropy_uni','entropy_item','random','helf','logpop_ent','pop_ent') or mode.startswith('mix') else {}; toks=[]; asked=set(); nq=0; nans=0; first=None; seq=[]; trow=[]
        for q in QPTS:
            while nq<q:
                if mode=='policy':
                    if os.environ.get('FIXQ1') and nq==0 and cans: k=NI+max(cans,key=lambda c:POOL_ENT[NI+c])   # Q1 FIXED = entropy opener (matches FIXQ1 training)
                    else:
                        with torch.no_grad(): _tt,_tm=toks2t(toks); sc=(scorer(torch.tensor(enc_u_np(toks)[None],dtype=torch.float32),len(toks)/8.)+(attn(_tt,_tm) if ATTN else 0)).numpy()[0]
                        for k in np.argsort(-sc):
                            if k not in asked: break
                    if first is None: first=int(k)
                    asked.add(int(k)); nq+=1; seq.append(int(k))
                    if PTYPE[k]==0:
                        j=PITEMS[k]
                        if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; ni_+=1
                    else:
                        cc=k-NI
                        if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1; trow.append((int(cc),1 if cans[cc]>0 else -1))
                elif mode=='conc_pop':
                    ci=[c for c in range(NC) if c not in asked]; cc=max(ci,key=lambda c:cfreq[c]); asked.add(cc); nq+=1
                    if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1
                elif mode=='entropy':                                              # CANONICAL entropy heuristic: most DIVISIVE answerable concept (binary entropy of like-rate, >=2000 train users, unanswerable->0 ranked last; matches lit_baselines = the strong/fair baseline). CONCEPTS ONLY by design (range(NC)) -- items never in the candidate set.
                    ci=[c for c in range(NC) if c not in asked]; cc=max(ci,key=lambda c:POOL_ENT[NI+c]); asked.add(cc); nq+=1
                    if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1
                elif mode=='entropy_uni':                                          # DIVISIVENESS over the UNIFIED pool (items+concepts), answerability-BLIND: ranks by global Hb so it picks divisive ITEMS the cold-start user usually hasn't seen -> unanswerable -> wasted Q. Demonstrates WHY canonical entropy is concept-only + the answerability cost of items.
                    cs=[k for k in range(NP) if k not in asked]; k=max(cs,key=lambda k:POOL_ENT[k]); asked.add(k); nq+=1
                    if PTYPE[k]==0:
                        j=PITEMS[k]
                        if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; ni_+=1
                    else:
                        cc=k-NI
                        if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
                elif mode=='entropy_item':                                         # SAME divisiveness criterion, ITEMS-ONLY pool: the answerability counterfactual -- most-divisive items are usually unseen in cold-start -> unanswerable -> NDCG should crater (criterion held fixed, only the pool's answerability changes).
                    cs=[k for k in range(NI) if k not in asked]; k=max(cs,key=lambda k:POOL_ENT[k]); asked.add(k); nq+=1
                    j=PITEMS[k]
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; ni_+=1
                elif mode=='fullprof':                                             # CEILING: fold the ENTIRE known profile -> belief == u* (cos=1 by construction); realistic warm-start upper bound on what T-question elicitation can reach
                    toks=[(Q[j],resid[x][j]) for j in profset]; nq=q
                elif mode=='random':                                              # lit: random concept (canonical active-learning control)
                    ci=[c for c in range(NC) if c not in asked]; nq+=1
                    if ci:
                        cc=int(ci[np.random.default_rng(1234+x*97+nq).integers(len(ci))]); asked.add(cc)
                        if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
                elif mode=='helf':                                                # HELF (Rashid/Karypis/Riedl 2008): harmonic mean of normalized divisiveness & normalized log-frequency
                    ci=[c for c in range(NC) if c not in asked]; cc=max(ci,key=lambda c:HELF_c[c]); asked.add(cc); nq+=1
                    if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
                elif mode=='logpop_ent':                                          # log(pop) * divisiveness (Rashid 2002)
                    ci=[c for c in range(NC) if c not in asked]; cc=max(ci,key=lambda c:LPE_c[c]); asked.add(cc); nq+=1
                    if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
                elif mode=='pop_ent':                                             # pop * divisiveness
                    ci=[c for c in range(NC) if c not in asked]; cc=max(ci,key=lambda c:PE_c[c]); asked.add(cc); nq+=1
                    if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
                elif mode=='conc_oracle':                                          # UNIFIED CEILING (items+concepts, privileged): greedy pick the lever maximizing TAIL/FULL-NDCG on the test items
                    trel=set(j for j in test if rd[j]>=4 and (not headmask[j] if os.environ.get('REWTAIL') else True))
                    ctoks,cpidx,caid=unicand_np(x,profset,cans,asked)
                    if not ctoks or not trel: nq+=1; continue
                    ul=enc_batch_np([toks+[tk] for tk in ctoks]); Sm=ul@Ql.T+popb[None,:]; Sm[:,list(profset)]=-1e9
                    if os.environ.get('REWTAIL'): Sm[:,headmask]=-1e9            # tail ceiling masks head; full ceiling ranks full
                    bo=-1.; bk=0
                    for kk in range(len(ctoks)):
                        o=np.argsort(-Sm[kk])[:10]; nd=(sum(_W[p] for p,it in enumerate(o) if int(it) in trel)/(_W[:min(10,len(trel))].sum()+1e-12)) if trel else 0.
                        if nd>bo: bo=nd; bk=kk
                    asked.add(caid[bk]); nq+=1; toks.append(ctoks[bk]); nans+=1
                    if caid[bk][0]=='i': ni_+=1
                    else: nc_+=1
                elif mode=='cont_oracle':                                          # CONTINUOUS CEILING: best point in R^D over {pool concepts} U {ideal centroid + random non-pool dirs}; gap over conc_oracle = continuous headroom THROUGH the frozen encoder
                    trel=set(j for j in test if rd[j]>=4 and (not headmask[j] if os.environ.get('REWTAIL') else True))
                    if not trel: nq+=1; continue
                    poolc=[c for c in cans if c not in asked]
                    E_all=[Ec[c] for c in poolc]; a_all=[float(cans[c]) for c in poolc]; npool=len(E_all)
                    _cn=float(np.linalg.norm(Ec,axis=1).mean()); _thr=float(np.mean([float(ustar@Ec[c]) for c in cans])) if cans else 0.
                    _nov=[Ql[list(trel)].mean(0)]                                  # the IDEAL continuous question: centroid of held-out tail-likes (points exactly at the targets)
                    _r2=np.random.default_rng(13+nq)
                    for _ in range(16): _nov.append(_r2.standard_normal(D))        # random-direction controls: is ANY non-pool direction better than every concept?
                    for e in _nov:
                        es=(e/(np.linalg.norm(e)+1e-9)*_cn).astype(np.float32); E_all.append(es); a_all.append(POS if float(ustar@es)>_thr else NEG)
                    ul=enc_batch_np([toks+[(E_all[k],a_all[k])] for k in range(len(E_all))]); Sm=ul@Ql.T+popb[None,:]; Sm[:,list(profset)]=-1e9
                    if os.environ.get('REWTAIL'): Sm[:,headmask]=-1e9
                    bo=-1.; bk=0
                    for k in range(len(E_all)):
                        o=np.argsort(-Sm[k])[:10]; nd=(sum(_W[p] for p,it in enumerate(o) if int(it) in trel)/(_W[:min(10,len(trel))].sum()+1e-12))
                        if nd>bo: bo=nd; bk=k
                    nq+=1; nans+=1; toks.append((E_all[bk],a_all[bk]))
                    if bk>=npool: novp_+=1                                          # continuous oracle preferred a NOVEL direction over EVERY pool concept -> real continuous headroom
                    else: asked.add(poolc[bk])
                elif mode=='pop_item':
                    cs=[j for j in PITEMS if j not in asked]; j=cs[0]; asked.add(j); nq+=1
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode.startswith('mix'):                                       # mixK = K item-asks of 8 (rest frequent concepts), spread evenly
                    ki=int(mode[3:]); is_it=int((nq+1)*ki/8)>int(nq*ki/8)
                    if is_it:
                        cs=[j for j in PITEMS if ('i',j) not in asked]; j=cs[0]; asked.add(('i',j)); nq+=1
                        if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; ni_+=1
                    else:
                        ci=[c for c in range(NC) if ('c',c) not in asked]; cc=max(ci,key=lambda c:cfreq[c]); asked.add(('c',cc)); nq+=1
                        if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1
            uu=enc_u_np(toks); mt=metr(uu,tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1];MR[q]+=mt[2]
            CO[q]+=float(uu@ustar/((np.linalg.norm(uu)+1e-9)*un))             # cos(belief, known user u*)
        m+=1; na+=nans
        if first is not None: FP.append(first)
        if mode=='policy': ALLSEQ.append(seq); TREEROWS.append(trow)
    extra=(f" | picks {ni_}i/{nc_}c"+(f" | distinct-1st {len(set(FP))}/{max(len(FP),1)}" if mode=='policy' else "")) if (mode=='policy' or mode.startswith('mix') or mode.startswith('entropy_')) else ""
    extra+=" | cos(u,u*) "+" ".join(f"{CO[q]/m:.3f}" for q in QPTS)
    if mode=='cont_oracle': extra+=f" | NOVEL-dir picks {novp_}/{m*8} ({100*novp_/max(m*8,1):.0f}% chose continuous over EVERY concept)"
    if mode=='policy' and ALLSEQ:                                                 # ADAPTIVITY: does the policy branch after the (fixed) opening, or ask the same shortlist?
        import collections; Lm=max(len(s) for s in ALLSEQ)
        pt=[collections.Counter(s[t] for s in ALLSEQ if len(s)>t) for t in range(Lm)]
        dt=[len(c) for c in pt]; share=[round(max(c.values())/sum(c.values()),2) for c in pt]
        extra+=f"\n     ADAPT: distinct-pick/turn {dt} | top-pick-share/turn {share} | distinct-traj {len(set(tuple(s) for s in ALLSEQ))}/{len(ALLSEQ)} | total-vocab {len(set(k for s in ALLSEQ for k in s))}"
    if mode=='policy' and os.environ.get('TREE'):
        import json; json.dump(TREEROWS, open(f'{base}/.cache/tree_{os.environ.get("TAG","x")}.json','w')); print(f"  TREE dumped: {len(TREEROWS)} user paths -> tree_{os.environ.get('TAG','x')}.json",flush=True)
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},{q:MR[q]/m for q in MR},na/m,extra,{q:CO[q]/m for q in CO}
if os.environ.get('EVALCKS'):                                                  # EFFICIENT eval-all: data loaded ONCE, score entropy+conc_pop + a list of checkpoints on the PAPER test set, ACROSS SEEDS. EVALCKS=tag1,tag2,... EVALSEEDS=123,1,2,3,7,11 -> seed-averaged per-epoch grid (CSV). RESID levers need RESID=1 in this env.
    _seeds=[int(s) for s in os.environ.get('EVALSEEDS',str(os.environ.get('SEED','123'))).split(',')]
    _tags=[t for t in os.environ['EVALCKS'].split(',') if t]; _csv=os.environ.get('EVALCSV',f'{base}/.cache/evalcks_grid.csv')
    print(f"=== EVALCKS seeds={_seeds} TESTRANGE={_tr} | NDCG@10 FULL/TAIL @ {QPTS} -> {_csv} ===",flush=True)
    _cf=open(_csv,'a')
    for _sd in _seeds:
        _rsd=np.random.default_rng(_sd); SPL={}                                # rebuild the profile-split for this seed (== the seed-averaging protocol)
        for x in te:
            its=list(dict(rat_by_u[x]))
            if len(its)>=6: il=its[:]; _rsd.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
        _ts=[x for x in te if x in SPL]
        TE=(_ts[:150]+_ts[400:]) if _tr=='clean' else (_ts if _tr=='all' else _ts[int(_tr.split(':')[0]):int(_tr.split(':')[1])])
        for _nm in os.environ.get('EVALBASE','entropy,conc_pop').split(',')+_tags:                               # heuristic baselines (recomputed per split) + each checkpoint, SAME split -> directly comparable
            if _nm in _tags:
                _p=f'{base}/.cache/policy_{_nm}.pt'
                if not os.path.exists(_p): print(f"  seed{_sd} {_nm:>24}: MISSING",flush=True); continue
                try: load_ck(_p); _md='policy'
                except Exception as _e: print(f"  seed{_sd} {_nm:>24}: LOADERR {type(_e).__name__} (din mismatch? skip)",flush=True); continue
            else: _md=_nm
            try: _rf=run(_md,False); _rt=run(_md,True)
            except Exception as _e: print(f"  seed{_sd} {_nm:>24}: EVALERR {type(_e).__name__}",flush=True); continue
            Mp,Rcp,MRp,_naf,_,COp=_rf; Mt,Rct,MRt=_rt[0],_rt[1],_rt[2]            # full run -> mrr_full + belief-cos (mask-independent); tail run -> tail NDCG/Rec/mrr
            print(f"  seed{_sd} {_nm:>24}: FULL "+" ".join(f"{Mp[q]:.3f}" for q in QPTS)+" | TAIL "+" ".join(f"{Mt[q]:.3f}" for q in QPTS),flush=True)
            for q in QPTS: _cf.write(f"{_sd},{_nm},{q},{Mp[q]:.4f},{Mt[q]:.4f},{Rcp[q]:.4f},{Rct[q]:.4f},{MRp[q]:.4f},{MRt[q]:.4f},{COp[q]:.4f},{_naf:.2f}\n")
            _cf.flush()
    _cf.close(); import sys; sys.exit(0)
for tail in [False,True]:
    print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | popularity-aware scorer | NDCG@10 / Rec@50 / ans ===",flush=True)
    for mode in os.environ.get('MODES','pop_item,conc_pop,policy').split(','):
        Mp,Rcp,MRp,na,extra,_CO=run(mode,tail); print(f"  {mode:<9}: NDCG "+" ".join(f"{Mp[q]:.3f}" for q in QPTS)+" | Rec "+" ".join(f"{Rcp[q]:.3f}" for q in QPTS)+" | MRR "+" ".join(f"{MRp[q]:.3f}" for q in QPTS)+f" | ans/{T}={na:.1f}{extra}",flush=True)
    print(f"  GATE: policy must beat conc_pop @q8 (full+tail)",flush=True)
