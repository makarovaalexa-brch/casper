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
class EncRA(nn.Module):                                                          # DAHCR recurrent+attention encoder (module-level so LOADREC can swap the GRU+attn recommender into run()/POLOPT)
    def __init__(s,H=128,heads=4):
        super().__init__(); s.inp=nn.Linear(D+1,H); s.attn=nn.MultiheadAttention(H,heads,batch_first=True); s.gru=nn.GRU(H,H,batch_first=True); s.out=nn.Linear(H,D)
    def forward(s,t,m):
        x=s.inp(t); a,_=s.attn(x,x,x,key_padding_mask=(m==0)); x=torch.relu(x+a)
        o,_=s.gru(x); L=m.sum(1).long().clamp(min=1); return s.out(o[torch.arange(len(o)),L-1])
class EncST(nn.Module):                                                          # set-transformer: MHSA + permutation-invariant mean-pool (order-invariant), module-level for LOADREC
    def __init__(s,H=128):
        super().__init__(); s.inp=nn.Linear(D+1,H); s.attn=nn.MultiheadAttention(H,4,batch_first=True); s.ff=nn.Sequential(nn.Linear(H,H),nn.ReLU(),nn.Linear(H,H)); s.out=nn.Linear(H,D)
    def forward(s,t,m):
        x=s.inp(t); a,_=s.attn(x,x,x,key_padding_mask=(m==0)); x=torch.relu(x+a); x=torch.relu(x+s.ff(x))
        m2=m.unsqueeze(-1); return s.out((x*m2).sum(1)/m2.sum(1).clamp(min=1))
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
if os.environ.get('LOADREC') and os.path.exists(os.environ['LOADREC']):          # swap in a FINE-TUNED recommender (enc+Ql) for ALL downstream blocks
    _r=torch.load(os.environ['LOADREC']); _sd=_r['st'] if 'st' in _r else (_r['era'] if 'era' in _r else _r['enc'])
    if 'st' in _r: enc=EncST(); _at='set-transformer'                           # MHSA + mean-pool (order-invariant)
    elif any('gru' in k for k in _sd): enc=EncRA(); _at='GRU+attn'              # GRU+attn
    else: _at='simple'                                                          # simple-attn (FTREC / recipe / V1)
    enc.load_state_dict(_sd); enc.eval()
    Ql=_r['Ql'].numpy().astype(np.float32); Qlt=torch.tensor(Ql)
    print(f"LOADREC: using {_at} recommender {os.path.basename(os.environ['LOADREC'])}",flush=True)
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
class Actor(nn.Module):                                                        # PAPER C continuous actor: state [belief u, turn] -> query q in R^D ; snap=argmax(q.POOL) [Wolpertinger replicate] OR fold q directly [continuous]
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,HID),nn.ReLU(),nn.Linear(HID,HID),nn.ReLU(),nn.Linear(HID,D))
    def forward(s,u,tt):
        B=u.shape[0]; tn=(tt.view(B,1) if torch.is_tensor(tt) else torch.full((B,1),float(tt))); return s.f(torch.cat([u,tn],1))
actor=Actor(); CONTMODE=os.environ.get('CONTMODE','snap'); _CN=float(np.linalg.norm(Ec,axis=1).mean())   # snap=Wolpertinger replicate / cont=off-pool fold ; _CN=mean concept norm (on-manifold scale)
POOLn=(POOL/(np.linalg.norm(POOL,axis=1,keepdims=True)+1e-9)).astype(np.float32); POOLnt=torch.tensor(POOLn)   # unit-norm pool for COSINE snap (consistent with cos-loss distillation; dot-product snap is norm-biased)
_GROUND=int(os.environ.get('GROUND',0)); _GTAU=float(os.environ.get('GTAU',0.2)); _GRADED=bool(os.environ.get('GRADED'))   # GOAL 2: GROUND>0 -> straight-through grounded fold (nearest real entity fwd, soft grad). GRADED=1 -> graded answer (affinity-interpolated NEG..POS) instead of one ±bit
_pp=list(actor.parameters())                                                   # the CONTINUOUS ACTOR is the policy (replaces discrete scorer+attn)
opt=torch.optim.Adam(_pp,1e-3,weight_decay=float(os.environ.get('WD',0)))     # WD = L2 regularization (anti-overfit)
class Critic(nn.Module):                                                      # P7: state-value baseline V(belief,turn) for REINFORCE variance reduction
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,1))
    def forward(s,u,tt): B=u.shape[0]; return s.f(torch.cat([u,torch.full((B,1),float(tt))],1)).squeeze(1)
critic=Critic(); CRITIC=os.environ.get('CRITIC')
if CRITIC: opt=torch.optim.Adam(_pp+list(critic.parameters()),1e-3)
def save_ck(path): torch.save({'actor':actor.state_dict()},path)              # PAPER C: persist the continuous actor
def load_ck(path):                                                            # load a continuous-actor ckpt ('actor') OR a legacy scorer ckpt
    sd=torch.load(path)
    if isinstance(sd,dict) and 'actor' in sd: actor.load_state_dict(sd['actor'])
    elif isinstance(sd,dict) and 'scorer' in sd: scorer.load_state_dict(sd['scorer'],strict=False)
    else: scorer.load_state_dict(sd,strict=False)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**float(os.environ.get('DBEXP',0.5)),1e-3,1.0)).astype(np.float32)   # DBEXP: de-bias strength (0.5=1/sqrt(pop) default; 1.0=1/pop pushes harder to tail)
IPOPt=torch.tensor(ipsw)                                                      # inverse-popularity weights (for the REW=dbf de-biased-full reward)
ML=64
def prep(users):
    B=len(users); ANS=torch.zeros(B,NP); USTAR=torch.zeros(B,D); tgt=torch.zeros(B,ni); wt=torch.zeros(B,ni); LIKED=torch.zeros(B,ML,dtype=torch.long); LMASK=torch.zeros(B,ML); RNEG=torch.tensor(rng.integers(0,ni,size=(B,ML)),dtype=torch.long); PROFM=torch.zeros(B,ni)  # random items ~= non-likes (ranking reward); PROFM=known profile to exclude
    for b,x in enumerate(users):
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); prof=set(allit[:max(len(allit)//2,4)]); held=[j for j in likes_by_u[x] if j not in prof] or likes_by_u[x][:1]
        for j in prof: PROFM[b,j]=1.
        _up=enc_u_np([(Q[j],resid[x][j]) for j in prof])                          # profile taste (for geometric answers to POPULAR non-profile items)
        for k,j in enumerate(PITEMS):
            if j in prof: ANS[b,k]=resid[x][j]
            elif os.environ.get('ITEMSANS'): ANS[b,k]=POS if float(_up@Q[j])>0 else NEG   # PAPER C: popular items ANSWERABLE on continuous scale (user derives like/dislike from taste, geometric sign)
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
def rollout(users,ANS,USTAR=None,explore=0.0):                                # PAPER C continuous-actor rollout (snap=Wolpertinger straight-through ; cont=fold off-pool point)
    B=len(users); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D); asked=torch.zeros(B,NP)
    for t in range(T):
        q=actor(u,t/8.)                                                      # (B,D) continuous query
        if CONTMODE=='cont':                                                 # GOAL 2 CONTINUOUS: assume answerable; one-bit geometric answer; GROUND=k folds the kNN-centroid of REAL entities (in-distribution)
            qn=q/(q.norm(dim=1,keepdim=True)+1e-9)                            # unit query
            if _GROUND:                                                      # STRAIGHT-THROUGH grounding: hard nearest real entity fwd (sharp, in-distribution), soft-attention grad bwd (actor still learns)
                sim=qn@POOLnt.t(); fes=torch.softmax(sim/_GTAU,1)@POOLt; fe=POOLt[sim.argmax(1)]+(fes-fes.detach())
            else: fe=qn*_CN                                                   # raw off-pool point (OOD baseline)
            cf=((fe*USTAR).sum(1)/(fe.norm(dim=1)*USTAR.norm(dim=1)+1e-9)) if USTAR is not None else torch.zeros(B)   # graded affinity cos(u*,fe) in [-1,1]
            ans=(NEG+(POS-NEG)*(cf+1)/2) if _GRADED else (torch.sign(cf)+(torch.tanh(4.*cf)-torch.tanh(4.*cf).detach()))   # GRADED: interpolate NEG..POS by affinity ; else one honest ±bit
            toks=toks.clone(); toks[:,t,:D]=fe; toks[:,t,D]=ans; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask); continue
        sc=q@POOLt.t()                                                       # SNAP score = query . pool
        if explore>0: sc=sc+explore*torch.randn_like(sc)
        sc=sc.masked_fill(asked>0,-1e9)                                      # no re-asking (matches eval)
        if os.environ.get('ANSMASK'): sc=sc.masked_fill(ANS.abs()<1e-6,-1e9)   # PAPER C FIX: snap ONLY to ANSWERABLE entities (ANS!=0) -> matches eval, kills un-askable cheating
        w_soft=torch.softmax(sc/TAU,1); idx=w_soft.argmax(1)
        w_hard=torch.zeros_like(w_soft).scatter_(1,idx.unsqueeze(1),1.0)
        w=w_hard+(w_soft-w_soft.detach())                                    # STRAIGHT-THROUGH snap (Wolpertinger): hard nearest entity == eval, soft gradient to the query
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
if os.environ.get('LOADCK') and os.path.exists(os.environ['LOADCK']): load_ck(os.environ['LOADCK']); print(f"LOADED explicit {os.environ['LOADCK']} -- skip training",flush=True); EP=0   # PAPER C: eval a SPECIFIC actor epoch ckpt
if os.environ.get('DISTILL') and not os.environ.get('LOADCK'):                     # PAPER C GOAL 1: distill CASPER-R (discrete winner) into the continuous actor -> snap reproduces its picks
    _tp=os.environ.get('TEACHER',f'{base}/.cache/policy_entdistill_ep4.pt'); _td=torch.load(_tp)
    scorer.load_state_dict(_td['scorer'] if isinstance(_td,dict) and 'scorer' in _td else _td,strict=False); scorer.eval()
    print(f"DISTILL: teacher={os.path.basename(_tp)} -> rolling out CASPER-R for demos",flush=True)
    _DC=f'{base}/.cache/distill_demos_casperR.npz'
    if os.path.exists(_DC) and not os.environ.get('REGEN'):
        _z=np.load(_DC); DSb=_z['S']; DTT=_z['TT']; DK=_z['K']; print(f"  loaded {len(DK)} cached demos",flush=True)
    else:
        DSb=[];DTT=[];DK=[]
        for x in [x for x in trbig][:int(os.environ.get('DEMON',1500))]:
            prof=set(j for j,_ in rat_by_u[x]); cans=cans_np(x,prof); toks=[]; asked=set()
            for t in range(T):
                u=enc_u_np(toks)
                with torch.no_grad(): sc=scorer(torch.tensor(u[None],dtype=torch.float32),t/8.).numpy()[0]
                k=next((int(kk) for kk in np.argsort(-sc) if int(kk) not in asked),None)            # CASPER-R = argmax over FULL pool (faithful, incl. its wasted picks)
                if k is None: break
                DSb.append(u.astype(np.float32)); DTT.append(t/8.); DK.append(k); asked.add(k)
                if PTYPE[k]==0:
                    j=PITEMS[k]
                    if j in prof: toks.append((Q[j],resid[x][j]))                                    # fold only if answerable (== CASPER-R eval)
                elif (k-NI) in cans: toks.append((Ec[k-NI],cans[k-NI]))
        DSb=np.array(DSb,np.float32);DTT=np.array(DTT,np.float32);DK=np.array(DK)
        np.savez(_DC,S=DSb,TT=DTT,K=DK); print(f"  saved {len(DK)} demos -> {os.path.basename(_DC)}",flush=True)
    DSt=torch.tensor(DSb); DTTt=torch.tensor(DTT); POOLk=POOLt[torch.tensor(DK).long()]            # BC: actor(u,turn) -> the chosen entity's embedding direction (so snap reproduces it)
    for ep in range(int(os.environ.get('DISTEP',40))):
        idx=torch.randperm(len(DSt)); tl=0.;nb=0
        for b0 in range(0,len(DSt),256):
            bb=idx[b0:b0+256]; q=actor(DSt[bb],DTTt[bb]); loss=(1-torch.nn.functional.cosine_similarity(q,POOLk[bb],dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl+=float(loss); nb+=1
        if ep%5==0 or ep==int(os.environ.get('DISTEP',40))-1: print(f"  DISTILL ep{ep+1} cos-loss {tl/nb:.4f}",flush=True)
    save_ck(_CK); save_ck(_CK.replace('.pt','_last.pt')); print(f"DISTILL done ({len(DK)} demos) -> {os.path.basename(_CK)}",flush=True); EP=0
if os.environ.get('ORADISTILL') and not os.environ.get('LOADCK'):                  # PAPER C WIN-PATH: distill the PRIVILEGED concept best-subset oracle (0.389 tail) into the continuous actor. The WHOLE 0.152->0.389 gap is SELECTION (same 761 concepts, same cans answers) -> does the oracle's selection generalise from the profile?
    _RT=os.environ.get('NORT') is None; _Wc=1./np.log2(np.arange(2,12)); _RECON=os.environ.get('RECON'); GRAW=os.environ.get('GRAW')
    _OD=f'{base}/.cache/oradistill_{"recon" if _RECON else ("tail" if _RT else "full")}_demos.npz'
    if os.path.exists(_OD) and not os.environ.get('REGEN'):
        _z=np.load(_OD); DSb=_z['S']; DTT=_z['TT']; DK=_z['K']; print(f"ORADISTILL: loaded {len(DK)} cached demos ({'RECON' if _RECON else 'NDCG'} teacher)",flush=True)
    elif _RECON:                                                                       # LEARNABLE teacher: greedy reconstruct-u* over UNIFIED answerable pool, graded answers (target=observable profile-encoding -> no imitation gap). Candidate pre-filter (top-RECONK by |alignment|) for speed.
        print(f"ORADISTILL: generating RECON-oracle demos (greedy min||u-u*||, GRAW={bool(GRAW)})...",flush=True)
        DSb=[];DTT=[];DK=[]; _RK=int(os.environ.get('RECONK','40'))
        def _gans(ustar,k): d=float(ustar@POOL[k]); return d if GRAW else (POS if d>0 else NEG)
        for x in [x for x in trbig][:int(os.environ.get('DEMON',800))]:
            allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); hh=max(len(allit)//2,4); prof=set(allit[:hh])
            ustar=enc_u_np([(Q[j],resid[x][j]) for j in prof]); cans=cans_np(x,prof)
            cset=list(range(NI))+[NI+c for c in cans]                                   # answerable pool: all popular items (ITEMSANS) + answerable concepts
            toks=[]; chosen=set()
            for t in range(T):
                u=enc_u_np(toks); cand=sorted([k for k in cset if k not in chosen],key=lambda k:-abs(float(ustar@POOL[k])))[:_RK]
                if not cand: break
                best=1e9; bk=cand[0]
                for k in cand:
                    u2=enc_u_np(toks+[(POOL[k],_gans(ustar,k))]); d=float(np.linalg.norm(u2-ustar))
                    if d<best: best=d; bk=k
                DSb.append(u.astype(np.float32)); DTT.append(t/8.); DK.append(bk); chosen.add(bk); toks.append((POOL[bk],_gans(ustar,bk)))
    else:
        print(f"ORADISTILL: generating concept-oracle demos ({'tail' if _RT else 'full'}-optimised, privileged greedy over training-user held splits)...",flush=True)
        DSb=[];DTT=[];DK=[]; _nu=0
        for x in [x for x in trbig][:int(os.environ.get('DEMON',800))]:
            allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); hh=max(len(allit)//2,4); prof=set(allit[:hh]); held=allit[hh:]
            rd=dict(rat_by_u[x]); trel=set(j for j in held if rd[j]>=4 and (not headmask[j] if _RT else True))
            if not trel: continue
            cans=cans_np(x,prof); cs=list(cans.keys())
            if not cs: continue
            toks=[]; chosen=set()
            for t in range(T):
                u=enc_u_np(toks); cand=[c for c in cs if c not in chosen]
                if not cand: break
                best=-1.; bk=cand[0]
                for c in cand:                                                          # GREEDY oracle pick: concept whose cans-answer most raises tail-NDCG vs THIS user's held targets (privileged teacher)
                    u2=enc_u_np(toks+[(Ec[c],cans[c])]); s=popb+Ql@u2; s[list(prof)]=-1e9
                    if _RT: s[headmask]=-1e9
                    o=np.argsort(-s)[:10]; nd=sum(_Wc[p] for p,it in enumerate(o) if int(it) in trel)
                    if nd>best: best=nd; bk=c
                DSb.append(u.astype(np.float32)); DTT.append(t/8.); DK.append(NI+bk); chosen.add(bk); toks.append((Ec[bk],cans[bk]))
            _nu+=1
        DSb=np.array(DSb,np.float32);DTT=np.array(DTT,np.float32);DK=np.array(DK)
        np.savez(_OD,S=DSb,TT=DTT,K=DK); print(f"  saved {len(DK)} demos from {_nu} users -> {os.path.basename(_OD)}",flush=True)
    DSt=torch.tensor(DSb); DTTt=torch.tensor(DTT); POOLk=POOLt[torch.tensor(DK).long()]    # student sees ONLY belief u + turn -> regress to the oracle-chosen concept embedding (cos), so snap reproduces it realizably
    for ep in range(int(os.environ.get('DISTEP',40))):
        idx=torch.randperm(len(DSt)); tl=0.;nb=0
        for b0 in range(0,len(DSt),256):
            bb=idx[b0:b0+256]; q=actor(DSt[bb],DTTt[bb]); loss=(1-torch.nn.functional.cosine_similarity(q,POOLk[bb],dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl+=float(loss); nb+=1
        if ep%5==0 or ep==int(os.environ.get('DISTEP',40))-1: print(f"  ORADISTILL ep{ep+1} cos-loss {tl/nb:.4f}",flush=True)
    save_ck(_CK); save_ck(_CK.replace('.pt','_last.pt')); print(f"ORADISTILL done ({len(DK)} demos) -> {os.path.basename(_CK)} ; eval flows into contactor below",flush=True); EP=0
if os.environ.get('INIT') and os.path.exists(os.environ['INIT']) and not os.environ.get('LOADCK'): load_ck(os.environ['INIT']); print(f"INIT warm-start from {os.path.basename(os.environ['INIT'])} (keep training)",flush=True)   # PAPER C Goal 2: continue-train from the Goal-1 distilled actor
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
            with torch.no_grad(): qv=actor(torch.tensor(enc_u_np(toks)[None],dtype=torch.float32),len(toks)/8.).numpy()[0]
            if CONTMODE=='cont':
                _us=enc_u_np([(Q[j],resid[x][j]) for j in profset]); _th=float(np.mean([float(_us@Ec[c]) for c in cans])) if cans else 0.
                _qn=(qv/(np.linalg.norm(qv)+1e-9)*_CN).astype(np.float32); toks.append((_qn, POS if float(_us@_qn)>_th else NEG))
            else:
                sc=qv@POOL.T; kk=next(int(k) for k in np.argsort(-sc) if int(k) not in asked); asked.add(kk)
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
            u=rollout(us,ANS,USTAR,explore=float(os.environ.get('EXPL',0.3)))
            loss=(1-torch.nn.functional.cosine_similarity(u,USTAR,dim=1)).mean() if OBJ=='ustar' else recon(u,tgt,wt)
            tot+=loss.item()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.0); opt.step(); nb+=1
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
if os.environ.get('ABOT'):                                                     # PHASE 0: learned answerer (ABot) — pretrain (mu,logsigma^2) on ITEMS via Gaussian NLL, then diagnostic vs cans
    import sys
    print("=== PHASE 0: ABot heteroscedastic answerer | pretrain on items (NLL) + diagnostic vs cans ===",flush=True)
    _ipop=(cnt/max(cnt.max(),1)).astype(np.float32); _cpop=(cfreq/max(cfreq.max(),1)).astype(np.float32)   # item & concept popularity in [0,1]
    Qn=torch.tensor((Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)).astype(np.float32)); _ipt=torch.tensor(_ipop); _NF=4   # unit item factors + nearest-movie features
    def afeat(ust,pe):                                                          # nearest-REAL-movie features (exclude exact self -> no leakage): distance, density, grounded affinity, popularity
        pu=pe/(pe.norm(dim=1,keepdim=True)+1e-9); tv,ti=(pu@Qn.t()).topk(9,dim=1); tv=tv[:,1:]; ti=ti[:,1:]
        return torch.stack([1-tv[:,0],1-tv.mean(1),(ust*Qn[ti].mean(1)).sum(1),_ipt[ti].mean(1)],1)
    _ADIN=D+D+1+_NF
    class ABot(nn.Module):                                                     # (u*, question-emb) -> (mu rating, log sigma^2). answerability = low sigma^2 (epistemic via nearest-movie distance)
        def __init__(s):
            super().__init__(); s.f=nn.Sequential(nn.Linear(_ADIN,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU()); s.mu=nn.Linear(256,1); s.lv=nn.Linear(256,1)
        def forward(s,ust,pe):
            x=torch.cat([ust,pe,(ust*pe).sum(1,keepdim=True),afeat(ust,pe)],1); h=s.f(x); return s.mu(h).squeeze(1),s.lv(h).squeeze(1)
    abot=ABot(); aopt=torch.optim.Adam(abot.parameters(),1e-3)
    AU=[];AP=[];AY=[]                                                           # build (u*, item-factor) -> resid, profile/held split (no leakage)
    for x in [x for x in trbig][:int(os.environ.get('ADEMON',3000))]:
        allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); hh=max(len(allit)//2,4); prof=allit[:hh]; held=allit[hh:]
        if not held: continue
        ust=enc_u_np([(Q[j],resid[x][j]) for j in prof]).astype(np.float32)
        for j in held: AU.append(ust); AP.append(Q[j].astype(np.float32)); AY.append(np.float32(resid[x][j]))
    AU=torch.tensor(np.array(AU)); AP=torch.tensor(np.array(AP)); AY=torch.tensor(np.array(AY))
    print(f"  pretrain pairs: {len(AY)} item ratings",flush=True)
    for ep in range(int(os.environ.get('AEP',30))):
        idx=torch.randperm(len(AY)); tl=0.;nb=0
        for b0 in range(0,len(AY),512):
            bb=idx[b0:b0+512]; mu,lv=abot(AU[bb],AP[bb]); lv=lv.clamp(-6,4)
            nll=(0.5*torch.exp(-lv)*(AY[bb]-mu)**2+0.5*lv).mean()
            aopt.zero_grad(); nll.backward(); aopt.step(); tl+=float(nll); nb+=1
        if ep%5==0 or ep==int(os.environ.get('AEP',30))-1: print(f"  ABot ep{ep+1} NLL {tl/nb:.4f}",flush=True)
    torch.save(abot.state_dict(),f'{base}/.cache/abot_items.pt'); abot.eval()
    with torch.no_grad():                                                      # sigma^2 sanity: real items should be MORE confident than random off-pool points
        _,lvi=abot(AU[:3000],AP[:3000])
        _rp=torch.tensor((np.random.default_rng(0).standard_normal((3000,D))).astype(np.float32)); _rp=_rp/(_rp.norm(dim=1,keepdim=True))*float(np.linalg.norm(Q,axis=1).mean())
        _,lvr=abot(AU[:3000],_rp)
        print(f"  sigma^2 check: item logvar {float(lvi.mean()):.3f} vs random-point logvar {float(lvr.mean()):.3f}  (random should be HIGHER = correctly less confident)",flush=True)
        mu_a,_=abot(AU[:8000],AP[:8000]); rmse_a=float(((mu_a-AY[:8000])**2).mean()**0.5); mae_a=float((mu_a-AY[:8000]).abs().mean())
        lin=(AU[:8000]*AP[:8000]).sum(1); aa=float((lin*AY[:8000]).sum()/((lin*lin).sum()+1e-9)); rmse_l=float(((aa*lin-AY[:8000])**2).mean()**0.5)
        print(f"  RATING-PREDICTION (held items, predicts from u* — NOT a lookup): ABot RMSE {rmse_a:.3f} MAE {mae_a:.3f} | best-linear u*.Q RMSE {rmse_l:.3f} | resid std {float(AY.std()):.3f} (=predict-zero baseline)",flush=True)
    nCf=nAf=nGf=m=0.                                                            # diagnostic: fold top-8 popular answerable concepts, cans vs ABot-mu, full NDCG@10
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        cans=cans_np(x,profset); cs=sorted(cans.keys(),key=lambda c:-cfreq[c])[:8]
        if not cs: continue
        ust=enc_u_np([(Q[j],resid[x][j]) for j in profset]).astype(np.float32)
        Ecs=torch.tensor(np.array([Ec[c] for c in cs],np.float32)); ut=torch.tensor(ust)[None].expand(len(cs),-1)
        with torch.no_grad(): mu,lv=abot(ut,Ecs); mu=mu.numpy(); lv=lv.numpy()
        b_cans=enc_u_np([(Ec[c],cans[c]) for c in cs])
        b_ab=enc_u_np([(Ec[c],float(mu[i])) for i,c in enumerate(cs)])
        keep=[i for i in range(len(cs)) if lv[i]<=np.median(lv)]
        b_g=enc_u_np([(Ec[cs[i]],float(mu[i])) for i in keep]) if keep else b_ab
        mc=metr(b_cans,tlike,profset,False); ma=metr(b_ab,tlike,profset,False); mg=metr(b_g,tlike,profset,False)
        if mc and ma and mg: nCf+=mc[0]; nAf+=ma[0]; nGf+=mg[0]; m+=1
    print(f"  DIAGNOSTIC te[300:] ({int(m)} users) | fold top-8 answerable concepts | FULL NDCG@10:",flush=True)
    print(f"    cans (geometric)     : {nCf/m:.4f}",flush=True)
    print(f"    ABot-mu (all 8)      : {nAf/m:.4f}",flush=True)
    print(f"    ABot-mu (conf-gated) : {nGf/m:.4f}",flush=True)
    print(f"  => learned answerer {'BEATS' if nAf>nCf else 'does NOT beat'} cans (full); gating {'helps' if nGf>nAf else 'no help'}",flush=True)
    sys.exit(0)
if os.environ.get('ABOTCO'):                                                   # PHASE 1: bot-play co-train the ABot (QBot=distilled actor FROZEN); ABot learns to answer maximising held-out NDCG
    import sys
    print("=== PHASE 1: bot-play co-train ABot | QBot frozen at distilled actor ===",flush=True)
    _ipop=(cnt/max(cnt.max(),1)).astype(np.float32); Qn=torch.tensor((Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)).astype(np.float32)); _ipt=torch.tensor(_ipop)
    def afeat(ust,pe):
        pu=pe/(pe.norm(dim=1,keepdim=True)+1e-9); tv,ti=(pu@Qn.t()).topk(9,dim=1); tv=tv[:,1:]; ti=ti[:,1:]
        return torch.stack([1-tv[:,0],1-tv.mean(1),(ust*Qn[ti].mean(1)).sum(1),_ipt[ti].mean(1)],1)
    class ABot(nn.Module):
        def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+D+1+4,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU()); s.mu=nn.Linear(256,1); s.lv=nn.Linear(256,1)
        def forward(s,ust,pe): x=torch.cat([ust,pe,(ust*pe).sum(1,keepdim=True),afeat(ust,pe)],1); h=s.f(x); return s.mu(h).squeeze(1),s.lv(h).squeeze(1)
    abot=ABot()
    if os.path.exists(f'{base}/.cache/abot_items.pt') and not os.environ.get('ASCRATCH'): abot.load_state_dict(torch.load(f'{base}/.cache/abot_items.pt')); print("  loaded PRETRAINED ABot init",flush=True)
    for p in actor.parameters(): p.requires_grad_(False)                       # FREEZE QBot (distilled, loaded via INIT)
    _CPHI=bool(os.environ.get('COTRAINPHI')); _ACANS=bool(os.environ.get('ACANS'))   # COTRAINPHI: co-train the recommender fold (enc) ; ACANS: cans-answer control (no learned ABot)
    if _CPHI:
        for p in enc.parameters(): p.requires_grad_(True)
    _prm=(list(abot.parameters()) if not _ACANS else [])+(list(enc.parameters()) if _CPHI else [])
    aopt=torch.optim.Adam(_prm,float(os.environ.get('COLR',3e-4)),weight_decay=float(os.environ.get('COWD',0)))   # COWD: regularise enc co-train (anti-overfit)
    def co_rollout(us,ANS,USTAR):                                              # frozen QBot snaps to answerable entity; ABot (sees TRUE u*) answers; fold; differentiable in ABot
        B=len(us); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D); asked=torch.zeros(B,NP)
        for t in range(T):
            with torch.no_grad():
                q=actor(u,t/8.); sc=(q@POOLnt.t()).masked_fill(asked>0,-1e9).masked_fill(ANS.abs()<1e-6,-1e9); idx=sc.argmax(1)
            oh=torch.zeros(B,NP).scatter_(1,idx.unsqueeze(1),1.0); asked=asked+oh; fe=POOLt[idx]
            a=(oh*ANS).sum(1) if _ACANS else abot(USTAR,fe)[0]                  # ACANS: real cans/resid answer (control) ; else learned ABot mu
            toks=toks.clone(); toks[:,t,:D]=fe; toks[:,t,D]=a; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask)
        return u
    for ep in range(int(os.environ.get('COEP',8))):
        rng.shuffle(trbig); tot=0.;nb=0
        for b0 in range(0,min(len(trbig),6000),96):
            us=trbig[b0:b0+96]; ANS,USTAR,tgt,wt,LIKED,LMASK,RNEG,PROFM=prep(us)
            u=co_rollout(us,ANS,USTAR); loss=recon(u,tgt,wt)
            aopt.zero_grad(); loss.backward(); aopt.step(); tot+=float(loss); nb+=1
        print(f"  co-train ep{ep+1} recon {tot/nb:.4f}",flush=True)
    torch.save(abot.state_dict(),f'{base}/.cache/abot_cotrain.pt'); enc.eval(); abot.eval()
    def test_elic(amode):                                                       # REAL 8-turn elicitation on te[300:] with the (co-trained) enc -> full/tail NDCG, directly comparable to CASPER-R 0.360/0.152
        nf=nt=mf=mt=0.
        for x in TE:
            profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
            if not tlike: continue
            cans=cans_np(x,profset); ut=torch.tensor(enc_u_np([(Q[j],resid[x][j]) for j in profset]).astype(np.float32))[None]; toks=[]; asked=set()
            for t in range(8):
                with torch.no_grad(): q=actor(torch.tensor(enc_u_np(toks)[None],dtype=torch.float32),len(toks)/8.); sc=(q@POOLnt.t()).numpy()[0]
                k=None
                for kk in np.argsort(-sc):
                    kk=int(kk)
                    if kk in asked: continue
                    if kk<NI and PITEMS[kk] in profset: k=kk; break
                    if kk>=NI and (kk-NI) in cans: k=kk; break
                if k is None: break
                asked.add(k)
                if k<NI: toks.append((Q[PITEMS[k]],resid[x][PITEMS[k]]))
                elif amode=='cans': toks.append((Ec[k-NI],cans[k-NI]))
                else:
                    with torch.no_grad(): mu,_=abot(ut,torch.tensor(Ec[k-NI][None])); toks.append((Ec[k-NI],float(mu[0])))
            u=enc_u_np(toks); mfu=metr(u,tlike,profset,False); mta=metr(u,tlike,profset,True)
            if mfu: nf+=mfu[0]; mf+=1
            if mta: nt+=mta[0]; mt+=1
        return nf/max(mf,1), nt/max(mt,1)
    cf,ct=test_elic('cans'); af,at=test_elic('abot')
    print(f"  TEST ELICITATION (te[300:], 8-turn, COTRAINPHI={_CPHI} ACANS={_ACANS}) | vs CASPER-R 0.360/0.152:",flush=True)
    print(f"    cans-answer : FULL {cf:.4f}  TAIL {ct:.4f}",flush=True)
    print(f"    ABot-answer : FULL {af:.4f}  TAIL {at:.4f}",flush=True)
    print(f"  => best FULL {max(cf,af):.4f} / TAIL {max(ct,at):.4f}  ({'BEATS' if max(cf,af)>0.360 or max(ct,at)>0.152 else 'below'} CASPER-R)",flush=True)
    sys.exit(0)
if os.environ.get('CONCORACLE'):                                               # DECISIVE: concept-only best-subset oracle (privileged) -> is there ANY realizable headroom in the answerable concept channel?
    import sys
    _RT=os.environ.get('NORT') is None
    print(f"=== CONCEPT-ONLY best-subset ORACLE (privileged greedy q=8 answerable concepts, cans answers, {'TAIL' if _RT else 'FULL'}-optimised) vs CASPER-R 0.360/0.152 ===",flush=True)
    nf=nt=mf=mt=0.; _Wc=1./np.log2(np.arange(2,12))
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        trel=set(j for j in test if rd[j]>=4 and (not headmask[j] if _RT else True))
        if not trel: continue
        cans=cans_np(x,profset); cs=list(cans.keys())
        if not cs: continue
        toks=[]; chosen=set()
        for t in range(8):
            best=-1.; bk=None
            for c in cs:
                if c in chosen: continue
                u=enc_u_np(toks+[(Ec[c],cans[c])]); s=popb+Ql@u; s[list(profset)]=-1e9
                if _RT: s[headmask]=-1e9
                o=np.argsort(-s)[:10]; nd=sum(_Wc[p] for p,it in enumerate(o) if int(it) in trel)
                if nd>best: best=nd; bk=c
            if bk is None: break
            chosen.add(bk); toks.append((Ec[bk],cans[bk]))
        u=enc_u_np(toks); mfu=metr(u,tlike,profset,False); mta=metr(u,tlike,profset,True)
        if mfu: nf+=mfu[0]; mf+=1
        if mta: nt+=mta[0]; mt+=1
    print(f"  CONCEPT-ONLY oracle: FULL {nf/mf:.4f}  TAIL {nt/mt:.4f}  vs CASPER-R 0.360/0.152  (>> CASPER-R = realizable concept headroom EXISTS; ~= means concepts exhausted)",flush=True)
    sys.exit(0)
if os.environ.get('REALCONC'):                                                 # REALIZABLE greedy concept policy (NO target knowledge) -> how much of the 0.389 privileged-oracle headroom is recoverable?
    import sys
    _RT=os.environ.get('NORT') is None; PSM=int(os.environ.get('PSM','10'))
    modes=os.environ.get('RCMODES','pop,eig,shift,pseudo').split(',')           # pop/eig = FULLY realizable; shift/pseudo = answer-peek (sees real answers, not targets) = semi-oracle
    print(f"=== REALIZABLE greedy concept policy ({'TAIL' if _RT else 'FULL'}-eval) modes={modes} vs CASPER-R 0.360/0.152, privileged-oracle 0.369/0.389 ===",flush=True)
    res={m:[0.,0.,0.,0.] for m in modes}
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        cans=cans_np(x,profset); cs=list(cans.keys())
        if not cs: continue
        for m in modes:
            toks=[]; chosen=set(); u=enc_u_np(toks)
            for t in range(8):
                cand=[c for c in cs if c not in chosen]
                if not cand: break
                if m=='pop': bk=max(cand,key=lambda c:cfreq[c])
                elif m=='eig': bk=min(cand,key=lambda c:abs(float(u@Ec[c])))     # most divisive given CURRENT belief (realizable EIG proxy)
                elif m=='shift':                                                 # answer-peek: fold real answer, pick max belief movement
                    best=-1.; bk=cand[0]
                    for c in cand:
                        d=float(np.linalg.norm(enc_u_np(toks+[(Ec[c],cans[c])])-u))
                        if d>best: best=d; bk=c
                elif m=='pseudo':                                                # answer-peek: belief top-M as pseudo-targets, greedy pseudo-NDCG
                    s=popb+Ql@u; s[list(profset)]=-1e9
                    if _RT: s[headmask]=-1e9
                    pm=set(int(i) for i in np.argsort(-s)[:PSM]); best=-1.; bk=cand[0]
                    for c in cand:
                        u2=enc_u_np(toks+[(Ec[c],cans[c])]); s2=popb+Ql@u2; s2[list(profset)]=-1e9
                        if _RT: s2[headmask]=-1e9
                        o=np.argsort(-s2)[:10]; nd=sum(_W[p] for p,it in enumerate(o) if int(it) in pm)
                        if nd>best: best=nd; bk=c
                chosen.add(bk); toks.append((Ec[bk],cans[bk])); u=enc_u_np(toks)
            mfu=metr(u,tlike,profset,False); mta=metr(u,tlike,profset,True)
            if mfu: res[m][0]+=mfu[0]; res[m][2]+=1
            if mta: res[m][1]+=mta[0]; res[m][3]+=1
    for m in modes:
        a=res[m]; print(f"  {m:8s}: FULL {a[0]/max(a[2],1):.4f}  TAIL {a[1]/max(a[3],1):.4f}",flush=True)
    sys.exit(0)
if os.environ.get('FTRA'):                                                     # USER'S DAHCR architecture: GRU(recurrent)+MHSA(attention) encoder, trained on ALL sequence lengths (1..full profile, real ratings). Goal: a GENERAL recommender (strong full-profile + robust), not narrow.
    import sys
    H=int(os.environ.get('RAH','128')); _CKra=f"{base}/.cache/{os.environ.get('RASAVE','ftra')}_best.pt"
    class EncRA(nn.Module):
        def __init__(s):
            super().__init__(); s.inp=nn.Linear(D+1,H); s.attn=nn.MultiheadAttention(H,int(os.environ.get('RAHEADS','4')),batch_first=True); s.gru=nn.GRU(H,H,batch_first=True); s.out=nn.Linear(H,D)
        def forward(s,t,m):
            x=s.inp(t); a,_=s.attn(x,x,x,key_padding_mask=(m==0)); x=torch.relu(x+a)
            o,_=s.gru(x); L=m.sum(1).long().clamp(min=1); return s.out(o[torch.arange(len(o)),L-1])
    class EncS(nn.Module):                                                     # ATTRIBUTION: simple attention-pool (frozen arch), trainable, SAME recipe -> isolates architecture vs training recipe
        def __init__(s):
            super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU()); s.att=nn.Linear(H,1); s.val=nn.Linear(H,D)
        def forward(s,t,m):
            h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
    era=(EncS() if os.environ.get('ARCH')=='simple' else EncRA()); Qlp=torch.nn.Parameter(torch.tensor(Ql)); popbt2=torch.tensor(popb)
    opt3=torch.optim.Adam(list(era.parameters())+[Qlp],float(os.environ.get('RALR','1e-3')),weight_decay=float(os.environ.get('RAWD','1e-5')))
    def udat(users,cap):
        D2=[]
        for x in users[:cap]:
            allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); hh=max(len(allit)//2,4); prof=allit[:hh]; held=allit[hh:]
            rd=dict(rat_by_u[x]); hl=[j for j in held if rd[j]>=4]
            if hl and len(prof)>=2: D2.append((x,prof,hl,set(prof)))
        return D2
    TR=udat([x for x in trbig],int(os.environ.get('RAN','2500'))); VA=udat([x for x in te if x in _VSPL],300)
    def buildtok(items_list,xs):
        mx=max(len(p) for p in items_list); B=len(items_list); tok=torch.zeros(B,mx,D+1); msk=torch.zeros(B,mx)
        for b,(its,x) in enumerate(zip(items_list,xs)):
            for q,j in enumerate(its): tok[b,q,:D]=torch.tensor(Q[j].astype(np.float32)); tok[b,q,D]=resid[x][j]; msk[b,q]=1
        return tok,msk
    def valnd():
        nd=0.;c=0
        for b0 in range(0,len(VA),128):
            ch=VA[b0:b0+128]; tok,msk=buildtok([d[1] for d in ch],[d[0] for d in ch])
            with torch.no_grad(): sc=(era(tok,msk)@Qlp.t()+popbt2).numpy()
            for i,(x,prof,hl,ps) in enumerate(ch):
                s=sc[i].copy(); s[list(ps)]=-1e9; o=np.argsort(-s); rel=set(hl); nd+=sum(_W[p] for p,it in enumerate(o[:10]) if int(it) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); c+=1
        return nd/c
    _RAMIX=float(os.environ.get('RAMIX','0')); _ordE=sorted(range(NP),key=lambda k:-POOL_ENT[k])
    enc_a2=Enc(); enc_a2.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc_a2.eval()   # frozen taste for graded answers
    for p in enc_a2.parameters(): p.requires_grad_(False)
    def _usof(prof,x):
        with torch.no_grad(): return enc_a2(*toks2t([(Q[j],resid[x][j]) for j in prof]))[0].numpy().astype(np.float32)
    TRus=[_usof(d[1],d[0]) for d in TR] if _RAMIX>0 else None
    def buildtok_graded(seqs,usl):                                            # (entity, graded answer us.entity) sequences
        mx=max(len(s) for s in seqs); B=len(seqs); tok=torch.zeros(B,mx,D+1); msk=torch.zeros(B,mx)
        for b,(seq,us) in enumerate(zip(seqs,usl)):
            for q,k in enumerate(seq): tok[b,q,:D]=torch.tensor(POOL[k].astype(np.float32)); tok[b,q,D]=float(us@POOL[k]); msk[b,q]=1
        return tok,msk
    best=-1.
    for ep in range(0 if os.environ.get('RALOAD') else int(os.environ.get('RAEP','30'))):
        oi=list(range(len(TR))); rng.shuffle(oi); tl=0.;nb=0
        for b0 in range(0,len(oi),128):
            bi=oi[b0:b0+128]; ch=[TR[i] for i in bi]
            if _RAMIX>0 and float(rng.random())<_RAMIX:                       # GRADED-QUESTION batch (elicitation distribution: entropy or random Qs over items+concepts)
                seqs=[]; usl=[]
                for i in bi:
                    Lq=int(rng.integers(1,9)); seqs.append(_ordE[:Lq] if float(rng.random())<0.5 else list(rng.integers(0,NP,Lq))); usl.append(TRus[i])
                tok,msk=buildtok_graded(seqs,usl)
            else:                                                            # REAL-RATING profile prefix (all lengths)
                its=[]; xs=[]
                for (x,prof,hl,ps) in ch:
                    L=int(rng.integers(1,len(prof)+1)); pp=prof[:]; rng.shuffle(pp); its.append(pp[:L]); xs.append(x)
                tok,msk=buildtok(its,xs)
            sc=era(tok,msk)@Qlp.t()+popbt2; loss=0.
            for ii,(x,prof,hl,ps) in enumerate(ch):
                pos=sc[ii,hl]; neg=sc[ii,torch.randint(0,ni,(len(hl)*5,))]; loss=loss-torch.log(torch.sigmoid(pos.unsqueeze(1)-neg.unsqueeze(0))+1e-9).mean()
            loss=loss/len(ch); opt3.zero_grad(); loss.backward(); opt3.step(); tl+=float(loss);nb+=1
        v=valnd()
        if v>best: best=v; torch.save({'era':era.state_dict(),'Ql':Qlp.detach().clone()},_CKra); _m=' *'
        else: _m=''
        if ep%5==0 or ep==int(os.environ.get('RAEP','30'))-1: print(f"  FTRA ep{ep+1} loss {tl/nb:.4f} val-fullprof {v:.4f}{_m}",flush=True)
    ck=torch.load(_CKra); era.load_state_dict(ck['era']); era.eval(); Qlf=ck['Ql'].numpy().astype(np.float32)
    def evtest(make_belief,label,ref):
        nf=nt=cf=ct=0.
        for x in TE:
            profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
            if not tlike: continue
            u=make_belief(profset,x)
            s=(popb+Qlf@u).copy(); s[list(profset)]=-1e9
            of=np.argsort(-s); relf=set(tlike); ndf=sum(_W[p] for p,it in enumerate(of[:10]) if int(it) in relf)/(_W[:min(10,len(relf))].sum()+1e-12)
            st=s.copy(); st[headmask]=-1e9; relt=set(t for t in tlike if not headmask[t])
            if relf: nf+=ndf; cf+=1
            if relt: ot=np.argsort(-st); nt+=sum(_W[p] for p,it in enumerate(ot[:10]) if int(it) in relt)/(_W[:min(10,len(relt))].sum()+1e-12); ct+=1
        print(f"  FTRA TEST {label}: FULL {nf/max(cf,1):.4f}  TAIL {nt/max(ct,1):.4f}  | {ref}",flush=True)
    def _bfull(profset,x):
        tok,msk=buildtok([list(profset)],[x])
        with torch.no_grad(): return era(tok,msk)[0].numpy()
    def _belic(profset,x):                                                   # entropy8 graded-answer elicitation via the trained encoder
        us=_usof(list(profset),x); tok,msk=buildtok_graded([_ordE[:8]],[us])
        with torch.no_grad(): return era(tok,msk)[0].numpy()
    evtest(_bfull,f"full-profile (val peak {best:.4f})","vs frozen 0.408/0.214")
    if _RAMIX>0 or os.environ.get('RAELIC'): evtest(_belic,"entropy8-elicitation","vs frozen-elicit 0.367/0.158, FTREC 0.385/0.165")
    sys.exit(0)
if os.environ.get('SHUF'):                                                     # ORDER-SENSITIVITY: fold the SAME entropy8 answers in many random orders -> per-user STD of NDCG across orders. attn-pool=order-invariant (STD~0); GRU=order-sensitive risk.
    import sys
    enc_a=Enc(); enc_a.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc_a.eval()
    for p in enc_a.parameters(): p.requires_grad_(False)
    _ordE=sorted(range(NP),key=lambda k:-POOL_ENT[k])[:8]; rngr=np.random.default_rng(1); _NS=int(os.environ.get('NSHUF','10'))
    def _bel(usf,seq): return enc_u_np([(POOL[k],(POS if float(usf@POOL[k])>0 else NEG)) for k in seq])
    def _nd(u,tlike,prof):
        s=(popb+Ql@u).copy(); s[list(prof)]=-1e9; o=np.argsort(-s); rel=set(tlike)
        return sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    stds=[]; means=[]; rng2=np.random.default_rng(0)
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        usf=enc_a(*toks2t([(Q[j],resid[x][j]) for j in profset]))[0].numpy()
        nds=[_nd(_bel(usf,list(rng2.permutation(_ordE))),tlike,profset) for _ in range(_NS)]
        stds.append(float(np.std(nds))); means.append(float(np.mean(nds)))
    print(f"=== SHUFFLE order-sensitivity [{os.path.basename(os.environ.get('LOADREC','FROZEN'))}] entropy8, {_NS} random orders/user ===",flush=True)
    print(f"  mean NDCG {np.mean(means):.4f} | mean per-user STD across orders {np.mean(stds):.4f}  (0 = order-invariant; larger = order-sensitive)",flush=True)
    sys.exit(0)
if os.environ.get('ROBUST'):                                                   # ROBUSTNESS: a recommender (LOADREC, or frozen) ranking beliefs from DIFFERENT question sequences (entropy/pop/random/full-profile), FIXED user answers (original enc). Shows generality vs brittleness.
    import sys
    enc_a=Enc(); enc_a.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc_a.eval()   # fixed user taste (answers)
    for p in enc_a.parameters(): p.requires_grad_(False)
    ordE=sorted(range(NP),key=lambda k:-POOL_ENT[k])[:8]; ordP=sorted(range(NP),key=lambda k:-PRIOR[k])[:8]
    ordC=[k for k in sorted(range(NP),key=lambda k:-POOL_ENT[k]) if k>=NI][:8]   # concept-only entropy (Paper-B style policy, graded answers here)
    nq=int(os.environ.get('RQ','8')); rngr=np.random.default_rng(1)
    _gw=os.environ.get('GRAW')                                                   # GRAW=1 -> graded answers (Paper-C); else BINARY geometric sign (Paper-A/B convention)
    def belief(usf,seq): return enc_u_np([(POOL[k],(float(usf@POOL[k]) if _gw else (POS if float(usf@POOL[k])>0 else NEG))) for k in seq])
    def ndcg(u,tlike,prof,tail):
        s=(popb+Ql@u).copy(); s[list(prof)]=-1e9
        if tail: s[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
        else: rel=set(tlike)
        if not rel: return None
        o=np.argsort(-s); return sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    keys=['item_entropy8','conc_entropy8','pop','random','fullprof']; agg={k:[0.,0.,0,0] for k in keys}
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        usf=enc_a(*toks2t([(Q[j],resid[x][j]) for j in profset]))[0].numpy()
        ev={'item_entropy8':belief(usf,ordE[:nq]),'conc_entropy8':belief(usf,ordC[:nq]),'pop':belief(usf,ordP[:nq]),
            'random':belief(usf,list(rngr.integers(0,NP,nq))),'fullprof':enc_u_np([(Q[j],resid[x][j]) for j in profset])}
        for k,u in ev.items():
            f=ndcg(u,tlike,profset,False); t=ndcg(u,tlike,profset,True)
            if f is not None: agg[k][0]+=f; agg[k][2]+=1
            if t is not None: agg[k][1]+=t; agg[k][3]+=1
    print(f"=== ROBUSTNESS [{os.path.basename(os.environ.get('LOADREC','FROZEN'))}] (fixed answers, varied sequences) ===",flush=True)
    for k in keys: f,t,cf,ct=agg[k]; print(f"  {k:11s}: FULL {f/max(cf,1):.4f}  TAIL {t/max(ct,1):.4f}",flush=True)
    sys.exit(0)
if os.environ.get('POLOPT'):                                                   # STAGE 2: OPTIMISE the policy against the (de-OOD'd, policy-agnostic) recommender loaded via LOADREC. Residual-on-entropy policy + REINFORCE on true NDCG. Answers from ORIGINAL frozen 'user' enc (taste fixed); belief/rank from loaded recommender.
    import sys
    enc_a=Enc(); enc_a.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc_a.eval()   # fixed user-taste encoder (answers)
    ENTt=torch.tensor(POOL_ENT.astype(np.float32)); _TAU=float(os.environ.get('PTAU','0.5'))
    class Pol(nn.Module):                                                      # score_k = beta*entropy_k + g(belief, entity)  (g init 0 -> pure entropy warm-start)
        def __init__(s):
            super().__init__(); s.g=nn.Sequential(nn.Linear(2*D,64),nn.ReLU(),nn.Linear(64,1)); s.b=nn.Parameter(torch.tensor(1.0))
            nn.init.zeros_(s.g[-1].weight); nn.init.zeros_(s.g[-1].bias)
        def forward(s,u):
            B=u.shape[0]; x=torch.cat([u.unsqueeze(1).expand(B,NP,D),POOLt.unsqueeze(0).expand(B,NP,D)],2); return s.b*ENTt.unsqueeze(0)+s.g(x).squeeze(-1)
    pol=Pol(); popt=torch.optim.Adam(pol.parameters(),float(os.environ.get('PLR','3e-5')))
    def udata(users,cap):
        US=[];HL=[];PR=[]
        for x in users[:cap]:
            allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); hh=max(len(allit)//2,4); prof=set(allit[:hh]); held=allit[hh:]
            rd=dict(rat_by_u[x]); hl=[j for j in held if rd[j]>=4]
            if not hl: continue
            with torch.no_grad(): us=enc_a(*toks2t([(Q[j],resid[x][j]) for j in prof]))[0].numpy()
            US.append(us.astype(np.float32)); HL.append(hl); PR.append(list(prof))
        return torch.tensor(np.array(US,np.float32)),HL,PR
    def rollout(usb,greedy=False):                                            # returns final belief u (B,D), logps list ; answers = usb.entity (graded), belief via loaded enc (frozen)
        B=usb.shape[0]; toks=torch.zeros(B,T,D+1); msk=torch.zeros(B,T); u=torch.zeros(B,D); asked=torch.zeros(B,NP); ar=torch.arange(B); logps=[]
        for t in range(T):
            sc=pol(u).masked_fill(asked>0,-1e9); p=torch.softmax(sc/_TAU,1)
            idx=p.argmax(1) if greedy else torch.multinomial(p,1).squeeze(1)
            logps.append(torch.log(p[ar,idx]+1e-9)); asked=asked+torch.zeros(B,NP).scatter_(1,idx.unsqueeze(1),1.)
            with torch.no_grad():
                ans=(usb*POOLt[idx]).sum(1); toks=toks.clone(); toks[:,t,:D]=POOLt[idx]; toks[:,t,D]=ans; msk=msk.clone(); msk[:,t]=1; u=enc(toks,msk)
        return u,logps
    def nd_batch(u,HL,PR,tail):
        s=(u@Qlt.t()+popbt).numpy(); out=np.zeros(len(HL))
        for i in range(len(HL)):
            ss=s[i].copy(); ss[PR[i]]=-1e9
            if tail: ss[headmask]=-1e9; rel=set(j for j in HL[i] if not headmask[j])
            else: rel=set(HL[i])
            if not rel: continue
            o=np.argsort(-ss)[:10]; out[i]=sum(_W[p] for p,it in enumerate(o) if int(it) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
        return torch.tensor(out,dtype=torch.float32)
    print("POLOPT: building user data...",flush=True)
    trUS,trHL,trPR=udata([x for x in trbig],int(os.environ.get('POLN','2500')))
    _RT=bool(os.environ.get('REWTAIL')); _EB=float(os.environ.get('PENT','0.01'))
    VAL_U=[x for x in te if x in _VSPL][:300]                                  # honest early-stop on VAL (te[:300]); TEST = te[300:] reported at best-val ckpt
    def evalpol(users):
        nf=nt=cf=ct=0.
        for x in users:
            allit=dict(rat_by_u[x])
            if x in SPL: profset,test=SPL[x]
            else:
                its=list(allit); _r2=np.random.default_rng(0); _r2.shuffle(its); profset=set(its[:len(its)//2]); test=its[len(its)//2:]
            tlike=set(j for j in test if allit[j]>=4)
            if not tlike: continue
            with torch.no_grad(): usb=enc_a(*toks2t([(Q[j],resid[x][j]) for j in profset]))[0]
            u,_=rollout(usb.unsqueeze(0),greedy=True)
            mf=nd_batch(u,[list(tlike)],[list(profset)],False); mt=nd_batch(u,[list(tlike)],[list(profset)],True)
            nf+=float(mf[0]); cf+=1; nt+=float(mt[0]); ct+=1
        return nf/max(cf,1),nt/max(ct,1)
    f0,t0=evalpol(TE); print(f"  POLOPT init (=entropy on this recommender) TEST: FULL {f0:.4f} TAIL {t0:.4f}",flush=True)
    best=-1.; bf=bt=0.; _CKp=f'{base}/.cache/polopt_best.pt'
    for ep in range(int(os.environ.get('POLEP','25'))):
        perm=torch.randperm(len(trUS)); tl=0.;nb=0
        for b0 in range(0,len(trUS),64):
            bb=perm[b0:b0+64]; usb=trUS[bb]; u,logps=rollout(usb,greedy=False)
            r=nd_batch(u,[trHL[i] for i in bb.tolist()],[trPR[i] for i in bb.tolist()],_RT)
            ent=-(torch.stack(logps,1)).mean()                                # crude entropy proxy (encourage exploration)
            adv=r-r.mean(); loss=-(torch.stack(logps,1).sum(1)*adv).mean()-_EB*ent
            popt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(),5.); popt.step(); tl+=float(r.mean());nb+=1
        vf,vt=evalpol(VAL_U); sel=vt if _RT else vf
        if sel>best: best=sel; tf,tt=evalpol(TE); bf,bt=tf,tt; torch.save(pol.state_dict(),_CKp); _m=f' * TEST {tf:.4f}/{tt:.4f}'
        else: _m=''
        print(f"  POLOPT ep{ep+1} train-NDCG {tl/nb:.4f} | VAL full {vf:.4f} tail {vt:.4f}{_m}",flush=True)
    print(f"  POLOPT done. init TEST {f0:.4f}/{t0:.4f} -> best-val TEST {bf:.4f}/{bt:.4f}",flush=True)
    sys.exit(0)
if os.environ.get('FTREC'):                                                    # USER IDEA: FINE-TUNE the recommender (enc+Ql) on the ELICITATION distribution (same 8 entropy Qs, graded answers, fold-curriculum t=1..8) -> remove the OOD/partial-belief mismatch + make the entropy signal maximally useful. Frozen enc generates the ANSWER (fixed user taste); trainable enc builds the BELIEF.
    import sys
    enc_f=Enc(); enc_f.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc_f.eval()
    order=sorted(range(NP),key=lambda k:-POOL_ENT[k])[:int(os.environ.get('FTQ','8'))]; Eord=torch.tensor(np.array([POOL[k] for k in order],np.float32))   # the fixed entropy questionnaire
    for p in enc.parameters(): p.requires_grad_(True)
    _fp=[p.detach().clone() for p in enc.parameters()]; _ANC=float(os.environ.get('FTANCHOR','0'))   # L2 anchor to frozen encoder (anti-drift regularizer)
    Qlp=torch.nn.Parameter(torch.tensor(Ql)); popbt2=torch.tensor(popb); _Qlf0=torch.tensor(Ql)
    opt2=torch.optim.Adam(list(enc.parameters())+[Qlp],float(os.environ.get('FTLR','1e-4')),weight_decay=float(os.environ.get('FTWD','1e-4')))
    Eordn=np.array([POOL[k] for k in order],np.float32)
    def prep_ft(users,cap,K=1):                                                # returns u* (frozen profile-encoding) per example; answer to ANY question = us.entity computed on the fly
        US=[];H=[];P=[]
        for x in users[:cap]:
            for _ in range(K):
                allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); hh=max(len(allit)//2,4); prof=set(allit[:hh]); held=allit[hh:]
                rd=dict(rat_by_u[x]); hl=[j for j in held if rd[j]>=4]
                if not hl: continue
                with torch.no_grad(): us=enc_f(*toks2t([(Q[j],resid[x][j]) for j in prof]))[0].numpy()
                US.append(us.astype(np.float32)); H.append(hl); P.append(list(prof))
        return torch.tensor(np.array(US,np.float32)),H,P
    _RP=os.environ.get('RANDPOL'); ordt=torch.tensor(order)
    if not os.environ.get('FTLOAD'):
        print(f"FTREC: building data (TRAIN policy={'RANDOM (policy-agnostic)' if _RP else 'entropy-static'}, aug K={os.environ.get('FTAUG','4')})...",flush=True)
        trUS,trH,_=prep_ft([x for x in trbig],int(os.environ.get('FTN','2500')),int(os.environ.get('FTAUG','4')))
        vaUS,vaH,vaP=prep_ft([x for x in te if x in _VSPL],300,1)
    def tok_from(usb,idx):                                                     # usb (B,D), idx (B,t) entity ids -> tokens (B,t,D+1) with GRADED answers us.entity
        Eemb=POOLt[idx]; ans=(usb.unsqueeze(1)*Eemb).sum(-1); return torch.cat([Eemb,ans.unsqueeze(-1)],-1),torch.ones(idx.shape[0],idx.shape[1])
    def val_nd():                                                             # eval ALWAYS on the deployed entropy policy (top divisive), regardless of train policy
        with torch.no_grad():
            idx=ordt.unsqueeze(0).expand(len(vaUS),len(order)); tok,m=tok_from(vaUS,idx); sc=(enc(tok,m)@Qlp.t()+popbt2).numpy(); nd=0.;c=0
            for i,hl in enumerate(vaH):
                s=sc[i].copy(); s[vaP[i]]=-1e9; o=np.argsort(-s); rel=set(hl); nd+=sum(_W[p] for p,it in enumerate(o[:10]) if int(it) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); c+=1
            return nd/c
    _CKf=f"{base}/.cache/{os.environ.get('FTSAVE','ftrec')}_best.pt"
    if not os.environ.get('FTLOAD'): print(f"  FTREC FROZEN-baseline val-nd {val_nd():.4f} (fine-tune must beat THIS to be real)",flush=True)
    best=-1.
    for ep in range(0 if os.environ.get('FTLOAD') else int(os.environ.get('FTEP','40'))):
        perm=torch.randperm(len(trUS)); tl=0.;nb=0
        for b0 in range(0,len(trUS),128):
            bb=perm[b0:b0+128]; usb=trUS[bb]; t=int(rng.integers(1,int(os.environ.get('FTMAXQ','8'))+1))
            _userand=_RP or (float(os.environ.get('RANDMIX','0'))>0 and float(rng.random())<float(os.environ.get('RANDMIX','0')))   # RANDMIX=p -> p fraction of batches use random questions (robustness), rest entropy (specialisation) = strong AND not brittle
            if _userand:                                                       # RANDTEMP=temp -> sample weighted by entropy (cover divisive region, not dilute on useless entities)
                if os.environ.get('RANDTEMP'): idx=torch.multinomial(torch.softmax(torch.tensor(POOL_ENT.astype(np.float32))/float(os.environ['RANDTEMP']),0).unsqueeze(0).expand(len(bb),NP),t,replacement=False)
                else: idx=torch.randint(0,NP,(len(bb),t))
            else: idx=ordt[:min(t,len(order))].unsqueeze(0).expand(len(bb),min(t,len(order)))
            tok,m=tok_from(usb,idx); sc=enc(tok,m)@Qlp.t()+popbt2
            loss=0.; _TW=os.environ.get('TAILW')
            for ii,gi in enumerate(bb.tolist()):
                hl=trH[gi]; pos=sc[ii,hl]; neg=sc[ii,torch.randint(0,ni,(len(hl)*5,))]
                bpr=-torch.log(torch.sigmoid(pos.unsqueeze(1)-neg.unsqueeze(0))+1e-9).mean(1)   # per-positive BPR
                if _TW: w=torch.tensor([float(ipsw[j]) for j in hl]); loss=loss+(bpr*w).sum()/w.sum()   # inverse-pop weight -> optimise TAIL ranking
                else: loss=loss+bpr.mean()
            loss=loss/len(bb)
            if _ANC>0: loss=loss+_ANC*(sum(((p-pf)**2).sum() for p,pf in zip(enc.parameters(),_fp))+((Qlp-_Qlf0)**2).sum())   # anchor to frozen recommender
            opt2.zero_grad(); loss.backward(); opt2.step(); tl+=float(loss);nb+=1
        v=val_nd()
        if v>best: best=v; torch.save({'enc':enc.state_dict(),'Ql':Qlp.detach().clone()},_CKf); _m=' *'
        else: _m=''
        if ep%5==0 or ep==int(os.environ.get('FTEP','40'))-1: print(f"  FTREC ep{ep+1} loss {tl/nb:.4f} val-nd {v:.4f}{_m}",flush=True)
    ck=torch.load(_CKf); enc.load_state_dict(ck['enc']); enc.eval(); Qlf=ck['Ql'].numpy().astype(np.float32)   # eval TEST with fine-tuned recommender
    print(f"=== FTREC eval (te[300:], same entropy+GRAW belief) | baseline 0.367/0.158 ===",flush=True)
    nf=nt=cf=ct=0.
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        with torch.no_grad():
            usf=enc_f(*toks2t([(Q[j],resid[x][j]) for j in profset]))[0].numpy()
            toks=[(POOL[k],float(usf@POOL[k])) for k in order]; u=enc(*toks2t(toks))[0].numpy()
        s_full=(popb+Qlf@u).copy(); s_full[list(profset)]=-1e9
        of=np.argsort(-s_full); relf=set(tlike); ndf=sum(_W[p] for p,it in enumerate(of[:10]) if int(it) in relf)/(_W[:min(10,len(relf))].sum()+1e-12)
        st=s_full.copy(); st[headmask]=-1e9; relt=set(t for t in tlike if not headmask[t])
        if relf: nf+=ndf; cf+=1
        if relt: ot=np.argsort(-st); nt+=sum(_W[p] for p,it in enumerate(ot[:10]) if int(it) in relt)/(_W[:min(10,len(relt))].sum()+1e-12); ct+=1
    print(f"  FTREC: FULL {nf/max(cf,1):.4f}  TAIL {nt/max(ct,1):.4f}  (val-nd peak {best:.4f})",flush=True)
    sys.exit(0)
if os.environ.get('READOUT'):                                                  # USER HYPOTHESIS: discriminative info is IN the belief but the RECOMMENDER readout (score=popb+Ql.u) MIS-WEIGHTS it (cos up, NDCG down). Learn a reweighting W on the ELICITED-belief distribution -> better NDCG, policy UNCHANGED. (recommender-side lever, orthogonal to the policy.)
    import sys
    def uent_belief(x,profset):                                                # elicited belief via the WINNER uent+GRAW (entropy selection over answerable pool, graded geometric answers)
        ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset]); order=sorted(range(NP),key=lambda k:-POOL_ENT[k]); toks=[]
        for k in order:
            if len(toks)>=8: break
            toks.append((POOL[k],float(ustar@POOL[k])))
        return enc_u_np(toks)
    def build(users,cap):
        U=[];Hs=[]
        for x in users[:cap]:
            allit=[j for j,_ in rat_by_u[x]]; rng.shuffle(allit); hh=max(len(allit)//2,4); prof=set(allit[:hh]); held=allit[hh:]
            rd=dict(rat_by_u[x]); hl=[j for j in held if rd[j]>=4]
            if not hl: continue
            U.append(uent_belief(x,prof)); Hs.append(hl)
        return np.array(U,np.float32),Hs
    print("READOUT: building elicited beliefs (train)...",flush=True)
    trUb,trH=build([x for x in trbig],int(os.environ.get('RDN','1500')))
    Ut=torch.tensor(trUb); Qlt2=torch.tensor(Ql); popbt2=torch.tensor(popb); ipt=torch.tensor(ipsw.astype(np.float32))
    TGT=torch.zeros(len(trUb),ni); WT=torch.ones(len(trUb),ni)
    for i,hl in enumerate(trH):
        TGT[i,hl]=1.
        for j in hl: WT[i,j]=float(ipsw[j])
    posw=WT*TGT; WT=WT*TGT + (1-TGT)*(posw.sum(1,keepdim=True)/max(ni,1))       # balance: total neg weight ~ total pos weight per user
    Wm=torch.eye(D,requires_grad=True); bvec=torch.zeros(D,requires_grad=True)  # init identity => exactly the baseline at ep0
    ro=torch.optim.Adam([Wm,bvec],float(os.environ.get('RDLR','3e-3')),weight_decay=float(os.environ.get('RDWD','1e-3')))
    for ep in range(int(os.environ.get('RDEP','80'))):
        perm=torch.randperm(len(Ut)); tl=0.;nb=0
        for b0 in range(0,len(Ut),128):
            bb=perm[b0:b0+128]; sc=(Ut[bb]@Wm.t()+bvec)@Qlt2.t()+popbt2
            loss=(WT[bb]*nn.functional.binary_cross_entropy_with_logits(sc,TGT[bb],reduction='none')).sum()/WT[bb].sum()
            ro.zero_grad(); loss.backward(); ro.step(); tl+=float(loss); nb+=1
        if ep%20==0 or ep==int(os.environ.get('RDEP','80'))-1: print(f"  READOUT ep{ep+1} BCE {tl/nb:.4f}",flush=True)
    Wd=Wm.detach().numpy().astype(np.float32); bd=bvec.detach().numpy().astype(np.float32)
    def nd_w(u,tlike,excl,tail,useW):
        uu=(Wd@u+bd) if useW else u; s=(popb+Ql@uu).copy(); s[list(excl)]=-1e9
        if tail: s[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
        else: rel=set(tlike)
        if not rel: return None
        o=np.argsort(-s); return sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    print(f"=== READOUT eval (uent+GRAW belief, te[300:]) | baseline 0.367/0.158 ===",flush=True)
    a=[0.,0.,0.,0.,0,0]
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        u=uent_belief(x,profset)
        f0=nd_w(u,tlike,profset,False,False); f1=nd_w(u,tlike,profset,False,True)
        t0=nd_w(u,tlike,profset,True,False); t1=nd_w(u,tlike,profset,True,True)
        if f0 is not None: a[0]+=f0; a[1]+=f1; a[4]+=1
        if t0 is not None: a[2]+=t0; a[3]+=t1; a[5]+=1
    print(f"  FULL: baseline {a[0]/a[4]:.4f} -> learned-readout {a[1]/a[4]:.4f}",flush=True)
    print(f"  TAIL: baseline {a[2]/a[5]:.4f} -> learned-readout {a[3]/a[5]:.4f}",flush=True)
    sys.exit(0)
if os.environ.get('ARECON'):                                                   # OPTIMIZATION GATE: is there realizable ADAPTIVE headroom? Greedy sequential reconstruction of u* (TARGET-INDEPENDENT objective -> u* is the OBSERVABLE profile-encoding, so this oracle's selection IS learnable, unlike the NDCG oracle's). If >> static uent+GRAW 0.367/0.158 -> adaptivity is worth learning.
    import sys
    GRAW=os.environ.get('GRAW'); _CONLY=os.environ.get('CONLY'); _IONLY=os.environ.get('IONLY')
    cand_all=[k for k in range(NP) if (not _CONLY or k>=NI) and (not _IONLY or k<NI)]
    def _ans(ustar,k):
        d=float(ustar@POOL[k]); return d if GRAW else (POS if d>0 else NEG)
    print(f"=== ADAPTIVE-RECON oracle (greedy min||u-u*||, GRAW={bool(GRAW)}, pool={'conc' if _CONLY else 'item' if _IONLY else 'unified'}) vs static uent+GRAW 0.367/0.158, ceiling 0.407/0.21 ===",flush=True)
    nf=nt=cf=ct=0.; ni_a=0.; nu=0
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset]); toks=[]; chosen=set(); nia=0
        for t in range(8):
            best=1e9; bk=None
            for k in cand_all:
                if k in chosen: continue
                u2=enc_u_np(toks+[(POOL[k],_ans(ustar,k))]); d=float(np.linalg.norm(u2-ustar))
                if d<best: best=d; bk=k
            if bk is None: break
            chosen.add(bk); toks.append((POOL[bk],_ans(ustar,bk)));
            if bk<NI: nia+=1
        u=enc_u_np(toks); mfu=metr(u,tlike,profset,False); mta=metr(u,tlike,profset,True)
        if mfu: nf+=mfu[0]; cf+=1
        if mta: nt+=mta[0]; ct+=1
        ni_a+=nia; nu+=1
    print(f"  ADAPTIVE-RECON: FULL {nf/max(cf,1):.4f}  TAIL {nt/max(ct,1):.4f}  | items {ni_a/max(nu,1):.2f}/8  ({nu} users)",flush=True)
    sys.exit(0)
if os.environ.get('UNIANS'):                                                   # ON-OBJECTIVE: unified pool (items+concepts); ITEMS answerable on a CONTINUOUS scale via the GEOMETRIC derivation (no trained ABot) -> unlock the ITEM headroom. Confidence = |cos(u*,emb)|; abstain when taste-alignment too weak to derive (tunable CTAU).
    import sys
    _RT=os.environ.get('NORT') is None; _Wc=1./np.log2(np.arange(2,12))
    modes=os.environ.get('UMODES','uent,upop,uoracle').split(',')
    GRAD=os.environ.get('GRADED'); CTAU=float(os.environ.get('CTAU','0'))       # GRADED=1 -> graded answer (NEG..POS by cos) else cans-style sign; CTAU = min |cos(u*,emb)| confidence to respond (0=always answer)
    POOLnrm=np.linalg.norm(POOL,axis=1)+1e-9
    GRAW=os.environ.get('GRAW')                                                 # GRAW: graded answer = RAW dot u*.entity = the PREDICTED rating residual (resid-scale, in-distribution, carries magnitude) — the real continuous-answer lever (vs CASPER-R's +-1 bit)
    def ansconf(ustar,k):                                                       # the 'old way' geometric answer, EXTENDED to items: derived affinity + confidence (|alignment|)
        d=float(ustar@POOL[k]); cf=d/(POOLnrm[k]*(np.linalg.norm(ustar)+1e-9))
        if GRAW: a=d                                                            # predicted-residual graded answer (clip to plausible rating range)
        elif GRAD: a=NEG+(POS-NEG)*(cf+1)/2
        else: a=POS if cf>0 else NEG
        return a,abs(cf)
    print(f"=== UNIFIED-ANSWERABLE geometric (items+concepts; GRADED={bool(GRAD)} CTAU={CTAU}) {'TAIL' if _RT else 'FULL'} vs CASPER-R 0.360/0.152, conc-oracle 0.369/0.389 ===",flush=True)
    res={m:[0.,0.,0.,0.,0.,0,0] for m in modes}                                 # nf,nt,cntf,cntt,nitems,nusers,nabstain
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        trel=set(j for j in test if rd[j]>=4 and (not headmask[j] if _RT else True))
        if not tlike or not trel: continue
        ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset])
        for m in modes:
            toks=[]; chosen=set(); nit=0; nab=0
            for t in range(8):
                cand=[k for k in range(NP) if k not in chosen and (not os.environ.get('CONLY') or k>=NI) and (not os.environ.get('IONLY') or k<NI)]
                if not cand: break
                if m=='uent': order=sorted(cand,key=lambda k:-POOL_ENT[k])
                elif m=='upop': order=sorted(cand,key=lambda k:-PRIOR[k])
                elif m=='uoracle':                                              # privileged headroom: entity whose derived answer most raises tail-NDCG
                    best=-1.; order=None
                    for k in cand:
                        a,cfc=ansconf(ustar,k)
                        if cfc<CTAU: continue
                        u2=enc_u_np(toks+[(POOL[k],a)]); s=popb+Ql@u2; s[list(profset)]=-1e9
                        if _RT: s[headmask]=-1e9
                        o=np.argsort(-s)[:10]; nd=sum(_Wc[p] for p,it in enumerate(o) if int(it) in trel)
                        if nd>best: best=nd; order=[k]
                    if order is None: break
                picked=None
                for k in order:                                                 # respond to first entity passing the confidence gate; abstain skips weak-alignment entities
                    a,cfc=ansconf(ustar,k)
                    if cfc<CTAU and m!='uoracle': nab+=1; continue
                    picked=(k,a); break
                if picked is None: break
                k,a=picked; chosen.add(k); toks.append((POOL[k],a))
                if k<NI: nit+=1
            u=enc_u_np(toks); mfu=metr(u,tlike,profset,False); mta=metr(u,tlike,profset,True)
            if mfu: res[m][0]+=mfu[0]; res[m][2]+=1
            if mta: res[m][1]+=mta[0]; res[m][3]+=1
            res[m][4]+=nit; res[m][5]+=1; res[m][6]+=nab
    for m in modes:
        a=res[m]; print(f"  {m:8s}: FULL {a[0]/max(a[2],1):.4f}  TAIL {a[1]/max(a[3],1):.4f}  | items {a[4]/max(a[5],1):.2f}/8 abstain {a[6]/max(a[5],1):.2f}",flush=True)
    sys.exit(0)
if os.environ.get('ACTORRL'):                                                  # BEAT CASPER-R: REINFORCE-finetune the continuous-snap actor on TRUE held-out NDCG (cans answers, frozen phi) from the distilled init
    import sys
    _RT=bool(os.environ.get('REWTAIL'))
    print(f"=== ACTOR-RL REINFORCE finetune (REWTAIL={_RT}) | snap+cans, frozen phi, from distilled init | target > CASPER-R 0.360/0.152 ===",flush=True)
    aopt=torch.optim.Adam(actor.parameters(),float(os.environ.get('RLLR',1e-4))); _disct=torch.tensor((1./np.log2(np.arange(2,12))).astype(np.float32))
    for ep in range(int(os.environ.get('RLEP',15))):
        rng.shuffle(trbig); tot=0.;nb=0
        for b0 in range(0,min(len(trbig),6000),96):
            us=trbig[b0:b0+96]; ANS,USTAR,tgt,wt,LIKED,LMASK,RNEG,PROFM=prep(us)
            B=len(us); toks=torch.zeros(B,T,D+1); tmask=torch.zeros(B,T); u=torch.zeros(B,D); asked=torch.zeros(B,NP); ar=torch.arange(B); logps=[]; ents=[]
            ansm=(ANS.abs()>1e-6).float()                                       # answerable entities only (items in profile + cans concepts)
            for t in range(T):
                q=actor(u,t/8.); sc=(q@POOLnt.t()).masked_fill(asked>0,-1e9).masked_fill(ansm<0.5,-1e9); pp=torch.softmax(sc/TAU,1)
                idx=torch.multinomial(pp,1).squeeze(1); logps.append(torch.log(pp[ar,idx]+1e-9)); ents.append(-(pp*torch.log(pp+1e-9)).sum(1).mean())
                asked=asked+torch.zeros(B,NP).scatter_(1,idx.unsqueeze(1),1.0)
                with torch.no_grad():
                    toks=toks.clone(); toks[:,t,:D]=POOLt[idx]; toks[:,t,D]=ANS[ar,idx]; tmask=tmask.clone(); tmask[:,t]=1; u=enc(toks,tmask)   # fold the REAL (cans/resid) answer
            with torch.no_grad():                                              # true final NDCG@10 reward (tail or full)
                sca=(u@Qlt.t()+popbt).masked_fill(PROFM>0,-1e9)
                if _RT: sca=sca.masked_fill(HEADt.unsqueeze(0),-1e9)
                topi=sca.topk(10,1).indices; vmask=(LMASK*(~HEADt[LIKED]).float()) if _RT else LMASK
                idcg=torch.cumsum(_disct,0)[(vmask.sum(1).clamp(1,10).long()-1)]; hit=((topi.unsqueeze(2)==LIKED.unsqueeze(1))&(vmask.unsqueeze(1)>0)).any(2).float()   # ideal DCG = cumulative sum of top-n discounts
                ndcg=(hit*_disct).sum(1)/idcg.clamp(min=1e-6)
            adv=ndcg-ndcg.mean(); loss=-(torch.stack(logps,1).sum(1)*adv).mean()-float(os.environ.get('ENT',0.01))*(sum(ents)/T)
            aopt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),5.); aopt.step(); tot+=float(ndcg.mean()); nb+=1
            save_ck(f"{base}/.cache/policy_{os.environ.get('TAG','actorrl')}_last.pt")
        print(f"  RL ep{ep+1} train-{'tail' if _RT else 'full'}-NDCG {tot/nb:.4f}",flush=True); save_ck(f"{base}/.cache/policy_{os.environ.get('TAG','actorrl')}_ep{ep+1}.pt")
    save_ck(f"{base}/.cache/policy_{os.environ.get('TAG','actorrl')}.pt"); print(f"saved policy_{os.environ.get('TAG','actorrl')}.pt -- eval via MODES=contactor COSSNAP=1 LOADCK=...",flush=True)
    sys.exit(0)
def run(mode,tail):
    M={q:0. for q in QPTS};Rc={q:0. for q in QPTS};CO={q:0. for q in QPTS};MR={q:0. for q in QPTS};m=0;na=0.;ni_=0;nc_=0;novp_=0;FP=[];ALLSEQ=[];TREEROWS=[]
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); held={j:rd[j] for j in test}
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        ustar=enc_u_np([(Q[j],resid[x][j]) for j in profset]); un=np.linalg.norm(ustar)+1e-9   # known user vector
        cans=cans_np(x,profset) if mode in('policy','conc_pop','conc_oracle','cont_oracle','entropy','entropy_uni','entropy_item','random','helf','logpop_ent','pop_ent','contactor','contactor_c') or mode.startswith('mix') else {}; toks=[]; asked=set(); nq=0; nans=0; first=None; seq=[]; trow=[]
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
                        elif os.environ.get('ITEMSANS'): toks.append((Q[j],POS if float(ustar@Q[j])>0 else NEG)); nans+=1; ni_+=1   # PAPER C: popular item answerable via geometric sign
                    else:
                        cc=k-NI
                        if cc in cans: toks.append((Ec[cc],cans[cc])); nans+=1; nc_+=1; trow.append((int(cc),1 if cans[cc]>0 else -1))
                elif mode in ('contactor','contactor_c'):                          # PAPER C continuous actor eval: snap (Wolpertinger) or continuous off-pool fold
                    with torch.no_grad(): qv=actor(torch.tensor(enc_u_np(toks)[None],dtype=torch.float32),len(toks)/8.).numpy()[0]
                    if mode=='contactor_c':                                         # GOAL 2 CONTINUOUS: assume answerable; one-bit geometric answer; GROUND=k folds kNN-centroid of real entities
                        qn=qv/(np.linalg.norm(qv)+1e-9)
                        if _GROUND: fe=POOL[int(np.argmax(qn@POOLn.T))].astype(np.float32)   # hard nearest real entity (in-distribution)
                        else: fe=(qn*_CN).astype(np.float32)
                        _cf=float(ustar@fe)/(np.linalg.norm(fe)*np.linalg.norm(ustar)+1e-9)
                        toks.append((fe, (NEG+(POS-NEG)*(_cf+1)/2) if _GRADED else (POS if _cf>0 else NEG))); nq+=1; nans+=1; seq.append(-1)
                    else:                                                           # SNAP (Wolpertinger): nearest unasked pool entity by q.POOL, fold its real answer
                        sc=qv@(POOLn.T if os.environ.get('COSSNAP') else POOL.T)   # COSSNAP: cosine snap (norm-unbiased, matches cos-loss distillation)
                        _ansk=((set(range(NI)) if os.environ.get('ITEMSANS') else set(kk for kk in range(NI) if PITEMS[kk] in profset))|set(NI+c for c in cans)) if os.environ.get('ANSMASK') else None   # PAPER C FIX: restrict snap to ANSWERABLE entities (ITEMSANS -> all popular items answerable)
                        k=next(int(kk) for kk in np.argsort(-sc) if int(kk) not in asked and (_ansk is None or int(kk) in _ansk))
                        if first is None: first=int(k)
                        asked.add(int(k)); nq+=1; seq.append(int(k))
                        if PTYPE[k]==0:
                            j=PITEMS[k]
                            if j in profset: toks.append((Q[j],resid[x][j])); nans+=1; ni_+=1
                            elif os.environ.get('ITEMSANS'): toks.append((Q[j],float(ustar@POOL[k]) if os.environ.get('GRAW') else (POS if float(ustar@Q[j])>0 else NEG))); nans+=1; ni_+=1   # PAPER C: popular item answerable; GRAW=graded predicted-rating else sign
                        else:
                            cc=k-NI
                            if cc in cans: toks.append((Ec[cc],float(ustar@POOL[k]) if os.environ.get('GRAW') else cans[cc])); nans+=1; nc_+=1   # graded concept answer under GRAW (consistency)
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
