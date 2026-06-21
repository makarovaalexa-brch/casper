"""
Answerability test with the CONCEPT-AWARE encoder (concepts trained in as a learned channel, fixes OOD bug).
ANSWER flag: 'mean' = mean-residual concept answer (step 1); 'geom' = user's geometric answer (step 2): like/dislike =
whichever fold moves the belief CLOSER to the user's true taste vector u* (=fold of known-half profile). Realistic
cold-start, fixed T-ask budget, FULL-first + tail. GATE: conc_eig > pop_item AND conc_eig beats q0 (concepts must HELP).
"""
import os, time, numpy as np, torch, torch.nn as nn
ANSWER=os.environ.get('ANSWER','mean'); LAM=float(os.environ.get('LAM',3.0)); base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=8; rng=np.random.default_rng(0)
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
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
Ql=np.load(f'{base}/.cache/Ql_concept.npy'); Ec=np.load(f'{base}/.cache/Ec_concept.npy'); ctags=list(np.load(f'{base}/.cache/ctags_concept.npy'))
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
p_seen=(cnt/max(len(trU),1)).astype(np.float32); R=order_pop[:500]
t0=time.time(); tagitems={}; tagset=set(int(t) for t in ctags)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1])
        if m in iids and tg in tagset and float(a[2])>0.5: tagitems.setdefault(tg,[]).append(iids[m])
citems=[set(tagitems.get(int(t),[])) for t in ctags]; cfreq=np.array([len(s) for s in citems])
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems.get(int(t),[]): item2c[j].append(ki)
pe_cnt=np.zeros(len(ctags))                                   # answerability PRIOR: fraction of train users who could answer concept c
for x in trU[:2500]:
    cn={}
    for j,_ in rat_by_u[x]:
        for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
    for ki,ct in cn.items():
        if ct>=2: pe_cnt[ki]+=1
pe_ans=(pe_cnt/min(len(trU),2500)).astype(np.float32)+1e-6
print(f"  {len(ctags)} concepts, ANSWER={ANSWER} ({time.time()-t0:.0f}s)",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
def sig(z): return 1/(1+np.exp(-z))
def enc_u(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_batch(revs):
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def user_answers(x,profset):                                 # the user simulator: state-independent concept answers
    ac=[c for c in range(len(ctags)) if len(citems[c]&profset)>=2]
    if ANSWER=='mean': return {c:float(np.mean([resid[x][j] for j in citems[c]&profset])) for c in ac}
    ustar=enc_u([(Q[j],resid[x][j]) for j in profset])       # true taste vector from the answerable (known-half) profile
    pr={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(pr.values())) if pr else 0.
    return {c:(POS if pr[c]>thr else NEG) for c in ac}        # geom: like iff concept aligns with true taste (above this user's mean)
def cfold_val(c,cans): return cans.get(c)                    # None if unanswerable
def infogain_items(toks,cands):
    u=enc_u(toks); p=sig(popb[cands]+Ql[cands]@u); ul=enc_batch([toks+[(Q[j],POS)] for j in cands]); ud=enc_batch([toks+[(Q[j],NEG)] for j in cands])
    return p*sig(popb[R]+ul@Ql[R].T).sum(1)+(1-p)*sig(popb[R]+ud@Ql[R].T).sum(1)
def infogain_concepts(toks,cidx):                            # PROPER answerability-aware EIG (per-candidate belief, no popb leak)
    u=enc_u(toks); ca=np.array(cidx); pc=sig(Ec[ca]@u)                                   # per-concept belief the user likes c
    ul=enc_batch([toks+[(Ec[c],POS)] for c in cidx]); ud=enc_batch([toks+[(Ec[c],NEG)] for c in cidx])
    cov=pc*sig(ul@Ql[R].T).sum(1)+(1-pc)*sig(ud@Ql[R].T).sum(1)                          # belief-weighted expected coverage over R
    return pe_ans[ca]*cov                                                                 # x P(answerable): don't waste turns on un-answerable concepts
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-sc); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel); return nd,rc
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
TE=[x for x in te if x in SPL][:150]; ITEMC=list(order_pop[:1500]); orc_pick={'i':0,'c':0}
def run(mode,tail):
    M={q:0. for q in [0,2,4,8]};Rc={q:0. for q in [0,2,4,8]};m=0;ans_tot=0.
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        cans=user_answers(x,profset)                          # the user simulator: fixed per-user concept answers
        toks=[]; asked=set(); nans=0; nq=0
        for q in [0,2,4,8]:
            while nq<q:
                if mode=='rand_item':
                    j=int(rng.integers(ni))
                    if j in asked: continue
                    asked.add(j); nq+=1
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode=='pop_item':
                    cs=[j for j in ITEMC if j not in asked]
                    if not cs: break
                    j=cs[0]; asked.add(j); nq+=1
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode=='eig_item':
                    cs=[j for j in ITEMC if j not in asked][:500]
                    if not cs: break
                    csa=np.array(cs); j=cs[int((p_seen[csa]*infogain_items(toks,csa)).argmax())]; asked.add(j); nq+=1
                    if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                elif mode in ('conc_pop','conc_eig'):
                    ci=[c for c in range(len(ctags)) if c not in asked]
                    if not ci: break
                    if mode=='conc_pop': c=max(ci,key=lambda c:cfreq[c])
                    else: c=ci[int(infogain_concepts(toks,ci).argmax())]
                    asked.add(c); nq+=1; v=cans.get(c)
                    if v is not None: toks.append((Ec[c],v)); nans+=1
                elif mode=='conc_mix':                                      # COMBINATION at fixed budget: interleave popular ITEM + frequent CONCEPT
                    if nq%2==0:
                        cs=[j for j in order_pop[:1500] if ('i',int(j)) not in asked]
                        if not cs: break
                        j=int(cs[0]); asked.add(('i',j)); nq+=1
                        if j in profset: toks.append((Q[j],resid[x][j])); nans+=1
                    else:
                        ci=[c for c in range(len(ctags)) if ('c',c) not in asked and cans.get(c) is not None]
                        if not ci: break
                        c=max(ci,key=lambda c:cfreq[c]); asked.add(('c',c)); nq+=1; toks.append((Ec[c],cans[c])); nans+=1
                elif mode=='conc_popbelief':                                # REALIZABLE COLD (user's fusion, non-learned): popularity prior + lambda*belief-alignment; no training
                    u0=enc_u(toks); ci=[c for c in range(len(ctags)) if c not in asked and cans.get(c) is not None]
                    if not ci: break
                    ca=np.array(ci); score=np.log(pe_ans[ca]+1e-6)+LAM*(Ec[ca]@u0)
                    c=ci[int(score.argmax())]; asked.add(c); nq+=1; toks.append((Ec[c],cans[c])); nans+=1
                elif mode=='conc_gprof':                                     # profile-privileged: greedy concept maximizing coverage of KNOWN profile (deployable warm; upper bound cold)
                    ci=[c for c in range(len(ctags)) if c not in asked and cans.get(c) is not None]
                    if not ci: break
                    ci=sorted(ci,key=lambda c:-cfreq[c])[:150]; profa=np.array(list(profset))
                    ul=enc_batch([toks+[(Ec[c],cans[c])] for c in ci]); cov=sig(popb[profa]+ul@Ql[profa].T).sum(1)
                    c=ci[int(cov.argmax())]; asked.add(c); nq+=1; toks.append((Ec[c],cans[c])); nans+=1
                elif mode=='conc_gbelief':                                   # REALIZABLE COLD: greedy concept covering the BELIEF's current top predictions; rank by EXPECTED answer, fold TRUE answer (no profile/test peek)
                    u0=enc_u(toks); ref=np.array(list(order_pop[:50])) if not toks else np.argsort(-(popb+Ql@u0))[:50]
                    ci=[c for c in range(len(ctags)) if c not in asked and cans.get(c) is not None]
                    if not ci: break
                    ci=sorted(ci,key=lambda c:-cfreq[c])[:150]
                    ul=enc_batch([toks+[(Ec[c], POS if float(u0@Ec[c])>0 else NEG)] for c in ci]); cov=sig(popb[ref]+ul@Ql[ref].T).sum(1)
                    c=ci[int(cov.argmax())]; asked.add(c); nq+=1; toks.append((Ec[c],cans[c])); nans+=1
                elif mode in ('oracle','conc_oracle'):
                    bestv=-1;bestt=None
                    if mode=='oracle':
                      for j in [jj for jj in ITEMC if jj in profset and ('i',jj) not in asked][:150]:
                        mt=metr(enc_u(toks+[(Q[j],resid[x][j])]),tlike,profset,tail)
                        if mt and mt[0]>bestv: bestv=mt[0];bestt=('i',j)
                    for c in [cc for cc in range(len(ctags)) if len(citems[cc]&profset)>=2 and ('c',cc) not in asked][:150]:
                        v=cans.get(c); mt=metr(enc_u(toks+[(Ec[c],v)]),tlike,profset,tail)
                        if mt and mt[0]>bestv: bestv=mt[0];bestt=('c',c)
                    if bestt is None: break
                    asked.add(bestt); nq+=1; orc_pick[bestt[0]]+=1
                    if bestt[0]=='i': toks.append((Q[bestt[1]],resid[x][bestt[1]]))
                    else: toks.append((Ec[bestt[1]],cans.get(bestt[1])))
                    nans+=1
            mt=metr(enc_u(toks),tlike,profset,tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1; ans_tot+=nans
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},m,ans_tot/m
for tail in [False,True]:
    print(f"\n=== {'FULL (MAIN)' if not tail else 'TAIL'} | concept-aware enc, ANSWER={ANSWER} | NDCG@10 / Rec@50 / ans ===",flush=True)
    for mode in ['pop_item','conc_pop','conc_mix']:
        M,Rc,m,na=run(mode,tail); print(f"  {mode:<10}: NDCG "+" ".join(f"{M[q]:.3f}" for q in [0,2,4,8])+" | Rec "+" ".join(f"{Rc[q]:.3f}" for q in [0,2,4,8])+f" | ans/{T}={na:.1f}",flush=True)
    print(f"  GATE: conc_eig must beat q0 ({'concepts HELP' if True else ''}) AND pop_item",flush=True)
print(f"\nORACLE PICK: items={orc_pick['i']} concepts={orc_pick['c']} -> {100*orc_pick['c']/max(orc_pick['i']+orc_pick['c'],1):.0f}% concepts",flush=True)
# ITEMS-PRESERVED regression: warm full known-half profile (items only), concept-enc vs canonical unified-enc => items must NOT degrade
enc2=Enc(); enc2.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc2.eval(); Ql2=np.load(f'{base}/.cache/Ql_unified.npy')
def fold_with(encm,pairs):
    if not pairs: return np.zeros(D)
    arr=np.zeros((1,len(pairs),D+1),np.float32); m=np.ones((1,len(pairs)),np.float32)
    for q,(f,v) in enumerate(pairs): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return encm(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def warm(encm,Qlm,tail):
    M=0.;Rc=0.;m=0
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        u=fold_with(encm,[(Q[j],resid[x][j]) for j in profset]); sc=(popb+Qlm@u).copy(); sc[list(profset)]=-1e9
        if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
        else: rel=set(tlike)
        if not rel: continue
        o=np.argsort(-sc); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel); M+=nd;Rc+=rc;m+=1
    return M/m,Rc/m
print("\n=== ITEMS-PRESERVED regression (warm full known-half profile, items only) ===",flush=True)
for tail in [False,True]:
    cn=warm(enc,Ql,tail); un=warm(enc2,Ql2,tail)
    print(f"  {'tail' if tail else 'full'}: concept-enc NDCG {cn[0]:.3f} Rec {cn[1]:.3f} | unified-enc(PaperA) NDCG {un[0]:.3f} Rec {un[1]:.3f}",flush=True)
# BELIEF CONVERGENCE: does folding T CONCEPT answers approach the true item-profile taste u*? (vs folding T ITEMS)
print("\n=== BELIEF CONVERGENCE: cos(belief after T answers, u*=fold(full profile items)) ===",flush=True)
POPSET=set(int(j) for j in order_pop[:400])                                  # realistically-askable popular items
for kind in ['items','concepts_freq','concepts_gprof','BOTH_pop_items+freq_concepts']:
    out=[]
    for Tn in [2,4,8,16,32]:
        cs=0.;m=0
        for x in TE:
            profset,_=SPL[x]; prof=list(profset)
            if len(prof)<8: continue
            ustar=enc_u([(Q[j],resid[x][j]) for j in prof]); profa=np.array(prof)
            if kind=='items':
                toks=[(Q[j],resid[x][j]) for j in prof[:Tn]]
            elif kind=='BOTH_pop_items+freq_concepts':                        # the COMBINATION: answerable popular items (fine) + frequent concepts (coarse)
                seen_pop=[j for j in order_pop if int(j) in POPSET and j in profset]; nit=min(len(seen_pop),max(Tn//3,1)); its=seen_pop[:nit]
                ac=[c for c in range(len(ctags)) if len(citems[c]&profset)>=2]; proj={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(proj.values())) if ac else 0.
                fc=sorted(ac,key=lambda c:-cfreq[c])[:Tn-len(its)]
                toks=[(Q[j],resid[x][j]) for j in its]+[(Ec[c],POS if proj[c]>thr else NEG) for c in fc]
            else:
                ac=[c for c in range(len(ctags)) if len(citems[c]&profset)>=2]
                if not ac: continue
                proj={c:float(ustar@Ec[c]) for c in ac}; thr=np.mean(list(proj.values())); cans={c:(POS if proj[c]>thr else NEG) for c in ac}
                if kind=='concepts_freq': sel=sorted(ac,key=lambda c:-cfreq[c])[:Tn]
                else:                                                                  # GREEDY profile-coverage order (the conc_gprof selection that WINS)
                    sel=[]; rem=set(ac); tk=[]
                    for _ in range(min(Tn,len(ac))):
                        cc=list(rem); ul=enc_batch([tk+[(Ec[c],cans[c])] for c in cc]); cov=sig(popb[profa]+ul@Ql[profa].T).sum(1)
                        b=cc[int(cov.argmax())]; sel.append(b); rem.discard(b); tk=tk+[(Ec[b],cans[b])]
                toks=[(Ec[c],cans[c]) for c in sel]
            if not toks: continue
            uT=enc_u(toks); cs+=float(uT@ustar/((np.linalg.norm(uT)+1e-9)*(np.linalg.norm(ustar)+1e-9))); m+=1
        out.append(f"T={Tn}:{cs/m:.3f}")
    print(f"  fold {kind:<16}: "+"  ".join(out),flush=True)
print("  (if concepts_gprof >> concepts_freq's 0.83 -> the ceiling is SELECTION, not fundamental coarseness => keep cracking realizable selection)",flush=True)
