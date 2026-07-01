"""Precompute LLM-asker answer-data over a user range x seed-set, on the EXACT harness ruler (rng(0) keep/te;
SPL rng(seed)). Default: TEST range te[300:] x 5 seeds {1,2,3,7,11} for the seed-averaged STATIC LLM rows that drop
into the main table. (Val/seed1 single-seed used earlier for the adaptive probe.)  -> llm_data_seedavg.pkl"""
import os, numpy as np, torch, torch.nn as nn, csv, pickle
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64
rng=np.random.default_rng(0)
U,I,Rr=[],[],[]
for line in open(f'{ml}/ratings.dat'):
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
popb=np.log(cnt+1.0).astype(np.float32); bi=np.load(f'{base}/.cache/bi_svd.npy')
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
POSl=[];NEGl=[]
for x in trU[:3000]:
    for j,r in rat_by_u[x]: (POSl if r>=4 else NEGl).append(r-mu-bi[j])
POS=float(np.mean(POSl)); NEG=float(np.mean(NEGl))
Q=np.load(f'{base}/.cache/Q_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_concept.npy'); Ec=np.load(f'{base}/.cache/Ec_concept.npy'); ctags=list(np.load(f'{base}/.cache/ctags_concept.npy')); NC=len(ctags)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
tagset=set(int(t) for t in ctags); tagitems={}
f=open(f'{base}/genome-scores.csv'); next(f)
for line in f:
    a=line.split(','); m=int(a[0]); tg=int(a[1])
    if m in iids and tg in tagset and float(a[2])>0.5: tagitems.setdefault(tg,[]).append(iids[m])
citems=[set(tagitems.get(int(t),[])) for t in ctags]
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
def enc_u(toks):
    if not toks: return np.zeros(D,np.float32)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(ff,v) in enumerate(toks): arr[0,q,:D]=ff; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
nm={}
for row in csv.reader(open(f'{base}/genome-tags.csv',encoding='utf-8')):
    if row and row[0].isdigit(): nm[int(row[0])]=row[1]
concept_name2c={nm.get(int(ctags[cc]),'?').strip().lower():cc for cc in range(NC)}
title2iid={}
for line in open(f'{ml}/movies.dat',encoding='latin-1'):
    a=line.strip().split('::'); mid=int(a[0])
    if mid in iids: title2iid[a[1].strip()]=int(iids[mid])
TR=os.environ.get('TRANGE','test'); rng_range = te[300:] if TR=='test' else te[:300]
SEEDS=[int(s) for s in os.environ.get('SEEDS','1,2,3,7,11').split(',')]
data={}
for sd in SEEDS:
    vr=np.random.default_rng(sd); SPL={}
    for x in te:
        its=list(dict(rat_by_u[x]))
        if len(its)>=6: il=its[:]; vr.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
    recs=[]
    for x in [x for x in rng_range if x in SPL]:
        known,held=SPL[x]; rd=dict(rat_by_u[x])
        ustar=enc_u([(Q[j],resid[x][j]) for j in known])
        ac=[cc for cc in range(NC) if len(citems[cc]&known)>=2]
        pr={cc:float(ustar@Ec[cc]) for cc in ac}; thr=float(np.mean(list(pr.values()))) if pr else 0.
        cans={int(cc):(POS if pr[cc]>thr else NEG) for cc in ac}
        recs.append({'known':[int(j) for j in known],'cans':cans,'resid':{int(j):float(resid[x][j]) for j in known},'tlike':[int(j) for j in held if rd[j]>=4]})
    data[sd]=recs; print(f"  seed {sd}: {len(recs)} users")
pickle.dump({'data':data,'POS':POS,'NEG':NEG,'Ec':Ec,'Q':Q,'Ql':Ql,'popb':popb,'headmask':headmask,
             'concept_name2c':concept_name2c,'title2iid':title2iid,'seeds':SEEDS}, open(f'{base}/.cache/llm_data_seedavg.pkl','wb'))
print(f"saved seedavg data: range={TR} seeds={SEEDS} -> llm_data_seedavg.pkl")
