"""PATH1 (PREREG_PATH1_GATE.md) analysis: per-trseed q-curves, tail-AUC(turns 1-8) deltas vs entropy,
and the pooled paired per-user bootstrap (PAIRBOOT-style, 10k resamples, two-sided p).
Input: ANSDUMP csvs written by continuous_policy2.py EVALCKS runs (rows: seed,mode,tail,user,q,answered,ndcg).
Usage: python path1_auc_boot.py <entropy_dump.csv> <ts0_dump.csv> <ts1_dump.csv> <ts2_dump.csv>
AUC = mean of per-user NDCG@10 over q=1..8 (QPTS=1..8). Primary=TAIL AUC; secondary=tail@q4; q8 reported only.
"""
import sys, numpy as np
from collections import defaultdict
QS=list(range(1,9))
def load(path,want=None):
    # {(evalseed,user)} -> {q: ndcg}, split by tail flag; want = required mode name (dump files can mix modes)
    D={0:defaultdict(dict),1:defaultdict(dict)}
    for line in open(path):
        a=line.strip().split(',')
        if len(a)<7: continue
        sd,mode,tail,x,q,na,nd=int(a[0]),a[1],int(a[2]),int(a[3]),int(a[4]),int(float(a[5])),float(a[6])
        if want and mode!=want: continue
        D[tail][(sd,x)][q]=nd
    return D
def curves(D,tail):
    # returns {(seed,user): auc}, {(seed,user):{q:nd}} for users with all 8 points
    A={}; C={}
    for k,qs in D[tail].items():
        if all(q in qs for q in QS): A[k]=np.mean([qs[q] for q in QS]); C[k]=qs
    return A,C
ent=load(sys.argv[1],'entropy'); pols=[load(p,'policy') for p in sys.argv[2:]]
np.random.seed(0)
print(f"== PATH1 tail-AUC analysis | AUC = mean NDCG@10 over q=1..8 | comparator = entropy (same seeds/splits) ==")
for tail in (1,0):
    lab='TAIL' if tail else 'FULL'
    eA,eC=curves(ent,tail)
    # entropy q-curve (seed-avg)
    eq={q:np.mean([c[q] for c in eC.values()]) for q in QS}
    print(f"\n[{lab}] entropy q-curve: "+" ".join(f"q{q}={eq[q]:.4f}" for q in QS)+f" | AUC={np.mean(list(eA.values())):.4f}")
    per_user_delta_all=defaultdict(list)   # user -> list of deltas (over eval seeds AND trseeds), for pooled bootstrap (TAIL only used)
    for ti,P in enumerate(pols):
        pA,pC=curves(P,tail)
        keys=sorted(set(eA)&set(pA))
        pq={q:np.mean([pC[k][q] for k in keys]) for q in QS}
        eqm={q:np.mean([eC[k][q] for k in keys]) for q in QS}
        dauc=np.mean([pA[k]-eA[k] for k in keys])
        print(f"[{lab}] ts{ti}: q-curve "+" ".join(f"q{q}={pq[q]:.4f}" for q in QS))
        print(f"[{lab}] ts{ti}: AUC={np.mean([pA[k] for k in keys]):.4f} dAUC={dauc:+.4f} | dq2={pq[2]-eqm[2]:+.4f} dq4={pq[4]-eqm[4]:+.4f} dq8={pq[8]-eqm[8]:+.4f} (n={len(keys)} user-seed pairs)")
        for (sd,x) in keys: per_user_delta_all[x].append(pA[(sd,x)]-eA[(sd,x)])
    # pooled paired bootstrap over USERS: each user's delta = mean over eval seeds x trseeds
    ud=np.array([np.mean(v) for v in per_user_delta_all.values()]); n=len(ud)
    if n:
        bs=np.array([np.mean(ud[np.random.randint(0,n,n)]) for _ in range(10000)])
        p=2*min((bs<=0).mean(),(bs>=0).mean())
        print(f"[{lab}] POOLED paired bootstrap (per-user, {n} users, 10k): mean dAUC={ud.mean():+.4f} 95%CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] p={p:.4f}")
