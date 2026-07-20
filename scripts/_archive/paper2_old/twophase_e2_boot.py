"""E2 paired per-user bootstrap @q8 (FULL primary, TAIL honest) + phase-2 answered-count.
Rows in dump: seed,mode,tail,user,q,answered,ndcg."""
import sys, numpy as np
from collections import defaultdict
dump=sys.argv[1]
# nd[(mode,tail)][(seed,user)][q]=ndcg ; na[(mode)][(seed,user)][q]=answered
nd=defaultdict(lambda:defaultdict(dict)); na=defaultdict(lambda:defaultdict(dict))
for line in open(dump):
    a=line.strip().split(',')
    if len(a)<7: continue
    sd,mode,tail,x,q,ans,v=int(a[0]),a[1],int(a[2]),int(a[3]),int(a[4]),int(float(a[5])),float(a[6])
    nd[(mode,tail)][(sd,x)][q]=v
    if tail==0: na[mode][(sd,x)][q]=ans
def boot(mode,tail):
    E=nd[('entropy',tail)]; M=nd[(mode,tail)]
    keys=[k for k in E if 8 in E[k] and k in M and 8 in M[k]]
    # pool per-user delta = mean over eval seeds
    per_user=defaultdict(list)
    for (sd,x) in keys: per_user[x].append(M[(sd,x)][8]-E[(sd,x)][8])
    d=np.array([np.mean(v) for v in per_user.values()]); n=len(d)
    np.random.seed(0); bs=np.array([d[np.random.randint(0,n,n)].mean() for _ in range(10000)])
    p=2*min((bs<=0).mean(),(bs>=0).mean())
    em=np.mean([E[k][8] for k in keys]); mm=np.mean([M[k][8] for k in keys])
    return em,mm,d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),p,n
print("=== E2 @q8 paired per-user bootstrap (10k resamples, pooled over seeds; comparator=entropy) ===")
for tail in (0,1):
    lab='FULL (PRIMARY)' if tail==0 else 'TAIL (honest)'
    print(f"\n[{lab}]  entropy@q8 = {np.mean([nd[('entropy',tail)][k][8] for k in nd[('entropy',tail)] if 8 in nd[('entropy',tail)][k]]):.4f}")
    for mode in ('twophase','twophase_exp','twophase_poponly'):
        em,mm,dm,lo,hi,p,n=boot(mode,tail)
        flag='' if tail else (' PASS' if (dm>=0.005 and p<0.05) else ' FAIL-GATE')
        print(f"  {mode:>18}@q8={mm:.4f}  d={dm:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p={p:.4f}  (n={n} users){flag}")
# phase-2 answered-count (turns 5-8) = answered@q8 - answered@q4, averaged over seed-users
print("\n=== answered-count at turns 5-8 (na@q8 - na@q4); oracle target = 4/4 ===")
for mode in ('entropy','twophase','twophase_exp','twophase_poponly'):
    vals=[na[mode][k][8]-na[mode][k][4] for k in na[mode] if 8 in na[mode][k] and 4 in na[mode][k]]
    tot=[na[mode][k][8] for k in na[mode] if 8 in na[mode][k]]
    print(f"  {mode:>18}: phase2 answered {np.mean(vals):.2f}/4   (total ans/8 = {np.mean(tot):.2f})")
