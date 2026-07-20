import sys,numpy as np,collections
csv=sys.argv[1]
rows=collections.defaultdict(lambda:collections.defaultdict(list))  # (mode,q) -> {full:[],tail:[]}
for line in open(csv):
    a=line.strip().split(',')
    if len(a)<5: continue
    sd,mode,q,full,tail=a[0],a[1],int(a[2]),float(a[3]),float(a[4])
    rows[(mode,q)]['f'].append(full); rows[(mode,q)]['t'].append(tail)
modes=[]
for (mode,q) in rows:
    if mode not in modes: modes.append(mode)
qs=sorted(set(q for (_,q) in rows))
print(f"seed-avg over {len(rows[(modes[0],qs[0])]['f'])} seeds\n")
for mode in modes:
    print(f"{mode}:")
    for q in qs:
        f=rows[(mode,q)]['f']; t=rows[(mode,q)]['t']
        print(f"  q={q}: FULL {np.mean(f):.4f}+/-{np.std(f):.4f}   TAIL {np.mean(t):.4f}+/-{np.std(t):.4f}")
    print()
# deltas vs entropy at q8
if ('entropy',8) in rows:
    eF=np.array(rows[('entropy',8)]['f']); eT=np.array(rows[('entropy',8)]['t'])
    print("=== @q8 delta vs entropy (paired over seeds) ===")
    for mode in modes:
        if mode=='entropy': continue
        mF=np.array(rows[(mode,8)]['f']); mT=np.array(rows[(mode,8)]['t'])
        dF=mF-eF; dT=mT-eT
        print(f"  {mode:>14}: dFULL {dF.mean():+.4f} (sd {dF.std():.4f}, {dF.mean()/(dF.std()/len(dF)**0.5+1e-9):+.1f}sigma)   dTAIL {dT.mean():+.4f} (sd {dT.std():.4f}, {dT.mean()/(dT.std()/len(dT)**0.5+1e-9):+.1f}sigma)")
