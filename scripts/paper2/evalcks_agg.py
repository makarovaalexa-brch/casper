"""Aggregate the EVALCKS seed grid CSV (seed,tag,q,full,tail) -> mean +/- std per (tag,q).
Usage: python evalcks_agg.py [csv]   (default .cache/evalcks_grid.csv); compares every tag vs entropy.
"""
import sys, csv, numpy as np, collections
fn=sys.argv[1] if len(sys.argv)>1 else 'C:/dev/phd/casper/data/movielens/.cache/evalcks_grid.csv'
rows=list(csv.reader(open(fn)))
G=collections.defaultdict(lambda:collections.defaultdict(list))     # (tag,q) -> {'full':[...],'tail':[...]} across seeds
seeds=set()
for r in rows:
    if len(r)<5: continue
    sd,tag,q,full,tail=r; seeds.add(sd); G[(tag,int(q))]['full'].append(float(full)); G[(tag,int(q))]['tail'].append(float(tail))
qs=sorted(set(q for (_,q) in G)); tags=sorted(set(t for (t,_) in G),key=lambda t:(t!='entropy',t!='conc_pop',t))
print(f"seeds={sorted(seeds)} (n={len(seeds)})\n")
ENT={q:(np.mean(G[('entropy',q)]['full']),np.mean(G[('entropy',q)]['tail'])) for q in qs if ('entropy',q) in G}
for metric in ['full','tail']:
    print(f"=== {metric.upper()} NDCG@10  (mean +/- std over seeds; vs entropy delta) ===")
    print("%-26s "%'tag'+" ".join(f"q{q:<13}"%() for q in qs))
    for tag in tags:
        cells=[]
        for q in qs:
            v=G.get((tag,q),{}).get(metric)
            if not v: cells.append("       -      "); continue
            m=np.mean(v); s=np.std(v); d=m-(ENT[q][0] if metric=='full' else ENT[q][1]) if q in ENT else 0
            mark='' if tag in('entropy','conc_pop') else (f" ({d:+.3f})")
            cells.append(f"{m:.3f}±{s:.3f}{mark}")
        print("%-26s "%tag+" ".join(f"{c:<14}" for c in cells))
    print()
# headline: best tag at q8 on tail vs entropy
q8=max(qs)
best=sorted([(t,np.mean(G[(t,q8)]['tail'])) for t in tags if t not in('entropy','conc_pop') and (t,q8) in G],key=lambda x:-x[1])[:5]
et=ENT.get(q8,(0,0))[1]
print(f"--- top-5 by TAIL@q{q8} (entropy={et:.3f}) ---")
for t,m in best: print(f"  {t:<26} {m:.3f}  ({m-et:+.3f} vs entropy)")
