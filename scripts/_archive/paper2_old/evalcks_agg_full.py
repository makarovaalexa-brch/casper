"""Aggregate the HARMONIZED EVALCKS grid CSV
(seed,tag,q,ndcg_full,ndcg_tail,rec_full,rec_tail,rmse,cos,ans) -> mean +/- std per (tag,q),
for EVERY metric, with delta vs entropy (signed so + = better, RMSE flipped).
Usage: python evalcks_agg_full.py [csv]
"""
import sys, csv, numpy as np, collections
fn=sys.argv[1] if len(sys.argv)>1 else 'C:/dev/phd/casper/data/movielens/.cache/evalcks_harmonized.csv'
COLS=['ndcg_full','ndcg_tail','rec_full','rec_tail','mrr_full','mrr_tail','cos','ans']
G=collections.defaultdict(lambda:collections.defaultdict(list)); seeds=set()
for r in csv.reader(open(fn)):
    if len(r)<11: continue
    sd,tag,q=r[0],r[1],int(r[2]); seeds.add(sd)
    for i,c in enumerate(COLS): G[(tag,q)][c].append(float(r[3+i]))
qs=sorted(set(q for (_,q) in G))
order=['fullprof','entropy','entropy_item','entropy_uni','conc_pop','pop_item','mix4','random','helf','logpop_ent','pop_ent']  # ceiling + baselines first, then policies/ablations
tags=sorted(set(t for (t,_) in G),key=lambda t:(order.index(t) if t in order else 99, t))
print(f"seeds={sorted(seeds)} (n={len(seeds)}) | te[300:] harmonized | + = better than entropy (RMSE sign-flipped)\n")
ENT={(q,c):np.mean(G[('entropy',q)][c]) for q in qs for c in COLS if ('entropy',q) in G}
for c in COLS:
    lower_better=False
    print(f"=== {c.upper()}{'  (lower better)' if lower_better else ''}  (mean+/-std over seeds; delta vs entropy) ===")
    print("%-16s "%'tag'+" ".join("q%-18d"%q for q in qs))
    for tag in tags:
        cells=[]
        for q in qs:
            v=G.get((tag,q),{}).get(c)
            if not v: cells.append("        -         "); continue
            m=float(np.mean(v)); s=float(np.std(v))
            d=(m-ENT[(q,c)]) if (q,c) in ENT else 0.0
            if lower_better: d=-d
            mark='' if tag=='entropy' else f" ({d:+.3f})"
            cells.append(f"{m:.3f}+/-{s:.3f}{mark}")
        print("%-16s "%tag+" ".join(f"{x:<19}" for x in cells))
    print()
# headline summary at max q
q8=max(qs)
print(f"--- HEADLINE @ q{q8} (vs entropy) ---")
for tag in tags:
    if tag=='entropy': continue
    row=[]
    for c in COLS:
        v=G.get((tag,q8),{}).get(c)
        if not v: continue
        m=float(np.mean(v)); d=m-ENT.get((q8,c),m)
        row.append(f"{c}={m:.3f}({d:+.3f})")
    print(f"  {tag:<16} "+" ".join(row))
