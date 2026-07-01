"""Q-curve figure: ADAPTIVE policy dominates the STATIC entropy heuristic (seed-averaged, te[300:]).
Reads evalcks_all.csv (seed,tag,q,full,tail) -> 2 panels (FULL, TAIL) NDCG@10 vs question budget.
"""
import csv, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fn='C:/dev/phd/casper/data/movielens/.cache/evalcks_all.csv'
G=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))  # tag->q->{full,tail}->[seeds]
for r in csv.reader(open(fn)):
    if len(r)<5: continue
    sd,tag,q,full,tail=r; G[tag][int(q)]['full'].append(float(full)); G[tag][int(q)]['tail'].append(float(tail))
def curve(tag,metric):
    qs=sorted(G[tag]); return qs,[np.mean(G[tag][q][metric]) for q in qs],[np.std(G[tag][q][metric]) for q in qs]
SHOW=[('entropy','static entropy (heuristic)','#777','o','--'),
      ('conc_pop','popular concepts','#bbb','s',':'),
      ('entdistill_ep4','learned adaptive (ours)','#1f77b4','D','-'),
      ('dbf_uni_ep7','adaptive, fast-start','#2ca02c','^','-.'),
      ('poracle_reg_best','adaptive, answerability','#d62728','v','-.')]
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for ax,metric,ttl in [(axes[0],'full','FULL-catalogue NDCG@10'),(axes[1],'tail','Long-tail NDCG@10')]:
    for tag,lab,c,mk,ls in SHOW:
        if tag not in G: continue
        qs,m,s=curve(tag,metric); ax.plot(qs,m,ls,color=c,marker=mk,ms=5,lw=2 if 'ours' in lab else 1.4,label=lab,zorder=3 if 'ours' in lab else 2)
        if 'ours' in lab: ax.fill_between(qs,np.array(m)-np.array(s),np.array(m)+np.array(s),color=c,alpha=0.15,zorder=1)
    ax.set_xlabel('questions asked ($q$)'); ax.set_ylabel(ttl); ax.set_title(ttl.split(' NDCG')[0]); ax.grid(alpha=0.3); ax.set_xlim(0,8)
axes[0].legend(fontsize=8,loc='lower right')
fig.suptitle('Adaptive elicitation dominates the static entropy heuristic — most at low budgets (efficiency) and on the tail (seed-avg, n=6, te[300:])',fontsize=9)
fig.tight_layout(); out='C:/dev/phd/papers/paper2_casper/fig_qcurve_adaptive.pdf'
import os; os.makedirs(os.path.dirname(out),exist_ok=True); fig.savefig(out,bbox_inches='tight'); fig.savefig(out.replace('.pdf','.png'),dpi=130,bbox_inches='tight')
print('saved',out)
# also print the headline deltas
for q in [1,2,3,8]:
    e=np.mean(G['entropy'][q]['tail']); o=np.mean(G['entdistill_ep4'][q]['tail']); ef=np.mean(G['entropy'][q]['full']); of=np.mean(G['entdistill_ep4'][q]['full'])
    print(f"q{q}: TAIL ours {o:.3f} vs entropy {e:.3f} ({o-e:+.3f}) | FULL ours {of:.3f} vs {ef:.3f} ({of-ef:+.3f})")
