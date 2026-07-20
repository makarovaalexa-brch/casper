"""Q-curve saturation figure (full + tail NDCG@10 vs q, to q20), seed-averaged. Reads evalcks_qcurve.csv.
Shows CASPER-R peaking at q8 then declining, heuristics flat, ceiling flat-high -> q8 is the sweet spot."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, csv, collections, numpy as np, os
base='C:/dev/phd/casper/data/movielens'; FIGS='C:/dev/phd/papers/paper2_casper/figs'; os.makedirs(FIGS,exist_ok=True)
G=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
for r in csv.reader(open(f'{base}/.cache/evalcks_qcurve.csv')):
    if len(r)<5: continue
    sd,tag,q=r[0],r[1],int(r[2]); G[tag][q]['full'].append(float(r[3])); G[tag][q]['tail'].append(float(r[4]))
series=[('entdistill_ep4','CASPER-R (ours)','#d62728','-','o'),
        ('entropy','divisiveness (entropy)','#1f77b4','--','s'),
        ('conc_pop','popular concepts','#2ca02c',':','^')]
qs=sorted({q for t in G for q in G[t]}); XMAX=10
fig,axs=plt.subplots(1,2,figsize=(9,3.6))
for ax,metric,ttl in [(axs[0],'full','FULL NDCG@10'),(axs[1],'tail','TAIL NDCG@10')]:
    for tag,lab,col,ls,mk in series:
        if tag not in G: continue
        y=[np.mean(G[tag][q][metric]) for q in qs]
        ax.plot(qs,y,color=col,ls=ls,marker=mk,ms=4.5,label=lab,lw=1.9)
    cval=float(np.mean([np.mean(G['fullprof'][q][metric]) for q in qs if q>=2]))   # ceiling = constant horizontal reference
    ax.axhline(cval,color='#7f7f7f',ls='-.',lw=1.6,label='full-profile ceiling')
    ax.axvline(8,color='k',lw=0.8,alpha=0.4,ls=':')
    ax.set_xlim(-0.3,XMAX+0.3); ax.set_xlabel('number of questions $q$'); ax.set_title(ttl); ax.grid(alpha=0.25); ax.set_xticks([0,2,4,6,8,10])
axs[1].legend(fontsize=7.5,loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIGS}/qcurve.pdf',bbox_inches='tight'); fig.savefig(f'{FIGS}/qcurve.png',dpi=130,bbox_inches='tight')
print('saved qcurve.pdf/.png to',FIGS)
for tag,lab,_,_,_ in series:
    if tag in G: print(f"  {lab:24} tail:", [round(np.mean(G[tag][q]['tail']),3) for q in qs])
