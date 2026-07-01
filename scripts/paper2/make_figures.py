"""
PAPER B figures. Renders to papers/paper2_casper/figs/ (PDF + PNG).
Data-light: concept_landscape + baseline_bars use cached arrays / hard-coded seed-avg results.
qcurve + ablation take dicts (fill from the FINAL policy run once locked).
Run:  python scripts/paper2/make_figures.py [landscape|baselines|qcurve|ablation|all]
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, csv, collections, os, sys
base='C:/dev/phd/casper/data/movielens'; FIGS='C:/dev/phd/papers/paper2_casper/figs'; os.makedirs(FIGS,exist_ok=True)
NI=600
def _save(fig,name): fig.savefig(f'{FIGS}/{name}.pdf',bbox_inches='tight'); fig.savefig(f'{FIGS}/{name}.png',dpi=130,bbox_inches='tight'); plt.close(fig); print(f'  saved {name}.pdf/.png')

# ---------- shared concept data ----------
def _concept_data():
    ctags=list(np.load(f'{base}/.cache/ctags_concept.npy')); NC=len(ctags)
    nm={}
    with open(f'{base}/genome-tags.csv',encoding='utf-8') as f:
        r=csv.reader(f); next(r)
        for row in r: nm[int(row[0])]=row[1]
    cn=lambda ci: nm.get(int(ctags[ci]),'?')
    IG=np.load(f'{base}/.cache/pool_ig2.npy')[NI:]                 # population info-gain (concept slice)
    ENT=np.load(f'{base}/.cache/pool_entavg.npy')[0][NI:]          # FAIR divisiveness (regen'd: unanswerable->0)
    catalog=set()
    with open(f'{base}/ml-1m/ratings.dat') as f:
        for line in f: catalog.add(int(line.split('::')[1]))
    tagset=set(int(t) for t in ctags); cfm=collections.defaultdict(int)
    with open(f'{base}/genome-scores.csv') as f:
        next(f)
        for line in f:
            a=line.split(',')
            if int(a[0]) in catalog and int(a[1]) in tagset and float(a[2])>0.5: cfm[int(a[1])]+=1
    COV=np.array([cfm.get(int(t),0) for t in ctags],float)
    return ctags,cn,IG,ENT,COV,NC

# ---------- FIG: concept landscape (why the opener is the opener) ----------
def concept_landscape():
    ctags,cn,IG,ENT,COV,NC=_concept_data()
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    sc=ax.scatter(ENT,IG,s=12+150*(COV/COV.max()),c=np.log1p(COV),cmap='viridis',alpha=0.45,edgecolor='none')
    def mark(ci,label,color,dx=8,dy=6):
        ax.scatter([ENT[ci]],[IG[ci]],s=160,facecolor='none',edgecolor=color,linewidth=2.0,zorder=5)
        ax.annotate(f'{label}: "{cn(ci)}"',(ENT[ci],IG[ci]),textcoords='offset points',xytext=(dx,dy),fontsize=8.5,color=color,weight='bold')
    opener=45                                                       # good soundtrack (learned opener)
    mark(opener,'learned opener','crimson',8,8)
    mark(int(ENT.argmax()),'max-divisiveness','#444',-4,-16)        # entropy heuristic's pick
    mark(int(COV.argmax()),'max-coverage','#1f6f3f',8,-14)          # conc_pop's pick
    mark(int(IG.argmax()),'max-info-gain','#9467bd',-10,10)         # raw IG pick
    ax.set_xlabel('divisiveness  (binary entropy of like-rate)'); ax.set_ylabel('population info-gain')
    cb=plt.colorbar(sc,ax=ax); cb.set_label('log coverage  (#movies tagged)')
    ax.set_title('Concept landscape: the learned opener sits at the\ninfo-gain $\\times$ divisiveness $\\times$ coverage intersection',fontsize=10)
    _save(fig,'concept_landscape')

# ---------- FIG: baseline comparison (seed-avg, te[300:]) ----------
# FAIR ruler (conc_pop matches across scripts). entropy = CANONICAL (lit_baselines, fair). dbf = current policy (provisional, vs fair bar).
BASELINES_FULL={'q0 (no ask)':0.299,'random':0.338,'pop-items':0.325,'conc-pop':0.345,'logpop*ent':0.343,'HELF':0.343,
                'Golbandi':0.350,'IGCN':0.340,'pop*ent':0.339,'tree(Golbandi-adapt)':0.342,'entropy (fair)':0.360,'CASPER (ours)':0.359}
BASELINES_TAIL={'q0 (no ask)':0.080,'random':0.126,'pop-items':0.111,'conc-pop':0.127,'logpop*ent':0.128,'HELF':0.128,
                'Golbandi':0.129,'IGCN':0.112,'pop*ent':0.134,'tree(Golbandi-adapt)':0.124,'entropy (fair)':0.142,'CASPER (ours)':0.152}
# te[300:], seed-avg n=6, q8. CASPER = entropy-distill+RL (entwin). TAIL: CASPER 0.152 > entropy 0.142 (+0.010, 6/6 seeds); FULL saturated/tied ~0.36. NICF/Wolpertinger/LLM rows pending reruns.
def baseline_bars(full=None,tail=None):
    full=full or BASELINES_FULL; tail=tail or BASELINES_TAIL
    order=sorted(full,key=lambda k:full[k])
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    for ax,d,ttl in [(axs[0],full,'FULL NDCG@10'),(axs[1],tail,'TAIL NDCG@10')]:
        vals=[d[k] for k in order]
        cols=['crimson' if k=='dbf policy' else ('#1f6f3f' if 'entropy' in k else '#888') for k in order]
        ax.barh(range(len(order)),vals,color=cols); ax.set_yticks(range(len(order))); ax.set_yticklabels(order,fontsize=8)
        ax.set_xlabel(ttl); ax.set_xlim(min(vals)-0.01,max(vals)+0.01)
        for i,v in enumerate(vals): ax.text(v+0.001,i,f'{v:.3f}',va='center',fontsize=7)
    fig.suptitle('Elicitation askers on the FAIR ruler (te[300:], seed-avg) — policy vs the strong divisiveness heuristic',fontsize=10)
    _save(fig,'baselines')

# ---------- FIG: q-curve saturation (fill from final policy run) ----------
def qcurve(qs,full,tail):                                           # full/tail: {method: [ndcg per q]}
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for ax,data,ttl in [(axs[0],full,'FULL NDCG@10'),(axs[1],tail,'TAIL NDCG@10')]:
        for m,ys in data.items():
            ax.plot(qs,ys,marker='o',ms=4,label=m,lw=2 if 'policy' in m else 1.3)
        ax.axvline(8,ls='--',c='#bbb',lw=1); ax.set_xlabel('questions q'); ax.set_ylabel(ttl); ax.legend(fontsize=8)
    fig.suptitle('Saturation: realizable askers plateau by q=8 (te[300:])',fontsize=10); _save(fig,'qcurve')

# ---------- FIG: BC -> RL ablation (fill from final 3-stage run) ----------
def ablation(stages):                                              # stages: [(name, full, tail, items, answered)]
    fig,axs=plt.subplots(1,2,figsize=(10,4.2)); names=[s[0] for s in stages]; x=range(len(stages))
    axs[0].bar([i-0.2 for i in x],[s[1] for s in stages],0.4,label='full',color='#888')
    axs[0].bar([i+0.2 for i in x],[s[2] for s in stages],0.4,label='tail',color='crimson')
    axs[0].set_xticks(list(x)); axs[0].set_xticklabels(names,fontsize=8); axs[0].set_ylabel('NDCG@10'); axs[0].legend(fontsize=8)
    axs[1].bar([i-0.2 for i in x],[s[3] for s in stages],0.4,label='items asked',color='#1f6f3f')
    axs[1].bar([i+0.2 for i in x],[s[4] for s in stages],0.4,label='answered/8',color='#9467bd')
    axs[1].set_xticks(list(x)); axs[1].set_xticklabels(names,fontsize=8); axs[1].legend(fontsize=8)
    fig.suptitle('BC floor imitates privileged items -> RL drops them for answerable concepts',fontsize=10); _save(fig,'ablation')

if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    if which in('landscape','all'): concept_landscape()
    if which in('baselines','all'): baseline_bars()
    print('done.')
