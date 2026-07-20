"""Paper C graphical abstract: the realization ladder.
x = richness of the realizable action set the continuous query must land in
    (named concepts 761 -> item pairs ~6.9M -> free continuous direction).
y = NDCG@10 (full-catalogue, seed-avg {1,2,3,7,11}, te[300:], q8, graded).
Two series:
  post-hoc snap        = take the free continuous D1 query and snap it into the set AFTER training
  trained-through      = a policy trained to act natively within that interface
Sources: SNAPLOSS_RESULT.md (concept-snap 0.341), PAIRSNAP_RESULT.md (pair-snap 0.337),
         PAIRTRAIN_RESULT.md (pair-native 0.368), canonical CASPER-R 0.360, D1 free 0.378.
Output: fig_ladder.pdf"""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.size':10,'font.family':'serif','axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':150,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#0072B2'; VERM='#D55E00'; GREY='#888888'
x=np.array([0,1,2])
xlab=['named concepts\n(761 discrete)','item pairs\n($\\approx$6.9M)','free direction\n($\\infty$, off-manifold)']
snap    =[0.341,0.337,0.378]     # post-hoc snap; at 'free' there is nothing to snap -> = D1 free
trained =[0.360,0.368,0.378]     # trained-through interface; CASPER-R / pair-native / D1

fig,ax=plt.subplots(figsize=(6.4,4.0))
ax.plot(x,snap,   '-o',color=VERM,lw=2,ms=7,label='post-hoc snap  (name the query after training)')
ax.plot(x,trained,'-o',color=BLUE,lw=2,ms=7,label='trained through the interface')
# discrete SOTA reference band
ax.axhline(0.360,color=GREY,ls=':',lw=1.1)
ax.text(2.02,0.360,'discrete SOTA\n(CASPER-R)',va='center',ha='left',fontsize=7.6,color=GREY)
# annotate the free headline
ax.annotate('the best question\nhas no name',xy=(2,0.378),xytext=(1.15,0.372),
            fontsize=8.6,color='#222222',ha='center',
            arrowprops=dict(arrowstyle='->',color='#222222',lw=1.0))
# snap-loss gap at concepts
ax.annotate('',xy=(0,0.341),xytext=(0,0.378),arrowprops=dict(arrowstyle='<->',color=GREY,lw=1.0))
ax.text(0.04,0.360,'snap-loss\n$-0.037$',fontsize=7.6,color=GREY,va='center')
for xi,yi in zip(x,snap):    ax.annotate(f'{yi:.3f}',(xi,yi),textcoords='offset points',xytext=(0,-13),ha='center',fontsize=7.4,color=VERM)
for xi,yi in zip(x,trained): ax.annotate(f'{yi:.3f}',(xi,yi),textcoords='offset points',xytext=(0,8),ha='center',fontsize=7.4,color=BLUE)
ax.set_xticks(x); ax.set_xticklabels(xlab,fontsize=8.4)
ax.set_xlim(-0.35,2.5); ax.set_ylim(0.325,0.386)
ax.set_ylabel('NDCG@10  (full catalogue)')
ax.set_xlabel('richness of the realizable action set  $\\longrightarrow$')
ax.legend(loc='lower right',frameon=False,fontsize=8.4)
ax.set_title('Realizing a continuous preference query: richer set, or don’t snap',fontsize=10.5)
fig.tight_layout()
fig.savefig('C:/dev/phd/papers/paper3_casper/figs/fig_ladder.pdf',bbox_inches='tight')
print('saved fig_ladder.pdf')
