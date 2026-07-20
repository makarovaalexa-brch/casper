"""Paper A 'same data, two verdicts' figure.
Left panel: absolute NDCG@10 under the honest full-catalogue protocol vs the field-standard
100-sampled-negative protocol (ML-1M, seed-avg {1,2,3,7,11}, te[300:], q8, graded) -- the sampled
protocol inflates every method ~1.2x.
Right panel: bump chart of the elicitation-baseline ordering that the protocol FLIPS -- under the
honest ruler entropy (divisiveness) beats the popularity heuristic on all 5 seeds; under sampled
negatives popularity overtakes entropy on all 5 seeds.
Source: casper/experiments/paper2/PROTOCOL_INFLATION_RESULT.md
Output: fig_protocol_inflation.pdf"""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.size':10,'font.family':'serif','axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':150,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#0072B2'; VERM='#D55E00'; GREEN='#009E73'; GREY='#888888'

methods=['MOSTPOP\n(q0)','popular\nconcept','entropy\n(divisive)','CASPER-R','D1\n(continuous)']
full   =[0.3099,0.3465,0.3609,0.3601,0.3779]
samp   =[0.3978,0.4243,0.4201,0.4257,0.4429]

fig,(axL,axR)=plt.subplots(1,2,figsize=(9.4,4.0),gridspec_kw={'width_ratios':[1.35,1]})

# ---- LEFT: paired bars, honest vs sampled ----
xi=np.arange(len(methods)); w=0.38
axL.bar(xi-w/2,full,w,color=BLUE,label='honest full-catalogue')
axL.bar(xi+w/2,samp,w,color=VERM,alpha=0.85,label='100 sampled negatives')
for x,f,s in zip(xi,full,samp):
    axL.text(x-w/2,f+0.004,f'{f:.2f}',ha='center',fontsize=7,color=BLUE)
    axL.text(x+w/2,s+0.004,f'{s:.2f}',ha='center',fontsize=7,color=VERM)
axL.set_xticks(xi); axL.set_xticklabels(methods,fontsize=8)
axL.set_ylabel('NDCG@10'); axL.set_ylim(0,0.52)
axL.legend(loc='upper left',frameon=False,fontsize=8.4)
axL.set_title('Same data, inflated scores\n(sampled negatives $\\times$1.16–1.28)',fontsize=9.6)

# ---- RIGHT: bump chart, the reordering ----
labels=['entropy (divisiveness)','popularity heuristic']
honest_val=[0.3609,0.3465]   # entropy > popular
samp_val  =[0.4201,0.4243]   # popular > entropy
cols=[GREEN,VERM]
dyr=[-0.0016,0.0016]         # split the near-tied right endpoints for legibility
for lab,hv,sv,c,dy in zip(labels,honest_val,samp_val,cols,dyr):
    axR.plot([0,1],[hv,sv],'-o',color=c,lw=2.2,ms=8)
    axR.text(-0.06,hv,lab,ha='right',va='center',fontsize=8.2,color=c)
    axR.text( 1.06,sv+dy,lab,ha='left', va='center',fontsize=8.2,color=c)
axR.text(0.5,0.392,'ordering FLIPS\n(all 5 seeds)',ha='center',fontsize=9,color='#222222')
axR.set_xticks([0,1]); axR.set_xticklabels(['honest\nfull-catalogue','100 sampled\nnegatives'],fontsize=8.6)
axR.set_xlim(-1.05,2.35); axR.set_ylim(0.338,0.435); axR.set_ylabel('NDCG@10')
axR.set_title('...and reordered verdicts',fontsize=9.6,pad=14)
axR.spines['bottom'].set_visible(True)

fig.tight_layout()
fig.savefig('C:/dev/phd/papers/paper1_casper/figs/fig_protocol_inflation.pdf',bbox_inches='tight')
print('saved fig_protocol_inflation.pdf')
