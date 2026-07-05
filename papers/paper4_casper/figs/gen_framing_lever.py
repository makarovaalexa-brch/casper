"""Paper D framing-lever figure: popularity-percentile distributions of the movie an LLM user
names under 'favourite/love' vs 'hidden gem' framing (FULL context, gpt-4o-mini).
Data: casper/experiments/paper2/llmuser_phase1_raw.pkl (key A). See LLMUSER_PHASE1_RESULT.md.
The headline statistic is the -33.4 pt PAIRED shift (n=181 users, Wilcoxon p~0); the pooled
medians (98.7 love, 71.5 gem) move consistently. Output: fig_framing_lever.pdf"""
import pickle, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.size':10,'font.family':'serif','axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':150,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#0072B2'; VERM='#D55E00'
d=pickle.load(open('C:/dev/phd/casper/experiments/paper2/llmuser_phase1_raw.pkl','rb'))
A=d['A']
love=np.array([t[1] for t in A[('main','FULL','love')] if t[1] is not None])
gem =np.array([t[1] for t in A[('main','FULL','gem')]  if t[1] is not None])

fig,ax=plt.subplots(figsize=(6.6,3.6))
bins=np.linspace(0,100,26)
ax.hist(love,bins=bins,color=BLUE,alpha=0.70,label=f'"name a movie you love"   (median {np.median(love):.0f})',edgecolor='white',linewidth=0.4)
ax.hist(gem, bins=bins,color=VERM,alpha=0.70,label=f'"name a hidden gem you love"   (median {np.median(gem):.0f})',edgecolor='white',linewidth=0.4)
ax.axvline(np.median(love),color=BLUE,ls='--',lw=1.3)
ax.axvline(np.median(gem), color=VERM,ls='--',lw=1.3)
ax.set_xlabel('Popularity percentile of the named movie  (0 = obscure tail, 100 = most-rated head)')
ax.set_ylabel('LLM-user responses')
ax.set_xlim(0,100)
ymax=ax.get_ylim()[1]
ax.legend(loc='upper left',frameon=False,fontsize=8.6)
ax.text(0.5, ymax*0.985, r'framing lever: $-33$ pt paired shift toward the tail  ($n{=}181$, Wilcoxon $p\!\approx\!0$)',
        ha='center',fontsize=9.2,color='#222222')
ax.annotate('', xy=(np.median(gem),ymax*0.72), xytext=(np.median(love),ymax*0.72),
            arrowprops=dict(arrowstyle='->',color='#444444',lw=1.3))
ax.annotate('The Matrix,\nBack to the Future', xy=(98.7,ymax*0.40), xytext=(70,ymax*0.52),
            fontsize=7.6,color=BLUE,ha='center',arrowprops=dict(arrowstyle='-',color=BLUE,lw=0.7))
ax.annotate('Paris, Texas;\nCarnival of Souls', xy=(52,ymax*0.16), xytext=(30,ymax*0.42),
            fontsize=7.6,color=VERM,ha='center',arrowprops=dict(arrowstyle='-',color=VERM,lw=0.7))
fig.tight_layout()
fig.savefig('C:/dev/phd/papers/paper4_casper/figs/fig_framing_lever.pdf',bbox_inches='tight')
print('love n=%d median=%.1f | gem n=%d median=%.1f'%(len(love),np.median(love),len(gem),np.median(gem)))
