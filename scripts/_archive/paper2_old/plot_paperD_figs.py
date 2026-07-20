"""Paper D figures: (1) anytime curve learned-vs-fixed, (2) the learned question TREE (branching, from greedy paths).
Reads experiments/paper2/tree_data.json (written by POLOPEN TREEOUT=...). Outputs PDFs to papers/paper4_casper/."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

D=json.load(open('experiments/paper2/tree_data.json')); seqs=D['seqs']; OUT='C:/dev/phd/papers/paper4_casper'
LBL={'gem':'hidden-gem','fav':'favourite','align':'all-time fav','hate':'disliked','genre':'fav genre',
     'avoidgenre':'avoid genre','actor':'fav actor','director':'fav director','whatdoyoulike':'what do you like?','closed':'cont. probe'}

# ---------- (1) anytime curve ----------
turns=np.arange(1,len(D['learned_full'])+1)
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
for a,key,ttl in [(ax[0],'full','NDCG@10 (full)'),(ax[1],'tail','NDCG@10 (long tail)')]:
    a.plot(turns,D['learned_'+key],'-o',lw=2,ms=4,color='#1f77b4',label='learned adaptive')
    a.plot(turns,D['fixed_'+key],'--s',lw=2,ms=4,color='#888',label='best fixed order')
    a.set_title(ttl,fontsize=10); a.set_xlabel('questions asked'); a.grid(alpha=.3); a.set_xticks(turns)
ax[0].set_ylabel('NDCG@10'); ax[0].legend(fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_anytime_curve.pdf'); print('wrote fig_anytime_curve.pdf')

# ---------- (2) learned tree (trie of greedy paths, pruned) ----------
N=len(seqs); DEPTH=5; MINFRAC=0.06
# build trie nodes: id -> {turn,type,count,children{type:id},x,y}
nodes=[{'turn':-1,'type':'START','count':N,'children':{}}]
for s in seqs:
    cur=0
    for d in range(min(DEPTH,len(s))):
        ty=s[d]; ch=nodes[cur]['children']
        if ty not in ch:
            nodes.append({'turn':d,'type':ty,'count':0,'children':{}}); ch[ty]=len(nodes)-1
        cur=ch[ty]; nodes[cur]['count']+=1
# prune children below MINFRAC of parent
def prune(i):
    keep={t:j for t,j in nodes[i]['children'].items() if nodes[j]['count']>=max(6,MINFRAC*nodes[i]['count'])}
    nodes[i]['children']=keep
    for j in keep.values(): prune(j)
prune(0)
# layout: leaves get sequential y; internal = mean(children); x=turn+1
ypos={}; _c=[0.]
def layout(i):
    ch=nodes[i]['children']
    if not ch: ypos[i]=_c[0]; _c[0]+=1; return ypos[i]
    ys=[layout(j) for j in ch.values()]; ypos[i]=float(np.mean(ys)); return ypos[i]
layout(0)
fig,ax=plt.subplots(figsize=(9,6))
def draw(i):
    x=nodes[i]['turn']+1; y=ypos[i]
    for t,j in nodes[i]['children'].items():
        xj=nodes[j]['turn']+1; yj=ypos[j]; w=0.5+4.0*nodes[j]['count']/N
        ax.plot([x+.42,xj-.42],[y,yj],'-',color='#bbb',lw=w,zorder=1); draw(j)
    if i>0:
        frac=100*nodes[i]['count']/N
        ax.add_patch(FancyBboxPatch((x-.42,y-.22),.84,.44,boxstyle='round,pad=0.02',fc='#dceefb',ec='#1f77b4',zorder=2))
        ax.text(x,y,f"{LBL.get(nodes[i]['type'],nodes[i]['type'])}\n{frac:.0f}%",ha='center',va='center',fontsize=7,zorder=3)
draw(0)
ax.text(0,ypos[0],'cold\nstart',ha='center',va='center',fontsize=8,style='italic')
for d in range(DEPTH): ax.text(d+1,-.8,f'Q{d+1}',ha='center',fontsize=9,weight='bold')
ax.set_xlim(-.5,DEPTH+.6); ax.set_ylim(-1.2,_c[0]); ax.axis('off')
ax.set_title('Learned question tree (greedy policy; % = users reaching node)',fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_learned_tree.pdf'); print('wrote fig_learned_tree.pdf')
