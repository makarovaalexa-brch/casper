#!/usr/bin/env python
"""HONEST question-space figure: PCA (linear, no clustering bias) of questions + movies + tags,
plus a cosine-overlap panel proving questions live IN the catalog cloud (not a separate galaxy).
t-SNE manufactured an island because the actor's outputs are a tight self-similar sub-manifold."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE='C:/dev/phd/casper'
d=np.load(f'{BASE}/data/movielens/.cache/qviz.npz',allow_pickle=True)
OUT='C:/dev/phd/papers/paper3_casper/fig_qspace'
def unit(X): return (X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)).astype(np.float32)
M=unit(d['movies']); C=unit(d['concepts']); Q=unit(d['questions']); qt=d['qturn']
rng=np.random.default_rng(0); mi=rng.choice(len(M),min(1800,len(M)),replace=False); Ms=M[mi]

# PCA on the JOINT set (so all three share axes); center first
X=np.vstack([Ms,C,Q]); mu=X.mean(0); Xc=X-mu
U,S,Vt=np.linalg.svd(Xc,full_matrices=False); P=Xc@Vt[:2].T
nM,nC=len(Ms),len(C); Pm,Pc,Pq=P[:nM],P[nM:nM+nC],P[nM+nC:]

fig,(ax,axh)=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={'width_ratios':[1.25,1]})
ax.scatter(Pm[:,0],Pm[:,1],s=6,c='#c8c8c8',alpha=0.5,lw=0,label=f'movies (n={len(M)})',zorder=1)
ax.scatter(Pc[:,0],Pc[:,1],s=24,c='#e69138',alpha=0.8,lw=0,marker='^',label=f'tags/concepts (n={nC})',zorder=2)
sc=ax.scatter(Pq[:,0],Pq[:,1],s=16,c=qt,cmap='winter',alpha=0.75,lw=0,label=f'learned questions (n={len(Q)})',zorder=3)
cb=fig.colorbar(sc,ax=ax,fraction=0.04,pad=0.02); cb.set_label('question turn (0=opener → 7)',fontsize=8)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('PCA (linear) — questions interleave with movies & tags',fontsize=10)
ax.legend(loc='upper right',fontsize=8,framealpha=0.92,markerscale=1.3)

# cosine-overlap panel: movie-movie vs movie-question vs question-question
def pc(A,B,n=40000):
    ia=rng.integers(0,len(A),n); ib=rng.integers(0,len(B),n); return np.einsum('ij,ij->i',A[ia],B[ib])
bins=np.linspace(-1,1,60)
axh.hist(pc(M,M),bins=bins,density=True,alpha=0.5,color='#777777',label='movie–movie')
axh.hist(pc(M,Q),bins=bins,density=True,alpha=0.5,color='#0b5394',label='movie–question')
axh.hist(pc(Q,Q),bins=bins,density=True,alpha=0.4,color='#6aa84f',label='question–question')
axh.axvline(0,color='k',lw=0.6,ls=':')
axh.set_xlabel('cosine similarity'); axh.set_ylabel('density')
axh.set_title('movie–question ≈ movie–movie (same neighbourhood)',fontsize=10)
axh.legend(fontsize=8)
fig.suptitle("Where the learned questions live in the recommender's space (the t-SNE \"island\" was a projection artifact)",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96])
for ext in ('pdf','png'):
    fig.savefig(f'{OUT}.{ext}',bbox_inches='tight',dpi=160); print('wrote',f'{OUT}.{ext}')
print(f'PC1/PC2 explain {(S[:2]**2).sum()/(S**2).sum()*100:.1f}% of variance')
