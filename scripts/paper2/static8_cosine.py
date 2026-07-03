"""Exp2 helper: per-turn cosine between the learned STATIC-8 direction bank and D1's turn-mean queries."""
import os,numpy as np,torch
base='C:/dev/phd/casper/data/movielens'
ck=os.environ.get('CK',f'{base}/.cache/policy_static8_best.pt')
sd=torch.load(ck); bank=(sd['actor']['bank'] if isinstance(sd,dict) and 'actor' in sd else sd['bank']).numpy()
d1=np.load(f'{base}/.cache/static8_d1means.npy')
def cos(a,b): return float(a@b/((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))
print(f"CK={os.path.basename(ck)}  bank{bank.shape} vs D1-means{d1.shape}")
print("per-turn cos(static_bank[t], D1_turnmean[t]):")
for t in range(bank.shape[0]):
    print(f"  turn {t}: cos={cos(bank[t],d1[t]):+.3f}  |bank|={np.linalg.norm(bank[t]):.3f} |d1|={np.linalg.norm(d1[t]):.3f}")
print(f"mean per-turn cos = {np.mean([cos(bank[t],d1[t]) for t in range(bank.shape[0])]):+.3f}")
# cross-turn cosine matrix of the learned static bank (are the 8 questions diverse or collapsed?)
B=bank/(np.linalg.norm(bank,axis=1,keepdims=True)+1e-9); G=B@B.T
print("learned static bank cross-turn |cos| stats: mean-offdiag=%.3f  max-offdiag=%.3f"%(
    (np.abs(G)-np.eye(8)).sum()/ (8*7), np.max(np.abs(G-np.eye(8)))))
