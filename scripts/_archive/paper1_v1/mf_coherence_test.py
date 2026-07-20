"""
Does the franchise-flip (T3) / genre-coherence test even WORK on MF & WRMF (ML-100k)?
If the strong baselines also fail, the test is too hard on this small data (not an
instrument-specific defect). Same anchors/logic as instrument_v2_tests.py.
"""
import numpy as np
base = 'C:/dev/phd/casper/data/movielens/ml-100k'
rng = np.random.default_rng(0)
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
GEN = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
title = {}; genres = {}
with open(f'{base}/u.item', encoding='latin-1') as f:
    for line in f:
        p = line.rstrip('\n').split('|'); mid = int(p[0])
        if mid not in iids: continue
        idx = iids[mid]; title[idx] = p[1]; fl = [int(x) for x in p[5:24]]
        genres[idx] = set(GEN[k] for k in range(19) if fl[k])
# warm = all users (no cold split needed for coherence; use whole data to train item factors)
mu = float(R.mean()); D = 32
bu = np.zeros(nu); bi = np.zeros(ni); P = 0.1*rng.standard_normal((nu, D)); Q = 0.1*rng.standard_normal((ni, D))
for ep in range(30):
    o = rng.permutation(len(U)); lr = 0.008/(1+0.05*ep)
    for s in range(0, len(o), 20000):
        b = o[s:s+20000]; uu, ii, rr = u[b], i[b], R[b]
        e = (rr-(mu+bu[uu]+bi[ii]+np.sum(P[uu]*Q[ii],1))).astype(np.float64)
        np.add.at(bu, uu, lr*(e-0.05*bu[uu])); np.add.at(bi, ii, lr*(e-0.05*bi[ii]))
        np.add.at(P, uu, lr*(e[:,None]*Q[ii]-0.05*P[uu])); np.add.at(Q, ii, lr*(e[:,None]*P[uu]-0.05*Q[ii]))
def mf_predict(rev):
    if not rev: return mu+bi
    A = np.array([[1.]+Q[j].tolist() for j,_ in rev]); y = np.array([r-mu-bi[j] for j,r in rev])
    x = np.linalg.solve(A.T@A+4.0*np.eye(D+1), A.T@y); return mu+x[0]+bi+Q@x[1:]
# WRMF
Rm = np.full((nu, ni), np.nan)
for k in range(len(u)): Rm[u[k], i[k]] = R[k]
pos = (Rm >= 4); obs = ~np.isnan(Rm); alpha = 40.; WD = 32
Pw = 0.01*rng.standard_normal((nu, WD)); Qw = 0.01*rng.standard_normal((ni, WD)); Id = 0.1*np.eye(WD)
uo = [np.where(obs[x])[0] for x in range(nu)]; up = [np.where(pos[x])[0] for x in range(nu)]
io = [np.where(obs[:, j])[0] for j in range(ni)]; ip = [np.where(pos[:, j])[0] for j in range(ni)]
for it_ in range(15):
    QtQ = Qw.T@Qw
    for x in range(nu): A = QtQ+Id+alpha*(Qw[uo[x]].T@Qw[uo[x]]); Pw[x] = np.linalg.solve(A, (1+alpha)*Qw[up[x]].sum(0) if len(up[x]) else np.zeros(WD))
    PtP = Pw.T@Pw
    for j in range(ni): A = PtP+Id+alpha*(Pw[io[j]].T@Pw[io[j]]); Qw[j] = np.linalg.solve(A, (1+alpha)*Pw[ip[j]].sum(0) if len(ip[j]) else np.zeros(WD))
QwtQw = Qw.T@Qw
def wrmf_predict(rev):
    A = QwtQw+Id; rhs = np.zeros(WD)
    for j, r in rev:
        q = Qw[j]; A = A+alpha*np.outer(q, q)
        if r >= 4: rhs = rhs+(1+alpha)*q
    return Qw@np.linalg.solve(A, rhs)
print("MF + WRMF trained on ML-100k\n")
def find(sub):
    for idx, t in title.items():
        if sub.lower() in t.lower(): return idx
    return None
fr = {'Star Wars': ['Empire Strikes Back', 'Return of the Jedi'], 'Godfather': ['Godfather: Part II'],
      'Raiders of the Lost Ark': ['Last Crusade', 'Temple of Doom'], 'Back to the Future': ['Back to the Future Part'],
      'Die Hard': ['Die Hard 2', 'Die Hard: With']}
def t3(predict, name):
    cor = tot = 0; rows = []
    for anc, subs in fr.items():
        a = find(anc)
        if a is None: continue
        rel = [r for r in {find(s) for s in subs} - {None, a} if r is not None]
        if not rel: continue
        sl = predict([(a, 5.0)]); sd = predict([(a, 1.0)])
        for r in rel:
            rl = 1+int((sl >= sl[r]).sum()); rd = 1+int((sd >= sd[r]).sum()); ok = rl < rd
            cor += ok; tot += 1; rows.append(f"   {title[r][:30]:<30} like#{rl:<4} dislike#{rd:<4} {'OK' if ok else 'X'}")
    print(f"--- {name}: T3 franchise flip {cor}/{tot} = {100*cor/max(tot,1):.0f}% ---")
    for r in rows: print(r)
def risers(predict, name):
    b0 = predict([]); print(f"\n--- {name}: risers (LIKE Star Wars) ---")
    a = find('Star Wars'); d = predict([(a, 5.0)]) - b0; d[a] = -1e9
    ag = genres.get(a, set()); share = np.mean([len(genres.get(int(e), set()) & ag) > 0 for e in np.argsort(-d)[:20]])
    print(f"   top-20 genre share: {share*100:.0f}%")
    for e in np.argsort(-d)[:6]: print(f"      +{d[e]:.3f} {title[int(e)][:40]:<40} {sorted(genres.get(int(e),set()))}")
t3(mf_predict, "MF"); t3(wrmf_predict, "WRMF")
risers(mf_predict, "MF"); risers(wrmf_predict, "WRMF")
