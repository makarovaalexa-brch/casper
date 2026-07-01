"""
Matrix-Factorization reference vs the learned set-encoder instrument.

Answers: (1) is our instrument a WEAKER recommender than standard MF?
         (2) does our instrument SUPPRESS elicitation gains vs MF?

MF = explicit ALS factorization (item factors + biases) on the training
partition; new test users folded in by ridge least squares on their revealed
(item,polarity) answers -> the classical cold-start recommender
(Golbandi 2011; Zhou 2011). Both recommenders scored on IDENTICAL held-out
users / reveal splits, two lenses:
  (A) recommendation quality at full reveal: NDCG@10, Recall@10, MRR,
      Brier, ROC-AUC over the held-out preference set H.
  (B) elicitation headroom: per-turn NDCG / Brier AUC of random & popularity
      question orders, run UNDER each recommender.
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

NPZ = 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'
INST = 'instrument_ml_stratified_rankcal'
SEED = 42; N_USERS = int(os.environ.get('N_USERS', 250)); HOLD = 0.40; T = 15
D = int(os.environ.get('MF_D', 24)); LAM = float(os.environ.get('MF_LAM', 0.1))
ITERS = int(os.environ.get('MF_ITERS', 12)); N_TRAIN_MF = int(os.environ.get('MF_USERS', 15000))
rng = np.random.default_rng(SEED)

d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
nt = int(d['n_targets']); ni = train.shape[1]
pop = (~np.isnan(train[:, :nt])).mean(0); p_rated = (~np.isnan(train)).mean(0)
movies = list(range(nt))


# ---------------- MF reference (ALS + bias) ----------------
class MFInstrument:
    def __init__(self, Q, b, mu, n_movies, n_items):
        self.Q = Q; self.b = b; self.mu = mu
        self.n_movies = n_movies; self.n_items = n_items
    def _foldin(self, revealed):
        if not revealed:
            return np.zeros(self.Q.shape[1])
        idx = np.array([e for e, _ in revealed]); y = np.array([p for _, p in revealed])
        Qi = self.Q[idx]; r = y - self.mu - self.b[idx]
        A = Qi.T @ Qi + LAM * np.eye(self.Q.shape[1]); c = Qi.T @ r
        return np.linalg.solve(A, c)
    def predict_full(self, revealed):
        u = self._foldin(revealed); return self.mu + self.b + self.Q @ u
    def predict(self, revealed):
        return self.predict_full(revealed)[:self.n_movies]
    def predict_rated(self, revealed):
        return p_rated                                  # MF has no answerability head -> popularity proxy


class WRMFInstrument:
    """Implicit-feedback MF (Hu/Koren/Volinsky 2008): a proper top-N RANKER.
    liked=positive, everything else (disliked+unrated)=negative; observed get
    higher confidence. Predictions are preference scores (not calibrated)."""
    def __init__(self, Q, QtQ, alpha, n_movies, n_items):
        self.Q = Q; self.QtQ = QtQ; self.alpha = alpha
        self.n_movies = n_movies; self.n_items = n_items
    def _foldin(self, revealed):
        dd = self.Q.shape[1]
        A = self.QtQ + LAM * np.eye(dd); rhs = np.zeros(dd)
        for e, p in revealed:
            q = self.Q[e]; A = A + self.alpha * np.outer(q, q)
            if p >= 0.5: rhs = rhs + (1 + self.alpha) * q
        return np.linalg.solve(A, rhs)
    def predict_full(self, revealed):
        return self.Q @ self._foldin(revealed)
    def predict(self, revealed):
        return self.predict_full(revealed)[:self.n_movies]
    def predict_rated(self, revealed):
        return p_rated


class PopRank:
    """Non-personalized popularity ranking reference."""
    def __init__(self, n_movies, n_items):
        self.n_movies = n_movies; self.n_items = n_items
    def predict_full(self, revealed): return p_rated.copy()
    def predict(self, revealed): return p_rated[:self.n_movies]
    def predict_rated(self, revealed): return p_rated


def train_wrmf(alpha=40.0):
    t0 = time.time()
    sub = rng.choice(len(train), min(N_TRAIN_MF, len(train)), replace=False)
    R = train[sub]; nu = len(R)
    pos = (R == 1)                                       # liked = positive
    obs = ~np.isnan(R)                                   # rated = high confidence
    P = 0.01 * rng.standard_normal((nu, D)); Q = 0.01 * rng.standard_normal((ni, D))
    user_obs = [np.where(obs[u])[0] for u in range(nu)]
    user_pos = [np.where(pos[u])[0] for u in range(nu)]
    item_obs = [np.where(obs[:, i])[0] for i in range(ni)]
    item_pos = [np.where(pos[:, i])[0] for i in range(ni)]
    Id = LAM * np.eye(D)
    for it in range(ITERS):
        QtQ = Q.T @ Q
        for u in range(nu):
            oi = user_obs[u]
            A = QtQ + Id + alpha * (Q[oi].T @ Q[oi])
            rhs = (1 + alpha) * Q[user_pos[u]].sum(0) if len(user_pos[u]) else np.zeros(D)
            P[u] = np.linalg.solve(A, rhs)
        PtP = P.T @ P
        for i in range(ni):
            ou = item_obs[i]
            A = PtP + Id + alpha * (P[ou].T @ P[ou])
            rhs = (1 + alpha) * P[item_pos[i]].sum(0) if len(item_pos[i]) else np.zeros(D)
            Q[i] = np.linalg.solve(A, rhs)
        if (it + 1) % 4 == 0:
            print(f"  WRMF it{it+1}/{ITERS} ({time.time()-t0:.0f}s)", flush=True)
    return WRMFInstrument(Q, Q.T @ Q, alpha, nt, ni)


def train_mf():
    t0 = time.time()
    sub = rng.choice(len(train), min(N_TRAIN_MF, len(train)), replace=False)
    R = train[sub]                                       # [nu, ni] with NaN
    nu = len(R)
    obs = ~np.isnan(R)
    mu = float(np.nanmean(R))
    P = 0.01 * rng.standard_normal((nu, D)); Q = 0.01 * rng.standard_normal((ni, D))
    b = np.zeros(ni)
    # precompute per-user and per-item observed index lists
    user_items = [np.where(obs[u])[0] for u in range(nu)]
    item_users = [np.where(obs[:, i])[0] for i in range(ni)]
    I = LAM * np.eye(D)
    for it in range(ITERS):
        for u in range(nu):
            ii = user_items[u]
            if len(ii) == 0: continue
            Qi = Q[ii]; r = R[u, ii] - mu - b[ii]
            P[u] = np.linalg.solve(Qi.T @ Qi + I, Qi.T @ r)
        for i in range(ni):
            uu = item_users[i]
            if len(uu) == 0: continue
            Pu = P[uu]; r = R[uu, i] - mu - b[i]
            Q[i] = np.linalg.solve(Pu.T @ Pu + I, Pu.T @ r)
            b[i] = float(np.mean(R[uu, i] - mu - Pu @ Q[i]))
        if (it + 1) % 4 == 0:
            # train RMSE
            se = 0.0; n = 0
            for u in range(0, nu, 50):
                ii = user_items[u]
                if len(ii) == 0: continue
                pred = mu + b[ii] + Q[ii] @ P[u]; se += np.sum((R[u, ii] - pred) ** 2); n += len(ii)
            print(f"  MF ALS it{it+1}/{ITERS} train RMSE={np.sqrt(se/n):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return MFInstrument(Q, b, mu, nt, ni)


# ---------------- shared eval machinery ----------------
def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    alls = np.concatenate([pos, neg]); order = alls.argsort()
    ranks = np.empty(len(alls)); ranks[order] = np.arange(1, len(alls) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

cases = []
for prof in test:
    rated = np.where(~np.isnan(prof[:nt]))[0]
    if len(rated) < 8: continue
    liked = rated[prof[rated] == 1]; disl = rated[prof[rated] == 0]
    if len(liked) < 4 or len(disl) < 2: continue
    perm = rng.permutation(rated); k = max(2, int(HOLD * len(rated)))
    H = perm[:k]; askable = set(int(x) for x in perm[k:])
    Hl = [int(x) for x in H if prof[x] == 1]
    if len(Hl) < 1 or sum(prof[x] == 0 for x in H) < 1: continue
    cases.append((prof, askable, np.array(H), np.array(Hl)))
    if len(cases) >= N_USERS: break
print(f"users={len(cases)}", flush=True)


def quality_full_reveal(rec, name):
    nd = []; rc = []; mr = []; br = []; au = []
    for (prof, askable, H, Hl) in cases:
        rated_set = set(int(x) for x in np.where(~np.isnan(prof[:nt]))[0])
        rev = [(int(e), float(prof[e])) for e in askable if e < nt]
        bel = np.asarray(rec.predict_full(rev))[:nt]
        negs = np.array([m for m in movies if m not in rated_set])
        # per-target NDCG/MRR
        nds = []; mrs = []
        for h in Hl:
            rank = 1 + int((bel[negs] >= bel[h]).sum())
            nds.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0); mrs.append(1.0 / rank)
        nd.append(np.mean(nds)); mr.append(np.mean(mrs))
        # Recall@10 over the held-out liked set vs non-interacted
        cand = np.array(list(Hl) + [m for m in negs])
        top = cand[np.argsort(-bel[cand])[:10]]
        rc.append(len(set(top.tolist()) & set(Hl)) / len(Hl))
        p = np.clip(bel[H], 1e-6, 1 - 1e-6); y = prof[H]
        br.append(np.mean((p - y) ** 2)); a = auc(bel[H], y)
        if not np.isnan(a): au.append(a)
    print(f"  {name:<22} NDCG@10={np.mean(nd):.4f}  Recall@10={np.mean(rc):.4f}  "
          f"MRR={np.mean(mr):.4f}  Brier={np.mean(br):.4f}  ROC-AUC={np.mean(au):.4f}", flush=True)


porder = [int(e) for e in np.argsort(-pop)]
def elicit(rec, policy, name):
    ndc = np.zeros(T + 1); brc = np.zeros(T + 1); c = 0
    rrng = np.random.default_rng(123)
    for (prof, askable, H, Hl) in cases:
        rated_set = set(int(x) for x in np.where(~np.isnan(prof[:nt]))[0])
        negs = np.array([m for m in movies if m not in rated_set])
        forb = set(int(x) for x in H); rev = []; asked = set()
        cand_movies = [m for m in movies if m not in forb]
        for t in range(T + 1):
            bel = np.asarray(rec.predict_full(rev))[:nt]
            nds = []
            for h in Hl:
                rank = 1 + int((bel[negs] >= bel[h]).sum())
                nds.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
            ndc[t] += np.mean(nds)
            p = np.clip(bel[H], 1e-6, 1 - 1e-6); brc[t] += np.mean((p - prof[H]) ** 2)
            if t == T: break
            if policy == 'random':
                a = int(rrng.choice([m for m in cand_movies if m not in asked]))
            else:  # popularity
                a = next((e for e in porder if e not in asked and e not in forb), None)
            if a is None: continue
            asked.add(a)
            if a in askable and not np.isnan(prof[a]): rev.append((a, float(prof[a])))
        c += 1
    ndc /= c; brc /= c
    print(f"  {name:<28} NDCG: t0={ndc[0]:.3f} t15={ndc[-1]:.3f} AUC={ndc.mean():.4f} | "
          f"Brier: t0={brc[0]:.3f} t15={brc[-1]:.3f} | gain(NDCG t0->t15)={ndc[-1]-ndc[0]:+.3f}", flush=True)


print("\n=== training MF references ===", flush=True)
mf = train_mf()
wrmf = train_wrmf()
poprank = PopRank(nt, ni)
inst, _ = load_instrument_by_name(INST)

print("\n=== (A) RECOMMENDATION QUALITY @ full reveal ===", flush=True)
print("  -- ranking lens (each model on the task it is built for) --", flush=True)
quality_full_reveal(poprank, "popularity-rank")
quality_full_reveal(mf, "explicit MF (recon)")
quality_full_reveal(wrmf, "WRMF (implicit, ranker)")
quality_full_reveal(inst, "Learned instrument")

print("\n=== (B) ELICITATION HEADROOM (per-turn under each recommender) ===", flush=True)
for rec, tag in [(wrmf, 'WRMF'), (inst, 'INSTR')]:
    elicit(rec, 'random', f'{tag} / random')
    elicit(rec, 'popularity', f'{tag} / popularity')
