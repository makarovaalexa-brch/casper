"""
Improved ranking instrument (rank2): set-encoder user tower + TWO-TOWER
dot-product item embeddings (L2-normalised + temperature), ranking loss.
Goal: widen the honest baseline->revealed spread by letting a revealed movie
propagate to similar movies via shared embedding space (vs the rank-v1 flat
output head). Bigger (d=256) + longer; saves best checkpoint DURING training.
Self-reports the hard popularity-matched LOO curve + permutation risers so we
can compare directly to rank-v1.

Usage: DATASET_NAME=ml1m poetry run python scripts/paper1/train_instrument_rank2.py
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_instrument_lib import CHECKPOINT_DIR
from train_instrument_rank import build_batch, liked_rank_loss, eval_ndcg

NAME = os.environ.get('DATASET_NAME', 'ml1m')
NPZ = os.environ.get('DATASET_NPZ',
                     {'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz',
                      'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'}[NAME])
INST = CHECKPOINT_DIR / f'instrument_{NAME}_rank2.pt'
D_MODEL = int(os.environ.get('D_MODEL', 256)); EPOCHS = int(os.environ.get('EPOCHS', 150))
PATIENCE = int(os.environ.get('PATIENCE', 30)); UPE = int(os.environ.get('USERS_PER_EPOCH', 8000))
MAX_REVEAL = 60; SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


class RankTwoTower(nn.Module):
    def __init__(self, n_items, n_movies, d_model=256, n_heads=4, n_layers=2):
        super().__init__()
        self.n_items, self.n_movies = n_items, n_movies
        self.item_emb = nn.Embedding(n_items, d_model)
        self.pol_emb = nn.Embedding(2, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)); nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.movie_out = nn.Embedding(n_movies, d_model)
        self.head_rated = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(),
                                        nn.Linear(2 * d_model, n_items))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def user_vec(self, idx, pol, pad):
        b = idx.shape[0]
        tok = self.item_emb(idx) + self.pol_emb(pol)
        x = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1)
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool), pad], dim=1)
        return self.encoder(x, src_key_padding_mask=mask)[:, 0]

    def forward(self, idx, pol, pad):
        h = F.normalize(self.user_vec(idx, pol, pad), dim=-1)
        m = F.normalize(self.movie_out.weight, dim=-1)
        liked = self.logit_scale.exp() * (h @ m.T)          # [b, n_movies]
        return liked, self.head_rated(h)


def predict_scores(model, revealed_list, nt):
    b = len(revealed_list); L = max(1, max((len(r) for r in revealed_list), default=1))
    idx = torch.zeros(b, L, dtype=torch.long); pol = torch.zeros(b, L, dtype=torch.long)
    pad = torch.ones(b, L, dtype=torch.bool)
    for i, rev in enumerate(revealed_list):
        for j, (e, p) in enumerate(rev[:MAX_REVEAL]):
            idx[i, j] = int(e); pol[i, j] = 1 if p >= 0.5 else 0; pad[i, j] = False
    with torch.no_grad():
        return model(idx, pol, pad)[0][:, :nt].numpy()


def hard_loo(model, test, train, nt, ks=(0, 1, 3, 5, 10, 20), n_neg=50, n_users=600):
    pop = (~np.isnan(train[:, :nt])).mean(0)
    dec = np.zeros(nt, int); order = np.argsort(pop)
    for q in range(10):
        dec[order[q * nt // 10:(q + 1) * nt // 10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}
    rng = np.random.default_rng(0); users = []
    for prof in test[:n_users]:
        liked = np.where(prof[:nt] == 1)[0]
        rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3:
            continue
        rng.shuffle(liked); tgt = int(liked[0])
        poolneg = [i for i in by[dec[tgt]] if i not in rated]
        if len(poolneg) < n_neg:
            continue
        negs = list(rng.choice(poolneg, n_neg, replace=False))
        rpool = [int(x) for x in liked[1:]] + [int(i) for i in np.where(prof[:nt] == 0)[0]]
        rng.shuffle(rpool)
        users.append((prof, tgt, np.array([tgt] + negs), rpool))
    rows = {k: [] for k in ks}
    for K in ks:
        revs = [[(e, float(p[e])) for e in rp[:K] if e != tgt] for (p, tgt, c, rp) in users]
        sc = predict_scores(model, revs, nt)
        for u, (prof, tgt, cand, rp) in enumerate(users):
            s = sc[u][cand]; rank = 1 + int((s[1:] >= s[0]).sum())
            rows[K].append(1.0 if rank <= 10 else 0.0)
    return {K: float(np.mean(v)) for K, v in rows.items()}


def risers(model, items, nt, names, typ, subs):
    def find(sub, kind):
        s = sub.lower()
        for i, n in enumerate(names):
            if s in n.lower() and typ[i] == kind:
                return i
        return None
    s0 = predict_scores(model, [[]], nt)[0]
    for sub, kind in subs:
        idx = find(sub, kind)
        if idx is None:
            print(f"    [{sub}] not found"); continue
        sp = predict_scores(model, [[(idx, 1.0)]], nt)[0]
        up = sp - s0
        if idx < nt:
            up[idx] = -1e9
        top = np.argsort(-up)[:8]
        print(f"    +{names[idx]} -> " + " | ".join(names[i] for i in top))


def main():
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets']); ni = train.shape[1]
    items = [tuple(x) for x in d['items'].tolist()]
    names = [str(it[2]) for it in items]; typ = [it[0] for it in items]
    print(f"rank2 {NAME}: {len(train)} train, {ni} items, {nt} movies, d={D_MODEL}", flush=True)
    model = RankTwoTower(ni, nt, D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    rng = np.random.default_rng(SEED); best, be = -1, -1; t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(len(train))[:UPE]
        for s in range(0, len(perm), 256):
            arr = [train[i] for i in perm[s:s + 256]]
            bi, bp, pad, tgt, revt, ratedm = build_batch(arr, nt, ni, rng)
            opt.zero_grad()
            lk, rt = model(bi, bp, pad)
            loss = liked_rank_loss(lk, tgt, revt, nt) + F.binary_cross_entropy_with_logits(rt, ratedm)
            loss.backward(); opt.step()
        if ep % 2 and ep < EPOCHS - 1:
            continue
        _, fn, _, fh, rn = eval_ndcg(model, test, nt, ni, n=300)
        if fn > best:
            best, be = fn, ep
            torch.save({'model_state_dict': model.state_dict(), 'arch': 'rank_two_tower',
                        'd_model': D_MODEL, 'n_items': ni, 'n_movies': nt}, INST)
        if (ep + 1) % 10 == 0:
            print(f"  ep{ep+1} attr-NDCG full={fn:.4f} best={best:.4f}@{be+1} {time.time()-t0:.0f}s", flush=True)
        if ep - be >= PATIENCE:
            break
    model.load_state_dict(torch.load(INST)['model_state_dict'])
    print("\n=== rank2 HARD popularity-matched LOO curve (Hit@10) ===", flush=True)
    curve = hard_loo(model, test, train, nt)
    for K, v in curve.items():
        print(f"   K={K:>3}  Hit@10={v:.4f}")
    print(f"   spread K0->K20 = {curve[20]-curve[0]:+.4f}  (rank-v1: 0.433->0.647, +0.214)")
    print("\n=== rank2 permutation risers ===")
    risers(model, items, nt, names, typ,
           [("Gladiator", 'movie'), ("Toy Story", 'movie'),
            ("L.A. Confidential", 'movie'), ("Romance", 'genre'), ("Horror", 'genre')])
    print(f"\nsaved {INST}", flush=True)


if __name__ == '__main__':
    main()
