"""
Instrument v7: TWO-TOWER SET-ENCODER trained with sampled-softmax + explicit
negatives -- fixes the rated-ness artifact while keeping embedding-space scoring
(so it supports continuous / LLM-generated probes).

Grounding:
  two-tower retrieval + sampled softmax + logQ correction  (Covington 2016; Yi 2019)
  implicit negatives / pairwise ranking                     (Rendle BPR 2009; Hu iALS 2008)
  permutation-invariant amortized set encoder              (Zaheer DeepSets 2017; Garnelo CNP 2018)
  embedding-space cold-start elicitation (closest precedent)(Nguyen et al. UAI 2024)

Design: user tower = mean-pool DeepSets over revealed (item_emb + polarity_emb)
tokens -> user embedding u; item tower = embedding table Z; score = u . z_i.
Loss = sampled softmax with held-out LIKED items as positives, the FULL catalog
(disliked + unrated) as negatives, logQ (popularity) correction; revealed items
masked out of the denominator.

Trains on a SMALL subset for fast iteration; compares to WRMF + the old instrument
on the artifact control (disliked-in-top10) and ranking (liked-vs-disliked AUC).
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from test_instrument_lib import load_instrument_by_name

NPZ = 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz'
SEED = 42; D = int(os.environ.get('D', 64)); EPOCHS = int(os.environ.get('EPOCHS', 25))
N_TRAIN = int(os.environ.get('TT_USERS', 8000)); BS = 256; LAM = 0.1
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
nt = int(d['n_targets']); ni = train.shape[1]
pop = (~np.isnan(train[:, :nt])).mean(0)
logpop_movies = torch.tensor(np.log(np.clip(pop, 1e-4, None)), dtype=torch.float32)
OUT = 'C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_twotower_ml_stratified.pt'


class TwoTower(nn.Module):
    def __init__(self, n_items, d):
        super().__init__()
        self.Z = nn.Embedding(n_items, d)            # item tower
        self.pol = nn.Embedding(2, d)                # polarity (0=disliked,1=liked)
        self.enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.post = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.u0 = nn.Parameter(torch.zeros(d))       # empty-context prior
        self.rated_head = nn.Linear(d, n_items)      # answerability aux head
    def user_emb(self, ctx_items, ctx_pol, ctx_mask):
        # ctx_*: [B,L]; mask True=pad
        tok = self.Z(ctx_items) + self.pol(ctx_pol)  # [B,L,d]
        h = self.enc(tok)
        h = h.masked_fill(ctx_mask.unsqueeze(-1), 0.0)
        n = (~ctx_mask).sum(1, keepdim=True).clamp(min=1)
        pooled = h.sum(1) / n                          # masked mean (DeepSets)
        empty = (n.squeeze(1) == 0)
        u = self.post(pooled)
        u = torch.where(empty.unsqueeze(1), self.u0.unsqueeze(0).expand_as(u), u)
        return u
    def forward(self, ctx_items, ctx_pol, ctx_mask):
        u = self.user_emb(ctx_items, ctx_pol, ctx_mask)
        logits = u @ self.Z.weight.t()                # [B, n_items]
        return logits, u


def make_batch(users):
    B = len(users); L = 1
    ctxs = []
    for prof in users:
        rated = np.where(~np.isnan(prof[:nt]))[0]
        k = int(np.clip(np.exp(rng.uniform(0, np.log(len(rated)))), 1, len(rated) - 1)) if len(rated) > 1 else 1
        ctx = rng.choice(rated, k, replace=False)
        ctxs.append(ctx); L = max(L, k)
    ci = np.zeros((B, L), np.int64); cp = np.zeros((B, L), np.int64); cm = np.ones((B, L), bool)
    pos_mask = np.zeros((B, nt), np.float32); seen = np.zeros((B, nt), bool)
    bpr_l = np.zeros(B, np.int64); bpr_d = np.zeros(B, np.int64); bpr_ok = np.zeros(B, np.float32)
    for b, (prof, ctx) in enumerate(zip(users, ctxs)):
        for j, e in enumerate(ctx):
            ci[b, j] = e; cp[b, j] = int(prof[e] >= 0.5); cm[b, j] = False
            if e < nt: seen[b, e] = True
        liked = np.where(prof[:nt] == 1)[0]
        held = [i for i in liked if not seen[b, i]]    # held-out liked = positives
        for i in held: pos_mask[b, i] = 1.0
        # BPR pair: held-out liked vs held-out disliked (artifact-targeting)
        disl_held = [i for i in np.where(prof[:nt] == 0)[0] if not seen[b, i]]
        if held and disl_held:
            bpr_l[b] = rng.choice(held); bpr_d[b] = rng.choice(disl_held); bpr_ok[b] = 1.0
    return (torch.from_numpy(ci), torch.from_numpy(cp), torch.from_numpy(cm),
            torch.from_numpy(pos_mask), torch.from_numpy(seen),
            torch.from_numpy(bpr_l), torch.from_numpy(bpr_d), torch.from_numpy(bpr_ok))


def train_model():
    model = TwoTower(ni, D)
    opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-6)
    tr = train[rng.choice(len(train), min(N_TRAIN, len(train)), replace=False)]
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(len(tr)); tot = 0; nb = 0
        for s in range(0, len(perm), BS):
            users = [tr[i] for i in perm[s:s + BS]]
            ci, cp, cm, pos, seen = make_batch(users)
            opt.zero_grad()
            logits, u = model(ci, cp, cm)
            mlog = logits[:, :nt] - logpop_movies.unsqueeze(0)     # logQ correction
            mlog = mlog.masked_fill(seen, -1e9)                    # don't score revealed
            logp = F.log_softmax(mlog, dim=1)
            npos = pos.sum(1).clamp(min=1)
            loss = -(logp * pos).sum(1) / npos                     # multi-positive softmax
            loss = loss[pos.sum(1) > 0].mean()
            # answerability aux
            ratedm = torch.from_numpy(np.stack([(~np.isnan(p)).astype(np.float32) for p in users]))
            loss = loss + 0.3 * F.binary_cross_entropy_with_logits(model.rated_head(u), ratedm)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if (ep + 1) % 5 == 0:
            print(f"  ep{ep+1}/{EPOCHS} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({'state': model.state_dict(), 'd': D, 'n_items': ni, 'n_movies': nt}, OUT)
    return model


class TTWrapper:
    def __init__(self, model): self.m = model; self.m.eval(); self.n_movies = nt
    def predict_full(self, revealed):
        if revealed:
            ci = torch.tensor([[e for e, _ in revealed]]); cp = torch.tensor([[int(p >= .5) for _, p in revealed]])
            cm = torch.zeros_like(ci, dtype=torch.bool)
        else:
            ci = torch.zeros(1, 1, dtype=torch.long); cp = torch.zeros(1, 1, dtype=torch.long); cm = torch.ones(1, 1, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = self.m(ci, cp, cm)
        return logits[0].numpy()
    def predict(self, revealed): return self.predict_full(revealed)[:nt]


# ---------------- evaluation: artifact control + ranking ----------------
def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    alls = np.concatenate([pos, neg]); order = np.argsort(alls, kind='mergesort'); sa = alls[order]
    ranks = np.empty(len(alls)); i = 0
    while i < len(alls):
        j = i
        while j + 1 < len(alls) and sa[j + 1] == sa[i]: j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0; i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def build_cases(n=150, hold=0.40):
    cases = []
    for prof in test:
        rated = np.where(~np.isnan(prof[:nt]))[0]
        if len(rated) < 8: continue
        if (prof[rated] == 1).sum() < 4 or (prof[rated] == 0).sum() < 3: continue
        perm = rng.permutation(rated); k = max(2, int(hold * len(rated)))
        H = perm[:k]; askable = set(int(x) for x in perm[k:])
        Hl = [int(x) for x in H if prof[x] == 1]; Hd = [int(x) for x in H if prof[x] == 0]
        if len(Hl) < 1 or len(Hd) < 1: continue
        cases.append((prof, askable, Hl, Hd))
        if len(cases) >= n: break
    return cases


def evaluate(rec, name, cases):
    ld = []; lik_t10 = []; dis_t10 = []; ndcg = []
    for (prof, askable, Hl, Hd) in cases:
        rated_set = set(int(x) for x in np.where(~np.isnan(prof[:nt]))[0])
        rev = [(int(e), float(prof[e])) for e in askable if e < nt]
        bel = np.asarray(rec.predict_full(rev))[:nt]
        negs = np.array([m for m in range(nt) if m not in rated_set])
        a = auc(np.array([bel[i] for i in Hl] + [bel[i] for i in Hd]), np.array([1]*len(Hl)+[0]*len(Hd)))
        if not np.isnan(a): ld.append(a)
        for h in Hl:
            r = 1 + int((bel[negs] >= bel[h]).sum()); lik_t10.append(1.0 if r <= 10 else 0.0)
            ndcg.append(1.0/np.log2(r+1) if r <= 10 else 0.0)
        for hd in Hd:
            r = 1 + int((bel[negs] >= bel[hd]).sum()); dis_t10.append(1.0 if r <= 10 else 0.0)
    print(f"  {name:<24} liked-vs-disl AUC={np.mean(ld):.4f} | NDCG@10={np.mean(ndcg):.4f} | "
          f"liked top10={np.mean(lik_t10):.3f}  DISLIKED top10={np.mean(dis_t10):.3f} (want LOW)", flush=True)


if __name__ == '__main__':
    print(f"two-tower instrument: {ni} items, d={D}, train {N_TRAIN} users", flush=True)
    model = train_model()
    cases = build_cases()
    print(f"\n== artifact control + ranking ({len(cases)} users) ==", flush=True)
    evaluate(TTWrapper(model), "two-tower (NEW)", cases)
    try:
        old, _ = load_instrument_by_name('instrument_ml_stratified_rankcal')
        evaluate(old, "old instrument", cases)
    except Exception as e:
        print("  (old instrument skipped:", e, ")")
