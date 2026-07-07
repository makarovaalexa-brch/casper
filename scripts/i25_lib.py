"""i25_lib.py -- shared machinery for I2.5: the LEARNED FOLD over the frozen RecVAE-d512 instrument.

The I2.5 fold q(z | answer set) is a permutation-invariant Deep-Sets set encoder mapping heterogeneous
answer tokens -> RecVAE's 512-d latent z; the RecVAE decoder/scorer stays FROZEN (score = z @ W.T + bdec).
Token vocabulary (data-side answer values, non-circular by construction):
  - item   +- real centered rating   (entity emb = decoder row Wn[j])
  - concept+- graded member-aggregate (entity emb = RecVAE member-bag encoding; value = relevance-weighted
             mean centered rating over the user's revealed member items)
  - attribute (decade / genre) +- graded member-aggregate (NEW channel; entity emb = member-bag encoding;
             value = mean centered rating over the user's revealed member items of that attribute)
Refusals are EXCLUDED from the fold (they inform answerability, not taste).

Reuses the certified gate machinery (scripts/llm_answerability_gate.py) for data, the pinned seed-123
answerer split, the RecVAE checkpoint, concept directions and the NDCG@10 metric. NO LLM calls.
NEVER modifies canonical scripts or caches. All new checkpoints -> .cache/i25_*.

Lit anchors: Partial-VAE/EDDI (Ma 2019, amortized posterior from partial sets), Deep Sets (Zaheer 2017),
DropoutNet (Volkovs 2017, input-dropout cold-start curriculum). See I25_FOLD_RESULTS.md.
"""
import os, sys, json, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "instrument2"))
sys.path.insert(0, _HERE)
import warnings
warnings.filterwarnings("ignore")

import llm_answerability_gate as G          # canonical gate machinery (unmodified)

META_FULL = "data/movielens/.cache/ml25m/meta.npz"     # full ratings (train-user profiles)
RECVAE_CKPT = G.RECVAE_CKPT
GATE_GRID = G.GRID_CACHE
D_LAT = 512
MBAG = 50                                    # member-bag size for concept/attribute directions (gate convention)
SIGMA_STAR = 0.70                            # fidelity noise (stars), study-design sigma
CKPT_DEFAULT = ".cache/i25_fold_best.pt"

# decade buckets (attribute channel) + the 20 genres from the gate
DECADES = list(range(1920, 2030, 10))
GENRES = G.GENRES


# ======================================================================== frozen RecVAE
class Frozen:
    """Frozen RecVAE-d512: decoder matrix W/bdec, native encoder enc(multi-hot)->mu, and cached
    entity embeddings for items / concepts / attributes (all in the 512-d latent)."""
    def __init__(self, D):
        from recvae import RecVAE
        blob = torch.load(RECVAE_CKPT, map_location="cpu")
        a = blob["args"]
        self.model = RecVAE(a["hidden"], a["latent"], int(D["ni"]))
        self.model.load_state_dict(blob["model"]); self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ni = int(D["ni"])
        self.W = self.model.decoder.weight.detach().to(torch.float32)          # (ni, d)
        self.bdec = self.model.decoder.bias.detach().to(torch.float32)         # (ni,)
        Wn = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-9)
        self.Wn = Wn
        import ml25m_arena as A
        self.headmask = A.load_arena(seed=G.ANSWERER_SPLIT_SEED)["headmask"]
        self.D = D
        self.item_tag = D["concepts"]["item_tag"]
        self._cdir = {}
        self._adir = {}

    # --- native encoder on a binary multi-hot of item ids -> latent mu (RecVAE's own interface) ---
    def enc_items(self, item_lists):
        """item_lists: list of iterables of item ids. Returns (B,d) native RecVAE mu (no dropout).
        Empty rows -> zero latent (RecVAE's encoder divides by input norm, which is 0 for empties)."""
        B = len(item_lists)
        x = torch.zeros((B, self.ni), dtype=torch.float32)
        nonempty = []
        for b, it in enumerate(item_lists):
            it = [j for j in it]
            if it:
                x[b, it] = 1.0; nonempty.append(b)
        out = torch.zeros((B, self.W.shape[1]), dtype=torch.float32)
        if nonempty:
            idx = torch.tensor(nonempty)
            with torch.no_grad():
                mu, _ = self.model.encoder(x[idx], dropout_rate=0.0)
            out[idx] = mu.to(torch.float32)
        return out

    def native_fold_np(self, items):
        """RecVAE native fold of ONE item set -> latent (numpy)."""
        return self.enc_items([items]).numpy()[0].astype(np.float64)

    def decode_np(self, Z):
        Z = torch.as_tensor(np.atleast_2d(Z), dtype=torch.float32)
        with torch.no_grad():
            S = Z @ self.W.T + self.bdec
        return S.numpy()

    # --- member-bag entity embedding (concept_dir convention: top-M by relevance, RecVAE-encode) ---
    def _bag_emb(self, weights):
        top = np.argpartition(-weights, min(MBAG, len(weights) - 1))[:MBAG]
        b = np.zeros(self.ni, np.float32); b[top] = weights[top]
        s = b.sum()
        if s > 0:
            b /= s
        with torch.no_grad():
            mu, _ = self.model.encoder(torch.tensor(b[None, :]), dropout_rate=0.0)
        z = mu.numpy()[0].astype(np.float32)
        return z / (np.linalg.norm(z) + 1e-9)

    def concept_emb(self, ctag):
        if ctag not in self._cdir:
            self._cdir[ctag] = self._bag_emb(self.item_tag[:, ctag].astype(np.float64))
        return self._cdir[ctag]

    def attr_emb(self, akey):
        """akey = ('dec', decade) or ('gen', genre_index). Entity emb = member-bag encoding."""
        if akey not in self._adir:
            members = attr_members(self.D, akey)
            w = np.zeros(self.ni, np.float64)
            if len(members):
                w[members] = self.D["cnt"][members] + 1.0     # popular members define the direction
            self._adir[akey] = self._bag_emb(w)
        return self._adir[akey]


# ======================================================================== attribute channel (data-side)
def attr_members(D, akey):
    """Item ids that belong to an attribute. Decade: by title year; Genre: by genre matrix."""
    kind, val = akey
    if kind == "dec":
        yrs = np.array([_year(D, j) for j in range(int(D["ni"]))])
        return np.where((yrs >= val) & (yrs < val + 10))[0]
    else:
        return np.where(D["Gmat"][:, val] > 0)[0]


_YEAR_CACHE = {}
def _year(D, j):
    import answerability_main_study as MS
    if j not in _YEAR_CACHE:
        y = MS.item_year(D["title"][j])
        _YEAR_CACHE[j] = y if y else 0
    return _YEAR_CACHE[j]


def item_attrs(D, j):
    """Attributes of item j: its decade + each of its genres."""
    out = []
    y = _year(D, j)
    if y:
        out.append(("dec", (y // 10) * 10))
    for gi in np.where(D["Gmat"][j] > 0)[0]:
        out.append(("gen", int(gi)))
    return out


# ======================================================================== token construction
def build_tokens(D, FR, revealed, channels=("item", "concept", "attribute"),
                 max_concept=6, max_attr=6, noise_rng=None, sigma=0.0):
    """From a REVEALED {item_id: rating} dict, emit answer tokens (data-side values only).

    Returns list of (type_id, emb(512), value, key). type_id: 0=item,1=concept,2=attribute.
    Ratings are centered on the revealed-set mean; optional star-noise sigma added pre-centering.
    """
    if not revealed:
        return []
    its = list(revealed.keys())
    r = np.array([revealed[j] for j in its], float)
    if sigma > 0 and noise_rng is not None:
        r = r + sigma * noise_rng.standard_normal(len(r))
    mu = float(r.mean())
    cr = {j: r[k] - mu for k, j in enumerate(its)}           # centered rating per revealed item
    toks = []

    if "item" in channels:
        for j in its:
            toks.append((0, FR.Wn[j].numpy().astype(np.float32), float(cr[j]), f"I:{j}"))

    if "concept" in channels:
        # concept aggregate: relevance-weighted mean centered rating over revealed members
        it_arr = np.array(its)
        rel = D["concepts"]["item_tag"][it_arr]              # (n_rev, 200)
        mass = rel.sum(0)
        cand = np.argsort(-mass)
        picked = 0
        crv = np.array([cr[j] for j in its])
        for ctag in cand:
            m = float(mass[ctag])
            if m < 1e-6 or picked >= max_concept:
                if picked >= max_concept:
                    break
                continue
            val = float((rel[:, ctag] * crv).sum() / m)
            toks.append((1, FR.concept_emb(int(ctag)), val, f"C:{int(ctag)}"))
            picked += 1

    if "attribute" in channels:
        agg = collections.defaultdict(lambda: [0.0, 0.0])   # akey -> [sum_cr, n]
        for j in its:
            for ak in item_attrs(D, j):
                agg[ak][0] += cr[j]; agg[ak][1] += 1.0
        ordered = sorted(agg.items(), key=lambda kv: -kv[1][1])[:max_attr]
        for ak, (s, n) in ordered:
            val = s / n if n > 0 else 0.0
            kk = f"A:{ak[0]}:{ak[1]}"
            toks.append((2, FR.attr_emb(ak), float(val), kk))

    return toks


# ======================================================================== the learned fold (Deep Sets)
class Fold(nn.Module):
    """Permutation-invariant Deep-Sets encoder over heterogeneous answer tokens -> RecVAE latent.

    Per token input = [type_onehot(3), value(1), value*emb(512)]. phi -> sum-pool. rho maps
    [pool(512), native_item_z(512), log(1+ntok)(1)] -> z. The native_item_z (frozen RecVAE encoding
    of the revealed LIKED items) is a strong prior input so the item channel matches/《beats》 the base
    model's own fold (G-fold3/6 by construction), while concepts+attributes are folded on top."""
    def __init__(self, d=D_LAT):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3 + 1 + d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d))
        # RESIDUAL fold: z = native_z + delta; zero-init the delta head so the fold STARTS at the native
        # RecVAE item interface (G-fold3/6 pass by construction) and training only adds beneficial delta.
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)
        self.d = d

    def forward(self, tok_type, tok_val, tok_emb, mask, native_z):
        """tok_type (B,K) long; tok_val (B,K); tok_emb (B,K,d); mask (B,K) 1=valid; native_z (B,d).
        Returns z = native_z + delta(tokens)."""
        B, K, d = tok_emb.shape
        oneh = F.one_hot(tok_type.clamp(min=0), num_classes=3).to(tok_emb.dtype)  # (B,K,3)
        ve = tok_val.unsqueeze(-1) * tok_emb                                       # (B,K,d)
        x = torch.cat([oneh, tok_val.unsqueeze(-1), ve], dim=-1)                   # (B,K,3+1+d)
        h = self.phi(x) * mask.unsqueeze(-1)                                       # zero out pads
        pool = h.sum(1)                                                            # (B,d)
        ntok = mask.sum(1, keepdim=True)                                           # (B,1)
        delta = self.rho(torch.cat([pool, native_z, torch.log1p(ntok)], dim=-1))
        return native_z + delta


def pack_batch(FR, tokens_per_user, native_item_lists, device="cpu"):
    """tokens_per_user: list of token-lists; native_item_lists: liked item ids for the native prior."""
    B = len(tokens_per_user)
    K = max((len(t) for t in tokens_per_user), default=1)
    K = max(K, 1)
    tt = torch.zeros((B, K), dtype=torch.long)
    tv = torch.zeros((B, K), dtype=torch.float32)
    te = torch.zeros((B, K, FR.W.shape[1]), dtype=torch.float32)
    mask = torch.zeros((B, K), dtype=torch.float32)
    for b, toks in enumerate(tokens_per_user):
        for k, (typ, emb, val, _key) in enumerate(toks):
            tt[b, k] = typ; tv[b, k] = val
            te[b, k] = torch.as_tensor(emb)
            mask[b, k] = 1.0
    nz = FR.enc_items(native_item_lists)
    return tt.to(device), tv.to(device), te.to(device), mask.to(device), nz.to(device)


def fold_np(FR, model, tokens, native_items):
    """Fold ONE token set -> latent z (numpy). native_items = revealed LIKED ids for the prior path."""
    model.eval()
    tt, tv, te, mask, nz = pack_batch(FR, [tokens], [native_items])
    with torch.no_grad():
        z = model(tt, tv, te, mask, nz)
    return z.numpy()[0].astype(np.float64)


# ======================================================================== NDCG helper (arena metric)
def ndcg10(FR, z, tlike, profset, tail=False):
    S = FR.decode_np(z[None, :])[0]
    return G._ndcg_top10(S, tlike, profset, FR.headmask, tail)


# ======================================================================== train-user profiles
def load_train_profiles(D, n_users, seed=0):
    """Build rat_by_u for a SAMPLE of train users from the full meta ratings. Returns {u: {j: r}}."""
    d = np.load(META_FULL)
    uu, ii, rr = d["uu"], d["ii"], d["rr"]
    trU = set(d["trU"].astype(np.int64).tolist())
    rng = np.random.default_rng(seed)
    want = set(rng.choice(sorted(trU), size=min(n_users, len(trU)), replace=False).tolist())
    prof = collections.defaultdict(dict)
    sel = np.isin(uu, np.fromiter(want, int))
    for u, j, r in zip(uu[sel].tolist(), ii[sel].tolist(), rr[sel].tolist()):
        prof[u][j] = float(r)
    # keep only users with enough ratings to make a split with held-out likes
    return {u: v for u, v in prof.items() if len(v) >= 8 and sum(1 for r in v.values() if r >= 4) >= 4}
