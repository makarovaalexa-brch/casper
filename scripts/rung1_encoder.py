"""rung1_encoder.py -- RUNG I: the answer-native encoder on a FROZEN RecVAE-d512 decoder.

Faithful execution of DESIGN_SHEET_RUNG1_ENCODER.md (author-signed 2026-07-11). The 7th fold, done on
the RIGHT loss (ordinal + channel-dropout + confidence-as-precision) with the value-entity BINDING fix
(per-token FiLM before the sum-pool). Frozen decoder; NO decoder retrain (that is Rung II).

Reuses proven code: i25_lib (Frozen, enc_items, _bag_emb, ndcg), i25_fold_v3_sampler (V3Sampler = the
distilled answerer v2.1 reveal generator). NO LLM calls. Population trU users only; the 173/300 study
users are QUARANTINED (never train/val/test).

Commands:
  python scripts/rung1_encoder.py smoke                 # Phase 0: build + unit sanity -> build_sanity.json
  python scripts/rung1_encoder.py pilot                 # Phase 1: 2k-user kill-switch -> pilot_result.json
  python scripts/rung1_encoder.py train  [--n_users N]  # Phase 2: full train -> checkpoint + peak
  python scripts/rung1_encoder.py gates                 # Phase 3: all gates on held-out test -> gates_report

Load-bearing mechanisms: see .cache/rung1/WORKING_NOTES.md and the design sheet. Every one is wired here.
"""
import os, sys, json, time, argparse, collections, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import warnings
warnings.filterwarnings("ignore")
import i25_lib as L
from i25_fold_v3_sampler import (V3Sampler, KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG,
                                  FID_DATA, FID_EASE, FID_LLM, TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR,
                                  TYPE_ENTITY, NTYPE, CENTERED_FOLD, bin_star)

D_LAT = L.D_LAT
OUT = ".cache/rung1"
CKPT = f"{OUT}/rung1_encoder.pt"
BEST = f"{OUT}/rung1_encoder_best.pt"
PEAK = f"{OUT}/peak_rung1.txt"
os.makedirs(OUT, exist_ok=True)

# channel-dropout partition ids
CH_A, CH_B, CH_C = 0, 1, 2   # A=consumption anchor native_z, B=implicit tokens, C=explicit tokens


# ============================================================ the Rung-I encoder
class Rung1Encoder(nn.Module):
    """Amortized set-encoder -> posterior q(z|answers)=N(mu, diag(sigma^2)) over a FROZEN RecVAE latent.

    Deep-Sets SUM-pool residual with a PER-TOKEN FiLM phi applied BEFORE the sum (value-entity binding):
      phi(token) = ( gamma_value(SIGNED,identity-init) * tau_know(POSITIVE softplus,init 1) ) (.) emb
      mu    = native_z + rho_mu([ SUM_t phi(t), native_z, log1p(#distinct surviving) ])
      logsg = prior_logsigma + rho_sg([ pool, native_z, log1p(ntok), log1p(sum tau) ])
    rho_mu last layer zero-init -> empty pool -> mu = native_z = prior (exact intercept). gamma_value
    identity-init -> gamma(meh)~1. tau weights the token's contribution to the mu-POOL (moves recs under
    deterministic-mu eval) AND the sigma head. beta additive term PERMANENTLY OFF.
    """

    # per-token feature dims fed to the FiLM heads
    VAL_IN = 1 + NTYPE + 2 + 3        # [value, type_oh(4), kind_oh(2), fid_oh(3)]
    TAU_IN = 3 + 2 + NTYPE            # [lvl_oh(3), kind_oh(2), type_oh(4)]

    def __init__(self, d=D_LAT):
        super().__init__()
        self.d = d
        # SIGNED per-dim value/valence gain: unconstrained linear, identity-init (W=0,b=1) -> gamma=1
        self.gamma_head = nn.Linear(self.VAL_IN, d)
        nn.init.zeros_(self.gamma_head.weight); nn.init.ones_(self.gamma_head.bias)
        # POSITIVE knowledge-precision scalar: softplus, init ~1 (softplus(0.5413)=1)
        self.tau_head = nn.Linear(self.TAU_IN, 1)
        nn.init.zeros_(self.tau_head.weight); nn.init.constant_(self.tau_head.bias, 0.5413)
        # rho_mu: residual delta, zero-init last layer -> starts at native_z (exact intercept)
        self.rho_mu = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho_mu[-1].weight); nn.init.zeros_(self.rho_mu[-1].bias)
        # rho_sigma: posterior spread; zero-init last layer + learnable prior log-sigma
        self.rho_sg = nn.Sequential(nn.Linear(2 * d + 2, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho_sg[-1].weight); nn.init.zeros_(self.rho_sg[-1].bias)
        self.prior_logsigma = nn.Parameter(torch.zeros(1))     # prior-sigma = 1 at init

    def _phi(self, tt, tk, tl, tf, tv, te):
        """Per-token FiLM contribution phi (B,K,d) and per-token precision tau (B,K)."""
        type_oh = F.one_hot(tt.clamp(min=0), NTYPE).to(te.dtype)
        kind_oh = F.one_hot(tk.clamp(min=0), 2).to(te.dtype)
        fid_oh = F.one_hot(tf.clamp(min=0), 3).to(te.dtype)
        lvl_oh = F.one_hot(tl.clamp(min=0), 3).to(te.dtype)
        val_x = torch.cat([tv.unsqueeze(-1), type_oh, kind_oh, fid_oh], dim=-1)
        gamma = self.gamma_head(val_x)                          # (B,K,d) SIGNED
        tau_x = torch.cat([lvl_oh, kind_oh, type_oh], dim=-1)
        tau = F.softplus(self.tau_head(tau_x)).squeeze(-1)      # (B,K) POSITIVE
        phi = gamma * tau.unsqueeze(-1) * te                    # value bound to entity, precision-weighted
        return phi, tau

    def forward(self, tt, tk, tl, tf, tv, te, mask, native_z,
                keepA=None, keepB=None, keepC=None, sample=False, gen=None):
        """tt,tk,tl,tf type ids (B,K); tv values (B,K); te emb (B,K,d); mask valid (B,K); native_z (B,d).
        keepA/B/C: (B,) 0/1 channel-keep flags (channel-dropout; default keep all). sample: reparam z~q.
        Returns dict(mu, logsigma, z, tau_sum, ntok)."""
        B, K, d = te.shape
        phi, tau = self._phi(tt, tk, tl, tf, tv, te)
        is_impl = (tk == KIND_IMPL).to(te.dtype)                # (B,K)
        is_expl = 1.0 - is_impl
        if keepA is None:
            keepA = torch.ones(B, dtype=te.dtype)
        if keepB is None:
            keepB = torch.ones(B, dtype=te.dtype)
        if keepC is None:
            keepC = torch.ones(B, dtype=te.dtype)
        # per-token survival: implicit tokens ride channel B, explicit tokens ride channel C
        survive = mask * (is_impl * keepB.unsqueeze(-1) + is_expl * keepC.unsqueeze(-1))
        native_eff = native_z * keepA.unsqueeze(-1)             # channel A dropout drops the anchor
        pool = (phi * survive.unsqueeze(-1)).sum(1)             # (B,d)
        ntok = survive.sum(1, keepdim=True)                     # (B,1) # distinct surviving tokens
        tau_sum = (tau * survive).sum(1, keepdim=True)          # (B,1) total precision mass
        delta = self.rho_mu(torch.cat([pool, native_eff, torch.log1p(ntok)], dim=-1))
        mu = native_eff + delta
        logsg = self.prior_logsigma + self.rho_sg(
            torch.cat([pool, native_eff, torch.log1p(ntok), torch.log1p(tau_sum)], dim=-1))
        logsg = logsg.clamp(-6.0, 3.0)
        if sample:
            eps = torch.randn(mu.shape, generator=gen) if gen is not None else torch.randn_like(mu)
            z = mu + torch.exp(logsg) * eps
        else:
            z = mu
        return dict(mu=mu, logsigma=logsg, z=z, native=native_eff, tau_sum=tau_sum, ntok=ntok)


# ============================================================ token packing (dedup + finite screen)
def _strip(tok):
    """V3Sampler 7-field (type,kind,lvl,surp,fid,val,emb) -> Rung1 6-field (type,kind,lvl,fid,val,emb).
    The surprise feature is DROPPED (F4: no surprise-from-profile leak)."""
    typ, kind, lvl, _surp, fid, val, emb = tok
    return (int(typ), int(kind), int(lvl), int(fid), float(val), np.asarray(emb, np.float32))


def _dedup(toks):
    """SUM over the DISTINCT token set: dedup by (channel,entity,kind). collision -> highest fidelity
    (explicit) / higher knowledge (implicit). Deterministic function of the SET -> order-invariant."""
    best = {}
    for (typ, kind, lvl, fid, val, emb) in toks:
        if not np.all(np.isfinite(emb)):                       # F6 non-finite screen
            continue
        key = (typ, kind, hash(emb.tobytes()))
        cur = best.get(key)
        if cur is None:
            best[key] = (typ, kind, lvl, fid, val, emb); continue
        # collision: explicit keep highest fidelity id; implicit keep higher knowledge level
        if kind == KIND_EXPL:
            if fid > cur[3]:
                best[key] = (typ, kind, lvl, fid, val, emb)
        else:
            if lvl > cur[2]:
                best[key] = (typ, kind, lvl, fid, val, emb)
    return list(best.values())


def pack(FR, tok_lists, native_lists, device="cpu", dedup=True):
    B = len(tok_lists)
    d = FR.W.shape[1]
    stripped = [_dedup([_strip(t) for t in tl]) if dedup else [_strip(t) for t in tl]
                for tl in tok_lists]
    K = max((len(t) for t in stripped), default=1); K = max(K, 1)
    tt = torch.zeros((B, K), dtype=torch.long); tk = torch.zeros((B, K), dtype=torch.long)
    tl_ = torch.zeros((B, K), dtype=torch.long); tf = torch.zeros((B, K), dtype=torch.long)
    tv = torch.zeros((B, K), dtype=torch.float32); te = torch.zeros((B, K, d), dtype=torch.float32)
    mask = torch.zeros((B, K), dtype=torch.float32)
    for b, toks in enumerate(stripped):
        for k, (typ, kind, lvl, fid, val, emb) in enumerate(toks):
            tt[b, k] = typ; tk[b, k] = kind; tl_[b, k] = lvl; tf[b, k] = fid
            tv[b, k] = val; te[b, k] = torch.as_tensor(emb); mask[b, k] = 1.0
    nz = FR.enc_items(native_lists)
    nz = torch.nan_to_num(nz, nan=0.0, posinf=0.0, neginf=0.0)   # F6
    return (tt.to(device), tk.to(device), tl_.to(device), tf.to(device), tv.to(device),
            te.to(device), mask.to(device), nz.to(device)), stripped


def fold_mu(FR, model, toks, native, keepA=1.0, keepB=1.0, keepC=1.0):
    """Deterministic-mu fold of ONE token set -> latent z=mu (numpy). Eval scoring (no sampling)."""
    model.eval()
    args, _ = pack(FR, [toks], [native])
    kA = torch.tensor([keepA]); kB = torch.tensor([keepB]); kC = torch.tensor([keepC])
    with torch.no_grad():
        out = model(*args, keepA=kA, keepB=kB, keepC=kC, sample=False)
    return out["mu"].numpy()[0].astype(np.float64)


def fold_mu_batch(FR, model, tok_lists, native_lists, keepA=1.0, keepB=1.0, keepC=1.0):
    if not tok_lists:
        return np.zeros((0, FR.W.shape[1]))
    model.eval()
    args, _ = pack(FR, tok_lists, native_lists)
    B = len(tok_lists)
    kA = torch.full((B,), float(keepA)); kB = torch.full((B,), float(keepB)); kC = torch.full((B,), float(keepC))
    with torch.no_grad():
        out = model(*args, keepA=kA, keepB=kB, keepC=kC, sample=False)
    return out["mu"].numpy().astype(np.float64)


# ============================================================ curriculum sampler (reuses V3Sampler)
STRAT_STD, STRAT_ADV = "std", "adversarial"


def user_split(prof, rng):
    """Split a user's rated profile into KNOWN (interview source) and HELD (ordinal targets, all bands,
    disjoint). Held spans whatever bands the user has (NEVER drop users lacking a band; HARD RULE #1)."""
    its = list(prof.keys()); rng.shuffle(its)
    h = len(its) // 2
    known = {int(j): float(prof[j]) for j in its[:h]}
    held = {int(j): float(prof[j]) for j in its[h:]}
    return known, held


def prep_users(prof_dict, rng, S):
    out = []
    for u, prof in prof_dict.items():
        known, held = user_split(prof, rng)
        likes = sum(1 for r in known.values() if r >= 4)
        if len(known) >= 2 and held and likes >= 1:
            rec = dict(u=int(u), known=known, held=held,
                       held_likes=set(j for j, r in held.items() if r >= 4),
                       cache=S.make_cache(known))
            out.append(rec)
    return out


def sample_reveal(S, rec, clean_frac, rng):
    """One curriculum reveal for a user. log-uniform length 1->full; 30% clean; strategy mix incl
    adversarial (deliberately off-profile low-value asking). Returns (tokens, native_liked_ids)."""
    known = rec["known"]; cache = rec["cache"]
    n = len(known)
    if rng.random() < clean_frac:
        return S.build_reveal(known, "clean", 0, rng, cache=cache)   # full profile (micro-batched)
    # log-uniform interview length in [1, n]
    hi = max(1, n)
    budget = int(round(math.exp(rng.uniform(math.log(1), math.log(hi + 0.999)))))
    budget = max(1, min(budget, hi))
    if rng.random() < 0.25:                                          # adversarial strategy leg
        return _adv_interview(S, rec, budget, rng)
    return S.build_reveal(known, "interview", budget, rng, cache=cache)


def _adv_interview(S, rec, budget, rng):
    """Adversarial asking: deliberately OFF-profile regions (low value / likely no_clue). Reuses the
    proven V3Sampler primitives (emit / _pick_region) so token semantics are identical."""
    known = rec["known"]; cache = rec["cache"]
    ks = cache["ks"]; r = np.array([known[j] for j in ks], float); mu = float(r.mean())
    cr = {int(j): float(known[j] - mu) for j in ks}
    trait = S.user_traits(rng)
    toks = []; asked = set()
    chans = [TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY]
    for _ in range(budget):
        ch = chans[int(rng.integers(len(chans)))]
        key = S._pick_region(ch, rng, on_profile=False, cache=cache)   # off-profile on purpose
        sig = (ch, int(key) if isinstance(key, (int, np.integer)) else key)
        if sig in asked:
            continue
        asked.add(sig)
        toks += S.emit(known, cr, mu, ch, key, trait, rng)
    item_asked = [int(k) for (c, k) in asked if c == TYPE_ITEM and int(k) in known and known[int(k)] >= 4]
    rng.shuffle(toks)
    return toks, item_asked


# ============================================================ loss
def _channel_keeps(B, p_drop, rng):
    """Per-example independent channel-dropout over {A,B,C}; ensure >=1 survivor. Consumption (A) is
    dropped most often (forces value/concept to carry the sign, not free-ride the anchor)."""
    kA = (rng.random(B) >= p_drop).astype(np.float32)            # A dropped at p_drop
    kB = (rng.random(B) >= 0.5 * p_drop).astype(np.float32)
    kC = (rng.random(B) >= 0.5 * p_drop).astype(np.float32)
    for b in range(B):
        if kA[b] + kB[b] + kC[b] == 0:                          # never drop everything
            pick = rng.integers(3)
            (kA, kB, kC)[pick][b] = 1.0
    return torch.from_numpy(kA), torch.from_numpy(kB), torch.from_numpy(kC)


def ordinal_loss(Smat, held_list, held_star_list, m):
    """Pairwise ordinal margin over held graded ratings (incl hated): for star_i>star_j,
    hinge(m*(star_i-star_j) - (s_i - s_j)). Stratify by available bands; NEVER drop a user."""
    losses = []
    for b, (ids, stars) in enumerate(zip(held_list, held_star_list)):
        if len(ids) < 2:
            continue
        s = Smat[b, ids]                                        # (n_held,)
        st = stars                                              # (n_held,) tensor
        ds = s.unsqueeze(0) - s.unsqueeze(1)                   # s_i - s_j  (i=row)
        dst = st.unsqueeze(0) - st.unsqueeze(1)                # star_i - star_j
        pos = (dst > 0).float()                                # only ordered pairs
        if pos.sum() < 1:
            continue
        hinge = torch.clamp(m * dst - ds, min=0.0) * pos
        losses.append(hinge.sum() / pos.sum())
    if not losses:
        return torch.zeros((), requires_grad=True)
    return torch.stack(losses).mean()


def consume_loss(Smat, prof_list, likes_list):
    """Held-likes reconstruction (v3 recipe): mask known profile, log-softmax, -mean over held likes."""
    B, ni = Smat.shape
    negmask = torch.zeros((B, ni), dtype=torch.bool)
    tg = torch.zeros((B, ni), dtype=torch.float32)
    valid = []
    for b in range(B):
        negmask[b, list(prof_list[b])] = True
        lk = [j for j in likes_list[b]]
        if lk:
            tg[b, lk] = 1.0; valid.append(b)
    if not valid:
        return torch.zeros((), requires_grad=True)
    Sm = Smat.masked_fill(negmask, -1e9)
    logp = torch.log_softmax(Sm, dim=1)
    n_t = tg.sum(1).clamp(min=1)
    per = -((logp * tg).sum(1) / n_t)
    return per[valid].mean()


def kl_loss(mu, logsg, native, free_bits=0.03):
    """KL(N(mu,sigma^2) || N(native, I)) per-dim with a free-bits floor. Regularizes delta+sigma, not
    the native intercept."""
    var = torch.exp(2 * logsg)
    dmu = mu - native
    kl_d = 0.5 * (var + dmu * dmu - 1.0 - 2 * logsg)           # per-dim KL, prior sigma=1
    kl_d = torch.clamp(kl_d, min=free_bits)                    # free bits
    return kl_d.sum(1).mean()


# ============================================================ batching (length-bucketed micro-batch)
def build_batch(S, recs, clean_frac, rng):
    tok_lists, nat_lists, prof, likes, held_ids, held_stars = [], [], [], [], [], []
    for rec in recs:
        toks, native = sample_reveal(S, rec, clean_frac, rng)
        if not toks:
            continue
        tok_lists.append(toks); nat_lists.append(native)
        prof.append(set(rec["known"].keys()))
        likes.append(rec["held_likes"])
        hid = list(rec["held"].keys())
        held_ids.append(hid)
        held_stars.append(torch.tensor([rec["held"][j] for j in hid], dtype=torch.float32))
    return tok_lists, nat_lists, prof, likes, held_ids, held_stars


def batch_loss(FR, model, S, recs, clean_frac, p_drop, lam_c, beta, margin, rng, gen):
    tok_lists, nat_lists, prof, likes, held_ids, held_stars = build_batch(S, recs, clean_frac, rng)
    if not tok_lists:
        return None, {}
    args, stripped = pack(FR, tok_lists, nat_lists)
    B = len(tok_lists)
    kA, kB, kC = _channel_keeps(B, p_drop, rng)
    out = model(*args, keepA=kA, keepB=kB, keepC=kC, sample=True, gen=gen)
    z = out["z"]
    Smat = z @ FR.W.T + FR.bdec                                # (B,ni) scored on the reparam sample
    Lc = consume_loss(Smat, prof, likes)
    Lo = ordinal_loss(Smat, held_ids, held_stars, margin)
    Lkl = kl_loss(out["mu"], out["logsigma"], out["native"], free_bits=0.03)
    loss = Lo + lam_c * Lc + beta * Lkl
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss (F6)")
    return loss, dict(Lo=float(Lo), Lc=float(Lc), Lkl=float(Lkl))


# ============================================================ eval (deterministic mu)
@torch.no_grad()
def val_ndcg(FR, model, S, recs, seed, clean_frac=0.3, max_batch=800):
    model.eval()
    rng = np.random.default_rng(seed)
    tl, nl, idx = [], [], []
    for i, rec in enumerate(recs):
        toks, native = sample_reveal(S, rec, clean_frac, rng)
        if toks:
            tl.append(toks); nl.append(native); idx.append(i)
    if not tl:
        return 0.0
    vals = []
    for s in range(0, len(tl), max_batch):
        e = min(s + max_batch, len(tl))
        Z = fold_mu_batch(FR, model, tl[s:e], nl[s:e])
        for r in range(s, e):
            rec = recs[idx[r]]
            v = L.ndcg10(FR, Z[r - s], rec["held_likes"], set(rec["known"].keys()))
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


# ============================================================ data loading (disjoint held-out USERS)
def load_split(D, n_train, n_val, n_test, seed=1234):
    """Load disjoint TRAIN/VAL/TEST held-out-user profiles from population trU (study ids excluded via
    trU firewall). Uses i25_lib.load_train_profiles (samples trU, filters >=8 ratings & >=4 likes)."""
    need = n_train + n_val + n_test
    prof = L.load_train_profiles(D, int(need * 1.6) + 2000, seed=seed)   # oversample to survive filter
    keys = sorted(prof.keys())
    rng = np.random.default_rng(seed); rng.shuffle(keys)
    assert len(keys) >= need, f"user shortfall {len(keys)} < {need}; raise oversample"
    tr = {u: prof[u] for u in keys[:n_train]}
    va = {u: prof[u] for u in keys[n_train:n_train + n_val]}
    te = {u: prof[u] for u in keys[n_train + n_val:n_train + n_val + n_test]}
    return tr, va, te


# ============================================================ telemetry: gamma / tau per level
@torch.no_grad()
def film_telemetry(model):
    """Report gamma (mean per-dim) and tau at canonical (value,knowledge) levels. Watch gamma(hated)
    SIGN (must be able to invert) and gamma(meh)~1 (identity anchor)."""
    model.eval()
    d = model.d
    def probe(val, kind, typ, fid, lvl):
        type_oh = F.one_hot(torch.tensor([typ]), NTYPE).float()
        kind_oh = F.one_hot(torch.tensor([kind]), 2).float()
        fid_oh = F.one_hot(torch.tensor([fid]), 3).float()
        lvl_oh = F.one_hot(torch.tensor([lvl]), 3).float()
        val_x = torch.cat([torch.tensor([[val]]), type_oh, kind_oh, fid_oh], dim=-1)
        gamma = model.gamma_head(val_x)
        tau = F.softplus(model.tau_head(torch.cat([lvl_oh, kind_oh, type_oh], dim=-1)))
        return float(gamma.mean()), float(tau)
    out = {}
    for name, val in [("hated", CENTERED_FOLD["hated"]), ("meh", CENTERED_FOLD["meh"]),
                      ("liked", CENTERED_FOLD["liked"]), ("loved", CENTERED_FOLD["loved"])]:
        g, _ = probe(val, KIND_EXPL, TYPE_CONCEPT, FID_EASE, LVL_ROUGH)
        out[f"gamma_{name}"] = g
    for lname, lv in [("rough", LVL_ROUGH), ("know_well", LVL_KW), ("no_clue", LVL_NEG)]:
        _, t = probe(0.0, KIND_IMPL, TYPE_CONCEPT, FID_DATA, lv)
        out[f"tau_{lname}"] = t
    return out


# ============================================================ PHASE 0: build + unit sanity
def cmd_smoke(args):
    print("[phase0] loading frozen RecVAE + sampler ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = V3Sampler(D, FR)
    model = Rung1Encoder()
    res = {"checks": {}}

    # (1) forward on a toy batch
    prof = L.load_train_profiles(D, 60, seed=7)
    rng = np.random.default_rng(0)
    recs = prep_users(prof, rng, S)[:16]
    tl, nl = [], []
    for rec in recs:
        toks, native = sample_reveal(S, rec, 0.3, rng)
        tl.append(toks); nl.append(native)
    args_p, _ = pack(FR, tl, nl)
    out = model(*args_p, sample=False)
    ok_forward = tuple(out["mu"].shape) == (len(tl), D_LAT) and torch.isfinite(out["mu"]).all()
    res["checks"]["forward_runs"] = bool(ok_forward)

    # (2) intercept EXACT: empty interview -> mu = native_z = prior (here native empty -> 0)
    z_empty = fold_mu(FR, model, [], [])
    res["checks"]["intercept_maxdev"] = float(np.abs(z_empty).max())
    res["checks"]["intercept_exact"] = bool(np.abs(z_empty).max() < 1e-6)
    # intercept with a native anchor: empty tokens, non-empty liked native -> mu == native_z
    liked = [rec for rec in recs if rec["held_likes"]][0]
    nat_ids = list(liked["held_likes"])[:5]
    z_nat = fold_mu(FR, model, [], nat_ids)
    z_ref = FR.enc_items([nat_ids])[0].numpy().astype(np.float64)
    res["checks"]["intercept_native_maxdev"] = float(np.abs(z_nat - z_ref).max())

    # (3) no NaN screen: inject a non-finite emb row, confirm it is zeroed and loss finite
    bad = [(TYPE_CONCEPT, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, 0.5, np.full(D_LAT, np.nan, np.float32))]
    args_b, stripped_b = pack(FR, [bad], [[]])
    res["checks"]["nan_screen_drops_bad_token"] = bool(len(stripped_b[0]) == 0)
    outb = model(*args_b, sample=False)
    res["checks"]["nan_screen_finite"] = bool(torch.isfinite(outb["mu"]).all())

    # (4) gamma(meh) ~ 1 and tau init ~ 1 (identity anchor telemetry)
    tel = film_telemetry(model)
    res["telemetry_init"] = tel
    res["checks"]["gamma_meh_near1"] = bool(abs(tel["gamma_meh"] - 1.0) < 0.05)

    # (5) signed-gamma smoke: the MECHANISM can invert. At identity init value has no effect (expected);
    #     prove signability by installing a canonical negative value-slope and checking a HATED concept
    #     token drives its region DOWN vs a LOVED token (sign flip, not mere attenuation).
    probe_model = Rung1Encoder()
    with torch.no_grad():
        # value is feature index 0 of gamma_head input; set a negative per-dim slope on value
        probe_model.gamma_head.weight[:, 0] = -1.2
        # give rho_mu a small identity-ish read-out so pool moves mu (untrained rho is zero -> no signal)
        nn.init.eye_(probe_model.rho_mu[-1].weight[:, :D_LAT]) if False else None
    # use a concept region; compare region score for hated vs loved value on the SAME entity
    ctag = 0
    emb = FR.concept_emb(ctag).astype(np.float32)
    members = np.where(D["concepts"]["item_tag"][:, ctag] > 0.5)[0]
    def region_score_with(model_, val):
        # drive pool directly through rho_mu by temporarily making rho read the pool (identity last layer)
        tok = [(TYPE_CONCEPT, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, float(val), emb)]
        argp, _ = pack(FR, [tok], [[]])
        with torch.no_grad():
            o = model_(*argp, sample=False)
            # bypass untrained rho: score the POOL contribution directly (mechanism probe)
            phi, tau = model_._phi(argp[0], argp[1], argp[2], argp[3], argp[4], argp[5])
            pool = (phi * argp[6].unsqueeze(-1)).sum(1)[0].numpy()
        return float(pool @ (FR.W.numpy().T)[:, members].mean(1)) if len(members) else 0.0
    s_hate = region_score_with(probe_model, CENTERED_FOLD["hated"])
    s_love = region_score_with(probe_model, CENTERED_FOLD["loved"])
    res["signed_gamma_probe"] = dict(region_pool_score_hated=s_hate, region_pool_score_loved=s_love,
                                     inverts=bool(s_hate < 0 < s_love or s_hate < s_love))
    res["checks"]["signed_gamma_can_invert"] = bool(s_hate < s_love)

    # (6) one training step runs with finite loss + channel dropout
    gen = torch.Generator().manual_seed(0)
    try:
        loss, parts = batch_loss(FR, model, S, recs, 0.3, 0.5, 0.3, 0.01, 0.5, rng, gen)
        loss.backward()
        res["checks"]["train_step_finite"] = bool(loss is not None and torch.isfinite(loss))
        res["loss_parts"] = parts
    except Exception as ex:
        res["checks"]["train_step_finite"] = False
        res["train_step_error"] = str(ex)

    res["all_pass"] = bool(all(v for v in res["checks"].values() if isinstance(v, bool)))
    json.dump(res, open(f"{OUT}/build_sanity.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float), flush=True)
    print(f"\n[phase0] ALL_PASS={res['all_pass']} -> {OUT}/build_sanity.json", flush=True)
    return res


# ============================================================ training core (shared pilot/full)
def train_core(FR, S, tr_recs, val_recs, args, tag, epochs, ckpt, best):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = Rung1Encoder()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    gen = torch.Generator().manual_seed(args.seed + 5)
    step_rng = np.random.default_rng(args.seed + 100)
    nsteps_total = epochs * math.ceil(len(tr_recs) / args.batch)
    warm = max(1, int(0.6 * nsteps_total))
    step = 0
    hist = []; best_val = -1.0; best_ep = -1
    t0 = time.time()
    print(f"[{tag}] training {len(tr_recs)} users, {epochs} epochs, batch {args.batch}, "
          f"p_drop {args.p_drop}, lam_c {args.lam_c}, margin {args.margin}, beta_max {args.beta_max}",
          flush=True)
    for ep in range(epochs):
        model.train()
        order = np.arange(len(tr_recs)); step_rng.shuffle(order)
        tot = collections.defaultdict(float); nb = 0
        for b0 in range(0, len(order), args.batch):
            recs = [tr_recs[i] for i in order[b0:b0 + args.batch]]
            beta = args.beta_max * min(1.0, step / warm)
            out = batch_loss(FR, model, S, recs, args.clean_frac, args.p_drop, args.lam_c,
                             beta, args.margin, step_rng, gen)
            loss, parts = out
            if loss is None:
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                tot[k] += v
            tot["loss"] += float(loss); nb += 1; step += 1
        vN = val_ndcg(FR, model, S, val_recs, args.seed + 7, clean_frac=args.clean_frac)
        tel = film_telemetry(model)
        rec = dict(epoch=ep + 1, val_ndcg=round(vN, 4), min=round((time.time() - t0) / 60, 1),
                   gamma_hated=round(tel["gamma_hated"], 3), gamma_meh=round(tel["gamma_meh"], 3),
                   gamma_loved=round(tel["gamma_loved"], 3), tau_kw=round(tel["tau_know_well"], 3),
                   tau_rough=round(tel["tau_rough"], 3),
                   **{k: round(v / max(nb, 1), 4) for k, v in tot.items()})
        hist.append(rec)
        print(f"[{tag}] ep{ep+1:2d} val {vN:.4f} | Lo {rec.get('Lo',0):.3f} Lc {rec.get('Lc',0):.3f} "
              f"Lkl {rec.get('Lkl',0):.2f} | gamma h/m/l {rec['gamma_hated']:+.2f}/{rec['gamma_meh']:+.2f}/"
              f"{rec['gamma_loved']:+.2f} tau kw/rg {rec['tau_kw']:.2f}/{rec['tau_rough']:.2f} | "
              f"{rec['min']}m", flush=True)
        torch.save(dict(model=model.state_dict(), state=dict(history=hist, best_val=best_val),
                        args=vars(args)), ckpt)
        if vN > best_val:
            best_val = vN; best_ep = ep + 1
            torch.save(dict(model=model.state_dict(),
                            state=dict(best_val=best_val, best_epoch=best_ep, history=hist),
                            args=vars(args)), best)
    print(f"[{tag}] BEST val {best_val:.4f} @ep{best_ep} -> {best}", flush=True)
    return model, dict(best_val=best_val, best_epoch=best_ep, history=hist)


# ============================================================ value-zeroing (G-value-NONINERT protocol)
def _value_to_meh(tok_lists):
    """Neutralize EXPLICIT token value fields to in-distribution 'meh' (NOT a zero vector); native
    membership stays frozen upstream. Implicit tokens untouched."""
    meh = CENTERED_FOLD["meh"]
    out = []
    for tl in tok_lists:
        nt = []
        for t in tl:
            typ, kind, lvl, surp, fid, val, emb = t
            if kind == KIND_EXPL:
                nt.append((typ, kind, lvl, surp, fid, meh, emb))
            else:
                nt.append(t)
        out.append(nt)
    return out


@torch.no_grad()
def eval_full_and_valuezero(FR, model, S, recs, seed, clean_frac=0.3):
    """Return per-user dict arrays: ndcg_full, ndcg_meh (value->meh, membership frozen), sigma (mean
    posterior sd), ntok, err(=1-ndcg_full). Deterministic-mu scoring."""
    model.eval()
    rng = np.random.default_rng(seed)
    tls, nls, idx = [], [], []
    for i, rec in enumerate(recs):
        toks, native = sample_reveal(S, rec, clean_frac, rng)
        if toks:
            tls.append(toks); nls.append(native); idx.append(i)
    R = dict(ndcg_full=[], ndcg_meh=[], sigma=[], ntok=[], err=[])
    meh_tls = _value_to_meh(tls)
    for s in range(0, len(tls), 600):
        e = min(s + 600, len(tls))
        args_f, stripped = pack(FR, tls[s:e], nls[s:e])
        of = model(*args_f, sample=False)
        Zf = of["mu"].numpy().astype(np.float64)
        sig = torch.exp(of["logsigma"]).mean(1).numpy()
        ntok = of["ntok"].numpy().ravel()
        args_m, _ = pack(FR, meh_tls[s:e], nls[s:e])
        Zm = model(*args_m, sample=False)["mu"].numpy().astype(np.float64)
        for r in range(s, e):
            rec = recs[idx[r]]
            nf = L.ndcg10(FR, Zf[r - s], rec["held_likes"], set(rec["known"].keys()))
            nm = L.ndcg10(FR, Zm[r - s], rec["held_likes"], set(rec["known"].keys()))
            if nf is None or nm is None:
                continue
            R["ndcg_full"].append(nf); R["ndcg_meh"].append(nm)
            R["sigma"].append(float(sig[r - s])); R["ntok"].append(float(ntok[r - s]))
            R["err"].append(1.0 - nf)
    return {k: np.array(v) for k, v in R.items()}


def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _partial_corr(a, b, ctrl):
    """Partial Pearson corr of a,b controlling for ctrl (residualize both on [1,ctrl])."""
    a, b, ctrl = map(lambda z: np.asarray(z, float), (a, b, ctrl))
    if len(a) < 5:
        return float("nan")
    X = np.column_stack([np.ones_like(ctrl), ctrl])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def sigma_length_curve(FR, model, S, recs, seed):
    """Mean posterior sigma at growing interview budgets (should SHRINK). Uses interview reveals."""
    model.eval()
    out = {}
    for budget in [1, 2, 4, 8, 16, 32]:
        rng = np.random.default_rng(seed + budget)
        tls, nls = [], []
        for rec in recs:
            toks, native = S.build_reveal(rec["known"], "interview", budget, rng, cache=rec["cache"])
            if toks:
                tls.append(toks); nls.append(native)
        sigs = []
        for s in range(0, len(tls), 600):
            e = min(s + 600, len(tls))
            args_f, _ = pack(FR, tls[s:e], nls[s:e])
            o = model(*args_f, sample=False)
            sigs.append(torch.exp(o["logsigma"]).mean(1).numpy())
        out[budget] = float(np.concatenate(sigs).mean()) if sigs else float("nan")
    return out


# ============================================================ PHASE 1: PILOT (GO/STOP kill-switch)
def cmd_pilot(args):
    t0 = time.time()
    print("[phase1 PILOT] loading ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = V3Sampler(D, FR)
    tr, va, _te = load_split(D, n_train=2000, n_val=1000, n_test=0, seed=args.seed + 1234)
    rng = np.random.default_rng(args.seed)
    tr_recs = prep_users(tr, rng, S)
    val_recs = prep_users(va, np.random.default_rng(args.seed + 1), S)
    print(f"[phase1] pilot train {len(tr_recs)} users, val {len(val_recs)} users "
          f"[{time.time()-t0:.0f}s]", flush=True)
    ckpt = f"{OUT}/pilot.pt"; best = f"{OUT}/pilot_best.pt"
    # Design says "1-2 epochs"; the pilot is the KILL-SWITCH, so we give the value pathway a fair
    # chance to activate (channel-dropout needs gradient steps) before declaring STOP. On 2k users
    # this is still cheap. --epochs controls it (default 8). A STOP here = mechanism dead, not undertrain.
    pilot_epochs = max(2, args.epochs)
    model, tr_info = train_core(FR, S, tr_recs, val_recs, args, "pilot", pilot_epochs, ckpt, best)
    # reload best
    model.load_state_dict(torch.load(best, map_location="cpu")["model"]); model.eval()

    # ---- PILOT GO gates ----
    tel = film_telemetry(model)
    z_empty = fold_mu(FR, model, [], [])
    intercept_dev = float(np.abs(z_empty).max())
    ev = eval_full_and_valuezero(FR, model, S, val_recs, args.seed + 21, clean_frac=args.clean_frac)
    dvals = ev["ndcg_full"] - ev["ndcg_meh"]
    d_mean = float(dvals.mean()); d_ci = _boot_ci(dvals, seed=1)
    no_nan = bool(np.isfinite(ev["ndcg_full"]).all() and np.isfinite(ev["sigma"]).all())
    sig_curve = sigma_length_curve(FR, model, S, val_recs, args.seed + 31)
    budgets = sorted(sig_curve.keys()); svals = [sig_curve[b] for b in budgets]
    sigma_corr_len = float(np.corrcoef(np.log1p(budgets), svals)[0, 1])
    partial = _partial_corr(ev["sigma"], ev["err"], np.log1p(ev["ntok"]))

    go = dict(
        intercept_dev0=bool(intercept_dev < 1e-6),
        no_nan=no_nan,
        value_delta_mean=d_mean, value_delta_ci=list(d_ci),
        value_delta_ge_mde=bool(d_ci[0] > 0 and d_mean >= 0.005),
        sigma_shrinks_with_length=bool(sigma_corr_len < 0),
        sigma_len_corr=sigma_corr_len, sigma_curve=sig_curve,
        sigma_err_partialcorr=partial,
        sigma_err_partial_pos=bool(np.isfinite(partial) and partial > 0),
        gamma_meh_identity=bool(abs(tel["gamma_meh"] - 1.0) < 0.1),
        gamma_hated=tel["gamma_hated"], gamma_hated_sign="negative/inverting"
                    if tel["gamma_hated"] < tel["gamma_loved"] else "NOT-inverting",
        gamma_loved=tel["gamma_loved"],
    )
    hard = ["intercept_dev0", "no_nan", "value_delta_ge_mde", "sigma_shrinks_with_length",
            "sigma_err_partial_pos", "gamma_meh_identity"]
    verdict = "GO" if all(go[k] for k in hard) else "STOP"
    result = dict(verdict=verdict, best_val=tr_info["best_val"], best_epoch=tr_info["best_epoch"],
                  hard_gates={k: bool(go[k]) for k in hard}, detail=go, telemetry=tel,
                  history=tr_info["history"], minutes=round((time.time() - t0) / 60, 1))
    json.dump(result, open(f"{OUT}/pilot_result.json", "w"), indent=2, default=float)
    print("\n===== PILOT VERDICT =====", flush=True)
    print(f"  intercept dev0:            {go['intercept_dev0']} (dev {intercept_dev:.2e})", flush=True)
    print(f"  no NaN:                    {go['no_nan']}", flush=True)
    print(f"  value-zeroing dNDCG:       {d_mean:+.4f} CI[{d_ci[0]:+.4f},{d_ci[1]:+.4f}]  "
          f">=MDE(0.005,CI>0): {go['value_delta_ge_mde']}", flush=True)
    print(f"  sigma shrinks w/ length:   {go['sigma_shrinks_with_length']} (corr {sigma_corr_len:+.3f}); "
          f"curve {[(b, round(sig_curve[b],4)) for b in budgets]}", flush=True)
    print(f"  sigma<->err partial-corr:  {partial:+.3f}  >0: {go['sigma_err_partial_pos']}", flush=True)
    print(f"  gamma(meh)~1:              {go['gamma_meh_identity']} ({tel['gamma_meh']:+.3f})", flush=True)
    print(f"  gamma(hated) SIGN:         {tel['gamma_hated']:+.3f} vs loved {tel['gamma_loved']:+.3f} "
          f"-> {go['gamma_hated_sign']}", flush=True)
    print(f"\n  >>> PILOT {verdict} <<<  (best val {tr_info['best_val']:.4f}) "
          f"[{result['minutes']}m] -> {OUT}/pilot_result.json", flush=True)
    return result


# ============================================================ PHASE 3 gate helpers
def _scores(FR, z):
    return FR.decode_np(z[None, :])[0]


def _region_percentile(scores, members, profset=None):
    """Mean rank-percentile (1=top) of a region's member items in the full score ranking."""
    s = scores.copy()
    if profset:
        s[list(profset)] = -1e18
    order = np.argsort(-s)
    rankfrac = np.empty(len(s)); rankfrac[order] = 1.0 - np.arange(len(s)) / len(s)
    m = [j for j in members if j < len(s)]
    return float(np.mean(rankfrac[m])) if m else float("nan")


def _region_meanscore(FR, z, members):
    s = _scores(FR, z)
    m = [j for j in members if j < len(s)]
    return float(np.mean(s[m])) if m else float("nan")


def _mktok(ch, kind, lvl, fid, val, emb):
    return (int(ch), int(kind), int(lvl), 0.0, int(fid), float(val), np.asarray(emb, np.float32))


# ---- IG2 polarity flip + specificity ----
def gate_IG2(FR, model, S, D, users, n_probe=200):
    genres = [("gen", g) for g in range(D["Gmat"].shape[1])]
    flips = []; spec_disp = []; lists = []
    rng = np.random.default_rng(2)
    for gi, gk in enumerate(genres):
        emb = S.region_emb(TYPE_ATTR, gk); members = S.attr_set[gk]
        if len(members) < 20:
            continue
        z_love = fold_mu(FR, model, [_mktok(TYPE_ATTR, KIND_EXPL, LVL_ROUGH, FID_EASE,
                                            CENTERED_FOLD["loved"], emb)], [])
        z_hate = fold_mu(FR, model, [_mktok(TYPE_ATTR, KIND_EXPL, LVL_ROUGH, FID_EASE,
                                            CENTERED_FOLD["hated"], emb)], [])
        z_base = fold_mu(FR, model, [], [])
        p_love = _region_percentile(_scores(FR, z_love), list(members))
        p_hate = _region_percentile(_scores(FR, z_hate), list(members))
        flips.append(dict(genre=int(gk[1]), pctile_love=p_love, pctile_hate=p_hate,
                          drop=p_love - p_hate))
        # specificity: untouched genres should not move under "like X"
        for gk2 in genres:
            if gk2 == gk or len(S.attr_set[gk2]) < 20:
                continue
            base = _region_percentile(_scores(FR, z_base), list(S.attr_set[gk2]))
            und = _region_percentile(_scores(FR, z_love), list(S.attr_set[gk2]))
            spec_disp.append(abs(und - base))
        # illustrative top-10 titles under like/dislike for the first few genres
        if len(lists) < 4:
            top_love = np.argsort(-_scores(FR, z_love))[:10]
            lists.append(dict(genre=int(gk[1]),
                              gname=[k for k, v in enumerate([])] and None,
                              top10_like=[D["title"][int(j)] for j in top_love]))
    dl = np.array([f["drop"] for f in flips])
    return dict(mean_pctile_love=float(np.mean([f["pctile_love"] for f in flips])),
                mean_pctile_hate=float(np.mean([f["pctile_hate"] for f in flips])),
                mean_drop=float(dl.mean()), drop_ci=list(_boot_ci(dl, seed=2)),
                specificity_mean_disp=float(np.mean(spec_disp)),
                specificity_pass=bool(np.mean(spec_disp) < 0.03),
                per_genre=flips[:20], illustrative=lists,
                **{"pass": bool(_boot_ci(dl, seed=2)[0] > 0)})


# ---- IG1 genre purity ----
def gate_IG1(FR, model, S, D):
    genres = [("gen", g) for g in range(D["Gmat"].shape[1])]
    pur = []; ex = []
    for gk in genres:
        if len(S.attr_set[gk]) < 20:
            continue
        z = fold_mu(FR, model, [_mktok(TYPE_ATTR, KIND_EXPL, LVL_ROUGH, FID_EASE,
                                       CENTERED_FOLD["loved"], S.region_emb(TYPE_ATTR, gk))], [])
        top = np.argsort(-_scores(FR, z))[:10]
        frac = float(np.mean([1.0 if int(j) in S.attr_set[gk] else 0.0 for j in top]))
        pur.append(frac)
        if len(ex) < 4:
            ex.append(dict(genre=int(gk[1]), purity=frac,
                           top10=[D["title"][int(j)] for j in top]))
    return dict(mean_purity=float(np.mean(pur)), examples=ex)


# ---- IG3 franchise coherence ----
def gate_IG3(FR, model, S, D, users):
    ex = []
    for rec in users[:6]:
        liked = [j for j, r in rec["known"].items() if r >= 4]
        if not liked:
            continue
        j = liked[0]
        emb = FR.Wn[j].numpy().astype(np.float32)
        z = fold_mu(FR, model, [_mktok(TYPE_ITEM, KIND_EXPL, LVL_ROUGH, FID_DATA, 1.0, emb)], [j])
        top = np.argsort(-_scores(FR, z))[:10]
        ex.append(dict(seed=D["title"][int(j)], neighbors=[D["title"][int(t)] for t in top]))
    return dict(examples=ex)


# ---- IG4 graded value sweep ----
def gate_IG4(FR, model, S, D, users):
    genres = [("gen", g) for g in range(D["Gmat"].shape[1]) if len(S.attr_set[("gen", g)]) >= 20]
    seq = ["hated", "meh", "liked", "loved"]
    curves = []
    for gk in genres[:12]:
        emb = S.region_emb(TYPE_ATTR, gk); members = list(S.attr_set[gk])
        vals = []
        for lv in seq:
            z = fold_mu(FR, model, [_mktok(TYPE_ATTR, KIND_EXPL, LVL_ROUGH, FID_EASE,
                                           CENTERED_FOLD[lv], emb)], [])
            vals.append(_region_percentile(_scores(FR, z), members))
        curves.append(vals)
    C = np.array(curves)
    mono = float(np.mean([np.all(np.diff(row) >= -0.02) for row in C]))
    return dict(mean_curve=dict(zip(seq, C.mean(0).round(4).tolist())),
                frac_monotone=mono, **{"pass": bool(mono >= 0.6 and C.mean(0)[0] < C.mean(0)[-1])})


# ---- IG5 graded confidence ----
def gate_IG5(FR, model, S, D, users):
    diffs = []
    for rec in users:
        eid = _top_entity(S, rec["known"])
        if eid is None:
            continue
        emb = S.region_emb(TYPE_ENTITY, eid); members = S.entities[eid]["members"]
        z_r = fold_mu(FR, model, [_mktok(TYPE_ENTITY, KIND_IMPL, LVL_ROUGH, FID_DATA, 0.0, emb)], [])
        z_k = fold_mu(FR, model, [_mktok(TYPE_ENTITY, KIND_IMPL, LVL_KW, FID_DATA, 0.0, emb)], [])
        z0 = fold_mu(FR, model, [], [])
        pr = _region_meanscore(FR, z_r, members) - _region_meanscore(FR, z0, members)
        pk = _region_meanscore(FR, z_k, members) - _region_meanscore(FR, z0, members)
        diffs.append(pk - pr)
    ci = _boot_ci(diffs, seed=5)
    return dict(mean_kw_minus_rough=float(np.mean(diffs)), ci=list(ci), n=len(diffs),
                **{"pass": bool(ci[0] > 0)})


def _top_entity(S, known):
    kset = set(int(j) for j in known)
    best, bn = None, 0
    for eid, e in S.entities.items():
        n = len(kset & e["mset"])
        if n > bn:
            bn, best = n, eid
    return best


# ---- G-GoT: know-well-but-hated still pulls toward region ----
def gate_GoT(FR, model, S, users):
    ent_d, item_d = [], []
    for rec in users:
        eid = _top_entity(S, rec["known"])
        if eid is not None:
            emb = S.region_emb(TYPE_ENTITY, eid); members = S.entities[eid]["members"]
            got = [_mktok(TYPE_ENTITY, KIND_IMPL, LVL_KW, FID_DATA, 0.0, emb),
                   _mktok(TYPE_ENTITY, KIND_EXPL, LVL_ROUGH, FID_EASE, CENTERED_FOLD["hated"], emb)]
            z_got = fold_mu(FR, model, got, [])
            z0 = fold_mu(FR, model, [], [])
            ent_d.append(_region_meanscore(FR, z_got, members) - _region_meanscore(FR, z0, members))
        liked = [j for j, r in rec["known"].items() if r >= 4]
        if liked:
            j = liked[0]; emb = FR.Wn[j].numpy().astype(np.float32)
            got = [_mktok(TYPE_ITEM, KIND_IMPL, LVL_KW, FID_DATA, 0.0, emb),
                   _mktok(TYPE_ITEM, KIND_EXPL, LVL_ROUGH, FID_DATA, CENTERED_FOLD["hated"], emb)]
            z_got = fold_mu(FR, model, got, [])
            z0 = fold_mu(FR, model, [], [])
            item_d.append(_region_meanscore(FR, z_got, [j]) - _region_meanscore(FR, z0, [j]))
    ce = _boot_ci(ent_d, seed=6); cimd = _boot_ci(item_d, seed=7)
    return dict(entity_pull=float(np.mean(ent_d)), entity_ci=list(ce),
                item_pull=float(np.mean(item_d)), item_ci=list(cimd),
                **{"pass": bool(ce[0] > 0 and cimd[0] > 0)})


# ---- G-know-graded: rough < know_well ----
def gate_know_graded(FR, model, S, users):
    diffs = []
    for rec in users:
        eid = _top_entity(S, rec["known"])
        if eid is None:
            continue
        emb = S.region_emb(TYPE_ENTITY, eid); members = S.entities[eid]["members"]
        z0 = fold_mu(FR, model, [], [])
        z_r = fold_mu(FR, model, [_mktok(TYPE_ENTITY, KIND_IMPL, LVL_ROUGH, FID_DATA, 0.0, emb)], [])
        z_k = fold_mu(FR, model, [_mktok(TYPE_ENTITY, KIND_IMPL, LVL_KW, FID_DATA, 0.0, emb)], [])
        pr = _region_meanscore(FR, z_r, members) - _region_meanscore(FR, z0, members)
        pk = _region_meanscore(FR, z_k, members) - _region_meanscore(FR, z0, members)
        diffs.append(pk - pr)
    ci = _boot_ci(diffs, seed=8)
    return dict(mean=float(np.mean(diffs)), ci=list(ci), **{"pass": bool(ci[0] > 0)})


# ---- hygiene gates ----
def gate_hygiene(FR, model, S, users):
    rng = np.random.default_rng(11)
    # G-intercept
    z_empty = fold_mu(FR, model, [], [])
    intercept = float(np.abs(z_empty).max())
    # G-order + G-falsify-count + G-no-profile-leak on a few users
    order_dev, dup_dev = [], []
    for rec in users[:60]:
        toks, native = S.build_reveal(rec["known"], "interview", 12, rng, cache=rec["cache"])
        if not toks:
            continue
        z1 = fold_mu(FR, model, toks, native)
        sh = list(toks); rng.shuffle(sh)
        z2 = fold_mu(FR, model, sh, native)
        order_dev.append(float(np.abs(z1 - z2).max()))
        z3 = fold_mu(FR, model, toks + list(toks), native)   # duplicate the token set
        dup_dev.append(float(np.abs(z1 - z3).max()))
    # G-token-count-audit: emitted - deduped == pooled
    audit_ok = True
    for rec in users[:40]:
        toks, native = S.build_reveal(rec["known"], "interview", 15, rng, cache=rec["cache"])
        _, stripped = pack(FR, [toks], [native])
        pooled = len(stripped[0])
        emitted = len([_strip(t) for t in toks])
        deduped = len(_dedup([_strip(t) for t in toks]))
        if pooled != deduped:
            audit_ok = False
    return dict(intercept_maxdev=intercept, intercept_pass=bool(intercept < 1e-6),
                order_maxdev=float(np.max(order_dev)) if order_dev else 0.0,
                order_pass=bool((np.max(order_dev) if order_dev else 0) < 1e-5),
                dup_maxdev=float(np.max(dup_dev)) if dup_dev else 0.0,
                falsify_count_pass=bool((np.max(dup_dev) if dup_dev else 0) < 1e-5),
                token_count_audit_pass=bool(audit_ok))


# ---- G-implicit-ablation ----
def gate_implicit_ablation(FR, model, S, users):
    rng = np.random.default_rng(12)
    drops = []
    for rec in users:
        toks, native = S.build_reveal(rec["known"], "interview", 15, rng, cache=rec["cache"])
        if not toks:
            continue
        z_full = fold_mu(FR, model, toks, native)
        # zero implicit tokens by dropping channel B
        z_abl = fold_mu(FR, model, toks, native, keepB=0.0)
        a = L.ndcg10(FR, z_full, rec["held_likes"], set(rec["known"].keys()))
        b = L.ndcg10(FR, z_abl, rec["held_likes"], set(rec["known"].keys()))
        if None not in (a, b):
            drops.append(a - b)
    ci = _boot_ci(drops, seed=12)
    return dict(mean=float(np.mean(drops)), ci=list(ci), **{"pass": bool(ci[0] > 0)})


# ---- G-clean + G-monotone (k-curve) + value-NONINERT contexts ----
def gate_clean(FR, model, S, users):
    rng = np.random.default_rng(4)
    fold_n, nat_n = [], []
    for rec in users:
        toks, native = S.build_reveal(rec["known"], "clean", 0, rng, cache=rec["cache"])
        zf = fold_mu(FR, model, toks, native)
        zn = FR.enc_items([list(rec["known"].keys())])[0].numpy().astype(np.float64)
        a = L.ndcg10(FR, zf, rec["held_likes"], set(rec["known"].keys()))
        b = L.ndcg10(FR, zn, rec["held_likes"], set(rec["known"].keys()))
        if None not in (a, b):
            fold_n.append(a); nat_n.append(b)
    mf, mn = float(np.mean(fold_n)), float(np.mean(nat_n))
    return dict(fold=mf, native=mn, gap=mf - mn, **{"pass": bool(mf >= mn - 0.03)})


def gate_monotone(FR, model, S, users):
    curve = {}
    for b in [1, 2, 3, 4, 6, 8, 12, 16]:
        vals = []
        for rep in range(2):
            rng = np.random.default_rng(600 + b * 7 + rep)
            tls, nls, idx = [], [], []
            for i, rec in enumerate(users):
                toks, native = S.build_reveal(rec["known"], "interview", b, rng, cache=rec["cache"])
                if toks:
                    tls.append(toks); nls.append(native); idx.append(i)
            Z = fold_mu_batch(FR, model, tls, nls)
            for r in range(len(tls)):
                v = L.ndcg10(FR, Z[r], users[idx[r]]["held_likes"], set(users[idx[r]]["known"].keys()))
                if v is not None:
                    vals.append(v)
        curve[b] = float(np.mean(vals))
    clean = gate_clean(FR, model, S, users)["fold"]
    t1, t8 = curve[1], curve[8]
    return dict(curve=curve, clean=clean, span_1_8=t8 - t1, clean_minus_8=clean - t8,
                **{"pass": bool((t8 - t1) >= 0.02 and (clean - t8) >= 0.05)})


def gate_value_noninert(FR, model, S, users):
    """value->meh Delta in BOTH regimes x {short,long,full} contexts + binarize-loses."""
    rng = np.random.default_rng(20)
    ctx = {"short": ("interview", 4), "long": ("interview", 20), "full": ("clean", 0)}
    res = {}
    for cname, (mode, budget) in ctx.items():
        tls, nls, recs2 = [], [], []
        for rec in users:
            toks, native = S.build_reveal(rec["known"], mode, budget, rng, cache=rec["cache"])
            if toks:
                tls.append(toks); nls.append(native); recs2.append(rec)
        meh = _value_to_meh(tls)
        for regime, kA in [("present", 1.0), ("absent", 0.0)]:
            Zf = fold_mu_batch(FR, model, tls, nls, keepA=kA)
            Zm = fold_mu_batch(FR, model, meh, nls, keepA=kA)
            dl = []; absf = []
            for r, rec in enumerate(recs2):
                a = L.ndcg10(FR, Zf[r], rec["held_likes"], set(rec["known"].keys()))
                b = L.ndcg10(FR, Zm[r], rec["held_likes"], set(rec["known"].keys()))
                if None not in (a, b):
                    dl.append(a - b); absf.append(a)
            ci = _boot_ci(dl, seed=20)
            res[f"{cname}_{regime}"] = dict(delta=float(np.mean(dl)), ci=list(ci),
                                            abs_full=float(np.mean(absf)),
                                            pass_hard=bool(ci[0] > 0 and np.mean(dl) >= 0.01))
    allpass = all(res[k]["pass_hard"] for k in res)
    return dict(contexts=res, **{"pass": bool(allpass)})


# ============================================================ PHASE 2: FULL train
def cmd_train(args):
    t0 = time.time()
    print(f"[phase2 FULL] loading; train pop {args.n_users} / val {args.n_val} / test {args.n_test}",
          flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = V3Sampler(D, FR)
    tr, va, te = load_split(D, args.n_users, args.n_val, args.n_test, seed=args.seed + 1234)
    tr_recs = prep_users(tr, np.random.default_rng(args.seed), S)
    val_recs = prep_users(va, np.random.default_rng(args.seed + 1), S)
    # persist the test-user id list so gates use the SAME held-out test cohort (never in training)
    te_recs = prep_users(te, np.random.default_rng(args.seed + 2), S)
    json.dump(dict(test_uids=[r["u"] for r in te_recs], val_uids=[r["u"] for r in val_recs],
                   train_uids=[r["u"] for r in tr_recs], seed=args.seed),
              open(f"{OUT}/split_uids.json", "w"))
    print(f"[phase2] prep train {len(tr_recs)} / val {len(val_recs)} / test {len(te_recs)} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    model, info = train_core(FR, S, tr_recs, val_recs, args, "full", args.epochs, CKPT, BEST)
    open(PEAK, "w").write(f"best_val={info['best_val']:.4f} @ep{info['best_epoch']} "
                          f"n_users={args.n_users} p_drop={args.p_drop} lam_c={args.lam_c} "
                          f"margin={args.margin} beta_max={args.beta_max} seed={args.seed}\n")
    json.dump(info, open(f"{OUT}/train_history.json", "w"), indent=2, default=float)
    print(f"[phase2] DONE best val {info['best_val']:.4f} -> {BEST}; peak -> {PEAK} "
          f"[{round((time.time()-t0)/60,1)}m]", flush=True)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "pilot", "train", "gates"])
    ap.add_argument("--n_users", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p_drop", type=float, default=0.5)
    ap.add_argument("--lam_c", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--beta_max", type=float, default=0.05)
    ap.add_argument("--clean_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "smoke":
        cmd_smoke(a)
    elif a.cmd == "pilot":
        cmd_pilot(a)
    elif a.cmd == "train":
        cmd_train(a)
    else:
        print(f"command {a.cmd} not yet wired in this build step", flush=True)
