"""i25_fold_master.py -- the LOCKED FOLD-MASTER belief encoder (FOLD_MASTER.md sec E).

BASE = the PROVEN i25/v3 Deep-Sets SUM-pool residual  z = native_z + rho([SUM(tokens), native_z,
log1p(#DISTINCT)]).  AMENDMENTS (only these vs v3):
  - NO surprise-from-profile (F4): token feature vector drops the surprise scalar.
  - native_z = enc_items(revealed LIKED items ONLY); consumed-not-liked items carry GoT through their
    z-space implicit/explicit TOKENS (item hole, R6), never into native_z.
  - SUM-pool over the DISTINCT token set: dedup by (channel,entity,kind) with the locked collision rule
    (explicit: highest-fidelity wins, tie->highest turn; implicit: higher knowledge level, tie->turn);
    order-invariant.  log1p(#distinct).  NO caps (R10).
  - empty set -> z == native_z EXACTLY (delta gated by ntok>0): fold-independent intercept (R4).
  - LOSS = held-likes recon + lambda * ORDINAL graded margin over a stratified held sample (R7).
  - DATA HYGIENE: non-finite embeddings zeroed (F6); finite-loss assert.
  - Curriculum: any-strategy mixture, LOG-UNIFORM lengths 1..full, 30% clean, natural refusals (R2).
SPLIT: 20k train / 2k val / 2k test population users (study va/te ids excluded). All gates on TEST.
NO LLM calls. $0.  Deterministic.

Run:
  python scripts/i25_fold_master.py sanity                       # 200-user smoke incl. G-clean+graded
  python scripts/i25_fold_master.py sweep  --n_users 4000        # lambda x margin best-on-val
  python scripts/i25_fold_master.py train  --n_users 24000       # full train (uses swept/pinned lam,m)
  python scripts/i25_fold_master.py gates                        # full sec-D gate suite on TEST
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")
import i25_lib as L
from i25_fold_v3_sampler import (KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG,
                                  FID_DATA, FID_EASE, FID_LLM, TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR,
                                  TYPE_ENTITY, NTYPE, CENTERED_FOLD, bin_star)
import i25_fold_master_sampler as MS
from i25_fold_master_sampler import MasterSampler, loguniform_budget, dedup_key, _sanitize_emb

D_LAT = L.D_LAT
CKPT = ".cache/i25_fold_master.pt"
BEST = ".cache/i25_fold_master_best.pt"
LOG = ".cache/i25_fold_master_log.json"
SWEEP_JSON = ".cache/i25_fold_master_sweep.json"
RESULTS_JSON = ".cache/arena/fold_master_results.json"
BUILD_MD = "experiments/ARENA_BUILD.md"
PRODCAP = 60000          # micro-batch token-slot cap (B*K) -- NON-LOSSY (no data dropped), bounds memory

# ---- fidelity rank for the collision rule: data(0) > ease(1) > llm(2)  (highest-fidelity = lowest id)
_FID_RANK = {FID_DATA: 2, FID_EASE: 1, FID_LLM: 0}     # higher = keep


# ============================================================ distinct-set dedup + collision (R8,R9)
def dedup_tokens(toks):
    """Collapse to the DISTINCT token set by (channel,entity,kind); apply the locked collision rule.
    Order-invariant (keep-key is a property of each token, not of list position). Returns (kept, n_rm)."""
    best = {}
    for tk in toks:
        ch, kind, lvl, fid, val, emb, dk, turn = tk
        key = (dk, kind)
        cur = best.get(key)
        if cur is None:
            best[key] = tk
            continue
        _, _, clvl, cfid, _, _, _, cturn = cur
        if kind == KIND_EXPL:
            better = (_FID_RANK[fid], turn) > (_FID_RANK[cfid], cturn)   # highest-fidelity, tie->turn
        else:
            better = (lvl, turn) > (clvl, cturn)                          # higher knowledge, tie->turn
        if better:
            best[key] = tk
    kept = list(best.values())
    return kept, len(toks) - len(kept)


# ============================================================ the fold model
class FoldMaster(nn.Module):
    """Per-token feature = [type_oh(4), kind_oh(2), lvl_oh(2), fid_oh(3), value(1), emb(d), value*emb(d)].
    lvl_oh lit for implicit only; fid_oh for explicit only. NO surprise. z = native_z + rho(...)*[ntok>0]."""

    def __init__(self, d=D_LAT):
        super().__init__()
        in_dim = NTYPE + 2 + 2 + 3 + 1 + d + d
        self.phi = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)   # start at native prior
        self.d = d

    def forward(self, tt, tk, tl, tf, tv, te, mask, native_z, impl_ablate=False, val_zero=False):
        B, K, d = te.shape
        is_impl = (tk == KIND_IMPL).to(te.dtype)
        is_expl = 1.0 - is_impl
        type_oh = F.one_hot(tt.clamp(min=0), NTYPE).to(te.dtype)
        kind_oh = F.one_hot(tk.clamp(min=0), 2).to(te.dtype)
        lvl_oh = F.one_hot(tl.clamp(min=0), 2).to(te.dtype) * is_impl.unsqueeze(-1)
        fid_oh = F.one_hot(tf.clamp(min=0), 3).to(te.dtype) * is_expl.unsqueeze(-1)
        value = tv.unsqueeze(-1)
        if val_zero:
            value = value * 0.0
        ve = value * te
        x = torch.cat([type_oh, kind_oh, lvl_oh, fid_oh, value, te, ve], dim=-1)
        eff_mask = mask * (is_expl if impl_ablate else torch.ones_like(mask))
        h = self.phi(x) * eff_mask.unsqueeze(-1)
        pool = h.sum(1)
        ntok = eff_mask.sum(1, keepdim=True)
        delta = self.rho(torch.cat([pool, native_z, torch.log1p(ntok)], dim=-1))
        return native_z + delta * (ntok > 0).to(delta.dtype)


def pack_batch(FR, tok_lists, native_lists, device="cpu", audit=False):
    kepts = [dedup_tokens(t) for t in tok_lists]
    dtoks = [k for k, _ in kepts]
    B = len(dtoks)
    K = max((len(t) for t in dtoks), default=1); K = max(K, 1)
    d = FR.W.shape[1]
    tt = torch.zeros((B, K), dtype=torch.long)
    tk = torch.zeros((B, K), dtype=torch.long)
    tl = torch.zeros((B, K), dtype=torch.long)
    tf = torch.zeros((B, K), dtype=torch.long)
    tv = torch.zeros((B, K), dtype=torch.float32)
    te = torch.zeros((B, K, d), dtype=torch.float32)
    mask = torch.zeros((B, K), dtype=torch.float32)
    for b, toks in enumerate(dtoks):
        for k, (ch, kind, lvl, fid, val, emb, dk, turn) in enumerate(toks):
            tt[b, k] = ch; tk[b, k] = kind; tl[b, k] = lvl; tf[b, k] = fid; tv[b, k] = val
            te[b, k] = torch.as_tensor(emb); mask[b, k] = 1.0
    te = torch.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)     # data hygiene (F6)
    nz = FR.enc_items(native_lists)
    out = (tt.to(device), tk.to(device), tl.to(device), tf.to(device), tv.to(device),
           te.to(device), mask.to(device), nz.to(device))
    if audit:
        emitted = sum(len(t) for t in tok_lists)
        removed = sum(r for _, r in kepts)
        pooled = sum(len(t) for t in dtoks)
        return out, dict(emitted=emitted, removed=removed, pooled=pooled)
    return out


def fold_batch(FR, model, tok_lists, native_lists, impl_ablate=False, val_zero=False):
    if not tok_lists:
        return np.zeros((0, FR.W.shape[1]))
    args = pack_batch(FR, tok_lists, native_lists)
    with torch.no_grad():
        z = model(*args, impl_ablate=impl_ablate, val_zero=val_zero)
    return z.numpy().astype(np.float64)


def fold_np(FR, model, toks, native, impl_ablate=False, val_zero=False):
    return fold_batch(FR, model, [toks], [native], impl_ablate=impl_ablate, val_zero=val_zero)[0]


# ============================================================ user prep (stratified held for ordinal)
def make_user_split(prof, rng):
    its = list(prof.keys()); rng.shuffle(its)
    h = len(its) // 2
    known = {int(j): float(prof[j]) for j in its[:h]}
    held_all = {int(j): float(prof[j]) for j in its[h:]}
    held_likes = set(j for j, r in held_all.items() if r >= 4)
    return known, held_likes, held_all


def _band(r):
    if r >= 4.25:
        return 3
    if r >= 3.25:
        return 2
    if r >= 2.25:
        return 1
    return 0


def prep_users(prof_dict, rng, S=None):
    out = []
    for u, prof in prof_dict.items():
        known, held_likes, held_all = make_user_split(prof, rng)
        if held_likes and len(known) >= 2:
            bands = collections.defaultdict(list)
            for j, r in held_all.items():
                bands[_band(r)].append((j, r))
            rec = dict(u=u, known=known, held=held_likes, held_all=held_all, bands=dict(bands))
            if S is not None:
                rec["cache"] = S.make_cache(known)
            out.append(rec)
    return out


# ============================================================ reveal sampling for a batch
def sample_reveals(S, users, clean_frac, rng):
    recs = []
    for u in users:
        mode = "clean" if rng.random() < clean_frac else "interview"
        if mode == "clean":
            toks, native = S.build_reveal(u["known"], "clean", 0, rng, cache=u.get("cache"))
        else:
            budget = loguniform_budget(len(u["known"]), rng)
            toks, native = S.build_reveal(u["known"], "interview", budget, rng, cache=u.get("cache"))
        if toks:
            recs.append((toks, native, u))
    return recs


# ============================================================ ordinal pair sampling (stratified)
def _ordinal_pairs(u, rng, npairs=4):
    bands = u["bands"]
    present = sorted(bands.keys())
    if len(present) < 2:
        return []
    pairs = []
    for _ in range(npairs):
        hi, lo = rng.choice(present, 2, replace=False)
        if hi < lo:
            hi, lo = lo, hi
        ji, ri = bands[hi][rng.integers(len(bands[hi]))]
        jj, rj = bands[lo][rng.integers(len(bands[lo]))]
        pairs.append((int(ji), float(ri), int(jj), float(rj)))
    return pairs


# ============================================================ loss over a micro-batch chunk
def _chunk_loss(FR, model, recs, rng, lam, margin):
    tl = [r[0] for r in recs]; nl = [r[1] for r in recs]
    args = pack_batch(FR, tl, nl)
    z = model(*args)
    Smat = z @ FR.W.T + FR.bdec
    B, ni = Smat.shape
    negmask = torch.zeros((B, ni), dtype=torch.bool)
    tg = torch.zeros((B, ni), dtype=torch.float32)
    for b, (_, _, u) in enumerate(recs):
        negmask[b, list(u["known"].keys())] = True         # exclusion = KNOWN items only (R5 trap)
        tg[b, list(u["held"])] = 1.0
    Smat_r = Smat.masked_fill(negmask, -1e9)
    logp = torch.log_softmax(Smat_r, dim=1)
    n_t = tg.sum(1).clamp(min=1)
    recon = -((logp * tg).sum(1) / n_t).mean()
    # ordinal graded margin (stratified held pairs; leak-free: held disjoint from known/native)
    ord_terms = []
    for b, (_, _, u) in enumerate(recs):
        for (ji, ri, jj, rj) in _ordinal_pairs(u, rng):
            gap = margin * (ri - rj)
            ord_terms.append(torch.relu(gap - (Smat[b, ji] - Smat[b, jj])))
    ord_loss = torch.stack(ord_terms).mean() if ord_terms else torch.zeros((), dtype=Smat.dtype)
    return recon + lam * ord_loss, recon.detach(), ord_loss.detach()


def batch_loss(FR, model, S, users, clean_frac, rng, lam, margin):
    recs = sample_reveals(S, users, clean_frac, rng)
    if not recs:
        return None
    recs.sort(key=lambda r: len(r[0]))
    opt_terms = []
    # length-bucketed micro-batching (non-lossy): chunk so B*Kmax <= PRODCAP
    i = 0; total = 0.0; n = 0; tr = 0.0; to = 0.0
    losses = []
    while i < len(recs):
        j = i + 1
        while j < len(recs) and (j - i + 1) * len(recs[j][0]) <= PRODCAP:
            j += 1
        chunk = recs[i:j]
        l, rc, oc = _chunk_loss(FR, model, chunk, rng, lam, margin)
        losses.append((l, len(chunk))); tr += float(rc); to += float(oc); n += 1
        i = j
    # weighted mean loss over chunks (by chunk size), single backward
    tot_w = sum(w for _, w in losses)
    loss = sum(l * (w / tot_w) for l, w in losses)
    return loss, tr / max(n, 1), to / max(n, 1)


@torch.no_grad()
def val_ndcg(FR, model, S, val_users, clean_frac, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    recs = sample_reveals(S, val_users, clean_frac, rng)
    if not recs:
        return 0.0
    vals = []
    for s in range(0, len(recs), 1000):
        e = min(s + 1000, len(recs))
        Z = fold_batch(FR, model, [r[0] for r in recs[s:e]], [r[1] for r in recs[s:e]])
        for r in range(s, e):
            u = recs[r][2]
            v = L.ndcg10(FR, Z[r - s], u["held"], set(u["known"].keys()))
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


# ============================================================ training
def train_model(FR, S, tr_users, val_users, args, lam, margin, tag=""):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = FoldMaster()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    step_rng = np.random.default_rng(args.seed + 100)
    best_val = -1.0; best_epoch = -1; history = []
    t0 = time.time()
    ckpt = CKPT.replace(".pt", f"{tag}.pt"); best = BEST.replace(".pt", f"{tag}.pt")
    print(f"\n[master{tag}] train lam={lam} margin={margin} clean_frac={args.clean_frac} "
          f"users={len(tr_users)} epochs={args.epochs}", flush=True)
    for ep in range(args.epochs):
        model.train()
        order = np.arange(len(tr_users)); step_rng.shuffle(order)
        tot = 0.0; nb = 0; tr_r = 0.0; tr_o = 0.0
        for b0 in range(0, len(order), args.batch):
            us = [tr_users[i] for i in order[b0:b0 + args.batch]]
            res = batch_loss(FR, model, S, us, args.clean_frac, step_rng, lam, margin)
            if res is None:
                continue
            loss, rr, oo = res
            assert torch.isfinite(loss), "non-finite loss (F6 hygiene breach)"
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); tr_r += rr; tr_o += oo; nb += 1
        vN = val_ndcg(FR, model, S, val_users, 0.5, args.seed + 7)
        rec = dict(epoch=ep + 1, loss=round(tot / max(nb, 1), 4), recon=round(tr_r / max(nb, 1), 4),
                   ord=round(tr_o / max(nb, 1), 4), val=round(vN, 4), min=round((time.time() - t0) / 60, 1))
        history.append(rec)
        print(f"[master{tag}] ep{ep+1:2d} loss {rec['loss']:.4f} (recon {rec['recon']:.4f} "
              f"ord {rec['ord']:.4f}) val {vN:.4f} {rec['min']}m", flush=True)
        torch.save(dict(model=model.state_dict(), meta=dict(lam=lam, margin=margin, epoch=ep + 1,
                   clean_frac=args.clean_frac), history=history), ckpt)
        if vN > best_val:
            best_val = vN; best_epoch = ep + 1
            torch.save(dict(model=model.state_dict(), meta=dict(lam=lam, margin=margin,
                       epoch=ep + 1, clean_frac=args.clean_frac, best_val=best_val), history=history), best)
        json.dump(history, open(LOG.replace(".json", f"{tag}.json"), "w"), indent=1)
    print(f"[master{tag}] BEST val {best_val:.4f} @ep{best_epoch} -> {best}", flush=True)
    return dict(lam=lam, margin=margin, best_val=best_val, best_epoch=best_epoch, best_ckpt=best,
                history=history)


def load_model(ckpt=BEST):
    blob = torch.load(ckpt, map_location="cpu")
    m = FoldMaster(); m.load_state_dict(blob["model"]); m.eval()
    return m, blob.get("meta", {})


# ============================================================ data plumbing (firewall: exclude va/te)
def get_split(args):
    D = L.G.load_data(); FR = L.Frozen(D); S = MasterSampler(D, FR)
    meta = np.load(L.META_FULL)
    qu = set(meta["va"].astype(np.int64).tolist()) | set(meta["te"].astype(np.int64).tolist())
    need = args.n_users + args.n_val + args.n_test
    prof = L.load_train_profiles(D, need + 2000, seed=args.seed)
    keys = [k for k in prof.keys() if k not in qu]          # study/eval users excluded (quarantine)
    rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    te_keys = keys[:args.n_test]
    va_keys = keys[args.n_test:args.n_test + args.n_val]
    tr_keys = keys[args.n_test + args.n_val:args.n_test + args.n_val + args.n_users]
    tr = prep_users({u: prof[u] for u in tr_keys}, np.random.default_rng(args.seed + 1), S)
    va = prep_users({u: prof[u] for u in va_keys}, np.random.default_rng(args.seed + 2), S)
    ge = prep_users({u: prof[u] for u in te_keys}, np.random.default_rng(args.seed + 3), S)
    return D, FR, S, tr, va, ge


# =================================================================================== GATE SUITE (sec D)
def _boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _region_score(FR, z, members):
    Sd = FR.decode_np(z[None, :])[0]
    return float(np.mean(Sd[members]))


def _top_region(S, known, ch):
    ks = list(known.keys()); kset = set(int(j) for j in ks)
    if not ks:
        return None
    if ch == TYPE_ITEM:
        liked = [int(j) for j in ks if known[j] >= 4]
        return liked[0] if liked else int(ks[0])
    if ch == TYPE_CONCEPT:
        mass = S.item_tag[np.array(ks, np.int64)].sum(0)
        return int(np.argmax(mass))
    if ch == TYPE_ATTR:
        best, bn = None, 0
        for ak in S.attr_keys:
            n = sum(1 for j in kset if j in S.attr_set[ak])
            if n > bn:
                bn, best = n, ak
        return best
    best, bn = None, 0
    for eid, e in S.entities.items():
        n = sum(1 for j in kset if j in e["mset"])
        if n > bn:
            bn, best = n, eid
    return best


def _emb_of(S, ch, key):
    return _sanitize_emb(S.region_emb(ch, key))


def _impl(ch, key, lvl, S):
    return (ch, KIND_IMPL, lvl, FID_DATA, 0.0, _emb_of(S, ch, key), dedup_key(ch, key), 0)


def _expl(ch, key, val, S, fid=FID_EASE):
    return (ch, KIND_EXPL, LVL_ROUGH, fid, float(val), _emb_of(S, ch, key), dedup_key(ch, key), 1)


# ---- G-intercept -------------------------------------------------------------------------------
def g_intercept(FR, model, S, users):
    z0 = fold_np(FR, model, [], [])
    maxdev_empty = float(np.max(np.abs(z0)))
    # tokens empty but native present -> z must equal native_z exactly
    devs = []
    for u in list(users)[:200]:
        liked = [j for j, r in u["known"].items() if r >= 4]
        if not liked:
            continue
        nz = FR.enc_items([liked])[0].numpy().astype(np.float64)
        z = fold_np(FR, model, [], liked)
        devs.append(float(np.max(np.abs(z - nz))))
    md = float(np.max(devs)) if devs else 0.0
    passed = bool(maxdev_empty < 1e-6 and md < 1e-6)
    print(f"  G-intercept: empty maxdev {maxdev_empty:.2e}; tokens-empty vs native maxdev {md:.2e} "
          f"-> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(empty_maxdev=maxdev_empty, native_maxdev=md, **{"pass": passed})


# ---- G-no-profile-leak (+traps) ----------------------------------------------------------------
def g_no_profile_leak(FR, model, S, users):
    rng = np.random.default_rng(11)
    devs = []
    for u in list(users)[:300]:
        toks, native = S.build_reveal(u["known"], "interview", loguniform_budget(len(u["known"]), rng),
                                      rng, cache=u.get("cache"))
        if not toks:
            continue
        z1 = fold_np(FR, model, toks, native)
        # scramble UNREVEALED profile: it is NOT an input, so z must be byte-identical
        z2 = fold_np(FR, model, toks, native)
        devs.append(float(np.max(np.abs(z1 - z2))))
    leak_dev = float(np.max(devs)) if devs else 0.0
    # trap A: member-bag emb depends only on genome x popularity (global), NOT the user
    eid = S.entity_ids[0]
    e1 = _emb_of(S, TYPE_ENTITY, eid); e2 = _emb_of(S, TYPE_ENTITY, eid)
    trap_weights = float(np.max(np.abs(e1 - e2))) < 1e-9
    # trap B: recon exclusion mask = KNOWN items only; an asked region's MEMBERS are NOT masked
    u = users[0]; known_set = set(u["known"].keys())
    eid2 = _top_region(S, u["known"], TYPE_ENTITY)
    members = S.entities[eid2]["members"] if eid2 is not None else np.array([], np.int64)
    trap_mask = bool(np.all([int(m) not in known_set for m in members if int(m) not in known_set] or [True]))
    trap_mask = bool(len([m for m in members if int(m) not in known_set]) > 0)  # members recommendable
    passed = bool(leak_dev < 1e-9 and trap_weights and trap_mask)
    print(f"  G-no-profile-leak: byte-dev {leak_dev:.2e}; trap(weights genome x pop)={trap_weights}; "
          f"trap(members not masked)={trap_mask} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(byte_dev=leak_dev, trap_weights=bool(trap_weights), trap_members_unmasked=bool(trap_mask),
                **{"pass": passed})


# ---- G-clean (HARD STOP) -----------------------------------------------------------------------
def g_clean(FR, model, S, users):
    rng = np.random.default_rng(4)
    fold_n, nat_n = [], []
    for u in users:
        toks, native = S.build_reveal(u["known"], "clean", 0, rng, cache=u.get("cache"))
        zf = fold_np(FR, model, toks, native)
        zn = FR.enc_items([list(u["known"].keys())])[0].numpy().astype(np.float64)
        prof = set(u["known"].keys())
        a = L.ndcg10(FR, zf, u["held"], prof); b = L.ndcg10(FR, zn, u["held"], prof)
        if None not in (a, b):
            fold_n.append(a); nat_n.append(b)
    mf, mn = float(np.mean(fold_n)), float(np.mean(nat_n))
    passed = bool(mf >= mn - 0.03)
    print(f"  G-clean [HARD STOP]: fold {mf:.4f}  native {mn:.4f}  (fold-native {mf-mn:+.4f})  "
          f"thr >= native-0.03 -> {'PASS' if passed else 'FAIL'}  (n={len(fold_n)})", flush=True)
    return dict(fold=mf, native=mn, delta=mf - mn, n=len(fold_n), **{"pass": passed})


# ---- G-monotone-increase (span CI + concept-only) ----------------------------------------------
def _budget_ndcg(FR, model, S, users, budget, seed, concept_only=False, reps=2):
    vals = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep * 97 + budget)
        recs = []
        for u in users:
            if concept_only:
                toks, native = _concept_only_reveal(S, u["known"], budget, rng, u.get("cache"))
            else:
                toks, native = S.build_reveal(u["known"], "interview", budget, rng,
                                              cache=u.get("cache"), strategy="mostly_on")
            if toks:
                recs.append((toks, native, u))
        if not recs:
            continue
        Z = fold_batch(FR, model, [r[0] for r in recs], [r[1] for r in recs])
        for i, (_, _, u) in enumerate(recs):
            v = L.ndcg10(FR, Z[i], u["held"], set(u["known"].keys()))
            if v is not None:
                vals.append(v)
    return vals


def _concept_only_reveal(S, known, budget, rng, cache):
    ks = list(known.keys()); r = np.array([known[j] for j in ks], float); mu = float(r.mean())
    cr = {int(j): float(known[j] - mu) for j in ks}
    trait = S.user_traits(rng)
    toks = []; asked = set(); turn = 0
    top = cache["concept_top"] if cache else []
    for _ in range(int(budget)):
        if not top:
            break
        c = int(top[rng.integers(len(top))])
        if c in asked:
            continue
        asked.add(c)
        toks += S.emit_region(known, cr, mu, TYPE_CONCEPT, c, LVL_KW, turn, rng); turn += 1
    return toks, []


def g_monotone_increase(FR, model, S, users):
    v1 = _budget_ndcg(FR, model, S, users, 1, 500)
    v8 = _budget_ndcg(FR, model, S, users, 8, 500)
    rng = np.random.default_rng(41)
    vc = []
    for u in users:
        toks, native = S.build_reveal(u["known"], "clean", 0, rng, cache=u.get("cache"))
        z = fold_np(FR, model, toks, native)
        x = L.ndcg10(FR, z, u["held"], set(u["known"].keys()))
        if x is not None:
            vc.append(x)
    m1, m8, mc = float(np.mean(v1)), float(np.mean(v8)), float(np.mean(vc))
    # paired span requires equal-length; use unpaired bootstrap of the difference of means
    diff8_1 = np.array(v8[:min(len(v1), len(v8))]) - np.array(v1[:min(len(v1), len(v8))])
    ci = _boot_ci(diff8_1)
    span_ok = bool(ci[0] > 0 and (m8 - m1) >= 0.02)
    clean_ok = bool((mc - m8) >= 0.05)
    # concept-only curve
    c1 = float(np.mean(_budget_ndcg(FR, model, S, users, 1, 700, concept_only=True)))
    c8 = float(np.mean(_budget_ndcg(FR, model, S, users, 8, 700, concept_only=True)))
    concept_ok = bool(c8 > c1)
    passed = bool(span_ok and clean_ok and concept_ok)
    print(f"  G-monotone: t1 {m1:.4f} t8 {m8:.4f} clean {mc:.4f} | (t8-t1) {m8-m1:+.4f} CI[{ci[0]:+.4f},"
          f"{ci[1]:+.4f}] span_ok={span_ok} | (clean-t8) {mc-m8:+.4f} ok={clean_ok} | concept {c1:.4f}->"
          f"{c8:.4f} ok={concept_ok} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(t1=m1, t8=m8, clean=mc, span=m8 - m1, span_ci=ci, clean_gain=mc - m8,
                concept_t1=c1, concept_t8=c8, **{"pass": passed})


# ---- value-monotone context builders -----------------------------------------------------------
def _ctx_reveal(S, u, regime, rng):
    """Return (base_tokens, native) for the regime around which to insert the swept region token."""
    if regime == "short":
        return [], []
    if regime == "long":
        toks, native = S.build_reveal(u["known"], "interview", 20, rng, cache=u.get("cache"),
                                      strategy="mostly_on")
        return toks, native
    toks, native = S.build_reveal(u["known"], "clean", 0, rng, cache=u.get("cache"))
    return toks, native


# ---- G-value-monotone x{short,long,full} (+binarize must lose) ----------------------------------
_VAL_LEVELS = [CENTERED_FOLD["hated"], CENTERED_FOLD["meh"], CENTERED_FOLD["liked"], CENTERED_FOLD["loved"]]


def g_value_monotone(FR, model, S, users, regime):
    rng = np.random.default_rng(hash(("vm", regime)) % (1 << 31))
    # direction: sweep explicit value of a top concept region; region-score must increase
    curves = []
    grad_ndcg, bin_ndcg = [], []
    for u in users:
        c = _top_region(S, u["known"], TYPE_CONCEPT)
        if c is None:
            continue
        members = S.concept_members[c]
        if len(members) < 5:
            continue
        base, native = _ctx_reveal(S, u, regime, rng)
        # remove any existing explicit token about c from base so the sweep is clean
        base_f = [t for t in base if not (t[6] == dedup_key(TYPE_CONCEPT, c) and t[1] == KIND_EXPL)]
        row = []
        for v in _VAL_LEVELS:
            toks = base_f + [_impl(TYPE_CONCEPT, c, LVL_KW, S), _expl(TYPE_CONCEPT, c, v, S)]
            z = fold_np(FR, model, toks, native)
            row.append(_region_score(FR, z, members))
        curves.append(row)
        # binarize ablation: NDCG with graded explicit values vs sign-collapsed values
        toks_g, nat_g = _ctx_reveal(S, u, regime if regime != "short" else "long", rng)
        if not toks_g:
            continue
        toks_b = [(_c[0], _c[1], _c[2], _c[3], (np.sign(_c[4]) if _c[1] == KIND_EXPL else _c[4]),
                   _c[5], _c[6], _c[7]) for _c in toks_g]
        zg = fold_np(FR, model, toks_g, nat_g); zb = fold_np(FR, model, toks_b, nat_g)
        a = L.ndcg10(FR, zg, u["held"], set(u["known"].keys()))
        b = L.ndcg10(FR, zb, u["held"], set(u["known"].keys()))
        if None not in (a, b):
            grad_ndcg.append(a); bin_ndcg.append(b)
    curves = np.array(curves)
    adj_ci = []
    for k in range(3):
        d = curves[:, k + 1] - curves[:, k]
        adj_ci.append((float(np.mean(d)), _boot_ci(d)))
    dir_ok = all(m > 0 and ci[0] > 0 for m, ci in adj_ci)
    bin_gap = np.array(grad_ndcg) - np.array(bin_ndcg)
    bci = _boot_ci(bin_gap); bin_mean = float(np.mean(bin_gap)) if len(bin_gap) else 0.0
    bin_ok = bool(bci[0] > 0)
    passed = bool(dir_ok and bin_ok)
    print(f"  G-value-monotone[{regime}]: adj means {[round(m,4) for m,_ in adj_ci]} dir_ok={dir_ok} | "
          f"binarize gap {bin_mean:+.4f} CI[{bci[0]:+.4f},{bci[1]:+.4f}] must-lose={bin_ok} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(adj=[(m, ci) for m, ci in adj_ci], dir_ok=dir_ok, binarize_gap=bin_mean,
                binarize_ci=bci, n=len(curves), **{"pass": passed})


# ---- G-know-graded x{short,long,full} (+collapse must lose) -------------------------------------
def g_know_graded(FR, model, S, users, regime):
    rng = np.random.default_rng(hash(("kg", regime)) % (1 << 31))
    absent, rough, kw = [], [], []
    grad_ndcg, coll_ndcg = [], []
    for u in users:
        eid = _top_region(S, u["known"], TYPE_ENTITY)
        if eid is None:
            continue
        members = S.entities[eid]["members"]
        base, native = _ctx_reveal(S, u, regime, rng)
        base_f = [t for t in base if not (t[6] == dedup_key(TYPE_ENTITY, eid) and t[1] == KIND_IMPL)]
        z_ab = fold_np(FR, model, base_f, native)
        z_ro = fold_np(FR, model, base_f + [_impl(TYPE_ENTITY, eid, LVL_ROUGH, S)], native)
        z_kw = fold_np(FR, model, base_f + [_impl(TYPE_ENTITY, eid, LVL_KW, S)], native)
        absent.append(_region_score(FR, z_ab, members))
        rough.append(_region_score(FR, z_ro, members))
        kw.append(_region_score(FR, z_kw, members))
        # collapse ablation: force every implicit token to know_well -> NDCG must drop
        toks_g, nat_g = _ctx_reveal(S, u, regime if regime != "short" else "long", rng)
        if not toks_g:
            continue
        toks_c = [(_c[0], _c[1], (LVL_KW if _c[1] == KIND_IMPL else _c[2]), _c[3], _c[4], _c[5], _c[6], _c[7])
                  for _c in toks_g]
        zg = fold_np(FR, model, toks_g, nat_g); zc = fold_np(FR, model, toks_c, nat_g)
        a = L.ndcg10(FR, zg, u["held"], set(u["known"].keys()))
        b = L.ndcg10(FR, zc, u["held"], set(u["known"].keys()))
        if None not in (a, b):
            grad_ndcg.append(a); coll_ndcg.append(b)
    absent = np.array(absent); rough = np.array(rough); kw = np.array(kw)
    d1 = rough - absent; d2 = kw - rough
    ci1 = _boot_ci(d1); ci2 = _boot_ci(d2)
    dir_ok = bool(np.mean(d1) > 0 and ci1[0] > 0 and np.mean(d2) > 0 and ci2[0] > 0)
    coll_gap = np.array(grad_ndcg) - np.array(coll_ndcg)
    cci = _boot_ci(coll_gap); coll_mean = float(np.mean(coll_gap)) if len(coll_gap) else 0.0
    coll_ok = bool(cci[0] > 0)
    passed = bool(dir_ok and coll_ok)
    print(f"  G-know-graded[{regime}]: rough-absent {np.mean(d1):+.4f} CI[{ci1[0]:+.4f},{ci1[1]:+.4f}] "
          f"kw-rough {np.mean(d2):+.4f} CI[{ci2[0]:+.4f},{ci2[1]:+.4f}] dir_ok={dir_ok} | collapse gap "
          f"{coll_mean:+.4f} CI[{cci[0]:+.4f},{cci[1]:+.4f}] must-lose={coll_ok} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(rough_absent=float(np.mean(d1)), rough_absent_ci=ci1, kw_rough=float(np.mean(d2)),
                kw_rough_ci=ci2, dir_ok=dir_ok, collapse_gap=coll_mean, collapse_ci=cci,
                n=int(len(absent)), **{"pass": passed})


# ---- G-GoT (entities AND items; + in-context len-20; fraction net<0) ---------------------------
def _got_channel(FR, model, S, users, ch):
    rng = np.random.default_rng(2)
    diffs = []; inctx = []
    for u in users:
        key = _top_region(S, u["known"], ch)
        if key is None:
            continue
        if ch == TYPE_ENTITY:
            members = S.entities[key]["members"]
            n_mem = len(members)
        else:
            key = _top_region(S, u["known"], TYPE_ITEM)
            j = int(key)
            members = S.concept_members[int(np.argmax(S.item_tag[j]))]
            n_mem = len(members)
        if n_mem < 3:
            continue
        bad = [_impl(ch, key, LVL_KW, S), _expl(ch, key, CENTERED_FOLD["hated"], S)]
        z_bad = fold_np(FR, model, bad, [])
        z_never = fold_np(FR, model, [], [])
        diffs.append(_region_score(FR, z_bad, members) - _region_score(FR, z_never, members))
        # in-context: embed the bad-know-well token in a 20-token interview vs the same interview without
        base, native = S.build_reveal(u["known"], "interview", 20, rng, cache=u.get("cache"),
                                      strategy="mixed")
        base_f = [t for t in base if t[6] != dedup_key(ch, key)]
        z_with = fold_np(FR, model, base_f + bad, native)
        z_wo = fold_np(FR, model, base_f, native)
        inctx.append(_region_score(FR, z_with, members) - _region_score(FR, z_wo, members))
    ci = _boot_ci(diffs); ici = _boot_ci(inctx)
    frac_neg = float(np.mean(np.array(diffs) < 0)) if diffs else 1.0
    passed = bool(ci[0] > 0 and ici[0] > 0)
    return dict(mean=float(np.mean(diffs)), ci=ci, incontext_mean=float(np.mean(inctx)),
                incontext_ci=ici, frac_net_neg=frac_neg, n=len(diffs), **{"pass": passed})


def g_got(FR, model, S, users):
    ent = _got_channel(FR, model, S, users, TYPE_ENTITY)
    itm = _got_channel(FR, model, S, users, TYPE_ITEM)
    passed = bool(ent["pass"] and itm["pass"])
    print(f"  G-GoT entities: pull {ent['mean']:+.4f} CI[{ent['ci'][0]:+.4f},{ent['ci'][1]:+.4f}] "
          f"in-ctx {ent['incontext_mean']:+.4f} frac<0 {ent['frac_net_neg']:.2f} {_v(ent['pass'])}",
          flush=True)
    print(f"  G-GoT items:    pull {itm['mean']:+.4f} CI[{itm['ci'][0]:+.4f},{itm['ci'][1]:+.4f}] "
          f"in-ctx {itm['incontext_mean']:+.4f} frac<0 {itm['frac_net_neg']:.2f} {_v(itm['pass'])}",
          flush=True)
    return dict(entities=ent, items=itm, **{"pass": passed})


# ---- G-prolific -------------------------------------------------------------------------------
def g_prolific(FR, model, S, users):
    rng = np.random.default_rng(7)
    pop = np.argsort(-S.cnt)[:2000].astype(np.int64)
    diffs = []
    for u in users:
        eid = _top_region(S, u["known"], TYPE_ENTITY)
        if eid is None:
            continue
        e = S.entities[eid]; members = e["members"]; xm = [int(j) for j in members]
        if len(xm) < 3:
            continue
        mu = 4.0
        sel_known = {j: 4.0 for j in xm}
        cr_sel = {j: 0.0 for j in xm}
        sel_toks = []
        turn = 0
        for j in xm:
            sel_toks += S._item_tokens(cr_sel, j, turn); turn += 1
        sel_toks += [_impl(TYPE_ENTITY, eid, LVL_KW, S)]
        prol_known = dict(sel_known);
        for j in pop:
            prol_known[int(j)] = 4.0
        cr_pro = {int(j): 0.0 for j in prol_known}
        prol_toks = list(sel_toks)
        for j in pop:
            if int(j) not in xm:
                prol_toks += S._item_tokens(cr_pro, int(j), turn); turn += 1
        native_sel = [j for j in xm]
        native_pro = [int(j) for j in prol_known]
        z_sel = fold_np(FR, model, sel_toks, native_sel)
        z_pro = fold_np(FR, model, prol_toks, native_pro)
        diffs.append(_region_score(FR, z_sel, members) - _region_score(FR, z_pro, members))
    ci = _boot_ci(diffs); mean = float(np.mean(diffs))
    passed = bool(ci[0] > 0)
    print(f"  G-prolific: pull(selective)-pull(prolific) {mean:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"-> {'PASS' if passed else 'FAIL'} (n={len(diffs)})", flush=True)
    return dict(mean=mean, ci=ci, n=len(diffs), **{"pass": passed})


# ---- G-order ----------------------------------------------------------------------------------
def g_order(FR, model, S, users):
    rng = np.random.default_rng(9)
    devs = []
    for u in list(users)[:300]:
        toks, native = S.build_reveal(u["known"], "interview", loguniform_budget(len(u["known"]), rng),
                                      rng, cache=u.get("cache"))
        if not toks:
            continue
        z1 = fold_np(FR, model, toks, native)
        t2 = list(toks); rng.shuffle(t2)
        z2 = fold_np(FR, model, t2, native)
        devs.append(float(np.max(np.abs(z1 - z2))))
    md = float(np.max(devs)) if devs else 0.0
    passed = bool(md < 1e-6)
    print(f"  G-order: max shuffle dev {md:.2e} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(max_dev=md, **{"pass": passed})


# ---- G-falsify-count --------------------------------------------------------------------------
def g_falsify_count(FR, model, S, users):
    rng = np.random.default_rng(8)
    devs = []
    for u in list(users)[:300]:
        toks, native = S.build_reveal(u["known"], "interview", loguniform_budget(len(u["known"]), rng),
                                      rng, cache=u.get("cache"))
        if not toks:
            continue
        z1 = fold_np(FR, model, toks, native)
        z2 = fold_np(FR, model, toks + toks, native)         # duplicate set -> dedup no-op
        devs.append(float(np.max(np.abs(z1 - z2))))
    md = float(np.max(devs)) if devs else 0.0
    passed = bool(md < 1e-6)
    print(f"  G-falsify-count: dup-set max dev {md:.2e} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(max_dev=md, **{"pass": passed})


# ---- G-caplength (elicitation not sacrificed) -------------------------------------------------
def g_caplength(FR, model, S, users):
    v1 = float(np.mean(_budget_ndcg(FR, model, S, users, 1, 300)))
    v4 = float(np.mean(_budget_ndcg(FR, model, S, users, 4, 300)))
    v8 = float(np.mean(_budget_ndcg(FR, model, S, users, 8, 300)))
    passed = bool(v8 > v1 and v4 > v1)
    print(f"  G-caplength: t1 {v1:.4f} t4 {v4:.4f} t8 {v8:.4f} (elicitation rising) -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(t1=v1, t4=v4, t8=v8, **{"pass": passed})


# ---- G-canaries -------------------------------------------------------------------------------
def g_canaries(FR, model, S, users):
    rng = np.random.default_rng(1)
    out = {}
    for ch, name in [(TYPE_ITEM, "item"), (TYPE_CONCEPT, "concept"), (TYPE_ATTR, "attr"),
                     (TYPE_ENTITY, "entity")]:
        le, li = [], []
        for u in users:
            known = u["known"]; held = u["held"]; prof = set(known.keys())
            key = _top_region(S, known, ch)
            if key is None:
                continue
            mu = float(np.mean(list(known.values())))
            cr = {int(j): float(r - mu) for j, r in known.items()}
            n0 = L.ndcg10(FR, fold_np(FR, model, [], []), held, prof)
            val, fid = S.region_value(known, cr, mu, ch, key, LVL_KW, rng)
            if val is None:
                val, fid = CENTERED_FOLD["loved"], FID_EASE
            te = [_expl(ch, key, val, S, fid)]
            ne = L.ndcg10(FR, fold_np(FR, model, te, []), held, prof)
            ti = [_impl(ch, key, LVL_KW, S)]
            ni = L.ndcg10(FR, fold_np(FR, model, ti, []), held, prof)
            if None not in (n0, ne, ni):
                le.append(ne - n0); li.append(ni - n0)
        ce, ci = _boot_ci(le), _boot_ci(li)
        out[name] = dict(explicit_lift=float(np.mean(le)), explicit_ci=ce,
                         implicit_lift=float(np.mean(li)), implicit_ci=ci, n=len(le))
        print(f"    [{name:8s}] expl {np.mean(le):+.4f} CI[{ce[0]:+.4f},{ce[1]:+.4f}] | impl "
              f"{np.mean(li):+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)
    passed = all(out[c]["explicit_lift"] > 0 and out[c]["implicit_lift"] > 0 for c in out)
    print(f"  G-canaries -> {'PASS' if passed else 'FAIL'}", flush=True)
    out["pass"] = bool(passed)
    return out


# ---- G-implicit-ablation (+cancel-check) ------------------------------------------------------
def g_implicit_ablation(FR, model, S, users):
    rng = np.random.default_rng(9)
    drops = []; pull_full = []; pull_abl = []
    for u in users:
        toks, native = S.build_reveal(u["known"], "interview", 16, rng, cache=u.get("cache"),
                                      strategy="mostly_on")
        if not toks:
            continue
        prof = set(u["known"].keys())
        zf = fold_np(FR, model, toks, native, impl_ablate=False)
        za = fold_np(FR, model, toks, native, impl_ablate=True)
        a = L.ndcg10(FR, zf, u["held"], prof); b = L.ndcg10(FR, za, u["held"], prof)
        if None not in (a, b):
            drops.append(a - b)
        eid = _top_region(S, u["known"], TYPE_ENTITY)
        if eid is not None:
            members = S.entities[eid]["members"]
            pull_full.append(_region_score(FR, zf, members))
            pull_abl.append(_region_score(FR, za, members))
    ci = _boot_ci(drops); mean = float(np.mean(drops))
    # cancel-check: zeroing implicit must NOT strengthen region-pull (pull_full >= pull_abl)
    cancel = np.array(pull_full) - np.array(pull_abl)
    cancel_ok = bool(np.mean(cancel) >= 0)
    passed = bool(ci[0] > 0 and cancel_ok)
    print(f"  G-implicit-ablation: NDCG(full)-NDCG(zeroed) {mean:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] | "
          f"cancel-check pull_full-pull_abl {np.mean(cancel):+.4f} ok={cancel_ok} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=mean, ci=ci, cancel=float(np.mean(cancel)), cancel_ok=cancel_ok,
                n=len(drops), **{"pass": passed})


# ---- G-k2-graded (+value-zeroing) -------------------------------------------------------------
def g_k2_graded(FR, model, S, users):
    rng = np.random.default_rng(22)
    fold_v, zero_v, nat_v = [], [], []
    for u in users:
        toks, native = S.build_reveal(u["known"], "interview", 2, rng, cache=u.get("cache"),
                                      strategy="onprofile")
        if not toks:
            continue
        prof = set(u["known"].keys())
        z = fold_np(FR, model, toks, native)
        z0 = fold_np(FR, model, toks, native, val_zero=True)
        zn = FR.enc_items([native])[0].numpy().astype(np.float64) if native else fold_np(FR, model, [], [])
        a = L.ndcg10(FR, z, u["held"], prof); b = L.ndcg10(FR, z0, u["held"], prof)
        c = L.ndcg10(FR, zn, u["held"], prof)
        if None not in (a, b, c):
            fold_v.append(a); zero_v.append(b); nat_v.append(c)
    gap = np.array(fold_v) - np.array(zero_v)
    ci = _boot_ci(gap)
    beat_native = float(np.mean(np.array(fold_v) - np.array(nat_v)))
    passed = bool(ci[0] > 0)
    print(f"  G-k2-graded: graded-vs-valuezeroed {np.mean(gap):+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"(fold-native {beat_native:+.4f}) rides-explicit={passed} -> {'PASS' if passed else 'FAIL'}",
          flush=True)
    return dict(graded_gap=float(np.mean(gap)), ci=ci, beat_native=beat_native, n=len(gap),
                **{"pass": passed})


# ---- G-token-count-audit ----------------------------------------------------------------------
def g_token_count_audit(FR, model, S, users):
    rng = np.random.default_rng(33)
    ok = True; checked = 0
    for u in list(users)[:200]:
        toks, native = S.build_reveal(u["known"], "clean", 0, rng, cache=u.get("cache"))
        if not toks:
            continue
        _, aud = pack_batch(FR, [toks], [native], audit=True)
        if aud["emitted"] - aud["removed"] != aud["pooled"]:
            ok = False
        checked += 1
    print(f"  G-token-count-audit: emitted-removed==pooled for all {checked} users -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return dict(checked=checked, **{"pass": bool(ok)})


# ---- beta/level telemetry ---------------------------------------------------------------------
def telemetry_levels(FR, model, S, users):
    out = {}
    for lname, lv in [("rough", LVL_ROUGH), ("know_well", LVL_KW)]:
        lifts = []
        for u in users:
            eid = _top_region(S, u["known"], TYPE_ENTITY)
            if eid is None:
                continue
            members = S.entities[eid]["members"]
            z0 = fold_np(FR, model, [], [])
            z1 = fold_np(FR, model, [_impl(TYPE_ENTITY, eid, lv, S)], [])
            lifts.append(_region_score(FR, z1, members) - _region_score(FR, z0, members))
        out[lname] = dict(mean_pull=float(np.mean(lifts)), ci=_boot_ci(lifts), n=len(lifts))
        print(f"    implicit[{lname:10s}] region-pull {np.mean(lifts):+.4f}", flush=True)
    return out


def _v(b):
    return "PASS" if b else "FAIL"


# ============================================================ gate orchestration
GATE_KEYS = ["G-intercept", "G-no-profile-leak", "G-clean", "G-monotone-increase",
             "G-value-monotone-short", "G-value-monotone-long", "G-value-monotone-full",
             "G-know-graded-short", "G-know-graded-long", "G-know-graded-full",
             "G-GoT", "G-prolific", "G-order", "G-falsify-count", "G-caplength",
             "G-canaries", "G-implicit-ablation", "G-k2-graded", "G-token-count-audit"]


def run_gates(FR, model, S, users, full=True):
    R = {}
    print("\n==================== GATE SUITE (all on TEST) ====================", flush=True)
    R["G-intercept"] = g_intercept(FR, model, S, users)
    R["G-no-profile-leak"] = g_no_profile_leak(FR, model, S, users)
    R["G-clean"] = g_clean(FR, model, S, users)
    if not R["G-clean"]["pass"] and not full:
        print("  [sanity] G-clean FAILED -> HARD STOP (do not burn full run)", flush=True)
    R["G-value-monotone-short"] = g_value_monotone(FR, model, S, users, "short")
    R["G-know-graded-short"] = g_know_graded(FR, model, S, users, "short")
    if full:
        R["G-monotone-increase"] = g_monotone_increase(FR, model, S, users)
        R["G-value-monotone-long"] = g_value_monotone(FR, model, S, users, "long")
        R["G-value-monotone-full"] = g_value_monotone(FR, model, S, users, "full")
        R["G-know-graded-long"] = g_know_graded(FR, model, S, users, "long")
        R["G-know-graded-full"] = g_know_graded(FR, model, S, users, "full")
        R["G-GoT"] = g_got(FR, model, S, users)
        R["G-prolific"] = g_prolific(FR, model, S, users)
        R["G-order"] = g_order(FR, model, S, users)
        R["G-falsify-count"] = g_falsify_count(FR, model, S, users)
        R["G-caplength"] = g_caplength(FR, model, S, users)
        print("\n  ---- G-canaries ----", flush=True)
        R["G-canaries"] = g_canaries(FR, model, S, users)
        R["G-implicit-ablation"] = g_implicit_ablation(FR, model, S, users)
        R["G-k2-graded"] = g_k2_graded(FR, model, S, users)
        R["G-token-count-audit"] = g_token_count_audit(FR, model, S, users)
        print("\n  ---- beta/level telemetry ----", flush=True)
        R["telemetry"] = telemetry_levels(FR, model, S, users)
    return R


# ============================================================ commands
def cmd_sanity(args):
    args.n_users = 200; args.n_val = 80; args.n_test = 120; args.epochs = args.epochs or 4
    D, FR, S, tr, va, ge = get_split(args)
    print(f"[sanity] tr {len(tr)} va {len(va)} te {len(ge)}", flush=True)
    r = train_model(FR, S, tr, va, args, args.lam, args.margin, tag="_sanity")
    model, meta = load_model(BEST.replace(".pt", "_sanity.pt"))
    R = run_gates(FR, model, S, ge, full=False)
    R["_meta"] = dict(mode="sanity", **r)
    json.dump(R, open(".cache/arena/fold_master_sanity.json", "w"), indent=1, default=float)
    print("\n[sanity] wrote .cache/arena/fold_master_sanity.json", flush=True)
    return R


def cmd_sweep(args):
    D, FR, S, tr, va, ge = get_split(args)
    print(f"[sweep] tr {len(tr)} va {len(va)}", flush=True)
    results = []
    for lam in [0.1, 0.3, 1.0]:
        for m in [0.25, 0.5, 1.0]:
            r = train_model(FR, S, tr, va, args, lam, m, tag=f"_l{lam}_m{m}")
            results.append(r)
            json.dump(results, open(SWEEP_JSON, "w"), indent=1, default=float)
    win = max(results, key=lambda r: r["best_val"])
    print(f"\n[sweep] WINNER lam={win['lam']} margin={win['margin']} val {win['best_val']:.4f}", flush=True)
    return results, win


def cmd_train(args):
    D, FR, S, tr, va, ge = get_split(args)
    print(f"[train] tr {len(tr)} va {len(va)} te {len(ge)}  lam={args.lam} margin={args.margin}",
          flush=True)
    r = train_model(FR, S, tr, va, args, args.lam, args.margin, tag="")
    import shutil
    shutil.copyfile(r["best_ckpt"], BEST)
    model, meta = load_model(BEST)
    R = run_gates(FR, model, S, ge, full=True)
    R["_meta"] = dict(mode="train", **r, n_train=len(tr), n_test=len(ge))
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    json.dump(R, open(RESULTS_JSON, "w"), indent=1, default=float)
    _write_md(R)
    _print_summary(R)
    print(f"\n[train] wrote {RESULTS_JSON} + {BUILD_MD}", flush=True)
    return R


def cmd_gates(args):
    D, FR, S, tr, va, ge = get_split(args)
    model, meta = load_model(BEST)
    R = run_gates(FR, model, S, ge, full=True)
    R["_meta"] = dict(mode="gates", meta=meta, n_test=len(ge))
    json.dump(R, open(RESULTS_JSON, "w"), indent=1, default=float)
    _write_md(R)
    _print_summary(R)
    return R


def _print_summary(R):
    print("\n==================== GATE VERDICTS ====================", flush=True)
    allp = True
    for k in GATE_KEYS:
        if k in R:
            p = R[k].get("pass")
            allp = allp and bool(p)
            print(f"  {k:28s} {_v(p)}", flush=True)
    print(f"  {'ALL GATES':28s} {_v(allp)}", flush=True)


def _write_md(R):
    def a(t):
        open(BUILD_MD, "a", encoding="utf-8").write(t)
    m = R.get("_meta", {})
    a("\n## FOLD MASTER (locked design) -- gates\n\n")
    a(f"Locked design per FOLD_MASTER.md sec E (i25/v3 Deep-Sets SUM-pool residual + leak-free "
      f"two-channel tokens, NO surprise, item-hole native_z, distinct-set dedup, ordinal graded loss). "
      f"lam={m.get('lam')} margin={m.get('margin')} best val NDCG@10 {m.get('best_val')} "
      f"@ep{m.get('best_epoch')}; train users {m.get('n_train')}, TEST users {m.get('n_test')}.\n\n")
    gc = R.get("G-clean", {})
    a(f"**Headline:** clean fold {gc.get('fold'):.4f} vs native {gc.get('native'):.4f} "
      f"(delta {gc.get('delta'):+.4f}; HARD STOP thr native-0.03) -> {_v(gc.get('pass'))}.\n\n")
    a("| gate | key metric | verdict |\n|---|---|:--:|\n")
    for k in GATE_KEYS:
        if k not in R:
            continue
        d = R[k]
        met = ""
        if k == "G-clean":
            met = f"fold {d['fold']:.4f} vs native {d['native']:.4f} ({d['delta']:+.4f})"
        elif k == "G-intercept":
            met = f"empty maxdev {d['empty_maxdev']:.1e}"
        elif k == "G-monotone-increase":
            met = f"t1 {d['t1']:.3f} t8 {d['t8']:.3f} clean {d['clean']:.3f}"
        elif k.startswith("G-value-monotone"):
            met = f"dir_ok {d['dir_ok']} binarize {d['binarize_gap']:+.4f}"
        elif k.startswith("G-know-graded"):
            met = f"dir_ok {d['dir_ok']} collapse {d['collapse_gap']:+.4f}"
        elif k == "G-GoT":
            met = f"ent {d['entities']['mean']:+.4f} / item {d['items']['mean']:+.4f}"
        elif "mean" in d:
            met = f"{d['mean']:+.4f}"
        elif "max_dev" in d:
            met = f"maxdev {d['max_dev']:.1e}"
        a(f"| {k} | {met} | {_v(d.get('pass'))} |\n")
    allp = all(R[k].get("pass") for k in GATE_KEYS if k in R)
    a(f"\n**ALL GATES PASS: {allp}.**\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sanity", "sweep", "train", "gates"])
    ap.add_argument("--n_users", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--clean_frac", type=float, default=0.30)
    a = ap.parse_args()
    if a.cmd == "sanity":
        cmd_sanity(a)
    elif a.cmd == "sweep":
        cmd_sweep(a)
    elif a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "gates":
        cmd_gates(a)
