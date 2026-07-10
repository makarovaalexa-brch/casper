"""i25_fold_v31.py -- FOLD-V3.1: the NO-CLUE ANTI-SURPRISE ablation.

Author-directed NIGHT TASK. Tests whether no-clue / refusal tokens can carry genuine NEGATIVE
implicit signal, before finalizing the decision to skip folding them.

HYPOTHESIS (author, quoted verbatim in experiments/NOCLUE_ABLATION.md):
  no-clue is informative exactly when knowledge was EXPECTED -- refusing a famous item your volume
  predicts you'd know = real negative taste/territory evidence; refusing obscurity = nothing. The
  current sampler gives no-clue tokens the same SURPRISE feature as consumption (low for real
  no-clues) -- the fold cannot express "expected but absent".

MINIMAL DELTA vs FOLD-V3 (reuse i25_fold_v3.py + i25_fold_v3_sampler.py):
  1. NEW FEATURE on no-clue (LVL_NEG implicit) tokens ONLY: ANTI-SURPRISE = expectedness of knowing
     = log-lift of predicted engagement (fame x volume) vs the actual (zero) engagement of a refusal.
        E[n_E] = V * p_E         (volume-predicted engagement; p_E = popmass(E)/totalpop; V = volume)
        n_E    = # revealed members of E (0 for a genuine refusal)
        ANTI   = clip( log( (E[n_E] + 0.5) / (n_E + 0.5) ), 0, 8 )
     HIGH when a famous/expected entity is refused; ~0 for obscure refusals. (Equals -surprise, but
     as a DEDICATED feature lit only on no-clue tokens it decouples "expected-but-absent" from the
     consumption-surprise weight, which v3 found inert on the LVL_NEG flag.)
  2. CURRICULUM: informative refusals must appear. The v2.1 answerer yields k=0 naturally; we MEASURE
     the natural expected-refusal rate and modestly oversample famous-off-profile probes if starved.
  3. TRAIN one run, winning v3 config (clean_frac 0.30, ~10 epochs). NO LLM calls. Population trU
     users only (the 173 study/eval users are NEVER touched). Deterministic. Reduced threads.

Run:
  python scripts/i25_fold_v31.py train                 # one run, clean_frac 0.30
  python scripts/i25_fold_v31.py probes                # P1/P2/P3 battery + write NOCLUE_ABLATION.md
"""
import os
# --- compute courtesy: reduced thread count (arena agent running heavy jobs) ---
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
import sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import i25_lib as L
import i25_fold_v3 as V3
import i25_fold_v3_sampler as SP
from i25_fold_v3_sampler import (KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG,
                                 FID_DATA, FID_EASE, FID_LLM, TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR,
                                 TYPE_ENTITY, NTYPE, V3Sampler)

D_LAT = L.D_LAT
CKPT = ".cache/i25_fold_v31_c30.pt"
BEST = ".cache/i25_fold_v31_best.pt"
LOGP = ".cache/i25_fold_v31_c30_log.json"
PROBES_JSON = ".cache/i25_fold_v31_probes.json"
ABL_MD = "experiments/NOCLUE_ABLATION.md"
CLEAN_FRAC = 0.30
VAL_CLEAN_FRAC = 0.5
MAX_CLEAN_ITEMS = 50
ANTI_CLIP = 8.0
EXPECTED_ANTI_THR = 1.0        # anti-surprise above which a refusal is "expected but absent"
STARVED_PER_INTERVIEW = 0.30   # if < this many expected-refusals per interview, oversample


# =============================================================== the v3.1 sampler (anti-surprise)
class V31Sampler(V3Sampler):
    """FOLD-V3 sampler + ANTI-SURPRISE on no-clue tokens + optional famous-off-profile oversampling.
    Emits 8-field tokens: (type, kind, level, surprise, fid, value, emb, ANTI)."""

    def __init__(self, D, FR):
        super().__init__(D, FR)
        self.oversample_famref = False     # training-only; OFF for probes (natural rates)
        self.p_famref = 0.5
        # famous pools for oversampling expected-refusal exposure (off a user's profile)
        self.famous_items = np.argsort(-self.cnt)[:1500].astype(np.int64)
        ent_by_pop = sorted(self.entity_ids, key=lambda e: -self.entities[e]["popmass"])
        self.famous_entities = ent_by_pop[:max(1, int(0.4 * len(ent_by_pop)))]

    # ---- ANTI-SURPRISE: expectedness of knowing (high for famous refusals, ~0 for obscure) ----
    def anti_surprise(self, known_set, V, region_type, region_key):
        if region_type == TYPE_CONCEPT:
            pm = self.concept_popmass[region_key]; mset = self.concept_set[region_key]
        elif region_type == TYPE_ATTR:
            pm = self.attr_popmass[region_key]; mset = self.attr_set[region_key]
        elif region_type == TYPE_ENTITY:
            pm = self.entities[region_key]["popmass"]; mset = self.entities[region_key]["mset"]
        else:
            j = int(region_key); pm = float(self.cnt[j] + 1.0); mset = {j}
        n_E = float(sum(1 for x in known_set if x in mset))
        p_E = pm / self.totalpop
        exp = max(V, 1) * p_E                       # volume-predicted engagement (fame x volume)
        a = float(np.log((exp + 0.5) / (n_E + 0.5)))
        return float(np.clip(a, 0.0, ANTI_CLIP))

    # ---- emit: same logic as v3 but 8-field tokens; anti lit only on the no-clue token ----
    def emit(self, known, cr, mu, region_type, region_key, trait, rng, force_level=None):
        known_set = set(int(j) for j in known)
        V = len(known)
        surprise = self._surprise_val(known_set, V, region_type, region_key)
        emb = self.region_emb(region_type, region_key)
        level = force_level if force_level is not None else \
            self.draw_knowledge(surprise, region_type, trait, rng)
        toks = []
        if level == LVL_NEG:
            anti = self.anti_surprise(known_set, V, region_type, region_key)
            toks.append((region_type, KIND_IMPL, LVL_NEG, surprise, FID_DATA, 0.0, emb, anti))
            return toks
        toks.append((region_type, KIND_IMPL, level, surprise, FID_DATA, 0.0, emb, 0.0))
        val, fid = self.region_value(known, cr, mu, region_type, region_key, level, rng)
        if val is not None:
            toks.append((region_type, KIND_EXPL, LVL_ROUGH, 0.0, fid, float(val), emb, 0.0))
        return toks

    # ---- clean full profile: 8-field item tokens (no neg tokens in clean mode) ----
    def _build_clean(self, known, cr, mu, native, trait, rng, cache, max_items=None):
        toks = []
        ks = cache["ks"]; kset = cache["kset"]
        if max_items is not None and len(ks) > max_items:
            idx = rng.choice(len(ks), size=max_items, replace=False)
            ks = [ks[i] for i in idx]
        for j in ks:
            j = int(j)
            surprise = self._surprise_val(kset, len(ks), TYPE_ITEM, j)
            emb = self.FR.Wn[j].numpy().astype(np.float32)
            toks.append((TYPE_ITEM, KIND_EXPL, LVL_ROUGH, 0.0, FID_DATA, float(cr[j]), emb, 0.0))
            toks.append((TYPE_ITEM, KIND_IMPL, LVL_KW, surprise, FID_DATA, 0.0, emb, 0.0))
        for c in cache["concept_top"][:6]:
            toks += self.emit(known, cr, mu, TYPE_CONCEPT, int(c), trait, rng, force_level=LVL_KW)
        agg = [(ak, sum(1 for j in kset if j in self.attr_set[ak])) for ak in cache["attr_hit"]]
        for ak, _ in sorted(agg, key=lambda kv: -kv[1])[:6]:
            toks += self.emit(known, cr, mu, TYPE_ATTR, ak, trait, rng, force_level=LVL_KW)
        ent_hit = [(sum(1 for j in kset if j in self.entities[eid]["mset"]), eid)
                   for eid in cache["ent_hit"]]
        for _, eid in sorted(ent_hit, reverse=True)[:4]:
            toks += self.emit(known, cr, mu, TYPE_ENTITY, eid, trait, rng, force_level=LVL_KW)
        rng.shuffle(toks)
        return toks, native

    # ---- interview: same as v3 + optional famous-off-profile probe (natural refusal decided) ----
    def _build_interview(self, known, cr, mu, native, budget, trait, rng, cache):
        toks = []
        asked = set()
        kset = cache["kset"]
        for _ in range(budget):
            ch = self._SLOT_CH[int(np.searchsorted(self._SLOT_CDF, rng.random()))]
            on_profile = rng.random() < 0.7
            key = self._pick_region(ch, rng, on_profile, cache)
            sig = (ch, key if not isinstance(key, np.integer) else int(key))
            if sig in asked:
                continue
            asked.add(sig)
            toks += self.emit(known, cr, mu, ch, key, trait, rng)
        # modest oversampling of EXPECTED-refusal exposure (training only): ask a famous
        # off-profile region and let the v2.1 knowledge model decide the refusal (realistic).
        if self.oversample_famref and rng.random() < self.p_famref:
            if rng.random() < 0.5 and len(self.famous_entities):
                for _ in range(3):
                    eid = self.famous_entities[rng.integers(len(self.famous_entities))]
                    if self.entities[eid]["mset"].isdisjoint(kset) and (TYPE_ENTITY, eid) not in asked:
                        asked.add((TYPE_ENTITY, eid))
                        toks += self.emit(known, cr, mu, TYPE_ENTITY, eid, trait, rng)
                        break
            else:
                for _ in range(3):
                    j = int(self.famous_items[rng.integers(len(self.famous_items))])
                    if j not in kset and (TYPE_ITEM, j) not in asked:
                        asked.add((TYPE_ITEM, j))
                        toks += self.emit(known, cr, mu, TYPE_ITEM, j, trait, rng)
                        break
        item_asked = [int(k) for (c, k) in asked
                      if c == TYPE_ITEM and int(k) in known and known[int(k)] >= 4]
        rng.shuffle(toks)
        return toks, item_asked


# =============================================================== the v3.1 fold (adds anti feature)
class FoldV31(nn.Module):
    """FOLD-V3 Deep-Sets residual + one extra scalar input (ANTI-SURPRISE), lit only on no-clue
    tokens. Per-token input = [type_oh(4), kind_oh(2), lvl_oh(3), surprise(1), ANTI(1), fid_oh(3),
    value(1), emb(d), value*emb(d)]. z = native_z + rho(pool, native_z, log1p(ntok))."""

    def __init__(self, d=D_LAT):
        super().__init__()
        in_dim = NTYPE + 2 + 3 + 1 + 1 + 3 + 1 + d + d
        self.phi = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)
        self.d = d

    def forward(self, tt, tk, tl, ts, ta, tf, tv, te, mask, native_z, impl_ablate=False):
        B, K, d = te.shape
        is_impl = (tk == KIND_IMPL).to(te.dtype)
        is_expl = 1.0 - is_impl
        is_neg = ((tl == LVL_NEG) & (tk == KIND_IMPL)).to(te.dtype)
        type_oh = F.one_hot(tt.clamp(min=0), NTYPE).to(te.dtype)
        kind_oh = F.one_hot(tk.clamp(min=0), 2).to(te.dtype)
        lvl_oh = F.one_hot(tl.clamp(min=0), 3).to(te.dtype) * is_impl.unsqueeze(-1)
        fid_oh = F.one_hot(tf.clamp(min=0), 3).to(te.dtype) * is_expl.unsqueeze(-1)
        surprise = ts.unsqueeze(-1)
        anti = (ta * is_neg).unsqueeze(-1)                    # lit only on no-clue tokens
        value = tv.unsqueeze(-1)
        ve = value * te
        x = torch.cat([type_oh, kind_oh, lvl_oh, surprise, anti, fid_oh, value, te, ve], dim=-1)
        eff_mask = mask * (is_expl if impl_ablate else torch.ones_like(mask))
        h = self.phi(x) * eff_mask.unsqueeze(-1)
        pool = h.sum(1)
        ntok = eff_mask.sum(1, keepdim=True)
        delta = self.rho(torch.cat([pool, native_z, torch.log1p(ntok)], dim=-1))
        return native_z + delta


# =============================================================== packing / folding (8-field tokens)
def pack_batch(FR, tok_lists, native_lists):
    B = len(tok_lists)
    K = max((len(t) for t in tok_lists), default=1); K = max(K, 1)
    d = FR.W.shape[1]
    tt = torch.zeros((B, K), dtype=torch.long)
    tk = torch.zeros((B, K), dtype=torch.long)
    tl = torch.zeros((B, K), dtype=torch.long)
    ts = torch.zeros((B, K), dtype=torch.float32)
    ta = torch.zeros((B, K), dtype=torch.float32)
    tf = torch.zeros((B, K), dtype=torch.long)
    tv = torch.zeros((B, K), dtype=torch.float32)
    te = torch.zeros((B, K, d), dtype=torch.float32)
    mask = torch.zeros((B, K), dtype=torch.float32)
    for b, toks in enumerate(tok_lists):
        for k, tok in enumerate(toks):
            typ, kind, lvl, surp, fid, val, emb, anti = tok
            tt[b, k] = typ; tk[b, k] = kind; tl[b, k] = lvl; ts[b, k] = surp
            tf[b, k] = fid; tv[b, k] = val; te[b, k] = torch.as_tensor(emb)
            ta[b, k] = anti; mask[b, k] = 1.0
    nz = FR.enc_items(native_lists)
    return tt, tk, tl, ts, ta, tf, tv, te, mask, nz


def fold_batch(FR, model, tok_lists, native_lists, impl_ablate=False):
    if not tok_lists:
        return np.zeros((0, FR.W.shape[1]))
    args = pack_batch(FR, tok_lists, native_lists)
    with torch.no_grad():
        z = model(*args, impl_ablate=impl_ablate)
    return z.numpy().astype(np.float64)


def fold_np(FR, model, toks, native, impl_ablate=False):
    return fold_batch(FR, model, [toks], [native], impl_ablate=impl_ablate)[0].astype(np.float64)


def strip7(toks, drop_noclue=False):
    """8-field v3.1 tokens -> 7-field v3 tokens (for folding with the v3 model). If drop_noclue,
    also removes the no-clue (LVL_NEG implicit) tokens = the current arena skip rule."""
    out = []
    for t in toks:
        if drop_noclue and t[1] == KIND_IMPL and t[2] == LVL_NEG:
            continue
        out.append((t[0], t[1], t[2], t[3], t[4], t[5], t[6]))
    return out


# =============================================================== training (one run, clean_frac 0.30)
def batch_loss(FR, model, S, users, clean_frac, rng):
    tl, nl, tgt, prof = [], [], [], []
    for u in users:
        mode = "clean" if rng.random() < clean_frac else "interview"
        budget = int(rng.integers(1, 25))
        toks, native = S.build_reveal(u["known"], mode, budget, rng, cache=u.get("cache"),
                                      max_items=MAX_CLEAN_ITEMS)
        if not toks:
            continue
        tl.append(toks); nl.append(native); tgt.append(u["held"]); prof.append(set(u["known"].keys()))
    if not tl:
        return None
    tt, tk, tll, ts, ta, tf, tv, te, mask, nz = pack_batch(FR, tl, nl)
    z = model(tt, tk, tll, ts, ta, tf, tv, te, mask, nz)
    Smat = z @ FR.W.T + FR.bdec
    B, ni = Smat.shape
    negmask = torch.zeros((B, ni), dtype=torch.bool)
    tg = torch.zeros((B, ni), dtype=torch.float32)
    for b in range(B):
        negmask[b, list(prof[b])] = True
        tg[b, list(tgt[b])] = 1.0
    Smat = Smat.masked_fill(negmask, -1e9)
    logp = torch.log_softmax(Smat, dim=1)
    n_t = tg.sum(1).clamp(min=1)
    return -((logp * tg).sum(1) / n_t).mean()


@torch.no_grad()
def val_ndcg(FR, model, S, val_users, clean_frac, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    tl, nl, idx = [], [], []
    for i, u in enumerate(val_users):
        mode = "clean" if rng.random() < clean_frac else "interview"
        budget = int(rng.integers(1, 25))
        toks, native = S.build_reveal(u["known"], mode, budget, rng, cache=u.get("cache"),
                                      max_items=MAX_CLEAN_ITEMS)
        if toks:
            tl.append(toks); nl.append(native); idx.append(i)
    if not tl:
        return 0.0
    vals = []
    for s in range(0, len(tl), 1500):
        e = min(s + 1500, len(tl))
        Z = fold_batch(FR, model, tl[s:e], nl[s:e])
        for r in range(s, e):
            u = val_users[idx[r]]
            v = L.ndcg10(FR, Z[r - s], u["held"], set(u["known"].keys()))
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def measure_refusals(S, users, seed, budget=24, n=400):
    """Natural expected-refusal exposure per interview (oversampling OFF)."""
    was = S.oversample_famref; S.oversample_famref = False
    rng = np.random.default_rng(seed)
    n_int = 0; n_neg = 0; n_exp = 0; antis = []
    for u in users[:n]:
        toks, _ = S.build_reveal(u["known"], "interview", budget, rng, cache=u.get("cache"))
        n_int += 1
        for t in toks:
            if t[1] == KIND_IMPL and t[2] == LVL_NEG:
                n_neg += 1
                if t[7] >= EXPECTED_ANTI_THR:
                    n_exp += 1
                antis.append(t[7])
    S.oversample_famref = was
    return dict(n_interviews=n_int, neg_per_int=n_neg / max(n_int, 1),
                expected_per_int=n_exp / max(n_int, 1),
                anti_mean=float(np.mean(antis)) if antis else 0.0,
                anti_p90=float(np.percentile(antis, 90)) if antis else 0.0)


def cmd_train(args):
    t0 = time.time()
    print("[v31] loading data + frozen RecVAE ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = V31Sampler(D, FR)
    print(f"[v31] sampler ready: {S.n_concept} concepts, {len(S.attr_keys)} attrs, "
          f"{len(S.entity_ids)} entities; famous pool {len(S.famous_entities)} ent / "
          f"{len(S.famous_items)} items", flush=True)
    prof = L.load_train_profiles(D, args.n_users + args.n_val, seed=args.seed)
    keys = list(prof.keys()); rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    val_keys = set(keys[:args.n_val]); tr_keys = keys[args.n_val:]
    tr_users = V3.prep_users({u: prof[u] for u in tr_keys}, np.random.default_rng(args.seed + 1), S)
    val_users = V3.prep_users({u: prof[u] for u in val_keys}, np.random.default_rng(args.seed + 2), S)
    print(f"[v31] train users {len(tr_users)}  val users {len(val_users)}  "
          f"(load {round(time.time()-t0,1)}s)", flush=True)

    # ---- curriculum diagnostic: natural refusal exposure, decide oversampling ----
    nat = measure_refusals(S, tr_users, args.seed + 5)
    starved = nat["expected_per_int"] < STARVED_PER_INTERVIEW
    S.oversample_famref = bool(starved)
    over = None
    if starved:
        over = measure_refusals(S, tr_users, args.seed + 5)  # note: measure_* forces OFF internally
    print(f"[v31] NATURAL refusals/int {nat['neg_per_int']:.3f}  EXPECTED(anti>={EXPECTED_ANTI_THR})"
          f"/int {nat['expected_per_int']:.3f}  anti_mean {nat['anti_mean']:.3f}  "
          f"anti_p90 {nat['anti_p90']:.3f}  -> starved={starved} oversample={S.oversample_famref}",
          flush=True)
    curr = dict(natural=nat, starved=bool(starved), oversample=bool(S.oversample_famref),
                p_famref=S.p_famref if S.oversample_famref else 0.0,
                threshold=STARVED_PER_INTERVIEW)
    json.dump(curr, open(".cache/i25_fold_v31_curriculum.json", "w"), indent=1, default=float)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = FoldV31()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    state = dict(epoch=0, best_val=-1.0, best_epoch=-1, history=[], clean_frac=CLEAN_FRAC,
                 curriculum=curr)
    ep0 = 0
    if getattr(args, "resume", False) and os.path.exists(CKPT):
        blob = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
        state = blob["state"]; state.setdefault("curriculum", curr)
        ep0 = int(state["epoch"])
        print(f"[v31] RESUME from {CKPT} @ep{ep0} (best {state['best_val']:.4f} "
              f"@ep{state['best_epoch']})", flush=True)
    step_rng = np.random.default_rng(args.seed + 130 + 1000 * ep0)
    print(f"[v31] === training clean_frac={CLEAN_FRAC:.2f}, epochs={args.epochs} ===", flush=True)
    for ep in range(ep0, args.epochs):
        model.train()
        order = np.arange(len(tr_users)); step_rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), args.batch):
            us = [tr_users[i] for i in order[b0:b0 + args.batch]]
            loss = batch_loss(FR, model, S, us, CLEAN_FRAC, step_rng)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        vN = val_ndcg(FR, model, S, val_users, VAL_CLEAN_FRAC, args.seed + 7)
        state["epoch"] = ep + 1
        rec = dict(epoch=ep + 1, train_loss=round(tot / max(nb, 1), 4), val_ndcg=round(vN, 4),
                   min=round((time.time() - t0) / 60, 1))
        state["history"].append(rec)
        print(f"[v31] ep {ep+1:2d} | loss {rec['train_loss']:.4f} | val NDCG@10 {vN:.4f} | "
              f"{rec['min']}m", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state,
                        args=vars(args)), CKPT)
        if vN > state["best_val"]:
            state["best_val"] = vN; state["best_epoch"] = ep + 1
            torch.save(dict(model=model.state_dict(), state=state, args=vars(args)), BEST)
        json.dump(state["history"], open(LOGP, "w"), indent=1)
    print(f"[v31] BEST val {state['best_val']:.4f} @ep{state['best_epoch']} -> {BEST}  "
          f"({round((time.time()-t0)/60,1)}m)", flush=True)
    return state


# =============================================================== probes P1 / P2 / P3
def _boot_ci(x, n_boot=2000, seed=0):
    return V3._boot_ci(x, n_boot=n_boot, seed=seed)


def _region_score(FR, z, members):
    S = FR.decode_np(z[None, :])[0]
    return float(np.mean(S[members]))


def load_v31(FR, ckpt=BEST):
    blob = torch.load(ckpt, map_location="cpu")
    m = FoldV31(); m.load_state_dict(blob["model"]); m.eval()
    return m, blob.get("state", {})


# ---- P1: the author's probe. No-clue on EXPECTED (high anti) vs OBSCURE (low anti) entities ----
def probe_P1(FR, S, m31, users):
    print("\n---- P1 AUTHOR PROBE (no-clue pull vs no token: expected must be NEGATIVE, obscure ~0) ----",
          flush=True)
    ent_by_pop = sorted(S.entity_ids, key=lambda e: -S.entities[e]["popmass"])
    famous = ent_by_pop[:60]                       # globally expected entities
    obscure = ent_by_pop[-400:]                    # globally obscure entities
    rng = np.random.default_rng(11)
    exp_pull, obs_pull, exp_anti, obs_anti = [], [], [], []
    z0 = fold_np(FR, m31, [], [])
    for u in users:
        known = u["known"]; kset = set(int(j) for j in known); V = len(known)
        # EXPECTED: a famous entity the user did NOT consume (n_E == 0) -> high anti
        fam = [e for e in famous if S.entities[e]["mset"].isdisjoint(kset)]
        if fam:
            eid = fam[int(rng.integers(len(fam)))]
            members = S.entities[eid]["members"]; emb = S.region_emb(TYPE_ENTITY, eid)
            surp = S._surprise_val(kset, V, TYPE_ENTITY, eid)
            anti = S.anti_surprise(kset, V, TYPE_ENTITY, eid)
            tok = [(TYPE_ENTITY, KIND_IMPL, LVL_NEG, surp, FID_DATA, 0.0, emb, anti)]
            z1 = fold_np(FR, m31, tok, [])
            exp_pull.append(_region_score(FR, z1, members) - _region_score(FR, z0, members))
            exp_anti.append(anti)
        # OBSCURE: an obscure entity the user did NOT consume -> anti ~ 0
        obs = [e for e in obscure if S.entities[e]["mset"].isdisjoint(kset)]
        if obs:
            eid = obs[int(rng.integers(len(obs)))]
            members = S.entities[eid]["members"]; emb = S.region_emb(TYPE_ENTITY, eid)
            surp = S._surprise_val(kset, V, TYPE_ENTITY, eid)
            anti = S.anti_surprise(kset, V, TYPE_ENTITY, eid)
            tok = [(TYPE_ENTITY, KIND_IMPL, LVL_NEG, surp, FID_DATA, 0.0, emb, anti)]
            z1 = fold_np(FR, m31, tok, [])
            obs_pull.append(_region_score(FR, z1, members) - _region_score(FR, z0, members))
            obs_anti.append(anti)
    ce = _boot_ci(exp_pull); co = _boot_ci(obs_pull)
    me, mo = float(np.mean(exp_pull)), float(np.mean(obs_pull))
    exp_pass = bool(ce[1] < 0)          # expected pull NEGATIVE (CI excludes 0 on the negative side)
    obs_pass = bool(co[0] <= 0 <= co[1] or abs(mo) < 0.01)   # obscure ~ 0
    print(f"  EXPECTED (anti_mean {np.mean(exp_anti):.2f}): pull {me:+.4f} CI[{ce[0]:+.4f},{ce[1]:+.4f}]"
          f"  n={len(exp_pull)}  -> {'NEG-PASS' if exp_pass else 'FAIL'}", flush=True)
    print(f"  OBSCURE  (anti_mean {np.mean(obs_anti):.2f}): pull {mo:+.4f} CI[{co[0]:+.4f},{co[1]:+.4f}]"
          f"  n={len(obs_pull)}  -> {'~0-PASS' if obs_pass else 'FAIL'}", flush=True)
    return dict(expected=dict(mean=me, ci=ce, anti=float(np.mean(exp_anti)), n=len(exp_pull),
                              **{"pass": exp_pass}),
                obscure=dict(mean=mo, ci=co, anti=float(np.mean(obs_anti)), n=len(obs_pull),
                             **{"pass": obs_pass}))


# ---- P2: interview-level justification. THE DECISION NUMBER: does (c) beat (a)? ----
def probe_P2(FR, S, m31, m3, users, budget=24, seed=22):
    print("\n---- P2 INTERVIEW JUSTIFICATION (T=24, natural k=0; (c) fold-w/-anti vs (a) skip) ----",
          flush=True)
    was = S.oversample_famref; S.oversample_famref = False    # natural deployment rates
    rng = np.random.default_rng(seed)
    a_v, b_v, c_v, cskip_v = [], [], [], []
    n_neg_tot = 0; n_users_neg = 0
    for u in users:
        known = u["known"]; held = u["held"]; prof = set(known.keys())
        toks, native = S.build_reveal(known, "interview", budget, rng, cache=u.get("cache"))
        if not toks:
            continue
        neg = [t for t in toks if t[1] == KIND_IMPL and t[2] == LVL_NEG]
        n_neg_tot += len(neg); n_users_neg += (len(neg) > 0)
        # (a) v3, SKIP no-clue (current arena rule)
        za = V3.fold_np(FR, m3, strip7(toks, drop_noclue=True), native)
        # (b) v3, FOLD no-clue as-trained (v3 saw standard surprise, no anti)
        zb = V3.fold_np(FR, m3, strip7(toks, drop_noclue=False), native)
        # (c) v3.1, FOLD no-clue WITH anti-surprise
        zc = fold_np(FR, m31, toks, native)
        # (c') v3.1, SKIP no-clue (isolates the fold-with-anti mechanism within the same model)
        zcs = fold_np(FR, m31, [t for t in toks if not (t[1] == KIND_IMPL and t[2] == LVL_NEG)], native)
        va = L.ndcg10(FR, za, held, prof); vb = L.ndcg10(FR, zb, held, prof)
        vc = L.ndcg10(FR, zc, held, prof); vcs = L.ndcg10(FR, zcs, held, prof)
        if None in (va, vb, vc, vcs):
            continue
        a_v.append(va); b_v.append(vb); c_v.append(vc); cskip_v.append(vcs)
    S.oversample_famref = was
    a_v, b_v, c_v, cskip_v = map(np.asarray, (a_v, b_v, c_v, cskip_v))
    dca = c_v - a_v; dcb = c_v - b_v; dccs = c_v - cskip_v; dba = b_v - a_v
    ci_ca = _boot_ci(dca); ci_cb = _boot_ci(dcb); ci_ccs = _boot_ci(dccs); ci_ba = _boot_ci(dba)
    decision = bool(ci_ca[0] > 0)      # DECISION: (c) beats (a) with CI excl 0
    n = len(a_v)
    print(f"  n={n} users; no-clue tokens total {n_neg_tot} ({n_users_neg} users had >=1)", flush=True)
    print(f"  (a) v3 skip-noclue     NDCG {a_v.mean():.4f}", flush=True)
    print(f"  (b) v3 fold-noclue     NDCG {b_v.mean():.4f}   d(b-a) {dba.mean():+.4f} "
          f"CI[{ci_ba[0]:+.4f},{ci_ba[1]:+.4f}]", flush=True)
    print(f"  (c') v31 skip-noclue   NDCG {cskip_v.mean():.4f}", flush=True)
    print(f"  (c) v31 fold+anti      NDCG {c_v.mean():.4f}   d(c-a) {dca.mean():+.4f} "
          f"CI[{ci_ca[0]:+.4f},{ci_ca[1]:+.4f}]  <== DECISION", flush=True)
    print(f"       d(c-c') within-v31 {dccs.mean():+.4f} CI[{ci_ccs[0]:+.4f},{ci_ccs[1]:+.4f}]  "
          f"d(c-b) {dcb.mean():+.4f} CI[{ci_cb[0]:+.4f},{ci_cb[1]:+.4f}]", flush=True)
    print(f"  DECISION (c beats a): {decision} -> "
          f"{'CHANGE ARENA RULE to fold-noclue-with-v3.1' if decision else 'SKIP RULE JUSTIFIED'}",
          flush=True)
    return dict(n=n, n_neg_tokens=n_neg_tot, n_users_with_neg=n_users_neg,
                ndcg_a=float(a_v.mean()), ndcg_b=float(b_v.mean()), ndcg_c=float(c_v.mean()),
                ndcg_cskip=float(cskip_v.mean()),
                d_ca=float(dca.mean()), ci_ca=ci_ca, d_cb=float(dcb.mean()), ci_cb=ci_cb,
                d_ccs=float(dccs.mean()), ci_ccs=ci_ccs, d_ba=float(dba.mean()), ci_ba=ci_ba,
                decision_c_beats_a=decision)


# ---- P3: must not regress the decisive v3 gates (quick G2 / G2b / G7 / G4 on v3.1) ----
def _top_region(S, known, ch):
    return V3._top_region(S, known, ch)


def p3_G2(FR, S, m31, users):
    diffs = []
    for u in users:
        known = u["known"]; kset = set(int(j) for j in known)
        eid = _top_region(S, known, TYPE_ENTITY)
        if eid is None:
            continue
        members = S.entities[eid]["members"]
        if sum(1 for j in kset if j in S.entities[eid]["mset"]) < 3:
            continue
        emb = S.region_emb(TYPE_ENTITY, eid)
        surp = S._surprise_val(kset, len(known), TYPE_ENTITY, eid)
        bad = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, surp, FID_DATA, 0.0, emb, 0.0),
               (TYPE_ENTITY, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, SP.CENTERED_FOLD["hated"], emb, 0.0)]
        z_bad = fold_np(FR, m31, bad, [])
        z_never = fold_np(FR, m31, [], [])
        diffs.append(_region_score(FR, z_bad, members) - _region_score(FR, z_never, members))
    ci = _boot_ci(diffs)
    return dict(mean=float(np.mean(diffs)), ci=ci, n=len(diffs), **{"pass": bool(ci[0] > 0)})


def p3_G2b(FR, S, m31, users):
    diffs = []
    pop = np.argsort(-S.cnt)[:4000].astype(np.int64)
    for u in users:
        known = u["known"]; kset = set(int(j) for j in known)
        eid = _top_region(S, known, TYPE_ENTITY)
        if eid is None:
            continue
        e = S.entities[eid]; members = e["members"]; xm = [int(j) for j in members]
        if len(xm) < 3:
            continue
        emb = S.region_emb(TYPE_ENTITY, eid)
        sel_known = set(xm)
        surp_sel = S._surprise_val(sel_known, len(sel_known), TYPE_ENTITY, eid)
        prol_known = sel_known | (set(int(j) for j in pop) - set(xm))
        surp_pro = S._surprise_val(prol_known, len(prol_known), TYPE_ENTITY, eid)
        t_sel = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, surp_sel, FID_DATA, 0.0, emb, 0.0)]
        t_pro = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, surp_pro, FID_DATA, 0.0, emb, 0.0)]
        z_sel = fold_np(FR, m31, t_sel, []); z_pro = fold_np(FR, m31, t_pro, [])
        diffs.append(_region_score(FR, z_sel, members) - _region_score(FR, z_pro, members))
    ci = _boot_ci(diffs)
    return dict(mean=float(np.mean(diffs)), ci=ci, n=len(diffs), **{"pass": bool(ci[0] > 0)})


def p3_G7(FR, S, m31, users):
    rng = np.random.default_rng(9)
    drops = []
    was = S.oversample_famref; S.oversample_famref = False
    for u in users:
        known = u["known"]; held = u["held"]; prof = set(known.keys())
        budget = int(rng.integers(6, 20))
        toks, native = S.build_reveal(known, "interview", budget, rng, cache=u.get("cache"))
        if not toks:
            continue
        z_full = fold_np(FR, m31, toks, native, impl_ablate=False)
        z_abl = fold_np(FR, m31, toks, native, impl_ablate=True)
        a = L.ndcg10(FR, z_full, held, prof); b = L.ndcg10(FR, z_abl, held, prof)
        if None not in (a, b):
            drops.append(a - b)
    S.oversample_famref = was
    ci = _boot_ci(drops)
    return dict(mean=float(np.mean(drops)), ci=ci, n=len(drops), **{"pass": bool(ci[0] > 0)})


def p3_G4(FR, S, m31, users):
    rng = np.random.default_rng(4)
    fold_n, nat_n = [], []
    for u in users:
        known = u["known"]; held = u["held"]; prof = set(known.keys())
        toks, native = S.build_reveal(known, "clean", 0, rng, cache=u.get("cache"))
        zf = fold_np(FR, m31, toks, native)
        zn = FR.enc_items([list(known.keys())])[0].numpy().astype(np.float64)
        a = L.ndcg10(FR, zf, held, prof); b = L.ndcg10(FR, zn, held, prof)
        if None not in (a, b):
            fold_n.append(a); nat_n.append(b)
    mf, mn = float(np.mean(fold_n)), float(np.mean(nat_n))
    gap = mn - mf
    return dict(fold_ndcg=mf, native_ndcg=mn, gap=gap, n=len(fold_n), **{"pass": bool(abs(gap) < 0.05)})


V3_REF = {"G2": 0.1843, "G2b": 0.3159, "G7": 0.0203, "G4_gap": -0.0175}


def probe_P3(FR, S, m31, users):
    print("\n---- P3 NON-REGRESSION (quick G2/G2b/G7/G4 on v3.1 vs v3 reference) ----", flush=True)
    R = dict(G2=p3_G2(FR, S, m31, users), G2b=p3_G2b(FR, S, m31, users),
             G7=p3_G7(FR, S, m31, users), G4=p3_G4(FR, S, m31, users))
    print(f"  G2  GoT      v31 {R['G2']['mean']:+.4f} CI[{R['G2']['ci'][0]:+.4f},{R['G2']['ci'][1]:+.4f}]"
          f"  (v3 {V3_REF['G2']:+.4f})  {'PASS' if R['G2']['pass'] else 'FAIL'}", flush=True)
    print(f"  G2b prolific v31 {R['G2b']['mean']:+.4f} CI[{R['G2b']['ci'][0]:+.4f},{R['G2b']['ci'][1]:+.4f}]"
          f"  (v3 {V3_REF['G2b']:+.4f})  {'PASS' if R['G2b']['pass'] else 'FAIL'}", flush=True)
    print(f"  G7  impl-abl v31 {R['G7']['mean']:+.4f} CI[{R['G7']['ci'][0]:+.4f},{R['G7']['ci'][1]:+.4f}]"
          f"  (v3 {V3_REF['G7']:+.4f})  {'PASS' if R['G7']['pass'] else 'FAIL'}", flush=True)
    print(f"  G4  clean    v31 gap {R['G4']['gap']:+.4f}  (v3 {V3_REF['G4_gap']:+.4f})  "
          f"{'PASS' if R['G4']['pass'] else 'FAIL'}", flush=True)
    R["all_pass"] = bool(all(R[k]["pass"] for k in ("G2", "G2b", "G7", "G4")))
    return R


def cmd_probes(args):
    t0 = time.time()
    print("[probes] loading ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = V31Sampler(D, FR)
    prof = L.load_train_profiles(D, args.n_users + args.n_val, seed=args.seed)
    keys = list(prof.keys()); rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    val_keys = set(keys[:args.n_val])
    val_users = V3.prep_users({u: prof[u] for u in val_keys}, np.random.default_rng(args.seed + 2), S)
    vg = val_users[:args.n_gate]
    m31, wstate = load_v31(FR, BEST)
    m3, _ = V3.load_winner(FR, V3.WINNER, V3.FoldV3)
    print(f"[probes] v3.1 clean_frac={wstate.get('clean_frac')} best_val={wstate.get('best_val')}; "
          f"{len(vg)} probe users (disjoint val cohort)", flush=True)
    # print pre-registered gates FIRST
    print("\n=== PRE-REGISTERED PROBES (printed before results) ===", flush=True)
    print("  P1 author probe: no-clue on EXPECTED entity (high anti) pulls AWAY (neg, CI excl 0); "
          "obscure ~0.", flush=True)
    print("  P2 DECISION: on realistic T=24 interviews, does (c) v3.1 fold-noclue-with-anti BEAT "
          "(a) v3 skip-noclue? paired CI excl 0.", flush=True)
    print("  P3 non-regression: v3.1 must keep G2/G2b/G7 PASS (CI excl 0) and G4 |gap|<0.05.",
          flush=True)
    P1 = probe_P1(FR, S, m31, vg)
    P2 = probe_P2(FR, S, m31, m3, val_users[:args.n_p2])
    P3 = probe_P3(FR, S, m31, vg)
    curr = json.load(open(".cache/i25_fold_v31_curriculum.json")) if \
        os.path.exists(".cache/i25_fold_v31_curriculum.json") else {}
    R = dict(winner=dict(clean_frac=wstate.get("clean_frac"), best_val=wstate.get("best_val"),
                         best_epoch=wstate.get("best_epoch")),
             curriculum=curr, P1=P1, P2=P2, P3=P3)
    json.dump(R, open(PROBES_JSON, "w"), indent=1, default=float)
    _write_md(R)
    print(f"\n[probes] wrote {PROBES_JSON} + {ABL_MD}  [{time.time()-t0:.0f}s]", flush=True)
    return R


# =============================================================== NOCLUE_ABLATION.md writer
def _ci(c):
    return f"[{c[0]:+.4f},{c[1]:+.4f}]"


def _write_md(R):
    P1, P2, P3, curr = R["P1"], R["P2"], R["P3"], R.get("curriculum", {})
    decision = P2["decision_c_beats_a"]
    rec = ("FOLD no-clue WITH v3.1 anti-surprise (CHANGE the arena rule)" if decision
           else "SKIP no-clue folds (the current arena rule is empirically justified)")
    o = []
    o.append("# NO-CLUE ABLATION -- FOLD-V3.1 anti-surprise\n\n")
    o.append("Author-directed night task. Tests whether no-clue / refusal tokens can carry genuine "
             "NEGATIVE implicit signal before finalizing the decision to skip folding them. "
             "NO LLM calls; population trU users only (the 173 study/eval users untouched); "
             "deterministic; reduced threads.\n\n")
    o.append("## Hypothesis (author, verbatim)\n\n")
    o.append("> no-clue is informative exactly when knowledge was EXPECTED -- refusing a famous item "
             "your volume predicts you'd know = real negative taste/territory evidence; refusing "
             "obscurity = nothing. The current sampler gives no-clue tokens the same SURPRISE feature "
             "as consumption (low for real no-clues) -- the fold cannot express \"expected but "
             "absent\".\n\n")
    o.append("## ANTI-SURPRISE feature (formula)\n\n")
    o.append("New scalar input lit ONLY on no-clue (LVL_NEG implicit) tokens = expectedness of "
             "knowing:\n\n")
    o.append("```\n")
    o.append("E[n_E] = V * p_E            # volume-predicted engagement (fame x volume)\n")
    o.append("           p_E = popmass(E) / total_popmass ;  V = revealed volume (len known)\n")
    o.append("n_E    = # revealed members of E   (0 for a genuine refusal)\n")
    o.append("ANTI   = clip( log( (E[n_E] + 0.5) / (n_E + 0.5) ), 0, 8 )\n")
    o.append("```\n\n")
    o.append("HIGH when a famous/expected entity is refused (large E[n_E], n_E=0); ~0 for obscure "
             "refusals. Equals -surprise, but as a DEDICATED feature lit only on no-clue tokens it "
             "decouples \"expected-but-absent\" from the consumption-surprise weight (v3 found the "
             "LVL_NEG level flag inert when it shared the surprise feature).\n\n")
    o.append("## Curriculum (informative-refusal exposure)\n\n")
    nat = curr.get("natural", {})
    o.append(f"Natural v2.1-answerer interview (T=24, oversampling OFF): "
             f"{nat.get('neg_per_int', float('nan')):.3f} no-clue tokens/interview, "
             f"{nat.get('expected_per_int', float('nan')):.3f} EXPECTED (anti>={EXPECTED_ANTI_THR})"
             f"/interview; anti mean {nat.get('anti_mean', float('nan')):.3f}, "
             f"p90 {nat.get('anti_p90', float('nan')):.3f}. "
             f"Starved (< {curr.get('threshold', STARVED_PER_INTERVIEW)}/int): {curr.get('starved')}. "
             f"Oversampling famous-off-profile probes: {curr.get('oversample')} "
             f"(p={curr.get('p_famref')}). Oversampling lets the v2.1 knowledge model DECIDE each "
             "refusal (realistic); P2 evaluation uses NATURAL rates (oversampling OFF).\n\n")
    w = R["winner"]
    o.append(f"Winner checkpoint `{BEST}`: clean_frac {w['clean_frac']}, val NDCG@10 "
             f"{w['best_val']:.4f} @ep{w['best_epoch']}.\n\n")

    o.append("## P1 -- author probe (no-clue pull vs no token; region-score)\n\n")
    o.append("| case | anti (mean) | pull vs no-token | 95% CI | n | verdict |\n")
    o.append("|---|--:|--:|---|--:|:--:|\n")
    e, ob = P1["expected"], P1["obscure"]
    o.append(f"| EXPECTED (famous refused) | {e['anti']:.2f} | {e['mean']:+.4f} | {_ci(e['ci'])} | "
             f"{e['n']} | {'NEG (CI<0)' if e['pass'] else 'FAIL'} |\n")
    o.append(f"| OBSCURE (obscure refused) | {ob['anti']:.2f} | {ob['mean']:+.4f} | {_ci(ob['ci'])} | "
             f"{ob['n']} | {'~0' if ob['pass'] else 'FAIL'} |\n")
    o.append("\nDirect test of the author's hypothesis: EXPECTED row must be negative with CI excl 0; "
             "OBSCURE row must be ~0. Interpret each sub-verdict separately (see any hand-written "
             "reading notes below if present).\n\n")

    o.append("## P2 -- interview-level justification (T=24, natural k=0). THE DECISION NUMBER\n\n")
    o.append(f"n={P2['n']} DEV users; {P2['n_neg_tokens']} no-clue tokens "
             f"({P2['n_users_with_neg']} users had >=1).\n\n")
    o.append("| condition | NDCG@10 | delta vs (a) | 95% CI |\n|---|--:|--:|---|\n")
    o.append(f"| (a) v3 SKIP no-clue (arena rule) | {P2['ndcg_a']:.4f} | -- | -- |\n")
    o.append(f"| (b) v3 FOLD no-clue (as-trained) | {P2['ndcg_b']:.4f} | {P2['d_ba']:+.4f} | "
             f"{_ci(P2['ci_ba'])} |\n")
    o.append(f"| (c') v3.1 SKIP no-clue | {P2['ndcg_cskip']:.4f} | -- | -- |\n")
    o.append(f"| **(c) v3.1 FOLD no-clue + anti** | **{P2['ndcg_c']:.4f}** | **{P2['d_ca']:+.4f}** | "
             f"**{_ci(P2['ci_ca'])}** |\n")
    o.append(f"\nWithin-v3.1 fold-vs-skip (c - c'): {P2['d_ccs']:+.4f} CI {_ci(P2['ci_ccs'])}. "
             f"(c) vs (b): {P2['d_cb']:+.4f} CI {_ci(P2['ci_cb'])}.\n\n")
    o.append(f"**DECISION -- does (c) beat (a)? {P2['decision_c_beats_a']}** "
             f"(delta {P2['d_ca']:+.4f}, CI {_ci(P2['ci_ca'])}).\n\n")

    o.append("## P3 -- non-regression of the decisive v3 gates\n\n")
    o.append("| gate | v3.1 | 95% CI | v3 ref | verdict |\n|---|--:|---|--:|:--:|\n")
    o.append(f"| G2 GoT | {P3['G2']['mean']:+.4f} | {_ci(P3['G2']['ci'])} | {V3_REF['G2']:+.4f} | "
             f"{'PASS' if P3['G2']['pass'] else 'FAIL'} |\n")
    o.append(f"| G2b prolific | {P3['G2b']['mean']:+.4f} | {_ci(P3['G2b']['ci'])} | "
             f"{V3_REF['G2b']:+.4f} | {'PASS' if P3['G2b']['pass'] else 'FAIL'} |\n")
    o.append(f"| G7 implicit-abl | {P3['G7']['mean']:+.4f} | {_ci(P3['G7']['ci'])} | "
             f"{V3_REF['G7']:+.4f} | {'PASS' if P3['G7']['pass'] else 'FAIL'} |\n")
    o.append(f"| G4 clean gap | {P3['G4']['gap']:+.4f} | (fold {P3['G4']['fold_ndcg']:.4f} vs native "
             f"{P3['G4']['native_ndcg']:.4f}) | {V3_REF['G4_gap']:+.4f} | "
             f"{'PASS' if P3['G4']['pass'] else 'FAIL'} |\n")
    o.append(f"\nP3 all-pass: {P3['all_pass']}.\n\n")

    o.append("## VERDICT + RECOMMENDATION\n\n")
    p1ok = P1["expected"]["pass"] and P1["obscure"]["pass"]
    o.append(f"- P1 (author probe): {'CONFIRMED' if p1ok else 'PARTIAL/NOT confirmed'} -- expected-"
             f"refusal pull {e['mean']:+.4f} CI {_ci(e['ci'])} "
             f"({'neg, CI excl 0' if e['pass'] else 'sub-test FAIL'}); obscure-refusal pull "
             f"{ob['mean']:+.4f} ({'~0 ok' if ob['pass'] else 'NOT ~0, sub-test FAIL'}).\n")
    o.append(f"- P2 (decision): (c) beats (a) = {decision} (delta {P2['d_ca']:+.4f} CI "
             f"{_ci(P2['ci_ca'])}); retrain component (c')-(a) = "
             f"{P2['ndcg_cskip']-P2['ndcg_a']:+.4f}, fold component (c)-(c') = {P2['d_ccs']:+.4f} "
             f"CI {_ci(P2['ci_ccs'])}.\n")
    o.append(f"- P3 (non-regression): all-pass = {P3['all_pass']}.\n\n")
    final_ok = decision and P3["all_pass"]
    o.append(f"**RECOMMENDATION: {rec}.**")
    if decision and not P3["all_pass"]:
        o.append(" NOTE: P2 favors folding but P3 regressed a decisive gate -> the variant is DEAD; "
                 "keep the SKIP rule.")
    o.append(f" (P2 decision={decision}, P3 all-pass={P3['all_pass']}, net actionable={final_ok}.)\n")
    open(ABL_MD, "w", encoding="utf-8").write("".join(o))
    assert os.path.exists(ABL_MD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "probes", "smoke"])
    ap.add_argument("--n_users", type=int, default=25000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--n_gate", type=int, default=400)
    ap.add_argument("--n_p2", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "probes":
        cmd_probes(a)
    elif a.cmd == "smoke":
        a.n_users = 500; a.n_val = 150; a.epochs = 1; a.n_gate = 60; a.n_p2 = 100
        cmd_train(a)
        cmd_probes(a)
