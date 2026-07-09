"""i25_fold_v3.py -- FOLD-V3: the TWO-CHANNEL belief encoder (per DESIGN_SHEET_FOLD_V3.md, signed
2026-07-10 incl. the PROLIFIC/SURPRISE author amendment + gate G2b).

Deep-Sets residual around the FROZEN RecVAE-d512 decoder (v2 recipe), extended so every answer speaks
BOTH channels: an IMPLICIT token (knows E at level {rough,know_well} + a SURPRISE lift feature -->
consumption-as-taste) and an EXPLICIT token (values E at v, fidelity {data,ease,llm-style}). The blend
is LEARNED (D1: two separate tokens). Trained on the v2.1-answerer reveal distribution over POPULATION
trU users (sampler = i25_fold_v3_sampler), MIXED CURRICULUM (partial interviews 1-24 answers + clean
full profiles), curriculum ratio swept {30/70, 50/50}, best-on-val wins.

NO LLM calls. Firewall: population trU users only (the 173 study/eval users are NEVER touched).
Checkpoints every epoch -> .cache/i25_fold_v3*. Deterministic. STOP after gates (no arena/policies).

Run:
  python scripts/i25_fold_v3.py train  --n_users 25000 --epochs 12          # sweeps both ratios
  python scripts/i25_fold_v3.py gates                                        # gate battery on the winner
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import warnings
warnings.filterwarnings("ignore")
import i25_lib as L
import i25_fold_v3_sampler as SP
from i25_fold_v3_sampler import (NTYPE, KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG,
                                  FID_DATA, FID_EASE, FID_LLM, TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR,
                                  TYPE_ENTITY, V3Sampler)

D_LAT = L.D_LAT
CKPT_TPL = ".cache/i25_fold_v3_c{cf}.pt"
BEST_TPL = ".cache/i25_fold_v3_c{cf}_best.pt"
WINNER = ".cache/i25_fold_v3_best.pt"
LOG_TPL = ".cache/i25_fold_v3_c{cf}_log.json"
BUILD_MD = "experiments/FOLD_V3_BUILD.md"
GATES_JSON = ".cache/i25_fold_v3_gates.json"
SWEEP_JSON = ".cache/i25_fold_v3_sweep.json"
CLEAN_FRACS = [30, 50]          # clean-profile % in the mixed curriculum (30/70 and 50/50)
VAL_CLEAN_FRAC = 0.5            # FIXED 50/50 val mix for cross-run model selection (comparable)
MAX_CLEAN_ITEMS = 50           # cap clean-mode per-item tokens for throughput (G4 gate uses full)


def _md(txt, mode="a"):
    open(BUILD_MD, mode, encoding="utf-8").write(txt)


# ================================================================= the v3 fold
class FoldV3(nn.Module):
    """Residual Deep-Sets fold with SEPARATE implicit/explicit tokens (D1). Per-token input =
    [type_oh(4), kind_oh(2), lvl_oh(3), surprise(1), fid_oh(3), value(1), emb(d), value*emb(d)].
    lvl_oh lit only for implicit tokens; fid_oh only for explicit. z = native_z + rho(pool,...)."""

    def __init__(self, d=D_LAT):
        super().__init__()
        in_dim = NTYPE + 2 + 3 + 1 + 3 + 1 + d + d
        self.phi = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)   # residual: start native
        self.d = d

    def forward(self, tt, tk, tl, ts, tf, tv, te, mask, native_z, impl_ablate=False):
        B, K, d = te.shape
        is_impl = (tk == KIND_IMPL).to(te.dtype)                    # (B,K)
        is_expl = 1.0 - is_impl
        type_oh = F.one_hot(tt.clamp(min=0), NTYPE).to(te.dtype)
        kind_oh = F.one_hot(tk.clamp(min=0), 2).to(te.dtype)
        lvl_oh = F.one_hot(tl.clamp(min=0), 3).to(te.dtype) * is_impl.unsqueeze(-1)
        fid_oh = F.one_hot(tf.clamp(min=0), 3).to(te.dtype) * is_expl.unsqueeze(-1)
        surprise = ts.unsqueeze(-1)
        value = tv.unsqueeze(-1)
        ve = value * te
        x = torch.cat([type_oh, kind_oh, lvl_oh, surprise, fid_oh, value, te, ve], dim=-1)
        eff_mask = mask * (is_expl if impl_ablate else torch.ones_like(mask))
        h = self.phi(x) * eff_mask.unsqueeze(-1)
        pool = h.sum(1)
        ntok = eff_mask.sum(1, keepdim=True)
        delta = self.rho(torch.cat([pool, native_z, torch.log1p(ntok)], dim=-1))
        return native_z + delta


def _token_x(tt, tk, tl, ts, tf, tv, te):
    """Shared per-token feature tensor + is_impl mask (used by Deep-Sets and Set-Transformer)."""
    is_impl = (tk == KIND_IMPL).to(te.dtype)
    is_expl = 1.0 - is_impl
    type_oh = F.one_hot(tt.clamp(min=0), NTYPE).to(te.dtype)
    kind_oh = F.one_hot(tk.clamp(min=0), 2).to(te.dtype)
    lvl_oh = F.one_hot(tl.clamp(min=0), 3).to(te.dtype) * is_impl.unsqueeze(-1)
    fid_oh = F.one_hot(tf.clamp(min=0), 3).to(te.dtype) * is_expl.unsqueeze(-1)
    ve = tv.unsqueeze(-1) * te
    x = torch.cat([type_oh, kind_oh, lvl_oh, ts.unsqueeze(-1), fid_oh, tv.unsqueeze(-1), te, ve], dim=-1)
    return x, is_impl


class FoldV3ST(nn.Module):
    """G6 CONTINGENCY (D3): Set-Transformer pooling (masked self-attention block + pooling-by-multihead-
    attention) replacing the Deep-Sets sum-pool. Same per-token features, same residual (z=native_z+rho).
    Set-Transformer can weight tokens by count/context, the anti-saturation remedy the sheet names."""

    IN_DIM = NTYPE + 2 + 3 + 1 + 3 + 1 + D_LAT + D_LAT

    def __init__(self, d=D_LAT, heads=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(self.IN_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.sab = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pma = nn.MultiheadAttention(d, heads, batch_first=True)
        self.rho = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)
        self.d = d

    def forward(self, tt, tk, tl, ts, tf, tv, te, mask, native_z, impl_ablate=False):
        B, K, d = te.shape
        x, is_impl = _token_x(tt, tk, tl, ts, tf, tv, te)
        is_expl = 1.0 - is_impl
        eff_mask = mask * (is_expl if impl_ablate else torch.ones_like(mask))
        h = self.proj(x) * eff_mask.unsqueeze(-1)
        kpm = eff_mask == 0                                   # True = ignore (pad/ablated)
        kpm = kpm.clone(); kpm[:, 0] = False                 # keep >=1 valid key to avoid NaN on empty
        a, _ = self.sab(h, h, h, key_padding_mask=kpm); h = self.ln1(h + a)
        h = self.ln2(h + self.ff(h))
        seed = self.seed.expand(B, -1, -1)
        p, _ = self.pma(seed, h, h, key_padding_mask=kpm); pool = p.squeeze(1)
        ntok = eff_mask.sum(1, keepdim=True)
        pool = pool * (ntok > 0).to(pool.dtype)              # empty set -> zero pool -> native prior
        delta = self.rho(torch.cat([pool, native_z, torch.log1p(ntok)], dim=-1))
        return native_z + delta


def pack_batch(FR, tok_lists, native_lists, device="cpu"):
    B = len(tok_lists)
    K = max((len(t) for t in tok_lists), default=1); K = max(K, 1)
    d = FR.W.shape[1]
    tt = torch.zeros((B, K), dtype=torch.long)
    tk = torch.zeros((B, K), dtype=torch.long)
    tl = torch.zeros((B, K), dtype=torch.long)
    ts = torch.zeros((B, K), dtype=torch.float32)
    tf = torch.zeros((B, K), dtype=torch.long)
    tv = torch.zeros((B, K), dtype=torch.float32)
    te = torch.zeros((B, K, d), dtype=torch.float32)
    mask = torch.zeros((B, K), dtype=torch.float32)
    for b, toks in enumerate(tok_lists):
        for k, (typ, kind, lvl, surp, fid, val, emb) in enumerate(toks):
            tt[b, k] = typ; tk[b, k] = kind; tl[b, k] = lvl; ts[b, k] = surp
            tf[b, k] = fid; tv[b, k] = val; te[b, k] = torch.as_tensor(emb); mask[b, k] = 1.0
    nz = FR.enc_items(native_lists)
    return (tt.to(device), tk.to(device), tl.to(device), ts.to(device), tf.to(device),
            tv.to(device), te.to(device), mask.to(device), nz.to(device))


def fold_batch(FR, model, tok_lists, native_lists, impl_ablate=False):
    if not tok_lists:
        return np.zeros((0, FR.W.shape[1]))
    args = pack_batch(FR, tok_lists, native_lists)
    with torch.no_grad():
        z = model(*args, impl_ablate=impl_ablate)
    return z.numpy().astype(np.float64)


def fold_np(FR, model, toks, native, impl_ablate=False):
    return fold_batch(FR, model, [toks], [native], impl_ablate=impl_ablate)[0].astype(np.float64)


# ================================================================= user prep (v2 recipe)
def make_user_split(prof, rng):
    its = list(prof.keys()); rng.shuffle(its)
    h = len(its) // 2
    known = {j: prof[j] for j in its[:h]}
    held = set(j for j in its[h:] if prof[j] >= 4)
    return known, held


def prep_users(prof_dict, rng, S=None):
    out = []
    for u, prof in prof_dict.items():
        known, held = make_user_split(prof, rng)
        if held and len(known) >= 2:
            rec = dict(u=u, known={int(j): float(r) for j, r in known.items()}, held=held)
            if S is not None:
                rec["cache"] = S.make_cache(rec["known"])
            out.append(rec)
    return out


# ================================================================= training
def batch_reveals(S, users, clean_frac, rng):
    tl, nl, tgt, prof = [], [], [], []
    for u in users:
        mode = "clean" if rng.random() < clean_frac else "interview"
        budget = int(rng.integers(1, 25))
        toks, native = S.build_reveal(u["known"], mode, budget, rng, cache=u.get("cache"),
                                      max_items=MAX_CLEAN_ITEMS)
        if not toks:
            continue
        tl.append(toks); nl.append(native); tgt.append(u["held"]); prof.append(set(u["known"].keys()))
    return tl, nl, tgt, prof


def batch_loss(FR, model, S, users, clean_frac, rng):
    tl, nl, tgt, prof = batch_reveals(S, users, clean_frac, rng)
    if not tl:
        return None
    tt, tk, tll, ts, tf, tv, te, mask, nz = pack_batch(FR, tl, nl)
    z = model(tt, tk, tll, ts, tf, tv, te, mask, nz)
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


def train_one(FR, S, D, tr_users, val_users, clean_frac, args, model_cls=FoldV3, tag=""):
    cf = int(round(clean_frac * 100))
    ckpt = CKPT_TPL.format(cf=f"{tag}{cf}"); best = BEST_TPL.format(cf=f"{tag}{cf}")
    logp = LOG_TPL.format(cf=f"{tag}{cf}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    state = dict(epoch=0, best_val=-1.0, best_epoch=-1, history=[], clean_frac=clean_frac)
    step_rng = np.random.default_rng(args.seed + 100 + cf)
    t0 = time.time()
    print(f"\n[v3 c{cf}] === training clean_frac={clean_frac:.2f} ({cf}/{100-cf} clean/interview) ===",
          flush=True)
    for ep in range(args.epochs):
        model.train()
        order = np.arange(len(tr_users)); step_rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), args.batch):
            us = [tr_users[i] for i in order[b0:b0 + args.batch]]
            loss = batch_loss(FR, model, S, us, clean_frac, step_rng)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        vN = val_ndcg(FR, model, S, val_users, VAL_CLEAN_FRAC, args.seed + 7)
        state["epoch"] = ep + 1
        rec = dict(epoch=ep + 1, train_loss=round(tot / max(nb, 1), 4), val_ndcg=round(vN, 4),
                   min=round((time.time() - t0) / 60, 1))
        state["history"].append(rec)
        print(f"[v3 c{cf}] ep {ep+1:2d} | loss {rec['train_loss']:.4f} | val NDCG@10 {vN:.4f} | "
              f"{rec['min']}m", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state,
                        args=vars(args)), ckpt)              # checkpoint EVERY epoch
        if vN > state["best_val"]:
            state["best_val"] = vN; state["best_epoch"] = ep + 1
            torch.save(dict(model=model.state_dict(), state=state, args=vars(args)), best)
        json.dump(state["history"], open(logp, "w"), indent=1)
    print(f"[v3 c{cf}] BEST val {state['best_val']:.4f} @ep{state['best_epoch']} -> {best}", flush=True)
    return dict(clean_frac=clean_frac, cf=cf, best_val=state["best_val"],
                best_epoch=state["best_epoch"], best_ckpt=best, history=state["history"])


WINNER_ST = ".cache/i25_fold_v3_st_best.pt"


def cmd_st(args):
    """G6 contingency: train ONE Set-Transformer variant at the winning curriculum, then re-gate."""
    t0 = time.time()
    print("[v3-ST] G6 CONTINGENCY: training Set-Transformer variant ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = V3Sampler(D, FR)
    prof = L.load_train_profiles(D, args.n_users + args.n_val, seed=args.seed)
    keys = list(prof.keys()); rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    val_keys = set(keys[:args.n_val]); tr_keys = keys[args.n_val:]
    tr_users = prep_users({u: prof[u] for u in tr_keys}, np.random.default_rng(args.seed + 1), S)
    val_users = prep_users({u: prof[u] for u in val_keys}, np.random.default_rng(args.seed + 2), S)
    # winning curriculum from the Deep-Sets sweep
    sweep = json.load(open(SWEEP_JSON))
    cf_win = max(sweep, key=lambda r: r["best_val"])["cf"]
    print(f"[v3-ST] winning curriculum clean_frac={cf_win/100:.2f}; {len(tr_users)} train users", flush=True)
    r = train_one(FR, S, D, tr_users, val_users, cf_win / 100.0, args, model_cls=FoldV3ST, tag="st")
    import shutil
    shutil.copyfile(r["best_ckpt"], WINNER_ST)
    print(f"[v3-ST] best val {r['best_val']:.4f} -> {WINNER_ST}  ({round((time.time()-t0)/60,1)}m)",
          flush=True)
    # re-gate G6 (and full battery) on the ST variant
    R = run_gates(FR, S, D, FoldV3ST, WINNER_ST, val_users[:args.n_gate], tag="ST")
    json.dump(R, open(".cache/i25_fold_v3_gates_st.json", "w"), indent=1, default=float)
    _write_st_md(r, R)
    return R


def cmd_train(args):
    t0 = time.time()
    print("[v3] loading data + frozen RecVAE ...", flush=True)
    D = L.G.load_data()
    FR = L.Frozen(D)
    S = V3Sampler(D, FR)
    print(f"[v3] sampler ready: {S.n_concept} concepts, {len(S.attr_keys)} attrs "
          f"({len(S.decades)} decades), {len(S.entity_ids)} IMDb entities", flush=True)
    prof = L.load_train_profiles(D, args.n_users + args.n_val, seed=args.seed)
    keys = list(prof.keys())
    rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    val_keys = set(keys[:args.n_val]); tr_keys = keys[args.n_val:]
    tr_users = prep_users({u: prof[u] for u in tr_keys}, np.random.default_rng(args.seed + 1), S)
    val_users = prep_users({u: prof[u] for u in val_keys}, np.random.default_rng(args.seed + 2), S)
    print(f"[v3] train users {len(tr_users)}  val users {len(val_users)}  "
          f"(load {round(time.time()-t0,1)}s)", flush=True)

    # ---- config section of the build doc ----
    _write_config_md(D, S, args, len(tr_users), len(val_users))

    results = []
    for cf in CLEAN_FRACS:
        r = train_one(FR, S, D, tr_users, val_users, cf / 100.0, args)
        results.append(r)
        json.dump(results, open(SWEEP_JSON, "w"), indent=1)
    winner = max(results, key=lambda r: r["best_val"])
    import shutil
    shutil.copyfile(winner["best_ckpt"], WINNER)
    print(f"\n[v3] CURRICULUM WINNER = clean_frac {winner['clean_frac']:.2f} "
          f"(val {winner['best_val']:.4f}) -> {WINNER}", flush=True)
    _write_sweep_md(results, winner)
    return results, winner


# ================================================================= build-doc writers
def _write_config_md(D, S, args, ntr, nval):
    _md("# FOLD-V3 BUILD -- the two-channel belief encoder\n\n", mode="w")
    _md("Per DESIGN_SHEET_FOLD_V3.md (signed 2026-07-10, incl. PROLIFIC/SURPRISE author amendment + "
        "gate G2b). Deep-Sets residual around the FROZEN RecVAE-d512 decoder. Two SEPARATE tokens per "
        "answer (D1): IMPLICIT (channel, entity, level in {rough,know_well} + SURPRISE lift) and "
        "EXPLICIT (channel, entity, 4-level value, fidelity {data,ease,llm-style}); no_clue-on-asked "
        "emits an implicit-NEGATIVE token. Sampler = v2.1 answerer distribution on POPULATION trU users "
        "(the 173 study/eval users are NEVER touched). NO LLM calls.\n\n")
    _md("## Config\n\n")
    _md(f"- data: ni={D['ni']} items; train users {ntr}, disjoint val users {nval} "
        f"(n_users={args.n_users}, n_val={args.n_val}, seed={args.seed})\n")
    _md(f"- vocab: {S.n_concept} concepts, {len(S.attr_keys)} attributes "
        f"({len(S.decades)} decades + {len(S.attr_keys)-len(S.decades)} genres), "
        f"{len(S.entity_ids)} IMDb entities (director/actor/composer/writer/franchise; "
        "prominence-weighted member-bag embeddings)\n")
    _md(f"- SURPRISE feature = log((n_E+0.5)/(V*p_E+0.5)); p_E = popmass(E)/total popmass; n_E = "
        "#revealed members of E; V = revealed volume (author amendment: engagement vs volume-"
        "predicted expectation)\n")
    _md(f"- knowledge model (v2.1 answerer on population users): P(knows)=sigmoid(logit(base_ans)+"
        f"{SP.K_ANS}*surprise+trait_u), P(kw|ans)=sigmoid(logit(base_kw)+{SP.K_KW}*surprise+trait_u); "
        "base rates from battery A1/A3 (item .95/.70, concept .74/.52, attr .82/.36, entity .55/.36); "
        "trait_u ~ N(0, sigma_equated) with sigma from .cache/dans/sigma_v21.json "
        f"(concept {S.trait_sigma[TYPE_CONCEPT]:.3f}, entity {S.trait_sigma[TYPE_ENTITY]:.3f}, "
        f"item {S.trait_sigma[TYPE_ITEM]:.3f}) = the CORRECTED (flutter-free) trait dial\n")
    _md(f"- value model (real U EASE): rated item->real centered rating (fid=data); inferred "
        "concept/attr/entity aggregate over revealed members (real ratings), binned 4-level; unrated-item "
        "base = EASE per-item mean (ease_v21.npz). Fidelity/noise tied to level: know_well->ease "
        f"(sigma {0.25*SP.SIGMA_STAR:.3f}), rough->llm-style (sigma {SP.SIGMA_STAR:.3f})\n")
    _md(f"- architecture: Deep-Sets residual (v2 recipe); frozen decoder; all checkpoints; disjoint val; "
        f"epochs={args.epochs}, batch={args.batch}, lr={args.lr}\n")
    _md(f"- curriculum sweep (D2): clean-profile fraction in {{{'/'.join(str(c)+'%' for c in CLEAN_FRACS)}}} "
        f"(2 runs), model selection on a FIXED 50/50 val mix; best-on-val wins\n\n")
    assert os.path.exists(BUILD_MD)


def _write_sweep_md(results, winner):
    _md("## Curriculum sweep result\n\n")
    _md("| clean_frac | best val NDCG@10 | best epoch | ckpt |\n|---|--:|--:|---|\n")
    for r in results:
        star = " **<-- WINNER**" if r is winner else ""
        _md(f"| {r['clean_frac']:.2f} ({r['cf']}/{100-r['cf']}) | {r['best_val']:.4f} | "
            f"{r['best_epoch']} | `{r['best_ckpt']}`{star} |\n")
    _md(f"\n**Winner: clean_frac {winner['clean_frac']:.2f}, val NDCG@10 {winner['best_val']:.4f}** "
        f"-> `{WINNER}`. Gates run on this checkpoint.\n\n")


# ================================================================= gate battery
def _boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def load_winner(FR, ckpt=WINNER, model_cls=FoldV3):
    blob = torch.load(ckpt, map_location="cpu")
    m = model_cls(); m.load_state_dict(blob["model"]); m.eval()
    return m, blob.get("state", {})


def run_gates(FR, S, D, model_cls, ckpt, vg, tag=""):
    t0 = time.time()
    model, wstate = load_winner(FR, ckpt, model_cls)
    lbl = f" [{tag}]" if tag else ""
    print(f"\n[gates{lbl}] {model_cls.__name__} clean_frac={wstate.get('clean_frac')} "
          f"best_val={wstate.get('best_val')}; {len(vg)} gate users (disjoint val cohort)", flush=True)
    _print_thresholds()
    R = {}
    R["G1"] = gate_G1(FR, model, S, vg)
    R["G2"] = gate_G2(FR, model, S, vg)
    R["G2b"] = gate_G2b(FR, model, S, vg)
    R["G3"] = gate_G3(FR, model, S, vg)
    R["G4"] = gate_G4(FR, model, S, vg)
    R["G5"] = gate_G5(FR, model, S, D, vg)
    R["G6"] = gate_G6(FR, model, S, vg)
    R["G7"] = gate_G7(FR, model, S, vg)
    R["implied_strengths"] = probe_implied_strengths(FR, model, S, vg)
    R["winner"] = dict(clean_frac=wstate.get("clean_frac"), best_val=wstate.get("best_val"),
                       best_epoch=wstate.get("best_epoch"), arch=model_cls.__name__)
    print(f"[gates{lbl}] battery done [{time.time()-t0:.0f}s]", flush=True)
    _print_verdicts(R)
    return R


def _print_thresholds():
    print("\n=== PRE-REGISTERED GATE THRESHOLDS (printed before results) ===", flush=True)
    print("  G1  canaries: per channel & token-type, one-token NDCG lift from cold > 0 (explicit AND "
          "implicit-only). PASS if lift>0 (report CI).", flush=True)
    print("  G2  GoT: watched-all-X-rated-BADLY pulled toward X MORE than never-heard. CI excl 0.", flush=True)
    print("  G2b PROLIFIC: SELECTIVE-all-X pulled toward X MORE than PROLIFIC-all-X-plus-everything. "
          "CI excl 0. (G2+G2b jointly decisive.)", flush=True)
    print("  G3  dilution: adding vague (rough/llm) tokens to a vivid set does not drop NDCG > 0.005.", flush=True)
    print("  G4  clean-profile: |fold(full) - native RecVAE| < 0.05.", flush=True)
    print("  G5  no-harm vs v2: on noisy interview reveals, v3 >= v2 - 0.005.", flush=True)
    print("  G6  anti-saturation: mean NDCG over budget 1->24, max per-step decline > -0.003 "
          "(else train ST contingency).", flush=True)
    print("  G7  implicit ablation: zeroing implicit tokens DROPS NDCG. CI excl 0.", flush=True)


def cmd_gates(args):
    t0 = time.time()
    print("[gates] loading ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = V3Sampler(D, FR)
    prof = L.load_train_profiles(D, args.n_users + args.n_val, seed=args.seed)
    keys = list(prof.keys()); rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    val_keys = set(keys[:args.n_val])
    val_users = prep_users({u: prof[u] for u in val_keys}, np.random.default_rng(args.seed + 2), S)
    vg = val_users[:args.n_gate]
    R = run_gates(FR, S, D, FoldV3, WINNER, vg, tag="")
    json.dump(R, open(GATES_JSON, "w"), indent=1, default=float)
    _write_gate_md(R)
    print(f"\n[gates] wrote {GATES_JSON} + {BUILD_MD}  [{time.time()-t0:.0f}s]", flush=True)
    return R


def _ndcg_of(FR, model, toks, native, held, prof, impl_ablate=False):
    z = fold_np(FR, model, toks, native, impl_ablate=impl_ablate)
    return L.ndcg10(FR, z, held, prof)


def _region_score(FR, z, members):
    S = FR.decode_np(z[None, :])[0]
    return float(np.mean(S[members]))


# ---- G1: cold one-token canaries per channel per token type ----
def gate_G1(FR, model, S, users):
    print("\n---- G1 CANARIES (cold one-token NDCG lift, per channel per token-type) ----", flush=True)
    rng = np.random.default_rng(1)
    out = {}
    for ch, name in [(TYPE_ITEM, "item"), (TYPE_CONCEPT, "concept"),
                     (TYPE_ATTR, "attr"), (TYPE_ENTITY, "entity")]:
        lifts_expl, lifts_impl = [], []
        for u in users:
            known = u["known"]; held = u["held"]; prof = set(known.keys())
            key = _top_region(S, known, ch)
            if key is None:
                continue
            mu = float(np.mean(list(known.values())))
            cr = {int(j): float(r - mu) for j, r in known.items()}
            z0 = fold_np(FR, model, [], [])                       # cold baseline
            n0 = L.ndcg10(FR, z0, held, prof)
            emb = S.region_emb(ch, key)
            surp = S._surprise_val(prof, len(known), ch, key)
            # explicit token (positive value)
            val, fid = S.region_value(known, cr, mu, ch, key, LVL_KW, rng)
            if val is None:
                val, fid = SP.CENTERED_FOLD["loved"], FID_EASE
            te = [(ch, KIND_EXPL, LVL_ROUGH, 0.0, fid, float(abs(val) if val else 0.7), emb)]
            ne = L.ndcg10(FR, fold_np(FR, model, te, []), held, prof)
            # implicit-only token (knows E, value withheld)
            ti = [(ch, KIND_IMPL, LVL_KW, surp, FID_DATA, 0.0, emb)]
            ni = L.ndcg10(FR, fold_np(FR, model, ti, []), held, prof)
            if None not in (n0, ne, ni):
                lifts_expl.append(ne - n0); lifts_impl.append(ni - n0)
        ce = _boot_ci(lifts_expl); ci = _boot_ci(lifts_impl)
        me, mi = float(np.mean(lifts_expl)), float(np.mean(lifts_impl))
        out[name] = dict(explicit_lift=me, explicit_ci=ce, implicit_lift=mi, implicit_ci=ci,
                         n=len(lifts_expl))
        print(f"  [{name:8s}] explicit lift {me:+.4f} CI[{ce[0]:+.4f},{ce[1]:+.4f}] | "
              f"implicit-only lift {mi:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}]  (n={len(lifts_expl)})",
              flush=True)
    passed = all(out[c]["explicit_lift"] > 0 and out[c]["implicit_lift"] > 0 for c in out)
    out["pass"] = bool(passed)
    return out


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


# ---- G2: the GoT gate ----
def gate_G2(FR, model, S, users):
    print("\n---- G2 GoT GATE (watched-all-X-rated-BADLY vs never-heard; pull toward X) ----", flush=True)
    diffs = []
    for u in users:
        known = u["known"]; kset = set(int(j) for j in known)
        eid = _top_region(S, known, TYPE_ENTITY)
        if eid is None:
            continue
        members = S.entities[eid]["members"]
        n_E = sum(1 for j in kset if j in S.entities[eid]["mset"])
        if n_E < 3:
            continue
        emb = S.region_emb(TYPE_ENTITY, eid)
        surp = S._surprise_val(kset, len(known), TYPE_ENTITY, eid)
        # watched-everything-rated-BADLY: implicit know_well (high surprise) + explicit HATED value
        bad = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, surp, FID_DATA, 0.0, emb),
               (TYPE_ENTITY, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, SP.CENTERED_FOLD["hated"], emb)]
        z_bad = fold_np(FR, model, bad, [])
        z_never = fold_np(FR, model, [], [])                 # never-heard: no token about X
        diffs.append(_region_score(FR, z_bad, members) - _region_score(FR, z_never, members))
    ci = _boot_ci(diffs); mean = float(np.mean(diffs))
    passed = bool(ci[0] > 0)
    print(f"  pull(bad-rater) - pull(never-heard) = {mean:+.4f}  CI[{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"(n={len(diffs)})  -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=mean, ci=ci, n=len(diffs), **{"pass": passed})


# ---- G2b: the prolific control ----
def gate_G2b(FR, model, S, users):
    print("\n---- G2b PROLIFIC CONTROL (selective-all-X vs prolific-all-X-plus-everything) ----", flush=True)
    rng = np.random.default_rng(7)
    diffs = []
    pop = np.argsort(-S.cnt)[:4000].astype(np.int64)
    for u in users:
        known = u["known"]; kset = set(int(j) for j in known)
        eid = _top_region(S, known, TYPE_ENTITY)
        if eid is None:
            continue
        e = S.entities[eid]; members = e["members"]; mset = e["mset"]
        xm = [int(j) for j in members]
        if len(xm) < 3:
            continue
        emb = S.region_emb(TYPE_ENTITY, eid)
        # SELECTIVE: profile == X's members only
        sel_known = set(xm)
        surp_sel = S._surprise_val(sel_known, len(sel_known), TYPE_ENTITY, eid)
        # PROLIFIC: X's members + everything else (large volume, same X consumption)
        extra = set(int(j) for j in pop) - set(xm)
        prol_known = sel_known | extra
        surp_pro = S._surprise_val(prol_known, len(prol_known), TYPE_ENTITY, eid)
        t_sel = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, surp_sel, FID_DATA, 0.0, emb)]
        t_pro = [(TYPE_ENTITY, KIND_IMPL, LVL_KW, surp_pro, FID_DATA, 0.0, emb)]
        z_sel = fold_np(FR, model, t_sel, [])
        z_pro = fold_np(FR, model, t_pro, [])
        diffs.append(_region_score(FR, z_sel, members) - _region_score(FR, z_pro, members))
    ci = _boot_ci(diffs); mean = float(np.mean(diffs))
    passed = bool(ci[0] > 0)
    print(f"  pull(selective) - pull(prolific) = {mean:+.4f}  CI[{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"(n={len(diffs)})  -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=mean, ci=ci, n=len(diffs), **{"pass": passed})


# ---- G3: dilution ----
def gate_G3(FR, model, S, users):
    print("\n---- G3 DILUTION (vague tokens must not drown a vivid set) ----", flush=True)
    rng = np.random.default_rng(3)
    drops = []
    for u in users:
        known = u["known"]; held = u["held"]; prof = set(known.keys())
        mu = float(np.mean(list(known.values())))
        cr = {int(j): float(r - mu) for j, r in known.items()}
        liked = [int(j) for j in known if known[j] >= 4][:4]
        if len(liked) < 2:
            continue
        vivid = []
        for j in liked:
            emb = FR.Wn[j].numpy().astype(np.float32)
            surp = S._surprise_val(prof, len(known), TYPE_ITEM, j)
            vivid.append((TYPE_ITEM, KIND_EXPL, LVL_ROUGH, 0.0, FID_DATA, float(cr[j]), emb))
            vivid.append((TYPE_ITEM, KIND_IMPL, LVL_KW, surp, FID_DATA, 0.0, emb))
        n_vivid = _ndcg_of(FR, model, vivid, liked, held, prof)
        vague = list(vivid)
        for _ in range(6):                    # add vague rough/llm-style low-info tokens
            c = int(rng.integers(S.n_concept))
            emb = S.region_emb(TYPE_CONCEPT, c)
            vague.append((TYPE_CONCEPT, KIND_IMPL, LVL_ROUGH, 0.0, FID_DATA, 0.0, emb))
            vague.append((TYPE_CONCEPT, KIND_EXPL, LVL_ROUGH, 0.0, FID_LLM,
                          float(rng.choice([-1 / 3, 1 / 3])), emb))
        n_vague = _ndcg_of(FR, model, vague, liked, held, prof)
        if None not in (n_vivid, n_vague):
            drops.append(n_vivid - n_vague)
    ci = _boot_ci(drops); mean = float(np.mean(drops))
    passed = bool(mean <= 0.005)
    print(f"  NDCG drop (vivid -> vivid+vague) = {mean:+.4f}  CI[{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"(n={len(drops)})  threshold <=0.005 -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean_drop=mean, ci=ci, n=len(drops), **{"pass": passed})


# ---- G4: clean profile vs native RecVAE ----
def gate_G4(FR, model, S, users):
    print("\n---- G4 CLEAN PROFILE (fold(full) vs native RecVAE) ----", flush=True)
    rng = np.random.default_rng(4)
    fold_n, nat_n = [], []
    for u in users:
        known = u["known"]; held = u["held"]; prof = set(known.keys())
        toks, native = S.build_reveal(known, "clean", 0, rng, cache=u.get("cache"))
        zf = fold_np(FR, model, toks, native)
        zn = FR.enc_items([list(known.keys())])[0].numpy().astype(np.float64)
        a = L.ndcg10(FR, zf, held, prof); b = L.ndcg10(FR, zn, held, prof)
        if None not in (a, b):
            fold_n.append(a); nat_n.append(b)
    mf, mn = float(np.mean(fold_n)), float(np.mean(nat_n))
    gap = mn - mf
    passed = bool(abs(gap) < 0.05)
    print(f"  fold(full) {mf:.4f}  native RecVAE {mn:.4f}  gap {gap:+.4f}  threshold |gap|<0.05 "
          f"-> {'PASS' if passed else 'FAIL'}  (n={len(fold_n)})", flush=True)
    return dict(fold_ndcg=mf, native_ndcg=mn, gap=gap, n=len(fold_n), **{"pass": passed})


# ---- G5: no-harm vs v2 on noisy interview reveals (SAME reveal, controlled) ----
_V2FID_TO_V3 = {0: FID_DATA, 1: FID_EASE, 2: FID_LLM}


def _v2_to_v3_explicit(v2toks):
    """Express a v2 reveal (type_id, fid_id, emb, value) as v3 EXPLICIT tokens (v2 has no implicit
    channel). This isolates the no-harm test to v3's base competency on v2's own evidence."""
    out = []
    for (typ, fid2, emb, val) in v2toks:
        out.append((int(typ), KIND_EXPL, LVL_ROUGH, 0.0, _V2FID_TO_V3.get(int(fid2), FID_EASE),
                    float(val), np.asarray(emb, np.float32)))
    return out


def gate_G5(FR, model, S, D, users):
    print("\n---- G5 NO-HARM vs v2 (SAME v2-regime reveal folded by v2 vs v3-as-explicit) ----",
          flush=True)
    import i25_fold_v2 as F2
    item_mean = F2.load_item_mean(D)
    famous = np.where(D["tier"] == "famous")[0].astype(np.int64)
    v2 = F2.FoldV2()
    v2.load_state_dict(torch.load(F2.CKPT_BEST, map_location="cpu")["model"]); v2.eval()
    rng = np.random.default_rng(5)
    v3n, v2n = [], []
    for u in users:
        known = u["known"]; held = u["held"]; prof = set(known.keys())
        # ONE v2-style noisy interview reveal, folded by BOTH models (identical evidence)
        toks2, native2 = F2.build_reveal_v2(D, FR, item_mean, famous, known, min(len(known), 16),
                                             np.random.default_rng(rng.integers(1 << 30)))
        if not toks2:
            continue
        z2 = F2.fold_np_v2(FR, v2, toks2, native2)
        z3 = fold_np(FR, model, _v2_to_v3_explicit(toks2), native2)
        a = L.ndcg10(FR, z3, held, prof); b = L.ndcg10(FR, z2, held, prof)
        if None not in (a, b):
            v3n.append(a); v2n.append(b)
    m3, m2 = float(np.mean(v3n)), float(np.mean(v2n))
    delta = m3 - m2
    passed = bool(delta >= -0.005)
    print(f"  v3 {m3:.4f}  v2 {m2:.4f}  delta {delta:+.4f}  threshold >= -0.005 "
          f"-> {'PASS' if passed else 'FAIL'}  (n={len(v3n)})", flush=True)
    return dict(v3_ndcg=m3, v2_ndcg=m2, delta=delta, n=len(v3n), **{"pass": passed})


# ---- G6: anti-saturation over budget 1..24 ----
def gate_G6(FR, model, S, users):
    print("\n---- G6 ANTI-SATURATION (NDCG over budget 1->24) ----", flush=True)
    curve = []
    for b in range(1, 25):
        vals = []
        for rep in range(3):                              # average seeds to de-noise the curve
            rng = np.random.default_rng(600 + b * 10 + rep)
            for u in users:
                toks, native = S.build_reveal(u["known"], "interview", b, rng, cache=u.get("cache"))
                if not toks:
                    continue
                v = L.ndcg10(FR, fold_np(FR, model, toks, native), u["held"], set(u["known"].keys()))
                if v is not None:
                    vals.append(v)
        curve.append(float(np.mean(vals)))
    steps = np.diff(curve)
    max_decline = float(steps.min())
    passed = bool(max_decline > -0.003)
    print("  budget:  " + " ".join(f"{c:.3f}" for c in curve), flush=True)
    print(f"  max per-step decline {max_decline:+.4f}  threshold > -0.003 "
          f"-> {'PASS' if passed else 'FAIL (ST contingency fires)'}", flush=True)
    return dict(curve=curve, max_step_decline=max_decline, ndcg1=curve[0], ndcg24=curve[-1],
                **{"pass": passed})


# ---- G7: implicit-channel ablation ----
def gate_G7(FR, model, S, users):
    print("\n---- G7 IMPLICIT ABLATION (zeroing implicit tokens must DROP) ----", flush=True)
    rng = np.random.default_rng(9)
    drops = []
    for u in users:
        known = u["known"]; held = u["held"]; prof = set(known.keys())
        budget = int(rng.integers(6, 20))
        toks, native = S.build_reveal(known, "interview", budget, rng, cache=u.get("cache"))
        if not toks:
            continue
        z_full = fold_np(FR, model, toks, native, impl_ablate=False)
        z_abl = fold_np(FR, model, toks, native, impl_ablate=True)
        a = L.ndcg10(FR, z_full, held, prof); b = L.ndcg10(FR, z_abl, held, prof)
        if None not in (a, b):
            drops.append(a - b)
    ci = _boot_ci(drops); mean = float(np.mean(drops))
    passed = bool(ci[0] > 0)
    print(f"  NDCG(full) - NDCG(implicit-zeroed) = {mean:+.4f}  CI[{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"(n={len(drops)})  -> {'PASS' if passed else 'FAIL'}", flush=True)
    return dict(mean=mean, ci=ci, n=len(drops), **{"pass": passed})


# ---- implied per-level implicit strengths (what did rough / know_well / no-clue become worth) ----
def probe_implied_strengths(FR, model, S, users):
    """For a matched implicit token on the SAME region, measure the pull toward the region at each
    knowledge level (rough / know_well / no_clue-negative), vs no token. Reports the mean region-score
    lift = the learned implicit strength per level."""
    print("\n---- IMPLIED IMPLICIT STRENGTHS (pull toward region by level) ----", flush=True)
    levels = {"rough": LVL_ROUGH, "know_well": LVL_KW, "no_clue_neg": LVL_NEG}
    out = {}
    for lname, lv in levels.items():
        lifts = []
        for u in users:
            known = u["known"]; kset = set(int(j) for j in known)
            eid = _top_region(S, known, TYPE_ENTITY)
            if eid is None:
                continue
            members = S.entities[eid]["members"]
            emb = S.region_emb(TYPE_ENTITY, eid)
            surp = S._surprise_val(kset, len(known), TYPE_ENTITY, eid)
            z0 = fold_np(FR, model, [], [])
            tok = [(TYPE_ENTITY, KIND_IMPL, lv, surp, FID_DATA, 0.0, emb)]
            z1 = fold_np(FR, model, tok, [])
            lifts.append(_region_score(FR, z1, members) - _region_score(FR, z0, members))
        m = float(np.mean(lifts)); ci = _boot_ci(lifts)
        out[lname] = dict(mean_pull=m, ci=ci, n=len(lifts))
        print(f"  {lname:12s} region-score pull {m:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)
    return out


def _v(b):
    return "PASS" if b else "FAIL"


def _write_gate_md(R):
    _md("## Gate table (pre-registered thresholds; CIs = 95% bootstrap over users)\n\n")
    _md("| gate | metric | value | 95% CI | threshold | verdict |\n|---|---|--:|---|---|:--:|\n")
    g1 = R["G1"]
    for ch in ("item", "concept", "attr", "entity"):
        d = g1[ch]
        _md(f"| G1 {ch} explicit | cold 1-tok NDCG lift | {d['explicit_lift']:+.4f} | "
            f"[{d['explicit_ci'][0]:+.4f},{d['explicit_ci'][1]:+.4f}] | lift>0 | "
            f"{_v(d['explicit_lift']>0)} |\n")
        _md(f"| G1 {ch} implicit-only | cold 1-tok NDCG lift | {d['implicit_lift']:+.4f} | "
            f"[{d['implicit_ci'][0]:+.4f},{d['implicit_ci'][1]:+.4f}] | lift>0 | "
            f"{_v(d['implicit_lift']>0)} |\n")
    g = R["G2"]; _md(f"| G2 GoT | pull(bad)-pull(never) | {g['mean']:+.4f} | "
                     f"[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] | CI excl 0 | {_v(g['pass'])} |\n")
    g = R["G2b"]; _md(f"| G2b prolific | pull(sel)-pull(prol) | {g['mean']:+.4f} | "
                      f"[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] | CI excl 0 | {_v(g['pass'])} |\n")
    g = R["G3"]; _md(f"| G3 dilution | NDCG drop vivid->+vague | {g['mean_drop']:+.4f} | "
                     f"[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] | <=0.005 | {_v(g['pass'])} |\n")
    g = R["G4"]; _md(f"| G4 clean | fold {g['fold_ndcg']:.4f} vs native {g['native_ndcg']:.4f} | "
                     f"{g['gap']:+.4f} | - | \\|gap\\|<0.05 | {_v(g['pass'])} |\n")
    g = R["G5"]; _md(f"| G5 no-harm | v3 {g['v3_ndcg']:.4f} vs v2 {g['v2_ndcg']:.4f} | "
                     f"{g['delta']:+.4f} | - | >=-0.005 | {_v(g['pass'])} |\n")
    g = R["G6"]; _md(f"| G6 anti-sat | max per-step decline (NDCG {g['ndcg1']:.3f}->{g['ndcg24']:.3f}) | "
                     f"{g['max_step_decline']:+.4f} | - | >-0.003 | {_v(g['pass'])} |\n")
    g = R["G7"]; _md(f"| G7 implicit ablation | NDCG(full)-NDCG(zeroed) | {g['mean']:+.4f} | "
                     f"[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] | CI excl 0 | {_v(g['pass'])} |\n")
    _md("\n### Implied per-level implicit strengths (region-score pull vs no token)\n\n")
    _md("| level | mean pull | 95% CI |\n|---|--:|---|\n")
    for lv, d in R["implied_strengths"].items():
        _md(f"| {lv} | {d['mean_pull']:+.4f} | [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] |\n")
    st_fired = not R["G6"]["pass"]
    _md(f"\n### Verdicts\n\n- ST contingency fired: {st_fired} "
        f"(G6 max per-step decline {R['G6']['max_step_decline']:+.4f}).\n")
    _md(f"- Winner curriculum: clean_frac {R['winner']['clean_frac']} "
        f"(val NDCG@10 {R['winner']['best_val']}).\n")
    allpass = all(R[k].get("pass") for k in ("G1", "G2", "G2b", "G3", "G4", "G5", "G6", "G7"))
    _md(f"- ALL GATES PASS: {allpass}.\n\n")
    assert os.path.exists(BUILD_MD)


def _write_st_md(train_r, R):
    _md("\n## G6 CONTINGENCY -- Set-Transformer variant (D3, fired because Deep-Sets G6 failed)\n\n")
    _md(f"Trained ONE Set-Transformer variant (masked SAB + PMA pooling replacing Deep-Sets sum-pool; "
        f"same token features + residual) at the winning curriculum. Best val NDCG@10 "
        f"{train_r['best_val']:.4f} @ep{train_r['best_epoch']} (`{WINNER_ST}`). Re-gated:\n\n")
    _md("| gate | value | 95% CI | verdict |\n|---|--:|---|:--:|\n")
    g = R["G6"]; _md(f"| G6 anti-sat (NDCG {g['ndcg1']:.3f}->{g['ndcg24']:.3f}) | "
                     f"max step {g['max_step_decline']:+.4f} | - | {_v(g['pass'])} |\n")
    for k, lab, key in [("G2", "G2 GoT", "mean"), ("G2b", "G2b prolific", "mean"),
                        ("G7", "G7 implicit-abl", "mean")]:
        d = R[k]; _md(f"| {lab} | {d[key]:+.4f} | [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] | {_v(d['pass'])} |\n")
    g = R["G4"]; _md(f"| G4 clean | gap {g['gap']:+.4f} | - | {_v(g['pass'])} |\n")
    g = R["G5"]; _md(f"| G5 no-harm | delta {g['delta']:+.4f} | - | {_v(g['pass'])} |\n")
    allpass = all(R[k].get("pass") for k in ("G1", "G2", "G2b", "G3", "G4", "G5", "G6", "G7"))
    _md(f"\n**ST variant G6: {_v(R['G6']['pass'])}** (Deep-Sets G6 had failed). ST all-gates-pass: {allpass}.\n\n")
    assert os.path.exists(BUILD_MD)


def _print_verdicts(R):
    print("\n=== GATE VERDICTS ===", flush=True)
    for k in ("G1", "G2", "G2b", "G3", "G4", "G5", "G6", "G7"):
        print(f"  {k:4s} {_v(R[k]['pass'])}", flush=True)
    print(f"  ST contingency fired: {not R['G6']['pass']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "gates", "smoke", "st"])
    ap.add_argument("--n_users", type=int, default=25000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--n_gate", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "gates":
        cmd_gates(a)
    elif a.cmd == "st":
        cmd_st(a)
    elif a.cmd == "smoke":
        a.n_users = 400; a.n_val = 120; a.epochs = 1; a.n_gate = 60
        globals()["CLEAN_FRACS"] = [30, 50]
        cmd_train(a)
