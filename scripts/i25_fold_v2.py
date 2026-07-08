"""i25_fold_v2.py -- THE REPAIR: retrain the I2.5 LEARNED FOLD on an answerer-v1-LIKE answer
distribution, with KNOWLEDGE/FIDELITY entering as a TOKEN FEATURE (the hand knob w_rough retired).

WHY (STATE_2026-07-08): the v1 fold (.cache/i25_fold_best.pt) was trained on DATA-SIDE-ONLY reveals
(zero-noise real ratings / member aggregates) and applied a hand weight (know_well=1.0, rough=0.5)
OUTSIDE the fold. In the answerer-v1 arena most know_well answers are LLM-inferred (sigma~0.70 stars),
so folding them at weight 1.0 amplifies fidelity noise -> "8 vivid concept answers ~= silence"
(VIVID_SWAP T1 FAIL) and the calibration fitted w_k1_llm=0. This fold is OOD for the arena's answers.

THE FIX. Retrain on a reveal sampler that SIMULATES the answerer-v1 answer distribution from POPULATION
data (firewall: trU users only; the LLM grid is NEVER a training input). Each answer token carries a
FIDELITY-CLASS one-hot {data, llm_know_well, llm_rough} so the fold LEARNS its own per-class trust
(the retired w_rough). Calibrated to measured v1 numbers (sigma=0.70 stars; per-stratum knowledge base
rates from experiments/adaptivity_battery_v1_A.json) -- every constant documented in CALIB below.

Token vocabulary (fidelity class in parens):
  - item, real centered rating           (data)         -- the user's own ratings, zero noise
  - item, unrated-famous inferred         (llm_kw/rough) -- CF item-mean + N(0,0.70) noise, binned
  - concept member-aggregate inferred     (llm_kw/rough) -- rel-weighted mean rating + noise, 4-level
  - attribute member-aggregate inferred   (llm_kw/rough) -- mean rating + noise, 4-level
Refusals (no_clue) are excluded (they inform answerability, not taste).

NO LLM calls. Canonical scripts/caches untouched. Checkpoints -> .cache/i25_fold_v2*.
Run:  python scripts/i25_fold_v2.py --n_users 25000 --epochs 14
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import i25_lib as L

D_LAT = L.D_LAT
CKPT = ".cache/i25_fold_v2.pt"
CKPT_BEST = ".cache/i25_fold_v2_best.pt"
LOG = ".cache/i25_fold_v2_log.json"
ITEM_MEAN_CACHE = ".cache/i25_v2_item_mean.npy"

CENTERED_FOLD = {"hated": -1.0, "meh": -1.0 / 3.0, "liked": 1.0 / 3.0, "loved": 1.0}
SIGMA_STAR = 0.70          # schema fidelity sigma == masked-value MAE (answerer_schema.json)

# ---- CALIBRATION (documented; from experiments/adaptivity_battery_v1_A.json A1/A3) ----------------
# P(know_well | answered) and P(rough | answered) per channel, and channel/fidelity slot mix.
# item famous: rate_k1 .9907 rate_k2 .6937 -> P(kw|ans)=.6937/.9907=.700 ; P(rough|ans)=.300
# concept all : rate_k1 .7407 rate_k2 .3856 -> P(kw|ans)=.3856/.7407=.521 ; P(rough|ans)=.479
# attribute   : base_k1 .8244 base_k2 .2993 -> P(kw|ans)=.2993/.8244=.363 ; P(rough|ans)=.637
CALIB = dict(
    sigma_star=SIGMA_STAR,
    p_kw_given_ans=dict(item=0.700, concept=0.521, attr=0.363),
    # per-slot channel/fidelity mix for a simulated interview answer set (design choice, documented):
    slot_mix=dict(item_data=0.25, item_llm=0.15, concept_llm=0.40, attr_llm=0.20),
    item_data_keep=0.70,       # a revealed real item is actually presented as a data token w.p. this
    n_itemllm_max=4, n_concept_max=6, n_attr_max=6,
)
FID = {"data": 0, "llm_kw": 1, "llm_rough": 2}
NFID = 3


def bin_star(r):
    """Continuous extension of answerer_schema rating_to_scale bins -> centered 4-level fold value."""
    if r >= 4.25:
        return CENTERED_FOLD["loved"]
    if r >= 3.25:
        return CENTERED_FOLD["liked"]
    if r >= 2.25:
        return CENTERED_FOLD["meh"]
    return CENTERED_FOLD["hated"]


# ================================================================= population item-mean (firewall: trU)
def load_item_mean(D):
    if os.path.exists(ITEM_MEAN_CACHE):
        return np.load(ITEM_MEAN_CACHE)
    d = np.load(L.META_FULL)
    uu, ii, rr = d["uu"], d["ii"], d["rr"]
    trU = d["trU"].astype(np.int64)
    sel = np.isin(uu, trU)                       # POPULATION only (study users are in va/te)
    ii_s = ii[sel]; rr_s = rr[sel].astype(np.float64)
    ni = int(D["ni"])
    s = np.bincount(ii_s, weights=rr_s, minlength=ni)
    c = np.bincount(ii_s, minlength=ni).astype(np.float64)
    glob = float(rr_s.mean())
    im = np.where(c > 0, s / np.maximum(c, 1), glob)
    im[c == 0] = glob
    np.save(ITEM_MEAN_CACHE, im)
    return im


# ================================================================= the v2 fold (fidelity-aware Deep-Sets)
class FoldV2(nn.Module):
    """Residual Deep-Sets fold with a FIDELITY-CLASS token feature so the fold learns per-class trust.
    Per-token input = [type_onehot(3), fid_onehot(3), value(1), value*emb(d)]. z = native_z + rho(...)."""
    def __init__(self, d=D_LAT):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3 + NFID + 1 + d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, d))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)   # residual: start at native
        self.d = d

    def forward(self, tok_type, tok_fid, tok_val, tok_emb, mask, native_z, fid_ablate=False):
        B, K, d = tok_emb.shape
        oneh = F.one_hot(tok_type.clamp(min=0), num_classes=3).to(tok_emb.dtype)
        fh = F.one_hot(tok_fid.clamp(min=0), num_classes=NFID).to(tok_emb.dtype)
        if fid_ablate:
            fh = torch.zeros_like(fh)                 # G5 ablation: remove the fidelity feature
        ve = tok_val.unsqueeze(-1) * tok_emb
        x = torch.cat([oneh, fh, tok_val.unsqueeze(-1), ve], dim=-1)
        h = self.phi(x) * mask.unsqueeze(-1)
        pool = h.sum(1)
        ntok = mask.sum(1, keepdim=True)
        delta = self.rho(torch.cat([pool, native_z, torch.log1p(ntok)], dim=-1))
        return native_z + delta


def pack_batch_v2(FR, tokens_per_user, native_item_lists, device="cpu"):
    """tokens: list of (type_id, fid_id, emb(d), value)."""
    B = len(tokens_per_user)
    K = max((len(t) for t in tokens_per_user), default=1); K = max(K, 1)
    tt = torch.zeros((B, K), dtype=torch.long)
    tf = torch.zeros((B, K), dtype=torch.long)
    tv = torch.zeros((B, K), dtype=torch.float32)
    te = torch.zeros((B, K, FR.W.shape[1]), dtype=torch.float32)
    mask = torch.zeros((B, K), dtype=torch.float32)
    for b, toks in enumerate(tokens_per_user):
        for k, (typ, fid, emb, val) in enumerate(toks):
            tt[b, k] = typ; tf[b, k] = fid; tv[b, k] = val
            te[b, k] = torch.as_tensor(emb); mask[b, k] = 1.0
    nz = FR.enc_items(native_item_lists)
    return (tt.to(device), tf.to(device), tv.to(device), te.to(device), mask.to(device), nz.to(device))


def fold_batch_v2(FR, model, tok_lists, native_lists, fid_ablate=False):
    if not tok_lists:
        return np.zeros((0, FR.W.shape[1]))
    tt, tf, tv, te, mask, nz = pack_batch_v2(FR, tok_lists, native_lists)
    with torch.no_grad():
        z = model(tt, tf, tv, te, mask, nz, fid_ablate=fid_ablate)
    return z.numpy().astype(np.float64)


def fold_np_v2(FR, model, tokens, native_items, fid_ablate=False):
    return fold_batch_v2(FR, model, [tokens], [native_items], fid_ablate=fid_ablate)[0].astype(np.float64)


# ================================================================= the reveal sampler (SIMULATED v1)
def sample_knowledge(channel, rng):
    """Return (fid_id, know_well_bool) for an inferred (llm) token in a channel."""
    p_kw = CALIB["p_kw_given_ans"][channel]
    if rng.random() < p_kw:
        return FID["llm_kw"], True
    return FID["llm_rough"], False


def build_reveal_v2(D, FR, item_mean, famous, known, k, rng):
    """Simulate an answerer-v1-like answer set from a user's known-half real ratings {j:r}.

    Returns (tokens, native_items). tokens = list of (type_id, fid_id, emb(d), value).
    - item-data:  a random subset of the k revealed real items, real centered rating, fid=data.
    - item-llm:   unrated famous distractors, value = item_mean+N(0,sigma) centered, knowledge sampled.
    - concept-llm:top rel-mass concepts over the reveal, rel-weighted mean rating + noise -> 4-level.
    - attr-llm:   top attributes (decade/genre) over the reveal, mean rating + noise -> 4-level.
    """
    ks = list(known.keys())
    if len(ks) > k:
        ks = [ks[i] for i in rng.choice(len(ks), size=k, replace=False)]
    R = {j: known[j] for j in ks}                          # the revealed real ratings
    mu = float(np.mean(list(R.values())))                  # deployable centering (revealed mean)
    cr = {j: R[j] - mu for j in R}
    toks = []; native = []

    # ---- item-data (real ratings) ----
    for j in ks:
        if rng.random() < CALIB["item_data_keep"]:
            toks.append((0, FID["data"], FR.Wn[j].numpy().astype(np.float32), float(cr[j])))
            if R[j] >= 4:
                native.append(j)

    # ---- item-llm distractors (unrated famous items; inferred value only) ----
    n_ill = int(rng.integers(0, CALIB["n_itemllm_max"] + 1))
    profset = set(R.keys())
    if n_ill and len(famous):
        picks = famous[rng.choice(len(famous), size=min(n_ill * 3, len(famous)), replace=False)]
        picks = [int(j) for j in picks if int(j) not in profset][:n_ill]
        for j in picks:
            fid, _ = sample_knowledge("item", rng)
            v = float(np.clip(item_mean[j] + rng.normal(0, SIGMA_STAR), 0.5, 5.0) - mu)
            toks.append((0, fid, FR.Wn[j].numpy().astype(np.float32), v))

    # ---- concept-llm (rel-weighted mean centered rating + noise, binned 4-level) ----
    it_arr = np.array(ks)
    rel = D["concepts"]["item_tag"][it_arr]                 # (n_rev, n_tags)
    mass = rel.sum(0); crv = np.array([cr[j] for j in ks])
    picked = 0
    for ctag in np.argsort(-mass):
        m = float(mass[ctag])
        if m < 1e-6 or picked >= CALIB["n_concept_max"]:
            break
        agg = float((rel[:, ctag] * crv).sum() / m)        # centered aggregate (stars-space delta)
        noisy = agg + rng.normal(0, SIGMA_STAR)
        val = bin_star(noisy + mu)                          # re-add mean to bin on the star scale
        fid, _ = sample_knowledge("concept", rng)
        toks.append((1, fid, FR.concept_emb(int(ctag)), float(val)))
        picked += 1

    # ---- attr-llm (decade/genre member aggregates + noise, binned) ----
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for j in ks:
        for ak in L.item_attrs(D, j):
            agg[ak][0] += cr[j]; agg[ak][1] += 1.0
    for ak, (s, n) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:CALIB["n_attr_max"]]:
        a = s / n if n > 0 else 0.0
        val = bin_star(a + rng.normal(0, SIGMA_STAR) + mu)
        fid, _ = sample_knowledge("attr", rng)
        toks.append((2, fid, FR.attr_emb(ak), float(val)))

    rng.shuffle(toks)
    return toks, native


# ================================================================= training
def make_user_split(prof, rng):
    its = list(prof.keys()); rng.shuffle(its)
    h = len(its) // 2
    known = {j: prof[j] for j in its[:h]}
    held = set(j for j in its[h:] if prof[j] >= 4)
    return known, held


def prep_users(prof_dict, rng):
    out = []
    for u, prof in prof_dict.items():
        known, held = make_user_split(prof, rng)
        if held and len(known) >= 2:
            out.append(dict(u=u, known=known, held=held))
    return out


def batch_loss(FR, model, D, item_mean, famous, users, rng, kmax=16):
    tok_lists, native_lists, targets, profs = [], [], [], []
    k = int(rng.integers(1, kmax + 1))
    for u in users:
        known, held = u["known"], u["held"]
        if not held:
            continue
        toks, native = build_reveal_v2(D, FR, item_mean, famous, known, k, rng)
        if not toks:
            continue
        tok_lists.append(toks); native_lists.append(native)
        targets.append(held); profs.append(set(known.keys()))
    if not tok_lists:
        return None
    tt, tf, tv, te, mask, nz = pack_batch_v2(FR, tok_lists, native_lists)
    z = model(tt, tf, tv, te, mask, nz)
    S = z @ FR.W.T + FR.bdec
    B, ni = S.shape
    negmask = torch.zeros((B, ni), dtype=torch.bool)
    tgt = torch.zeros((B, ni), dtype=torch.float32)
    for b in range(B):
        negmask[b, list(profs[b])] = True
        tgt[b, list(targets[b])] = 1.0
    S = S.masked_fill(negmask, -1e9)
    logp = torch.log_softmax(S, dim=1)
    n_t = tgt.sum(1).clamp(min=1)
    loss = -(logp * tgt).sum(1) / n_t
    return loss.mean()


@torch.no_grad()
def val_ndcg(FR, model, D, item_mean, famous, val_users, rng):
    """Fold a SIMULATED full-length reveal (k=16) of each val user's known half; mean NDCG@10 full."""
    model.eval()
    vals = []
    tl, nl, idx = [], [], []
    for i, u in enumerate(val_users):
        toks, native = build_reveal_v2(D, FR, item_mean, famous, u["known"], 16, rng)
        if not toks:
            continue
        tl.append(toks); nl.append(native); idx.append(i)
    if not tl:
        return 0.0
    for s in range(0, len(tl), 2000):
        e = min(s + 2000, len(tl))
        Z = fold_batch_v2(FR, model, tl[s:e], nl[s:e])
        for r in range(s, e):
            u = val_users[idx[r]]
            v = L.ndcg10(FR, Z[r - s], u["held"], set(u["known"].keys()))
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=25000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()
    print(f"[v2] loading data + frozen RecVAE ...", flush=True)
    D = L.G.load_data()
    FR = L.Frozen(D)
    item_mean = load_item_mean(D)
    famous = np.where(D["tier"] == "famous")[0].astype(np.int64)
    print(f"[v2] item_mean ready ({(item_mean>0).sum()} items); {len(famous)} famous items; "
          f"loading {args.n_users} train profiles ...", flush=True)
    prof = L.load_train_profiles(D, args.n_users + args.n_val, seed=args.seed)
    keys = list(prof.keys())
    rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    val_keys = set(keys[:args.n_val]); tr_keys = keys[args.n_val:]
    tr_users = prep_users({u: prof[u] for u in tr_keys}, np.random.default_rng(args.seed + 1))
    val_users = prep_users({u: prof[u] for u in val_keys}, np.random.default_rng(args.seed + 2))
    print(f"[v2] train users {len(tr_users)}  val users {len(val_users)}  "
          f"(load {round(time.time()-t0,1)}s)", flush=True)

    model = FoldV2()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    state = dict(epoch=0, best_val=-1.0, best_epoch=-1, history=[], calib=CALIB)
    if args.resume and os.path.exists(CKPT):
        blob = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        print(f"[v2] resumed at epoch {state['epoch']} best_val={state['best_val']:.4f}", flush=True)

    step_rng = np.random.default_rng(args.seed + 100)
    for ep in range(state["epoch"], args.epochs):
        model.train()
        order = np.arange(len(tr_users)); step_rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), args.batch):
            us = [tr_users[i] for i in order[b0:b0 + args.batch]]
            loss = batch_loss(FR, model, D, item_mean, famous, us, step_rng, kmax=args.kmax)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        vN = val_ndcg(FR, model, D, item_mean, famous, val_users, np.random.default_rng(args.seed + 7))
        state["epoch"] = ep + 1
        rec = dict(epoch=ep + 1, train_loss=round(tot / max(nb, 1), 4), val_ndcg=round(vN, 4),
                   min=round((time.time() - t0) / 60, 1))
        state["history"].append(rec)
        print(f"[v2] ep {ep+1:2d} | loss {rec['train_loss']:.4f} | val NDCG@10 {vN:.4f} | {rec['min']}m",
              flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state, args=vars(args)), CKPT)
        if vN > state["best_val"]:
            state["best_val"] = vN; state["best_epoch"] = ep + 1
            torch.save(dict(model=model.state_dict(), state=state, args=vars(args)), CKPT_BEST)
        json.dump(state["history"], open(LOG, "w"), indent=1)
    print(f"[v2] BEST val NDCG@10 {state['best_val']:.4f} @ep{state['best_epoch']} -> {CKPT_BEST}  "
          f"(wall {round((time.time()-t0)/60,1)}m)", flush=True)


if __name__ == "__main__":
    main()
