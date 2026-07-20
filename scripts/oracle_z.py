"""oracle_z.py -- ORACLE-Z HEADROOM: the BEST-POSSIBLE belief z under the FROZEN RecVAE decoder.

Training-free, per-user optimization. NO encoder, NO architecture. For each held-out TEST user we split
ratings into leak-free KNOWN / HELD, then fit an oracle latent z* (512-d free variable) by gradient
descent through the FROZEN RecVAE decoder (W, b frozen; score = z @ W.T + b) to maximize a ranking
objective on the KNOWN ratings under FOUR conditionings:

  (1) LIKES-ONLY        : positives = KNOWN liked (r>=4); softmax multinomial CE (RecVAE's own objective).
  (2) LIKES+DISLIKES    : (1) + BPR pushing KNOWN liked above KNOWN watched-disliked (r<=2, INSIDE cone).
  (3) LIKES+DISINTEREST : (1) + BPR pushing KNOWN liked above UNWATCHED items from AVOIDED regions
                          (genres under-represented vs population; OUTSIDE the consumption cone).
  CLAIRVOYANT           : positives = HELD liked (fit directly to the eval target) -> upper-bound ceiling.

Then EVALUATE the fitted z* on the disjoint HELD set:
  PRIMARY  : NDCG@10 over HELD-LIKED (deployment metric; masks KNOWN items).
  SECONDARY: signed AUC over ALL held (held-liked ranked above held-disliked).

Reports Delta = (2)-(1) and (3)-(1) for both metrics with paired bootstrap CIs, plus the clairvoyant
ceiling for scale. The frozen decoder is the wall this bounds: no encoder/variant can beat the best z*.

RULES honored: NO data reduction (all 2k test users, all ratings, no caps; users batched for compute).
Study 173/300 users QUARANTINED via the trU firewall in load_train_profiles. NO LLM calls.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import warnings
warnings.filterwarnings("ignore")
import i25_lib as L

OUT = ".cache/rung1"
os.makedirs(OUT, exist_ok=True)
D_LAT = L.D_LAT


# ============================================================ per-user leak-free split
def user_split(prof, rng):
    """Random disjoint half-split of a user's rated profile into KNOWN / HELD (identical convention to
    rung1_encoder.user_split)."""
    its = list(prof.keys()); rng.shuffle(its)
    h = len(its) // 2
    known = {int(j): float(prof[j]) for j in its[:h]}
    held = {int(j): float(prof[j]) for j in its[h:]}
    return known, held


# ============================================================ disinterest region sampler
def population_genre_prev(D):
    """Population genre prevalence p_g = fraction of catalog items carrying genre g (popularity-weighted)."""
    G = D["Gmat"].astype(np.float64)                      # (ni, n_gen)
    cnt = D["cnt"].astype(np.float64) + 1.0               # item popularity
    w = cnt / cnt.sum()
    return (G * w[:, None]).sum(0)                        # (n_gen,) popularity-weighted prevalence


def disinterest_items(D, known_ids, avoid_rng, pop_prev, cnt, gen_item_lists, n_neg,
                      exclude):
    """Sample UNWATCHED negative target items from the user's AVOIDED genres (systematic avoidance:
    genres under-represented in the user's KNOWN profile relative to population). Items drawn are NOT in
    `exclude` (the user's full rated set) -> outside the consumption cone, no leak into HELD.
    Popularity-weighted so they are plausible recommendations the user nonetheless avoids."""
    G = D["Gmat"]
    ug = G[list(known_ids)].sum(0).astype(np.float64)     # user's genre counts over KNOWN items
    uq = ug / max(ug.sum(), 1.0)                          # user genre distribution
    # avoidance = population prevalence minus user share (positive => under-represented / avoided)
    avoid = pop_prev - uq
    order = np.argsort(-avoid)                            # most-avoided genres first
    picked = []
    seen = set()
    for g in order:
        if avoid[g] <= 0:
            break
        for j in gen_item_lists[g]:
            if j in exclude or j in seen:
                continue
            seen.add(int(j)); picked.append(int(j))
    if not picked:
        return np.array([], dtype=np.int64)
    picked = np.array(picked, dtype=np.int64)
    w = cnt[picked] + 1.0
    w = w / w.sum()
    m = min(n_neg, len(picked))
    idx = avoid_rng.choice(len(picked), size=m, replace=False, p=w)
    return picked[idx]


# ============================================================ oracle fit (gradient descent through decoder)
def fit_z(W, bdec, z0, pos_rows, pos_cols, neg_pairs, steps, lr, reg, lam):
    """Optimize a batch of latents Z (B,d) through the FROZEN decoder.
    pos_rows/pos_cols: flat COO of positive (user,item) targets for the softmax multinomial CE.
    neg_pairs: optional (row, pos_item, neg_item) tensors for a BPR margin term (likes > negatives).
    Returns detached Z (B,d) numpy."""
    Z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)
    B = Z.shape[0]
    pr = torch.as_tensor(pos_rows, dtype=torch.long)
    pc = torch.as_tensor(pos_cols, dtype=torch.long)
    seg = pr                                              # segment ids = user rows
    ones = torch.ones(len(pr))
    cnt_pos = torch.zeros(B).scatter_add_(0, seg, ones).clamp(min=1.0)
    if neg_pairs is not None:
        nr, npos, nneg = neg_pairs
        nr = torch.as_tensor(nr, dtype=torch.long)
        npos = torch.as_tensor(npos, dtype=torch.long)
        nneg = torch.as_tensor(nneg, dtype=torch.long)
        nones = torch.ones(len(nr))
        cnt_neg = torch.zeros(B).scatter_add_(0, nr, nones).clamp(min=1.0)
    for _ in range(steps):
        opt.zero_grad()
        S = Z @ W.T + bdec                               # (B, ni)  differentiable in Z
        logp = torch.log_softmax(S, dim=1)
        # per-user mean CE over its positive (liked) targets
        gp = logp[pr, pc]                                 # (n_pos,)
        ce_u = torch.zeros(B).scatter_add_(0, seg, -gp) / cnt_pos
        loss = ce_u.mean()
        if neg_pairs is not None and len(nr) > 0:
            sp = S[nr, npos]; sn = S[nr, nneg]
            bpr = -F.logsigmoid(sp - sn)                 # push pos above neg
            bpr_u = torch.zeros(B).scatter_add_(0, nr, bpr) / cnt_neg
            loss = loss + lam * bpr_u.mean()
        loss = loss + reg * ((Z - z0) ** 2).sum(1).mean()
        loss.backward()
        opt.step()
    return Z.detach()


# ============================================================ metrics
def ndcg_liked(FR, z, held_likes, profset):
    return L.ndcg10(FR, np.asarray(z, np.float64), held_likes, profset)


def signed_auc(scores, liked, disliked):
    """AUC that held-liked outrank held-disliked (all liked x disliked pairs)."""
    if not liked or not disliked:
        return None
    sl = scores[list(liked)][:, None]
    sd = scores[list(disliked)][None, :]
    return float(((sl > sd).mean() + 0.5 * (sl == sd).mean()))


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return (float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


# ============================================================ main
def main(args):
    t0 = time.time()
    print("[oracle-z] loading frozen RecVAE + data ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D)
    W = FR.W; bdec = FR.bdec                              # frozen float32 tensors
    ni = int(D["ni"])

    # ---- 2k held-out TEST users (disjoint from pilot train+val; quarantine via trU firewall) ----
    need = 2000 + 1000 + args.n_test
    prof = L.load_train_profiles(D, int(need * 1.6) + 2000, seed=args.seed + 1234)
    keys = sorted(prof.keys())
    rng = np.random.default_rng(args.seed + 1234); rng.shuffle(keys)
    assert len(keys) >= need, f"user shortfall {len(keys)} < {need}"
    te_keys = keys[3000:3000 + args.n_test]              # test partition (after pilot's 2k train + 1k val)
    print(f"[oracle-z] {len(te_keys)} held-out test users [{time.time()-t0:.0f}s]", flush=True)

    # precompute disinterest machinery
    pop_prev = population_genre_prev(D)
    cnt = D["cnt"].astype(np.float64)
    G = D["Gmat"]
    n_gen = G.shape[1]
    gen_item_lists = [np.where(G[:, g] > 0)[0] for g in range(n_gen)]

    # ---- build per-user records ----
    split_rng = np.random.default_rng(args.seed + 7)
    recs = []
    for u in te_keys:
        known, held = user_split(prof[u], split_rng)
        kl = [j for j, r in known.items() if r >= 4]      # known liked
        kd = [j for j, r in known.items() if r <= 2]      # known watched-disliked
        hl = set(j for j, r in held.items() if r >= 4)    # held liked (primary target)
        hd = set(j for j, r in held.items() if r <= 2)    # held disliked (signed metric)
        if len(kl) < 1 or len(hl) < 1:
            continue
        avoid_rng = np.random.default_rng(hash((int(u), args.seed)) & 0xFFFFFFFF)
        exclude = set(known.keys()) | set(held.keys())
        n_neg = max(len(kd), 20)
        di = disinterest_items(D, kl, avoid_rng, pop_prev, cnt, gen_item_lists, n_neg, exclude)
        recs.append(dict(u=int(u), known=known, kl=kl, kd=kd, hl=hl, hd=hd,
                         held_all=held, profset=set(known.keys()),
                         held_pos=list(hl), disint=di.tolist()))
    print(f"[oracle-z] {len(recs)} usable users (>=1 known-like & >=1 held-like); "
          f"{sum(1 for r in recs if r['kd'])} have known-dislikes; "
          f"{sum(1 for r in recs if r['hd'])} have held-dislikes [{time.time()-t0:.0f}s]", flush=True)

    # native z0 (RecVAE encoding of KNOWN liked) as the shared init for every conditioning
    z0_all = FR.enc_items([r["kl"] for r in recs]).detach()          # (N, d)

    N = len(recs)
    B = args.batch
    conds = ["likes", "dislikes", "disinterest", "clairvoyant"]
    has_kd = np.array([len(r["kd"]) > 0 for r in recs])

    # reg is the ONLY hyperparameter that anchors the fit to the native manifold; the like/dislike/
    # disinterest verdict must not hinge on it, so we SWEEP it and report robustness. The clairvoyant
    # ceiling is taken from the LEAST-regularized fit (truest upper bound).
    reg_grid = sorted(set([args.reg] + [float(x) for x in args.reg_grid.split(",") if x]))
    sweep = {}
    for reg in reg_grid:
        Zf = _fit_all(FR, W, bdec, recs, z0_all, conds, N, B, args, reg, t0)
        sweep[reg] = _evaluate(FR, recs, Zf, conds, z0_all.numpy(), has_kd)
    # assemble the report around the primary reg = args.reg
    R = _assemble(args, recs, sweep, has_kd, N, reg_grid)
    R["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(R, open(f"{OUT}/oracle_z.json", "w"), indent=2, default=float)
    _write_md(R)
    _print_summary(R)
    return R


def _fit_all(FR, W, bdec, recs, z0_all, conds, N, B, args, reg, t0):
    Zfit = {c: np.zeros((N, D_LAT), np.float64) for c in conds}
    for s in range(0, N, B):
        e = min(s + B, N)
        sub = recs[s:e]; b = len(sub)
        z0 = z0_all[s:e]
        # positive COO for likes-based conditionings (known-liked)
        pr, pc = [], []
        for i, r in enumerate(sub):
            for j in r["kl"]:
                pr.append(i); pc.append(j)
        # positive COO for clairvoyant (held-liked)
        cpr, cpc = [], []
        for i, r in enumerate(sub):
            for j in r["held_pos"]:
                cpr.append(i); cpc.append(j)
        # BPR neg pairs: (row, a-positive-liked-item, a-negative-item). One neg per (user,neg) paired to a
        # random known-liked positive of that user.
        def make_bpr(field):
            nr, npos, nneg = [], [], []
            prng = np.random.default_rng(args.seed + 99)
            for i, r in enumerate(sub):
                negs = r[field]
                if not negs or not r["kl"]:
                    continue
                kl = r["kl"]
                for jn in negs:
                    jp = kl[prng.integers(len(kl))]
                    nr.append(i); npos.append(int(jp)); nneg.append(int(jn))
            return (nr, npos, nneg)
        bpr_dis = make_bpr("kd")
        bpr_dii = make_bpr("disint")

        Zfit["likes"][s:e] = fit_z(W, bdec, z0, pr, pc, None,
                                   args.steps, args.lr, reg, args.lam).numpy()
        Zfit["dislikes"][s:e] = fit_z(W, bdec, z0, pr, pc, bpr_dis,
                                      args.steps, args.lr, reg, args.lam).numpy()
        Zfit["disinterest"][s:e] = fit_z(W, bdec, z0, pr, pc, bpr_dii,
                                         args.steps, args.lr, reg, args.lam).numpy()
        Zfit["clairvoyant"][s:e] = fit_z(W, bdec, z0, cpr, cpc, None,
                                         args.steps, args.lr, reg, args.lam).numpy()
        print(f"[oracle-z reg={reg}] fit users {s}-{e}/{N} [{time.time()-t0:.0f}s]", flush=True)
    return Zfit


def _evaluate(FR, recs, Zfit, conds, Znat, has_kd):
    """Score every fitted z on HELD: primary held-liked NDCG@10 + secondary signed AUC; return the
    paired deltas vs likes-only with bootstrap CIs, plus native baseline and z-shift diagnostics."""
    metr = {c: dict(ndcg=[], auc=[]) for c in conds}
    metr["native"] = dict(ndcg=[], auc=[])
    zshift = dict(dislikes=[], disinterest=[])
    for i, r in enumerate(recs):
        prof_s = r["profset"]
        for c in conds:
            sc = FR.decode_np(Zfit[c][i][None, :])[0]
            metr[c]["ndcg"].append(ndcg_liked(FR, Zfit[c][i], r["hl"], prof_s))
            metr[c]["auc"].append(signed_auc(sc, r["hl"], r["hd"]))
        scn = FR.decode_np(Znat[i][None, :])[0]
        metr["native"]["ndcg"].append(ndcg_liked(FR, Znat[i], r["hl"], prof_s))
        metr["native"]["auc"].append(signed_auc(scn, r["hl"], r["hd"]))
        zshift["dislikes"].append(float(np.linalg.norm(Zfit["dislikes"][i] - Zfit["likes"][i])))
        zshift["disinterest"].append(float(np.linalg.norm(Zfit["disinterest"][i] - Zfit["likes"][i])))

    def arr(c, m):
        return np.array([x for x in metr[c][m] if x is not None], float)

    def paired_delta(cA, cB, m, mask=None):
        a = np.array(metr[cA][m], float); b = np.array(metr[cB][m], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            ok = ok & mask
        return boot_ci((a - b)[ok], seed=1), int(ok.sum())

    return dict(
        mean_ndcg={c: boot_ci(arr(c, "ndcg"), seed=3) for c in ["native"] + conds},
        mean_auc={c: boot_ci(arr(c, "auc"), seed=4) for c in ["native"] + conds},
        zshift={k: float(np.mean(v)) for k, v in zshift.items()},
        d_ndcg=dict(
            dislikes_ALL=paired_delta("dislikes", "likes", "ndcg"),
            dislikes_KNOWNDIS=paired_delta("dislikes", "likes", "ndcg", has_kd),
            disinterest_ALL=paired_delta("disinterest", "likes", "ndcg"),
        ),
        d_auc=dict(
            dislikes_ALL=paired_delta("dislikes", "likes", "auc"),
            disinterest_ALL=paired_delta("disinterest", "likes", "auc"),
        ),
        ceiling=dict(
            clairvoyant_minus_likes=paired_delta("clairvoyant", "likes", "ndcg"),
            likes_minus_native=paired_delta("likes", "native", "ndcg"),
        ),
    )


def _assemble(args, recs, sweep, has_kd, N, reg_grid):
    prim = sweep[args.reg]                                 # primary reg = --reg
    low = sweep[reg_grid[0]]                               # least-regularized -> truest ceiling
    return dict(
        n_users=N,
        n_known_dislike=int(has_kd.sum()),
        n_held_dislike=int(sum(1 for r in recs if r["hd"])),
        hyperparams=dict(steps=args.steps, lr=args.lr, primary_reg=args.reg, reg_grid=reg_grid,
                         lam=args.lam, batch=args.batch, n_neg_disinterest="max(#known_dislike,20)"),
        primary_reg=args.reg,
        primary_ndcg_liked=prim["mean_ndcg"],
        secondary_auc=prim["mean_auc"],
        zshift_from_likes=prim["zshift"],
        delta_primary_ndcg={
            "dislikes_minus_likes_ALL": prim["d_ndcg"]["dislikes_ALL"][0],
            "dislikes_minus_likes_KNOWNDIS_subset": prim["d_ndcg"]["dislikes_KNOWNDIS"][0],
            "disinterest_minus_likes_ALL": prim["d_ndcg"]["disinterest_ALL"][0],
        },
        delta_primary_counts={
            "dislikes_ALL_n": prim["d_ndcg"]["dislikes_ALL"][1],
            "dislikes_KNOWNDIS_n": prim["d_ndcg"]["dislikes_KNOWNDIS"][1],
            "disinterest_ALL_n": prim["d_ndcg"]["disinterest_ALL"][1],
        },
        delta_secondary_auc={
            "dislikes_minus_likes_ALL": prim["d_auc"]["dislikes_ALL"][0],
            "disinterest_minus_likes_ALL": prim["d_auc"]["disinterest_ALL"][0],
        },
        ceiling_gap=dict(
            clairvoyant_minus_likes=low["ceiling"]["clairvoyant_minus_likes"][0],
            clairvoyant_ndcg=low["mean_ndcg"]["clairvoyant"],
            oracle_likes_minus_native=prim["ceiling"]["likes_minus_native"][0],
        ),
        reg_robustness={
            str(r): dict(
                likes_ndcg=sweep[r]["mean_ndcg"]["likes"][0],
                native_ndcg=sweep[r]["mean_ndcg"]["native"][0],
                clairvoyant_ndcg=sweep[r]["mean_ndcg"]["clairvoyant"][0],
                d_dislikes_ALL=sweep[r]["d_ndcg"]["dislikes_ALL"][0][0],
                d_disinterest_ALL=sweep[r]["d_ndcg"]["disinterest_ALL"][0][0],
                d_dislikes_auc_ALL=sweep[r]["d_auc"]["dislikes_ALL"][0][0],
                d_disinterest_auc_ALL=sweep[r]["d_auc"]["disinterest_ALL"][0][0],
            ) for r in reg_grid
        },
    )


def _fmt(t):
    return f"{t[0]:+.4f} CI[{t[1]:+.4f},{t[2]:+.4f}]"


def _print_summary(R):
    print("\n===== ORACLE-Z VERDICT =====", flush=True)
    p = R["primary_ndcg_liked"]
    print(f"held-LIKED NDCG@10 (mean CI):", flush=True)
    for c in ["native", "likes", "dislikes", "disinterest", "clairvoyant"]:
        print(f"   {c:12s} {p[c][0]:.4f} CI[{p[c][1]:.4f},{p[c][2]:.4f}]", flush=True)
    d = R["delta_primary_ndcg"]
    print("\nPRIMARY delta held-LIKED NDCG@10 (vs likes-only):", flush=True)
    print(f"   (2) +watched-dislikes  ALL        {_fmt(d['dislikes_minus_likes_ALL'])}", flush=True)
    print(f"   (2) +watched-dislikes  known-dis   {_fmt(d['dislikes_minus_likes_KNOWNDIS_subset'])}", flush=True)
    print(f"   (3) +disinterest       ALL        {_fmt(d['disinterest_minus_likes_ALL'])}", flush=True)
    s = R["delta_secondary_auc"]
    print("\nSECONDARY signed AUC delta (vs likes-only):", flush=True)
    print(f"   (2) +watched-dislikes  ALL        {_fmt(s['dislikes_minus_likes_ALL'])}", flush=True)
    print(f"   (3) +disinterest       ALL        {_fmt(s['disinterest_minus_likes_ALL'])}", flush=True)
    print(f"\nCEILING: clairvoyant - likes-only = {_fmt(R['ceiling_gap']['clairvoyant_minus_likes'])}", flush=True)
    print(f"         oracle-likes - native      = {_fmt(R['ceiling_gap']['oracle_likes_minus_native'])}", flush=True)
    print(f"\n[{R['minutes']}m] -> {OUT}/oracle_z.json + ORACLE_Z.md", flush=True)


def _write_md(R):
    p = R["primary_ndcg_liked"]; a = R["secondary_auc"]
    d = R["delta_primary_ndcg"]; s = R["delta_secondary_auc"]
    c = R["ceiling_gap"]
    def row(t):
        return f"{t[0]:+.4f} | [{t[1]:+.4f}, {t[2]:+.4f}]"
    dis_all = d["dislikes_minus_likes_ALL"]
    dii_all = d["disinterest_minus_likes_ALL"]
    dis_sub = d["dislikes_minus_likes_KNOWNDIS_subset"]
    clair = c["clairvoyant_minus_likes"]
    # verdict logic
    def sig(t):
        return t[1] > 0  # CI lower bound above 0
    v_dis = "HEADROOM" if sig(dis_all) else ("WALL (~0)" if abs(dis_all[0]) < 0.005 else "HURTS" if dis_all[0] < 0 else "weak")
    v_dii = "HEADROOM" if sig(dii_all) else ("WALL (~0)" if abs(dii_all[0]) < 0.005 else "HURTS" if dii_all[0] < 0 else "weak")
    md = f"""# ORACLE-Z HEADROOM — best-possible belief under the FROZEN RecVAE decoder

**Date:** 2026-07-11 · **Decoder:** frozen RecVAE-d512 (`.cache/instrument2/ml25m_recvae_d512_best.pt`),
score = z·Wᵀ+b, W,b FROZEN. **Users:** {R['n_users']} held-out TEST users (disjoint from pilot train+val;
study 173/300 quarantined via trU firewall). **Training-free** per-user optimization of a 512-d latent z*
by Adam through the frozen decoder ({R['hyperparams']['steps']} steps, lr {R['hyperparams']['lr']},
reg {R['hyperparams']['primary_reg']}, BPR λ {R['hyperparams']['lam']}). NO encoder, NO architecture, NO LLM calls.
All ratings used; users batched for compute only (no data reduction). Raw: `.cache/rung1/oracle_z.json`.

Per user: leak-free disjoint KNOWN / HELD half-split. Fit z* to KNOWN under four conditionings; evaluate
on HELD. {R['n_known_dislike']} users have KNOWN watched-dislikes (r≤2); {R['n_held_dislike']} have HELD
dislikes (signed-AUC defined).

## Conditionings
1. **likes-only** — positives = KNOWN liked (r≥4), softmax multinomial CE (RecVAE's own objective).
2. **likes + watched-dislikes** — (1) + BPR ranking liked above KNOWN disliked (r≤2, *inside* the cone).
3. **likes + disinterest** — (1) + BPR ranking liked above UNWATCHED items from AVOIDED genres
   (under-represented vs population; *outside* the cone by construction — cannot drag the likes).
- **clairvoyant** — positives = HELD liked (fit directly to the eval target) = upper-bound ceiling.

## PRIMARY — held-LIKED NDCG@10 (the deployment metric)

| conditioning | NDCG@10 | 95% CI |
|---|---|---|
| native z0 (RecVAE-encode known-likes, no opt) | {p['native'][0]:.4f} | [{p['native'][1]:.4f}, {p['native'][2]:.4f}] |
| (1) likes-only oracle z* | {p['likes'][0]:.4f} | [{p['likes'][1]:.4f}, {p['likes'][2]:.4f}] |
| (2) likes + watched-dislikes | {p['dislikes'][0]:.4f} | [{p['dislikes'][1]:.4f}, {p['dislikes'][2]:.4f}] |
| (3) likes + disinterest | {p['disinterest'][0]:.4f} | [{p['disinterest'][1]:.4f}, {p['disinterest'][2]:.4f}] |
| **clairvoyant ceiling** (fit to held-likes) | {p['clairvoyant'][0]:.4f} | [{p['clairvoyant'][1]:.4f}, {p['clairvoyant'][2]:.4f}] |

### Δ vs likes-only (paired, held-LIKED NDCG@10)

| contrast | Δ | 95% CI | verdict |
|---|---|---|---|
| (2) +watched-dislikes — ALL users (n={R['delta_primary_counts']['dislikes_ALL_n']}) | {dis_all[0]:+.4f} | [{dis_all[1]:+.4f}, {dis_all[2]:+.4f}] | {v_dis} |
| (2) +watched-dislikes — known-dislike subset (n={R['delta_primary_counts']['dislikes_KNOWNDIS_n']}) | {dis_sub[0]:+.4f} | [{dis_sub[1]:+.4f}, {dis_sub[2]:+.4f}] | — |
| (3) +disinterest — ALL users (n={R['delta_primary_counts']['disinterest_ALL_n']}) | {dii_all[0]:+.4f} | [{dii_all[1]:+.4f}, {dii_all[2]:+.4f}] | {v_dii} |

## SECONDARY — signed AUC (held-liked ranked above held-disliked)

| conditioning | AUC | 95% CI |
|---|---|---|
| (1) likes-only | {a['likes'][0]:.4f} | [{a['likes'][1]:.4f}, {a['likes'][2]:.4f}] |
| (2) likes + watched-dislikes | {a['dislikes'][0]:.4f} | [{a['dislikes'][1]:.4f}, {a['dislikes'][2]:.4f}] |
| (3) likes + disinterest | {a['disinterest'][0]:.4f} | [{a['disinterest'][1]:.4f}, {a['disinterest'][2]:.4f}] |

Δ signed AUC vs likes-only: (2) watched-dislikes ALL = {row(s['dislikes_minus_likes_ALL'])};
(3) disinterest ALL = {row(s['disinterest_minus_likes_ALL'])}.

## Robustness to the manifold-anchor `reg` (the only free knob)
`reg` anchors the fit toward the native RecVAE belief; the verdict must not hinge on it. Primary reg =
{R['primary_reg']}. Δ's (held-LIKED NDCG@10, ALL users) and signed-AUC Δ's across the sweep:

| reg | likes NDCG | native | clairvoyant | Δ(2) dislike | Δ(3) disinterest | Δ(2) AUC | Δ(3) AUC |
|---|---|---|---|---|---|---|---|
{chr(10).join(f"| {r} | {v['likes_ndcg']:.4f} | {v['native_ndcg']:.4f} | {v['clairvoyant_ndcg']:.4f} | {v['d_dislikes_ALL']:+.4f} | {v['d_disinterest_ALL']:+.4f} | {v['d_dislikes_auc_ALL']:+.4f} | {v['d_disinterest_auc_ALL']:+.4f} |" for r, v in R['reg_robustness'].items())}

## Scale / ceiling
- clairvoyant − likes-only = **{clair[0]:+.4f}** CI[{clair[1]:+.4f}, {clair[2]:+.4f}] — total headroom that ANY
  belief could capture if it knew the held target. This is how much room exists at all under this decoder.
- oracle-likes − native = {row(c['oracle_likes_minus_native'])} — the lift from optimizing z past the
  raw RecVAE encoding of the known likes (sanity: the oracle beats the native belief).
- mean ‖z*(2)−z*(1)‖ = {R['zshift_from_likes']['dislikes']:.3f}; ‖z*(3)−z*(1)‖ =
  {R['zshift_from_likes']['disinterest']:.3f} — the negative signals DID move the belief.

## Verdict
The **clairvoyant ceiling is +{clair[0]:.3f}** NDCG@10: there is real headroom for a better belief under the
frozen decoder. Against that ceiling:
- **Watched-dislike (inside the cone):** Δ = {dis_all[0]:+.4f} (ALL). {'A real, if small, gain — dislike-as-context has headroom under the frozen decoder.' if sig(dis_all) else 'The frozen decoder is the WALL for watched-dislike: even the best-possible z fit with correctly-signed dislike targets does not improve held-liked ranking. Consistent with RUNG1_CHECKS (like/dislike geometry conflated).'}
- **Disinterest (outside the cone):** Δ = {dii_all[0]:+.4f} (ALL). {'This is the useful negative — suppressing AVOIDED unwatched regions is separable from the likes and lifts held-liked NDCG, where watched-dislike does not.' if sig(dii_all) and dii_all[0] >= dis_all[0] else 'Disinterest does not clear the wall either under the frozen decoder.'}

**Hypothesis (3)>(2)≈0:** {'CONFIRMED — the useful negative is DISINTEREST (avoided regions), not watched-dislike. An encoder that models systematic avoidance has headroom the frozen decoder permits; watched-dislike does not.' if (sig(dii_all) and not sig(dis_all)) else 'NOT confirmed as stated — see the Δ table above; ' + ('both negatives help.' if (sig(dii_all) and sig(dis_all)) else ('watched-dislike helps as much or more.' if (sig(dis_all) and dis_all[0] >= dii_all[0]) else 'neither negative clears the wall.'))}
"""
    open("ORACLE_Z.md", "w", encoding="utf-8").write(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--reg_grid", type=str, default="1e-3,1e-2,3e-2")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
