r"""concept_fold.py -- ARM C-LITE: a small TRAINED gated FOLD-TO-POINT concept module on the FROZEN i25
tower (author directive 2026-07-24: the Arm-A additive numbers are unacceptable for the vision --
concepts must carry whole interviews).

JULY RECORD THIS IMPLEMENTS (memory `concept-fold-additive-union-vs-point-intersection`): the additive
UNION saturates (+0.011 concepts-only here; +0.028 for the old trained additive head) while fold-to-POINT
intersection COMPOUNDS (+0.052 tail@kc32, the Paper-B u1 precedent). Recipe mirrored from the July gated
fold-in: gated zero-init module on the FROZEN tower, learned concept embeddings (whitened-centroid init),
per-user normalization (anti-saturation).

ARCHITECTURE (few hundred k params; tower and decoder untouched):
    z_run = z_items (the frozen tower fold; 0 = intercept when no items)
    for each concept answer (c, v) in CANONICAL descending-value order:
        p      = emb[c] * v                       # value-scaled candidate point (emb trainable,
                                                  #   init = whitened member centroid * init_scale)
        g      = sigmoid(gate([z_run, p, v]))     # context-dependent intersection gate
        delta  = mlp([z_run, p, v])               # fold-to-point update (FINAL LAYER ZERO-INIT)
        z_run  = z_run + g * delta                # recursive intersection with the running point
    U = z_run - z_items;  U = U * min(1, cap/||U||)   # PER-USER NORM (anti-saturation, trainable cap)
    return z_items + U
  Zero-init => output == z_items BIT-EXACTLY at init (no-op start; the tower's certified behaviour is
  the floor, exactly the July gated-fold recipe). ORDER: the recursion is order-dependent in principle;
  answers are folded in canonical DESC-value order (ties by concept id) -- deterministic and documented
  (a permutation-invariant two-pass variant is the fallback arm if order artifacts show).

TRAINING (evening class -- DO NOT LAUNCH until tonight's t2final retrain lands; this file only builds):
  item-masked curriculum (per the author brief; DESIGN_RECONCILED doc not found in docs/design -- the
  coordinator's inline spec is the source): per example
    * split the user's graded profile into INPUT pool and TARGET held likes (leak-free);
    * labels = SEL watch-lift concepts computed FROM THE INPUT POOL ONLY (targets excluded -- C2 clause),
      m ~ U{1..M_MAX} revealed with SEL-graded values;
    * THE MEMBER ITEMS THAT GENERATED EACH REVEALED CONCEPT ARE DROPPED FROM THE ITEM INPUT (the mask:
      concepts must carry INDEPENDENT signal, not echo the items that produced them);
    * item budget k ~ {0, 1..8, dropout regime} on the remaining items (k=0 concepts-only examples
      included -- the acceptance bar is the concepts-only curve);
    * loss = NLL(frozen decoder(z_out), held likes). Tower + decoder frozen, excluded from the optimizer;
      drift-asserted per epoch.
ACCEPTANCE (author bar): concepts-only curve MONOTONE to m=8+ on FULL and TAIL; a meaningful fraction of
the item curve; mixed additivity preserved (m2k2 >= Arm A's +0.0125); G5 member-specificity AUC +
G-collinearity still pass. Arm A stays in the eval as the untrained comparator row.

Usage:
  python src/instrument/concept_fold.py --smoke
  python src/instrument/concept_fold.py --train --tag cfold            # DO NOT LAUNCH YET (author gate)
  python src/instrument/concept_fold.py --eval_curve .cache/instrument/cfold_best.pt [--full_threads]
"""
import os
import sys
_FULL = "--full_threads" in sys.argv or "--train" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import level_to_sv, pack_tokens, NLEV, log
from belief_layer import build_concept_dirs, ConceptMean

assert "load_answerer" not in globals()                      # retired-answerer ban

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
M_MAX = 8                                                    # concepts revealed per training example
SEED = 4242


# ============================================================================= the module
class ConceptFoldNet(nn.Module):
    """Gated recursive fold-to-point over concept answers, composing with the frozen tower fold.
    Zero-init (delta final layer) => bit-exact no-op at init. Per-user norm cap (anti-saturation)."""

    def __init__(self, nc, d=200, h=256, conc_init=None, init_scale=1.0):
        super().__init__()
        self.nc, self.d = nc, d
        self.emb = nn.Embedding(nc, d)
        if conc_init is not None:                            # whitened member centroids (July: LEARNED
            with torch.no_grad():                            # emb beats frozen; init from the geometry)
                self.emb.weight.copy_(torch.as_tensor(conc_init, dtype=torch.float32) * init_scale)
        else:
            nn.init.normal_(self.emb.weight, std=0.02)
        d_in = 2 * d + 1                                     # [z_run, p, v]
        self.gate = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, 1))
        self.mlp = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)   # no-op at init
        self.log_cap = nn.Parameter(torch.tensor(1.6094))    # softplus-> cap ~ 5.0 update-norm ceiling

    @staticmethod
    def canonical_order(cids, vals):
        """DESC value, ties by concept id -- deterministic fold order (documented)."""
        idx = sorted(range(len(cids)), key=lambda i: (-float(vals[i]), int(cids[i])))
        return [cids[i] for i in idx], [vals[i] for i in idx]

    def forward(self, z_items, conc_ids, conc_vals):
        """z_items: (B,d) frozen tower folds. conc_ids/vals: length-B lists of per-user answer lists.
        Returns (B,d). Sequential over the max answer count; users with fewer answers stop updating."""
        B = z_items.shape[0]
        z = z_items
        mmax = max((len(c) for c in conc_ids), default=0)
        for step in range(mmax):
            has = torch.tensor([len(conc_ids[r]) > step for r in range(B)])
            if not bool(has.any()):
                break
            cid = torch.tensor([conc_ids[r][step] if len(conc_ids[r]) > step else 0 for r in range(B)])
            val = torch.tensor([float(conc_vals[r][step]) if len(conc_vals[r]) > step else 0.0
                                for r in range(B)]).unsqueeze(1)
            p = self.emb(cid) * val
            x = torch.cat([z, p, val], dim=1)
            g = torch.sigmoid(self.gate(x))
            delta = self.mlp(x)
            upd = g * delta * has.float().unsqueeze(1)
            z = z + upd
        U = z - z_items                                      # per-user norm (anti-saturation)
        cap = F.softplus(self.log_cap)
        nrm = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        U = U * torch.minimum(torch.ones_like(nrm), cap / nrm)
        return z_items + U

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================= shared data machinery
def build_member_matrix(members, tags, ni):
    """(ni, C) binary CSR + per-concept train-mass rate."""
    Mm = sparse.csr_matrix((np.ones(sum(len(members[t]) for t in tags), np.float32),
                            (np.concatenate([members[t] for t in tags]),
                             np.concatenate([np.full(len(members[t]), i)
                                             for i, t in enumerate(tags)]))),
                           shape=(ni, len(tags)))
    return Mm


def sel_concepts_from_items(item_sids, Mm, grate, m, member_sets):
    """SEL top-m concepts computed from an ITEM SET ONLY (the input pool; targets never enter).
    Returns (concept idx list, SEL-graded value list, the member items supporting each) or ([],[],[])."""
    if len(item_sids) == 0:
        return [], [], []
    x = np.zeros(Mm.shape[0], np.float32); x[item_sids] = 1.0
    counts = x @ Mm                                          # dense (C,)
    counts = np.asarray(counts).ravel()
    lift = (counts / max(len(item_sids), 1)) / np.maximum(grate, 1e-12)
    lift[counts < 2] = -np.inf
    pos = np.flatnonzero(np.isfinite(lift) & (lift > 1.0))
    if len(pos) == 0:
        return [], [], []
    order = pos[np.argsort(-lift[pos])][:m]
    l1 = np.log(lift[order[0]])
    vals = np.clip(np.log(lift[order]) / max(l1, 1e-9), 0.25, 1.0)
    support = [np.intersect1d(item_sids, member_sets[int(c)]) for c in order]
    return [int(c) for c in order], [float(v) for v in vals], support


# ============================================================================= training (DO NOT LAUNCH)
def make_example(u, rng, Mm, grate, member_sets, p_conc_only=0.35):
    """Item-masked curriculum example: (input item sids/levels AFTER member-drop, concept ids, concept
    vals, target likes). None if unusable."""
    items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(items) < 4 or len(liked) < 2:
        return None
    # input pool / target split (leak-free: targets never in the pool)
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(items, tg)
    pool_s, pool_l = items[keep], lv[keep]
    if len(pool_s) < 2:
        return None
    # SEL labels FROM THE INPUT POOL ONLY
    m = int(rng.integers(1, M_MAX + 1))
    cids, cvals, support = sel_concepts_from_items(pool_s, Mm, grate, m, member_sets)
    if len(cids) == 0:
        return None
    # ITEM MASK: drop every member item that generated a revealed concept (independent-signal curriculum)
    drop = np.unique(np.concatenate(support)) if support else np.empty(0, np.int64)
    mkeep = ~np.isin(pool_s, drop)
    rem_s, rem_l = pool_s[mkeep], pool_l[mkeep]
    # item budget: concepts-only with prob p_conc_only, else k ~ U{1..8} of the remaining items
    if rng.random() < p_conc_only or len(rem_s) == 0:
        in_s, in_l = np.empty(0, np.int64), np.empty(0, np.int64)
    else:
        k = int(rng.integers(1, min(8, len(rem_s)) + 1))
        pick = rng.choice(len(rem_s), size=k, replace=False)
        in_s, in_l = rem_s[pick], rem_l[pick]
    tgt = np.setdiff1d(tg, in_s)
    if len(tgt) == 0:
        return None
    return in_s, in_l, cids, cvals, tgt


def fold_items_frozen(enc, seqs, batch=256):
    """Frozen tower fold (no grad)."""
    Z = torch.zeros(len(seqs), enc.d_out)
    enc.eval()
    with torch.no_grad():
        for st in range(0, len(seqs), batch):
            rows = [(np.asarray(s, np.int64), np.asarray(l, np.int64), level_to_sv(np.asarray(l)))
                    for s, l in seqs[st:st + batch]]
            ids, vals, pad, lvs = pack_tokens(rows)
            Z[st:st + len(rows)] = enc(ids, vals, pad, lvs)
    return Z


def train(args):
    """Evening-class training run. AUTHOR GATE: do not launch until tonight's t2final retrain lands --
    the module retrains cheaply on the new tower (ckpt-swap by --snapshot)."""
    from train_tower_t2 import reproduce_partition, build_train_profiles, build_model, compute_head_mask
    import metrics as M
    import pandas as pd
    PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
    GENOME = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    head_mask, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    users = build_train_profiles(raw, tr_set, show2id, max_users=(args.max_users or None))
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, _, _, _ = build_model(a, ni, cnt)
    blob = torch.load(args.snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"]); enc.eval()
    for p in list(enc.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)                              # frozen; excluded from the optimizer
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    W0 = Wd.clone(); b0 = bd.clone()
    # genome concepts + whitened init
    m2s = {int(m): i for i, m in enumerate(usid)}
    g = pd.read_csv(GENOME); g = g[g["relevance"] >= 0.5]; g = g[g["movieId"].isin(m2s)]
    g["sid"] = g["movieId"].map(m2s).astype(np.int64)
    members = {int(t): np.sort(sub["sid"].values) for t, sub in g.groupby("tagId")}
    members = {t: mm for t, mm in members.items() if len(mm) >= 30}
    tags = sorted(members.keys())
    d_c, d_raw, w_c = build_concept_dirs(Wd, members, tags)
    Mm = build_member_matrix(members, tags, ni)
    grate = (cnt @ np.asarray(Mm.todense())) / max(cnt.sum(), 1e-9)
    member_sets = {i: members[t] for i, t in enumerate(tags)}
    net = ConceptFoldNet(len(tags), d=enc.d_out, h=args.hidden, conc_init=d_c.numpy())
    log(f"[cfold] TRAINABLE params = {net.n_params():,} (emb {len(tags)}x{enc.d_out} + gate + mlp + cap)")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    ckb = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    best = -1.0; start_ep = 0
    if args.resume_from_best and os.path.exists(ckb):
        # resume from the best ckpt's weights (fresh optimizer; the 2026-07-24 harness-kill recovery).
        blob = torch.load(ckb, map_location="cpu")
        assert blob["tags"] == tags, "concept vocabulary drift on resume"
        net.load_state_dict(blob["net"])
        start_ep = int(blob["epoch"]); best = float(blob["val"]["m8_full"])
        log(f"[cfold] RESUMED FROM BEST ep{start_ep} (val m8={best:.4f}); fresh optimizer")
    # val harness: concepts-only m={2,8} on the val cohort (built once; the author acceptance metric)
    import run_battery_phaseA as PA_
    for ep in range(start_ep, args.epochs):
        rng = np.random.default_rng(1000 + ep)
        order = rng.permutation(len(users))
        net.train(); run = 0.0; nb = 0; t0 = time.time()
        for st in range(0, len(order), args.batch):
            uix = order[st:st + args.batch]
            exs = [make_example(users[i], rng, Mm, grate, member_sets) for i in uix]
            exs = [e for e in exs if e is not None]
            if not exs:
                continue
            Z = fold_items_frozen(enc, [(e[0], e[1]) for e in exs])
            cids = [ConceptFoldNet.canonical_order(e[2], e[3])[0] for e in exs]
            cvals = [ConceptFoldNet.canonical_order(e[2], e[3])[1] for e in exs]
            zout = net(Z, cids, cvals)
            logits = zout @ Wd.T + bd
            logsm = F.log_softmax(logits, dim=-1)
            tgt = torch.zeros(len(exs), ni)
            for r, e in enumerate(exs):
                tgt[r, e[4]] = 1.0
            nll = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(nll); nb += 1
            if nb % 100 == 0:
                log(f"  ep{ep} b{nb} NLL={run/nb:.4f} cap={float(F.softplus(net.log_cap)):.2f} "
                    f"({(time.time()-t0)/60:.1f}m)")
        assert torch.equal(Wd, W0) and torch.equal(bd, b0), "frozen decoder drifted"
        # epoch val: concepts-only m=8 (the acceptance axis) -- cheap, no encoder at k=0
        vm = quick_val(enc, Wd, bd, net, args)
        score = vm["m8_full"]
        log(f"[ep{ep+1}] NLL={run/max(nb,1):.4f} val concepts-only m2={vm['m2_full']:.4f} "
            f"m8={vm['m8_full']:.4f} (best {best:.4f}) ({(time.time()-t0)/60:.1f}m)")
        if score > best:
            best = score
            torch.save({"net": net.state_dict(), "tags": tags, "epoch": ep + 1, "val": vm,
                        "hidden": args.hidden, "snapshot": os.path.basename(args.snapshot)}, ckb)
            log(f"[ep{ep+1}] best saved -> {ckb}")
    log(f"[cfold] done; best val m8 full = {best:.4f}")


_VAL_CACHE = {}


def quick_val(enc, Wd, bd, net, args):
    """Concepts-only m in {2,8} full@10 on the val cohort (cheap: k=0 needs no encoder)."""
    import run_battery_phaseA as PA_
    from concepts_only_curve import sel_top_concepts
    from run_battery_phaseA import load_genome, ndcg10_from_scores
    if "ctx" not in _VAL_CACHE:
        ctx = PA_.build_real_ctx(args.snapshot)
        members = load_genome(ctx)
        tags = sorted(members.keys())
        _VAL_CACHE["sel"] = sel_top_concepts(ctx, members, tags, M_MAX)
        _VAL_CACHE["ctx"] = ctx
    ctx = _VAL_CACHE["ctx"]; sel = _VAL_CACHE["sel"]
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel]
    out = {}
    net.eval()
    with torch.no_grad():
        for m in (2, 8):
            f = np.full(ctx.n, np.nan)
            for st in range(0, len(rows), 500):
                chunk = rows[st:st + 500]
                Z0 = torch.zeros(len(chunk), ctx.enc.d_out)
                cids = []; cvals = []
                for r in chunk:
                    cs, vs = sel[r]
                    ci, vv = ConceptFoldNet.canonical_order(list(cs[:m]), list(vs[:m]))
                    cids.append(ci); cvals.append(vv)
                zo = net(Z0, cids, cvals)
                S = (zo @ Wd.T + bd).numpy().astype(np.float32)
                ff, _ = ndcg10_from_scores(S, ctx.va_tr[chunk], ctx.va_te[chunk], None)
                f[np.asarray(chunk)] = ff
            out[f"m{m}_full"] = float(np.nanmean(f[rows]))
    return out


# ============================================================================= acceptance eval
def eval_curve(args):
    """The author-bar acceptance eval: Arm C concepts-only curve m in {1,2,4,8} (full+tail, monotone
    verdict) with the UNTRAINED Arm A as comparator row, items-at-k, mixed m2k2, G5 member AUC +
    G-collinearity. -> experiments/battery/concepts_only_curve_armC.json"""
    import run_battery_phaseA as PA_
    from run_battery_phaseA import build_real_ctx, load_genome, ndcg10_from_scores, eval_tokens, \
        bootstrap_ci, spearman
    from concepts_only_curve import sel_top_concepts, concepts_only_scores, diversify_sel, K_SEED
    from train_tower_t2 import truncate_graded
    blob = torch.load(args.eval_curve, map_location="cpu")
    ctx = build_real_ctx(args.snapshot)
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    members = load_genome(ctx)
    tags = sorted(members.keys())
    assert tags == blob["tags"], "concept vocabulary drift between train and eval"
    d_c, d_raw, w_c = build_concept_dirs(ctx.Wd, members, tags)
    net = ConceptFoldNet(len(tags), d=ctx.d, h=blob["hidden"], conc_init=None)
    net.load_state_dict(blob["net"]); net.eval()
    # selection: DIVERSIFIED default (author ruling; ask-order upside) for the main curve; the
    # CORRELATED top-SEL set feeds the REDUNDANCY-ROBUSTNESS acceptance row (m8 !< m4).
    sel40 = sel_top_concepts(ctx, members, tags, 40)
    sel_top = {r: (cs[:M_MAX], vs[:M_MAX]) for r, (cs, vs) in sel40.items()}
    sel = diversify_sel(sel40, torch.as_tensor(d_c), m_max=M_MAX)
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel and r in sel_top]
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    f_int, t_int = eval_tokens(ctx, empty, ctx.va_tr, rows=rows)
    out = {"analysis": "concepts_only_curve_armC", "ckpt": os.path.basename(args.eval_curve),
           "n_users": len(rows), "trainable_params": net.n_params(),
           "intercept": {"full": float(np.nanmean(f_int[rows])), "tail": float(np.nanmean(t_int[rows]))},
           "armC": {"full@10": {}, "tail@10": {}}, "armA_comparator": {"full@10": {}, "tail@10": {}}}
    cmA = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0, beta_cold=1.0, floor_rho=0.0)   # curve winner cfg
    with torch.no_grad():
        for m in (1, 2, 4, 8):
            fC = np.full(ctx.n, np.nan); tC = np.full(ctx.n, np.nan)
            for st in range(0, len(rows), 500):
                chunk = rows[st:st + 500]
                Z0 = torch.zeros(len(chunk), ctx.d)
                cids = []; cvals = []
                for r in chunk:
                    cs, vs = sel[r]
                    ci, vv = ConceptFoldNet.canonical_order(list(cs[:m]), list(vs[:m]))
                    cids.append(ci); cvals.append(vv)
                zo = net(Z0, cids, cvals)
                S = (zo @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                ff, tt = ndcg10_from_scores(S, ctx.va_tr[chunk], ctx.va_te[chunk], ctx.head_mask)
                fC[np.asarray(chunk)] = ff; tC[np.asarray(chunk)] = tt
            out["armC"]["full@10"][str(m)] = float(np.nanmean(fC[rows]))
            out["armC"]["tail@10"][str(m)] = float(np.nanmean(tC[rows]))
            fA, tA = concepts_only_scores(ctx, cmA, sel, rows, m)
            out["armA_comparator"]["full@10"][str(m)] = float(np.nanmean(fA[rows]))
            out["armA_comparator"]["tail@10"][str(m)] = float(np.nanmean(tA[rows]))
            log(f"[eval m={m}] armC {out['armC']['full@10'][str(m)]:.4f}/"
                f"{out['armC']['tail@10'][str(m)]:.4f} | armA "
                f"{out['armA_comparator']['full@10'][str(m)]:.4f}/"
                f"{out['armA_comparator']['tail@10'][str(m)]:.4f}")
        # REDUNDANCY ROBUSTNESS (author acceptance clause 2026-07-24): folding the CORRELATED top-SEL
        # m=8 set must NOT decline vs its m=4 prefix (full AND tail) -- the operator itself must fix
        # redundancy; div-selection is ask-order upside, not a crutch.
        red = {}
        for m in (4, 8):
            fR = np.full(ctx.n, np.nan); tR = np.full(ctx.n, np.nan)
            for st in range(0, len(rows), 500):
                chunk = rows[st:st + 500]
                Z0 = torch.zeros(len(chunk), ctx.d)
                cids = []; cvals = []
                for r in chunk:
                    cs, vs = sel_top[r]
                    ci, vv = ConceptFoldNet.canonical_order(list(cs[:m]), list(vs[:m]))
                    cids.append(ci); cvals.append(vv)
                zo = net(Z0, cids, cvals)
                S = (zo @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                ff, tt = ndcg10_from_scores(S, ctx.va_tr[chunk], ctx.va_te[chunk], ctx.head_mask)
                fR[np.asarray(chunk)] = ff; tR[np.asarray(chunk)] = tt
            red[str(m)] = {"full": float(np.nanmean(fR[rows])), "tail": float(np.nanmean(tR[rows]))}
        out["redundancy_topSEL"] = red
        out["redundancy_robust_full"] = bool(red["8"]["full"] >= red["4"]["full"] - 1e-9)
        out["redundancy_robust_tail"] = bool(red["8"]["tail"] >= red["4"]["tail"] - 1e-9)
        log(f"[eval redundancy topSEL] m4 {red['4']['full']:.4f}/{red['4']['tail']:.4f} -> "
            f"m8 {red['8']['full']:.4f}/{red['8']['tail']:.4f} robust="
            f"{out['redundancy_robust_full']}|{out['redundancy_robust_tail']}")
        # acceptance verdicts
        for arm in ("armC", "armA_comparator"):
            sf = [out[arm]["full@10"][str(m)] for m in (1, 2, 4, 8)]
            stl = [out[arm]["tail@10"][str(m)] for m in (1, 2, 4, 8)]
            out[arm + "_monotone_full"] = bool(all(sf[i + 1] >= sf[i] - 1e-9 for i in range(3)))
            out[arm + "_monotone_tail"] = bool(all(stl[i + 1] >= stl[i] - 1e-9 for i in range(3)))
        out["G_collinearity_PASS"] = bool(out["armC"]["full@10"]["1"] > out["intercept"]["full"])
        # items-at-k + fraction-of-item-curve
        items = {}
        for m in (1, 2, 4, 8):
            Lm = truncate_graded(ctx.L_val, m, K_SEED[m])
            toks = [(Lm[i].indices.astype(np.int64), (Lm[i].data - 1).astype(np.int64))
                    for i in range(ctx.n)]
            f, t = eval_tokens(ctx, toks, ctx.va_tr, rows=rows)
            items[str(m)] = {"full": float(np.nanmean(f[rows])), "tail": float(np.nanmean(t[rows]))}
        out["items_at_k"] = items
        out["fraction_of_item_gain_m8"] = ((out["armC"]["full@10"]["8"] - out["intercept"]["full"])
                                           / max(items["8"]["full"] - out["intercept"]["full"], 1e-9))
        # mixed m2k2: tower fold of k2 + concept fold on top (composition)
        Zk2 = torch.zeros(ctx.n, ctx.d)
        from run_battery_phaseA import fold_z
        Zk2 = torch.from_numpy(fold_z(ctx.enc, ctx.k2_tokens, list(range(ctx.n))))
        f_mix = np.full(ctx.n, np.nan); t_mix = np.full(ctx.n, np.nan)
        f_k2, _ = eval_tokens(ctx, ctx.k2_tokens, ctx.va_tr, rows=rows)
        for st in range(0, len(rows), 500):
            chunk = rows[st:st + 500]
            cids = []; cvals = []
            for r in chunk:
                cs, vs = sel[r]
                ci, vv = ConceptFoldNet.canonical_order(list(cs[:2]), list(vs[:2]))
                cids.append(ci); cvals.append(vv)
            zo = net(Zk2[chunk], cids, cvals)
            S = (zo @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            ff, tt = ndcg10_from_scores(S, ctx.va_tr[chunk], ctx.va_te[chunk], ctx.head_mask)
            f_mix[np.asarray(chunk)] = ff; t_mix[np.asarray(chunk)] = tt
        d_mix = bootstrap_ci(f_mix[rows] - f_k2[rows])
        out["mixed_m2_k2"] = {"full@10": float(np.nanmean(f_mix[rows])),
                              "tail@10": float(np.nanmean(t_mix[rows])),
                              "vs_items_k2": {"mean": d_mix[0], "ci95": d_mix[1]},
                              "armA_bar": 0.0125}
        # G5 member-specificity AUC at k=0 (popularity-matched non-members, top concept)
        aucs = []
        rngm = np.random.default_rng(SEED)
        for r in rows[:2000]:
            c = int(sel[r][0][0])
            mem = members[tags[c]]
            non = np.setdiff1d(np.arange(ctx.ni), mem)
            non_sorted = non[np.argsort(ctx.cnt[non])]
            pos = np.searchsorted(ctx.cnt[non_sorted], ctx.cnt[mem])
            match = non_sorted[np.clip(pos, 0, len(non_sorted) - 1)]
            ci, vv = ConceptFoldNet.canonical_order([c], [float(sel[r][1][0])])
            zo = net(torch.zeros(1, ctx.d), [ci], [vv])
            S = (zo @ ctx.Wd.T + ctx.bd)[0].numpy()
            sm = S[mem]; sn = S[match]; k = len(sm)
            ranks = np.argsort(np.argsort(np.concatenate([sm, sn])))[:k].sum()
            aucs.append((ranks - k * (k - 1) / 2) / (k * k))
        out["g5_member_auc_k0"] = float(np.mean(aucs))
        out["acceptance"] = {
            "monotone_full_to_m8": out["armC_monotone_full"],
            "monotone_tail_to_m8": out["armC_monotone_tail"],
            "redundancy_robust_full": out["redundancy_robust_full"],
            "redundancy_robust_tail": out["redundancy_robust_tail"],
            "mixed_ge_armA_bar": bool(out["mixed_m2_k2"]["vs_items_k2"]["mean"] >= 0.0125),
            "g5_auc_gt_0.8": bool(out["g5_member_auc_k0"] > 0.8),
            "G_collinearity": out["G_collinearity_PASS"]}
        out["ACCEPT"] = bool(all(out["acceptance"].values()))
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "concepts_only_curve_armC.json")
    json.dump(out, open(path, "w"), indent=2)
    log(f"[out] -> {path} ACCEPT={out['ACCEPT']}")
    return out


# ============================================================================= smoke
def _smoke():
    import recvae as R
    from train_tower_t2 import build_model
    print("[SMOKE] ConceptFoldNet synthetic (NOT canonical data)")
    rng = np.random.RandomState(0)
    ni, d, nc = 130, 16, 8
    cnt = rng.rand(ni) * 100
    fake = R.RecVAE(24, d, ni)
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=24, t_latent=d, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, _, _, _ = build_model(a, ni, cnt, teacher_override=fake)
    enc.eval()
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    members = {t: np.sort(rng.choice(ni, size=rng.randint(20, 40), replace=False)) for t in range(nc)}
    tags = list(range(nc))
    d_c, d_raw, w_c = build_concept_dirs(Wd, members, tags)
    net = ConceptFoldNet(nc, d=d, h=32, conc_init=d_c.numpy())
    print(f"[SMOKE] params = {net.n_params():,} (toy dims)")
    # 1) NO-OP AT INIT: output bit-exact z_items for any answers (delta zero-init)
    Z = torch.randn(4, d)
    zo = net(Z, [[0, 1], [2], [], [3, 4, 5]], [[1.0, 0.5], [0.8], [], [1.0, 0.7, 0.3]])
    assert torch.equal(zo, Z), "zero-init no-op broken"
    print("[SMOKE] no-op at init PASS (output == z_items bit-exact)")
    # 2) canonical order deterministic + value-sorted
    ci, vv = ConceptFoldNet.canonical_order([3, 1, 2], [0.5, 1.0, 0.5])
    assert ci == [1, 2, 3] and vv == [1.0, 0.5, 0.5], f"canonical order wrong: {ci} {vv}"
    print("[SMOKE] canonical order PASS (desc value, ties by id)")
    # 3) trains: 2-cluster synthetic -- concept 0 -> items 0..19, concept 1 -> items 100..119;
    #    concepts-only input must learn to rank the right cluster (fold-to-point compounding)
    real_nc = 2
    net2 = ConceptFoldNet(real_nc, d=d, h=32, conc_init=d_c.numpy()[:2])
    opt = torch.optim.AdamW(net2.parameters(), lr=5e-3)
    tgtA = np.arange(0, 20); tgtB = np.arange(100, 120)
    Z0 = torch.zeros(32, d)
    nll0 = None
    for it in range(300):
        cids = [[0] if i % 2 == 0 else [1] for i in range(32)]
        cvals = [[1.0]] * 32
        zo = net2(Z0, cids, cvals)
        logsm = F.log_softmax(zo @ Wd.T + bd, -1)
        tgt = torch.zeros(32, ni)
        for i in range(32):
            tgt[i, tgtA if i % 2 == 0 else tgtB] = 1.0
        nll = -((logsm * tgt).sum(-1) / 20).mean()
        if it == 0:
            nll0 = float(nll)
        opt.zero_grad(); nll.backward(); opt.step()
    print(f"[SMOKE] fold-to-point training PASS: NLL {nll0:.3f} -> {float(nll):.3f}")
    assert float(nll) < nll0 - 0.1, "concept fold did not learn"
    # 4) per-user norm cap actually caps
    with torch.no_grad():
        net2.log_cap.fill_(-6.0)                             # cap ~ 0.0025
        zo = net2(Z0[:4], [[0], [1], [0], [1]], [[1.0]] * 4)
        assert float((zo - Z0[:4]).norm(dim=1).max()) <= float(F.softplus(net2.log_cap)) + 1e-5
    print("[SMOKE] per-user norm cap PASS (update norm <= cap)")
    # 5) composition with a nonzero item fold: output differs from both inputs, item fold preserved at cap->0
    with torch.no_grad():
        net2.log_cap.fill_(1.6094)
        Zi = torch.randn(2, d)
        zo = net2(Zi, [[0], [1]], [[1.0], [1.0]])
        assert not torch.allclose(zo, Zi), "trained fold inert on item context"
    print("[SMOKE] composition PASS (concept fold moves a nonzero item-fold point)")
    # 6) make_example: item-masked curriculum drops the supporting members
    Mm = build_member_matrix(members, tags, ni)
    grate = (cnt @ np.asarray(Mm.todense())) / max(cnt.sum(), 1e-9)
    member_sets = {i: members[t] for i, t in enumerate(tags)}
    hit = 0
    for trial in range(200):
        k = rng.randint(12, 40)
        its = rng.choice(ni, size=k, replace=False).astype(np.int64)
        lvs = rng.randint(0, NLEV, size=k).astype(np.int64); lvs[:6] = 8
        u = {"items": its, "levels": lvs, "liked": its[lvs >= 7]}
        e = make_example(u, np.random.default_rng(trial), Mm, grate, member_sets)
        if e is None:
            continue
        in_s, in_l, cids, cvals, tgt = e
        for c in cids:
            assert not np.isin(in_s, member_sets[c]).any(), \
                "member items of a revealed concept leaked into the item input"
        assert len(np.intersect1d(in_s, tgt)) == 0, "target leaked into input"
        hit += 1
    assert hit > 20, f"make_example produced too few usable examples ({hit}/200)"
    print(f"[SMOKE] item-masked curriculum PASS ({hit}/200 usable; member-drop + leak asserts hold)")
    # 7) real-dims param count (the reportable number)
    net_real = ConceptFoldNet(1031, d=200, h=256, conc_init=None)
    print(f"[SMOKE] REAL-dims param count: {net_real.n_params():,} "
          f"(emb 1031x200 + gate(401->256->1) + mlp(401->256->200) + cap)")
    assert net_real.n_params() < 500_000, "param budget exceeded (author: few hundred k max)"
    print("[SMOKE] COMPLETE (no-op init, canonical order, fold-to-point learns, norm cap, "
          "composition, item-masked curriculum, param budget)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true", help="EVENING CLASS -- do not launch until t2final")
    ap.add_argument("--eval_curve", default=None, metavar="CKPT")
    ap.add_argument("--snapshot", default=os.path.join(CKPT_DIR, "t2i25_EP4_SNAP.pt"),
                    help="frozen tower ckpt (swap to t2final when it lands)")
    ap.add_argument("--tag", default="cfold")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_users", type=int, default=0, help="dev cap only (0=all; HARD RULE #1)")
    ap.add_argument("--resume_from_best", action="store_true",
                    help="resume from <tag>_best.pt weights (fresh optimizer; continue epochs)")
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    elif args.train:
        train(args)
    elif args.eval_curve:
        eval_curve(args)
    else:
        ap.error("one of --smoke / --train / --eval_curve required")


if __name__ == "__main__":
    main()
