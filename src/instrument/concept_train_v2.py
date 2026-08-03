r"""concept_train_v2.py -- train the concept channel on the FRESH v2 curriculum, pure NLL.

Differences from concept_distill_train.py, and only these:
  1. FRESH DRAWS. Examples come from concept_curriculum_v2.draw() on a live RNG, a new one every step.
     The old trainer materialised ONE example per user under a fixed seed, cached its latents, and looped
     that frozen set every epoch -- one draw per user for the whole run. That is the concept-side version
     of what limited the item tower, where even nine draws per user proved thin.
  2. NO DISTILLATION. lam>0 never beat lam=0 at equal budget, so the KL term and its teacher fold are
     gone. That is what makes (1) affordable: with no teacher there is nothing to precompute, and the
     base fold the student needs anyway is batched through fold_items_frozen.
  3. REFUSALS ARE FED, and the net is built with use_refusal=True so a refusal is distinguishable from a
     genuine MEH -- both carry v=0.0, so without the flag the model could not learn the distinction the
     interview regime exists to teach.

The frozen tower, the signed four-band answer, member-drop, the answerability rule and the target
hold-out are all unchanged.

  python src/instrument/concept_train_v2.py --snapshot models/i26c_ep31/t2i26c_ep31.pt \
         --tag cv2 --epochs 12 --steps_per_epoch 400 --batch 256
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_curriculum_v2 as CV2
from concept_fold import ConceptFoldNet, fold_items_frozen
import concept_distill_train as CD

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
OUT = os.path.join(_ROOT, "experiments", "instrument")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)


def _member_counts(S):
    """Per-concept member count, the weight the rare-concept sampler uses."""
    return np.array([len(S["member_sets"][i]) for i in range(len(S["tags"]))], np.float64)


def build_val_interviews(S, args, mc, n_users=1500, seed=4242):
    """PRE-REGISTERED selection set, built ONCE and cached so every epoch is scored on identical
    interviews (Fable 2026-08-03). Held-out users are a slice of the TRAIN cohort -- the canonical val
    and test cohorts stay quarantined, the same pattern the item tower's smoke used.

    Deployment realism lives HERE, not in the training mixture: interviews are drawn under the
    BROAD-FIRST ask policy, with refusals decided by the user's real history.
    """
    rng = np.random.default_rng(seed)
    val_users = S["users"][-n_users:]
    packs = {}
    for m in (0, 1, 4, 8):
        rows = []
        for u in val_users:
            CV2_saved = (CV2.P_INTERVIEW, CV2.P_DENSE, CV2.P_KC0, CV2.P_KC1)
            ex = None
            for _ in range(6):
                ex = _one_val_example(u, rng, S, mc, m)
                if ex is not None:
                    break
            if ex is not None:
                rows.append(ex)
        packs[m] = rows
        log(f"[val] m={m}: {len(rows)} interviews cached (broad-first policy)")
    return packs


def _one_val_example(u, rng, S, mc, m):
    """One val interview at exactly m asked concepts under the broad policy (m=0 -> no concepts)."""
    items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(items) < 4 or len(liked) < 2:
        return None
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(items, tg)
    pool_s, pool_l = items[keep], lv[keep]
    if len(pool_s) < 2:
        return None
    stars = (pool_l.astype(np.float32) + 1.0) / 2.0
    from signed_answers import signed_for_pool, cap_negatives, BAND_REFUSE
    V, B, ans = signed_for_pool(pool_s, stars, S["item_mean"], S["Mm"], S["grate"], S["prereg"])
    if m == 0:
        cids = np.empty(0, np.int64); ok = np.empty(0, bool)
    else:
        cids = CV2.ask_pool("broad", len(S["tags"]), mc, rng, m)
        ok = np.array([bool(ans[c]) and B[c] != BAND_REFUSE for c in cids])
    cvals = cap_negatives([float(V[c]) if ok[j] else 0.0 for j, c in enumerate(cids)])
    crefs = [0.0 if ok[j] else 1.0 for j in range(len(cids))]
    ansd = [int(c) for j, c in enumerate(cids) if ok[j]]
    drop = (np.unique(np.concatenate([np.intersect1d(pool_s, S["member_sets"][c]) for c in ansd]))
            if ansd else np.empty(0, np.int64))
    rem_s = pool_s[~np.isin(pool_s, drop)]; rem_l = pool_l[~np.isin(pool_s, drop)]
    k = min(4, len(rem_s))
    if k > 0:
        pick = rng.choice(len(rem_s), size=k, replace=False)
        in_s, in_l = rem_s[pick], rem_l[pick]
    else:
        in_s = np.empty(0, np.int64); in_l = np.empty(0, np.int64)
    tgt = np.setdiff1d(tg, in_s)
    if len(tgt) == 0:
        return None
    return (in_s, in_l, [int(c) for c in cids], [float(v) for v in cvals],
            [float(x) for x in crefs], tgt)


def eval_val(net, S, packs, head_mask, batch=256):
    """NDCG@10 full+tail on the cached interviews. Input items are masked out of the ranking.

    @10 not @100: every other number in this project is @10, and the no-crater constraint compares
    against the 0.1279 NO-ANSWER INTERCEPT which is itself @10. Mixing depths would make that gate
    unfireable."""
    Wd, bd, ni = S["Wd"], S["bd"], S["ni"]
    out = {}
    net.eval()
    with torch.no_grad():
        for m, rows in packs.items():
            f_all, t_all = [], []
            for st in range(0, len(rows), batch):
                blk = rows[st:st + batch]
                Zi = fold_items_frozen(S["enc"], [(e[0], e[1]) for e in blk], batch=batch)
                z = net(Zi, [e[2] for e in blk], [e[3] for e in blk], [e[4] for e in blk])
                sc = (z @ Wd.T + bd).numpy()
                for r, e in enumerate(blk):
                    sc[r, e[0]] = -np.inf                      # never re-rank the fold-in
                    order = np.argsort(-sc[r])[:10]
                    rel = np.isin(order, e[5]).astype(np.float64)
                    disc = 1.0 / np.log2(np.arange(2, 12))
                    idcg = (np.ones(min(len(e[5]), 10)) * disc[:min(len(e[5]), 10)]).sum()
                    f_all.append((rel * disc).sum() / max(idcg, 1e-9))
                    tm = ~head_mask[order]
                    ridx = np.isin(order, e[5]) & tm
                    nt = int((~head_mask[e[5]]).sum())
                    if nt > 0:
                        it = (np.ones(min(nt, 10)) * disc[:min(nt, 10)]).sum()
                        t_all.append((ridx * disc).sum() / max(it, 1e-9))
            out[m] = (float(np.mean(f_all)), float(np.mean(t_all)) if t_all else float("nan"))
    net.train()
    return out


def run_epoch(net, opt, S, args, ep, mc):
    Wd, bd, ni = S["Wd"], S["bd"], S["ni"]
    W0, b0 = Wd.clone(), bd.clone()
    users = S["users"]
    # Seed is a pure function of (run seed, epoch) and is LOGGED: fresh draws without a recorded seed
    # would be unreproducible.
    seed = args.seed * 100003 + ep
    rng = np.random.default_rng(seed)
    net.train()
    tot_nll, nb, t0 = 0.0, 0, time.time()
    regimes = {}
    n_ref_tok = 0
    n_tok = 0
    for step in range(args.steps_per_epoch):
        batch = []
        guard = 0
        while len(batch) < args.batch and guard < args.batch * 50:
            guard += 1
            u = users[int(rng.integers(len(users)))]
            ex = CV2.draw(u, rng, S["Mm"], S["grate"], S["member_sets"], S["item_mean"],
                          S["prereg"], member_count=mc, n_concepts=len(S["tags"]))
            if ex is not None:
                batch.append(ex)
        if len(batch) < 2:
            continue
        seqs = [(e[0], e[1]) for e in batch]
        Zi = fold_items_frozen(S["enc"], seqs, batch=args.batch)      # batched frozen folds
        cids = [e[2] for e in batch]
        cvals = [e[3] for e in batch]
        crefs = [e[4] for e in batch]
        zout = net(Zi, cids, cvals, crefs)
        stu = zout @ Wd.T + bd
        logsm = F.log_softmax(stu, dim=-1)
        tgt = torch.zeros(len(batch), ni)
        for r, e in enumerate(batch):
            tgt[r, e[5]] = 1.0
        nll = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot_nll += float(nll); nb += 1
        for e in batch:
            regimes[e[6]] = regimes.get(e[6], 0) + 1
            n_ref_tok += e[7]; n_tok += len(e[2])
        if nb % 100 == 0:
            log(f"  ep{ep} step {nb}/{args.steps_per_epoch} NLL={tot_nll/nb:.4f} "
                f"({(time.time()-t0)/60:.1f}m)")
    assert torch.equal(Wd, W0) and torch.equal(bd, b0), "frozen decoder drifted"
    mix = {k: round(v / max(sum(regimes.values()), 1), 3) for k, v in sorted(regimes.items())}
    return {"nll": tot_nll / max(nb, 1), "minutes": (time.time() - t0) / 60, "seed": seed,
            "regime_mix": mix, "refusal_token_share": round(n_ref_tok / max(n_tok, 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--tag", default="cv2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--m_max", type=int, default=16)
    ap.add_argument("--mix", default=None, help="p_broad,p_dense,p_interview,p_kc0,p_kc1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.mix:
        CV2.set_mixture(*[float(x) for x in a.mix.split(",")])   # p_interview,p_dense,p_kc0,p_kc1
    log(f"[cv3] mixture interview={CV2.P_INTERVIEW} dense={CV2.P_DENSE} kc0={CV2.P_KC0} "
        f"kc1={CV2.P_KC1} | ask policies={CV2.ASK_POLICIES} dense_rare={CV2.P_DENSE_RARE} "
        f"| rare pow={CV2.RARE_POW} clip={CV2.RARE_CLIP}")

    # reuse the existing stack builder verbatim (tower, tags, member sets, SEL valuation, users)
    a.lam = 0.0; a.arm = "s1"; a.tau = 1.0; a.eps_l2 = 0.0
    S = CD._build_train_stack(a)
    mc = _member_counts(S)
    log(f"[cv2] member counts: min {mc.min():.0f} median {np.median(mc):.0f} max {mc.max():.0f}")

    from train_tower_t2 import compute_head_mask
    import metrics as _M
    _tr = _M.load_train(S["ni"], os.path.join(_ROOT, "data", "ml-25m", "proc"))
    head_mask, _ = compute_head_mask(_tr, S["ni"])
    N_VAL = 1500
    packs = build_val_interviews(S, a, mc, n_users=N_VAL) if not a.smoke else None
    S["users"] = S["users"][:-N_VAL]          # val slice NEVER trained on
    log(f"[cv3] train users {len(S['users'])} | val users {N_VAL} (held out of training)")

    net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=a.hidden,
                         conc_init=S["d_c"].numpy(), use_refusal=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    log(f"[cv2] net params {sum(p.numel() for p in net.parameters()):,} (use_refusal=True)")

    if a.smoke:
        a.epochs, a.steps_per_epoch = 1, 3
    hist = []
    for ep in range(1, a.epochs + 1):
        r = run_epoch(net, opt, S, a, ep, mc)
        hist.append(r)
        log(f"[cv2] ep{ep} NLL={r['nll']:.4f} {r['minutes']:.1f}m seed={r['seed']} "
            f"mix={r['regime_mix']} refusal_tokens={r['refusal_token_share']:.3f}")
        if packs is not None:
            v = eval_val(net, S, packs, head_mask)
            r["val"] = {str(k): v[k] for k in v}
            sel = float(np.mean([v[m][0] for m in (1, 4, 8)]))
            r["sel"] = sel
            crater = min(v[m][0] for m in (1, 4, 8)) < 0.1279      # no-crater constraint
            log(f"[cv3] ep{ep} VAL m0={v[0][0]:.4f}/{v[0][1]:.4f} m1={v[1][0]:.4f} "
                f"m4={v[4][0]:.4f} m8={v[8][0]:.4f} | SEL={sel:.4f}"
                + ("  !! CRATER (below 0.1279 intercept)" if crater else ""))
        torch.save({"net": net.state_dict(), "epoch": ep, "nll": r["nll"], "hist": hist,
                    "use_refusal": True, "snapshot": os.path.basename(a.snapshot)},
                   os.path.join(CKPT_DIR, f"{a.tag}_ep{ep:02d}.pt"))
    os.makedirs(OUT, exist_ok=True)
    json.dump(hist, open(os.path.join(OUT, f"{a.tag}_hist.json"), "w"), indent=2)
    log(f"[cv2] done -> {a.tag}")


if __name__ == "__main__":
    main()
