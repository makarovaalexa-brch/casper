r"""a22_concept_curve_paired.py -- re-measure the concept m-curve with each MONITOR DEFECT switchable.

Design sheet: docs/design/DS_A22_CONCEPT_MONITOR_AND_ASKSET.md

The curve that declared the Goodreads concept channel a failure came from the in-training monitor
(concept_train_v2.build_val_interviews). That monitor has three defects, none of which was a problem
while it was only picking a checkpoint on one dataset, and all of which bite when it is read as a
cross-dataset result:

  1. MEMBER-DROP AT EVAL (concept_train_v2.py:105-107). Items supporting an answered concept are
     removed from the pool BEFORE the 4 context items are drawn. That is a TRAINING device -- it forces
     the concept answer to carry independent signal -- and at evaluation it strips the context of
     exactly the taste the targets lie on, with a penalty that grows in m.
  2. UNPAIRED LEVELS (:87). Targets are redrawn inside each call, so m0-vs-m8 carries target-draw noise.
  3. ACTIVITY-BIASED SLICE (:63). `users[-1500:]` is a positional tail of a userId-ordered list.

Defect 3 is NOT repaired here and cannot be: that slice is held out of training (:248), so re-drawing it
for an existing checkpoint evaluates users the net trained on. It is reported, not fixed; fixing it
belongs to the next run's holdout rule.

This script does not modify the training path. For the as-shipped configuration it CALLS the shipped
builder, so the reproduction gate G1 is exact rather than re-implemented.

  python src/instrument/a22_concept_curve_paired.py --snapshot .cache/instrument/t2i26g_ep53.pt \
         --ckpt .cache/instrument/cvg_ep12.pt --tag gr --seeds 5
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_curriculum_v2 as CV2
import concept_train_v2 as CT
from concept_fold import ConceptFoldNet, fold_items_frozen

OUT = os.path.join(_ROOT, "experiments", "instrument")
N_VAL = 1500                      # must match concept_train_v2.N_VAL -- the training holdout


def log(m):
    print("[" + time.strftime("%H:%M:%S") + "] " + m, flush=True)


# ------------------------------------------------------------------ paired interview construction
def ask_set(policy, S, mc, rng, m):
    """Concept ids to ask. 'broad' = the shipped member-weighted sampler. 'topN:<N>' = uniform over
    the N broadest concepts, which is the member-count hypothesis under test."""
    C = len(S["tags"])
    if m == 0:
        return np.empty(0, np.int64)
    if policy == "broad":
        return CV2.ask_pool("broad", C, mc, rng, m)
    if policy.startswith("topN:"):
        N = min(int(policy.split(":")[1]), C)
        pool = np.argsort(-mc)[:N]
        return rng.choice(pool, size=min(m, N), replace=False)
    if policy.startswith("band:"):
        # MIDDLE BAND by member-count rank: concepts large enough to be answerable, small enough to
        # still discriminate. The top of the bank is answerable but contentless (topN) and the tail is
        # informative but unanswerable -- this asks whether anything survives in between.
        lo, hi = (int(x) for x in policy.split(":")[1].split("-"))
        pool = np.argsort(-mc)[lo:min(hi, C)]
        return rng.choice(pool, size=min(m, len(pool)), replace=False)
    raise ValueError(policy)


def paired_user(u, rng, S, mc, ms, drop, policy, scramble, no_refusals=False):
    """ONE target draw per user, reused at every m. Returns {m: pack row} or None.

    Mirrors concept_train_v2._one_val_example except that the target draw and the residual pool are
    hoisted out of the m loop (that is the PAIRING fix) and member-drop is optional.
    """
    from signed_answers import signed_for_pool, cap_negatives, BAND_REFUSE
    items = np.asarray(u["items"], np.int64)
    lv = np.asarray(u["levels"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(items) < 4 or len(liked) < 2:
        return None
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)             # DRAWN ONCE
    keep = ~np.isin(items, tg)
    pool_s, pool_l = items[keep], lv[keep]
    if len(pool_s) < 2:
        return None
    stars = (pool_l.astype(np.float32) + 1.0) / 2.0
    V, B, ans = signed_for_pool(pool_s, stars, S["item_mean"], S["Mm"], S["grate"], S["prereg"])

    # the context items are also drawn ONCE when member-drop is off, so every m sees the same context
    # and the ONLY thing that varies across m is how many concepts were asked.
    base_pick = rng.permutation(len(pool_s))
    out = {}
    for m in ms:
        cids = ask_set(policy, S, mc, rng, m)
        if m:
            ok = np.array([bool(ans[c]) and B[c] != BAND_REFUSE for c in cids], bool)
        else:
            ok = np.empty(0, bool)
        cvals = cap_negatives([float(V[c]) if ok[j] else 0.0 for j, c in enumerate(cids)])
        crefs = [0.0 if ok[j] else 1.0 for j in range(len(cids))]
        if drop:
            ansd = [int(c) for j, c in enumerate(cids) if ok[j]]
            if ansd:
                dr = np.unique(np.concatenate(
                    [np.intersect1d(pool_s, S["member_sets"][c]) for c in ansd]))
            else:
                dr = np.empty(0, np.int64)
            m_keep = ~np.isin(pool_s, dr)
            rem_s, rem_l = pool_s[m_keep], pool_l[m_keep]
            k = min(4, len(rem_s))
            if k > 0:
                pick = rng.choice(len(rem_s), size=k, replace=False)
                in_s, in_l = rem_s[pick], rem_l[pick]
            else:
                in_s = np.empty(0, np.int64)
                in_l = np.empty(0, np.int64)
        else:
            k = min(4, len(pool_s))
            in_s, in_l = pool_s[base_pick[:k]], pool_l[base_pick[:k]]
        tgt = np.setdiff1d(tg, in_s)
        if len(tgt) == 0:
            return None
        cid_out = [int(c) for c in cids]
        if no_refusals and m:
            # Feed ONLY the concepts the user actually answered. On Goodreads an 8-question interview
            # delivers ~2.2 answers and ~5.8 REFUSAL tokens (ML-25M: ~6.0 and ~2.0), and refusals are
            # fed to the net (use_refusal=True). If the anti-accumulation is the refusal tokens rather
            # than the answers, the decay must disappear here.
            k_ok = [j for j in range(len(cid_out)) if crefs[j] == 0.0]
            cid_out = [cid_out[j] for j in k_ok]
            cvals = [cvals[j] for j in k_ok]
            crefs = [crefs[j] for j in k_ok]
        if scramble and m:
            # EXISTENTIAL CONTROL (G3): keep the values and refusal flags, replace WHICH concept they
            # are attached to. A fold that is concept-identity-sensitive must lose its lift here.
            cid_out = [int(c) for c in rng.choice(len(S["tags"]), size=len(cid_out), replace=False)]
        out[m] = (in_s, in_l, cid_out, [float(v) for v in cvals], [float(x) for x in crefs], tgt)
    return out


def build_paired(S, mc, ms, n_users, seed, drop, policy, scramble, no_refusals=False):
    """Paired packs over the SAME held-out users, one independent rng per user."""
    val_users = S["users"][-n_users:]
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2 ** 31 - 1, size=len(val_users))
    packs = {m: [] for m in ms}
    kept = 0
    for i, u in enumerate(val_users):
        r = paired_user(u, np.random.default_rng(int(seeds[i])), S, mc, ms, drop, policy, scramble,
                        no_refusals)
        if r is None:
            continue
        kept += 1
        for m in ms:
            packs[m].append(r[m])
    return packs, kept


# ------------------------------------------------------------------------------------ evaluation
def ndcg_rows(net, S, rows, head_mask, strip, batch=256):
    """Per-user NDCG@10 full and tail. `strip` zeroes the concept channel while holding the CONTEXT
    fixed -- the difference is the paired FOLD EFFECT."""
    Wd, bd = S["Wd"], S["bd"]
    disc = 1.0 / np.log2(np.arange(2, 12))
    f_all, t_all = [], []
    net.eval()
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            blk = rows[st:st + batch]
            Zi = fold_items_frozen(S["enc"], [(e[0], e[1]) for e in blk], batch=batch)
            if strip:
                z = net(Zi, [[] for _ in blk], [[] for _ in blk], [[] for _ in blk])
            else:
                z = net(Zi, [e[2] for e in blk], [e[3] for e in blk], [e[4] for e in blk])
            sc = (z @ Wd.T + bd).numpy()
            for r, e in enumerate(blk):
                sc[r, e[0]] = -np.inf
                order = np.argsort(-sc[r])[:10]
                rel = np.isin(order, e[5]).astype(np.float64)
                idcg = (disc[:min(len(e[5]), 10)]).sum()
                f_all.append((rel * disc).sum() / max(idcg, 1e-9))
                nt = int((~head_mask[e[5]]).sum())
                if nt > 0:
                    ridx = np.isin(order, e[5]) & ~head_mask[order]
                    t_all.append((ridx * disc).sum() / max((disc[:min(nt, 10)]).sum(), 1e-9))
                else:
                    t_all.append(np.nan)
    return np.asarray(f_all), np.asarray(t_all)


def boot_ci(d, n=10000, seed=98765):
    """Percentile bootstrap over users -- the project's standard, seed 98765."""
    if len(d) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    b = np.nanmean(d[idx], axis=1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ask", default="broad", help="broad | topN:<N>")
    ap.add_argument("--skip_g1", action="store_true")
    ap.add_argument("--no_refusals", action="store_true",
                    help="feed ONLY answered concepts; isolates the refusal tokens' contribution")
    a = ap.parse_args()
    ms = (0, 1, 4, 8)

    import metrics as _M
    from train_tower_t2 import compute_head_mask, PROC as _PROC
    import concept_distill_train as CD

    st_args = argparse.Namespace(snapshot=a.snapshot, max_users=None)
    S = CD._build_train_stack(st_args)
    mc = CT._member_counts(S)
    head_mask, _ = compute_head_mask(_M.load_train(S["ni"], _PROC), S["ni"])

    blob = torch.load(a.ckpt, map_location="cpu")
    net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=int(blob.get("hidden", 256)),
                         conc_init=S["d_c"].numpy(), use_refusal=bool(blob.get("use_refusal", True)))
    net.load_state_dict(blob["net"])
    log("[a22] tower=" + os.path.basename(a.snapshot) + " net=" + os.path.basename(a.ckpt) +
        " ep=" + str(blob.get("epoch")) + " concepts=" + str(len(S["tags"])) +
        " users=" + str(len(S["users"])))

    # ---- G2 LEAK: the eval users must be exactly the slice training excluded.
    assert len(S["users"]) > N_VAL, "not enough users to hold out"
    assert len({id(u) for u in S["users"][-N_VAL:]}) == N_VAL
    log("[G2] LEAK CHECK PASS: eval on users[-" + str(N_VAL) + ":], the slice "
        "concept_train_v2:248 removes from training")

    res = {"tag": a.tag, "ckpt": os.path.basename(a.ckpt), "snapshot": os.path.basename(a.snapshot),
           "epoch": blob.get("epoch"), "ask": a.ask, "n_concepts": len(S["tags"]), "cells": {}}

    # ---- G1 REPRODUCTION: call the SHIPPED builder, seed 4242, and match the training log.
    if not a.skip_g1:
        t0 = time.time()
        packs = CT.build_val_interviews(S, argparse.Namespace(), mc, n_users=N_VAL, seed=4242)
        v = CT.eval_val(net, S, packs, head_mask)
        res["G1_shipped_seed4242"] = {str(m): [float(v[m][0]), float(v[m][1])] for m in ms}
        log("[G1] shipped monitor, seed 4242: " +
            " ".join("m" + str(m) + "=" + format(v[m][0], ".4f") for m in ms) +
            "   (" + format(time.time() - t0, ".0f") + "s)")

    # ---- the factorial: (drop on/off) x (fold on/stripped), paired, several seeds
    for drop in (True, False):
        for scr in ((False, True) if not drop else (False,)):
            key = "drop=" + str(int(drop)) + ",scramble=" + str(int(scr))
            per_seed = []
            for si in range(a.seeds):
                seed = 4242 + 1000 * si
                packs, kept = build_paired(S, mc, ms, N_VAL, seed, drop, a.ask, scr, a.no_refusals)
                cell = {}
                for m in ms:
                    f_on, t_on = ndcg_rows(net, S, packs[m], head_mask, strip=False)
                    f_off, t_off = ndcg_rows(net, S, packs[m], head_mask, strip=True)
                    d = f_on - f_off
                    cell[str(m)] = {"full": float(f_on.mean()), "tail": float(np.nanmean(t_on)),
                                    "full_stripped": float(f_off.mean()),
                                    "fold_effect": float(d.mean()), "fold_ci": boot_ci(d),
                                    "n": int(len(f_on))}
                per_seed.append(cell)
                log("[" + key + "] seed " + str(seed) + " n=" + str(kept) + " " + " ".join(
                    "m" + str(m) + "=" + format(cell[str(m)]["full"], ".4f") +
                    "(fold" + format(cell[str(m)]["fold_effect"], "+.4f") + ")" for m in ms))
            agg = {}
            for m in ms:
                fs = np.array([c[str(m)]["full"] for c in per_seed])
                fe = np.array([c[str(m)]["fold_effect"] for c in per_seed])
                agg[str(m)] = {"full_mean": float(fs.mean()),
                               "full_sd": float(fs.std(ddof=1)) if len(fs) > 1 else 0.0,
                               "fold_effect_mean": float(fe.mean()),
                               "fold_effect_sd": float(fe.std(ddof=1)) if len(fe) > 1 else 0.0,
                               "fold_ci_seed0": per_seed[0][str(m)]["fold_ci"]}
            res["cells"][key] = {"per_seed": per_seed, "agg": agg}
            log("[" + key + "] AGG " + " ".join(
                "m" + str(m) + "=" + format(agg[str(m)]["full_mean"], ".4f") +
                "+-" + format(agg[str(m)]["full_sd"], ".4f") for m in ms))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "a22_" + a.tag + "_" + a.ask.replace(":", "") +
                     ("_norefuse" if a.no_refusals else "") + ".json")
    json.dump(res, open(p, "w"), indent=2)
    log("[a22] -> " + p)


if __name__ == "__main__":
    main()
