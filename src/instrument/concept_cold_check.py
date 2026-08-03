r"""concept_cold_check.py -- score concept checkpoints with concepts folded CLEAN (zero items).

WHY THIS EXISTS. The in-training val metric handed every user FOUR ITEMS of context and then folded
concepts on top. That is the situation in which concepts matter LEAST, and it introduces a confound: with
items present, member-drop removes the answered concept's member films from the input, so the channel has
to repay a debt that only exists because items were supplied. It is what made m=1 look like a crater.

Concept elicitation is a COLD-START mechanism -- you ask about concepts precisely because you have
nothing else. Folded clean:
  * m=0 is the true no-evidence intercept, so the no-crater constraint is finally within-protocol
  * m=1/4/8 measure what concepts ALONE buy
  * member-drop is vacuous (no items to drop), so the confound disappears

Training needs no redo: the curriculum samples item context k ~ U{0..8}, so pure-concept examples are
already ~11% of it. Only the measurement was wrong, and every epoch is checkpointed.

  python src/instrument/concept_cold_check.py --tag cv3 --snapshot models/i26c_ep31/t2i26c_ep31.pt
"""
import os
import sys
import json
import glob
import time
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_curriculum_v2 as CV2
from concept_fold import ConceptFoldNet, fold_items_frozen
import concept_distill_train as CD
from signed_answers import signed_for_pool, cap_negatives, BAND_REFUSE

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
MS = (0, 1, 2, 4, 8, 16)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cold_interviews(S, mc, n_users=1500, seed=4242, policy="broad"):
    """One cold interview per user per budget: NO items, m concepts asked under `policy`."""
    rng = np.random.default_rng(seed)
    users = S["users"][-n_users:]
    packs = {m: [] for m in MS}
    for u in users:
        items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
        liked = np.asarray(u["liked"], np.int64)
        if len(items) < 4 or len(liked) < 2:
            continue
        ntg = max(1, len(liked) // 3)
        tg = rng.choice(liked, size=ntg, replace=False)
        keep = ~np.isin(items, tg)
        pool_s, pool_l = items[keep], lv[keep]
        if len(pool_s) < 2:
            continue
        stars = (pool_l.astype(np.float32) + 1.0) / 2.0
        V, B, ans = signed_for_pool(pool_s, stars, S["item_mean"], S["Mm"], S["grate"], S["prereg"])
        for m in MS:
            if m == 0:
                cids = np.empty(0, np.int64); ok = np.empty(0, bool)
            else:
                cids = CV2.ask_pool(policy, len(S["tags"]), mc, rng, m)
                ok = np.array([bool(ans[c]) and B[c] != BAND_REFUSE for c in cids])
            cvals = cap_negatives([float(V[c]) if ok[j] else 0.0 for j, c in enumerate(cids)])
            crefs = [0.0 if ok[j] else 1.0 for j in range(len(cids))]
            packs[m].append((np.empty(0, np.int64), np.empty(0, np.int64),      # ZERO ITEMS
                             [int(c) for c in cids], [float(v) for v in cvals],
                             [float(x) for x in crefs], tg))
    return packs


def ndcg_at(sc_row, targets, head_mask, k=10):
    order = np.argsort(-sc_row)[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rel = np.isin(order, targets).astype(np.float64)
    idcg = (np.ones(min(len(targets), k)) * disc[:min(len(targets), k)]).sum()
    full = (rel * disc).sum() / max(idcg, 1e-9)
    nt = int((~head_mask[targets]).sum())
    if nt == 0:
        return full, None
    rt = (np.isin(order, targets) & ~head_mask[order]).astype(np.float64)
    it = (np.ones(min(nt, k)) * disc[:min(nt, k)]).sum()
    return full, (rt * disc).sum() / max(it, 1e-9)


def score(net, S, packs, head_mask, batch=256):
    Wd, bd = S["Wd"], S["bd"]
    out = {}
    if net is not None:
        net.eval()
    with torch.no_grad():
        for m, rows in packs.items():
            fs, ts = [], []
            for st in range(0, len(rows), batch):
                blk = rows[st:st + batch]
                Zi = fold_items_frozen(S["enc"], [(e[0], e[1]) for e in blk], batch=batch)
                z = Zi if net is None else net(Zi, [e[2] for e in blk], [e[3] for e in blk],
                                               [e[4] for e in blk])
                sc = (z @ Wd.T + bd).numpy()
                for r, e in enumerate(blk):
                    f, t = ndcg_at(sc[r], e[5], head_mask)
                    fs.append(f)
                    if t is not None:
                        ts.append(t)
            out[m] = (float(np.mean(fs)), float(np.mean(ts)) if ts else float("nan"))
    if net is not None:
        net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cv3")
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--policy", default="broad", choices=("broad", "uniform", "niche"))
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_users", type=int, default=0)
    a = ap.parse_args()
    a.lam = 0.0; a.arm = "s1"; a.tau = 1.0; a.eps_l2 = 0.0; a.m_max = 16

    S = CD._build_train_stack(a)
    mc = np.array([len(S["member_sets"][i]) for i in range(len(S["tags"]))], np.float64)
    from train_tower_t2 import compute_head_mask
    import metrics as M
    tr = M.load_train(S["ni"], os.path.join(_ROOT, "data", "ml-25m", "proc"))
    head_mask, _ = compute_head_mask(tr, S["ni"])

    log(f"building COLD interviews (zero items, policy={a.policy}) ...")
    packs = cold_interviews(S, mc, n_users=a.n_users, policy=a.policy)
    log(f"  {len(packs[0])} users x budgets {MS}")

    # CANARY: with no net at all, every budget must give the SAME number -- the no-evidence prior.
    base = score(None, S, packs, head_mask)
    spread = max(v[0] for v in base.values()) - min(v[0] for v in base.values())
    log(f"CANARY no-net: m0={base[0][0]:.4f} spread across budgets={spread:.2e} "
        f"{'PASS' if spread < 1e-9 else 'FAIL -- concepts leaking without a net'}")

    rows = []
    for p in sorted(glob.glob(os.path.join(CKPT_DIR, f"{a.tag}_ep*.pt"))):
        blob = torch.load(p, map_location="cpu")
        net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=a.hidden,
                             conc_init=S["d_c"].numpy(), use_refusal=blob.get("use_refusal", False))
        net.load_state_dict(blob["net"])
        r = score(net, S, packs, head_mask)
        sel = float(np.mean([r[m][0] for m in (1, 4, 8)]))
        ok = all(r[m][0] >= r[0][0] for m in (1, 2, 4, 8, 16))       # WITHIN-protocol no-crater
        rows.append({"epoch": blob["epoch"], "sel": sel, "no_crater": ok,
                     **{f"m{m}": r[m] for m in MS}})
        log(f"ep{blob['epoch']:02d} " + "  ".join(f"m{m}={r[m][0]:.4f}/{r[m][1]:.4f}" for m in MS)
            + f" | SEL={sel:.4f} {'' if ok else 'CRATER'}")
    if rows:
        best = max([r for r in rows if r["no_crater"]] or rows, key=lambda r: r["sel"])
        log(f"SELECTED ep{best['epoch']} SEL={best['sel']:.4f} no_crater={best['no_crater']}")
        json.dump(rows, open(os.path.join(_ROOT, "experiments", "instrument",
                                          f"{a.tag}_coldcheck.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
