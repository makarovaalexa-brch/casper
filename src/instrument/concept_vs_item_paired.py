r"""concept_vs_item_paired.py -- paired per-user CIs on the concept-minus-item delta, per bank, per budget.

The bank sweep gives point estimates. The load-bearing claim -- that under a discrimination-ranked bank
(`polar`) the concept channel beats the item channel on BOTH axes at every budget from m=2 -- rests on
deltas of +.0120 to +.0297 tail, which need intervals before they reach the chapter.

PAIRED is correct and not a preference: both arms score the IDENTICAL users with the IDENTICAL held-out
targets (the cold-check RNG streams are split so the target draw does not depend on which arm is built),
and at m=0 they are bit-identical. Per-user differences are therefore strongly correlated.

  python src/instrument/concept_vs_item_paired.py --ckpt .cache/instrument/cv5_ep24.pt --boot 10000
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_distill_train as CD
from concept_fold import ConceptFoldNet, fold_items_frozen
from concept_cold_check import (build_banks, cold_interviews, cold_item_interviews, ndcg_at,
                                apply_pool_mask, MS, log)

BANKS = ("pop", "entropy", "polar", "expinfo")


def per_user(net, S, packs, head_mask, batch=256, tag="cold_check_paired"):
    """Like concept_cold_check.score but keeps the PER-USER vectors, which is what pairing needs.

    The candidate mask is `concept_cold_check.apply_pool_mask` -- the SAME function the point-estimate
    sweep uses -- so the CIs and the table they qualify can never drift onto different pools. It was a
    copy of the old `sc[r, e[6]] = -inf` line; migrated with `score` on 2026-08-05 (policy B).
    """
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
                    apply_pool_mask(sc[r], e, st + r, tag)
                    f, t = ndcg_at(sc[r], e[5], head_mask)
                    fs.append(f)
                    ts.append(np.nan if t is None else t)
            out[m] = (np.asarray(fs, float), np.asarray(ts, float))
    return out


def boot_ci(d, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, ".cache", "instrument", "cv5_ep24.pt"))
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_ROOT, "experiments", "instrument",
                                                  "concept_vs_item_paired.json"),
                    help="output JSON; use a NEW filename when the pool policy changes")
    a = ap.parse_args()
    a.lam = 0.0; a.arm = "s1"; a.tau = 1.0; a.eps_l2 = 0.0; a.m_max = 16

    S = CD._build_train_stack(a)
    mc = np.array([len(S["member_sets"][i]) for i in range(len(S["tags"]))], np.float64)
    from train_tower_t2 import compute_head_mask
    import metrics as M
    tr = M.load_train(S["ni"], os.path.join(_ROOT, "data", "ml-25m", "proc"))
    head_mask, _ = compute_head_mask(tr, S["ni"])
    banks = build_banks(S)
    banks["item_pop"] = np.argsort(-np.asarray(tr.sum(axis=0)).ravel())
    banks["conc_pop"] = np.argsort(-mc)

    blob = torch.load(a.ckpt, map_location="cpu")
    keep_meh = bool(blob.get("keep_meh", False))
    net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=blob.get("hidden", a.hidden),
                         conc_init=S["d_c"].numpy(), use_refusal=blob.get("use_refusal", False))
    net.load_state_dict(blob["net"])
    log(f"net={os.path.basename(a.ckpt)} ep{blob['epoch']} hidden={blob.get('hidden')} "
        f"keep_meh={keep_meh}")

    out = {}
    for b in BANKS:
        ci = per_user(None, S, cold_item_interviews(S, banks[f"item_{b}"], n_users=a.n_users), head_mask)
        cc = per_user(net, S, cold_interviews(S, mc, n_users=a.n_users, rank=banks[f"conc_{b}"],
                                              keep_meh=keep_meh), head_mask)
        assert abs(ci[0][0].mean() - cc[0][0].mean()) < 1e-9, "m=0 arms-matched canary FAILED"
        out[b] = {}
        log(f"--- bank={b}")
        for m in MS[1:]:
            for ax, i in (("full", 0), ("tail", 1)):
                g, h = cc[m][i], ci[m][i]
                keep = ~(np.isnan(g) | np.isnan(h))
                d = g[keep] - h[keep]
                lo, hi = boot_ci(d, a.boot)
                sep = "SEP" if (lo > 0 or hi < 0) else "ns "
                out[b][f"m{m}_{ax}"] = {"n": int(keep.sum()), "delta": float(d.mean()),
                                        "ci": [lo, hi], "separated": bool(lo > 0 or hi < 0)}
                log(f"    m{m:2d} {ax:4s} conc-minus-item {d.mean():+.4f} "
                    f"CI [{lo:+.4f},{hi:+.4f}] {sep} (n={int(keep.sum())})")
    out["_pool_policy"] = "B (see concept_cold_check.apply_pool_mask; 2026-08-05 ruling)"
    json.dump(out, open(a.out, "w"), indent=2)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
