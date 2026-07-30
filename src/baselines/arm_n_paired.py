r"""arm_n_paired.py -- paired comparison of the instrument against a named rival on arm N.

WHY. The arm-N table's headline question is whether the gap to RecVAE is a tie, as it is on arm A
(-0.0057, CI 0.0069). Point estimates plus per-arm standard errors cannot answer that: the two models
are scored on the SAME 10,000 users, so the comparison is paired and an unpaired SE-of-difference is
the wrong test. This computes the per-user difference and bootstraps its CI.

Reports full@10 and tail@10 separately. The tail vector covers a smaller user set (users whose
held-out items are all head are dropped), so the two are bootstrapped independently.

  python src/baselines/arm_n_paired.py [--rival recvae] [--boot 10000]
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
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import metrics as M
from arm_n import load_arm_n
from arm_n_tower import graded_to_levels, SNAP_DEFAULT, OUT
from run_arm_n import build
from train_tower_t2 import build_model, make_graded_predict_fn, compute_head_mask


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(OUT, "paired.log"), "a") as f:
        f.write(line + "\n")


def boot_ci(d, n_boot, seed=0):
    """Percentile bootstrap over the per-user differences."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rival", default="recvae")
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--arm", default="N", choices=["A", "N"],
                    help="A re-tests the CHAPTER's tie claim with the correct paired test; the "
                         "recorded CI 0.0069 is 1.96*sqrt(2)*SE, i.e. an UNPAIRED interval.")
    a = ap.parse_args()

    D = load_arm_n(log=logln)
    ni = D["n_items"]
    _hm, cnt = compute_head_mask(D["train"], ni)

    ma = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                            train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                            warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, _t, _p, _g = build_model(ma, ni, cnt)
    blob = torch.load(a.snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"]); enc.eval()
    if a.arm == "N":
        L_ours, pool = graded_to_levels(D["g_te_tr"]), D["pool"]
    else:
        # ARM A: the canonical likes-only fold-in with real star levels, canonical mask. Reproduces
        # the chapter's 0.3482 and lets the recorded tie be re-tested with the correct statistic.
        from train_tower_t2 import build_graded_eval_matrix, reproduce_partition
        unique_uid, _a, _b, _c, _d, raw, show2id, usid = reproduce_partition()
        L_ours, _n = build_graded_eval_matrix(raw, unique_uid, show2id, usid, "test")
        pool = D["te_tr"]
    logln(f"[paired] arm {a.arm}: fold-in nnz={L_ours.nnz} pool nnz={pool.nnz}")

    pr_ours = make_graded_predict_fn(enc, decoder.weight.detach(), decoder.bias.detach(),
                                     L_ours, check_nnz=False)
    r_ours = M.evaluate(pr_ours, D["te_tr"], D["te_te"], batch_size=500, head_mask=D["head_mask"],
                        mask_X=pool, per_user=True)
    logln(f"[paired] ours  full={r_ours['ndcg@10']:.4f} tail={r_ours['tail_ndcg@10']:.4f}")

    pr_rival, _hp = build(a.rival, D, 1e9, logln)
    r_rival = M.evaluate(pr_rival, D["te_tr"], D["te_te"], batch_size=500, head_mask=D["head_mask"],
                         mask_X=pool, per_user=True)
    logln(f"[paired] {a.rival} full={r_rival['ndcg@10']:.4f} tail={r_rival['tail_ndcg@10']:.4f}")

    out = {"rival": a.rival, "n_boot": a.boot}
    for key in ("ndcg@10", "tail_ndcg@10"):
        vo, vr = r_ours[key + "_per_user"], r_rival[key + "_per_user"]
        if len(vo) != len(vr):
            logln(f"[paired] {key}: LENGTH MISMATCH {len(vo)} vs {len(vr)} -- not pairable, skipped")
            continue
        d = vo - vr
        lo, hi = boot_ci(d, a.boot)
        tie = lo <= 0.0 <= hi
        out[key] = {"ours": float(vo.mean()), "rival": float(vr.mean()), "diff": float(d.mean()),
                    "ci95": [lo, hi], "n_users": int(len(d)), "tie": bool(tie)}
        logln(f"[paired] {key}: ours-{a.rival} = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"n={len(d)}  -> {'TIE (CI spans 0)' if tie else 'SEPARATED'}")

    with open(os.path.join(OUT, f"paired_{a.rival}_arm{a.arm}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
