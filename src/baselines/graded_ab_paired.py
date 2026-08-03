r"""graded_ab_paired.py -- paired per-user CI on the graded-vs-binarised premium at full profile.

The point estimate (+0.0039 full, +0.0020 tail on the certified tower) FLIPS the direction of the claim
the chapter used to make, so it needs an interval and not a point. Both arms score the IDENTICAL users
with the IDENTICAL nonzero support -- only the star levels differ -- so the per-user differences are
strongly correlated and the test must be PAIRED. An unpaired interval here would be ~4.5x too wide and
would hide a real effect (the 2026-07-30 scar, in the opposite direction).

  python src/baselines/graded_ab_paired.py [--boot 10000]
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
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import metrics as M
from arm_n import load_arm_n
from arm_n_paired import boot_ci
from arm_n_tower import logln, LIKE_LEVEL
from train_tower_t2 import (build_graded_eval_matrix, make_graded_predict_fn, reproduce_partition,
                            compute_head_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    D = load_arm_n(log=logln)
    ni, head_mask = D["n_items"], D["head_mask"]
    _hm, cnt = compute_head_mask(D["train"], ni)

    import argparse as _ap
    ma = _ap.Namespace(arch="i26", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                       train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                       warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    from i26_encoder import build_i26
    from train_tower_t2 import load_recvae_teacher, apply_sign_prior
    src = load_recvae_teacher(ni, hidden=600, latent=200)
    enc, decoder, _p, _g = build_i26(ni, src, ma, log=logln)
    apply_sign_prior(enc)
    blob = torch.load(a.snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"], strict=False)
    decoder.load_state_dict(blob["decoder"])
    enc.eval()
    Wd, bd = decoder.weight.detach(), decoder.bias.detach()

    unique_uid, _tr, _vd, _te, _n, raw, show2id, usid = reproduce_partition()
    L_g, _ = build_graded_eval_matrix(raw, unique_uid, show2id, usid, "test")
    L_b = L_g.copy()
    L_b.data = np.full_like(L_b.data, LIKE_LEVEL)
    assert L_b.nnz == L_g.nnz, "support must be identical"

    out = {}
    per = {}
    for tag, L in (("graded", L_g), ("binarised", L_b)):
        pr = make_graded_predict_fn(enc, Wd, bd, L, check_nnz=False)
        r = M.evaluate(pr, D["te_tr"], D["te_te"], batch_size=500, head_mask=head_mask,
                       mask_X=D["te_tr"], per_user=True)
        per[tag] = r
        logln(f"[{tag:9s}] full@10={np.nanmean(r['ndcg@10_per_user']):.4f} "
              f"tail@10={np.nanmean(r['tail_ndcg@10_per_user']):.4f}")

    for metric in ("ndcg@10", "tail_ndcg@10"):
        g = np.asarray(per["graded"][metric + "_per_user"], float)
        b = np.asarray(per["binarised"][metric + "_per_user"], float)
        keep = ~(np.isnan(g) | np.isnan(b))          # tail is NaN for users with no tail target
        d = g[keep] - b[keep]
        lo, hi = boot_ci(d, a.boot)
        sep = "SEPARATED" if (lo > 0 or hi < 0) else "not separated"
        out[metric] = {"n": int(keep.sum()), "mean_diff": float(d.mean()),
                       "ci": [float(lo), float(hi)], "separated": bool(lo > 0 or hi < 0)}
        logln(f"[paired] {metric:14s} n={int(keep.sum())} graded-minus-binarised "
              f"{d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] -> {sep}")

    os.makedirs(os.path.join(_ROOT, "experiments", "baselines", "arm_n"), exist_ok=True)
    json.dump(out, open(os.path.join(_ROOT, "experiments", "baselines", "arm_n",
                                    "graded_ab_paired.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
