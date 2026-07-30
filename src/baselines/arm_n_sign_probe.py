r"""arm_n_sign_probe.py -- the clean sign test: SAME QUESTIONS, different answer expressiveness.

WHY THIS EXISTS (confound found 2026-07-30, mid-run). arm_n_tower.py compares the tower at budget k
under arm A (k items drawn from the user's LIKES) against arm N (k items drawn from their FULL rated
history). Those two arms ask DIFFERENT QUESTIONS: arm N spends ~56% of its budget on items the user
disliked, arm A spends all of it on items they liked. So "N_k2 < A_k2" cannot be read as "sign does not
help" -- it mostly says a like is a more informative answer than a dislike, which is a statement about
question SELECTION, not about answer expressiveness.

An interview asks about an ITEM. What differs between the protocols is what the user is allowed to say
back. So hold the question set fixed and vary only the answer contract:

  ASK   the same k items, drawn once per user from their full rated history (fixed seed).
  ARM A the Liang contract: sub-3.5 answers do not exist, so a disliked item comes back as UNKNOWN and
        is simply not folded. Only the likes among the k are input.
  ARM N the native contract: every one of the k answers is folded with its real graded level, so a
        dislike enters as a NEGATIVE observation (the tower's gamma is negative for levels 0-4).

gap = N - A at fixed k is then exactly what a "no" is worth, with question selection held constant and
nothing else moving. Pool is the protocol pool in both, so demoting the observed dislikes wins nothing.

This is the number R5 should be scoped on.

  python src/baselines/arm_n_sign_probe.py [--snapshot .cache/instrument/t2final_best.pt]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import metrics as M
from arm_n import load_arm_n
from arm_n_tower import graded_to_levels, SNAP_DEFAULT, COLD_SEED, OUT
from train_tower_t2 import (build_model, make_graded_predict_fn, truncate_graded, compute_head_mask)

BUDGETS = [1, 2, 4, 8, 16, 32]
LIKE_MIN_LEVEL = 7          # level 7 == 4.0 stars; the Liang filter keeps r > 3.5


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sign_probe.log"), "a") as f:
        f.write(line + "\n")


def keep_likes_only(L):
    """The Liang answer contract applied to an already-chosen question set: a sub-3.5 answer does not
    exist in that protocol, so drop it. The QUESTION was still asked -- it just came back unusable."""
    K = L.tocsr().copy()
    K.data[K.data < LIKE_MIN_LEVEL + 1] = 0.0     # L carries level+1
    K.eliminate_zeros()
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    a = ap.parse_args()

    t0 = time.time()
    D = load_arm_n(log=logln)
    ni = D["n_items"]
    _hm, cnt = compute_head_mask(D["train"], ni)
    ma = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                            train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                            warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, _t, _p, _g = build_model(ma, ni, cnt)
    blob = torch.load(a.snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"]); enc.eval()
    Wd, bd = decoder.weight.detach(), decoder.bias.detach()
    logln(f"[probe] snapshot {os.path.basename(a.snapshot)} ep={blob.get('epoch', '?')}")

    L_full = graded_to_levels(D["g_te_tr"])
    res = {}
    for k in BUDGETS:
        Lk = truncate_graded(L_full, k, COLD_SEED)          # the SHARED question set
        Lk_A = keep_likes_only(Lk)                          # same questions, Liang answer contract
        askedA = Lk_A.nnz / Lk.shape[0]
        row = {}
        for tag, Lx in (("A", Lk_A), ("N", Lk)):
            pr = make_graded_predict_fn(enc, Wd, bd, Lx, check_nnz=False)
            r = M.evaluate(pr, D["te_tr"], D["te_te"], batch_size=500, head_mask=D["head_mask"],
                           mask_X=D["pool"])
            row[tag] = r
        g_full = row["N"]["ndcg@10"] - row["A"]["ndcg@10"]
        g_tail = row["N"]["tail_ndcg@10"] - row["A"]["tail_ndcg@10"]
        # Paired SE of the gap is not available from the aggregate dict; report each arm's SE.
        row["gap_full"], row["gap_tail"] = g_full, g_tail
        row["asked"] = k
        row["usable_A"] = askedA
        res[f"k{k}"] = row
        logln(f"[probe] k={k:3d}  A(likes only, {askedA:4.1f}/{k} usable) full={row['A']['ndcg@10']:.4f} "
              f"tail={row['A']['tail_ndcg@10']:.4f}  |  N(all signed) full={row['N']['ndcg@10']:.4f} "
              f"tail={row['N']['tail_ndcg@10']:.4f}  |  GAP full={g_full:+.4f} tail={g_tail:+.4f}")

    with open(os.path.join(OUT, "sign_probe.json"), "w") as f:
        json.dump({"snapshot": os.path.basename(a.snapshot), "seed": COLD_SEED,
                   "like_min_level": LIKE_MIN_LEVEL, "results": res}, f, indent=2)
    logln(f"[probe] done in {(time.time() - t0) / 60:.1f} m")


if __name__ == "__main__":
    main()
