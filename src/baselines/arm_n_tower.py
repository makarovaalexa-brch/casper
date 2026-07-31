r"""arm_n_tower.py -- the instrument on its native input contract (arm N), at full profile AND at
interview budgets.

SPEC: docs/results/PROTOCOL_DISLIKE_DISCARD.md section 11 (arm N).

WHAT ARM A GAVE THE TOWER, precisely -- because this is easy to get wrong and the whole comparison
rests on it. `build_graded_eval_matrix` asserts `L.nnz == test_tr.csv nnz`: the tower's canonical
0.3482 folds EXACTLY the same (user, item) pairs as every baseline, with the real half-star level
substituted for the 1.0. That is the INTENSITY channel (4.0 / 4.5 / 5.0 within likes), worth -0.0002 at
full profile. The tower was never handed extra support in arm A.

WHAT ARM N ADDS: the fold-in becomes the user's full rated history (all bands, held-out targets
removed), so a dislike is finally foldable -- levels 0..6 map to NEGATIVE signed values through the
tower's own level_to_sv (star 2.75 is the zero). The candidate pool is the protocol's, identical for
every model, so nothing is won by demoting the observed dislikes: they are already out of everyone's
list. Any gain is off-support generalisation.

BUDGETS. Full profile is the least interesting row here and the most dangerous to over-read: the
chapter's claim lives in the scarce-evidence regime. So this reports k in {2, 4, 8, 16, full} under the
SAME pool, with answers drawn from the full rated history (a "no" is now a legal answer). The k-subset
is a fixed random draw per user (COLD_SEED), identical across arms by construction.

  python src/baselines/arm_n_tower.py [--snapshot .cache/instrument/t2final_best.pt]
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
from train_tower_t2 import (build_model, make_graded_predict_fn, build_graded_eval_matrix,
                            truncate_graded, reproduce_partition, star_to_level, compute_head_mask)

OUT = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
SNAP_DEFAULT = os.path.join(_ROOT, ".cache", "instrument", "t2final_best.pt")
COLD_SEED = 4242
BUDGETS = [2, 4, 8, 16]


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tower.log"), "a") as f:
        f.write(line + "\n")


def graded_to_levels(G):
    """Ratings CSR -> the tower's level+1 encoding (its graded matrices carry level+1, 0 == absent)."""
    L = G.tocsr().copy().astype(np.float32)
    L.data = (star_to_level(L.data).astype(np.float32) + 1.0)
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--arch", default="i25", choices=["i25", "i26"],
                    help="i26 = the interview-native tower (exposure branch over asked-but-unseen "
                         "tokens). Its state dict cannot be loaded into an i25 model. NOTE: the arm-A "
                         "row is a LIKES-ONLY fold-in with no asked-but-unseen items, so the exposure "
                         "branch is inert on that row -- this measures the same quantity as the "
                         "certified 0.3482 and is directly comparable to it.")
    a = ap.parse_args()

    t0 = time.time()
    D = load_arm_n(log=logln)
    ni, head_mask = D["n_items"], D["head_mask"]
    _hm, cnt = compute_head_mask(D["train"], ni)

    ma = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                            train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                            warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    if a.arch == "i26":
        sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))
        from i26_encoder import build_i26
        from train_tower_t2 import load_recvae_teacher, apply_sign_prior
        ma.arch = "i26"
        src = load_recvae_teacher(ni, hidden=600, latent=200)
        enc, decoder, _p, _g = build_i26(ni, src, ma, log=logln)
        apply_sign_prior(enc)
    else:
        enc, decoder, _teacher, _p, _g = build_model(ma, ni, cnt)
    blob = torch.load(a.snapshot, map_location="cpu")
    miss = enc.load_state_dict(blob["enc"], strict=(a.arch != "i26"))
    if a.arch == "i26" and (miss.missing_keys or miss.unexpected_keys):
        raise SystemExit(f"[tower] REFUSING to load: missing={sorted(miss.missing_keys)} "
                         f"unexpected={sorted(miss.unexpected_keys)}")
    decoder.load_state_dict(blob["decoder"])
    enc.eval()
    Wd, bd = decoder.weight.detach(), decoder.bias.detach()
    logln(f"[tower] snapshot {os.path.basename(a.snapshot)} ep={blob.get('epoch', '?')} "
          f"val_full={blob.get('val_full', float('nan')):.4f}")

    # ARM A fold-in: the canonical likes-only support, real star levels. Reproduces 0.3482.
    unique_uid, _tr, _vd, _te, _n, raw, show2id, usid = reproduce_partition()
    L_A, _ = build_graded_eval_matrix(raw, unique_uid, show2id, usid, "test")
    # ARM N fold-in: full rated history, all bands, targets already excluded by arm_n.
    L_N = graded_to_levels(D["g_te_tr"])
    logln(f"[tower] fold-in nnz: arm A {L_A.nnz} (likes only) vs arm N {L_N.nnz} (all bands, "
          f"{float((L_N.data <= 7).mean()):.3f} of tokens are dislikes at level<=6)")

    res = {}

    def run(tag, L, pool, k=None):
        ts = time.time()
        Lk = L if k is None else truncate_graded(L, k, COLD_SEED)
        pr = make_graded_predict_fn(enc, Wd, bd, Lk, check_nnz=False)
        r = M.evaluate(pr, D["te_tr"], D["te_te"], batch_size=500, head_mask=head_mask, mask_X=pool)
        res[tag] = r
        logln(f"[tower] {tag:22s} full@10={r['ndcg@10']:.4f} tail@10={r['tail_ndcg@10']:.4f} "
              f"ndcg@100={r['ndcg@100']:.4f} ({(time.time() - ts) / 60:.1f} m)")
        return r

    # Row 1: arm A reproduction. MUST snap to 0.3482 / 0.2462 -- proves the checkpoint and path are the
    # ones that produced the chapter's number before anything else is believed.
    run("A_full", L_A, D["te_tr"])
    # Row 2: the pool change alone, likes-only input. Isolates "what the fairer pool is worth".
    run("A_input_N_pool", L_A, D["pool"])
    # Row 3: the full arm-N contract.
    run("N_full", L_N, D["pool"])
    # Rows 4+: the regime the chapter's claim actually lives in.
    for k in BUDGETS:
        run(f"A_k{k}_N_pool", L_A, D["pool"], k=k)
        run(f"N_k{k}", L_N, D["pool"], k=k)

    snap_ok = abs(res["A_full"]["ndcg@10"] - 0.3482) <= 1e-3
    logln(f"[gate] SNAP arm-A full@10 {res['A_full']['ndcg@10']:.4f} vs chapter 0.3482 -> "
          f"{'PASS' if snap_ok else '*** FAIL -- stop, wrong checkpoint or path ***'}")
    with open(os.path.join(OUT, "tower.json"), "w") as f:
        json.dump({"snapshot": os.path.basename(a.snapshot), "snap_ok": bool(snap_ok),
                   "results": res}, f, indent=2)
    logln(f"[tower] done in {(time.time() - t0) / 60:.1f} m")
    return 0 if snap_ok else 1


if __name__ == "__main__":
    sys.exit(main())
