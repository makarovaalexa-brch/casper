r"""posterior_blend.py -- B2+B3: put R4 in the scoring path, anchored on the population marginal.

THE PROBLEM (measured 2026-07-30). Our zero-evidence decode scores 0.1553 while counting (Most-Popular)
scores 0.1626. We start 0.0073 BELOW the trivial estimator, and the residual short-budget deficit to
Golbandi's lookup after its unknown branch is stripped is +0.0046 at k=2 / +0.0021 at k=4 -- SMALLER
than the prior deficit. We are not losing to his tree; we are losing to our own empty-set decode.

WHY PLAIN R4 SHRINKAGE CANNOT FIX IT. The decoder is LINEAR (`z @ Wd.T + bd`), so the posterior-expected
decode equals the decode of the posterior mean, and shrinking z toward the prior mean shrinks the SCORES
toward that same 0.1553 object. The load-bearing quantity is the ANCHOR, not the shrinkage.

THE FIX. For a user about whom nothing is known, the correct Bayesian prediction is the POPULATION ITEM
MARGINAL -- which is exactly Most-Popular, computed by counting on TRAIN data (no eval contact). So
blend the decode toward log-popularity, with the weight supplied by the belief itself:

    Lambda_0 = diag(1 / var_emp)              prior precision; var_emp = per-dim variance of the fold
                                              over TRAIN users, recomputed for THIS snapshot
    Lambda_t = Lambda_0 + sum_j phi_j phi_j^T  unit observation precision along each answered item's
                                              decoder row phi_j = Wd[j]      (UNTUNED)
    w_i      = 1 - (phi_i' Sigma_t phi_i) / (phi_i' Sigma_0 phi_i)      per ITEM, in [0,1)
    score_i  = w_i * zscore(decode)_i + (1 - w_i) * zscore(log popcount)_i

w_i is the fractional reduction in PREDICTIVE VARIANCE along item i's own direction. At zero evidence
Sigma_t == Sigma_0, so w = 0 and the model returns exactly the popularity ranking -- the floor becomes
0.1626 by construction, not by a bolted-on gate. As evidence accumulates along directions near item i,
w_i -> 1 and the decode is recovered untouched, so nothing at full profile moves.

NOTHING HERE IS TUNED. No alpha, no threshold, no per-budget weight. That discipline is the whole point:
a fitted blend weight would be the trivial MostPop gate wearing a Bayesian costume, and a reviewer would
say so. The two z-scores put the components on a common scale; z-scoring is monotone, so neither
component's own ranking is altered -- only their relative influence.

PROVENANCE NOTE. `.cache/instrument/belief_i25.pt` holds a FITTED belief, but it was fitted against
`t2i25_EP4_SNAP.pt` and not the certified `t2final_best.pt`, and its confidence cells are inverted
(vague 27.66 precision vs know-well 0.013). We do NOT inherit that fit. var_emp is recomputed here for
the snapshot actually in use; the observation precision is unit.

  python src/baselines/posterior_blend.py --anchor pop      # B2
  python src/baselines/posterior_blend.py --anchor empty    # the anchor-swap CONTROL
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
from arm_n_tower import SNAP_DEFAULT, graded_to_levels
import interview_strategies as ST
from train_tower_t2 import (build_model, compute_head_mask, pack_tokens, level_to_sv, truncate_graded)

OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
VAR_CACHE = os.path.join(_ROOT, ".cache", "instrument", "var_emp_t2final.npy")
BUDGETS = [1, 2, 4, 8, 16]
ARMS = ["helf", "entropy0", "popularity"]


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "posterior_blend.log"), "a") as f:
        f.write(line + "\n")


def fold_z(enc, rows, batch=256):
    """z for a list of (sids, levels) token rows."""
    out = np.zeros((len(rows), 200), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            pack = [(np.asarray(a, np.int64), np.asarray(b, np.int64),
                     level_to_sv(np.asarray(b, np.int64))) for a, b in chunk]
            ids, vals, pad, lvs = pack_tokens(pack, binarize=False)
            out[s:s + len(chunk)] = enc(ids, vals, pad, lvs).numpy()
    return out


def empirical_var(enc, g_train, n_sample_users, log=logln):
    """var_emp: per-dim variance of the tower fold over TRAIN users. Snapshot-specific, so it is
    recomputed rather than taken from the belief checkpoint (which was fitted on another snapshot)."""
    if os.path.exists(VAR_CACHE):
        v = np.load(VAR_CACHE)
        log(f"[blend] var_emp from cache: mean={v.mean():.4f}")
        return v
    G = g_train.tocsr()
    rows = []
    for u in range(n_sample_users):
        s, e = G.indptr[u], G.indptr[u + 1]
        sid = G.indices[s:e].astype(np.int64)
        lv = np.clip(np.rint(G.data[s:e] * 2).astype(np.int64) - 1, 0, 9)
        rows.append((sid, lv))
    log(f"[blend] folding {len(rows)} TRAIN users for var_emp ...")
    Z = fold_z(enc, rows)
    v = np.maximum(Z.var(axis=0), 1e-6).astype(np.float32)
    os.makedirs(os.path.dirname(VAR_CACHE), exist_ok=True)
    np.save(VAR_CACHE, v)
    log(f"[blend] var_emp computed: mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}")
    return v


def item_weights(phis, v0, Wd_np, q0):
    """w_i = 1 - phi_i' Sigma_t phi_i / phi_i' Sigma_0 phi_i, for ONE user.
    phis: (m, d) decoder rows of the answered items (the observation directions). Woodbury:
        Sigma_t = V0 - V0 P' (I + P V0 P')^{-1} P V0 ,  P = phis
    so phi_i' Sigma_t phi_i = q0_i - || L^{-1} (P V0 phi_i) ||^2 with L = chol(I + P V0 P')."""
    if phis.shape[0] == 0:
        return np.zeros(Wd_np.shape[0], dtype=np.float32)
    P = phis.astype(np.float64)
    PV = P * v0[None, :]                              # (m,d)
    Mx = np.eye(P.shape[0]) + PV @ P.T                # (m,m)
    L = np.linalg.cholesky(Mx)
    T = PV @ Wd_np.T                                  # (m, n_items)
    S = np.linalg.solve(L, T)                         # (m, n_items)
    quad_t = q0 - (S ** 2).sum(axis=0)
    return np.clip(1.0 - quad_t / np.maximum(q0, 1e-12), 0.0, 1.0).astype(np.float32)


def zs(x):
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    return (x - m) / np.maximum(s, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--anchor", default="pop", choices=["pop", "empty"],
                    help="pop = population marginal (B2). empty = the LEARNED empty-set decode "
                         "(the CONTROL: if this works equally well the mechanism is variance, not "
                         "prior quality).")
    ap.add_argument("--var_users", type=int, default=20000)
    ap.add_argument("--budgets", default="1,2,4,8,16")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]

    D = load_arm_n(log=logln)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    _hm, cnt = compute_head_mask(D["train"], ni)
    ma = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                            train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                            warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, dec, _t, _p, _g = build_model(ma, ni, cnt)
    blob = torch.load(a.snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"]); dec.load_state_dict(blob["decoder"]); enc.eval()
    Wd = dec.weight.detach(); bd = dec.bias.detach()
    Wd_np = Wd.numpy().astype(np.float64); bd_np = bd.numpy().astype(np.float64)

    var_emp = empirical_var(enc, D["g_train"], min(a.var_users, D["g_train"].shape[0]))
    v0 = var_emp.astype(np.float64)
    q0 = (Wd_np ** 2) @ v0                                    # phi_i' Sigma_0 phi_i, per item

    # The two anchors. Population marginal = counting on TRAIN (no eval contact).
    pop_anchor = zs(np.log1p(cnt.astype(np.float64)))
    z_empty = fold_z(enc, [(np.empty(0, np.int64), np.empty(0, np.int64))])[0]
    empty_anchor = zs(z_empty.astype(np.float64) @ Wd_np.T + bd_np)
    anchor = pop_anchor if a.anchor == "pop" else empty_anchor
    logln(f"[blend] anchor={a.anchor}; q0 mean={q0.mean():.4f}")

    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    lvl = ST.level_lookup(D["g_te_tr"])
    res = {}
    for arm in ARMS:
        res[arm] = {}
        for k in budgets:
            ts = time.time()
            asked, answered = ST.ask(arm, glob, bank, n, ni, lvl, k)
            rows = [(np.asarray([i for i, _ in t], np.int64),
                     np.asarray([l for _, l in t], np.int64)) for t in answered]
            Z = fold_z(enc, rows)
            base = Z.astype(np.float64) @ Wd_np.T + bd_np
            base_z = zs(base)
            scores = np.empty((n, ni), dtype=np.float32)
            wbar = 0.0
            for u in range(n):
                ids = rows[u][0]
                w = item_weights(Wd_np[ids], v0, Wd_np, q0) if len(ids) else np.zeros(ni, np.float32)
                wbar += float(w.mean())
                scores[u] = (w * base_z[u] + (1.0 - w) * anchor).astype(np.float32)
            st = {"c": 0}

            def pr(X, _s=scores, _st=st):
                o = _s[_st["c"]:_st["c"] + X.shape[0]].copy(); _st["c"] += X.shape[0]; return o
            r = M.evaluate(pr, D["te_tr"], D["te_te"], batch_size=500,
                           head_mask=D["head_mask"], mask_X=D["pool"])
            res[arm][f"k{k}"] = {"full": r["ndcg@10"], "tail": r["tail_ndcg@10"],
                                 "mean_w": wbar / n}
            logln(f"[blend:{a.anchor}] {arm:11s} k={k:2d} mean_w={wbar/n:.4f} "
                  f"full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f} "
                  f"({(time.time()-ts)/60:.1f}m)")
            json.dump(res, open(os.path.join(OUT, f"posterior_blend_{a.anchor}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
