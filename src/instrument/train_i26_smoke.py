r"""train_i26_smoke.py -- the smoke run: does the interview-native design learn at all?

SPEC: docs/design/RETRAIN_DESIGN_SHEET.md section 9 step 4. Short run on a SUBSET. Its job is to fail
loudly and cheaply if the design is wrong, before a multi-hour train and a two-day cascade.

WHAT IT WATCHES, AND WHY BOTH (author correction): TRAIN loss and HELD-OUT metric, separately per regime
bucket. Watching held-out alone cannot distinguish "the design is wrong" from "it has not seen enough
examples yet", and those need opposite responses:

    train loss FLAT                  -> NOT a design failure. Give it more steps/data before judging.
    train down, held-out WORSE       -> memorising, or the design is wrong. Stop.
    both improving                   -> proceed to the full run.

Held-out users here are a slice of the TRAIN cohort. The canonical val/test cohorts stay quarantined;
the real gates run later, once.

  python src/instrument/train_i26_smoke.py [--users 20000] [--steps 400]
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import metrics as M
from arm_n import load_arm_n
import interview_strategies as ST
from i26_encoder import build_i26, UNSEEN_LEVEL
from interview_curriculum import StrategyFamily, draw, P_FULL, P_INTERVIEW
from train_tower_t2 import (load_recvae_teacher, pack_tokens, level_to_sv, apply_sign_prior,
                            compute_head_mask, PROC)

OUT = os.path.join(_ROOT, "experiments", "instrument")
W_NEG = 0.1
NEG_CLAMP = -8.0


def logln(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "i26_smoke.log"), "a") as f:
        f.write(line + "\n")


def profiles(G, idx):
    """Train-user profiles from the cached graded matrix: items / levels / vals / liked / disliked."""
    G = G.tocsr(); out = []
    for u in idx:
        s, e = G.indptr[u], G.indptr[u + 1]
        sid = G.indices[s:e].astype(np.int64)
        lv = np.clip(np.rint(G.data[s:e] * 2).astype(np.int64) - 1, 0, 9)
        if len(sid) < 6 or (lv >= 7).sum() < 2:
            continue
        vals = (((lv.astype(np.float64) + 1.0) / 2.0 - 2.75) / 2.25).astype(np.float32)
        out.append({"items": sid, "levels": lv, "vals": vals,
                    "liked": sid[lv >= 7], "disliked": sid[lv < 7]})
    return out


def full_dropout(u, rng):
    """The EXISTING full-profile/dropout regime, unchanged (protects the floor)."""
    its = u["items"]
    keep = rng.random(len(its)) >= rng.uniform(0.0, 0.8)
    if not keep.any():
        keep[rng.integers(0, len(its))] = True
    tgt = np.setdiff1d(u["liked"], its[keep])
    if len(tgt) == 0:
        return None
    negs = np.setdiff1d(u["disliked"], its[keep])
    return (its[keep], u["levels"][keep], u["vals"][keep], tgt, negs)


def pack(batch):
    """(inp, lv, unseen) -> token tensors. Unseen items ride the sentinel level."""
    rows = []
    for (inp, lv, sv, unseen, tgt, negs) in batch:
        sid = np.concatenate([inp, unseen]).astype(np.int64)
        lvl = np.concatenate([lv, np.full(len(unseen), UNSEEN_LEVEL)]).astype(np.int64)
        rows.append((sid, lvl, level_to_sv(np.clip(lvl, 0, 9))))
    return pack_tokens(rows, binarize=False)


def loss_of(enc, Wd, bd, batch, ni):
    ids, vals, padm, lvs = pack(batch)
    z = enc(ids, vals, padm, lvs)
    logits = enc.logits(z, Wd, bd)
    logsm = F.log_softmax(logits, dim=-1)
    B = len(batch)
    tgt = torch.zeros((B, ni)); neg = torch.zeros((B, ni))
    for r, (_i, _l, _v, _u, t, n) in enumerate(batch):
        tgt[r, t] = 1.0
        if len(n):
            neg[r, n] = 1.0
    nll = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0))
    npen = (logsm * neg).sum(-1) / neg.sum(-1).clamp_min(1.0)
    return (nll + W_NEG * npen.clamp(min=NEG_CLAMP)).mean()


def regime_of(ex):
    """Label an example by regime so train/val loss can be tracked per bucket."""
    inp, lv, sv, unseen, tgt, negs = ex
    if len(unseen) > 0:
        return "interview"
    if len(inp) <= 1:
        return "empty"
    return "full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=100)
    a = ap.parse_args()

    D = load_arm_n(log=logln)
    ni = D["n_items"]
    _hm, cnt = compute_head_mask(D["train"], ni)
    rng = np.random.default_rng(0)
    idx = rng.choice(D["g_train"].shape[0], size=a.users, replace=False)
    us = profiles(D["g_train"], idx)
    n_val = max(500, len(us) // 10)
    val_us, tr_us = us[:n_val], us[n_val:]
    logln(f"[smoke] {len(tr_us)} train / {len(val_us)} held-out profiles (held-out is a slice of the "
          f"TRAIN cohort; canonical val/test stay quarantined)")

    H, H0 = ST.entropies(D["g_train"], ni, float(D["g_train"].shape[0]))
    Hn = H / max(H.max(), 1e-9); Fn = np.log1p(cnt) / max(np.log1p(cnt).max(), 1e-9)
    helf = 2.0 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)
    fam = StrategyFamily(cnt, H, H0, helf)

    ma = argparse.Namespace(arch="i26", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                            train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=a.lr,
                            warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    src = load_recvae_teacher(ni, hidden=ma.t_hidden, latent=ma.t_latent)
    enc, dec, params = build_i26(ni, src, ma, log=logln)
    apply_sign_prior(enc)
    # Warm-start the SHARED taste path from the certified checkpoint, so the smoke measures the NEW
    # parts learning rather than the old parts re-learning from scratch.
    blob = torch.load(os.path.join(_ROOT, ".cache", "instrument", "t2final_best.pt"), map_location="cpu")
    sd = {k: v for k, v in blob["enc"].items()}
    missing = enc.load_state_dict(sd, strict=False)
    logln(f"[smoke] warm-started taste path from t2final_best; new params left at init: "
          f"{[k for k in missing.missing_keys]}")
    Wd = dec.weight.detach(); bd = dec.bias.detach()
    opt = torch.optim.Adam(params, lr=a.lr)

    def make_batch(pool, r):
        out = []
        while len(out) < a.batch:
            ex = draw(pool[r.integers(len(pool))], r, fam, full_dropout)
            if ex is not None:
                out.append(ex)
        return out

    def val_report(tag):
        enc.eval()
        r = np.random.default_rng(1234)
        buckets = {"full": [], "interview": [], "empty": []}
        with torch.no_grad():
            for _ in range(12):
                b = make_batch(val_us, r)
                for reg in buckets:
                    sub = [e for e in b if regime_of(e) == reg]
                    if sub:
                        buckets[reg].append(float(loss_of(enc, Wd, bd, sub, ni)))
            e_ids, e_v, e_p, e_l = pack_tokens([(np.empty(0, np.int64), np.empty(0, np.int64),
                                                np.empty(0, np.float32))], binarize=False)
            z0 = enc(e_ids, e_v, e_p, e_l)
            sc0 = enc.logits(z0, Wd, bd).numpy()[0]

        def cpred(X, _s=sc0):
            return np.repeat(_s[None, :], X.shape[0], axis=0).astype(np.float32)
        r0 = M.evaluate(cpred, D["te_tr"], D["te_te"], batch_size=500,
                        head_mask=D["head_mask"], mask_X=D["pool"])
        enc.train()
        logln(f"[smoke] {tag} HELD-OUT loss  " +
              "  ".join(f"{k}={np.mean(v):.4f}" if v else f"{k}=--" for k, v in buckets.items()) +
              f"  | empty-set NDCG@10={r0['ndcg@10']:.4f} (target >= 0.1626)")

    logln("[smoke] ---- step 0 (should equal the certified model: all new parts are zero-init) ----")
    val_report("step   0")
    r = np.random.default_rng(7)
    run = {"full": [], "interview": [], "empty": []}
    enc.train()
    for step in range(1, a.steps + 1):
        b = make_batch(tr_us, r)
        opt.zero_grad()
        L = loss_of(enc, Wd, bd, b, ni)
        L.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        with torch.no_grad():
            for reg in run:
                sub = [e for e in b if regime_of(e) == reg]
                if sub:
                    run[reg].append(float(loss_of(enc, Wd, bd, sub, ni)))
        if step % a.eval_every == 0:
            logln(f"[smoke] step {step:3d} TRAIN loss    " +
                  "  ".join(f"{k}={np.mean(v[-40:]):.4f}" if v else f"{k}=--" for k, v in run.items()))
            val_report(f"step {step:3d}")
    logln("[smoke] done")


if __name__ == "__main__":
    main()
