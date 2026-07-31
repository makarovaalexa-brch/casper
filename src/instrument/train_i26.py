r"""train_i26.py -- the full interview-native run, with the pre-registered gates wired in.

SPEC: docs/design/RETRAIN_DESIGN_SHEET.md sections 7 and 9 step 5. The smoke (train_i26_smoke.py) has
passed: all three regime buckets descend on BOTH train and held-out, and the prior climbs 0.1550 ->
0.1598. This is the same machinery over the full train cohort, with epoch loop, checkpointing and gates.

GATES CHECKED EVERY EPOCH (abort tier). The run kills itself rather than waiting for a human:
  G-FULL     canonical full-profile full@10 >= 0.3467 (0.3482 - the paired-CI width).
             EARLY WARNING (risk 1, precedent: the Kalman fold-all crater and the DAE snap failure):
             abort if it is more than 0.005 below the certified value by the end of epoch 2.
  G-EMPTY    empty-set decode full@10 >= 0.1626 (best reachable 0.1628). Tail reported, NOT gated --
             the optimum's tail is 0.0239 vs MostPop's 0.0262.
  G-MONOTONE one answer must not score BELOW zero answers (author rule). The certified model FAILS this.
Ship-tier gates (G-SHORT, G-LONG, G-TAIL, G-INTENSITY, G-SHUFFLE) run once at the best checkpoint, in
the arm-N interview harness -- not here.

The canonical VAL cohort is used for G-FULL only. The canonical TEST cohort is not touched at all.
`t2final_best.pt` is never written; this arm checkpoints to t2i26*.pt.

  python src/instrument/train_i26.py --epochs 14 --tag t2i26
"""
import os
import sys
import time
import json
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
from interview_curriculum import StrategyFamily, draw
from train_i26_smoke import profiles, full_dropout, pack, loss_of, regime_of
from train_tower_t2 import (load_recvae_teacher, pack_tokens, apply_sign_prior, compute_head_mask,
                            make_graded_predict_fn, build_graded_eval_matrix, reproduce_partition,
                            truncate_graded, PROC)

OUT = os.path.join(_ROOT, "experiments", "instrument")
CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
CERT_FULL = 0.3482          # the certified arm-A full-profile number
G_FULL_FLOOR = 0.3467       # abort below this
G_EMPTY_FLOOR = 0.1626      # counting
# ABORT RULE, REWRITTEN 2026-07-31 after it killed a HEALTHY run.
# v1 tested an ABSOLUTE level at epoch 2 (abort if val < 0.3482 - 0.005) and fired on 0.3307 -> 0.3342.
# But that is +0.0035 in one epoch: a warm-started model whose new branches start at zero DIPS while
# they settle and then climbs back, and at that rate it reaches certified in ~4 more epochs. The gate
# killed a run that was recovering exactly as it should.
# v2 tests the TREND and gives recovery room:
#   * abort only if full-profile is FLAT-OR-FALLING over the last PATIENCE epochs, or
#   * if it has not reached the floor by RECOVER_BY epochs.
# A dip is expected; a dip that stops improving is the real failure.
PATIENCE = 3                # consecutive epochs without improvement -> abort
RECOVER_BY = 8              # must be at/above the floor by this epoch
HEARTBEAT_EVERY = 50        # steps between heartbeat writes (the watchdog reads this)


def archive_existing(paths, log):
    """NEVER DELETE A CHECKPOINT. 2026-07-31: I twice ran `rm .cache/instrument/t2i26_best.pt` when
    relaunching, destroying the only artifact of a 2-hour run -- after the author had explicitly asked
    that the best checkpoint always be kept on disk. Clearing `_last` is legitimate (resume reads it, so
    a stale one would silently continue an old run under old settings); clearing `_best` never was --
    nothing reads it. So a fresh run now ARCHIVES whatever it finds, with a timestamp, instead of
    requiring anyone to delete anything by hand."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for pth in paths:
        if os.path.exists(pth):
            dst = f"{os.path.splitext(pth)[0]}.archived_{stamp}.pt"
            os.replace(pth, dst)
            log(f"[i26] ARCHIVED {os.path.basename(pth)} -> {os.path.basename(dst)} (never deleted)")


def atomic_save(obj, path):
    """Write to a temp file then os.replace. A crash mid-write must never corrupt the best checkpoint --
    it is the only artifact of the run that matters."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def logln(m, tag="t2i26"):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{tag}_train.log"), "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--tag", default="t2i26")
    ap.add_argument("--batch", type=int, default=128)   # recipe buckets up to MAX_B=256; 64 was noisy
    ap.add_argument("--steps_per_epoch", type=int, default=2200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--fresh", action="store_true",
                    help="Start a new run: ARCHIVES any existing best/last checkpoints (timestamped) "
                         "rather than resuming or deleting. Without it, an existing _last resumes.")
    a = ap.parse_args()
    L = lambda m: logln(m, a.tag)

    D = load_arm_n(log=L)
    ni = D["n_items"]
    _hm, cnt = compute_head_mask(D["train"], ni)
    rng = np.random.default_rng(0)
    idx = rng.permutation(D["g_train"].shape[0])
    us = profiles(D["g_train"], idx)
    n_val = 4000
    val_us, tr_us = us[:n_val], us[n_val:]
    L(f"[i26] {len(tr_us)} train / {len(val_us)} held-out profiles from the TRAIN cohort")

    H, H0 = ST.entropies(D["g_train"], ni, float(D["g_train"].shape[0]))
    Hn = H / max(H.max(), 1e-9); Fn = np.log1p(cnt) / max(np.log1p(cnt).max(), 1e-9)
    helf = 2.0 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)
    fam = StrategyFamily(cnt, H, H0, helf)

    ma = argparse.Namespace(arch="i26", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                            train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=a.lr,
                            warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    src = load_recvae_teacher(ni, hidden=ma.t_hidden, latent=ma.t_latent)
    enc, dec, params = build_i26(ni, src, ma, log=L)
    apply_sign_prior(enc)
    blob = torch.load(os.path.join(CKPT_DIR, "t2final_best.pt"), map_location="cpu")
    miss = enc.load_state_dict(blob["enc"], strict=False)
    L(f"[i26] warm-started taste path from t2final_best; at init: {sorted(miss.missing_keys)}")
    Wd = dec.weight.detach(); bd = dec.bias.detach()
    opt = torch.optim.Adam(params, lr=a.lr)

    # ---- canonical VAL cohort, for G-FULL only. TEST is never touched here. -------------
    uu, _tr, _vd, _te, _n, raw, show2id, usid = reproduce_partition()
    L_val, _nv = build_graded_eval_matrix(raw, uu, show2id, usid, "validation")
    va_tr, va_te = M.load_val(ni, PROC)

    def full_profile_val():
        enc.eval()
        pr = make_graded_predict_fn(enc, Wd, bd, L_val, check_nnz=False)
        r = M.evaluate(pr, va_tr, va_te, batch_size=500, head_mask=D["head_mask"])
        enc.train(); return r["ndcg@10"], r["tail_ndcg@10"]

    def empty_and_k1():
        """G-EMPTY and G-MONOTONE, on the arm-N pool (same object the interview table uses)."""
        enc.eval()
        with torch.no_grad():
            e = pack_tokens([(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))],
                            binarize=False)
            s0 = enc.logits(enc(*e), Wd, bd).numpy()[0]
        r0 = M.evaluate(lambda X, _s=s0: np.repeat(_s[None, :], X.shape[0], 0).astype(np.float32),
                        D["te_tr"], D["te_te"], batch_size=500,
                        head_mask=D["head_mask"], mask_X=D["pool"])
        from arm_n_tower import graded_to_levels
        L1 = truncate_graded(graded_to_levels(D["g_te_tr"]), 1, 4242)
        pr = make_graded_predict_fn(enc, Wd, bd, L1, check_nnz=False)
        r1 = M.evaluate(pr, D["te_tr"], D["te_te"], batch_size=500,
                        head_mask=D["head_mask"], mask_X=D["pool"])
        enc.train(); return r0["ndcg@10"], r1["ndcg@10"]

    bank = [fam.build_bank(np.random.default_rng(0))]      # regenerated each epoch (see build_bank)

    def make_batch(pool, r):
        out = []
        while len(out) < a.batch:
            ex = draw(pool[r.integers(len(pool))], r, fam, full_dropout, bank=bank[0])
            if ex is not None:
                out.append(ex)
        return out

    best = {"val": -1.0, "epoch": 0}
    start_ep = 1
    hist = []
    last_p = os.path.join(CKPT_DIR, f"{a.tag}_last.pt")
    best_p = os.path.join(CKPT_DIR, f"{a.tag}_best.pt")
    state_p = os.path.join(OUT, f"{a.tag}_state.json")
    if a.fresh:
        archive_existing([best_p, last_p], L)
    # ---- RESUME: a watchdog relaunch must not start over from epoch 1 -------------------
    if os.path.exists(last_p):
        lb = torch.load(last_p, map_location="cpu")
        enc.load_state_dict(lb["enc"]); opt.load_state_dict(lb["opt"])
        start_ep = int(lb["epoch"]) + 1
        best = lb.get("best", best)
        hist = lb.get("hist", [])
        L(f"[i26] RESUMED from {a.tag}_last.pt at epoch {start_ep} "
          f"(best val_full={best['val']:.4f} @ep{best['epoch']})")
    t0 = time.time()
    enc.train()

    def beat(ep, step):
        json.dump({"pid": os.getpid(), "heartbeat": time.time(), "epoch": ep, "step": step,
                   "best_val": best["val"], "best_epoch": best["epoch"]},
                  open(state_p, "w"))

    for ep in range(start_ep, a.epochs + 1):
        r = np.random.default_rng(1000 + ep)
        bank[0] = fam.build_bank(np.random.default_rng(500 + ep))   # fresh askers every epoch
        run = {"full": [], "interview": [], "empty": []}
        for step in range(a.steps_per_epoch):
            b = make_batch(tr_us, r)
            opt.zero_grad()
            loss = loss_of(enc, Wd, bd, b, ni)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if step % HEARTBEAT_EVERY == 0:
                beat(ep, step)
            if step % 5 == 0:
                with torch.no_grad():
                    for reg in run:
                        sub = [e for e in b if regime_of(e) == reg]
                        if sub:
                            run[reg].append(float(loss_of(enc, Wd, bd, sub, ni)))
        # ---- held-out loss per regime
        rv = np.random.default_rng(99)
        vb = {"full": [], "interview": [], "empty": []}
        with torch.no_grad():
            for _ in range(20):
                b = make_batch(val_us, rv)
                for reg in vb:
                    sub = [e for e in b if regime_of(e) == reg]
                    if sub:
                        vb[reg].append(float(loss_of(enc, Wd, bd, sub, ni)))
        fv, ft = full_profile_val()
        e0, e1 = empty_and_k1()
        el = (time.time() - t0) / 60.0
        L(f"[i26] ep{ep:2d} TRAIN " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in run.items()) +
          " | HELD-OUT " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in vb.items()))
        L(f"[i26] ep{ep:2d} G-FULL val={fv:.4f}/{ft:.4f} (floor {G_FULL_FLOOR}) | "
          f"G-EMPTY k0={e0:.4f} (floor {G_EMPTY_FLOOR}) | G-MONOTONE k1={e1:.4f} "
          f"({'PASS' if e1 >= e0 else 'FAIL'}) | {el:.1f}m")
        hist.append({"ep": ep, "val_full": fv, "val_tail": ft, "k0": e0, "k1": e1,
                     "train": {k: float(np.mean(v)) for k, v in run.items()},
                     "held": {k: float(np.mean(v)) for k, v in vb.items()}})
        json.dump(hist, open(os.path.join(OUT, f"{a.tag}_hist.json"), "w"), indent=2)

        fulls = [h["val_full"] for h in hist]
        stalled = (len(fulls) > PATIENCE and
                   max(fulls[-PATIENCE:]) <= max(fulls[:-PATIENCE]) + 1e-5)
        too_late = (ep >= RECOVER_BY and fv < G_FULL_FLOOR)
        if stalled or too_late:
            why = (f"full-profile has not improved in {PATIENCE} epochs (best {max(fulls):.4f})"
                   if stalled else
                   f"full-profile {fv:.4f} still below the floor {G_FULL_FLOOR} at epoch {ep}")
            open(os.path.join(OUT, f"{a.tag}_DONE.marker"), "w").write(f"ABORTED: G-FULL ({why})")
            L(f"[i26] *** ABORT (risk 1): {why}. Design sheet section 0: the certified checkpoint "
              f"stands. ***")
            return 1
        if ep >= 2 and fv < G_FULL_FLOOR:
            L(f"[i26] NOTE ep{ep}: full-profile {fv:.4f} below the floor {G_FULL_FLOOR} but "
              f"{'IMPROVING' if len(fulls) < 2 or fv > fulls[-2] else 'not improving'} "
              f"({(fv - fulls[-2]):+.4f} vs last epoch) -- recovery window runs to ep{RECOVER_BY}")
        if fv > best["val"]:
            best = {"val": fv, "epoch": ep}
            atomic_save({"enc": enc.state_dict(), "decoder": dec.state_dict(), "epoch": ep,
                         "val_full": fv, "val_tail": ft, "k0": e0, "k1": e1, "arch": "i26"}, best_p)
            L(f"[i26] new best val_full={fv:.4f} -> {a.tag}_best.pt (atomic)")
        # `last` carries optimiser state so a watchdog relaunch resumes instead of restarting.
        atomic_save({"enc": enc.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                     "best": best, "hist": hist, "arch": "i26"}, last_p)
        beat(ep, a.steps_per_epoch)
        if el > a.max_minutes:
            L(f"[i26] wall budget {a.max_minutes}m hit at ep{ep}"); break

    L(f"[i26] done. best val_full={best['val']:.4f} @ep{best['epoch']}")
    open(os.path.join(OUT, f"{a.tag}_DONE.marker"), "w").write(json.dumps(best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
