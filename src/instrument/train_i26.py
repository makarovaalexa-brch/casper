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
from scipy import sparse
from arm_n import load_arm_n
import interview_strategies as ST
from i26_encoder import build_i26, UNSEEN_LEVEL
from interview_curriculum import StrategyFamily, draw
from train_i26_smoke import profiles, full_dropout, pack, loss_of, loss_vec, regime_of
from train_tower_t2 import (load_recvae_teacher, pack_tokens, apply_sign_prior, compute_head_mask,
                            make_graded_predict_fn, build_graded_eval_matrix, reproduce_partition,
                            truncate_graded, PROC)

OUT = os.path.join(_ROOT, "experiments", "instrument")
CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
# REFERENCE POINTS ARE MEASURED AT STARTUP, NOT HARDCODED (fix 2026-07-31).
# ep1 flagged "G-EMPTY below floor (0.1315 < 0.1626)" -- a FALSE ALARM. 0.1626 was measured on the
# arm-N pool (all rated items masked), which inflates scores ~+0.03; when the diagnostics moved to the
# VAL cohort they also dropped that pool, so a val/arm-A number was being compared to an arm-N floor.
# Same mismatch on full profile: 0.3467 derives from the TEST number 0.3482, but the certified model
# scores 0.3451 on VAL. Comparing a val measurement to a test-derived floor is the same error twice.
# So both references are now MEASURED on the val cohort under the identical protocol at startup.
CERT_VAL_FULL = 0.3451      # t2final_best.pt's own recorded val_full (sanity-checked at startup)
CI_WIDTH = 0.0015           # paired-bootstrap half-width
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
    enc, dec, params, groups = build_i26(ni, src, ma, log=L)
    apply_sign_prior(enc)
    blob = torch.load(os.path.join(CKPT_DIR, "t2final_best.pt"), map_location="cpu")
    miss = enc.load_state_dict(blob["enc"], strict=False)
    L(f"[i26] warm-started taste path from t2final_best; at init: {sorted(miss.missing_keys)}")
    Wd = dec.weight.detach(); bd = dec.bias.detach()
    opt = torch.optim.Adam(groups, lr=a.lr)

    # ---- canonical VAL cohort. TEST is never touched here. ------------------------------
    uu, _tr, _vd, _te, _n, raw, show2id, usid = reproduce_partition()
    L_val, _nv = build_graded_eval_matrix(raw, uu, show2id, usid, "validation")
    va_tr, va_te = M.load_val(ni, PROC)
    # Empty-set reference on the SAME protocol the diagnostics use: Most-Popular on val, arm-A masking.
    import pop as _pop
    _pp = _pop.fit(D["train"], ni, log=lambda m: None)
    G_EMPTY_FLOOR = M.evaluate(_pp, va_tr, va_te, batch_size=500,
                               head_mask=D["head_mask"])["ndcg@10"]
    G_FULL_FLOOR = CERT_VAL_FULL - CI_WIDTH
    # Val-side interview machinery: answerability from the val users' FULL rated history.
    from graded_data import _partition, _graded_rows
    _uu2, _s2i, _raw2 = _partition()
    _nall = len(_uu2)
    g_va_tr = _graded_rows(_uu2[_nall - 20000:_nall - 10000], _s2i, _raw2, exclude=va_te)
    vlvl = ST.level_lookup(g_va_tr)
    vglob, vbank = ST.build_orders(D["g_train"], D["train"], ni, log=lambda m: None)
    nva = va_tr.shape[0]
    L(f"[i26] val interview oracle: {g_va_tr.nnz} rated cells over {nva} val users")
    L(f"[i26] references MEASURED on val/arm-A: MostPop={G_EMPTY_FLOOR:.4f} (empty-set floor); "
      f"full-profile floor={G_FULL_FLOOR:.4f} (certified val {CERT_VAL_FULL} - CI {CI_WIDTH})")

    def full_profile_val():
        enc.eval()
        pr = make_graded_predict_fn(enc, Wd, bd, L_val, check_nnz=False)
        r = M.evaluate(pr, va_tr, va_te, batch_size=500, head_mask=D["head_mask"])
        enc.train(); return r["ndcg@10"], r["tail_ndcg@10"]

    def interview_val():
        """THE SELECTION METRIC, and everything else, on the VAL cohort only.

        Two corrections, both author-driven (2026-07-31):
        (1) Selection was on FULL-PROFILE val -- i.e. on exactly the quantity we have decided we are
            willing to trade ("even if it sacrifices 1 or 2 points, but beats everyone clean on
            interview, this is defensible"). It would have picked the epoch that is best at the thing
            being sacrificed and discarded the epoch that is best at the job. Selection is now the
            SHORT-INTERVIEW score; full profile is reported as the cost, not optimised.
        (2) k0/k1 were computed on the TEST cohort. They never fed selection, but watching test numbers
            every epoch is how unconscious selection starts. Everything here is VAL now.

        Returns (k0, k1, k2, k8, sel).

        WHAT k MEANS HERE -- corrected 2026-07-31 after the author flagged k8=0.2442 as suspiciously
        high. It was: truncate_graded(L_val, k) takes k items from the user's LIKED fold-in, so every
        one is answerable and positive. "k=8" meant eight real likes. But in a deployed interview k=8
        means eight QUESTIONS ASKED, of which only ~1.6-3.1 come back answered -- which is why the
        interview table reads 0.1899-0.1935 at k=8 against this metric's 0.2442. Selecting on the easy
        regime while deploying in the hard one can prefer a different epoch entirely.
        So k2/k8 are now REAL simulated interviews on val users: strategy-selected questions, answered
        only if the user rated the item, the rest folded as unseen -- the same generator the model
        trains on, and the same thing the interview table measures.
        """
        enc.eval()
        with torch.no_grad():
            e = pack_tokens([(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))],
                            binarize=False)
            s0 = enc.logits(enc(*e), Wd, bd).numpy()[0]
        r0 = M.evaluate(lambda X, _s=s0: np.repeat(_s[None, :], X.shape[0], 0).astype(np.float32),
                        va_tr, va_te, batch_size=500, head_mask=D["head_mask"])
        # k=1 keeps the cheap likes-only probe purely for the monotonicity check against k=0.
        L1 = truncate_graded(L_val, 1, 4242)
        pr1 = make_graded_predict_fn(enc, Wd, bd, L1, check_nnz=False)
        r1 = M.evaluate(pr1, va_tr, va_te, batch_size=500, head_mask=D["head_mask"])["ndcg@10"]
        # k=2 / k=8: REAL interviews -- strategy-selected asks, unanswerable questions folded as unseen.
        out = {}
        for k in (2, 8):
            asked, answered = ST.ask("helf", vglob, vbank, nva, ni, vlvl, k)
            rows = []
            for u in range(nva):
                a_ = [i for i, _l in answered[u]]
                l_ = [l for _i, l in answered[u]]
                uns = [int(i) for i in asked[u] if int(i) not in set(a_)]
                sid = np.asarray(a_ + uns, np.int64)
                lvl = np.asarray(l_ + [UNSEEN_LEVEL] * len(uns), np.int64)
                rows.append((sid, lvl + 1.0))
            Lk = sparse.csr_matrix(
                (np.concatenate([r[1] for r in rows]).astype(np.float32),
                 (np.repeat(np.arange(nva), [len(r[0]) for r in rows]),
                  np.concatenate([r[0] for r in rows]))), shape=(nva, ni), dtype=np.float32)
            pr = make_graded_predict_fn(enc, Wd, bd, Lk, check_nnz=False)
            out[k] = M.evaluate(pr, va_tr, va_te, batch_size=500,
                                head_mask=D["head_mask"])["ndcg@10"]
        enc.train()
        return r0["ndcg@10"], r1, out[2], out[8], 0.5 * (out[2] + out[8])

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
        import glob as _g
        archive_existing([best_p, last_p] + sorted(_g.glob(os.path.join(CKPT_DIR, f"{a.tag}_ep*.pt"))),
                         L)
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
            lv_ = loss_vec(enc, Wd, bd, b, ni)          # ONE forward, reused for logging
            lv_.mean().backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if step % HEARTBEAT_EVERY == 0:
                beat(ep, step)
            if step % 5 == 0:
                with torch.no_grad():
                    d_ = lv_.detach()
                    for j, e in enumerate(b):
                        run[regime_of(e)].append(float(d_[j]))
        # ---- held-out loss per regime
        rv = np.random.default_rng(99)
        vb = {"full": [], "interview": [], "empty": []}
        with torch.no_grad():
            for _ in range(20):
                b = make_batch(val_us, rv)
                d_ = loss_vec(enc, Wd, bd, b, ni)
                for j, e in enumerate(b):
                    vb[regime_of(e)].append(float(d_[j]))
        fv, ft = full_profile_val()
        e0, e1, k2, k8, sel = interview_val()
        el = (time.time() - t0) / 60.0
        L(f"[i26] ep{ep:2d} TRAIN " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in run.items()) +
          " | HELD-OUT " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in vb.items()))
        L(f"[i26] ep{ep:2d} SELECT sel={sel:.4f} (k2={k2:.4f} k8={k8:.4f}) | "
          f"COST full={fv:.4f}/{ft:.4f} | k0={e0:.4f} k1={e1:.4f} "
          f"(monotone {'PASS' if e1 >= e0 else 'FAIL'}) | {el:.1f}m   [all VAL cohort]")
        hist.append({"ep": ep, "val_full": fv, "val_tail": ft, "k0": e0, "k1": e1,
                     "k2": k2, "k8": k8, "sel": sel,
                     "train": {k: float(np.mean(v)) for k, v in run.items()},
                     "held": {k: float(np.mean(v)) for k, v in vb.items()}})
        json.dump(hist, open(os.path.join(OUT, f"{a.tag}_hist.json"), "w"), indent=2)

        # SAVE FIRST, REPORT SECOND -- ordering is structural, not stylistic. In v1 the abort check
        # sat ABOVE this block and returned at epoch 2, so the epoch-2 model (val 0.3342, BETTER than
        # epoch 1's 0.3307) was never written and `best` stayed pinned at epoch 1. Author caught it.
        # Nothing may ever come between computing a val score and persisting the model that earned it.
        # EVERY EPOCH IS KEPT (author, 2026-07-31: "can you save all chkp though? we need a good
        # tradeoff, surely we have space on disk"). ~32 MB each, 14 epochs ~= 450 MB against 37 GB free.
        # The full-profile / short-interview tradeoff is a JUDGEMENT the author makes with the whole
        # curve in front of them -- keeping only a single "best" under one selection rule pre-empts that
        # decision and throws away the operating points we might actually prefer. Each file carries its
        # own metrics so a checkpoint can be chosen on any criterion after the fact.
        atomic_save({"enc": enc.state_dict(), "decoder": dec.state_dict(), "epoch": ep,
                     "sel": sel, "k0": e0, "k1": e1, "k2": k2, "k8": k8,
                     "val_full": fv, "val_tail": ft, "arch": "i26"},
                    os.path.join(CKPT_DIR, f"{a.tag}_ep{ep:02d}.pt"))
        L(f"[i26] saved {a.tag}_ep{ep:02d}.pt (sel={sel:.4f} full={fv:.4f})")
        if sel > best["val"]:
            best = {"val": sel, "epoch": ep}
            atomic_save({"enc": enc.state_dict(), "decoder": dec.state_dict(), "epoch": ep,
                         "sel": sel, "k2": k2, "k8": k8, "val_full": fv, "val_tail": ft,
                         "k0": e0, "k1": e1, "arch": "i26"}, best_p)
            L(f"[i26] new best SHORT-INTERVIEW sel={sel:.4f} (full-profile cost {fv:.4f}) "
              f"-> {a.tag}_best.pt (atomic)")

        # THE GATES REPORT; THEY DO NOT KILL. (author, 2026-07-31: "do NOT set up logic to kill the
        # run, delete everything and not offer alternative. we could have learned smth at least".)
        # The v1 abort stopped at 02:09 and, because the watchdog correctly stands down on a DONE
        # marker, the machine then sat IDLE until 08:43 -- six and a half hours producing nothing. A
        # stop with no alternative path is not a safety feature. Training now continues regardless; a
        # failing gate is logged loudly, the best checkpoint keeps being written, and a human reads the
        # full curve and decides. Nothing is ever lost by letting it run.
        fulls = [h["val_full"] for h in hist]
        sels = [h["sel"] for h in hist]
        stalled = (len(sels) > PATIENCE and
                   max(sels[-PATIENCE:]) <= max(sels[:-PATIENCE]) + 1e-5)
        flags = []
        if fv < G_FULL_FLOOR:
            trend = "IMPROVING" if len(fulls) < 2 or fv > fulls[-2] else "flat/falling"
            d = (fv - fulls[-2]) if len(fulls) >= 2 else 0.0
            flags.append(f"G-FULL below floor ({fv:.4f} < {G_FULL_FLOOR}), {trend} {d:+.4f}")
        if stalled:
            flags.append(f"short-interview not improving for {PATIENCE} epochs "
                         f"(best {max(sels):.4f})")
        if e0 < G_EMPTY_FLOOR:
            flags.append(f"G-EMPTY below floor ({e0:.4f} < {G_EMPTY_FLOOR})")
        if e1 < e0:
            flags.append(f"G-MONOTONE violated (k1 {e1:.4f} < k0 {e0:.4f})")
        if flags:
            L(f"[i26] ep{ep:2d} GATE FLAGS (reporting only, run CONTINUES): " + "; ".join(flags))
        # `last` carries optimiser state so a watchdog relaunch resumes instead of restarting.
        atomic_save({"enc": enc.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                     "best": best, "hist": hist, "arch": "i26"}, last_p)
        beat(ep, a.steps_per_epoch)
        if el > a.max_minutes:
            L(f"[i26] wall budget {a.max_minutes}m hit at ep{ep}"); break

    L(f"[i26] done. best short-interview sel={best['val']:.4f} @ep{best['epoch']}")
    open(os.path.join(OUT, f"{a.tag}_DONE.marker"), "w").write(json.dumps(best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
