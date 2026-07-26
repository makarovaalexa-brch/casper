r"""concept_distill_train.py -- DISTILLATION-trained concept fold student
(DESIGN_CONCEPT_FOLD_DISTILLATION.md sections 3-9; Step-0 PASSED with qualified-go). Fixes the lossy
concept operator: the plain-NDCG C-lite fold captures only 1-6% of the member-item information; this
distills a PRIVILEGED member-item-fold TEACHER into the same frozen-tower gated fold-to-point student.

ARCHITECTURE (UNCHANGED from concept_fold.ConceptFoldNet; item-safe by construction): frozen i25 tower
  (t2i25_EP4_SNAP.pt) + frozen decoder + gated ZERO-INIT fold-to-point concept module. Empty concepts =>
  bit-identical item behavior (G0 item-tie is mechanical). Tokens are NOT co-trained into the tower.

TEACHER (design section 3, LUPI, TRAIN-ONLY): u_teacher = the FROZEN-TOWER fold of the user's rated
  MEMBER ITEMS of the answered concept(s), composed on top of the current item belief, expressed as a
  DELTA over that base belief in DECODER-SCORE space:
      base   = decode(fold(in_s))                 # current interview belief (items so far)
      teach  = decode(fold(in_s UNION member_items_of_answered_concepts))
      teacher_delta = teach - base                # dense, 18,359 items of signal (the credit fix)
  The revealed concepts' member items are DROPPED from the student's item input (member-drop curriculum)
  so the concept channel must carry INDEPENDENT signal. kc>1 teacher = the JOINT union member-fold.
  Student input = the concept ANSWER(s) ALONE (SEL signed four-band values), NO member ratings.

LOSS (design section 5): L = L_rank (held-like multinomial NLL, kept) + lam * L_distill (KL over the
  ni items between student delta-scores and teacher delta-scores, DELTA form, temperature tau)
  [+ eps * L_coord_l2 (secondary raw-coord regularizer, default off)].
  lam grid {0.1, 1, 10}, val-selected. Answer = SEL signed four-band, IDENTICAL train and eval.

SCHEDULE ARMS (design section 5b):
  S1 JOINT (PRIMARY): L_rank + lam*L_distill throughout.
  S2 DISTILL-THEN-SHARPEN: phase1 distill-dominant (lam high) to convergence -> phase2 NDCG finetune with
    lam LOWERED (not zeroed) + early-stop on capture-proxy erosion.

CONTROLS (design section 7): (a) lam=0 same-budget retrain (isolates the distillation claim -- the
  comparator); (b) shuffled-teacher canary (--shuffle_teacher; must NOT help); (c) SEL answer fixed,
  train==eval; canonical-snap item G0; leak known INTERSECT held == empty.

Usage:
  python src/instrument/concept_distill_train.py --smoke
  python src/instrument/concept_distill_train.py --train --tag cd_s1_l1 --lam 1.0 --arm s1 --epochs 4
  python src/instrument/concept_distill_train.py --eval_capture .cache/instrument/cd_s1_l1_best.pt
  python src/instrument/concept_distill_train.py --poc              # full single-seed POC (detached)
"""
import os
import sys
_FULL = "--full_threads" in sys.argv or "--train" in sys.argv or "--poc" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import level_to_sv, pack_tokens, NLEV, log
from belief_layer import build_concept_dirs
from concept_fold import ConceptFoldNet, fold_items_frozen, build_member_matrix
import concept_fold as CF

assert "load_answerer" not in globals()                          # retired-answerer ban

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
SNAP_DEFAULT = os.path.join(CKPT_DIR, "t2i25_EP4_SNAP.pt")
STEP0_JSON = os.path.join(OUTDIR, "concept_distill_step0.json")
SEED = 4242
M_MAX_DEFAULT = 16                                               # kc curriculum U{1..M_MAX} (design: 1..32)
KC_EVAL = (1, 2, 4)                                              # capture budgets (crater + composition)
KC_CURVE = (1, 2, 4, 8)                                          # deployment concepts-only curve
INTERCEPT_REF = (0.12793773315625726, 0.01923195232020529)
EMPTY = (np.empty(0, np.int64), np.empty(0, np.int64))


# ============================================================================= distillation loss
def kl_delta(stu_delta, teach_delta, tau):
    """KL(softmax(teacher_delta/tau) || softmax(student_delta/tau)) * tau^2, over the ni item axis.
    DELTA form: the logits are the score DELTAs (teacher - base, student - base), not absolute scores."""
    t_log = F.log_softmax(teach_delta / tau, dim=-1)
    s_log = F.log_softmax(stu_delta / tau, dim=-1)
    p_t = t_log.exp()
    return (tau * tau) * (p_t * (t_log - s_log)).sum(-1).mean()


# ============================================================================= curriculum example
def make_example_distill(u, rng, Mm, grate, member_sets, item_mean, prereg, m_max=M_MAX_DEFAULT,
                         p_conc_only=0.35):
    """SIGNED four-band curriculum example WITH the teacher's member items. Mirrors
    concept_fold.make_example_signed exactly, and ALSO returns (mem_s, mem_l) = the union of the revealed
    concepts' rated member items in the pool (dropped from the item input) with their levels -- the
    teacher fold basis. Returns (in_s, in_l, cids, cvals, tgt, mem_s, mem_l) or None."""
    from signed_answers import signed_for_pool, cap_negatives, BAND_REFUSE
    items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(items) < 4 or len(liked) < 2:
        return None
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(items, tg)
    pool_s = items[keep]; pool_l = lv[keep]
    if len(pool_s) < 2:
        return None
    stars = (pool_l.astype(np.float32) + 1.0) / 2.0
    V, B, ans = signed_for_pool(pool_s, stars, item_mean, Mm, grate, prereg)
    cand = np.flatnonzero(ans & (B != BAND_REFUSE))
    if len(cand) == 0:
        return None
    m = int(rng.integers(1, m_max + 1)); m = min(m, len(cand))
    order = cand[np.argsort(-np.abs(V[cand]))]
    top = order[: int(np.ceil(m / 2))]
    rest_pool = np.setdiff1d(cand, top)
    rest = rng.choice(rest_pool, size=min(m - len(top), len(rest_pool)), replace=False) \
        if len(rest_pool) and m > len(top) else np.empty(0, np.int64)
    cids = np.concatenate([top, rest]).astype(int)
    cvals = cap_negatives([float(V[c]) for c in cids])
    support = [np.intersect1d(pool_s, member_sets[int(c)]) for c in cids]
    drop = np.unique(np.concatenate(support)) if support else np.empty(0, np.int64)
    mk = ~np.isin(pool_s, drop)
    rem_s, rem_l = pool_s[mk], pool_l[mk]
    if rng.random() < p_conc_only or len(rem_s) == 0:
        in_s = np.empty(0, np.int64); in_l = np.empty(0, np.int64)
    else:
        k = int(rng.integers(1, min(8, len(rem_s)) + 1))
        pick = rng.choice(len(rem_s), size=k, replace=False)
        in_s, in_l = rem_s[pick], rem_l[pick]
    tgt = np.setdiff1d(tg, in_s)
    if len(tgt) == 0:
        return None
    lmap = {int(s): int(l) for s, l in zip(pool_s, pool_l)}
    mem_s = drop.astype(np.int64)
    mem_l = np.array([lmap[int(s)] for s in mem_s], np.int64)
    if len(mem_s) == 0:                                          # every revealed concept has >=2 members
        return None
    return in_s, in_l, [int(c) for c in cids], [float(v) for v in cvals], tgt, mem_s, mem_l


# ============================================================================= training
def _build_train_stack(args):
    """Load the frozen tower + decoder, genome concepts, member matrix, signed prereg, curriculum users.
    Returns a dict of everything the training loop needs."""
    from train_tower_t2 import (reproduce_partition, build_train_profiles, build_model,
                                 compute_head_mask)
    from signed_answers import compute_prereg
    import metrics as M
    import pandas as pd
    PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
    GENOME = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    head_mask, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    users = build_train_profiles(raw, tr_set, show2id, max_users=(args.max_users or None))
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, _, _, _ = build_model(a, ni, cnt)
    blob = torch.load(args.snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"]); enc.eval()
    for p in list(enc.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    m2s = {int(m): i for i, m in enumerate(usid)}
    g = pd.read_csv(GENOME); g = g[g["relevance"] >= 0.5]; g = g[g["movieId"].isin(m2s)]
    g["sid"] = g["movieId"].map(m2s).astype(np.int64)
    members = {int(t): np.sort(sub["sid"].values) for t, sub in g.groupby("tagId")}
    members = {t: mm for t, mm in members.items() if len(mm) >= 30}
    tags = sorted(members.keys())
    d_c, d_raw, w_c = build_concept_dirs(Wd, members, tags)
    Mm = build_member_matrix(members, tags, ni)
    grate = (cnt @ np.asarray(Mm.todense())) / max(cnt.sum(), 1e-9)
    member_sets = {i: members[t] for i, t in enumerate(tags)}
    prereg, item_mean = compute_prereg(raw, tr_set, show2id, ni, Mm, grate)
    log(f"[cd] tower={os.path.basename(args.snapshot)} ni={ni} concepts={len(tags)} "
        f"users={len(users)} w_val={prereg['w_val']:.3f}")
    return dict(enc=enc, decoder=decoder, Wd=Wd, bd=bd, ni=ni, tags=tags, members=members,
                member_sets=member_sets, Mm=Mm, grate=grate, prereg=prereg, item_mean=item_mean,
                d_c=d_c, users=users)


def _run_epoch(net, opt, S, users, args, lam, ep):
    """One curriculum epoch. lam = distillation weight (0 = no-distill control)."""
    enc, Wd, bd, ni = S["enc"], S["Wd"], S["bd"], S["ni"]
    W0, b0 = Wd.clone(), bd.clone()
    rng = np.random.default_rng(1000 + ep)
    order = rng.permutation(len(users))
    net.train(); run = {"nll": 0.0, "kl": 0.0}; nb = 0; t0 = time.time()
    for st in range(0, len(order), args.batch):
        uix = order[st:st + args.batch]
        exs = [make_example_distill(users[i], rng, S["Mm"], S["grate"], S["member_sets"],
                                    S["item_mean"], S["prereg"], m_max=args.m_max) for i in uix]
        exs = [e for e in exs if e is not None]
        if not exs:
            continue
        Z_items = fold_items_frozen(enc, [(e[0], e[1]) for e in exs])
        teach_seqs = [(np.concatenate([e[0], e[5]]), np.concatenate([e[1], e[6]])) for e in exs]
        Z_teach = fold_items_frozen(enc, teach_seqs)
        if args.shuffle_teacher:                                 # canary: random teacher per student
            perm = rng.permutation(len(exs))
            Z_teach = Z_teach[perm]
        with torch.no_grad():
            base = Z_items @ Wd.T + bd
            teach_delta = (Z_teach @ Wd.T + bd) - base
        cids = [ConceptFoldNet.canonical_order(e[2], e[3])[0] for e in exs]
        cvals = [ConceptFoldNet.canonical_order(e[2], e[3])[1] for e in exs]
        zout = net(Z_items, cids, cvals)
        stu = zout @ Wd.T + bd
        logsm = F.log_softmax(stu, dim=-1)
        tgt = torch.zeros(len(exs), ni)
        for r, e in enumerate(exs):
            tgt[r, e[4]] = 1.0
        nll = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
        loss = nll
        klv = torch.tensor(0.0)
        if lam > 0:
            klv = kl_delta(stu - base, teach_delta, args.tau)
            loss = loss + lam * klv
        if args.eps_l2 > 0:                                      # secondary raw-coord regularizer
            u_t = (Z_teach - Z_items).detach()
            loss = loss + args.eps_l2 * ((zout - Z_items - u_t) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run["nll"] += float(nll); run["kl"] += float(klv); nb += 1
        if nb % 100 == 0:
            log(f"  ep{ep} b{nb} NLL={run['nll']/nb:.4f} KL={run['kl']/nb:.4f} "
                f"cap={float(F.softplus(net.log_cap)):.2f} ({(time.time()-t0)/60:.1f}m)")
    assert torch.equal(Wd, W0) and torch.equal(bd, b0), "frozen decoder drifted"
    return {"nll": run["nll"] / max(nb, 1), "kl": run["kl"] / max(nb, 1), "nb": nb,
            "minutes": (time.time() - t0) / 60}


def train(args, S=None):
    """Train ONE config (tag/lam/arm). Returns best-val checkpoint path. S = prebuilt stack (POC reuse)."""
    if S is None:
        S = _build_train_stack(args)
    net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=args.hidden, conc_init=S["d_c"].numpy())
    log(f"[cd:{args.tag}] arm={args.arm} lam={args.lam} eps={args.eps_l2} tau={args.tau} "
        f"shuffle={args.shuffle_teacher} params={net.n_params():,}")
    ckb = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    hb = os.path.join(OUTDIR, f"{args.tag}.heartbeat")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    vargs = argparse.Namespace(snapshot=args.snapshot, signed=True)
    best = -1.0

    def _val_and_save(ep, lam_now, phase):
        nonlocal best
        vm = CF.quick_val(S["enc"], S["Wd"], S["bd"], net, vargs)
        score = vm["m8_full"]
        open(hb, "w").write(f"{args.tag} {phase} ep{ep+1} m2={vm['m2_full']:.4f} "
                            f"m8={vm['m8_full']:.4f} best={best:.4f} {time.strftime('%H:%M:%S')}\n")
        log(f"[cd:{args.tag} {phase} ep{ep+1}] val concepts-only m2={vm['m2_full']:.4f} "
            f"m8={vm['m8_full']:.4f} (best {best:.4f})")
        if score > best:
            best = score
            torch.save({"net": net.state_dict(), "tags": S["tags"], "epoch": ep + 1, "val": vm,
                        "hidden": args.hidden, "lam": lam_now, "arm": args.arm,
                        "snapshot": os.path.basename(args.snapshot)}, ckb)
            log(f"[cd:{args.tag} {phase} ep{ep+1}] best saved -> {ckb}")
        return vm

    if args.arm == "s1":
        for ep in range(args.epochs):
            st = _run_epoch(net, opt, S, S["users"], args, args.lam, ep)
            log(f"[cd:{args.tag} ep{ep+1}] NLL={st['nll']:.4f} KL={st['kl']:.4f} "
                f"({st['minutes']:.1f}m, {st['nb']} batches)")
            _val_and_save(ep, args.lam, "S1")
    else:  # s2 distill-then-sharpen
        for ep in range(args.epochs):                            # phase 1: distill-dominant
            st = _run_epoch(net, opt, S, S["users"], args, args.lam, ep)
            log(f"[cd:{args.tag} P1 ep{ep+1}] NLL={st['nll']:.4f} KL={st['kl']:.4f} ({st['minutes']:.1f}m)")
            _val_and_save(ep, args.lam, "P1")
        p1_end = best                                            # capture-proxy at end of phase 1
        lam2 = args.lam_phase2
        for ep in range(args.phase2_epochs):                     # phase 2: NDCG sharpen, retained anchor
            st = _run_epoch(net, opt, S, S["users"], args, lam2, args.epochs + ep)
            log(f"[cd:{args.tag} P2 ep{ep+1}] NLL={st['nll']:.4f} KL={st['kl']:.4f} lam={lam2} "
                f"({st['minutes']:.1f}m)")
            vm = CF.quick_val(S["enc"], S["Wd"], S["bd"], net, vargs)
            open(hb, "w").write(f"{args.tag} P2 ep{ep+1} m8={vm['m8_full']:.4f} p1_end={p1_end:.4f}\n")
            log(f"[cd:{args.tag} P2 ep{ep+1}] m8={vm['m8_full']:.4f} (p1_end {p1_end:.4f})")
            if vm["m8_full"] > best:
                best = vm["m8_full"]
                torch.save({"net": net.state_dict(), "tags": S["tags"], "epoch": args.epochs + ep + 1,
                            "val": vm, "hidden": args.hidden, "lam": lam2, "arm": "s2",
                            "snapshot": os.path.basename(args.snapshot)}, ckb)
                log(f"[cd:{args.tag} P2 ep{ep+1}] best saved -> {ckb}")
            elif vm["m8_full"] < p1_end - args.erosion_tol:      # capture-proxy erosion -> stop
                log(f"[cd:{args.tag}] P2 early-stop (erosion {p1_end - vm['m8_full']:.4f} "
                    f"> tol {args.erosion_tol})")
                break
    log(f"[cd:{args.tag}] done; best val m8 = {best:.4f} -> {ckb}")
    return ckb


# ============================================================================= capture eval (headline)
def _tiers(sh):
    mc = np.array([len(sh["members"][t]) for t in sh["tags"]], np.int64)
    b1, b2 = np.percentile(mc, [33.333, 66.667])
    tier_mask = {"fine": mc <= b1, "medium": (mc > b1) & (mc <= b2), "broad": mc > b2}
    pools = {"overall": np.ones(len(mc), bool), "fine": tier_mask["fine"],
             "medium": tier_mask["medium"], "broad": tier_mask["broad"]}
    return pools, [float(b1), float(b2)]


def eval_capture(args, ckpt=None):
    """The headline test: does the TRAINED student beat the tabular floor per tier? Reuses the Step-0
    teacher machinery. Reports student capture vs floor vs teacher-ceiling, G0 tie, deployment curve."""
    import run_battery_phaseA as PA
    from run_battery_phaseA import build_real_ctx, ndcg10_from_scores
    from tradeoff_ledger import build_shared
    from concept_distill_step0 import select_topk, union_member_seqs, score_item_fold
    from signed_answers import cap_negatives
    ckpt = ckpt or args.eval_capture
    blob = torch.load(ckpt, map_location="cpu")
    ctx = build_real_ctx(args.snapshot)
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    sh = build_shared(ctx)
    assert sh.get("signed_available"), "signed prereg cache required (run training first)"
    tags = sh["tags"]
    assert tags == blob["tags"], "concept vocabulary drift between train and eval"
    net = ConceptFoldNet(len(tags), d=ctx.d, h=blob["hidden"], conc_init=None)
    net.load_state_dict(blob["net"]); net.eval()

    # --- G0 item-tie (mechanical, zero-init): empty concepts => bit-exact tower ---
    zr = torch.randn(8, ctx.d)
    g0_ok = bool(torch.equal(net(zr, [[]] * 8, [[]] * 8), zr))
    assert g0_ok, "G0 broken: empty-concept fold is not bit-identical to the tower"

    z0 = fold_items_frozen(ctx.enc, [EMPTY])                     # (1,d) intercept latent
    base_vec = (z0[0] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    row_pos = {r: j for j, r in enumerate(rows)}

    # intercept per user (q0 snap)
    base_full = np.full(len(rows), np.nan); base_tail = np.full(len(rows), np.nan)
    for st in range(0, len(rows), 500):
        ch = rows[st:st + 500]
        S = np.repeat(base_vec[None, :], len(ch), axis=0)
        f, t = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
        base_full[st:st + len(ch)] = f; base_tail[st:st + len(ch)] = t
    icept = {"full": float(np.nanmean(base_full)), "tail": float(np.nanmean(base_tail))}
    snap = (abs(icept["full"] - INTERCEPT_REF[0]) < 5e-3 and abs(icept["tail"] - INTERCEPT_REF[1]) < 5e-3)
    log(f"[cd-eval] intercept {icept['full']:.5f}/{icept['tail']:.5f} snap={snap} G0_tie={g0_ok}")

    Vs = sh["conc_value_signed"]
    memberset = {c: set(sh["members"][tags[c]].tolist()) for c in range(len(tags))}
    pools, tbounds = _tiers(sh)

    def student_ndcg(users, picks, batch=500):
        full = np.full(len(users), np.nan); tail = np.full(len(users), np.nan)
        with torch.no_grad():
            for st in range(0, len(users), batch):
                ch = users[st:st + batch]
                Z0 = z0.repeat(len(ch), 1)
                cids = []; cvals = []
                for j, r in enumerate(ch):
                    cs = picks[st + j]
                    vv = cap_negatives([float(Vs[r, c]) for c in cs])
                    ci, v2 = ConceptFoldNet.canonical_order(list(cs), list(vv))
                    cids.append(ci); cvals.append(v2)
                zo = net(Z0, cids, cvals)
                Sr = (zo @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                f, t = ndcg10_from_scores(Sr, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
                full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
        return full, tail

    def cap(stud, teach, base):
        return (stud - base) / (teach - base) if abs(teach - base) > 1e-9 else float("nan")

    capture = {}
    for pname, pmask in pools.items():
        capture[pname] = {}
        for kc in KC_EVAL:
            users, picks = select_topk(sh, rows, pmask, kc)
            if not users:
                capture[pname][str(kc)] = {"n_users": 0}; continue
            seqs = union_member_seqs(sh, memberset, users, picks)
            tf, tt = score_item_fold(ctx, users, seqs)          # teacher ceiling (matches Step-0)
            sf, st_ = student_ndcg(users, picks)
            idx = np.array([row_pos[r] for r in users])
            i_full = float(np.nanmean(base_full[idx])); i_tail = float(np.nanmean(base_tail[idx]))
            tfm, ttm = float(np.nanmean(tf)), float(np.nanmean(tt))
            sfm, stm = float(np.nanmean(sf)), float(np.nanmean(st_))
            capture[pname][str(kc)] = {
                "n_users": len(users), "intercept_full": i_full, "intercept_tail": i_tail,
                "teacher_full@10": tfm, "teacher_tail@10": ttm,
                "student_full@10": sfm, "student_tail@10": stm,
                "capture_full": cap(sfm, tfm, i_full), "capture_tail": cap(stm, ttm, i_tail)}
            r_ = capture[pname][str(kc)]
            log(f"[cd-cap {pname:>7} kc={kc}] teach={tfm:.4f} stu={sfm:.4f} "
                f"cap_full={r_['capture_full']:.1%} cap_tail={r_['capture_tail']:.1%} n={len(users)}")

    # --- deployment concepts-only curve (diversified selection), kc=1,2,4,8 ---
    sel = sh["sel_div"]
    crows = [r for r in rows if r in sel]
    curve = {"full@10": {}, "tail@10": {}}
    with torch.no_grad():
        for m in KC_CURVE:
            fC = np.full(len(crows), np.nan); tC = np.full(len(crows), np.nan)
            for st in range(0, len(crows), 500):
                ch = crows[st:st + 500]
                Z0 = z0.repeat(len(ch), 1)
                cids = []; cvals = []
                for r in ch:
                    cs, vs = sel[r]
                    ci, v2 = ConceptFoldNet.canonical_order(list(cs[:m]), list(vs[:m]))
                    cids.append(ci); cvals.append(v2)
                zo = net(Z0, cids, cvals)
                Sr = (zo @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                f, t = ndcg10_from_scores(Sr, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
                fC[st:st + len(ch)] = f; tC[st:st + len(ch)] = t
            curve["full@10"][str(m)] = float(np.nanmean(fC)); curve["tail@10"][str(m)] = float(np.nanmean(tC))
    sfull = [curve["full@10"][str(m)] for m in KC_CURVE]
    curve["monotone_full"] = bool(all(sfull[i + 1] >= sfull[i] - 1e-9 for i in range(len(sfull) - 1)))
    curve["opener_q1_gt_intercept"] = bool(curve["full@10"]["1"] > icept["full"])
    log(f"[cd-eval] deployment curve full {sfull} monotone={curve['monotone_full']}")

    out = {"analysis": "concept_distill_capture", "ckpt": os.path.basename(ckpt),
           "arm": blob.get("arm"), "lam": blob.get("lam"), "epoch": blob.get("epoch"),
           "trainable_params": net.n_params(), "n_users_cohort": len(rows),
           "intercept": icept, "q0_snap": bool(snap), "G0_item_tie": g0_ok,
           "tier_bounds_membercount": tbounds, "capture": capture, "deployment_curve": curve}
    # floor comparison (Step-0 tabular capture)
    if os.path.exists(STEP0_JSON):
        s0 = json.load(open(STEP0_JSON))
        floor = {}
        for pname in ("overall", "fine", "medium", "broad"):
            floor[pname] = {}
            for kc in KC_EVAL:
                d = s0["part_A_capture"].get(pname, {}).get(str(kc), {})
                floor[pname][str(kc)] = d.get("capture_full")
        out["floor_capture_full_step0"] = floor
        beats = {}
        for pname in ("overall", "fine", "medium", "broad"):
            beats[pname] = {}
            for kc in KC_EVAL:
                fl = floor[pname].get(str(kc)); stu = capture[pname].get(str(kc), {}).get("capture_full")
                beats[pname][str(kc)] = (None if fl is None or stu is None
                                         else {"student": stu, "floor": fl, "delta": stu - fl,
                                               "beats_floor": bool(stu > fl)})
        out["beats_floor"] = beats
    return out


# ============================================================================= POC orchestration
def _cfg(tag, lam, arm="s1", shuffle=False, **kw):
    d = dict(tag=tag, lam=lam, arm=arm, shuffle_teacher=shuffle)
    d.update(kw)
    return d


def poc(args):
    """Single-seed proof-of-concept: S1 lam-sweep + no-distill control + shuffled canary, eval each,
    pick best, then S2 on the best lam. Writes concept_distill_train.json incrementally (resume-safe:
    a config whose checkpoint already exists AND whose result is recorded is skipped)."""
    os.makedirs(OUTDIR, exist_ok=True)
    respath = os.path.join(OUTDIR, "concept_distill_train.json")
    res = json.load(open(respath)) if os.path.exists(respath) else {
        "analysis": "concept_distill_train", "snapshot": os.path.basename(args.snapshot),
        "seed": SEED, "epochs": args.epochs, "m_max": args.m_max, "tau": args.tau,
        "configs": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    S = _build_train_stack(args)                                 # built ONCE, reused across configs
    done_marker = os.path.join(OUTDIR, "concept_distill_train_DONE.marker")

    def _do(cfg):
        tag = cfg["tag"]
        ckb = os.path.join(CKPT_DIR, f"{tag}_best.pt")
        if tag in res["configs"] and os.path.exists(ckb):
            log(f"[poc] SKIP {tag} (already done)"); return
        cargs = argparse.Namespace(**vars(args)); cargs.__dict__.update(cfg)
        train(cargs, S=S)
        ev = eval_capture(cargs, ckpt=ckb)
        res["configs"][tag] = {"cfg": {k: cfg[k] for k in cfg},
                               "capture": ev["capture"], "beats_floor": ev.get("beats_floor"),
                               "deployment_curve": ev["deployment_curve"], "G0_item_tie": ev["G0_item_tie"],
                               "intercept": ev["intercept"], "val_m8": float(torch.load(ckb)["val"]["m8_full"])}
        res["floor_capture_full_step0"] = ev.get("floor_capture_full_step0")
        json.dump(res, open(respath, "w"), indent=2, default=float)
        log(f"[poc] recorded {tag} -> {respath}")

    # --- S1 lam sweep (PRIMARY) + controls ---
    s1_cfgs = [_cfg(f"cd_s1_l{str(l).replace('.', '')}", l, "s1") for l in (0.1, 1.0, 10.0)]
    s1_cfgs.append(_cfg("cd_ctrl_nodistill", 0.0, "s1"))        # control (a): lam=0
    for cfg in s1_cfgs:
        _do(cfg)

    # pick best S1 distill lam by overall kc4 capture_full (val-cohort selection)
    def _score(tag):
        c = res["configs"].get(tag, {}).get("capture", {}).get("overall", {}).get("4", {})
        return c.get("capture_full", -9.9)
    lam_tags = {0.1: "cd_s1_l01", 1.0: "cd_s1_l10", 10.0: "cd_s1_l100"}
    best_lam = max(lam_tags, key=lambda l: _score(lam_tags[l]))
    res["best_S1_lam"] = best_lam
    log(f"[poc] best S1 lam={best_lam} (overall kc4 capture "
        f"{_score(lam_tags[best_lam]):.1%}); nodistill ctrl {_score('cd_ctrl_nodistill'):.1%}")

    # --- shuffled-teacher canary at the best lam (control b) ---
    _do(_cfg("cd_canary_shuffle", best_lam, "s1", shuffle=True))

    # --- S2 distill-then-sharpen on the best lam ---
    s2 = _cfg("cd_s2", best_lam, "s2")
    _do(s2)

    json.dump(res, open(respath, "w"), indent=2, default=float)
    open(done_marker, "w").write(f"done {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log(f"[poc] COMPLETE -> {respath}; marker {done_marker}")


# ============================================================================= smoke
def _smoke():
    print("[SMOKE] concept_distill_train (synthetic; NOT canonical data)")
    import recvae as R
    from train_tower_t2 import build_model
    rng = np.random.RandomState(0)
    ni, d, nc = 130, 16, 8
    cnt = rng.rand(ni) * 100
    fake = R.RecVAE(24, d, ni)
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=24, t_latent=d, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, _, _, _ = build_model(a, ni, cnt, teacher_override=fake)
    enc.eval()
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    members = {t: np.sort(rng.choice(ni, size=rng.randint(20, 40), replace=False)) for t in range(nc)}
    tags = list(range(nc))
    d_c, _, _ = build_concept_dirs(Wd, members, tags)
    Mm = build_member_matrix(members, tags, ni)
    grate = (cnt @ np.asarray(Mm.todense())) / max(cnt.sum(), 1e-9)
    member_sets = {i: members[t] for i, t in enumerate(tags)}
    item_mean = np.full(ni, 3.5, np.float32)
    prereg = {"w_val": 0.3, "t_like_p60pos": 0.15, "t_neg_absp25": 0.1, "C_NEG": 2.0}

    # 1) kl_delta zero when identical, positive otherwise
    a1 = torch.randn(4, ni)
    assert float(kl_delta(a1, a1.clone(), 1.0)) < 1e-6, "kl_delta not zero on identical"
    assert float(kl_delta(torch.randn(4, ni), torch.randn(4, ni), 1.0)) > 0, "kl_delta not positive"
    print("[SMOKE] kl_delta PASS (0 on identical, >0 otherwise)")

    # 2) make_example_distill: teacher members are exactly the dropped members; leak-free
    hit = 0
    for t in range(300):
        k = rng.randint(14, 60)
        its = rng.choice(ni, size=k, replace=False).astype(np.int64)
        lvs = rng.randint(0, NLEV, size=k).astype(np.int64); lvs[:6] = 8
        u = {"items": its, "levels": lvs, "liked": its[lvs >= 7]}
        e = make_example_distill(u, np.random.default_rng(t), Mm, grate, member_sets, item_mean, prereg)
        if e is None:
            continue
        in_s, in_l, cids, cvals, tgt, mem_s, mem_l = e
        for c in cids:                                           # members dropped from item input
            assert not np.isin(in_s, member_sets[c]).any(), "member leaked into item input"
        assert len(np.intersect1d(in_s, tgt)) == 0, "target leaked into input"
        assert len(np.intersect1d(in_s, mem_s)) == 0, "teacher members overlap item input"
        assert len(mem_s) == len(mem_l) and len(mem_s) > 0, "teacher member seq malformed"
        union = np.union1d(in_s, mem_s)                         # teacher fold basis is well-formed
        assert len(union) == len(in_s) + len(mem_s)
        hit += 1
    assert hit > 30, f"too few usable examples ({hit}/300)"
    print(f"[SMOKE] make_example_distill PASS ({hit}/300 usable; teacher members = dropped members)")

    # 3) one training step reduces loss on a tiny 2-cluster synthetic (concepts carry the signal)
    net = ConceptFoldNet(nc, d=d, h=32, conc_init=d_c.numpy())
    opt = torch.optim.AdamW(net.parameters(), lr=5e-3)
    Z0 = torch.zeros(16, d)
    # teacher_delta = boost cluster A for concept 0, cluster B for concept 1
    tgtA = np.arange(0, 20); tgtB = np.arange(60, 80)
    l0 = None
    for it in range(200):
        cids = [[0] if i % 2 == 0 else [1] for i in range(16)]
        cvals = [[1.0]] * 16
        teach_delta = torch.full((16, ni), -1.0)
        for i in range(16):
            teach_delta[i, tgtA if i % 2 == 0 else tgtB] = 5.0
        base = Z0 @ Wd.T + bd
        zout = net(Z0, cids, cvals)
        stu_delta = (zout @ Wd.T + bd) - base
        loss = kl_delta(stu_delta, teach_delta, 1.0)
        if it == 0:
            l0 = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[SMOKE] distill step PASS: KL {l0:.4f} -> {float(loss):.4f}")
    assert float(loss) < l0 - 0.01, "distillation did not reduce KL"

    # 4) tier split + G0 tie
    sh_like = {"members": members, "tags": tags}
    pools, _ = _tiers(sh_like)
    assert set(pools) == {"overall", "fine", "medium", "broad"}
    zr = torch.randn(5, d)
    assert torch.equal(net(zr, [[]] * 5, [[]] * 5), zr), "G0 tie broken"
    print("[SMOKE] tiers + G0 item-tie PASS")
    print("[SMOKE] COMPLETE")


def _base_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval_capture", default=None, metavar="CKPT")
    ap.add_argument("--poc", action="store_true", help="full single-seed POC (detached)")
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--tag", default="cd_dev")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--arm", choices=("s1", "s2"), default="s1")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--eps_l2", type=float, default=0.0, help="secondary raw-coord L2 weight (0=off)")
    ap.add_argument("--shuffle_teacher", action="store_true", help="canary: random teacher per student")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--phase2_epochs", type=int, default=3, help="S2 sharpen epochs")
    ap.add_argument("--lam_phase2", type=float, default=0.1, help="S2 retained-anchor lam")
    ap.add_argument("--erosion_tol", type=float, default=0.003, help="S2 capture-proxy early-stop tol")
    ap.add_argument("--m_max", type=int, default=M_MAX_DEFAULT)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_users", type=int, default=0, help="dev cap only (0=all; HARD RULE #1)")
    ap.add_argument("--full_threads", action="store_true")
    return ap


def main():
    args = _base_argparser().parse_args()
    if args.smoke:
        _smoke()
    elif args.poc:
        poc(args)
    elif args.train:
        train(args)
    elif args.eval_capture:
        out = eval_capture(args)
        path = os.path.join(OUTDIR, "concept_distill_capture_single.json")
        json.dump(out, open(path, "w"), indent=2, default=float)
        log(f"[out] -> {path}")
    else:
        _base_argparser().error("one of --smoke / --train / --eval_capture / --poc required")


if __name__ == "__main__":
    main()
