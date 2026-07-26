r"""concept_distill_step0.py -- STEP 0 GO/NO-GO for the distillation-trained concept fold
(DESIGN_CONCEPT_FOLD_DISTILLATION.md sections 1 + 1b). ZERO TRAINING. Decides whether the whole
distillation plan is viable BEFORE any train.

STACK: FROZEN i25 tower (t2i25_EP4_SNAP.pt) + decoder; the SIGNED SEL answer simulator (values/bands
from build_shared). NO concept fold net is used -- Step 0 measures the CEILING of the answer channel,
which the trained fold could at best inherit. Canonical ML-25M Liang split, 10k COLD_SEED users,
full+tail NDCG@10, credit-neutral masking, leak-safe (fold-in only; held-out targets excluded).

PART A -- TABULAR-STUDENT GO/NO-GO PROBE (the ceiling of the whole plan):
  TEACHER (privileged, design section 3): for user u and answerable concept c (>=2 rated members in
    fold-in), u_teacher = the FROZEN-TOWER fold of u's RATED MEMBER ITEMS of c (real ratings), expressed
    as a DELTA over the base/empty belief in DECODER-SCORE space (the ni item scores minus base scores).
  TABULAR STUDENT = the closed-form optimum of the distillation loss for the (concept, answer) input:
    bucket users by (concept c, signed SEL band), store the MEAN teacher delta-scores per bucket, and at
    eval predict delta = bucket_mean[(c, band)] -> base + delta -> rank -> NDCG. Bucket means are
    estimated by 2-FOLD CROSS-FITTING within the 10k cohort (a user's prediction uses ONLY the OTHER
    fold's mean -> leak-safe, out-of-sample; a faithful, tractable realization of design section 1's
    "bucket train users", since the built context holds only the 10k cold cohort -- NOTED, not a stop).
  CAPTURE-RATE = (tabular_NDCG - intercept) / (teacher_NDCG - intercept), OVERALL and per coarseness
    tier (fine/medium/broad by member count), at kc=1 (crater budget) and kc=4. kc>1 teacher = JOINT
    member-fold over the UNION of the answered concepts' rated members; kc>1 student = SUM of the per-
    concept bucket means (the additive closed form of a per-(concept,answer) input) -- NOTED.
  WITHIN-CELL teacher variance: mean cos(individual teacher delta, bucket-mean delta) per tier.
  VERDICT (pre-registered): capture >= ~20-30% -> PROCEED; <= ~10% -> STOP (answer channel info-limited,
    fix = richer student input); in between -> report + flag for author call.

PART B -- TEACHER CEILING SWEEP (design section 1b, the true student bar): the member-fold teacher's OWN
  NDCG@10 (full+tail) per tier at kc=1,2,4,8, plus the full-profile fold ceiling (fold ALL fold-in
  items). A perfect student only inherits the teacher.

CONTROLS (HARD RULE 5): SHUFFLED-TEACHER canary (bucket by a random WRONG concept id -> capture ~0);
  q0 == cold intercept (canonical-snap 0.12794/0.01923); leak check known INTERSECT held == empty;
  teacher uses fold-in member ratings only (te excluded by construction).

Usage:
  python src/instrument/concept_distill_step0.py --smoke
  python src/instrument/concept_distill_step0.py [--full_threads]
"""
import os
import sys
_FULL = "--full_threads" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, ndcg10_from_scores, bootstrap_ci,
                                load_genome, OUTDIR, SEED)
from tradeoff_ledger import build_shared, fold_items_enc

assert not hasattr(sys.modules[__name__], "load_answerer")

INTERCEPT_REF = (0.12793773315625726, 0.01923195232020529)   # canonical-snap
KC_TEACHER = (1, 2, 4, 8)                                     # Part B ceiling budgets
KC_STUDENT = (1, 4)                                           # Part A capture budgets
N_BANDS = 3                                                   # like / meh / dislike (refuse never folds)
EMPTY = (np.empty(0, np.int64), np.empty(0, np.int64))


# ============================================================================= scoring helpers
def score_item_fold(ctx, users, seqs, batch=500):
    """Fold item seqs through the FROZEN tower -> per-user full/tail NDCG@10, credit-neutral masking of
    the folded (fold-in) items. Mirrors adaptive_concept_arms.score_users exactly."""
    Wd, bd = ctx.Wd, ctx.bd
    Z = fold_items_enc(ctx.enc, seqs)
    full = np.full(len(users), np.nan); tail = np.full(len(users), np.nan)
    for st in range(0, len(users), batch):
        ch = users[st:st + batch]
        S = (Z[st:st + len(ch)] @ Wd.T + bd).numpy().astype(np.float32)
        mask = ctx.va_tr[ch].copy(); te = ctx.va_te[ch].tolil()
        for j in range(len(ch)):
            s = seqs[st + j][0]
            if len(s):
                extra = sparse.csr_matrix((np.ones(len(s), np.float32), (np.zeros(len(s)), s)),
                                          shape=(1, ctx.ni))
                mask[j] = mask[j] + extra
                for i in np.asarray(s).tolist():
                    te[j, i] = 0
        mask.data[:] = 1.0; te = te.tocsr(); te.eliminate_zeros()
        f, t = ndcg10_from_scores(S, mask.tocsr(), te, ctx.head_mask)
        full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
    return full, tail


def score_student(ctx, base_vec, users, delta, batch=500):
    """Student is in DECODER-SCORE space: scores = base + delta. Concepts reveal no items -> mask = va_tr
    only (honest deployment scoring). Returns per-user full/tail NDCG@10 aligned to `users`."""
    Wd_none = None                                                 # (no tower here)
    full = np.full(len(users), np.nan); tail = np.full(len(users), np.nan)
    for st in range(0, len(users), batch):
        ch = users[st:st + batch]
        S = (base_vec[None, :] + delta[st:st + len(ch)]).astype(np.float32)
        f, t = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
        full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
    return full, tail


# ============================================================================= bucket construction
def build_buckets(ctx, sh, rows, memberset, base_vec, parity, tier_of, decode_chunk=1500):
    """The expensive pass. For every answerable&foldable (user, concept) cell, fold the user's rated
    MEMBER items of that concept through the frozen tower, decode to a delta-score vector, and:
      (1) accumulate 2-fold cross-fit bucket sums SumA/SumB keyed by (concept, band);
      (2) accumulate the within-cell cos(delta, pooled band-mean) per coarseness tier.
    Returns meanA, meanB (C, N_BANDS, ni) float32 cross-fit bucket means, cnt (C, N_BANDS), and the
    within-cell cos aggregates. Encoder fold happens ONCE per concept (Z stored, ni-decode repeated)."""
    C = len(sh["tags"]); ni = ctx.ni
    Wd, bd = ctx.Wd, ctx.bd
    ans = sh["conc_answerable"]; Fs = sh["conc_fold_signed"]; Bs = sh["conc_band_signed"]
    lvl = sh["lvl_lookup"]
    rows_arr = np.asarray(rows)
    SumA = np.zeros((C, N_BANDS, ni), np.float32); SumB = np.zeros((C, N_BANDS, ni), np.float32)
    cntA = np.zeros((C, N_BANDS), np.int64); cntB = np.zeros((C, N_BANDS), np.int64)
    cos_sum = {"fine": 0.0, "medium": 0.0, "broad": 0.0, "overall": 0.0}
    cos_n = {"fine": 0, "medium": 0, "broad": 0, "overall": 0}
    t0 = time.time()
    for c in range(C):
        elig = ans[rows_arr, c] & Fs[rows_arr, c]
        rc = rows_arr[elig]
        if len(rc) < 1:
            continue
        ms = memberset[c]
        seqs = []
        for r in rc:
            d = lvl[r]
            mi = [i for i in d if i in ms]
            seqs.append((np.asarray(mi, np.int64), np.asarray([d[i] for i in mi], np.int64)))
        Z = fold_items_enc(ctx.enc, seqs)                          # (len(rc), d)  -- encoder ONCE
        bands = Bs[rc, c].astype(np.int64)                         # in {0,1,2} (Fs excludes refuse=3)
        par = parity[rc]
        # ---- pass 1: bucket sums (cross-fit) + pooled band means ----
        pooled_mean = {}
        for b in range(N_BANDS):
            selb = bands == b
            if not selb.any():
                continue
            Zb = Z[selb]; parb = par[selb]
            ssum = np.zeros(ni, np.float32)
            for st in range(0, len(Zb), decode_chunk):
                zc = Zb[st:st + decode_chunk]
                dc = (zc @ Wd.T + bd).numpy().astype(np.float32) - base_vec[None, :]
                ssum += dc.sum(0)
                pc = parb[st:st + decode_chunk]
                if (pc == 0).any():
                    SumA[c, b] += dc[pc == 0].sum(0)
                if (pc == 1).any():
                    SumB[c, b] += dc[pc == 1].sum(0)
            cntA[c, b] += int((parb == 0).sum()); cntB[c, b] += int((parb == 1).sum())
            pooled_mean[b] = ssum / max(len(Zb), 1)
        # ---- pass 2: within-cell cos to pooled band mean ----
        tkey = tier_of[c]
        for b in range(N_BANDS):
            if b not in pooled_mean:
                continue
            mbar = pooled_mean[b]; nb = float(np.linalg.norm(mbar))
            if nb < 1e-12:
                continue
            Zb = Z[bands == b]
            for st in range(0, len(Zb), decode_chunk):
                dc = (Zb[st:st + decode_chunk] @ Wd.T + bd).numpy().astype(np.float32) - base_vec[None, :]
                num = dc @ mbar
                den = np.linalg.norm(dc, axis=1) * nb + 1e-12
                cosv = num / den
                s = float(cosv.sum()); k = int(len(cosv))
                cos_sum[tkey] += s; cos_n[tkey] += k
                cos_sum["overall"] += s; cos_n["overall"] += k
        if c % 100 == 0:
            log(f"[buckets] concept {c}/{C} rc={len(rc)} ({(time.time()-t0)/60:.1f}m)")
    cnt = cntA + cntB
    with np.errstate(invalid="ignore", divide="ignore"):
        SumA /= np.maximum(cntA, 1)[:, :, None].astype(np.float32)   # in place -> meanA (halves peak RAM)
        SumB /= np.maximum(cntB, 1)[:, :, None].astype(np.float32)   # in place -> meanB
    meanA, meanB = SumA, SumB
    within = {k: (cos_sum[k] / cos_n[k] if cos_n[k] else float("nan")) for k in cos_sum}
    within_n = cos_n
    log(f"[buckets] done {C} concepts ({(time.time()-t0)/60:.1f}m); within-cell cos {within}")
    return meanA, meanB, cntA, cntB, cnt, within, within_n


# ============================================================================= selection
def select_topk(sh, rows, cand_mask_C, kc):
    """Per user: top-kc answerable&foldable concepts of the candidate pool (cand_mask_C over concepts),
    ranked by |signed value| desc. Returns (users_with_ge_kc, per_user_concept_lists) aligned."""
    ans = sh["conc_answerable"]; Fs = sh["conc_fold_signed"]; Vs = sh["conc_value_signed"]
    users = []; picks = []
    for r in rows:
        elig = np.flatnonzero(ans[r] & Fs[r] & cand_mask_C)
        if len(elig) < kc:
            continue
        order = elig[np.argsort(-np.abs(Vs[r, elig]))][:kc]
        users.append(r); picks.append([int(cc) for cc in order])
    return users, picks


def union_member_seqs(sh, memberset, users, picks):
    """Joint teacher fold basis: union of each user's rated member items across the selected concepts."""
    lvl = sh["lvl_lookup"]
    seqs = []
    for r, cs in zip(users, picks):
        d = lvl[r]; items = set()
        for c in cs:
            items |= (set(d.keys()) & memberset[c])
        it = list(items)
        seqs.append((np.asarray(it, np.int64), np.asarray([d[i] for i in it], np.int64)))
    return seqs


def student_delta(sh, meanA, meanB, parity, users, picks, shuffle_perm=None):
    """Student predicted delta-score for each user = SUM over selected concepts of the CROSS-FIT bucket
    mean for (concept, that user's band). Fold-A users read meanB, fold-B users read meanA. shuffle_perm
    (canary): look up perm[c] instead of c -> a wrong concept's bucket."""
    Bs = sh["conc_band_signed"]; ni = meanA.shape[2]
    delta = np.zeros((len(users), ni), np.float32)
    for j, (r, cs) in enumerate(zip(users, picks)):
        src = meanB if parity[r] == 0 else meanA                   # cross-fit: use the OTHER fold
        acc = np.zeros(ni, np.float32)
        for c in cs:
            b = int(Bs[r, c])
            cc = int(shuffle_perm[c]) if shuffle_perm is not None else c
            acc += src[cc, b]
        delta[j] = acc
    return delta


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--out", default="concept_distill_step0.json")
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    if args.smoke and not sh.get("signed_available"):              # fabricate signed arrays (code path)
        rng = np.random.RandomState(4); C = sh["d_c"].shape[0]
        V = (rng.rand(ctx.n, C).astype(np.float32) * 2 - 1)
        B = np.full((ctx.n, C), 1, np.int8)                        # meh
        B[V > 0.3] = 0; B[V < -0.3] = 2                            # like / dislike
        F = rng.rand(ctx.n, C) > 0.15                              # some refuse
        B[~F] = 3
        sh["conc_value_signed"] = V; sh["conc_band_signed"] = B
        sh["conc_fold_signed"] = B != 3
        sh["conc_answerable"] = rng.rand(ctx.n, C) > 0.3
        sh["signed_available"] = True
    assert sh.get("signed_available"), "signed prereg cache required"

    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    rows_arr = np.asarray(rows)

    # ---- leak check (HARD RULE 5): fold-in INTERSECT held-out == empty ----
    leaks = sum(1 for r in rows if len(np.intersect1d(ctx.allb[r][0], ctx.va_te[r].indices)) > 0)
    assert leaks == 0, f"LEAK: {leaks} users have fold-in/held-out overlap"
    log(f"[leak] fold-in INTERSECT held-out empty for all {len(rows)} users: PASS")

    # ---- base / intercept (q0 snap) ----
    z0 = fold_items_enc(ctx.enc, [EMPTY])
    base_vec = (z0[0] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
    fb = np.full(len(rows), np.nan); tb = np.full(len(rows), np.nan)
    for st in range(0, len(rows), 500):
        ch = rows[st:st + 500]
        S = np.repeat(base_vec[None, :], len(ch), axis=0)
        f, t = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
        fb[st:st + len(ch)] = f; tb[st:st + len(ch)] = t
    intercept = {"full": float(np.nanmean(fb)), "tail": float(np.nanmean(tb))}
    snap_ok = (abs(intercept["full"] - INTERCEPT_REF[0]) < (5e-3 if not args.smoke else 1e9)
               and abs(intercept["tail"] - INTERCEPT_REF[1]) < (5e-3 if not args.smoke else 1e9))
    log(f"[intercept] {intercept['full']:.5f}/{intercept['tail']:.5f} snap={snap_ok}")
    base_full = fb; base_tail = tb                                 # per-user intercept (aligned to rows)
    row_pos = {r: j for j, r in enumerate(rows)}

    # ---- coarseness tiers (member-count terciles) ----
    C = len(sh["tags"])
    member_count = np.array([len(sh["members"][t]) for t in sh["tags"]], np.int64)
    b1, b2 = np.percentile(member_count, [33.333, 66.667])
    tier_mask = {"fine": member_count <= b1,
                 "medium": (member_count > b1) & (member_count <= b2),
                 "broad": member_count > b2}
    tier_of = np.where(member_count <= b1, "fine",
                       np.where(member_count <= b2, "medium", "broad"))
    memberset = {c: set(sh["members"][sh["tags"][c]].tolist()) for c in range(C)}
    parity = (np.arange(ctx.n) % 2).astype(np.int64)               # 2-fold cross-fit assignment

    # ---- BUCKETS (the expensive pass) ----
    meanA, meanB, cntA, cntB, cnt, within, within_n = build_buckets(
        ctx, sh, rows, memberset, base_vec, parity, tier_of)

    # ============================ PART B: teacher ceiling sweep ============================
    log("=== PART B: teacher ceiling sweep ===")
    pools = {"overall": np.ones(C, bool), "fine": tier_mask["fine"],
             "medium": tier_mask["medium"], "broad": tier_mask["broad"]}
    teacher_ceiling = {}
    teacher_cache = {}                                             # (pool,kc) -> (users, full, tail)
    for pname, pmask in pools.items():
        teacher_ceiling[pname] = {}
        for kc in KC_TEACHER:
            users, picks = select_topk(sh, rows, pmask, kc)
            if not users:
                teacher_ceiling[pname][str(kc)] = {"n_users": 0}; continue
            seqs = union_member_seqs(sh, memberset, users, picks)
            tf, tt = score_item_fold(ctx, users, seqs)
            teacher_cache[(pname, kc)] = (users, picks, tf, tt)
            teacher_ceiling[pname][str(kc)] = {
                "n_users": len(users), "teacher_full@10": float(np.nanmean(tf)),
                "teacher_tail@10": float(np.nanmean(tt)),
                "mean_members_folded": float(np.mean([len(s[0]) for s in seqs]))}
            log(f"[teacher {pname:>7} kc={kc}] full={np.nanmean(tf):.4f} tail={np.nanmean(tt):.4f} "
                f"n={len(users)}")
    # full-profile fold ceiling (fold ALL fold-in items)
    allseq = [(np.asarray(list(sh["lvl_lookup"][r].keys()), np.int64),
               np.asarray(list(sh["lvl_lookup"][r].values()), np.int64)) for r in rows]
    ff, ft = score_item_fold(ctx, rows, allseq)
    full_profile_ceiling = {"full@10": float(np.nanmean(ff)), "tail@10": float(np.nanmean(ft)),
                            "mean_items": float(np.mean([len(s[0]) for s in allseq]))}
    log(f"[full-profile ceiling] {full_profile_ceiling['full@10']:.4f}/"
        f"{full_profile_ceiling['tail@10']:.4f} ({full_profile_ceiling['mean_items']:.0f} items)")

    # ============================ PART A: capture-rate ============================
    log("=== PART A: tabular-student capture-rate ===")
    rng = np.random.default_rng(SEED)
    shuffle_perm = rng.permutation(C)                             # shuffled-teacher canary
    capture = {}
    for pname, pmask in pools.items():
        capture[pname] = {}
        for kc in KC_STUDENT:
            if (pname, kc) in teacher_cache:
                users, picks, tf, tt = teacher_cache[(pname, kc)]
            else:
                users, picks = select_topk(sh, rows, pmask, kc)
                if not users:
                    capture[pname][str(kc)] = {"n_users": 0}; continue
                seqs = union_member_seqs(sh, memberset, users, picks)
                tf, tt = score_item_fold(ctx, users, seqs)
            if not users:
                capture[pname][str(kc)] = {"n_users": 0}; continue
            idx = np.array([row_pos[r] for r in users])
            i_full = float(np.nanmean(base_full[idx])); i_tail = float(np.nanmean(base_tail[idx]))
            # student
            d_stu = student_delta(sh, meanA, meanB, parity, users, picks)
            sf, st_ = score_student(ctx, base_vec, users, d_stu)
            # shuffled-teacher canary
            d_shuf = student_delta(sh, meanA, meanB, parity, users, picks, shuffle_perm=shuffle_perm)
            cf, ct = score_student(ctx, base_vec, users, d_shuf)
            tfm = float(np.nanmean(tf)); ttm = float(np.nanmean(tt))
            sfm = float(np.nanmean(sf)); stm = float(np.nanmean(st_))
            cfm = float(np.nanmean(cf)); ctm = float(np.nanmean(ct))

            def cap(stud, teach, base):
                return (stud - base) / (teach - base) if abs(teach - base) > 1e-9 else float("nan")
            capture[pname][str(kc)] = {
                "n_users": len(users),
                "intercept_full": i_full, "intercept_tail": i_tail,
                "teacher_full@10": tfm, "teacher_tail@10": ttm,
                "student_full@10": sfm, "student_tail@10": stm,
                "capture_full": cap(sfm, tfm, i_full), "capture_tail": cap(stm, ttm, i_tail),
                "shuffled_student_full@10": cfm, "shuffled_capture_full": cap(cfm, tfm, i_full),
                "shuffled_student_tail@10": ctm, "shuffled_capture_tail": cap(ctm, ttm, i_tail)}
            r_ = capture[pname][str(kc)]
            log(f"[capture {pname:>7} kc={kc}] teach={tfm:.4f} stu={sfm:.4f} "
                f"cap_full={r_['capture_full']:.1%} cap_tail={r_['capture_tail']:.1%} "
                f"shuf_cap={r_['shuffled_capture_full']:.1%} n={len(users)}")

    # ---- pre-registered verdict (overall, full, at kc=1 and kc=4) ----
    def verdict_for(cap_val):
        if np.isnan(cap_val):
            return "NA"
        if cap_val >= 0.20:
            return "PROCEED"
        if cap_val <= 0.10:
            return "STOP"
        return "FLAG_AUTHOR"
    cap_k1 = capture["overall"]["1"]["capture_full"]
    cap_k4 = capture["overall"]["4"]["capture_full"]
    verdict = {"kc1_overall_capture_full": cap_k1, "kc1_verdict": verdict_for(cap_k1),
               "kc4_overall_capture_full": cap_k4, "kc4_verdict": verdict_for(cap_k4),
               "rule": "capture>=~0.20-0.30 PROCEED; <=~0.10 STOP (answer channel info-limited, fix = "
                       "richer student input: answer + kc-so-far belief context); between = FLAG_AUTHOR"}

    out = {"analysis": "concept_distill_step0", "n_users": len(rows),
           "stack": "FROZEN i25 tower (t2i25_EP4_SNAP.pt) + decoder; SIGNED SEL answer simulator; "
                    "NO concept-fold net (Step 0 = answer-channel CEILING, zero training)",
           "n_concepts": C, "tier_bounds_membercount": [float(b1), float(b2)],
           "bucketing": "(concept, signed SEL band in {like,meh,dislike}); 2-fold cross-fit means "
                        "(user reads the OTHER fold) -> leak-safe out-of-sample; kc>1 student = SUM of "
                        "per-concept bucket means; kc>1 teacher = JOINT union member-fold",
           "selection": "per-user top-kc answerable&foldable concepts of the pool by |signed value|",
           "intercept": intercept, "intercept_ref": list(INTERCEPT_REF),
           "controls": {"q0_canonical_snap_PASS": bool(snap_ok),
                        "leak_foldin_heldout_overlap_users": int(leaks),
                        "shuffled_teacher_canary": "bucket by a random WRONG concept id; capture must ~0 "
                                                   "(see capture[*][*].shuffled_capture_full)"},
           "part_B_teacher_ceiling": teacher_ceiling,
           "full_profile_fold_ceiling": full_profile_ceiling,
           "part_A_capture": capture,
           "within_cell_teacher_cos": within, "within_cell_n": within_n,
           "verdict": verdict,
           "seconds": round(time.time() - t00, 1)}
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, args.out)
    json.dump(out, open(jpath, "w"), indent=2, default=float)
    log(f"[out] -> {jpath}")

    # ---- printed summary ----
    print(f"\n=== CONCEPT-DISTILL STEP 0 (intercept {intercept['full']:.4f}/{intercept['tail']:.4f}; "
          f"snap={snap_ok}; n={len(rows)}; C={C}) ===")
    print("\nPART B -- TEACHER CEILING (full@10 / tail@10, member-fold):")
    print(f"{'pool':>8} | " + " ".join(f"kc={kc:<11}" for kc in KC_TEACHER))
    for pname in ("overall", "fine", "medium", "broad"):
        cells = []
        for kc in KC_TEACHER:
            d = teacher_ceiling[pname].get(str(kc), {})
            cells.append(f"{d.get('teacher_full@10', float('nan')):.4f}/"
                         f"{d.get('teacher_tail@10', float('nan')):.4f}" if d.get("n_users") else "  --  ")
        print(f"{pname:>8} | " + " ".join(f"{c:<14}" for c in cells))
    print(f"full-profile fold ceiling: {full_profile_ceiling['full@10']:.4f}/"
          f"{full_profile_ceiling['tail@10']:.4f} ({full_profile_ceiling['mean_items']:.0f} items)")
    print("\nPART A -- CAPTURE-RATE (tier x kc; full | tail; shuffled-canary in parens):")
    print(f"{'pool':>8} | " + " ".join(f"kc={kc:<22}" for kc in KC_STUDENT))
    for pname in ("overall", "fine", "medium", "broad"):
        cells = []
        for kc in KC_STUDENT:
            d = capture[pname].get(str(kc), {})
            if d.get("n_users"):
                cells.append(f"{d['capture_full']:+.1%}|{d['capture_tail']:+.1%} "
                             f"(shuf {d['shuffled_capture_full']:+.1%})")
            else:
                cells.append("  --  ")
        print(f"{pname:>8} | " + " ".join(f"{c:<25}" for c in cells))
    print(f"\nWITHIN-CELL teacher cos-to-mean: overall={within['overall']:.3f} "
          f"fine={within['fine']:.3f} medium={within['medium']:.3f} broad={within['broad']:.3f}")
    print(f"\nVERDICT  kc=1 overall capture_full={cap_k1:.1%} -> {verdict['kc1_verdict']}  |  "
          f"kc=4 overall capture_full={cap_k4:.1%} -> {verdict['kc4_verdict']}")
    print(f"seconds={out['seconds']}")


if __name__ == "__main__":
    main()
