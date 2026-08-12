r"""SEPARATION (1), ISOLATED: an UNTRUNCATED, UNDISCOUNTED ranking loss vs NDCG@10, measured the
same way, at the same state, over the same 1,000 candidates. Label `pwheldout_root`.

WHY THIS EXISTS. The three-way ablation (`split_criterion_ablation.py`) varies the criterion, but
arm (b) differs from arm (c) in TWO ways at once: it is a pairwise loss rather than a truncated
discounted one (separation 1), AND it is a training loss over the node's users rather than the
measured value of asking (separation 2). This job separates them, by changing ONLY the loss.

WHAT IT DOES. At the ROOT state (empty prefix, all 10,000 canonical validation users), for every one
of the 1,000 candidate questions:
  * fold the candidate's answers through the certified pair exactly as `genv_eval` does -- same
    policy-B mask, same learned decoder bias, same `cap_negatives` over the revealed set, same
    refusal token;
  * from the SAME score matrix compute BOTH
      - full NDCG@10, through `ndcg10_from_scores`, i.e. the identical funnel with the identical
        mask contract; and
      - the HELD-OUT PAIRWISE RANK LOSS in its 0-1 (AUC) form:
            L_u = #{(t, n) : t is a target, n is a non-target in the policy-B pool, s_n >= s_t}
                  / (|T_u| |N_u|)
        which is untruncated (every non-target counts, not only the top 10) and undiscounted (a pair
        at rank 5,000 counts exactly as much as a pair at rank 2). This is the 0-1 limit of the
        pairwise rank loss Hu & Yu's framework lists, evaluated on held-out targets on the FULL
        18,359-item catalogue -- their LOSS under OUR protocol.

So the two numbers come from one fold, one score matrix, one mask, one state, one candidate set. The
only thing that differs is the function that turns scores into a scalar. Any disagreement in how they
rank the 1,000 candidates is attributable to truncation and discounting and to nothing else.

THE FREE CONTRACT CHECK. The banked CELF heap sidecar of the tree's root node holds the exact
measured NDCG@10 marginal gain of every one of those 1,000 candidates, from the real 10.3 h build.
This job recomputes them and asserts agreement to <= 1e-12 before any correlation is reported. If the
NDCG column does not come back bit-for-bit, the AUC column measured beside it is not trustworthy
either and the run stops (memory: `banked-global-greedy-is-a-policy-A-artifact` -- assert the banked
number whenever a new arm nests in an old one).

COST. One tower fold + one full-catalogue rank pass per candidate, 1,000 candidates, 10,000 users,
nothing sampled (HARD RULE 1). Resumable: every candidate is appended to disk as it completes, and a
re-run skips what is already banked (HARD RULE 11).

NOT RUN, AND WHY. The full `pairwise_heldout` TREE -- growing all 16 questions under this loss -- is
the same 10.3 h that the NDCG tree cost, because it needs this fold at every node and every leaf-tail
step. The root is where the whole cohort sits, where every arm is nested, and where the answer is
least confounded, so it is the affordable slice that carries the separation.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import argparse
import numpy as np
import torch

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_mean,
                                    contract_selftest, _env_item_state, CERT_TOWER, CERT_CONC)
from run_battery_phaseA import ndcg10_from_scores
from ndcg_tree import build_pool, leak_assert

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
HEAP_ROOT = os.path.join(OUTDIR, "ndcgtree_heaps", "node_root.json")
LABEL = "pwheldout_root"


def eval_both(E, seq, rows, batch=500):
    """(full NDCG@10, tail NDCG@10, pairwise-rank-loss) per user, from ONE score matrix.

    The NDCG half is `genv_eval`'s code path, line for line, including the mask contract. The
    pairwise half reads the same masked score matrix: a candidate is any item the policy-B mask
    leaves finite, a positive is a held-out target, and the loss is the fraction of (target,
    non-target) pairs that are mis-ordered -- ties counted as errors, which is the conservative
    convention and is stated rather than chosen silently.
    """
    item_ids = [int(i) for ch, i in seq if ch == 0]
    conc_ids = [int(c) for ch, c in seq if ch == 1]
    Zi, _ans = _env_item_state(E, item_ids)
    ctx = E["ctx"]; net = E["net"]
    ridx = np.asarray(rows, np.int64)
    jj = E["posmap"][ridx]
    assert (jj >= 0).all()
    n = len(rows)
    full = np.full(n, np.nan); tail = np.full(n, np.nan); pw = np.full(n, np.nan)
    cvals_all = crefs_all = None
    if conc_ids:
        cc = np.asarray(conc_ids, np.int64)
        ok = E["foldable"][np.ix_(ridx, cc)]
        vv = np.where(ok, E["V"][np.ix_(ridx, cc)], 0.0).astype(np.float64)
        s = -np.minimum(vv, 0.0).sum(1)
        sc = np.minimum(1.0, E["cneg"] / np.maximum(s, 1e-9))
        cvals_all = np.where(vv < 0, vv * sc[:, None], vv)
        crefs_all = (~ok).astype(np.float64)
    with torch.no_grad():
        for st in range(0, n, batch):
            ch = ridx[st:st + batch]; cj = jj[st:st + batch]
            Zb = Zi[torch.from_numpy(cj)]
            if conc_ids:
                B = len(ch)
                z = net(Zb, [conc_ids] * B, cvals_all[st:st + B].tolist(),
                        crefs_all[st:st + B].tolist() if net.use_refusal else None)
            else:
                z = Zb
            S = (z @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            f, t = ndcg10_from_scores(S, E["pool"][ch], ctx.va_te[ch], ctx.head_mask,
                                      contract=(E["contract_tag"], ch))
            full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
            # ---- the untruncated, undiscounted half, off the SAME masked scores.
            # The whole batch is sorted in ONE numpy call (a per-user sort of 18,359 floats, ten
            # thousand times, is the dominant cost otherwise). Masked candidates are given a finite
            # sentinel below every real score rather than -inf, so one sorted row serves every
            # target lookup; policy B guarantees a held-out target is never masked, so a target's
            # score is always a real one and the sentinel can never be counted above it.
            pool_nz = E["pool"][ch].nonzero()
            Sm = S.astype(np.float64)
            n_mask = np.zeros(Sm.shape[0], np.int64)
            np.add.at(n_mask, pool_nz[0], 1)
            SENT = -1e30
            Sm[pool_nz] = SENT
            srt = np.sort(Sm, axis=1)
            te = ctx.va_te[ch]
            ni = Sm.shape[1]
            for u in range(Sm.shape[0]):
                tg = te[u].indices
                npos = len(tg)
                if npos == 0:
                    continue
                nc = ni - int(n_mask[u])                 # candidates the mask leaves alive
                nneg = nc - npos
                if nneg <= 0:
                    continue
                sv = Sm[u, tg]
                assert (sv > SENT).all(), "a held-out target was inside the policy-B mask"
                # #candidates scoring >= s, for each target; TIES COUNT AS ERRORS (conservative)
                ge = ni - np.searchsorted(srt[u], sv, side="left")
                # subtract the targets that are themselves >= s -- they are not negatives
                svs = np.sort(sv)
                ge_pos = npos - np.searchsorted(svs, sv, side="left")
                pw[st + u] = float((ge - ge_pos).sum()) / float(npos * nneg)
    E["n_evals"] += 1
    return full, tail, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 1,000 candidates")
    args = ap.parse_args()

    t00 = time.time()
    OUT = os.path.join(OUTDIR, args.label + ".json")
    PART = os.path.join(OUTDIR, args.label + "_partial.json")
    assert os.path.exists(HEAP_ROOT), "the tree root's banked CELF heap is required: " + HEAP_ROOT
    heap = json.load(open(HEAP_ROOT))
    banked = {}
    for negg, q, upd in heap["heap"]:
        if int(upd) == 0:
            banked[(int(q[0]), int(q[1]))] = -float(negg)
    banked[(int(heap["seq"][0][0]), int(heap["seq"][0][1]))] = float(heap["gains"][0])
    base_banked = float(heap["base"])
    log("[banked] root heap holds exact measured NDCG@10 marginal gains for "
        + str(len(banked)) + " candidates, base " + repr(base_banked))

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "pwheldout(val)")
    rows = E["rows"]
    contract_selftest(E, rows); leak_assert(E, rows, "validation")
    assert len(rows) == 10000
    set_active_rows(E, rows); E["memo"] = {}
    pool, _ip, _cp = build_pool(E, 500, 500)
    assert len(pool) == 1000

    f0, t0_, p0 = eval_both(E, [], rows)
    base_full = float(np.nanmean(f0)); base_pw = float(np.nanmean(p0))
    log("[base] m=0: full NDCG@10 " + format(base_full, ".12f") + " | pairwise held-out rank loss "
        + format(base_pw, ".8f") + " (0 = every target above every non-target)")
    assert abs(base_full - base_banked) < 1e-12, (
        "the m=0 full NDCG " + repr(base_full) + " does not reproduce the banked root heap base "
        + repr(base_banked) + " -- different ruler, STOP")
    log("[gate] PASS: the m=0 intercept reproduces the banked root heap base to <= 1e-12")

    done = {}
    if os.path.exists(PART):
        blob = json.load(open(PART))
        if blob.get("base_full") == base_full:
            done = {tuple(int(x) for x in k.split(",")): v for k, v in blob["cand"].items()}
            log("[resume] " + str(len(done)) + " candidates already banked")
    todo = [q for q in pool if tuple(q) not in done]
    if args.limit:
        todo = todo[:args.limit]
    log("[run] " + str(len(todo)) + " candidate(s) to evaluate at the ROOT state, 10,000 users, "
        "full 18,359-item catalogue, nothing sampled")

    dev_max = 0.0
    for i, q in enumerate(todo):
        f, _t, p = eval_both(E, [tuple(q)], rows)
        gf = float(np.nanmean(f)) - base_full
        gp = base_pw - float(np.nanmean(p))            # a GAIN: reduction in the pairwise loss
        rec = {"ndcg_gain": gf, "pairwise_gain": gp,
               "ndcg_abs": float(np.nanmean(f)), "pairwise_abs": float(np.nanmean(p))}
        bk = banked.get(tuple(q))
        if bk is not None:
            rec["banked_ndcg_gain"] = bk
            rec["banked_abs_dev"] = abs(gf - bk)
            dev_max = max(dev_max, rec["banked_abs_dev"])
            assert rec["banked_abs_dev"] < 1e-12, (
                "candidate " + str(q) + ": recomputed NDCG gain " + repr(gf) + " != banked "
                + repr(bk) + ". The measured column does not reproduce the tree's own build -- STOP.")
        done[tuple(q)] = rec
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t00
            log("[run] " + str(i + 1) + "/" + str(len(todo)) + " (" + format(el / 60, ".1f")
                + "m, " + format(el / (i + 1), ".2f") + "s/cand); banked-NDCG max dev "
                + format(dev_max, ".1e"))
            with open(PART + ".tmp", "w") as fh:
                json.dump({"base_full": base_full, "base_pw": base_pw,
                           "cand": {str(k[0]) + "," + str(k[1]): v for k, v in done.items()}}, fh)
            os.replace(PART + ".tmp", PART)

    ks = sorted(done)
    gn = np.asarray([done[k]["ndcg_gain"] for k in ks])
    gp = np.asarray([done[k]["pairwise_gain"] for k in ks])
    from scipy.stats import spearmanr, kendalltau
    an, ap_ = int(np.argmax(gn)), int(np.argmax(gp))
    res = {"analysis": "separation_1_untruncated_pairwise_vs_ndcg10_at_the_root",
           "label": args.label, "citable": len(done) == 1000,
           "n_candidates": len(ks), "n_users": len(rows),
           "state": "ROOT (empty prefix)", "mask_policy": "B",
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "base_full_ndcg10": base_full, "base_pairwise_rank_loss": base_pw,
           "banked_ndcg_max_abs_dev": dev_max,
           "spearman_ndcg_vs_pairwise": float(spearmanr(gn, gp).correlation),
           "kendall_ndcg_vs_pairwise": float(kendalltau(gn, gp).correlation),
           "ndcg_argmax": list(ks[an]), "pairwise_argmax": list(ks[ap_]),
           "argmax_agrees": bool(ks[an] == ks[ap_]),
           "ndcg_gain_of_pairwise_argmax": float(gn[ap_]),
           "best_ndcg_gain": float(gn[an]),
           "regret_of_following_the_pairwise_loss": float(gn[an] - gn[ap_]),
           "expected_regret_of_a_uniform_random_pick": float(gn[an] - gn.mean()),
           "rank_of_ndcg_argmax_under_pairwise": int(1 + (gp > gp[an]).sum()),
           "top10_recall": float(len(set(np.argsort(-gp)[:10].tolist())
                                     & set(np.argsort(-gn)[:10].tolist())) / 10.0),
           "per_candidate": {str(k[0]) + "," + str(k[1]): done[k] for k in ks},
           "seconds": round(time.time() - t00, 1)}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    log("=" * 110)
    log("[result] Spearman(NDCG@10 gain, untruncated pairwise gain) over " + str(len(ks))
        + " candidates = " + format(res["spearman_ndcg_vs_pairwise"], "+.4f")
        + " (Kendall " + format(res["kendall_ndcg_vs_pairwise"], "+.4f") + ")")
    log("[result] NDCG argmax " + str(res["ndcg_argmax"]) + " vs pairwise argmax "
        + str(res["pairwise_argmax"]) + " -> agree=" + str(res["argmax_agrees"]))
    log("[result] following the untruncated pairwise loss costs "
        + format(res["regret_of_following_the_pairwise_loss"], "+.6f")
        + " of measured NDCG@10 at the root; a uniform random pick costs "
        + format(res["expected_regret_of_a_uniform_random_pick"], "+.6f"))
    log("[result] top-10 recall " + format(res["top10_recall"], ".2f")
        + "; the NDCG argmax sits at rank " + str(res["rank_of_ndcg_argmax_under_pairwise"])
        + " under the pairwise loss")
    log("[done] wrote " + os.path.basename(OUT) + " in " + format((time.time() - t00) / 60, ".1f")
        + "m")
    return res


if __name__ == "__main__":
    main()
