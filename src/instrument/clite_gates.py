r"""clite_gates.py -- TRAINED-MODULE concept gates for Arm C-lite (S3 of the overnight queue,
author directive 2026-07-24). Eval-only, on cfold_best.pt over the frozen ep4 tower, VAL cohort.

GATES (concept-answer analogues of the battery's G3a/G6/G8, applied to the TRAINED fold -- the Phase-A
passes certified the TOWER's item channel; a trained concept operator needs its own sign/existential/
coherence receipts):
  G3a-c FLIP        negate every concept answer's value (+v -> -v) through the net: dNDCG < 0 CI-clean.
  G6-c  EXISTENTIAL wrong-user (derangement: fold user r-1's concepts; must NOT beat the intercept),
                    shuffled values (within-user permutation across concepts; true - shuffled reported),
                    placebo-constant (all values = 1.0, membership kept; true - placebo reported),
                    DUPLICATE x2/x3 (same concept answer repeated; |dNDCG| must stay ~0 -- information,
                    not cardinality; the recursive fold could over-count -- this is the receipt it
                    does not).
  G8-c  ORDER       permute the RAW answer order K=5 through the net's sequential recursion (bypassing
                    canonical_order): max |dz| and dNDCG spread reported; the canonical-order path is
                    invariant BY CONSTRUCTION (sorted) -- both facts recorded honestly.

Evidence per user: top-4 diversified SEL concepts (the ledger's default axis), k=0 (concepts-only,
the channel under certification). Stats: paired per-user bootstrap CIs. Output:
experiments/battery/clite_gates.json.

Usage:
  python src/instrument/clite_gates.py --smoke
  python src/instrument/clite_gates.py [--ckpt .cache/instrument/cfold_best.pt] [--full_threads]
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

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, ndcg10_from_scores, bootstrap_ci,
                                load_genome, OUTDIR, SEED)
from train_tower_t2 import log
from belief_layer import build_concept_dirs
from concepts_only_curve import sel_top_concepts, diversify_sel
from concept_fold import ConceptFoldNet

assert not hasattr(sys.modules[__name__], "load_answerer")

M_EV = 4          # evidence size per user (top-4 div concepts)


def score_rows(net, ctx, rows, conc_lists, order_raw=False):
    """Fold each user's concept answers (k=0) through the net -> per-user full NDCG@10.
    order_raw=True feeds the lists AS GIVEN (bypasses canonical ordering) for the G8-c probe."""
    full = np.full(ctx.n, np.nan)
    Z_all = torch.zeros(len(rows), ctx.d)
    with torch.no_grad():
        for st in range(0, len(rows), 500):
            chunk = rows[st:st + 500]
            cids = []; cvals = []
            for r in chunk:
                cl = conc_lists[r]
                if order_raw:
                    ci = [c for c, _ in cl]; vv = [v for _, v in cl]
                else:
                    ci, vv = ConceptFoldNet.canonical_order([c for c, _ in cl], [v for _, v in cl])
                cids.append(ci); cvals.append(vv)
            Z = net(torch.zeros(len(chunk), ctx.d), cids, cvals)
            Z_all[st:st + len(chunk)] = Z
            S = (Z @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            f, _ = ndcg10_from_scores(S, ctx.va_tr[chunk], ctx.va_te[chunk], None)
            full[np.asarray(chunk)] = f
    return full, Z_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, ".cache", "instrument", "cfold_best.pt"))
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, ".cache", "instrument",
                                                       "t2i25_EP4_SNAP.pt"))
    ap.add_argument("--signed", action="store_true",
                    help="SIGNED evidence (four-band values incl. real dislike-band flips); "
                         "requires the signed prereg cache")
    ap.add_argument("--out_tag", default="", help="output JSON suffix")
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    members = load_genome(ctx)
    tags = sorted(members.keys())
    d_c, d_raw, w_c = build_concept_dirs(ctx.Wd, members, tags)
    if args.smoke:
        net = ConceptFoldNet(len(tags), d=ctx.d, h=32, conc_init=d_c.numpy())
        with torch.no_grad():                                 # non-zero so the gates have signal
            for p in net.mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.05)
    else:
        blob = torch.load(args.ckpt, map_location="cpu")
        assert blob["tags"] == tags, "concept vocabulary drift"
        net = ConceptFoldNet(len(tags), d=ctx.d, h=blob["hidden"])
        net.load_state_dict(blob["net"])
    net.eval()
    if args.signed:
        # SIGNED evidence: top-4 by |v| among answerable non-refuse; C_NEG cap; REAL dislike-band
        # members in the evidence -> the flip test flips both directions
        from signed_answers import load_prereg, signed_values, cap_negatives, BAND_REFUSE
        from concept_fold import build_member_matrix
        import numpy as _np
        prereg, item_mean = load_prereg()
        Mm = build_member_matrix(members, tags, ctx.ni)
        pexp = (ctx.cnt @ _np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
        items_l = [_np.asarray(s, _np.int64) for s, l in ctx.allb]
        stars_l = [((_np.asarray(l, _np.float64) + 1) / 2).astype(_np.float32)
                   for s, l in ctx.allb]
        V, F, B, ans = signed_values(items_l, stars_l, item_mean, Mm, pexp, ctx.ni, prereg,
                                     apply_neg_cap=False)
        sel = {}
        for r in range(ctx.n):
            cand = _np.flatnonzero(ans[r] & (B[r] != BAND_REFUSE))
            if len(cand) >= 2:
                o = cand[_np.argsort(-_np.abs(V[r, cand]))][:M_EV]
                sel[r] = (o.astype(_np.int64),
                          _np.asarray(cap_negatives([float(V[r, c]) for c in o]), _np.float32))
        rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel]
        conc_true = {r: list(zip([int(c) for c in sel[r][0]],
                                 [float(v) for v in sel[r][1]])) for r in rows}
        n_dis = sum(1 for r in rows for _, v in conc_true[r] if v < 0)
        log(f"[gates] SIGNED evidence: {n_dis} dislike-band answers in the evidence "
            f"({n_dis / max(sum(len(conc_true[r]) for r in rows), 1):.1%} of folds)")
    else:
        sel40 = sel_top_concepts(ctx, members, tags, 40)
        sel = diversify_sel(sel40, d_c, m_max=8)
        rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel
                and len(sel[r][0]) >= 2]
        conc_true = {r: list(zip([int(c) for c in sel[r][0][:M_EV]],
                                 [float(v) for v in sel[r][1][:M_EV]])) for r in rows}
    log(f"[gates] cohort {len(rows)} val users, evidence = top-{M_EV} div concepts, k=0")
    # intercept + true
    empty = {r: [] for r in rows}
    f_int, _ = score_rows(net, ctx, rows, empty)
    f_true, Z_true = score_rows(net, ctx, rows, conc_true)
    m_int = float(np.nanmean(f_int[rows])); m_true = float(np.nanmean(f_true[rows]))
    out = {"gates": "clite_concept_gates", "signed_evidence": bool(args.signed),
           "ckpt": os.path.basename(args.ckpt),
           "n_users": len(rows), "evidence": f"top-{M_EV} diversified SEL concepts, k=0",
           "intercept_full@10": m_int, "true_full@10": m_true}
    # ---- G3a-c FLIP ----
    conc_flip = {r: [(c, -v) for c, v in conc_true[r]] for r in rows}
    f_flip, _ = score_rows(net, ctx, rows, conc_flip)
    d = bootstrap_ci(f_flip[rows] - f_true[rows])
    out["g3a_flip"] = {"flipped_full@10": float(np.nanmean(f_flip[rows])), "delta": d[0],
                       "ci95": d[1], "PASS": bool(d[0] < 0 and d[1][1] < 0),
                       "note": "net trained on v in [0.25,1]; -v = negated candidate point"}
    # ---- G6-c existential ----
    rlist = list(rows)
    conc_wrong = {r: conc_true[rlist[(i - 1) % len(rlist)]] for i, r in enumerate(rlist)}
    f_wrong, _ = score_rows(net, ctx, rows, conc_wrong)
    dw = bootstrap_ci(f_wrong[rows] - f_int[rows])
    rng = np.random.default_rng(SEED)
    conc_shuf = {}
    for r in rows:
        vs = [v for _, v in conc_true[r]]
        p = rng.permutation(len(vs))
        conc_shuf[r] = [(c, vs[p[i]]) for i, (c, _) in enumerate(conc_true[r])]
    f_shuf, _ = score_rows(net, ctx, rows, conc_shuf)
    ds = bootstrap_ci(f_true[rows] - f_shuf[rows])
    conc_plac = {r: [(c, 1.0) for c, _ in conc_true[r]] for r in rows}
    f_plac, _ = score_rows(net, ctx, rows, conc_plac)
    dp = bootstrap_ci(f_true[rows] - f_plac[rows])
    out["g6_wrong_user"] = {"full@10": float(np.nanmean(f_wrong[rows])),
                            "delta_vs_intercept": dw[0], "ci95": dw[1],
                            "PASS_no_gain_over_intercept": bool(dw[0] <= 0 or dw[1][0] <= 0)}
    out["g6_shuffled_values"] = {"true_minus_shuffled": ds[0], "ci95": ds[1],
                                 "note": "graded-value binding within the concept set (reported)"}
    out["g6_placebo_constant1"] = {"true_minus_placebo": dp[0], "ci95": dp[1],
                                   "note": "membership kept, values flattened to 1.0 (reported)"}
    # ---- G6-c duplicates x2/x3 ----
    dup_res = {}
    for k in (2, 3):
        conc_dup = {r: conc_true[r] + [conc_true[r][0]] * (k - 1) for r in rows}
        f_dup, _ = score_rows(net, ctx, rows, conc_dup, order_raw=True)
        dd = bootstrap_ci(f_dup[rows] - f_true[rows])
        dup_res[f"x{k}"] = {"delta": dd[0], "ci95": dd[1],
                            "PASS_near_zero": bool(abs(dd[0]) < 0.002)}
    out["g6_duplicates"] = dup_res
    # ---- G8-c order permutation (raw sequential order, K=5) ----
    devs_z = []; devs_f = []
    f_base, Z_base = score_rows(net, ctx, rows, conc_true, order_raw=True)
    for kperm in range(4):
        rngp = np.random.default_rng(100 + kperm)
        conc_perm = {}
        for r in rows:
            p = rngp.permutation(len(conc_true[r]))
            conc_perm[r] = [conc_true[r][i] for i in p]
        f_p, Z_p = score_rows(net, ctx, rows, conc_perm, order_raw=True)
        devs_z.append(float((Z_p - Z_base).norm(dim=1).max()))
        devs_f.append(float(np.nanmax(np.abs(f_p[rows] - f_base[rows]))))
    dmean = bootstrap_ci(np.abs(f_p[rows] - f_base[rows]))
    out["g8_order"] = {"raw_order_max_dz": max(devs_z), "raw_order_max_dNDCG": max(devs_f),
                       "raw_order_mean_absdNDCG": dmean[0],
                       "canonical_order_invariant": True,
                       "note": "recursion is order-dependent in principle; the deployed path sorts "
                               "canonically (desc value, ties by id) = exact invariance; raw-order "
                               "sensitivity reported as the honest residual"}
    out["PASS"] = bool(out["g3a_flip"]["PASS"]
                       and out["g6_wrong_user"]["PASS_no_gain_over_intercept"]
                       and all(v["PASS_near_zero"] for v in dup_res.values()))
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"clite_gates{(chr(95)+args.out_tag) if args.out_tag else ''}.json")
    json.dump(out, open(path, "w"), indent=2)
    log(f"[gates] -> {path} PASS={out['PASS']}")
    print(f"\nG3a-c flip {d[0]:+.4f} PASS={out['g3a_flip']['PASS']} | wrong-user vs int {dw[0]:+.4f} "
          f"| shuf {ds[0]:+.4f} | plac {dp[0]:+.4f} | dup x2 {dup_res['x2']['delta']:+.5f} "
          f"x3 {dup_res['x3']['delta']:+.5f} | order max dNDCG {max(devs_f):.5f} "
          f"| OVERALL PASS={out['PASS']}")


if __name__ == "__main__":
    main()
