r"""concept_granularity.py -- companion diagnostic to answer_contrast.py (author 2026-07-25:
"coarseness does NOT work -- we have 1031 concepts of every granularity"). Tests whether FINE concepts
carry more per-answer information than BROAD ones, on the strong stack (frozen i25 tower + signed
C-lite). Member count = the granularity proxy (few members = fine/specific, many = broad/crude).

A1 GRANULARITY OF SELECTED CONCEPTS: member-count distribution (median + terciles) of the concepts
   actually ASKED by (i) the generic polarization bank, (ii) the member-mass bank, (iii) oracle
   per-user selection (top-|behavioral v|). Hypothesis: oracle selection prefers FINE concepts while
   the realizable banks systematically ask BROAD ones.

A2 PER-ANSWER LIFT BY GRANULARITY: bin the 1031 concepts into member-count terciles (fine/medium/
   broad). Per bin, the mean SINGLE-concept cold NDCG@10 lift (fold ONE concept, honest behavioral B
   answer, over ALL users who can answer it, minus the intercept). Non-lossy (every answerable user).
   Tests coarseness-as-ceiling (fine ~= broad, both weak) vs coarseness-as-selection-artifact (fine
   strong but under-selected by the realizable banks). Also Spearman(per-concept lift, member count).

OPTIONAL: signed C-FULL (cfull_signed_best.pt) concept-ask row (polarization bank, honest B answers)
   head-to-head vs signed C-lite under IDENTICAL answers on this harness.

Output: experiments/battery/concept_granularity.json + printed tables. Diagnostic-for-understanding.
Usage: python src/instrument/concept_granularity.py [--smoke] [--full_threads] [--no_scfull]
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

from train_tower_t2 import log, I25Encoder
import run_battery_phaseA as PA
from run_battery_phaseA import build_real_ctx, build_smoke_ctx, OUTDIR, spearman
from tradeoff_ledger import build_shared, Rung
from adaptive_concept_arms import train_concept_stats, score_users
from answer_contrast import Behavioral, walk_static_concept, score_snapshots, BUDGETS
from scipy import sparse

MIN_ANS = 10                                                    # min answerable users for a per-concept mean


def dist(vals):
    """median + terciles (p33/p66) + mean + n of a value list."""
    v = np.asarray(vals, np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
            "p33": float(np.percentile(v, 33)), "p66": float(np.percentile(v, 66)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--sclite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfold_signed_best.pt"))
    ap.add_argument("--scfull_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfull_signed_best.pt"))
    ap.add_argument("--no_scfull", action="store_true")
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    if args.smoke and not sh.get("signed_available"):
        rng = np.random.RandomState(4); C = sh["d_c"].shape[0]
        sh["conc_value_signed"] = (rng.rand(ctx.n, C).astype(np.float32) * 2 - 1)
        sh["conc_fold_signed"] = rng.rand(ctx.n, C) > 0.15
        sh["signed_available"] = True
    assert sh.get("signed_available")
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    rows_arr = np.asarray(rows)
    C = len(sh["tags"])
    member_count = np.array([len(sh["members"][t]) for t in sh["tags"]], np.int64)
    Vb = sh["conc_value_signed"]; ansB = sh["conc_answerable"]; foldB = sh["conc_fold_signed"]
    AF = (ansB & foldB)                                          # (n, C) answerable-and-folds under B

    # rung: signed C-lite
    from concept_fold import ConceptFoldNet
    if args.smoke:
        net = ConceptFoldNet(C, d=ctx.d, h=32, conc_init=sh["d_c"].numpy())
        with torch.no_grad():
            for p in net.mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.05)
    else:
        blob = torch.load(args.sclite_ckpt, map_location="cpu")
        net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"]); net.load_state_dict(blob["net"])
    net.eval()
    rung = Rung("sclite", ctx, sh, clite_net=net, val_source="signed")

    Mm = sparse.csr_matrix((np.ones(sum(len(sh["members"][t]) for t in sh["tags"]), np.float32),
                            (np.concatenate([sh["members"][t] for t in sh["tags"]]),
                             np.concatenate([np.full(len(sh["members"][t]), i)
                                             for i, t in enumerate(sh["tags"])]))),
                           shape=(ctx.ni, C))
    pexp = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
    _, std_v = train_concept_stats(ctx, Mm, pexp, smoke=args.smoke)
    conc_order = np.argsort(-std_v)                             # polarization bank
    mass_order = sh["bank_conc"]                                # member-mass bank

    # ======================================================================= A1: granularity of asked
    NA = min(8, C)
    polar_counts = member_count[conc_order[:NA]]
    mass_counts = member_count[mass_order[:NA]]
    ksel_counts = []
    per_q_k_median = {}
    for r in rows:
        elig = np.flatnonzero(AF[r])
        order = elig[np.argsort(-np.abs(Vb[r, elig]))][:NA]
        ksel_counts.extend(member_count[order].tolist())
    a1 = {"proxy": "genome member count (fine=few, broad=many)", "asked_first_k": NA,
          "polarization_bank": dist(polar_counts), "member_mass_bank": dist(mass_counts),
          "oracle_selection_pooled": dist(ksel_counts),
          "all_1031_concepts": dist(member_count)}
    log(f"[A1] member-count median asked: polar={a1['polarization_bank'].get('median')} "
        f"mass={a1['member_mass_bank'].get('median')} oracle-sel={a1['oracle_selection_pooled'].get('median')} "
        f"(all-concepts median {a1['all_1031_concepts']['median']})")

    # ======================================================================= A2: per-answer lift by gran
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    f0, t0, _ = score_users(rung, [empty] * len(rows), [[] for _ in rows], rows)
    int_full = np.full(ctx.n, np.nan); int_tail = np.full(ctx.n, np.nan)
    int_full[rows_arr] = f0; int_tail[rows_arr] = t0
    per_c = {"member_count": member_count.tolist(), "n_answerable": np.zeros(C, np.int64).tolist(),
             "lift_full@10": np.full(C, np.nan).tolist(), "lift_tail@10": np.full(C, np.nan).tolist()}
    lf = np.full(C, np.nan); lt = np.full(C, np.nan); nans = np.zeros(C, np.int64)
    t0a = time.time()
    for c in range(C):
        sel = AF[rows_arr, c]
        rc = rows_arr[sel]
        nans[c] = len(rc)
        if len(rc) < MIN_ANS:
            continue
        conc = [[(c, float(Vb[r, c]))] for r in rc]
        fc, tc, _ = score_users(rung, [empty] * len(rc), conc, list(rc))
        lf[c] = float(np.nanmean(fc - int_full[rc]))
        lt[c] = float(np.nanmean(tc - int_tail[rc]))
        if c % 200 == 0:
            log(f"[A2] concept {c}/{C} ({(time.time()-t0a)/60:.1f}m)")
    per_c["n_answerable"] = nans.tolist()
    per_c["lift_full@10"] = lf.tolist(); per_c["lift_tail@10"] = lt.tolist()
    # terciles by member count over concepts with a stable estimate
    valid = ~np.isnan(lf)
    mc_valid = member_count[valid]
    b1, b2 = np.percentile(mc_valid, [33.333, 66.667])
    bins = {"fine": member_count <= b1, "medium": (member_count > b1) & (member_count <= b2),
            "broad": member_count > b2}
    a2 = {"tercile_bounds_membercount": [float(b1), float(b2)], "min_answerable_users": MIN_ANS,
          "n_concepts_scored": int(valid.sum()), "bins": {}}
    for name, m in bins.items():
        mm = m & valid
        a2["bins"][name] = {"n_concepts": int(mm.sum()),
                            "median_member_count": float(np.median(member_count[mm])) if mm.any() else None,
                            "mean_answerable_users": float(nans[mm].mean()) if mm.any() else None,
                            "mean_lift_full@10": float(np.nanmean(lf[mm])) if mm.any() else None,
                            "mean_lift_tail@10": float(np.nanmean(lt[mm])) if mm.any() else None}
    a2["spearman_lift_full_vs_membercount"] = spearman(member_count[valid], lf[valid])
    a2["spearman_lift_tail_vs_membercount"] = spearman(member_count[valid], lt[valid])
    log(f"[A2] mean lift full fine/med/broad = "
        f"{a2['bins']['fine']['mean_lift_full@10']:.4f}/{a2['bins']['medium']['mean_lift_full@10']:.4f}/"
        f"{a2['bins']['broad']['mean_lift_full@10']:.4f} | spearman(lift,memcount)="
        f"{a2['spearman_lift_full_vs_membercount']:.3f}")

    # ======================================================================= optional: scfull head-to-head
    head2head = None
    if not args.no_scfull and (args.smoke or os.path.exists(args.scfull_ckpt)):
        try:
            if args.smoke:
                head2head = "SKIPPED in smoke (scfull needs concept-token ckpt)"
            else:
                blob = torch.load(args.scfull_ckpt, map_location="cpu")
                native = ctx.enc._native[0] if hasattr(ctx.enc, "_native") else None
                enc_sf = I25Encoder(ctx.ni, native, d_lat=ctx.d, token_mode="film", n_concepts=C)
                enc_sf.load_state_dict(blob["enc"]); enc_sf.eval()
                rung_sf = Rung("scfull", ctx, sh, cfull_enc=enc_sf, val_source="signed")
                bmodel = Behavioral(sh, sh["lvl_lookup"])
                head2head = {}
                for name, rg in (("sclite", rung), ("scfull", rung_sf)):
                    ev = walk_static_concept(rg, rows, bmodel, conc_order)
                    sc = score_snapshots(rg, rows, ev, "concept")
                    head2head[name] = {str(q): {"full@10": sc[q]["full@10"], "tail@10": sc[q]["tail@10"],
                                                "full@100": sc[q]["full@100"]} for q in BUDGETS}
                log("[scfull] concept-ask B (polarization) q8 full: "
                    f"sclite={head2head['sclite']['8']['full@10']:.4f} "
                    f"scfull={head2head['scfull']['8']['full@10']:.4f}")
        except Exception as e:
            head2head = f"scfull load failed: {type(e).__name__}: {e}"

    out = {"analysis": "concept_granularity", "n_users": len(rows), "n_concepts": C,
           "stack": "frozen t2i25_EP4 tower + signed C-lite", "diagnostic": True,
           "A1_granularity_of_selected": a1, "A2_per_answer_lift_by_granularity": a2,
           "scfull_vs_sclite_concept_ask_B": head2head, "seconds": round(time.time() - t00, 1)}
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "concept_granularity.json")
    json.dump(out, open(jpath, "w"), indent=2, default=float)
    log(f"[out] -> {jpath}")

    print("\n=== A1  GRANULARITY OF ASKED CONCEPTS (member count; fine=few / broad=many) ===")
    for k in ("polarization_bank", "member_mass_bank", "oracle_selection_pooled", "all_1031_concepts"):
        d = a1[k]
        print(f"  {k:>24}: median={d.get('median')} terciles[{d.get('p33')},{d.get('p66')}] "
              f"mean={d.get('mean'):.0f} n={d['n']}")
    print("\n=== A2  PER-ANSWER COLD LIFT BY GRANULARITY (single concept, honest B answer) ===")
    for name in ("fine", "medium", "broad"):
        b = a2["bins"][name]
        print(f"  {name:>6}: n_conc={b['n_concepts']:4d} med_members={b['median_member_count']:.0f} "
              f"mean_ans_users={b['mean_answerable_users']:.0f} "
              f"lift_full@10={b['mean_lift_full@10']:+.4f} lift_tail@10={b['mean_lift_tail@10']:+.4f}")
    print(f"  Spearman(lift_full, member_count) = {a2['spearman_lift_full_vs_membercount']:+.3f} "
          f"(negative => FINE concepts lift MORE per answer)")
    if isinstance(head2head, dict):
        print("\n=== C-LITE vs C-FULL (concept-ask B, polarization, identical answers) full@10 ===")
        for name in ("sclite", "scfull"):
            print(f"  {name:>6}: " + " ".join(f"q{q}={head2head[name][str(q)]['full@10']:.4f}"
                                              for q in BUDGETS))


if __name__ == "__main__":
    main()
