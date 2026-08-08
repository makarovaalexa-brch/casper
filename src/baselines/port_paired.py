r"""port_paired.py -- PAIRED percentile-bootstrap CIs for the DRE, DPE and IGCN port cells (task #21).

WHY. `docs/results/DRE_PORT_ON_OUR_RULER.md` owes "Paired CIs against entropy0 and greedy items-only";
`docs/results/SNAPIGCN_NATIVE.md` owes the port cells' intervals; the DPE table contains the two most
load-bearing unmeasured claims in the battery (the ILD arm that DEGRADES with budget, and the oracle
CEILING).

THE FORM IS PAIRED AND IT IS NOT OPTIONAL. `ARM_N_PHASE1.md` sec 4b records the scar: the unpaired
`1.96*sqrt(2)*SE` interval overstated the true interval ~4.6x and forced a retraction. Here we
resample USERS, recompute the per-user difference, and take the 2.5/97.5 percentiles.

REUSE, NOT A FOURTH IMPLEMENTATION. `boot_ci` is imported from `gtport_paired` and used as the
reference; the chunked variant below exists ONLY because this box has ~2 GB free and the reference
allocates a 10000x10000 float64 (~1.2 GB) per call. `--equiv_check` proves the two are BIT-IDENTICAL
on real difference vectors before any interval is reported; if they ever differ the run aborts.

COMPARATOR DISCIPLINE. The IGCN run carries `entropy0-static` and `fixedorder:greedy_items` INSIDE
itself, so every IGCN gap is exact by construction. DRE and DPE ran without comparators, so their
gaps against entropy0 / greedy items-only are CROSS-RUN and are admitted only on the control below:
the IGCN run's in-run `entropy0-static` must be per-user identical (0.0e+00) to the banked act1
vector, which proves the item-only static arm reproduces bit-for-bit across independent harness runs
on this contract. Every such row is stamped `pairing: cross-run (admitted)`.

  python src/baselines/port_paired.py --boot 10000 --seed 98765
"""
import os
import sys
import json
import time
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from gtport_paired import boot_ci as REF_BOOT_CI          # noqa: E402  (the reference form)

OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
LABEL = "port_paired"                                     # unique output label (HARD RULE 11e)
LOGP = os.path.join(OUT, f"{LABEL}.log")
OUTP = os.path.join(OUT, f"{LABEL}.json")

VEC_DRE = os.path.join(OUT, "adaptive_table_dre_port_v1_per_user.npz")
VEC_DPE = os.path.join(OUT, "adaptive_table_dpe_v1_per_user.npz")
VEC_IGCN = os.path.join(OUT, "adaptive_table_igcn_port01_per_user.npz")
ACT1 = os.path.join(OUT, "act1_per_user_vectors.npz")
BUDGETS = (1, 2, 4, 8, 16)
AXES = ("full", "tail")

E0 = "entropy0-static"
GI = "fixedorder:greedy_items"


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def boot_ci(d, n_boot, seed, chunk=250):
    """Chunked percentile bootstrap. BIT-IDENTICAL to gtport_paired.boot_ci: RandomState fills
    size=(n_boot, n) row-major, so drawing it in consecutive (chunk, n) blocks from the same stream
    yields exactly the same resamples. Proven, not asserted -- see equiv_check()."""
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    n = len(d)
    means = np.empty(n_boot, dtype=float)
    i = 0
    while i < n_boot:
        m = min(chunk, n_boot - i)
        means[i:i + m] = d[rng.randint(0, n, size=(m, n))].mean(axis=1)
        i += m
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a, b, n_boot, seed, label, out, pairing="in-run"):
    if len(a) != len(b):
        raise ValueError(f"{label}: vectors misaligned ({len(a)} vs {len(b)}) -- pairing invalid")
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d, n_boot, seed)
    sep = bool(lo > 0 or hi < 0)
    out[label] = {"n": int(len(d)), "delta": float(d.mean()), "lo": lo, "hi": hi,
                  "separates": sep, "pairing": pairing,
                  "direction": ("+" if d.mean() > 0 else "-") if sep else "ns"}
    logln(f"    {label:72s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] "
          f"{'SEP' if sep else 'ns '} {pairing}")
    return out[label]


def equiv_check(V, n_boot, seed):
    """The chunked bootstrap must equal the reference EXACTLY on real difference vectors."""
    probes = [(("igcn", 16, "full"), (E0, 16, "full")),
              (("igcn", 8, "tail"), (GI, 8, "tail"))]
    rows = []
    for x, y in probes:
        d = V[x] - V[y]
        r = REF_BOOT_CI(d, n_boot, seed)
        c = boot_ci(d, n_boot, seed)
        ok = (r == c)
        rows.append({"probe": f"{x[0]}|k{x[1]}|{x[2]} - {y[0]}", "ref_lo": r[0], "ref_hi": r[1],
                     "chunked_lo": c[0], "chunked_hi": c[1], "identical": bool(ok)})
        logln(f"[equiv] {rows[-1]['probe']:52s} ref={r} chunked={c} "
              f"{'IDENTICAL' if ok else 'DIFFER'}")
    return rows, all(r["identical"] for r in rows)


def load(path, tag):
    z = np.load(path, allow_pickle=False)
    V = {}
    for key in z.files:
        cons, spec, k, ax = key.split("|")
        V[(cons, spec, int(k.lstrip("k")), ax)] = z[key]
    logln(f"[{tag}] {len(V)} vectors from {os.path.basename(path)}")
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=98765)
    a = ap.parse_args()

    DRE = load(VEC_DRE, "dre")
    DPE = load(VEC_DPE, "dpe")
    IG = load(VEC_IGCN, "igcn")

    # one flat namespace for the "ours" consumer, which is where every cross-arm gap lives
    V = {}
    for src in (DRE, DPE, IG):
        for (cons, spec, k, ax), v in src.items():
            if cons == "ours":
                V[(spec, k, ax)] = v

    # ------------------------------------------------------------------ canary: row alignment
    nf = {len(v) for kk, v in V.items() if kk[2] == "full"}
    nt = {len(v) for kk, v in V.items() if kk[2] == "tail"}
    assert len(nf) == 1 and len(nt) == 1, f"user subsets differ: full={nf} tail={nt}"
    n_full, n_tail = nf.pop(), nt.pop()
    logln(f"[canary] all cells row-aligned: n_full={n_full}, n_tail={n_tail}")

    res = {"meta": {"label": LABEL, "boot": a.boot, "seed": a.seed,
                    "form": "paired percentile bootstrap over the per-user difference vector",
                    "sources": {"dre": os.path.basename(VEC_DRE),
                                "dpe": os.path.basename(VEC_DPE),
                                "igcn": os.path.basename(VEC_IGCN)},
                    "contract": ("canonical Liang ML-25M, policy-B pool, NO-FLOOR, 10,000 test "
                                 "users, consumer = certified tower i26c ep31; DPE ran under "
                                 "--concept_neg_cap 0 (certified-era answers), which cannot touch "
                                 "the item-only comparators"),
                    "n_full_scored": n_full, "n_tail_scored": n_tail},
           "points": {}, "equiv_check": {}, "self_test": {},
           "C1_dpe_ild_degrades_with_budget": {},
           "C1b_budget_monotonicity_controls": {},
           "C2_oracle_ceiling": {},
           "C3_dre": {}, "C3b_dre_consumer_agreement": {},
           "C4_hellinger_vs_entropy0": {},
           "C5_igcn": {},
           "C6_dpe_secondary": {}}

    for (spec, k, ax), v in V.items():
        res["points"].setdefault(f"{spec}|k{k}", {})[ax] = float(v.mean())
    for (cons, spec, k, ax), v in DRE.items():
        res["points"].setdefault(f"{cons}::{spec}|k{k}", {})[ax] = float(v.mean())

    def flush():
        json.dump(res, open(OUTP, "w"), indent=2)

    flush()

    # ------------------------------------------------------------------ EQUIVALENCE of the two RNGs
    logln("=== equivalence: chunked bootstrap vs the gtport_paired reference (must be IDENTICAL) ===")
    rows, eq_ok = equiv_check(V, a.boot, a.seed)
    res["equiv_check"] = {"probes": rows, "pass": bool(eq_ok)}
    res["meta"]["equiv_check_pass"] = bool(eq_ok)
    flush()
    if not eq_ok:
        raise SystemExit("[port_paired] chunked bootstrap != reference -- refusing to report.")

    # ------------------------------------------------------------------ ADMISSION (cross-run)
    # The item-only static order must reproduce BIT-FOR-BIT across independent harness runs before
    # any DRE/DPE row may be differenced against the IGCN run's comparators.
    logln("=== admission: in-run entropy0-static vs banked act1 entropy0, per user ===")
    ref = np.load(ACT1, allow_pickle=False)
    worst = 0.0
    for k in BUDGETS:
        for ax in AXES:
            worst = max(worst, float(np.max(np.abs(
                IG[("ours", E0, k, ax)] - ref[f"ours|{E0}|{k}|{ax}"]))))
    res["meta"]["admission_entropy0_max_abs_per_user_diff"] = worst
    admitted = worst <= 0.0
    res["meta"]["cross_run_rows_admitted"] = bool(admitted)
    logln(f"[admission] max per-user |in-run entropy0 - banked act1 entropy0| = {worst:.3e} "
          f"({'PASS' if admitted else 'REFUSED'})")
    flush()
    if not admitted:
        raise SystemExit("[port_paired] cross-run admission FAILED -- DRE/DPE gaps not reportable.")
    XR = "cross-run (admitted)"

    # ------------------------------------------------------------------ SELF-TEST (arm vs itself)
    logln("=== self-test: an arm against ITSELF must give delta 0.0000 with CI [0.0000,0.0000] ===")
    for spec in ("dre", "dpe", "igcn"):
        for k in BUDGETS:
            for ax in AXES:
                paired(V[(spec, k, ax)], V[(spec, k, ax)], a.boot, a.seed,
                       f"SELFTEST/{ax}/k{k}/{spec}-minus-itself", res["self_test"])
    st_ok = all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0
                for v in res["self_test"].values())
    res["meta"]["self_test_pass"] = bool(st_ok)
    logln(f"[self-test] {'PASS' if st_ok else 'FAIL'}")
    flush()

    # ------------------------------------------------------- C1: DOES DPE-ILD DEGRADE WITH BUDGET?
    logln("=== C1: dpe (ILD) budget ladder -- is MORE questions WORSE, per user? ===")
    for k in (2, 4, 8, 16):
        for ax in AXES:
            paired(V[("dpe", k, ax)], V[("dpe", 1, ax)], a.boot, a.seed,
                   f"{ax}/dpe/k{k}-minus-k1", res["C1_dpe_ild_degrades_with_budget"])
    for ax in AXES:
        paired(V[("dpe", 16, ax)], V[("dpe", 8, ax)], a.boot, a.seed,
               f"{ax}/dpe/k16-minus-k8", res["C1_dpe_ild_degrades_with_budget"])
    flush()

    logln("=== C1b: the same k16-minus-k1 contrast for every other arm (the control) ===")
    for spec in ("dpe_hellinger", "dpe_oracle", "dpe_concept", "dre", "igcn", "igcn_i1", E0, GI):
        for ax in AXES:
            paired(V[(spec, 16, ax)], V[(spec, 1, ax)], a.boot, a.seed,
                   f"{ax}/{spec}/k16-minus-k1", res["C1b_budget_monotonicity_controls"])
    flush()

    # ------------------------------------------------------------------ C2: THE ORACLE CEILING
    logln("=== C2: dpe_oracle (a CEILING -- it reads the user's own ratings at selection time) ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(V[("dpe_oracle", k, ax)], V[(GI, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dpe_oracle-minus-greedy_items", res["C2_oracle_ceiling"], XR)
            paired(V[("dpe_oracle", k, ax)], V[(E0, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dpe_oracle-minus-entropy0", res["C2_oracle_ceiling"], XR)
    flush()

    # ------------------------------------------------------------------ C3: DRE
    logln("=== C3: DRE (WWW'20 learned static) vs entropy0 (2008) and vs greedy items-only ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(V[("dre", k, ax)], V[(E0, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dre-minus-entropy0", res["C3_dre"], XR)
            paired(V[("dre", k, ax)], V[(GI, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dre-minus-greedy_items", res["C3_dre"], XR)
    flush()

    logln("=== C3b: DRE's two consumers against each other (IN-RUN, exact) ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(DRE[("ours", "dre", k, ax)], DRE[("dre", "dre", k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/ours_tower-minus-dre_own_decoder", res["C3b_dre_consumer_agreement"])
    flush()

    # ------------------------------------------------------------------ C4: hellinger vs entropy0
    logln("=== C4: dpe_hellinger vs entropy0 (the k=16 TAIL dead heat, .0925 vs .0924) ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(V[("dpe_hellinger", k, ax)], V[(E0, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dpe_hellinger-minus-entropy0", res["C4_hellinger_vs_entropy0"], XR)
            paired(V[("dpe_hellinger", k, ax)], V[(GI, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dpe_hellinger-minus-greedy_items",
                   res["C4_hellinger_vs_entropy0"], XR)
    flush()

    # ------------------------------------------------------------------ C5: IGCN (all IN-RUN)
    logln("=== C5: IGCN vs entropy0 and greedy items-only (exact by construction, same run) ===")
    for spec in ("igcn", "igcn_i1", "igcn_page15"):
        for k in BUDGETS:
            for ax in AXES:
                paired(V[(spec, k, ax)], V[(E0, k, ax)], a.boot, a.seed,
                       f"{ax}/k{k}/{spec}-minus-entropy0", res["C5_igcn"])
                paired(V[(spec, k, ax)], V[(GI, k, ax)], a.boot, a.seed,
                       f"{ax}/k{k}/{spec}-minus-greedy_items", res["C5_igcn"])
    logln("=== C5b: what the i=1 deviation and the page-15 period are worth against `igcn` ===")
    for spec in ("igcn_i1", "igcn_page15"):
        for k in BUDGETS:
            for ax in AXES:
                paired(V[(spec, k, ax)], V[("igcn", k, ax)], a.boot, a.seed,
                       f"{ax}/k{k}/{spec}-minus-igcn", res["C5_igcn"])
    flush()

    # ------------------------------------------------------------------ C6: DPE secondary
    logln("=== C6: dpe / dpe_concept, secondary rows ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(V[("dpe", k, ax)], V[(E0, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dpe-minus-entropy0", res["C6_dpe_secondary"], XR)
            paired(V[("dpe_concept", k, ax)], V[("dpe", k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/dpe_concept-minus-dpe", res["C6_dpe_secondary"])
    flush()

    logln(f"[port_paired] analysis written -> {OUTP}")


if __name__ == "__main__":
    main()
