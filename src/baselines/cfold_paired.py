r"""cfold_paired.py -- PAIRED percentile-bootstrap CIs for the concept-fold key-term REPRESENTATION
three-way (uniform_member_mean / representative / whitened_centroid).

WHY. Three separate runs now exist under ONE answer convention (`--concept_neg_cap 0`, CERTIFIED-ERA
values) with `--per_user` vectors on disk. The point estimates at k=16 (.2168/.0877, .2135/.0864,
.2211/.0942) order whitened > uniform > representative, but point estimates are not intervals --
`ARM_N_PHASE1.md` sec 4b records the scar where an UNPAIRED `1.96*sqrt(2)*SE` interval overstated
the true interval ~4.6x and forced a retraction. Here we resample USERS, recompute the per-user
difference, and take the 2.5/97.5 percentiles.

PAIRABILITY IS PROVEN, NOT ASSUMED. These are three SEPARATE RUNS. Under the key-term schedule
b(t)=5*floor(log t) no key-term turn fires before t=3, so the key-term REPRESENTATION cannot touch
any arm at k=1 or k=2 -- the three runs must therefore be BIT-IDENTICAL there, per user. That test
doubles as the row-alignment canary: it can only pass if the same 10,000 users were scored in the
same order. If it fails, the runs are not pairable and only point estimates may be reported.

  python src/baselines/cfold_paired.py --boot 10000 --seed 98765
"""
import os
import json
import time
import argparse
import itertools

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
LABEL = "cfold_paired"                            # unique output label (HARD RULE 11e)
LOGP = os.path.join(OUT, f"{LABEL}.log")

# arm name -> (per-user npz, table json). The representative run is the PARTIAL one: its ConUCB arm
# was killed, but the conlinucb_mcr cells we read here are complete.
ARMS = {
    "uniform": ("adaptive_table_cfold_uniform_negcap0_per_user.npz",
                "adaptive_table_cfold_uniform_negcap0.json", "uniform_member_mean"),
    "representative": ("adaptive_table_cfold_representative_per_user.npz",
                       "adaptive_table_cfold_representative_PARTIAL.json", "representative"),
    "whitened": ("adaptive_table_cfold_whitened_per_user.npz",
                 "adaptive_table_cfold_whitened.json", "whitened_centroid"),
}
SPEC = "conlinucb_mcr"          # the selector all three runs share
BUDGETS = (1, 2, 4, 8, 16)
TEST_BUDGETS = (4, 8, 16)       # the briefed comparisons
FREE_BUDGETS = (1, 2)           # no key-term turn fires here => must be identical across arms
AXES = ("full", "tail")


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def boot_ci(d, n_boot, seed):
    """Percentile bootstrap over the per-user DIFFERENCE vector (act1_paired.boot_ci)."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a, b, n_boot, seed, label, out):
    if len(a) != len(b):
        raise ValueError(f"{label}: vectors misaligned ({len(a)} vs {len(b)}) -- pairing invalid")
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d, n_boot, seed)
    sep = bool(lo > 0 or hi < 0)
    out[label] = {"n": int(len(d)), "delta": float(d.mean()), "lo": lo, "hi": hi,
                  "separates": sep, "direction": ("+" if d.mean() > 0 else "-") if sep else "ns"}
    logln(f"    {label:60s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] {'SEP' if sep else 'ns '}")
    return out[label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=98765)
    a = ap.parse_args()

    # ------------------------------------------------------------------ load + CONTRACT check
    V, meta = {}, {}
    for arm, (npz, tbl, repr_expected) in ARMS.items():
        z = np.load(os.path.join(OUT, npz), allow_pickle=False)
        for key in z.files:
            cons, spec, k, ax = key.split("|")
            V[(arm, spec, int(k.lstrip("k")), ax)] = z[key]
        m = json.load(open(os.path.join(OUT, tbl)))["meta"]
        c = m.get("_contract", {})
        # The per-cell record is `keyterm_repr_by_selector` where it exists; the older
        # representative run predates that field and carries a STALE top-level `keyterm_repr`
        # (uniform_member_mean) that its own _contract and run log both contradict. Trust
        # _contract.keyterm_repr, which is the value the harness was actually invoked with.
        by_sel = m.get("keyterm_repr_by_selector", {})
        meta[arm] = {
            "table": tbl, "npz": npz,
            "keyterm_repr_contract": c.get("keyterm_repr"),
            "keyterm_repr_by_selector": by_sel.get(SPEC),
            "keyterm_repr_toplevel_STALE": m.get("keyterm_repr"),
            "concept_answer_convention": m.get("concept_answer_convention"),
            "concept_neg_cap": c.get("concept_neg_cap"),
            "snapshot": m.get("snapshot"), "concept_ckpt": c.get("concept_ckpt"),
            "n_test_users": m.get("n_test_users"), "no_floor": m.get("no_floor"),
            "ucb_alpha": m.get("ucb_alpha"), "conv_b_schedule": c.get("conv_b_schedule"),
            "n_concepts": m.get("n_concepts"),
        }
        if c.get("keyterm_repr") != repr_expected:
            raise SystemExit(f"[cfold] {arm}: _contract.keyterm_repr={c.get('keyterm_repr')!r} "
                             f"!= expected {repr_expected!r} -- wrong file, stopping.")
        if by_sel and by_sel.get(SPEC) != repr_expected:
            raise SystemExit(f"[cfold] {arm}: keyterm_repr_by_selector[{SPEC}]="
                             f"{by_sel.get(SPEC)!r} != {repr_expected!r} -- stopping.")
    logln(f"[cfold] loaded {len(V)} vectors from {len(ARMS)} runs")

    # every contract field except keyterm_repr must MATCH across the three runs
    shared = ["concept_answer_convention", "concept_neg_cap", "snapshot", "concept_ckpt",
              "n_test_users", "no_floor", "ucb_alpha", "conv_b_schedule", "n_concepts"]
    mismatch = {f: {arm: meta[arm][f] for arm in ARMS}
                for f in shared if len({str(meta[arm][f]) for arm in ARMS}) != 1}
    if mismatch:
        raise SystemExit(f"[cfold] contract fields differ across runs: {mismatch} -- stopping.")
    logln("[contract] all three runs agree on " + ", ".join(shared))
    logln(f"[contract] answer convention = {meta['uniform']['concept_answer_convention']} "
          f"(concept_neg_cap={meta['uniform']['concept_neg_cap']})")

    res = {"meta": {"label": LABEL, "boot": a.boot, "seed": a.seed, "spec": SPEC,
                    "form": "paired percentile bootstrap over the per-user difference vector",
                    "arms": meta,
                    "contract": ("canonical Liang ML-25M, 10,000 test users, NO-FLOOR, tower i26c "
                                 "ep31 + concept channel cv3 ep08, key-term schedule "
                                 "b(t)=5*floor(log t), selector conlinucb_mcr, "
                                 "concept_neg_cap=0 (one answer convention across all three)")},
           "points": {}, "self_test": {}, "pairability_canary": {},
           "A_whitened_minus_uniform": {}, "B_whitened_minus_representative": {},
           "C_uniform_minus_representative": {}}

    for (arm, spec, k, ax), v in V.items():
        if spec == SPEC:
            res["points"].setdefault(f"{arm}|k{k}", {})[ax] = float(v.mean())

    # ------------------------------------------------------------------ SELF-TEST (arm vs itself)
    logln("=== self-test: an arm against ITSELF must give delta 0.0000 with CI [0.0000,0.0000] ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(V[("whitened", SPEC, k, ax)], V[("whitened", SPEC, k, ax)], a.boot, a.seed,
                   f"SELFTEST/{ax}/k{k}/whitened-minus-itself", res["self_test"])
    st_ok = all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0
                for v in res["self_test"].values())
    res["meta"]["self_test_pass"] = bool(st_ok)
    logln(f"[self-test] {'PASS' if st_ok else 'FAIL'}")

    # ------------------------------------------------------------------ PAIRABILITY CANARY
    # These are three SEPARATE RUNS. Under b(t)=5*floor(log t) the first key-term turn is t=3, so
    # at k=1 and k=2 the key-term representation is never consulted and the three runs must be
    # BIT-IDENTICAL per user. Passing requires the same users in the same order -- which is exactly
    # the property pairing needs. Failing means we may report point estimates ONLY.
    logln("=== pairability canary: the three runs must be BIT-IDENTICAL at k=1,2 (no key-term "
          "turn fires before t=3), which can only hold if users are row-aligned ===")
    pair_ok = True
    for x, y in itertools.combinations(ARMS, 2):
        for k in FREE_BUDGETS:
            for ax in AXES:
                va, vb = V[(x, SPEC, k, ax)], V[(y, SPEC, k, ax)]
                same = len(va) == len(vb) and bool(np.array_equal(va, vb))
                mx = float(np.max(np.abs(va - vb))) if len(va) == len(vb) else float("nan")
                res["pairability_canary"][f"{ax}/k{k}/{x}-vs-{y}"] = {
                    "n_x": int(len(va)), "n_y": int(len(vb)),
                    "max_abs_per_user_diff": mx, "identical": same}
                logln(f"    {ax}/k{k}/{x}-vs-{y:16s} n={len(va):5d} maxabs={mx:.3e} "
                      f"{'IDENTICAL' if same else 'DIFFERS'}")
                pair_ok &= same
    res["meta"]["pairability_pass"] = bool(pair_ok)
    logln(f"[pairability] {'PASS -- runs are row-aligned, pairing is valid' if pair_ok else 'FAIL'}")
    if not pair_ok:
        res["meta"]["verdict"] = ("NOT PAIRABLE -- the three runs are not per-user alignable; "
                                  "point estimates only, no intervals reported")
        json.dump(res, open(os.path.join(OUT, f"{LABEL}.json"), "w"), indent=2)
        raise SystemExit("[cfold] PAIRABILITY CANARY FAILED -- refusing to fake pairing. "
                         "Point estimates are in the JSON; no CI may be read.")

    # ------------------------------------------------------------------ the briefed comparisons
    logln("=== A: whitened_centroid MINUS uniform_member_mean ===")
    for k in TEST_BUDGETS:
        for ax in AXES:
            paired(V[("whitened", SPEC, k, ax)], V[("uniform", SPEC, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/whitened-minus-uniform", res["A_whitened_minus_uniform"])

    logln("=== B: whitened_centroid MINUS representative ===")
    for k in TEST_BUDGETS:
        for ax in AXES:
            paired(V[("whitened", SPEC, k, ax)], V[("representative", SPEC, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/whitened-minus-representative", res["B_whitened_minus_representative"])

    # third leg, for completeness: it fixes the ORDER of the two losers
    logln("=== C (completes the three-way): uniform_member_mean MINUS representative ===")
    for k in TEST_BUDGETS:
        for ax in AXES:
            paired(V[("uniform", SPEC, k, ax)], V[("representative", SPEC, k, ax)], a.boot, a.seed,
                   f"{ax}/k{k}/uniform-minus-representative", res["C_uniform_minus_representative"])

    outp = os.path.join(OUT, f"{LABEL}.json")
    json.dump(res, open(outp, "w"), indent=2)
    logln(f"[cfold] analysis written -> {outp}")


if __name__ == "__main__":
    main()
