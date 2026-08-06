r"""gtport_paired.py -- PAIRED percentile-bootstrap CIs for the Golbandi tree PORT+CROSS cells.

WHY. `docs/results/GOLBANDI_PORT_ON_OUR_RULER.md` states its own gap under "Owed before citing":
"Paired CIs on every gap here. The headline tail difference is +.0068; nothing in this table has an
interval yet."

THE FORM IS PAIRED AND IT IS NOT OPTIONAL. `ARM_N_PHASE1.md` sec 4b records the scar: the unpaired
`1.96*sqrt(2)*SE` interval overstated the true interval ~4.6x and forced a retraction. Here we
resample USERS, recompute the per-user difference, and take the 2.5/97.5 percentiles of the
difference distribution -- the same `boot_ci`/`paired` pattern as `act1_paired.py`.

PAIRING IS VALID WITHIN THE RUN. Every cell of `adaptive_table_gtport13_per_user.npz` scores the SAME
10,000 held-out users against the SAME targets under the SAME policy-B pool and NO-FLOOR contract;
only the asked questions and the consumer change. Row alignment is asserted by a canary, not assumed.

CROSS-RUN (greedy). The greedy items-only rows live in `MASK_CONTRACT_DIAGNOSTIC.md`, a different
harness (`strategy_channel_suite.genv_eval`). Those per-user vectors are only admitted if
`--greedy_npz` is supplied AND the entropy0 vector inside it matches this run's
`ours|treestatic:entropy0` per user (not merely in the mean). Without that file the greedy gaps are
reported as point estimates with NO interval, and said so.

  python src/baselines/gtport_paired.py --boot 10000 --seed 98765
  python src/baselines/gtport_paired.py --greedy_npz <path>   # adds the greedy comparisons
"""
import os
import sys
import json
import time
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
LABEL = "gtport13_paired"                      # unique output label (HARD RULE 11e)
LOGP = os.path.join(OUT, f"{LABEL}.log")

VEC = os.path.join(OUT, "adaptive_table_gtport13_per_user.npz")
BUDGETS = (1, 2, 4, 8)
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
    logln(f"    {label:72s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] {'SEP' if sep else 'ns '}")
    return out[label]


def load_cache(path):
    vecs = {}
    z = np.load(path, allow_pickle=False)
    for key in z.files:
        cons, spec, k, ax = key.split("|")
        vecs.setdefault((cons, spec, int(k.lstrip("k"))), {})[ax] = z[key]
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", default=VEC)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--greedy_npz", default=os.path.join(OUT, "gtport13_greedy_per_user.npz"),
                    help="per-user vectors for the greedy sequences, scored in THIS contract")
    a = ap.parse_args()

    vecs = load_cache(a.vec)
    logln(f"[gtport] {len(vecs)} cells from {os.path.basename(a.vec)}")

    # ---------------------------------------------------------------- canary: row alignment
    nf = {len(c["full"]) for c in vecs.values()}
    nt = {len(c["tail"]) for c in vecs.values()}
    assert len(nf) == 1 and len(nt) == 1, f"user subsets differ: full={nf} tail={nt}"
    n_full, n_tail = nf.pop(), nt.pop()
    logln(f"[canary] all {len(vecs)} cells row-aligned: n_full={n_full}, n_tail={n_tail}")

    have = set(vecs)
    missing = []

    def V(cons, spec, k, ax):
        return vecs[(cons, spec, k)][ax]

    def M_(cons, spec, k, ax):
        return float(np.mean(V(cons, spec, k, ax)))

    def P(cons, x, y, k, ax, label, bucket, cons_y=None):
        cy = cons_y or cons
        if (cons, x, k) not in have or (cy, y, k) not in have:
            missing.append(label)
            return None
        return paired(V(cons, x, k, ax), V(cy, y, k, ax), a.boot, a.seed, label, bucket)

    res = {"meta": {"label": LABEL, "boot": a.boot, "seed": a.seed,
                    "source_vectors": os.path.basename(a.vec),
                    "form": "paired percentile bootstrap over the per-user difference vector",
                    "contract": "canonical Liang ML-25M, policy-B pool, NO-FLOOR, 10,000 test users",
                    "n_full_scored": n_full, "n_tail_scored": n_tail,
                    "cells_present": sorted(f"{c}|{s}|k{k}" for c, s, k in have)},
           "points": {f"{c}|{s}|k{k}": {"full": M_(c, s, k, "full"), "tail": M_(c, s, k, "tail")}
                      for (c, s, k) in vecs},
           "self_test": {}, "C1_tree_vs_entropy0_ours": {}, "C2_tree_vs_greedy_ours": {},
           "C3_tree_vs_helf_golbandi": {}, "C4_static_ladder": {}, "C5_leaf_shrinkage": {},
           "C6_identity_checks": {}}

    # ---------------------------------------------------------------- SELF-TEST
    logln("=== self-test: an arm against ITSELF must give delta 0.0000 with CI [0.0000,0.0000] ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(V("ours", "golbandi_tree", k, ax), V("ours", "golbandi_tree", k, ax),
                   a.boot, a.seed, f"SELFTEST/ours/{ax}/k{k}/golbandi_tree-minus-itself",
                   res["self_test"])
    st_ok = bool(res["self_test"]) and all(
        v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for v in res["self_test"].values())
    res["meta"]["self_test_pass"] = st_ok
    logln(f"[self-test] {'PASS' if st_ok else 'FAIL'}: {len(res['self_test'])} self-comparisons")

    # ---------------------------------------------------------------- identity check
    logln("=== identity: treestatic:entropy0 vs the independent entropy0-static arm ===")
    for cons in ("ours", "golbandi_leaf", "golbandi_leaf_lam200"):
        for k in BUDGETS:
            for ax in AXES:
                if (cons, "entropy0-static", k) not in have:
                    continue
                d = np.abs(V(cons, "treestatic:entropy0", k, ax)
                           - V(cons, "entropy0-static", k, ax))
                res["C6_identity_checks"][f"{cons}/{ax}/k{k}"] = {
                    "max_abs_per_user_diff": float(d.max()),
                    "mean_diff": float(V(cons, "treestatic:entropy0", k, ax).mean()
                                       - V(cons, "entropy0-static", k, ax).mean())}
    mx = max(v["max_abs_per_user_diff"] for v in res["C6_identity_checks"].values())
    logln(f"[identity] max per-user |treestatic:entropy0 - entropy0-static| over all cells = {mx:.2e}")

    # ---------------------------------------------------------------- C1
    logln("=== C1: consumer `ours`, golbandi_tree MINUS treestatic:entropy0 (the CROSSOVER) ===")
    for k in BUDGETS:
        for ax in AXES:
            P("ours", "golbandi_tree", "treestatic:entropy0", k, ax,
              f"ours/{ax}/k{k}/golbandi_tree-minus-entropy0", res["C1_tree_vs_entropy0_ours"])

    # ---------------------------------------------------------------- C3
    logln("=== C3: consumer `golbandi_leaf`, golbandi_tree MINUS treestatic:HELF (the 'tie') ===")
    for k in BUDGETS:
        for ax in AXES:
            P("golbandi_leaf", "golbandi_tree", "treestatic:helf", k, ax,
              f"golbandi_leaf/{ax}/k{k}/golbandi_tree-minus-helf", res["C3_tree_vs_helf_golbandi"])
    logln("--- and against entropy0 / popularity on its own consumer ---")
    for y in ("treestatic:entropy0", "treestatic:popularity"):
        for k in BUDGETS:
            for ax in AXES:
                P("golbandi_leaf", "golbandi_tree", y, k, ax,
                  f"golbandi_leaf/{ax}/k{k}/golbandi_tree-minus-{y.split(':')[1]}",
                  res["C3_tree_vs_helf_golbandi"])

    # ---------------------------------------------------------------- C4
    logln("=== C4: the static ladder, on BOTH consumers (instrument dependence) ===")
    pairs = [("entropy0", "popularity"), ("entropy0", "helf"), ("popularity", "helf")]
    for cons in ("ours", "golbandi_leaf", "golbandi_leaf_lam200"):
        for x, y in pairs:
            for k in BUDGETS:
                for ax in AXES:
                    P(cons, f"treestatic:{x}", f"treestatic:{y}", k, ax,
                      f"{cons}/{ax}/k{k}/{x}-minus-{y}", res["C4_static_ladder"])

    # ---------------------------------------------------------------- C5
    logln("=== C5: leaf shrinkage lam=200 MINUS lam=8, same asked questions ===")
    for spec in ("golbandi_tree", "treestatic:helf", "treestatic:entropy0", "treestatic:popularity"):
        for k in BUDGETS:
            for ax in AXES:
                P("golbandi_leaf_lam200", spec, spec, k, ax,
                  f"lam200-minus-lam8/{ax}/k{k}/{spec}", res["C5_leaf_shrinkage"],
                  cons_y="golbandi_leaf")

    # ---------------------------------------------------------------- C2: greedy (cross-run)
    logln("=== C2: consumer `ours`, golbandi_tree MINUS greedy items-only (CROSS-RUN) ===")
    gstat = {"greedy_npz": a.greedy_npz, "admitted": False}
    if os.path.exists(a.greedy_npz):
        gz = load_cache(a.greedy_npz)
        # THE ADMISSION TEST: the greedy run's entropy0 vector must equal this run's, PER USER.
        worst = 0.0
        for k in BUDGETS:
            for ax in AXES:
                if ("greedy_env", "entropy0-top16", k) not in gz:
                    worst = float("inf")
                    continue
                worst = max(worst, float(np.max(np.abs(
                    gz[("greedy_env", "entropy0-top16", k)][ax] - V("ours", "treestatic:entropy0", k, ax)))))
        gstat["entropy0_canary_max_abs_per_user_diff"] = worst
        logln(f"[C2 canary] max per-user |greedy-env entropy0 - this-run entropy0| = {worst:.3e}")
        if worst <= 1e-6:
            gstat["admitted"] = True
            for k in BUDGETS:
                for ax in AXES:
                    key = ("greedy_env", "greedy-items-only", k)
                    if key not in gz:
                        missing.append(f"ours/{ax}/k{k}/golbandi_tree-minus-greedy_items")
                        continue
                    paired(V("ours", "golbandi_tree", k, ax), gz[key][ax], a.boot, a.seed,
                           f"ours/{ax}/k{k}/golbandi_tree-minus-greedy_items",
                           res["C2_tree_vs_greedy_ours"])
                    paired(V("ours", "treestatic:entropy0", k, ax), gz[key][ax], a.boot, a.seed,
                           f"ours/{ax}/k{k}/entropy0-minus-greedy_items",
                           res["C2_tree_vs_greedy_ours"])
        else:
            logln("[C2] REFUSED: the greedy vectors are not per-user identical on the shared "
                  "entropy0 anchor, so they are NOT in this contract. No interval reported.")
    else:
        logln(f"[C2] no greedy per-user vectors at {a.greedy_npz} -- point estimates only, no CI")
    res["meta"]["greedy"] = gstat

    res["meta"]["skipped_for_missing_cells"] = sorted(set(missing))
    outp = os.path.join(OUT, f"{LABEL}.json")
    json.dump(res, open(outp, "w"), indent=2)
    logln(f"[gtport] analysis written -> {outp}")


if __name__ == "__main__":
    main()
