r"""convbandit_paired.py -- PAIRED percentile-bootstrap CIs for every gap in the convbandit port.

WHY. `docs/results/CONVBANDIT_PORT_ON_OUR_RULER.md` states its own debt under "Owed":
"Paired CIs on every gap here -- `--per_user` vectors are on disk. The load-bearing number is the
+.0582/+.0454 concept-channel gap."

THE FORM IS PAIRED AND IT IS NOT OPTIONAL. `ARM_N_PHASE1.md` sec 4b records the scar: the unpaired
`1.96*sqrt(2)*SE` interval overstated the true interval ~4.6x and forced a retraction. Here we
resample USERS, recompute the per-user difference, and take the 2.5/97.5 percentiles -- the same
`boot_ci`/`paired` pattern as `act1_paired.py` / `gtport_paired.py`.

CROSS-RUN ROWS ARE RE-SCORED, NOT ADMITTED ON A MEAN. entropy0, greedy items-only and greedy
concepts-only were re-scored inside this contract by `convbandit_greedy_rescore.py`; that script's
entropy0 vector must be per-user identical to the banked `ours|entropy0-static` (act1), which is the
admission test enforced again here before any greedy difference is reported.

  python src/baselines/convbandit_paired.py --boot 10000 --seed 98765
"""
import os
import json
import time
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
LABEL = "convbandit_paired"                       # unique output label (HARD RULE 11e)
LOGP = os.path.join(OUT, f"{LABEL}.log")

VEC = os.path.join(OUT, "adaptive_table_convbandit_uniform_v2_per_user.npz")
RESCORE = os.path.join(OUT, "convbandit_rescore_per_user.npz")
RESCORE_NC = os.path.join(OUT, "convbandit_rescore_nocap_per_user.npz")
ACT1 = os.path.join(OUT, "act1_per_user_vectors.npz")
BUDGETS = (1, 2, 4, 8, 16)
AXES = ("full", "tail")
FAMILY = ("clime", "conlinucb_bs", "conlinucb_mcr", "conlinucb_bs_static", "conucb")


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
    logln(f"    {label:66s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] {'SEP' if sep else 'ns '}")
    return out[label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=98765)
    a = ap.parse_args()

    z = np.load(VEC, allow_pickle=False)
    V = {}
    for key in z.files:
        cons, spec, k, ax = key.split("|")
        V[(spec, int(k.lstrip("k")), ax)] = z[key]
    logln(f"[convbandit] {len(V)} vectors from {os.path.basename(VEC)}")

    # ------------------------------------------------------------------ canary: row alignment
    nf = {len(V[k]) for k in V if k[2] == "full"}
    nt = {len(V[k]) for k in V if k[2] == "tail"}
    assert len(nf) == 1 and len(nt) == 1, f"user subsets differ: full={nf} tail={nt}"
    n_full, n_tail = nf.pop(), nt.pop()
    logln(f"[canary] all cells row-aligned: n_full={n_full}, n_tail={n_tail}")

    res = {"meta": {"label": LABEL, "boot": a.boot, "seed": a.seed,
                    "source_vectors": os.path.basename(VEC),
                    "rescore_vectors": os.path.basename(RESCORE),
                    "form": "paired percentile bootstrap over the per-user difference vector",
                    "contract": ("canonical Liang ML-25M, policy-B pool, NO-FLOOR, 10,000 test "
                                 "users, tower i26c ep31 + concept channel cv3 ep08, key-term "
                                 "schedule b(t)=5*floor(log t), repr uniform_member_mean"),
                    "n_full_scored": n_full, "n_tail_scored": n_tail},
           "points": {}, "self_test": {}, "schedule_self_test": {},
           "C1_concept_channel_in_contract": {}, "C1b_concept_channel_certified_answers": {},
           "C2_clime_vs_static": {},
           "C3_conucb_vs_successors": {}, "C4_family_pairwise": {},
           "C5_answer_convention_cost": {}}

    for (spec, k, ax), v in V.items():
        res["points"].setdefault(f"{spec}|k{k}", {})[ax] = float(v.mean())

    # ------------------------------------------------------------------ SELF-TEST (arm vs itself)
    logln("=== self-test: an arm against ITSELF must give delta 0.0000 with CI [0.0000,0.0000] ===")
    for k in BUDGETS:
        for ax in AXES:
            paired(V[("clime", k, ax)], V[("clime", k, ax)], a.boot, a.seed,
                   f"SELFTEST/{ax}/k{k}/clime-minus-itself", res["self_test"])
    st_ok = all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0
                for v in res["self_test"].values())
    res["meta"]["self_test_pass"] = bool(st_ok)
    logln(f"[self-test] {'PASS' if st_ok else 'FAIL'}")

    # ------------------------------------------------------------------ SCHEDULE SELF-TEST
    # Under b(t)=5*floor(log t) no key-term turn fires before t=3, so conlinucb_mcr, conlinucb_bs and
    # clime must be NUMERICALLY IDENTICAL at k=1 and k=2. If they are not, the schedule or the
    # selectors are wrong and nothing below is interpretable.
    logln("=== schedule self-test: mcr == bs == clime at k=1,2 (no key-term turn before t=3) ===")
    sched_ok = True
    for x, y in (("conlinucb_mcr", "conlinucb_bs"), ("conlinucb_bs", "clime"),
                 ("conlinucb_mcr", "clime")):
        for k in (1, 2):
            for ax in AXES:
                r = paired(V[(x, k, ax)], V[(y, k, ax)], a.boot, a.seed,
                           f"SCHEDULE/{ax}/k{k}/{x}-minus-{y}", res["schedule_self_test"])
                sched_ok &= (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0)
    res["meta"]["schedule_self_test_pass"] = bool(sched_ok)
    logln(f"[schedule self-test] {'PASS' if sched_ok else 'FAIL'}")
    if not sched_ok:
        json.dump(res, open(os.path.join(OUT, f"{LABEL}.json"), "w"), indent=2)
        raise SystemExit("[convbandit] SCHEDULE SELF-TEST FAILED -- stopping, as briefed. The three "
                         "adaptive variants are NOT identical at k=1/k=2, so either the key-term "
                         "schedule or a selector is wrong. No gap below may be read.")

    # ------------------------------------------------------------------ the re-scored static rows
    R, admitted, note = {}, False, ""
    if os.path.exists(RESCORE):
        zr = np.load(RESCORE, allow_pickle=False)
        for key in zr.files:
            _, name, k, ax = key.split("|")
            R[(name, int(k.lstrip("k")), ax)] = zr[key]
        ref = np.load(ACT1, allow_pickle=False)
        worst = 0.0
        for k in BUDGETS:
            for ax in AXES:
                if ("entropy0-top16", k, ax) not in R:
                    worst = float("inf")
                    continue
                worst = max(worst, float(np.max(np.abs(
                    R[("entropy0-top16", k, ax)] - ref[f"ours|entropy0-static|{k}|{ax}"]))))
        res["meta"]["rescore_entropy0_canary_max_abs_per_user_diff"] = worst
        logln(f"[admission] max per-user |re-scored entropy0 - banked act1 entropy0| = {worst:.3e}")
        # SECOND control, on the CONCEPT side: ConLinUCB-BS-static is a fixed key-term order in this
        # run, so re-scoring it here must reproduce the banked convbandit vectors per user. That is
        # what licenses re-scoring a concept sequence at all.
        worst_c = max(float(np.max(np.abs(R[("conlinucb_bs_static", k, ax)] - V[("conlinucb_bs_static", k, ax)])))
                      for k in BUDGETS for ax in AXES) if ("conlinucb_bs_static", 1, "full") in R \
            else float("inf")
        res["meta"]["rescore_bs_static_canary_max_abs_per_user_diff"] = worst_c
        logln(f"[admission] max per-user |re-scored conlinucb_bs_static - banked| = {worst_c:.3e}")
        admitted = worst <= 0.0 and worst_c <= 0.0
        if not admitted:
            note = ("the re-scored entropy0 control is not per-user identical to the banked ladder "
                    "vector, so the re-scored rows are not in this contract")
            logln(f"[admission] REFUSED: {note}")
    else:
        note = f"no re-scored vectors at {RESCORE}"
        logln(f"[admission] {note} -- static-row gaps are point estimates with NO interval")
    res["meta"]["rescored_rows_admitted"] = bool(admitted)
    res["meta"]["rescore_note"] = note

    def PR(spec, name, k, ax, label, bucket, sign=+1):
        """paired: convbandit arm MINUS re-scored static row (sign=+1) or the reverse."""
        if not admitted or (name, k, ax) not in R:
            return None
        x, y = V[(spec, k, ax)], R[(name, k, ax)]
        if sign < 0:
            x, y = y, x
        return paired(x, y, a.boot, a.seed, label, bucket)

    # ------------------------------------------------------------------ C1: THE LOAD-BEARING ONE
    logln("=== C1 (IN CONTRACT): greedy concepts-only MINUS conlinucb_bs_static, both under the "
          "convbandit run's own concept-answer convention (population neg-cap) ===")
    for k in BUDGETS:
        for ax in AXES:
            PR("conlinucb_bs_static", "greedy-concepts-only", k, ax,
               f"{ax}/k{k}/greedy_concepts-minus-conlinucb_bs_static",
               res["C1_concept_channel_in_contract"], sign=-1)

    # C1b. The SAME two fixed orders under the CERTIFIED-ERA answer convention (revealed-set cap),
    # i.e. the values the concept channel cv3 was trained on and the ones the published greedy row
    # was measured under. Both arms move, so this is still like-for-like; it is the other honest
    # reading of the same comparison, and the two together bracket the doc's cross-contract number.
    logln("=== C1b (CERTIFIED-ERA ANSWERS): the same two fixed orders, both re-scored ===")
    NC, nc_ok = {}, False
    if os.path.exists(RESCORE_NC):
        zn = np.load(RESCORE_NC, allow_pickle=False)
        for key in zn.files:
            _, name, k, ax = key.split("|")
            NC[(name, int(k.lstrip("k")), ax)] = zn[key]
        # control: entropy0 is item-only, so the concept convention cannot touch it -- it must still
        # be per-user identical to the banked ladder vector.
        w = max(float(np.max(np.abs(NC[("entropy0-top16", k, ax)] -
                                    np.load(ACT1)[f"ours|entropy0-static|{k}|{ax}"])))
                for k in BUDGETS for ax in AXES)
        res["meta"]["nocap_entropy0_canary_max_abs_per_user_diff"] = w
        nc_ok = w <= 0.0
        logln(f"[admission/nocap] entropy0 control max per-user diff = {w:.3e} "
              f"({'PASS' if nc_ok else 'REFUSED'})")
    res["meta"]["certified_answer_rows_admitted"] = bool(nc_ok)
    if nc_ok:
        for k in BUDGETS:
            for ax in AXES:
                paired(NC[("greedy-concepts-only", k, ax)], NC[("conlinucb_bs_static", k, ax)],
                       a.boot, a.seed,
                       f"{ax}/k{k}/greedy_concepts-minus-conlinucb_bs_static",
                       res["C1b_concept_channel_certified_answers"])
        # C5: what the answer convention alone is worth, same order, same consumer, same users.
        logln("=== C5: certified-era MINUS in-contract answers, the SAME fixed order ===")
        for name in ("greedy-concepts-only", "conlinucb_bs_static"):
            for k in BUDGETS:
                for ax in AXES:
                    paired(NC[(name, k, ax)], R[(name, k, ax)], a.boot, a.seed,
                           f"{ax}/k{k}/{name}/certified_answers-minus-in_contract",
                           res["C5_answer_convention_cost"])

    # ------------------------------------------------------------------ C2: CLiME vs the statics
    logln("=== C2: CLiME MINUS entropy0 / greedy items-only (the Act-1 null) ===")
    for k in BUDGETS:
        for ax in AXES:
            PR("clime", "entropy0-top16", k, ax, f"{ax}/k{k}/clime-minus-entropy0",
               res["C2_clime_vs_static"])
            PR("clime", "greedy-items-only", k, ax, f"{ax}/k{k}/clime-minus-greedy_items",
               res["C2_clime_vs_static"])
            PR("clime", "greedy-concepts-only", k, ax, f"{ax}/k{k}/clime-minus-greedy_concepts",
               res["C2_clime_vs_static"])

    # ------------------------------------------------------------------ C3: ConUCB vs successors
    logln("=== C3: ConUCB MINUS its own successors (the answerability gap) ===")
    for y in ("conlinucb_mcr", "conlinucb_bs", "clime", "conlinucb_bs_static"):
        for k in BUDGETS:
            for ax in AXES:
                paired(V[("conucb", k, ax)], V[(y, k, ax)], a.boot, a.seed,
                       f"{ax}/k{k}/conucb-minus-{y}", res["C3_conucb_vs_successors"])

    # ------------------------------------------------------------------ C4: family pairwise
    logln("=== C4: the rest of the family, pairwise ===")
    for i, x in enumerate(FAMILY):
        for y in FAMILY[i + 1:]:
            if x == "conucb" or y == "conucb":
                continue
            for k in BUDGETS:
                for ax in AXES:
                    paired(V[(x, k, ax)], V[(y, k, ax)], a.boot, a.seed,
                           f"{ax}/k{k}/{x}-minus-{y}", res["C4_family_pairwise"])

    outp = os.path.join(OUT, f"{LABEL}.json")
    json.dump(res, open(outp, "w"), indent=2)
    logln(f"[convbandit] analysis written -> {outp}")


if __name__ == "__main__":
    main()
