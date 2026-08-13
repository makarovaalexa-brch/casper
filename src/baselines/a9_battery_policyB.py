r"""a9_battery_policyB.py -- A9: put `tab:battery`'s three greedy static rows on the REBUILT
policy-B orders, and re-verify the narrative sentences that quantify over them.

WHAT THIS IS AND IS NOT. This is a RE-SCORE, not a search. The three orders are fixed artifacts
(`greedy_static_policyB_{combined,items,concepts}.json`, A1/A3/A4) and their per-user TEST vectors
were already produced by `src/instrument/channel_orders_on_test.py` under the same contract as the
rest of the table (certified pair, policy-B pool, NO-FLOOR, behavioural answers, 10,000 test users,
no sampling). Nothing is re-selected and no recommender is re-fit here: this script pairs those
banked vectors against the battery's banked selector vectors and computes the intervals.

THE FORM IS PAIRED AND IT IS NOT OPTIONAL (`ARM_N_PHASE1.md` sec 4b): resample USERS, recompute the
per-user difference, take the 2.5/97.5 percentiles. `boot_ci` is IMPORTED from `gtport_paired` (the
reference) and the chunked variant from `port_paired` is proven bit-identical to it on real
difference vectors before any interval is reported.

THE TWO ADMISSION GATES, because this crosses two harness families for the first time:

  G1 CROSS-FAMILY. The instrument harness (`strategy_channel_suite.genv_eval`, which produced the
     three static rows) and the arm-N harness (`run_interview_table.score_arm`, which produced every
     selector row) must agree PER USER, not in the mean. They share exactly one arm: the rebuilt
     items order and the banked policy-A items order both open with item 83, so at k=1 they are the
     same interview. Required: max per-user |armN - instrument| == 0.0 after the instrument's own
     float32 storage round-trip, on BOTH axes, and the tail row-compaction (10,000 -> 9,281) must
     line up. Anything else and the pairing is invalid and nothing is reported.

  G2 CROSS-RUN. Every selector source file that carries a shared static arm must reproduce the
     banked act1 `entropy0-static` vector per user to 0.0e+00 (the `port_paired` admission test).
     Files with no shared arm (DRE, DPE) ride the chain the IGCN run establishes and are stamped.

  Plus the free contract check the author asked for: the three rebuilt orders' K=16 test numbers
  must reproduce the banked headline figures 0.25900/0.12370, 0.24470/0.10171, 0.24467/0.11607.

EXTENDED 2026-08-13 for TODO 1.3(b), the contract-scale three-outcome EDDI run. The two new rows
(`eddi_ig3` / `eddi_ig3_zero`) are differenced against exactly the same three rebuilt static orders
by exactly this machinery -- extended rather than forked, so no second copy of the pairing, the
gates or the bootstrap can drift from this one. The new source npz carries TWO extra gates of its
own: it re-ran `entropy0-static` (G2, the cross-run admission test every other source pays) AND it
re-ran `eddi_ig`, the two-outcome row already in `tab:battery`, which must reproduce the banked
act1 vectors per user at 0.0e+00 -- a free contract check on the whole chain, and it makes the
two-outcome vs three-outcome comparison IN-RUN rather than cross-run. Both are asserted below.
The source is skipped, by name, if its npz is absent, so this script still runs standalone.

  python src/baselines/a9_battery_policyB.py --boot 10000 --seed 98765
  python src/baselines/a9_battery_policyB.py --label a9_battery_policyB_eddi3   # + the EDDI rows
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
from port_paired import boot_ci as CHUNKED_BOOT_CI        # noqa: E402  (memory-bounded twin)

IV = os.path.join(_ROOT, "experiments", "baselines", "interview")
BAT = os.path.join(_ROOT, "experiments", "battery")
LABEL = "a9_battery_policyB"                              # unique output label (HARD RULE 11e)
LOGP = os.path.join(IV, f"{LABEL}.log")
OUTP = os.path.join(IV, f"{LABEL}.json")

STATIC_NPZ = os.path.join(BAT, "channel_orders_policyB_test_per_user.npz")
STATIC_JSON = os.path.join(BAT, "channel_orders_policyB_test.json")
ACT1 = os.path.join(IV, "act1_per_user_vectors.npz")

BUDGETS = (1, 2, 4, 8, 16)
AXES = ("full", "tail")
STATICS = ("combined", "items", "concepts")

# the K=16 test figures the author gave as the contract check
TARGETS = {"combined": (0.25900, 0.12370),
           "items": (0.24470, 0.10171),
           "concepts": (0.24467, 0.11607)}

# selector rows of tab:battery -> (source npz, consumer, spec, kind)
SOURCES = {
    "act1": os.path.join(IV, "act1_per_user_vectors.npz"),
    "gtport": os.path.join(IV, "adaptive_table_gtport13_per_user.npz"),
    "convbandit": os.path.join(IV, "adaptive_table_convbandit_uniform_v2_per_user.npz"),
    "convrescore": os.path.join(IV, "convbandit_rescore_per_user.npz"),
    "dpe": os.path.join(IV, "adaptive_table_dpe_v1_per_user.npz"),
    "dre": os.path.join(IV, "adaptive_table_dre_port_v1_per_user.npz"),
    "baldaopt": os.path.join(IV, "adaptive_table_bald_aopt_v1_per_user.npz"),
    "igcn": os.path.join(IV, "adaptive_table_igcn_port01_per_user.npz"),
    # TODO 1.3(b), 2026-08-13: the contract-scale three-outcome EDDI run. Carries its own
    # `entropy0-static` (G2) AND its own `eddi_ig` (the banked two-outcome row, re-derived in-run).
    "eddi3": os.path.join(IV, "adaptive_table_eddi3_policyB_per_user.npz"),
}
# Sources whose npz may legitimately be absent (the script predates them); skipped BY NAME, never
# silently. Everything else is required, exactly as before.
OPTIONAL_SOURCES = ("eddi3",)

# (display name, source, key-consumer, key-spec, type, admission)
ARMS = [
    ("GLM-UCB",              "act1",       "ours", "glm_ucb",             "adaptive", "in-family"),
    ("Thompson",             "act1",       "ours", "thompson",            "adaptive", "in-family"),
    ("latent-only info-gain", "act1",      "ours", "eddi_ig",             "adaptive", "in-family"),
    ("Golbandi tree",        "gtport",     "ours", "golbandi_tree",       "adaptive", "in-family"),
    ("DRE",                  "dre",        "ours", "dre",             "learned static", "chain"),
    ("CLiME",                "convbandit", "ours", "clime",               "adaptive", "chain"),
    ("ConLinUCB-BS-static",  "convbandit", "ours", "conlinucb_bs_static", "static (concepts)",
                                                                                       "chain"),
    ("ConLinUCB-MCR",        "convbandit", "ours", "conlinucb_mcr",       "adaptive", "chain"),
    ("ConLinUCB-BS",         "convbandit", "ours", "conlinucb_bs",        "adaptive", "chain"),
    ("ConUCB",               "convbandit", "ours", "conucb",              "adaptive", "chain"),
    ("DPE-Hellinger",        "dpe",        "ours", "dpe_hellinger",       "adaptive", "chain"),
    ("DPE-ILD",              "dpe",        "ours", "dpe",                 "adaptive", "chain"),
    ("DPE-concept",          "dpe",        "ours", "dpe_concept",         "adaptive", "chain"),
    ("IGCN",                 "igcn",       "ours", "igcn",                "adaptive", "in-family"),
    ("BALD",                 "baldaopt",   "ours", "bald",              "value-free", "in-family"),
    ("A-optimal",            "baldaopt",   "ours", "aopt",              "value-free", "in-family"),
    ("clairvoyant oracle",   "dpe",        "ours", "dpe_oracle",          "ceiling",  "chain"),
    # TODO 1.3(b). `latent-only info-gain (re-run)` is the SAME arm as the row above, recomputed in
    # the eddi3 process; it is here so its intervals are read off the same machinery, and its
    # per-user identity with the banked row is asserted as a gate before any of this is reported.
    ("latent-only info-gain (re-run)", "eddi3", "ours", "eddi_ig",        "adaptive", "in-family"),
    ("EDDI three-outcome (zero)",      "eddi3", "ours", "eddi_ig3_zero",  "adaptive", "in-family"),
    ("EDDI three-outcome (signed)",    "eddi3", "ours", "eddi_ig3",       "adaptive", "in-family"),
]

# the literature-anchored static heuristics, carried so the "gap that matters" sentences can be
# re-checked against the rebuilt rows too
HEURISTICS = [
    ("entropy0",        "act1", "ours", "entropy0-static",      "static"),
    ("popularity",      "act1", "ours", "popularity-static",    "static"),
    ("HELF",            "act1", "ours", "helf-static",          "static"),
    ("pure entropy",    "act1", "ours", "pure_entropy-static",  "static"),
    ("random (bank)",   "act1", "ours", "random_bank-static",   "static"),
    ("random (catalogue)", "act1", "ours", "random_catalog-static", "static"),
]


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(IV, exist_ok=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def key_variants(cons, spec, k, ax):
    """The two key spellings in use across the banked npz files."""
    return (f"{cons}|{spec}|{k}|{ax}", f"{cons}|{spec}|k{k}|{ax}")


def get(z, cons, spec, k, ax):
    for key in key_variants(cons, spec, k, ax):
        if key in z.files:
            return np.asarray(z[key], np.float64)
    return None


def paired(a, b, n_boot, seed, label, out, pairing):
    if len(a) != len(b):
        raise ValueError(f"{label}: vectors misaligned ({len(a)} vs {len(b)}) -- pairing invalid")
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    lo, hi = CHUNKED_BOOT_CI(d, n_boot, seed)
    sep = bool(lo > 0 or hi < 0)
    out[label] = {"n": int(len(d)), "delta": float(d.mean()), "lo": lo, "hi": hi,
                  "separates": sep, "pairing": pairing,
                  "direction": ("+" if d.mean() > 0 else "-") if sep else "ns"}
    return out[label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--label", default=LABEL,
                    help="output label. A run that ADDS rows must use a fresh one -- it must never "
                         "overwrite the banked table another document already cites (HARD RULE 11e).")
    a = ap.parse_args()
    t00 = time.time()
    global LOGP, OUTP
    LOGP = os.path.join(IV, f"{a.label}.log")
    OUTP = os.path.join(IV, f"{a.label}.json")

    # Sources present on disk. An optional one that is absent is NAMED and skipped; a required one
    # that is absent still raises, as before.
    have_src = {tag: os.path.exists(p) for tag, p in SOURCES.items()}
    for tag, ok in have_src.items():
        if not ok:
            assert tag in OPTIONAL_SOURCES, f"required source npz missing: {SOURCES[tag]}"
            logln(f"[source] SKIPPED (absent): {tag} -> {os.path.basename(SOURCES[tag])}")

    res = {"meta": {"label": a.label, "analysis": "A9: tab:battery greedy rows on the rebuilt "
                                                "policy-B orders, with paired CIs",
                    "boot": a.boot, "seed": a.seed,
                    "form": "paired percentile bootstrap over the per-user difference vector",
                    "contract": ("canonical Liang ML-25M TEST cohort (10,000 users), consumer = "
                                 "certified pair t2i26c_ep31 + concept_cv3_ep08, policy-B candidate "
                                 "pool, NO-FLOOR, behavioural answers, learned decoder bias, no "
                                 "sampling"),
                    "static_source": os.path.basename(STATIC_NPZ)},
           "gates": {}, "static_rows": {}, "selector_points": {},
           "vs_static": {}, "narrative": {}}

    def flush():
        json.dump(res, open(OUTP, "w"), indent=2)

    # ------------------------------------------------------------------ the three rebuilt rows
    zs = np.load(STATIC_NPZ, allow_pickle=False)
    sj = json.load(open(STATIC_JSON))
    assert sj["mask_policy"] == "B" and sj["pool_policy"] == "B" and sj["split"] == "test"
    assert sj["n_test_users"] == 10000 and sj["tower"] == "t2i26c_ep31.pt"
    assert sj["banked_assert"]["all_bit_identical"], "the source run's banked assert did not pass"
    S = {}
    tail_nan = np.isnan(np.asarray(zs["combined|16|tail"], np.float64))
    keep = ~tail_nan
    logln(f"[static] tail scored users: {int(keep.sum())} of {len(keep)} "
          f"({int(tail_nan.sum())} have no tail target)")
    for arm in STATICS:
        for K in BUDGETS:
            f = np.asarray(zs[f"{arm}|{K}|full"], np.float64)
            t = np.asarray(zs[f"{arm}|{K}|tail"], np.float64)
            assert np.array_equal(np.isnan(t), tail_nan), f"{arm}|{K}: tail NaN mask differs"
            S[(arm, K, "full")] = f
            S[(arm, K, "tail")] = t[keep]
            res["static_rows"].setdefault(arm, {})[str(K)] = {
                "full@10": float(f.mean()), "tail@10": float(t[keep].mean())}
        res["static_rows"][arm]["ans@16"] = {
            "items": sj["test_curves"][arm]["16"]["items_answered"],
            "concepts": sj["test_curves"][arm]["16"]["concepts_answered"]}

    # ------------------------------------------------------------------ CONTRACT CHECK (targets)
    tg = {}
    for arm, (wf, wt) in TARGETS.items():
        gf = res["static_rows"][arm]["16"]["full@10"]
        gt = res["static_rows"][arm]["16"]["tail@10"]
        ok = (round(gf, 5) == round(wf, 5)) and (round(gt, 5) == round(wt, 5))
        tg[arm] = {"want_full": wf, "got_full": gf, "want_tail": wt, "got_tail": gt, "pass": bool(ok)}
        logln(f"[target] {arm:<9} K=16 full {gf:.5f} (want {wf:.5f})  tail {gt:.5f} "
              f"(want {wt:.5f})  {'PASS' if ok else 'FAIL'}")
    res["gates"]["banked_targets"] = tg
    assert all(v["pass"] for v in tg.values()), "banked K=16 targets do not reproduce -- STOP"

    # ------------------------------------------------------------------ G1 CROSS-FAMILY GATE
    zb = np.load(SOURCES["baldaopt"], allow_pickle=False)
    g1 = {}
    for ax in AXES:
        armn = get(zb, "ours", "fixedorder:greedy_items", 1, ax)
        inst = S[("items", 1, ax)]
        assert armn is not None and len(armn) == len(inst), \
            f"G1: length mismatch on {ax} ({0 if armn is None else len(armn)} vs {len(inst)})"
        raw = float(np.max(np.abs(armn - inst)))
        # the instrument banks float32; round-trip the arm-N vector through float32 before the
        # bit-identity claim, exactly as the source run's own banked assert does
        rt = float(np.max(np.abs(armn.astype(np.float32).astype(np.float64) - inst)))
        g1[ax] = {"n": len(inst), "max_abs_diff_float64": raw, "max_abs_diff_after_f32": rt,
                  "bit_identical_after_f32": bool(rt == 0.0)}
        logln(f"[gate G1] armN(fixedorder:greedy_items,k=1) vs instrument(items,K=1) {ax}: "
              f"max|d| {raw:.3e}, after float32 round-trip {rt:.3e} "
              f"{'PASS' if rt == 0.0 else 'FAIL'}")
    res["gates"]["G1_cross_family"] = g1
    assert all(v["bit_identical_after_f32"] for v in g1.values()), \
        "G1 FAILED: the two harnesses do not agree per user -- cross-family pairing is invalid"

    # ------------------------------------------------------------------ G2 CROSS-RUN GATE
    ref = np.load(ACT1, allow_pickle=False)
    g2 = {}
    for tag, path, cons, spec in (("gtport", SOURCES["gtport"], "ours", "entropy0-static"),
                                  ("baldaopt", SOURCES["baldaopt"], "ours", "fixedorder:entropy0"),
                                  ("igcn", SOURCES["igcn"], "ours", "entropy0-static"),
                                  ("convrescore", SOURCES["convrescore"], "rescore",
                                   "entropy0-top16"),
                                  ("eddi3", SOURCES["eddi3"], "ours", "entropy0-static")):
        if not have_src.get(tag, True):
            continue
        z = np.load(path, allow_pickle=False)
        worst = 0.0
        ks = BUDGETS if tag != "gtport" else (1, 2, 4, 8)
        for k in ks:
            for ax in AXES:
                v = get(z, cons, spec, k, ax)
                if v is None:
                    continue
                worst = max(worst, float(np.max(np.abs(v - ref[f"ours|entropy0-static|{k}|{ax}"]))))
        g2[tag] = {"max_abs_per_user_diff": worst, "pass": bool(worst <= 0.0)}
        logln(f"[gate G2] {tag:<12} entropy0 vs banked act1: max per-user |d| = {worst:.3e} "
              f"{'PASS' if worst <= 0.0 else 'REFUSED'}")
    res["gates"]["G2_cross_run_entropy0"] = g2
    assert all(v["pass"] for v in g2.values()), "G2 FAILED -- cross-run rows not admissible"

    # ------------------------------------------------------- G3 BANKED-ARM GATE (TODO 1.3(b))
    # The eddi3 run re-derived `eddi_ig`, the two-outcome row ALREADY IN `tab:battery`. If the
    # re-run does not reproduce the banked per-user vectors exactly, the three-outcome numbers
    # produced beside it are not on the same ruler and nothing here may be differenced.
    # This is the "assert the banked number" contract check
    # (memory: banked-global-greedy-is-a-policy-A-artifact).
    if have_src.get("eddi3"):
        z3 = np.load(SOURCES["eddi3"], allow_pickle=False)
        g3, worst = {}, 0.0
        for k in BUDGETS:
            for ax in AXES:
                v = get(z3, "ours", "eddi_ig", k, ax)
                assert v is not None, f"eddi3 run has no eddi_ig|{k}|{ax} -- the gate cannot run"
                d = float(np.max(np.abs(v - ref[f"ours|eddi_ig|{k}|{ax}"])))
                g3[f"k{k}|{ax}"] = {"max_abs_per_user_diff": d,
                                    "mean_rerun": float(v.mean()),
                                    "mean_banked": float(ref[f"ours|eddi_ig|{k}|{ax}"].mean())}
                worst = max(worst, d)
        g3["worst"] = worst
        g3["pass"] = bool(worst <= 0.0)
        logln(f"[gate G3] eddi3 re-run of the BANKED two-outcome eddi_ig vs act1: max per-user "
              f"|d| = {worst:.3e} {'PASS' if worst <= 0.0 else 'REFUSED'}")
        res["gates"]["G3_banked_eddi_ig"] = g3
        assert g3["pass"], "G3 FAILED: the re-run does not reproduce the banked two-outcome row"

    # ------------------------------------------------------------------ bootstrap equivalence
    probe = S[("combined", 16, "full")] - get(np.load(SOURCES["act1"]), "ours", "glm_ucb", 16,
                                              "full")
    r = REF_BOOT_CI(probe, a.boot, a.seed)
    c = CHUNKED_BOOT_CI(probe, a.boot, a.seed)
    res["gates"]["bootstrap_equivalence"] = {"ref": list(r), "chunked": list(c),
                                             "identical": bool(r == c)}
    logln(f"[gate] chunked bootstrap vs reference: ref={r} chunked={c} "
          f"{'IDENTICAL' if r == c else 'DIFFER'}")
    assert r == c, "chunked bootstrap != reference -- refusing to report"
    flush()

    # ------------------------------------------------------------------ self-test
    st = {}
    for arm in STATICS:
        for ax in AXES:
            paired(S[(arm, 16, ax)], S[(arm, 16, ax)], a.boot, a.seed,
                   f"SELFTEST/{arm}/{ax}", st, "self")
    res["gates"]["self_test"] = st
    ok = all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for v in st.values())
    logln(f"[gate] self-test (arm minus itself == exactly 0): {'PASS' if ok else 'FAIL'}")
    assert ok
    flush()

    # ------------------------------------------------------------------ the comparisons
    zc = {tag: np.load(p, allow_pickle=False) for tag, p in SOURCES.items() if have_src[tag]}
    for name, src, cons, spec, kind, adm in ARMS + [(n, s, c, sp, k, "in-family")
                                                    for n, s, c, sp, k in HEURISTICS]:
        if src not in zc:
            logln(f"[arm] {name:<32} SKIPPED -- source '{src}' is absent")
            continue
        z = zc[src]
        pts = {}
        for K in BUDGETS:
            for ax in AXES:
                v = get(z, cons, spec, K, ax)
                if v is None:
                    continue
                pts.setdefault(str(K), {})[ax] = float(v.mean())
                for arm in STATICS:
                    lab = f"{name}|K{K}|{ax}|minus|{arm}"
                    paired(v, S[(arm, K, ax)], a.boot, a.seed, lab, res["vs_static"],
                           "cross-family (G1)" if adm == "in-family"
                           else "cross-family + cross-run (G1+G2 chain)")
        res["selector_points"][name] = {"type": kind, "source": os.path.basename(SOURCES[src]),
                                        "admission": adm, "points": pts}
        logln(f"[arm] {name:<22} {kind:<17} {len(pts)} budgets, "
              f"{sum(len(x) for x in pts.values()) * 3} comparisons")
        flush()

    # ------------------------------------------------------------------ the narrative claims
    # TODO 1.3(b): the head-to-head the mechanism claim actually rests on. Both variants are
    # differenced against the two-outcome row FROM THE SAME PROCESS (G3 proves that row is the
    # banked one), so this is an in-run paired comparison, not a cross-run one.
    if have_src.get("eddi3"):
        for spec, nm in (("eddi_ig3", "signed"), ("eddi_ig3_zero", "zero")):
            for K in BUDGETS:
                for ax in AXES:
                    v3 = get(zc["eddi3"], "ours", spec, K, ax)
                    v2 = get(zc["eddi3"], "ours", "eddi_ig", K, ax)
                    if v3 is None or v2 is None:
                        continue
                    paired(v3, v2, a.boot, a.seed,
                           f"three-outcome ({nm})|K{K}|{ax}|minus|two-outcome",
                           res["vs_static"], "in-run (same process)")
        flush()

    adaptive_named = ["GLM-UCB", "Thompson", "latent-only info-gain"]
    # the "(re-run)" row is the SAME arm as "latent-only info-gain" -- counting it would double-count
    # one selector in a claim that quantifies over selectors.
    all_adaptive = [n for n, _, _, _, k, _ in ARMS
                    if k == "adaptive" and not n.endswith("(re-run)")]
    for claim, arms_in in (("thirty_comparisons_three_named_adaptive", adaptive_named),
                           ("every_adaptive_selector_in_the_table", all_adaptive)):
        for comparator in STATICS:
            cells, lead, sep, bad = 0, 0, 0, []
            for name in arms_in:
                for K in BUDGETS:
                    for ax in AXES:
                        lab = f"{name}|K{K}|{ax}|minus|{comparator}"
                        if lab not in res["vs_static"]:
                            continue
                        r0 = res["vs_static"][lab]
                        cells += 1
                        lead += int(r0["delta"] < 0)
                        sep += int(r0["separates"] and r0["delta"] < 0)
                        if not (r0["separates"] and r0["delta"] < 0):
                            bad.append({"cell": lab, "delta": r0["delta"], "lo": r0["lo"],
                                        "hi": r0["hi"], "separates": r0["separates"]})
            res["narrative"][f"{claim}|vs_{comparator}"] = {
                "cells": cells, "static_leads": lead, "static_leads_and_separates": sep,
                "claim_holds": bool(cells == sep), "exceptions": bad}
            logln(f"[narrative] {claim} vs {comparator}: {sep}/{cells} lead AND separate "
                  f"({lead}/{cells} lead on the point estimate)")
    res["meta"]["seconds"] = round(time.time() - t00, 1)
    flush()
    logln(f"[done] {OUTP} in {(time.time() - t00) / 60:.1f}m")


if __name__ == "__main__":
    main()
