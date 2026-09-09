"""A12 - is the concept channel's bank gain ANSWERABILITY, or BREADTH per answer?

Design sheet: docs/design/DS_A12_ANSWERABILITY_VS_BREADTH.md

The paper attributes the +0.01430 bank gain to answerability (16 concept questions collect 12.36
answers where 16 film questions collect 3.11). Both round-2 reviewers said the attribution is
unearned, because a concept answer also CARRIES more than an item answer: it summarises many films.
Two explanations, one number.

The code trace (2026-09-09) established that the reviewers' proposed control -- re-score item
questions under a WATCH-based answerability predicate -- IS NOT CONSTRUCTIBLE. In ML-25M as this
project processes it a rating is the only evidence of a watch (`signed_answers.py:6`: "'watched' =
rated at any band"), so "watched" and "rated" are the same event for a film and the item predicate
`i in lvl[u]` is already the widest one available. There is no knob.

What IS constructible, and is what this script does, is to DECOMPOSE the two channels' value into a
RATE and a PER-ANSWER WORTH, from artifacts already on disk:

    lift(arm, K)  =  answers(arm, K)  x  worth-per-answer(arm, K)

If the concept channel wins on rate and loses on worth, its advantage is answerability, not breadth,
and the reviewers' alternative hypothesis is not merely unsupported but contradicted. If it wins on
both, breadth is real and the paper must say so. The decomposition cannot be gamed after the fact:
both factors are read off the same banked run that produced the published number.

The load-bearing comparison is ANSWER-MATCHED: hold the number of answers collected roughly fixed
and ask which channel bought more with them. Answer counts are recomputed here per user, not taken
from the cohort means in the JSON, so the match can be made per user rather than on averages.

NOTHING IS REFITTED. The orders are the banked policy-B validation-selected sequences, the cohort is
the same 10,000 test users, and the gate below must reproduce the published contrast before any new
number is read.

Run:
    python src/instrument/a12_answerability_vs_breadth.py
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(_HERE) == "instrument" else None
sys.path.insert(0, _HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, contract_selftest,
                                    CERT_TOWER, CERT_CONC)
from group_greedy_genre import leak_assert

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
SRC_LABEL = "channel_orders_policyB_test"
LABEL = "a12_answerability_vs_breadth"
BUDGETS = (1, 2, 4, 8, 16)
N_BOOT = 10000
BOOT_SEED = 98765
ARMS = ("items", "concepts", "combined")

# The published cells this run must reproduce before it may report anything new.
GATE = {
    ("items", 16, "full"): 0.24470, ("items", 16, "tail"): 0.10171,
    ("concepts", 16, "full"): 0.24467, ("concepts", 16, "tail"): 0.11607,
    ("combined", 16, "full"): 0.25900, ("combined", 16, "tail"): 0.12370,
}
GATE_CONTRAST = {"combined_minus_items@K16|full": 0.01430,
                 "combined_minus_items@K16|tail": 0.02199}
GATE_TOL = 5e-5


def paired_bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired percentile bootstrap over per-user differences, resampling users (the cluster)."""
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "n_boot": n_boot, "seed": seed, "excludes_zero": bool(lo > 0 or hi < 0)}


def load_order(path, channel, n_pool, snapshot, conc_ckpt):
    b = json.load(open(path))
    assert b.get("citable") and not b.get("SANITY"), f"{path} is not a citable artifact"
    assert b["split"] == "validation" and not b["test_touched"], f"{path} was not built on validation"
    assert b["mask_policy"] == "B", f"{path} is not a policy-B build"
    assert b["objective"] == "full", f"{path} objective is {b['objective']}"
    assert b["n_pool"] == n_pool, f"{path} pool is {b['n_pool']}, expected {n_pool}"
    assert b["tower"] == os.path.basename(snapshot) and b["concept_ckpt"] == os.path.basename(conc_ckpt)
    seq = [(int(a), int(c)) for a, c in b["sequence"]]
    assert len(seq) >= max(BUDGETS), f"{path} has {len(seq)} picks"
    if channel in ("items", "concepts"):
        want = 0 if channel == "items" else 1
        assert all(q[0] == want for q in seq), f"{path} is not a {channel}-only order"
    return seq[:max(BUDGETS)], b


def per_user_answered(E, seq, rows):
    """Per-user count of answerable questions in `seq`, split by channel.

    Item side: the answer oracle `E['lvl'][r]`, i.e. the user rated that film at any band
    (strategy_channel_suite.py:522 uses the same membership test).
    Concept side: `E['foldable']`, which is (n_c >= 2) AND (band != REFUSE, i.e. e_c >= 1.5)
    (signed_answers.py:130,138 -> strategy_channel_suite.py:461).
    """
    ridx = np.asarray(rows, np.int64)
    item_ids = [int(i) for ch, i in seq if ch == 0]
    conc_ids = [int(c) for ch, c in seq if ch == 1]
    lvl = E["lvl"]
    ia = (np.array([sum(1 for i in item_ids if i in lvl[r]) for r in rows], np.float64)
          if item_ids else np.zeros(len(rows)))
    ca = (np.asarray(E["foldable"][np.ix_(ridx, np.asarray(conc_ids, np.int64))].sum(1),
                     np.float64).ravel()
          if conc_ids else np.zeros(len(rows)))
    return ia, ca


def main():
    t00 = time.time()
    src_json = os.path.join(OUTDIR, f"{SRC_LABEL}.json")
    src_npz = os.path.join(OUTDIR, f"{SRC_LABEL}_per_user.npz")
    assert os.path.exists(src_json) and os.path.exists(src_npz), "banked channel orders missing"
    src = json.load(open(src_json))
    Z = np.load(src_npz)
    assert src["split"] == "test" and src["n_test_users"] == 10000
    log(f"[src] {SRC_LABEL}: {src['n_test_users']} users, budgets {src['budgets']}")

    # ---------- GATE 1: the banked per-user vectors reproduce the published cells ----------
    fails = []
    for (arm, K, axis), want in GATE.items():
        got = float(Z[f"{arm}|{K}|{axis}"].mean())
        if abs(got - want) > GATE_TOL:
            fails.append(f"{arm}|{K}|{axis}: {got:.5f} vs published {want:.5f}")
    d_full = Z["combined|16|full"] - Z["items|16|full"]
    d_tail = Z["combined|16|tail"] - Z["items|16|tail"]
    for key, want, d in (("combined_minus_items@K16|full", GATE_CONTRAST["combined_minus_items@K16|full"], d_full),
                         ("combined_minus_items@K16|tail", GATE_CONTRAST["combined_minus_items@K16|tail"], d_tail)):
        got = float(d.mean())
        if abs(got - want) > GATE_TOL:
            fails.append(f"{key}: {got:.5f} vs published {want:.5f}")
    if fails:
        raise SystemExit("GATE FAILED, nothing new may be read:\n  " + "\n  ".join(fails))
    log("[gate] PASSED: banked per-user vectors reproduce every published cell to 5 dp")

    # ---------- answer counts, recomputed per user ----------
    orders, srcs = {}, {}
    for arm, fn, npool in (("items", "greedy_static_policyB_items.json", 500),
                           ("concepts", "greedy_static_policyB_concepts.json", 500),
                           ("combined", "greedy_static_policyB_combined.json", 1000)):
        path = os.path.join(OUTDIR, fn)
        orders[arm], srcs[arm] = load_order(path, arm, npool, CERT_TOWER, CERT_CONC)
        log(f"[order] {arm:<9} {fn}")

    E = build_greedy_env(CERT_TOWER, "test", CERT_CONC, f"{LABEL}(test)")
    rows = E["rows"]
    contract_selftest(E, rows)
    leak_assert(E, rows, "test")
    assert len(rows) == 10000, f"test cohort is {len(rows)} users"
    set_active_rows(E, rows)
    assert np.array_equal(np.asarray(rows, np.int64), Z["__rows__"]), \
        "row order differs from the banked npz; per-user pairing would be wrong"

    ans = {}
    for arm in ARMS:
        for K in BUDGETS:
            ia, ca = per_user_answered(E, orders[arm][:K], rows)
            ans[(arm, K)] = {"items": ia, "concepts": ca, "total": ia + ca}

    # ---------- GATE 2: recomputed cohort answer means match the banked curves ----------
    fails = []
    for arm in ARMS:
        for K in BUDGETS:
            for ch, key in (("items", "items_answered"), ("concepts", "concepts_answered")):
                got = float(ans[(arm, K)][ch].mean())
                want = float(src["test_curves"][arm][str(K)][key])
                if abs(got - want) > 1e-6:
                    fails.append(f"{arm}|K{K}|{key}: {got:.4f} vs banked {want:.4f}")
    if fails:
        raise SystemExit("ANSWER-COUNT GATE FAILED:\n  " + "\n  ".join(fails))
    log("[gate] PASSED: recomputed per-user answer counts reproduce the banked cohort means")

    m0f, m0t = Z["m0|full"], Z["m0|tail"]
    out = {"label": LABEL, "citable": True, "split": "test", "n_users": 10000,
           "design_sheet": "docs/design/DS_A12_ANSWERABILITY_VS_BREADTH.md",
           "source_artifact": f"{SRC_LABEL}.json",
           "tower": src["tower"], "concept_ckpt": src["concept_ckpt"],
           "mask_policy": src["mask_policy"], "budgets": list(BUDGETS),
           "inference": f"paired percentile bootstrap, {N_BOOT} resamples clustered over users, "
                        f"seed {BOOT_SEED}",
           "gate": "PASSED: published cells and cohort answer means both reproduced",
           "not_constructible": (
               "A watch-based item answerability predicate does not exist in this pipeline: in "
               "ML-25M as processed here a rating is the only evidence of a watch, so 'watched' and "
               "'rated' are the same event for a film (signed_answers.py:6). The item predicate is "
               "already the widest available (all-bands rated). The reviewers' proposed control is "
               "therefore not implementable, and this decomposition is run in its place."),
           "rate_vs_worth": {}, "answer_matched": {}, "curves": {}}

    # ---------- the decomposition: rate x worth ----------
    log("")
    log("RATE x WORTH  (lift over the answer-free reference, per answer collected)")
    log(f"{'arm':10s} {'K':>3s} {'full':>9s} {'lift':>9s} {'answers':>8s} {'per-answer':>11s}"
        f" {'tail lift':>10s} {'per-answer':>11s}")
    for arm in ARMS:
        for K in BUDGETS:
            f = Z[f"{arm}|{K}|full"]; t = Z[f"{arm}|{K}|tail"]
            a = ans[(arm, K)]["total"]
            lf, lt = float((f - m0f).mean()), float((t - m0t).mean())
            na = float(a.mean())
            wf, wt = (lf / na if na > 0 else float("nan")), (lt / na if na > 0 else float("nan"))
            out["rate_vs_worth"][f"{arm}|K{K}"] = {
                "full": float(f.mean()), "tail": float(t.mean()),
                "lift_full": lf, "lift_tail": lt, "answers": na,
                "worth_per_answer_full": wf, "worth_per_answer_tail": wt}
            log(f"{arm:10s} {K:>3d} {float(f.mean()):9.5f} {lf:9.5f} {na:8.3f} {wf:11.5f}"
                f" {lt:10.5f} {wt:11.5f}")

    # ---------- the load-bearing comparison: answer-matched ----------
    # Items reach 3.108 answers only at K=16. Concepts pass that between K=2 (1.751) and K=4 (3.568).
    # Both neighbours are reported, so the reading does not depend on an interpolation.
    log("")
    log("ANSWER-MATCHED: what did each channel buy with a comparable number of answers?")
    pairs = [("items", 16, "concepts", 4), ("items", 16, "concepts", 2),
             ("items", 8, "concepts", 2), ("items", 4, "concepts", 1)]
    for a1, k1, a2, k2 in pairs:
        n1 = float(ans[(a1, k1)]["total"].mean()); n2 = float(ans[(a2, k2)]["total"].mean())
        rec = {"arm_a": f"{a1}@K{k1}", "answers_a": n1, "arm_b": f"{a2}@K{k2}", "answers_b": n2}
        for axis in ("full", "tail"):
            d = Z[f"{a1}|{k1}|{axis}"].astype(np.float64) - Z[f"{a2}|{k2}|{axis}"].astype(np.float64)
            rec[axis] = paired_bootstrap(d)
        out["answer_matched"][f"{a1}@K{k1}_minus_{a2}@K{k2}"] = rec
        s = "*" if rec["full"]["excludes_zero"] else " "
        log(f"  {a1}@K{k1} ({n1:.3f} answers) - {a2}@K{k2} ({n2:.3f} answers): "
            f"full {rec['full']['mean']:+.5f}{s} [{rec['full']['lo']:+.5f},{rec['full']['hi']:+.5f}]"
            f"  tail {rec['tail']['mean']:+.5f} "
            f"[{rec['tail']['lo']:+.5f},{rec['tail']['hi']:+.5f}]")

    # per-user exact match: users whose concept answers at K=4 bracket their item answers at K=16
    ia16 = ans[("items", 16)]["total"]
    ca4 = ans[("concepts", 4)]["total"]
    band = np.abs(ca4 - ia16) <= 0.5
    if band.sum() >= 200:
        rec = {"n": int(band.sum()),
               "definition": "users whose concept answers at K=4 are within 0.5 of their item "
                             "answers at K=16, i.e. matched per user rather than on the cohort mean"}
        for axis in ("full", "tail"):
            d = (Z[f"items|16|{axis}"].astype(np.float64)
                 - Z[f"concepts|4|{axis}"].astype(np.float64))[band]
            rec[axis] = paired_bootstrap(d)
        out["answer_matched"]["per_user_matched_band"] = rec
        log(f"  per-user matched band (n={rec['n']}): "
            f"full {rec['full']['mean']:+.5f} [{rec['full']['lo']:+.5f},{rec['full']['hi']:+.5f}]")

    for arm in ARMS:
        out["curves"][arm] = {str(K): {"answers_items": float(ans[(arm, K)]["items"].mean()),
                                       "answers_concepts": float(ans[(arm, K)]["concepts"].mean())}
                              for K in BUDGETS}

    out["seconds"] = time.time() - t00
    op = os.path.join(OUTDIR, f"{LABEL}.json")
    json.dump(out, open(op, "w"), indent=2)
    np.savez_compressed(os.path.join(OUTDIR, f"{LABEL}_per_user.npz"),
                        __rows__=np.asarray(rows, np.int64),
                        **{f"ans|{arm}|{K}": ans[(arm, K)]["total"].astype(np.float32)
                           for arm in ARMS for K in BUDGETS})
    log(f"\n[done] {op}  ({out['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
