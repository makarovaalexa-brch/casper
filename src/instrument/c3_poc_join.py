r"""PAPER C / STAGE C3 -- PoC STAGE 5: THE UNBLINDED JOIN.

Design sheet: `docs/design/DS_C3_POC.md`. This is the ONLY stage licensed to open the ground-truth
key, and it is deliberately the last one: the packets (`c3_poc_extract.py`), the renders and the
judgments were produced by agents that never saw it, and the re-encode geometry
(`c3_poc_reencode.py`) is per-packet and label-free by construction.

WHY IT EXISTS (repair, 2026-08-24). The C3 PoC's headline table was assembled by hand from four JSON
files during the unblinding conversation. Every number in `docs/results/C3_POC.md` therefore had no
committed producer, no intervals, and no paired test -- on n=10 per arm, where a 2/10 and a 5/10 are
not distinguishable and a mean of ten cosines is dominated by two packets. This file recomputes the
whole table from the artifacts, and adds what n=10 makes mandatory:

  * WILSON 95% intervals on every judge rate (overall accuracy, the true-positive rate, and the
    twin-pair rate), so "11/30 correct" is read with its width;
  * the ten PAIRED real-minus-scrambled re-encode differences with an exact SIGN TEST -- paired,
    because each scrambled packet carries a specific real point and a specific real context, so the
    arms are matched and an unpaired comparison throws that away (memory
    `paired-test-kills-the-tie-claim`);
  * MEDIANS beside every mean, because the random arm's mean (+0.371) and median (+0.486) disagree
    by more than the gap the table was being read for.

READS: `_quarantine/KEY.json`, `reencode.json`, `renders.json`, `judgments.json`.
WRITES: `experiments/paperC/c3_poc/join.json`.
No model, no training, no test user, no LLM call. Pure arithmetic over banked artifacts.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:
    sys.argv += ["--threads", "2"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import json
import argparse
import numpy as np

from policy_net import log
from c0_geometry import PAPERC, git_block, wilson, sign_test

OUT_DIR = os.path.join(PAPERC, "c3_poc")
QUARANTINE = os.path.join(OUT_DIR, "_quarantine")
TYPES = ("real", "scrambled", "random")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def stat(x):
    """Mean AND median AND the range, for every cell. n=10 arms do not get a bare mean."""
    v = np.asarray([t for t in x if t is not None], np.float64)
    if not len(v):
        return {"n": 0}
    p = np.percentile(v, [0, 25, 50, 75, 100])
    return {"n": int(len(v)), "n_missing": int(len(x) - len(v)), "mean": float(v.mean()),
            "median": float(p[2]), "min": float(p[0]), "q1": float(p[1]), "q3": float(p[3]),
            "max": float(p[4]), "sd": float(v.std(ddof=1)) if len(v) > 1 else None}


def run(args):
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads)
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]}")

    key_path = os.path.join(QUARANTINE, f"KEY{args.tag}.json")
    KEY = load(key_path)
    RE = load(os.path.join(OUT_DIR, f"reencode{args.tag}.json"))
    RD = load(os.path.join(OUT_DIR, f"renders{args.render_tag}.json"))
    JG = load(os.path.join(OUT_DIR, f"judgments{args.judge_tag}.json"))

    k = {int(r["id"]): r for r in KEY["packets"]}
    re_ = {int(r["id"]): r for r in RE["rows"]}
    rd = {int(r["id"]): r for r in RD["renders"]}
    jg = {int(r["id"]): r for r in JG["judgments"]}
    ids = sorted(k)
    assert set(ids) == set(re_) == set(rd) == set(jg), (
        "the four artifacts do not cover the same packet ids: "
        f"key {len(k)}, reencode {len(re_)}, renders {len(rd)}, judgments {len(jg)}")
    n = len(ids)
    log(f"[join] {n} packets; types " + ", ".join(
        f"{t}={sum(1 for i in ids if k[i]['type'] == t)}" for t in TYPES))

    # ---- the per-packet joined table ----------------------------------------------------------
    rows = []
    for i in ids:
        rows.append({
            "id": i, "type": k[i]["type"], "point_kind": k[i]["point_kind"],
            "context_state_sid": k[i]["context_state_sid"],
            "point_source_state_sid": k[i]["point_source_state_sid"],
            "context_state_depth": k[i]["context_state_depth"],
            "context_state_coherence_mean_removed": k[i]["context_state_coherence_mean_removed"],
            "question": rd[i]["question"],
            "renderability_self": rd[i].get("renderability"),
            "judge_guess": jg[i]["guess"], "judge_confidence": jg[i].get("confidence"),
            "judge_answerability": jg[i].get("answerability"),
            "judge_correct": bool((jg[i]["guess"] == "real") == (k[i]["type"] == "real")),
            "encodable": re_[i]["encodable"],
            "n_concepts_selected": re_[i]["n_concepts_selected"],
            "cos_uniform": re_[i]["cos_uniform"], "cos_weighted": re_[i]["cos_weighted"],
            "anchor_cos_uniform": re_[i]["anchor_reference"]["cos_uniform"],
            "anchor_cos_weighted": re_[i]["anchor_reference"]["cos_weighted"]})

    # ---- the per-type table --------------------------------------------------------------------
    per_type = {}
    for t in TYPES:
        sub = [r for r in rows if r["type"] == t]
        per_type[t] = {
            "n": len(sub),
            "cos_uniform": stat([r["cos_uniform"] for r in sub]),
            "cos_weighted": stat([r["cos_weighted"] for r in sub]),
            "anchor_cos_uniform": stat([r["anchor_cos_uniform"] for r in sub]),
            "renderability_self": stat([r["renderability_self"] for r in sub]),
            "judge_answerability": stat([r["judge_answerability"] for r in sub]),
            "n_encodable": int(sum(bool(r["encodable"]) for r in sub)),
            "p_judged_real": wilson(sum(r["judge_guess"] == "real" for r in sub), len(sub)),
            "judge_accuracy": wilson(sum(r["judge_correct"] for r in sub), len(sub)),
            "question_tracks_its_anchor_reference": wilson(
                sum(1 for r in sub if r["cos_uniform"] is not None
                    and r["anchor_cos_uniform"] is not None
                    and r["cos_uniform"] >= r["anchor_cos_uniform"]), len(sub))}

    # ---- the judge, with intervals ---------------------------------------------------------------
    n_real = sum(1 for r in rows if r["type"] == "real")
    n_ctrl = n - n_real
    judge = {
        "overall_accuracy": wilson(sum(r["judge_correct"] for r in rows), n),
        "all_control_baseline": {"p": n_ctrl / n, "note": (
            "the accuracy of always answering 'control': the arms are 10 real vs 20 control, so "
            "chance is not 0.5 and the judge's 0.367 must be read against 0.667")},
        "true_positive_rate": wilson(
            sum(1 for r in rows if r["type"] == "real" and r["judge_guess"] == "real"), n_real),
        "false_positive_rate": wilson(
            sum(1 for r in rows if r["type"] != "real" and r["judge_guess"] == "real"), n_ctrl),
        "by_guess": {g: sum(1 for r in rows if r["judge_guess"] == g)
                     for g in sorted({r["judge_guess"] for r in rows})},
        "confusion": {t: {g: sum(1 for r in rows if r["type"] == t and r["judge_guess"] == g)
                          for g in sorted({r["judge_guess"] for r in rows})} for t in TYPES}}

    # ---- the twin pairs: each real point appears twice, as itself and inside its scrambled twin --
    real_by_src = {r["point_source_state_sid"]: r for r in rows if r["type"] == "real"}
    twins, pairs_ok = [], 0
    for r in rows:
        if r["type"] != "scrambled":
            continue
        m = real_by_src.get(r["point_source_state_sid"])
        if m is None:
            continue
        pairs_ok += 1
        twins.append({"real_id": m["id"], "scrambled_id": r["id"],
                      "point_source_state_sid": r["point_source_state_sid"],
                      "real_cos_uniform": m["cos_uniform"],
                      "scrambled_cos_uniform": r["cos_uniform"],
                      "diff_real_minus_scrambled": (
                          None if (m["cos_uniform"] is None or r["cos_uniform"] is None)
                          else m["cos_uniform"] - r["cos_uniform"]),
                      "judge_picked_real_and_rejected_twin": bool(
                          m["judge_guess"] == "real" and r["judge_guess"] != "real")})
    assert pairs_ok == n_real, (
        f"{pairs_ok} twin pairs matched, expected {n_real} -- the derangement should give every "
        "real point exactly one scrambled carrier")
    judge["twin_pairs_real_picked_and_twin_rejected"] = wilson(
        sum(t["judge_picked_real_and_rejected_twin"] for t in twins), len(twins))

    # ---- ★ the PAIRED re-encode comparison -------------------------------------------------------
    a = np.asarray([t["real_cos_uniform"] for t in twins], np.float64)
    b = np.asarray([t["scrambled_cos_uniform"] for t in twins], np.float64)
    paired = {
        "_comment": ("PAIRED because a scrambled packet carries a named real point into a named "
                     "other state's context: the two arms are matched on the point, so the "
                     "difference is within-point and an unpaired comparison discards that. NOTE "
                     "the structural limit already recorded in C3_POC.md -- under variant (a) the "
                     "anchors travel WITH the point, so this difference is expected to be ~0 by "
                     "construction and a null here is not evidence about rendering."),
        "n_pairs": len(twins),
        "real": stat(list(a)), "scrambled": stat(list(b)),
        "differences": [None if not np.isfinite(x) else float(x) for x in (a - b)],
        "diff": stat(list(a - b)),
        "sign_test": sign_test(a, b),
        "n_real_greater": int((a > b).sum()), "n_scrambled_greater": int((b > a).sum()),
        "n_exactly_equal": int((a == b).sum())}

    out = {"analysis": "paperC_C3_poc_stage5_join",
           "design_sheet": "docs/design/DS_C3_POC.md",
           "produced_by": "src/instrument/c3_poc_join.py",
           "citable": True, "test_touched": False, "split": "validation",
           "_comment": ("THE UNBLINDED JOIN -- the only stage that opens the key. Every rate here "
                        "carries a Wilson 95% interval and every arm carries a median beside its "
                        "mean, because n=10 per arm."),
           "git": GB,
           "inputs": {"key": os.path.relpath(key_path, _ROOT),
                      "reencode": f"reencode{args.tag}.json",
                      "renders": f"renders{args.render_tag}.json",
                      "judgments": f"judgments{args.judge_tag}.json",
                      "key_seed": KEY.get("seed"),
                      "reencode_git_rev": RE.get("git", {}).get("rev"),
                      "renderer": RD.get("renderer"), "judge": JG.get("judge")},
           "n_packets": n, "per_type": per_type, "judge": judge,
           "paired_real_vs_scrambled_reencode": paired,
           "twin_pairs": twins, "rows": rows}

    p = os.path.join(OUT_DIR, f"join{args.tag}.json")
    with open(p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(p + ".tmp", p)

    log("-" * 96)
    log("type       n  cos_uniform mean/med   anchor mean   render  answerab  P(judged real) [Wilson]")
    for t in TYPES:
        c = per_type[t]
        log(f"{t:<10} {c['n']:>2}  {c['cos_uniform']['mean']:+.4f}/{c['cos_uniform']['median']:+.4f}"
            f"      {c['anchor_cos_uniform']['mean']:+.4f}      "
            f"{c['renderability_self']['mean']:.2f}     {c['judge_answerability']['mean']:.2f}"
            f"     {c['p_judged_real']['p']:.2f} "
            f"[{c['p_judged_real']['lo']:.3f}, {c['p_judged_real']['hi']:.3f}]")
    j = judge["overall_accuracy"]
    log(f"[judge] accuracy {j['k']}/{j['n']} = {j['p']:.4f} Wilson [{j['lo']:.4f}, {j['hi']:.4f}] "
        f"vs an all-control baseline of {judge['all_control_baseline']['p']:.4f}")
    tp = judge["true_positive_rate"]
    log(f"[judge] true positives {tp['k']}/{tp['n']} Wilson [{tp['lo']:.4f}, {tp['hi']:.4f}]; "
        f"twin pairs {judge['twin_pairs_real_picked_and_twin_rejected']['k']}/"
        f"{judge['twin_pairs_real_picked_and_twin_rejected']['n']} Wilson "
        f"[{judge['twin_pairs_real_picked_and_twin_rejected']['lo']:.4f}, "
        f"{judge['twin_pairs_real_picked_and_twin_rejected']['hi']:.4f}]")
    st = paired["sign_test"]
    log(f"[paired] real - scrambled: median {paired['diff']['median']:+.4f} mean "
        f"{paired['diff']['mean']:+.4f}; real greater in {paired['n_real_greater']}/"
        f"{paired['n_pairs']}, sign test p = {st['p_two_sided']}")
    log(f"[done] wrote {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C / C3 PoC stage 5 -- the unblinded join")
    ap.add_argument("--tag", default="", help="packet/reencode/key set suffix")
    ap.add_argument("--render_tag", default="")
    ap.add_argument("--judge_tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with an uncommitted src/instrument (HARD RULE 10)")
    args = ap.parse_args()
    log("#" * 100)
    log("# PAPER C / C3 PoC -- STAGE 5 JOIN. The unblinding step: arithmetic over banked "
        "artifacts, no model, no LLM call, no test user.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
