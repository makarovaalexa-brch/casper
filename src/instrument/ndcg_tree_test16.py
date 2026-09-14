r"""THE TEST LOOK — label `ndcgtree16test`. Author-authorised, 2026-08-12, ONE look.

AUTHOR SAID, verbatim: "add 16th step, run on test. we are changing comparator, we are NOT changing
this tree, don't see a problem".

WHAT THIS DOES, IN ORDER
  1. Extends each of the 15 leaf orders from 15 to 16 questions by RESUMING its banked CELF heap
     (the tree structure is NOT regrown -- same 22 nodes, same splitters, same leaves, same depth 5).
     Every leaf asserts its first 15-depth picks are UNCHANGED element-wise; a faithful resume must
     reproduce its own prefix.
  2. Re-measures on VALIDATION at K in {1,2,4,8,16}.
  3. Routes the 10,000 canonical TEST users through the frozen tree by their OWN behavioural answers
     to the splitter questions -- the deployable path -- and applies each user's leaf order.
  4. Reports validation and test side by side, per leaf, with the SHRINKAGE column, because that is
     the headline: the genre arm shrank ~2.5x val->test (+0.0205 -> +0.0082) with LARGER groups than
     these leaves.

WHY MEASURING A FROZEN TREE AGAINST A SECOND COMPARATOR LATER IS NOT TUNING ON TEST. The tree was
grown entirely on validation and is frozen: its 22 nodes and their splitters cannot change. The
decision to rebuild the static comparator under policy B is already committed with its reasoning
(commit ba51e8a), timestamped before any test number exists. Swapping in that comparator afterwards
changes what the tree is compared against, not what the tree IS.

THE COMPARATOR HERE IS THE POLICY-A-SELECTED ORDER, AND IT IS LABELLED AS SUCH EVERYWHERE. The
`combined` arm of greedy_static_i26c.json was SELECTED under the retired policy-A candidate mask
(confirmed in docs/results/MASK_POLICY_B_MIGRATION.md: its intercept_eval 0.13139/0.02343 IS the
policy-A bridge, against policy B's 0.15855/0.02819). Its SEQUENCE is used and re-scored in-run under
policy B, so the comparison is question-for-question at equal K -- but its ORDER was chosen on a
different ruler, so the K=1 difference carries a +0.00093 selection-protocol offset that is NOT
conditioning. The clean policy-B comparator is paused mid-rebuild with 7 picks banked and resumes
after this look; these numbers must never be quoted as if that rebuild had happened.

CONTRACT. Full AND tail NDCG@10, learned decoder bias, NO FLOOR, policy-B candidate pool, all 10,000
canonical test users, nothing sampled (HARD RULE 1). Paired PERCENTILE bootstrap over per-user
differences, 10,000 resamples, seed 98765 -- never the unpaired 1.96*sqrt(2)*SE form. Leak asserted
on BOTH cohorts: no user may have a held-out target inside the routing lookup or the fold.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import gc
import json
import time
import shutil
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, greedy_certified,
                                    genv_eval, genv_answered, genv_mean, contract_selftest,
                                    CERT_TOWER, CERT_CONC)
from run_battery_phaseA import bootstrap_ci
from ndcg_tree import build_pool, split_rows, leak_assert, read_global, BRANCHES, HEADLINE_ARM

# Per-dataset results dir, single source of truth (run_battery_phaseA keys it off
# CASPER_DATASET). Was hardcoded here, so a Goodreads run would have written on top of the
# certified ML-25M tree/greedy artifacts -- HARD RULE 11e.
from run_battery_phaseA import OUTDIR                  # noqa: E402
# The tree artifacts carry the label they were grown under. ml-25m grew L=15 as "ndcgtree" and
# extended to 16; a tree grown directly at L=16 (--label ndcgtree16) writes those names instead.
# Take whichever exists so either provenance feeds this step.
_TJ = [os.path.join(OUTDIR, n) for n in ("ndcgtree.json", "ndcgtree16.json")]
TREE_JSON = next((p for p in _TJ if os.path.exists(p)), _TJ[0])
_TH = [os.path.join(OUTDIR, n) for n in ("ndcgtree_heaps", "ndcgtree16_heaps")]
TREE_HEAPS = next((p for p in _TH if os.path.exists(p)), _TH[0])
GLOBAL_JSON = os.path.join(OUTDIR, "greedy_static_i26c.json")

LABEL = "ndcgtree16test"
TOTAL_L = 16
BUDGETS = (1, 2, 4, 8, 16)
BOOT, BOOT_SEED = 10_000, 98765
ENV_LABEL_VAL = "ndcgtree(val)"          # MUST match the label the tree's heap signatures carry
ENV_LABEL_TEST = "ndcgtree16test(test)"

COMPARATOR_LABEL = "combined_POLICY_A_SELECTED"
COMPARATOR_NOTE = (
    "THE POLICY-A-SELECTED COMPARATOR. This is the `combined` arm of greedy_static_i26c.json, whose "
    "ORDER was selected under the retired policy-A candidate mask (docs/results/"
    "MASK_POLICY_B_MIGRATION.md records its intercept_eval 0.13139/0.02343 as the policy-A bridge; "
    "policy B is 0.15855/0.02819). Its SEQUENCE is re-scored in-run under policy B here, so the "
    "comparison is question-for-question at equal K -- but because its order was CHOSEN on a "
    "different ruler, the K=1 difference carries a selection-protocol offset (+0.00093 full on "
    "validation) that is NOT conditioning and that sits inside every K>=2 number. The clean policy-B "
    "comparator is paused mid-rebuild with 7 of 16 picks banked. DO NOT quote these gains as if that "
    "rebuild had happened.")
NEGATIVE_LEAF_NOTE = (
    "A NEGATIVE leaf is a RESULT, not a bug. A leaf must carry its ancestors' questions, which were "
    "chosen for its PARENTS' cohorts, so the static order is NOT inside a leaf's search space and a "
    "leaf can legitimately score below it.")


def leaf_rows(E, nodes, nid, rows_all):
    """Route a cohort to one leaf by walking the FROZEN tree with the banked splitters and the
    harness's own behavioural answer rule. This is the deployable path: a user is branched by the
    answer they would actually give, and an unanswerable question sends them down DID-NOT-ANSWER
    having still spent the turn."""
    parts = nid.split(".")
    assert parts[0] == "root", nid
    rows, cur = list(rows_all), "root"
    for br in parts[1:]:
        assert br in BRANCHES, f"bad branch '{br}' in {nid}"
        q = tuple(int(x) for x in nodes[cur]["splitter"])
        kids, _ar, _z = split_rows(E, rows, q)
        rows = kids[br]
        cur = cur + "." + br
    assert cur == nid
    return rows


def measure(E, leaves, comp_seq, budgets, bank, tag):
    """Per-leaf and cohort-weighted curves on ONE cohort. Every comparison is the leaf's first K
    questions against the comparator's first K, with K asserted equal on both sides."""
    for rec in leaves:
        nid = rec["nid"]
        rows = rec[f"rows_{tag}"]
        if not rows:
            rec[f"curve_{tag}"] = {}
            continue
        own = [(int(a), int(b)) for a, b in rec["own_sequence"]]
        set_active_rows(E, rows); E["memo"] = {}
        curve = {}
        for K in budgets:
            a, b = own[:K], [tuple(q) for q in comp_seq[:K]]
            assert len(a) == len(b) == K, f"UNEQUAL BUDGET at K={K}: {len(a)} vs {len(b)}"
            of, ot = genv_eval(E, a, rows)
            gf, gt = genv_eval(E, b, rows)
            ia, ca = genv_answered(E, a, rows)
            bank[f"{tag}|{nid}|{K}|own_full"] = of.astype(np.float32)
            bank[f"{tag}|{nid}|{K}|own_tail"] = ot.astype(np.float32)
            bank[f"{tag}|{nid}|{K}|cmp_full"] = gf.astype(np.float32)
            bank[f"{tag}|{nid}|{K}|cmp_tail"] = gt.astype(np.float32)
            curve[str(K)] = {"tree_full@10": float(np.nanmean(of)),
                             "tree_tail@10": float(np.nanmean(ot)),
                             "cmp_full@10": float(np.nanmean(gf)),
                             "cmp_tail@10": float(np.nanmean(gt)),
                             "diff_full@10": float(np.nanmean(of - gf)),
                             "diff_tail@10": float(np.nanmean(ot - gt)),
                             "items_answered": round(ia, 3), "concepts_answered": round(ca, 3),
                             "set_overlap": len(set(a) & set(b))}
        bank[f"{tag}|{nid}|rows"] = np.asarray(rows, np.int64)
        rec[f"curve_{tag}"] = curve
        log(f"[{tag}/{nid}] n={len(rows)} | @K{budgets[-1]} tree "
            f"{curve[str(budgets[-1])]['tree_full@10']:.5f} vs cmp "
            f"{curve[str(budgets[-1])]['cmp_full@10']:.5f} "
            f"({curve[str(budgets[-1])]['diff_full@10']:+.5f} full, "
            f"{curve[str(budgets[-1])]['diff_tail@10']:+.5f} tail)")
    weighted = {}
    for K in budgets:
        df = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_full"]
                             - bank[f"{tag}|{r['nid']}|{K}|cmp_full"]
                             for r in leaves if r[f"rows_{tag}"]])
        dt = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_tail"]
                             - bank[f"{tag}|{r['nid']}|{K}|cmp_tail"]
                             for r in leaves if r[f"rows_{tag}"]])
        tf = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_full"]
                             for r in leaves if r[f"rows_{tag}"]])
        tt = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_tail"]
                             for r in leaves if r[f"rows_{tag}"]])
        cf = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|cmp_full"]
                             for r in leaves if r[f"rows_{tag}"]])
        ct = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|cmp_tail"]
                             for r in leaves if r[f"rows_{tag}"]])
        mf, cif, nf = bootstrap_ci(df, n_boot=BOOT, seed=BOOT_SEED)
        mt, cit, _ = bootstrap_ci(dt, n_boot=BOOT, seed=BOOT_SEED)
        # ABSOLUTE cohort-weighted NDCG on BOTH arms, not only the difference. These are what let a
        # reader judge the "the current static is weak" assumption for themselves: if the tree's
        # absolute sits close to the comparator's, a weak comparator cannot be hiding much; if it is
        # far above, the comparator's weakness matters more. Weighted by cohort-side leaf sizes,
        # which is automatic here because the per-user vectors are concatenated across leaves.
        weighted[str(K)] = {
            "tree_full@10": float(np.nanmean(tf)), "tree_tail@10": float(np.nanmean(tt)),
            "cmp_full@10": float(np.nanmean(cf)), "cmp_tail@10": float(np.nanmean(ct)),
            "diff_full@10": mf, "ci95_full": list(cif),
            "diff_tail@10": mt, "ci95_tail": list(cit),
            "n_users": int(nf),
            "separates_full": bool(cif[0] > 0 or cif[1] < 0),
            "separates_tail": bool(cit[0] > 0 or cit[1] < 0)}
    return weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()

    t00 = time.time()
    OUT = os.path.join(OUTDIR, f"{args.label}.json")
    PARTIAL = os.path.join(OUTDIR, f"{args.label}_partial.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}_per_user.npz")
    HEAPDIR = os.path.join(OUTDIR, f"{args.label}_heaps")
    os.makedirs(HEAPDIR, exist_ok=True)

    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("tree json", TREE_JSON), ("tree heaps", TREE_HEAPS),
                  ("global greedy json", GLOBAL_JSON)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"

    tree = json.load(open(TREE_JSON))
    assert tree["citable"] and not tree["SANITY"] and tree["split"] == "validation"
    # The 15 is legacy: ml-25m grew its tree under a 15-question budget grid and this step exists
    # to reconcile it with the 16-question comparison. A tree grown directly at L=16 is a
    # stronger object -- it actually spent 16 questions rather than carrying a bolted-on 16th --
    # so accept either, and take the leaf tail length from the tree instead of assuming 15.
    _TL = int(tree["total_L"])
    assert _TL in (15, TOTAL_L) and tree["tower"] == os.path.basename(args.snapshot)
    nodes, leaves_meta = tree["nodes"], tree["leaf_table"]
    log("#" * 110)
    log(f"# THE TEST LOOK (author-authorised, ONE look). Tree FROZEN: {tree['n_nodes']} nodes, "
        f"{tree['n_leaves']} leaves, depth {tree['depth_reached']} -- NOT regrown. Budget "
        f"{tree['total_L']} -> {TOTAL_L} by extending each leaf's tail from its banked CELF heap.")
    log("# " + COMPARATOR_NOTE)
    log("#" * 110)

    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {"leaves": {}}

    def flush():
        with open(PARTIAL + ".tmp", "w") as fh:
            json.dump(part, fh, indent=2, default=float)
        os.replace(PARTIAL + ".tmp", PARTIAL)

    bank = {}
    # ================================================================= VALIDATION
    Ev = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, ENV_LABEL_VAL)
    assert Ev["contract_tag"] == f"genv:{ENV_LABEL_VAL}:validation", Ev["contract_tag"]
    rows_val = Ev["rows"]
    contract_selftest(Ev, rows_val)
    assert len(rows_val) == tree["n_cohort"]
    leak_assert(Ev, rows_val, "validation")
    set_active_rows(Ev, rows_val); Ev["memo"] = {}
    icpt_val = genv_mean(Ev, [], rows_val)
    log(f"[intercept/val] m=0, NO questions asked, policy-B pool: {icpt_val[0]:.5f} full / "
        f"{icpt_val[1]:.5f} tail -- the floor both arms are measured against")
    pool, item_pool, conc_pool = build_pool(Ev, args.n_items, args.n_concepts)
    assert len(item_pool) == tree["n_items_pool"] and len(conc_pool) == tree["n_concepts_pool"]
    comp, _g = read_global(GLOBAL_JSON, TOTAL_L)
    cmp_seq = comp[HEADLINE_ARM]["sequence"][:TOTAL_L]
    assert len(cmp_seq) == TOTAL_L
    log(f"[comparator] {COMPARATOR_LABEL}: {[[int(a), int(b)] for a, b in cmp_seq]}")

    # ---- extend every leaf 15 -> 16 by resuming its banked heap ---------------------------------
    log("=" * 110)
    log(f"[extend] {len(leaves_meta)} leaves, 15 -> {TOTAL_L} questions, resuming banked CELF heaps")
    leaves = []
    for meta in leaves_meta:
        nid, depth = meta["nid"], int(meta["depth"])
        rows = leaf_rows(Ev, nodes, nid, rows_val)
        assert len(rows) == meta["n_users"], \
            f"{nid}: routed {len(rows)} val users, tree recorded {meta['n_users']}"
        prefix = [(int(a), int(b)) for a, b in nodes[nid]["path"]]
        pool_n = [q for q in pool if q not in set(prefix)]
        old_steps = [(int(a), int(b)) for a, b in nodes[nid]["steps"]]
        # An INHERITING leaf records depth+1 but keeps its PARENT's path, so its tail is
        # _TL - len(path), not _TL - depth. ml-25m never grew deep enough to create one.
        assert len(old_steps) == _TL - len(prefix)
        src = os.path.join(TREE_HEAPS, f"node_{nid}.json")
        dst = os.path.join(HEAPDIR, f"node_{nid}.json")
        assert os.path.exists(src), f"leaf heap missing: {src}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)            # HARD RULE 11(e): never share an output path
        bnev = int(json.load(open(dst))["nev"])
        set_active_rows(Ev, rows); Ev["memo"] = {}; Ev["n_evals"] = 0; Ev["t_evals"] = 0.0
        t0 = time.time()
        want = TOTAL_L - depth
        seq, gains, nev = greedy_certified(Ev, rows, pool_n, want, args.objective, f"{nid}/ext",
                                           prefix=prefix, heap_path=dst)
        assert len(seq) == want, f"{nid}: {len(seq)} picks, wanted {want}"
        assert nev >= bnev, (f"{nid}: did NOT resume ({nev} evals vs {bnev} banked); a cold reseed "
                             f"restarts the counter")
        assert [tuple(q) for q in seq[:len(old_steps)]] == old_steps, \
            (f"{nid}: RESUME INFIDELITY -- first {len(old_steps)} picks CHANGED.\n"
             f"  banked  {old_steps}\n  resumed {[tuple(q) for q in seq[:len(old_steps)]]}")
        own = prefix + [(int(a), int(b)) for a, b in seq]
        assert len(own) == TOTAL_L
        rec = {"nid": nid, "depth": depth, "n_val": len(rows), "rows_val": rows,
               "path": [[int(a), int(b)] for a, b in prefix],
               "own_sequence": [[int(a), int(b)] for a, b in own],
               "new_question": [int(seq[-1][0]), int(seq[-1][1])],
               "new_question_gain": float(gains[-1]), "prefix_unchanged": True,
               "n_evals_new": int(nev - bnev),
               "minutes_extend": round((time.time() - t0) / 60, 2),
               "genre_face_validity": meta["genre_face_validity"],
               "val_gain_at15_vs_policyA": meta["diff_full@10"]}
        leaves.append(rec)
        part["leaves"][nid] = {k: v for k, v in rec.items() if not k.startswith("rows_")}
        flush()
        log(f"[extend {nid}] n={len(rows)} d={depth}: +1 = "
            f"{'concept' if seq[-1][0] else 'item'} {seq[-1][1]} gain {gains[-1]:+.5f} | prefix "
            f"UNCHANGED | {nev - bnev} new evals ({rec['minutes_extend']:.1f}m)")

    log("=" * 110)
    log("[validation] measuring the extended orders at K in %s" % (list(BUDGETS),))
    w_val = measure(Ev, leaves, cmp_seq, BUDGETS, bank, "val")
    np.savez_compressed(NPZ, **bank)
    for r in leaves:
        r["val_gain_at16"] = r["curve_val"][str(TOTAL_L)]["diff_full@10"]
        r["val_gain_at16_tail"] = r["curve_val"][str(TOTAL_L)]["diff_tail@10"]
    flush()

    # free the validation environment before building the test one (two 10k cohorts at once is
    # ~6 GB of tower state on a 16 GB box with ~6.5 GB free -- HARD RULE 11(d)/(c))
    del Ev
    gc.collect()
    log("[mem] validation environment released before building the test cohort")

    # ================================================================= TEST (ONE LOOK)
    log("=" * 110)
    log("[test] building the TEST environment -- the ONE authorised look")
    Et = build_greedy_env(args.snapshot, "test", args.concept_ckpt, ENV_LABEL_TEST)
    rows_test = Et["rows"]
    contract_selftest(Et, rows_test)
    leak_assert(Et, rows_test, "test")
    log(f"[test] {len(rows_test)} canonical TEST users")
    set_active_rows(Et, rows_test); Et["memo"] = {}
    icpt_test = genv_mean(Et, [], rows_test)
    # The m=0 intercept has never been printed for the test cohort. It is what puts both arms in
    # proportion: it is what a user gets for answering NOTHING, so tree-minus-intercept and
    # static-minus-intercept are what the 16-question interview actually buys on each arm.
    log(f"[intercept/test] m=0, NO questions asked, policy-B pool: {icpt_test[0]:.5f} full / "
        f"{icpt_test[1]:.5f} tail -- what the whole interview is measured against")

    # route every test user through the FROZEN tree by their own behavioural answers
    assigned = 0
    for rec in leaves:
        rec["rows_test"] = leaf_rows(Et, nodes, rec["nid"], rows_test)
        rec["n_test"] = len(rec["rows_test"])
        assigned += rec["n_test"]
    assert assigned == len(rows_test), \
        f"routing lost users: {assigned} assigned vs {len(rows_test)} test users"
    seen = set()
    for rec in leaves:
        s = set(rec["rows_test"])
        assert not (seen & s), "test routing is not disjoint"
        seen |= s
    log(f"[test] routed {assigned}/{len(rows_test)} users into {len(leaves)} leaves, "
        f"disjoint and covering (HARD RULE 1: nothing dropped)")
    for r in sorted(leaves, key=lambda r: -r["n_val"]):
        log(f"[test/route] {r['nid']:<48} val n={r['n_val']:5d}  test n={r['n_test']:5d}")

    w_test = measure(Et, leaves, cmp_seq, BUDGETS, bank, "test")
    np.savez_compressed(NPZ, **bank)
    for r in leaves:
        c = r.get("curve_test", {}).get(str(TOTAL_L))
        r["test_gain_at16"] = c["diff_full@10"] if c else None
        r["test_gain_at16_tail"] = c["diff_tail@10"] if c else None
        r["test_full@10_at16"] = c["tree_full@10"] if c else None
        r["test_tail@10_at16"] = c["tree_tail@10"] if c else None
        r["shrinkage_val_to_test"] = (
            (r["val_gain_at16"] / r["test_gain_at16"])
            if (c and r["test_gain_at16"] not in (None, 0.0)) else None)
    flush()

    # ---- size vs gain, on TEST ------------------------------------------------------------------
    ok = [r for r in leaves if r["n_test"] > 0]
    nn = np.array([r["n_test"] for r in ok], float)
    gt = np.array([r["test_gain_at16"] for r in ok])
    gv = np.array([r["val_gain_at16"] for r in ok])
    try:
        from scipy.stats import spearmanr
        rho_t, p_t = spearmanr(nn, gt)
        rho_v, p_v = spearmanr(np.array([r["n_val"] for r in ok], float), gv)
    except Exception:
        rho_t = p_t = rho_v = p_v = float("nan")
    big = nn >= 700
    size_gain = {
        "test_spearman_rho": float(rho_t), "test_p_value": float(p_t),
        "val_spearman_rho": float(rho_v), "val_p_value": float(p_v),
        "test_weighted_gain_leaves_ge_700": float((nn[big] * gt[big]).sum() / max(nn[big].sum(), 1)),
        "test_weighted_gain_leaves_lt_700": float((nn[~big] * gt[~big]).sum() / max(nn[~big].sum(), 1)),
        "n_test_users_ge_700": int(nn[big].sum()), "n_test_users_lt_700": int(nn[~big].sum()),
        "interpretation": (
            "On validation the relationship was Spearman -0.55 (p=0.035): small leaves gained most, "
            "which is the overfitting signature the genre experiment was built to detect. Whether it "
            "PERSISTS on test is what decides how much of the validation gain is real. A negative rho "
            "surviving to test would mean the small leaves are genuinely better served; a rho that "
            "goes to zero or flips means the validation pattern was fitted noise.")}

    n_neg_val = sum(1 for r in leaves if r["val_gain_at16"] < 0)
    n_neg_test = sum(1 for r in ok if r["test_gain_at16"] < 0)

    # ---- report ---------------------------------------------------------------------------------
    log("=" * 110)
    log("PER-LEAF: VALIDATION vs TEST at K=16, against " + COMPARATOR_LABEL)
    log(NEGATIVE_LEAF_NOTE)
    log("-" * 110)
    log(f"{'leaf':<46}{'d':>3}{'n_val':>7}{'n_test':>7}{'val_d':>10}{'test_d':>10}{'shrink':>9}"
        f"{'testNDCG':>10}{'testTail':>10}")
    for r in sorted(leaves, key=lambda r: -r["n_val"]):
        sh = r["shrinkage_val_to_test"]
        log(f"{r['nid']:<46}{r['depth']:>3}{r['n_val']:>7}{r['n_test']:>7}"
            f"{r['val_gain_at16']:>10.5f}"
            f"{(r['test_gain_at16'] if r['test_gain_at16'] is not None else float('nan')):>10.5f}"
            f"{(sh if sh is not None else float('nan')):>9.2f}"
            f"{(r['test_full@10_at16'] if r['test_full@10_at16'] is not None else float('nan')):>10.5f}"
            f"{(r['test_tail@10_at16'] if r['test_tail@10_at16'] is not None else float('nan')):>10.5f}")
    log("-" * 110)
    for K in BUDGETS:
        v, t = w_val[str(K)], w_test[str(K)]
        log(f"K={K:<3} VAL  {v['diff_full@10']:+.5f} [{v['ci95_full'][0]:+.5f},{v['ci95_full'][1]:+.5f}] "
            f"full | {v['diff_tail@10']:+.5f} [{v['ci95_tail'][0]:+.5f},{v['ci95_tail'][1]:+.5f}] tail")
        log(f"K={K:<3} TEST {t['diff_full@10']:+.5f} [{t['ci95_full'][0]:+.5f},{t['ci95_full'][1]:+.5f}] "
            f"full | {t['diff_tail@10']:+.5f} [{t['ci95_tail'][0]:+.5f},{t['ci95_tail'][1]:+.5f}] tail "
            f"| separates full={t['separates_full']} tail={t['separates_tail']}")
        log(f"K={K:<3} TEST ABSOLUTE  tree {t['tree_full@10']:.5f}/{t['tree_tail@10']:.5f}  "
            f"static {t['cmp_full@10']:.5f}/{t['cmp_tail@10']:.5f}  "
            f"m=0 {icpt_test[0]:.5f}/{icpt_test[1]:.5f}  | lift over m=0: tree "
            f"{t['tree_full@10'] - icpt_test[0]:+.5f}, static {t['cmp_full@10'] - icpt_test[0]:+.5f} "
            f"(the interview buys this much; the tree-vs-static difference is "
            f"{t['diff_full@10'] / max(t['cmp_full@10'] - icpt_test[0], 1e-9) * 100:.1f}% of it)")
    log("-" * 110)
    log(f"[size-gain] TEST spearman(n, gain@16) = {rho_t:.3f} (p={p_t:.4f}) | VAL {rho_v:.3f} "
        f"(p={p_v:.4f}) | test >=700 {size_gain['test_weighted_gain_leaves_ge_700']:+.5f} vs <700 "
        f"{size_gain['test_weighted_gain_leaves_lt_700']:+.5f}")
    log(f"[negatives] leaves below the comparator at K=16: {n_neg_val}/{len(leaves)} on validation, "
        f"{n_neg_test}/{len(ok)} on test. " + NEGATIVE_LEAF_NOTE)
    log("=" * 110)

    out = {"analysis": "ndcg_objective_interview_tree_TEST_LOOK_budget16",
           "stage": "2_test", "label": args.label, "citable": True,
           "authorisation": ("author-authorised single test look, 2026-08-12: 'add 16th step, run on "
                             "test. we are changing comparator, we are NOT changing this tree'"),
           "tree_regrown": False, "tree_source": os.path.basename(TREE_JSON),
           "depth_reached": tree["depth_reached"], "n_nodes": tree["n_nodes"], "n_leaves": len(leaves),
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "mask_policy": "B",
           # DERIVED stamp (2026-08-12 audit): "mask_policy" above is a label; these are read off the
           # masks this run actually scored through, so the artifact can state its own ruler.
           "pool_contract_val": Ev["pool_contract"], "pool_contract_test": Et["pool_contract"],
           "total_L": TOTAL_L,
           "budgets": list(BUDGETS), "n_val": len(rows_val), "n_test": len(rows_test),
           "comparator_label": COMPARATOR_LABEL, "comparator_note": COMPARATOR_NOTE,
           "comparator_sequence": [[int(a), int(b)] for a, b in cmp_seq],
           "policyB_comparator_status": ("PAUSED mid-rebuild with 7 of 16 picks banked in "
                                         "ndcgtree16_heaps/greedy_static_policyB_combined_heap.json; "
                                         "resumes after this look."),
           "bootstrap": {"kind": "paired percentile over per-user differences",
                         "n_resamples": BOOT, "seed": BOOT_SEED},
           "negative_leaf_note": NEGATIVE_LEAF_NOTE,
           "n_leaves_negative_val": n_neg_val, "n_leaves_negative_test": n_neg_test,
           "intercept_val_m0": {"full@10": icpt_val[0], "tail@10": icpt_val[1]},
           "intercept_test_m0": {"full@10": icpt_test[0], "tail@10": icpt_test[1]},
           "absolute_reporting_note": (
               "Absolute cohort-weighted NDCG@10 is reported for BOTH arms at every K, alongside the "
               "m=0 intercept (what a user gets for answering nothing). The author intends to read "
               "these against the current static ON THE EXPLICIT ASSUMPTION THAT IT IS WEAK, because "
               "its order was selected under policy A. Absolutes make that assumption inspectable: if "
               "the tree's absolute sits close to the static's, a weak comparator cannot be hiding "
               "much; if it is far above, the comparator's weakness matters more. Differences alone "
               "do not let a reader make that judgement, which is why the intercept is included -- it "
               "is the first time it has been printed for the test cohort."),
           "cohort_weighted_val": w_val, "cohort_weighted_test": w_test,
           "size_vs_gain": size_gain,
           "leaf_table": [{k: v for k, v in r.items() if not k.startswith("rows_")}
                          for r in sorted(leaves, key=lambda r: -r["n_val"])],
           "per_user_npz": os.path.basename(NPZ),
           "seconds": round(time.time() - t00, 1)}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    np.savez_compressed(NPZ, **bank)
    log(f"[done] wrote {os.path.basename(OUT)} + {os.path.basename(NPZ)} in "
        f"{(time.time() - t00) / 60:.1f}m")
    return out


if __name__ == "__main__":
    main()
