r"""A2 — RE-MEASURE THE FROZEN NDCG TREE AGAINST THE REBUILT POLICY-B STATIC COMPARATOR.
Label `ndcgtree16test_policyB`. Validation AND test, K in {1,2,4,8,16}, full AND tail NDCG@10.

AUTHOR SAID (2026-08-12), verbatim: "add 16th step, run on test. we are changing comparator, we are
NOT changing this tree, don't see a problem".

WHAT CHANGES AND WHAT DOES NOT.
  The TREE IS NOT REGROWN and NOT MODIFIED: same 22 nodes, same splitters, same 15 leaves, same
  depth 5, same 16-question leaf orders as the banked test look. The only thing that changes is the
  STATIC COMPARATOR: `greedy_static_i26c.json`'s combined arm (SELECTED under the retired policy-A
  candidate mask) is replaced by `greedy_static_policyB_combined.json`, rebuilt under policy B on the
  same 10,000 validation users, the same 1,000-candidate pool, the same objective and the same
  certified stack, by resuming the tree root's own CELF heap.

THE CONTRACT GATE, WHICH IS THE POINT OF THIS RUN.
  At K=1 no conditioning is possible: every user is asked the same single question by both arms. The
  rebuilt comparator's first pick and the tree's root splitter are the same argmax over the same
  pool, cohort and objective (the rebuild was literally seeded from the root's heap), so they MUST be
  the same question, and the per-user difference MUST be EXACTLY 0.0 for every user on BOTH cohorts,
  on BOTH axes. This is asserted with `== 0.0`, not a tolerance. Any non-zero stops the run before a
  single gain is written.

REUSE IS GATED, NOT ASSUMED (memory: `verify-the-artifact-not-a-proxy`).
  Nothing is taken on trust from the banked artifacts. The tree's own per-user vectors and the
  policy-A comparator's per-user vectors are RECOMPUTED here from the environment, and each is
  asserted BIT-IDENTICAL (max |diff| == 0.0e+00 per user) against
  `ndcgtree16test_per_user.npz`, along with the routed row ids per leaf. If the harness cannot
  reproduce the banked test look exactly, the new comparison is not on the same ruler and this run
  stops. The banked numbers are therefore also a free contract check on the whole chain (memory:
  `banked-global-greedy-is-a-policy-A-artifact`).

CONTRACT. Full AND tail NDCG@10, learned decoder bias, no floor, policy-B candidate pool, all 10,000
canonical validation users and all 10,000 canonical test users, nothing sampled (HARD RULE 1).
Paired PERCENTILE bootstrap over per-user DIFFERENCE vectors, 10,000 resamples, seed 98765 — never
the unpaired 1.96*sqrt(2)*SE form.

The validation side needs NO environment: the running `ndcg_tree_extend16` job already banked the
per-user vectors for the tree, the policy-B comparator and the policy-A comparator on all 10,000
validation users. This job only adds the paired CIs to them. The test side is the only compute.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_eval, genv_answered,
                                    genv_mean, contract_selftest, CERT_TOWER, CERT_CONC)
from run_battery_phaseA import bootstrap_ci
from ndcg_tree import split_rows, leak_assert, BRANCHES

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
TREE_JSON = os.path.join(OUTDIR, "ndcgtree.json")
VAL16_JSON = os.path.join(OUTDIR, "ndcgtree16.json")
VAL16_NPZ = os.path.join(OUTDIR, "ndcgtree16_per_user.npz")
TEST16_JSON = os.path.join(OUTDIR, "ndcgtree16test.json")
TEST16_NPZ = os.path.join(OUTDIR, "ndcgtree16test_per_user.npz")
STATICB_JSON = os.path.join(OUTDIR, "greedy_static_policyB_combined.json")

LABEL = "ndcgtree16test_policyB"
TOTAL_L = 16
BUDGETS = (1, 2, 4, 8, 16)
BOOT, BOOT_SEED = 10_000, 98765
ENV_LABEL_TEST = "ndcgtree16test(test)"     # MUST match the banked test look's env label exactly


def route(E, nodes, nid, rows_all):
    """Walk the FROZEN tree with the banked splitters, branching each user by the answer they would
    actually give. Identical to the banked test look's routing function."""
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


def exact(a, b, what):
    """The reuse gate: bit-identical, per user, no tolerance."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    fa, fb = np.isnan(a), np.isnan(b)
    assert (fa == fb).all(), f"{what}: NaN pattern differs"
    d = np.abs(a[~fa] - b[~fb])
    mx = float(d.max()) if d.size else 0.0
    assert mx == 0.0, (f"{what}: RECOMPUTATION DOES NOT REPRODUCE THE BANKED VECTORS "
                       f"(max |diff| = {mx:.3e}, {int((d > 0).sum())} users differ). The banked "
                       f"artifact and this harness are not on the same ruler -- STOP.")
    return mx


def weighted_table(bank, leaves, tag, arms, budgets=BUDGETS):
    """Cohort-weighted absolutes + paired-bootstrap differences of the tree against each comparator."""
    out = {}
    use = [r for r in leaves if r[f"n_{tag}"] > 0]
    for K in budgets:
        row = {}
        tf = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_full"] for r in use])
        tt = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_tail"] for r in use])
        row["tree_full@10"] = float(np.nanmean(tf))
        row["tree_tail@10"] = float(np.nanmean(tt))
        for arm in arms:
            cf = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|{arm}_full"] for r in use])
            ct = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|{arm}_tail"] for r in use])
            df = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_full"]
                                 - bank[f"{tag}|{r['nid']}|{K}|{arm}_full"] for r in use])
            dt = np.concatenate([bank[f"{tag}|{r['nid']}|{K}|own_tail"]
                                 - bank[f"{tag}|{r['nid']}|{K}|{arm}_tail"] for r in use])
            mf, cif, nf = bootstrap_ci(df, n_boot=BOOT, seed=BOOT_SEED)
            mt, cit, _ = bootstrap_ci(dt, n_boot=BOOT, seed=BOOT_SEED)
            row[f"{arm}_full@10"] = float(np.nanmean(cf))
            row[f"{arm}_tail@10"] = float(np.nanmean(ct))
            row[f"diff_full@10_vs_{arm}"] = mf
            row[f"ci95_full_vs_{arm}"] = list(cif)
            row[f"diff_tail@10_vs_{arm}"] = mt
            row[f"ci95_tail_vs_{arm}"] = list(cit)
            row[f"separates_full_vs_{arm}"] = bool(cif[0] > 0 or cif[1] < 0)
            row[f"separates_tail_vs_{arm}"] = bool(cit[0] > 0 or cit[1] < 0)
            row["n_users"] = int(nf)
            # the exact-per-user difference vectors, for the K=1 gate
            row[f"_maxabs_diff_full_vs_{arm}"] = float(np.abs(df[~np.isnan(df)]).max())
            row[f"_maxabs_diff_tail_vs_{arm}"] = float(np.abs(dt[~np.isnan(dt)]).max())
        out[str(K)] = row
    return out


def k1_gate(table, tag):
    """K=1: no conditioning is possible, both arms ask the identical question, so EVERY user's
    difference must be exactly zero. Asserted with ==, not a tolerance."""
    r = table["1"]
    for ax in ("full", "tail"):
        mx = r[f"_maxabs_diff_{ax}_vs_policyB"]
        mean = r[f"diff_{ax}@10_vs_policyB"]
        assert mx == 0.0 and mean == 0.0, (
            f"K=1 GATE FAILED on {tag}/{ax}: mean {mean!r}, max per-user |diff| {mx!r}. The tree's "
            f"root splitter and the rebuilt comparator's first pick are the same question on the "
            f"same users, so every per-user score must be bit-identical. STOP -- do not report gains.")
    log(f"[gate/{tag}] PASS: K=1 difference vs the policy-B comparator is EXACTLY 0.0 for every one "
        f"of the {r['n_users']} users, on BOTH axes (max per-user |diff| = 0.0e+00).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()

    t00 = time.time()
    OUT = os.path.join(OUTDIR, f"{args.label}.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}_per_user.npz")
    assert not os.path.exists(OUT + ".lock"), "another run holds this output"

    for nm, p in (("tree", TREE_JSON), ("val16", VAL16_JSON), ("val16 npz", VAL16_NPZ),
                  ("test16", TEST16_JSON), ("test16 npz", TEST16_NPZ), ("policy-B static", STATICB_JSON),
                  ("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt)):
        assert os.path.exists(p), f"{nm} missing: {p}"

    tree = json.load(open(TREE_JSON))
    v16 = json.load(open(VAL16_JSON))
    t16 = json.load(open(TEST16_JSON))
    stb = json.load(open(STATICB_JSON))
    nodes = tree["nodes"]

    # ---------------- contract checks on the inputs ------------------------------------------------
    assert tree["citable"] and not tree["SANITY"] and tree["split"] == "validation"
    assert v16["citable"] and v16["split"] == "validation" and v16["total_L"] == TOTAL_L
    assert t16["citable"] and t16["total_L"] == TOTAL_L and not t16["tree_regrown"]
    assert stb["citable"] and stb["mask_policy"] == "B" and stb["split"] == "validation"
    assert stb["L"] == TOTAL_L and stb["n_cohort"] == 10000 and stb["objective"] == "full"
    assert stb["n_pool"] == 1000 and stb["n_items_pool"] == 500 and stb["n_concepts_pool"] == 500
    for b in (v16, t16, stb):
        assert b["tower"] == os.path.basename(args.snapshot), b["tower"]
        assert b["concept_ckpt"] == os.path.basename(args.concept_ckpt), b["concept_ckpt"]

    seq_B = [(int(a), int(b)) for a, b in stb["sequence"]][:TOTAL_L]
    seq_A = [(int(a), int(b)) for a, b in t16["comparator_sequence"]][:TOTAL_L]
    assert len(seq_B) == len(seq_A) == TOTAL_L
    root_q = tuple(int(x) for x in nodes["root"]["splitter"])
    assert seq_B[0] == root_q, (f"the rebuilt comparator opens with {seq_B[0]} but the tree root "
                                f"splitter is {root_q}")
    log("#" * 110)
    log(f"# A2 -- the frozen tree re-measured against the REBUILT POLICY-B comparator. Tree NOT "
        f"regrown, NOT modified: {tree['n_nodes']} nodes, {tree['n_leaves']} leaves, depth "
        f"{tree['depth_reached']}, 16-question leaf orders identical to the banked test look.")
    log(f"# policy-B comparator: {[[a, b] for a, b in seq_B]}")
    log(f"# policy-A comparator: {[[a, b] for a, b in seq_A]}  (superseded, reported beside it)")
    log(f"# first picks: tree root splitter {root_q} == policy-B pick 1 {seq_B[0]}; policy-A pick 1 "
        f"was {seq_A[0]} -- THAT is why the old K=1 was not zero.")
    log(f"# positions identical between the two orders: {stb['positions_identical_to_policyA_order']}"
        f"/{TOTAL_L}; set overlap {len(set(seq_A) & set(seq_B))}/{TOTAL_L}")
    log("#" * 110)

    # the 16-question leaf orders must be the same object in both extend jobs (free contract check)
    own_v = {r["nid"]: [tuple(q) for q in r["own_sequence"]] for r in v16["leaf_table"]}
    own_t = {r["nid"]: [tuple(q) for q in r["own_sequence"]] for r in t16["leaf_table"]}
    assert set(own_v) == set(own_t), "leaf sets differ between the val and test extend jobs"
    for nid in own_v:
        assert own_v[nid] == own_t[nid], (f"{nid}: the two extend jobs produced DIFFERENT 16-question "
                                          f"orders\n  val  {own_v[nid]}\n  test {own_t[nid]}")
    log(f"[contract] all {len(own_v)} leaf 16-question orders are identical across the two "
        f"independent extend jobs")

    # ================================================================= VALIDATION (banked, no env)
    log("=" * 110)
    log("[validation] paired CIs from the vectors the extend16 job banked -- no environment is built, "
        "no score is recomputed, so nothing can drift.")
    zv = np.load(VAL16_NPZ)
    vbank, vleaves = {}, []
    for r in sorted(v16["leaf_table"], key=lambda r: -r["n_users"]):
        nid = r["nid"]
        vleaves.append({"nid": nid, "depth": r["depth"], "n_val": r["n_users"]})
        for K in BUDGETS:
            for arm in ("own", "policyB", "policyA"):
                for ax in ("full", "tail"):
                    k = f"{nid}|{K}|{arm}_{ax}"
                    assert k in zv.files, f"missing banked validation vector {k}"
                    vbank[f"val|{nid}|{K}|{arm}_{ax}"] = zv[k].astype(np.float64)
        assert len(zv[f"{nid}|rows"]) == r["n_users"]
    w_val = weighted_table(vbank, vleaves, "val", ("policyB", "policyA"))
    k1_gate(w_val, "validation")

    # ================================================================= TEST
    log("=" * 110)
    log("[test] building the TEST environment. The tree's own vectors and the policy-A comparator's "
        "are RECOMPUTED and gated bit-identical against the banked test look before any new number "
        "is believed.")
    Et = build_greedy_env(args.snapshot, "test", args.concept_ckpt, ENV_LABEL_TEST)
    assert Et["contract_tag"] == f"genv:{ENV_LABEL_TEST}:test", Et["contract_tag"]
    rows_test = Et["rows"]
    contract_selftest(Et, rows_test)
    leak_assert(Et, rows_test, "test")
    assert len(rows_test) == t16["n_test"] == 10000, f"{len(rows_test)} test users"
    set_active_rows(Et, rows_test); Et["memo"] = {}
    icpt_t = genv_mean(Et, [], rows_test)
    log(f"[intercept/test] m=0, policy-B pool: {icpt_t[0]:.5f} full / {icpt_t[1]:.5f} tail")
    assert abs(icpt_t[0] - t16["intercept_test_m0"]["full@10"]) == 0.0, "test intercept drifted"

    zt = np.load(TEST16_NPZ)
    tbank, tleaves = {}, []
    meta_by_nid = {r["nid"]: r for r in t16["leaf_table"]}
    assigned, seen = 0, set()
    routed = {}
    for nid, meta in meta_by_nid.items():
        rows = route(Et, nodes, nid, rows_test)
        assert len(rows) == meta["n_test"], f"{nid}: routed {len(rows)}, banked {meta['n_test']}"
        banked_rows = zt[f"test|{nid}|rows"].astype(np.int64)
        assert np.array_equal(np.asarray(rows, np.int64), banked_rows), \
            f"{nid}: ROUTED ROW IDS DIFFER from the banked test look -- STOP"
        routed[nid] = rows
        assigned += len(rows)
        s = set(rows)
        assert not (seen & s), "test routing is not disjoint"
        seen |= s
    assert assigned == len(rows_test), f"routing lost users: {assigned} vs {len(rows_test)}"
    log(f"[test] routed {assigned}/{len(rows_test)} users into {len(routed)} leaves, disjoint and "
        f"covering, row ids BIT-IDENTICAL to the banked test look (HARD RULE 1: nothing dropped)")

    gate_max = 0.0
    per_leaf = []
    for nid in sorted(routed, key=lambda n: -meta_by_nid[n]["n_test"]):
        rows = routed[nid]
        meta = meta_by_nid[nid]
        own = own_t[nid]
        tleaves.append({"nid": nid, "depth": meta["depth"], "n_test": len(rows),
                        "n_val": meta["n_val"]})
        if not rows:
            continue
        set_active_rows(Et, rows); Et["memo"] = {}
        curve = {}
        for K in BUDGETS:
            a = own[:K]
            assert len(a) == K
            of, ot = genv_eval(Et, a, rows)
            # ---- REUSE GATE: the tree's own vectors must reproduce the banked ones exactly
            gate_max = max(gate_max, exact(of, zt[f"test|{nid}|{K}|own_full"], f"{nid} K={K} own_full"))
            gate_max = max(gate_max, exact(ot, zt[f"test|{nid}|{K}|own_tail"], f"{nid} K={K} own_tail"))
            tbank[f"test|{nid}|{K}|own_full"] = of
            tbank[f"test|{nid}|{K}|own_tail"] = ot
            row = {"tree_full@10": float(np.nanmean(of)), "tree_tail@10": float(np.nanmean(ot))}
            ia, ca = genv_answered(Et, a, rows)
            row["items_answered"] = round(ia, 3); row["concepts_answered"] = round(ca, 3)
            for arm, seq_c in (("policyB", seq_B), ("policyA", seq_A)):
                b = seq_c[:K]
                assert len(b) == K, f"UNEQUAL BUDGET: {arm} {len(b)} at K={K}"
                gf, gt = genv_eval(Et, b, rows)
                if arm == "policyA":       # banked under key `cmp_*` in the test look
                    gate_max = max(gate_max, exact(gf, zt[f"test|{nid}|{K}|cmp_full"],
                                                   f"{nid} K={K} policyA_full"))
                    gate_max = max(gate_max, exact(gt, zt[f"test|{nid}|{K}|cmp_tail"],
                                                   f"{nid} K={K} policyA_tail"))
                tbank[f"test|{nid}|{K}|{arm}_full"] = gf
                tbank[f"test|{nid}|{K}|{arm}_tail"] = gt
                row[f"{arm}_full@10"] = float(np.nanmean(gf))
                row[f"{arm}_tail@10"] = float(np.nanmean(gt))
                row[f"diff_full@10_vs_{arm}"] = float(np.nanmean(of - gf))
                row[f"diff_tail@10_vs_{arm}"] = float(np.nanmean(ot - gt))
                row[f"set_overlap_vs_{arm}"] = len(set(a) & set(b))
            curve[str(K)] = row
        tbank[f"test|{nid}|rows"] = np.asarray(rows, np.int64)
        np.savez_compressed(NPZ, **tbank)                     # HARD RULE 11(a): persist per leaf
        c = curve[str(TOTAL_L)]
        per_leaf.append({"nid": nid, "depth": meta["depth"], "n_val": meta["n_val"],
                         "n_test": len(rows), "own_sequence": [[a, b] for a, b in own],
                         "curve_test": curve,
                         "test_gain_at16_vs_policyB": c["diff_full@10_vs_policyB"],
                         "test_gain_at16_tail_vs_policyB": c["diff_tail@10_vs_policyB"],
                         "test_gain_at16_vs_policyA": c["diff_full@10_vs_policyA"],
                         "val_gain_at16_vs_policyA": meta["val_gain_at16"]})
        log(f"[test/{nid}] n={len(rows)} | @K16 vs policyB "
            f"{c['diff_full@10_vs_policyB']:+.5f}/{c['diff_tail@10_vs_policyB']:+.5f} | vs policyA "
            f"{c['diff_full@10_vs_policyA']:+.5f}/{c['diff_tail@10_vs_policyA']:+.5f}")

    log(f"[reuse-gate] PASS: every recomputed tree and policy-A per-user vector on all 10,000 test "
        f"users reproduces the banked test look EXACTLY (max |diff| over the whole run = "
        f"{gate_max:.1e}). The two runs are on the same ruler; only the comparator changed.")

    w_test = weighted_table(tbank, tleaves, "test", ("policyB", "policyA"))
    k1_gate(w_test, "test")

    # ---------------- report -----------------------------------------------------------------------
    n_neg_B = sum(1 for r in per_leaf if r["test_gain_at16_vs_policyB"] < 0)
    n_neg_A = sum(1 for r in per_leaf if r["test_gain_at16_vs_policyA"] < 0)
    nn = np.array([r["n_test"] for r in per_leaf], float)
    gg = np.array([r["test_gain_at16_vs_policyB"] for r in per_leaf])
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(nn, gg)
    except Exception:
        rho = pval = float("nan")
    big = nn >= 700
    size_gain = {"test_spearman_rho_vs_policyB": float(rho), "test_p_value": float(pval),
                 "weighted_gain_leaves_ge_700": float((nn[big] * gg[big]).sum() / max(nn[big].sum(), 1)),
                 "weighted_gain_leaves_lt_700": float((nn[~big] * gg[~big]).sum() / max(nn[~big].sum(), 1)),
                 "n_test_users_ge_700": int(nn[big].sum()), "n_test_users_lt_700": int(nn[~big].sum())}

    log("=" * 110)
    log(f"{'leaf':<46}{'d':>3}{'n_val':>7}{'n_test':>7}{'testB_full':>12}{'testB_tail':>12}"
        f"{'testA_full':>12}")
    for r in per_leaf:
        log(f"{r['nid']:<46}{r['depth']:>3}{r['n_val']:>7}{r['n_test']:>7}"
            f"{r['test_gain_at16_vs_policyB']:>12.5f}{r['test_gain_at16_tail_vs_policyB']:>12.5f}"
            f"{r['test_gain_at16_vs_policyA']:>12.5f}")
    log("-" * 110)
    for K in BUDGETS:
        for tag, w in (("VAL ", w_val), ("TEST", w_test)):
            log(f"K={K:<3} {tag} vs policy-B {w[str(K)]['diff_full@10_vs_policyB']:+.5f} "
                f"[{w[str(K)]['ci95_full_vs_policyB'][0]:+.5f},"
                f"{w[str(K)]['ci95_full_vs_policyB'][1]:+.5f}] full | "
                f"{w[str(K)]['diff_tail@10_vs_policyB']:+.5f} "
                f"[{w[str(K)]['ci95_tail_vs_policyB'][0]:+.5f},"
                f"{w[str(K)]['ci95_tail_vs_policyB'][1]:+.5f}] tail | sep "
                f"full={w[str(K)]['separates_full_vs_policyB']} "
                f"tail={w[str(K)]['separates_tail_vs_policyB']}   "
                f"(vs policy-A {w[str(K)]['diff_full@10_vs_policyA']:+.5f}/"
                f"{w[str(K)]['diff_tail@10_vs_policyA']:+.5f})")
    log("-" * 110)
    for K in BUDGETS:
        w = w_test[str(K)]
        log(f"K={K:<3} TEST ABSOLUTE tree {w['tree_full@10']:.5f}/{w['tree_tail@10']:.5f}  "
            f"policyB {w['policyB_full@10']:.5f}/{w['policyB_tail@10']:.5f}  "
            f"policyA {w['policyA_full@10']:.5f}/{w['policyA_tail@10']:.5f}  m=0 "
            f"{icpt_t[0]:.5f}/{icpt_t[1]:.5f} | lift over m=0: tree "
            f"{w['tree_full@10'] - icpt_t[0]:+.5f}, policyB "
            f"{w['policyB_full@10'] - icpt_t[0]:+.5f} (the tree-vs-policyB difference is "
            f"{w['diff_full@10_vs_policyB'] / max(w['policyB_full@10'] - icpt_t[0], 1e-9) * 100:.1f}% "
            f"of what the interview itself buys)")
    log(f"[size-gain] TEST spearman(n, gain@16 vs policy-B) = {rho:.3f} (p={pval:.4f}); >=700 users "
        f"{size_gain['weighted_gain_leaves_ge_700']:+.5f} vs <700 "
        f"{size_gain['weighted_gain_leaves_lt_700']:+.5f}")
    log(f"[negatives] leaves below the comparator at K=16 on test: {n_neg_B}/{len(per_leaf)} vs "
        f"policy-B, {n_neg_A}/{len(per_leaf)} vs policy-A")
    log("=" * 110)

    out = {"analysis": "ndcg_tree_remeasured_against_policyB_static_comparator",
           "stage": "A2", "label": args.label, "citable": True,
           "authorisation": ("author ruling 2026-08-12: 'we are changing comparator, we are NOT "
                             "changing this tree'"),
           "tree_regrown": False, "tree_modified": False, "tree_source": os.path.basename(TREE_JSON),
           "n_nodes": tree["n_nodes"], "n_leaves": len(per_leaf),
           "depth_reached": tree["depth_reached"],
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": "full", "mask_policy": "B", "total_L": TOTAL_L, "budgets": list(BUDGETS),
           "n_val": 10000, "n_test": len(rows_test),
           "comparator_policyB": {"label": stb["label"], "file": os.path.basename(STATICB_JSON),
                                  "sequence": [[a, b] for a, b in seq_B],
                                  "gains": stb["gains"],
                                  "curve_policyB": stb["curve_policyB"],
                                  "positions_identical_to_policyA": stb[
                                      "positions_identical_to_policyA_order"],
                                  "set_overlap_with_policyA": len(set(seq_A) & set(seq_B))},
           "comparator_policyA_superseded": {"sequence": [[a, b] for a, b in seq_A],
                                             "note": t16["comparator_note"]},
           "k1_gate": {"root_splitter": list(root_q), "policyB_first_question": list(seq_B[0]),
                       "policyA_first_question": list(seq_A[0]),
                       "val_K1_diff_full": w_val["1"]["diff_full@10_vs_policyB"],
                       "val_K1_diff_tail": w_val["1"]["diff_tail@10_vs_policyB"],
                       "test_K1_diff_full": w_test["1"]["diff_full@10_vs_policyB"],
                       "test_K1_diff_tail": w_test["1"]["diff_tail@10_vs_policyB"],
                       "exactly_zero_per_user_both_cohorts_both_axes": True},
           "reuse_gate": {"what": ("tree own + policy-A comparator per-user vectors on all 10,000 test "
                                   "users, and the routed row ids per leaf, recomputed and compared to "
                                   "ndcgtree16test_per_user.npz"),
                          "max_abs_diff": gate_max, "bit_identical": gate_max == 0.0},
           "bootstrap": {"kind": "paired percentile over per-user differences",
                         "n_resamples": BOOT, "seed": BOOT_SEED},
           "intercept_test_m0": {"full@10": icpt_t[0], "tail@10": icpt_t[1]},
           "cohort_weighted_val": w_val, "cohort_weighted_test": w_test,
           "size_vs_gain": size_gain,
           "n_leaves_negative_test_vs_policyB": n_neg_B,
           "n_leaves_negative_test_vs_policyA": n_neg_A,
           "leaf_table": per_leaf, "per_user_npz": os.path.basename(NPZ),
           "seconds": round(time.time() - t00, 1)}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    np.savez_compressed(NPZ, **tbank)
    log(f"[done] wrote {os.path.basename(OUT)} + {os.path.basename(NPZ)} in "
        f"{(time.time() - t00) / 60:.1f}m")
    return out


if __name__ == "__main__":
    main()
