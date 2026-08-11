"""Genre-split group-conditioned greedy, STAGE 1 (build) -- label `groupgreedy_genre16`.

WHY THIS EXISTS. `group_greedy_d1.py` branched on the answer to ONE popular ITEM (item 83, the global
greedy's own first pick). 70.45% of the validation cohort could not answer it, and in exactly that
70% the conditioning gain was ~0.000 (+0.0019); the two groups that DID answer gained +0.0129 and
+0.0237. So the depth-1 result was budget-limited by ANSWERABILITY, not by conditioning. This run
replaces the branch question with one everyone can answer: "what is your favourite genre?".

THE BRANCH.
  * An OPEN first question (no closed vocabulary is presented), simulated by reading the user's own
    FOLD-IN profile: among their LIKED fold-in items (level >= LIKE_MIN_LEVEL = 7, i.e. >= 4.0
    stars) take the most frequent MovieLens genre, ties broken ALPHABETICALLY. Held-out targets are
    excluded by construction -- `sh["lvl_lookup"]` comes from `ctx.allb`, which `build_real_ctx`
    builds with `keepm = ~g["sid"].isin(te_sets[r])` -- and this is ASSERTED at runtime.
  * Users with NO liked fold-in item (or none carrying a genre) form an explicit `none` group. They
    are run, not dropped (HARD RULE 1).
  * The answer selects the group, i.e. it conditions the SELECTOR. It is NOT folded into the
    recommender: the pool is items-only, the scored sequence is items only, no genre/concept token
    enters the fold.

*** THE GENEROUS ASSUMPTION, STATED UP FRONT (author ruling 2026-08-11) ***
  The genre answer is treated as FREE SIDE INFORMATION. It costs the interview NOTHING: no turn is
  charged for it, and the comparison is 16 ITEMS vs 16 ITEMS. A real system would have to ask for it
  and would pay a turn. This is deliberately GENEROUS TO THE CONDITIONED ARM and must be visible to
  any reader: it is repeated in the output JSON under `generous_assumption`, in the design sheet and
  in the log. Every number this script produces is an UPPER BOUND on what genre conditioning buys.

WHAT THE SANITY GATE DOES AND DOES NOT MEAN.
  At 16-vs-16 on the SELECTION cohort the group's own order is GUARANTEED to be >= the global order,
  because the global order is inside every group's search space (modulo CELF's step-wise argmax). So
  `sanity_pass` is a pure HARNESS check -- A WIN IS NOT EVIDENCE THAT CONDITIONING PAYS. The three
  quantities that carry information are:
    (1) the MAGNITUDE of the per-group and cohort-weighted gain -- the CEILING on conditioning here;
    (2) GAIN vs GROUP SIZE -- the overfitting signal. Large gains on small groups next to ~zero on
        the big ones is fitting noise, not structure. The end-of-run table sorts by n for exactly
        this reading;
    (3) ORDER OVERLAP with the global order. If the conditioned orders are nearly identical to the
        global one, that is the cleanest possible statement that there was nothing to condition on.

STAGE 2 (`group_greedy_genre_test.py`) carries these orders to the TEST cohort. NOTHING HERE TOUCHES
TEST: the test cohort is not built, not loaded and not read.

PERSISTENCE (HARD RULE 11a): `*_partial.json` is rewritten after EVERY pick. A kill costs one pick.
Per-user score vectors are banked to `*_per_user.npz` so stage 2 and any paired CI can reuse them.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import csv
import json
import time
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, greedy_certified,
                                    genv_mean, genv_eval, genv_answered, contract_selftest,
                                    _ranked_pool, CERT_TOWER, CERT_CONC)
from i26_encoder import LIKE_MIN_LEVEL

LABEL = "groupgreedy_genre16"
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
GLOBAL_JSON = os.path.join(OUTDIR, "greedy_static_i26c.json")
MOVIES_CSV = os.path.join(_ROOT, "data", "movielens", "movies.csv")     # the ML-25M raw movies file
UNIQUE_SID = os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")
NO_GENRE = "(no genres listed)"
NONE_GROUP = "none"
ITEM_BUDGETS = (1, 2, 4, 8, 16)            # ITEM questions on BOTH sides -- 16 vs 16, nothing charged
SMALL_GROUP = 200                          # below this a group is flagged untrustworthy, never merged

GENEROUS = ("The favourite-genre answer is treated as FREE SIDE INFORMATION: it costs NO turn and is "
            "NOT folded into the recommender, so the comparison is 16 ITEMS vs 16 ITEMS. A deployed "
            "system would have to ask for it and would pay a turn. This is GENEROUS TO THE "
            "CONDITIONED ARM by construction -- every gain here is an UPPER BOUND.")
SANITY_MEANING = ("`sanity_pass` is a HARNESS check, NOT evidence. On the SELECTION cohort the "
                  "group's own 16-item order is GUARANTEED to be >= the global 16-item order, "
                  "because the global order lies inside every group's search space. A pass means the "
                  "search, the cohort masking and the memo are correct and NOTHING ELSE. What "
                  "carries information: the MAGNITUDE of the gain, the gain-vs-group-size relation "
                  "(overfitting), and the order overlap with the global order.")


def _read_global(path):
    """The frozen global 16-step items-only order. READ, never re-derived."""
    blob = json.load(open(path))
    arm = blob["arms"]["items-only"]
    seq = [(int(a), int(b)) for a, b in arm["sequence"]]
    assert len(seq) == 16, f"global items-only order is {len(seq)} steps, expected 16"
    assert all(ch == 0 for ch, _ in seq), "the items-only arm contains a non-item question"
    assert blob["objective"] == "full", f"global build objective is {blob['objective']}, expected full"
    assert blob["item_bank"] == "pop" and blob["n_items_pool"] == 500, \
        f"global build pool is {blob['item_bank']}/{blob['n_items_pool']}, expected pop/500"
    return seq, blob


def load_item_genres(ni):
    """sid -> sorted tuple of MovieLens genres, for the project's item vocabulary.

    sid is the ROW INDEX into data/ml-25m/proc/unique_sid.txt, which is exactly the `show2id` the
    canonical Liang split built (asserted against ctx.show2id by the caller).
    """
    assert os.path.exists(MOVIES_CSV), f"genre source missing: {MOVIES_CSV}"
    assert os.path.exists(UNIQUE_SID), f"item vocabulary missing: {UNIQUE_SID}"
    with open(UNIQUE_SID) as f:
        sids = [int(x) for x in f.read().split()]
    assert len(sids) == ni, f"unique_sid.txt has {len(sids)} items, ctx.ni = {ni}"
    movie2sid = {m: i for i, m in enumerate(sids)}
    assert len(movie2sid) == ni, "unique_sid.txt contains duplicate movieIds"
    gen = {}
    n_rows = 0
    with open(MOVIES_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            s = movie2sid.get(int(r["movieId"]))
            if s is None:
                continue
            gs = [g.strip() for g in r["genres"].split("|") if g.strip() and g.strip() != NO_GENRE]
            if gs:
                gen[s] = tuple(sorted(set(gs)))
    vocab = sorted({g for v in gen.values() for g in v})
    log(f"[genre] source {os.path.relpath(MOVIES_CSV, _ROOT)} ({n_rows} rows) x "
        f"{os.path.relpath(UNIQUE_SID, _ROOT)} ({ni} catalogue items): "
        f"{len(gen)}/{ni} ({100.0 * len(gen) / ni:.2f}%) catalogue items carry >=1 genre; "
        f"{ni - len(gen)} carry none. Genre vocabulary ({len(vocab)}): {vocab}")
    return gen, vocab


def favourite_genre(liked_sids, gen):
    """Most frequent genre over the user's LIKED FOLD-IN items; ties broken ALPHABETICALLY."""
    cnt = {}
    for s in liked_sids:
        for g in gen.get(s, ()):
            cnt[g] = cnt.get(g, 0) + 1
    if not cnt:
        return None
    best = max(cnt.values())
    return sorted(g for g, c in cnt.items() if c == best)[0]


def partition_by_genre(E, rows, gen):
    """Partition a cohort by favourite genre off the GRADED FOLD-IN lookup (targets excluded).

    Shared by stage 1 (validation) and stage 2 (test) -- the rule must be bit-identical on both
    sides, so it lives in ONE function and stage 2 imports it rather than re-implementing it.
    """
    lvl = E["lvl"]
    g = {}
    n_no_liked = 0
    n_liked_but_ungenred = 0
    for r in rows:
        d = lvl[r]
        liked = [int(i) for i, L in d.items() if int(L) >= LIKE_MIN_LEVEL]
        fav = favourite_genre(liked, gen)
        if fav is None:
            fav = NONE_GROUP
            if liked:
                n_liked_but_ungenred += 1
            else:
                n_no_liked += 1
        g.setdefault(fav, []).append(int(r))
    tot = sum(len(v) for v in g.values())
    assert tot == len(rows), f"partition lost users: {tot} != {len(rows)}"
    seen = set()
    for v in g.values():
        assert not (seen & set(v)), "partition is not disjoint"
        seen |= set(v)
    assert len(seen) == len(rows), "partition is not a cover"
    log(f"[partition] '{NONE_GROUP}' = {n_no_liked} users with NO liked fold-in item + "
        f"{n_liked_but_ungenred} with liked items but no genre on any of them")
    return g, n_no_liked, n_liked_but_ungenred


def leak_assert(E, rows, tag):
    """The grouping must never see a held-out target. lvl_lookup is built from ctx.allb, which is
    built with the te rows removed -- assert it rather than trust the comment. Run on BOTH cohorts."""
    ctx = E["ctx"]; lvl = E["lvl"]
    bad = 0
    for r in rows:
        te = set(ctx.va_te[r].indices.tolist())
        if te & set(lvl[r].keys()):
            bad += 1
    assert bad == 0, f"LEAK ({tag}): {bad} users have a held-out target inside the fold-in lookup"
    log(f"[leak/{tag}] 0/{len(rows)} users have any held-out target inside the grouping lookup -- "
        f"the favourite-genre answer is derived from FOLD-IN ratings only")


def order_overlap(own_items, global_seq, K):
    """How much of the group's first-K order is the global order? Set overlap + positional matches."""
    a = [int(i) for _, i in own_items[:K]]
    b = [int(i) for _, i in global_seq[:K]]
    inter = len(set(a) & set(b))
    same_pos = sum(1 for x, y in zip(a, b) if x == y)
    return {"K": K, "set_overlap": inter, "set_overlap_frac": round(inter / K, 4),
            "positions_identical": same_pos, "positions_differ": K - same_pos,
            "identical_order": bool(a == b)}


def _log_gain_table(rows_tbl, weighted):
    """The two readings that carry information, printed so they cannot be missed."""
    log("=" * 108)
    log("GAIN vs GROUP SIZE  (validation/SELECTION cohort -- a win here is GUARANTEED, see below)")
    log(SANITY_MEANING)
    log("-" * 108)
    log(f"{'group':<14}{'n':>7}{'diff_full@16':>14}{'diff_tail@16':>14}"
        f"{'overlap@16':>12}{'pos_differ@16':>15}{'small?':>9}")
    for t in rows_tbl:
        log(f"{t['group']:<14}{t['n_users']:>7}{t['diff_full@16']:>14.5f}{t['diff_tail@16']:>14.5f}"
            f"{t['set_overlap@16']:>12}{t['positions_differ@16']:>15}"
            f"{('YES' if t['small_group'] else ''):>9}")
    log("-" * 108)
    for K in sorted(weighted, key=int):
        w = weighted[K]
        log(f"COHORT-WEIGHTED (all {w['n_users']} users) @K={K:<3} "
            f"full {w['diff_full@10']:+.5f}   tail {w['diff_tail@10']:+.5f}")
    log("=" * 108)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--L", type=int, default=16, help="ITEM steps per group (vs the global order's L)")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--import_banked", default=None,
                    help="A *_partial.json from an earlier run whose COMPLETED groups' orders are "
                         "reused instead of re-searched. The contract is asserted FIELD BY FIELD "
                         "first; any mismatch aborts. Curves/overlap/per-user vectors are always "
                         "recomputed here, so imported and freshly-searched groups are measured by "
                         "the same code on the same env.")
    ap.add_argument("--sanity", action="store_true",
                    help="end-to-end pipeline check: L=2 on ONE group from a 12-item pool. Writes "
                         "*_SANITY.json and is NEVER citable.")
    args = ap.parse_args()

    t00 = time.time()
    sfx = "_SANITY" if args.sanity else ""
    OUT = os.path.join(OUTDIR, f"{args.label}{sfx}.json")
    PARTIAL = os.path.join(OUTDIR, f"{args.label}{sfx}_partial.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}{sfx}_per_user.npz")
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- phase 1: every FILE path exists before anything expensive runs ------------------------
    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("global greedy json", GLOBAL_JSON), ("movies.csv", MOVIES_CSV),
                  ("unique_sid.txt", UNIQUE_SID)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"
    log("#" * 108)
    log("# GENEROUS ASSUMPTION: " + GENEROUS)
    log("#" * 108)
    if args.sanity:
        log("# SANITY MODE: L=2 on ONE group, 12-item pool. Pipeline check ONLY, NOT citable.")

    global_seq, gblob = _read_global(GLOBAL_JSON)
    log(f"[global] read from {os.path.basename(GLOBAL_JSON)} (tower={gblob['tower']}, "
        f"conc={gblob['concept_ckpt']}): 16-step items-only order = {[i for _, i in global_seq]}")
    assert os.path.basename(args.snapshot) == gblob["tower"], \
        f"tower mismatch: {os.path.basename(args.snapshot)} vs global build's {gblob['tower']}"
    assert os.path.basename(args.concept_ckpt) == gblob["concept_ckpt"], \
        f"concept ckpt mismatch vs the global build ({gblob['concept_ckpt']})"

    # ---- the ONE environment: validation. The test cohort is never built. ----------------------
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"{args.label}(val)")
    rows = E["rows"]
    contract_selftest(E, rows)
    log(f"[cohort] {len(rows)} canonical VALIDATION users (rows with a non-empty target). "
        f"TEST IS NOT BUILT AND NOT TOUCHED.")
    leak_assert(E, rows, "validation")

    gen, gvocab = load_item_genres(E["ctx"].ni)
    s2i = getattr(E["ctx"], "show2id", None)
    if s2i is not None:
        with open(UNIQUE_SID) as f:
            sids = [int(x) for x in f.read().split()]
        mism = sum(1 for i, m in enumerate(sids) if int(s2i.get(m, -1)) != i)
        assert mism == 0, f"unique_sid.txt disagrees with ctx.show2id on {mism} items"
        log(f"[genre] sid mapping verified against ctx.show2id: 0 mismatches over {len(sids)} items")

    groups, n_no_liked, n_ungenred = partition_by_genre(E, rows, gen)
    sizes = {k: len(v) for k, v in groups.items()}
    order_groups = sorted(sizes, key=lambda k: (-sizes[k], k))     # biggest first
    log(f"[partition] favourite genre from LIKED fold-in items (level >= {LIKE_MIN_LEVEL}, "
        f">= 4.0 stars); most frequent genre wins; TIES BROKEN ALPHABETICALLY; no liked/genred item "
        f"-> '{NONE_GROUP}'. {len(sizes)} groups over {len(rows)} users.")
    for k in order_groups:
        flag = (f"   <<< UNDER {SMALL_GROUP} USERS: STATISTICALLY UNTRUSTWORTHY (reported, NOT "
                f"merged, NOT dropped)") if sizes[k] < SMALL_GROUP else ""
        log(f"[partition]   {k:22s} = {sizes[k]:6d} users ({100.0 * sizes[k] / len(rows):5.2f}%){flag}")
    n_small = sum(1 for v in sizes.values() if v < SMALL_GROUP)
    log(f"[partition] {n_small}/{len(sizes)} groups fall below {SMALL_GROUP} users. "
        f"HARD RULE 1: none merged, none dropped.")

    # ---- the pool: the SAME _ranked_pool call the global build makes -----------------------------
    item_tables = {"pop": np.asarray(E["ctx"].cnt, np.float64)}
    pool_all = [(0, int(i)) for i in _ranked_pool("pop", item_tables, args.n_items, "item")]
    pool = list(pool_all)
    L = args.L
    if args.sanity:
        L = 2
        # The comparator's own head must stay IN the truncated pool or the check is not a check: the
        # group would lose because the reference order was outside the search space, not because
        # conditioning failed. (Defect caught 2026-08-10.)
        pool = list(dict.fromkeys(list(global_seq[:L]) + pool[:12]))
    _poolset = set(pool)
    assert all(q in _poolset for q in global_seq[:L]), \
        f"the global order's first {L} items are not all inside the candidate pool"
    budgets = sorted(set([K for K in ITEM_BUDGETS if K <= L] + [L]))
    log(f"[pool] items-only, bank='pop', top-{args.n_items} -> {len(pool)} candidates; "
        f"L = {L} item steps per group, compared ITEM-FOR-ITEM against the global order at K in "
        f"{budgets}. Nothing is charged for the genre answer.")

    sig = {"tower": os.path.basename(args.snapshot), "conc": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "n_items": len(pool_all),
           "branch": "favourite_genre_from_liked_foldin", "L_items": L, "charged_for_genre": 0,
           "split": "validation", "n_cohort": len(rows), "sizes": sizes, "sanity": bool(args.sanity)}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    if part.get("sig") != sig:
        if part:
            log(f"[resume] sidecar {os.path.basename(PARTIAL)} carries a DIFFERENT contract -- ignored")
        part = {"sig": sig, "groups": {}}

    def flush():
        json.dump(part, open(PARTIAL, "w"), indent=2, default=float)

    # ---- IMPORT BANKED ORDERS, but only across an IDENTICAL contract -----------------------------
    # Field-by-field, with the older sidecar's field NAMES mapped explicitly. A silent pool across
    # two contracts is exactly the failure mode HARD RULE 8 and the stale-default scar are about.
    imported, contract_check = {}, None
    if args.import_banked:
        bp = args.import_banked if os.path.isabs(args.import_banked) \
            else os.path.join(OUTDIR, args.import_banked)
        assert os.path.exists(bp), f"--import_banked file missing: {bp}"
        bank = json.load(open(bp))
        bsig = bank.get("sig", {})
        # ("field", value_here, value_there) -- the older sidecar called L `L_items_after_genre`.
        checks = [("tower", sig["tower"], bsig.get("tower")),
                  ("concept_ckpt", sig["conc"], bsig.get("conc")),
                  ("objective", sig["objective"], bsig.get("objective")),
                  ("item_bank", sig["item_bank"], bsig.get("item_bank")),
                  ("n_items_pool", sig["n_items"], bsig.get("n_items")),
                  ("branch_rule", sig["branch"], bsig.get("branch")),
                  ("split", sig["split"], bsig.get("split")),
                  ("n_cohort", sig["n_cohort"], bsig.get("n_cohort")),
                  ("L_items", L, bsig.get("L_items", bsig.get("L_items_after_genre"))),
                  ("sanity", sig["sanity"], bsig.get("sanity")),
                  ("group_sizes", sig["sizes"], bsig.get("sizes"))]
        bad = [(f, a, b) for f, a, b in checks if a != b]
        contract_check = {"source": os.path.basename(bp),
                          "fields": [{"field": f, "this_run": a, "banked": b, "match": a == b}
                                     for f, a, b in checks],
                          "all_match": not bad}
        log("=" * 108)
        log(f"[contract] asserting the banked contract in {os.path.basename(bp)} field by field:")
        for f, a, b in checks:
            mark = "OK " if a == b else "*** MISMATCH ***"
            sa = str(a) if f != "group_sizes" else f"<{len(a)} groups>"
            sb = str(b) if f != "group_sizes" else f"<{len(b or {})} groups>"
            log(f"[contract]   {mark} {f:<16} this_run={sa}   banked={sb}")
        if bad:
            log("!" * 108)
            for f, a, b in bad:
                log(f"[contract] MISMATCH on '{f}': this run {a!r} vs banked {b!r}")
            raise SystemExit("REFUSING to pool results across two different contracts. "
                             "Nothing was merged; nothing was written.")
        log("[contract] ALL FIELDS MATCH -- the banked orders were produced under this exact "
            "contract, so they may be reused.")
        for g, r in bank.get("groups", {}).items():
            if r.get("done") and len(r.get("items", [])) == L:
                imported[g] = r
        log(f"[contract] importing {len(imported)} completed group order(s): "
            + ", ".join(f"{g}(n={imported[g]['n_users']})" for g in imported))
        log(f"[contract] {len(order_groups) - len(imported)} group(s) still to search.")
        log("=" * 108)

    run_groups = order_groups
    if args.sanity:
        big = [k for k in order_groups if sizes[k] >= SMALL_GROUP]
        run_groups = [(big[-1] if big else order_groups[0])]      # smallest TRUSTWORTHY group
        log(f"[sanity] running ONE group: {run_groups[0]} ({sizes[run_groups[0]]} users)")

    banked = {}
    for gname in run_groups:
        grows = groups[gname]
        if part["groups"].get(gname, {}).get("done"):
            log(f"[{gname}] already complete in the sidecar -- skipping")
            continue
        set_active_rows(E, grows)
        E["memo"] = {}                       # memo is keyed on the SEQUENCE only: it MUST NOT cross
        E["n_evals"] = 0; E["t_evals"] = 0.0  # cohorts. Reset with the cohort, every time.
        t0 = time.time()
        rec = {"n_users": len(grows), "branch_answer": gname, "done": False,
               "small_group": bool(len(grows) < SMALL_GROUP)}
        part["groups"][gname] = rec
        flush()

        def on_pick(seq, gains, nev, _r=rec, _t=t0):
            _r["items"] = [[int(a), int(b)] for a, b in seq]
            _r["order"] = [int(b) for _, b in seq]
            _r["gains"] = [float(x) for x in gains]
            _r["n_evals"] = int(nev)
            _r["sec_per_eval"] = round(E["t_evals"] / max(E["n_evals"], 1), 3)
            _r["minutes"] = round((time.time() - _t) / 60, 2)
            flush()                          # HARD RULE 11(a): on disk after EVERY pick

        if gname in imported:
            b = imported[gname]
            assert b["n_users"] == len(grows), \
                f"banked '{gname}' has n_users={b['n_users']} but this cohort has {len(grows)}"
            own_items = [(int(a), int(b2)) for a, b2 in b["items"]]
            gains, nev, spe = b.get("gains", []), int(b.get("n_evals", 0)), b.get("sec_per_eval", 0.0)
            rec["order_source"] = f"IMPORTED from {os.path.basename(args.import_banked)} " \
                                  f"(identical contract, asserted field by field)"
            rec["search_minutes_original"] = b.get("minutes")
            log(f"[{gname}] order IMPORTED (not re-searched): {[i for _, i in own_items]}")
        else:
            rec["order_source"] = "searched in this run"
            own_items, gains, nev = greedy_certified(E, grows, pool, L, args.objective, gname,
                                                     on_pick=on_pick)
            spe = E["t_evals"] / max(E["n_evals"], 1)
        rec.update({"items": [[int(a), int(b)] for a, b in own_items],
                    "order": [int(b) for _, b in own_items],
                    "gains": [float(g) for g in gains], "n_evals": int(nev),
                    "sec_per_eval": round(spe, 3),
                    "minutes": round((time.time() - t0) / 60, 2),
                    "rows": [int(r) for r in grows]})

        # ---- ITEM-FOR-ITEM curve, K vs K, plus PER-USER vectors banked for paired CIs ----------
        curve = {}
        for K in budgets:
            of, ot = genv_eval(E, own_items[:K], grows)
            lf, lt = genv_eval(E, global_seq[:K], grows)
            ia, _ = genv_answered(E, own_items[:K], grows)
            banked[f"{gname}|{K}|own_full"] = of.astype(np.float32)
            banked[f"{gname}|{K}|own_tail"] = ot.astype(np.float32)
            banked[f"{gname}|{K}|glob_full"] = lf.astype(np.float32)
            banked[f"{gname}|{K}|glob_tail"] = lt.astype(np.float32)
            curve[str(K)] = {"group_full@10": float(np.nanmean(of)),
                             "group_tail@10": float(np.nanmean(ot)),
                             "global_full@10": float(np.nanmean(lf)),
                             "global_tail@10": float(np.nanmean(lt)),
                             "diff_full@10": float(np.nanmean(of - lf)),
                             "diff_tail@10": float(np.nanmean(ot - lt)),
                             "n_full": int(np.isfinite(of).sum()),
                             "n_tail": int(np.isfinite(ot - lt).sum()),
                             "group_item_answered": round(ia, 3),
                             "overlap": order_overlap(own_items, global_seq, K)}
        rec["curve_item_matched"] = curve
        banked[f"{gname}|rows"] = np.asarray(grows, np.int64)

        jj = 0 if args.objective == "full" else 1
        K = len(own_items)
        o_k = genv_mean(E, own_items, grows)[jj]
        g_k = genv_mean(E, global_seq[:K], grows)[jj]
        rec["n_items_compared"] = K
        rec[f"own_minus_global@{K}"] = o_k - g_k
        rec["sanity_pass"] = bool(o_k >= g_k - 1e-9)
        rec["sanity_meaning"] = SANITY_MEANING
        rec["done"] = True
        flush()
        np.savez_compressed(NPZ, **banked)
        ov = curve[str(K)]["overlap"]
        log(f"[{gname}] DONE {len(grows)} users, {nev} evals, {spe:.3f}s/eval "
            f"({(time.time() - t0) / 60:.1f}m) | order {[i for _, i in own_items]}")
        log(f"[{gname}] @K={K}: own {o_k:.5f} vs global {g_k:.5f} ({o_k - g_k:+.5f}) | "
            f"overlap with global {ov['set_overlap']}/{K} items, {ov['positions_differ']} positions "
            f"differ, identical_order={ov['identical_order']}")
        if not rec["sanity_pass"]:
            log("!" * 108)
            log(f"[{gname}] SANITY *** FAIL *** : the group's OWN {K}-item greedy order scores BELOW "
                f"the GLOBAL {K}-item order on the group's OWN selection users: own {o_k:.5f} < "
                f"global {g_k:.5f} ({o_k - g_k:+.5f}). The global order is inside the search space, "
                f"so this cannot happen unless the search, the cohort masking or the memo is broken "
                f"-- or the group is small enough for CELF's step-wise argmax to be beaten by a "
                f"sequence it never enumerated (n_users={len(grows)}). DO NOT INTERPRET THIS GROUP.")
            log("!" * 108)

    # ---- the two readings that carry information -------------------------------------------------
    done = [g for g in part["groups"] if part["groups"][g].get("done")]
    tbl = []
    for g in sorted(done, key=lambda k: -part["groups"][k]["n_users"]):
        r = part["groups"][g]; c = r["curve_item_matched"]; Kc = str(r["n_items_compared"])
        tbl.append({"group": g, "n_users": r["n_users"], "small_group": r["small_group"],
                    "diff_full@16": c[Kc]["diff_full@10"], "diff_tail@16": c[Kc]["diff_tail@10"],
                    "set_overlap@16": c[Kc]["overlap"]["set_overlap"],
                    "positions_differ@16": c[Kc]["overlap"]["positions_differ"],
                    "sanity_pass": r["sanity_pass"], "sec_per_eval": r["sec_per_eval"]})
    weighted = {}
    for K in budgets:
        df, dt = [], []
        for g in done:
            k = f"{g}|{K}|"
            if k + "own_full" in banked:
                df.append(banked[k + "own_full"] - banked[k + "glob_full"])
                dt.append(banked[k + "own_tail"] - banked[k + "glob_tail"])
        if df:
            df = np.concatenate(df); dt = np.concatenate(dt)
            weighted[str(K)] = {"diff_full@10": float(np.nanmean(df)),
                                "diff_tail@10": float(np.nanmean(dt)),
                                "n_users": int(np.isfinite(df).sum())}
    _log_gain_table(tbl, weighted)

    out = {"analysis": "group_conditioned_greedy_genre_split_16v16",
           "stage": "1_build_on_validation",
           "label": args.label, "SANITY": bool(args.sanity), "citable": not args.sanity,
           "split": "validation", "test_touched": False,
           "generous_assumption": GENEROUS, "sanity_meaning": SANITY_MEANING,
           "tower": os.path.basename(args.snapshot), "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "n_items_pool": len(pool_all),
           "branch_question": "OPEN: what is your favourite genre? (FREE -- no turn charged)",
           "branch_rule": (f"most frequent MovieLens genre over the user's LIKED FOLD-IN items "
                           f"(lvl_lookup level >= {LIKE_MIN_LEVEL} == >= 4.0 stars); ties broken "
                           f"ALPHABETICALLY; no liked/genred fold-in item -> group '{NONE_GROUP}'. "
                           f"Held-out targets are excluded by construction and asserted at runtime."),
           "genre_source": os.path.relpath(MOVIES_CSV, _ROOT),
           "sid_source": os.path.relpath(UNIQUE_SID, _ROOT),
           "n_catalogue_items": int(E["ctx"].ni),
           "n_items_with_genre": len(gen), "genre_vocabulary": gvocab,
           "n_users_no_liked_folding": n_no_liked, "n_users_liked_but_ungenred": n_ungenred,
           "comparison": ("ITEM-FOR-ITEM: the group's first K items vs the global order's first K "
                          "items, K in %s. There is no total-budget curve: the genre answer is "
                          "free." % (budgets,)),
           "global_sequence": [[int(a), int(b)] for a, b in global_seq],
           "n_cohort": len(rows), "n_groups": len(sizes), "group_sizes": sizes,
           "small_group_threshold": SMALL_GROUP,
           "n_groups_below_threshold": n_small,
           "budgets_items": budgets, "L_items": L,
           "contract_assertion": contract_check,
           "gain_vs_group_size": tbl, "cohort_weighted": weighted,
           "per_user_npz": os.path.basename(NPZ),
           "groups": part["groups"], "sig": sig,
           "seconds": round(time.time() - t00, 1),
           "baseline_sec_per_eval_10k": gblob.get("sec_per_eval_build")}
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    np.savez_compressed(NPZ, **banked)
    log(f"[done] wrote {os.path.basename(OUT)} + {os.path.basename(NPZ)} "
        f"in {(time.time() - t00) / 60:.1f}m")
    log("[cost] s/eval by group: " + " | ".join(
        f"{t['group']}: n={t['n_users']} {t['sec_per_eval']}s" for t in tbl))
    return out


if __name__ == "__main__":
    main()
