"""Genre-split group-conditioned greedy — the answerable-branch version of the depth-1 control.

WHY THIS EXISTS. `group_greedy_d1.py` branched on the answer to ONE popular ITEM (item 83, the global
greedy's own first pick). 70.45% of the validation cohort could not answer it, and in exactly that
70% the conditioning gain was ~0.000 (+0.0019); the two groups that DID answer gained +0.0129 and
+0.0237. So the depth-1 result was budget-limited by ANSWERABILITY, not by conditioning. This run
replaces the branch question with one nearly everyone can answer: "what is your favourite genre?".

WHAT THE GENRE QUESTION IS, PRECISELY:
  * It is an OPEN first question (no closed vocabulary is presented to the user), simulated by reading
    the user's own FOLD-IN profile: among their LIKED fold-in items (level >= LIKE_MIN_LEVEL = 7,
    i.e. >= 4.0 stars) take the most frequent MovieLens genre, ties broken ALPHABETICALLY. Held-out
    targets are excluded by construction: `sh["lvl_lookup"]` is built from `ctx.allb`, which
    `build_real_ctx` builds with `keepm = ~g["sid"].isin(te_sets[r])`. This is asserted at runtime.
  * Users with NO liked fold-in item (or no liked item carrying a genre) form an explicit `none`
    group. They are run, not dropped (HARD RULE 1).
  * The answer is used ONLY to select the group, i.e. to condition the SELECTOR. It is NOT folded
    into the recommender: the pool is items-only (design point 4), the scored sequence is items only,
    and there is no concept/genre token in it. The genre turn therefore buys the interview nothing
    except a better ordering of the remaining items -- and it still COSTS A TURN. That is the whole
    point of the budget accounting below.

BUDGET ACCOUNTING (design point 3).
  * A group's interview is [genre question] + L=15 items => 16 turns total.
  * The comparator is the global 16-step items-only order from `greedy_static_i26c.json`.
  * HEADLINE COMPARISON, at MATCHED TOTAL BUDGET T: the group's first (T-1) ITEMS against the global
    order's first T ITEMS. Reported at T in {1, 2, 4, 8, 16}. At T=1 the group has asked only the
    genre question and has zero items -- that cell is the cold intercept and is kept for completeness.
  * CONTAINMENT / SANITY GATE, at MATCHED ITEM COUNT K: the group's K items against the global
    order's first K items, on the group's OWN users. This -- and only this -- is the check whose
    rationale is "the global order is inside the search space the group's greedy explored". At
    matched TOTAL budget the global order has one MORE item than the group, so it is NOT inside the
    group's search space and a shortfall there is a real finding, not a bug. Both numbers are
    recorded; `sanity_pass` gates on the item-matched one and logs a screaming FAIL.

VALIDATION ONLY. The TEST cohort is not built, not loaded and not touched.

PERSISTENCE (HARD RULE 11a): `*_partial.json` is rewritten after EVERY pick. A kill costs one pick.
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
                                    genv_mean, genv_answered, contract_selftest, _ranked_pool,
                                    CERT_TOWER, CERT_CONC)
from i26_encoder import LIKE_MIN_LEVEL

LABEL = "groupgreedy_genre"
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
GLOBAL_JSON = os.path.join(OUTDIR, "greedy_static_i26c.json")
MOVIES_CSV = os.path.join(_ROOT, "data", "movielens", "movies.csv")     # the ML-25M raw movies file
UNIQUE_SID = os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")
NO_GENRE = "(no genres listed)"
NONE_GROUP = "none"
TOTAL_BUDGETS = (1, 2, 4, 8, 16)          # TOTAL turns, genre question INCLUDED
SMALL_GROUP = 200                          # below this a group is flagged untrustworthy, never merged


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


def _partition(E, rows, gen):
    """Partition the cohort by favourite genre off the GRADED FOLD-IN lookup (targets excluded)."""
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


def _leak_assert(E, rows):
    """The grouping must never see a held-out target. lvl_lookup is built from ctx.allb, which is
    built with the te rows removed -- assert it here rather than trust the comment."""
    ctx = E["ctx"]; lvl = E["lvl"]
    bad = 0
    for r in rows:
        te = set(ctx.va_te[r].indices.tolist())
        if te & set(lvl[r].keys()):
            bad += 1
    assert bad == 0, f"LEAK: {bad} users have a held-out target inside the fold-in level lookup"
    log(f"[leak] 0/{len(rows)} users have any held-out target inside the grouping lookup -- the "
        f"favourite-genre answer is derived from FOLD-IN ratings only")


def _curves(E, own_items, global_seq, grows):
    """Both budget conventions, side by side, on this group's own users."""
    tot = {}
    for T in TOTAL_BUDGETS:
        gi = own_items[:max(T - 1, 0)]                 # genre turn burns 1 of the T turns
        gl = global_seq[:T]                            # global spends all T on items
        gf, gt = genv_mean(E, gi, grows)
        lf, lt = genv_mean(E, gl, grows)
        ia, _ = genv_answered(E, gi, grows)
        tot[str(T)] = {"group_items_asked": len(gi), "global_items_asked": len(gl),
                       "group_full@10": gf, "group_tail@10": gt,
                       "global_full@10": lf, "global_tail@10": lt,
                       "diff_full@10": gf - lf, "diff_tail@10": gt - lt,
                       "group_item_answered": round(ia, 3)}
    itm = {}
    for K in (1, 2, 4, 8, len(own_items)):
        if K > len(own_items):
            continue
        gf, gt = genv_mean(E, own_items[:K], grows)
        lf, lt = genv_mean(E, global_seq[:K], grows)
        itm[str(K)] = {"group_full@10": gf, "global_full@10": lf, "diff_full@10": gf - lf,
                       "group_tail@10": gt, "global_tail@10": lt, "diff_tail@10": gt - lt}
    return tot, itm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--L", type=int, default=15,
                    help="ITEM steps after the genre question (total interview = 1 + L = 16)")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--sanity", action="store_true",
                    help="end-to-end pipeline check: L=2 on ONE group from a 12-item pool. Writes "
                         "*_SANITY.json and is NEVER citable.")
    args = ap.parse_args()

    t00 = time.time()
    sfx = "_SANITY" if args.sanity else ""
    OUT = os.path.join(OUTDIR, f"{args.label}{sfx}.json")
    PARTIAL = os.path.join(OUTDIR, f"{args.label}{sfx}_partial.json")
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- phase 1: every FILE path exists before anything expensive runs ------------------------
    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("global greedy json", GLOBAL_JSON), ("movies.csv", MOVIES_CSV),
                  ("unique_sid.txt", UNIQUE_SID)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"
    if args.sanity:
        log("#" * 100)
        log("# SANITY MODE: L=2 on ONE group, 12-item pool. Pipeline check ONLY, NOT citable.")
        log("#" * 100)

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
    _leak_assert(E, rows)

    gen, gvocab = load_item_genres(E["ctx"].ni)
    s2i = getattr(E["ctx"], "show2id", None)
    if s2i is not None:
        with open(UNIQUE_SID) as f:
            sids = [int(x) for x in f.read().split()]
        mism = sum(1 for i, m in enumerate(sids) if int(s2i.get(m, -1)) != i)
        assert mism == 0, f"unique_sid.txt disagrees with ctx.show2id on {mism} items"
        log("[genre] sid mapping verified against ctx.show2id: 0 mismatches over 18359 items")

    groups, n_no_liked, n_ungenred = _partition(E, rows, gen)
    sizes = {k: len(v) for k, v in groups.items()}
    order_groups = sorted(sizes, key=lambda k: (-sizes[k], k))     # biggest first
    log(f"[partition] favourite genre from LIKED fold-in items (level >= {LIKE_MIN_LEVEL}, "
        f">= 4.0 stars); most frequent genre wins; TIES BROKEN ALPHABETICALLY; no liked/genred item "
        f"-> '{NONE_GROUP}'. {len(sizes)} groups over {len(rows)} users.")
    n_small = 0
    for k in order_groups:
        flag = ""
        if sizes[k] < SMALL_GROUP:
            flag = f"   <<< UNDER {SMALL_GROUP} USERS: STATISTICALLY UNTRUSTWORTHY (reported, NOT merged, NOT dropped)"
            n_small += 1
        log(f"[partition]   {k:22s} = {sizes[k]:6d} users ({100.0 * sizes[k] / len(rows):5.2f}%){flag}")
    log(f"[partition] {n_small}/{len(sizes)} groups fall below {SMALL_GROUP} users. "
        f"HARD RULE 1: none merged, none dropped.")

    # ---- the pool: the SAME _ranked_pool call the global build makes -----------------------------
    item_tables = {"pop": np.asarray(E["ctx"].cnt, np.float64)}
    pool_all = [(0, int(i)) for i in _ranked_pool("pop", item_tables, args.n_items, "item")]
    pool = list(pool_all)
    L = args.L
    if args.sanity:
        L = 2
        # The comparator's own head must stay IN the truncated pool or neither budget check is a
        # check: it would fail because the reference order was outside the search space, not because
        # anything is wrong. L+1 because the TOTAL-budget comparison reads global_seq[:L+1].
        pool = list(dict.fromkeys(list(global_seq[:L + 1]) + pool[:12]))
    # DEFECT GUARD (caught on 2026-08-10): a truncated pool that excludes the comparator's own head
    # makes both budget checks meaningless -- the group would lose because the reference order was
    # outside the search space, not because conditioning failed.
    _poolset = set(pool)
    assert all(q in _poolset for q in global_seq[:L + 1]), \
        f"the global order's first {L + 1} items are not all inside the candidate pool"
    log(f"[pool] items-only, bank='pop', top-{args.n_items} -> {len(pool)} candidates "
        f"(top-5 {[i for _, i in pool[:5]]}); L = {L} item steps after the genre question")

    sig = {"tower": os.path.basename(args.snapshot), "conc": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "n_items": len(pool_all),
           "branch": "favourite_genre_from_liked_foldin", "L_items_after_genre": L,
           "split": "validation", "n_cohort": len(rows), "sizes": sizes, "sanity": bool(args.sanity)}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    if part.get("sig") != sig:
        if part:
            log(f"[resume] sidecar {os.path.basename(PARTIAL)} carries a DIFFERENT contract -- ignored")
        part = {"sig": sig, "groups": {}}

    def flush():
        json.dump(part, open(PARTIAL, "w"), indent=2, default=float)

    run_groups = order_groups
    if args.sanity:
        big = [k for k in order_groups if sizes[k] >= SMALL_GROUP]
        run_groups = [(big[-1] if big else order_groups[0])]      # smallest TRUSTWORTHY group
        log(f"[sanity] running ONE group: {run_groups[0]} ({sizes[run_groups[0]]} users)")

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
            _r["interview"] = ["GENRE:" + gname] + [int(b) for _, b in seq]
            _r["gains"] = [float(x) for x in gains]
            _r["n_evals"] = int(nev)
            _r["sec_per_eval"] = round(E["t_evals"] / max(E["n_evals"], 1), 3)
            _r["minutes"] = round((time.time() - _t) / 60, 2)
            flush()                          # HARD RULE 11(a): on disk after EVERY pick

        own_items, gains, nev = greedy_certified(E, grows, pool, L, args.objective, gname,
                                                 on_pick=on_pick)
        spe = E["t_evals"] / max(E["n_evals"], 1)
        rec.update({"items": [[int(a), int(b)] for a, b in own_items],
                    "interview": ["GENRE:" + gname] + [int(b) for _, b in own_items],
                    "gains": [float(g) for g in gains], "n_evals": int(nev),
                    "sec_per_eval": round(spe, 3),
                    "minutes": round((time.time() - t0) / 60, 2)})
        ct, ci = _curves(E, own_items, global_seq, grows)
        rec["curve_total_budget"] = ct        # HEADLINE: genre turn charged
        rec["curve_item_matched"] = ci        # CONTAINMENT: same number of items on both sides

        jj = 0 if args.objective == "full" else 1
        K = len(own_items)
        o_k = genv_mean(E, own_items, grows)[jj]
        g_k = genv_mean(E, global_seq[:K], grows)[jj]           # item-matched: K vs K
        g_t = genv_mean(E, global_seq[:K + 1], grows)[jj]       # total-budget: K vs K+1
        rec["n_items_compared"] = K
        rec["own_minus_global_ITEM_MATCHED@%d" % K] = o_k - g_k
        rec["own_minus_global_TOTAL_BUDGET@%d" % (K + 1)] = o_k - g_t
        rec["sanity_pass"] = bool(o_k >= g_k - 1e-9)            # gates on CONTAINMENT only
        rec["done"] = True
        flush()
        log(f"[{gname}] DONE {len(grows)} users, {nev} evals, {spe:.3f}s/eval "
            f"({(time.time() - t0) / 60:.1f}m) | items {[i for _, i in own_items]}")
        log(f"[{gname}] TOTAL-BUDGET @{K + 1} turns: group [genre + {K} items] {o_k:.5f} vs global "
            f"[{K + 1} items] {g_t:.5f} ({o_k - g_t:+.5f})  <-- the genre turn is CHARGED here")
        if rec["sanity_pass"]:
            log(f"[{gname}] SANITY OK (item-matched, {K} vs {K}): own {o_k:.5f} >= global {g_k:.5f} "
                f"({o_k - g_k:+.5f})")
        else:
            log("!" * 100)
            log(f"[{gname}] SANITY *** FAIL *** : the group's OWN greedy order scores BELOW the "
                f"GLOBAL order's first {K} items on the group's OWN selection users: own {o_k:.5f} "
                f"< global {g_k:.5f} ({o_k - g_k:+.5f}). The global head is inside the search space, "
                f"so this cannot happen if the search, the cohort masking and the memo are correct "
                f"-- UNLESS the group is small enough for greedy's step-wise argmax to be beaten by "
                f"a sequence it never enumerated (n_users={len(grows)}). DO NOT INTERPRET THIS GROUP.")
            log("!" * 100)

    out = {"analysis": "group_conditioned_greedy_genre_split",
           "label": args.label, "SANITY": bool(args.sanity), "citable": not args.sanity,
           "split": "validation", "test_touched": False,
           "tower": os.path.basename(args.snapshot), "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "n_items_pool": len(pool_all),
           "branch_question": "OPEN: what is your favourite genre?",
           "branch_rule": (f"most frequent MovieLens genre over the user's LIKED FOLD-IN items "
                           f"(lvl_lookup level >= {LIKE_MIN_LEVEL} == >= 4.0 stars); ties broken "
                           f"ALPHABETICALLY; no liked/genred fold-in item -> group '{NONE_GROUP}'. "
                           f"Held-out targets are excluded by construction and asserted at runtime."),
           "genre_source": os.path.relpath(MOVIES_CSV, _ROOT),
           "sid_source": os.path.relpath(UNIQUE_SID, _ROOT),
           "n_catalogue_items": int(E["ctx"].ni),
           "n_items_with_genre": len(gen), "genre_vocabulary": gvocab,
           "n_users_no_liked_folding": n_no_liked, "n_users_liked_but_ungenred": n_ungenred,
           "budget_accounting": (
               "The genre question COSTS ONE TURN and is NOT folded into the recommender (it only "
               "selects the group). HEADLINE `curve_total_budget[T]`: the group's first T-1 ITEMS vs "
               "the global order's first T ITEMS. CONTAINMENT `curve_item_matched[K]`: K items vs K "
               "items -- this is what `sanity_pass` gates on, because only there is the comparator "
               "inside the group's search space."),
           "global_sequence": [[int(a), int(b)] for a, b in global_seq],
           "n_cohort": len(rows), "n_groups": len(sizes), "group_sizes": sizes,
           "small_group_threshold": SMALL_GROUP,
           "n_groups_below_threshold": int(sum(1 for v in sizes.values() if v < SMALL_GROUP)),
           "total_budgets": list(TOTAL_BUDGETS),
           "groups": part["groups"], "sig": sig,
           "seconds": round(time.time() - t00, 1),
           "baseline_sec_per_eval_10k": gblob.get("sec_per_eval_build")}
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    log(f"[done] wrote {os.path.basename(OUT)} in {(time.time() - t00) / 60:.1f}m")
    log("[cost] s/eval by group vs the 13.543 s/eval global build at 10,000 users: " + " | ".join(
        f"{g}: n={part['groups'][g].get('n_users', 0)} "
        f"{part['groups'][g].get('sec_per_eval', float('nan'))}s" for g in part["groups"]))
    return out


if __name__ == "__main__":
    main()
