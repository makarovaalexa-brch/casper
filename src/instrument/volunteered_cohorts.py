r"""EXPERIMENT V -- VOLUNTEERED COHORTS. Does evidence the user VOLUNTEERS support cohort
conditioning as well as the answer to an ASKED question does?

> **AUTHOR SAID (2026-08-16, verbatim): "ok, go ahead."**

THE QUESTION
------------
The banked genre arm (`group_greedy_genre.py` -> `groupgreedy_genre15*`) splits users by their answer
to ONE asked genre question, gives each group its own optimised 15-item order, and gains
+0.01831 full / +0.00730 tail on the 10,000 VALIDATION users against the rebuilt policy-B items
comparator (`groupgreedy_genre15_VAL_policyB.json`), and +0.010153 on TEST.

Experiment R showed the distilled policy is INERT when the user volunteers evidence, and the z-only
ablation showed that when forced to respond it responds BADLY. Neither measured whether there is
anything TO respond to. **V measures the CEILING, not a policy**: it gives each volunteered-evidence
cohort its own optimised static order and asks how much that conditioning is worth.

THE ONLY THING THAT CHANGES vs THE GENRE ARM IS HOW A COHORT IS FORMED
---------------------------------------------------------------------
  genre arm  cohort = the answer to one ASKED genre question (favourite genre over the user's LIKED
             FOLD-IN items, ties alphabetical, no liked/genred item -> 'none').
  V          cohort = the same genre signature computed over the user's VOLUNTEERED items only --
             k=4 items drawn from the fold-in half, popularity-weighted, `default_rng(98765)`,
             EXACTLY R's draw (`policy_resume.draw_volunteered`, imported, not re-implemented, and
             asserted field-by-field against R's banked `resume_k4_static.json` draw block).

Everything else is the genre arm's: the same 500-item pop pool, the same L=15, the same budget list
{1,2,4,8,15}, the same 200-user small-group threshold, the same two pre-specified arms (all_groups
and fallback_200), the same paired percentile bootstrap (10,000 resamples, seed 98765), the same
certified pair, the same policy-B mask.

THE ONE STRUCTURAL DIFFERENCE THAT IS FORCED, AND WHY
----------------------------------------------------
In V the volunteered items are FOLDED as ordinary evidence before the interview starts (that is what
"volunteered" means -- R's state structure). So EVERY score in this file, for the group orders AND
for the global comparator, is measured FROM THE SEEDED STATE: user u's scored item sequence is
`dedup(vol_u + asked[:K])`. A re-asked volunteered item is folded ONCE and the turn is burnt -- R's
semantics, `policy_resume.build_seqs`, imported. Because every user carries a different sequence the
scorer is `genv_eval_multi`, not `genv_eval`; it is the SAME contract (asserted here: with identical
sequences the two agree bit-for-bit).

THE PLACEBO ARM, WHICH IS NOT OPTIONAL (HARD RULE 5)
----------------------------------------------------
On the SELECTION cohort a group's own order is GUARANTEED to be >= the global order, because the
global order lies inside every group's search space. So an in-sample cohort gain is >= 0 by
construction and its MAGNITUDE is the only thing that carries information -- against WHAT is the
question. `--cohorts placebo` re-runs the identical machinery on RANDOM cohorts with the IDENTICAL
size profile (a permutation of the real label vector, `default_rng(98765)`). That is the null level
for "how much does splitting 10,000 users into groups of these sizes and re-optimising buy, when the
split carries no information". V minus placebo is the reading; V alone is not.

VALIDATION ONLY. The 10,000 canonical TEST users are NOT built, NOT loaded, NOT named. The one
unspent test look is the author's to spend.

PERSISTENCE (HARD RULE 11): `*_partial.json` after every pick, a per-group CELF heap sidecar under a
contract signature, per-user vectors banked to `*_per_user.npz`.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import heapq
import hashlib
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_eval,
                                    genv_eval_multi, contract_selftest, _ranked_pool,
                                    pool_contract, CERT_TOWER, CERT_CONC)
from group_greedy_genre import (load_item_genres, favourite_genre, leak_assert, order_overlap,
                                UNIQUE_SID, SMALL_GROUP, NONE_GROUP)
from policy_resume import draw_volunteered, build_seqs, dedup_keep_order
from i26_encoder import LIKE_MIN_LEVEL

LABEL = "volcohorts_k4"
OUTDIR = os.path.join(_ROOT, "experiments", "battery", "volcohorts")
COMPARATOR_JSON = os.path.join(_ROOT, "experiments", "battery", "greedy_static_policyB_items.json")
GENRE_VAL_JSON = os.path.join(_ROOT, "experiments", "battery", "groupgreedy_genre15_VAL_policyB.json")
GENRE_VAL_NPZ = os.path.join(_ROOT, "experiments", "battery",
                             "groupgreedy_genre15_VAL_policyB_per_user.npz")
GENRE_S1_JSON = os.path.join(_ROOT, "experiments", "battery", "groupgreedy_genre15.json")
GENRE_S1_NPZ = os.path.join(_ROOT, "experiments", "battery", "groupgreedy_genre15_per_user.npz")
R_K4_STATIC = os.path.join(_ROOT, "experiments", "battery", "resume", "resume_k4_static.json")

VAL_M0_FULL = 0.15686538118354684          # canonical m=0 snap, policy B, 10,000 validation users
N_BOOT = 10000
BOOT_SEED = 98765
SEED = 98765
K_DEFAULT = 4                              # volunteered items per user (R's middle rung)
BUDGETS = (1, 2, 4, 8, 15)
L_ITEMS = 15

CEILING_NOTE = (
    "V measures a CEILING. Each cohort is handed its OWN greedy-optimised static order, fitted on "
    "the very users it is scored on. It does NOT show that any learned policy can capture this, and "
    "the gap between the ceiling and what a policy achieves is the chapter's open question. Do not "
    "blur the two.")
SANITY_MEANING = (
    "On the SELECTION cohort a group's own K-item order is GUARANTEED to be >= the global order's "
    "first K items, because the global order lies inside every group's search space. `sanity_pass` "
    "is therefore a HARNESS check and NEVER evidence. What carries information: the MAGNITUDE of the "
    "gain against the PLACEBO arm's magnitude, the gain-vs-group-size relation, and the order "
    "overlap with the global order.")


# ============================================================================ inference
def paired_bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CI over the PER-USER paired difference vector. Users, not cells, are resampled.
    Never the unpaired 1.96*sqrt(2)*SE form (memory `paired-test-kills-the-tie-claim`)."""
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


# ============================================================================ seeded scoring
def seeded_eval(E, grows, vol, item_ids, score=True):
    """(full, tail) per user for ONE asked ITEM order, scored FROM THE SEEDED STATE.

    `vol` is the volunteered list for each row of `grows`, in the same order. The scored sequence is
    dedup(volunteered + asked): a re-asked volunteered item folds once and burns the turn.
    """
    iseqs = [dedup_keep_order(list(v) + [int(i) for i in item_ids]) for v in vol]
    cseqs = [[] for _ in grows]
    return genv_eval_multi(E, grows, iseqs, cseqs, score=score)


def redundancy(vol, item_ids):
    """(mean re-asks per user, frac of users with >=1) for one asked order against the seeds."""
    ids = [int(i) for i in item_ids]
    c = np.asarray([sum(1 for i in ids if i in set(v)) for v in vol], np.int64)
    return float(c.mean()), float((c > 0).mean()), int(c.sum())


# ============================================================================ CELF, seeded
def _sha(o):
    return hashlib.sha1(json.dumps(o, separators=(",", ":")).encode()).hexdigest()


def heap_sig(E, grows, vol, pool, objective, k_vol):
    return {"v": 1, "kind": "seeded", "split": E["split"], "objective": objective,
            "tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "k_vol": int(k_vol),
            "mask_pool": pool_contract(E["pool"], grows, policy=E["pool_policy"]),
            "n_pool": len(pool), "pool_sha1": _sha([int(i) for i in pool]),
            "n_rows": len(grows), "rows_sha1": _sha(sorted(int(r) for r in grows)),
            "vol_sha1": _sha([[int(x) for x in v] for v in vol])}


def greedy_seeded(E, grows, vol, pool, L, objective, tag, memo, log_every=50,
                  on_pick=None, heap_path=None, sig=None):
    """Lazy-greedy (CELF) best static ITEM order of length L, maximising the cohort-mean seeded
    NDCG@10. Structurally identical to `strategy_channel_suite.greedy_certified`; the ONLY change is
    that the objective is evaluated from the seeded state (`seeded_eval`) instead of the empty one."""
    assert objective in ("full", "tail")
    j = 0 if objective == "full" else 1
    t0 = time.time()

    def val(ids):
        key = tuple(int(x) for x in ids)
        if key in memo:
            return memo[key]
        f, t = seeded_eval(E, grows, vol, key)
        v = (float(np.nanmean(f)), float(np.nanmean(t)))
        memo[key] = v
        return v

    def _save(heap, seq, gains, base, cur, nev):
        if not heap_path:
            return
        blob = {"sig": sig, "tag": tag, "base": float(base), "cur": float(cur), "nev": int(nev),
                "seq": [int(x) for x in seq], "gains": [float(g) for g in gains],
                "heap": [[float(a), int(q), int(u)] for a, q, u in heap]}
        os.makedirs(os.path.dirname(heap_path) or ".", exist_ok=True)
        with open(heap_path + ".tmp", "w") as fh:
            json.dump(blob, fh)
        os.replace(heap_path + ".tmp", heap_path)

    banked = None
    if heap_path and os.path.exists(heap_path):
        try:
            blob = json.load(open(heap_path))
        except Exception as e:
            blob = None
            log(f"[greedy {tag}] heap sidecar unreadable ({e}) -- ignored, seeding cold")
        if blob is not None and blob.get("sig") == sig:
            banked = blob
        elif blob is not None:
            log(f"[greedy {tag}] heap sidecar carries a DIFFERENT contract -- IGNORED, seeding cold")

    if banked is not None:
        base = float(banked["base"]); cur = float(banked["cur"]); nev = int(banked["nev"])
        seq = [int(x) for x in banked["seq"]]; gains = [float(g) for g in banked["gains"]]
        heap = [(float(a), int(q), int(u)) for a, q, u in banked["heap"]]
        heapq.heapify(heap)
        log(f"[greedy {tag}] RESUMED: {len(seq)} pick(s), {len(heap)} heap entries, {nev} evals paid")
        if len(seq) >= L:
            return seq[:L], gains[:L], nev
    else:
        base = val(())[j]
        heap = []; nev = 0
        for q in pool:
            v = val((q,))[j]; nev += 1
            heapq.heappush(heap, (-(v - base), int(q), 0))
            if nev % log_every == 0:
                el = time.time() - t0
                log(f"[greedy {tag}] seed {nev}/{len(pool)} ({el / 60:.1f}m, {el / nev:.2f}s/eval)")
        seq = []; gains = []; cur = base
        _save(heap, seq, gains, base, cur, nev)
    while len(seq) < L and heap:
        neg, q, upd = heapq.heappop(heap)
        if upd == len(seq):
            seq.append(int(q)); gains.append(-neg); cur += -neg
            log(f"[greedy {tag}] pick {len(seq)}/{L} = item {q} gain {-neg:+.5f} -> {cur:.5f} "
                f"(nev={nev}, {(time.time() - t0) / 60:.1f}m)")
            _save(heap, seq, gains, base, cur, nev)
            if on_pick is not None:
                on_pick(list(seq), list(gains), nev)
        else:
            v = val(tuple(seq) + (int(q),))[j]; nev += 1
            heapq.heappush(heap, (-(v - cur), int(q), len(seq)))
    return seq, gains, nev


# ============================================================================ cohorts
def volunteered_genre(vol_ids, lvl_row, gen):
    """The genre signature of what the user VOLUNTEERED. Same rule as the genre arm, different
    evidence: most frequent MovieLens genre over the LIKED volunteered items (level >=
    LIKE_MIN_LEVEL), ties broken ALPHABETICALLY; no liked/genred volunteered item -> 'none'."""
    liked = [int(i) for i in vol_ids if int(lvl_row.get(int(i), -1)) >= LIKE_MIN_LEVEL]
    g = favourite_genre(liked, gen)
    return g if g is not None else NONE_GROUP


def partition(E, rows, vol, gen):
    lvl = E["lvl"]
    labels = []
    n_no_liked = n_ungenred = 0
    for r, v in zip(rows, vol):
        d = lvl[int(r)]
        liked = [int(i) for i in v if int(d.get(int(i), -1)) >= LIKE_MIN_LEVEL]
        g = favourite_genre(liked, gen)
        if g is None:
            g = NONE_GROUP
            if liked:
                n_ungenred += 1
            else:
                n_no_liked += 1
        labels.append(g)
    return labels, n_no_liked, n_ungenred


# ============================================================================ the run
def main():
    ap = argparse.ArgumentParser(description="Experiment V: volunteered-evidence cohorts")
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--k_vol", type=int, default=K_DEFAULT)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--L", type=int, default=L_ITEMS)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--cohorts", default="volunteered", choices=["volunteered", "placebo"],
                    help="`placebo` = random cohorts with the IDENTICAL size profile (HARD RULE 5)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--census_only", action="store_true",
                    help="print the cohort composition and stop -- no search, no orders")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.census_only or args.i_have_author_goahead, \
        "REFUSING to run the search without --i_have_author_goahead"
    label = args.label or f"{LABEL}_{args.cohorts}"
    t00 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    OUT = os.path.join(OUTDIR, f"{label}.json")
    PARTIAL = os.path.join(OUTDIR, f"{label}_partial.json")
    NPZ = os.path.join(OUTDIR, f"{label}_per_user.npz")

    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("comparator", COMPARATOR_JSON), ("unique_sid", UNIQUE_SID),
                  ("R k=4 static cell", R_K4_STATIC)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"

    log("#" * 110)
    log("# EXPERIMENT V -- VOLUNTEERED COHORTS. VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("# " + CEILING_NOTE)
    log("#" * 110)

    cmp_blob = json.load(open(COMPARATOR_JSON))
    assert cmp_blob["channel"] == "items" and cmp_blob["mask_policy"] == "B" and cmp_blob["citable"]
    comparator = [int(i) for ch, i in cmp_blob["sequence"]]
    assert all(int(ch) == 0 for ch, _ in cmp_blob["sequence"]), "comparator is not items-only"

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"{label}(val)")
    rows = E["rows"]
    assert len(rows) == 10000, f"cohort is {len(rows)} users, expected 10,000"
    contract_selftest(E, rows)
    leak_assert(E, rows, "validation")
    set_active_rows(E, rows)

    # ---- GATE 1: canonical m=0 snap ------------------------------------------------------------
    f0, _ = genv_eval(E, [], rows)
    f0 = float(np.nanmean(f0))
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r} != {VAL_M0_FULL}"
    log(f"[gate 1] canonical m=0 snap {f0:.17g} == banked {VAL_M0_FULL:.17g}")

    # ---- GATE 2: the comparator reproduces its banked UNSEEDED curve exactly --------------------
    gate2 = {}
    for K in (1, 2, 4, 8, 16):
        f, t = genv_eval(E, [(0, i) for i in comparator[:K]], rows)
        got, want = float(np.nanmean(f)), cmp_blob["curve_policyB"][str(K)]["full@10"]
        gate2[str(K)] = {"got": got, "banked": want, "exact": got == want}
        assert got == want, f"COMPARATOR GATE FAILED at K={K}: {got!r} != {want!r}"
    log(f"[gate 2] the banked global items-only comparator reproduces its curve EXACTLY at "
        f"K in {sorted(gate2, key=int)} (unseeded, 5/5 bit-identical)")

    # ---- GATE 3: genv_eval_multi == genv_eval when every user carries the same sequence ---------
    probe = [(0, i) for i in comparator[:4]]
    fa, ta = genv_eval(E, probe, rows)
    fb, tb = genv_eval_multi(E, rows, [[i for _, i in probe]] * len(rows),
                             [[] for _ in rows])
    assert np.array_equal(fa, fb, equal_nan=True) and np.array_equal(ta, tb, equal_nan=True), \
        "genv_eval_multi disagrees with genv_eval on an identical-sequence cohort"
    log("[gate 3] genv_eval_multi == genv_eval bit-for-bit on an identical-sequence cohort "
        "(the seeded scorer is the same ruler, not a second one)")

    # ---- the draw: R's, imported, and asserted against R's banked block -------------------------
    vol, draw = draw_volunteered(E, rows, args.k_vol, SEED)
    rbank = json.load(open(R_K4_STATIC))["draw"]
    if args.k_vol == 4:
        for f in ("k_requested", "n_users", "seed_size_mean", "n_users_short_of_k",
                  "frac_volunteered_items_liked", "draw", "rng"):
            assert draw[f] == rbank[f], f"DRAW GATE: '{f}' {draw[f]!r} != R's banked {rbank[f]!r}"
        log(f"[gate 4] the volunteered draw is BIT-IDENTICAL to experiment R's banked k=4 draw "
            f"(mean seed {draw['seed_size_mean']}, {draw['frac_volunteered_items_liked']:.6f} liked)")

    gen, gvocab = load_item_genres(E["ctx"].ni)
    labels, n_no_liked, n_ungenred = partition(E, rows, vol, gen)

    placebo_note = None
    if args.cohorts == "placebo":
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(labels))
        labels = [labels[i] for i in perm]
        placebo_note = ("PLACEBO: the real volunteered-genre label vector was PERMUTED with "
                        f"default_rng({SEED}). Group SIZES are identical to the real cohorts; the "
                        "assignment carries no information. This is the null level for an in-sample "
                        "cohort-conditioned greedy.")
        log("!" * 110)
        log("! " + placebo_note)
        log("!" * 110)

    groups = {}
    for r, g in zip(rows, labels):
        groups.setdefault(g, []).append(int(r))
    sizes = {k: len(v) for k, v in groups.items()}
    order_groups = sorted(sizes, key=lambda k: (-sizes[k], k))
    n_small = sum(1 for v in sizes.values() if v < SMALL_GROUP)
    n_fallback = sum(v for v in sizes.values() if v < SMALL_GROUP)

    # ---- COMPOSITION, PRINTED BEFORE ANY ORDER IS BUILT ----------------------------------------
    genre_sizes = json.load(open(GENRE_S1_JSON))["group_sizes"]
    log("=" * 110)
    log(f"COHORT COMPOSITION ({args.cohorts}) -- k={args.k_vol} volunteered items per user, "
        f"{len(sizes)} groups over {len(rows)} users")
    log(f"rule: most frequent MovieLens genre over the user's LIKED VOLUNTEERED items "
        f"(level >= {LIKE_MIN_LEVEL} == >= 4.0 stars), ties ALPHABETICAL, none -> '{NONE_GROUP}'")
    log("-" * 110)
    log(f"{'group':<16}{'n (V)':>9}{'% (V)':>9}{'n (genre arm)':>16}{'% (genre)':>11}{'small?':>9}")
    for k in order_groups:
        ng = genre_sizes.get(k, 0)
        log(f"{k:<16}{sizes[k]:>9}{100.0 * sizes[k] / len(rows):>8.2f}%{ng:>16}"
            f"{100.0 * ng / len(rows):>10.2f}%{('YES' if sizes[k] < SMALL_GROUP else ''):>9}")
    for k in sorted(genre_sizes, key=lambda x: -genre_sizes[x]):
        if k not in sizes:
            log(f"{k:<16}{0:>9}{0.0:>8.2f}%{genre_sizes[k]:>16}"
                f"{100.0 * genre_sizes[k] / len(rows):>10.2f}%{'YES':>9}")
    log("-" * 110)
    log(f"[census] '{NONE_GROUP}' = {n_no_liked} users with NO liked volunteered item + "
        f"{n_ungenred} whose liked volunteered items carry no genre")
    log(f"[census] {n_small}/{len(sizes)} groups below {SMALL_GROUP} users, holding {n_fallback} "
        f"users ({100.0 * n_fallback / len(rows):.2f}%) -- these are the fallback_200 users. "
        f"HARD RULE 1: none merged, none dropped; the all_groups arm runs every one of them.")
    log("=" * 110)
    if args.census_only:
        cen = {"analysis": "volunteered_cohort_census", "cohorts": args.cohorts, "citable": False,
               "split": "validation", "test_touched": False, "k_vol": args.k_vol,
               "n_cohort": len(rows), "sizes": sizes, "genre_arm_sizes": genre_sizes,
               "n_users_no_liked_volunteered": n_no_liked,
               "n_users_liked_but_ungenred": n_ungenred,
               "n_groups_below_threshold": n_small, "n_fallback_users": n_fallback,
               "draw": draw, "small_group_threshold": SMALL_GROUP}
        p = os.path.join(OUTDIR, f"{label}_census.json")
        json.dump(cen, open(p, "w"), indent=2, default=float)
        log(f"[census-only] wrote {os.path.basename(p)} -- stopping before any search")
        return cen

    # ---- pool ----------------------------------------------------------------------------------
    item_tables = {"pop": np.asarray(E["ctx"].cnt, np.float64)}
    pool = [int(i) for i in _ranked_pool("pop", item_tables, args.n_items, "item")]
    L = args.L
    budgets = sorted(set([K for K in BUDGETS if K <= L] + [L]))
    assert all(q in set(pool) for q in comparator[:L]), \
        "the comparator's first L items are not all inside the candidate pool"
    log(f"[pool] items-only, bank='pop', top-{args.n_items} -> {len(pool)} candidates; L={L}; "
        f"compared ITEM-FOR-ITEM against the global order at K in {budgets}, both scored FROM THE "
        f"SEEDED STATE. Nothing is charged for the volunteered items -- they were free.")

    sig = {"tower": os.path.basename(args.snapshot), "conc": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "n_items": len(pool),
           "cohorts": args.cohorts, "k_vol": args.k_vol, "L_items": L,
           "split": "validation", "n_cohort": len(rows), "sizes": sizes,
           "comparator": cmp_blob["label"], "seed": SEED,
           "pool_contract": pool_contract(E["pool"], rows, policy=E["pool_policy"])}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    if part.get("sig") != sig:
        if part:
            log("[resume] the partial on disk carries a DIFFERENT contract -- starting cold")
        part = {"sig": sig, "groups": {}}

    def flush():
        with open(PARTIAL + ".tmp", "w") as fh:
            json.dump(part, fh, indent=2, default=float)
        os.replace(PARTIAL + ".tmp", PARTIAL)

    banked = {"__rows__": np.asarray(rows, np.int64)}
    vol_by_row = {int(r): [int(x) for x in v] for r, v in zip(rows, vol)}

    for gname in order_groups:
        grows = groups[gname]
        gvol = [vol_by_row[r] for r in grows]
        if part["groups"].get(gname, {}).get("done"):
            log(f"[{gname}] already complete in the sidecar -- skipping the search")
        set_active_rows(E, grows)
        memo = {}
        rec = part["groups"].setdefault(gname, {})
        rec.update({"n_users": len(grows), "cohort_label": gname,
                    "small_group": bool(len(grows) < SMALL_GROUP)})
        t0 = time.time()

        def on_pick(seq, gains, nev, _r=rec, _t=t0):
            _r["order"] = [int(x) for x in seq]
            _r["gains"] = [float(x) for x in gains]
            _r["n_evals"] = int(nev)
            _r["minutes"] = round((time.time() - _t) / 60, 2)
            flush()

        if rec.get("done") and len(rec.get("order", [])) == L:
            own = [int(x) for x in rec["order"]]
            gains, nev = rec.get("gains", []), int(rec.get("n_evals", 0))
        else:
            hp = os.path.join(OUTDIR, "heaps", f"{label}_{gname}.json")
            hsig = heap_sig(E, grows, gvol, pool, args.objective, args.k_vol)
            own, gains, nev = greedy_seeded(E, grows, gvol, pool, L, args.objective, gname, memo,
                                            on_pick=on_pick, heap_path=hp, sig=hsig)
            rec.update({"order": [int(x) for x in own], "gains": [float(x) for x in gains],
                        "n_evals": int(nev), "minutes": round((time.time() - t0) / 60, 2)})

        # ---- item-for-item curve + per-user vectors, both arms scored FROM THE SEEDED STATE -----
        curve = {}
        for K in budgets:
            of, ot = seeded_eval(E, grows, gvol, own[:K])
            lf, lt = seeded_eval(E, grows, gvol, comparator[:K])
            banked[f"{gname}|{K}|own_full"] = of.astype(np.float32)
            banked[f"{gname}|{K}|own_tail"] = ot.astype(np.float32)
            banked[f"{gname}|{K}|glob_full"] = lf.astype(np.float32)
            banked[f"{gname}|{K}|glob_tail"] = lt.astype(np.float32)
            r_own = redundancy(gvol, own[:K])
            r_glb = redundancy(gvol, comparator[:K])
            curve[str(K)] = {
                "group_full@10": float(np.nanmean(of)), "group_tail@10": float(np.nanmean(ot)),
                "global_full@10": float(np.nanmean(lf)), "global_tail@10": float(np.nanmean(lt)),
                "diff_full@10": float(np.nanmean(of - lf)),
                "diff_tail@10": float(np.nanmean(ot - lt)),
                "n_full": int(np.isfinite(of).sum()),
                "reask_own_mean": r_own[0], "reask_own_frac_users": r_own[1],
                "reask_global_mean": r_glb[0], "reask_global_frac_users": r_glb[1],
                "overlap": order_overlap([(0, i) for i in own], [(0, i) for i in comparator], K)}
        rec["curve_item_matched"] = curve
        rec["rows"] = [int(r) for r in grows]
        banked[f"{gname}|rows"] = np.asarray(grows, np.int64)
        jj = 0 if args.objective == "full" else 1
        d = curve[str(L)]["diff_full@10"] if jj == 0 else curve[str(L)]["diff_tail@10"]
        rec["sanity_pass"] = bool(d >= -1e-9)
        rec["done"] = True
        flush()
        np.savez_compressed(NPZ, **banked)
        ov = curve[str(L)]["overlap"]
        log(f"[{gname}] DONE n={len(grows)} evals={rec.get('n_evals')} "
            f"({(time.time() - t0) / 60:.1f}m) order={own}")
        log(f"[{gname}] @K={L}: own {curve[str(L)]['group_full@10']:.5f} vs global "
            f"{curve[str(L)]['global_full@10']:.5f} ({d:+.5f}) | overlap {ov['set_overlap']}/{L}, "
            f"{ov['positions_differ']} positions differ | re-asks own {curve[str(L)]['reask_own_mean']:.3f} "
            f"vs global {curve[str(L)]['reask_global_mean']:.3f}")
        if not rec["sanity_pass"]:
            log("!" * 110)
            log(f"[{gname}] SANITY *** FAIL *** own < global on its OWN selection users ({d:+.5f}). "
                f"The global order is inside the search space so this cannot happen unless the "
                f"search, the cohort masking or the memo is broken. DO NOT INTERPRET THIS GROUP.")
            log("!" * 110)

    # ---- cohort-weighted results, both pre-specified arms, paired CIs ---------------------------
    pos = {int(r): i for i, r in enumerate(rows)}
    weighted, per_group = {}, {}
    for K in budgets:
        dF = np.full(len(rows), np.nan); dT = np.full(len(rows), np.nan)
        dFf = np.full(len(rows), np.nan); dTf = np.full(len(rows), np.nan)
        for g, rec in part["groups"].items():
            jj = np.asarray([pos[int(r)] for r in rec["rows"]], np.int64)
            of = banked[f"{g}|{K}|own_full"].astype(np.float64)
            ot = banked[f"{g}|{K}|own_tail"].astype(np.float64)
            lf = banked[f"{g}|{K}|glob_full"].astype(np.float64)
            lt = banked[f"{g}|{K}|glob_tail"].astype(np.float64)
            dF[jj] = of - lf; dT[jj] = ot - lt
            if rec["small_group"]:                      # fallback_200: asked the GLOBAL order
                dFf[jj] = 0.0; dTf[jj] = 0.0
            else:
                dFf[jj] = of - lf; dTf[jj] = ot - lt
        weighted[str(K)] = {
            "all_groups": {"full": paired_bootstrap(dF), "tail": paired_bootstrap(dT)},
            "fallback_200": {"full": paired_bootstrap(dFf), "tail": paired_bootstrap(dTf)}}
        log(f"[weighted] K={K:<3} all_groups full {weighted[str(K)]['all_groups']['full']['mean']:+.5f} "
            f"[{weighted[str(K)]['all_groups']['full']['lo']:+.5f}, "
            f"{weighted[str(K)]['all_groups']['full']['hi']:+.5f}]  tail "
            f"{weighted[str(K)]['all_groups']['tail']['mean']:+.5f}  | fallback_200 full "
            f"{weighted[str(K)]['fallback_200']['full']['mean']:+.5f}")

    log("-" * 110)
    log(f"{'group':<16}{'n':>7}{'diff_full':>12}{'ci_lo':>11}{'ci_hi':>11}{'diff_tail':>12}"
        f"{'overlap':>9}{'reask':>8}{'small?':>8}")
    for g in sorted(part["groups"], key=lambda k: -part["groups"][k]["n_users"]):
        rec = part["groups"][g]; c = rec["curve_item_matched"][str(L)]
        ci = paired_bootstrap(banked[f"{g}|{L}|own_full"].astype(np.float64)
                              - banked[f"{g}|{L}|glob_full"].astype(np.float64))
        cit = paired_bootstrap(banked[f"{g}|{L}|own_tail"].astype(np.float64)
                               - banked[f"{g}|{L}|glob_tail"].astype(np.float64))
        per_group[g] = {"n_users": rec["n_users"], "small_group": rec["small_group"],
                        "diff_full": ci, "diff_tail": cit,
                        "set_overlap": c["overlap"]["set_overlap"],
                        "positions_differ": c["overlap"]["positions_differ"],
                        "reask_own_mean": c["reask_own_mean"],
                        "reask_global_mean": c["reask_global_mean"],
                        "order": rec["order"], "n_evals": rec.get("n_evals")}
        log(f"{g:<16}{rec['n_users']:>7}{ci['mean']:>12.5f}{ci['lo']:>11.5f}{ci['hi']:>11.5f}"
            f"{cit['mean']:>12.5f}{c['overlap']['set_overlap']:>9}{c['reask_own_mean']:>8.3f}"
            f"{('YES' if rec['small_group'] else ''):>8}")
    log("-" * 110)

    out = {"analysis": "volunteered_evidence_cohort_conditioned_greedy",
           "label": label, "citable": True, "cohorts": args.cohorts,
           "split": "validation", "test_touched": False,
           "CEILING_NOTE": CEILING_NOTE, "sanity_meaning": SANITY_MEANING,
           "placebo_note": placebo_note,
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "mask_policy": "B", "pool_policy": E["pool_policy"], "pool_contract": E["pool_contract"],
           "objective": args.objective, "item_bank": "pop", "n_items_pool": len(pool),
           "k_vol": args.k_vol, "draw": draw, "L_items": L, "budgets_items": budgets,
           "n_cohort": len(rows), "n_groups": len(sizes), "group_sizes": sizes,
           "genre_arm_sizes": genre_sizes,
           "small_group_threshold": SMALL_GROUP, "n_groups_below_threshold": n_small,
           "n_fallback_users": n_fallback,
           "n_users_no_liked_volunteered": n_no_liked, "n_users_liked_but_ungenred": n_ungenred,
           "cohort_rule": ("most frequent MovieLens genre over the user's LIKED VOLUNTEERED items "
                           f"(level >= {LIKE_MIN_LEVEL}); ties ALPHABETICAL; none -> "
                           f"'{NONE_GROUP}'. Identical to the genre arm's rule, computed over the "
                           "volunteered set instead of the whole fold-in half."),
           "scoring": ("every arm is scored FROM THE SEEDED STATE: user u's folded item sequence is "
                       "dedup(volunteered_u + asked[:K]). A re-asked volunteered item folds ONCE and "
                       "the turn is burnt (experiment R's semantics)."),
           "comparator": {"file": os.path.basename(COMPARATOR_JSON), "label": cmp_blob["label"],
                          "sequence": comparator},
           "gates": {"canonical_m0_full": f0, "canonical_m0_banked": VAL_M0_FULL,
                     "comparator_unseeded_curve": gate2,
                     "genv_eval_multi_equals_genv_eval": True,
                     "draw_identical_to_R_k4": bool(args.k_vol == 4)},
           "cohort_weighted": weighted, "per_group": per_group, "groups": part["groups"],
           "sig": sig, "per_user_npz": os.path.basename(NPZ),
           "seconds": round(time.time() - t00, 1)}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    np.savez_compressed(NPZ, **banked)
    log(f"[done] wrote {os.path.basename(OUT)} in {(time.time() - t00) / 60:.1f}m")
    return out


if __name__ == "__main__":
    main()
